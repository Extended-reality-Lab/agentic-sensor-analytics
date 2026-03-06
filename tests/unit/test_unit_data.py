"""
Unit tests for the data layer.
Written against the actual source files:
  - models.py      (SensorReading, SensorMetadata, NodeMetadata, DataQuery,
                    DataQueryResult, TimeRange, SystemState)
  - config.py      (APISettings, CacheSettings, ValidationSettings,
                    SensorMapping, DataConfig)
  - api_client.py  (SMTAPIClient, SMTAPIError, SMTAuthenticationError)
  - repository.py  (SensorDataRepository, RepositoryError)
  - llm_bridge.py  (LLMDataBridge)

Every test is fully offline — no live API or YAML files on disk required.
"""

import pytest
import yaml
import tempfile
import requests as req_lib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timezone, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

# --- models ---------------------------------------------------------------
from data.models import (
    SensorReading,
    SensorMetadata,
    NodeMetadata,
    DataQuery,
    DataQueryResult,
    TimeRange,
    SystemState,
)

# --- config ---------------------------------------------------------------
from data.config import (
    APISettings,
    CacheSettings,
    ValidationSettings,
    SensorMapping,
    DataConfig,
)

# --- api_client -----------------------------------------------------------
from data.api_client import SMTAPIClient, SMTAPIError, SMTAuthenticationError

# --- repository -----------------------------------------------------------
from data.repository import SensorDataRepository, RepositoryError

# --- llm_bridge -----------------------------------------------------------
from data.llm_bridge import LLMDataBridge


# ==========================================================================
# Shared helpers
# ==========================================================================

def _dt(year=2025, month=1, day=14, **kw):
    """Shorthand for a timezone-aware datetime."""
    return datetime(year, month, day, tzinfo=timezone.utc, **kw)


def _make_config(job_id=1):
    """Minimal DataConfig that bypasses all file I/O."""
    return DataConfig(
        api=APISettings(
            username="testuser",
            password="testpass",
            base_url="http://test.example.com",
            job_id=job_id,
        )
    )


def _make_node(node_id=9279, name="15_9279", location="15_9279"):
    return NodeMetadata(node_id=node_id, physical_id=node_id, name=name, location=location)


def _make_sensor(
    sensor_id=1,
    sensor_type="temperature",
    location="15_9279",
    unit="°C",
    node_id=9279,
):
    return SensorMetadata(
        sensor_id=sensor_id,
        name="sensor_" + str(sensor_id),
        sensor_type=sensor_type,
        location=location,
        unit=unit,
        node_id=node_id,
    )


def _make_readings(count=24):
    start = _dt()
    return [
        SensorReading(
            timestamp=start + timedelta(hours=i),
            value=20.0 + i * 0.5,
            unit="°C",
            raw_value=20.0 + i * 0.5,
            quality_flag=0,
        )
        for i in range(count)
    ]


def _make_mock_api(nodes=None, sensors=None, readings=None):
    """
    Build a Mock that matches SMTAPIClient's interface exactly.
    login() sets authenticated=True; logout() sets it False.
    """
    api = Mock(spec=SMTAPIClient)
    api.authenticated = False

    def _login():
        api.authenticated = True
        return True

    def _logout():
        api.authenticated = False
        return True

    api.login.side_effect = _login
    api.logout.side_effect = _logout
    api.list_nodes.return_value = nodes if nodes is not None else [_make_node()]
    api.list_sensors.return_value = sensors if sensors is not None else [_make_sensor()]
    api.get_sensor_data.return_value = readings if readings is not None else _make_readings()
    return api


def _make_repo(api=None, config=None):
    if config is None:
        config = _make_config()
    if api is None:
        api = _make_mock_api()
    return SensorDataRepository(api_client=api, config=config)


# ==========================================================================
# SensorReading
# ==========================================================================

class TestSensorReading:

    def test_basic_creation(self):
        r = SensorReading(timestamp=_dt(), value=22.5, unit="°C")
        assert r.value == 22.5
        assert r.unit == "°C"

    def test_quality_flag_0_valid(self):
        r = SensorReading(timestamp=_dt(), value=0.0, unit="°C", quality_flag=0)
        assert r.quality_flag == 0

    def test_quality_flag_1_valid(self):
        r = SensorReading(timestamp=_dt(), value=0.0, unit="°C", quality_flag=1)
        assert r.quality_flag == 1

    def test_quality_flag_2_valid(self):
        r = SensorReading(timestamp=_dt(), value=0.0, unit="°C", quality_flag=2)
        assert r.quality_flag == 2

    def test_quality_flag_3_raises(self):
        with pytest.raises(ValueError):
            SensorReading(timestamp=_dt(), value=0.0, unit="°C", quality_flag=3)

    def test_quality_flag_negative_raises(self):
        with pytest.raises(ValueError):
            SensorReading(timestamp=_dt(), value=0.0, unit="°C", quality_flag=-1)

    def test_raw_value_optional_defaults_none(self):
        r = SensorReading(timestamp=_dt(), value=0.0, unit="°C")
        assert r.raw_value is None

    def test_raw_value_set(self):
        r = SensorReading(timestamp=_dt(), value=22.5, unit="°C", raw_value=22.4)
        assert r.raw_value == 22.4

    def test_default_quality_flag_is_zero(self):
        r = SensorReading(timestamp=_dt(), value=0.0, unit="°C")
        assert r.quality_flag == 0


# ==========================================================================
# TimeRange
# ==========================================================================

class TestTimeRange:

    def test_valid_range(self):
        tr = TimeRange(start_time=_dt(day=1), end_time=_dt(day=2))
        assert tr.start_time < tr.end_time

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError):
            TimeRange(start_time=_dt(day=2), end_time=_dt(day=1))

    def test_end_equal_start_raises(self):
        t = _dt()
        with pytest.raises(ValueError):
            TimeRange(start_time=t, end_time=t)


# ==========================================================================
# DataQuery
# ==========================================================================

class TestDataQuery:

    def test_single_location_string(self):
        q = DataQuery(
            sensor_type="temperature",
            location="Node 15",
            start_time=_dt(day=1),
            end_time=_dt(day=2),
        )
        assert q.location == "Node 15"

    def test_multiple_locations_list(self):
        q = DataQuery(
            sensor_type="humidity",
            location=["Node 14", "Node 15"],
            start_time=_dt(day=1),
            end_time=_dt(day=2),
        )
        assert isinstance(q.location, list)
        assert len(q.location) == 2

    def test_get_locations_list_from_string(self):
        q = DataQuery(
            sensor_type="temperature",
            location="Node 15",
            start_time=_dt(day=1),
            end_time=_dt(day=2),
        )
        assert q.get_locations_list() == ["Node 15"]

    def test_get_locations_list_from_list(self):
        q = DataQuery(
            sensor_type="temperature",
            location=["Node 14", "Node 15"],
            start_time=_dt(day=1),
            end_time=_dt(day=2),
        )
        assert q.get_locations_list() == ["Node 14", "Node 15"]

    def test_operation_optional(self):
        q = DataQuery(
            sensor_type="temperature",
            location="Node 15",
            start_time=_dt(day=1),
            end_time=_dt(day=2),
        )
        assert q.operation is None


# ==========================================================================
# DataQueryResult
# ==========================================================================

class TestDataQueryResult:

    def _query(self):
        return DataQuery(
            sensor_type="temperature",
            location="Node 15",
            start_time=_dt(day=14),
            end_time=_dt(day=15),
        )

    def test_valid_result(self):
        readings = _make_readings(5)
        result = DataQueryResult(
            sensor_metadata=_make_sensor(),
            readings=readings,
            query_params=self._query(),
            total_readings=5,
        )
        assert result.total_readings == 5

    def test_wrong_total_readings_raises(self):
        readings = _make_readings(5)
        with pytest.raises(ValueError):
            DataQueryResult(
                sensor_metadata=_make_sensor(),
                readings=readings,
                query_params=self._query(),
                total_readings=99,
            )

    def test_has_quality_issues_defaults_false(self):
        readings = _make_readings(3)
        result = DataQueryResult(
            sensor_metadata=_make_sensor(),
            readings=readings,
            query_params=self._query(),
            total_readings=3,
        )
        assert result.has_quality_issues is False


# ==========================================================================
# APISettings
# ==========================================================================

class TestAPISettings:

    def test_defaults(self):
        s = APISettings(job_id=1)
        assert s.timeout == 30
        assert s.max_retries == 3

    def test_job_id_required(self):
        with pytest.raises(Exception):
            APISettings()  # missing job_id

    def test_custom_timeout(self):
        s = APISettings(job_id=5, timeout=60)
        assert s.timeout == 60

    def test_custom_max_retries(self):
        s = APISettings(job_id=5, max_retries=5)
        assert s.max_retries == 5


# ==========================================================================
# CacheSettings
# ==========================================================================

class TestCacheSettings:

    def test_enabled_default_true(self):
        assert CacheSettings().enabled is True

    def test_ttl_default(self):
        assert CacheSettings().ttl_seconds == 3600

    def test_max_size_default(self):
        assert CacheSettings().max_size_mb == 100


# ==========================================================================
# ValidationSettings
# ==========================================================================

class TestValidationSettings:

    def test_check_ranges_default_true(self):
        assert ValidationSettings().check_ranges is True

    def test_flag_outliers_default_true(self):
        assert ValidationSettings().flag_outliers is True

    def test_outlier_std_devs_default(self):
        assert ValidationSettings().outlier_std_devs == 3.0


# ==========================================================================
# SensorMapping
# ==========================================================================

class TestSensorMapping:

    def test_temperature_keywords_include_temp(self):
        assert "temp" in SensorMapping().temperature_keywords

    def test_temperature_keywords_include_temperature(self):
        assert "temperature" in SensorMapping().temperature_keywords

    def test_temperature_keywords_include_tmp(self):
        assert "tmp" in SensorMapping().temperature_keywords

    def test_humidity_keywords_include_humidity(self):
        assert "humidity" in SensorMapping().humidity_keywords

    def test_humidity_keywords_include_rh(self):
        assert "rh" in SensorMapping().humidity_keywords

    def test_co2_keywords_include_co2(self):
        assert "co2" in SensorMapping().co2_keywords

    def test_moisture_keywords_include_moisture(self):
        assert "moisture" in SensorMapping().moisture_keywords

    def test_moisture_keywords_include_mc(self):
        assert "mc" in SensorMapping().moisture_keywords


# ==========================================================================
# DataConfig
# ==========================================================================

class TestDataConfig:

    def test_from_yaml_loads_correctly(self):
        data = {
            "api": {
                "username": "user",
                "password": "pass",
                "base_url": "http://example.com",
                "job_id": 5,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            config = DataConfig.from_yaml(path)
            assert config.api.username == "user"
            assert config.api.job_id == 5
        finally:
            Path(path).unlink()

    def test_from_yaml_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            DataConfig.from_yaml("does_not_exist.yaml")

    def test_from_yaml_empty_file_raises_value_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name
        try:
            with pytest.raises(ValueError):
                DataConfig.from_yaml(path)
        finally:
            Path(path).unlink()

    def test_save_yaml_redacts_password(self):
        config = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.yaml"
            config.save_yaml(out)
            text = out.read_text()
            assert "testpass" not in text
            assert "REDACTED" in text

    def test_save_yaml_preserves_username(self):
        config = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.yaml"
            config.save_yaml(out)
            loaded = yaml.safe_load(out.read_text())
            assert loaded["api"]["username"] == "testuser"

    def test_save_yaml_creates_parent_directories(self):
        config = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "subdir" / "nested" / "out.yaml"
            config.save_yaml(out)
            assert out.exists()

    def test_cache_defaults_applied(self):
        assert _make_config().cache.enabled is True
        assert _make_config().cache.ttl_seconds == 3600

    def test_validation_defaults_applied(self):
        assert _make_config().validation.check_ranges is True
        assert _make_config().validation.outlier_std_devs == 3.0


# ==========================================================================
# SMTAPIClient — _determine_unit (static, no network)
# ==========================================================================

class TestDetermineUnit:
    """_determine_unit is @staticmethod — call without an instance."""

    def test_temperature_keyword_returns_celsius(self):
        assert SMTAPIClient._determine_unit("temperature") == "°C"

    def test_temp_prefix_returns_celsius(self):
        assert SMTAPIClient._determine_unit("temp_sensor") == "°C"

    def test_humidity_returns_percent(self):
        assert SMTAPIClient._determine_unit("humidity") == "%"

    def test_rh_in_name_returns_percent(self):
        assert SMTAPIClient._determine_unit("rh sensor") == "%"

    def test_co2_returns_ppm(self):
        assert SMTAPIClient._determine_unit("co2") == "ppm"

    def test_carbon_in_name_returns_ppm(self):
        assert SMTAPIClient._determine_unit("carbon dioxide") == "ppm"

    def test_moisture_returns_percent(self):
        assert SMTAPIClient._determine_unit("moisture") == "%"

    def test_mc_in_name_returns_percent(self):
        assert SMTAPIClient._determine_unit("mc sensor") == "%"

    def test_load_returns_newtons(self):
        assert SMTAPIClient._determine_unit("load cell") == "N"

    def test_force_returns_newtons(self):
        assert SMTAPIClient._determine_unit("force gauge") == "N"

    def test_strain_returns_microstrain(self):
        assert SMTAPIClient._determine_unit("strain gauge") == "με"

    def test_unrecognised_returns_units(self):
        assert SMTAPIClient._determine_unit("unknown_widget") == "units"

    def test_case_insensitive(self):
        # The implementation lowercases internally
        assert SMTAPIClient._determine_unit("TEMPERATURE") == "°C"
        assert SMTAPIClient._determine_unit("Humidity") == "%"


# ==========================================================================
# SMTAPIClient — init and from_config
# ==========================================================================

class TestSMTAPIClientInit:

    def test_authenticated_starts_false(self):
        c = SMTAPIClient(username="u", password="p", base_url="http://x.com")
        assert c.authenticated is False

    def test_session_id_starts_none(self):
        c = SMTAPIClient(username="u", password="p", base_url="http://x.com")
        assert c.session_id is None

    def test_defaults_timeout_and_retries(self):
        c = SMTAPIClient(username="u", password="p", base_url="http://x.com")
        assert c.timeout == 30
        assert c.max_retries == 3

    def test_from_config_maps_fields(self):
        config = _make_config()
        c = SMTAPIClient.from_config(config)
        assert c.username == "testuser"
        assert c.base_url == "http://test.example.com"
        assert c.timeout == 30
        assert c.max_retries == 3


# ==========================================================================
# SMTAPIClient — _make_request retry logic (mock the session)
# ==========================================================================

class TestMakeRequestRetry:

    def _client(self, retries=3):
        c = SMTAPIClient(
            username="u", password="p",
            base_url="http://test.com",
            max_retries=retries,
        )
        c.session = Mock()
        return c

    def test_retries_exactly_max_retries_times_on_connection_error(self):
        c = self._client(retries=3)
        c.session.get.side_effect = req_lib.ConnectionError("down")
        with pytest.raises(SMTAPIError):
            c._make_request("listNode")
        assert c.session.get.call_count == 3

    def test_raises_smt_api_error_message_contains_attempts(self):
        c = self._client(retries=2)
        c.session.get.side_effect = req_lib.Timeout("timeout")
        with pytest.raises(SMTAPIError, match="2 attempts"):
            c._make_request("listNode")

    def test_xml_error_element_raises_smt_api_error(self):
        import xml.etree.ElementTree as ET
        c = self._client(retries=1)
        xml_with_error = b"<response><error>Unauthorised</error></response>"
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = xml_with_error
        c.session.get.return_value = mock_response
        with pytest.raises(SMTAPIError, match="Unauthorised"):
            c._make_request("listNode")

    def test_successful_response_returned_on_first_attempt(self):
        import xml.etree.ElementTree as ET
        c = self._client(retries=3)
        xml_ok = b"<response><ok/></response>"
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = xml_ok
        c.session.get.return_value = mock_response
        result = c._make_request("listNode")
        assert isinstance(result, ET.Element)
        c.session.get.assert_called_once()

    def test_action_param_always_added_to_request(self):
        import xml.etree.ElementTree as ET
        c = self._client(retries=1)
        xml_ok = b"<response/>"
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = xml_ok
        c.session.get.return_value = mock_response
        c._make_request("myAction")
        call_kwargs = c.session.get.call_args[1]
        assert call_kwargs["params"]["action"] == "myAction"


# ==========================================================================
# SMTAPIClient — login / logout / context manager
# ==========================================================================

class TestLoginLogout:

    def _client_with_mock_request(self, xml_bytes):
        import xml.etree.ElementTree as ET
        c = SMTAPIClient(username="u", password="p", base_url="http://x.com")
        root = ET.fromstring(xml_bytes)
        c._make_request = Mock(return_value=root)
        return c

    def test_successful_login_sets_authenticated(self):
        c = self._client_with_mock_request(
            b"<response><login>success</login><PHPSESSID>abc123</PHPSESSID></response>"
        )
        result = c.login()
        assert result is True
        assert c.authenticated is True
        assert c.session_id == "abc123"

    def test_failed_login_raises_auth_error(self):
        c = self._client_with_mock_request(
            b"<response><login>fail</login></response>"
        )
        with pytest.raises(SMTAuthenticationError):
            c.login()

    def test_logout_when_not_authenticated_returns_true(self):
        c = SMTAPIClient(username="u", password="p", base_url="http://x.com")
        assert c.authenticated is False
        assert c.logout() is True

    def test_successful_logout_clears_session(self):
        c = self._client_with_mock_request(
            b"<response><logout>success</logout></response>"
        )
        c.authenticated = True
        c.session_id = "abc123"
        result = c.logout()
        assert result is True
        assert c.authenticated is False
        assert c.session_id is None

    def test_context_manager_calls_login_and_logout(self):
        c = SMTAPIClient(username="u", password="p", base_url="http://x.com")
        c.login = Mock(return_value=True)
        c.logout = Mock(return_value=True)
        with c:
            c.login.assert_called_once()
        c.logout.assert_called_once()

    def test_list_nodes_raises_when_not_authenticated(self):
        c = SMTAPIClient(username="u", password="p", base_url="http://x.com")
        assert c.authenticated is False
        with pytest.raises(SMTAPIError, match="Not authenticated"):
            c.list_nodes(1)

    def test_list_sensors_raises_when_not_authenticated(self):
        c = SMTAPIClient(username="u", password="p", base_url="http://x.com")
        with pytest.raises(SMTAPIError, match="Not authenticated"):
            c.list_sensors(1)

    def test_get_sensor_data_raises_when_not_authenticated(self):
        c = SMTAPIClient(username="u", password="p", base_url="http://x.com")
        with pytest.raises(SMTAPIError, match="Not authenticated"):
            c.get_sensor_data(1, _dt(day=1), _dt(day=2))


# ==========================================================================
# Repository — connect / disconnect
# ==========================================================================

class TestRepositoryConnectDisconnect:

    def test_connect_calls_login_when_not_authenticated(self):
        api = _make_mock_api()
        repo = _make_repo(api=api)
        repo.connect()
        api.login.assert_called_once()

    def test_connect_skips_login_when_already_authenticated(self):
        api = _make_mock_api()
        api.authenticated = True
        repo = _make_repo(api=api)
        repo.connect()
        api.login.assert_not_called()

    def test_disconnect_calls_logout_when_authenticated(self):
        api = _make_mock_api()
        api.authenticated = True
        repo = _make_repo(api=api)
        repo.disconnect()
        api.logout.assert_called_once()

    def test_disconnect_skips_logout_when_not_authenticated(self):
        api = _make_mock_api()
        api.authenticated = False
        repo = _make_repo(api=api)
        repo.disconnect()
        api.logout.assert_not_called()

    def test_context_manager_connects_on_enter(self):
        api = _make_mock_api()
        repo = _make_repo(api=api)
        with repo:
            api.login.assert_called_once()

    def test_context_manager_disconnects_on_exit(self):
        api = _make_mock_api()
        repo = _make_repo(api=api)
        with repo:
            pass
        api.logout.assert_called_once()


# ==========================================================================
# Repository — metadata caching
# ==========================================================================

class TestMetadataCaching:

    def test_list_nodes_called_only_once_on_repeated_access(self):
        api = _make_mock_api()
        repo = _make_repo(api=api)
        repo.connect()
        repo._get_all_nodes()
        repo._get_all_nodes()
        repo._get_all_nodes()
        api.list_nodes.assert_called_once()

    def test_list_sensors_called_only_once_on_repeated_access(self):
        api = _make_mock_api()
        repo = _make_repo(api=api)
        repo.connect()
        repo._get_all_sensors()
        repo._get_all_sensors()
        api.list_sensors.assert_called_once()

    def test_clear_cache_resets_nodes_cache_to_none(self):
        repo = _make_repo()
        repo.connect()
        repo._get_all_nodes()
        assert repo._nodes_cache is not None
        repo.clear_cache()
        assert repo._nodes_cache is None

    def test_clear_cache_resets_sensors_cache_to_none(self):
        repo = _make_repo()
        repo.connect()
        repo._get_all_sensors()
        assert repo._sensors_cache is not None
        repo.clear_cache()
        assert repo._sensors_cache is None

    def test_after_clear_cache_sensors_refetched_from_api(self):
        api = _make_mock_api()
        repo = _make_repo(api=api)
        repo.connect()
        repo._get_all_sensors()
        repo.clear_cache()
        repo._get_all_sensors()
        assert api.list_sensors.call_count == 2

    def test_get_all_sensors_sets_location_from_parent_node(self):
        """_get_all_sensors must overwrite sensor.location with node.location."""
        node = _make_node(node_id=9279, location="15_9279")
        sensor = _make_sensor(node_id=9279, location="stale_location")
        api = _make_mock_api(nodes=[node], sensors=[sensor])
        repo = _make_repo(api=api)
        repo.connect()
        sensors = repo._get_all_sensors()
        assert sensors[0].location == "15_9279"

    def test_sensor_from_failed_node_silently_skipped(self):
        """If list_sensors raises SMTAPIError for a node, it should be skipped."""
        from data.api_client import SMTAPIError as _Err
        node1 = _make_node(node_id=1, location="loc1")
        node2 = _make_node(node_id=2, location="loc2")
        api = _make_mock_api(nodes=[node1, node2])
        api.list_sensors.side_effect = [_Err("timeout"), [_make_sensor(node_id=2)]]
        repo = _make_repo(api=api)
        repo.connect()
        sensors = repo._get_all_sensors()
        # Only node2's sensor survives
        assert len(sensors) == 1
        assert sensors[0].node_id == 2


# ==========================================================================
# Repository — _get_human_readable_location
# ==========================================================================

class TestGetHumanReadableLocation:

    def test_underscore_format_extracts_first_segment(self):
        assert _make_repo()._get_human_readable_location("15_9279") == "Node 15"

    def test_two_digit_node_number(self):
        assert _make_repo()._get_human_readable_location("14_1234") == "Node 14"

    def test_single_digit_node_number(self):
        assert _make_repo()._get_human_readable_location("3_0042") == "Node 3"

    def test_multiple_underscores_uses_first_segment_only(self):
        assert _make_repo()._get_human_readable_location("15_9279_extra") == "Node 15"

    def test_no_underscore_returns_name_unchanged(self):
        assert _make_repo()._get_human_readable_location("Node 15") == "Node 15"

    def test_plain_string_without_underscore_unchanged(self):
        assert _make_repo()._get_human_readable_location("hallway") == "hallway"


# ==========================================================================
# Repository — _normalize_sensor_type
# ==========================================================================

class TestNormalizeSensorType:

    def test_temp_returns_temperature(self):
        assert _make_repo()._normalize_sensor_type("temp") == "temperature"

    def test_temperature_returns_temperature(self):
        assert _make_repo()._normalize_sensor_type("temperature") == "temperature"

    def test_tmp_returns_temperature(self):
        assert _make_repo()._normalize_sensor_type("tmp") == "temperature"

    def test_temperature_case_insensitive(self):
        assert _make_repo()._normalize_sensor_type("TEMPERATURE") == "temperature"
        assert _make_repo()._normalize_sensor_type("Temp_Node") == "temperature"

    def test_humidity_returns_humidity(self):
        assert _make_repo()._normalize_sensor_type("humidity") == "humidity"

    def test_rh_returns_humidity(self):
        assert _make_repo()._normalize_sensor_type("rh") == "humidity"

    def test_co2_returns_co2(self):
        assert _make_repo()._normalize_sensor_type("co2") == "co2"

    def test_co2_case_insensitive(self):
        assert _make_repo()._normalize_sensor_type("CO2") == "co2"

    def test_carbon_dioxide_returns_co2(self):
        assert _make_repo()._normalize_sensor_type("carbon dioxide") == "co2"

    def test_moisture_returns_moisture(self):
        assert _make_repo()._normalize_sensor_type("moisture") == "moisture"

    def test_mc_returns_moisture(self):
        assert _make_repo()._normalize_sensor_type("mc") == "moisture"

    def test_strain_returns_strain(self):
        assert _make_repo()._normalize_sensor_type("strain") == "strain"

    def test_equation_returns_strain(self):
        assert _make_repo()._normalize_sensor_type("equation") == "strain"

    def test_quadratic_returns_strain(self):
        assert _make_repo()._normalize_sensor_type("quadratic") == "strain"

    def test_unknown_string_returns_none(self):
        assert _make_repo()._normalize_sensor_type("unknown") is None

    def test_completely_unrecognised_returns_none(self):
        assert _make_repo()._normalize_sensor_type("xyz_widget") is None


# ==========================================================================
# Repository — _find_sensor
# ==========================================================================

class TestFindSensor:

    def test_finds_matching_sensor_by_human_readable_location(self):
        sensor = _make_sensor(sensor_type="temperature", location="15_9279")
        repo = _make_repo(api=_make_mock_api(sensors=[sensor]))
        repo.connect()
        result = repo._find_sensor("temperature", "Node 15")
        assert result is not None
        assert result.sensor_id == sensor.sensor_id

    def test_returns_none_for_wrong_sensor_type(self):
        sensor = _make_sensor(sensor_type="humidity", location="15_9279")
        repo = _make_repo(api=_make_mock_api(sensors=[sensor]))
        repo.connect()
        assert repo._find_sensor("temperature", "Node 15") is None

    def test_returns_none_for_wrong_location(self):
        sensor = _make_sensor(sensor_type="temperature", location="15_9279")
        repo = _make_repo(api=_make_mock_api(sensors=[sensor]))
        repo.connect()
        assert repo._find_sensor("temperature", "Node 99") is None

    def test_location_match_case_insensitive(self):
        sensor = _make_sensor(sensor_type="temperature", location="15_9279")
        repo = _make_repo(api=_make_mock_api(sensors=[sensor]))
        repo.connect()
        # "node 15" lowercase should still find "Node 15"
        assert repo._find_sensor("temperature", "node 15") is not None

    def test_raw_location_fallback_match(self):
        """If sensor.location is already 'Node 15', raw match should work."""
        sensor = _make_sensor(sensor_type="temperature", location="Node 15")
        repo = _make_repo(api=_make_mock_api(sensors=[sensor]))
        repo.connect()
        assert repo._find_sensor("temperature", "Node 15") is not None

    def test_returns_none_when_sensor_list_empty(self):
        repo = _make_repo(api=_make_mock_api(sensors=[]))
        repo.connect()
        assert repo._find_sensor("temperature", "Node 15") is None


# ==========================================================================
# Repository — get_available_sensors
# ==========================================================================

class TestGetAvailableSensors:

    def test_returns_list(self):
        assert isinstance(_make_repo().get_available_sensors(), list)

    def test_temperature_sensor_included(self):
        assert "temperature" in _make_repo().get_available_sensors()

    def test_unknown_type_excluded(self):
        sensor = _make_sensor(sensor_type="unknown")
        repo = _make_repo(api=_make_mock_api(sensors=[sensor]))
        assert repo.get_available_sensors() == []

    def test_unrecognised_type_excluded(self):
        sensor = _make_sensor(sensor_type="xyz_widget")
        repo = _make_repo(api=_make_mock_api(sensors=[sensor]))
        assert repo.get_available_sensors() == []

    def test_deduplicates_sensor_types(self):
        sensors = [
            _make_sensor(sensor_id=1, sensor_type="temperature"),
            _make_sensor(sensor_id=2, sensor_type="temperature"),
            _make_sensor(sensor_id=3, sensor_type="humidity"),
        ]
        result = _make_repo(api=_make_mock_api(sensors=sensors)).get_available_sensors()
        assert result.count("temperature") == 1

    def test_result_is_sorted(self):
        sensors = [
            _make_sensor(sensor_id=1, sensor_type="temperature"),
            _make_sensor(sensor_id=2, sensor_type="co2"),
            _make_sensor(sensor_id=3, sensor_type="humidity"),
        ]
        result = _make_repo(api=_make_mock_api(sensors=sensors)).get_available_sensors()
        assert result == sorted(result)

    def test_multiple_sensor_types_all_returned(self):
        sensors = [
            _make_sensor(sensor_id=1, sensor_type="temperature"),
            _make_sensor(sensor_id=2, sensor_type="humidity"),
            _make_sensor(sensor_id=3, sensor_type="co2"),
        ]
        result = _make_repo(api=_make_mock_api(sensors=sensors)).get_available_sensors()
        assert "temperature" in result
        assert "humidity" in result
        assert "co2" in result


# ==========================================================================
# Repository — get_available_locations
# ==========================================================================

class TestGetAvailableLocations:

    def test_returns_list(self):
        assert isinstance(_make_repo().get_available_locations(), list)

    def test_underscore_format_converted_to_human_readable(self):
        node = _make_node(name="15_9279", location="15_9279")
        result = _make_repo(api=_make_mock_api(nodes=[node])).get_available_locations()
        assert "Node 15" in result

    def test_result_is_sorted(self):
        nodes = [
            _make_node(node_id=1, name="16_001", location="16_001"),
            _make_node(node_id=2, name="14_002", location="14_002"),
            _make_node(node_id=3, name="15_003", location="15_003"),
        ]
        result = _make_repo(api=_make_mock_api(nodes=nodes)).get_available_locations()
        assert result == sorted(result)

    def test_deduplicates_locations(self):
        """Two nodes that both map to 'Node 15' should appear only once."""
        nodes = [
            _make_node(node_id=1, name="15_001", location="15_001"),
            _make_node(node_id=2, name="15_002", location="15_002"),
        ]
        result = _make_repo(api=_make_mock_api(nodes=nodes)).get_available_locations()
        assert result.count("Node 15") == 1

    def test_node_with_empty_location_excluded(self):
        # The repository filters on `if node.location`
        node = NodeMetadata(node_id=1, physical_id=1, name="test", location="")
        result = _make_repo(api=_make_mock_api(nodes=[node])).get_available_locations()
        assert result == []


# ==========================================================================
# Repository — get_sensors_by_node
# ==========================================================================

class TestGetSensorsByNode:

    def test_returns_dict(self):
        assert isinstance(_make_repo().get_sensors_by_node(), dict)

    def test_key_is_human_readable_location(self):
        node = _make_node(name="15_9279", location="15_9279")
        sensor = _make_sensor(sensor_type="temperature", location="15_9279")
        result = _make_repo(api=_make_mock_api(nodes=[node], sensors=[sensor])).get_sensors_by_node()
        assert "Node 15" in result

    def test_values_are_sorted_lists(self):
        node = _make_node(name="15_9279", location="15_9279")
        sensors = [
            _make_sensor(sensor_id=1, sensor_type="temperature", location="15_9279"),
            _make_sensor(sensor_id=2, sensor_type="co2",         location="15_9279"),
            _make_sensor(sensor_id=3, sensor_type="humidity",    location="15_9279"),
        ]
        result = _make_repo(api=_make_mock_api(nodes=[node], sensors=sensors)).get_sensors_by_node()
        types = result["Node 15"]
        assert types == sorted(types)

    def test_unknown_sensor_types_excluded_from_values(self):
        node = _make_node(name="15_9279", location="15_9279")
        sensor = _make_sensor(sensor_type="unknown", location="15_9279")
        result = _make_repo(api=_make_mock_api(nodes=[node], sensors=[sensor])).get_sensors_by_node()
        assert result.get("Node 15", []) == []

    def test_each_type_appears_once_per_node(self):
        node = _make_node(name="15_9279", location="15_9279")
        sensors = [
            _make_sensor(sensor_id=1, sensor_type="temperature", location="15_9279"),
            _make_sensor(sensor_id=2, sensor_type="temperature", location="15_9279"),
        ]
        result = _make_repo(api=_make_mock_api(nodes=[node], sensors=sensors)).get_sensors_by_node()
        assert result["Node 15"].count("temperature") == 1


# ==========================================================================
# Repository — get_time_range
# ==========================================================================

class TestGetTimeRange:

    def test_returns_tuple_of_two_datetimes(self):
        result = _make_repo().get_time_range()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, datetime) for x in result)

    def test_start_before_end(self):
        start, end = _make_repo().get_time_range()
        assert start < end

    def test_earliest_is_2019_05_01(self):
        start, _ = _make_repo().get_time_range()
        assert start.year == 2019
        assert start.month == 5
        assert start.day == 1

    def test_latest_is_close_to_now(self):
        _, end = _make_repo().get_time_range()
        now = datetime.now(timezone.utc)
        assert abs((end - now).total_seconds()) < 5


# ==========================================================================
# Repository — validate_parameters
# ==========================================================================

class TestValidateParameters:

    def _valid_time_range(self):
        return (_dt(2025, 1, 14), _dt(2025, 1, 15))

    def test_valid_params_returns_empty_list(self):
        errors = _make_repo().validate_parameters(
            sensor_type="temperature",
            location="Node 15",
            time_range=self._valid_time_range(),
        )
        assert errors == []

    def test_invalid_sensor_type_returns_error(self):
        errors = _make_repo().validate_parameters(
            sensor_type="invalid_sensor",
            location="Node 15",
            time_range=self._valid_time_range(),
        )
        assert any("invalid_sensor" in e for e in errors)

    def test_invalid_location_returns_error(self):
        errors = _make_repo().validate_parameters(
            sensor_type="temperature",
            location="Node 999",
            time_range=self._valid_time_range(),
        )
        assert any("Node 999" in e for e in errors)

    def test_both_invalid_returns_two_errors(self):
        errors = _make_repo().validate_parameters(
            sensor_type="bad_sensor",
            location="Node 999",
            time_range=self._valid_time_range(),
        )
        assert len(errors) >= 2

    def test_start_before_2019_returns_error(self):
        errors = _make_repo().validate_parameters(
            sensor_type="temperature",
            location="Node 15",
            time_range=(
                datetime(2000, 1, 1, tzinfo=timezone.utc),
                datetime(2000, 1, 2, tzinfo=timezone.utc),
            ),
        )
        assert any("before available data" in e for e in errors)

    def test_error_message_exact_format_for_bad_sensor(self):
        """Error must say 'Unknown sensor type 'X''."""
        errors = _make_repo().validate_parameters(
            sensor_type="bad_sensor",
            location="Node 15",
            time_range=self._valid_time_range(),
        )
        assert any("Unknown sensor type 'bad_sensor'" in e for e in errors)

    def test_error_message_exact_format_for_bad_location(self):
        """Error must say 'Unknown location 'X''."""
        errors = _make_repo().validate_parameters(
            sensor_type="temperature",
            location="Node 999",
            time_range=self._valid_time_range(),
        )
        assert any("Unknown location 'Node 999'" in e for e in errors)


# ==========================================================================
# Repository — get_readings
# ==========================================================================

class TestGetReadings:

    def _call(self, repo=None, sensor_type="temperature", location="Node 15"):
        if repo is None:
            repo = _make_repo()
        return repo.get_readings(
            sensor_type=sensor_type,
            location=location,
            start_time=_dt(day=14),
            end_time=_dt(day=15),
        )

    def test_returns_dataframe(self):
        assert isinstance(self._call(), pd.DataFrame)

    def test_required_columns_present(self):
        df = self._call()
        for col in ("timestamp", "value", "unit", "location", "quality_flag"):
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_matches_api_readings(self):
        assert len(self._call()) == 24

    def test_location_column_equals_requested_location(self):
        df = self._call()
        assert (df["location"] == "Node 15").all()

    def test_unit_overridden_from_sensor_metadata(self):
        """Repository must replace the 'unknown' unit from api readings
        with the unit stored on the SensorMetadata object."""
        sensor = _make_sensor(unit="°F")
        api = _make_mock_api(sensors=[sensor])
        df = self._call(repo=_make_repo(api=api))
        assert (df["unit"] == "°F").all()

    def test_calls_api_get_sensor_data(self):
        api = _make_mock_api()
        self._call(repo=_make_repo(api=api))
        api.get_sensor_data.assert_called_once()

    def test_passes_correct_date_range_to_api(self):
        api = _make_mock_api()
        start = _dt(day=14)
        end = _dt(day=15)
        _make_repo(api=api).get_readings("temperature", "Node 15", start, end)
        call_kw = api.get_sensor_data.call_args[1]
        assert call_kw["start_date"] == start
        assert call_kw["end_date"] == end

    def test_no_sensor_found_raises_repository_error(self):
        repo = _make_repo(api=_make_mock_api(sensors=[]))
        with pytest.raises(RepositoryError):
            self._call(repo=repo)

    def test_api_error_wrapped_as_repository_error(self):
        api = _make_mock_api()
        api.get_sensor_data.side_effect = SMTAPIError("timeout")
        with pytest.raises(RepositoryError, match="API error"):
            self._call(repo=_make_repo(api=api))


# ==========================================================================
# Repository — get_readings_multiple_locations
# ==========================================================================

class TestGetReadingsMultipleLocations:

    def _two_location_repo(self):
        sensors = [
            _make_sensor(sensor_id=1, sensor_type="temperature", location="15_9279"),
            _make_sensor(sensor_id=2, sensor_type="temperature", location="14_1234"),
        ]
        nodes = [
            _make_node(node_id=9279, name="15_9279", location="15_9279"),
            _make_node(node_id=1234, name="14_1234", location="14_1234"),
        ]
        return _make_repo(api=_make_mock_api(nodes=nodes, sensors=sensors))


    def test_raises_repository_error_when_no_location_has_data(self):
        repo = _make_repo(api=_make_mock_api(sensors=[]))
        with pytest.raises(RepositoryError, match="No data found for any location"):
            repo.get_readings_multiple_locations(
                sensor_type="temperature",
                locations=["Node 99", "Node 100"],
                start_time=_dt(day=14),
                end_time=_dt(day=15),
            )

    def test_partial_failure_uses_available_data(self):
        """If only one location fails, the other's data should still be returned."""
        sensor1 = _make_sensor(sensor_id=1, sensor_type="temperature", location="15_9279")
        node1 = _make_node(node_id=9279, name="15_9279", location="15_9279")
        api = _make_mock_api(nodes=[node1], sensors=[sensor1])
        # Node 99 won't be found, but Node 15 should succeed
        repo = _make_repo(api=api)
        df = repo.get_readings_multiple_locations(
            sensor_type="temperature",
            locations=["Node 15", "Node 99"],
            start_time=_dt(day=14),
            end_time=_dt(day=15),
        )
        assert len(df) == 24  # only Node 15's readings


# ==========================================================================
# LLMDataBridge
# ==========================================================================

def _make_task_spec(intent="query", location="Node 15", operation="mean",
                     aggregation_level=None):
    from llm import TaskSpecification, AggregationLevel
    if intent == "comparison" and isinstance(location, str):
        location = [location, "Node 16"]
    return TaskSpecification(
        intent_type=intent,
        sensor_type="temperature",
        location=location,
        start_time=_dt(day=14),
        end_time=_dt(day=15),
        operation=operation,
        aggregation_level=aggregation_level,
    )


def _empty_df():
    return pd.DataFrame(
        columns=["timestamp", "value", "unit", "location", "quality_flag"]
    )


class TestLLMDataBridge:

    def _bridge(self, **repo_return_overrides):
        repo = Mock(spec=SensorDataRepository)
        repo.get_readings.return_value = _empty_df()
        repo.get_readings_multiple_locations.return_value = _empty_df()
        repo.get_available_sensors.return_value = ["temperature"]
        repo.get_available_locations.return_value = ["Node 15"]
        repo.get_time_range.return_value = (_dt(2025, 1, 1), _dt(2025, 12, 31))
        repo.validate_parameters.return_value = []
        repo.connect.return_value = None
        for k, v in repo_return_overrides.items():
            setattr(repo, k, Mock(return_value=v))
        return LLMDataBridge(repo), repo

    # --- execute_task routing ---

    def test_query_intent_calls_get_readings(self):
        bridge, repo = self._bridge()
        bridge.execute_task(_make_task_spec(intent="query"))
        repo.get_readings.assert_called_once()

    def test_comparison_intent_calls_get_readings_multiple_locations(self):
        bridge, repo = self._bridge()
        bridge.execute_task(_make_task_spec(intent="comparison"))
        repo.get_readings_multiple_locations.assert_called_once()

    def test_aggregation_intent_calls_get_readings(self):
        bridge, repo = self._bridge()
        from llm import TaskSpecification
        spec = TaskSpecification(
            intent_type="aggregation",
            sensor_type="temperature",
            location="Node 15",
            start_time=_dt(day=14),
            end_time=_dt(day=15),
            operation="mean",
            aggregation_level="daily",
        )
        bridge.execute_task(spec)
        repo.get_readings.assert_called_once()

    def test_unknown_intent_raises_value_error(self):
        bridge, _ = self._bridge()
        spec = Mock()
        spec.intent_type = "completely_unknown"
        with pytest.raises((ValueError, AttributeError)):
            bridge.execute_task(spec)

    def test_query_passes_sensor_type_and_location(self):
        bridge, repo = self._bridge()
        bridge.execute_task(_make_task_spec(intent="query", location="Node 15"))
        call_kw = repo.get_readings.call_args[1]
        assert call_kw["sensor_type"] == "temperature"
        assert call_kw["location"] == "Node 15"

    def test_comparison_passes_list_of_locations(self):
        bridge, repo = self._bridge()
        bridge.execute_task(_make_task_spec(intent="comparison"))
        call_kw = repo.get_readings_multiple_locations.call_args[1]
        assert isinstance(call_kw["locations"], list)
        assert len(call_kw["locations"]) == 2

    # --- get_system_context ---

    def test_get_system_context_returns_required_keys(self):
        bridge, _ = self._bridge()
        ctx = bridge.get_system_context()
        assert "available_sensors" in ctx
        assert "available_locations" in ctx
        assert "time_range" in ctx

    def test_get_system_context_calls_repository_connect(self):
        bridge, repo = self._bridge()
        bridge.get_system_context()
        repo.connect.assert_called_once()

    def test_get_system_context_values_match_repository(self):
        bridge, _ = self._bridge()
        ctx = bridge.get_system_context()
        assert ctx["available_sensors"] == ["temperature"]
        assert ctx["available_locations"] == ["Node 15"]

    # --- validate_task ---

    def test_valid_spec_returns_empty_errors(self):
        bridge, _ = self._bridge()
        errors = bridge.validate_task(_make_task_spec())
        assert errors == []

    def test_invalid_spec_returns_errors_from_repository(self):
        bridge, repo = self._bridge()
        repo.validate_parameters.return_value = ["Unknown location 'Node 999'"]
        errors = bridge.validate_task(_make_task_spec())
        assert len(errors) > 0
        assert "Unknown location 'Node 999'" in errors

    def test_single_location_query_calls_validate_parameters_once(self):
        bridge, repo = self._bridge()
        bridge.validate_task(_make_task_spec(intent="query"))
        repo.validate_parameters.assert_called_once()

    def test_comparison_calls_validate_parameters_once_per_location(self):
        """Comparison has 2 locations → validate_parameters called twice."""
        bridge, repo = self._bridge()
        bridge.validate_task(_make_task_spec(intent="comparison"))
        assert repo.validate_parameters.call_count == 2

    def test_validate_task_passes_time_range_as_tuple(self):
        bridge, repo = self._bridge()
        bridge.validate_task(_make_task_spec())
        call_kw = repo.validate_parameters.call_args[1]
        time_range = call_kw["time_range"]
        assert isinstance(time_range, tuple)
        assert len(time_range) == 2

    def test_errors_from_multiple_locations_accumulated(self):
        """Both locations in a comparison failing → errors from both are returned."""
        bridge, repo = self._bridge()
        repo.validate_parameters.return_value = ["bad location"]
        errors = bridge.validate_task(_make_task_spec(intent="comparison"))
        # Two locations × one error each = 2 errors total
        assert len(errors) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])