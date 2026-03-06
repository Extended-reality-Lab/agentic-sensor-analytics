"""
Unit tests for the agent layer.
Tests each AgentNodes method and the three routing functions in isolation
using mocked dependencies — no live LLM, API, or LangGraph execution needed.

Matches the actual implementation in nodes.py and graph.py exactly:
  - AgentNodes.__init__ takes (llm, repository, bridge) — registry is built internally
  - generate_explanation reads use_streaming from state, NOT from its own parameter
  - validate_task calls TaskSpecificationParser.validate_against_context via context_dict
    fetched from bridge.get_system_context()
  - Routing functions return node name strings: "retrieve_data"/"handle_error", etc.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime, timedelta, timezone
from types import GeneratorType

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.state import AgentState, create_initial_state, get_execution_summary, add_trace_entry
from agent.graph import should_continue, should_continue_after_retrieval, should_continue_after_analytics
from agent.nodes import AgentNodes
from llm import TaskSpecification, IntentType, Operation, AggregationLevel, LLMError, LLMGenerationError


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_task_spec(
    intent="query",
    sensor="temperature",
    location="Node 15",
    operation="mean",
    aggregation_level=None,
    confidence=0.95,
):
    """Build a TaskSpecification with sensible defaults."""
    if intent == "comparison" and isinstance(location, str):
        location = [location, "Node 16"]
    return TaskSpecification(
        intent_type=intent,
        sensor_type=sensor,
        location=location,
        start_time=datetime(2025, 1, 14, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 15, tzinfo=timezone.utc),
        operation=operation,
        aggregation_level=aggregation_level,
        confidence=confidence,
    )


def _make_df(rows=24):
    """Return a minimal DataFrame matching the repository output schema."""
    start = datetime(2025, 1, 14, tzinfo=timezone.utc)
    return pd.DataFrame({
        "timestamp": [start + timedelta(hours=i) for i in range(rows)],
        "value": [20.0 + i * 0.5 for i in range(rows)],
        "unit": "°C",
        "location": "Node 15",
        "quality_flag": 0,
    })


def _make_analytics_result_dict(value=22.5, success=True):
    """Return a dict matching result.model_dump() output stored in state."""
    return {
        "value": value,
        "unit": "°C",
        "metadata": {"operation": "mean", "sample_size": 24},
        "success": success,
        "error_message": None if success else "Tool failed",
        "execution_time_ms": 10.0,
    }


_SYSTEM_CONTEXT = {
    "available_sensors": ["temperature", "humidity", "co2"],
    "available_locations": ["Node 14", "Node 15", "Node 16"],
    "time_range": (
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 12, 31, tzinfo=timezone.utc),
    ),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.extract_intent.return_value = _make_task_spec()
    llm.explain_results.return_value = "The average temperature in Node 15 was 22.5°C."
    llm.explain_error.return_value = "The location you specified was not found."
    return llm


@pytest.fixture
def mock_bridge():
    bridge = Mock()
    bridge.get_system_context.return_value = _SYSTEM_CONTEXT
    bridge.execute_task.return_value = _make_df()
    return bridge


@pytest.fixture
def mock_repository():
    return Mock()


@pytest.fixture
def mock_tool():
    """A mock analytics tool whose execute() returns a successful result."""
    tool = Mock()
    tool.name = "temporal_mean"

    # Build a mock AnalyticsResult whose model_dump() returns a plain dict
    result = Mock()
    result.success = True
    result.value = 22.5
    result.unit = "°C"
    result.error_message = None
    result.execution_time_ms = 10.0
    result.model_dump.return_value = _make_analytics_result_dict()
    tool.execute.return_value = result
    return tool


@pytest.fixture
def mock_registry(mock_tool):
    registry = Mock()
    registry.get_tool.return_value = mock_tool
    return registry


@pytest.fixture
def nodes(mock_llm, mock_repository, mock_bridge, mock_registry):
    """AgentNodes with all external dependencies mocked."""
    with patch("agent.nodes.get_registry", return_value=mock_registry):
        n = AgentNodes(
            llm=mock_llm,
            repository=mock_repository,
            bridge=mock_bridge,
        )
    return n


# ---------------------------------------------------------------------------
# create_initial_state
# ---------------------------------------------------------------------------

class TestCreateInitialState:

    def test_user_query_is_set(self):
        state = create_initial_state("What is the temperature?")
        assert state["user_query"] == "What is the temperature?"

    def test_nullable_fields_are_none(self):
        state = create_initial_state("q")
        assert state["task_spec"] is None
        assert state["data"] is None
        assert state["analytics_result"] is None
        assert state["explanation"] is None
        assert state["explanation_stream"] is None
        assert state["error_explanation"] is None
        assert state["end_time"] is None

    def test_list_fields_are_empty(self):
        state = create_initial_state("q")
        assert state["validation_errors"] == []
        assert state["execution_trace"] == []

    def test_success_starts_false(self):
        state = create_initial_state("q")
        assert state["success"] is False

    def test_start_time_is_set(self):
        before = datetime.now()
        state = create_initial_state("q")
        after = datetime.now()
        assert before <= state["start_time"] <= after

    def test_use_streaming_default_is_false(self):
        """Default must be False so explanation is always a string in tests."""
        state = create_initial_state("q")
        assert state["use_streaming"] is False

    def test_use_streaming_can_be_set_true(self):
        state = create_initial_state("q", use_streaming=True)
        assert state["use_streaming"] is True


# ---------------------------------------------------------------------------
# add_trace_entry
# ---------------------------------------------------------------------------

class TestAddTraceEntry:

    def test_entry_appended_to_trace(self):
        state = create_initial_state("q")
        add_trace_entry(state, "my_step", "completed", {"key": "val"}, 42.0)
        assert len(state["execution_trace"]) == 1

    def test_entry_fields(self):
        state = create_initial_state("q")
        add_trace_entry(state, "my_step", "completed", {"key": "val"}, 42.0)
        entry = state["execution_trace"][0]
        assert entry["step"] == "my_step"
        assert entry["status"] == "completed"
        assert entry["details"] == {"key": "val"}
        assert entry["duration_ms"] == 42.0
        assert isinstance(entry["timestamp"], datetime)

    def test_multiple_entries_accumulate(self):
        state = create_initial_state("q")
        add_trace_entry(state, "step_a", "started", {})
        add_trace_entry(state, "step_a", "completed", {}, 10.0)
        assert len(state["execution_trace"]) == 2

    def test_missing_trace_key_is_initialised(self):
        """If execution_trace is absent, add_trace_entry should create it."""
        state = create_initial_state("q")
        del state["execution_trace"]
        add_trace_entry(state, "step", "started", {})
        assert len(state["execution_trace"]) == 1


# ---------------------------------------------------------------------------
# Node 1: interpret_query
# ---------------------------------------------------------------------------

class TestInterpretQueryNode:

    def test_task_spec_set_on_success(self, nodes, mock_llm):
        state = create_initial_state("What is the average temperature in Node 15 yesterday?")
        result = nodes.interpret_query(state)
        assert result["task_spec"] is not None
        assert result["task_spec"].sensor_type == "temperature"

    def test_calls_bridge_get_system_context(self, nodes, mock_bridge):
        state = create_initial_state("q")
        nodes.interpret_query(state)
        mock_bridge.get_system_context.assert_called_once()

    def test_calls_llm_extract_intent(self, nodes, mock_llm):
        state = create_initial_state("q")
        nodes.interpret_query(state)
        mock_llm.extract_intent.assert_called_once()

    def test_selected_node_passed_to_extract_intent(self, nodes, mock_llm):
        state = create_initial_state("q")
        state["selected_node"] = "Node 15"
        nodes.interpret_query(state)
        call_args = mock_llm.extract_intent.call_args
        # third positional arg or 'selected_node' kwarg
        passed_node = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("selected_node")
        assert passed_node == "Node 15"

    def test_trace_contains_interpret_query_entries(self, nodes):
        state = create_initial_state("q")
        result = nodes.interpret_query(state)
        steps = [e["step"] for e in result["execution_trace"]]
        assert "interpret_query" in steps

    def test_trace_has_started_and_completed(self, nodes):
        state = create_initial_state("q")
        result = nodes.interpret_query(state)
        statuses = [e["status"] for e in result["execution_trace"]
                    if e["step"] == "interpret_query"]
        assert "started" in statuses
        assert "completed" in statuses

    def test_completed_trace_contains_intent_details(self, nodes):
        state = create_initial_state("q")
        result = nodes.interpret_query(state)
        completed = [e for e in result["execution_trace"]
                     if e["step"] == "interpret_query" and e["status"] == "completed"]
        assert len(completed) == 1
        details = completed[0]["details"]
        assert "intent_type" in details
        assert "sensor_type" in details
        assert "confidence" in details

    def test_llm_error_sets_validation_errors(self, nodes, mock_llm):
        mock_llm.extract_intent.side_effect = LLMError("model offline")
        state = create_initial_state("q")
        result = nodes.interpret_query(state)
        assert len(result["validation_errors"]) > 0
        assert any("Failed to understand query" in e for e in result["validation_errors"])

    def test_llm_error_sets_failed_trace(self, nodes, mock_llm):
        mock_llm.extract_intent.side_effect = LLMError("failure")
        state = create_initial_state("q")
        result = nodes.interpret_query(state)
        failed = [e for e in result["execution_trace"]
                  if e["step"] == "interpret_query" and e["status"] == "failed"]
        assert len(failed) == 1

    def test_generic_exception_sets_validation_errors(self, nodes, mock_llm):
        mock_llm.extract_intent.side_effect = RuntimeError("unexpected")
        state = create_initial_state("q")
        result = nodes.interpret_query(state)
        assert len(result["validation_errors"]) > 0
        assert any("Unexpected error" in e for e in result["validation_errors"])

    def test_success_does_not_set_validation_errors(self, nodes):
        state = create_initial_state("q")
        result = nodes.interpret_query(state)
        assert result["validation_errors"] == []


# ---------------------------------------------------------------------------
# Node 2: validate_task
# ---------------------------------------------------------------------------

class TestValidateTaskNode:

    def test_valid_spec_produces_no_errors(self, nodes, mock_bridge):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()  # Node 15 is in _SYSTEM_CONTEXT
        result = nodes.validate_task(state)
        assert result["validation_errors"] == []

    def test_none_task_spec_produces_error(self, nodes):
        state = create_initial_state("q")
        state["task_spec"] = None
        result = nodes.validate_task(state)
        assert len(result["validation_errors"]) > 0

    def test_invalid_location_produces_error(self, nodes, mock_bridge):
        state = create_initial_state("q")
        # Node 999 is not in _SYSTEM_CONTEXT available_locations
        state["task_spec"] = _make_task_spec(location="Node 999")
        result = nodes.validate_task(state)
        assert len(result["validation_errors"]) > 0
        assert any("Node 999" in e for e in result["validation_errors"])

    def test_invalid_sensor_produces_error(self, nodes, mock_bridge):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec(sensor="invalid_sensor")
        result = nodes.validate_task(state)
        assert len(result["validation_errors"]) > 0
        assert any("invalid_sensor" in e for e in result["validation_errors"])

    def test_calls_bridge_get_system_context(self, nodes, mock_bridge):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        nodes.validate_task(state)
        mock_bridge.get_system_context.assert_called()

    def test_trace_has_started_entry(self, nodes):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        result = nodes.validate_task(state)
        steps_statuses = [(e["step"], e["status"]) for e in result["execution_trace"]]
        assert ("validate_task", "started") in steps_statuses

    def test_trace_completed_on_valid_spec(self, nodes):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        result = nodes.validate_task(state)
        steps_statuses = [(e["step"], e["status"]) for e in result["execution_trace"]]
        assert ("validate_task", "completed") in steps_statuses

    def test_trace_failed_on_invalid_spec(self, nodes):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec(location="Node 999")
        result = nodes.validate_task(state)
        steps_statuses = [(e["step"], e["status"]) for e in result["execution_trace"]]
        assert ("validate_task", "failed") in steps_statuses

    def test_time_before_available_data_produces_error(self, nodes, mock_bridge):
        """start_time before the system's earliest available data should fail."""
        spec = TaskSpecification(
            intent_type="query",
            sensor_type="temperature",
            location="Node 15",
            start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2020, 1, 2, tzinfo=timezone.utc),
            operation="mean",
        )
        state = create_initial_state("q")
        state["task_spec"] = spec
        result = nodes.validate_task(state)
        assert len(result["validation_errors"]) > 0


# ---------------------------------------------------------------------------
# Node 3: retrieve_data
# ---------------------------------------------------------------------------

class TestRetrieveDataNode:

    def test_data_set_on_success(self, nodes, mock_bridge):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        result = nodes.retrieve_data(state)
        assert result["data"] is not None
        assert isinstance(result["data"], pd.DataFrame)
        assert len(result["data"]) == 24

    def test_calls_bridge_execute_task(self, nodes, mock_bridge):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        nodes.retrieve_data(state)
        mock_bridge.execute_task.assert_called_once()

    def test_empty_dataframe_adds_validation_error(self, nodes, mock_bridge):
        mock_bridge.execute_task.return_value = pd.DataFrame()
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        result = nodes.retrieve_data(state)
        assert len(result["validation_errors"]) > 0

    def test_empty_dataframe_does_not_set_data(self, nodes, mock_bridge):
        mock_bridge.execute_task.return_value = pd.DataFrame()
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        result = nodes.retrieve_data(state)
        assert result.get("data") is None

    def test_trace_completed_entry_has_rows_retrieved(self, nodes):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        result = nodes.retrieve_data(state)
        completed = [e for e in result["execution_trace"]
                     if e["step"] == "retrieve_data" and e["status"] == "completed"]
        assert len(completed) == 1
        assert "rows_retrieved" in completed[0]["details"]
        assert completed[0]["details"]["rows_retrieved"] == 24

    def test_repository_error_sets_validation_error(self, nodes, mock_bridge):
        from data import RepositoryError
        mock_bridge.execute_task.side_effect = RepositoryError("API down")
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        result = nodes.retrieve_data(state)
        assert len(result["validation_errors"]) > 0
        assert any("Data retrieval failed" in e for e in result["validation_errors"])

    def test_generic_exception_sets_validation_error(self, nodes, mock_bridge):
        mock_bridge.execute_task.side_effect = RuntimeError("network error")
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        result = nodes.retrieve_data(state)
        assert len(result["validation_errors"]) > 0
        assert any("Unexpected error" in e for e in result["validation_errors"])

    def test_trace_failed_on_empty_data(self, nodes, mock_bridge):
        mock_bridge.execute_task.return_value = pd.DataFrame()
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        result = nodes.retrieve_data(state)
        failed = [e for e in result["execution_trace"]
                  if e["step"] == "retrieve_data" and e["status"] == "failed"]
        assert len(failed) == 1


# ---------------------------------------------------------------------------
# Node 4: execute_analytics — tool selection logic
# ---------------------------------------------------------------------------

class TestExecuteAnalyticsNode:

    def test_analytics_result_set_on_success(self, nodes):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec(intent="query", operation="mean")
        state["data"] = _make_df()
        result = nodes.execute_analytics(state)
        assert result["analytics_result"] is not None
        assert isinstance(result["analytics_result"], dict)

    def test_query_uses_temporal_mean_tool(self, nodes, mock_registry):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec(intent="query", operation="mean")
        state["data"] = _make_df()
        nodes.execute_analytics(state)
        mock_registry.get_tool.assert_called_with("temporal_mean")

    def test_aggregation_uses_temporal_aggregation_tool(self, nodes, mock_registry):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec(
            intent="aggregation", operation="mean", aggregation_level="daily"
        )
        state["data"] = _make_df()
        nodes.execute_analytics(state)
        mock_registry.get_tool.assert_called_with("temporal_aggregation")

    def test_comparison_uses_spatial_comparison_tool(self, nodes, mock_registry):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec(intent="comparison", operation="mean")
        state["data"] = _make_df()
        nodes.execute_analytics(state)
        mock_registry.get_tool.assert_called_with("spatial_comparison")

    def test_summary_operation_uses_statistical_summary_tool(self, nodes, mock_registry):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec(intent="query", operation="summary")
        state["data"] = _make_df()
        nodes.execute_analytics(state)
        mock_registry.get_tool.assert_called_with("statistical_summary")

    def test_aggregation_level_passed_to_tool(self, nodes, mock_registry, mock_tool):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec(
            intent="aggregation", operation="mean", aggregation_level="daily"
        )
        state["data"] = _make_df()
        nodes.execute_analytics(state)
        call_kwargs = mock_tool.execute.call_args[1]
        assert "aggregation_level" in call_kwargs
        assert call_kwargs["aggregation_level"].value == "daily"

    def test_failed_tool_result_adds_validation_error(self, nodes, mock_registry, mock_tool):
        mock_tool.execute.return_value.success = False
        mock_tool.execute.return_value.error_message = "column missing"
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        state["data"] = _make_df()
        result = nodes.execute_analytics(state)
        assert len(result["validation_errors"]) > 0

    def test_none_tool_raises_error(self, nodes, mock_registry):
        mock_registry.get_tool.return_value = None
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        state["data"] = _make_df()
        result = nodes.execute_analytics(state)
        assert len(result["validation_errors"]) > 0

    def test_no_data_adds_validation_error(self, nodes):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        state["data"] = None
        result = nodes.execute_analytics(state)
        assert len(result["validation_errors"]) > 0

    def test_trace_completed_contains_tool_name(self, nodes):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec()
        state["data"] = _make_df()
        result = nodes.execute_analytics(state)
        completed = [e for e in result["execution_trace"]
                     if e["step"] == "execute_analytics" and e["status"] == "completed"]
        assert len(completed) == 1
        assert "tool" in completed[0]["details"]

    def test_operation_kwarg_passed_to_tool(self, nodes, mock_tool):
        state = create_initial_state("q")
        state["task_spec"] = _make_task_spec(operation="max")
        state["data"] = _make_df()
        nodes.execute_analytics(state)
        call_kwargs = mock_tool.execute.call_args[1]
        assert call_kwargs.get("operation") == "max"


# ---------------------------------------------------------------------------
# Node 5: generate_explanation
# ---------------------------------------------------------------------------

class TestGenerateExplanationNode:

    def _state_ready_for_explanation(self, streaming=False):
        state = create_initial_state("q", use_streaming=streaming)
        state["task_spec"] = _make_task_spec()
        state["analytics_result"] = _make_analytics_result_dict()
        return state

    def test_explanation_set_when_not_streaming(self, nodes, mock_llm):
        state = self._state_ready_for_explanation(streaming=False)
        result = nodes.generate_explanation(state)
        assert result["explanation"] is not None
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0

    def test_success_true_when_not_streaming(self, nodes):
        state = self._state_ready_for_explanation(streaming=False)
        result = nodes.generate_explanation(state)
        assert result["success"] is True

    def test_end_time_set_when_not_streaming(self, nodes):
        state = self._state_ready_for_explanation(streaming=False)
        result = nodes.generate_explanation(state)
        assert result["end_time"] is not None

    def test_calls_llm_explain_results(self, nodes, mock_llm):
        state = self._state_ready_for_explanation(streaming=False)
        nodes.generate_explanation(state)
        mock_llm.explain_results.assert_called_once()

    def test_explain_results_called_with_stream_false(self, nodes, mock_llm):
        state = self._state_ready_for_explanation(streaming=False)
        nodes.generate_explanation(state)
        call_kwargs = mock_llm.explain_results.call_args[1]
        assert call_kwargs.get("stream") is False

    def test_streaming_stores_generator_not_string(self, nodes, mock_llm):
        """When use_streaming=True, explanation_stream is set, explanation is None."""
        def fake_stream(*args, **kwargs):
            yield "chunk1"
            yield "chunk2"
        mock_llm.explain_results.return_value = fake_stream()

        state = self._state_ready_for_explanation(streaming=True)
        result = nodes.generate_explanation(state)
        assert result.get("explanation") is None
        assert result["explanation_stream"] is not None

    def test_streaming_sets_success_true(self, nodes, mock_llm):
        def fake_stream(*args, **kwargs):
            yield "chunk"
        mock_llm.explain_results.return_value = fake_stream()
        state = self._state_ready_for_explanation(streaming=True)
        result = nodes.generate_explanation(state)
        assert result["success"] is True

    def test_llm_error_triggers_fallback_explanation(self, nodes, mock_llm):
        """LLMError should produce a fallback 'Result: X unit' explanation, not raise."""
        mock_llm.explain_results.side_effect = LLMError("offline")
        state = self._state_ready_for_explanation(streaming=False)
        result = nodes.generate_explanation(state)
        assert result["explanation"] is not None
        assert "22.5" in result["explanation"] or "Result" in result["explanation"]
        assert result["success"] is True

    def test_generic_exception_with_analytics_result_still_succeeds(self, nodes, mock_llm):
        mock_llm.explain_results.side_effect = RuntimeError("unexpected")
        state = self._state_ready_for_explanation(streaming=False)
        result = nodes.generate_explanation(state)
        assert result["success"] is True
        assert result["explanation"] is not None

    def test_generic_exception_without_analytics_result_fails(self, nodes, mock_llm):
        mock_llm.explain_results.side_effect = RuntimeError("unexpected")
        state = create_initial_state("q", use_streaming=False)
        state["task_spec"] = _make_task_spec()
        state["analytics_result"] = None  # no result
        result = nodes.generate_explanation(state)
        assert result["success"] is False

    def test_trace_completed_contains_explanation_length(self, nodes):
        state = self._state_ready_for_explanation(streaming=False)
        result = nodes.generate_explanation(state)
        completed = [e for e in result["execution_trace"]
                     if e["step"] == "generate_explanation" and e["status"] == "completed"]
        assert len(completed) == 1
        assert "explanation_length" in completed[0]["details"]

    def test_essential_metadata_keys_passed_to_llm(self, nodes, mock_llm):
        """Node should strip verbose keys like time_series before calling LLM."""
        analytics = _make_analytics_result_dict()
        analytics["metadata"]["time_series"] = [{"t": "x", "v": 1}] * 1000
        analytics["metadata"]["operation"] = "mean"
        analytics["metadata"]["sample_size"] = 24

        state = self._state_ready_for_explanation(streaming=False)
        state["analytics_result"] = analytics
        nodes.generate_explanation(state)

        passed_results = mock_llm.explain_results.call_args[0][2]
        assert isinstance(passed_results, list)
        result_summary = passed_results[0]
        assert "time_series" not in result_summary.get("metadata", {})
        assert "operation" in result_summary.get("metadata", {})


# ---------------------------------------------------------------------------
# Node 6: handle_error
# ---------------------------------------------------------------------------

class TestHandleErrorNode:

    def test_error_explanation_set(self, nodes, mock_llm):
        state = create_initial_state("q")
        state["validation_errors"] = ["Unknown location 'Node 999'"]
        result = nodes.handle_error(state)
        assert result["error_explanation"] is not None
        assert len(result["error_explanation"]) > 0

    def test_success_is_false(self, nodes):
        state = create_initial_state("q")
        state["validation_errors"] = ["some error"]
        result = nodes.handle_error(state)
        assert result["success"] is False

    def test_end_time_set(self, nodes):
        state = create_initial_state("q")
        state["validation_errors"] = ["some error"]
        result = nodes.handle_error(state)
        assert result["end_time"] is not None

    def test_calls_llm_explain_error(self, nodes, mock_llm):
        state = create_initial_state("q")
        state["validation_errors"] = ["some error"]
        nodes.handle_error(state)
        mock_llm.explain_error.assert_called_once()

    def test_errors_passed_to_llm(self, nodes, mock_llm):
        state = create_initial_state("q")
        state["validation_errors"] = ["Error A", "Error B"]
        nodes.handle_error(state)
        call_args = mock_llm.explain_error.call_args[0]
        passed_errors = call_args[1]
        assert "Error A" in passed_errors
        assert "Error B" in passed_errors

    def test_llm_error_fallback_uses_bullet_list(self, nodes, mock_llm):
        """If LLMError is raised, fallback should include the raw error text."""
        mock_llm.explain_error.side_effect = LLMError("offline")
        state = create_initial_state("q")
        state["validation_errors"] = ["Error A"]
        result = nodes.handle_error(state)
        assert result["error_explanation"] is not None
        assert "Error A" in result["error_explanation"]

    def test_empty_validation_errors_uses_fallback_message(self, nodes):
        """No validation errors should still produce a generic error explanation."""
        state = create_initial_state("q")
        state["validation_errors"] = []
        result = nodes.handle_error(state)
        assert result["error_explanation"] is not None

    def test_trace_has_started_and_completed(self, nodes):
        state = create_initial_state("q")
        state["validation_errors"] = ["error"]
        result = nodes.handle_error(state)
        steps_statuses = [(e["step"], e["status"]) for e in result["execution_trace"]]
        assert ("handle_error", "started") in steps_statuses
        assert ("handle_error", "completed") in steps_statuses

    def test_trace_completed_shows_errors_handled_count(self, nodes):
        state = create_initial_state("q")
        state["validation_errors"] = ["E1", "E2", "E3"]
        result = nodes.handle_error(state)
        completed = [e for e in result["execution_trace"]
                     if e["step"] == "handle_error" and e["status"] == "completed"]
        assert completed[0]["details"]["errors_handled"] == 3

    def test_absolute_fallback_on_unexpected_exception(self, nodes, mock_llm):
        """Even if explain_error raises a non-LLMError, node must not crash."""
        mock_llm.explain_error.side_effect = RuntimeError("total failure")
        state = create_initial_state("q")
        state["validation_errors"] = ["error"]
        result = nodes.handle_error(state)
        assert result["error_explanation"] is not None
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Routing functions — return actual node name strings
# ---------------------------------------------------------------------------

class TestShouldContinue:
    """Tests for should_continue (post validate_task routing)."""

    def test_no_errors_routes_to_retrieve_data(self):
        state = create_initial_state("q")
        state["validation_errors"] = []
        assert should_continue(state) == "retrieve_data"

    def test_with_errors_routes_to_handle_error(self):
        state = create_initial_state("q")
        state["validation_errors"] = ["bad location"]
        assert should_continue(state) == "handle_error"

    def test_multiple_errors_routes_to_handle_error(self):
        state = create_initial_state("q")
        state["validation_errors"] = ["err1", "err2"]
        assert should_continue(state) == "handle_error"


class TestShouldContinueAfterRetrieval:
    """Tests for should_continue_after_retrieval (post retrieve_data routing)."""

    def test_data_present_no_errors_routes_to_execute_analytics(self):
        state = create_initial_state("q")
        state["validation_errors"] = []
        state["data"] = _make_df()
        assert should_continue_after_retrieval(state) == "execute_analytics"

    def test_errors_routes_to_handle_error(self):
        state = create_initial_state("q")
        state["validation_errors"] = ["no data"]
        state["data"] = None
        assert should_continue_after_retrieval(state) == "handle_error"

    def test_no_data_routes_to_handle_error(self):
        state = create_initial_state("q")
        state["validation_errors"] = []
        state["data"] = None
        assert should_continue_after_retrieval(state) == "handle_error"

    def test_errors_with_data_still_routes_to_handle_error(self):
        """If validation_errors exist, should go to error even if data is present."""
        state = create_initial_state("q")
        state["validation_errors"] = ["something went wrong"]
        state["data"] = _make_df()
        assert should_continue_after_retrieval(state) == "handle_error"


class TestShouldContinueAfterAnalytics:
    """Tests for should_continue_after_analytics (post execute_analytics routing)."""

    def test_result_present_no_errors_routes_to_generate_explanation(self):
        state = create_initial_state("q")
        state["validation_errors"] = []
        state["analytics_result"] = _make_analytics_result_dict()
        assert should_continue_after_analytics(state) == "generate_explanation"

    def test_errors_routes_to_handle_error(self):
        state = create_initial_state("q")
        state["validation_errors"] = ["tool failed"]
        state["analytics_result"] = None
        assert should_continue_after_analytics(state) == "handle_error"

    def test_no_result_routes_to_handle_error(self):
        state = create_initial_state("q")
        state["validation_errors"] = []
        state["analytics_result"] = None
        assert should_continue_after_analytics(state) == "handle_error"

    def test_errors_with_result_still_routes_to_handle_error(self):
        state = create_initial_state("q")
        state["validation_errors"] = ["partial failure"]
        state["analytics_result"] = _make_analytics_result_dict()
        assert should_continue_after_analytics(state) == "handle_error"


# ---------------------------------------------------------------------------
# get_execution_summary
# ---------------------------------------------------------------------------

class TestGetExecutionSummary:

    def _state_with_trace(self, success=True):
        state = create_initial_state("q")
        state["success"] = success
        state["start_time"] = datetime(2025, 1, 1, 12, 0, 0)
        state["end_time"] = datetime(2025, 1, 1, 12, 0, 5)
        state["execution_trace"] = [
            {
                "step": "interpret_query",
                "timestamp": datetime(2025, 1, 1, 12, 0, 0),
                "status": "started",
                "details": {},
                "duration_ms": None,
            },
            {
                "step": "interpret_query",
                "timestamp": datetime(2025, 1, 1, 12, 0, 1),
                "status": "completed",
                "details": {},
                "duration_ms": 200.0,
            },
            {
                "step": "validate_task",
                "timestamp": datetime(2025, 1, 1, 12, 0, 2),
                "status": "started",
                "details": {},
                "duration_ms": None,
            },
            {
                "step": "validate_task",
                "timestamp": datetime(2025, 1, 1, 12, 0, 3),
                "status": "completed",
                "details": {},
                "duration_ms": 50.0,
            },
        ]
        return state

    def test_returns_required_fields(self):
        state = self._state_with_trace()
        summary = get_execution_summary(state)
        for key in ("total_steps", "steps_completed", "steps_failed",
                    "total_duration_ms", "success"):
            assert key in summary

    def test_total_steps_counts_unique_step_names(self):
        state = self._state_with_trace()
        summary = get_execution_summary(state)
        # interpret_query and validate_task = 2 unique steps
        assert summary["total_steps"] == 2

    def test_steps_completed_counts_unique_completed_steps(self):
        state = self._state_with_trace()
        summary = get_execution_summary(state)
        assert summary["steps_completed"] == 2

    def test_steps_failed_is_zero_when_none_failed(self):
        state = self._state_with_trace()
        summary = get_execution_summary(state)
        assert summary["steps_failed"] == 0

    def test_failed_step_counted_correctly(self):
        state = self._state_with_trace(success=False)
        state["execution_trace"][3]["status"] = "failed"
        summary = get_execution_summary(state)
        assert summary["steps_failed"] == 1

    def test_total_duration_sums_duration_ms(self):
        state = self._state_with_trace()
        summary = get_execution_summary(state)
        assert summary["total_duration_ms"] == pytest.approx(250.0)

    def test_none_duration_not_added(self):
        """Entries with duration_ms=None (started entries) should not inflate total."""
        state = self._state_with_trace()
        summary = get_execution_summary(state)
        # Only the two completed entries have duration_ms set
        assert summary["total_duration_ms"] == pytest.approx(250.0)

    def test_success_matches_state(self):
        assert get_execution_summary(self._state_with_trace(success=True))["success"] is True
        assert get_execution_summary(self._state_with_trace(success=False))["success"] is False

    def test_empty_trace_returns_empty_dict(self):
        state = create_initial_state("q")
        summary = get_execution_summary(state)
        assert summary == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])