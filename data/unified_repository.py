"""
data/unified_repository.py

Routes sensor data queries to the correct backend:
  - SMT API           → temperature, humidity, co2, moisture (existing)
  - RefTek RT-130     → accelerometer (TrimbleAccelerometers/reftek/...)
  - File-based CSV    → strain (Loadcells), weather (WeatherStation)

Drop-in replacement for SensorDataRepository anywhere it is used.
The LLMDataBridge, AgentNodes, and AgentExecutor require no changes.
"""

from __future__ import annotations

import re
import struct
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# RT-130 BINARY PARSER
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_PACKET_TYPES = {
    b"DT", b"SH", b"EH", b"ET", b"AD",
    b"CD", b"SC", b"DS", b"OM", b"PD", b"OK",
}

# The dominant (stream_byte, channel_byte) observed in these files.
# All real DT data packets carry this combination; the handful of stray
# packets with other values are parse artefacts and should be skipped.
_PRIMARY_STREAM_CHAN = (0x15, 0x10)


def _decode_steim2(frame_bytes: bytes, num_samples: int) -> np.ndarray:
    """Decode STEIM-2 compressed integer samples from RT-130 DT packets."""
    samples: list[int] = []
    last = 0

    num_frames = len(frame_bytes) // 64
    for fi in range(num_frames):
        frame = frame_bytes[fi * 64 : (fi + 1) * 64]
        if len(frame) < 64:
            break

        descriptor = struct.unpack(">I", frame[0:4])[0]

        if fi == 0:
            x0 = struct.unpack(">i", frame[4:8])[0]
            samples.append(x0)
            last = x0
            start_word = 3
        else:
            start_word = 1

        for wi in range(start_word, 16):
            if len(samples) >= num_samples:
                break

            cnib = (descriptor >> (30 - wi * 2)) & 0x3
            wb = frame[wi * 4 : (wi + 1) * 4]
            word = struct.unpack(">I", wb)[0]
            sword = struct.unpack(">i", wb)[0]

            if cnib == 0:
                continue

            elif cnib == 1:                          # variable-length diffs
                dnib = (word >> 30) & 0x3
                if dnib == 1:                        # 4 × 8-bit diffs
                    for shift in (22, 14, 6, -1):
                        if shift == -1:
                            d = word & 0xFF
                        else:
                            d = (word >> shift) & 0xFF
                        if d > 127:
                            d -= 256
                        last += d
                        samples.append(last)
                        if len(samples) >= num_samples:
                            break
                elif dnib == 2:                      # 5 × 6-bit diffs
                    for shift in (24, 18, 12, 6, 0):
                        d = (word >> shift) & 0x3F
                        if d > 31:
                            d -= 64
                        last += d
                        samples.append(last)
                        if len(samples) >= num_samples:
                            break
                elif dnib == 3:                      # 6 × 5-bit diffs
                    for shift in (25, 20, 15, 10, 5, 0):
                        d = (word >> shift) & 0x1F
                        if d > 15:
                            d -= 32
                        last += d
                        samples.append(last)
                        if len(samples) >= num_samples:
                            break

            elif cnib == 2:                          # 2 × 16-bit diffs
                for shift in (16, 0):
                    d = (word >> shift) & 0xFFFF
                    if d > 32767:
                        d -= 65536
                    last += d
                    samples.append(last)
                    if len(samples) >= num_samples:
                        break

            elif cnib == 3:                          # 1 × 32-bit diff
                last += sword
                samples.append(last)

        if len(samples) >= num_samples:
            break

    return np.array(samples[:num_samples], dtype=np.int32)


def _parse_eh_sample_rate(payload: bytes) -> int:
    """Extract sample rate (sps) from an EH (Event Header) packet payload."""
    try:
        text = payload.decode("ascii", errors="replace")
        m = re.search(r"(\d+)\s+(CON|EVT|COT)", text)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 200   # safe default based on observed data


def _parse_filename_start_time(filename: str, recording_date: datetime) -> datetime:
    """
    Decode the hex start-time embedded in an RT-130 filename.

    Filename format:  SSXXXXXXX_TTTTTTTT
      SS        = stream number (2 digits)
      XXXXXXX   = event/sequence number (7 digits)
      TTTTTTTT  = milliseconds from midnight UTC (8 hex chars)

    Example: 000000003_0036EE80
      stream=0, event=3, offset=3_600_000 ms → 01:00:00 UTC
    """
    try:
        stem = Path(filename).stem
        parts = stem.split("_")
        if len(parts) == 2 and len(parts[0]) == 9:
            ms_offset = int(parts[1], 16)
            return recording_date + timedelta(milliseconds=ms_offset)
    except Exception:
        pass
    return recording_date


def _parse_rt130_file(
    filepath: Path,
    recording_date: datetime,
    sample_rate_override: Optional[int] = None,
) -> tuple[list[np.ndarray], datetime, float]:
    """
    Parse one RT-130 binary file.

    Returns
    -------
    (sample_blocks, file_start_time, sample_rate)
      sample_blocks : list of 1-D int32 arrays, one per DT packet
      file_start_time : UTC datetime derived from filename
      sample_rate     : samples per second
    """
    file_start = _parse_filename_start_time(filepath.name, recording_date)

    with open(filepath, "rb") as fh:
        raw = fh.read()

    sample_rate = sample_rate_override or 200.0
    sample_blocks: list[np.ndarray] = []
    offset = 0

    while offset < len(raw) - 16:
        ptype = raw[offset : offset + 2]
        if ptype not in _VALID_PACKET_TYPES:
            offset += 1
            continue

        plen = struct.unpack(">H", raw[offset + 2 : offset + 4])[0] * 16
        if plen == 0 or offset + plen > len(raw):
            offset += 2
            continue

        payload = raw[offset + 16 : offset + plen]

        if ptype == b"EH" and sample_rate_override is None:
            detected = _parse_eh_sample_rate(payload)
            if detected:
                sample_rate = float(detected)

        elif ptype == b"DT":
            stream_byte = raw[offset + 6]
            chan_byte = raw[offset + 7]
            if (stream_byte, chan_byte) == _PRIMARY_STREAM_CHAN:
                if len(payload) >= 64:
                    # STEIM-2 frames begin at payload byte 48 (confirmed empirically)
                    samples = _decode_steim2(payload[48:], 256)
                    if len(samples) > 0:
                        sample_blocks.append(samples)

        offset += plen

    return sample_blocks, file_start, sample_rate


# ═══════════════════════════════════════════════════════════════════════════════
# REFTEK REPOSITORY  (accelerometer data)
# ═══════════════════════════════════════════════════════════════════════════════


class RefTekRepository:
    """
    Reads accelerometer waveform data stored in RefTek RT-130 binary format.

    Expected directory tree under base_path:
        TrimbleAccelerometers/
        └── reftek/
            └── YYYYDDD/          ← Julian day folder (e.g. 2023151)
                └── <UNIT_ID>/    ← datalogger ID (e.g. E2DB)
                    ├── 0/        ← stream 0 files
                    └── 2/        ← stream 2 files
    """

    def __init__(
        self,
        base_path: str,
        sensitivity_counts_per_g: Optional[float] = None,
        unit_labels: Optional[dict] = None,
        stream_axis_map: Optional[dict] = None,
    ) -> None:
        if sensitivity_counts_per_g is None:
            raise ValueError("sensitivity_counts_per_g must be set in data_config.yaml")
        if unit_labels is None:
            raise ValueError("unit_labels must be set in data_config.yaml under file_data.unit_labels")
        if stream_axis_map is None:
            raise ValueError("stream_axis_map must be set in data_config.yaml under file_data.stream_axis_map")
        
        self.reftek_root = Path(base_path) / "TrimbleAccelerometers" / "reftek"
        self.sensitivity = sensitivity_counts_per_g
        self.unit_labels = unit_labels
        self.stream_axis_map = stream_axis_map
        self._label_to_id: dict[str, str] = {
            v.lower(): k for k, v in self.unit_labels.items()
        }

    # ── public interface ────────────────────────────────────────────────────

    def get_readings(
        self,
        sensor_type: str,          # 'accelerometer'
        location: str,             # unit ID, friendly name, or 'all'
        start_time: datetime,
        end_time: datetime,
        streams: Optional[list[str]] = None,   # e.g. ['0', '2'] — None = all
        axes: Optional[list[str]] = None,      # e.g. ['X','Z'] — None = all
    ) -> pd.DataFrame:
        """
        Return a tidy DataFrame for the requested accelerometer data.

        Columns
        -------
        timestamp    : datetime (UTC)
        value        : float (g-force, converted from raw counts)
        raw_counts   : int
        unit         : str  ('g')
        location     : str  (unit ID, e.g. 'E2DB')
        axis         : str  ('X' | 'Y' | 'Z' | 'unknown')
        stream       : str  folder name ('0', '1', '2')
        sample_rate  : float
        sensor_type  : str  ('accelerometer')
        """
        unit_ids = self._resolve_units(location)
        day_folders = self._day_folders_in_range(start_time, end_time)
        frames: list[pd.DataFrame] = []

        for day_folder in day_folders:
            year = int(day_folder.name[:4])
            julian_day = int(day_folder.name[4:])
            recording_date = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
                days=julian_day - 1
            )

            for unit_id in unit_ids:
                unit_path = day_folder / unit_id
                try:
                    if not unit_path.exists():
                        continue
                except PermissionError:
                    logger.debug("Permission denied stat-ing: %s (skipping)", unit_path)
                    continue

                try:
                    stream_dirs = sorted(unit_path.iterdir())
                except PermissionError:
                    logger.debug("Permission denied entering: %s (skipping)", unit_path)
                    continue

                for stream_dir in stream_dirs:
                    try:
                        if not stream_dir.is_dir():
                            continue
                    except PermissionError:
                        logger.debug("Permission denied stat-ing: %s (skipping)", stream_dir)
                        continue
                    stream_name = stream_dir.name
                    if streams and stream_name not in streams:
                        continue

                    axis = self.stream_axis_map.get(stream_name, "unknown")
                    if axes and axis not in axes:
                        continue

                    try:
                        files = sorted(stream_dir.iterdir())
                    except PermissionError:
                        logger.debug("Permission denied entering: %s (skipping)", stream_dir)
                        continue

                    for fpath in files:
                        try:
                            if not fpath.is_file():
                                continue
                        except PermissionError:
                            logger.debug("Permission denied stat-ing: %s (skipping)", fpath)
                            continue
                        try:
                            df = self._file_to_dataframe(
                                fpath, recording_date, unit_id, axis, stream_name
                            )
                            if not df.empty:
                                frames.append(df)
                        except PermissionError:
                            logger.debug("Permission denied reading: %s (skipping)", fpath)
                        except Exception as exc:
                            logger.warning("Failed to parse %s: %s", fpath, exc)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        mask = (combined["timestamp"] >= start_time) & (
            combined["timestamp"] <= end_time
        )
        return combined.loc[mask].sort_values("timestamp").reset_index(drop=True)

    def get_available_sensors(self) -> list[str]:
        return ["accelerometer"]

    def get_available_locations(self) -> list[str]:
        """Return friendly names for all unit IDs found on disk."""
        ids = self._discover_unit_ids()
        return [self.unit_labels.get(uid, uid) for uid in sorted(ids)]

    def get_available_unit_ids(self) -> list[str]:
        return sorted(self._discover_unit_ids())

    def get_date_range(self) -> tuple[Optional[datetime], Optional[datetime]]:
        if not self.reftek_root.exists():
            return None, None
        day_dirs = []
        try:
            for d in self.reftek_root.iterdir():
                try:
                    if d.is_dir() and re.match(r"^\d{7}$", d.name):
                        day_dirs.append(d)
                except PermissionError:
                    logger.debug("Permission denied stat-ing: %s (skipping)", d)
        except PermissionError as e:
            logger.warning("Permission denied listing reftek root: %s", e)
            return None, None
        if not day_dirs:
            return None, None
        day_dirs = sorted(day_dirs, key=lambda p: p.name)
        return (
            self._folder_name_to_dt(day_dirs[0].name),
            self._folder_name_to_dt(day_dirs[-1].name),
        )
    
    def get_sensors_by_node(self) -> dict:
        """Return {friendly_name: ['accelerometer']} for sidebar display."""
        return {
            self.unit_labels.get(uid, uid): ["accelerometer"]
            for uid in sorted(self._discover_unit_ids())
        }

    # ── private helpers ─────────────────────────────────────────────────────

    def _file_to_dataframe(
        self,
        fpath: Path,
        recording_date: datetime,
        unit_id: str,
        axis: str,
        stream_name: str,
    ) -> pd.DataFrame:
        sample_blocks, file_start, sample_rate = _parse_rt130_file(
            fpath, recording_date
        )
        if not sample_blocks:
            return pd.DataFrame()

        all_samples = np.concatenate(sample_blocks)
        n = len(all_samples)
        dt_us = 1_000_000.0 / sample_rate

        timestamps = [
            file_start + timedelta(microseconds=i * dt_us) for i in range(n)
        ]

        return pd.DataFrame(
            {
                "timestamp":   timestamps,
                "value":       all_samples.astype(float) / self.sensitivity,
                "raw_counts":  all_samples,
                "unit":        "g",
                "location":    unit_id,
                "axis":        axis,
                "stream":      stream_name,
                "sample_rate": sample_rate,
                "sensor_type": "accelerometer",
            }
        )

    def _resolve_units(self, location: str) -> list[str]:
        if not location or location.lower() in ("all", ""):
            return self.get_available_unit_ids()
        # Check direct unit ID match (e.g. 'E2DB')
        upper = location.upper()
        if re.match(r"^[0-9A-F]{4}$", upper):
            return [upper]
        # Check friendly name (e.g. 'Accelerometer Node 1')
        uid = self._label_to_id.get(location.lower())
        if uid:
            return [uid]
        logger.warning("Unknown accelerometer location '%s'; returning all.", location)
        return self.get_available_unit_ids()

    def _discover_unit_ids(self) -> set[str]:
        ids: set[str] = set()
        if not self.reftek_root.exists():
            return ids
        try:
            root_entries = list(self.reftek_root.iterdir())
        except PermissionError as e:
            logger.warning("Permission denied listing reftek root: %s", e)
            return ids
        for day_dir in root_entries:
            try:
                if not day_dir.is_dir():
                    continue
            except PermissionError:
                logger.debug("Permission denied stat-ing: %s (skipping)", day_dir)
                continue
            try:
                for unit_dir in day_dir.iterdir():
                    try:
                        if unit_dir.is_dir():
                            ids.add(unit_dir.name)
                    except PermissionError:
                        logger.debug("Permission denied stat-ing: %s (skipping)", unit_dir)
            except PermissionError:
                logger.debug("Permission denied entering: %s (skipping)", day_dir)
        return ids

    def _day_folders_in_range(
        self, start: datetime, end: datetime
    ) -> list[Path]:
        if not self.reftek_root.exists():
            return []
        result = []
        try:
            root_entries = list(self.reftek_root.iterdir())
        except PermissionError as e:
            logger.warning("Permission denied listing reftek root: %s", e)
            return []
        for d in root_entries:
            try:
                if not d.is_dir() or not re.match(r"^\d{7}$", d.name):
                    continue
            except PermissionError:
                logger.debug("Permission denied stat-ing: %s (skipping)", d)
                continue
            try:
                folder_dt = self._folder_name_to_dt(d.name)
            except Exception:
                continue
            if folder_dt.date() >= start.date() and folder_dt.date() <= end.date():
                result.append(d)
        return sorted(result, key=lambda p: p.name)

    @staticmethod
    def _folder_name_to_dt(name: str) -> datetime:
        year = int(name[:4])
        jday = int(name[4:])
        return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=jday - 1)


# ═══════════════════════════════════════════════════════════════════════════════
# FILE-BASED CSV REPOSITORY  (Loadcells, WeatherStation)
# ═══════════════════════════════════════════════════════════════════════════════

_CSV_SOURCE_MAP = {
    "strain":  "Loadcells",
    "weather": "WeatherStation",
}


class FileDataRepository:
    """
    Reads CSV-based sensor data from PeavySensorData sub-folders.
    Expected columns (adjust _normalise_df if files differ):
        timestamp, value, unit, location
    """

    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)

    def get_readings(
        self,
        sensor_type: str,
        location: Optional[str],
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        folder_name = _CSV_SOURCE_MAP.get(sensor_type)
        if not folder_name:
            return pd.DataFrame()

        folder = self.base_path / folder_name
        if not folder.exists():
            logger.warning("Folder not found: %s", folder)
            return pd.DataFrame()

        frames = []
        for csv_file in sorted(folder.glob("**/*.csv")):
            try:
                df = pd.read_csv(csv_file)
                df = self._normalise_df(df, sensor_type)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                logger.warning("Could not read %s: %s", csv_file, exc)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        mask = (combined["timestamp"] >= start_time) & (
            combined["timestamp"] <= end_time
        )
        combined = combined.loc[mask]

        if location and location.lower() not in ("all", "") and "location" in combined.columns:
            combined = combined[combined["location"] == location]

        return combined.sort_values("timestamp").reset_index(drop=True)

    def get_available_sensors(self) -> list[str]:
        return [k for k, v in _CSV_SOURCE_MAP.items()
                if (self.base_path / v).exists()]

    def get_available_locations(self) -> list[str]:
        locs: set[str] = set()
        for folder in _CSV_SOURCE_MAP.values():
            for f in (self.base_path / folder).glob("**/*.csv"):
                try:
                    df = pd.read_csv(f, nrows=20)
                    if "location" in df.columns:
                        locs.update(df["location"].dropna().unique())
                except Exception:
                    pass
        return sorted(locs)

    @staticmethod
    def _normalise_df(df: pd.DataFrame, sensor_type: str) -> pd.DataFrame:
        """
        Normalise raw CSV columns to the standard schema.
        Adjust column aliases here once the actual Loadcell/Weather file
        headers are known.
        """
        aliases = {
            "time":       "timestamp",
            "datetime":   "timestamp",
            "date_time":  "timestamp",
            "val":        "value",
            "reading":    "value",
            "node":       "location",
            "site":       "location",
        }
        df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

        if "timestamp" not in df.columns:
            logger.warning("No timestamp column found in CSV; skipping.")
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])

        if "value" not in df.columns and len(df.columns) >= 2:
            # Fallback: use second column as value
            df = df.rename(columns={df.columns[1]: "value"})

        if "unit" not in df.columns:
            df["unit"] = ""
        if "location" not in df.columns:
            df["location"] = "unknown"

        df["sensor_type"] = sensor_type
        return df[["timestamp", "value", "unit", "location", "sensor_type"]]


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED REPOSITORY  (public interface consumed by LLMDataBridge)
# ═══════════════════════════════════════════════════════════════════════════════

#  Sensor types handled by each backend
_REFTEK_SENSORS  = {"accelerometer"}
_FILE_SENSORS    = {"weather", "lc"}
# Everything else falls through to the existing SMT API repository


class UnifiedRepository:
    """
    Single repository that routes every sensor query to the correct backend.

    Parameters
    ----------
    smt_repo   : your existing SensorDataRepository (SMT API)
    base_path  : root of PeavySensorData on the network drive
                 e.g. "Z:/Data/COF/PeavySensorData"
    reftek_kw  : optional kwargs forwarded to RefTekRepository.__init__
    """

    def __init__(
        self,
        smt_repo,                          # SensorDataRepository instance
        base_path: str,
        reftek_kw: Optional[dict] = None,
    ) -> None:
        self._smt   = smt_repo
        self._rt130 = RefTekRepository(base_path, **(reftek_kw or {}))
        self._file  = FileDataRepository(base_path)

    # ── primary data method ─────────────────────────────────────────────────

    def get_readings(
        self,
        sensor_type: str,
        location,
        start_time: datetime,
        end_time: datetime,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Route to the correct backend and return a unified DataFrame.

        Extra kwargs (e.g. streams=['0','2'], axes=['X','Z']) are forwarded
        to RefTekRepository.get_readings for fine-grained accelerometer queries.
        """
        st = sensor_type.lower()
        if st in _REFTEK_SENSORS:
            loc = location if isinstance(location, str) else location[0]
            return self._rt130.get_readings(st, loc, start_time, end_time, **kwargs)
        if st in _FILE_SENSORS:
            loc = location if isinstance(location, str) else location[0]
            return self._file.get_readings(st, loc, start_time, end_time)
        # Default: SMT API (temperature, humidity, co2, moisture, …)
        return self._smt.get_readings(sensor_type, location, start_time, end_time)

    def get_readings_multiple_locations(
        self,
        sensor_type: str,
        locations: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Multi-location query — mirrors SensorDataRepository signature."""
        st = sensor_type.lower()
        if st in _REFTEK_SENSORS:
            frames = [
                self._rt130.get_readings(st, loc, start_time, end_time)
                for loc in locations
            ]
            non_empty = [f for f in frames if not f.empty]
            return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
        if st in _FILE_SENSORS:
            frames = [
                self._file.get_readings(st, loc, start_time, end_time)
                for loc in locations
            ]
            non_empty = [f for f in frames if not f.empty]
            return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
        return self._smt.get_readings_multiple_locations(
            sensor_type, locations, start_time, end_time
        )

    # ── metadata methods (used by LLMDataBridge.get_system_context) ─────────

    def get_available_sensors(self) -> list[str]:
        smt_sensors = self._smt.get_available_sensors()
        # Add accelerometer statically from config, no folder scan
        combined = list(dict.fromkeys(smt_sensors + ["accelerometer"]))
        return combined

    def get_available_locations(self) -> list[str]:
        smt_locs = self._smt.get_available_locations()
        # Add accelerometer unit IDs from config, no folder scan
        accel_locs = list(self._rt130.unit_labels.values())
        seen = set()
        result = []
        for loc in smt_locs + accel_locs:
            if loc not in seen:
                seen.add(loc)
                result.append(loc)
        return result

    def get_sensors_by_node(self) -> dict:
        combined = {}
        combined.update(self._smt.get_sensors_by_node())
        # Add accelerometer nodes from config, no folder scan
        for label in self._rt130.unit_labels.values():
            combined[label] = ["accelerometer"]
        return combined

    def get_date_range_for_sensor(self, sensor_type: str) -> tuple:
        """Return (start, end) datetime for a specific sensor type."""
        st = sensor_type.lower()
        if st in _REFTEK_SENSORS:
            return self._rt130.get_date_range()
        # Fall back to SMT range for everything else
        return self._smt.get_date_range() if hasattr(self._smt, "get_date_range") else (None, None)
    
    def get_time_range(self) -> tuple:
        """
        Return (earliest, latest) datetime from SMT API only.
        RefTek folders are not accessed until explicitly queried.
        """
        return self._smt.get_time_range() if hasattr(self._smt, "get_time_range") else (None, None)


    # ── pass-through methods for full SensorDataRepository compatibility ────

    def validate_parameters(self, sensor_type, location, time_range):
        st = sensor_type.lower()
        if st in _REFTEK_SENSORS or st in _FILE_SENSORS:
            return []   # basic pass — add domain validation here if needed
        return self._smt.validate_parameters(sensor_type, location, time_range)

    def clear_cache(self):
        if hasattr(self._smt, "clear_cache"):
            self._smt.clear_cache()

    @property
    def connected(self) -> bool:
        return getattr(self._smt, "connected", True)

    # ── compatibility shims so graph.py needs no changes ────────────────────

    def connect(self):
        """Delegate connect() to the SMT repository."""
        if hasattr(self._smt, "connect"):
            self._smt.connect()

    class _ApiClientShim:
        """Minimal shim so graph.py's api_client.authenticated check works."""
        def __init__(self, smt_repo):
            self._smt = smt_repo
        @property
        def authenticated(self) -> bool:
            if hasattr(self._smt, "api_client"):
                return self._smt.api_client.authenticated
            return getattr(self._smt, "connected", True)

    @property
    def api_client(self):
        return self._ApiClientShim(self._smt)

    # ── updated factory classmethod ─────────────────────────────────────────

    @classmethod
    def from_config(cls, data_config_path=None) -> "UnifiedRepository":
        """
        Build a UnifiedRepository from data_config.yaml.
        Replaces AgentExecutor.from_config's direct use of SensorDataRepository.

        Usage in graph.py AgentExecutor.from_config():
            repository = UnifiedRepository.from_config(data_config_path)
            bridge = LLMDataBridge(repository)
        """
        import yaml

        config_path = data_config_path or "config/data_config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        from data import SensorDataRepository  # existing SMT repo
        smt_repo = SensorDataRepository.from_config(data_config_path)

        file_cfg = cfg.get("file_data", {})
        base_path = file_cfg.get("base_path")
        if not base_path:
            raise ValueError(
                "file_data.base_path must be set in data_config.yaml. "
                "Example: Z:/Data/COF/PeavySensorData"
            )

        sensitivity = file_cfg.get("sensitivity_counts_per_g")
        if not sensitivity:
            raise ValueError(
                "file_data.sensitivity_counts_per_g must be set in data_config.yaml. "
                "Obtain the correct value from the accelerometer sensor spec sheet."
            )

        reftek_kw = {
            "sensitivity_counts_per_g": float(sensitivity),
            "unit_labels": file_cfg.get("unit_labels"),
            "stream_axis_map": file_cfg.get("stream_axis_map"),
        }

        return cls(smt_repo=smt_repo, base_path=base_path, reftek_kw=reftek_kw)