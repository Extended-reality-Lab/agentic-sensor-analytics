# data/reftek_parser.py
import struct
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional

PACKET_TYPES = {
    b'DT': 'data',
    b'SH': 'state_of_health',
    b'EH': 'event_header',
    b'ET': 'event_trailer',
    b'AD': 'auxiliary_data',
    b'CD': 'calibration_data',
    b'SC': 'station_channel',
    b'DS': 'data_stream',
    b'OM': 'operation_mode',
    b'PD': 'parameter_data',
    b'OK': 'acknowledgement',
}

VALID_PACKET_BYTES = set(PACKET_TYPES.keys())


@dataclass
class RT130DataPacket:
    unit_id: str            # e.g. 'E2DB'
    stream: int             # stream byte from header
    channel: int            # channel byte from header
    packet_index: int       # sequential index within file
    samples: np.ndarray     # decoded integer counts
    sample_rate: float      # samples per second


@dataclass
class EventHeader:
    unit_id: str
    trigger_time: Optional[datetime]
    sample_rate: int
    recording_mode: str     # e.g. 'CON' for continuous
    raw_text: str


def _decode_steim2(frame_bytes: bytes, num_samples: int) -> np.ndarray:
    """Decode STEIM-2 compressed accelerometer data."""
    samples = []
    x0 = None

    num_frames = len(frame_bytes) // 64
    for fi in range(num_frames):
        frame = frame_bytes[fi * 64:(fi + 1) * 64]
        if len(frame) < 64:
            break

        descriptor = struct.unpack('>I', frame[0:4])[0]

        if fi == 0:
            x0 = struct.unpack('>i', frame[4:8])[0]
            samples.append(x0)
            start_word = 3
        else:
            start_word = 1

        last = samples[-1] if samples else 0

        for wi in range(start_word, 16):
            if len(samples) >= num_samples:
                break
            cnib = (descriptor >> (30 - wi * 2)) & 0x3
            word_bytes = frame[wi * 4:(wi + 1) * 4]
            word = struct.unpack('>I', word_bytes)[0]
            sword = struct.unpack('>i', word_bytes)[0]

            if cnib == 0:
                continue
            elif cnib == 1:
                dnib = (word >> 30) & 0x3
                if dnib == 1:    # 4 x 8-bit differences
                    for shift in [22, 14, 6]:
                        d = (word >> shift) & 0xFF
                        if d > 127: d -= 256
                        last += d
                        samples.append(last)
                        if len(samples) >= num_samples: break
                    d = word & 0xFF
                    if d > 127: d -= 256
                    last += d
                    samples.append(last)
                elif dnib == 2:  # 5 x 6-bit differences
                    for shift in [24, 18, 12, 6, 0]:
                        d = (word >> shift) & 0x3F
                        if d > 31: d -= 64
                        last += d
                        samples.append(last)
                        if len(samples) >= num_samples: break
                elif dnib == 3:  # 6 x 5-bit differences
                    for shift in [25, 20, 15, 10, 5, 0]:
                        d = (word >> shift) & 0x1F
                        if d > 15: d -= 32
                        last += d
                        samples.append(last)
                        if len(samples) >= num_samples: break
            elif cnib == 2:      # 2 x 16-bit differences
                for shift in [16, 0]:
                    d = (word >> shift) & 0xFFFF
                    if d > 32767: d -= 65536
                    last += d
                    samples.append(last)
                    if len(samples) >= num_samples: break
            elif cnib == 3:      # 1 x 32-bit difference
                last += sword
                samples.append(last)

        if len(samples) >= num_samples:
            break

    return np.array(samples[:num_samples], dtype=np.int32)


def parse_event_header(payload: bytes) -> EventHeader:
    """Extract metadata from EH (Event Header) packet payload."""
    unit_id = ''
    trigger_time = None
    sample_rate = 0
    recording_mode = ''

    try:
        text = payload.decode('ascii', errors='replace')
        
        # Trigger time: 'Trigger Time = 2023151000000000'
        m = re.search(r'Trigger Time\s*=\s*(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d{3})', text)
        if m:
            year, jday, hh, mm, ss, ms = map(int, m.groups())
            trigger_time = (datetime(year, 1, 1, tzinfo=timezone.utc)
                            + timedelta(days=jday - 1, hours=hh,
                                        minutes=mm, seconds=ss, milliseconds=ms))

        # Sample rate: '200 CON'
        m = re.search(r'(\d+)\s+(CON|EVT|COT)', text)
        if m:
            sample_rate = int(m.group(1))
            recording_mode = m.group(2)

        return EventHeader(
            unit_id=unit_id,
            trigger_time=trigger_time,
            sample_rate=sample_rate,
            recording_mode=recording_mode,
            raw_text=text,
        )
    except Exception:
        return EventHeader('', None, 0, '', '')


def parse_rt130_filename(filename: str) -> dict:
    """
    Decode RT-130 filename: SSXXXXXXX_TTTTTTTT
      SS       = stream number (2 digits)
      XXXXXXX  = event/sequence number (7 digits)
      TTTTTTTT = start time offset from midnight in milliseconds (hex)

    Returns dict with stream, event_number, start_ms, start_offset_td
    """
    try:
        name = Path(filename).stem
        parts = name.split('_')
        if len(parts) != 2 or len(parts[0]) != 9:
            return {}
        prefix, time_hex = parts
        stream = int(prefix[0:2])
        event_num = int(prefix[2:9])
        start_ms = int(time_hex, 16)
        return {
            'stream': stream,
            'event_number': event_num,
            'start_ms_from_midnight': start_ms,
            'start_offset': timedelta(milliseconds=start_ms),
        }
    except Exception:
        return {}


def parse_rt130_file(
    filepath: Path,
    recording_date: datetime,           # midnight UTC of the recording day
    primary_stream_channel: tuple = (0x15, 0x10),  # (stream, channel) to extract
) -> tuple[list[RT130DataPacket], Optional[EventHeader]]:
    """
    Parse a single RT-130 binary data file.

    Returns (data_packets, event_header)
    data_packets: list of RT130DataPacket with decoded samples
    event_header: metadata extracted from EH packet if present
    """
    with open(filepath, 'rb') as f:
        raw = f.read()

    file_info = parse_rt130_filename(filepath.name)
    start_offset = file_info.get('start_offset', timedelta(0))
    file_start_time = recording_date + start_offset

    data_packets = []
    event_header = None
    packet_index = 0
    offset = 0

    while offset < len(raw) - 16:
        ptype = raw[offset:offset + 2]
        if ptype not in VALID_PACKET_BYTES:
            offset += 1
            continue

        plen = struct.unpack('>H', raw[offset + 2:offset + 4])[0] * 16
        if plen == 0 or offset + plen > len(raw):
            offset += 2
            continue

        unit_id = raw[offset + 4:offset + 6].hex().upper()
        stream_byte = raw[offset + 6]
        chan_byte = raw[offset + 7]
        payload = raw[offset + 16:offset + plen]

        if ptype == b'EH' and event_header is None:
            event_header = parse_event_header(payload)
            event_header.unit_id = unit_id

        elif ptype == b'DT':
            # Only decode the dominant stream/channel (filters out noise hits)
            if (stream_byte, chan_byte) == primary_stream_channel:
                # Payload layout:
                # [0:2]  = rate code / flags
                # [2:4]  = packet sequence number  
                # [4]    = encoding (0x04 = STEIM-2 in this instrument)
                # [5]    = flags
                # [6:8]  = channel flags
                # [8:16] = 8 more header bytes
                # [16:]  starts at offset 16 within payload for some RT-130 variants
                # We determined STEIM-2 frames start at payload offset 48
                if len(payload) >= 64:
                    frames = payload[48:]  # STEIM-2 frame data
                    samples = _decode_steim2(frames, 256)
                    if len(samples) > 0:
                        data_packets.append(RT130DataPacket(
                            unit_id=unit_id,
                            stream=stream_byte,
                            channel=chan_byte,
                            packet_index=packet_index,
                            samples=samples,
                            sample_rate=200.0,  # from EH '200 CON'
                        ))
                packet_index += 1

        offset += plen

    return data_packets, event_header


def packets_to_dataframe(
    packets: list[RT130DataPacket],
    file_start_time: datetime,
    unit_id: str,
    axis_label: str = 'unknown',
    sensitivity_counts_per_g: float = 419430.0,
) -> pd.DataFrame:
    """
    Convert RT130DataPacket list to a tidy DataFrame.
    Timestamps are computed from file start time + sequential sample index.
    axis_label: 'X', 'Y', or 'Z' — determined by stream/folder context.
    sensitivity_counts_per_g: convert raw counts to g-force.
    """
    if not packets:
        return pd.DataFrame()

    sample_rate = packets[0].sample_rate
    dt_us = 1_000_000 / sample_rate  # microseconds per sample

    all_samples = np.concatenate([p.samples for p in packets])
    n = len(all_samples)

    timestamps = [
        file_start_time + timedelta(microseconds=i * dt_us)
        for i in range(n)
    ]

    g_values = all_samples.astype(float) / sensitivity_counts_per_g

    return pd.DataFrame({
        'timestamp':   timestamps,
        'value':       g_values,
        'raw_counts':  all_samples,
        'unit':        'g',
        'location':    unit_id,
        'axis':        axis_label,
        'sample_rate': sample_rate,
        'sensor_type': 'accelerometer',
    })