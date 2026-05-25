"""Small shared conversion helpers + sensor unit pickers for the radar adapter.

The radar sensor uses **non-SI units on purpose** (master plan §1): ``slope`` in
MHz/us, times in us, ``sample_rate`` in ksps, ``power`` in dBm, antenna locations in
half-wavelength units. The unit pickers below keep the stored value in the platform's
native unit (the option with multiplier 1.0 is the base), so the round trip is exact
no matter which unit the editor displays.
"""
from typing import Any, List, Optional

# Carrier frequency: stored in Hz (RadarConfig.fc), shown in GHz by default.
FREQUENCY_UNITS = [
    {"label": "Hz", "multiplier": 1.0},
    {"label": "kHz", "multiplier": 1e3},
    {"label": "MHz", "multiplier": 1e6},
    {"label": "GHz", "multiplier": 1e9},
]

# Chirp slope: stored in MHz/us (RadarConfig.slope).
SLOPE_UNITS = [
    {"label": "MHz/us", "multiplier": 1.0},
    {"label": "GHz/us", "multiplier": 1e3},
]

# ADC sample rate: stored in ksps (RadarConfig.sample_rate).
SAMPLE_RATE_UNITS = [
    {"label": "sps", "multiplier": 1e-3},
    {"label": "ksps", "multiplier": 1.0},
    {"label": "Msps", "multiplier": 1e3},
]

# Chirp/frame times: stored in us (RadarConfig.adc_start_time / idle_time / ramp_end_time).
TIME_UNITS = [
    {"label": "ns", "multiplier": 1e-3},
    {"label": "us", "multiplier": 1.0},
    {"label": "ms", "multiplier": 1e3},
]


def num(value: Any) -> float:
    """Component scalar (tensor or number) -> python float."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def vec(value: Any) -> List[float]:
    """Component vector (tensor or list) -> python floats."""
    seq = value.tolist() if hasattr(value, "tolist") else value
    return [float(v) for v in seq]


def vec_list(value: Any) -> List[List[float]]:
    """Component list-of-vectors -> list of [x, y, z] python floats."""
    return [vec(entry) for entry in (value or [])]


def opt_dict_equal(a: Optional[dict], b: Optional[dict]) -> bool:
    """Compare two optional sub-config dicts (None == None)."""
    return a == b
