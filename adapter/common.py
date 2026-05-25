"""Small shared conversion helpers + sensor unit pickers for the radar adapter.

The radar sensor uses **non-SI units on purpose** (master plan §1): ``slope`` in
MHz/us, times in us, ``sample_rate`` in ksps, ``power`` in dBm, antenna locations in
half-wavelength units. The unit pickers below keep the stored value in the platform's
native unit (the option with multiplier 1.0 is the base), so the round trip is exact
no matter which unit the editor displays.
"""
from typing import Any, Dict, List, Optional

C0 = 299792458.0  # speed of light (m/s), matching witwin.radar.Radar

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


class Derived:
    """Read-only derived sensor values, computed exactly as ``witwin.radar.Radar`` does."""

    @staticmethod
    def compute(comp: Any) -> Dict[str, float]:
        """RadarConfig component -> range/doppler resolution + max range/doppler."""
        fc = num(comp.fc)
        fs = num(comp.sample_rate) * 1e3
        slope_hz = num(comp.slope) * 1e12
        adc_samples = num(comp.adc_samples)
        chirp_period = (num(comp.idle_time) + num(comp.ramp_end_time)) * 1e-6
        num_tx = num(comp.num_tx)
        num_doppler = num(comp.num_doppler_bins)
        lam = C0 / fc
        return {
            "range_resolution_m": C0 * fs / (2 * slope_hz * adc_samples),
            "max_range_m": C0 * fs / (2 * slope_hz),
            "doppler_resolution_mps": lam / (2 * num_doppler * chirp_period * num_tx),
            "max_doppler_mps": lam / (4 * chirp_period * num_tx),
        }

