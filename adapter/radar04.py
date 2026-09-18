"""Narrow WiTwin Radar 0.4 boundary used by the Studio adapters.

Studio keeps the TI-style component fields (MHz/us, kSPS and microseconds),
while Radar 0.4 owns their conversion through :meth:`Radar.from_dict`.  Keeping
that conversion here prevents the snapshot and animation paths from depending
on deleted 0.3 configuration and simulation internals.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


RADAR04_CONFIG_KEYS = (
    "num_tx",
    "num_rx",
    "fc",
    "slope",
    "power",
    "adc_samples",
    "adc_start_time",
    "sample_rate",
    "idle_time",
    "ramp_end_time",
    "chirp_per_frame",
    "tx_loc",
    "rx_loc",
    "antenna_pattern",
    "output_domain",
)


def flat_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the public Radar 0.4 flat FMCW interchange fields."""

    return {
        key: value
        for key in RADAR04_CONFIG_KEYS
        if key in config and value_is_present(value := config[key])
    }


def value_is_present(value: Any) -> bool:
    """Keep false-y numeric values while omitting optional ``None`` blocks."""

    return value is not None


def build_radar(
    config: Mapping[str, Any],
    *,
    position: Sequence[float],
    look_at: Sequence[float],
    up: Sequence[float],
    polarization: str | Sequence[float] = "up",
    device: str = "cuda",
):
    """Construct the immutable Radar 0.4 instrument from Studio state."""

    from witwin.radar import Radar

    return Radar.from_dict(
        flat_config(config),
        position=tuple(float(value) for value in position),
        look_at=tuple(float(value) for value in look_at),
        up=tuple(float(value) for value in up),
        polarization=(
            polarization
            if isinstance(polarization, str)
            else tuple(float(value) for value in polarization)
        ),
        device=device,
    )

