"""Whole-recording views for measured or exported Radar results.

The saved cube remains the source of truth.  This module uses the official
Radar processing functions for every frame, then reduces the Range-Doppler
power along one axis to make time-visible review products.  It does not alter
or re-save the recording.
"""
from dataclasses import dataclass

import numpy as np


MAX_OVERVIEW_BYTES = 64 * 1024**2
DISPLAY_FLOOR_DB = -60.0


@dataclass(frozen=True)
class RecordingOverview:
    times_s: np.ndarray
    range_m: np.ndarray
    velocity_mps: np.ndarray
    micro_doppler_db: np.ndarray
    range_time_db: np.ndarray
    motion_energy: np.ndarray
    tx: int
    rx: int


def _relative_db(values):
    values = np.asarray(values, dtype=np.float64)
    peak = float(values.max(initial=0.0))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.full(values.shape, DISPLAY_FLOOR_DB, dtype=np.float32)
    floor = peak * 10.0 ** (DISPLAY_FLOOR_DB / 20.0)
    return np.maximum(20.0 * np.log10(np.maximum(values, floor) / peak), DISPLAY_FLOOR_DB).astype(np.float32)


def build_recording_overview(result, tx=0, rx=0):
    """Compute complete-recording time products without dropping frames/bins."""
    from witwin.radar.processing import ProcessingCube, range_doppler_map, range_profile

    axes = result.axes
    if not 0 <= tx < axes.num_tx or not 0 <= rx < axes.num_rx:
        raise ValueError("Requested antenna is outside the recorded array.")
    times = np.asarray(result.times_s, dtype=np.float64)
    ranges = np.asarray(axes.range_m, dtype=np.float64)
    velocities = np.asarray(axes.velocity_mps, dtype=np.float64)
    if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Recording overview requires at least two increasing finite timestamps.")
    output_bytes = len(times) * (len(ranges) + len(velocities) + 1) * 4
    if output_bytes > MAX_OVERVIEW_BYTES:
        raise ValueError("Recording overview exceeds 64 MiB; split the recording into smaller files.")

    micro_amplitude = np.empty((len(times), len(velocities)), dtype=np.float64)
    range_amplitude = np.empty((len(times), len(ranges)), dtype=np.float64)
    motion_amplitude = np.empty(len(times), dtype=np.float64)
    zero_bin = int(np.argmin(np.abs(velocities)))
    moving_bins = np.ones(len(velocities), dtype=bool)
    moving_bins[zero_bin] = False

    for frame in range(len(times)):
        processing = ProcessingCube(result.cube[frame], axes)
        profile = range_profile(processing, window="rectangular", remove_dc=False)
        rd = range_doppler_map(profile, window="hann")
        magnitude = rd.data[tx, rx].abs().detach().cpu().numpy().astype(np.float64, copy=False)
        power = np.square(magnitude)
        micro_amplitude[frame] = np.sqrt(power.sum(axis=1))
        range_amplitude[frame] = np.sqrt(power.sum(axis=0))
        motion_amplitude[frame] = np.sqrt(power[moving_bins].sum()) if moving_bins.any() else 0.0

    peak_motion = float(motion_amplitude.max(initial=0.0))
    motion_energy = (
        motion_amplitude / peak_motion if peak_motion > 0.0 else np.zeros_like(motion_amplitude)
    ).astype(np.float32)
    return RecordingOverview(
        times_s=times.copy(),
        range_m=ranges.copy(),
        velocity_mps=velocities.copy(),
        micro_doppler_db=_relative_db(micro_amplitude),
        range_time_db=_relative_db(range_amplitude),
        motion_energy=motion_energy,
        tx=int(tx),
        rx=int(rx),
    )


def _image_plot(values, x, y, title, ylabel):
    from witwin_server.api import PlotData

    # Studio's heatmap requires uniformly spaced bins. Hardware timestamps are
    # increasing but contain sub-frame USB/host jitter, so render them on the
    # measured recording endpoints while retaining every frame in order.
    display_time = np.linspace(float(x[0]), float(x[-1]), len(x), dtype=np.float64)
    return PlotData(
        "imshow",
        {
            "values": values.T.tolist(),
            "x": display_time.tolist(),
            "y": y.tolist(),
            "origin": "lower",
            "range": {
                "min": DISPLAY_FLOOR_DB,
                "max": 0.0,
                "label": "Relative amplitude (dB; recording peak = 0 dB)",
            },
        },
        title=title,
        xlabel="Time (s)",
        ylabel=ylabel,
    )


def recording_overview_plot(overview, view):
    """Return one whole-recording plot for the shared Radar Replay figure."""
    from witwin_server.api import PlotData

    antenna = f"TX{overview.tx} RX{overview.rx}"
    if view == "micro_doppler":
        return _image_plot(
            overview.micro_doppler_db,
            overview.times_s,
            overview.velocity_mps,
            f"Micro-Doppler | all {len(overview.times_s)} frames | {antenna}",
            "Closing velocity (m/s)",
        )
    if view == "range_time":
        return _image_plot(
            overview.range_time_db,
            overview.times_s,
            overview.range_m,
            f"Range-Time | all {len(overview.times_s)} frames | {antenna}",
            "Range (m)",
        )
    if view == "motion_energy":
        return PlotData(
            "line",
            {
                "series": [{
                    "x": overview.times_s.tolist(),
                    "y": overview.motion_energy.tolist(),
                    "label": "Non-zero-Doppler energy",
                }],
                "xlim": [float(overview.times_s[0]), float(overview.times_s[-1])],
                "ylim": [0.0, 1.05],
            },
            title=f"Motion Energy | all {len(overview.times_s)} frames | {antenna}",
            xlabel="Time (s)",
            ylabel="Relative motion energy (recording peak = 1)",
        )
    raise ValueError(f"Unsupported complete-recording view: {view}")
