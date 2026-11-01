"""Radar output checks (design 7.6, stage S7) on a finished animation cube.

They check that a run is real and correct, not that it is pretty: the cube is
finite and non-zero, has the measured frame count, each target returns energy
near its ground-truth range on the frames it is in view and moving, and no
static (zero-Doppler) return dominates the moving targets. Static dominance is a
soft warning, ``RADAR_STATIC_DOMINANT``, naming the range of the static peak so a
bright spot the installation rule missed is caught.

Processing uses the official ``witwin.radar.processing`` range profile and
Range-Doppler map; no DSP is implemented here.
"""
from __future__ import annotations

import math

import numpy as np


# A static peak this far above the strongest moving return is a warning.
# Calibrated on meeting sample 03 (2.8-8.6 s, 58 frames): the installation-rule
# mount peaks at 17.3 dB (0 frames over 20 dB); a north-wall mount looking
# straight at the south wall puts its foot point at 4.51 m and exceeds 20 dB on
# 57 of 58 frames (median 23.7 dB, max 44.4 dB).
STATIC_DOMINANT_DB = 20.
# A target is present when its moving return near the true range clears the
# median moving floor of the frame by this much.
PRESENCE_DB = 10.
PRESENCE_MIN_FRACTION = .8
MIN_TARGET_SPEED_MPS = .2
RANGE_WINDOW_M = .6
# Doppler bins either side of zero that still hold a Hann-windowed static return.
STATIC_DOPPLER_HALF_BINS = 1


def _frame_power(cube_frame, axes):
    """Incoherent Range-Doppler power over every TX/RX channel: (doppler, range)."""
    from witwin.radar.processing import ProcessingCube, range_doppler_map, range_profile

    profile = range_profile(ProcessingCube(cube_frame, axes), window="rectangular", remove_dc=False)
    rd = range_doppler_map(profile, window="hann")
    power = rd.data.abs().square().sum(dim=(0, 1))
    return power.detach().cpu().double().numpy(), rd.axes


def _db(value):
    return 10. * math.log10(max(float(value), 1e-300))


def radar_output_checks(cube, axes, times_s, *, labels=None, expected_frame_count=None,
                        static_dominant_db=STATIC_DOMINANT_DB, presence_db=PRESENCE_DB,
                        min_speed_mps=MIN_TARGET_SPEED_MPS, range_window_m=RANGE_WINDOW_M):
    """Checks on a ``[frame, TX, RX, slow, fast]`` cube with its producer axes."""
    import torch

    cube = torch.as_tensor(cube)
    times = np.asarray(times_s, dtype=np.float64)
    finite = bool(torch.isfinite(cube).all())
    nonzero = bool(finite and (cube.abs() > 0).any())
    frame_count = int(cube.shape[0])
    checks = {
        "all_finite": finite,
        "nonzero_return": nonzero,
        "frame_count_matches": frame_count == len(times) and (
            expected_frame_count is None or frame_count == int(expected_frame_count)),
    }
    frames, warnings = [], []
    targets = {}
    label_frames = {}
    if labels is not None:
        for target in labels.get("targets") or ():
            label_frames[str(target["object_id"])] = target["frames"]
            targets[str(target["object_id"])] = {"checked": 0, "present": 0, "missing_frames": [],
                                                 "skipped_static": 0, "not_in_view": 0}
    if finite and nonzero:
        for index in range(frame_count):
            power, rd_axes = _frame_power(cube[index], axes)
            velocity = rd_axes.velocity_mps.detach().cpu().double().numpy()
            ranges = rd_axes.range_m.detach().cpu().double().numpy()
            zero = int(np.argmin(np.abs(velocity)))
            static_rows = np.zeros(len(velocity), dtype=bool)
            static_rows[max(0, zero - STATIC_DOPPLER_HALF_BINS):zero + STATIC_DOPPLER_HALF_BINS + 1] = True
            static = power[static_rows]
            moving = power[~static_rows]
            static_bin = int(np.argmax(static.max(axis=0)))
            static_peak = float(static.max())
            moving_peak = float(moving.max()) if moving.size else 0.
            floor = float(np.median(moving)) if moving.size else 0.
            ratio_db = _db(static_peak) - _db(moving_peak)
            record = {"frame": index, "t_s": float(times[index]) if index < len(times) else None,
                      "static_peak_db": _db(static_peak), "static_peak_range_m": float(ranges[static_bin]),
                      "moving_peak_db": _db(moving_peak), "static_to_moving_db": ratio_db}
            # With every target standing still the bodies themselves are the
            # zero-Doppler return (a fallen person lying down), and the ratio
            # says nothing about the room; without labels every frame counts.
            record["static_check"] = not label_frames or any(
                float(rows[index].get("speed_mps") or 0.) >= min_speed_mps
                for rows in label_frames.values() if index < len(rows))
            moving_by_range = moving.max(axis=0) if moving.size else np.zeros(len(ranges))
            for object_id, rows in label_frames.items():
                if index >= len(rows):
                    continue
                row = rows[index]
                summary = targets[object_id]
                if row["label"] != "in_view":
                    summary["not_in_view"] += 1
                    continue
                if float(row.get("speed_mps") or 0.) < min_speed_mps:
                    summary["skipped_static"] += 1
                    continue
                window = np.abs(ranges - float(row["range_m"])) <= range_window_m
                margin = _db(moving_by_range[window].max()) - _db(floor) if window.any() else -math.inf
                summary["checked"] += 1
                if margin >= presence_db:
                    summary["present"] += 1
                else:
                    summary["missing_frames"].append({"frame": index, "range_m": float(row["range_m"]),
                                                      "margin_db": margin})
            frames.append(record)
    dominant = [item for item in frames
                if item["static_check"] and item["static_to_moving_db"] > static_dominant_db]
    if dominant:
        worst = max(dominant, key=lambda item: item["static_to_moving_db"])
        warnings.append({
            "code": "RADAR_STATIC_DOMINANT",
            "message": (f"A static return {worst['static_to_moving_db']:.1f} dB above the moving targets at "
                        f"{worst['static_peak_range_m']:.2f} m in {len(dominant)} of {len(frames)} frames."),
            "next_step": "Check the named range bin; consider another mount.",
            "frames": len(dominant),
            "static_peak_range_m": worst["static_peak_range_m"],
            "worst_static_to_moving_db": worst["static_to_moving_db"],
        })
    for object_id, summary in targets.items():
        summary["present_fraction"] = (summary["present"] / summary["checked"]) if summary["checked"] else None
    checks["targets_present_when_in_view"] = all(
        summary["present_fraction"] is None or summary["present_fraction"] >= PRESENCE_MIN_FRACTION
        for summary in targets.values()) if targets else None
    hard = [name for name, value in checks.items() if value is False]
    ratios = [item["static_to_moving_db"] for item in frames]
    return {
        "passed": not hard,
        "checks": checks,
        "failed_checks": hard,
        "warnings": warnings,
        "targets": targets,
        "static_to_moving_db": {"median": float(np.median(ratios)), "max": float(np.max(ratios))} if ratios else None,
        "static_check_skipped_frames": sum(not item["static_check"] for item in frames),
        "thresholds": {"static_dominant_db": static_dominant_db, "presence_db": presence_db,
                       "presence_min_fraction": PRESENCE_MIN_FRACTION, "min_speed_mps": min_speed_mps,
                       "range_window_m": range_window_m},
        "frames": frames,
    }


def check_saved_result(saved, *, labels=None, expected_frame_count=None, **thresholds):
    """Run the checks on a :class:`~wt_radar.adapter.saved_result.SavedRadarResult`."""
    return radar_output_checks(saved.cube, saved.axes, saved.times_s, labels=labels,
                               expected_frame_count=expected_frame_count, **thresholds)


__all__ = ["PRESENCE_DB", "STATIC_DOMINANT_DB", "check_saved_result", "radar_output_checks"]
