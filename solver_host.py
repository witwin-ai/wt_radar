"""Out-of-process Radar solver entry."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PLUGIN_PARENT = Path(__file__).resolve().parent.parent
if str(PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_PARENT))

import wt_radar  # noqa: E402,F401  (registers Radar components before Scene.from_dict)
from witwin_server.features.solvers.sdk import query, serve  # noqa: E402


def _as_numpy(value: Any, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def solve(ctx, scene, config):
    config = config or {}
    from wt_radar.adapter.animation import MODEL as ANIMATION_MODEL, solve_animation
    if config.get("adapter") == ANIMATION_MODEL:
        return solve_animation(ctx, scene, config)
    from wt_radar.adapter.snapshot import MODEL, solve_snapshot
    if config.get("adapter") == MODEL:
        return solve_snapshot(ctx, scene, config)
    adapter = str(config.get("adapter") or "<missing>")
    raise RuntimeError(
        f"Unsupported Radar solver adapter {adapter!r}. "
        "The legacy Radar 0.3 Scene/RadarConfig/trace path has been removed; "
        "submit an explicit Radar 0.4 snapshot or animation request."
    )


@query("snapshot_view")
def snapshot_view(ctx, result, params):
    from wt_radar.adapter.snapshot import SnapshotResult, snapshot_view as view
    if not isinstance(result, SnapshotResult):
        raise ValueError("Result is not a Radar 0.4 frozen-point snapshot")
    return view(result, params)


@query("animation_view")
def animation_view(ctx, result, params):
    from wt_radar.adapter.animation import AnimationResult, animation_view as view
    if not isinstance(result, AnimationResult):
        raise ValueError("Result is not a Studio skin animation")
    return view(result, params)


@query("animation_export")
def animation_export(ctx, result, params):
    from wt_radar.adapter.animation import AnimationResult, export_animation
    if not isinstance(result, AnimationResult):
        raise ValueError("Result is not a Studio skin animation")
    return export_animation(ctx, result)


@query("animation_replay")
def animation_replay(ctx, result, params):
    from wt_radar.adapter.animation import AnimationResult
    from wt_radar.adapter.replay import build_recording
    if not isinstance(result, AnimationResult):
        raise ValueError("Result is not a Studio skin animation")
    manifest, data = build_recording(result, int(params.get("tx", 0)), int(params.get("rx", 0)))
    return {"recording": manifest, "reference": ctx.result_ref(data, kind="radar.replay", extension=".bin")}


@query("animation_manifest")
def animation_manifest(ctx, result, params):
    """Small verification receipt for Agent orchestration; no RF/DSP work."""
    from wt_radar.adapter.animation import AnimationResult
    if not isinstance(result, AnimationResult):
        raise ValueError("Result is not a Studio skin animation")
    cube = result.cube
    frame_count = int(len(result.times_s))
    site_count = int(len(result.metadata.get("site_vertex_indices") or []))
    topology = dict(result.metadata.get("topology_preflight") or {})
    declared_site_count = int(topology.get("declared_site_count", -1))
    active_site_count = int(topology.get("active_site_count", -1))
    occluded_site_count = int(topology.get("occluded_site_count", -1))
    active_site_ids = [int(value) for value in (topology.get("active_site_ids") or [])]
    diagnostics = list(result.metadata.get("frame_diagnostics") or [])
    positions = np.asarray(result.positions_m)
    velocities = np.asarray(result.velocities_mps)
    cube_values = _as_numpy(cube)
    complete_site_shapes = (
        positions.shape == (frame_count, site_count, 3)
        and velocities.shape == (frame_count, site_count, 3)
    )
    times = np.asarray(result.times_s, dtype=np.float64)
    nonzero_return_frames = sum(bool(row.get("nonzero_return")) for row in diagnostics)
    dynamic_delay_rate_frames = sum(
        float(row.get("delay_rate_abs_max") or 0.0) > 0.0 for row in diagnostics
    )
    coupled_dynamic_return_frames = sum(
        bool(row.get("nonzero_return"))
        and float(row.get("delay_rate_abs_max") or 0.0) > 0.0
        for row in diagnostics
    )
    environment = dict(result.metadata.get("environment_reflection") or {})
    return {
        "frame_count": frame_count,
        "cube_shape": [int(value) for value in cube.shape],
        "all_finite": bool(np.isfinite(cube_values).all()),
        "signal_nonzero": bool(np.any(np.abs(cube_values) > 0)),
        "signal_abs_max": float(np.max(np.abs(cube_values))) if cube_values.size else 0.0,
        "site_count": site_count,
        "declared_site_count": declared_site_count,
        "active_site_count": active_site_count,
        "occluded_site_count": occluded_site_count,
        "active_site_ids": active_site_ids,
        "visibility_coverage": float(topology.get("visibility_coverage", 0.0)),
        "visibility_quality": str(topology.get("visibility_quality") or "unknown"),
        "rcs_policy": str(topology.get("rcs_policy") or ""),
        "rcs_per_site_m2": float(result.metadata.get("rcs_per_site_m2", float("nan"))),
        "frame_diagnostics_count": len(diagnostics),
        "no_zero_fill_or_dropped_active_sites": bool(
            len(diagnostics) == frame_count and complete_site_shapes
            and site_count == active_site_count == len(active_site_ids)
        ),
        "timebase_finite": bool(times.shape == (frame_count,) and np.isfinite(times).all()),
        "motion_arrays_finite": bool(
            complete_site_shapes and np.isfinite(positions).all() and np.isfinite(velocities).all()
        ),
        "motion_speed_max_mps": float(np.linalg.norm(velocities, axis=-1).max())
        if complete_site_shapes and velocities.size else 0.0,
        "moving_site_count": int(np.count_nonzero(
            np.linalg.norm(velocities, axis=-1).max(axis=0) > 1e-5
        )) if complete_site_shapes and velocities.size else 0,
        "motion_position_extent_m": float(np.linalg.norm(
            positions - positions[:1], axis=-1
        ).max()) if complete_site_shapes and positions.size else 0.0,
        "nonzero_return_frames": nonzero_return_frames,
        "dynamic_delay_rate_frames": dynamic_delay_rate_frames,
        "coupled_dynamic_return_frames": coupled_dynamic_return_frames,
        "environment_reflection_model": str(environment.get("model") or ""),
        "environment_reflection_max_depth": int(environment.get("max_depth", -1)),
        "environment_reflected_path_count": int(environment.get("reflected_path_count", -1)),
        "environment_material_slot_count": int(environment.get("material_slot_count", -1)),
        "environment_coherent_with_target": bool(environment.get("coherent_with_target")),
        "solver_completion_contract": "atomic_active_sites_no_zero_fill_v2",
        "device": str(result.metadata.get("device") or ""),
        "input_fingerprint": result.metadata.get("input_fingerprint"),
        "scene_id": result.metadata.get("scene_id"),
        "model": result.metadata.get("model"),
        "versions": dict(result.metadata.get("versions") or {}),
        "times_s": times.tolist(),
    }


if __name__ == "__main__":
    serve(solve)
