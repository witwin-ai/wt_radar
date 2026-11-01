"""Studio baked skin motion -> native Radar 0.4, with no RF/DSP reimplementation."""
from dataclasses import asdict, dataclass, replace
from contextlib import contextmanager
from importlib.metadata import version
import io
import hashlib
import json
from pathlib import Path
import re
import threading
import uuid

import numpy as np
import torch

from .snapshot import (MODEL as SNAPSHOT_MODEL, SnapshotResult, prepare_snapshot,
                       processing_cube, snapshot_request, snapshot_view)
from .studio_motion import (
    MAX_TARGET_SITE_BUDGET, MultiTargetSkinSampler, StudioSkinSampler,
    resolved_target_site_budget,
)
from .target_contract import component_target_specs
from .memory_budget import MAX_RESULT_BYTES, require_result_memory
from .timebase import aligned_frame_count

MODEL = "studio_visible_skinned_surface_sites_v2"
TARGET_MOTION_SAMPLING = frozenset({"adaptive", "chirp"})


@contextmanager
def progress_heartbeat(ctx, fraction, message, *, interval_s=20.0):
    """Keep Studio's solver watchdog informed during one blocking native call."""
    stopped = threading.Event()

    def publish():
        while not stopped.wait(interval_s):
            ctx.progress(fraction, message)

    ctx.progress(fraction, message)
    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=1.0)


class AnimationTopologyError(ValueError):
    """A declared skin site is unreachable in native Radar topology discovery."""

    def __init__(self, message, *, detail):
        super().__init__(message)
        self.detail = detail


def animation_request(component):
    targets = component_target_specs(component)
    site_budget = sum(spec["site_budget"] for spec in targets)
    request = snapshot_request(component)
    request.update(adapter=MODEL, duration_s=float(component.animation_duration_s),
                   fps=float(component.animation_fps),
                   target_multipath=bool(component.target_multipath),
                   target_motion_sampling=str(component.target_motion_sampling),
                   target_site_budget=site_budget, targets=targets)
    fingerprint = str(getattr(component, "_agent_input_fingerprint", "") or "")
    if not fingerprint and getattr(component, "scene", None) is not None:
        # Component-button and Agent execution must publish the same immutable
        # result identity.  Import lazily to keep adapter/module registration
        # acyclic while reusing the one canonical authored-input contract.
        from ..agent_tools import _measurement_input_fingerprint
        fingerprint = _measurement_input_fingerprint(component.scene, component.owner)
    if fingerprint:
        request["input_fingerprint"] = fingerprint
    native_preflight = getattr(component, "_agent_native_preflight", None)
    if isinstance(native_preflight, dict):
        # The Agent binds this evidence to the same authored-input fingerprint
        # immediately before submission. JSON-normalize it so the immutable
        # solver request owns its copy after the transient component field is
        # removed.
        request["native_preflight"] = json.loads(json.dumps(native_preflight))
    fidelity = getattr(component, "_motion_fidelity_provenance", None)
    if fidelity is not None:
        from .fidelity_ab import normalize_fidelity_provenance
        request["motion_fidelity"] = normalize_fidelity_provenance(fidelity)
    frame_times(request, component)
    return request


def target_reflection_depth(request):
    """Return the explicitly authored dynamic-target multipath depth."""
    value = request.get("target_multipath", False)
    if not isinstance(value, bool):
        raise ValueError("target_multipath must be a boolean.")
    return 1 if value else 0


def target_motion_sampling(request):
    """Return the explicit Radar 0.4 intra-frame target motion policy."""
    value = str(request.get("target_motion_sampling", "chirp"))
    if value not in TARGET_MOTION_SAMPLING:
        raise ValueError(
            f"target_motion_sampling must be one of {sorted(TARGET_MOTION_SAMPLING)}, got {value!r}."
        )
    return value


def target_motion_policy(Motion, request):
    mode = target_motion_sampling(request)
    return Motion.adaptive() if mode == "adaptive" else Motion.chirp(rediscover_every_frames=1)


def target_site_budget(request):
    value = request.get("target_site_budget", 24)
    if (isinstance(value, (bool, np.bool_)) or int(value) != value
            or not 1 <= int(value) <= MAX_TARGET_SITE_BUDGET):
        raise ValueError(
            f"target_site_budget must be an integer from 1 to {MAX_TARGET_SITE_BUDGET}."
        )
    return int(value)


def frame_times(request, component):
    start, duration, fps = (float(request[key]) for key in ("time_s", "duration_s", "fps"))
    if (not np.isfinite([start, duration, fps]).all() or start < 0 or duration <= 0
            or not 1 <= fps <= 30 or duration > 30):
        raise ValueError("Animation requires start >= 0, duration (0,30] seconds and FPS [1,30].")
    count = aligned_frame_count(duration, fps)
    if count is None:
        raise ValueError(f"Duration × FPS must be an integer of at least two frames. "
                         f"Actual duration={duration:.12g}, FPS={fps:.12g}, frames={duration * fps:.12g}; "
                         "re-enter exact values (the UI may round their display).")
    size = count * int(component.num_tx) * int(component.num_rx) * int(component.chirp_per_frame) * int(component.adc_samples) * 8
    require_result_memory(size)
    manager = component.scene.timeline_manager
    if manager is None or start + duration > float(manager.clip.duration) + 1e-7:
        raise ValueError("Requested interval extends beyond the baked Studio timeline.")
    return start + np.arange(count, dtype=np.float64) / fps


def solve_native_frame(
    radar, world, rcs_m2, time_s, positions, velocities, polarization,
    *, stable_site_ids=None, environment_cube=None, target_reflections=0,
    target_motion_sampling_mode="chirp",
):
    """Simulate one Studio frame through Radar 0.4's public moving-target API."""
    from witwin.radar import Motion, PointTargets

    p = torch.as_tensor(positions, dtype=torch.float32, device=radar.device).contiguous()
    v = torch.as_tensor(velocities, dtype=torch.float32, device=radar.device).contiguous()
    if environment_cube is not None:
        raise ValueError("Radar 0.4 owns coherent reflections; a separate environment cube is not accepted.")
    radar_position = torch.as_tensor(radar.position, dtype=p.dtype, device=p.device)
    displacement = p - radar_position
    distance = torch.linalg.vector_norm(displacement, dim=1)
    if distance.numel() and bool((distance <= 1e-12).any()):
        raise ValueError("Radar position intersects an authored target site.")
    radial_speed = (v * displacement).sum(dim=1) / distance.clamp_min(1e-12)
    max_speed = float(radar.system_config.waveform_spec().max_unambiguous_speed_mps)
    if radial_speed.numel() and float(radial_speed.abs().max()) > max_speed:
        raise ValueError(
            f"Target radial speed exceeds Radar's {max_speed:.6g} m/s unambiguous velocity; "
            "lower the motion speed or change radar timing explicitly."
        )

    base_time = float(time_s)

    def trajectory(sample_time):
        return p + v * (float(sample_time) - base_time)

    targets = PointTargets(
        positions=p,
        rcs=float(rcs_m2),
        trajectory=trajectory,
        ids=(None if stable_site_ids is None else tuple(int(value) for value in stable_site_ids)),
    )
    simulation = radar.simulate(
        world,
        targets,
        times=(base_time,),
        los=True,
        reflections=int(target_reflections),
        motion=target_motion_policy(Motion, {"target_motion_sampling": target_motion_sampling_mode}),
    )
    primal = simulation.cube.detach()
    if primal.device.type != "cuda" or not torch.isfinite(primal).all():
        raise RuntimeError(f"Nonfinite native CUDA frame at t={time_s:.6f}s; no zero-fill or discarded sites.")
    processed = processing_cube(simulation, radar)
    delay_rate_abs_max = float((2.0 * radial_speed.abs() / 299_792_458.0).max()) if p.numel() else 0.0
    return processed, {
        "delay_rate_abs_max": delay_rate_abs_max,
        "path_count": int(simulation.last_radar_paths.path_count),
        "sample_count": len(simulation.sample_times_s[0]),
        "path_set_complete": bool(simulation.path_set_complete),
        "motion_sampling": simulation.motion_sampling,
        "chirp_change_max": float((primal - primal[..., :1, :]).abs().max()),
        "nonzero_return": bool((primal.abs() > 0).any()),
    }


@dataclass
class AnimationResult:
    cube: object
    axes: object
    times_s: object
    positions_m: object
    velocities_mps: object
    metadata: dict


def prepare_animation(scene, request):
    """Run the exact non-simulating checks shared by Agent preflight and solve."""
    from .radar04 import build_radar

    if request.get("adapter") != MODEL or str(scene.scene_id) != request.get("scene_id"):
        raise ValueError("Animation request and solver scene identity disagree.")
    sensor = scene.get_object(request["radar_object_id"])
    component = sensor.get_component("Radar") if sensor else None
    if component is None:
        raise ValueError("Radar object no longer exists.")
    times = frame_times(request, component)
    targets = request.get("targets") or component_target_specs(component)
    sampler = MultiTargetSkinSampler(scene, targets)
    from .replay import motion_fingerprint
    authored_motion_fingerprint = motion_fingerprint(scene)
    if str(sensor.id) in sampler.moving_ids:
        raise ValueError("Radar must be static and outside the animated target hierarchy.")
    animated_ids = {track.object_id for track in sampler.tracks}
    for obj in scene.objects.values():
        if obj in sampler.targets:
            continue
        chain, ancestor = {str(obj.id)}, obj
        while ancestor.parent_id:
            chain.add(str(ancestor.parent_id))
            ancestor = scene.get_object(ancestor.parent_id)
        has_mesh = obj.get_component("Mesh") is not None
        if ((has_mesh or obj is sensor) and chain & animated_ids) or (
            has_mesh and any(target_id in chain for target_id in sampler.target_ids)
        ):
            raise ValueError(f"Animated/attached geometry outside the selected skin is unsupported: {obj.id}")
    # The static export also validates legacy settings, material support and pose.
    world, config, meta = prepare_snapshot(scene, {**request, "adapter": SNAPSHOT_MODEL})
    if not torch.cuda.is_available():
        raise RuntimeError("Animation simulation requires CUDA; no CPU fallback.")
    radar = build_radar(
        config,
        device="cuda",
        position=meta["radar_position_m"],
        look_at=meta["radar_target_m"],
        up=meta["radar_up"],
        polarization=meta["world_polarization"],
    )
    spec = radar.system_config.waveform_spec()
    # These are output sampling intervals, not playback FPS. Refuse overlapping
    # CPIs in this first interface rather than silently change radar timing.
    cpi = spec.num_chirps * spec.num_tx * spec.chirp_period_s
    if cpi > 1 / float(request["fps"]):
        raise ValueError("Radar CPI exceeds requested frame spacing; lower Animation FPS explicitly.")
    return times, sampler, radar, world, config, meta, authored_motion_fingerprint


def sample_authored_motion(times, sampler):
    """Sample positions and velocities once at the authored output frames."""
    sampled = [sampler.sample(float(time_s)) for time_s in times]
    return (
        np.stack([sample[0] for sample in sampled]),
        np.stack([sample[1] for sample in sampled]),
    )


def validate_unambiguous_velocity(radar, positions, velocities):
    """Fail before topology discovery when authored radial motion must alias."""
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if positions.shape != velocities.shape or positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("Authored target positions and velocities must have matching frame/site/XYZ shape.")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("Authored target motion contains non-finite position or velocity samples.")
    displacement = positions - np.asarray(radar.position, dtype=np.float64)
    distance = np.linalg.norm(displacement, axis=2)
    if (distance <= 1e-12).any():
        raise ValueError("Radar position intersects an authored target site.")
    radial_speed = np.sum(velocities * displacement, axis=2) / distance
    speed = np.linalg.norm(velocities, axis=2)
    observed_max = float(np.abs(radial_speed).max()) if radial_speed.size else 0.0
    supported_max = float(radar.system_config.waveform_spec().max_unambiguous_speed_mps)
    if observed_max > supported_max:
        raise ValueError(
            f"Target radial speed {observed_max:.6g} m/s exceeds Radar's {supported_max:.6g} m/s "
            "unambiguous velocity; lower the motion speed or change radar timing explicitly."
        )
    return {
        "motion_speed_max_mps": float(speed.max()) if speed.size else 0.0,
        "radial_motion_speed_max_mps": observed_max,
        "max_unambiguous_speed_mps": supported_max,
    }


def animation_preflight(scene, request, *, sampled_motion=None):
    """Return solve evidence after native topology discovery, before synthesis.

    ``sampled_motion`` is ``(times_s, positions_m, velocities_mps)`` the caller
    already sampled for these targets (automatic placement does); it is reused
    only when its timebase and site count match this request exactly.
    """
    times, sampler, radar, world, _config, meta, _motion = prepare_animation(scene, request)
    positions = velocities = None
    if sampled_motion is not None:
        cached_times, cached_positions, cached_velocities = (
            np.asarray(value, dtype=np.float64) for value in sampled_motion)
        expected_shape = (len(times), len(sampler.vertex_ids), 3)
        if (cached_times.shape == np.asarray(times).shape
                and np.array_equal(cached_times, np.asarray(times, dtype=np.float64))
                and cached_positions.shape == expected_shape
                and cached_velocities.shape == expected_shape):
            positions, velocities = cached_positions, cached_velocities
    if positions is None:
        positions, velocities = sample_authored_motion(times, sampler)
    velocity_evidence = validate_unambiguous_velocity(radar, positions, velocities)
    topology = visibility_preflight(
        radar, world, positions, sampler,
        polarization=meta["world_polarization"],
        times_s=times,
        target_reflections=target_reflection_depth(request),
    )
    target_evidence = []
    active_ranks = set(topology["active_site_ranks"])
    for target in sampler.target_slices:
        target_evidence.append({
            **target,
            "active_site_count": sum(
                rank in active_ranks
                for rank in range(target["site_start"], target["site_end"])
            ),
        })
    return {
        "frame_count": int(len(times)),
        "site_count": int(topology["active_site_count"]),
        "declared_site_count": int(topology["declared_site_count"]),
        "moving_object_ids": sorted(str(value) for value in sampler.moving_ids),
        "targets": target_evidence,
        "radar_device": str(radar.device),
        "radar_position_m": list(meta["radar_position_m"]),
        "radar_target_m": list(meta["radar_target_m"]),
        "model": MODEL,
        "topology": topology,
        "input_fingerprint": str(request.get("input_fingerprint") or ""),
        "sampled_motion": {
            "times_s": np.asarray(times, dtype=np.float64).tolist(),
            "positions_m": positions.tolist(),
            "velocities_mps": velocities.tolist(),
        },
        **velocity_evidence,
    }


def topology_preflight(
    radar, world, positions, sampler, *, polarization, time_s, target_reflections=0,
):
    """Single-frame compatibility wrapper around interval visibility preflight."""
    return visibility_preflight(
        radar, world, [positions], sampler,
        polarization=polarization, times_s=[time_s], target_reflections=target_reflections,
    )


def visibility_preflight(
    radar, world, positions_by_frame, sampler, *, polarization, times_s,
    target_reflections=0,
):
    """Select sites reachable for the whole interval using native discovery.

    This deliberately does not ray-cast or approximate visibility in the
    Studio plugin. Channel remains the sole owner of path discovery. The
    interface keeps the intersection of sites with both native legs at every
    requested frame, preserves each original stable ID, and never redistributes
    an occluded site's RCS onto the visible sites.
    """
    from witwin.radar import Motion, PointTargets

    positions_by_frame = tuple(np.asarray(value, dtype=np.float64) for value in positions_by_frame)
    times_s = tuple(float(value) for value in times_s)
    if not positions_by_frame or len(positions_by_frame) != len(times_s):
        raise ValueError("Topology preflight requires one site array per requested frame.")
    declared_ids = tuple(3_000_000 + rank for rank in range(len(positions_by_frame[0])))
    if not tuple(getattr(world, "structures", (None,))):
        # Free space has no static geometry to occlude a leg, so every site
        # keeps both paths for the whole interval; target self-occlusion is not
        # modelled by either branch.
        return {
            "status": "reachable",
            "method": "free_space_no_static_occluders",
            "frame_count": len(times_s),
            "declared_site_count": len(declared_ids),
            "active_site_count": len(declared_ids),
            "occluded_site_count": 0,
            "active_site_ranks": list(range(len(declared_ids))),
            "active_site_ids": [int(value) for value in declared_ids],
            "occluded_sites": [],
            "frames": [{"frame_index": index, "time_s": float(time_s), "reachable_site_count": len(declared_ids),
                        "round_trip_rows": len(declared_ids), "path_set_complete": True}
                       for index, time_s in enumerate(times_s)],
            "rcs_policy": "total_rcs_divided_by_declared_sites_no_visible_renormalization",
            "visibility_coverage": 1.0,
            "visibility_quality": "complete",
            "target_reflections": int(target_reflections),
        }
    positions_by_frame = np.stack(positions_by_frame)
    timeline = np.asarray(times_s, dtype=np.float64)
    # Visibility is a propagation question at Studio frame instants, not a
    # waveform-synthesis question.  A one-chirp immutable Radar preserves the
    # carrier, arrays, pose, pattern and propagation request while making
    # Motion.chirp schedule exactly one observation per output frame.  Using
    # the authored chirp count here would repeat the same topology check up to
    # 128 times per frame before any synthesis is requested.
    preflight_radar = radar.replace(
        waveform=replace(radar.waveform, chirps_per_frame=1),
    )

    def trajectory_for(ranks):
        samples = positions_by_frame[:, ranks]

        def trajectory(sample_time):
            value = float(sample_time)
            if value <= timeline[0]:
                positions = samples[0]
            elif value >= timeline[-1]:
                positions = samples[-1]
            else:
                right = int(np.searchsorted(timeline, value, side="right"))
                left = right - 1
                weight = (value - timeline[left]) / (timeline[right] - timeline[left])
                positions = samples[left] + weight * (samples[right] - samples[left])
            return torch.as_tensor(positions, dtype=torch.float32, device=preflight_radar.device).contiguous()

        return trajectory

    # Radar 0.4 deliberately refuses a declared target that lacks either leg.
    # The exception names that stable ID, so remove only that interval-ineligible
    # site and retry the entire time sequence in one compiled session.  This
    # preserves the old full-interval intersection semantics without importing
    # Channel internals or recompiling the room once per frame.
    active_ranks = list(range(len(declared_ids)))
    excluded_ids = set()
    trace = None
    while active_ranks:
        active_ids = tuple(declared_ids[rank] for rank in active_ranks)
        initial = torch.as_tensor(
            positions_by_frame[0, active_ranks], dtype=torch.float32, device=radar.device,
        ).contiguous()
        try:
            trace = preflight_radar.trace(
                world,
                PointTargets(
                    positions=initial,
                    rcs=1.0,
                    ids=active_ids,
                    trajectory=trajectory_for(active_ranks),
                ),
                times=times_s,
                los=True,
                reflections=int(target_reflections),
                motion=Motion.chirp(rediscover_every_frames=1),
            )
            break
        except ValueError as exc:
            match = re.search(r"site (\d+) has no (?:inbound|outbound) leg row", str(exc))
            if match is None:
                raise
            missing_id = int(match.group(1))
            if missing_id not in active_ids:
                raise
            excluded_ids.add(missing_id)
            active_ranks = [rank for rank in active_ranks if declared_ids[rank] != missing_id]

    frame_diagnostics = []
    if trace is not None:
        for frame_index, time_s in enumerate(times_s):
            frame = trace.frame(frame_index)
            frame_diagnostics.append({
                "frame_index": frame_index,
                "time_s": float(time_s),
                "reachable_site_count": len(active_ranks),
                "round_trip_rows": int(frame.last_radar_paths.path_count),
                "path_set_complete": bool(frame.path_set_complete),
            })

    occluded_ranks = [rank for rank, site_id in enumerate(declared_ids) if site_id in excluded_ids]
    occluded_sites = [{
        "site_id": int(declared_ids[rank]),
        "site_rank": rank,
        "bone_name": str(sampler.site_names[rank]),
        "vertex_index": int(sampler.vertex_ids[rank]),
        "visible_frame_count": 0,
    } for rank in occluded_ranks]
    if not active_ranks:
        detail = {
            "code": "sensor_embedded_or_all_sites_occluded",
            "declared_site_count": len(declared_ids),
            "frame_count": len(times_s),
            "occluded_sites": occluded_sites,
            "frames": frame_diagnostics,
        }
        raise AnimationTopologyError(
            "Native Radar topology preflight found no skin site with both "
            "inbound and outbound paths throughout the requested interval. "
            "The Radar may be embedded in room geometry or the animal is fully "
            "occluded. No GPU waveform synthesis was submitted.",
            detail=detail,
        )
    return {
        "status": "reachable",
        "method": "radar04_public_trace_interval_visibility_intersection_no_synthesis",
        "frame_count": len(times_s),
        "declared_site_count": len(declared_ids),
        "active_site_count": len(active_ranks),
        "occluded_site_count": len(occluded_ranks),
        "active_site_ranks": active_ranks,
        "active_site_ids": [int(declared_ids[rank]) for rank in active_ranks],
        "occluded_sites": occluded_sites,
        "frames": frame_diagnostics,
        "rcs_policy": "total_rcs_divided_by_declared_sites_no_visible_renormalization",
        "visibility_coverage": len(active_ranks) / len(declared_ids),
        "visibility_quality": (
            "complete" if len(active_ranks) == len(declared_ids)
            else "degraded_interval_global_subset"
        ),
        "target_reflections": int(target_reflections),
    }


def validated_cached_topology(request, times, sampler, meta):
    """Validate topology evidence bound to this immutable solve request."""
    preflight = request.get("native_preflight")
    if preflight is None:
        return None
    if not isinstance(preflight, dict) or preflight.get("model") != MODEL:
        raise ValueError("Cached Radar preflight model does not match this animation adapter.")
    expected_fingerprint = str(request.get("input_fingerprint") or "")
    if expected_fingerprint and str(preflight.get("input_fingerprint") or "") != expected_fingerprint:
        raise ValueError("Cached Radar preflight fingerprint does not match this solve request.")
    topology = preflight.get("topology")
    if not isinstance(topology, dict) or topology.get("status") != "reachable":
        raise ValueError("Cached Radar preflight does not contain reachable topology evidence.")
    declared_count = int(topology.get("declared_site_count", -1))
    if declared_count != len(sampler.vertex_ids):
        raise ValueError("Cached Radar preflight site count does not match the selected skin.")
    if int(preflight.get("frame_count", -1)) != len(times):
        raise ValueError("Cached Radar preflight frame count does not match the requested interval.")
    if sorted(str(value) for value in preflight.get("moving_object_ids", ())) != sorted(
        str(value) for value in sampler.moving_ids
    ):
        raise ValueError("Cached Radar preflight moving-object set does not match the solver scene.")
    try:
        pose_matches = np.allclose(
            np.asarray(preflight.get("radar_position_m"), dtype=np.float64),
            np.asarray(meta["radar_position_m"], dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        ) and np.allclose(
            np.asarray(preflight.get("radar_target_m"), dtype=np.float64),
            np.asarray(meta["radar_target_m"], dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        )
    except (TypeError, ValueError):
        pose_matches = False
    if not pose_matches:
        raise ValueError("Cached Radar preflight pose does not match the solver scene.")
    active_ranks = [int(value) for value in topology.get("active_site_ranks", ())]
    if not active_ranks or len(active_ranks) != len(set(active_ranks)) or any(
        rank < 0 or rank >= declared_count for rank in active_ranks
    ):
        raise ValueError("Cached Radar preflight contains invalid active site ranks.")
    expected_ids = [3_000_000 + rank for rank in active_ranks]
    if [int(value) for value in topology.get("active_site_ids", ())] != expected_ids:
        raise ValueError("Cached Radar preflight stable site IDs do not match active ranks.")
    if int(topology.get("active_site_count", -1)) != len(active_ranks):
        raise ValueError("Cached Radar preflight active site count is inconsistent.")
    if int(topology.get("target_reflections", -1)) != target_reflection_depth(request):
        raise ValueError("Cached Radar preflight target multipath setting does not match this solve request.")
    if int(topology.get("occluded_site_count", -1)) != declared_count - len(active_ranks):
        raise ValueError("Cached Radar preflight occluded site count is inconsistent.")
    return json.loads(json.dumps(topology))


def validated_cached_motion(request, times, sampler):
    """Return structurally valid sampled motion from the bound preflight."""
    preflight = request.get("native_preflight")
    if not isinstance(preflight, dict):
        return None
    sampled = preflight.get("sampled_motion")
    if sampled is None:
        return None
    if not isinstance(sampled, dict):
        raise ValueError("Cached Radar preflight sampled motion is malformed.")
    cached_times = np.asarray(sampled.get("times_s"), dtype=np.float64)
    positions = np.asarray(sampled.get("positions_m"), dtype=np.float64)
    velocities = np.asarray(sampled.get("velocities_mps"), dtype=np.float64)
    expected_shape = (len(times), len(sampler.vertex_ids), 3)
    if (cached_times.shape != np.asarray(times).shape
            or not np.array_equal(cached_times, np.asarray(times, dtype=np.float64))
            or positions.shape != expected_shape
            or velocities.shape != expected_shape
            or not np.isfinite(positions).all()
            or not np.isfinite(velocities).all()):
        raise ValueError("Cached Radar preflight sampled motion does not match this solve request.")
    return positions, velocities


MAX_SOLVE_SITE_EXCLUSIONS = 16
_LEG_ROW_GAP = re.compile(r"site (\d+) has no (?:inbound|outbound) leg row")


class _SolveTopologyGap(Exception):
    """The native solve's frozen topology has no leg row for one active site."""

    def __init__(self, site_id, frame_index):
        super().__init__(f"site {site_id} lost at frame {frame_index}")
        self.site_id = int(site_id)
        self.frame_index = int(frame_index)


def _topology_gap_site(exc, active_site_ids):
    """Return the active site a native leg-row error names, if any."""
    active = {int(value) for value in active_site_ids}
    current, seen = exc, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        match = _LEG_ROW_GAP.search(str(current))
        if match and int(match.group(1)) in active:
            return int(match.group(1))
        current = current.__cause__ or current.__context__
    return None


def _exclude_solve_site(topology, sampler, site_id, frame_index):
    """Move one site from the active set to the interval-global occluded set."""
    rank = int(site_id) - 3_000_000
    remaining = [int(value) for value in topology["active_site_ranks"] if int(value) != rank]
    declared = int(topology["declared_site_count"])
    occluded_sites = list(topology.get("occluded_sites", ())) + [{
        "site_id": int(site_id),
        "site_rank": rank,
        "bone_name": str(sampler.site_names[rank]),
        "vertex_index": int(sampler.vertex_ids[rank]),
        "visible_frame_count": None,
        "excluded_by": "native_solve_frozen_topology",
    }]
    if not remaining:
        raise AnimationTopologyError(
            "The native solve lost every skin site from its frozen topology. "
            "No completed result published.",
            detail={"code": "sensor_embedded_or_all_sites_occluded", "declared_site_count": declared,
                    "occluded_sites": occluded_sites, "frames": topology.get("frames", [])},
        )
    return {
        **topology,
        "active_site_count": len(remaining),
        "occluded_site_count": declared - len(remaining),
        "active_site_ranks": remaining,
        "active_site_ids": [3_000_000 + value for value in remaining],
        "occluded_sites": occluded_sites,
        "visibility_coverage": len(remaining) / declared,
        "visibility_quality": "degraded_interval_global_subset",
        "solve_exclusions": list(topology.get("solve_exclusions", ())) + [
            {"site_id": int(site_id), "site_rank": rank, "frame_index": int(frame_index)},
        ],
    }


def solve_animation(ctx, scene, request):
    from witwin.radar import Motion, PointTargets
    from .environment_clutter import (
        add_environment_to_fmcw_result,
        single_bounce_environment_cube,
    )

    ctx.progress(0.01, "Preparing Radar scene")
    times, sampler, radar, world, config, meta, authored_motion_fingerprint = prepare_animation(scene, request)
    cached_motion = validated_cached_motion(request, times, sampler)
    motion_samples_reused = cached_motion is not None
    if cached_motion is None:
        with progress_heartbeat(ctx, 0.03, "Sampling authored target motion"):
            poses, velocities = sample_authored_motion(times, sampler)
    else:
        poses, velocities = cached_motion
        ctx.progress(0.03, "Reusing verified authored motion samples")
    velocity_evidence = validate_unambiguous_velocity(radar, poses, velocities)
    topology = validated_cached_topology(request, times, sampler, meta)
    preflight_reused = topology is not None
    if topology is None:
        with progress_heartbeat(ctx, 0.08, "Checking room reflections and motion visibility"):
            topology = visibility_preflight(
                radar, world,
                poses,
                sampler,
                polarization=meta["world_polarization"],
                times_s=times,
                target_reflections=target_reflection_depth(request),
            )
    else:
        ctx.progress(0.08, "Reusing verified room-reflection visibility")
    with progress_heartbeat(ctx, 0.1, "Compiling single-bounce room reflections"):
        environment = single_bounce_environment_cube(
            radar, world, polarization=meta["world_polarization"],
        )
    all_poses, all_velocities, base_meta = poses, velocities, meta

    def solve_active_sites(topology):
        meta = dict(base_meta)
        active_ranks = np.asarray(topology["active_site_ranks"], dtype=np.int64)
        active_site_ids = tuple(topology["active_site_ids"])
        rcs_per_site = meta["rcs_m2"] / topology["declared_site_count"]
        for key in ("snapshot_time_s", "target_local_point", "target_world_point_m", "velocity_m_per_s"):
            meta.pop(key, None)
        target_reflections = target_reflection_depth(request)
        motion_sampling = target_motion_sampling(request)
        target_components = ["los"] + (["reflection"] if target_reflections else [])
        active_rank_set = set(active_ranks.tolist())
        target_evidence = [{
            **target,
            "active_site_count": sum(
                rank in active_rank_set
                for rank in range(target["site_start"], target["site_end"])
            ),
        } for target in sampler.target_slices]
        meta.update(model=MODEL, motion_fingerprint=authored_motion_fingerprint, frame_count=len(times), fps=float(request["fps"]),
                    duration_s=float(request["duration_s"]),
                    targets=target_evidence,
                    declared_site_vertex_indices=sampler.vertex_ids.tolist(),
                    declared_site_bone_names=sampler.site_names,
                    available_bone_site_count=sampler.available_site_count,
                    target_site_budget=sampler.site_budget,
                    site_selection="anatomical_priority_then_bind_pose_coverage",
                    site_vertex_indices=sampler.vertex_ids[active_ranks].tolist(),
                    site_bone_names=[sampler.site_names[rank] for rank in active_ranks],
                    active_site_ids=list(active_site_ids),
                    topology_preflight=topology, rcs_per_site_m2=rcs_per_site,
                    components=["los", "reflection"], max_depth=1,
                    target_propagation={
                        "components": target_components,
                        "max_depth": target_reflections,
                        "multipath_enabled": bool(target_reflections),
                        "motion_sampling": motion_sampling,
                    },
                    environment_reflection={
                        "model": "radar04_native_single_bounce",
                        "max_depth": 1,
                        "reflected_path_count": environment.reflected_path_count,
                        "material_slot_count": environment.material_slot_count,
                        "coherent_with_target": True,
                    },
                    native_preflight_reused=preflight_reused,
                    native_motion_samples_reused=motion_samples_reused,
                    **velocity_evidence,
                    velocity_method="Studio LINEAR timeline + CPU skinning; finite difference <=1ms, right-hand at knots",
                    slow_time_model=f"Radar 0.4 public {motion_sampling} motion sampling per Studio frame",
                    limitations="Uncalibrated equal-RCS surface samples, not full electromagnetic skin. "
                                "Only sites with native inbound+outbound paths throughout the interval are active; "
                                "occluded sites keep their original RCS share and are not renormalized. "
                                "Static room geometry contributes native single-bounce specular reflections; "
                                + ("dynamic target-related single-bounce multipath is enabled; "
                                   if target_reflections else "dynamic target-related multipath is disabled; ")
                                + "no diffuse or multi-bounce room clutter and no cat self-occlusion. "
                                "Frozen weights within each CPI; no acceleration/range migration within CPI.",
                    versions={name: version(name) for name in ("witwin-radar", "witwin-channel", "witwin")})
        if request.get("input_fingerprint"):
            meta["input_fingerprint"] = str(request["input_fingerprint"])
        from .fidelity_ab import attach_fidelity_provenance
        meta = attach_fidelity_provenance(
            meta, request,
            authored_motion_fingerprint=authored_motion_fingerprint,
        )
        ctx.log(f"Studio skin animation model: {meta}")
        cubes = None
        poses = all_poses[:, active_ranks]
        velocities = all_velocities[:, active_ranks]
        timeline = np.asarray(times, dtype=np.float64)

        def trajectory(sample_time):
            value = float(sample_time)
            if value <= timeline[0]:
                index = 0
            elif value >= timeline[-1]:
                index = len(timeline) - 1
            else:
                index = int(np.searchsorted(timeline, value, side="right")) - 1
            position = poses[index] + velocities[index] * (value - timeline[index])
            return torch.as_tensor(position, dtype=torch.float32, device=radar.device).contiguous()

        targets = PointTargets(
            positions=torch.as_tensor(poses[0], dtype=torch.float32, device=radar.device).contiguous(),
            rcs=float(rcs_per_site),
            trajectory=trajectory,
            ids=active_site_ids,
        )
        stream = iter(radar.stream(
            world,
            targets,
            times=times,
            los=True,
            reflections=target_reflections,
            motion=target_motion_policy(Motion, request),
        ))
        diagnostics = []
        result = None
        for index, time_s in enumerate(times):
            ctx.throw_if_cancelled()
            try:
                pending_fraction = 0.15 + 0.8 * index / len(times)
                with progress_heartbeat(
                    ctx,
                    pending_fraction,
                    f"Simulating Radar frame {index + 1}/{len(times)}",
                ):
                    simulation = next(stream)
                result = processing_cube(simulation, radar)
                result = replace(
                    result,
                    data=add_environment_to_fmcw_result(radar, result.data, environment),
                )
            except Exception as exc:
                gap_site = _topology_gap_site(exc, active_site_ids)
                if gap_site is not None:
                    raise _SolveTopologyGap(gap_site, index) from exc
                raise RuntimeError(f"Animation failed at frame {index}/{len(times)}, t={time_s:.6f}s: {exc}. "
                                   "No completed result published; native error retained.") from exc
            p, v = poses[index], velocities[index]
            radar_position = np.asarray(radar.position, dtype=np.float64)
            displacement = p - radar_position
            distance = np.linalg.norm(displacement, axis=1)
            radial_speed = np.sum(v * displacement, axis=1) / np.maximum(distance, 1e-12)
            stats = {
                "delay_rate_abs_max": float(np.max(2.0 * np.abs(radial_speed) / 299_792_458.0)),
                "path_count": int(simulation.last_radar_paths.path_count),
                "sample_count": len(simulation.sample_times_s[0]),
                "path_set_complete": bool(simulation.path_set_complete),
                "motion_sampling": simulation.motion_sampling,
                "chirp_change_max": float(
                    (simulation.cube - simulation.cube[..., :1, :]).abs().max()
                ),
                "nonzero_return": bool((simulation.cube.abs() > 0).any()),
                "compile_count": int(simulation.compile_count),
                "discovery_count": int(simulation.discovery_count),
            }
            if cubes is None:
                cubes = torch.empty((len(times), *result.data.shape), dtype=result.data.dtype, device="cpu")
            cubes[index].copy_(result.data)
            diagnostics.append(stats)
            completed_fraction = 0.15 + 0.8 * (index + 1) / len(times)
            ctx.progress(completed_fraction, f"GPU frame {index+1}/{len(times)} | Studio t={time_s:.3f}s")
        if result is None or cubes is None:
            raise RuntimeError("Radar animation produced no frames.")
        axes = replace(result.axes, range_m=result.axes.range_m.cpu(), velocity_mps=result.axes.velocity_mps.cpu())
        meta["frame_diagnostics"] = diagnostics
        meta["tx_positions_m"] = radar.tx_pos.detach().cpu().tolist()
        meta["rx_positions_m"] = radar.rx_pos.detach().cpu().tolist()
        meta["device"] = str(radar.device)
        return AnimationResult(cubes, axes, times, poses, velocities, meta)

    # The native solve is the final authority on reachability. Adaptive motion
    # freezes its topology on its own cadence, so a site the chirp preflight saw
    # at every Studio frame (a foot grazing the floor) can still lose its leg
    # row mid-solve. Drop that site for the whole interval, the same rule the
    # preflight applies, and restart rather than fail the measurement.
    while True:
        try:
            return solve_active_sites(topology)
        except _SolveTopologyGap as gap:
            if len(topology.get("solve_exclusions", ())) >= MAX_SOLVE_SITE_EXCLUSIONS:
                raise RuntimeError(
                    f"Animation failed: native solve lost more than {MAX_SOLVE_SITE_EXCLUSIONS} "
                    "skin sites from its frozen topology. No completed result published."
                ) from gap
            topology = _exclude_solve_site(topology, sampler, gap.site_id, gap.frame_index)
            ctx.progress(0.15, f"Excluding skin site {gap.site_id} lost by the native solve; restarting")


def animation_view(result, params):
    from witwin.radar.processing import ProcessingCube
    frame = int(params.get("frame", 0))
    if not 0 <= frame < len(result.times_s):
        raise ValueError(f"Animation frame must be 0..{len(result.times_s)-1}.")
    meta = {key: value for key, value in result.metadata.items() if key != "frame_diagnostics"}
    meta.update(frame_index=frame, time_s=float(result.times_s[frame]),
                frame_diagnostics=result.metadata["frame_diagnostics"][frame])
    payload = snapshot_view(SnapshotResult(ProcessingCube(result.cube[frame], result.axes), meta), params)
    payload["title"] = f"Studio skin motion | frame {frame}/{len(result.times_s)-1} | t={result.times_s[frame]:.3f}s"
    return payload


def export_animation(ctx, result):
    axes = asdict(result.axes)
    for key in ("range_m", "velocity_mps"):
        axes[key] = axes[key].tolist()
    stream = io.BytesIO()
    np.savez(stream, cube=result.cube.numpy(), times_s=result.times_s,
             positions_m=result.positions_m, velocities_mps=result.velocities_mps,
             result_schema_version=np.asarray(1), processing_axes_json=json.dumps(axes),
             producer_metadata_json=json.dumps(result.metadata, allow_nan=False))
    return ctx.result_ref(stream.getvalue(), kind="radar.animation", extension=".npz",
                          metadata={"frameCount": len(result.times_s), "model": MODEL})


def persist_export(api, reference, *, existing_path=None):
    """Validate the solver blob and copy it to a user-owned, unique project file.

    Solver result refs are disposable on host shutdown, not permanent exports.
    Exclusive creation never overwrites an existing experiment or result.
    """
    project = Path(api.server.default_scene_dir).resolve()
    folder = project / "results" / "radar-animation"
    if not folder.resolve().is_relative_to(project):
        raise ValueError("Animation export directory escapes the current project.")
    if existing_path:
        existing = Path(existing_path).resolve()
        if not existing.is_relative_to(folder.resolve()) or existing.suffix.lower() != '.npz':
            raise ValueError('Existing export is outside the Project Radar results directory.')
        if existing.is_file():
            # Verify against the reference bound to this solver run, not merely
            # the displayed filename. The disposable solver blob is not needed.
            digest = hashlib.sha256()
            size = 0
            with existing.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
                    size += len(chunk)
            if size != reference.get('size'):
                raise ValueError('Existing export size changed; the file was not overwritten.')
            if f'sha256:{digest.hexdigest()}' != reference.get('contentHash'):
                raise ValueError('Existing export checksum changed; the file was not overwritten.')
            return str(existing)
    payload = api.solvers.read_result_ref("witwin.radar.simulate", reference)
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"studio-animation-{uuid.uuid4().hex}.npz"
    with destination.open("xb") as stream:
        stream.write(payload)
    return str(destination)


def show_animation(component):
    params = animation_view_params(component)
    payload = component._query_result("animation_view", params)
    return apply_animation_view(component, payload)


def animation_view_params(component):
    return {"view": str(component.view), "tx": int(component.tx_index), "rx": int(component.rx_index),
              "frame": int(component.animation_frame_index),
              "static_clutter_removal": bool(component.static_clutter_removal), "show_cfar": bool(component.show_cfar)}


def apply_animation_view(component, payload):
    from .replay import numeric_plot
    component._last_animation_view_metadata = dict(payload.get("metadata") or {})
    component.signal_figure.set_plot_data(numeric_plot(payload))
    return payload["title"]
