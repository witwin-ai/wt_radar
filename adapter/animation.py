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
from .studio_motion import StudioSkinSampler
from .memory_budget import MAX_RESULT_BYTES, require_result_memory
from .timebase import aligned_frame_count

MODEL = "studio_visible_skinned_surface_sites_v2"


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
    request = snapshot_request(component)
    request.update(adapter=MODEL, duration_s=float(component.animation_duration_s),
                   fps=float(component.animation_fps))
    fingerprint = str(getattr(component, "_agent_input_fingerprint", "") or "")
    if not fingerprint and getattr(component, "scene", None) is not None:
        # Component-button and Agent execution must publish the same immutable
        # result identity.  Import lazily to keep adapter/module registration
        # acyclic while reusing the one canonical authored-input contract.
        from ..agent_tools import _measurement_input_fingerprint
        fingerprint = _measurement_input_fingerprint(component.scene, component.owner)
    if fingerprint:
        request["input_fingerprint"] = fingerprint
    frame_times(request, component)
    return request


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
    *, stable_site_ids=None, environment_cube=None,
):
    """Simulate one Studio frame through Radar 0.4's public moving-target API."""
    from witwin.radar import Motion, PointTargets

    p = torch.as_tensor(positions, dtype=torch.float32, device=radar.device).contiguous()
    v = torch.as_tensor(velocities, dtype=torch.float32, device=radar.device).contiguous()
    if environment_cube is not None:
        raise ValueError("Radar 0.4 owns coherent reflections; a separate environment cube is not accepted.")
    speed = torch.linalg.vector_norm(v, dim=1)
    max_speed = float(radar.system_config.waveform_spec().max_unambiguous_speed_mps)
    if speed.numel() and float(speed.max()) > max_speed:
        raise ValueError(
            f"Target speed exceeds Radar's {max_speed:.6g} m/s unambiguous velocity; "
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
        reflections=1,
        motion=Motion.chirp(),
    )
    primal = simulation.cube.detach()
    if primal.device.type != "cuda" or not torch.isfinite(primal).all():
        raise RuntimeError(f"Nonfinite native CUDA frame at t={time_s:.6f}s; no zero-fill or discarded sites.")
    processed = processing_cube(simulation, radar)
    radar_position = torch.as_tensor(radar.position, dtype=p.dtype, device=p.device)
    displacement = p - radar_position
    distance = torch.linalg.vector_norm(displacement, dim=1).clamp_min(1e-12)
    radial_speed = (v * displacement).sum(dim=1) / distance
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
    sampler = StudioSkinSampler(scene, str(component.snapshot_target_id).strip())
    from .replay import motion_fingerprint
    authored_motion_fingerprint = motion_fingerprint(scene)
    if str(sensor.id) in sampler.moving_ids:
        raise ValueError("Radar must be static and outside the animated target hierarchy.")
    animated_ids = {track.object_id for track in sampler.tracks}
    for obj in scene.objects.values():
        if obj is sampler.target:
            continue
        chain, ancestor = {str(obj.id)}, obj
        while ancestor.parent_id:
            chain.add(str(ancestor.parent_id))
            ancestor = scene.get_object(ancestor.parent_id)
        has_mesh = obj.get_component("Mesh") is not None
        if ((has_mesh or obj is sensor) and chain & animated_ids) or (has_mesh and str(sampler.target.id) in chain):
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


def animation_preflight(scene, request):
    """Return solve evidence after native topology discovery, before synthesis."""
    times, sampler, radar, world, _config, meta, _motion = prepare_animation(scene, request)
    positions = [sampler.positions(float(time_s)) for time_s in times]
    topology = visibility_preflight(
        radar, world, positions, sampler,
        polarization=meta["world_polarization"],
        times_s=times,
    )
    return {
        "frame_count": int(len(times)),
        "site_count": int(topology["active_site_count"]),
        "declared_site_count": int(topology["declared_site_count"]),
        "moving_object_ids": sorted(str(value) for value in sampler.moving_ids),
        "radar_device": str(radar.device),
        "radar_position_m": list(meta["radar_position_m"]),
        "radar_target_m": list(meta["radar_target_m"]),
        "model": MODEL,
        "topology": topology,
    }


def topology_preflight(radar, world, positions, sampler, *, polarization, time_s):
    """Single-frame compatibility wrapper around interval visibility preflight."""
    return visibility_preflight(
        radar, world, [positions], sampler,
        polarization=polarization, times_s=[time_s],
    )


def visibility_preflight(radar, world, positions_by_frame, sampler, *, polarization, times_s):
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
                reflections=1,
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
    }


def solve_animation(ctx, scene, request):
    from witwin.radar import Motion, PointTargets

    ctx.progress(0.01, "Preparing Radar scene")
    times, sampler, radar, world, config, meta, authored_motion_fingerprint = prepare_animation(scene, request)
    with progress_heartbeat(ctx, 0.03, "Sampling authored cat motion"):
        sampled = [sampler.sample(float(time_s)) for time_s in times]
        poses = np.stack([sample[0] for sample in sampled])
        velocities = np.stack([sample[1] for sample in sampled])
    with progress_heartbeat(ctx, 0.08, "Checking room reflections and motion visibility"):
        topology = visibility_preflight(
            radar, world,
            poses,
            sampler,
            polarization=meta["world_polarization"],
            times_s=times,
        )
    active_ranks = np.asarray(topology["active_site_ranks"], dtype=np.int64)
    active_site_ids = tuple(topology["active_site_ids"])
    rcs_per_site = meta["rcs_m2"] / topology["declared_site_count"]
    for key in ("snapshot_time_s", "target_local_point", "target_world_point_m", "velocity_m_per_s"):
        meta.pop(key, None)
    meta.update(model=MODEL, motion_fingerprint=authored_motion_fingerprint, frame_count=len(times), fps=float(request["fps"]),
                duration_s=float(request["duration_s"]),
                declared_site_vertex_indices=sampler.vertex_ids.tolist(),
                declared_site_bone_names=sampler.site_names,
                site_vertex_indices=sampler.vertex_ids[active_ranks].tolist(),
                site_bone_names=[sampler.site_names[rank] for rank in active_ranks],
                active_site_ids=list(active_site_ids),
                topology_preflight=topology, rcs_per_site_m2=rcs_per_site,
                components=["los", "reflection"], max_depth=1,
                environment_reflection={
                    "model": "radar04_native_single_bounce",
                    "max_depth": 1,
                    "coherent_with_target": True,
                },
                velocity_method="Studio LINEAR timeline + CPU skinning; finite difference <=1ms, right-hand at knots",
                slow_time_model="Radar 0.4 public chirp-time motion sampling per Studio frame",
                limitations="Uncalibrated equal-RCS surface samples, not full electromagnetic skin. "
                            "Only sites with native inbound+outbound paths throughout the interval are active; "
                            "occluded sites keep their original RCS share and are not renormalized. "
                            "Static room geometry contributes native single-bounce specular reflections; "
                            "no diffuse or multi-bounce room clutter and no cat self-occlusion. "
                            "Frozen weights within each CPI; no acceleration/range migration within CPI.",
                versions={name: version(name) for name in ("witwin-radar", "witwin-channel", "witwin")})
    if request.get("input_fingerprint"):
        meta["input_fingerprint"] = str(request["input_fingerprint"])
    ctx.log(f"Studio skin animation model: {meta}")
    cubes = None
    poses = poses[:, active_ranks]
    velocities = velocities[:, active_ranks]
    speed = np.linalg.norm(velocities, axis=2)
    max_speed = float(radar.system_config.waveform_spec().max_unambiguous_speed_mps)
    if speed.size and float(speed.max()) > max_speed:
        raise ValueError(
            f"Target speed exceeds Radar's {max_speed:.6g} m/s unambiguous velocity; "
            "lower the motion speed or change radar timing explicitly."
        )
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
        reflections=1,
        motion=Motion.chirp(rediscover_every_frames=1),
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
        except Exception as exc:
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
    np.savez_compressed(stream, cube=result.cube.numpy(), times_s=result.times_s,
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
