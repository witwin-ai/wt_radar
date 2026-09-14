"""Studio baked skin motion -> native Radar 0.3, with no RF/DSP reimplementation."""
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
import io
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import torch
from torch.autograd import forward_ad

from .snapshot import (MODEL as SNAPSHOT_MODEL, SnapshotResult, prepare_snapshot,
                       processing_cube, snapshot_request, snapshot_view)
from .studio_motion import StudioSkinSampler
from .memory_budget import MAX_RESULT_BYTES, require_result_memory
from .timebase import aligned_frame_count

MODEL = "studio_visible_skinned_surface_sites_v2"


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
    radar, world, response, time_s, positions, velocities, polarization,
    *, stable_site_ids=None,
):
    from witwin.radar.propagation import Kinematics, two_way_duals
    from witwin.radar.simulation import ScatterSitePolicy

    p = torch.as_tensor(positions, dtype=torch.float32, device=radar.device).contiguous()
    v = torch.as_tensor(velocities, dtype=torch.float32, device=radar.device).contiguous()
    with two_way_duals(sites=Kinematics(positions_m=p, velocities_m_per_s=v)) as duals:
        simulation = radar.simulate(world, times=(float(time_s),), response=response,
                                    sites=ScatterSitePolicy.explicit(
                                        duals.sites,
                                        stable_ids=(None if stable_site_ids is None else tuple(stable_site_ids)),
                                    ), ad_mode="jvp",
                                    components=frozenset({"los"}), max_depth=0,
                                    polarization=tuple(polarization))
        primal = forward_ad.unpack_dual(simulation.cube).primal.detach().clone()
        rates = radar.last_radar_paths.delay_rate
        if rates is None:
            raise RuntimeError("Native Radar did not publish motion delay rates.")
        rates = forward_ad.unpack_dual(rates).primal.detach().clone()
    if primal.device.type != "cuda" or not torch.isfinite(primal).all() or not torch.isfinite(rates).all():
        raise RuntimeError(f"Nonfinite native CUDA frame at t={time_s:.6f}s; no zero-fill or discarded sites.")
    spec = radar.system_config.waveform_spec()
    if rates.numel() and float(rates.abs().max()) * spec.carrier_rate_hz >= .5 / (spec.num_tx * spec.chirp_period_s):
        raise ValueError(f"Doppler exceeds Nyquist at t={time_s:.6f}s; change radar timing explicitly.")
    processed = processing_cube(replace(simulation, cube=primal), radar)
    return processed, {"delay_rate_abs_max": float(rates.abs().max()) if rates.numel() else 0.,
                       "chirp_change_max": float((primal - primal[..., :1, :]).abs().max()),
                       "nonzero_return": bool((primal.abs() > 0).any())}


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
    from witwin.radar import Radar

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
    radar = Radar(config, device="cuda", position=meta["radar_position_m"],
                  target=meta["radar_target_m"], up=meta["radar_up"])
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
    from witwin.radar.channel import ChannelPropagationAdapter, compile_scene
    from witwin.radar.simulation import ScatterSitePolicy, bind_radar_world

    positions_by_frame = tuple(np.asarray(value, dtype=np.float64) for value in positions_by_frame)
    times_s = tuple(float(value) for value in times_s)
    if not positions_by_frame or len(positions_by_frame) != len(times_s):
        raise ValueError("Topology preflight requires one site array per requested frame.")
    solve_config = radar.system_config.with_propagation(
        components=frozenset({"los"}), max_depth=0,
    )
    propagation = solve_config.propagation
    compiled = compile_scene(
        world, reference_frequency_hz=propagation.reference_frequency_hz,
    )
    adapter = ChannelPropagationAdapter(
        compiled,
        reference_frequency_hz=propagation.reference_frequency_hz,
        components=propagation.components,
        max_depth=propagation.max_depth,
    )
    declared_ids = None
    active_ids = None
    visible_frame_counts = None
    frame_diagnostics = []
    for frame_index, (time_s, positions) in enumerate(zip(times_s, positions_by_frame)):
        site_positions = torch.as_tensor(
            positions, dtype=torch.float32, device=radar.device,
        ).contiguous()
        binding = bind_radar_world(
            radar,
            world,
            sites=ScatterSitePolicy.explicit(site_positions),
            polarization=tuple(polarization),
        )
        if declared_ids is None:
            declared_ids = tuple(binding.site_ids)
            active_ids = set(declared_ids)
            visible_frame_counts = {site_id: 0 for site_id in declared_ids}
        elif tuple(binding.site_ids) != declared_ids:
            raise RuntimeError("Native Radar site IDs changed across visibility preflight frames.")
        inbound = adapter.freeze(binding.transmitters, binding.site_sinks)
        outbound = adapter.freeze(binding.site_sources, binding.receivers)
        reachable_in = {int(value) for value in inbound.sink_id.detach().cpu().tolist()}
        reachable_out = {int(value) for value in outbound.source_id.detach().cpu().tolist()}
        reachable = set(declared_ids) & reachable_in & reachable_out
        active_ids &= reachable
        for site_id in reachable:
            visible_frame_counts[site_id] += 1
        frame_diagnostics.append({
            "frame_index": frame_index,
            "time_s": time_s,
            "reachable_site_count": len(reachable),
            "inbound_leg_rows": int(inbound.row_count),
            "outbound_leg_rows": int(outbound.row_count),
        })

    active_ranks = [rank for rank, site_id in enumerate(declared_ids) if site_id in active_ids]
    occluded_ranks = [rank for rank, site_id in enumerate(declared_ids) if site_id not in active_ids]
    occluded_sites = [{
        "site_id": int(declared_ids[rank]),
        "site_rank": rank,
        "bone_name": str(sampler.site_names[rank]),
        "vertex_index": int(sampler.vertex_ids[rank]),
        "visible_frame_count": int(visible_frame_counts[declared_ids[rank]]),
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
        "method": "native_channel_interval_visibility_intersection_no_synthesis",
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
    from witwin.radar.scattering import ScalarRcsResponse

    times, sampler, radar, world, config, meta, authored_motion_fingerprint = prepare_animation(scene, request)
    topology = visibility_preflight(
        radar, world,
        [sampler.positions(float(time_s)) for time_s in times],
        sampler,
        polarization=meta["world_polarization"],
        times_s=times,
    )
    active_ranks = np.asarray(topology["active_site_ranks"], dtype=np.int64)
    active_site_ids = tuple(topology["active_site_ids"])
    rcs_per_site = meta["rcs_m2"] / topology["declared_site_count"]
    response = ScalarRcsResponse.from_rcs(rcs_per_site, reference_frequency_hz=config.fc, device=radar.device)
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
                velocity_method="Studio LINEAR timeline + CPU skinning; finite difference <=1ms, right-hand at knots",
                slow_time_model="native frozen-weight first-order carrier-rate per frame",
                limitations="Uncalibrated equal-RCS surface samples, not full electromagnetic skin. "
                            "Only sites with native inbound+outbound paths throughout the interval are active; "
                            "occluded sites keep their original RCS share and are not renormalized. "
                            "Static room occlusion; no room clutter, extra bounces or cat self-occlusion. "
                            "Frozen weights within each CPI; no acceleration/range migration within CPI.",
                versions={name: version(name) for name in ("witwin-radar", "witwin-channel", "witwin")})
    if request.get("input_fingerprint"):
        meta["input_fingerprint"] = str(request["input_fingerprint"])
    ctx.log(f"Studio skin animation model: {meta}")
    cubes = None
    poses, velocities, diagnostics = [], [], []
    for index, time_s in enumerate(times):
        ctx.throw_if_cancelled()
        p, v = sampler.sample(float(time_s))
        p, v = p[active_ranks], v[active_ranks]
        try:
            result, stats = solve_native_frame(
                radar, world, response, time_s, p, v,
                meta["world_polarization"], stable_site_ids=active_site_ids,
            )
        except Exception as exc:
            raise RuntimeError(f"Animation failed at frame {index}/{len(times)}, t={time_s:.6f}s: {exc}. "
                               "No completed result published; native error retained.") from exc
        if cubes is None:
            cubes = torch.empty((len(times), *result.data.shape), dtype=result.data.dtype, device="cpu")
        cubes[index].copy_(result.data)
        poses.append(p)
        velocities.append(v)
        diagnostics.append(stats)
        ctx.progress((index + 1) / len(times), f"GPU frame {index+1}/{len(times)} | Studio t={time_s:.3f}s")
    axes = replace(result.axes, range_m=result.axes.range_m.cpu(), velocity_mps=result.axes.velocity_mps.cpu())
    meta["frame_diagnostics"] = diagnostics
    meta["tx_positions_m"] = radar.tx_pos.detach().cpu().tolist()
    meta["rx_positions_m"] = radar.rx_pos.detach().cpu().tolist()
    meta["device"] = str(radar.device)
    return AnimationResult(cubes, axes, times, np.stack(poses), np.stack(velocities), meta)


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
