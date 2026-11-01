"""Static Radar capture (gap G4): the room with no moving target, as N frames.

Two modes, both through Radar 0.4's public API and Channel's single-bounce
environment route, with no RF or DSP implemented here:

- ``room``: the room alone. Every skinned actor is left out of the export, so
  the cube is the static background and clutter a dataset needs.
- ``frozen``: the declared actors held at one Timeline time, as zero-velocity
  skin sites (the same sites, active-site rule and RCS split as an animation
  measurement), plus the room.

Nothing moves, and the configured instrument has no noise stage, so every frame
of a static capture is the same native product; one frame is synthesized and
repeated N times on the measured timebase (``frames_identical_by_construction``).
"""
from dataclasses import replace
from importlib.metadata import version

import numpy as np
import torch

from .animation import AnimationResult, target_reflection_depth
from .common import num, vec
from .config_map import ConfigMap
from .memory_budget import require_result_memory
from .snapshot import _hierarchy, processing_cube, validate_runtime_options
from .subconfig_map import SubConfigMap


STATIC_MODEL = "studio_static_capture_v1"
STATIC_MODES = ("room", "frozen")
DEFAULT_FRAME_COUNT = 10
MAX_FRAME_COUNT = 300


def static_capture_request(component, *, mode="room", frame_count=DEFAULT_FRAME_COUNT, freeze_time_s=None):
    """An immutable solver request for a static capture of the component's scene."""
    from .target_contract import component_target_specs

    validate_runtime_options(component)
    if component.scene is None or component.owner is None:
        raise ValueError("Radar must belong to an open Studio scene.")
    if mode not in STATIC_MODES:
        raise ValueError(f"Static capture mode must be one of {STATIC_MODES}, got {mode!r}.")
    manager = component.scene.timeline_manager
    if manager is not None and (manager.is_playing or manager.is_recording):
        raise ValueError("Pause timeline playback and recording before a static capture.")
    count = int(frame_count)
    if count != frame_count or not 1 <= count <= MAX_FRAME_COUNT:
        raise ValueError(f"Static capture needs 1 to {MAX_FRAME_COUNT} frames.")
    fps = float(component.animation_fps)
    if not np.isfinite(fps) or not 1 <= fps <= 30:
        raise ValueError("Static capture uses the Radar's animation FPS, which must be 1 to 30.")
    require_result_memory(count * int(component.num_tx) * int(component.num_rx)
                          * int(component.chirp_per_frame) * int(component.adc_samples) * 8)
    request = {
        "adapter": STATIC_MODEL,
        "mode": mode,
        "scene_id": str(component.scene.scene_id),
        "radar_object_id": str(component.owner.id),
        "frame_count": count,
        "fps": fps,
        "duration_s": count / fps,
        "start_s": 0.0,
        "freeze_time_s": None,
        "targets": [],
        "target_multipath": bool(component.target_multipath),
    }
    if mode == "frozen":
        freeze = float(component.t0 if freeze_time_s is None else freeze_time_s)
        duration = float(manager.clip.duration) if manager is not None else 0.
        if not np.isfinite(freeze) or freeze < 0 or freeze > duration + 1e-7:
            raise ValueError("Freeze time must lie on the baked Studio timeline.")
        rcs = num(component.snapshot_rcs_m2)
        if not np.isfinite(rcs) or rcs <= 0:
            raise ValueError("Frozen actors need a positive finite Snapshot RCS (split over their sites).")
        request.update(freeze_time_s=freeze, start_s=freeze, targets=component_target_specs(component))
    return request


def _build(scene, request):
    from .radar04 import build_radar, flat_config
    from .studio_geometry import export_static_mesh_scene

    if request.get("adapter") != STATIC_MODEL or str(scene.scene_id) != request.get("scene_id"):
        raise ValueError("Static capture request and solver scene identity disagree.")
    sensor = scene.get_object(request["radar_object_id"])
    component = sensor.get_component("Radar") if sensor is not None else None
    if component is None:
        raise ValueError("Radar object no longer exists.")
    validate_runtime_options(component)
    if (int(component.num_range_bins) != int(component.adc_samples)
            or int(component.num_doppler_bins) != int(component.chirp_per_frame)):
        raise ValueError("Static capture uses native sample/chirp bins: set Range Bins=ADC Samples "
                         "and Doppler Bins=Chirps.")
    matrices, visible = _hierarchy(scene)
    skinned = {str(oid) for oid, obj in scene.objects.items() if obj.get_component("SkinnedMesh") is not None}
    declared = {str(item["object_id"]) for item in request.get("targets") or ()}
    if request["mode"] == "frozen":
        undeclared = sorted(skinned - declared)
        if undeclared:
            raise ValueError("Every skinned actor must be a frozen target or be removed: " + ", ".join(undeclared))
        excluded = {str(sensor.id)} | declared
    else:
        # The room alone: actors are not part of the background.
        excluded = {str(sensor.id)} | skinned
    excluded |= {oid for oid, enabled in visible.items() if not enabled}
    world = export_static_mesh_scene(scene, excluded_object_ids=excluded, frequency_hz=float(component.fc))
    matrix = matrices[str(sensor.id)]
    position = matrix[:3, 3]
    forward, up = -matrix[:3, 2], matrix[:3, 1]
    if min(np.linalg.norm(forward), np.linalg.norm(up)) < 1e-8 or np.linalg.norm(np.cross(forward, up)) < 1e-8:
        raise ValueError("Radar pose has degenerate forward/up vectors.")
    config = ConfigMap._core_dict(component)
    config["antenna_pattern"] = SubConfigMap.antenna_build(component)
    if not torch.cuda.is_available():
        raise RuntimeError("Static capture requires CUDA; no CPU fallback.")
    polarization = vec(component.snapshot_polarization)
    radar = build_radar(flat_config(config), device="cuda", position=position.tolist(),
                        look_at=(position + forward).tolist(), up=up.tolist(), polarization=polarization)
    meta = {
        "scene_id": str(scene.scene_id), "radar_object_id": str(sensor.id),
        "radar_position_m": position.tolist(), "radar_target_m": (position + forward).tolist(),
        "radar_up": up.tolist(), "world_polarization": polarization, "rcs_m2": num(component.snapshot_rcs_m2),
        "room_mesh_count": world.mesh_count, "room_triangle_count": world.face_count,
        "room_object_ids": sorted(world.object_to_structure_id),
        "room_material_frequency_hz": world.material_frequency_hz, "radar_config": config,
        "excluded_object_ids": sorted(excluded),
    }
    return radar, world.scene, meta


def solve_static_capture(ctx, scene, request):
    """Synthesize one static frame and publish it as an N-frame animation result."""
    from witwin.radar import Motion, PointTargets
    from .animation import visibility_preflight
    from .environment_clutter import add_environment_to_fmcw_result, single_bounce_environment_cube
    from .studio_motion import MultiTargetSkinSampler

    ctx.progress(0.01, "Preparing static Radar scene")
    radar, world, meta = _build(scene, request)
    count, fps = int(request["frame_count"]), float(request["fps"])
    times = float(request["start_s"]) + np.arange(count, dtype=np.float64) / fps
    frozen = request["mode"] == "frozen"
    target_evidence, topology = [], None
    if frozen:
        sampler = MultiTargetSkinSampler(scene, request["targets"])
        sites, _velocity = sampler.sample(float(request["freeze_time_s"]))
        sites = np.asarray(sites, dtype=np.float64)
        ctx.progress(0.05, "Checking frozen-site visibility")
        topology = visibility_preflight(
            radar, world, sites[None], sampler, polarization=meta["world_polarization"],
            times_s=np.asarray([float(request["freeze_time_s"])]),
            target_reflections=target_reflection_depth(request))
        active = np.asarray(topology["active_site_ranks"], dtype=np.int64)
        active_set = set(active.tolist())
        target_evidence = [{**item, "active_site_count": sum(
            rank in active_set for rank in range(item["site_start"], item["site_end"]))}
            for item in sampler.target_slices]
        positions = sites[active]
        ids = tuple(topology["active_site_ids"])
        rcs = meta["rcs_m2"] / topology["declared_site_count"]
    else:
        positions, ids, rcs = np.zeros((0, 3)), (), None
    ctx.progress(0.1, "Compiling single-bounce room reflections")
    environment = single_bounce_environment_cube(radar, world, polarization=meta["world_polarization"])
    ctx.progress(0.5, "Synthesizing the static frame")
    if len(positions):
        simulation = radar.simulate(
            world,
            PointTargets(positions=torch.as_tensor(positions, dtype=torch.float32, device=radar.device),
                         rcs=float(rcs), ids=ids),
            times=(float(times[0]),), los=True, reflections=target_reflection_depth(request),
            motion=Motion.static())
        processed = processing_cube(simulation, radar)
        path_count = int(simulation.last_radar_paths.path_count)
        processed = replace(processed, data=add_environment_to_fmcw_result(radar, processed.data, environment))
        data, axes = processed.data, processed.axes
    else:
        # Native synthesis needs at least one target site; the room alone is the
        # environment product with the axes of its own synthesis cube.
        data, axes, path_count = environment.cube, environment.axes, 0
    if data.device.type != "cuda" or not bool(torch.isfinite(data).all()):
        raise RuntimeError("Native CUDA static frame is missing or nonfinite.")
    frame = data.detach().cpu()
    cube = frame.unsqueeze(0).repeat((count,) + (1,) * frame.dim()).contiguous()
    axes = replace(axes, range_m=axes.range_m.cpu(), velocity_mps=axes.velocity_mps.cpu())
    site_positions = np.repeat(positions[None], count, axis=0)
    meta.update(
        model=STATIC_MODEL, capture="static", mode=request["mode"], frame_count=count, fps=fps,
        freeze_time_s=request.get("freeze_time_s"), frames_identical_by_construction=True,
        targets=target_evidence, topology_preflight=topology, rcs_per_site_m2=rcs,
        site_count=int(len(positions)),
        environment_reflection={
            "model": "radar04_native_single_bounce", "max_depth": 1,
            "reflected_path_count": environment.reflected_path_count,
            "material_slot_count": environment.material_slot_count,
            "coherent_with_target": True,
        },
        frame_diagnostics=[{"nonzero_return": bool((frame.abs() > 0).any()), "delay_rate_abs_max": 0.0,
                            "target_path_count": path_count}] * count,
        tx_positions_m=radar.tx_pos.detach().cpu().tolist(),
        rx_positions_m=radar.rx_pos.detach().cpu().tolist(),
        device=str(radar.device),
        limitations=("Static capture: nothing moves and no noise stage is configured, so every frame is the "
                     "same native product. Room geometry contributes native single-bounce specular "
                     "reflections only; frozen actors are uncalibrated equal-RCS skin sites."),
        versions={name: version(name) for name in ("witwin-radar", "witwin-channel", "witwin")},
    )
    if request.get("input_fingerprint"):
        meta["input_fingerprint"] = str(request["input_fingerprint"])
    ctx.progress(1.0, f"Static capture complete: {count} frames")
    return AnimationResult(cube, axes, times, site_positions, np.zeros_like(site_positions), meta)


__all__ = ["DEFAULT_FRAME_COUNT", "STATIC_MODEL", "STATIC_MODES", "solve_static_capture",
           "static_capture_request"]
