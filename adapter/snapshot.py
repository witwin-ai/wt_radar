"""Studio snapshot transport to Radar 0.3; no propagation or DSP implementation."""
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
import torch

from .common import num, vec
from .config_map import ConfigMap
from .subconfig_map import SubConfigMap


MODEL = "explicit_point_frozen_snapshot"
MAX_CUBE_ELEMENTS = 2_000_000
SUPPORTED_VIEWS = {"range_profile", "range_spectrum", "range_doppler"}


def validate_options(component):
    """Old settings must not silently acquire a different meaning."""
    unsupported = [name for name in (
        "enable_thermal", "enable_quantization", "enable_phase", "enable_lna",
        "enable_agc", "enable_adc", "pol_enabled", "multipath", "use_seed",
    ) if bool(getattr(component, name))]
    if unsupported:
        raise ValueError("Snapshot adapter does not support these enabled legacy settings: " + ", ".join(unsupported))
    if int(component.max_reflections) != 0:
        raise ValueError("Snapshot diagnosis currently requires max_reflections=0 (no extra environment bounces).")
    if str(component.device) != "cuda" or str(component.backend) != "dirichlet":
        raise ValueError("Snapshot diagnosis requires native CUDA FMCW spectrum synthesis; no alternate backend.")
    for name, default in {"sampling": "triangle", "resolution": 128, "epsilon_r": 5.0,
                          "ray_batch_size": 65536, "pad_factor": 16, "motion_sampling": "per_chirp"}.items():
        if getattr(component, name) != default:
            raise ValueError(f"Legacy {name} is not used by point snapshots; restore its default {default!r}.")
    if component.post_processors:
        raise ValueError("Legacy post-processors are not supported by the snapshot adapter.")
    if not str(component.snapshot_target_id).strip():
        raise ValueError("Set Snapshot Target ID to an existing object; no target is selected automatically.")
    rcs = num(component.snapshot_rcs_m2)
    if not np.isfinite(rcs) or rcs <= 0:
        raise ValueError("Snapshot RCS must be positive finite square metres; it is an explicit modelling assumption.")
    for name in ("snapshot_local_point", "snapshot_polarization"):
        values = np.asarray(vec(getattr(component, name)), dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain three finite values.")
    if np.linalg.norm(vec(component.snapshot_polarization)) <= 1e-12:
        raise ValueError("Snapshot world polarization must be nonzero.")
    count = int(component.num_tx) * int(component.num_rx) * int(component.chirp_per_frame) * int(component.adc_samples)
    if count < 1 or count > MAX_CUBE_ELEMENTS:
        raise ValueError(f"Snapshot cube exceeds the {MAX_CUBE_ELEMENTS}-element diagnostic budget.")


def snapshot_request(component):
    validate_options(component)
    if component.scene is None or component.owner is None:
        raise ValueError("Radar must belong to an open Studio scene.")
    manager = component.scene.timeline_manager
    if manager is not None and (manager.is_playing or manager.is_recording):
        raise ValueError("Pause timeline playback and recording before taking a scene snapshot.")
    time_s = num(component.t0)
    if not np.isfinite(time_s) or time_s < 0:
        raise ValueError("Snapshot time must be finite and nonnegative.")
    return {"adapter": MODEL, "scene_id": str(component.scene.scene_id),
            "radar_object_id": str(component.owner.id), "time_s": time_s}


def _hierarchy(scene):
    """Check the authoritative Transform tree before the shared affine exporter."""
    owners = {id(obj.get_component("Transform")): obj for obj in scene.objects.values()
              if obj.get_component("Transform") is not None}
    matrices, visible = {}, {}

    def visit(obj, visiting):
        oid = str(obj.id)
        if oid in visiting:
            raise ValueError(f"Transform parent cycle at {oid}")
        if oid in matrices:
            return matrices[oid]
        transform = obj.get_component("Transform")
        if transform is None:
            local = np.eye(4, dtype=np.float32)
            parent = None
        else:
            local = transform.get_transformation_matrix().detach().cpu().numpy()
            parent = getattr(transform, "parent", None)
        if local.shape != (4, 4) or not np.isfinite(local).all():
            raise ValueError(f"Nonfinite or invalid world transform for {oid}")
        enabled = bool(obj.visible)
        if parent is not None:
            parent_obj = owners.get(id(parent))
            if parent_obj is None:
                raise ValueError(f"Unresolved Transform parent for {oid}")
            local = visit(parent_obj, visiting | {oid}) @ local
            enabled = enabled and visible[str(parent_obj.id)]
        matrices[oid], visible[oid] = local, enabled
        return local

    for obj in scene.objects.values():
        visit(obj, set())
    return matrices, visible


def prepare_snapshot(scene, request):
    """Read a solver-owned scene copy; never seek the user's live editor scene."""
    from witwin.radar import RadarConfig
    from .studio_geometry import export_static_mesh_scene

    if request.get("adapter") != MODEL or str(scene.scene_id) != request.get("scene_id"):
        raise ValueError("Snapshot request and solver scene identity disagree.")
    sensor_obj = scene.get_object(request["radar_object_id"])
    component = sensor_obj.get_component("Radar") if sensor_obj is not None else None
    if component is None:
        raise ValueError("The selected Radar object is no longer present in this scene.")
    validate_options(component)
    time_s = float(request["time_s"])
    if not np.isfinite(time_s) or time_s < 0:
        raise ValueError("Invalid snapshot time.")
    manager = scene.timeline_manager
    if manager is None and time_s != 0:
        raise ValueError("A scene without a timeline supports only snapshot time zero.")
    if manager is not None and time_s > float(manager.clip.duration):
        raise ValueError("Snapshot time is outside the scene timeline; no endpoint clamping.")
    if manager is not None:
        manager.set_time(time_s, apply_to_scene=True)
    validate_options(component)
    matrices, visible = _hierarchy(scene)
    target_id = str(component.snapshot_target_id).strip()
    target = scene.get_object(target_id)
    if target is None or target_id == str(sensor_obj.id):
        raise ValueError(f"Snapshot Target ID {target_id!r} must name a separate existing object.")
    if not visible[target_id] or not visible[str(sensor_obj.id)]:
        raise ValueError("Snapshot target and radar must be visible through their whole parent hierarchy.")
    excluded = {str(sensor_obj.id), target_id} | {oid for oid, enabled in visible.items() if not enabled}
    for obj in scene.objects.values():
        if str(obj.id) in excluded:
            continue
        if obj.get_component("SkinnedMesh") is not None:
            raise ValueError(f"Additional visible SkinnedMesh {obj.id} has no declared scattering model.")
        if obj.get_component("PlatformGeometry") is not None and obj.get_component("Mesh") is None:
            raise ValueError(f"PlatformGeometry-only object {obj.id} is not supported by this Mesh snapshot adapter.")
    world = export_static_mesh_scene(scene, excluded_object_ids=excluded)
    local_point = np.asarray(vec(component.snapshot_local_point), dtype=np.float32)
    point = (matrices[target_id] @ np.append(local_point, np.float32(1)))[:3]
    sensor_matrix = matrices[str(sensor_obj.id)]
    position = sensor_matrix[:3, 3]
    forward, up = -sensor_matrix[:3, 2], sensor_matrix[:3, 1]
    if min(np.linalg.norm(forward), np.linalg.norm(up)) < 1e-8 or np.linalg.norm(np.cross(forward, up)) < 1e-8:
        raise ValueError("Radar pose has degenerate forward/up vectors.")
    config = ConfigMap._core_dict(component)
    config["antenna_pattern"] = SubConfigMap.antenna_build(component)
    radar_config = RadarConfig.from_dict(config)
    if int(component.num_range_bins) != int(component.adc_samples) or int(component.num_doppler_bins) != int(component.chirp_per_frame):
        raise ValueError("Snapshot uses native sample/chirp bins: set Range Bins=ADC Samples and Doppler Bins=Chirps.")
    metadata = {
        "model": MODEL, "scene_id": str(scene.scene_id), "scene_name": str(scene.name),
        "radar_object_id": str(sensor_obj.id), "target_object_id": target_id,
        "snapshot_time_s": time_s, "target_local_point": local_point.tolist(),
        "target_world_point_m": point.tolist(), "target_world_matrix": matrices[target_id].tolist(),
        "radar_position_m": position.tolist(), "radar_target_m": (position + forward).tolist(),
        "radar_up": up.tolist(), "rcs_m2": num(component.snapshot_rcs_m2),
        "world_polarization": vec(component.snapshot_polarization),
        "velocity_m_per_s": [0.0, 0.0, 0.0], "components": ["los"], "max_depth": 0,
        "room_mesh_count": world.mesh_count, "room_triangle_count": world.face_count,
        "room_object_ids": sorted(world.object_to_structure_id),
        "radar_config": config,
        "limitations": "One explicit RCS point, not skin scattering; frozen pose, no gait Doppler; "
                       "room occludes paths, no static room clutter or target self-occlusion; RCS is uncalibrated.",
    }
    return world.scene, radar_config, metadata


def processing_cube(simulation, radar):
    """Repack published axes into the official processing metadata constructor."""
    from witwin.radar.processing import ProcessingAxes, ProcessingCube
    from witwin.radar.synthesis.assembly import SynthesisResult

    spec, array = radar.system_config.waveform_spec(), radar.system_config.sensors.array
    expected = ("frame", "tx", "rx", "chirp", "range_bin")
    if tuple(simulation.axes) != expected or simulation.kind != "fmcw" or spec.output_domain != "spectrum":
        raise ValueError("Snapshot result is not the requested FMCW spectrum product.")
    if len(simulation.times_s) != 1:
        raise ValueError("Snapshot adapter requires exactly one frame.")
    frame = simulation.cube[0]
    # The native synthesis pair rank is RX-major. This is structural packing,
    # checked by the official inverse below, not a second signal transform.
    packed = frame.permute(2, 1, 0, 3).reshape(spec.num_chirps, array.sensor_pair_count, spec.num_samples)
    synthesis = SynthesisResult(
        cube=packed, kind=simulation.kind, axes=("chirp", "sensor_pair", "range_bin"),
        phasor=simulation.phasor, time_dependence=simulation.time_dependence,
        reference_frequency_hz=simulation.reference_frequency_hz, output_domain=spec.output_domain,
    )
    axes = ProcessingAxes.from_synthesis(synthesis, spec, array)
    if not torch.equal(ProcessingCube.from_synthesis(synthesis, axes).data, frame):
        raise RuntimeError("Simulation/processing array packing mismatch.")
    return ProcessingCube(frame, axes)


@dataclass
class SnapshotResult:
    processing: object
    metadata: dict


def solve_snapshot(ctx, scene, request):
    from witwin.radar import Radar
    from witwin.radar.scattering import ScalarRcsResponse
    from witwin.radar.simulation import ScatterSitePolicy

    ctx.progress(0.0, "Exporting frozen Studio scene and explicit point target")
    world, config, metadata = prepare_snapshot(scene, request)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; snapshot simulation has no CPU fallback.")
    radar = Radar(config, device="cuda", position=metadata["radar_position_m"],
                  target=metadata["radar_target_m"], up=metadata["radar_up"])
    response = ScalarRcsResponse.from_rcs(metadata["rcs_m2"], reference_frequency_hz=config.fc, device=radar.device)
    sites = ScatterSitePolicy.explicit(torch.tensor([metadata["target_world_point_m"]], device=radar.device))
    ctx.log(f"Radar 0.3 snapshot: {metadata}")
    ctx.progress(0.3, "Running native GPU propagation and FMCW synthesis")
    simulation = radar.simulate(world, times=(metadata["snapshot_time_s"],), response=response,
                                sites=sites, components=frozenset({"los"}), max_depth=0,
                                polarization=tuple(metadata["world_polarization"]))
    processed = processing_cube(simulation, radar)
    if processed.data.device.type != "cuda" or not bool(torch.isfinite(processed.data).all()):
        raise RuntimeError("Native CUDA result is missing or nonfinite.")
    metadata.update({"cube_shape": list(processed.data.shape), "device": str(processed.data.device),
                     "versions": {name: version(name) for name in ("witwin-radar", "witwin-channel", "witwin")},
                     "compile_count": simulation.compile_count, "discovery_count": simulation.discovery_count,
                     "tx_positions_m": radar.tx_pos.detach().cpu().tolist(),
                     "rx_positions_m": radar.rx_pos.detach().cpu().tolist()})
    ctx.progress(1.0, "GPU snapshot complete; all velocities frozen to zero")
    return SnapshotResult(processed, metadata)


def snapshot_view(result, params):
    from witwin.radar.processing import range_doppler_map, range_profile

    view = str(params.get("view", "range_profile"))
    if view not in SUPPORTED_VIEWS:
        raise ValueError("Snapshot supports Range Profile, Range Spectrum and Range Doppler only.")
    if params.get("static_clutter_removal") or params.get("show_cfar"):
        raise ValueError("Snapshot review does not apply clutter removal or CFAR; disable those options.")
    tx, rx = int(params.get("tx", 0)), int(params.get("rx", 0))
    axes = result.processing.axes
    if not 0 <= tx < axes.num_tx or not 0 <= rx < axes.num_rx:
        raise ValueError("TX/RX selection is outside the recorded array.")
    profile = range_profile(result.processing, window="rectangular", remove_dc=False)
    selected = profile.data[tx, rx, 0]
    out = {"view": view, "tx": tx, "rx": rx, "metadata": result.metadata,
           "range_m": axes.range_m.detach().cpu().tolist()}
    if view == "range_doppler":
        rd = range_doppler_map(profile, window="hann")
        out.update({"magnitude": rd.data[tx, rx].abs().detach().cpu().tolist(),
                    "velocity_mps": rd.axes.velocity_mps.detach().cpu().tolist()})
    else:
        out.update({"real": selected.real.detach().cpu().tolist(), "imag": selected.imag.detach().cpu().tolist(),
                    "magnitude": selected.abs().detach().cpu().tolist()})
    return out


def show_snapshot(component):
    from .replay import numeric_plot
    params = {"view": str(component.view), "tx": int(component.tx_index), "rx": int(component.rx_index),
              "static_clutter_removal": bool(component.static_clutter_removal), "show_cfar": bool(component.show_cfar)}
    payload = component._query_result(
        "snapshot_view",
        params,
        result_handle=component._snapshot_solver_result_handle,
        run_id=component._snapshot_solver_run_id,
    )
    meta = payload["metadata"]
    payload["title"] = (f"Frozen point snapshot | {meta['target_object_id']} | t={meta['snapshot_time_s']:.3f}s | "
                        f"TX{payload['tx']} RX{payload['rx']} | linear amplitude")
    component.snapshot_figure.set_plot_data(numeric_plot(payload))
    return "Native GPU snapshot displayed; not a moving-cat measurement"
