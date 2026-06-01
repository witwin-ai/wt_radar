"""Solve hand-off + sigproc views (export -> solve -> in-component display, master §7.2).

``SolveRunner`` is the export->solve half of the edit loop: it rebuilds the platform
``(Scene, RadarConfig)`` pair from the live studio scene (on a CUDA scene, because the
RayD tracer is CUDA-only), constructs a ``Radar`` from the RadarSensor pose/backend,
and runs ``radar.simulate(...)`` to produce the MIMO data cube. ``SigProc`` wraps the
platform ``sigproc`` (range-doppler / point cloud / MUSIC / CFAR) for the in-component
figure. Result tensors are runtime state — never serialized into the scene.

Live solve needs CUDA for renderable scenes regardless of the solver backend: the tracer
binds to ``Radar.device``, so simulation uses the scene device even when a stored sensor
spec was authored as CPU-only for inspection.
"""
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any, List, Optional, Tuple

import numpy as np

from witwin_server.utils.logging import get_logger

from .common import num

logger = get_logger("RadarSolve")


@dataclass
class SensorSpec:
    """Radar pose + backend for ``Radar(...)``.

    The pose (position / target / up) is derived from the owner SceneObject's
    ``Transform`` so the simulated radar follows the viewport gizmo: the radar sits
    at the Transform's world position, looks along the rotated local -Z (matching the
    cone gizmo), and the up vector is the rotated local +Y.
    """

    position: List[float]
    target: Optional[List[float]]
    up: List[float]
    fov: float
    backend: str
    pad_factor: int
    device: str

    @classmethod
    def from_component(cls, radar: Any) -> "SensorSpec":
        position, target, up = _world_pose(radar.owner)
        return cls(
            position=position,
            target=target,
            up=up,
            fov=num(radar.fov),
            backend=str(radar.backend),
            pad_factor=int(radar.pad_factor),
            device=str(radar.device),
        )


def _world_pose(obj: Any) -> tuple:
    """World position + look-at target + up read off the owner's Transform.

    Returns ``(position, target, up)`` in world coordinates: position is the
    Transform's world translation; target is ``position + R @ (0,0,-1)`` (looking
    down the rotated local -Z, matching the cone gizmo's local -Z direction); up is
    the rotated local +Y. Both direction vectors are normalized so a Transform with
    non-uniform scale still produces a unit look-at axis.
    """
    from witwin_server.utils.mitsuba_utils import get_world_transform

    matrix = np.asarray(get_world_transform(obj), dtype=np.float64)
    position = matrix[:3, 3].tolist()
    rotation = matrix[:3, :3]
    forward = rotation @ np.array([0.0, 0.0, -1.0])
    upvec = rotation @ np.array([0.0, 1.0, 0.0])
    forward = _normalize(forward, fallback=(0.0, 0.0, -1.0))
    upvec = _normalize(upvec, fallback=(0.0, 1.0, 0.0))
    target = [position[i] + forward[i] for i in range(3)]
    obj_name = getattr(obj, "name", "?")
    logger.info(
        f"_world_pose[{obj_name}]: position={[round(p, 4) for p in position]} "
        f"forward={[round(f, 4) for f in forward]} up={[round(u, 4) for u in upvec]}")
    return position, target, list(upvec)


def _normalize(vector: np.ndarray, *, fallback: tuple) -> tuple:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return fallback
    return tuple((vector / norm).tolist())


@dataclass
class TracerSpec:
    """Ray-tracing parameters read off the unified Radar component (for ``radar.simulate``)."""

    resolution: int = 128
    epsilon_r: float = 5.0
    sampling: str = "triangle"
    multipath: bool = False
    max_reflections: int = 0
    ray_batch_size: int = 65536

    @classmethod
    def from_component(cls, radar: Any) -> "TracerSpec":
        if radar is None:
            return cls()
        return cls(
            resolution=int(radar.resolution),
            epsilon_r=num(radar.epsilon_r),
            sampling=str(radar.sampling),
            multipath=bool(radar.multipath),
            max_reflections=int(radar.max_reflections),
            ray_batch_size=int(radar.ray_batch_size),
        )


@dataclass
class SolveResult:
    """A completed solve: the live ``Radar`` and its MIMO signal ``(TX,RX,chirps,ADC)``."""

    radar: Any
    signal: Any


@dataclass
class RDView:
    """A range-doppler map ready for display + the complex map for CFAR."""

    mag_db: np.ndarray       # (num_doppler_bins, adc_samples), dB
    ranges: np.ndarray       # (num_range_bins // 2,) meters
    velocities: np.ndarray   # (num_doppler_bins,) m/s
    rd_map: np.ndarray       # complex, for CFAR detection


class SolveRunner:
    """Build + run a ``Radar.simulate`` from the live studio scene."""

    @staticmethod
    def run(studio_scene: Any, *, sensor: SensorSpec, tracer: TracerSpec,
            motion_sampling: str, t0: float, live_cache: Optional[dict] = None,
            cache_key: Optional[str] = None,
            platform_cache_key: Optional[str] = None) -> SolveResult:
        """Rebuild the (Scene, RadarConfig) pair on CUDA, construct the Radar, and simulate."""
        import torch
        from .radar_adapter import RadarAdapter

        scene_device = "cuda" if torch.cuda.is_available() else "cpu"
        if live_cache is not None and cache_key is not None:
            cached = SolveRunner._try_live_cache(
                live_cache,
                cache_key=cache_key,
                t0=t0,
            )
            if cached is not None:
                return cached

        sensor = SolveRunner._sensor_for_scene_device(sensor, scene_device)
        build_started = time.perf_counter()
        platform_reused = False
        scene_config = None
        if live_cache is not None and cache_key is not None and platform_cache_key is not None:
            scene_config = SolveRunner._try_reuse_platform_scene(
                live_cache,
                platform_cache_key=platform_cache_key,
            )
        if scene_config is None:
            scene, config = RadarAdapter().to_platform(studio_scene, device=scene_device)
        else:
            scene, config = scene_config
            platform_reused = True
        logger.info(
            f"SolveRunner.run: scene_device={scene_device}, "
            f"{len(scene.structures)} structures, tracer={tracer.sampling}@{tracer.resolution} "
            f"platform_reused={platform_reused}")
        radar_config_key = SolveRunner._radar_config_key(config)
        radar_sensor_runtime_key = SolveRunner._radar_sensor_runtime_key(sensor)
        radar_reused = False
        radar = None
        if live_cache is not None and cache_key is not None:
            radar = SolveRunner._try_reuse_pose_radar(
                live_cache,
                config_key=radar_config_key,
                sensor_runtime_key=radar_sensor_runtime_key,
                sensor=sensor,
            )
            radar_reused = radar is not None
        if radar is None:
            radar = SolveRunner.build_radar(config, sensor)
            logger.info(
                f"  wr.Radar built: position={radar.position.tolist()} "
                f"target={radar.target.tolist()} tx_pos[0]={radar.tx_pos[0].tolist()}")
        else:
            logger.info(
                f"  wr.Radar pose reused: position={radar.position.tolist()} "
                f"target={radar.target.tolist()} tx_pos[0]={radar.tx_pos[0].tolist()}")
        if live_cache is not None and cache_key is not None:
            result = SolveRunner._run_live_cache_miss(
                live_cache,
                cache_key=cache_key,
                platform_cache_key=platform_cache_key,
                scene=scene,
                platform_config=config,
                radar=radar,
                radar_config_key=radar_config_key,
                radar_sensor_runtime_key=radar_sensor_runtime_key,
                radar_reused=radar_reused,
                platform_reused=platform_reused,
                tracer=tracer,
                motion_sampling=motion_sampling,
                t0=t0,
                build_elapsed=time.perf_counter() - build_started,
            )
        else:
            simulate_started = time.perf_counter()
            signal = radar.simulate(
                scene, resolution=tracer.resolution, epsilon_r=tracer.epsilon_r,
                sampling=tracer.sampling, multipath=tracer.multipath,
                max_reflections=tracer.max_reflections, ray_batch_size=tracer.ray_batch_size,
                t0=t0, motion_sampling=motion_sampling)
            result = SolveResult(radar=radar, signal=signal)
            logger.info(
                f"  radar.simulate elapsed_ms={(time.perf_counter() - simulate_started) * 1000.0:.2f}")
        SolveRunner._log_signal("radar solve done", result.signal)
        return result

    @staticmethod
    def _try_live_cache(live_cache: dict, *, cache_key: str, t0: float) -> Optional[SolveResult]:
        """Reuse fixed scene/radar/path state for repeated live frames with the same signature."""
        if live_cache.get("cache_key") != cache_key:
            return None
        if not bool(live_cache.get("fast_frame_reusable", False)):
            return None
        radar = live_cache.get("radar")
        if radar is None:
            return None

        started = time.perf_counter()
        try:
            path_cache = live_cache.get("path_cache")
            if path_cache is not None:
                signal = radar.mimo_from_paths(path_cache)
                mode = "mimo_from_paths"
            else:
                trace = live_cache.get("trace")
                if trace is None:
                    return None
                signal = radar.mimo_from_trace(trace, t0=t0)
                mode = "mimo_from_trace"
        except Exception as exc:  # noqa: BLE001 - stale/incompatible fast path should rebuild
            logger.warning(f"  live cache fast path failed; rebuilding: {type(exc).__name__}: {exc}")
            live_cache.clear()
            return None

        logger.info(
            f"  live cache hit: mode={mode} elapsed_ms={(time.perf_counter() - started) * 1000.0:.2f}")
        return SolveResult(radar=radar, signal=signal)

    @staticmethod
    def _try_reuse_pose_radar(live_cache: dict, *, config_key: tuple,
                              sensor_runtime_key: tuple, sensor: SensorSpec) -> Any | None:
        radar = live_cache.get("radar")
        if radar is None or not hasattr(radar, "set_pose"):
            return None
        if live_cache.get("radar_config_key") != config_key:
            return None
        if live_cache.get("radar_sensor_runtime_key") != sensor_runtime_key:
            return None
        try:
            return radar.set_pose(
                position=sensor.position,
                target=sensor.target,
                up=sensor.up,
                fov=sensor.fov,
            )
        except Exception as exc:  # noqa: BLE001 - stale/incompatible radar should rebuild
            logger.warning(f"  live radar pose reuse failed; rebuilding: {type(exc).__name__}: {exc}")
            return None

    @staticmethod
    def _try_reuse_platform_scene(live_cache: dict, *, platform_cache_key: str) -> Optional[tuple[Any, Any]]:
        if live_cache.get("platform_cache_key") != platform_cache_key:
            return None
        scene = live_cache.get("platform_scene")
        if scene is None or "platform_config" not in live_cache:
            return None
        return scene, live_cache.get("platform_config")

    @staticmethod
    def _run_live_cache_miss(live_cache: dict, *, cache_key: str,
                             platform_cache_key: Optional[str],
                             scene: Any, platform_config: Any, radar: Any,
                             radar_config_key: tuple, radar_sensor_runtime_key: tuple,
                             radar_reused: bool, platform_reused: bool,
                             tracer: TracerSpec, motion_sampling: str, t0: float,
                             build_elapsed: float) -> SolveResult:
        """Trace once and cache the MIMO path representation for live repeats."""
        tracer_obj, tracer_reused = SolveRunner._live_tracer(
            live_cache,
            scene,
            radar,
            tracer,
        )
        trace_started = time.perf_counter()
        trace_time = t0 if bool(getattr(scene, "has_motion", False)) else None
        trace = tracer_obj.trace(time=trace_time)
        trace_elapsed = time.perf_counter() - trace_started

        path_cache = None
        signal_started = time.perf_counter()
        mode = "mimo_from_trace"
        can_cache_paths = not (bool(getattr(scene, "has_motion", False)) and motion_sampling == "per_chirp")
        if can_cache_paths:
            try:
                path_cache = radar.path_cache_from_trace(trace)
                signal = radar.mimo_from_paths(path_cache)
                mode = "mimo_from_paths"
            except (AttributeError, NotImplementedError) as exc:
                logger.info(f"  live path cache unavailable; using mimo_from_trace: {type(exc).__name__}")
                signal = radar.mimo_from_trace(trace, t0=t0)
            except Exception as exc:  # noqa: BLE001 - fall back to the documented trace path
                logger.warning(f"  live path cache build failed; using mimo_from_trace: {type(exc).__name__}: {exc}")
                signal = radar.mimo_from_trace(trace, t0=t0)
        else:
            signal = radar.mimo_from_trace(trace, t0=t0)
        signal_elapsed = time.perf_counter() - signal_started

        live_cache.clear()
        live_cache.update({
            "cache_key": cache_key,
            "platform_cache_key": platform_cache_key,
            "platform_scene": scene,
            "platform_config": platform_config,
            "radar_config_key": radar_config_key,
            "radar_sensor_runtime_key": radar_sensor_runtime_key,
            "radar": radar,
            "trace": trace,
            "path_cache": path_cache,
            "tracer": tracer_obj,
            "tracer_spec_key": SolveRunner._tracer_spec_key(tracer),
            "scene_topology_key": SolveRunner._scene_topology_key(scene),
            "fast_frame_reusable": not bool(getattr(scene, "has_motion", False)),
        })
        logger.info(
            "  live cache miss: "
            f"build_ms={build_elapsed * 1000.0:.2f} trace_ms={trace_elapsed * 1000.0:.2f} "
            f"signal_ms={signal_elapsed * 1000.0:.2f} mode={mode} "
            f"path_cached={path_cache is not None} tracer_reused={tracer_reused} "
            f"radar_reused={radar_reused} platform_reused={platform_reused}")
        return SolveResult(radar=radar, signal=signal)

    @staticmethod
    def _log_signal(prefix: str, signal: Any) -> None:
        sig_abs = signal.detach().cpu().abs()
        logger.info(
            f"  {prefix}: shape={tuple(signal.shape)} "
            f"mean|sig|={sig_abs.mean().item():.6e} max|sig|={sig_abs.max().item():.6e}")

    @staticmethod
    def _live_tracer(live_cache: dict, scene: Any, radar: Any, tracer: TracerSpec) -> tuple[Any, bool]:
        from witwin.radar.trace import Tracer

        tracer_spec_key = SolveRunner._tracer_spec_key(tracer)
        topology_key = SolveRunner._scene_topology_key(scene)
        cached = live_cache.get("tracer")
        if (
            cached is not None
            and live_cache.get("tracer_spec_key") == tracer_spec_key
            and live_cache.get("scene_topology_key") == topology_key
        ):
            cached.scene = scene
            cached.radar = radar
            SolveRunner._mark_scene_vertices_dirty(scene)
            return cached, True

        return Tracer(
            scene,
            radar,
            resolution=tracer.resolution,
            epsilon_r=tracer.epsilon_r,
            sampling=tracer.sampling,
            multipath=tracer.multipath,
            max_reflections=tracer.max_reflections,
            ray_batch_size=tracer.ray_batch_size,
        ), False

    @staticmethod
    def _tracer_spec_key(tracer: TracerSpec) -> tuple:
        return (
            int(tracer.resolution),
            float(tracer.epsilon_r),
            str(tracer.sampling),
            bool(tracer.multipath),
            int(tracer.max_reflections),
            int(tracer.ray_batch_size),
        )

    @staticmethod
    def _mark_scene_vertices_dirty(scene: Any) -> None:
        dirty_full = getattr(scene, "DIRTY_FULL", None)
        dirty_vertices = getattr(scene, "DIRTY_VERTICES", None)
        if dirty_full is None or dirty_vertices is None:
            return
        if int(getattr(scene, "dirty_level", dirty_full)) >= int(dirty_full):
            try:
                scene._dirty_level = int(dirty_vertices)
            except Exception:  # noqa: BLE001 - dirty hint is an optimization only
                return

    @staticmethod
    def _scene_topology_key(scene: Any) -> tuple:
        rows = []
        for structure in getattr(scene, "structures", []) or []:
            geometry = getattr(structure, "geometry", None)
            material = getattr(structure, "material", None)
            rows.append((
                str(getattr(structure, "name", "")),
                type(geometry).__module__ + "." + type(geometry).__name__,
                SolveRunner._array_shape(getattr(geometry, "vertices", None)),
                SolveRunner._array_shape_and_bytes(getattr(geometry, "faces", None)),
                SolveRunner._material_eps(material),
                bool(getattr(structure, "enabled", True)),
            ))
        return tuple(rows)

    @staticmethod
    def _array_shape(value: Any):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        try:
            return tuple(np.asarray(value).shape)
        except Exception:  # noqa: BLE001 - non-array geometry metadata is not topology-critical
            return None

    @staticmethod
    def _array_shape_and_bytes(value: Any):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        try:
            array = np.asarray(value)
        except Exception:  # noqa: BLE001
            return None
        return tuple(array.shape), array.tobytes()

    @staticmethod
    def _material_eps(material: Any) -> Optional[float]:
        try:
            return float(material.evaluate_static().eps_r)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def run_group(studio_scene: Any, *, sensors: "dict[str, SensorSpec]", tracer: TracerSpec,
                  motion_sampling: str, t0: float) -> "dict[str, Any]":
        """Run several named radars on one rebuilt scene (``Radar.simulate_group``)."""
        import torch
        import witwin.radar as wr

        from .radar_adapter import RadarAdapter

        scene_device = "cuda" if torch.cuda.is_available() else "cpu"
        scene, config = RadarAdapter().to_platform(studio_scene, device=scene_device)
        radars = {
            name: SolveRunner.build_radar(config, SolveRunner._sensor_for_scene_device(spec, scene_device))
            for name, spec in sensors.items()
        }
        return wr.Radar.simulate_group(
            scene, radars=radars, resolution=tracer.resolution, epsilon_r=tracer.epsilon_r,
            sampling=tracer.sampling, multipath=tracer.multipath,
            max_reflections=tracer.max_reflections, ray_batch_size=tracer.ray_batch_size,
            t0=t0, motion_sampling=motion_sampling)

    @staticmethod
    def _sensor_for_scene_device(sensor: SensorSpec, scene_device: str) -> SensorSpec:
        """Return the sensor device needed by the active platform scene."""
        if str(scene_device).startswith("cuda") and str(sensor.device) != "cuda":
            return replace(sensor, device="cuda")
        return sensor

    @staticmethod
    def _radar_sensor_runtime_key(sensor: SensorSpec) -> tuple:
        return (
            str(sensor.backend),
            int(sensor.pad_factor),
            str(sensor.device),
        )

    @staticmethod
    def _radar_config_key(config: Any) -> tuple:
        return ("config", SolveRunner._stable_key(config))

    @staticmethod
    def _stable_key(value: Any):
        if is_dataclass(value):
            value = asdict(value)
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, dict):
            return tuple((str(k), SolveRunner._stable_key(v)) for k, v in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(SolveRunner._stable_key(item) for item in value)
        if isinstance(value, (float, np.floating)):
            return round(float(value), 12)
        if isinstance(value, (int, bool, str)) or value is None:
            return value
        if hasattr(value, "__dict__"):
            public = {
                str(k): v
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
            return SolveRunner._stable_key(public)
        return repr(value)

    @staticmethod
    def build_radar(config: Any, sensor: SensorSpec) -> Any:
        """Construct a ``Radar`` from a RadarConfig + the sensor pose/backend."""
        import witwin.radar as wr

        return wr.Radar(
            config, backend=sensor.backend, pad_factor=sensor.pad_factor, device=sensor.device,
            position=sensor.position, target=sensor.target, up=sensor.up, fov=sensor.fov)


class SigProc:
    """Thin wrappers over ``witwin.radar.sigproc`` for the in-component views."""

    @staticmethod
    def range_doppler(radar: Any, signal: Any, *, tx: int, rx: int,
                      static_clutter_removal: bool) -> RDView:
        """Range-doppler map (dB) for one TX-RX pair + the complex map for CFAR."""
        from witwin.radar.sigproc import process_rd

        mag_db, rd_map, ranges, velocities = process_rd(
            radar, signal, tx=tx, rx=rx, static_clutter_removal=static_clutter_removal)
        return RDView(mag_db, ranges, velocities, rd_map)

    @staticmethod
    def point_cloud(radar: Any, signal: Any, *, detector: str, static_clutter_removal: bool,
                    guard: Tuple[int, int], training: Tuple[int, int], pfa: float,
                    energy_top_k: int) -> np.ndarray:
        """Filtered point cloud ``(N, 6)`` = x, y, z, v, energy_dB, r."""
        from witwin.radar.sigproc import process_pc

        return process_pc(
            radar, signal, static_clutter_removal=static_clutter_removal, detector=detector,
            guard_cells=guard, training_cells=training, pfa=pfa, energy_top_k=energy_top_k)

    @staticmethod
    def cfar_mask(rd_map: np.ndarray, *, guard: Tuple[int, int], training: Tuple[int, int],
                  pfa: float) -> np.ndarray:
        """CA-CFAR detection mask over a range-doppler magnitude map."""
        import torch
        from witwin.radar.sigproc import ca_cfar_2d_fast

        mag = torch.as_tensor(np.abs(rd_map))
        mask, _ = ca_cfar_2d_fast(mag, guard_cells=guard, training_cells=training, pfa=pfa)
        return mask.detach().cpu().numpy()

    @staticmethod
    def music_image(radar: Any, signal: Any, *, num_pixels: int = 64) -> np.ndarray:
        """MUSIC 2D image (max-projected over range). Needs a large UPA (e.g. 20x20)."""
        from witwin.radar.sigproc import MUSICImager

        cfg = radar.config
        smooth = max(1, min(3, cfg.num_tx - 1, cfg.num_rx - 1))
        imager = MUSICImager(
            num_tx=cfg.num_tx, num_rx=cfg.num_rx,
            num_signals=min(7, cfg.num_tx * cfg.num_rx - 1),
            spatial_smooth=smooth, num_pixels=num_pixels, num_chirps=cfg.chirp_per_frame)
        image3d = imager.radar_image(signal)
        if hasattr(image3d, "detach"):
            image3d = image3d.detach().cpu().numpy()
        return np.asarray(image3d).max(axis=2)
