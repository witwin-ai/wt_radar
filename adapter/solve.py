"""Solve hand-off + sigproc views (export -> solve -> in-component display, master §7.2).

``SolveRunner`` is the export->solve half of the edit loop: it rebuilds the platform
``(Scene, RadarConfig)`` pair from the live studio scene (on a CUDA scene, because the
mitsuba ray tracer is CUDA-only), constructs a ``Radar`` from the RadarSensor pose/backend,
and runs ``radar.simulate(...)`` to produce the MIMO data cube. ``SigProc`` wraps the
platform ``sigproc`` (range-doppler / point cloud / MUSIC / CFAR) for the in-component
figure. Result tensors are runtime state — never serialized into the scene.

Live solve needs CUDA regardless of the solver backend (the tracer is mitsuba-CUDA); the
dirichlet/slang solver backends also need CUDA, while pytorch runs its solve math on CPU.
"""
from dataclasses import dataclass
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
            motion_sampling: str, t0: float) -> SolveResult:
        """Rebuild the (Scene, RadarConfig) pair on CUDA, construct the Radar, and simulate."""
        import torch
        import witwin.radar as wr

        from .radar_adapter import RadarAdapter

        scene_device = "cuda" if torch.cuda.is_available() else "cpu"
        scene, config = RadarAdapter().to_platform(studio_scene, device=scene_device)
        logger.info(
            f"SolveRunner.run: scene_device={scene_device}, "
            f"{len(scene.structures)} structures, tracer={tracer.sampling}@{tracer.resolution}")
        radar = SolveRunner.build_radar(config, sensor)
        logger.info(
            f"  wr.Radar built: position={radar.position.tolist()} "
            f"target={radar.target.tolist()} tx_pos[0]={radar.tx_pos[0].tolist()}")
        signal = radar.simulate(
            scene, resolution=tracer.resolution, epsilon_r=tracer.epsilon_r,
            sampling=tracer.sampling, multipath=tracer.multipath,
            max_reflections=tracer.max_reflections, ray_batch_size=tracer.ray_batch_size,
            t0=t0, motion_sampling=motion_sampling)
        sig_abs = signal.detach().cpu().abs()
        logger.info(
            f"  radar.simulate done: shape={tuple(signal.shape)} "
            f"mean|sig|={sig_abs.mean().item():.6e} max|sig|={sig_abs.max().item():.6e}")
        return SolveResult(radar=radar, signal=signal)

    @staticmethod
    def run_group(studio_scene: Any, *, sensors: "dict[str, SensorSpec]", tracer: TracerSpec,
                  motion_sampling: str, t0: float) -> "dict[str, Any]":
        """Run several named radars on one rebuilt scene (``Radar.simulate_group``)."""
        import torch
        import witwin.radar as wr

        from .radar_adapter import RadarAdapter

        scene_device = "cuda" if torch.cuda.is_available() else "cpu"
        scene, config = RadarAdapter().to_platform(studio_scene, device=scene_device)
        radars = {name: SolveRunner.build_radar(config, spec) for name, spec in sensors.items()}
        return wr.Radar.simulate_group(
            scene, radars=radars, resolution=tracer.resolution, epsilon_r=tracer.epsilon_r,
            sampling=tracer.sampling, multipath=tracer.multipath,
            max_reflections=tracer.max_reflections, ray_batch_size=tracer.ray_batch_size,
            t0=t0, motion_sampling=motion_sampling)

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
