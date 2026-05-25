"""Multi-frame timeline generation (master §5.2) -> ``witwin.radar.Timeline``.

Builds a ``Timeline`` from the ``RadarTimeline`` component's source descriptor (a point
cloud sequence file, or an SMPL motion file rendered against the scene) and generates the
MIMO frame stack ``(num_radar_frames, TX, RX, chirps, ADC)``. Keyframe tensors are large
runtime/result state — the editor stores only the source path, never baked keyframes.

The point-cloud-sequence source needs no ray tracing (the points drive ``radar.mimo``
directly). The motion source renders SMPL keyframes through a ``Tracer`` and therefore
needs CUDA + SMPL model files and a structure named ``human``.
"""
from typing import Any, Optional, Tuple

import numpy as np

from .common import num
from .solve import SensorSpec, SolveRunner, TracerSpec


class TimelineRunner:
    """Build a ``Timeline`` + ``Radar`` from a RadarTimeline component and generate frames."""

    @staticmethod
    def generate(comp: Any, *, sensor: SensorSpec, tracer: TracerSpec,
                 studio_scene: Any) -> Tuple[Any, Any]:
        """Return ``(radar, frames)`` where frames is ``(num_radar_frames, TX, RX, chirps, ADC)``."""
        import torch
        import witwin.radar as wr

        from .radar_adapter import RadarAdapter

        device = "cuda" if torch.cuda.is_available() else "cpu"
        scene, config = RadarAdapter().to_platform(studio_scene, device=device)
        radar = SolveRunner.build_radar(config, sensor)

        timeline = wr.Timeline(frame_rate=num(comp.frame_rate), device=device)
        if str(comp.source) == "pointcloud_sequence":
            positions, intensities = TimelineRunner._load_pointclouds(str(comp.pointcloud_path), device)
            timeline.add_pointcloud_sequence(positions, intensities)
        else:
            tracer_obj = wr.Tracer(scene, radar, resolution=tracer.resolution,
                                   epsilon_r=tracer.epsilon_r, sampling="triangle")
            motion = dict(np.load(str(comp.motion_path)))
            timeline.from_motion(scene, tracer_obj, motion)

        frames = timeline.generate(radar, progress=False, velocity_corrected=bool(comp.velocity_corrected))
        return radar, frames

    @staticmethod
    def _load_pointclouds(path: str, device: str) -> Tuple[Any, Optional[Any]]:
        # Load a (F, N, 3) point-cloud sequence (+ optional (F, N) intensities) from .npy/.npz.
        import torch

        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            positions = torch.as_tensor(data["positions"], dtype=torch.float32, device=device)
            intensities = (torch.as_tensor(data["intensities"], dtype=torch.float32, device=device)
                           if "intensities" in data.files else None)
            return positions, intensities
        return torch.as_tensor(data, dtype=torch.float32, device=device), None
