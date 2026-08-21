"""Multi-frame generation through repeated Radar 0.3 scene-driven solves."""
from typing import Any, Optional, Tuple

import numpy as np

from .solve import PropagationSpec, SensorSpec, SolveRunner


class TimelineRunner:
    """Generate a frame cube from explicit point sites or Core scene motion."""

    @staticmethod
    def generate(
        comp: Any,
        *,
        sensor: SensorSpec,
        propagation: PropagationSpec,
        studio_scene: Any,
    ) -> Tuple[Any, Any]:
        import torch

        frame_rate = max(float(comp.frame_rate), 1e-6)
        source = str(comp.timeline_source)
        point_frames = None
        if source == "pointcloud_sequence":
            point_frames, _ = TimelineRunner._load_pointclouds(str(comp.pointcloud_path), "cuda")
            frame_count = int(point_frames.shape[0])
        elif source == "scene_motion":
            frame_count = int(comp.motion_frame_count)
        else:
            raise ValueError("timeline_source must be 'pointcloud_sequence' or 'scene_motion'")

        cache = {}
        signals = []
        last = None
        for index in range(frame_count):
            time_s = index / frame_rate
            sites = None if point_frames is None else point_frames[index]
            last = SolveRunner.run(
                studio_scene,
                sensor=sensor,
                propagation=propagation,
                t0=time_s,
                live_cache=cache,
                cache_key=f"timeline:{index}",
                platform_cache_key="timeline:scene",
                site_positions=sites,
            )
            signals.append(last.signal)
        if last is None:
            raise ValueError("timeline requires at least one frame")
        return last.radar, torch.stack(signals, dim=0)

    @staticmethod
    def _load_pointclouds(path: str, device: str) -> Tuple[Any, Optional[Any]]:
        import torch

        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            positions = torch.as_tensor(data["positions"], dtype=torch.float32, device=device)
            intensities = (
                torch.as_tensor(data["intensities"], dtype=torch.float32, device=device)
                if "intensities" in data.files
                else None
            )
            return positions, intensities
        return torch.as_tensor(data, dtype=torch.float32, device=device), None
