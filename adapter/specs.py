"""Studio-authored sensor pose and tracing request values.

These small transport types are shared by the Radar 0.4 snapshot/animation UI
and agent contracts.  They deliberately contain no solver implementation and
do not import ``witwin.radar`` or ``witwin.channel``.
"""
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from witwin_server.utils.logging import get_logger

from .common import num

logger = get_logger("RadarSpecs")


@dataclass
class SensorSpec:
    """Radar pose and runtime selection derived from the owner object."""

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
    """Return world position, look-at target, and up from a Studio transform."""
    from witwin_server.utils.mitsuba_utils import get_world_transform

    matrix = np.asarray(get_world_transform(obj), dtype=np.float64)
    position = matrix[:3, 3].tolist()
    rotation = matrix[:3, :3]
    forward = _normalize(rotation @ np.array([0.0, 0.0, -1.0]), fallback=(0.0, 0.0, -1.0))
    upvec = _normalize(rotation @ np.array([0.0, 1.0, 0.0]), fallback=(0.0, 1.0, 0.0))
    target = [position[i] + forward[i] for i in range(3)]
    logger.info(
        f"_world_pose[{getattr(obj, 'name', '?')}]: "
        f"position={[round(p, 4) for p in position]} "
        f"forward={[round(f, 4) for f in forward]} "
        f"up={[round(u, 4) for u in upvec]}"
    )
    return position, target, list(upvec)


def _normalize(vector: np.ndarray, *, fallback: tuple) -> tuple:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return fallback
    return tuple((vector / norm).tolist())


@dataclass
class TracerSpec:
    """Authored propagation controls carried in a Radar solver request."""

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


__all__ = ["SensorSpec", "TracerSpec"]
