"""Core ``LinearTrajectory`` <-> Studio ``RadarMotion`` interchange."""
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .common import num, vec


class MotionMap:
    """Map the rigid-motion subset represented by Core ``LinearTrajectory``."""

    @staticmethod
    def to_studio(comp: Any, trajectory: Any) -> None:
        comp.offset = trajectory.origin.detach().cpu().tolist()
        comp.velocity = trajectory.velocity.detach().cpu().tolist()
        comp.t_ref = float(trajectory.reference_time_s)

        rotation = trajectory.rotation
        angular = trajectory.angular_velocity
        if rotation is None and angular is None:
            return

        comp.use_rotation = True
        rotvec = MotionMap._rotation_vector(rotation)
        angular_vec = np.zeros(3, dtype=np.float64) if angular is None else np.asarray(
            angular.detach().cpu(), dtype=np.float64
        )
        angle = float(np.linalg.norm(rotvec))
        speed = float(np.linalg.norm(angular_vec))
        axis_source = rotvec if angle > 1e-12 else angular_vec
        axis_norm = float(np.linalg.norm(axis_source))
        axis = np.array((0.0, 0.0, 1.0)) if axis_norm <= 1e-12 else axis_source / axis_norm
        comp.axis = axis.tolist()
        comp.angle = angle
        comp.angular_velocity = speed * (-1.0 if float(np.dot(angular_vec, axis)) < 0.0 else 1.0)

    @staticmethod
    def build(comp: Any) -> Optional[Any]:
        """Build a Core trajectory, refusing legacy graph features Core does not represent."""
        import torch
        from witwin.core import LinearTrajectory

        offset = vec(comp.offset)
        velocity = vec(comp.velocity)
        use_rotation = bool(comp.use_rotation)
        if not any(abs(value) > 0.0 for value in offset + velocity) and not use_rotation:
            return None

        rotation = angular_velocity = None
        if use_rotation:
            axis = np.asarray(vec(comp.axis), dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                raise ValueError("RadarMotion.axis must be non-zero")
            axis /= norm
            quaternion_xyzw = Rotation.from_rotvec(axis * num(comp.angle)).as_quat()
            rotation = torch.tensor(
                (quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]),
                dtype=torch.float32,
            )
            angular_velocity = torch.tensor(axis * num(comp.angular_velocity), dtype=torch.float32)

        return LinearTrajectory(
            origin=torch.tensor(offset, dtype=torch.float32),
            velocity=torch.tensor(velocity, dtype=torch.float32),
            rotation=rotation,
            angular_velocity=angular_velocity,
            reference_time_s=num(comp.t_ref),
        )

    @staticmethod
    def _rotation_vector(rotation: Any) -> np.ndarray:
        if rotation is None:
            return np.zeros(3, dtype=np.float64)
        value = np.asarray(rotation.detach().cpu(), dtype=np.float64)
        if value.shape == (3,):
            return Rotation.from_euler("zyx", value).as_rotvec()
        if value.shape == (4,):
            return Rotation.from_quat((value[1], value[2], value[3], value[0])).as_rotvec()
        raise ValueError(f"Core trajectory rotation must have shape (3,) or (4,), got {value.shape}")
