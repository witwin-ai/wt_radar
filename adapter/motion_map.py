"""Motion-graph interchange: ``witwin.radar.TransformMotion`` <-> ``RadarMotion``.

``to_studio`` fills a ``RadarMotion`` component from a platform ``TransformMotion``;
``build`` reconstructs a ``TransformMotion`` (or ``None`` when the component is trivial,
so an empty motion the user never edited is simply ignored). Acyclicity / parent-exists
are validated by ``scene.add_structure_motion`` on the platform side, not duplicated here.
"""
from typing import Any, Optional

from .common import num, vec


class MotionMap:
    """``TransformMotion`` <-> ``RadarMotion`` component."""

    @staticmethod
    def to_studio(comp: Any, motion: Any) -> None:
        """Fill a ``RadarMotion`` component from a platform ``TransformMotion``."""
        comp.offset = motion.offset.tolist()
        comp.velocity = motion.velocity.tolist()
        comp.space = str(motion.space)
        comp.t_ref = float(motion.t_ref.item())
        comp.parent = motion.parent or ""
        if motion.axis is not None:
            comp.use_rotation = True
            comp.axis = motion.axis.tolist()
            comp.angular_velocity = float(motion.angular_velocity.item())
            comp.angle = float(motion.angle.item())
            if motion.origin is not None:
                comp.use_origin = True
                comp.origin = motion.origin.tolist()

    @staticmethod
    def build(comp: Any) -> Optional[Any]:
        """``RadarMotion`` -> ``TransformMotion``, or ``None`` when the motion is trivial."""
        from witwin.radar import TransformMotion

        offset = vec(comp.offset)
        velocity = vec(comp.velocity)
        use_rotation = bool(comp.use_rotation)
        parent = str(comp.parent).strip() or None
        has_translation = any(abs(c) > 0.0 for c in offset + velocity)
        if not has_translation and not use_rotation and parent is None:
            return None  # nothing to move: treat as no motion

        axis = vec(comp.axis) if use_rotation else None
        origin = vec(comp.origin) if (use_rotation and bool(comp.use_origin)) else None
        return TransformMotion(
            offset=offset,
            velocity=velocity,
            axis=axis,
            angular_velocity=num(comp.angular_velocity) if use_rotation else 0.0,
            angle=num(comp.angle) if use_rotation else 0.0,
            origin=origin,
            space=str(comp.space),
            t_ref=num(comp.t_ref),
            parent=parent,
        )
