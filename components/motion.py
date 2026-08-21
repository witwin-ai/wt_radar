"""Per-structure Core ``LinearTrajectory`` authoring component."""
from witwin_server.core.components import (
    Component,
    bool_field,
    component,
    float_field,
    vector3_field,
)

_CAT = "Simulation/Radar"


@component(name="RadarMotion", category=_CAT)
class RadarMotionComponent(Component):
    """A Core world-frame linear and angular trajectory for one structure."""

    offset = vector3_field([0.0, 0.0, 0.0], description="Initial position delta")
    velocity = vector3_field([0.0, 0.0, 0.0], description="Linear velocity (units/s)")
    t_ref = float_field(0.0, description="Reference time: delta = offset + velocity*(t - t_ref)")

    use_rotation = bool_field(False, description="Rotate about an axis over time")
    axis = vector3_field([0.0, 0.0, 1.0], show_if="use_rotation", description="Rotation axis (non-zero)")
    angular_velocity = float_field(0.0, show_if="use_rotation", description="Angular velocity (rad/s)")
    angle = float_field(0.0, show_if="use_rotation", description="Initial angle (rad)")
