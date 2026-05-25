"""Per-structure radar dynamics -> ``RadarMotion`` component <-> ``TransformMotion``.

This is the radar **motion graph** (``scene._structure_motions``): a separate, acyclic
rigid-motion graph that is **never** the studio ``Transform`` parent tree (master §5.3).
A structure's delta at time t is ``offset + velocity*(t - t_ref)`` plus an optional
rotation about ``axis`` through ``origin`` at ``angular_velocity``; ``parent`` chains one
structure's motion onto another's (by structure name).

Only structures that actually move carry this component; the adapter adds it on
``to_studio`` from ``scene.get_structure_motion(name)`` and rebuilds the motion on
``to_platform`` via ``scene.add_structure_motion`` (which validates parent-exists +
acyclicity).
"""
from witwin_server.components import (
    Component,
    bool_field,
    component,
    float_field,
    string_field,
    vector3_field,
)

_CAT = "Simulation/Radar"


@component(name="RadarMotion", category=_CAT)
class RadarMotionComponent(Component):
    """A single rigid ``TransformMotion`` for one structure (radar motion graph)."""

    offset = vector3_field([0.0, 0.0, 0.0], description="Initial position delta")
    velocity = vector3_field([0.0, 0.0, 0.0], description="Linear velocity (units/s)")
    space = string_field("world", options=["world", "local"], enum_toggle=True,
                         description="Frame the delta/axis are expressed in")
    t_ref = float_field(0.0, description="Reference time: delta = offset + velocity*(t - t_ref)")

    use_rotation = bool_field(False, description="Rotate about an axis over time")
    axis = vector3_field([0.0, 0.0, 1.0], show_if="use_rotation", description="Rotation axis (non-zero)")
    angular_velocity = float_field(0.0, show_if="use_rotation", description="Angular velocity (rad/s)")
    angle = float_field(0.0, show_if="use_rotation", description="Initial angle (rad)")
    use_origin = bool_field(False, show_if="use_rotation",
                            description="Explicit rotation pivot (else the geometry position)")
    origin = vector3_field([0.0, 0.0, 0.0], show_if="use_origin", description="Rotation pivot")

    parent = string_field("", description="Parent structure name in the motion graph (blank = none)")
