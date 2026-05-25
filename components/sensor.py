"""Radar sensor pose + solver backend -> the ``Radar(...)`` constructor (master §3.6).

These fields are **not** part of ``RadarConfig`` (the platform passes them straight to
``Radar(position=, target=, up=, fov=, backend=, pad_factor=, device=)``), so they are
editor-authored state consumed at solve time (R3) rather than round-tripped from the
``(Scene, RadarConfig)`` pair. The pose here is the radar's own look-at frame; it is
independent of any structure's studio ``Transform`` and of the radar motion graph.

The gizmo reuses the old plugin's pattern: a sphere at the sensor + an FOV/range cone,
with the range taken from the derived max range of the sibling ``RadarConfig``.
"""
from witwin_server.components import (
    Component,
    GizmoContext,
    bool_field,
    component,
    float_field,
    int_field,
    string_field,
    vector3_field,
)

from ..adapter.common import Derived, vec

_CAT = "Simulation/Radar"
_DIRICHLET = {"field_name": "backend", "operator": "eq", "value": "dirichlet"}


@component(name="RadarSensor", category=_CAT)
class RadarSensorComponent(Component):
    """Radar pose (look-at) + backend selection for ``Radar(...)``."""

    position = vector3_field([0.0, 0.0, 0.0], description="Radar origin in world coords (m)")
    use_target = bool_field(False, description="Explicit look-at target (else position + (0,0,-1))")
    target = vector3_field([0.0, 0.0, -1.0], show_if="use_target",
                           description="Look-at target in world coords (m)")
    up = vector3_field([0.0, 1.0, 0.0], description="World-space up vector")
    fov = float_field(60.0, min=1.0, max=179.0, description="Ray-tracing field of view (deg)")

    backend = string_field("dirichlet", options=["dirichlet", "slang", "pytorch"], enum_toggle=True,
                           description="Solver backend (dirichlet/slang need CUDA; pytorch runs on CPU)")
    pad_factor = int_field(16, min=1, show_if=_DIRICHLET, description="FFT zero-padding factor (dirichlet backend)")
    device = string_field("cuda", options=["cuda", "cpu"], enum_toggle=True,
                          description="Compute device (cpu allowed only for the pytorch backend)")

    def on_draw_gizmos(self, ctx: GizmoContext) -> None:
        """Sphere at the sensor + an FOV/range cone toward the look-at direction."""
        ctx.color = "#ffd400"
        ctx.draw_sphere(radius=0.05)
        ctx.draw_cone(fov="fov", range=self._max_range(), segments=4, direction=self._forward())

    def _forward(self):
        # Unit look-at direction (defaults to -Z when no explicit target).
        if not bool(self.use_target):
            return (0.0, 0.0, -1.0)
        pos, tgt = vec(self.position), vec(self.target)
        d = [tgt[i] - pos[i] for i in range(3)]
        norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) ** 0.5
        return (0.0, 0.0, -1.0) if norm <= 1e-12 else (d[0] / norm, d[1] / norm, d[2] / norm)

    def _max_range(self) -> float:
        # Derived max range from the sibling RadarConfig (fallback to a default).
        config = self.owner.get_component("RadarConfig") if self.owner else None
        return Derived.compute(config)["max_range_m"] if config is not None else 10.0
