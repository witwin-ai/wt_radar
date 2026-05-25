"""Ray-tracing config -> the ``Tracer`` built inside ``radar.simulate`` (master §6.1).

Lives on the Radar Settings object. These are not part of the ``(Scene, RadarConfig)``
pair (they are solve-time tracer parameters), so they are editor state consumed by the
solve, read by ``adapter.solve.TracerSpec``.
"""
from witwin_server.components import (
    Component,
    bool_field,
    component,
    float_field,
    int_field,
    string_field,
)

_CAT = "Simulation/Radar"
_PIXEL = {"field_name": "sampling", "operator": "eq", "value": "pixel"}


@component(name="RadarTracer", category=_CAT)
class RadarTracerComponent(Component):
    """Ray tracer parameters for ``radar.simulate(...)``."""

    resolution = int_field(128, min=1, description="Ray-tracing film width = height")
    epsilon_r = float_field(5.0, min=1.0, description="Default Fresnel permittivity (per-structure eps_r overrides)")
    sampling = string_field("triangle", options=["pixel", "triangle"], enum_toggle=True,
                            description="Triangle (mesh facets) or pixel (camera rays) sampling")
    multipath = bool_field(False, show_if=_PIXEL, description="Trace reflections (pixel sampling only)")
    max_reflections = int_field(0, min=0, show_if="multipath", description="Reflection bounce count")
    ray_batch_size = int_field(65536, min=1, description="Rays per batch (multipath)")
