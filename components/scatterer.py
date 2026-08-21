"""Explicit Radar 0.3 scatter-site authoring for one Studio object."""
from witwin_server.core.components import (
    Component,
    bool_field,
    component,
    int_field,
    string_field,
    vector3_field,
)

_CAT = "Simulation/Radar"


@component(name="RadarScatterer", display_name="Radar Scatterer", category=_CAT)
class RadarScattererComponent(Component):
    """Explicit point or deterministic surface sites attached to a structure."""

    enabled = bool_field(True, description="Include this explicit site in Radar simulation")
    mode = string_field(
        "point",
        options=["point", "surface"],
        description="Use one local point or stable area-weighted sites over the current mesh surface",
    )
    site_offset = vector3_field(
        [0.0, 0.0, 0.0],
        description="Site position in the object's local frame; place it on or just outside a surface",
    )
    surface_site_count = int_field(
        128, min=1, max=4096, description="Number of stable area-weighted sites in surface mode"
    )
    surface_seed = int_field(0, description="Deterministic surface sampling seed")
