"""Explicit Radar 0.3 scatter-site authoring for one Studio object."""
from witwin_server.core.components import Component, bool_field, component, vector3_field

_CAT = "Simulation/Radar"


@component(name="RadarScatterer", display_name="Radar Scatterer", category=_CAT)
class RadarScattererComponent(Component):
    """One explicitly authored scatter site attached to a structure."""

    enabled = bool_field(True, description="Include this explicit site in Radar simulation")
    site_offset = vector3_field(
        [0.0, 0.0, 0.0],
        description="Site position in the object's local frame; place it on or just outside a surface",
    )
