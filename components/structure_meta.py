"""Per-structure radar metadata -> ``RadarStructureMeta`` component.

Radar ``Structure`` objects carry two extra metadata keys beyond the shared
``StructureMeta`` (priority/enabled/tags/opaque metadata): ``metadata["dynamic"]``
(moving / triangle-sampled) and ``metadata["bsdf"]`` (an opaque Mitsuba BSDF override
dict). The adapter lifts those two keys out of the base ``StructureMeta`` into this
dedicated component for a cleaner UI; the remaining opaque metadata stays on
``StructureMeta``.
"""
import json
from typing import Optional

from witwin_server.components import Component, bool_field, component, string_field

_CAT = "Simulation/Radar"


@component(name="RadarStructureMeta", category=_CAT)
class RadarStructureMetaComponent(Component):
    """Radar ``metadata['dynamic']`` + ``metadata['bsdf']`` for one structure."""

    dynamic = bool_field(False, description="Moving / triangle-sampled (metadata['dynamic'])")
    bsdf_json = string_field("", widget="textarea",
                             description="Mitsuba BSDF override as JSON (metadata['bsdf']); blank = none")

    @property
    def bsdf(self) -> Optional[dict]:
        """Parsed BSDF override dict, or ``None`` when blank."""
        text = str(self.bsdf_json).strip()
        return json.loads(text) if text else None

    @bsdf.setter
    def bsdf(self, value: Optional[dict]) -> None:
        self.bsdf_json = "" if not value else json.dumps(dict(value))
