"""Radar scene components (auto-registered via their ``@component`` decorators).

Importing this package registers every radar component with the core
``ComponentRegistry``. The base layer's shared components (PlatformGeometry /
EMMaterial / StructureMeta) stay in core; these are the radar-specific additions.
"""
from .config import RadarConfigComponent
from .structure_meta import RadarStructureMetaComponent

__all__ = [
    "RadarConfigComponent",
    "RadarStructureMetaComponent",
]
