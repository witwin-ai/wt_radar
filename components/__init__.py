"""Radar scene components (auto-registered via their ``@component`` decorators).

Importing this package registers every radar component with the core
``ComponentRegistry``. The base layer's shared components (PlatformGeometry /
EMMaterial / StructureMeta) stay in core; these are the radar-specific additions.
"""
from .config import RadarConfigComponent
from .motion import RadarMotionComponent
from .result import RadarResultComponent
from .sensor import RadarSensorComponent
from .structure_meta import RadarStructureMetaComponent
from .subconfigs import (
    RadarAntennaPatternComponent,
    RadarNoiseModelComponent,
    RadarPolarizationComponent,
    RadarReceiverChainComponent,
)
from .tracer import RadarTracerComponent

__all__ = [
    "RadarConfigComponent",
    "RadarMotionComponent",
    "RadarResultComponent",
    "RadarSensorComponent",
    "RadarStructureMetaComponent",
    "RadarTracerComponent",
    "RadarAntennaPatternComponent",
    "RadarNoiseModelComponent",
    "RadarPolarizationComponent",
    "RadarReceiverChainComponent",
]
