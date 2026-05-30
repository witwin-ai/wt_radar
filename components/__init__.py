"""Radar scene components (auto-registered via their ``@component`` decorators).

The unified ``Radar`` component carries every settings-level radar field
(FMCW config + sensor pose + tracer + sub-configs + solve + timeline) under
foldout groups; optional motion and the polymorphic post-processor hierarchy stay
separate because they live on other scene objects.
"""
from .motion import RadarMotionComponent
from .post_processing import (
    PointCloudProcessorComponent,
    RadarPostProcessorComponent,
    RangeDopplerProcessorComponent,
)
from .radar import RadarComponent

__all__ = [
    "RadarComponent",
    "RadarMotionComponent",
    "RadarPostProcessorComponent",
    "RangeDopplerProcessorComponent",
    "PointCloudProcessorComponent",
]
