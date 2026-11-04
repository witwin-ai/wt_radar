"""Public Python API of wt_radar for other Plugins (D16 layer C).

Other Plugins import only this module, and only after listing
``witwin.radar`` in their ``pluginDependencies``. Every other module here is
private and may change without notice.
"""
from __future__ import annotations

from .install_rule import (
    InstallRule,
    InstallRuleError,
    SensingZone,
    event_in_view,
    plan_installation,
    room_geometry_from_scene,
)
# Samples radar scatter sites from a Studio skinned mesh; wt_genesis checks
# its baked playback proxies against it.
from .adapter.studio_motion import StudioSkinSampler

__all__ = [
    "StudioSkinSampler",
    "InstallRule",
    "InstallRuleError",
    "SensingZone",
    "event_in_view",
    "plan_installation",
    "room_geometry_from_scene",
]
