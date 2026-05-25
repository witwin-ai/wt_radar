"""Radar library prefabs — drag-to-create scene building blocks.

R0 registers the ``Radar`` category and a **Radar Settings** prefab (the FMCW sensor
config singleton, built from the same component defaults the adapter round-trips). The
full starter (settings + a moving target so it is immediately simulable) plus the
sensor / moving-target / SMPL items land in R3 alongside the solve + result views.
"""
from witwin_server import Library

from .adapter.config_map import ConfigMap

_CATEGORY = "radar"


def _make_settings(ctx):
    # A Radar Settings object with default 77 GHz FMCW config.
    obj = ConfigMap.settings_to_studio(None)
    obj.name = ctx.name
    return obj


def register() -> None:
    """Register the Radar library category + prefab items."""
    Library.register_category(_CATEGORY, "Radar", icon="radar", order=30)
    Library.register_item("radar_settings", "Radar Settings", _CATEGORY,
                          icon="settings", object_type="empty", factory=_make_settings,
                          description="FMCW sensor config singleton (build your own scene)")


register()
