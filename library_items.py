"""Radar library prefabs — drag-to-create scene building blocks (master §8).

Registers a ``Radar`` category with:

- **Radar (Demo)** — a Radar Settings object (77 GHz FMCW defaults + sensor pose/backend
  + tracer + the Simulate button) plus a moving box target, so dropping it in is
  immediately simulable: press **Simulate** on the Radar Settings object to see the
  range-doppler / point-cloud views in-component.
- **Radar Settings** — just the sensor singleton, to compose a custom scene.
- **Radar Target** — a plain box mesh with radar-friendly material defaults.

Human/SMPL bodies are owned by the wt-human plugin, not radar; a radar scene that contains
one round-trips through the shared base geometry map with no radar-side SMPL code.

Each prefab reuses the tested adapter maps so geometry / materials / components match the
round trip. Runtime/result state is never part of a prefab.
"""
from witwin_server import Library

from .adapter.config_map import ConfigMap

_CATEGORY = "radar"


def _settings_object(name="Radar Settings"):
    # The Radar Settings singleton with default 77 GHz FMCW config + sensor/tracer/result.
    obj = ConfigMap.settings_to_studio(None)
    obj.name = name
    return obj


def _target_object(name="Radar Target"):
    # A plain Studio mesh: the adapter exports visible meshes directly as radar structures.
    from witwin_server import SceneObject

    obj = SceneObject(name=name, mesh_type="Cube")
    transform = obj.get_component("Transform")
    transform.position = [0.6, 0.0, -2.5]
    transform.scale = [0.6, 0.4, 0.4]
    material = obj.get_component("Material")
    if material is not None:
        material.eps_r = 8.0
        material.material_name = "metal"
    return obj


# --- library factories (context.scene gets the extras; return the primary obj) ---

def _make_demo(ctx):
    settings = _settings_object(ctx.name)
    # Unique the target name so dropping the demo more than once does not produce two
    # structures with the same name (the platform wr.Scene.add_structure rejects dups).
    base = "Radar Target"
    existing = {obj.name for obj in ctx.scene.objects.values()}
    name = base
    i = 2
    while name in existing:
        name = f"{base} {i}"
        i += 1
    ctx.scene.add_object(_target_object(name))
    return settings


def _make_settings(ctx):
    return _settings_object(ctx.name)


def _make_target(ctx):
    obj = _target_object(ctx.name)
    return obj


def register() -> None:
    """Register the Radar library category + prefab items."""
    Library.register_category(_CATEGORY, "Radar", icon="radar", order=30)
    Library.register_item("radar_demo", "Radar (Demo)", _CATEGORY,
                          icon="radar", object_type="empty", factory=_make_demo,
                          description="Sensor + a plain mesh target; Simulate to see the range-doppler signal")
    Library.register_item("radar_settings", "Radar Settings", _CATEGORY,
                          icon="settings", object_type="empty", factory=_make_settings,
                          description="FMCW sensor config + Simulate button (build your own scene)")
    Library.register_item("radar_target", "Radar Target", _CATEGORY,
                          icon="box", object_type="mesh", factory=_make_target,
                          description="A plain box mesh exported directly as a radar structure")


register()
