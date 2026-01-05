"""
Library items provided by the Radar Simulation extension.

Registers the Radar object in the RF category.
"""
from witwin import Library, CreateContext
from witwin.core import SceneObject


# Factory function for creating radar objects
def create_radar(ctx: CreateContext) -> SceneObject:
    """Create a radar object with the Radar component."""
    obj = SceneObject(name=ctx.name, mesh_type="Empty")
    obj.add_component("Radar")
    # Radar has direction but no meaningful scale
    obj["Transform"].freeze_scale = True
    return obj


# Register the library item in the RF category
Library.register_item(
    id="radar",
    name="Radar",
    category="rf",
    icon="radar",
    description="Radar sensor for simulation",
    factory=create_radar,
    object_type="radar"
)
