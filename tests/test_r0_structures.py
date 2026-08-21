"""Core Scene/DynamicScene and explicit-site adapter coverage."""
import numpy as np

from witwin_server import Scene, SceneObject

from _helpers import assert_config_equal, bounds, make_config, make_scene


def _settings(adapter):
    studio = adapter.to_studio((make_scene(), make_config()))
    return next(obj for obj in studio.objects.values() if obj.get_component("Radar") is not None)


def test_detects_current_core_scene_types(adapter):
    assert adapter.detect(make_scene())
    assert adapter.detect(make_scene(dynamic=True))
    assert adapter.detect((make_scene(), make_config()))
    assert not adapter.detect(object())


def test_scene_and_config_round_trip(adapter):
    original = make_scene()
    config = make_config()
    rebuilt, rebuilt_config = adapter.to_platform(adapter.to_studio((original, config)))

    assert_config_equal(config, rebuilt_config)
    assert [item.name for item in rebuilt.structures] == ["wall", "car"]
    for left, right in zip(original.structures, rebuilt.structures, strict=True):
        np.testing.assert_allclose(bounds(left.geometry), bounds(right.geometry), rtol=1e-6, atol=1e-5)
        assert right.structure_id is not None
        assert right.material_id is not None


def test_imported_targets_are_normal_meshes(adapter):
    studio = adapter.to_studio((make_scene(), make_config()))
    targets = [obj for obj in studio.objects.values() if obj.name in {"wall", "car"}]
    assert len(targets) == 2
    assert all(obj.get_component("Mesh") is not None for obj in targets)
    assert all(obj.get_component("PlatformGeometry") is None for obj in targets)


def test_plain_mesh_exports_structure_and_explicit_site(adapter):
    from wt_radar.components.scatterer import RadarScattererComponent

    studio = Scene()
    studio.add_object(_settings(adapter))
    target = SceneObject(name="target", mesh_type="Cube")
    target.get_component("Transform").position = [0.5, 0.0, -2.0]
    target.get_component("Transform").scale = [0.2, 0.4, 0.6]
    target.add_component(RadarScattererComponent())
    studio.add_object(target)

    exported = adapter.build_export(studio)

    assert len(exported.scene.structures) == 1
    assert exported.site_positions_m == ((0.5, 0.0, -2.0),)
    assert exported.site_structure_ids == (1,)
    assert exported.scene.structures[0].metadata["dynamic"] is True


def test_library_demo_exports_with_latest_core(adapter, wtr):
    studio = Scene()
    studio.add_object(wtr.library_items._settings_object())
    studio.add_object(wtr.library_items._target_object())
    exported = adapter.build_export(studio)
    assert len(exported.scene.structures) == 1
    assert len(exported.site_positions_m) == 1
    assert exported.config.num_tx == 3
