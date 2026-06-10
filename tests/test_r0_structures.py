"""R0 round-trip: scaffolding + structures + the (Scene, RadarConfig) pair contract.

Builds a radar ``Scene`` paired with a ``RadarConfig``, runs the pair through
``to_studio`` then ``to_platform``, and asserts the rebuilt pair reproduces the
Studio-editable config, geometry, and scalar material fields.
"""
import witwin.radar as wr
import numpy as np
from witwin_server import Scene, SceneObject
from witwin_server.core.components import PlatformGeometryComponent, SkinnedMeshComponent
from witwin_server.features.solvers.scene_ref import load_scene_ref, make_scene_ref

from _helpers import assert_radar_config_equal, assert_structures_equal, approx

_CONFIG = {
    "num_tx": 3, "num_rx": 4,
    "fc": 77e9, "slope": 60.012, "power": 12.0,
    "adc_samples": 256, "adc_start_time": 6.0, "sample_rate": 4400.0,
    "idle_time": 7.0, "ramp_end_time": 58.0, "chirp_per_frame": 128,
    "frame_per_second": 10.0,
    "num_doppler_bins": 128, "num_range_bins": 256, "num_angle_bins": 64,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
}


def _build_pair():
    # A radar (Scene, RadarConfig) pair. Input metadata is deliberately not exported
    # back into Studio-authored radar structures.
    scene = wr.Scene(device="cpu")
    scene.add_mesh(
        name="wall",
        geometry=wr.Box(position=(0.0, 0.0, -4.0), size=(2.0, 2.0, 0.1)),
        material=wr.Material(eps_r=5.0, name="concrete"),
    )
    scene.add_mesh(
        name="car",
        geometry=wr.Box(position=(0.6, 0.0, -2.5), size=(0.6, 0.4, 0.4)),
        material=wr.Material(eps_r=8.0, sigma_e=0.02, name="metal"),
        bsdf={"type": "conductor", "material": "Cu"},
        dynamic=True,
    )
    scene.add_mesh(
        name="ball",
        geometry=wr.Sphere(position=(-1.0, 0.0, -2.0), radius=0.3),
        material=wr.Material(eps_r=2.0, name="plastic"),
    )
    config = wr.RadarConfig.from_dict(_CONFIG)
    return scene, config


def test_detect_claims_radar(adapter):
    scene, config = _build_pair()
    assert adapter.detect((scene, config)) is True
    assert adapter.detect(scene) is True
    assert adapter.detect(object()) is False
    assert adapter.detect((scene,)) is False


def test_config_round_trip(adapter):
    scene, config = _build_pair()
    rebuilt_scene, rebuilt_config = adapter.to_platform(adapter.to_studio((scene, config)))
    assert_radar_config_equal(config, rebuilt_config)


def test_structures_round_trip(adapter):
    scene, config = _build_pair()
    rebuilt_scene, rebuilt_config = adapter.to_platform(adapter.to_studio((scene, config)))
    assert_structures_equal(scene.structures, rebuilt_scene.structures)


def test_imported_structures_are_regular_meshes(adapter):
    studio = adapter.to_studio(_build_pair())

    targets = [obj for obj in studio.objects.values() if obj.name in {"wall", "car", "ball"}]

    assert len(targets) == 3
    for obj in targets:
        mesh = obj.get_component("Mesh")
        assert mesh is not None and not mesh.is_empty
        assert obj.get_component("PlatformGeometry") is None
        assert obj.get_component("StructureMeta") is None
        assert obj.get_component("RadarStructureMeta") is None


def test_library_target_is_plain_mesh_without_radar_sidecars(wtr):
    target = wtr.library_items._target_object()

    mesh = target.get_component("Mesh")
    assert mesh is not None and not mesh.is_empty
    assert target.get_component("RadarStructureMeta") is None
    assert target.get_component("RadarMotion") is None


def test_library_demo_target_is_plain_static_mesh(wtr):
    class Ctx:
        name = "Radar Demo"
        scene = Scene()

    settings = wtr.library_items._make_demo(Ctx)
    targets = [obj for obj in Ctx.scene.objects.values() if obj.name.startswith("Radar Target")]

    assert settings.get_component("Radar") is not None
    assert len(targets) == 1
    assert targets[0].get_component("RadarMotion") is None


def test_plain_mesh_exports_as_radar_structure(adapter):
    _, config = _build_pair()
    studio = adapter.to_studio((wr.Scene(device="cpu"), config))
    cube = SceneObject(name="plain target", mesh_type="Cube")
    cube.get_component("Transform").position = [0.5, 0.0, -1.0]
    cube.get_component("Transform").scale = [0.2, 0.4, 0.6]
    cube.get_component("Material").eps_r = 7.0
    studio.add_object(cube)

    rebuilt_scene, _ = adapter.to_platform(studio)

    assert len(rebuilt_scene.structures) == 1
    target = rebuilt_scene.structures[0]
    assert target.name == "plain target"
    assert str(target.geometry.kind) == "mesh"
    assert target.metadata.get("dynamic") is True
    assert approx(target.material.eps_r, 7.0)
    expected_bounds = ((0.4, 0.6), (-0.2, 0.2), (-1.3, -0.7))
    for got_axis, expected_axis in zip(target.geometry.bounds_world, expected_bounds):
        assert approx(got_axis[0], expected_axis[0])
        assert approx(got_axis[1], expected_axis[1])


def test_skinned_mesh_exports_posed_surface(adapter):
    _, config = _build_pair()
    studio = adapter.to_studio((wr.Scene(device="cpu"), config))
    _add_posed_skinned_target(studio)

    rebuilt_scene, _ = adapter.to_platform(studio)

    assert len(rebuilt_scene.structures) == 1
    bounds = rebuilt_scene.structures[0].geometry.bounds_world
    assert approx(bounds[0][0], 1.0)
    assert approx(bounds[0][1], 1.0)


def test_scene_ref_skinned_mesh_prefers_runtime_skin_over_smpl_geometry(adapter):
    _, config = _build_pair()
    studio = adapter.to_studio((wr.Scene(device="cpu"), config))
    body = _add_posed_skinned_target(studio)
    geom = body.add_component(PlatformGeometryComponent())
    geom.kind = "smpl"
    geom.model_root = "E:/missing/smpl/body_models"

    restored = load_scene_ref(make_scene_ref(studio))
    rebuilt_scene, _ = adapter.to_platform(restored)

    assert len(rebuilt_scene.structures) == 1
    bounds = rebuilt_scene.structures[0].geometry.bounds_world
    assert approx(bounds[0][0], 1.0)
    assert approx(bounds[0][1], 1.0)


def _add_posed_skinned_target(studio):
    body = SceneObject(name="skinned target", mesh_type="Empty")
    studio.add_object(body)
    bone = SceneObject(name="skinned target bone", mesh_type="Empty")
    studio.add_object(bone)
    studio.set_parent(bone.id, body.id, keep_local_transform=False)
    bone.get_component("Transform").position = [1.0, 0.0, 0.0]

    skinned = SkinnedMeshComponent()
    body.add_component(skinned)
    skinned.set_mesh_data(
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.uint32),
        notify=False,
    )
    skinned.set_skinning_data(
        [bone.id],
        np.zeros((3, 4), dtype=np.int64),
        np.array([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=np.float32),
        np.eye(4, dtype=np.float32).reshape(1, 4, 4),
        root_bone_id=bone.id,
    )
    return body


def test_scene_ref_preserves_in_memory_custom_mesh_for_solver(adapter):
    _, config = _build_pair()
    studio = adapter.to_studio((wr.Scene(device="cpu"), config))
    custom = SceneObject(name="custom target", mesh_type="Custom")
    custom.get_component("Mesh").set_mesh_data(
        np.array([
            [-0.25, -0.25, -1.25],
            [0.25, -0.25, -1.25],
            [0.25, 0.25, -1.25],
            [-0.25, 0.25, -1.25],
            [-0.25, -0.25, -0.75],
            [0.25, -0.25, -0.75],
            [0.25, 0.25, -0.75],
            [-0.25, 0.25, -0.75],
        ], dtype=np.float32),
        np.array([
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 7, 3], [0, 4, 7],
            [1, 2, 6], [1, 6, 5],
            [3, 7, 6], [3, 6, 2],
            [0, 1, 5], [0, 5, 4],
        ], dtype=np.uint32),
    )
    studio.add_object(custom)

    restored = load_scene_ref(make_scene_ref(studio))
    rebuilt_scene, _ = adapter.to_platform(restored)

    assert len(rebuilt_scene.structures) == 1
    assert rebuilt_scene.structures[0].name == "custom target"
    assert rebuilt_scene.structures[0].metadata.get("dynamic") is True


def test_radar_only_bsdf_metadata_is_dropped_and_meshes_are_traceable(adapter):
    scene, config = _build_pair()
    rebuilt_scene, _ = adapter.to_platform(adapter.to_studio((scene, config)))
    by_name = {s.name: s for s in rebuilt_scene.structures}
    assert by_name["car"].metadata.get("dynamic") is True
    assert "bsdf" not in by_name["car"].metadata
    assert by_name["wall"].metadata.get("dynamic") is True
    assert "bsdf" not in by_name["wall"].metadata


def test_bare_scene_defaults_config(adapter):
    scene, _ = _build_pair()
    rebuilt_scene, rebuilt_config = adapter.to_platform(adapter.to_studio(scene))
    # Bare-scene load uses the component defaults (a valid 3TX/4RX 77 GHz config).
    assert rebuilt_config.num_tx == 3 and rebuilt_config.num_rx == 4
    assert_structures_equal(scene.structures, rebuilt_scene.structures)
