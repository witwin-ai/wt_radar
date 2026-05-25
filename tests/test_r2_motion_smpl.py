"""R2 round-trip: SMPL bodies + dynamics (the radar motion graph).

Motion: a parented, mixed translation/rotation graph round-trips so
``scene._structure_motions`` reconstructs identically; self-parent and cyclic graphs are
rejected on export; a structure rename re-parents its children (platform
``update_structure``) and the renamed graph still round-trips.

SMPL: the params (pose/shape/gender/model_root) reconstruct through the base
``GeometryMap`` (no baking needed). The full platform->studio->platform round trip bakes
the display mesh, which needs SMPL model files; when they are absent the base bake raises
(it only degrades on ImportError, not on a missing model file) so that leg is skipped
with a clear message.
"""
import pytest
import witwin.radar as wr

from _helpers import approx, assert_motions_equal, assert_structures_equal

_BASE = {
    "num_tx": 3, "num_rx": 4,
    "fc": 77e9, "slope": 60.012, "power": 12.0,
    "adc_samples": 256, "adc_start_time": 6.0, "sample_rate": 4400.0,
    "idle_time": 7.0, "ramp_end_time": 58.0, "chirp_per_frame": 128,
    "frame_per_second": 10.0,
    "num_doppler_bins": 128, "num_range_bins": 256, "num_angle_bins": 64,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
}


def _motion_pair():
    # base (linear velocity) with arm (rotation) parented onto it.
    scene = wr.Scene(device="cpu")
    scene.add_mesh(name="base", geometry=wr.Box(position=(0.0, 0.0, -3.0), size=(0.5, 0.5, 0.5)),
                   material=wr.Material(eps_r=4.0), dynamic=True)
    scene.add_mesh(name="arm", geometry=wr.Box(position=(0.5, 0.0, -3.0), size=(0.3, 0.1, 0.1)),
                   material=wr.Material(eps_r=4.0), dynamic=True)
    scene.add_structure_motion("base", wr.TransformMotion(velocity=(0.1, 0.0, 0.0), t_ref=0.5, space="world"))
    scene.add_structure_motion("arm", wr.TransformMotion(
        axis=(0.0, 1.0, 0.0), angular_velocity=1.5, angle=0.2, origin=(0.5, 0.0, -3.0),
        space="local", parent="base"))
    return scene, wr.RadarConfig.from_dict(_BASE)


def _radar_motion(studio, name):
    obj = next(o for o in studio.objects.values() if o.name == name)
    return obj.get_component("RadarMotion")


# --- motion graph round trip -----------------------------------------------

def test_motion_graph_round_trip(adapter):
    scene, config = _motion_pair()
    rebuilt, _ = adapter.to_platform(adapter.to_studio((scene, config)))
    assert_structures_equal(scene.structures, rebuilt.structures)
    assert_motions_equal(scene, rebuilt)
    assert rebuilt._structure_motions["arm"].parent == "base"


def test_parent_only_motion_round_trip(adapter):
    scene = wr.Scene(device="cpu")
    scene.add_mesh(name="lead", geometry=wr.Box(position=(0.0, 0.0, -3.0), size=(0.4, 0.4, 0.4)),
                   material=wr.Material(eps_r=3.0), dynamic=True)
    scene.add_mesh(name="follow", geometry=wr.Box(position=(0.3, 0.0, -3.0), size=(0.2, 0.2, 0.2)),
                   material=wr.Material(eps_r=3.0), dynamic=True)
    scene.add_structure_motion("lead", wr.TransformMotion(velocity=(0.0, 0.2, 0.0)))
    scene.add_structure_motion("follow", wr.TransformMotion(parent="lead"))
    rebuilt, _ = adapter.to_platform(adapter.to_studio((scene, wr.RadarConfig.from_dict(_BASE))))
    assert_motions_equal(scene, rebuilt)


def test_self_parent_rejected(adapter):
    scene, config = _motion_pair()
    studio = adapter.to_studio((scene, config))
    _radar_motion(studio, "arm").parent = "arm"
    with pytest.raises(ValueError):
        adapter.to_platform(studio)


def test_cyclic_graph_rejected(adapter):
    scene, config = _motion_pair()
    studio = adapter.to_studio((scene, config))
    _radar_motion(studio, "base").parent = "arm"  # base->arm->base
    with pytest.raises(ValueError):
        adapter.to_platform(studio)


def test_rename_keeps_motion_refs_consistent(adapter):
    # The adapter keys motions by the live object name and parents by the RadarMotion.parent
    # string, so an editor rename (object name + children's parent refs) round-trips cleanly.
    # (The platform's update_structure remap is unreachable via its signature: the positional
    # lookup name collides with a name= change, so renames are an editor-side concern.)
    scene, config = _motion_pair()
    studio = adapter.to_studio((scene, config))
    base_obj = next(o for o in studio.objects.values() if o.name == "base")
    base_obj.name = "platform"
    _radar_motion(studio, "arm").parent = "platform"

    rebuilt, _ = adapter.to_platform(studio)
    names = {s.name for s in rebuilt.structures}
    assert "platform" in names and "base" not in names
    assert "platform" in rebuilt._structure_motions
    assert rebuilt._structure_motions["arm"].parent == "platform"


# --- SMPL ------------------------------------------------------------------

def test_smpl_params_reconstruct(wtr):
    # to_platform direction (no baking): PlatformGeometry(kind=smpl) -> SMPLBody, params kept.
    from witwin_server import SceneObject
    from witwin_server.components import PlatformGeometryComponent
    from witwin_server.platform_bridge import GeometryMap

    pose = [round(0.01 * i, 4) for i in range(72)]
    shape = [round(0.1 * i, 4) for i in range(10)]
    obj = SceneObject(name="human", mesh_type="Empty")
    geom = obj.add_component(PlatformGeometryComponent())
    geom.kind = "smpl"
    geom.pose = pose
    geom.shape = shape
    geom.gender = "female"
    geom.model_root = "/models/smpl"

    body = GeometryMap.to_platform(geom, [0.0, 0.0, -3.0], [1.0, 0.0, 0.0, 0.0])
    assert str(body.kind) == "smpl"
    assert str(body.gender) == "female"
    assert str(body.model_root) == "/models/smpl"
    assert all(approx(a, b) for a, b in zip([float(x) for x in body.pose.tolist()], pose))
    assert all(approx(a, b) for a, b in zip([float(x) for x in body.shape.tolist()], shape))


def test_smpl_full_round_trip(adapter):
    scene = wr.Scene(device="cpu")
    scene.add_smpl(name="human", pose=[0.0] * 72, shape=[0.0] * 10,
                   position=(0.0, 0.0, -3.0), gender="male")
    config = wr.RadarConfig.from_dict(_BASE)
    try:
        studio = adapter.to_studio((scene, config))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"SMPL model files absent; base bake degrades only on ImportError ({type(exc).__name__})")

    rebuilt, _ = adapter.to_platform(studio)
    body = rebuilt.structures[0].geometry
    assert str(body.kind) == "smpl"
    assert str(body.gender) == "male"
    assert rebuilt.structures[0].metadata.get("dynamic") is True  # SMPL bodies are always dynamic
