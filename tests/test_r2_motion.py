"""R2 round-trip: dynamics (the radar motion graph).

A parented, mixed translation/rotation graph round-trips so ``scene._structure_motions``
reconstructs identically; self-parent and cyclic graphs are rejected on export; an editor
rename (object name + child parent refs) keeps the motion graph consistent.

Human/SMPL bodies are owned by the wt-human plugin, not radar. A radar scene that contains
one round-trips through the shared base geometry map with no radar-side SMPL code, so SMPL
is not exercised here.
"""
import pytest
import witwin.radar as wr

from _helpers import assert_motions_equal, assert_structures_equal

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
