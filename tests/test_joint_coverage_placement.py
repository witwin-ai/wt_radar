"""whole_path_visible plans one Radar pose for the union of every dynamic target.

These regressions describe the R47 demo failure: the Radar was placed one metre
from the person while only one of the moving actors was registered, and
"whole path visible" was satisfied by a single skin site of one target.
"""
import numpy as np
import pytest

from test_snapshot import scene
from test_animation import add_track, make_rig
from wt_radar import sensor_placement
from wt_radar.adapter import animation
from witwin_server.api import SceneObject, SkinnedMeshComponent, ToolError


def add_walking_rig(scene, object_id, start, end, *, duration=2.):
    """A second independently skinned actor whose root walks from start to end."""
    actor = SceneObject(id=object_id, name=object_id.title(), mesh_type="Empty")
    scene.add_object(actor)
    skin = actor.add_component(SkinnedMeshComponent())
    vertices = np.array([
        [-.1, 0, -.15], [0, .1, -.15], [.1, 0, -.15],
        [-.1, 0, .15], [0, .1, .15], [.1, 0, .15],
    ], dtype=np.float32)
    skin.set_mesh_data(vertices, np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32))
    bones = [f"{object_id}-bone0", f"{object_id}-bone1"]
    for bone in bones:
        scene.add_object(SceneObject(id=bone, name=bone, mesh_type="Empty"))
        scene.set_parent(bone, object_id, keep_local_transform=True)
    indices = np.zeros((6, 4), dtype=np.int64)
    indices[3:, 0] = 1
    weights = np.zeros((6, 4), dtype=np.float32)
    weights[:, 0] = 1
    skin.set_skinning_data(
        bones, indices, weights, np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)), root_bone_id=bones[0],
    )
    add_track(scene, bones[0], "position", [(0., list(start)), (duration, list(end))])


def fully_visible_preflight(calls=None):
    def preflight(_detached, request, **_kwargs):
        if calls is not None:
            calls.append(request)
        sites = sum(int(target["site_budget"]) for target in request["targets"])
        return {
            "frame_count": 2, "site_count": sites, "radar_device": "cuda",
            "topology": {
                "declared_site_count": sites, "active_site_count": sites,
                "active_site_ranks": list(range(sites)), "visibility_coverage": 1.,
                "frames": [{"reachable_site_count": sites}],
            },
        }
    return preflight


def angles_from(position, aim, points):
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(aim, dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    rays = np.asarray(points, dtype=np.float64).reshape(-1, 3) - position
    cosine = rays @ forward / np.linalg.norm(rays, axis=1)
    return np.degrees(np.arccos(np.clip(cosine, -1., 1.)))


def human_and_cat(original, component):
    # Person: target at [0, 1, -3] walking 2 m along -Z. Cat: walks 1.5 m along +X
    # on the floor 2.5 m to the side, so a pose hugging the person misses the cat.
    make_rig(original, component)
    add_walking_rig(original, "cat", [2.5, -1.7, -2.5], [4., -1.7, -2.5])
    component.animation_duration_s = 2.
    return [{"object_id": "target", "site_budget": 2}, {"object_id": "cat", "site_budget": 2}]


def joint_sites(original, targets, times):
    from wt_radar.adapter.studio_motion import MultiTargetSkinSampler
    from wt_radar.adapter.target_contract import normalize_target_specs
    sampler = MultiTargetSkinSampler(original, normalize_target_specs(original, targets))
    return np.asarray([sampler.sample(float(t))[0] for t in times])


def test_whole_path_keeps_every_dynamic_target_inside_the_fov_with_margin(scene, monkeypatch):
    original, component = scene
    targets = human_and_cat(original, component)
    monkeypatch.setattr(animation, "animation_preflight", fully_visible_preflight())

    result = sensor_placement.choose_sensor_placement(original, {
        "targets": targets, "duration_s": 2., "fps": 10, "height_m": 1.,
        "placement_objective": "whole_path_visible",
    }, radar_object_id="radar")

    sites = joint_sites(original, targets, np.arange(20) / 10.)
    angles = angles_from(result["position_m"], result["aim_point_m"], sites)
    assert angles.max() <= 30. - 5. + 1e-6, "every site of every target stays 5 degrees inside the FOV"
    coverage = result["coverage"]
    assert [item["object_id"] for item in coverage["targets"]] == ["target", "cat"]
    for item in coverage["targets"]:
        assert item["root_frames_in_fov"] == item["frame_count"] == 20
        assert item["min_root_margin_deg"] >= 5.
        assert item["native_active_site_count"] > 0
        assert item["whole_path_visible"] is True
    assert coverage["all_targets_whole_path_visible"] is True
    assert result["predicted_metrics"]["all_targets_whole_path_in_fov"] is True


def test_whole_path_prefers_a_distant_pose_with_margin_over_the_nearest_ring(scene, monkeypatch):
    original, component = scene
    make_rig(original, component)
    component.animation_duration_s = 2.
    monkeypatch.setattr(animation, "animation_preflight", fully_visible_preflight())

    result = sensor_placement.choose_sensor_placement(original, {
        "target_object_id": "target", "duration_s": 2., "fps": 10, "height_m": 1.,
        "placement_objective": "whole_path_visible",
    }, radar_object_id="radar")

    assert result["coverage"]["min_site_range_m"] >= 1.5
    assert result["actual_distance_m"] >= 2.
    assert result["coverage"]["min_root_margin_deg"] >= 5.


def test_whole_path_keeps_clear_of_furniture_and_walls(scene, monkeypatch):
    original, component = scene
    targets = human_and_cat(original, component)
    wardrobe = SceneObject(id="wardrobe", name="Wardrobe", mesh_type="Cube")
    wardrobe.get_component("Transform").position = [1.5, 0., 0.5]
    wardrobe.get_component("Transform").scale = [1.2, 2., .6]
    original.add_object(wardrobe)
    monkeypatch.setattr(animation, "animation_preflight", fully_visible_preflight())

    result = sensor_placement.choose_sensor_placement(original, {
        "targets": targets, "duration_s": 2., "fps": 10, "height_m": 1.,
        "placement_objective": "whole_path_visible",
    }, radar_object_id="radar")

    clearance = result["coverage"]["clearance"]
    assert clearance["nearest_furniture_m"] >= .3
    assert clearance["room_edge_m"] >= .3
    x, _, z = result["position_m"]
    assert not (1.5 - .6 - .3 <= x <= 1.5 + .6 + .3 and .5 - .3 - .3 <= z <= .5 + .3 + .3)
    assert -5. + .3 <= x <= 5. - .3 and -5. + .3 <= z <= 5. - .3


def test_whole_path_fails_explicitly_and_suggests_multiple_radars(scene, monkeypatch):
    original, component = scene
    make_rig(original, component)
    # Two actors at opposite room corners cannot share one 20-degree FOV.
    add_walking_rig(original, "cat", [4.2, -1.7, 4.2], [4.4, -1.7, 4.4])
    add_track(original, "bone0", "position", [(0., [-1.2, 0., -1.2]), (2., [-1.4, 0., -1.4])])
    component.animation_duration_s = 2.
    monkeypatch.setattr(animation, "animation_preflight", fully_visible_preflight())

    with pytest.raises(ToolError) as caught:
        sensor_placement.choose_sensor_placement(original, {
            "targets": [{"object_id": "target", "site_budget": 2}, {"object_id": "cat", "site_budget": 2}],
            "duration_s": 2., "fps": 10, "height_m": 1., "fov_deg": 20.,
            "placement_objective": "whole_path_visible",
        }, radar_object_id="radar")

    assert caught.value.code == "sensor_placement_objective_unavailable"
    detail = caught.value.detail
    assert detail["suggestion"] == "multiple_radars"
    assert {item["object_id"] for item in detail["per_target_coverage"]} == {"target", "cat"}
    assert all(item["best_single_target_pose_m"] for item in detail["per_target_coverage"])


def test_whole_path_does_not_count_one_target_for_another_occluded_target(scene, monkeypatch):
    original, component = scene
    targets = human_and_cat(original, component)

    def person_only(_detached, request, **_kwargs):
        return {
            "frame_count": 2, "site_count": 2, "radar_device": "cuda",
            "targets": [
                {"object_id": "target", "site_start": 0, "site_end": 2, "active_site_count": 2},
                {"object_id": "cat", "site_start": 2, "site_end": 4, "active_site_count": 0},
            ],
            "topology": {
                "declared_site_count": 4, "active_site_count": 2, "active_site_ranks": [0, 1],
                "visibility_coverage": .5, "frames": [{"reachable_site_count": 2}],
            },
        }

    monkeypatch.setattr(animation, "animation_preflight", person_only)
    with pytest.raises(ToolError) as caught:
        sensor_placement.choose_sensor_placement(original, {
            "targets": targets, "duration_s": 2., "fps": 10, "height_m": 1.,
            "placement_objective": "whole_path_visible",
        }, radar_object_id="radar")
    assert caught.value.code == "sensor_placement_objective_unavailable"
    assert "cat" in caught.value.detail["targets_without_whole_path"]


def test_joint_candidates_cover_the_whole_room_floor_not_only_rings():
    # Actors near one wall: rings around their centre mostly leave the room, so
    # the search must also sample the open floor (R47 v7: 3 ranked candidates).
    floor = ("floor", "Floor", np.array([0., -.1, 0.]), np.array([5.7, 0., 4.9]))
    roots = np.array([[1.8, 1., 2.1], [3.1, .3, 4.3]])
    positions = list(sensor_placement.joint_candidate_positions(roots, 1., floor=floor))
    inside = [p for p in positions if .3 <= p[0] <= 5.4 and .3 <= p[2] <= 4.6]
    assert len(inside) >= 80
    # Every 0.5 m cell of the inset floor has a candidate, including the far west side.
    for x in np.arange(.3, 5.4, .5):
        for z in np.arange(.3, 4.6, .5):
            assert any(abs(p[0] - x) <= .26 and abs(p[2] - z) <= .26 for p in inside)
    assert len({(round(p[0], 2), round(p[2], 2)) for p in positions}) == len(positions)


def test_whole_path_native_preflights_are_bounded(scene, monkeypatch):
    original, component = scene
    targets = human_and_cat(original, component)
    calls = []

    def occluded_cat(_detached, request, **_kwargs):
        calls.append(1)
        return {
            "frame_count": 2, "site_count": 2, "radar_device": "cuda",
            "targets": [
                {"object_id": "target", "site_start": 0, "site_end": 2, "active_site_count": 2},
                {"object_id": "cat", "site_start": 2, "site_end": 4, "active_site_count": 0},
            ],
            "topology": {
                "declared_site_count": 4, "active_site_count": 2, "active_site_ranks": [0, 1],
                "visibility_coverage": .5, "frames": [{"reachable_site_count": 2}],
            },
        }

    monkeypatch.setattr(animation, "animation_preflight", occluded_cat)
    with pytest.raises(ToolError):
        sensor_placement.choose_sensor_placement(original, {
            "targets": targets, "duration_s": 2., "fps": 10, "height_m": 1.,
            "placement_objective": "whole_path_visible",
        }, radar_object_id="radar")
    assert 1 < len(calls) <= sensor_placement.MAX_WHOLE_PATH_NATIVE_PREFLIGHTS


def test_a_target_needs_seventy_percent_of_its_path_inside_the_fov():
    # Radar at the origin looking along +X with a 60 degree FOV (5 degree margin).
    position, aim = [0., 0., 0.], [1., 0., 0.]
    inside, outside = [3., 0., 0.], [0., 0., 3.]
    def frames(inside_count, total=10):
        return np.array([[inside if index < inside_count else outside] for index in range(total)])
    slices = [{"object_id": "cat", "site_start": 0, "site_end": 1}]
    seventy = sensor_placement._target_coverage(position, aim, frames(7), slices, fov_deg=60.)
    sixty = sensor_placement._target_coverage(position, aim, frames(6), slices, fov_deg=60.)
    assert seventy["targets"][0]["path_fov_fraction"] == pytest.approx(.7)
    assert seventy["targets"][0]["fov_whole_path"] is True
    assert sixty["targets"][0]["fov_whole_path"] is False
    assert sensor_placement.MIN_PATH_FOV_FRACTION == .7
