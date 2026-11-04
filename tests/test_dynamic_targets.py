"""Radar targets cover every moving actor of the scene, never just the first one."""
import pytest

from witwin_server.api import SceneObject, SkinnedMeshComponent, ToolError

from test_agent_tools import harness, run


def add_moving_person(scene):
    person = SceneObject(id="person", name="Person", mesh_type="Empty")
    person.add_component(SkinnedMeshComponent())
    scene.add_object(person)
    bone = SceneObject(id="person-smpl-bone-00", name="pelvis", mesh_type="Empty")
    scene.add_object(bone)
    scene.set_parent(bone.id, "person", keep_local_transform=True)
    track = scene.timeline_manager.clip.get_or_create_track(bone.id, "Transform", "position")
    track.add_keyframe(0., [0., 0., 0.])
    track.add_keyframe(8., [0., 0., -1.])


def animate_cat(scene):
    track = scene.timeline_manager.clip.get_or_create_track("catstray", "Transform", "position")
    track.add_keyframe(0., [2., 0., 0.])
    track.add_keyframe(8., [3., 0., 0.])


def test_plan_sensor_placement_declares_every_moving_actor(harness):
    scene, tools = harness
    add_moving_person(scene)
    animate_cat(scene)

    result = run(tools, "plan_sensor_placement", {
        "scene_id": scene.scene_id,
        "placement_mode": "automatic",
        "placement_objective": "whole_path_visible",
    })

    assert result["status"] == "ready_for_confirmation"
    assert result["dynamic_object_ids"] == ["catstray", "person"]
    assert [target["object_id"] for target in result["proposed_ensure_sensor_arguments"]["targets"]] == [
        "catstray", "person",
    ]


def test_ensure_sensor_rejects_a_target_list_that_omits_a_moving_actor(harness):
    scene, tools = harness
    add_moving_person(scene)
    animate_cat(scene)

    with pytest.raises(ToolError) as caught:
        run(tools, "ensure_sensor", {
            "scene_id": scene.scene_id,
            "operation_id": "person-only-radar",
            "targets": [{"object_id": "person", "site_budget": 720}],
            "placement_mode": "automatic",
            "placement_objective": "whole_path_visible",
        })

    assert caught.value.code == "undeclared_dynamic_target"
    assert caught.value.detail["missing_object_ids"] == ["catstray"]
    assert [target["object_id"] for target in caught.value.detail["suggested_targets"]] == [
        "person", "catstray",
    ]
    assert not any(obj.get_component("Radar") for obj in scene.objects.values())


def test_static_skinned_objects_do_not_have_to_be_declared(harness):
    scene, tools = harness
    add_moving_person(scene)

    result = run(tools, "plan_sensor_placement", {
        "scene_id": scene.scene_id,
        "placement_mode": "automatic",
        "placement_objective": "whole_path_visible",
    })

    assert result["dynamic_object_ids"] == ["person"]
    assert [target["object_id"] for target in result["targets"]] == ["person"]
