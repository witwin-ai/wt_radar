"""Agent Radar façade tests; native RF/propagation/DSP is intentionally untouched."""
import asyncio
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from witwin_server import SceneObject
from witwin_server.core.components import MeshComponent, SkinnedMeshComponent
from witwin_server.core.scene import Scene
from witwin_server.core import wtscene
from witwin_server.features.timeline import TimelineManager
from witwin_server.tools.base import ToolError

from wt_radar import agent_tools as radar_tools
from wt_radar.adapter.solve import SensorSpec


class ToolCollector:
    def __init__(self):
        self.values = {}

    def register(self, value):
        self.values[value.name] = value


@pytest.fixture
def harness():
    scene = Scene(scene_id="bedroom_004")
    scene.timeline_manager = TimelineManager(scene)
    scene.timeline_manager.clip.duration = 10.0
    target = SceneObject(id="catstray", name="CatStray", mesh_type="Empty")
    target.add_component(SkinnedMeshComponent())
    scene.add_object(target)
    collector = ToolCollector()
    server = SimpleNamespace(get_scene=lambda scene_id: scene if scene_id == scene.scene_id else None)
    context = SimpleNamespace(
        scene=scene,
        api=SimpleNamespace(server=server),
        tools=collector,
    )
    scene._radar_test_context = context
    radar_tools.register(context)
    return scene, collector.values


def run(tools, name, args):
    return tools[name].run(args)


@pytest.mark.parametrize("apply_to_scene", [False, True])
def test_measurement_fingerprint_tracks_authored_inputs_not_playback(harness, apply_to_scene):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id, "operation_id": "fingerprint-playback",
        "target_object_id": "catstray", "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
    })
    sensor = scene.get_object(ensured["radar"]["object_id"])
    position = scene.timeline_manager.clip.get_or_create_track("catstray", "Transform", "position")
    position.add_keyframe(0., [0., 0., 0.])
    position.add_keyframe(1., [1., 0., 0.])
    rotation = scene.timeline_manager.clip.get_or_create_track("catstray", "Transform", "rotation")
    rotation.add_keyframe(0., [0., 0., 0.])
    rotation.add_keyframe(1., [0., 90., 0.])
    expected = radar_tools._measurement_input_fingerprint(scene, sensor)
    inspected = run(tools, "inspect_pipeline", {
        "scene_id": scene.scene_id,
        "radar_object_id": sensor.id,
    })
    assert inspected["selected_radar"]["measurement_input_fingerprint"] == expected
    scene.timeline_manager.set_time(.5, apply_to_scene=apply_to_scene)
    scene.timeline_manager.is_looping = True
    assert radar_tools._measurement_input_fingerprint(scene, sensor) == expected
    position.keyframes[1].value = [2., 0., 0.]
    assert radar_tools._measurement_input_fingerprint(scene, sensor) != expected
    position.keyframes[1].value = [1., 0., 0.]
    assert radar_tools._measurement_input_fingerprint(scene, sensor) == expected
    scene.get_object("catstray").get_component("Transform").scale = [0.5, 0.5, 0.5]
    assert radar_tools._measurement_input_fingerprint(scene, sensor) != expected


@pytest.mark.parametrize("change, expected_status", [
    (None, "available"), ("missing", "missing_payload"),
    ("hash", "payload_mismatch"), ("motion", "stale_motion"),
    ("unverified", "unverified_motion"), ("unsafe_id", "invalid_manifest"),
])
def test_inspect_distinguishes_persisted_replay_from_lost_native_handle(harness, tmp_path, change, expected_status):
    from witwin_server.core.components.core.plot import PlotData
    from witwin_server.features.assets.binary.service import BinaryAssetService
    from wt_radar.adapter.replay import motion_fingerprint

    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id, "operation_id": "replay-inspect-ensure",
        "target_object_id": "catstray", "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
    })
    sensor = scene.get_object(ensured["radar"]["object_id"])
    persistence = SimpleNamespace(runtime_volumes_dir=tmp_path, volume_candidate_dirs=lambda: [tmp_path])
    service = BinaryAssetService(persistence=persistence)
    data = np.arange(8, dtype="<f4").tobytes()
    asset = service.register_bytes(kind="radar-replay", format="witwin.radar.replay.float32.v1",
                                  data=data, scene_id=scene.scene_id, persistent=True).to_dict()
    # A fresh service has no native worker or memory cache, only the saved file.
    scene._radar_test_context.api.server.handlers = {
        "binary_assets": SimpleNamespace(service=BinaryAssetService(persistence=persistence)),
    }
    record = {"schema": 1, "sceneId": scene.scene_id, "timesS": [0., .1], "endTimeS": .2,
              "rangeM": [0., 1.], "velocityMps": [0.], "frameStride": 4,
              "asset": asset, "sceneFingerprint": motion_fingerprint(scene)}
    if change == "missing":
        asset["assetId"] = "missing-replay"
        asset["uri"] = f"/scenes/{scene.scene_id}/assets/missing-replay"
    elif change == "hash":
        asset["contentHash"] = "sha256:wrong"
    elif change == "motion":
        scene.timeline_manager.clip.duration = 11.0
    elif change == "unverified":
        record.pop("sceneFingerprint")
    elif change == "unsafe_id":
        asset["assetId"] = "../outside-project"
    sensor.get_component("Radar").signal_figure.set_plot_data(PlotData("line", {"recording": record}))
    sensor.get_component("Radar")._figure_instances.clear()
    assert "recording" not in sensor.get_component("Radar").signal_figure.to_dict()["data"]
    before = copy.deepcopy(scene.to_dict())
    inspected = run(tools, "inspect_pipeline", {"scene_id": scene.scene_id, "radar_object_id": sensor.id})
    selected = inspected["selected_radar"]
    assert selected["result"]["present"] is False
    assert selected["result"]["result_handle"] is None
    assert selected["recorded_replay"]["status"] == expected_status
    assert selected["recorded_replay"]["ready_for_timeline"] is (change is None)
    if change is None:
        assert selected["recorded_replay"]["storage"] == "persistent_file"
        assert selected["recorded_replay"]["payload_integrity_verified"] is True
    assert scene.to_dict() == before


RCS_POLICY = "total_rcs_divided_by_declared_sites_no_visible_renormalization"


def native_preflight(*, frame_count=50, site_count=4, moving_object_ids=None):
    ids = [3_000_000 + index for index in range(site_count)]
    return {
        "frame_count": frame_count,
        "site_count": site_count,
        "declared_site_count": site_count,
        "moving_object_ids": moving_object_ids or ["catstray"],
        "radar_device": "cuda",
        "radar_position_m": [1.0, 1.0, 2.0],
        "radar_target_m": [0.0, 0.4, 0.0],
        "model": "studio_visible_skinned_surface_sites_v2",
        "topology": {
            "declared_site_count": site_count,
            "active_site_count": site_count,
            "occluded_site_count": 0,
            "active_site_ids": ids,
            "rcs_policy": RCS_POLICY,
            "visibility_coverage": 1.0,
            "visibility_quality": "complete",
        },
    }


def topology_manifest(*, site_count=4):
    return {
        "declared_site_count": site_count,
        "active_site_count": site_count,
        "occluded_site_count": 0,
        "active_site_ids": [3_000_000 + index for index in range(site_count)],
        "rcs_policy": RCS_POLICY,
        "rcs_per_site_m2": float(np.float32(0.1)) / site_count,
        "visibility_coverage": 1.0,
        "visibility_quality": "complete",
        "no_zero_fill_or_dropped_active_sites": True,
        "environment_reflection_model": "native_channel_direct_single_bounce",
        "environment_reflection_max_depth": 1,
        "environment_reflected_path_count": 3,
        "environment_material_slot_count": 7,
        "environment_coherent_with_target": True,
        "solver_completion_contract": "atomic_active_sites_no_zero_fill_v2",
    }


def remember_preflight(scene, sensor, fingerprint, *, site_count=4):
    radar_tools._remember_preflight(
        scene._radar_test_context,
        scene_id=str(scene.scene_id),
        radar_object_id=str(sensor.id),
        fingerprint=fingerprint,
        native_preflight=native_preflight(site_count=site_count),
    )


def test_registers_narrow_domain_tools(harness):
    _scene, tools = harness
    assert set(tools) == {
        "runtime_diagnostics",
        "inspect_pipeline",
        "plan_sensor_placement",
        "ensure_sensor",
        "plan_animation_measurement",
        "submit_animation_measurement",
        "get_simulation",
        "cancel_simulation",
        "verify_result",
        "prepare_replay",
        "export_result",
        "describe_pipeline_contract",
    }
    assert tools["inspect_pipeline"].permission_tier == "read"
    assert tools["plan_sensor_placement"].permission_tier == "read"
    assert tools["plan_sensor_placement"].side_effects is False
    assert tools["plan_sensor_placement"].requires_confirmation is False
    assert tools["ensure_sensor"].permission_tier == "scene_write"
    assert not tools["ensure_sensor"].requires_confirmation
    assert tools["ensure_sensor"].idempotent
    assert not tools["ensure_sensor"].durable_confirmation
    assert tools["cancel_simulation"].permission_tier == "soft_write"
    assert not tools["cancel_simulation"].requires_confirmation
    for name, tier in (
        ("submit_animation_measurement", "scene_write"),
        ("prepare_replay", "soft_write"),
        ("export_result", "file_write"),
    ):
        assert tools[name].side_effects is True
        assert tools[name].requires_confirmation is (name in {"prepare_replay", "export_result"})
        assert tools[name].permission_tier == tier
        assert tools[name].idempotent is True
        assert tools[name].durable_confirmation is (name in {"prepare_replay", "export_result"})
    assert "hint-explicit-intent:只运行雷达" in tools[
        "submit_animation_measurement"
    ].tags
    assert "hint-explicit-intent:只回放" in tools["prepare_replay"].tags
    assert "hint-explicit-intent:只导出" in tools["export_result"].tags
    assert "word-intent:previous recording" in tools["prepare_replay"].tags
    assert "hint-explicit-intent:watch" in tools["prepare_replay"].tags
    assert "hint-explicit-intent:npz" in tools["export_result"].tags
    assert "hint-explicit-only" in tools["submit_animation_measurement"].tags
    assert "hint-explicit-only" not in tools["plan_animation_measurement"].tags
    for term in (
        "rerun radar", "measure again", "current radar position", "moved radar",
        "重新测量", "当前位置", "移动雷达",
    ):
        assert f"hint-term:{term}" in tools["plan_animation_measurement"].tags


@pytest.mark.parametrize("term", ["saved recording", "replay", "npz", "export", "download"])
def test_saved_result_discovery_hints_belong_only_to_read_only_inspection(harness, term):
    _scene, tools = harness
    inspect = tools["inspect_pipeline"]
    assert f"hint-term:{term}" in inspect.tags
    assert inspect.permission_tier == "read" and not inspect.side_effects
    assert not inspect.requires_confirmation
    assert inspect.input_schema["required"] == ["scene_id"]
    assert "operation_id" in inspect.description and "recorded_replay" in inspect.description
    for name in ("runtime_diagnostics", "ensure_sensor", "submit_animation_measurement", "prepare_replay", "export_result"):
        assert f"hint-term:{term}" not in tools[name].tags


def test_pipeline_capability_is_derived_from_registered_action_contracts(harness):
    _scene, tools = harness
    capability = run(tools, "describe_pipeline_contract", {})

    assert capability["schema"] == "witwin.radar.pipeline-capabilities.v1"
    assert capability["actions"]["replay"]["input_schema"] == (
        tools["prepare_replay"].input_schema
    )
    assert capability["actions"]["replay"]["requires_confirmation"] is True
    assert capability["actions"]["preflight"]["requires_confirmation"] is False
    assert capability["dependencies"]["export"] == ["verify"]

    capability["actions"]["replay"]["input_schema"]["properties"].clear()
    assert run(tools, "describe_pipeline_contract", {})["actions"]["replay"][
        "input_schema"
    ]["properties"]


def test_operation_receipt_corruption_fails_closed_without_reset(harness):
    scene, tools = harness
    run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "receipt-corrupt",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
    })
    plugin_ctx = scene._radar_test_context
    original = copy.deepcopy(plugin_ctx._radar_operation_fallback)
    plugin_ctx._radar_operation_fallback["receipt-corrupt"]["operation_id"] = "other"
    with pytest.raises(ToolError) as raised:
        run(tools, "ensure_sensor", {
            "scene_id": scene.scene_id,
            "operation_id": "receipt-corrupt",
            "target_object_id": "catstray",
            "position_m": [1.0, 1.0, 2.0],
            "aim_point_m": [0.0, 0.4, 0.0],
        })
    assert raised.value.code == "radar_operation_state_corrupt"
    assert plugin_ctx._radar_operation_fallback["receipt-corrupt"]["operation_id"] == "other"
    assert original["receipt-corrupt"]["operation_id"] == "receipt-corrupt"


def test_inspect_reports_missing_radar_without_mutating_scene(harness):
    scene, tools = harness
    before = set(scene.objects)
    result = run(tools, "inspect_pipeline", {"scene_id": scene.scene_id})
    assert result["status"] == "blocked"
    assert result["blockers"][0]["code"] == "missing_radar"
    assert set(scene.objects) == before


def test_plan_sensor_placement_is_read_only_and_resolves_single_skinned_target(harness):
    scene, tools = harness
    before = copy.deepcopy(scene.to_dict())
    result = run(tools, "plan_sensor_placement", {
        "scene_id": scene.scene_id,
        "position_m": [3, 1, 2],
        "aim_point_m": [0, 0.4, 0],
    })

    assert result["status"] == "ready_for_confirmation"
    assert result["target_object_id"] == "catstray"
    assert result["placement"]["position_m"] == [3.0, 1.0, 2.0]
    assert result["proposed_ensure_sensor_arguments"]["aim_point_m"] == [0.0, 0.4, 0.0]


def test_plan_sensor_placement_preserves_natural_measurement_objective(harness):
    scene, tools = harness
    before = scene.to_dict()
    result = run(tools, "plan_sensor_placement", {
        "scene_id": scene.scene_id,
        "target_object_id": "catstray",
        "placement_mode": "automatic",
        "placement_objective": "bidirectional_radial_velocity",
    })
    assert result["proposed_ensure_sensor_arguments"]["placement_objective"] == "bidirectional_radial_velocity"
    assert result["placement"]["mode"] == "automatic"
    assert result["proposed_ensure_sensor_arguments"]["start_s"] == 0.0
    assert result["proposed_ensure_sensor_arguments"]["duration_s"] == 10.0
    assert result["proposed_ensure_sensor_arguments"]["fps"] == 30.0
    assert scene.to_dict() == before
    assert result["next_step"]["requires_confirmation"] is False
    assert result["mutates_scene"] is False


def test_plan_sensor_placement_covers_a_long_frame_aligned_timeline(harness):
    scene, tools = harness
    scene.timeline_manager.clip.duration = 217 / 30
    before = scene.to_dict()

    result = run(tools, "plan_sensor_placement", {
        "scene_id": scene.scene_id,
        "placement_mode": "automatic",
        "placement_objective": "whole_path_visible",
    })

    proposed = result["proposed_ensure_sensor_arguments"]
    assert proposed["start_s"] == 0.0
    assert proposed["duration_s"] == pytest.approx(217 / 30)
    assert proposed["fps"] == 30.0
    assert proposed["duration_s"] * proposed["fps"] == pytest.approx(217)
    assert proposed["start_s"] + proposed["duration_s"] == pytest.approx(
        scene.timeline_manager.clip.duration
    )
    assert result["placement"]["timeline_interval"]["start_s"] == 0.0
    assert result["placement"]["timeline_interval"]["duration_s"] == pytest.approx(217 / 30)
    assert result["placement"]["timeline_interval"]["fps"] == 30.0
    assert scene.to_dict() == before


def test_plan_sensor_placement_blocks_instead_of_truncating_unaligned_timeline(harness):
    scene, tools = harness
    scene.timeline_manager.clip.duration = 3.47
    before = scene.to_dict()

    result = run(tools, "plan_sensor_placement", {
        "scene_id": scene.scene_id,
        "placement_mode": "automatic",
        "placement_objective": "whole_path_visible",
    })

    assert result["status"] == "blocked"
    assert result["blockers"][0]["code"] == "timeline_interval_unavailable"
    assert scene.to_dict() == before


def test_plan_sensor_placement_blocks_without_usable_motion_interval(harness):
    scene, tools = harness
    scene.timeline_manager.clip.duration = 0.01
    before = scene.to_dict()

    result = run(tools, "plan_sensor_placement", {
        "scene_id": scene.scene_id,
        "placement_mode": "automatic",
    })

    assert result["status"] == "blocked"
    assert result["blockers"][0]["code"] == "timeline_interval_unavailable"
    assert scene.to_dict() == before


def test_plan_sensor_placement_blocks_ambiguous_target_without_writing(harness):
    scene, tools = harness
    second = SceneObject(id="other-cat", name="Other Cat", mesh_type="Empty")
    second.add_component(SkinnedMeshComponent())
    scene.add_object(second)
    before = copy.deepcopy(scene.to_dict())

    result = run(tools, "plan_sensor_placement", {
        "scene_id": scene.scene_id,
        "position_m": [3, 1, 2],
        "aim_point_m": [0, 0.4, 0],
    })

    assert result["status"] == "blocked"
    assert result["blockers"][0]["code"] == "ambiguous_target"
    assert result["blockers"][0]["target_object_ids"] == ["catstray", "other-cat"]
    assert scene.to_dict() == before


def test_ensure_sensor_creates_once_and_aims_local_minus_z(harness):
    scene, tools = harness
    args = {
        "scene_id": scene.scene_id,
        "operation_id": "pipeline-1:radar",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
        "duration_s": 5.0,
        "fps": 10.0,
    }
    first = run(tools, "ensure_sensor", args)
    second = run(tools, "ensure_sensor", args)
    assert first["created"] is True
    assert second["created"] is False
    assert first["radar"]["object_id"] == second["radar"]["object_id"]
    sensors = [obj for obj in scene.objects.values() if obj.get_component("Radar")]
    assert len(sensors) == 1
    radar = sensors[0].get_component("Radar")
    assert radar.snapshot_target_id == "catstray"
    assert radar.animation_duration_s == pytest.approx(5.0)
    assert radar.animation_fps == pytest.approx(10.0)
    pose = SensorSpec.from_component(radar)
    actual = np.asarray(pose.target) - np.asarray(pose.position)
    expected = np.asarray(args["aim_point_m"]) - np.asarray(args["position_m"])
    np.testing.assert_allclose(actual / np.linalg.norm(actual), expected / np.linalg.norm(expected), atol=1e-6)


def test_ensure_sensor_never_guesses_when_multiple_exist(harness):
    scene, tools = harness
    for name in ("Radar A", "Radar B"):
        scene.add_object(radar_tools._settings_object(name))
    with pytest.raises(ToolError) as error:
        run(tools, "ensure_sensor", {
            "scene_id": scene.scene_id,
            "operation_id": "ambiguous",
            "target_object_id": "catstray",
            "position_m": [1.0, 1.0, 2.0],
            "aim_point_m": [0.0, 0.4, 0.0],
        })
    assert error.value.code == "ambiguous_radar"


def test_ensure_sensor_rejects_missing_target_before_creation(harness):
    scene, tools = harness
    with pytest.raises(ToolError) as error:
        run(tools, "ensure_sensor", {
            "scene_id": scene.scene_id,
            "operation_id": "missing-target",
            "target_object_id": "missing-cat",
            "position_m": [1.0, 1.0, 2.0],
            "aim_point_m": [0.0, 0.4, 0.0],
        })
    assert error.value.code == "missing_target"
    assert not any(obj.get_component("Radar") for obj in scene.objects.values())


def test_ensure_sensor_operation_id_rejects_different_payload(harness):
    scene, tools = harness
    base = {
        "scene_id": scene.scene_id,
        "operation_id": "stable-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
    }
    run(tools, "ensure_sensor", base)
    with pytest.raises(ToolError) as error:
        run(tools, "ensure_sensor", {**base, "aim_point_m": [2.0, 0.4, 0.0]})
    assert error.value.code == "operation_conflict"
    assert len([obj for obj in scene.objects.values() if obj.get_component("Radar")]) == 1


def test_ensure_sensor_retry_rejects_drifted_sensor_state(harness):
    scene, tools = harness
    args = {
        "scene_id": scene.scene_id,
        "operation_id": "drifted-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
    }
    result = run(tools, "ensure_sensor", args)
    sensor = scene.get_object(result["radar"]["object_id"])
    scene.update_transform(sensor.id, position=[3.0, 1.0, 2.0])
    with pytest.raises(ToolError) as error:
        run(tools, "ensure_sensor", args)
    assert error.value.code == "operation_state_stale"


def test_review_controls_do_not_change_measurement_fingerprint(harness):
    scene, tools = harness
    result = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "review-fingerprint",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
    })
    sensor = scene.get_object(result["radar"]["object_id"])
    radar = sensor.get_component("Radar")
    before = radar_tools._measurement_input_fingerprint(scene, sensor)
    radar.view = "range_profile"
    radar.tx_index = 1
    radar.rx_index = 2
    radar.static_clutter_removal = True
    radar.show_cfar = True
    assert radar_tools._measurement_input_fingerprint(scene, sensor) == before


def test_measurement_fingerprint_is_stable_across_studio_float32_persistence(harness):
    scene, tools = harness
    result = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "float32-persistence",
        "target_object_id": "catstray",
        "position_m": [1.123456789, 1.0, 2.987654321],
        "aim_point_m": [0.0, 0.4, 0.0],
    })
    sensor = scene.get_object(result["radar"]["object_id"])
    radar = sensor.get_component("Radar")
    before = radar_tools._measurement_input_fingerprint(scene, sensor)

    # Match the exact numeric coercion performed by .wtscene persistence.
    for field in ("fc", "slope", "snapshot_rcs_m2", "animation_duration_s", "animation_fps"):
        setattr(radar, field, float(np.float32(getattr(radar, field))))
    transform = sensor.get_component("Transform")
    transform.position = [float(np.float32(value)) for value in transform.position]
    transform.rotation = [float(np.float32(value)) for value in transform.rotation]
    assert radar_tools._measurement_input_fingerprint(scene, sensor) == before

    restored = Scene.from_dict(wtscene.loads(wtscene.dumps(scene.to_dict())))
    restored_sensor = restored.get_object(sensor.id)
    assert radar_tools._measurement_input_fingerprint(restored, restored_sensor) == before

    # One durable authored float32 step is still a real input change.
    radar.fov = float(np.nextafter(np.float32(radar.fov), np.float32(180.0)))
    assert radar_tools._measurement_input_fingerprint(scene, sensor) != before


def test_measurement_fingerprint_distinguishes_adjacent_timeline_double(harness):
    scene, tools = harness
    result = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "timeline-double-identity",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
    })
    sensor = scene.get_object(result["radar"]["object_id"])
    track = scene.timeline_manager.clip.get_or_create_track(
        "catstray", "Transform", "position",
    )
    track.add_keyframe(0.0, [0.0, 0.0, 0.0])
    track.add_keyframe(0.5, [1.0, 0.0, 0.0])
    before = radar_tools._measurement_input_fingerprint(scene, sensor)

    track.keyframes[1].time = float(np.nextafter(np.float64(0.5), np.float64(1.0)))

    assert radar_tools._measurement_input_fingerprint(scene, sensor) != before


def test_measurement_fingerprint_distinguishes_adjacent_sensor_pose_ulp(harness):
    scene, tools = harness
    result = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "sensor-pose-ulp-identity",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
    })
    sensor = scene.get_object(result["radar"]["object_id"])
    transform = sensor.get_component("Transform")
    before = radar_tools._measurement_input_fingerprint(scene, sensor)
    rotation = transform.rotation.detach().cpu().numpy().astype(np.float32)
    rotation[0] = np.nextafter(rotation[0], np.float32(np.inf))
    transform.rotation = rotation.tolist()

    assert radar_tools._measurement_input_fingerprint(scene, sensor) != before


def test_scene_fingerprint_tracks_static_mesh_vertex_content(harness):
    scene, _tools = harness
    room = SceneObject(id="room-mesh", name="Room Mesh", mesh_type="Empty")
    mesh = MeshComponent()
    room.add_component(mesh)
    scene.add_object(room)
    faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    mesh.set_mesh_data(vertices, faces)
    before = radar_tools._scene_input_fingerprint(scene)
    moved = vertices.copy()
    moved[1, 0] += 0.123
    mesh.set_mesh_data(moved, faces)
    assert radar_tools._scene_input_fingerprint(scene) != before


def test_scene_fingerprint_tracks_skin_vertex_and_weight_content(harness):
    scene, _tools = harness
    skin = scene.get_object("catstray").get_component("SkinnedMesh")
    faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.asarray([[0, 1, 0, 1]] * 3, dtype=np.int64)
    weights = np.asarray([[0.8, 0.2, 0.0, 0.0]] * 3, dtype=np.float32)
    inverse_bind = np.stack([np.eye(4), np.eye(4)]).astype(np.float32)
    skin.set_mesh_data(vertices, faces)
    skin.set_skinning_data(["bone-a", "bone-b"], indices, weights, inverse_bind)
    initial = radar_tools._scene_input_fingerprint(scene)
    moved = vertices.copy()
    moved[0, 2] += 0.123
    skin.set_mesh_data(moved, faces)
    after_vertex = radar_tools._scene_input_fingerprint(scene)
    assert after_vertex != initial
    changed_weights = np.asarray([[0.6, 0.4, 0.0, 0.0]] * 3, dtype=np.float32)
    skin.set_skinning_data(["bone-a", "bone-b"], indices, changed_weights, inverse_bind)
    assert radar_tools._scene_input_fingerprint(scene) != after_vertex


def test_ensure_sensor_rejects_parented_world_pose(harness):
    scene, tools = harness
    parent = SceneObject(name="Table", mesh_type="Empty")
    scene.add_object(parent)
    sensor = radar_tools._settings_object("Parented Radar")
    scene.add_object(sensor)
    sensor.get_component("Transform").set_parent(parent.get_component("Transform"))
    with pytest.raises(ToolError) as error:
        run(tools, "ensure_sensor", {
            "scene_id": scene.scene_id,
            "operation_id": "parented",
            "radar_object_id": sensor.id,
            "target_object_id": "catstray",
            "position_m": [1.0, 1.0, 2.0],
            "aim_point_m": [0.0, 0.4, 0.0],
        })
    assert error.value.code == "parented_radar_unsupported"


def test_plan_preflight_is_read_only_and_reports_exact_cube(harness, monkeypatch):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "preflight-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
        "duration_s": 5.0,
        "fps": 10.0,
    })
    sensor_id = ensured["radar"]["object_id"]
    monkeypatch.setattr(
        "wt_radar.adapter.animation.animation_preflight",
        lambda *_: native_preflight(moving_object_ids=["catstray", "spine"]),
    )
    monkeypatch.setattr(radar_tools, "_runtime_evidence", lambda: {
        "torch": {"cuda_available": True, "device_name": "test-gpu"}
    })
    before = scene.to_dict()
    result = run(tools, "plan_animation_measurement", {
        "scene_id": scene.scene_id,
        "radar_object_id": sensor_id,
        "start_s": 0.0,
        "duration_s": 5.0,
        "fps": 10.0,
    })
    assert result["status"] == "ready"
    assert result["measurement"]["frame_count"] == 50
    assert result["measurement"]["cube_shape"] == [50, 3, 4, 128, 256]
    assert result["target"]["declared_site_count"] == 4
    assert result["target"]["active_site_count"] == 4
    assert result["target"]["occluded_site_count"] == 0
    assert result["target"]["visibility_coverage"] == 1.0
    assert result["target"]["visibility_quality"] == "complete"
    assert result["algorithm_unchanged"] is True
    assert result["completion_requirements"] == {
        "actual_frame_count": 50,
        "all_frames_finite": True,
        "device": "cuda",
        "result_handle_required": True,
        "no_zero_fill_for_active_sites": True,
        "stable_active_site_ids_for_full_interval": True,
        "occluded_declared_sites_reported": True,
        "no_visible_rcs_renormalization": True,
    }
    assert scene.to_dict() == before


def test_plan_reuses_automatic_placement_native_preflight(harness, monkeypatch):
    scene, tools = harness
    evidence = native_preflight(moving_object_ids=["catstray", "spine"])
    monkeypatch.setattr(
        "wt_radar.sensor_placement.choose_sensor_placement",
        lambda *_args, **_kwargs: {
            "mode": "automatic",
            "position_m": [1.0, 1.0, 2.0],
            "aim_point_m": [0.0, 0.4, 0.0],
            "native_preflight": copy.deepcopy(evidence),
        },
    )
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "cached-placement-ensure",
        "target_object_id": "catstray",
        "placement_mode": "automatic",
        "duration_s": 5.0,
        "fps": 10.0,
    })
    monkeypatch.setattr(
        "wt_radar.adapter.animation.animation_preflight",
        lambda *_: pytest.fail("planning repeated automatic placement preflight"),
    )
    monkeypatch.setattr(radar_tools, "_runtime_evidence", lambda: {
        "torch": {"cuda_available": True, "device_name": "test-gpu"}
    })

    planned = run(tools, "plan_animation_measurement", {
        "scene_id": scene.scene_id,
        "radar_object_id": ensured["radar"]["object_id"],
    })

    assert planned["status"] == "ready"
    assert planned["native_preflight"] == evidence


def test_manual_radar_move_preflight_uses_current_pose_without_reconfiguration(harness, monkeypatch):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "manual-move-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
        "duration_s": 5.0,
        "fps": 10.0,
    })
    sensor_id = ensured["radar"]["object_id"]
    scene.update_transform(sensor_id, position=[-2.0, 1.25, 1.5], rotation=[0.1, 0.2, 0.3])
    before = copy.deepcopy(scene.to_dict())
    observed = {}

    def preflight(detached, request):
        observed["position"] = list(
            detached.get_object(sensor_id).get_component("Transform").position
        )
        return native_preflight(moving_object_ids=["catstray", "spine"])

    monkeypatch.setattr("wt_radar.adapter.animation.animation_preflight", preflight)
    monkeypatch.setattr(radar_tools, "_runtime_evidence", lambda: {
        "torch": {"cuda_available": True, "device_name": "test-gpu"}
    })
    result = run(tools, "plan_animation_measurement", {
        "scene_id": scene.scene_id,
        "radar_object_id": sensor_id,
    })

    np.testing.assert_allclose(observed["position"], [-2.0, 1.25, 1.5])
    assert result["input_fingerprint"] == radar_tools._measurement_input_fingerprint(
        scene, scene.get_object(sensor_id),
    )
    assert scene.to_dict() == before


def test_plan_preserves_structured_native_topology_failure(harness, monkeypatch):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "topology-failure-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
        "duration_s": 5.0,
        "fps": 10.0,
    })
    from wt_radar.adapter.animation import AnimationTopologyError

    detail = {
        "code": "sensor_embedded_or_all_sites_occluded",
        "declared_site_count": 44,
        "frame_count": 50,
    }

    def fail_preflight(*_args, **_kwargs):
        raise AnimationTopologyError("native topology blocked", detail=detail)

    monkeypatch.setattr("wt_radar.adapter.animation.animation_preflight", fail_preflight)
    with pytest.raises(ToolError) as error:
        run(tools, "plan_animation_measurement", {
            "scene_id": scene.scene_id,
            "radar_object_id": ensured["radar"]["object_id"],
            "start_s": 0.0,
            "duration_s": 5.0,
            "fps": 10.0,
        })
    assert error.value.code == "radar_topology_unreachable"
    assert error.value.detail == detail


def test_plan_explicitly_marks_low_visibility_as_degraded(harness, monkeypatch):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "low-coverage-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
        "duration_s": 5.0,
        "fps": 10.0,
    })
    partial = native_preflight(site_count=1)
    partial["declared_site_count"] = 44
    partial["topology"].update({
        "declared_site_count": 44,
        "active_site_count": 1,
        "occluded_site_count": 43,
        "visibility_coverage": 1 / 44,
        "visibility_quality": "degraded_interval_global_subset",
    })
    monkeypatch.setattr(
        "wt_radar.adapter.animation.animation_preflight", lambda *_: copy.deepcopy(partial),
    )
    monkeypatch.setattr(radar_tools, "_runtime_evidence", lambda: {
        "torch": {"cuda_available": True, "device_name": "test-gpu"}
    })
    result = run(tools, "plan_animation_measurement", {
        "scene_id": scene.scene_id,
        "radar_object_id": ensured["radar"]["object_id"],
        "start_s": 0.0,
        "duration_s": 5.0,
        "fps": 10.0,
    })
    assert result["status"] == "ready"
    assert result["target"]["declared_site_count"] == 44
    assert result["target"]["active_site_count"] == 1
    assert result["target"]["occluded_site_count"] == 43
    assert result["target"]["visibility_coverage"] == pytest.approx(1 / 44)
    assert result["target"]["visibility_quality"] == "degraded_interval_global_subset"


def test_plan_accepts_float32_round_trip_for_fractional_start(harness, monkeypatch):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "fractional-start-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
        "start_s": 1.4,
        "duration_s": 4.0,
        "fps": 10.0,
    })
    monkeypatch.setattr(
        "wt_radar.adapter.animation.animation_preflight",
        lambda *_: native_preflight(frame_count=40),
    )
    monkeypatch.setattr(radar_tools, "_runtime_evidence", lambda: {
        "torch": {"cuda_available": True, "device_name": "test-gpu"}
    })
    result = run(tools, "plan_animation_measurement", {
        "scene_id": scene.scene_id,
        "radar_object_id": ensured["radar"]["object_id"],
        "start_s": 1.4,
        "duration_s": 4.0,
        "fps": 10.0,
    })
    assert result["status"] == "ready"
    assert result["measurement"]["start_s"] == pytest.approx(1.4)
    assert result["measurement"]["frame_count"] == 40


def test_submit_requires_matching_plan_preflight_receipt(harness):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "missing-preflight-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
        "duration_s": 5.0,
        "fps": 10.0,
    })
    sensor = scene.get_object(ensured["radar"]["object_id"])
    fingerprint = radar_tools._measurement_input_fingerprint(scene, sensor)

    async def exercise():
        with pytest.raises(ToolError) as error:
            await tools["submit_animation_measurement"].run({
                "scene_id": scene.scene_id,
                "radar_object_id": sensor.id,
                "operation_id": "missing-preflight-job",
                "expected_input_fingerprint": fingerprint,
            })
        return error.value

    failure = asyncio.run(exercise())
    assert failure.code == "preflight_receipt_missing"
    assert radar_tools._get_operation(scene._radar_test_context, "missing-preflight-job") is None


def test_trusted_direct_submit_can_wait_for_completed_operation(harness, monkeypatch):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "wait-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
        "duration_s": 5.0,
        "fps": 10.0,
    })
    sensor = scene.get_object(ensured["radar"]["object_id"])
    fingerprint = radar_tools._measurement_input_fingerprint(scene, sensor)
    remember_preflight(scene, sensor, fingerprint)

    async def complete(ctx, *, operation_id, **_kwargs):
        receipt = radar_tools._get_operation(ctx, operation_id)
        radar_tools._put_operation(ctx, {
            **receipt,
            "status": "replay_ready",
            "run_id": "native-run",
            "result_handle": "native-result",
            "next_step": "complete",
        })

    monkeypatch.setattr(radar_tools, "_run_animation_job", complete)

    async def exercise():
        return await tools["submit_animation_measurement"].run({
            "scene_id": scene.scene_id,
            "radar_object_id": sensor.id,
            "operation_id": "wait-job",
            "expected_input_fingerprint": fingerprint,
            "wait_for_completion": True,
        })

    result = asyncio.run(exercise())
    assert result["status"] == "replay_ready"
    assert result["run_id"] == "native-run"
    assert result["result_handle"] == "native-result"


def test_plan_refuses_interval_beyond_timeline_before_motion_sampling(harness):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "interval-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
        "start_s": 8.0,
        "duration_s": 5.0,
        "fps": 10.0,
    })
    with pytest.raises(ToolError) as error:
        run(tools, "plan_animation_measurement", {
            "scene_id": scene.scene_id,
            "radar_object_id": ensured["radar"]["object_id"],
            "start_s": 8.0,
            "duration_s": 5.0,
            "fps": 10.0,
        })
    assert error.value.code == "timeline_interval_unavailable"


@pytest.mark.parametrize("completed_status", ["verified", "replay_ready", "exported"])
@pytest.mark.parametrize("change", [None, "scene", "identity", "manifest"])
def test_submit_observe_and_verify_requires_native_evidence(harness, monkeypatch, completed_status, change):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "job-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
        "duration_s": 5.0,
        "fps": 10.0,
    })
    sensor_id = ensured["radar"]["object_id"]
    sensor = scene.get_object(sensor_id)
    radar = sensor.get_component("Radar")
    monkeypatch.setattr(
        "wt_radar.adapter.animation.animation_preflight", lambda *_: native_preflight(),
    )

    async def fake_simulate(*, animation=False, on_submitted=None):
        assert animation is True
        assert radar._agent_native_preflight == native_preflight()
        if on_submitted is not None:
            on_submitted("native-run")
        await asyncio.sleep(0)
        radar._solver_run_id = "native-run"
        radar._solver_result_handle = "native-result"
        radar._animation_result = True
        return "complete"

    monkeypatch.setattr(radar, "_simulate_async", fake_simulate)
    expected = radar_tools._measurement_input_fingerprint(scene, sensor)
    remember_preflight(scene, sensor, expected)
    expected_versions = {
        name: radar_tools._package_version(name)
        for name in ("witwin-radar", "witwin-channel", "witwin")
    }
    current_manifest = [{
        **topology_manifest(),
        "frame_count": 50,
        "cube_shape": [50, 3, 4, 128, 256],
        "all_finite": True,
        "site_count": 4,
        "frame_diagnostics_count": 50,
        "timebase_finite": True,
        "motion_arrays_finite": True,
        "signal_nonzero": True,
        "signal_abs_max": 1e-6,
        "moving_site_count": 4,
        "motion_speed_max_mps": 0.8,
        "motion_position_extent_m": 1.2,
        "nonzero_return_frames": 40,
        "dynamic_delay_rate_frames": 40,
        "coupled_dynamic_return_frames": 40,
        "device": "cuda",
        "input_fingerprint": expected,
        "scene_id": scene.scene_id,
        "model": "studio_visible_skinned_surface_sites_v2",
        "versions": expected_versions,
        "times_s": (np.arange(50, dtype=np.float64) / 10.0).tolist(),
    }]
    monkeypatch.setattr(radar, "_query_result", lambda *_: current_manifest[0])
    ready_replay = {
        "status": "available", "payload_available": True,
        "payload_integrity_verified": True, "motion_current": True,
        "ready_for_timeline": True, "frame_count": 50,
    }
    monkeypatch.setattr(radar_tools, "_recorded_replay_summary", lambda *_: ready_replay)

    async def exercise():
        submitted = await tools["submit_animation_measurement"].run({
            "scene_id": scene.scene_id,
            "radar_object_id": sensor_id,
            "operation_id": "job-1",
            "expected_input_fingerprint": expected,
        })
        assert submitted["status"] == "queued"
        task = radar_tools._ACTIVE_JOBS[
            radar_tools._job_key(scene._radar_test_context, "job-1")
        ]
        await task

    asyncio.run(exercise())
    observed = run(tools, "get_simulation", {
        "scene_id": scene.scene_id, "radar_object_id": sensor_id, "operation_id": "job-1",
    })
    assert observed["status"] == "replay_ready"
    assert not hasattr(radar, "_agent_native_preflight")
    assert observed["verification"]["passed"] is True
    assert observed["can_claim_success"] is True
    verified = asyncio.run(tools["verify_result"].run({
        "scene_id": scene.scene_id, "radar_object_id": sensor_id, "operation_id": "job-1",
    }))
    assert verified["status"] == "replay_ready"
    assert verified["verification"]["passed"] is True
    assert verified["can_claim_success"] is True

    # A later replay/export receipt must remain verifiable without losing that
    # milestone, but an earlier successful check is not a freshness guarantee.
    ctx = scene._radar_test_context
    receipt = radar_tools._get_operation(ctx, "job-1")
    radar_tools._put_operation(ctx, {**receipt, "status": completed_status, "next_step": "complete"})
    persisted_before = copy.deepcopy(radar_tools._get_operation(ctx, "job-1"))
    if change == "scene":
        scene.get_object("catstray").name = "Changed Cat"
    elif change == "identity":
        radar._solver_result_handle = "other-result"
    elif change == "manifest":
        current_manifest[0] = {}
    args = {"scene_id": scene.scene_id, "radar_object_id": sensor_id, "operation_id": "job-1"}
    if change == "identity":
        with pytest.raises(ToolError) as error:
            asyncio.run(tools["verify_result"].run(args))
        assert error.value.code == "result_handle_unavailable"
    else:
        repeated = asyncio.run(tools["verify_result"].run(args))
        if change == "scene":
            assert repeated["status"] == "stale"
            assert repeated["can_claim_success"] is False
        elif change == "manifest":
            assert repeated["status"] == "failed"
            assert repeated["verification"]["passed"] is False
            assert repeated["can_claim_success"] is False
        else:
            assert repeated["status"] == completed_status
            assert repeated["verification"]["passed"] is True
            assert repeated["next_step"] == "complete"
    assert radar_tools._get_operation(ctx, "job-1") == persisted_before


def test_prepare_replay_refuses_component_status_without_published_asset(harness, monkeypatch):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "unpublished-replay-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
        "duration_s": 5.0,
        "fps": 10.0,
    })
    sensor = scene.get_object(ensured["radar"]["object_id"])
    radar = sensor.get_component("Radar")
    fingerprint = radar_tools._measurement_input_fingerprint(scene, sensor)
    radar._animation_result = True
    radar._solver_run_id = "unpublished-run"
    radar._solver_result_handle = "unpublished-result"
    radar._last_result_input_fingerprint = fingerprint
    radar_tools._put_operation(scene._radar_test_context, {
        "operation_id": "unpublished-replay-job",
        "scene_id": scene.scene_id,
        "radar_object_id": sensor.id,
        "input_fingerprint": fingerprint,
        "status": "verified",
        "run_id": "unpublished-run",
        "result_handle": "unpublished-result",
        "verification": {"passed": True},
        "next_step": "prepare_replay_or_export",
    })

    async def source_changed():
        return "Source changed; old replay was not published."

    monkeypatch.setattr(radar, "prepare_synchronized_replay", source_changed)
    with pytest.raises(ToolError) as error:
        asyncio.run(tools["prepare_replay"].run({
            "scene_id": scene.scene_id,
            "radar_object_id": sensor.id,
            "operation_id": "unpublished-replay-job",
        }))
    assert error.value.code == "replay_preparation_unverified"
    assert error.value.detail["component_status"].startswith("Source changed")
    assert error.value.detail["recorded_replay"]["status"] == "not_prepared"
    assert radar_tools._get_operation(
        scene._radar_test_context, "unpublished-replay-job",
    )["status"] == "verified"


def test_cancel_simulation_waits_for_interrupted_receipt(harness, monkeypatch):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "cancel-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
        "duration_s": 5.0,
        "fps": 10.0,
    })
    sensor_id = ensured["radar"]["object_id"]
    sensor = scene.get_object(sensor_id)
    radar = sensor.get_component("Radar")
    monkeypatch.setattr(
        "wt_radar.adapter.animation.animation_preflight", lambda *_: native_preflight(),
    )
    started = asyncio.Event()
    never = asyncio.Event()

    async def fake_simulate(*, animation=False, on_submitted=None):
        assert animation is True
        if on_submitted is not None:
            on_submitted("cancel-native-run")
        started.set()
        await never.wait()

    monkeypatch.setattr(radar, "_simulate_async", fake_simulate)
    expected = radar_tools._measurement_input_fingerprint(scene, sensor)
    remember_preflight(scene, sensor, expected)

    async def exercise():
        await tools["submit_animation_measurement"].run({
            "scene_id": scene.scene_id,
            "radar_object_id": sensor_id,
            "operation_id": "cancel-job",
            "expected_input_fingerprint": expected,
        })
        await started.wait()
        return await tools["cancel_simulation"].run({
            "scene_id": scene.scene_id,
            "radar_object_id": sensor_id,
            "operation_id": "cancel-job",
        })

    cancelled = asyncio.run(exercise())
    assert cancelled["status"] == "interrupted"
    assert cancelled["error"]["code"] == "task_cancelled"
    assert cancelled["can_claim_success"] is False


def test_verify_rejects_wrong_shape_scene_model_versions_and_sites(harness, monkeypatch):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "bad-evidence-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
        "duration_s": 5.0,
        "fps": 10.0,
    })
    sensor = scene.get_object(ensured["radar"]["object_id"])
    radar = sensor.get_component("Radar")
    monkeypatch.setattr(
        "wt_radar.adapter.animation.animation_preflight", lambda *_: native_preflight(),
    )

    async def fake_simulate(*, animation=False, on_submitted=None):
        radar._solver_run_id = "bad-run"
        if on_submitted is not None:
            on_submitted("bad-run")
        radar._solver_result_handle = "bad-result"
        radar._animation_result = True

    monkeypatch.setattr(radar, "_simulate_async", fake_simulate)
    expected = radar_tools._measurement_input_fingerprint(scene, sensor)
    remember_preflight(scene, sensor, expected)
    monkeypatch.setattr(radar, "_query_result", lambda *_: {
        **topology_manifest(site_count=999),
        "frame_count": 50,
        "cube_shape": [50, 1],
        "all_finite": True,
        "site_count": 999,
        "timebase_finite": True,
        "motion_arrays_finite": True,
        "device": "cuda",
        "input_fingerprint": expected,
        "scene_id": "wrong-scene",
        "model": "wrong-model",
        "versions": {},
        "times_s": (np.arange(50, dtype=np.float64) / 10.0).tolist(),
    })

    async def exercise():
        submitted = await tools["submit_animation_measurement"].run({
            "scene_id": scene.scene_id,
            "radar_object_id": sensor.id,
            "operation_id": "bad-evidence-job",
            "expected_input_fingerprint": expected,
        })
        assert submitted["can_claim_success"] is False
        task = radar_tools._ACTIVE_JOBS[
            radar_tools._job_key(scene._radar_test_context, "bad-evidence-job")
        ]
        await task
        return tools["get_simulation"].run({
            "scene_id": scene.scene_id,
            "radar_object_id": sensor.id,
            "operation_id": "bad-evidence-job",
        })

    result = asyncio.run(exercise())
    assert result["status"] == "failed"
    assert result["can_claim_success"] is False
    checks = result["verification"]["checks"]
    assert checks["cube_shape_matches"] is False
    assert checks["site_count_matches"] is False
    assert checks["scene_id_matches"] is False
    assert checks["model_matches"] is False
    assert checks["versions_match"] is False


@pytest.mark.parametrize(
    ("patch", "failed_check"),
    [
        ({"signal_nonzero": False, "signal_abs_max": 0.0}, "nonzero_signal_present"),
        ({"moving_site_count": 0, "motion_speed_max_mps": 0.0}, "target_motion_present"),
        ({"coupled_dynamic_return_frames": 0}, "native_motion_coupling_present"),
        ({"active_site_ids": [3_000_001, 3_000_000, 3_000_002, 3_000_003]},
         "active_site_ids_match"),
    ],
)
def test_native_manifest_checks_reject_empty_or_static_measurements(patch, failed_check):
    fingerprint = "a" * 64
    versions = {"witwin-radar": "0.3.0", "witwin-channel": "0.5.0", "witwin": "0.4.0"}
    expected = {
        "scene_id": "bedroom_004",
        "model": "studio_visible_skinned_surface_sites_v2",
        "frame_count": 2,
        "cube_shape": [2, 3, 4, 128, 256],
        "site_count": 4,
        **topology_manifest(),
        "versions": versions,
        "start_s": 1.0,
        "fps": 10.0,
    }
    receipt = {
        "input_fingerprint": fingerprint,
        "result_handle": "result",
        "run_id": "run",
    }
    manifest = {
        **expected,
        "device": "cuda",
        "all_finite": True,
        "timebase_finite": True,
        "motion_arrays_finite": True,
        "signal_nonzero": True,
        "signal_abs_max": 1e-6,
        "moving_site_count": 4,
        "motion_speed_max_mps": 0.8,
        "motion_position_extent_m": 1.0,
        "nonzero_return_frames": 2,
        "dynamic_delay_rate_frames": 2,
        "coupled_dynamic_return_frames": 2,
        "input_fingerprint": fingerprint,
        "times_s": [1.0, 1.1],
        **patch,
    }
    checks = radar_tools._native_manifest_checks(manifest, receipt, expected)
    assert checks[failed_check] is False
    assert not all(checks.values())


@pytest.mark.parametrize("fake_device", ["cpu", "cuda-fake-cpu", "cuda_cpu", "", "CUDA:0"])
def test_verify_rejects_noncanonical_cuda_device(harness, monkeypatch, fake_device):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id, "operation_id": f"device-ensure-{fake_device}",
        "target_object_id": "catstray", "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0], "duration_s": 5.0, "fps": 10.0,
    })
    sensor = scene.get_object(ensured["radar"]["object_id"])
    radar = sensor.get_component("Radar")
    monkeypatch.setattr(
        "wt_radar.adapter.animation.animation_preflight", lambda *_: native_preflight(),
    )
    async def fake_simulate(*, animation=False, on_submitted=None):
        radar._solver_run_id = "device-run"
        if on_submitted is not None:
            on_submitted("device-run")
        radar._solver_result_handle = "device-result"
        radar._animation_result = True
    monkeypatch.setattr(radar, "_simulate_async", fake_simulate)
    expected = radar_tools._measurement_input_fingerprint(scene, sensor)
    remember_preflight(scene, sensor, expected)
    versions = {name: radar_tools._package_version(name) for name in ("witwin-radar", "witwin-channel", "witwin")}
    monkeypatch.setattr(radar, "_query_result", lambda *_: {
        **topology_manifest(),
        "frame_count": 50, "cube_shape": [50, 3, 4, 128, 256],
        "all_finite": True, "site_count": 4, "frame_diagnostics_count": 50,
        "timebase_finite": True,
        "motion_arrays_finite": True,
        "device": fake_device, "input_fingerprint": expected,
        "scene_id": scene.scene_id, "model": "studio_visible_skinned_surface_sites_v2",
        "versions": versions, "times_s": (np.arange(50) / 10.0).tolist(),
    })

    async def exercise():
        await tools["submit_animation_measurement"].run({
            "scene_id": scene.scene_id, "radar_object_id": sensor.id,
            "operation_id": f"device-job-{fake_device}", "expected_input_fingerprint": expected,
        })
        task = radar_tools._ACTIVE_JOBS[
            radar_tools._job_key(scene._radar_test_context, f"device-job-{fake_device}")
        ]
        await task
        return tools["get_simulation"].run({
            "scene_id": scene.scene_id, "radar_object_id": sensor.id,
            "operation_id": f"device-job-{fake_device}",
        })

    result = asyncio.run(exercise())
    assert result["status"] == "failed"
    assert result["verification"]["checks"]["cuda_device"] is False


def test_submit_rejects_scene_changed_after_preflight(harness):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "stale-ensure",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
    })
    sensor = scene.get_object(ensured["radar"]["object_id"])
    expected = radar_tools._measurement_input_fingerprint(scene, sensor)
    scene.get_object("catstray").name = "Changed Cat"

    async def exercise():
        with pytest.raises(ToolError) as error:
            await tools["submit_animation_measurement"].run({
                "scene_id": scene.scene_id,
                "radar_object_id": sensor.id,
                "operation_id": "stale-job",
                "expected_input_fingerprint": expected,
            })
        assert error.value.code == "stale_preflight"

    asyncio.run(exercise())
