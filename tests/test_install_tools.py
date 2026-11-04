"""Agent tools for the installation rule, labels and output checks (step 3g)."""
import asyncio
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_agent_tools import ToolCollector, run
from test_animation import make_rig
from test_install_rule import box_room
from test_snapshot import scene  # noqa: F401
from witwin_server.api import TimelineManager
from wt_radar import agent_tools as radar_tools


def register(scene, project_dir="."):
    collector = ToolCollector()
    server = SimpleNamespace(get_scene=lambda scene_id: scene if scene_id == scene.scene_id else None,
                             default_scene_dir=str(project_dir))
    context = SimpleNamespace(scene=scene, api=SimpleNamespace(server=server), tools=collector)
    radar_tools.register(context)
    return collector.values


@pytest.fixture
def room():
    scene = box_room(scene_id="install_room")
    scene.timeline_manager = TimelineManager(scene)
    scene.timeline_manager.clip.duration = 5.0
    return scene, register(scene)


def test_installation_plan_needs_the_room_only_and_proposes_fixed_ensure_arguments(room):
    scene, tools = room
    before = scene.to_dict()
    result = run(tools, "plan_sensor_placement", {
        "scene_id": scene.scene_id, "placement_mode": "automatic", "placement_objective": "installation"})
    assert result["ok"] and result["status"] == "ready_for_confirmation"
    placement = result["placement"]
    assert placement["mode"] == "installation"
    assert placement["mount"]["position_m"][1] == pytest.approx(2.0)
    assert placement["zones"]["person"]["area_m2"] > 0
    assert placement["rejected_counts"]["planar_normal_in_beam"] > 0
    proposed = result["proposed_ensure_sensor_arguments"]
    assert proposed["placement_mode"] == "fixed"
    assert proposed["placement_objective"] == "installation"
    assert proposed["position_m"] == placement["mount"]["position_m"]
    assert proposed["aim_point_m"] == placement["mount"]["aim_point_m"]
    assert proposed["height_m"] == 2.0
    # Default measured interval: clip length minus 0.2 s at 10 fps.
    assert (proposed["start_s"], proposed["duration_s"], proposed["fps"]) == (0.0, 4.8, 10.0)
    assert "targets" not in proposed
    assert result["requires_targets_before_ensure"] is True
    assert scene.to_dict() == before


def test_installation_plan_reports_a_missed_key_event_as_blocked(room):
    scene, tools = room
    result = run(tools, "plan_sensor_placement", {
        "scene_id": scene.scene_id, "placement_mode": "automatic", "placement_objective": "installation",
        "key_events": [{"id": "inside_wall", "actor_kind": "person", "position_m": [2.5, 0., -1.]}],
    })
    assert not result["ok"] and result["status"] == "blocked"
    blocker = result["blockers"][0]
    assert blocker["code"] == "RADAR_ZONE_MISSES_EVENT"
    assert blocker["next_step"]
    assert blocker["detail"]["missed_events"][0]["id"] == "inside_wall"


def test_installation_rule_fields_are_validated(room):
    scene, tools = room
    with pytest.raises(Exception) as raised:
        run(tools, "plan_sensor_placement", {
            "scene_id": scene.scene_id, "placement_mode": "automatic", "placement_objective": "installation",
            "install_rule": {"height_m": 2.0, "aim_height_m": 2.5}})
    assert getattr(raised.value, "code", "") == "invalid_sensor_placement"


def test_ensure_sensor_installs_at_the_rule_mount(scene):
    original, component = scene
    make_rig(original, component)
    original.remove_object("radar")
    tools = register(original)
    result = run(tools, "ensure_sensor", {
        "scene_id": original.scene_id, "operation_id": "install-1", "target_object_id": "target",
        "placement_mode": "automatic", "placement_objective": "installation",
        "install_rule": {"height_m": 1.5, "aim_height_m": .3},
        "duration_s": .2, "fps": 10,
    })
    assert result["ok"] and result["placement"]["mode"] == "installation"
    sensor = original.get_object(result["radar"]["object_id"])
    position = list(sensor.get_component("Transform").position)
    assert position == pytest.approx(result["placement"]["mount"]["position_m"])
    assert position[1] == pytest.approx(original.get_object("floor").get_component("Transform").position[1]
                                        + .05 + 1.5)


def test_label_visibility_tool_returns_labels_and_the_key_event_gate(scene):
    original, component = scene
    make_rig(original, component)
    component.animation_targets_json = '[{"object_id": "target", "site_budget": 6}]'
    tools = register(original)
    assert tools["label_visibility"].permission_tier == "read"
    assert tools["label_visibility"].side_effects is False
    result = run(tools, "label_visibility", {
        "scene_id": original.scene_id, "radar_object_id": "radar",
        "key_events": [{"id": "start", "object_id": "target", "t_s": 0.}]})
    assert result["ok"] and result["label_method"] == "studio_los_v1"
    assert result["gate"]["key_events_in_view"] is True
    assert result["targets"][0]["label_counts"]["in_view"] == 2
    compact = run(tools, "label_visibility", {
        "scene_id": original.scene_id, "radar_object_id": "radar", "include_frames": False})
    assert "frames" not in compact["targets"][0]


def _write_result(path, frame_count, *, amplitude=1.):
    from witwin.radar import Radar
    from witwin.radar.processing import ProcessingAxes
    from witwin.radar.synthesis.assembly import SynthesisResult

    radar = Radar.from_dict({
        "num_tx": 1, "num_rx": 1, "tx_loc": [[0, 0, 0]], "rx_loc": [[0, 0, 0]],
        "adc_samples": 64, "chirp_per_frame": 32,
        "fc": 77e9, "slope": 60.012, "adc_start_time": 6, "sample_rate": 4400,
        "idle_time": 7, "ramp_end_time": 65, "power": 15,
    }, device="cpu")
    system = radar.system_config
    axes = ProcessingAxes.from_synthesis(
        SynthesisResult.from_fmcw(torch.zeros((32, 1, 64), dtype=torch.complex64), system.waveform_spec()),
        system.waveform_spec(), system.sensors.array)
    record = dataclasses.asdict(axes)
    for key in ("range_m", "velocity_mps"):
        record[key] = record[key].tolist()
    cube = np.zeros((frame_count, 1, 1, 32, 64), dtype=np.complex64)
    cube[:, 0, 0, :, 5] = amplitude * np.exp(2j * np.pi * 4 * np.arange(32) / 32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, cube=cube, times_s=np.arange(frame_count) / 10., result_schema_version=np.asarray(1),
             processing_axes_json=np.asarray(json.dumps(record)),
             producer_metadata_json=np.asarray(json.dumps({"scene_id": "snapshot-test"})),
             range_m=axes.range_m.numpy(), velocity_mps=axes.velocity_mps.numpy())
    return path


def test_check_radar_output_runs_the_checks_on_a_project_result(scene, tmp_path):
    original, component = scene
    make_rig(original, component)
    component.animation_targets_json = '[{"object_id": "target", "site_budget": 6}]'
    tools = register(original, tmp_path)
    assert tools["check_radar_output"].permission_tier == "read"
    path = _write_result(tmp_path / "results" / "radar-animation" / "studio-animation-test.npz", 2)
    result = asyncio.run(run(tools, "check_radar_output", {
        "scene_id": original.scene_id, "radar_object_id": "radar", "result_path": str(path)}))
    assert result["ok"]
    assert result["checks"]["all_finite"] and result["checks"]["frame_count_matches"]
    assert result["checks"]["replay_ready"] is False
    assert result["status"] == "failed" and "replay_ready" in result["failed_checks"]
    assert result["key_events_in_view"] is True
    assert "frames" not in result

    outside = tmp_path / "elsewhere.npz"
    _write_result(outside, 2)
    with pytest.raises(Exception) as raised:
        asyncio.run(run(tools, "check_radar_output", {
            "scene_id": original.scene_id, "radar_object_id": "radar", "result_path": str(outside)}))
    assert getattr(raised.value, "code", "") == "result_path_outside_project"


def test_installation_plan_predicts_the_peak_radial_speed_from_the_baked_motion(scene):
    original, component = scene
    make_rig(original, component)
    original.remove_object("radar")
    tools = register(original)
    result = run(tools, "plan_sensor_placement", {
        "scene_id": original.scene_id, "placement_mode": "automatic", "placement_objective": "installation",
        "target_object_id": "target", "install_rule": {"profile": "fall"},
    })
    assert result["ok"], result
    velocity = result["placement"]["velocity"]
    assert velocity["checked"] is True
    assert 0. < velocity["peak_radial_mps"] <= velocity["limit_mps"] < velocity["max_unambiguous_speed_mps"]
    assert result["placement"]["rule"]["profile"] == "fall"

    blocked = run(tools, "plan_sensor_placement", {
        "scene_id": original.scene_id, "placement_mode": "automatic", "placement_objective": "installation",
        "target_object_id": "target", "install_rule": {"profile": "fall", "max_speed_fraction": .01},
    })
    assert blocked["status"] == "blocked"
    assert blocked["blockers"][0]["code"] == "RADAR_VELOCITY_ALIASING"


def test_a_fall_key_event_selects_the_fall_profile(room):
    scene, tools = room
    result = run(tools, "plan_sensor_placement", {
        "scene_id": scene.scene_id, "placement_mode": "automatic", "placement_objective": "installation",
        "key_events": [{"id": "impact", "kind": "fall_impact", "actor_kind": "person",
                        "position_m": [2.5, 0., 2.0], "test_height_m": .3}],
    })
    assert result["ok"], result
    assert result["placement"]["rule"]["profile"] == "fall"
    assert result["placement"]["mount"]["position_m"][1] <= 1.3 + 1e-9
    assert result["proposed_ensure_sensor_arguments"]["height_m"] <= 1.3


def test_velocity_check_catches_a_peak_between_radar_frames(scene):
    from test_animation import add_track

    original, component = scene
    make_rig(original, component)
    original.remove_object("radar")
    # Replace the slow walk with a 25 m/s burst between the 10 fps frames 0.0 and 0.1 s.
    original.timeline_manager.clip.tracks.clear()
    add_track(original, "bone0", "position", [(0., [0., 0., 0.]), (.04, [0., 0., 0.]), (.07, [0., 0., -.75]),
                                             (2., [0., 0., -.75])])
    tools = register(original)
    args = {"scene_id": original.scene_id, "placement_mode": "automatic", "placement_objective": "installation",
            "target_object_id": "target", "install_rule": {"profile": "fall"}}
    velocity = run(tools, "plan_sensor_placement", args)["placement"]["velocity"]
    # The peak lies between the 10 fps Radar frames; frame-only sampling would see nothing.
    assert velocity["sample_fps"] == 30.0
    assert velocity["t_s"] == pytest.approx(1 / 15)
    assert velocity["peak_radial_mps"] > 1.
    # The chosen mount looks nearly across the burst; a stricter share leaves none.
    blocked = run(tools, "plan_sensor_placement", {**args, "install_rule": {"profile": "fall",
                                                                           "max_speed_fraction": .5}})
    assert blocked["blockers"][0]["code"] == "RADAR_VELOCITY_ALIASING"
    assert blocked["blockers"][0]["detail"]["sample_fps"] == 30.0
