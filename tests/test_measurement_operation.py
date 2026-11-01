"""One Radar measurement per authored input: deterministic and traceable.

R47 v16: the Agent reused its ensure_sensor operation_id for the submit, got a
misleading "different Radar inputs" conflict, invented a second id and the
Session ended with two receipts. The preflight now names the one measurement
operation and a re-submit of the same inputs resolves to that operation.
"""
import asyncio

import pytest

from wt_radar import agent_tools as radar_tools
from witwin_server.tools.base import ToolError

from test_agent_tools import harness, native_preflight, run


def planned_measurement(scene, tools, monkeypatch):
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id,
        "operation_id": "human_wave_radar_01",
        "target_object_id": "catstray",
        "position_m": [1.0, 1.0, 2.0],
        "aim_point_m": [0.0, 0.4, 0.0],
        "duration_s": 5.0,
        "fps": 10.0,
    })
    monkeypatch.setattr(
        "wt_radar.adapter.animation.animation_preflight",
        lambda *_: native_preflight(moving_object_ids=["catstray"]),
    )
    monkeypatch.setattr(radar_tools, "_runtime_evidence", lambda: {
        "torch": {"cuda_available": True, "device_name": "test-gpu"}
    })
    started = []

    async def job(ctx, *, operation_id, **_kwargs):
        started.append(operation_id)

    monkeypatch.setattr(radar_tools, "_run_animation_job", job)
    plan = run(tools, "plan_animation_measurement", {
        "scene_id": scene.scene_id,
        "radar_object_id": ensured["radar"]["object_id"],
    })
    return ensured, plan, started


def submit(tools, args):
    return asyncio.run(tools["submit_animation_measurement"].run(args))


def test_preflight_names_one_deterministic_measurement_operation(harness, monkeypatch):
    scene, tools = harness
    _, plan, _ = planned_measurement(scene, tools, monkeypatch)
    fingerprint = plan["input_fingerprint"]

    next_step = plan["next_step"]
    assert next_step["tool"] == "witwin.radar.submit_animation_measurement"
    assert next_step["arguments"] == {
        "scene_id": scene.scene_id,
        "radar_object_id": plan["radar"]["object_id"],
        "operation_id": f"radar-measure-{fingerprint[:24]}",
        "expected_input_fingerprint": fingerprint,
    }
    assert plan["measurement_operation_id"] == next_step["arguments"]["operation_id"]


def test_resubmitting_the_same_inputs_never_starts_a_second_simulation(harness, monkeypatch):
    scene, tools = harness
    _, plan, started = planned_measurement(scene, tools, monkeypatch)
    first = submit(tools, plan["next_step"]["arguments"])
    second = submit(tools, {
        **plan["next_step"]["arguments"], "operation_id": "human_wave_measurement_02",
    })

    assert first["status"] == "queued"
    assert second["operation_id"] == first["operation_id"]
    assert second["deduplicated_from_operation_id"] == "human_wave_measurement_02"
    assert started == [first["operation_id"]]
    measurements = [
        receipt for receipt in radar_tools._operation_map(scene._radar_test_context).values()
        if receipt.get("kind") == "animation_measurement"
    ]
    assert [receipt["operation_id"] for receipt in measurements] == [first["operation_id"]]


def test_an_ensure_sensor_operation_id_is_rejected_with_the_measurement_id(harness, monkeypatch):
    scene, tools = harness
    _, plan, started = planned_measurement(scene, tools, monkeypatch)

    with pytest.raises(ToolError) as caught:
        submit(tools, {**plan["next_step"]["arguments"], "operation_id": "human_wave_radar_01"})

    assert caught.value.code == "operation_id_reserved"
    assert caught.value.detail["reserved_by"] == "ensure_sensor"
    assert caught.value.detail["measurement_operation_id"] == plan["measurement_operation_id"]
    assert started == []


def test_a_failed_measurement_gets_a_fresh_retry_operation(harness, monkeypatch):
    scene, tools = harness
    ensured, plan, started = planned_measurement(scene, tools, monkeypatch)
    first = submit(tools, plan["next_step"]["arguments"])
    receipt = radar_tools._get_operation(scene._radar_test_context, first["operation_id"])
    radar_tools._put_operation(scene._radar_test_context, {**receipt, "status": "failed"})

    replanned = run(tools, "plan_animation_measurement", {
        "scene_id": scene.scene_id,
        "radar_object_id": ensured["radar"]["object_id"],
    })
    retry_id = replanned["next_step"]["arguments"]["operation_id"]
    assert retry_id == f"{first['operation_id']}-retry-1"
    retried = submit(tools, replanned["next_step"]["arguments"])
    assert retried["operation_id"] == retry_id
    assert started == [first["operation_id"], retry_id]
