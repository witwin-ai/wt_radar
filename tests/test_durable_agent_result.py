"""Synthetic transport unit tests; native execution is verified separately."""
import asyncio
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_agent_tools import harness, run, ToolCollector  # noqa: F401
from test_saved_result import payload  # noqa: F401
from wt_radar import agent_tools
from wt_radar.adapter.replay import motion_fingerprint
from wt_radar.components import radar as radar_module
from witwin_server.core import wtscene
from witwin_server.core.scene import Scene
from witwin_server.features.assets.binary.service import BinaryAssetService
from witwin_server.tools.base import ToolError


@pytest.fixture
def cold_result(harness, payload, tmp_path, monkeypatch):
    scene, tools = harness
    ensured = run(tools, "ensure_sensor", {
        "scene_id": scene.scene_id, "operation_id": "ensure",
        "target_object_id": "catstray", "position_m": [1, 1, 2],
        "aim_point_m": [0, .4, 0], "duration_s": .2, "fps": 10,
    })
    sensor = scene.get_object(ensured["radar"]["object_id"])
    fingerprint = agent_tools._measurement_input_fingerprint(scene, sensor)
    payload["producer_metadata_json"] = np.asarray(json.dumps({
        "scene_id": scene.scene_id, "input_fingerprint": fingerprint,
        "motion_fingerprint": motion_fingerprint(scene), "duration_s": .2,
    }))
    folder = tmp_path / "results" / "radar-animation"
    folder.mkdir(parents=True)
    path = folder / "recording.npz"
    np.savez(path, **payload)
    old_context = scene._radar_test_context
    old_context.api.server.default_scene_dir = tmp_path
    evidence = {**agent_tools._export_evidence(old_context, path),
                "input_fingerprint": fingerprint, "run_id": "native-run", "result_handle": "native-result"}
    receipt = {"operation_id": "recording", "scene_id": scene.scene_id,
               "radar_object_id": sensor.id, "input_fingerprint": fingerprint,
               "run_id": "native-run", "result_handle": "native-result", "status": "exported",
               "verification": {"passed": True}, "exports": {"first": evidence}}
    agent_tools._put_operation(old_context, receipt)
    state = copy.deepcopy(old_context._radar_operation_fallback)
    scene = Scene.from_dict(wtscene.loads(wtscene.dumps(scene.to_dict())))
    collector = ToolCollector()
    persistence = SimpleNamespace(runtime_volumes_dir=tmp_path / "volumes", volume_candidate_dirs=lambda: [tmp_path / "volumes"])
    server = SimpleNamespace(default_scene_dir=tmp_path, scenes={scene.scene_id: scene},
                             get_scene=lambda _: scene,
                             handlers={"binary_assets": SimpleNamespace(service=BinaryAssetService(persistence=persistence))})
    ctx = SimpleNamespace(scene=scene, api=SimpleNamespace(server=server), tools=collector,
                          _radar_operation_fallback=state)
    agent_tools.register(ctx)
    monkeypatch.setattr(radar_module, "api", ctx.api)
    sensor = scene.get_object(sensor.id)
    radar = sensor.get_component("Radar")
    monkeypatch.setattr(radar, "_query_result", lambda *a, **k: pytest.fail("Unexpected native worker call"))
    assert not radar._animation_result and not radar._solver_result_handle
    args = {"scene_id": scene.scene_id, "radar_object_id": sensor.id, "operation_id": "recording"}
    return scene, radar, collector.values, ctx, args, path


def test_cold_result_replay_and_export_without_solver(cold_result):
    scene, radar, tools, ctx, args, path = cold_result
    before_file, before_timeline = path.read_bytes(), copy.deepcopy(scene.timeline_manager.clip.to_dict())
    before_inspect = scene.to_dict()
    selected = run(tools, "inspect_pipeline", {"scene_id": scene.scene_id})["selected_radar"]
    assert selected["result"]["present"] is False
    assert selected["durable_results"][0]["operation_id"] == "recording"
    assert selected["durable_results"][0]["status"] == "available"
    assert scene.to_dict() == before_inspect
    # A caller who knows only the Scene can discover both required result IDs.
    discovered = {"scene_id": scene.scene_id, "radar_object_id": selected["object_id"],
                  "operation_id": selected["durable_results"][0]["operation_id"]}
    replay = asyncio.run(run(tools, "prepare_replay", discovered))
    assert replay["completion_status"] == "exported"
    assert agent_tools._recorded_replay_summary(ctx, radar.owner)["ready_for_timeline"]
    assert not radar._animation_result and not radar._solver_result_handle and radar._saved_result is None
    for export_id in ("first", "second", "second"):
        exported = asyncio.run(run(tools, "export_result", {**discovered, "export_operation_id": export_id}))
        assert exported["export"]["path"] == str(path)
        assert exported["reused"] is True
        assert path.read_bytes() == before_file
    assert list(path.parent.iterdir()) == [path]
    assert scene.timeline_manager.clip.to_dict() == before_timeline


def test_matching_live_result_reuses_verified_durable_export(cold_result):
    _scene, radar, tools, ctx, args, path = cold_result
    before = path.read_bytes()
    radar._animation_result = True
    radar._solver_run_id = "native-run"
    radar._solver_result_handle = "native-result"

    exported = asyncio.run(run(tools, "export_result", {
        **args, "export_operation_id": "warm-reuse",
    }))

    assert exported["reused"] is True
    assert exported["export"]["path"] == str(path)
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]
    assert list(ctx._radar_operation_fallback["recording"]["exports"]) == ["first"]


@pytest.mark.parametrize("action", ["prepare_replay", "export_result"])
@pytest.mark.parametrize("change", ["motion", "file", "producer", "schema", "receipt", "outside"])
def test_cold_result_refuses_stale_or_invalid_source(cold_result, action, change):
    scene, radar, tools, ctx, args, path = cold_result
    receipt = ctx._radar_operation_fallback["recording"]
    export = receipt["exports"]["first"]
    if change == "motion":
        scene.timeline_manager.clip.get_or_create_track("catstray", "Transform", "position").add_keyframe(0, [2, 0, 0])
    elif change == "file":
        path.write_bytes(b"corrupted unit-test result")
    elif change in {"producer", "schema"}:
        with np.load(path, allow_pickle=False) as archive:
            data = dict(archive)
        key = "producer_metadata_json" if change == "producer" else "result_schema_version"
        data[key] = np.asarray('{}' if change == "producer" else 999)
        np.savez(path, **data)
        export.update(agent_tools._export_evidence(ctx, path))
    elif change == "receipt":
        export["run_id"] = "wrong-run"
    else:
        export["path"] = str(path.parent.parent.parent.parent / "outside.npz")
    before_scene, before_file = copy.deepcopy(scene.to_dict()), path.read_bytes()
    selected = run(tools, "inspect_pipeline", {"scene_id": scene.scene_id})["selected_radar"]
    assert selected["durable_results"][0]["current"] is False
    with pytest.raises(ToolError) as error:
        asyncio.run(run(tools, action, {**args, **({"export_operation_id": "next"} if action == "export_result" else {})}))
    assert error.value.code == ("stale_result" if change == "motion" else "export_path_invalid" if change == "outside" else "saved_result_invalid")
    assert scene.to_dict() == before_scene and path.read_bytes() == before_file


@pytest.mark.parametrize("action", ["prepare_replay", "export_result"])
def test_saved_export_does_not_hide_a_different_live_result(cold_result, action):
    _scene, radar, tools, _ctx, args, _path = cold_result
    radar._animation_result = True
    radar._solver_run_id = "different-run"
    radar._solver_result_handle = "different-result"
    with pytest.raises(ToolError) as error:
        asyncio.run(run(tools, action, {**args, **({"export_operation_id": "next"} if action == "export_result" else {})}))
    assert error.value.code == "result_handle_unavailable"
