"""UI/lifecycle regressions; solver transport is stubbed, RF/DSP is untouched."""
import asyncio
import hashlib
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from test_saved_result import payload as saved_payload
from wt_radar.components.radar import RadarComponent


@pytest.fixture
def component(monkeypatch):
    value = RadarComponent()
    value.view = "range_profile"
    value._solver_result_handle = "recorded-animation"
    value._solver_run_id = "run-animation"
    value._animation_result = True
    # Isolated lifecycle cases have no ProjectServer; live identity is tested below.
    monkeypatch.setattr(value, "_scene_is_live", lambda: True, raising=False)
    return value


def test_saved_load_leaves_animation_mode(component, monkeypatch):
    from wt_radar.adapter import animation, saved_result

    saved = SimpleNamespace(times_s=[0.0, 0.1], producer={},
                            axes=SimpleNamespace(waveform="fmcw", output_domain="spectrum"))
    monkeypatch.setattr(saved_result.SavedRadarResult, "load", lambda path: saved)
    monkeypatch.setattr(saved_result, "show_saved_result", lambda value: "saved preview")

    def unexpected_animation(value):
        pytest.fail("Loading a saved result still dispatched the previous animation")

    monkeypatch.setattr(animation, "show_animation", unexpected_animation)
    assert component.load_saved_result() == "saved preview"
    assert not component._animation_result
    assert component._saved_result is saved
    assert not component._solver_result_handle


@pytest.mark.parametrize("operation,expected", [("animation_export", 180.), ("animation_replay", 180.), ("animation_view", None)])
def test_only_large_animation_transfers_extend_query_wait(component, monkeypatch, operation, expected):
    import wt_radar.components.radar as module
    calls = []
    def query(*args, **kwargs):
        calls.append(kwargs)
        return {"data": {"ok": True}}
    monkeypatch.setattr(module, "api", SimpleNamespace(solvers=SimpleNamespace(query=query)))
    assert component._query_result(operation, {}) == {"ok": True}
    assert calls[0].get("timeout") == expected
    assert calls[0]["run_id"] == "run-animation"


def test_saved_load_cannot_replace_inflight_solve(component, monkeypatch):
    from wt_radar.adapter import saved_result

    component._snapshot_running = True
    component._animation_result = False
    calls = []
    saved = SimpleNamespace(times_s=[0.0, 0.1], producer={},
                            axes=SimpleNamespace(waveform="fmcw", output_domain="spectrum"))

    def load(path):
        calls.append(path)
        return saved

    monkeypatch.setattr(saved_result.SavedRadarResult, "load", load)
    monkeypatch.setattr(saved_result, "show_saved_result", lambda value: "saved preview")
    with pytest.raises(ValueError, match="(?i)(solve|active|running)"):
        component.load_saved_result()
    assert calls == []
    assert component._solver_result_handle == "recorded-animation"


def test_failed_animation_preview_clears_previous_frame(component, monkeypatch):
    component.signal_figure.line([0, 1], [2, 3])

    def rejected(operation, params):
        raise ValueError("Animation frame must be 0..49")

    monkeypatch.setattr(component, "_query_result", rejected)
    component.update_view()
    assert not component.signal_figure._series
    assert "view failed" in component.animation_status.lower()


def test_export_completion_cannot_overwrite_new_result(component, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def delayed_query(operation, params):
        assert operation == "animation_export"
        entered.set()
        assert release.wait(3)
        return {"path": "C:/old-run.npz"}

    monkeypatch.setattr(component, "_query_result", delayed_query)

    async def exercise():
        task = asyncio.create_task(component.export_animation_result())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            component._solver_result_handle = "new-animation"
            component._solver_run_id = "new-run"
            component.animation_status = "New run active"
            component.animation_export_path = ""
        finally:
            release.set()
        try:
            await task
        except (ValueError, RuntimeError):
            pass  # Explicit stale-result refusal is also acceptable.
        assert component.animation_status == "New run active"
        assert component.animation_export_path == ""

    asyncio.run(exercise())


def stub_solve_transport(component, monkeypatch, solve, *, status=None, cancel=None):
    from wt_radar.components import radar
    from wt_radar.adapter import animation
    from witwin_server.features.solvers import scene_ref

    request = {"duration_s": 1.0, "fps": 2.0, "time_s": 0.0}
    monkeypatch.setattr(animation, "animation_request", lambda value: request)
    monkeypatch.setattr(scene_ref, "make_scene_ref", lambda value: {"scene": {}})
    monkeypatch.setattr(radar, "api", SimpleNamespace(solvers=SimpleNamespace(
        solve=solve,
        status=status or (lambda *_args, **_kwargs: {"activeRunId": None}),
        cancel=cancel or (lambda *_args, **_kwargs: False),
    )))
    monkeypatch.setattr(radar.Notifications, "success", lambda *args: None)


def test_initial_preview_query_is_not_on_asyncio_loop(component, monkeypatch):
    from wt_radar.adapter import animation

    run = SimpleNamespace(status="succeeded", outputs={"resultHandle": "next"}, run_id="next-run")
    stub_solve_transport(component, monkeypatch, lambda *args, **kwargs: run)
    observed = []

    def query(operation, params):
        try:
            asyncio.get_running_loop()
            observed.append("event-loop")
        except RuntimeError:
            observed.append("worker")
        return {}

    monkeypatch.setattr(component, "_query_result", query)
    monkeypatch.setattr(animation, "apply_animation_view", lambda value, payload: "preview")
    asyncio.run(component._simulate_async(animation=True))
    assert observed == ["worker"]


def test_completed_animation_automatically_prepares_replay(component, monkeypatch):
    from wt_radar.components import radar
    from wt_radar.adapter import animation

    run = SimpleNamespace(status="succeeded", outputs={"resultHandle": "next"}, run_id="next-run")
    stub_solve_transport(component, monkeypatch, lambda *args, **kwargs: run)
    radar.api.server = SimpleNamespace(handlers={"binary_assets": object()})
    monkeypatch.setattr(component, "_query_result", lambda *_: {})
    monkeypatch.setattr(animation, "apply_animation_view", lambda *_: "preview")
    prepared = []

    async def prepare():
        prepared.append((component._solver_run_id, component._solver_result_handle))
        return "Replay ready"

    monkeypatch.setattr(component, "prepare_synchronized_replay", prepare)
    asyncio.run(component._simulate_async(animation=True))

    assert prepared == [("next-run", "next")]


def test_animation_status_reports_interval_visibility_coverage(component, monkeypatch):
    from wt_radar.adapter import animation

    run = SimpleNamespace(status="succeeded", outputs={"resultHandle": "next"}, run_id="next-run")
    stub_solve_transport(component, monkeypatch, lambda *args, **kwargs: run)
    payload = {
        "title": "preview",
        "metadata": {
            "topology_preflight": {
                "declared_site_count": 44,
                "active_site_count": 19,
                "visibility_quality": "degraded_interval_global_subset",
            }
        },
    }
    monkeypatch.setattr(component, "_query_result", lambda *_: payload)

    def apply(value, received):
        value._last_animation_view_metadata = dict(received["metadata"])
        return received["title"]

    monkeypatch.setattr(animation, "apply_animation_view", apply)
    asyncio.run(component._simulate_async(animation=True))
    assert "19/44" in component.animation_status
    assert "43.2%" in component.animation_status
    assert "degraded_interval_global_subset" in component.animation_status


def test_cancelled_async_call_cancels_the_exact_native_solver_run(component, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    run = SimpleNamespace(status="cancelled", outputs={}, error={"message": "cancelled"}, run_id="next-run")
    cancelled = []

    def solve(*args, **kwargs):
        kwargs["on_submitted"]("next-run")
        entered.set()
        assert release.wait(3)
        return run

    def cancel(run_id, reason):
        cancelled.append((run_id, reason))
        release.set()
        return True

    stub_solve_transport(
        component,
        monkeypatch,
        solve,
        status=lambda *_args, **_kwargs: {"activeRunId": "next-run"},
        cancel=cancel,
    )

    async def exercise():
        task = asyncio.create_task(component._simulate_async(animation=True))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
        finally:
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not component._snapshot_running
        assert not component._animation_result
        assert not component._solver_result_handle
        assert "cancel" in component.animation_status.lower()
        assert cancelled == [("next-run", "Studio Radar animation request cancelled")]

    asyncio.run(exercise())


def test_native_completion_that_wins_cancel_race_is_preserved(component, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    run = SimpleNamespace(status="succeeded", outputs={"resultHandle": "kept"}, run_id="next-run")

    def solve(*args, **kwargs):
        kwargs["on_submitted"]("next-run")
        entered.set()
        assert release.wait(3)
        return run

    def status(*_args, **_kwargs):
        release.set()
        return {"activeRunId": "next-run"}

    def cancel(*_args):
        release.set()
        return False

    stub_solve_transport(component, monkeypatch, solve, status=status, cancel=cancel)
    monkeypatch.setattr(component, "_refresh_animation_view", lambda: asyncio.sleep(0))

    async def exercise():
        task = asyncio.create_task(component._simulate_async(animation=True))
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        result = await task
        assert "completion won" in result.lower()
        assert component._animation_result
        assert component._solver_result_handle == "kept"
        assert component._solver_run_id == "next-run"

    asyncio.run(exercise())


def test_initial_preview_failure_is_not_hidden_by_solve_success(component, monkeypatch):
    run = SimpleNamespace(status="succeeded", outputs={"resultHandle": "next"}, run_id="next-run")
    stub_solve_transport(component, monkeypatch, lambda *args, **kwargs: run)

    def query(operation, params):
        raise RuntimeError("Preview failed to decode result")

    monkeypatch.setattr(component, "_query_result", query)
    asyncio.run(component._simulate_async(animation=True))
    # Native data remains exportable, but UI must not hide that rendering failed.
    assert component._animation_result
    assert component._solver_result_handle == "next"
    assert "view failed" in component.animation_status.lower()
    assert not component.signal_figure._series


def test_new_preview_wins_while_old_query_is_still_running(component, monkeypatch):
    from wt_radar.adapter import animation

    entered, release = threading.Event(), threading.Event()
    applied = []

    def query(operation, params):
        if params["frame"] == 0:
            entered.set()
            assert release.wait(3)
        return {"frame": params["frame"]}

    def apply(value, payload):
        applied.append(payload["frame"])
        value.signal_figure.clear().line([0, 1], [payload["frame"]] * 2)
        return str(payload["frame"])

    monkeypatch.setattr(component, "_query_result", query)
    monkeypatch.setattr(animation, "apply_animation_view", apply)

    async def exercise():
        first = asyncio.create_task(component._refresh_animation_view())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            component._animation_result = False  # Suppress automatic callback in this unit test.
            component.animation_frame_index = 1
            component._animation_result = True
            assert await component._refresh_animation_view() == "1"
            assert applied == [1]
        finally:
            release.set()
            await first
        assert applied == [1]

    asyncio.run(exercise())


@pytest.mark.parametrize("status,outputs,message", [
    ("failed", {}, "native CUDA rejected geometry"),
    ("cancelled", {}, "cancelled by user"),
    ("succeeded", {}, "no result handle"),
])
def test_nonresult_solver_completion_cannot_publish_previous_animation(
        component, monkeypatch, status, outputs, message):
    run = SimpleNamespace(status=status, outputs=outputs,
                          error={"message": message}, run_id="new-run")
    stub_solve_transport(component, monkeypatch, lambda *args, **kwargs: run)
    component.signal_figure.line([0, 1], [2, 3])
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(component._simulate_async(animation=True))
    assert not component._snapshot_running
    assert not component._animation_result
    assert not component._solver_result_handle
    assert not component.signal_figure._series
    assert message in component.animation_status


@pytest.fixture
def export_source(tmp_path, saved_payload):
    from witwin_server.features.solvers.result_ref import read_result_ref

    cache, project = tmp_path / "solver-cache", tmp_path / "project"
    cache.mkdir()
    project.mkdir()
    source = cache / "native-result.npz"
    np.savez_compressed(source, **saved_payload)
    raw = source.read_bytes()
    reference = {"path": str(source), "size": len(raw),
                 "contentHash": "sha256:" + hashlib.sha256(raw).hexdigest()}

    def validated_read(solver_id, ref):
        assert solver_id == "witwin.radar.simulate"
        return read_result_ref(ref, root=cache)

    api = SimpleNamespace(server=SimpleNamespace(default_scene_dir=str(project)),
                          solvers=SimpleNamespace(read_result_ref=validated_read))
    return api, reference, source, project, raw


def test_permanent_export_survives_solver_cache_removal_and_reloads(export_source):
    from wt_radar.adapter.animation import persist_export
    from wt_radar.adapter.saved_result import SavedRadarResult

    api, reference, source, project, raw = export_source
    destination = Path(persist_export(api, reference))
    assert destination.is_relative_to(project / "results" / "radar-animation")
    assert destination.read_bytes() == raw
    source.unlink()  # Only this pytest-owned temporary cache file is removed.
    restored = SavedRadarResult.load(destination)
    assert restored.cube.shape == (2, 1, 1, 4, 8)
    np.testing.assert_array_equal(restored.times_s, [0.0, 0.1])
    assert destination.read_bytes() == raw


def test_repeated_export_reuses_verified_file_without_solver_cache(export_source):
    from wt_radar.adapter.animation import persist_export

    api, reference, source, project, raw = export_source
    destination = Path(persist_export(api, reference))
    source.unlink()  # Only the pytest-owned temporary solver result.
    assert persist_export(api, reference, existing_path=str(destination)) == str(destination)
    assert list((project / 'results' / 'radar-animation').glob('*.npz')) == [destination]
    assert destination.read_bytes() == raw


def test_export_button_reuses_same_run_but_exports_new_run_separately(component, export_source, monkeypatch):
    from wt_radar.components import radar

    api, reference, source, project, raw = export_source
    monkeypatch.setattr(radar, 'api', api)
    notices = []
    monkeypatch.setattr(radar.Notifications, 'success', lambda *args: notices.append(args))
    monkeypatch.setattr(component, '_query_result', lambda *_: reference)
    first = asyncio.run(component.export_animation_result())
    first_path = component.animation_export_path
    assert 'Exported native' in first
    source.unlink()  # Only the pytest-owned temporary cache, not the permanent file.
    repeated = asyncio.run(component.export_animation_result())
    assert 'existing NPZ verified (no new copy)' in repeated
    assert component.animation_export_path == first_path
    assert len(list((project / 'results' / 'radar-animation').glob('*.npz'))) == 1
    assert notices[-1][1] == 'Existing NPZ verified; no new copy created.'
    assert not component._export_running
    # A new simulation clears this path; its export must not reuse the prior run.
    component._solver_result_handle = 'new-result'
    component._solver_run_id = 'new-run'
    component.animation_export_path = ''
    source.write_bytes(raw)
    asyncio.run(component.export_animation_result())
    assert component.animation_export_path != first_path
    assert len(list((project / 'results' / 'radar-animation').glob('*.npz'))) == 2


@pytest.mark.parametrize('tamper', ['contents', 'size', 'outside_project'])
def test_repeated_export_never_trusts_path_alone(export_source, tmp_path, tamper):
    from wt_radar.adapter.animation import persist_export

    api, reference, _, _, raw = export_source
    destination = Path(persist_export(api, reference))
    if tamper == 'outside_project':
        destination = tmp_path / 'unrelated.npz'
        destination.write_bytes(raw)
    elif tamper == 'contents':
        changed = bytearray(raw)
        changed[-1] ^= 1
        destination.write_bytes(changed)
    else:
        destination.write_bytes(raw[:-1])
    before = destination.read_bytes()
    with pytest.raises(ValueError, match='(?i)(checksum|size|outside)'):
        persist_export(api, reference, existing_path=str(destination))
    assert destination.read_bytes() == before


@pytest.mark.parametrize("tamper", ["hash", "size", "path_escape"])
def test_export_refuses_invalid_source_without_publishing(export_source, tmp_path, tamper):
    from wt_radar.adapter.animation import persist_export

    api, reference, _, project, _ = export_source
    invalid = dict(reference)
    if tamper == "hash":
        invalid["contentHash"] = "sha256:" + "0" * 64
    elif tamper == "size":
        invalid["size"] += 1
    else:
        invalid["path"] = str(tmp_path / "outside-solver.npz")
    with pytest.raises(ValueError):
        persist_export(api, invalid)
    assert not (project / "results").exists()


def test_export_refuses_resolved_destination_escape(export_source, tmp_path, monkeypatch):
    from wt_radar.adapter.animation import persist_export

    api, reference, _, project, _ = export_source
    folder = project / "results" / "radar-animation"
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == folder:
            return tmp_path / "outside-project"
        return original_resolve(path, *args, **kwargs)

    # Model a pre-existing junction without requiring Windows symlink privileges.
    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="escapes"):
        persist_export(api, reference)
    assert not folder.exists()


def test_export_never_overwrites_existing_destination(export_source, monkeypatch):
    from wt_radar.adapter import animation

    api, reference, _, _, raw = export_source
    monkeypatch.setattr(animation.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed-test-id"))
    destination = Path(animation.persist_export(api, reference))
    with pytest.raises(FileExistsError):
        animation.persist_export(api, reference)
    assert destination.read_bytes() == raw


def attached_component(monkeypatch):
    from witwin_server import Scene, SceneObject
    from wt_radar.components import radar

    scene = Scene(scene_id="radar-owner-scene")
    owner = SceneObject(id="radar", mesh_type="Empty")
    value = owner.add_component(RadarComponent())
    scene.add_object(owner)
    server = SimpleNamespace(scenes={scene.scene_id: scene})
    monkeypatch.setattr(radar, "api", SimpleNamespace(server=server))
    return scene, owner, value, server


def test_live_identity_rejects_closed_reopened_or_removed_owner(monkeypatch):
    from witwin_server import Scene

    scene, owner, value, server = attached_component(monkeypatch)
    assert value._scene_is_live()
    del server.scenes[scene.scene_id]
    assert not value._scene_is_live()
    server.scenes[scene.scene_id] = Scene(scene_id=scene.scene_id)
    assert not value._scene_is_live()
    server.scenes[scene.scene_id] = scene
    scene.remove_object(owner.id)
    assert not value._scene_is_live()


def test_live_identity_rejects_removed_replaced_component(monkeypatch):
    _, owner, value, _ = attached_component(monkeypatch)
    assert value._scene_is_live()
    assert owner.remove_component("Radar")
    assert not value._scene_is_live()
    owner.add_component(RadarComponent())
    assert not value._scene_is_live()


def test_solve_does_not_publish_into_closed_reopened_scene(monkeypatch):
    from witwin_server import Scene
    from wt_radar.components import radar

    scene, _, value, server = attached_component(monkeypatch)
    value.view = "range_profile"
    entered, release = threading.Event(), threading.Event()
    run = SimpleNamespace(status="succeeded", outputs={"resultHandle": "late-result"}, run_id="late-run")

    def solve(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return run

    stub_solve_transport(value, monkeypatch, solve)
    radar.api.server = server

    async def exercise():
        task = asyncio.create_task(value._simulate_async(animation=True))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            server.scenes[scene.scene_id] = Scene(scene_id=scene.scene_id)
        finally:
            release.set()
        await task
        assert not value._solver_result_handle
        assert not value._animation_result
        assert not value.signal_figure._series
        assert not value._snapshot_running

    asyncio.run(exercise())


def test_preview_does_not_publish_after_scene_replaced(monkeypatch):
    from witwin_server import Scene
    from wt_radar.adapter import animation

    scene, _, value, server = attached_component(monkeypatch)
    value.view = "range_profile"
    value._animation_result = True
    value._solver_result_handle = "recorded"
    value._solver_run_id = "recorded-run"
    entered, release = threading.Event(), threading.Event()
    applied = []

    def query(operation, params):
        entered.set()
        assert release.wait(3)
        return {}

    monkeypatch.setattr(value, "_query_result", query)
    monkeypatch.setattr(animation, "apply_animation_view", lambda *args: applied.append(args))

    async def exercise():
        task = asyncio.create_task(value._refresh_animation_view())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            server.scenes[scene.scene_id] = Scene(scene_id=scene.scene_id)
        finally:
            release.set()
        await task
        assert applied == []

    asyncio.run(exercise())
