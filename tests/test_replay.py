"""Numeric replay transport, provenance and bounded loading; no RF substitutes."""
from dataclasses import replace
import asyncio
from hashlib import sha256
import io
import json
import threading
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from test_saved_result import payload, save  # noqa: F401
from witwin.radar.processing import ProcessingCube, range_doppler_map, range_profile
from witwin_server import Scene, SceneObject
from witwin_server.features.timeline import TimelineManager
from wt_radar.adapter import memory_budget, replay
from wt_radar.adapter.saved_result import SavedRadarResult, show_saved_result
from wt_radar.components.radar import RadarComponent


@pytest.fixture
def saved(tmp_path, payload):
    return SavedRadarResult.load(save(tmp_path, payload))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_recording_is_exact_official_selected_rp_rd_float32(saved, dtype):
    axes = replace(saved.axes, num_tx=2, num_rx=3,
                   tx_loc_half_wavelength=((0., 0., 0.), (1., 0., 0.)),
                   rx_loc_half_wavelength=((0., 0., 0.), (1., 0., 0.), (2., 0., 0.)))
    weights = torch.arange(1, 7).reshape(1, 2, 3, 1, 1)
    cube = saved.cube.to(dtype) * weights
    result = replace(saved, cube=cube, axes=axes)
    record, data = replay.build_recording(result, tx=1, rx=2)
    nr, nd = len(axes.range_m), len(axes.velocity_mps)
    assert record["schema"] == 1
    assert record["frameStride"] == nr * (nd + 1)
    assert len(data) == 2 * nr * (nd + 1) * 4
    actual = np.frombuffer(data, dtype="<f4").reshape(2, record["frameStride"])
    for frame in range(2):
        profile = range_profile(ProcessingCube(cube[frame], axes), window="rectangular", remove_dc=False)
        rd = range_doppler_map(profile, window="hann")
        np.testing.assert_array_equal(actual[frame, :nr], profile.data[1, 2, 0].abs().numpy().astype("<f4"))
        np.testing.assert_array_equal(actual[frame, nr:].reshape(nd, nr), rd.data[1, 2].abs().numpy().astype("<f4"))
    assert record["rangeM"] == axes.range_m.tolist()
    assert record["velocityMps"] == axes.velocity_mps.tolist()
    assert record["profileMax"] == float(actual[:, :nr].max())
    assert record["dopplerMax"] == float(actual[:, nr:].max())
    assert record["timesS"] == [0., .1]
    assert record["endTimeS"] == pytest.approx(.2)
    assert "TX1 RX2" in record["sourceLabel"]


def test_recording_accepts_animation_metadata_and_measured_duration(saved):
    result = SimpleNamespace(cube=saved.cube, axes=saved.axes, times_s=saved.times_s + 7.,
                             metadata={"scene_id": "room", "duration_s": .2, "motion_fingerprint": "digest"})
    record, _ = replay.build_recording(result)
    assert record["sceneId"] == "room"
    assert record["sceneFingerprint"] == "digest"
    assert record["endTimeS"] == pytest.approx(7.2)


@pytest.mark.parametrize("times", [np.array([0.]), np.array([0., 0.]), np.array([.1, 0.]), np.array([0., np.nan])])
def test_recording_refuses_bad_timestamps(saved, times):
    with pytest.raises(ValueError, match="timestamps"):
        replay.build_recording(replace(saved, times_s=times))


@pytest.mark.parametrize("tx,rx", [(-1, 0), (1, 0), (0, -1), (0, 1)])
def test_recording_refuses_bad_antenna_without_clamping(saved, tx, rx):
    with pytest.raises(ValueError, match="antenna"):
        replay.build_recording(saved, tx, rx)


def test_recording_budget_is_checked_before_processing(saved, monkeypatch):
    import witwin.radar.processing as processing
    monkeypatch.setattr(replay, "MAX_PREVIEW_BYTES", 1)
    monkeypatch.setattr(processing, "range_profile", lambda *a, **k: pytest.fail("DSP ran before budget rejection"))
    with pytest.raises(ValueError, match="128 MiB"):
        replay.build_recording(saved)


def test_recording_rejects_nonfinite_and_invalid_duration(saved):
    bad = saved.cube.clone()
    bad[0, 0, 0, 0, 0] = complex(float("nan"), 0.)
    with pytest.raises(ValueError, match="Nonfinite"):
        replay.build_recording(replace(saved, cube=bad))
    with pytest.raises(ValueError, match="end time"):
        replay.build_recording(replace(saved, producer={"duration_s": .05}))


def test_memory_guards_absolute_budget_and_free_headroom(monkeypatch):
    monkeypatch.setattr(memory_budget, "available_memory_bytes", lambda: 20 * 1024**3)
    memory_budget.require_result_memory(memory_budget.MAX_RESULT_BYTES)
    with pytest.raises(ValueError, match="1536 MiB"):
        memory_budget.require_result_memory(memory_budget.MAX_RESULT_BYTES + 1)
    size = 400 * 1024**2
    monkeypatch.setattr(memory_budget, "available_memory_bytes", lambda: 3 * size - 1)
    with pytest.raises(ValueError, match="Insufficient free"):
        memory_budget.require_result_memory(size)
    monkeypatch.setattr(memory_budget, "available_memory_bytes", lambda: 3 * size)
    memory_budget.require_result_memory(size)


def test_saved_header_rejects_shape_bomb_before_np_load(tmp_path, monkeypatch):
    import wt_radar.adapter.saved_result as transport
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {
        "descr": np.dtype("complex64").str, "fortran_order": False,
        "shape": (1000000, 3, 4, 128, 256),
    })
    path = tmp_path / "shape-bomb.npz"
    with ZipFile(path, "w") as archive:
        archive.writestr("cube.npy", stream.getvalue())
    monkeypatch.setattr(transport.np, "load", lambda *a, **k: pytest.fail("Large array allocation path reached"))
    with pytest.raises(ValueError, match="header shape"):
        SavedRadarResult.load(path)


def test_saved_header_checks_headroom_before_np_load(tmp_path, payload, monkeypatch):
    import wt_radar.adapter.saved_result as transport
    path = save(tmp_path, payload)
    monkeypatch.setattr(memory_budget, "available_memory_bytes", lambda: 0)
    monkeypatch.setattr(transport.np, "load", lambda *a, **k: pytest.fail("Array load ran without memory headroom"))
    with pytest.raises(ValueError, match="Insufficient free"):
        SavedRadarResult.load(path)


def test_saved_cube_wraps_loaded_array_without_second_full_copy(tmp_path, payload, monkeypatch):
    import wt_radar.adapter.saved_result as transport
    path = save(tmp_path, payload)
    arrays = []
    original_load = transport.np.load

    class RecordingArchive:
        def __init__(self, archive):
            self.archive, self.files = archive, archive.files

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            array = self.archive[key]
            if key == "cube":
                arrays.append(array)
            return array

    monkeypatch.setattr(transport.np, "load", lambda *a, **k: RecordingArchive(original_load(*a, **k)))
    result = SavedRadarResult.load(path)
    assert len(arrays) == 1
    assert result.cube.data_ptr() == arrays[0].__array_interface__["data"][0]
    np.testing.assert_array_equal(result.cube.numpy(), payload["cube"])


def replay_scene():
    room = Scene(scene_id="test-room")
    room.timeline_manager = TimelineManager(room)
    room.timeline_manager.clip.duration = 1.
    obj = SceneObject(id="radar", mesh_type="Empty")
    component = obj.add_component(RadarComponent())
    room.add_object(obj)
    room.timeline_manager.clip.get_or_create_track("radar", "Transform", "position").add_keyframe(0., [0., 1., 0.])
    return room, component


class BinaryService:
    def __init__(self):
        self.registered, self.cleared = [], []

    def register_bytes(self, **kwargs):
        self.registered.append(kwargs)
        return SimpleNamespace(asset_id="new-asset", to_dict=lambda: {"asset_id": "new-asset", "byte_length": len(kwargs["data"])})

    def clear(self, asset_id):
        self.cleared.append(asset_id)


class DataSources:
    def __init__(self):
        self.registered = []

    def register(self, descriptor):
        self.registered.append(descriptor)
        return descriptor


def publish_fixture(saved):
    room, component = replay_scene()
    metadata = {"scene_id": room.scene_id, "motion_fingerprint": replay.motion_fingerprint(room)}
    record, data = replay.build_recording(replace(saved, producer=metadata))
    service = BinaryService()
    data_sources = DataSources()
    api = SimpleNamespace(
        server=SimpleNamespace(handlers={"binary_assets": SimpleNamespace(service=service)}),
        data_sources=data_sources,
    )
    return room, component, record, data, service, api


def test_publish_uses_binary_asset_and_bound_figure_not_rgb(saved):
    room, component, record, data, service, api = publish_fixture(saved)
    component.signal_figure.line([0, 1], [2, 3], label="stale")
    component._replay_asset_id = "old-asset"
    original_record = json.loads(json.dumps(record))
    replay.publish_recording(component, api, record, data)
    assert service.registered == [{"kind": "radar-replay", "format": "witwin.radar.replay.float32.v1",
                                   "data": data, "scene_id": room.scene_id, "persistent": True,
                                   "asset_id": f"radar-replay-{sha256(data).hexdigest()}"}]
    assert service.cleared == ["old-asset"]
    figure = component.signal_figure.to_dict()
    assert not component.signal_figure._series
    assert figure["type"] == "line"
    assert figure["data"]["recording"]["asset"]["asset_id"] == "new-asset"
    assert figure["data"]["recording"]["motionVerifiedAtPreparation"] is True
    descriptor = figure["data"]["recording"]["dataSource"]
    assert descriptor["mode"] == "timeline"
    assert descriptor["retention"] == "persistent"
    assert {channel["channelId"] for channel in descriptor["channels"]} == {
        "range_profile", "range_doppler",
    }
    assert component.signal_source["sourceId"] == descriptor["sourceId"]
    assert api.data_sources.registered == [descriptor]
    assert "recording" in figure["data"] and "image" not in figure["data"]
    assert record == original_record  # metadata supplied by producer is not mutated


@pytest.mark.parametrize("change", ["scene", "motion", "bytes", "service"])
def test_publish_refuses_mismatch_before_asset_or_figure_update(saved, change):
    room, component, record, data, service, api = publish_fixture(saved)
    component.signal_figure.line([0], [1], label="previous")
    before = component.signal_figure.to_dict()
    if change == "scene":
        record["sceneId"] = "other-room"
    elif change == "motion":
        room.timeline_manager.clip.tracks["radar/Transform/position"].add_keyframe(.5, [1., 1., 0.])
    elif change == "bytes":
        data = data[:-4]
    else:
        api.server.handlers.clear()
    with pytest.raises((ValueError, RuntimeError)):
        replay.publish_recording(component, api, record, data)
    assert not service.registered and not service.cleared
    after = component.signal_figure.to_dict()
    assert {k: v for k, v in before.items() if k != "timestamp"} == {k: v for k, v in after.items() if k != "timestamp"}


def test_motion_fingerprint_does_not_change_when_only_seeking_clock_moves():
    room, _ = replay_scene()
    before = replay.motion_fingerprint(room)
    room.timeline_manager.set_time(.3, apply_to_scene=True)
    assert replay.motion_fingerprint(room) == before


@pytest.mark.parametrize("view", ["range_profile", "range_spectrum", "range_doppler"])
def test_numeric_plot_and_figure_setter_preserve_data_semantics(saved, view):
    from wt_radar.adapter.snapshot import SnapshotResult, snapshot_view
    payload = snapshot_view(SnapshotResult(ProcessingCube(saved.cube[0], saved.axes), {}), {"view": view})
    component = RadarComponent()
    component.signal_figure.line([1], [99], label="obsolete")
    plot = replay.numeric_plot(payload)
    component.signal_figure.set_plot_data(plot)
    figure = component.signal_figure.to_dict()
    assert not component.signal_figure._series
    if view == "range_doppler":
        assert figure["type"] == "imshow"
        assert np.asarray(figure["data"]["values"]).shape == (4, 8)
        np.testing.assert_array_equal(figure["data"]["values"], payload["magnitude"])
        assert figure["data"]["x"] == payload["range_m"]
        assert figure["data"]["y"] == payload["velocity_mps"]
        assert figure["data"]["origin"] == "lower"
    else:
        assert figure["type"] == "line"
        series = figure["data"]["series"]
        expected_keys = ["magnitude"] if view == "range_profile" else ["real", "imag"]
        for row, key in zip(series, expected_keys):
            assert row["x"] == payload["range_m"]
            assert row["y"] == payload[key]


def test_saved_rd_is_numeric_and_matches_official_map(saved):
    component = RadarComponent()
    component._saved_result = saved
    component.view = "range_doppler"
    component.saved_frame_index = 1
    show_saved_result(component)
    data = component.signal_figure.to_dict()["data"]
    expected = range_doppler_map(saved.profile(1), window="hann").data[0, 0].abs().numpy()
    np.testing.assert_array_equal(data["values"], expected)


def test_publish_failure_clears_new_asset_and_preserves_old_id(saved, monkeypatch):
    _, component, record, data, service, api = publish_fixture(saved)
    component._replay_asset_id = "old-asset"
    component.signal_figure.line([0], [1], label="previous")
    before = component.signal_figure.to_dict()

    def fail_set_plot_data(_plot):
        raise RuntimeError("injected figure publication failure")

    monkeypatch.setattr(component.signal_figure, "set_plot_data", fail_set_plot_data)
    with pytest.raises(RuntimeError, match="injected figure publication failure"):
        replay.publish_recording(component, api, record, data)
    assert len(service.registered) == 1
    assert service.cleared == ["new-asset"]
    assert component._replay_asset_id == "old-asset"
    after = component.signal_figure.to_dict()
    assert {k: v for k, v in before.items() if k != "timestamp"} == {k: v for k, v in after.items() if k != "timestamp"}


@pytest.mark.parametrize("field", ["static_clutter_removal", "show_cfar"])
def test_prepare_refuses_processing_setting_enabled_during_worker(saved, monkeypatch, field):
    _, component = replay_scene()
    component._saved_result = saved
    monkeypatch.setattr(component, "_scene_is_live", lambda: True)
    started, release = threading.Event(), threading.Event()
    generation = component._animation_view_generation

    def blocked_build(result, tx, rx):
        assert result is saved and (tx, rx) == (0, 0)
        started.set()
        if not release.wait(5):
            raise TimeoutError("test did not release replay worker")
        return {}, b""

    def forbidden_publish(*args, **kwargs):
        pytest.fail("Replay published despite unsupported processing setting")

    monkeypatch.setattr(replay, "build_recording", blocked_build)
    monkeypatch.setattr(replay, "publish_recording", forbidden_publish)

    async def exercise():
        task = asyncio.create_task(component.prepare_synchronized_replay())
        try:
            assert await asyncio.to_thread(started.wait, 5)
            assert component._replay_preparing
            # Simulate a changed field without unrelated automatic view refresh.
            component.set_field(field, True, silent=True)
            release.set()
            with pytest.raises(ValueError, match="enabled during preparation"):
                await asyncio.wait_for(task, 5)
        finally:
            release.set()
            if not task.done():
                await asyncio.wait_for(task, 5)

    asyncio.run(exercise())
    assert not component._replay_preparing
    assert component._saved_result is saved
    assert component._animation_view_generation == generation
    assert "Replay preparation failed" in str(component.replay_status)


@pytest.mark.parametrize("busy_flag", ["_replay_preparing", "_export_running"])
def test_load_saved_refuses_concurrent_prepare_or_export_without_mutation(saved, monkeypatch, busy_flag):
    _, component = replay_scene()
    component._saved_result = saved
    component._solver_result_handle = "existing-result"
    component._solver_run_id = "existing-run"
    component._replay_asset_id = "old-asset"
    component.signal_figure.line([0], [1], label="previous")
    before = component.signal_figure.to_dict()
    generation = component._animation_view_generation
    setattr(component, busy_flag, True)
    monkeypatch.setattr(SavedRadarResult, "load", lambda *a, **k: pytest.fail("Load started during another operation"))

    with pytest.raises(ValueError, match="before loading a saved result"):
        component.load_saved_result()
    assert component._saved_result is saved
    assert component._solver_result_handle == "existing-result"
    assert component._solver_run_id == "existing-run"
    assert component._replay_asset_id == "old-asset"
    assert component._animation_view_generation == generation
    assert getattr(component, busy_flag)
    after = component.signal_figure.to_dict()
    assert {k: v for k, v in before.items() if k != "timestamp"} == {k: v for k, v in after.items() if k != "timestamp"}
