import time
import threading

import numpy as np
import witwin.radar as wr
from witwin_server import Scene

from wt_radar.adapter.config_map import ConfigMap


_CONFIG = {
    "num_tx": 3,
    "num_rx": 4,
    "fc": 77e9,
    "slope": 60.012,
    "power": 15.0,
    "adc_samples": 8,
    "adc_start_time": 6.0,
    "sample_rate": 4400.0,
    "idle_time": 7.0,
    "ramp_end_time": 58.0,
    "chirp_per_frame": 4,
    "frame_per_second": 10.0,
    "num_doppler_bins": 4,
    "num_range_bins": 8,
    "num_angle_bins": 8,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
}


class _FakeSolvers:
    def __init__(self):
        self.solve_calls = []
        self.query_calls = []
        self.call_calls = []

    def solve(self, solver_id, **kwargs):
        self.solve_calls.append((solver_id, kwargs))
        raise AssertionError("live stream path must not call api.solvers.solve")

    def query(self, solver_id, result_handle, op, params, **kwargs):
        self.query_calls.append((solver_id, result_handle, op, params, kwargs))
        raise AssertionError("live stream path must not call api.solvers.query")

    def call(self, solver_id, method, params=None, **kwargs):
        self.call_calls.append((solver_id, method, params or {}, kwargs))
        if method == "live_start":
            return {"sessionId": (params or {}).get("session_id"), "status": "started"}
        if method == "live_status":
            return {"status": "running"}
        return {"ok": True}


class _FakeStream:
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.statuses = []
        self.refs = []

    def update(self, status=None, **kwargs):
        self.statuses.append(status)

    def ref(self, channel):
        ref = {"streamId": self.stream_id, "channelId": channel}
        self.refs.append(ref)
        return ref


class _FakeStreams:
    def __init__(self):
        self.open_calls = []
        self.closed = []
        self.errors = []
        self.streams = {}

    def open(self, stream_id, **kwargs):
        self.open_calls.append((stream_id, kwargs))
        return self.streams.setdefault(stream_id, _FakeStream(stream_id))

    def close(self, stream_id, reason=None):
        self.closed.append((stream_id, reason))

    def error(self, stream_id, message, **kwargs):
        self.errors.append((stream_id, message, kwargs))


class _FakeDataSources:
    def __init__(self):
        self.registered = []
        self.timeline_datasets = []

    def register(self, descriptor):
        self.registered.append(descriptor)
        return descriptor

    def register_timeline_dataset(self, descriptor):
        self.timeline_datasets.append(descriptor)
        return descriptor


class _FakeApi:
    def __init__(self):
        self.solvers = _FakeSolvers()
        self.streams = _FakeStreams()
        self.data_sources = _FakeDataSources()


def _radar_in_scene():
    settings = ConfigMap.settings_to_studio(wr.RadarConfig.from_dict(_CONFIG))
    studio = Scene()
    studio.begin_batch()
    studio.add_object(settings)
    studio.end_batch()
    return settings.get_component("Radar"), settings


def _wait_for(predicate, label, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {label}")


def _calls(fake, method):
    return [call for call in fake.solvers.call_calls if call[1] == method]


def test_realtime_stream_uses_solver_live_session_and_updates_on_transform(monkeypatch):
    import wt_radar.components.radar as radar_mod

    fake = _FakeApi()
    monkeypatch.setattr(radar_mod, "api", fake)
    radar, settings = _radar_in_scene()
    radar.stream_max_fps = 20.0
    radar.stream_on_change_only = True
    radar.stream_channels = "raw,rd,pc"

    assert radar.start_stream() == "Stream started"
    try:
        _wait_for(lambda: len(_calls(fake, "live_start")) == 1, "live_start call")
        assert fake.solvers.solve_calls == []
        assert fake.solvers.query_calls == []

        start = _calls(fake, "live_start")[0]
        assert start[0] == "witwin.radar.simulate"
        assert start[2]["session_id"] == "radar.radar_settings.signal"
        assert start[2]["stream_id"] == "radar.radar_settings.signal"
        assert start[2]["channels"] == ["raw", "rd", "pc"]
        assert start[2]["max_fps"] == 20.0
        assert start[2]["config"]["live"]["stream_id"] == "radar.radar_settings.signal"
        assert start[2]["config"]["live"]["motion_sampling"] == "per_frame"
        assert start[3]["scene"] is radar.scene
        assert fake.streams.streams["radar.radar_settings.signal"].refs[0]["viewport"]["maxFps"] == 20.0

        settings.get_component("Transform").position = [1.0, 2.0, 3.0]
        _wait_for(lambda: len(_calls(fake, "live_update")) >= 1, "live_update after transform")
        update = _calls(fake, "live_update")[-1]
        assert update[2]["session_id"] == "radar.radar_settings.signal"
        assert update[2]["config"]["sensor"]["position"] == [1.0, 2.0, 3.0]
        assert "scene" not in update[3]

        assert radar.pause_stream() == "Stream paused"
        assert len(_calls(fake, "live_pause")) == 1
        paused_updates = len(_calls(fake, "live_update"))
        settings.get_component("Transform").position = [2.0, 2.0, 3.0]
        time.sleep(0.15)
        assert len(_calls(fake, "live_update")) == paused_updates

        assert radar.start_stream() == "Stream running"
        assert len(_calls(fake, "live_resume")) == 1
        settings.get_component("Transform").position = [3.0, 2.0, 3.0]
        _wait_for(lambda: len(_calls(fake, "live_update")) > paused_updates, "live_update after resume")
    finally:
        radar.stop_stream()
        _wait_for(lambda: radar._live_thread is None or not radar._live_thread.is_alive(), "live thread stop")

    assert radar.stream_status == "stopped"
    assert radar.signal_stream is None
    assert fake.streams.closed[-1] == ("radar.radar_settings.signal", "stopped")
    assert len(_calls(fake, "live_stop")) == 1
    assert fake.streams.errors == []
    assert fake.solvers.solve_calls == []
    assert fake.solvers.query_calls == []


def test_realtime_stream_defaults_to_change_only_rd_preview(monkeypatch):
    import wt_radar.components.radar as radar_mod

    fake = _FakeApi()
    monkeypatch.setattr(radar_mod, "api", fake)
    radar, _settings = _radar_in_scene()

    assert radar.stream_max_fps == 30.0
    assert radar.stream_on_change_only is True
    assert radar.stream_channels == "rd"
    assert radar.start_stream() == "Stream started"
    try:
        _wait_for(lambda: len(_calls(fake, "live_start")) == 1, "live_start call")
    finally:
        radar.stop_stream()
        _wait_for(lambda: radar._live_thread is None or not radar._live_thread.is_alive(), "live thread stop")

    params = _calls(fake, "live_start")[0][2]
    assert params["max_fps"] == 30.0
    assert params["channels"] == ["rd"]
    assert params["config"]["live"]["channels"] == ["rd"]
    assert params["config"]["live"]["stream_on_change_only"] is True
    assert radar.signal_stream is None
    stream = fake.streams.streams["radar.radar_settings.signal"]
    assert stream.refs[0]["viewport"]["maxFps"] == 30.0


def test_realtime_stream_descriptors_declare_generic_plot_specs():
    radar, _settings = _radar_in_scene()

    descriptors = {item["channelId"]: item for item in radar._stream_channel_descriptors()}

    assert descriptors["raw"]["plot"] == {
        "kind": "line",
        "x": {"mode": "index", "label": "ADC sample"},
        "y": {"label": "Amplitude"},
        "complex": "real_imag",
        "series": [
            {"component": "real", "label": "Real", "color": "#ff9500"},
            {"component": "imag", "label": "Imag", "color": "#00aaff"},
        ],
    }
    assert descriptors["rd"]["plot"] == {
        "kind": "heatmap",
        "x": {"label": "Range bin"},
        "y": {"label": "Doppler bin"},
        "colormap": "imshow",
        "range": {"mode": "full"},
    }
    assert descriptors["pc"]["plot"] == {
        "kind": "points",
        "stride": 6,
        "axes": ["x", "y", "z"],
        "colorBy": "intensity",
    }


def test_realtime_stream_preview_field_hides_label_and_stream_header():
    radar, _settings = _radar_in_scene()

    fields = {field["name"]: field for field in radar.to_dict()["fields"]}
    signal_stream = fields["signal_stream"]
    signal_source = fields["signal_source"]

    assert signal_stream.get("hide_label") is True
    assert signal_stream.get("hidden") is True
    assert "title" not in signal_stream
    assert signal_stream["widget"]["show_header"] is False
    assert signal_source["field_type"] == "data_source_ref"
    assert signal_source["widget"]["widget_type"] == "data-source"
    assert signal_source.get("transient") is True
    assert signal_source.get("hide_label") is True


def test_realtime_stream_registers_radar_result_data_source(monkeypatch):
    import wt_radar.components.radar as radar_mod

    fake = _FakeApi()
    monkeypatch.setattr(radar_mod, "api", fake)
    radar, _settings = _radar_in_scene()

    stream = radar._open_signal_stream(radar._stream_channel_descriptors())

    assert stream.stream_id == "radar.radar_settings.signal"
    assert fake.data_sources.registered
    descriptor = fake.data_sources.registered[-1]
    assert descriptor["sourceId"] == "radar.radar_settings.result"
    assert descriptor["mode"] == "stream"
    assert descriptor["kind"] == "radar.result"
    assert descriptor["streamId"] == "radar.radar_settings.signal"
    assert descriptor["defaultChannel"] == "rd"
    assert {channel["channelId"] for channel in descriptor["channels"]} == {"raw", "rd", "pc"}
    assert radar.signal_source["sourceId"] == "radar.radar_settings.result"
    assert radar.signal_source["streamId"] == "radar.radar_settings.signal"


def test_local_raw_stream_publishes_selected_adc_trace_for_line_plot():
    radar, _settings = _radar_in_scene()

    class FakeStream:
        def __init__(self):
            self.published = []

        def publish(self, channel_id, payload, metadata=None):
            self.published.append((channel_id, np.asarray(payload), metadata or {}))

    sig = np.zeros((2, 2, 1, 3), dtype=np.complex64)
    sig[1, 0, 0] = np.asarray([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
    radar.tx_index = 1
    radar.rx_index = 0
    radar._signal = type(
        "FakeTensor",
        (),
        {
            "shape": sig.shape,
            "detach": lambda self: self,
            "cpu": lambda self: self,
            "numpy": lambda self: sig,
        },
    )()
    stream = FakeStream()

    radar._publish_raw_stream(stream)

    assert len(stream.published) == 1
    channel_id, payload, metadata = stream.published[0]
    assert channel_id == "raw"
    assert payload.dtype == np.float32
    assert payload.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert metadata == {"dtype": "complex64", "shape": [3], "tx": 1, "rx": 0, "chirp": 0}


def test_realtime_stream_view_change_switches_signal_channel(monkeypatch):
    import wt_radar.components.radar as radar_mod

    fake = _FakeApi()
    monkeypatch.setattr(radar_mod, "api", fake)
    radar, _settings = _radar_in_scene()

    assert radar.start_stream() == "Stream started"
    try:
        _wait_for(lambda: len(_calls(fake, "live_start")) == 1, "live_start call")
        stream = fake.streams.streams["radar.radar_settings.signal"]
        assert stream.refs[-1]["channelId"] == "rd"

        live_updates_before = len(_calls(fake, "live_update"))
        radar.view = "raw_signal"

        _wait_for(lambda: stream.refs[-1]["channelId"] == "raw", "raw stream ref")
        assert radar.signal_stream["defaultChannel"] == "raw"
        _wait_for(lambda: len(_calls(fake, "live_update")) > live_updates_before, "live_update after view change")
    finally:
        radar.stop_stream()
        _wait_for(lambda: radar._live_thread is None or not radar._live_thread.is_alive(), "live thread stop")

    update = _calls(fake, "live_update")[-1]
    assert update[2]["channels"] == ["raw", "rd"]
    assert update[2]["config"]["live"]["channels"] == ["raw", "rd"]


def test_realtime_stream_migrates_legacy_continuous_defaults(monkeypatch):
    import wt_radar.components.radar as radar_mod

    fake = _FakeApi()
    monkeypatch.setattr(radar_mod, "api", fake)
    radar, _settings = _radar_in_scene()
    radar.stream_on_change_only = False
    radar.stream_channels = "raw,rd,pc"

    assert radar.start_stream() == "Stream started"
    try:
        _wait_for(lambda: len(_calls(fake, "live_start")) == 1, "live_start call")
    finally:
        radar.stop_stream()
        _wait_for(lambda: radar._live_thread is None or not radar._live_thread.is_alive(), "live thread stop")

    params = _calls(fake, "live_start")[0][2]
    assert radar.stream_on_change_only is True
    assert radar.stream_channels == "rd"
    assert params["channels"] == ["rd"]
    assert params["config"]["live"]["stream_on_change_only"] is True


def test_post_processing_field_change_refreshes_existing_preview():
    radar, _settings = _radar_in_scene()
    rendered = []
    published = []
    radar._solver_result_handle = "radar-handle"
    radar._solver_run_id = "radar-run"
    radar.update_view = lambda: rendered.append((str(radar.view), int(radar.tx_index))) or "rendered"
    radar._publish_signal_stream = lambda channels=None: published.append(channels)

    radar.view = "point_cloud"

    assert rendered == [("point_cloud", 0)]
    assert published == [None]


def test_update_view_is_internal_not_a_properties_button():
    radar, _settings = _radar_in_scene()

    buttons = radar.to_dict()["buttons"]

    assert "update_view" not in {button["name"] for button in buttons}
    assert "Update View" not in {button["display_name"] for button in buttons}


def test_solver_live_session_publishes_channels_from_one_radar_frame(monkeypatch):
    import wt_radar.solver_host as solver_host

    class FakeCtx:
        run_id = "live-run"

        def __init__(self):
            self.published = []
            self.logs = []

        def stream_publish(self, stream_id, channel_id, payload=None, **kwargs):
            self.published.append((stream_id, channel_id, np.asarray(payload).shape, kwargs.get("metadata") or {}))

        def log(self, message, level="info"):
            self.logs.append((level, message))

        def stream_error(self, stream_id, message, **kwargs):
            self.logs.append(("error", message))

    calls = []
    fake_signal = np.ones((1, 1, 2, 8), dtype=np.complex64)
    fake_result = type("Result", (), {"radar": object(), "signal": fake_signal})()

    def fake_run(scene, *, sensor, tracer, motion_sampling, t0, live_cache=None,
                 cache_key=None, platform_cache_key=None):
        calls.append({
            "scene": scene,
            "motion_sampling": motion_sampling,
            "t0": t0,
            "cache": live_cache,
            "cache_key": cache_key,
            "platform_cache_key": platform_cache_key,
        })
        return fake_result

    class FakeRD:
        mag_db = np.ones((4, 8), dtype=np.float32)

    monkeypatch.setattr(solver_host, "load_scene_ref", lambda ref: {"scene": ref})
    monkeypatch.setattr(solver_host.SolveRunner, "run", fake_run)
    monkeypatch.setattr(solver_host.SigProc, "range_doppler", lambda *args, **kwargs: FakeRD())
    monkeypatch.setattr(
        solver_host.SigProc,
        "point_cloud",
        lambda *args, **kwargs: np.zeros((2, 6), dtype=np.float32),
    )

    ctx = FakeCtx()
    result = solver_host.live_start(ctx, {
        "session_id": "session-1",
        "stream_id": "radar.demo.signal",
        "channels": ["raw", "rd", "pc"],
        "max_fps": 30.0,
        "sceneRef": {"initial": True},
        "config": {
            "sensor": {
                "position": [0.0, 0.0, 0.0],
                "target": [0.0, 0.0, -1.0],
                "up": [0.0, 1.0, 0.0],
                "fov": 70.0,
                "backend": "dirichlet",
                "pad_factor": 1,
                "device": "cuda",
            },
            "tracer": {},
            "t0": 2.5,
            "live": {
                "signature": "sig-1",
                "stream_on_change_only": True,
                "motion_sampling": "per_frame",
            },
        },
    })
    try:
        assert result["status"] == "started"
        _wait_for(lambda: len(ctx.published) >= 3, "initial live frame publish")
        assert len(calls) == 1
        assert calls[0]["motion_sampling"] == "per_frame"
        assert calls[0]["t0"] == 2.5
        assert calls[0]["cache_key"] == "sig-1"
        assert {item[1] for item in ctx.published} == {"raw", "rd", "pc"}
        assert any(item[1] == "raw" and item[3]["dtype"] == "complex64" for item in ctx.published)
        assert any(item[1] == "rd" and item[2] == (4, 8) for item in ctx.published)
        assert any(item[1] == "pc" and item[2] == (2, 6) for item in ctx.published)

        solver_host.live_update(ctx, {
            "session_id": "session-1",
            "sceneRef": {"updated": True},
            "config": {
                "sensor": {
                    "position": [1.0, 0.0, 0.0],
                    "target": [1.0, 0.0, -1.0],
                    "up": [0.0, 1.0, 0.0],
                    "fov": 70.0,
                    "backend": "dirichlet",
                    "pad_factor": 1,
                    "device": "cuda",
                },
                "tracer": {},
                "t0": 3.0,
                "live": {
                    "signature": "sig-2",
                    "stream_on_change_only": True,
                    "motion_sampling": "per_frame",
                },
            },
        })
        _wait_for(lambda: len(calls) == 2, "live frame after update")
        assert calls[-1]["cache_key"] == "sig-2"
    finally:
        solver_host.live_stop(ctx, {"session_id": "session-1"})


def test_solver_live_stop_suppresses_inflight_publish(monkeypatch):
    import wt_radar.solver_host as solver_host

    class FakeCtx:
        run_id = "live-run"

        def __init__(self):
            self.published = []

        def stream_publish(self, *args, **kwargs):
            self.published.append((args, kwargs))

        def log(self, message, level="info"):
            pass

        def stream_error(self, stream_id, message, **kwargs):
            pass

    entered = threading.Event()
    release = threading.Event()
    fake_signal = np.ones((1, 1, 2, 8), dtype=np.complex64)
    fake_result = type("Result", (), {"radar": object(), "signal": fake_signal})()

    def fake_run(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=3.0)
        return fake_result

    class FakeRD:
        mag_db = np.ones((4, 8), dtype=np.float32)

    monkeypatch.setattr(solver_host, "load_scene_ref", lambda ref: {"scene": ref})
    monkeypatch.setattr(solver_host.SolveRunner, "run", fake_run)
    monkeypatch.setattr(solver_host.SigProc, "range_doppler", lambda *args, **kwargs: FakeRD())

    ctx = FakeCtx()
    solver_host.live_start(ctx, {
        "session_id": "stop-session",
        "stream_id": "radar.demo.signal",
        "channels": ["rd"],
        "sceneRef": {},
        "config": {
            "sensor": {
                "position": [0.0, 0.0, 0.0],
                "target": [0.0, 0.0, -1.0],
                "up": [0.0, 1.0, 0.0],
                "fov": 70.0,
                "backend": "dirichlet",
                "pad_factor": 1,
                "device": "cuda",
            },
            "tracer": {},
            "live": {"signature": "sig-stop", "stream_on_change_only": True},
        },
    })
    assert entered.wait(timeout=3.0)
    stop_done = threading.Event()

    def stop_session():
        solver_host.live_stop(ctx, {"session_id": "stop-session"})
        stop_done.set()

    threading.Thread(target=stop_session, daemon=True).start()
    time.sleep(0.05)
    release.set()
    assert stop_done.wait(timeout=3.0)
    time.sleep(0.05)
    assert ctx.published == []


def test_solver_live_stop_does_not_trigger_extra_idle_solve(monkeypatch):
    import wt_radar.solver_host as solver_host

    class FakeCtx:
        run_id = "live-run"

        def __init__(self):
            self.published = []

        def stream_publish(self, stream_id, channel_id, payload=None, **kwargs):
            self.published.append((stream_id, channel_id))

        def log(self, message, level="info"):
            pass

        def stream_error(self, stream_id, message, **kwargs):
            pass

    calls = []
    fake_signal = np.ones((1, 1, 2, 8), dtype=np.complex64)
    fake_result = type("Result", (), {"radar": object(), "signal": fake_signal})()

    def fake_run(*args, **kwargs):
        calls.append(kwargs.get("cache_key"))
        return fake_result

    class FakeRD:
        mag_db = np.ones((4, 8), dtype=np.float32)

    monkeypatch.setattr(solver_host, "load_scene_ref", lambda ref: {"scene": ref})
    monkeypatch.setattr(solver_host.SolveRunner, "run", fake_run)
    monkeypatch.setattr(solver_host.SigProc, "range_doppler", lambda *args, **kwargs: FakeRD())

    ctx = FakeCtx()
    solver_host.live_start(ctx, {
        "session_id": "idle-stop-session",
        "stream_id": "radar.demo.signal",
        "channels": ["rd"],
        "max_fps": 30.0,
        "sceneRef": {},
        "config": {
            "sensor": {
                "position": [0.0, 0.0, 0.0],
                "target": [0.0, 0.0, -1.0],
                "up": [0.0, 1.0, 0.0],
                "fov": 70.0,
                "backend": "dirichlet",
                "pad_factor": 1,
                "device": "cuda",
            },
            "tracer": {},
            "live": {"signature": "sig-idle", "stream_on_change_only": True},
        },
    })
    _wait_for(lambda: len(calls) == 1, "initial live frame")
    time.sleep(0.08)
    solver_host.live_stop(ctx, {"session_id": "idle-stop-session"})
    time.sleep(0.05)

    assert calls == ["sig-idle"]


def test_live_wait_until_keeps_a_fine_grained_deadline_window():
    import pytest
    import wt_radar.solver_host as solver_host

    now = [0.0]

    class FakeStop:
        waits = []

        def is_set(self):
            return False

        def wait(self, timeout):
            self.waits.append(timeout)
            now[0] += timeout
            return False

    stop = FakeStop()

    def fake_sleep(_timeout):
        now[0] += 0.001

    stopped = solver_host._wait_until(
        stop,
        0.010,
        clock=lambda: now[0],
        sleep=fake_sleep,
        fine_window=0.003,
        spin_only_threshold=0.003,
    )

    assert stopped is False
    assert stop.waits == pytest.approx([0.007])
    assert now[0] >= 0.010


def test_live_wait_until_yields_during_default_short_wait():
    import wt_radar.solver_host as solver_host

    now = [0.0]

    class FakeStop:
        waits = []

        def is_set(self):
            return False

        def wait(self, timeout):
            self.waits.append(timeout)
            now[0] += timeout
            return False

    stop = FakeStop()

    stopped = solver_host._wait_until(
        stop,
        0.003,
        clock=lambda: now[0],
        fine_window=0.020,
        spin_only_threshold=0.050,
    )

    assert stopped is False
    assert stop.waits
    assert all(timeout > 0.0 for timeout in stop.waits)
    assert now[0] >= 0.003


def test_next_frame_deadline_keeps_fixed_cadence_after_oversleep():
    import pytest
    import wt_radar.solver_host as solver_host

    interval = 1.0 / 30.0
    first = solver_host._next_frame_deadline(0.0, 10.0, interval)
    second = solver_host._next_frame_deadline(first, 10.046, interval, now=10.047)

    assert first == pytest.approx(10.0 + interval)
    assert second == pytest.approx(10.0 + interval * 2)
    assert second < 10.046 + interval


def test_next_frame_deadline_does_not_catch_up_after_slow_first_frame():
    import pytest
    import wt_radar.solver_host as solver_host

    interval = 1.0 / 30.0
    deadline = solver_host._next_frame_deadline(0.0, 10.0, interval, now=13.0)

    assert deadline == pytest.approx(13.0 + interval)


def test_solver_live_restart_suppresses_previous_session_publish(monkeypatch):
    import wt_radar.solver_host as solver_host

    class FakeCtx:
        run_id = "live-run"

        def __init__(self):
            self.published = []

        def stream_publish(self, stream_id, channel_id, payload=None, **kwargs):
            self.published.append((stream_id, channel_id, kwargs.get("metadata") or {}))

        def log(self, message, level="info"):
            pass

        def stream_error(self, stream_id, message, **kwargs):
            pass

    entered = threading.Event()
    release = threading.Event()
    calls = []
    fake_signal = np.ones((1, 1, 2, 8), dtype=np.complex64)
    fake_result = type("Result", (), {"radar": object(), "signal": fake_signal})()

    def fake_run(*args, **kwargs):
        calls.append(kwargs.get("cache_key"))
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=3.0)
        return fake_result

    class FakeRD:
        mag_db = np.ones((4, 8), dtype=np.float32)

    monkeypatch.setattr(solver_host, "load_scene_ref", lambda ref: {"scene": ref})
    monkeypatch.setattr(solver_host.SolveRunner, "run", fake_run)
    monkeypatch.setattr(solver_host.SigProc, "range_doppler", lambda *args, **kwargs: FakeRD())

    ctx = FakeCtx()
    base_config = {
        "sensor": {
            "position": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, -1.0],
            "up": [0.0, 1.0, 0.0],
            "fov": 70.0,
            "backend": "dirichlet",
            "pad_factor": 1,
            "device": "cuda",
        },
        "tracer": {},
    }
    solver_host.live_start(ctx, {
        "session_id": "restart-session",
        "stream_id": "radar.demo.signal",
        "channels": ["rd"],
        "sceneRef": {},
        "config": {**base_config, "live": {"signature": "old", "stream_on_change_only": True}},
    })
    assert entered.wait(timeout=3.0)
    solver_host.live_start(ctx, {
        "session_id": "restart-session",
        "stream_id": "radar.demo.signal",
        "channels": ["rd"],
        "sceneRef": {},
        "config": {**base_config, "live": {"signature": "new", "stream_on_change_only": True}},
    })
    release.set()
    _wait_for(lambda: "new" in calls, "new live session frame")
    solver_host.live_stop(ctx, {"session_id": "restart-session"})

    assert len(ctx.published) == 1
    assert calls.count("old") == 1
    assert "new" in calls


def test_solver_live_session_publishes_completed_frame_when_update_arrives_mid_solve(monkeypatch):
    import wt_radar.solver_host as solver_host

    class FakeCtx:
        run_id = "live-run"

        def __init__(self):
            self.published = []

        def stream_publish(self, stream_id, channel_id, payload=None, **kwargs):
            self.published.append((stream_id, channel_id, kwargs.get("metadata") or {}))

        def stream_error(self, stream_id, message, **kwargs):
            pass

    fake_signal = np.ones((1, 1, 2, 8), dtype=np.complex64)
    fake_result = type("Result", (), {"radar": object(), "signal": fake_signal})()

    class FakeRD:
        mag_db = np.ones((4, 8), dtype=np.float32)

    ctx = FakeCtx()
    config = {
        "sensor": {
            "position": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, -1.0],
            "up": [0.0, 1.0, 0.0],
            "fov": 70.0,
            "backend": "dirichlet",
            "pad_factor": 1,
            "device": "cuda",
        },
        "tracer": {},
        "live": {"signature": "old", "stream_on_change_only": False},
    }
    session = solver_host.LiveSession(
        ctx,
        "mid-solve",
        scene={"scene": "old"},
        config=config,
        params={"stream_id": "radar.demo.signal", "channels": ["rd"]},
    )
    snapshot = session._snapshot()

    def fake_run(*args, **kwargs):
        updated = {
            **config,
            "live": {"signature": "new", "stream_on_change_only": False},
        }
        session.update(ctx, scene={"scene": "new"}, config=updated, params={"stream_id": "radar.demo.signal"})
        return fake_result

    monkeypatch.setattr(solver_host.SolveRunner, "run", fake_run)
    monkeypatch.setattr(solver_host.SigProc, "range_doppler", lambda *args, **kwargs: FakeRD())

    assert session._solve_and_publish(snapshot) is True
    assert len(ctx.published) == 1
    assert ctx.published[0][2]["signature"] == "old"


def test_realtime_live_update_does_not_send_elapsed_t0_to_solver(monkeypatch):
    import wt_radar.components.radar as radar_mod

    fake = _FakeApi()
    monkeypatch.setattr(radar_mod, "api", fake)
    radar, settings = _radar_in_scene()
    radar.stream_on_change_only = False
    radar.t0 = 1.5

    assert radar.start_stream() == "Stream started"
    try:
        _wait_for(lambda: len(_calls(fake, "live_start")) == 1, "live_start call")
        time.sleep(0.05)
        settings.get_component("Transform").position = [9.0, 0.0, 0.0]
        _wait_for(lambda: len(_calls(fake, "live_update")) >= 1, "live_update call")
    finally:
        radar.stop_stream()
        _wait_for(lambda: radar._live_thread is None or not radar._live_thread.is_alive(), "live thread stop")

    assert _calls(fake, "live_start")[0][2]["config"]["t0"] == 1.5
    assert _calls(fake, "live_update")[-1][2]["config"]["t0"] == 1.5
