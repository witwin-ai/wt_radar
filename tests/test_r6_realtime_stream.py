import time

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


class _FakeRun:
    status = "succeeded"
    error = None

    def __init__(self, index):
        self.outputs = {"resultHandle": f"handle-{index}"}
        self.run_id = f"run-{index}"


class _FakeSolvers:
    def __init__(self):
        self.solve_calls = []
        self.query_calls = []

    def solve(self, solver_id, **kwargs):
        self.solve_calls.append((solver_id, kwargs))
        return _FakeRun(len(self.solve_calls))

    def query(self, solver_id, result_handle, op, params, **kwargs):
        self.query_calls.append((op, result_handle, params, kwargs))
        if op == "raw_signal":
            return {"data": {"tx": 0, "rx": 0, "real": [1.0, 0.0], "imag": [0.0, 1.0]}}
        if op == "range_doppler":
            return {"data": {"mag_db": [[1.0, 2.0], [3.0, 4.0]]}}
        if op == "point_cloud":
            return {"data": {"points": [[1.0, 2.0, 3.0, 0.0, -10.0, 3.7]]}}
        raise AssertionError(op)


class _FakeStream:
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.published = []
        self.statuses = []
        self.refs = []

    def update(self, status=None, **kwargs):
        self.statuses.append(status)

    def ref(self, channel):
        ref = {"streamId": self.stream_id, "channelId": channel}
        self.refs.append(ref)
        return ref

    def publish(self, channel, payload, metadata=None):
        self.published.append((channel, tuple(np.asarray(payload).shape), metadata or {}))


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

    def error(self, stream_id, message):
        self.errors.append((stream_id, message))


class _FakeApi:
    def __init__(self):
        self.solvers = _FakeSolvers()
        self.streams = _FakeStreams()


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


def test_realtime_stream_republishes_when_radar_transform_changes(monkeypatch):
    import wt_radar.components.radar as radar_mod

    fake = _FakeApi()
    monkeypatch.setattr(radar_mod, "api", fake)
    radar, settings = _radar_in_scene()
    radar.stream_max_fps = 20.0
    radar.stream_on_change_only = True
    radar.stream_channels = "raw,rd,pc"

    assert radar.start_stream() == "Stream started"
    try:
        _wait_for(lambda: len(fake.solvers.solve_calls) >= 1, "initial realtime solve")
        stream = next(iter(fake.streams.streams.values()))
        _wait_for(
            lambda: {entry[0] for entry in stream.published} >= {"raw", "rd", "pc"},
            "initial raw/rd/pc stream publish",
        )
        first_count = len(fake.solvers.solve_calls)

        settings.get_component("Transform").position = [1.0, 2.0, 3.0]
        _wait_for(lambda: len(fake.solvers.solve_calls) >= first_count + 1, "solve after transform move")

        assert radar.pause_stream() == "Stream paused"
        paused_count = len(fake.solvers.solve_calls)
        settings.get_component("Transform").position = [2.0, 2.0, 3.0]
        time.sleep(0.2)
        assert len(fake.solvers.solve_calls) == paused_count

        assert radar.start_stream() == "Stream running"
        settings.get_component("Transform").position = [3.0, 2.0, 3.0]
        _wait_for(lambda: len(fake.solvers.solve_calls) >= paused_count + 1, "solve after resume")
    finally:
        radar.stop_stream()
        _wait_for(lambda: radar._live_thread is None or not radar._live_thread.is_alive(), "live thread stop")

    assert radar.stream_status == "stopped"
    assert radar.signal_stream is None
    assert fake.streams.closed[-1] == ("radar.radar_settings.signal", "stopped")
    assert fake.streams.errors == []
    assert fake.solvers.solve_calls[-1][1]["config"]["sensor"]["position"] == [3.0, 2.0, 3.0]


def test_realtime_stream_continuous_mode_advances_t0(monkeypatch):
    import wt_radar.components.radar as radar_mod

    fake = _FakeApi()
    monkeypatch.setattr(radar_mod, "api", fake)
    radar, _settings = _radar_in_scene()
    radar.stream_max_fps = 20.0
    radar.stream_on_change_only = False
    radar.stream_channels = "rd"
    radar.t0 = 1.5

    assert radar.start_stream() == "Stream started"
    try:
        _wait_for(lambda: len(fake.solvers.solve_calls) >= 2, "continuous realtime solves")
    finally:
        radar.stop_stream()
        _wait_for(lambda: radar._live_thread is None or not radar._live_thread.is_alive(), "live thread stop")

    t0_values = [call[1]["config"]["t0"] for call in fake.solvers.solve_calls]
    assert len(t0_values) >= 2
    assert t0_values[0] >= 1.5
    assert t0_values[-1] > t0_values[0]
