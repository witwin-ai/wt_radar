import types
from dataclasses import asdict

import torch
import witwin.radar as wr

from wt_radar.adapter.solve import SensorSpec, SolveRunner, TracerSpec


def _sensor():
    return SensorSpec(
        position=[0.0, 0.0, 0.0],
        target=[0.0, 0.0, -1.0],
        up=[0.0, 1.0, 0.0],
        fov=60.0,
        backend="dirichlet",
        pad_factor=4,
        device="cuda",
    )


def test_live_cache_reuses_trace_and_path_cache(monkeypatch):
    import witwin.radar.trace as trace_mod
    import wt_radar.adapter.radar_adapter as adapter_mod

    counters = {
        "to_platform": 0,
        "build_radar": 0,
        "tracer_init": 0,
        "trace": 0,
        "path_cache": 0,
        "mimo_from_paths": 0,
    }

    class FakeAdapter:
        def to_platform(self, studio_scene, *, device="cpu"):
            counters["to_platform"] += 1
            return wr.Scene(device="cpu"), object()

    class FakeRadar:
        position = torch.tensor([0.0, 0.0, 0.0])
        target = torch.tensor([0.0, 0.0, -1.0])
        tx_pos = torch.zeros((1, 3))

        def path_cache_from_trace(self, trace, *, velocities=None):
            counters["path_cache"] += 1
            return types.SimpleNamespace(trace=trace)

        def mimo_from_paths(self, cache):
            counters["mimo_from_paths"] += 1
            return torch.full((1, 1, 1, 1), float(counters["mimo_from_paths"]))

    class FakeTracer:
        def __init__(self, *args, **kwargs):
            counters["tracer_init"] += 1

        def trace(self, time=None):
            counters["trace"] += 1
            return types.SimpleNamespace(points=torch.zeros((1, 3)), intensities=torch.ones(1))

    monkeypatch.setattr(adapter_mod, "RadarAdapter", FakeAdapter)
    monkeypatch.setattr(SolveRunner, "build_radar", staticmethod(lambda config, sensor: counters.__setitem__("build_radar", counters["build_radar"] + 1) or FakeRadar()))
    monkeypatch.setattr(trace_mod, "Tracer", FakeTracer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    cache = {}
    tracer = TracerSpec(resolution=16)

    first = SolveRunner.run("studio", sensor=_sensor(), tracer=tracer, motion_sampling="per_frame",
                            t0=0.0, live_cache=cache, cache_key="same")
    second = SolveRunner.run("studio", sensor=_sensor(), tracer=tracer, motion_sampling="per_frame",
                             t0=1.0, live_cache=cache, cache_key="same")

    assert first.signal.item() == 1.0
    assert second.signal.item() == 2.0
    assert counters["to_platform"] == 1
    assert counters["build_radar"] == 1
    assert counters["tracer_init"] == 1
    assert counters["trace"] == 1
    assert counters["path_cache"] == 1
    assert counters["mimo_from_paths"] == 2


def test_live_cache_invalidates_when_signature_changes(monkeypatch):
    import witwin.radar.trace as trace_mod
    import wt_radar.adapter.radar_adapter as adapter_mod

    counters = {"to_platform": 0, "tracer_init": 0, "trace": 0}

    class FakeAdapter:
        def to_platform(self, studio_scene, *, device="cpu"):
            counters["to_platform"] += 1
            return wr.Scene(device="cpu"), object()

    class FakeRadar:
        position = torch.tensor([0.0, 0.0, 0.0])
        target = torch.tensor([0.0, 0.0, -1.0])
        tx_pos = torch.zeros((1, 3))

        def path_cache_from_trace(self, trace, *, velocities=None):
            return object()

        def mimo_from_paths(self, cache):
            return torch.zeros((1, 1, 1, 1))

    class FakeTracer:
        def __init__(self, *args, **kwargs):
            counters["tracer_init"] += 1

        def trace(self, time=None):
            counters["trace"] += 1
            return types.SimpleNamespace(points=torch.zeros((1, 3)), intensities=torch.ones(1))

    monkeypatch.setattr(adapter_mod, "RadarAdapter", FakeAdapter)
    monkeypatch.setattr(SolveRunner, "build_radar", staticmethod(lambda config, sensor: FakeRadar()))
    monkeypatch.setattr(trace_mod, "Tracer", FakeTracer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    cache = {}
    tracer = TracerSpec(resolution=16)
    SolveRunner.run("studio", sensor=_sensor(), tracer=tracer, motion_sampling="per_frame",
                    t0=0.0, live_cache=cache, cache_key="a")
    SolveRunner.run("studio", sensor=_sensor(), tracer=tracer, motion_sampling="per_frame",
                    t0=0.0, live_cache=cache, cache_key="b")

    assert counters["to_platform"] == 2
    assert counters["tracer_init"] == 1
    assert counters["trace"] == 2


def test_solver_host_reuses_live_session_cache(monkeypatch):
    import solver_host

    class FakeCtx:
        def log(self, message):
            pass

        def progress(self, value, message):
            pass

    calls = []

    def fake_run(studio_scene, **kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(radar=object(), signal=torch.zeros((1, 1, 1, 1)))

    solver_host._LIVE_CACHES.clear()
    monkeypatch.setattr(solver_host.SolveRunner, "run", staticmethod(fake_run))
    config = {
        "sensor": asdict(_sensor()),
        "tracer": asdict(TracerSpec(resolution=16)),
        "motion_sampling": "per_frame",
        "t0": 0.0,
        "live": {"session_id": "radar.test.signal", "signature": "same"},
    }

    solver_host.solve(FakeCtx(), "scene", config)
    solver_host.solve(FakeCtx(), "scene", config)

    assert calls[0]["cache_key"] == "same"
    assert calls[1]["cache_key"] == "same"
    assert calls[0]["live_cache"] is calls[1]["live_cache"]
