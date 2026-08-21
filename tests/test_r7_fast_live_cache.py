"""Live cache behavior after removal of legacy trace/path caches."""
from dataclasses import asdict
from types import SimpleNamespace

import torch

from _helpers import make_config
from wt_radar.adapter.solve import PropagationSpec, SensorSpec, SolveRunner


def _sensor(position=None):
    return SensorSpec(
        position or [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        60.0,
    )


def test_live_cache_reuses_export_and_radar(monkeypatch):
    import wt_radar.adapter.radar_adapter as adapter_mod

    counters = {"export": 0, "radar": 0, "pose": 0, "simulate": 0}

    class FakeExport:
        config = make_config()
        scene = object()

        @staticmethod
        def positions_at(time_s, *, device):
            return torch.tensor([[0.0, 0.0, -2.0]], device=device)

    class FakeAdapter:
        def build_export(self, scene):
            counters["export"] += 1
            return FakeExport()

    class FakeRadar:
        device = "cpu"
        config = FakeExport.config

        def set_pose(self, **kwargs):
            counters["pose"] += 1

        def simulate(self, scene, **kwargs):
            counters["simulate"] += 1
            return SimpleNamespace(cube=torch.full((1, 1, 1, 2, 4), counters["simulate"]))

    monkeypatch.setattr(adapter_mod, "RadarAdapter", FakeAdapter)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        SolveRunner,
        "build_radar",
        staticmethod(lambda config, sensor: counters.__setitem__("radar", counters["radar"] + 1) or FakeRadar()),
    )
    monkeypatch.setattr(SolveRunner, "processing_axes", staticmethod(lambda radar: "axes"))

    cache = {}
    first = SolveRunner.run(
        "scene",
        sensor=_sensor(),
        propagation=PropagationSpec(),
        t0=0.0,
        live_cache=cache,
        cache_key="frame-a",
        platform_cache_key="scene-a",
    )
    second = SolveRunner.run(
        "scene",
        sensor=_sensor([1.0, 0.0, 0.0]),
        propagation=PropagationSpec(),
        t0=0.1,
        live_cache=cache,
        cache_key="frame-b",
        platform_cache_key="scene-a",
    )

    assert first.signal.flatten()[0].item() == 1
    assert second.signal.flatten()[0].item() == 2
    assert counters == {"export": 1, "radar": 1, "pose": 1, "simulate": 2}
    assert "trace" not in cache and "path_cache" not in cache


def test_solver_host_reuses_session_cache(monkeypatch):
    import solver_host

    class Context:
        def log(self, message):
            pass

        def progress(self, value, message):
            pass

    calls = []

    def fake_run(scene, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(signal=torch.zeros((1, 1, 2, 4)))

    monkeypatch.setattr(solver_host.SolveRunner, "run", staticmethod(fake_run))
    solver_host._LIVE_CACHES.clear()
    config = {
        "sensor": asdict(_sensor()),
        "propagation": asdict(PropagationSpec()),
        "t0": 0.0,
        "live": {
            "session_id": "radar.test",
            "signature": "settings-a",
            "scene_payload_signature": "scene-a",
        },
    }

    solver_host.solve(Context(), "scene", config)
    solver_host.solve(Context(), "scene", config)

    assert calls[0]["live_cache"] is calls[1]["live_cache"]
    assert calls[0]["cache_key"] == "settings-a"
    assert calls[0]["platform_cache_key"] == "scene-a"
