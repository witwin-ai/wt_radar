"""Independent multi-radar execution and post-processor defaults."""
from dataclasses import replace
from types import SimpleNamespace

import torch

from wt_radar.adapter.solve import PropagationSpec, SensorSpec, SolveRunner


def _sensor():
    return SensorSpec([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], 60.0)


def test_run_group_uses_independent_current_solves(monkeypatch):
    calls = []

    def fake_run(scene, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(signal=torch.full((1, 1, 2, 4), float(len(calls))))

    monkeypatch.setattr(SolveRunner, "run", staticmethod(fake_run))
    base = _sensor()
    signals = SolveRunner.run_group(
        "scene",
        sensors={"front": base, "side": replace(base, position=[1.0, 0.0, 0.0])},
        propagation=PropagationSpec(),
        t0=0.0,
    )

    assert set(signals) == {"front", "side"}
    assert len(calls) == 2
    assert calls[0]["sensor"].position == [0.0, 0.0, 0.0]
    assert calls[1]["sensor"].position == [1.0, 0.0, 0.0]


def test_post_processors_keep_static_returns_visible_by_default():
    from wt_radar.components.post_processing import (
        PointCloudProcessorComponent,
        RangeDopplerProcessorComponent,
    )

    assert bool(RangeDopplerProcessorComponent().static_clutter_removal) is False
    assert bool(PointCloudProcessorComponent().static_clutter_removal) is False
