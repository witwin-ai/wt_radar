"""Multi-frame generation uses repeated Radar 0.3 scene solves."""
from types import SimpleNamespace

import torch

from wt_radar.adapter.solve import PropagationSpec, SensorSpec
from wt_radar.adapter.timeline_run import TimelineRunner


def _component(source: str, count: int = 3):
    return SimpleNamespace(
        frame_rate=10.0,
        timeline_source=source,
        pointcloud_path="unused.npz",
        motion_frame_count=count,
    )


def _sensor():
    return SensorSpec([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], 60.0)


def test_scene_motion_runs_one_solve_per_frame(monkeypatch):
    calls = []

    def fake_run(scene, **kwargs):
        calls.append(kwargs)
        index = len(calls)
        return SimpleNamespace(radar="radar", signal=torch.full((1, 1, 2, 4), float(index)))

    monkeypatch.setattr("wt_radar.adapter.timeline_run.SolveRunner.run", fake_run)
    radar, frames = TimelineRunner.generate(
        _component("scene_motion", 3),
        sensor=_sensor(),
        propagation=PropagationSpec(),
        studio_scene="scene",
    )

    assert radar == "radar"
    assert tuple(frames.shape) == (3, 1, 1, 2, 4)
    assert [call["t0"] for call in calls] == [0.0, 0.1, 0.2]
    assert all(call["platform_cache_key"] == "timeline:scene" for call in calls)


def test_pointcloud_sequence_passes_explicit_sites(monkeypatch):
    points = torch.tensor([[[0.0, 0.0, -2.0]], [[0.1, 0.0, -2.0]]])
    calls = []
    monkeypatch.setattr(
        TimelineRunner,
        "_load_pointclouds",
        staticmethod(lambda path, device: (points, None)),
    )

    def fake_run(scene, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(radar="radar", signal=torch.zeros((1, 1, 2, 4)))

    monkeypatch.setattr("wt_radar.adapter.timeline_run.SolveRunner.run", fake_run)
    _, frames = TimelineRunner.generate(
        _component("pointcloud_sequence"),
        sensor=_sensor(),
        propagation=PropagationSpec(),
        studio_scene="scene",
    )
    assert tuple(frames.shape) == (2, 1, 1, 2, 4)
    assert torch.equal(calls[0]["site_positions"], points[0])
    assert torch.equal(calls[1]["site_positions"], points[1])
