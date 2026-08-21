"""Radar bake sequence kinematics and channel emission tests."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import wt_radar.solver_host as solver_host


class _Context:
    def __init__(self):
        self.frames = []
        self.logs = []
        self.progress_updates = []

    def log(self, message):
        self.logs.append(str(message))

    def throw_if_cancelled(self):
        return None

    def frame(self, time_s, channel, value, metadata=None):
        self.frames.append(
            (float(time_s), str(channel), np.asarray(value), dict(metadata or {}))
        )

    def progress(self, value, message):
        self.progress_updates.append((float(value), str(message)))


def _settings():
    return {
        "sensor": SimpleNamespace(device="cuda"),
        "propagation": object(),
        "tx": 0,
        "rx": 0,
        "static_clutter_removal": False,
        "velocity_corrected": True,
        "detector": "topk",
        "guard": (2, 4),
        "training": (4, 8),
        "pfa": 1e-3,
        "energy_top_k": 8,
    }


def test_bake_uses_one_velocity_corrected_solve_for_all_selected_channels(monkeypatch):
    scenes = {"a": object(), "b": object()}
    monkeypatch.setattr(solver_host, "load_scene_ref", lambda ref: scenes[ref])
    monkeypatch.setattr(solver_host, "_bake_settings", lambda config, scene: _settings())

    positions = np.asarray(
        [
            [[0.0, 0.0, -2.0], [0.2, 0.0, -2.0]],
            [[0.1, 0.0, -2.0], [0.3, 0.0, -2.0]],
        ],
        dtype=np.float32,
    )
    velocities = np.full_like(positions, (1.0, 0.0, 0.0))
    stable_ids = np.asarray([3_000_000, 3_000_001], dtype=np.int64)
    sequence = SimpleNamespace(
        positions_m=positions,
        velocities_m_per_s=velocities,
        stable_ids=stable_ids,
        frame_count=2,
        site_count=2,
    )
    captured_sequence = {}

    def fake_kinematics(sampled_scenes, times):
        captured_sequence["scenes"] = tuple(sampled_scenes)
        captured_sequence["times"] = tuple(times)
        return sequence

    monkeypatch.setattr(solver_host, "scene_scatterer_kinematics", fake_kinematics)

    solve_calls = []

    def fake_run(scene, **kwargs):
        solve_calls.append((scene, kwargs))
        signal = torch.ones((1, 1, 2, 4), dtype=torch.complex64)
        return SimpleNamespace(radar=object(), signal=signal)

    monkeypatch.setattr(solver_host.SolveRunner, "run", staticmethod(fake_run))
    monkeypatch.setattr(
        solver_host.SigProc,
        "range_doppler",
        staticmethod(lambda *args, **kwargs: SimpleNamespace(mag_db=np.ones((2, 4), dtype=np.float32))),
    )
    monkeypatch.setattr(
        solver_host.SigProc,
        "point_cloud",
        staticmethod(lambda *args, **kwargs: np.ones((3, 6), dtype=np.float32)),
    )

    ctx = _Context()
    result = solver_host.bake_sequence(
        ctx,
        {
            "frames": [
                {"time": 0.0, "sceneRef": "a"},
                {"time": 0.1, "sceneRef": "b"},
            ],
            "channels": ["raw", "rd", "pc"],
        },
    )

    assert result == {
        "frames": 2,
        "channels": ["raw", "rd", "pc"],
        "siteCount": 2,
    }
    assert captured_sequence == {"scenes": (scenes["a"], scenes["b"]), "times": (0.0, 0.1)}
    assert len(solve_calls) == 2
    for index, (_, kwargs) in enumerate(solve_calls):
        np.testing.assert_array_equal(kwargs["site_positions"], positions[index])
        np.testing.assert_array_equal(kwargs["site_velocities"], velocities[index])
        np.testing.assert_array_equal(kwargs["site_stable_ids"], stable_ids)

    assert [item[1] for item in ctx.frames] == ["raw", "rd", "pc", "raw", "rd", "pc"]
    assert ctx.frames[0][3]["semantic"] == "radar.raw"
    assert ctx.frames[1][3]["semantic"] == "radar.range_doppler"
    assert ctx.frames[2][3]["semantic"] == "pointcloud_xyz"
    assert ctx.frames[0][2].dtype == np.complex64
    assert ctx.frames[2][2].shape == (3, 6)


def test_bake_defaults_to_range_doppler_and_can_disable_velocity(monkeypatch):
    scene = object()
    monkeypatch.setattr(solver_host, "load_scene_ref", lambda ref: scene)
    settings = _settings()
    settings["velocity_corrected"] = False
    monkeypatch.setattr(solver_host, "_bake_settings", lambda config, value: settings)
    monkeypatch.setattr(
        solver_host,
        "scene_scatterer_kinematics",
        lambda scenes, times: SimpleNamespace(
            positions_m=np.zeros((1, 1, 3), dtype=np.float32),
            velocities_m_per_s=np.ones((1, 1, 3), dtype=np.float32),
            stable_ids=np.asarray([3_000_000], dtype=np.int64),
            frame_count=1,
            site_count=1,
        ),
    )
    calls = []

    def fake_run(value, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            radar=object(),
            signal=torch.ones((1, 1, 2, 4), dtype=torch.complex64),
        )

    monkeypatch.setattr(solver_host.SolveRunner, "run", staticmethod(fake_run))
    monkeypatch.setattr(
        solver_host.SigProc,
        "range_doppler",
        staticmethod(lambda *args, **kwargs: SimpleNamespace(mag_db=np.ones((2, 4), dtype=np.float32))),
    )

    ctx = _Context()
    result = solver_host.bake_sequence(
        ctx,
        {"frames": [{"time": 0.0, "sceneRef": "one"}]},
    )

    assert result["channels"] == ["rd"]
    assert calls[0]["site_velocities"] is None
    assert [item[1] for item in ctx.frames] == ["rd"]
