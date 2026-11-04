"""The plugin must not silently fall back to the removed Radar 0.3 API."""
from pathlib import Path
import tomllib

import pytest

from wt_radar import solver_host

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def test_package_does_not_register_radar03_platform_adapter():
    # The adapter registry is a Studio internal; register_platform_adapter is
    # the only public way in, so the package must never call it.
    callers = [
        path.relative_to(PLUGIN_ROOT).as_posix()
        for path in PLUGIN_ROOT.rglob("*.py")
        if "tests" not in path.relative_to(PLUGIN_ROOT).parts
        and "register_platform_adapter" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert callers == []


def test_solver_rejects_non_radar04_requests():
    with pytest.raises(RuntimeError, match="legacy Radar 0.3"):
        solver_host.solve(object(), object(), {"adapter": "unknown"})


def test_bare_solve_snapshots_the_scene_radar(monkeypatch):
    """An empty config (witwin solve with no config) runs the scene's first Radar's snapshot."""
    from types import SimpleNamespace
    from wt_radar.adapter import snapshot

    radar = object()
    scene = SimpleNamespace(objects={
        "box": SimpleNamespace(get_component=lambda name: None),
        "radar": SimpleNamespace(get_component=lambda name: radar if name == "Radar" else None),
    })
    monkeypatch.setattr(snapshot, "snapshot_request", lambda component: {"adapter": "picked", "component": component})
    assert solver_host._snapshot_request_from_scene(scene) == {"adapter": "picked", "component": radar}
    with pytest.raises(RuntimeError, match="no Radar"):
        solver_host.solve(object(), SimpleNamespace(objects={}), {})


def test_solver_host_does_not_publish_radar03_live_session_methods():
    for name in ("live_start", "live_update", "live_pause", "live_resume", "live_stop", "live_status"):
        assert not hasattr(solver_host, name)


def test_solver_host_does_not_publish_radar03_signal_queries():
    for name in ("raw_signal", "range_doppler", "point_cloud", "music"):
        assert not hasattr(solver_host, name)

    manifest = tomllib.loads((Path(__file__).parents[1] / "witwin.toml").read_text())
    advertised = set(manifest["contributes"]["solvers"][0]["capabilities"]["query"])
    assert not advertised.intersection({"raw_signal", "range_doppler", "point_cloud", "music"})
