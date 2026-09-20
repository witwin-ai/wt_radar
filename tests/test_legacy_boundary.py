"""The plugin must not silently fall back to the removed Radar 0.3 API."""
import pytest

from witwin_server.features.platform.bridge.registry import AdapterRegistry
from wt_radar import solver_host


def test_package_does_not_register_radar03_platform_adapter():
    assert not AdapterRegistry.has("radar")


@pytest.mark.parametrize("config", [{}, {"adapter": "unknown"}])
def test_solver_rejects_non_radar04_requests(config):
    with pytest.raises(RuntimeError, match="legacy Radar 0.3"):
        solver_host.solve(object(), object(), config)


def test_solver_host_does_not_publish_radar03_live_session_methods():
    for name in ("live_start", "live_update", "live_pause", "live_resume", "live_stop", "live_status"):
        assert not hasattr(solver_host, name)
