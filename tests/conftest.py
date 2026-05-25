"""Pytest harness for the wt_radar round-trip suite.

Runs against the real environment: ``witwin_server`` (with the frozen
``platform_bridge``) and the radar solver (``witwin.radar`` / ``witwin.core``, which
pull in drjit + mitsuba) must be importable. The conftest adds the plugins directory so
``importlib.import_module("wt_radar")`` resolves (the plugin itself is not pip-installed;
the server adds the same directory at startup). ``witwin.radar.__init__`` eagerly
initializes the mitsuba CUDA variant, so if the radar stack is not importable the whole
suite is skipped rather than erroring (mark as needs-user-verification).
"""
import importlib
import sys
from pathlib import Path

import pytest

# .../plugins/wt_radar/tests -> .../plugins
PLUGINS_DIR = Path(__file__).resolve().parents[2]
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))

PKG = "wt_radar"

try:
    import witwin_server  # noqa: F401
    import witwin.radar  # noqa: F401
    _DEPS_OK = True
except Exception:  # noqa: BLE001 - radar stack (mitsuba/drjit/CUDA) may be absent
    _DEPS_OK = False

# Skip collection of every test module when the base/solver deps are absent.
if not _DEPS_OK:
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture(scope="session")
def wtr():
    """Import the plugin package (registers components + the radar adapter)."""
    return importlib.import_module(PKG)


@pytest.fixture(scope="session")
def adapter(wtr):
    """A fresh :class:`RadarAdapter` instance."""
    return importlib.import_module(f"{PKG}.adapter.radar_adapter").RadarAdapter()


@pytest.fixture(scope="session")
def device():
    """'cuda' when available, else 'cpu'."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def cuda_ready():
    """Skip a test when no CUDA device is present (dirichlet/slang backends need it)."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("radar dirichlet/slang solve requires a CUDA device")
    return True
