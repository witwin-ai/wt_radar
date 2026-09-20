import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

from wt_radar.adapter import runtime_contract


def test_installed_runtime_matches_supported_release_lines():
    runtime_contract.require_supported_runtime.cache_clear()
    observed = runtime_contract.require_supported_runtime()

    assert observed == {
        distribution: version(distribution)
        for distribution in runtime_contract.SUPPORTED_RUNTIME
    }


@pytest.mark.parametrize(
    ("distribution", "reported"),
    [("witwin-radar", "0.3.0"), ("witwin-channel", "0.6.0")],
)
def test_incompatible_runtime_is_refused(monkeypatch, distribution, reported):
    installed = {"witwin-radar": "0.4.0", "witwin-channel": "0.5.0"}
    installed[distribution] = reported
    monkeypatch.setattr(runtime_contract, "version", installed.__getitem__)
    runtime_contract.require_supported_runtime.cache_clear()

    with pytest.raises(RuntimeError, match=f"{distribution} {reported} is unsupported"):
        runtime_contract.require_supported_runtime()
    runtime_contract.require_supported_runtime.cache_clear()


def test_manifest_declares_supported_simulation_packages():
    manifest = tomllib.loads((Path(__file__).resolve().parents[1] / "witwin.toml").read_text(encoding="utf-8"))

    assert "witwin-radar>=0.4,<0.5" in manifest["dependencies"]
    assert "witwin-channel>=0.5,<0.6" in manifest["dependencies"]
