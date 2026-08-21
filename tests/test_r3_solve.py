"""Radar 0.3 scene-driven solve and typed processing coverage."""
from pathlib import Path
import tomllib

import pytest
from witwin_server import Scene

from _helpers import make_config
from wt_radar.adapter.config_map import ConfigMap
from wt_radar.adapter.solve import PropagationSpec, SensorSpec, SigProc, SolveRunner
from wt_radar.library_items import _target_object


def _studio(adapter):
    studio = Scene()
    studio.add_object(ConfigMap.settings_to_studio(make_config()))
    studio.add_object(_target_object())
    return studio


def _settings(studio):
    return next(obj for obj in studio.objects.values() if obj.get_component("Radar") is not None)


def test_sensor_pose_follows_settings_transform(adapter):
    studio = _studio(adapter)
    settings = _settings(studio)
    settings.get_component("Transform").position = [1.5, -2.0, 0.5]
    spec = SensorSpec.from_component(settings.get_component("Radar"))
    assert spec.position == pytest.approx([1.5, -2.0, 0.5])
    assert spec.target == pytest.approx([1.5, -2.0, -0.5])
    assert spec.up == pytest.approx([0.0, 1.0, 0.0])
    assert spec.device == "cuda"


def test_propagation_is_explicit_and_current(adapter):
    radar = _settings(_studio(adapter)).get_component("Radar")
    radar.target_rcs_m2 = 2.5
    radar.multipath = True
    radar.max_reflections = 2
    radar.polarization = [1.0, 0.0, 0.0]
    spec = PropagationSpec.from_component(radar)
    assert spec.target_rcs_m2 == 2.5
    assert spec.components == frozenset({"los", "reflection"})
    assert spec.max_depth == 2
    assert spec.polarization == (1.0, 0.0, 0.0)


@pytest.mark.gpu
def test_latest_radar_solve_and_processing(adapter, cuda_ready):
    studio = _studio(adapter)
    component = _settings(studio).get_component("Radar")
    result = SolveRunner.run(
        studio,
        sensor=SensorSpec.from_component(component),
        propagation=PropagationSpec.from_component(component),
        t0=0.0,
    )

    assert tuple(result.signal.shape) == (1, 2, 8, 16)
    assert tuple(result.simulation.cube.shape) == (1, 1, 2, 8, 16)
    view = SigProc.range_doppler(
        result.radar,
        result.signal,
        tx=0,
        rx=0,
        static_clutter_removal=False,
    )
    assert view.mag_db.shape == (8, 16)
    assert view.ranges.shape == (16,)
    assert view.velocities.shape == (8,)
    mask = SigProc.cfar_mask(view.rd_map, guard=(0, 0), training=(1, 1), pfa=1e-3)
    assert mask.shape == view.mag_db.shape



@pytest.mark.gpu
def test_radial_site_velocity_reaches_two_way_delay_rate(adapter, cuda_ready):
    import torch

    studio = _studio(adapter)
    component = _settings(studio).get_component("Radar")
    result = SolveRunner.run(
        studio,
        sensor=SensorSpec.from_component(component),
        propagation=PropagationSpec.from_component(component),
        t0=0.0,
        site_positions=torch.tensor([[0.0, 0.0, -2.5]], device="cuda"),
        site_velocities=torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
        site_stable_ids=[3_000_000],
    )

    delay_rate = result.radar.last_radar_paths.delay_rate

    assert delay_rate is not None
    assert tuple(delay_rate.shape) == (2,)
    assert bool(torch.all(delay_rate.abs() > 6.0e-9))
def test_manifest_declares_dedicated_runtime():
    manifest = tomllib.loads((Path(__file__).parents[1] / "witwin.toml").read_text(encoding="utf-8"))
    solver = manifest["contributes"]["solvers"][0]
    assert solver["runtime"]["conda"] == "witwin_runtime"
    assert manifest["version"] == "0.3.0"
