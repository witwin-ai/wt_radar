from types import SimpleNamespace

import pytest
import torch


def test_direct_environment_composer_is_owned_by_wt_radar():
    from wt_radar.adapter.direct_environment import DirectEnvironmentComposer

    assert DirectEnvironmentComposer.__module__ == "wt_radar.adapter.direct_environment"


def test_environment_adapter_does_not_require_radar_direct_composer():
    import inspect
    from wt_radar.adapter import environment_clutter

    source = inspect.getsource(environment_clutter)
    assert "witwin.radar.paths import DirectComposer" not in source
    assert "RadarComponentIndex.from_direct" not in source


def test_local_composer_orders_channel_rows_by_radar_sensor_pair():
    from wt_radar.adapter.direct_environment import DirectEnvironmentComposer

    frozen = SimpleNamespace(
        source_id=torch.tensor([11, 10, 11, 10], dtype=torch.int64),
        sink_id=torch.tensor([21, 20, 20, 21], dtype=torch.int64),
        component_id=torch.tensor([1, 1, 1, 1], dtype=torch.int64),
        depth=torch.tensor([1, 1, 1, 1], dtype=torch.int64),
        primitive_sequence=torch.tensor([[4], [1], [2], [3]], dtype=torch.int64),
        material_sequence=torch.tensor([[0], [0], [0], [0]], dtype=torch.int64),
    )
    live = SimpleNamespace(
        delay_s=torch.tensor([4.0, 1.0, 2.0, 3.0], dtype=torch.float32),
        coefficient=torch.tensor([4j, 1j, 2j, 3j], dtype=torch.complex64),
        row_valid=torch.tensor([True, True, False, True]),
    )
    composer = DirectEnvironmentComposer.freeze(
        frozen,
        radar_source_ids=(10, 11),
        radar_sink_ids=(20, 21),
        reference_frequency_hz=77e9,
    )
    paths = composer.compose(live)

    assert composer.site_count == 0
    assert composer.row_index.tolist() == [1, 2, 3, 0]
    assert paths.sensor_pair_index.tolist() == [0, 1, 2, 3]
    assert paths.pair_offsets.tolist() == [0, 1, 2, 3, 4]
    assert paths.total_delay_s.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert paths.row_valid.tolist() == [True, False, True, True]


def _wall_scene():
    import witwin.core as wc

    mesh = wc.Mesh(
        torch.tensor(
            [
                [4.0, -2.0, -2.0],
                [4.0, 2.0, -2.0],
                [4.0, 2.0, 2.0],
                [4.0, -2.0, 2.0],
            ],
            dtype=torch.float32,
        ),
        torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    wall = wc.Structure(
        geometry=mesh,
        material=wc.PhysicalMaterial(name="concrete", eps_r=5.24, sigma_e=0.0462),
        structure_id=1,
        material_id=1,
        assignment_id=1,
        surface_id=1,
    )
    return wc.Scene(structures=(wall,))


def _radar():
    from witwin.radar import Radar

    config = {
        "num_tx": 1,
        "num_rx": 1,
        "fc": 77.0e9,
        "slope": 60.012,
        "adc_samples": 64,
        "adc_start_time": 6,
        "sample_rate": 4400,
        "idle_time": 7,
        "ramp_end_time": 58,
        "chirp_per_frame": 8,
        "power": 12,
        "tx_loc": [[0, 0, 0]],
        "rx_loc": [[0, 0, 0]],
    }
    return Radar.from_dict(
        config,
        device="cuda",
        position=(0, 0, 0),
        look_at=(1, 0, 0),
        up=(0, 0, 1),
        polarization=(0, 0, 1),
    )


def test_single_bounce_environment_uses_native_reflection_and_no_leakage(cuda_ready):
    from wt_radar.adapter.environment_clutter import single_bounce_environment_cube

    result = single_bounce_environment_cube(
        _radar(), _wall_scene(), polarization=(0.0, 0.0, 1.0),
    )

    assert result.max_depth == 1
    assert result.reflected_path_count == 1
    assert result.material_slot_count == 1
    assert result.cube.shape == (1, 1, 8, 64)
    assert result.cube.device.type == "cuda"
    assert torch.isfinite(result.cube).all()
    assert torch.count_nonzero(result.cube) > 0


def test_single_bounce_policy_is_reflection_only():
    from wt_radar.adapter.environment_clutter import SINGLE_BOUNCE_COMPONENTS

    assert SINGLE_BOUNCE_COMPONENTS == frozenset({"reflection"})
    assert "los" not in SINGLE_BOUNCE_COMPONENTS


def test_environment_cube_is_added_in_native_fmcw_result_domain(cuda_ready):
    from wt_radar.adapter.environment_clutter import (
        add_environment_to_fmcw_result,
        single_bounce_environment_cube,
    )
    radar = _radar()
    environment = single_bounce_environment_cube(
        radar, _wall_scene(), polarization=(0.0, 0.0, 1.0),
    )
    target = torch.zeros_like(environment.cube)

    combined = add_environment_to_fmcw_result(radar, target, environment)

    torch.testing.assert_close(combined, environment.cube)
    assert torch.count_nonzero(combined) > 0
