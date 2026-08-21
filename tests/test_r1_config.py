"""RadarConfig, antenna pattern, and typed frontend round trips."""
from dataclasses import replace

import pytest

from _helpers import assert_config_equal, make_config


def _round_trip(config_map, config):
    return config_map.build_config(config_map.settings_to_studio(config))


def test_core_config_round_trip(config_map):
    config = make_config()
    assert_config_equal(config, _round_trip(config_map, config))


@pytest.mark.parametrize(
    "pattern",
    [
        {
            "kind": "separable",
            "x_angles_deg": [-90.0, 0.0, 90.0],
            "y_angles_deg": [-90.0, 90.0],
            "x_values": [0.5, 1.0, 0.5],
            "y_values": [1.0, 1.0],
        },
        {
            "kind": "map",
            "x_angles_deg": [-90.0, 0.0, 90.0],
            "y_angles_deg": [-45.0, 45.0],
            "values": [[0.1, 0.2, 0.1], [0.3, 0.4, 0.3]],
        },
    ],
)
def test_antenna_pattern_round_trip(config_map, pattern):
    config = make_config(antenna_pattern=pattern)
    assert_config_equal(config, _round_trip(config_map, config))


def test_typed_frontend_round_trip(config_map):
    from witwin.radar.frontend import (
        AdcSpec,
        AgcSpec,
        FrontendSpec,
        LnaSpec,
        NoiseSpec,
        PortSpec,
        SeedSpec,
    )

    frontend = FrontendSpec(
        port=PortSpec(reference_impedance_ohm=75.0),
        noise=NoiseSpec(
            noise_figure_db=4.0,
            antenna_temperature_k=300.0,
            bandwidth_hz=4.4e6,
            phase_noise_dbc_per_hz=-92.0,
            phase_offset_hz=1e5,
            phase_sample_rate_hz=4.4e6,
        ),
        lna=LnaSpec(gain_db=20.0),
        agc=AgcSpec(target_rms=0.5, mode="global", min_gain_db=-20.0, max_gain_db=40.0),
        adc=AdcSpec(bits=12, full_scale=1.5),
        seed=SeedSpec(seed_base=42),
    )
    config = replace(make_config(), frontend=frontend)
    assert_config_equal(config, _round_trip(config_map, config))


def test_default_frontend_stays_absent(config_map):
    assert _round_trip(config_map, make_config()).frontend is None


def test_location_count_validation(config_map):
    settings = config_map.settings_to_studio(make_config())
    settings.get_component("Radar").tx_loc = []
    with pytest.raises(ValueError):
        config_map.build_config(settings)


def test_frontend_validation_is_preserved(config_map):
    settings = config_map.settings_to_studio(make_config())
    radar = settings.get_component("Radar")
    radar.enable_agc = True
    radar.agc_target_rms = 0.0
    with pytest.raises(ValueError):
        config_map.build_config(settings)
