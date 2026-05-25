"""R1 round-trip: full sensor config — the 4 sub-configs + derived values + validation.

Each sub-config (antenna pattern separable/map, every noise block, per-antenna +
uniform polarization, the lna/agc/adc receiver chain) round-trips through the Radar
Settings object and back into a validated ``RadarConfig``; a constructed ``Radar``
reports the derived range/doppler resolution + max range/doppler the editor shows; and
each validation rule (loc counts, adc-vs-quantization exclusivity, monotonic antenna
angles) triggers on export.
"""
import pytest
import witwin.radar as wr

from _helpers import approx, assert_radar_config_equal

_BASE = {
    "num_tx": 3, "num_rx": 4,
    "fc": 77e9, "slope": 60.012, "power": 12.0,
    "adc_samples": 256, "adc_start_time": 6.0, "sample_rate": 4400.0,
    "idle_time": 7.0, "ramp_end_time": 58.0, "chirp_per_frame": 128,
    "frame_per_second": 10.0,
    "num_doppler_bins": 128, "num_range_bins": 256, "num_angle_bins": 64,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
}


def _config(**sub):
    return wr.RadarConfig.from_dict({**_BASE, **sub})


def _roundtrip(config_map, config):
    return config_map.build_config(config_map.settings_to_studio(config))


# --- sub-config round trips ------------------------------------------------

def test_core_round_trip(config_map):
    config = _config()
    assert_radar_config_equal(config, _roundtrip(config_map, config))


def test_antenna_separable_round_trip(config_map):
    config = _config(antenna_pattern={
        "kind": "separable",
        "x_angles_deg": [-90, 0, 90], "y_angles_deg": [-90, -10, 30, 90],
        "x_values": [0.5, 1.0, 0.5], "y_values": [0.2, 0.8, 1.0, 0.3]})
    assert_radar_config_equal(config, _roundtrip(config_map, config))


def test_antenna_map_round_trip(config_map):
    config = _config(antenna_pattern={
        "kind": "map",
        "x_angles_deg": [-90, 0, 90], "y_angles_deg": [-45, 45],
        "values": [[0.1, 0.2, 0.1], [0.3, 0.4, 0.3]]})
    assert_radar_config_equal(config, _roundtrip(config_map, config))


def test_antenna_default_round_trip(config_map):
    # No antenna_pattern -> the use-default-dipole toggle round-trips to None.
    config = _config()
    rebuilt = _roundtrip(config_map, config)
    assert rebuilt.antenna_pattern is None


def test_noise_all_blocks_round_trip(config_map):
    config = _config(noise_model={
        "thermal": {"std": 0.05}, "quantization": {"bits": 10, "full_scale": 2.0},
        "phase": {"std": 0.01}, "seed": 42})
    assert_radar_config_equal(config, _roundtrip(config_map, config))


def test_polarization_round_trip(config_map):
    # Per-antenna tx (distinct vectors) + uniform rx (alias 'v' -> expanded bank).
    config = _config(polarization={
        "tx": [[1, 0, 0], [0, 1, 0], [1, 0, 0]], "rx": "v", "reflection_flip": False})
    assert_radar_config_equal(config, _roundtrip(config_map, config))


def test_receiver_chain_round_trip(config_map):
    config = _config(receiver_chain={
        "lna": {"gain_db": 20.0},
        "agc": {"target_rms": 0.5, "max_gain_db": 40.0, "min_gain_db": -20.0, "mode": "global"},
        "adc": {"bits": 12, "full_scale": 1.5},
        "reference_impedance_ohm": 75.0})
    assert_radar_config_equal(config, _roundtrip(config_map, config))


def test_combined_round_trip(config_map):
    # All four sub-configs together (noise without quantization so adc is allowed).
    config = _config(
        antenna_pattern={"kind": "separable", "x_angles_deg": [-90, 90], "y_angles_deg": [-90, 90],
                         "x_values": [1.0, 1.0], "y_values": [1.0, 1.0]},
        noise_model={"thermal": {"std": 0.02}, "phase": {"std": 0.005}},
        polarization={"tx": "h", "rx": "h"},
        receiver_chain={"adc": {"bits": 14, "full_scale": 1.0}})
    assert_radar_config_equal(config, _roundtrip(config_map, config))


# --- derived values --------------------------------------------------------

def test_derived_matches_constructed_radar(config_map):
    from wt_radar.adapter.common import Derived

    config = _config()
    settings = config_map.settings_to_studio(config)
    rebuilt = config_map.build_config(settings)
    radar = wr.Radar(rebuilt, backend="pytorch", device="cpu")
    derived = Derived.compute(settings.get_component("RadarConfig"))
    assert approx(radar.range_resolution, derived["range_resolution_m"])
    assert approx(radar.max_range, derived["max_range_m"])
    assert approx(radar.doppler_resolution, derived["doppler_resolution_mps"])
    assert approx(radar.max_doppler, derived["max_doppler_mps"])


# --- validation rules ------------------------------------------------------

def test_loc_count_validation(config_map):
    settings = config_map.settings_to_studio(_config())
    settings.get_component("RadarConfig").tx_loc = [[0, 0, 0], [4, 0, 0]]  # 2 != num_tx (3)
    with pytest.raises(ValueError):
        config_map.build_config(settings)


def test_adc_quantization_exclusive(config_map):
    settings = config_map.settings_to_studio(_config())
    settings.get_component("RadarNoiseModel").enable_quantization = True
    settings.get_component("RadarReceiverChain").enable_adc = True
    with pytest.raises(ValueError):
        config_map.build_config(settings)


def test_monotonic_antenna_angles(config_map):
    settings = config_map.settings_to_studio(_config())
    antenna = settings.get_component("RadarAntennaPattern")
    antenna.use_default = False
    antenna.pattern_kind = "separable"
    antenna.x_angles_deg = [10.0, 5.0]  # not strictly increasing
    antenna.y_angles_deg = [-90.0, 90.0]
    antenna.x_values = [1.0, 1.0]
    antenna.y_values = [1.0, 1.0]
    with pytest.raises(ValueError):
        config_map.build_config(settings)
