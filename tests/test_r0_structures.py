"""R0 round-trip: scaffolding + structures + the (Scene, RadarConfig) pair contract.

Builds a radar ``Scene`` (three structures with bsdf/dynamic metadata) paired with a
``RadarConfig``, runs the pair through ``to_studio`` then ``to_platform``, and asserts
the rebuilt pair reproduces every ``RadarConfig`` field (incl. tx_loc/rx_loc and the
non-SI unit conventions) and every structure (geometry + scalar material + bsdf/dynamic
metadata).
"""
import witwin.radar as wr

from _helpers import assert_radar_config_equal, assert_structures_equal

_CONFIG = {
    "num_tx": 3, "num_rx": 4,
    "fc": 77e9, "slope": 60.012, "power": 12.0,
    "adc_samples": 256, "adc_start_time": 6.0, "sample_rate": 4400.0,
    "idle_time": 7.0, "ramp_end_time": 58.0, "chirp_per_frame": 128,
    "frame_per_second": 10.0,
    "num_doppler_bins": 128, "num_range_bins": 256, "num_angle_bins": 64,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
}


def _build_pair():
    # A radar (Scene, RadarConfig) pair with three structures + radar metadata.
    scene = wr.Scene(device="cpu")
    scene.add_mesh(
        name="wall",
        geometry=wr.Box(position=(0.0, 0.0, -4.0), size=(2.0, 2.0, 0.1)),
        material=wr.Material(eps_r=5.0, name="concrete"),
    )
    scene.add_mesh(
        name="car",
        geometry=wr.Box(position=(0.6, 0.0, -2.5), size=(0.6, 0.4, 0.4)),
        material=wr.Material(eps_r=8.0, sigma_e=0.02, name="metal"),
        bsdf={"type": "conductor", "material": "Cu"},
        dynamic=True,
    )
    scene.add_mesh(
        name="ball",
        geometry=wr.Sphere(position=(-1.0, 0.0, -2.0), radius=0.3),
        material=wr.Material(eps_r=2.0, name="plastic"),
    )
    config = wr.RadarConfig.from_dict(_CONFIG)
    return scene, config


def test_detect_claims_radar(adapter):
    scene, config = _build_pair()
    assert adapter.detect((scene, config)) is True
    assert adapter.detect(scene) is True
    assert adapter.detect(object()) is False
    assert adapter.detect((scene,)) is False


def test_config_round_trip(adapter):
    scene, config = _build_pair()
    rebuilt_scene, rebuilt_config = adapter.to_platform(adapter.to_studio((scene, config)))
    assert_radar_config_equal(config, rebuilt_config)


def test_structures_round_trip(adapter):
    scene, config = _build_pair()
    rebuilt_scene, rebuilt_config = adapter.to_platform(adapter.to_studio((scene, config)))
    assert_structures_equal(scene.structures, rebuilt_scene.structures)


def test_metadata_round_trip(adapter):
    scene, config = _build_pair()
    rebuilt_scene, _ = adapter.to_platform(adapter.to_studio((scene, config)))
    by_name = {s.name: s for s in rebuilt_scene.structures}
    assert by_name["car"].metadata["dynamic"] is True
    assert by_name["car"].metadata["bsdf"] == {"type": "conductor", "material": "Cu"}
    assert "dynamic" not in by_name["wall"].metadata
    assert "bsdf" not in by_name["wall"].metadata


def test_bare_scene_defaults_config(adapter):
    scene, _ = _build_pair()
    rebuilt_scene, rebuilt_config = adapter.to_platform(adapter.to_studio(scene))
    # Bare-scene load uses the component defaults (a valid 3TX/4RX 77 GHz config).
    assert rebuilt_config.num_tx == 3 and rebuilt_config.num_rx == 4
    assert_structures_equal(scene.structures, rebuilt_scene.structures)
