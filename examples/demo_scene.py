"""A radar ``(Scene, RadarConfig)`` factory for the ``load_platform_scene`` demo.

Load it from the editor with factory string ``wt_radar.examples.demo_scene:radar_demo``:
the radar adapter builds a Radar Settings object (77 GHz FMCW defaults) plus a static
wall and a moving car target. Press **Simulate** on the Radar Settings object to run a
backend and see the range-doppler / point-cloud / MUSIC views in-component (R3).

Units are the platform's native sensor units (MHz/us, us, ksps, dBm, half-wavelength
antenna locations). The scene device is ``cpu`` so the round trip is portable; the solve
device comes from the RadarSensor component.
"""
from typing import Any, Dict, Tuple

# 77 GHz automotive FMCW defaults (matches examples/single_point.py).
DEMO_CONFIG: Dict[str, Any] = {
    "num_tx": 3,
    "num_rx": 4,
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 6,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 58,
    "chirp_per_frame": 128,
    "frame_per_second": 10,
    "num_doppler_bins": 128,
    "num_range_bins": 256,
    "num_angle_bins": 64,
    "power": 15,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
}


def radar_demo() -> Tuple[Any, Any]:
    """Build the demo ``(radar.Scene, RadarConfig)`` pair."""
    import witwin.radar as wr

    scene = wr.Scene(device="cpu")
    scene.add_mesh(
        name="Wall",
        geometry=wr.Box(position=(0.0, 0.0, -4.0), size=(2.0, 2.0, 0.1)),
        material=wr.Material(eps_r=5.0, name="concrete"),
    )
    scene.add_mesh(
        name="Car",
        geometry=wr.Box(position=(0.6, 0.0, -2.5), size=(0.6, 0.4, 0.4)),
        material=wr.Material(eps_r=8.0, name="metal"),
        bsdf={"type": "conductor"},
        dynamic=True,
    )
    config = wr.RadarConfig.from_dict(DEMO_CONFIG)
    return scene, config
