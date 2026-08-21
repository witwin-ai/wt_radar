"""A current Core ``(DynamicScene, RadarConfig)`` factory for the load demo.

Load it from the editor with factory string ``wt_radar.examples.demo_scene:radar_demo``:
the radar adapter builds a Radar Settings object (77 GHz FMCW defaults) plus a static
wall and a moving car target. Press **Simulate** on the Radar Settings object to run a
backend and see the range-doppler / point-cloud / MUSIC views in-component (R3).

Units are the platform's native sensor units (MHz/us, us, ksps, dBm, half-wavelength
antenna locations). Core owns the world and Radar owns the sensor configuration.
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
    """Build the demo ``(Core DynamicScene, RadarConfig)`` pair."""
    import torch
    import witwin.core as wc
    from witwin.radar import RadarConfig

    wall = wc.Structure(
        wc.Box(position=(0.0, 0.0, -4.0), size=(2.0, 2.0, 0.1)),
        wc.PhysicalMaterial(eps_r=5.0, name="concrete", material_id=1),
        name="Wall",
        structure_id=1,
        material_id=1,
        assignment_id=1,
        surface_id=1,
    )
    car = wc.Structure(
        wc.Box(position=(0.6, 0.0, -2.5), size=(0.6, 0.4, 0.4)),
        wc.PhysicalMaterial(eps_r=8.0, name="metal", material_id=2),
        name="Car",
        structure_id=2,
        material_id=2,
        assignment_id=2,
        surface_id=2,
    )
    scene = wc.Scene(structures=(wall, car))
    dynamic = wc.DynamicScene(
        scene,
        structure_trajectories={
            2: wc.LinearTrajectory(
                origin=torch.zeros(3),
                velocity=torch.tensor((0.2, 0.0, 0.0)),
            )
        },
    )
    return dynamic, RadarConfig.from_dict(DEMO_CONFIG)
