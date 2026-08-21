"""Shared fixtures and comparisons for the Radar 0.3 plugin tests."""
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import torch

ATOL = 1e-5
RTOL = 1e-6

BASE_CONFIG = {
    "num_tx": 1,
    "num_rx": 2,
    "fc": 77e9,
    "slope": 60.012,
    "power": 12.0,
    "adc_samples": 16,
    "adc_start_time": 6.0,
    "sample_rate": 4400.0,
    "idle_time": 7.0,
    "ramp_end_time": 58.0,
    "chirp_per_frame": 8,
    "frame_per_second": 10.0,
    "num_doppler_bins": 8,
    "num_range_bins": 16,
    "num_angle_bins": 8,
    "tx_loc": [[0.0, 0.0, 0.0]],
    "rx_loc": [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
}


def approx(a: Any, b: Any) -> bool:
    a, b = float(a), float(b)
    return abs(a - b) <= ATOL + RTOL * max(abs(a), abs(b))


def make_config(**updates: Any) -> Any:
    import witwin.radar as wr

    return wr.RadarConfig.from_dict({**BASE_CONFIG, **updates})


def make_scene(*, dynamic: bool = False) -> Any:
    import witwin.core as wc

    structures = (
        wc.Structure(
            wc.Box(position=(0.0, 0.0, -4.0), size=(2.0, 2.0, 0.1)),
            wc.PhysicalMaterial(eps_r=5.0, name="concrete", material_id=1),
            name="wall",
            structure_id=1,
            material_id=1,
            assignment_id=1,
            surface_id=1,
        ),
        wc.Structure(
            wc.Box(position=(0.6, 0.0, -2.5), size=(0.6, 0.4, 0.4)),
            wc.PhysicalMaterial(eps_r=8.0, sigma_e=0.02, name="metal", material_id=2),
            name="car",
            structure_id=2,
            material_id=2,
            assignment_id=2,
            surface_id=2,
        ),
    )
    scene = wc.Scene(structures=structures)
    if not dynamic:
        return scene
    trajectory = wc.LinearTrajectory(
        origin=torch.tensor((0.2, 0.0, 0.0), dtype=torch.float32),
        velocity=torch.tensor((0.1, 0.0, 0.0), dtype=torch.float32),
        reference_time_s=0.5,
    )
    return wc.DynamicScene(scene, structure_trajectories={2: trajectory})


def normalized(value: Any) -> Any:
    if is_dataclass(value):
        return normalized(asdict(value))
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(normalized(item) for item in value)
    if hasattr(value, "detach"):
        return normalized(value.detach().cpu().tolist())
    if isinstance(value, (float, np.floating)):
        return round(float(value), 7)
    return value


def assert_config_equal(left: Any, right: Any) -> None:
    _assert_value_equal(left, right)


def _assert_value_equal(left: Any, right: Any) -> None:
    if is_dataclass(left) or is_dataclass(right):
        assert is_dataclass(left) and is_dataclass(right)
        _assert_value_equal(asdict(left), asdict(right))
    elif isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_value_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_value_equal(left_item, right_item)
    elif isinstance(left, float) or isinstance(right, float):
        assert approx(left, right)
    else:
        assert left == right


def bounds(geometry: Any) -> np.ndarray:
    vertices, _ = geometry.to_mesh()
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    vertices = np.asarray(vertices).reshape(-1, 3)
    return np.stack((vertices.min(axis=0), vertices.max(axis=0)), axis=1)
