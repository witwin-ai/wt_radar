"""Stable surface-site identity and kinematics regression tests."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from wt_radar.adapter import surface_sites


@dataclass
class _Frame:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    matrix: np.ndarray


def _triangle(z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, z]],
        dtype=np.float32,
    )
    return vertices, np.asarray([[0, 1, 2]], dtype=np.int64)


def _frame(z: float = 0.0, x: float = 0.0) -> _Frame:
    vertices, faces = _triangle(z)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = x
    return _Frame("body", vertices, faces, matrix)


@pytest.fixture
def frame_adapter(monkeypatch):
    monkeypatch.setattr(
        surface_sites,
        "render_mesh_arrays",
        lambda frame: (frame.vertices, frame.faces),
    )
    monkeypatch.setattr(surface_sites, "get_world_transform", lambda frame: frame.matrix)


def test_surface_layout_is_deterministic_and_preserves_stable_ids():
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    left = surface_sites.SurfaceSiteLayout.sample(
        vertices, faces, count=32, seed=19, stable_id_base=7_000_000
    )
    right = surface_sites.SurfaceSiteLayout.sample(
        vertices, faces, count=32, seed=19, stable_id_base=7_000_000
    )

    np.testing.assert_array_equal(left.face_indices, right.face_indices)
    np.testing.assert_allclose(left.barycentric, right.barycentric)
    np.testing.assert_array_equal(left.stable_ids, np.arange(7_000_000, 7_000_032))
    np.testing.assert_allclose(left.barycentric.sum(axis=1), 1.0, atol=1e-6)
    points = left.evaluate(vertices, faces)
    assert points.shape == (32, 3)
    assert np.all(points[:, :2] >= 0.0)
    assert np.all(points[:, :2] <= 1.0)


def test_world_translation_produces_exact_site_velocity(frame_adapter):
    frames = [_frame(x=0.0), _frame(x=0.5), _frame(x=1.0)]

    sequence = surface_sites.surface_kinematics_sequence(
        frames,
        (0.0, 0.5, 1.0),
        count=8,
        seed=2,
    )

    assert sequence.positions_m.shape == (3, 8, 3)
    np.testing.assert_allclose(sequence.velocities_m_per_s[:, :, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(sequence.velocities_m_per_s[:, :, 1:], 0.0, atol=1e-6)
    np.testing.assert_array_equal(sequence.stable_ids, np.arange(3_000_000, 3_000_008))


def test_articulated_vertex_motion_produces_nonzero_surface_velocity(frame_adapter):
    frames = [_frame(z=0.0), _frame(z=1.0), _frame(z=2.0)]

    sequence = surface_sites.surface_kinematics_sequence(
        frames,
        (0.0, 0.5, 1.0),
        count=16,
        seed=5,
    )

    assert np.all(sequence.velocities_m_per_s[:, :, 2] > 0.0)
    np.testing.assert_allclose(
        sequence.positions_m[2, :, :2],
        sequence.positions_m[0, :, :2],
        atol=1e-6,
    )


def test_topology_change_is_refused_instead_of_relabeling_sites():
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    layout = surface_sites.SurfaceSiteLayout.sample(vertices, faces, count=4, seed=0)

    with pytest.raises(ValueError, match="topology changed"):
        layout.evaluate(vertices, faces[::-1])


def test_degenerate_surface_is_refused():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="non-degenerate"):
        surface_sites.SurfaceSiteLayout.sample(vertices, faces, count=1, seed=0)


def test_surface_scatterer_defaults_are_backward_compatible():
    from wt_radar.components.scatterer import RadarScattererComponent

    scatterer = RadarScattererComponent()

    assert scatterer.mode == "point"
    assert scatterer.surface_site_count == 128
    assert scatterer.surface_seed == 0

class _SceneObject:
    def __init__(self, frame, scatterer):
        self.name = frame.name
        self.vertices = frame.vertices
        self.faces = frame.faces
        self.matrix = frame.matrix
        self.scatterer = scatterer

    def get_component(self, name):
        if name == "RadarScatterer":
            return self.scatterer
        return None


def test_scene_sequence_combines_point_and_surface_targets(frame_adapter):
    point_scatterer = SimpleNamespace(
        enabled=True,
        mode="point",
        site_offset=(0.0, 0.0, 0.0),
    )
    surface_scatterer = SimpleNamespace(
        enabled=True,
        mode="surface",
        surface_site_count=4,
        surface_seed=7,
    )
    scenes = []
    for index, time_s in enumerate((0.0, 0.5, 1.0)):
        scenes.append(
            SimpleNamespace(
                objects={
                    "point": _SceneObject(_frame(x=time_s), point_scatterer),
                    "surface": _SceneObject(_frame(z=float(index)), surface_scatterer),
                }
            )
        )

    sequence = surface_sites.scene_scatterer_kinematics(
        scenes,
        (0.0, 0.5, 1.0),
    )

    assert sequence.positions_m.shape == (3, 5, 3)
    np.testing.assert_allclose(sequence.velocities_m_per_s[:, 0, 0], 1.0, atol=1e-6)
    assert np.all(sequence.velocities_m_per_s[:, 1:, 2] > 0.0)
    np.testing.assert_array_equal(
        sequence.stable_ids,
        np.arange(3_000_000, 3_000_005),
    )
