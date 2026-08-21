"""Deterministic surface sites and frame-wise kinematics for Studio meshes."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from witwin_server.utils.mitsuba_utils import get_world_transform

from .common import render_mesh_arrays


def _mesh_arrays(vertices: Any, faces: Any) -> tuple[np.ndarray, np.ndarray]:
    verts = np.ascontiguousarray(np.asarray(vertices, dtype=np.float32).reshape(-1, 3))
    tris = np.ascontiguousarray(np.asarray(faces, dtype=np.int64).reshape(-1, 3))
    if verts.shape[0] == 0 or tris.shape[0] == 0:
        raise ValueError("surface sampling requires a non-empty triangle mesh")
    if int(tris.min()) < 0 or int(tris.max()) >= int(verts.shape[0]):
        raise ValueError("mesh faces contain a vertex index outside the vertex array")
    if not np.isfinite(verts).all():
        raise ValueError("mesh vertices must be finite")
    return verts, tris


def _topology_digest(vertex_count: int, faces: np.ndarray) -> str:
    count = np.asarray([int(vertex_count)], dtype="<i8")
    canonical = np.ascontiguousarray(faces, dtype="<i8")
    return hashlib.sha256(count.tobytes() + canonical.tobytes()).hexdigest()


@dataclass(frozen=True)
class SurfaceSiteLayout:
    """Stable triangle and barycentric identities sampled from one mesh topology."""

    vertex_count: int
    topology_digest: str
    face_indices: np.ndarray
    barycentric: np.ndarray
    stable_ids: np.ndarray

    @classmethod
    def sample(
        cls,
        vertices: Any,
        faces: Any,
        *,
        count: int,
        seed: int,
        stable_id_base: int = 3_000_000,
    ) -> "SurfaceSiteLayout":
        if type(count) is not int or count < 1:
            raise ValueError(f"count must be a positive int, got {count!r}")
        if type(seed) is not int:
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        verts, tris = _mesh_arrays(vertices, faces)
        triangle_vertices = verts[tris]
        cross = np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        )
        areas = np.linalg.norm(cross, axis=1).astype(np.float64) * 0.5
        total_area = float(areas.sum())
        if not np.isfinite(total_area) or total_area <= np.finfo(np.float32).eps:
            raise ValueError("surface sampling requires at least one non-degenerate triangle")

        rng = np.random.default_rng(seed)
        face_indices = rng.choice(tris.shape[0], size=count, replace=True, p=areas / total_area)
        first = rng.random(count)
        second = rng.random(count)
        root = np.sqrt(first)
        barycentric = np.stack(
            (1.0 - root, root * (1.0 - second), root * second),
            axis=1,
        ).astype(np.float32)
        return cls(
            vertex_count=int(verts.shape[0]),
            topology_digest=_topology_digest(verts.shape[0], tris),
            face_indices=np.ascontiguousarray(face_indices, dtype=np.int64),
            barycentric=np.ascontiguousarray(barycentric),
            stable_ids=np.arange(
                int(stable_id_base),
                int(stable_id_base) + count,
                dtype=np.int64,
            ),
        )

    @property
    def count(self) -> int:
        return int(self.face_indices.shape[0])

    def evaluate(self, vertices: Any, faces: Any) -> np.ndarray:
        """Evaluate the same sites on a deformed mesh with identical topology."""
        verts, tris = _mesh_arrays(vertices, faces)
        digest = _topology_digest(verts.shape[0], tris)
        if int(verts.shape[0]) != self.vertex_count or digest != self.topology_digest:
            raise ValueError(
                "surface site topology changed after sampling; stable site identity "
                "requires identical vertex and face indexing for every frame"
            )
        triangles = verts[tris[self.face_indices]]
        return np.ascontiguousarray(
            np.einsum("si,sij->sj", self.barycentric, triangles),
            dtype=np.float32,
        )


@dataclass(frozen=True)
class SurfaceKinematicsSequence:
    """World positions and finite-difference velocities for stable surface sites."""

    positions_m: np.ndarray
    velocities_m_per_s: np.ndarray
    stable_ids: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def site_count(self) -> int:
        return int(self.positions_m.shape[1])


def world_surface_positions(obj: Any, layout: SurfaceSiteLayout) -> np.ndarray:
    """Evaluate a layout on the object's current skinned mesh in world coordinates."""
    mesh_arrays = render_mesh_arrays(obj)
    if mesh_arrays is None:
        raise ValueError(f"surface RadarScatterer object {obj.name!r} has no non-empty triangle mesh")
    local = layout.evaluate(*mesh_arrays)
    ones = np.ones((local.shape[0], 1), dtype=np.float32)
    homogeneous = np.concatenate((local, ones), axis=1)
    world = (np.asarray(get_world_transform(obj), dtype=np.float64) @ homogeneous.T).T[:, :3]
    return np.ascontiguousarray(world, dtype=np.float32)


def differentiate_site_positions(positions_m: Any, times_s: Sequence[float]) -> np.ndarray:
    """Differentiate stable site rows with centred interior and one-sided edges."""
    positions = np.ascontiguousarray(np.asarray(positions_m, dtype=np.float32))
    times = np.asarray(tuple(float(value) for value in times_s), dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"positions_m must have shape (F, S, 3), got {positions.shape}")
    if positions.shape[0] != times.shape[0]:
        raise ValueError(
            f"positions_m has {positions.shape[0]} frames but times_s has {times.shape[0]} entries"
        )
    if positions.shape[0] == 0:
        raise ValueError("surface kinematics requires at least one frame")
    if not np.isfinite(positions).all() or not np.isfinite(times).all():
        raise ValueError("surface positions and times must be finite")
    if positions.shape[0] == 1:
        return np.zeros_like(positions)
    intervals = np.diff(times)
    if np.any(intervals <= 0.0):
        raise ValueError("times_s must be strictly increasing")

    velocities = np.empty_like(positions)
    velocities[0] = (positions[1] - positions[0]) / intervals[0]
    velocities[-1] = (positions[-1] - positions[-2]) / intervals[-1]
    if positions.shape[0] > 2:
        spans = (times[2:] - times[:-2]).reshape(-1, 1, 1)
        velocities[1:-1] = (positions[2:] - positions[:-2]) / spans
    return np.ascontiguousarray(velocities)


def surface_kinematics_sequence(
    objects: Sequence[Any],
    times_s: Sequence[float],
    *,
    count: int,
    seed: int,
    stable_id_base: int = 3_000_000,
) -> SurfaceKinematicsSequence:
    """Sample the first object once, then evaluate that identity over all frames."""
    frames = tuple(objects)
    times = tuple(float(value) for value in times_s)
    if not frames:
        raise ValueError("surface kinematics requires at least one object frame")
    if len(frames) != len(times):
        raise ValueError(f"received {len(frames)} object frames for {len(times)} times")

    first_mesh = render_mesh_arrays(frames[0])
    if first_mesh is None:
        raise ValueError(f"surface RadarScatterer object {frames[0].name!r} has no non-empty triangle mesh")
    layout = SurfaceSiteLayout.sample(
        *first_mesh,
        count=count,
        seed=seed,
        stable_id_base=stable_id_base,
    )
    positions = np.stack(
        [world_surface_positions(obj, layout) for obj in frames],
        axis=0,
    )
    return SurfaceKinematicsSequence(
        positions_m=np.ascontiguousarray(positions, dtype=np.float32),
        velocities_m_per_s=differentiate_site_positions(positions, times),
        stable_ids=layout.stable_ids,
    )
