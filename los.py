"""Studio-side static line of sight (``studio_los_v1``) for placement and labels.

A segment test against the static room meshes, used to approximate what the
Radar can see before and after a simulation. It is a label and placement
approximation only: the native simulation keeps its own path discovery, and no
propagation, scattering or antenna response is modelled here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LOS_METHOD = "studio_los_v1"
# A hit this close to either end of a segment does not block it: the start sits
# on the mount's wall stand-off and the end often lies on a supporting surface
# (a paw on a bed top, a foot point on a wall face).
START_TOLERANCE_M = .01
END_TOLERANCE_M = .02
_PAIR_BUDGET = 2_000_000


@dataclass
class StaticMesh:
    """One static mesh in world coordinates."""

    object_id: str
    name: str
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def lo(self) -> np.ndarray:
        return self.vertices.min(axis=0)

    @property
    def hi(self) -> np.ndarray:
        return self.vertices.max(axis=0)

    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]


def static_meshes(scene, *, excluded_ids=()) -> list[StaticMesh]:
    """Visible static meshes of a Studio scene; skinned actors are never occluders."""
    from .adapter.snapshot import _hierarchy

    matrices, visible = _hierarchy(scene)
    excluded = {str(value) for value in excluded_ids}
    meshes = []
    for object_id, obj in scene.objects.items():
        object_id = str(object_id)
        if object_id in excluded or not visible.get(object_id, True):
            continue
        if obj.get_component("SkinnedMesh") is not None:
            continue
        mesh = obj.get_component("Mesh")
        if mesh is None or mesh.is_empty:
            continue
        vertices = np.asarray(mesh.get_vertices_numpy(), dtype=np.float64)
        faces = np.asarray(mesh.get_faces_numpy(), dtype=np.int64).reshape(-1, 3)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not len(faces):
            continue
        world = (np.column_stack((vertices, np.ones(len(vertices)))) @ matrices[object_id].T)[:, :3]
        meshes.append(StaticMesh(object_id, str(obj.name), world, faces))
    return meshes


class StaticOccluders:
    """Segment-vs-triangle occlusion with a per-mesh bounding-box cull."""

    def __init__(self, meshes):
        self.meshes = [mesh for mesh in meshes if len(mesh.faces)]
        self._lo = np.array([mesh.lo for mesh in self.meshes]).reshape(-1, 3)
        self._hi = np.array([mesh.hi for mesh in self.meshes]).reshape(-1, 3)
        self._triangles = [mesh.triangles() for mesh in self.meshes]

    @classmethod
    def from_scene(cls, scene, *, excluded_ids=()):
        return cls(static_meshes(scene, excluded_ids=excluded_ids))

    @property
    def triangle_count(self) -> int:
        return int(sum(len(item) for item in self._triangles))

    def segments_blocked(self, origins, points, *, exclude_ids=()) -> np.ndarray:
        """True where a static triangle lies strictly between origin and point.

        ``origins`` is one point or one per segment; ``points`` is ``(N, 3)``.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        origins = np.broadcast_to(np.asarray(origins, dtype=np.float64), points.shape)
        blocked = np.zeros(len(points), dtype=bool)
        if not len(points) or not self.meshes:
            return blocked
        direction = points - origins
        length = np.linalg.norm(direction, axis=1)
        valid = length > START_TOLERANCE_M + END_TOLERANCE_M
        t_min = np.where(valid, START_TOLERANCE_M / np.maximum(length, 1e-12), 1.)
        t_max = np.where(valid, 1. - END_TOLERANCE_M / np.maximum(length, 1e-12), 0.)
        excluded = {str(value) for value in exclude_ids}
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = 1. / direction
        for index, mesh in enumerate(self.meshes):
            if mesh.object_id in excluded:
                continue
            candidates = np.flatnonzero(valid & ~blocked & _slab_hits(
                origins, inverse, self._lo[index], self._hi[index], t_min, t_max))
            if not len(candidates):
                continue
            triangles = self._triangles[index]
            step = max(1, _PAIR_BUDGET // max(1, len(triangles)))
            for start in range(0, len(candidates), step):
                rows = candidates[start:start + step]
                hit = _any_triangle_hit(
                    origins[rows], direction[rows], triangles, t_min[rows], t_max[rows])
                blocked[rows[hit]] = True
        return blocked

    def visible(self, origin, points, *, exclude_ids=()) -> np.ndarray:
        return ~self.segments_blocked(origin, points, exclude_ids=exclude_ids)


def _slab_hits(origins, inverse, lo, hi, t_min, t_max):
    # A segment parallel to an axis gets +-inf slab times: unbounded inside the
    # slab, empty outside it. fmin/fmax drop the nan of an origin on the plane.
    with np.errstate(invalid="ignore"):
        t1 = (lo[None, :] - origins) * inverse
        t2 = (hi[None, :] - origins) * inverse
    near = np.fmin(t1, t2)
    far = np.fmax(t1, t2)
    enter = np.maximum(near.max(axis=1), t_min)
    leave = np.minimum(far.min(axis=1), t_max)
    return enter <= leave


def _any_triangle_hit(origins, direction, triangles, t_min, t_max):
    """Moller-Trumbore for every (segment, triangle) pair; any hit per segment."""
    v0 = triangles[:, 0][None]
    e1 = (triangles[:, 1] - triangles[:, 0])[None]
    e2 = (triangles[:, 2] - triangles[:, 0])[None]
    d = direction[:, None, :]
    p = np.cross(d, e2)
    det = np.sum(e1 * p, axis=2)
    ok = np.abs(det) > 1e-12
    inv_det = np.where(ok, 1. / np.where(ok, det, 1.), 0.)
    s = origins[:, None, :] - v0
    u = np.sum(s * p, axis=2) * inv_det
    q = np.cross(s, e1)
    v = np.sum(d * q, axis=2) * inv_det
    t = np.sum(e2 * q, axis=2) * inv_det
    hit = (ok & (u >= 0.) & (v >= 0.) & (u + v <= 1.)
           & (t > t_min[:, None]) & (t < t_max[:, None]))
    return hit.any(axis=1)


__all__ = ["LOS_METHOD", "StaticMesh", "StaticOccluders", "static_meshes"]
