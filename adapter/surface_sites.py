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

def _enabled_scatterers(scene: Any) -> tuple[tuple[str, Any, Any], ...]:
    objects = getattr(scene, "objects", None)
    if not objects:
        return ()
    found = []
    for object_id, obj in sorted(objects.items(), key=lambda item: str(item[0])):
        scatterer = obj.get_component("RadarScatterer") if hasattr(obj, "get_component") else None
        if scatterer is not None and bool(scatterer.enabled):
            found.append((str(object_id), obj, scatterer))
    return tuple(found)


def _apply_declared_motion(obj: Any, positions: np.ndarray, time_s: float) -> np.ndarray:
    component = obj.get_component("RadarMotion") if hasattr(obj, "get_component") else None
    if component is None:
        return positions
    from scipy.spatial.transform import Rotation

    from .motion_map import MotionMap

    trajectory = MotionMap.build(component)
    if trajectory is None:
        return positions
    motion = trajectory.at(float(time_s))
    moved = np.asarray(positions, dtype=np.float64)
    if motion.rotation is not None:
        value = np.asarray(motion.rotation.detach().cpu(), dtype=np.float64)
        rotation = (
            Rotation.from_quat((value[1], value[2], value[3], value[0]))
            if value.shape == (4,)
            else Rotation.from_euler("zyx", value)
        )
        moved = rotation.apply(moved)
    if motion.translation is not None:
        moved = moved + np.asarray(motion.translation.detach().cpu(), dtype=np.float64)
    return np.ascontiguousarray(moved, dtype=np.float32)


def _point_site_position(obj: Any, scatterer: Any, time_s: float) -> np.ndarray:
    local = np.asarray([*scatterer.site_offset, 1.0], dtype=np.float64)
    world = (np.asarray(get_world_transform(obj), dtype=np.float64) @ local)[:3]
    return _apply_declared_motion(obj, world.reshape(1, 3), time_s)


def scene_scatterer_kinematics(
    scenes: Sequence[Any],
    times_s: Sequence[float],
    *,
    stable_id_base: int = 3_000_000,
) -> SurfaceKinematicsSequence:
    """Build one stable explicit-site sequence from all enabled scene scatterers."""
    frames = tuple(scenes)
    times = tuple(float(value) for value in times_s)
    if not frames:
        raise ValueError("radar scatterer kinematics requires at least one scene frame")
    if len(frames) != len(times):
        raise ValueError(f"received {len(frames)} scene frames for {len(times)} times")

    declared = _enabled_scatterers(frames[0])
    if not declared:
        raise ValueError("radar bake requires at least one enabled RadarScatterer")
    identity = tuple(item[0] for item in declared)
    for index, scene in enumerate(frames[1:], start=1):
        current = tuple(item[0] for item in _enabled_scatterers(scene))
        if current != identity:
            raise ValueError(
                f"enabled RadarScatterer identity changed at frame {index}; "
                "a baked sequence requires the same ordered targets in every frame"
            )

    tracks = []
    next_stable_id = int(stable_id_base)
    for object_id, _, first_scatterer in declared:
        object_frames = []
        components = []
        for scene in frames:
            objects = getattr(scene, "objects", {})
            match = next(
                (obj for key, obj in objects.items() if str(key) == object_id),
                None,
            )
            if match is None:
                raise ValueError(f"RadarScatterer object {object_id!r} is missing from a sampled frame")
            component = match.get_component("RadarScatterer")
            object_frames.append(match)
            components.append(component)

        mode = str(first_scatterer.mode)
        if any(str(component.mode) != mode for component in components):
            raise ValueError(f"RadarScatterer {object_id!r} changed mode during the baked sequence")
        if mode == "surface":
            count = int(first_scatterer.surface_site_count)
            seed = int(first_scatterer.surface_seed)
            if any(
                int(component.surface_site_count) != count or int(component.surface_seed) != seed
                for component in components
            ):
                raise ValueError(
                    f"surface RadarScatterer {object_id!r} changed site count or seed during the baked sequence"
                )
            track = surface_kinematics_sequence(
                object_frames,
                times,
                count=count,
                seed=seed,
                stable_id_base=next_stable_id,
            )
            positions = np.stack(
                [
                    _apply_declared_motion(obj, track.positions_m[index], times[index])
                    for index, obj in enumerate(object_frames)
                ],
                axis=0,
            )
            track = SurfaceKinematicsSequence(
                positions_m=np.ascontiguousarray(positions, dtype=np.float32),
                velocities_m_per_s=differentiate_site_positions(positions, times),
                stable_ids=track.stable_ids,
            )
        elif mode == "point":
            positions = np.concatenate(
                [
                    _point_site_position(obj, component, times[index])[None, ...]
                    for index, (obj, component) in enumerate(zip(object_frames, components, strict=True))
                ],
                axis=0,
            )
            track = SurfaceKinematicsSequence(
                positions_m=np.ascontiguousarray(positions, dtype=np.float32),
                velocities_m_per_s=differentiate_site_positions(positions, times),
                stable_ids=np.asarray([next_stable_id], dtype=np.int64),
            )
        else:
            raise ValueError(f"RadarScatterer {object_id!r} has unsupported mode {mode!r}")
        tracks.append(track)
        next_stable_id += track.site_count

    return SurfaceKinematicsSequence(
        positions_m=np.ascontiguousarray(
            np.concatenate([track.positions_m for track in tracks], axis=1),
            dtype=np.float32,
        ),
        velocities_m_per_s=np.ascontiguousarray(
            np.concatenate([track.velocities_m_per_s for track in tracks], axis=1),
            dtype=np.float32,
        ),
        stable_ids=np.ascontiguousarray(
            np.concatenate([track.stable_ids for track in tracks]),
            dtype=np.int64,
        ),
    )
