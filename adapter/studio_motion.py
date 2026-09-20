"""Sample Studio's existing timeline and skinning, never the live editor scene.

Velocities are numerical derivatives of the same rendered world vertices.
Differences stay inside a LINEAR key interval, with a right-hand derivative at
knots. The step (at most 1 ms) and approximation are reported, not RF algorithms.
"""
import re

import numpy as np
import torch

from .snapshot import _hierarchy


def spatially_spread_ranks(points, count):
    """Select a deterministic, spatially spread subset of bind-pose sites."""
    points = np.asarray(points, dtype=np.float64)
    count = int(count)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Site selection requires finite Nx3 bind-pose points.")
    if not 1 <= count <= len(points):
        raise ValueError("Site selection count must be within the available point count.")
    if count == len(points):
        return np.arange(len(points), dtype=np.int64)
    center = points.mean(axis=0)
    selected = [int(np.argmin(np.sum((points - center) ** 2, axis=1)))]
    nearest_distance = np.sum((points - points[selected[0]]) ** 2, axis=1)
    while len(selected) < count:
        candidates = nearest_distance.copy()
        candidates[selected] = -np.inf
        rank = int(np.argmax(candidates))
        selected.append(rank)
        nearest_distance = np.minimum(
            nearest_distance,
            np.sum((points - points[rank]) ** 2, axis=1),
        )
    return np.asarray(sorted(selected), dtype=np.int64)


def anatomical_site_priority(name):
    """Rank useful articulated-body sites above cosmetic micro-bones."""
    name = str(name).casefold()
    if any(token in name for token in ("mouth", "tongue", "mustache", "eyelid", "eye", "claw", "ear")):
        return 0
    if any(token in name for token in ("head", "neck", "spine")):
        return 100
    if any(token in name for token in ("shoulder", "hip", "leg", "foot")):
        return 90
    if "tail" in name:
        numbers = re.findall(r"\d+", name)
        segment = int(numbers[-1]) if numbers else 1
        return 80 if segment % 2 else 60
    return 70


def anatomically_prioritized_ranks(points, names, count):
    """Select semantic priority tiers, using spatial coverage to break ties."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) != len(names):
        raise ValueError("Site names and bind-pose points must have equal length.")
    count = min(int(count), len(points))
    priorities = np.asarray([anatomical_site_priority(name) for name in names])
    selected = []
    for priority in sorted(set(priorities.tolist()), reverse=True):
        tier = np.flatnonzero(priorities == priority)
        remaining = count - len(selected)
        if remaining <= 0:
            break
        if len(tier) <= remaining:
            selected.extend(tier.tolist())
        else:
            local = spatially_spread_ranks(points[tier], remaining)
            selected.extend(tier[local].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


class StudioSkinSampler:
    def __init__(self, scene, target_id, *, site_budget=64):
        self.scene = scene
        self.target = scene.get_object(target_id)
        self.skin = self.target.get_component("SkinnedMesh") if self.target else None
        self.manager = scene.timeline_manager
        if self.skin is None or self.skin.is_empty or self.manager is None:
            raise ValueError("Animation requires an existing hydrated SkinnedMesh and baked Studio timeline.")
        self.duration = float(self.manager.clip.duration)
        _hierarchy(scene)
        skin = self.skin.get_skinning_data()
        self.bone_ids = list(skin["bone_ids"])
        vertices = np.asarray(self.skin.get_vertices_numpy())
        indices = np.asarray(skin.get("skin_indices"), dtype=np.int64)
        weights = np.asarray(skin.get("skin_weights"), dtype=np.float64)
        ibm = np.asarray(skin.get("inverse_bind_matrices"), dtype=np.float64)
        if (not self.bone_ids or indices.shape != (len(vertices), 4) or weights.shape != indices.shape
                or ibm.shape != (len(self.bone_ids), 4, 4) or indices.min() < 0
                or indices.max() >= len(self.bone_ids) or not np.isfinite(ibm).all()
                or not np.isfinite(weights).all() or (weights < 0).any()
                or not np.allclose(weights.sum(1), 1, atol=2e-5, rtol=0)):
            raise ValueError("Invalid skin influences / inverse bind matrices; no rigid or identity fallback.")
        self.moving_ids = {str(target_id), *self.bone_ids}
        for oid in list(self.moving_ids):
            obj = scene.get_object(oid)
            if obj is None or obj.get_component("Transform") is None:
                raise ValueError(f"Missing skin bone / Transform: {oid}")
            while obj.parent_id:
                self.moving_ids.add(str(obj.parent_id))
                obj = scene.get_object(obj.parent_id)
                if obj is None:
                    raise ValueError("Unresolved skin hierarchy")
        # CPU skinning uses parent_id; static export uses Transform.parent.
        for obj in scene.objects.values():
            transform = obj.get_component("Transform")
            parent = scene.get_object(obj.parent_id) if obj.parent_id else None
            expected = parent.get_component("Transform") if parent else None
            if transform is not None and transform.parent is not expected:
                raise ValueError(f"Inconsistent Studio parent representations at {obj.id}")
        knots = [0.0, self.duration]
        self.tracks = []
        for track in self.manager.clip.tracks.values():
            if not track.keyframes:
                continue
            if (track.object_id not in self.moving_ids or track.component_name != "Transform"
                    or track.field_name not in {"position", "rotation", "scale"}):
                raise ValueError(f"Only target/bone Transform animation is supported; unsupported track {track.path}")
            times = np.asarray([key.time for key in track.keyframes])
            values = np.asarray([key.value for key in track.keyframes], dtype=np.float64)
            if (not np.isfinite(times).all() or (np.diff(times) <= 0).any()
                    or values.shape != (len(times), 3) or not np.isfinite(values).all()
                    or any(key.interpolation.value != "linear" for key in track.keyframes)):
                raise ValueError(f"Animation requires finite, ordered LINEAR vector tracks: {track.path}")
            self.tracks.append(track)
            knots.extend(times.tolist())
        if not self.tracks:
            raise ValueError("No baked target/bone motion tracks. Run Motion Matching in the room first.")
        self.knots = np.unique(knots)
        # One representative surface vertex per influenced bone, fixed in bind
        # geometry. Prefer high-influence vertices nearest the weighted centroid.
        chosen, names = [], []
        for bone, oid in enumerate(self.bone_ids):
            influence = np.where(indices == bone, weights, 0).sum(1)
            if influence.max() <= 0:
                continue
            candidates = np.flatnonzero(influence >= .95 * influence.max())
            center = (vertices * influence[:, None]).sum(0) / influence.sum()
            vertex = int(candidates[np.argmin(np.linalg.norm(vertices[candidates] - center, axis=1))])
            if vertex not in chosen:
                chosen.append(vertex)
                names.append(str(scene.get_object(oid).name))
        if not 1 <= len(chosen) <= 64:
            raise ValueError("Animation supports 1..64 influenced-bone surface sites.")
        if isinstance(site_budget, (bool, np.bool_)) or not 1 <= int(site_budget) <= 64:
            raise ValueError("Target site budget must be an integer from 1 to 64.")
        self.available_site_count = len(chosen)
        self.site_budget = int(site_budget)
        ranks = anatomically_prioritized_ranks(
            vertices[np.asarray(chosen)], names, min(self.site_budget, len(chosen)),
        )
        chosen = [chosen[rank] for rank in ranks]
        names = [names[rank] for rank in ranks]
        self.vertex_ids, self.site_names = np.asarray(chosen), names
        selected = self.vertex_ids
        self._base_vertices = torch.as_tensor(vertices[selected], dtype=torch.float32).contiguous()
        self._bone_indices = torch.as_tensor(indices[selected], dtype=torch.int64).contiguous()
        self._bone_weights = torch.as_tensor(weights[selected], dtype=torch.float32).contiguous()
        self._inverse_bind_matrices = torch.as_tensor(ibm, dtype=torch.float32).contiguous()

    def _bone_world_transforms_once(self):
        """Use Studio's exact transform math while memoizing each hierarchy node."""
        compute = getattr(self.skin, "_compute_world_matrix_torch", None)
        collect = getattr(self.skin, "_get_bone_world_transforms", None)
        if compute is None or collect is None:
            raise ValueError("Studio SkinnedMesh lacks the required world-transform contract.")
        cache = {}

        def memoized(obj, scene):
            key = str(obj.id)
            if key not in cache:
                cache[key] = compute(obj, scene)
            return cache[key]

        had_override = "_compute_world_matrix_torch" in vars(self.skin)
        previous_override = vars(self.skin).get("_compute_world_matrix_torch")
        self.skin._compute_world_matrix_torch = memoized
        try:
            return collect()
        finally:
            if had_override:
                self.skin._compute_world_matrix_torch = previous_override
            else:
                del self.skin._compute_world_matrix_torch

    def _sparse_world_vertices(self):
        """Apply Studio-equivalent linear blend skinning only to selected sites."""
        skinning = self._bone_world_transforms_once() @ self._inverse_bind_matrices
        vertices = torch.cat(
            (self._base_vertices, torch.ones((len(self._base_vertices), 1), dtype=torch.float32)),
            dim=1,
        )
        output = torch.zeros((len(vertices), 4), dtype=torch.float32)
        for influence in range(4):
            matrices = skinning[self._bone_indices[:, influence]]
            transformed = torch.bmm(matrices, vertices.unsqueeze(-1)).squeeze(-1)
            output += self._bone_weights[:, influence:influence + 1] * transformed
        return output[:, :3]

    def positions(self, time_s):
        if not np.isfinite(time_s) or not 0 <= time_s <= self.duration:
            raise ValueError("Sample time outside Studio timeline")
        self.manager.set_time(float(time_s), apply_to_scene=True)
        # TimelineManager logs and swallows setter errors. Refuse such failures
        # here rather than simulate a stale pose under a new timestamp.
        for track in self.tracks:
            component = self.scene.get_object(track.object_id).get_component("Transform")
            expected = track.get_value_at(float(time_s))
            actual = getattr(component, track.field_name).detach().cpu().numpy()
            if expected is not None and not np.allclose(actual, expected, atol=2e-6, rtol=1e-6):
                raise ValueError(f"Timeline did not apply {track.path} at {time_s}")
        world = self._sparse_world_vertices().detach().cpu().numpy()
        if not np.isfinite(world).all():
            raise ValueError("Nonfinite Studio skin geometry")
        return world.astype(np.float64)

    def sample(self, time_s, step_s=.001):
        positions = self.positions(time_s)
        # Baked frame clocks can differ by floating-point roundoff (<0.1 us).
        # Do not turn that roundoff into a near-zero difference denominator.
        i = int(np.searchsorted(self.knots, time_s + 1e-7, side="right"))
        lo = float(self.knots[max(i - 1, 0)])
        hi = float(self.knots[min(i, len(self.knots) - 1)])
        if hi <= time_s:  # last endpoint: no right-hand motion beyond the clip
            velocity = np.zeros_like(positions)
        else:
            dt = min(float(step_s), (hi - time_s) / 3)
            if time_s - lo >= dt:
                velocity = (self.positions(time_s + dt) - self.positions(time_s - dt)) / (2 * dt)
            else:
                # Second-order forward derivative at the knot, no interval mixing.
                velocity = (-3 * positions + 4 * self.positions(time_s + dt)
                            - self.positions(time_s + 2 * dt)) / (2 * dt)
        self.manager.set_time(float(time_s), apply_to_scene=True)
        return positions, velocity
