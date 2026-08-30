"""Sample Studio's existing timeline and skinning, never the live editor scene.

Velocities are numerical derivatives of the same rendered world vertices.
Differences stay inside a LINEAR key interval, with a right-hand derivative at
knots. The step (at most 1 ms) and approximation are reported, not RF algorithms.
"""
import numpy as np

from .snapshot import _hierarchy


class StudioSkinSampler:
    def __init__(self, scene, target_id):
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
        self.vertex_ids, self.site_names = np.asarray(chosen), names

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
        world = self.skin.compute_skinned_vertices(in_local_space=False).detach().cpu().numpy()
        if not np.isfinite(world).all():
            raise ValueError("Nonfinite Studio skin geometry")
        return world[self.vertex_ids].astype(np.float64)

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
