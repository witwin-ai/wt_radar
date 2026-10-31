"""Per-frame, per-target visibility labels (design 7.5, ``studio_los_v1``).

Every Radar frame of every declared target gets one label: ``in_view``,
``occluded``, ``out_of_fov`` or ``out_of_range``, from the same skin sites the
simulation declares. Labels describe a sample; they never reject it. The only
hard gate is that each key event is ``in_view`` (decision Q3).

Occlusion is a Studio-side static line-of-sight test against the room meshes
(decision Q2a). The native preflight's interval-global active-site share is
reported next to it so the two can be compared.
"""
from __future__ import annotations

import numpy as np

from .install_rule import antenna_angles
from .los import LOS_METHOD, StaticOccluders


LABELS_SCHEMA = "witwin.radar.labels/v1"
LABELS = ("in_view", "occluded", "out_of_fov", "out_of_range")
# A frame is in view when at least this share of the sites is in the beam, in
# range and in static line of sight.
IN_VIEW_SITE_FRACTION = .3
MIN_RANGE_M = 1.


def _native_shares(native_preflight):
    shares = {}
    if not isinstance(native_preflight, dict):
        return shares
    for item in native_preflight.get("targets") or ():
        if not isinstance(item, dict) or "object_id" not in item:
            continue
        declared = int(item.get("declared_site_count", 0) or 0)
        active = int(item.get("active_site_count", 0) or 0)
        if declared:
            shares[str(item["object_id"])] = active / declared
    return shares


def label_frames(*, radar_position, forward, up, times_s, positions_by_frame, target_slices, occluders,
                 beam_half_deg=30., range_m=(MIN_RANGE_M, 11.), key_events=(), native_preflight=None,
                 target_kinds=None):
    """Label sampled skin sites ``(frames, sites, 3)``; pure geometry, no scene."""
    position = np.asarray(radar_position, dtype=np.float64)
    points = np.asarray(positions_by_frame, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if points.ndim != 3 or points.shape[0] != len(times) or points.shape[2] != 3:
        raise ValueError("positions_by_frame must be (frames, sites, 3) and match times_s.")
    low, high = float(range_m[0]), float(range_m[1])
    half = float(beam_half_deg)
    flat = points.reshape(-1, 3)
    azimuth, elevation = antenna_angles(position, forward, flat, up=up)
    azimuth = azimuth.reshape(points.shape[:2])
    elevation = elevation.reshape(points.shape[:2])
    distance = np.linalg.norm(points - position, axis=2)
    in_beam = (np.abs(azimuth) <= half) & (np.abs(elevation) <= half)
    in_range = (distance >= low) & (distance <= high)
    candidates = (in_beam & in_range).reshape(-1)
    line_of_sight = np.zeros(len(flat), dtype=bool)
    rows = np.flatnonzero(candidates)
    if len(rows):
        line_of_sight[rows] = occluders.visible(position, flat[rows])
    line_of_sight = line_of_sight.reshape(points.shape[:2])
    period = float(np.median(np.diff(times))) if len(times) > 1 else 0.

    events_by_target = {}
    gate_failures, event_results = [], []
    for event in key_events or ():
        events_by_target.setdefault(str(event["object_id"]), []).append(event)
    shares = _native_shares(native_preflight)
    kinds = target_kinds or {}

    targets = []
    for item in target_slices:
        object_id = str(item["object_id"])
        sites = slice(int(item["site_start"]), int(item["site_end"]))
        root = points[:, sites].mean(axis=1)
        root_azimuth, root_elevation = antenna_angles(position, forward, root, up=up)
        root_range = np.linalg.norm(root - position, axis=1)
        visible = (in_beam[:, sites] & in_range[:, sites] & line_of_sight[:, sites]).mean(axis=1)
        beam_share = in_beam[:, sites].mean(axis=1)
        frames, counts = [], {label: 0 for label in LABELS}
        for index in range(len(times)):
            if not low <= root_range[index] <= high:
                label = "out_of_range"
            elif max(abs(root_azimuth[index]), abs(root_elevation[index])) > half:
                label = "out_of_fov"
            elif visible[index] >= IN_VIEW_SITE_FRACTION:
                label = "in_view"
            else:
                label = "occluded"
            counts[label] += 1
            frames.append({
                "frame": index,
                "t_s": float(times[index]),
                "label": label,
                "visible_site_fraction": float(visible[index]),
                "beam_site_fraction": float(beam_share[index]),
                "range_m": float(root_range[index]),
                "azimuth_deg": float(root_azimuth[index]),
                "elevation_deg": float(root_elevation[index]),
                "key_event": None,
            })
        for event in events_by_target.get(object_id, ()):
            t_s = float(event["t_s"])
            index = int(np.argmin(np.abs(times - t_s))) if len(times) else -1
            outside = index < 0 or abs(times[index] - t_s) > period / 2. + 1e-6
            result = {"id": str(event.get("id", "")), "object_id": object_id, "t_s": t_s,
                      "kind": event.get("kind"), "frame": None if outside else index}
            if outside:
                result.update(in_view=False, label=None, reason="outside_measured_interval")
            else:
                label = frames[index]["label"]
                frames[index]["key_event"] = result["id"]
                result.update(in_view=label == "in_view", label=label,
                              reason=None if label == "in_view" else label)
            event_results.append(result)
            if not result["in_view"]:
                gate_failures.append({"id": result["id"], "object_id": object_id, "t_s": t_s,
                                      "reason": result["reason"]})
        targets.append({
            "object_id": object_id,
            "kind": kinds.get(object_id),
            "site_count": int(sites.stop - sites.start),
            "frames": frames,
            "label_counts": counts,
            "in_view_fraction": counts["in_view"] / len(times) if len(times) else 0.,
            "mean_visible_site_fraction": float(visible.mean()) if len(times) else 0.,
            "native_active_site_share": shares.get(object_id),
        })
    unknown = sorted(set(events_by_target) - {str(item["object_id"]) for item in target_slices})
    for object_id in unknown:
        for event in events_by_target[object_id]:
            gate_failures.append({"id": str(event.get("id", "")), "object_id": object_id,
                                  "t_s": float(event["t_s"]), "reason": "not_a_declared_target"})
    return {
        "schema": LABELS_SCHEMA,
        "label_method": LOS_METHOD,
        "radar": {"position_m": position.tolist(), "forward": np.asarray(forward, float).tolist(),
                  "up": np.asarray(up, float).tolist(), "beam_half_deg": half,
                  "range_m": [low, high]},
        "in_view_site_fraction": IN_VIEW_SITE_FRACTION,
        "frame_times_s": times.tolist(),
        "targets": targets,
        "key_events": event_results,
        "gate": {"key_events_in_view": not gate_failures, "failures": gate_failures},
    }


def _beam_half_deg(component):
    """First null of a separable pattern (the smaller axis), else half the configured FOV."""
    if str(getattr(component, "pattern_kind", "")) == "separable" and not bool(component.use_default):
        halves = []
        for angles, values in ((component.x_angles_deg, component.x_values),
                               (component.y_angles_deg, component.y_values)):
            nulls = [abs(float(a)) for a, v in zip(angles, values) if float(v) <= 0. and float(a) != 0.]
            # Outside the tabulated support the pattern is zero.
            halves.append(min(nulls) if nulls else max(abs(float(a)) for a in angles))
        return min(halves)
    return float(component.fov) / 2.


def label_visibility(scene, radar_object, *, key_events=(), native_preflight=None):
    """Labels for the Radar's declared targets over its authored measurement interval."""
    from .adapter.common import Derived
    from .adapter.specs import _world_pose
    from .adapter.studio_motion import MultiTargetSkinSampler, target_kind
    from .adapter.target_contract import component_target_specs
    from .adapter.timebase import aligned_frame_count

    component = radar_object.get_component("Radar")
    specs = component_target_specs(component)
    duration, fps, start = (float(component.animation_duration_s), float(component.animation_fps),
                            float(component.t0))
    count = aligned_frame_count(duration, fps)
    if count is None:
        raise ValueError("Radar duration times FPS must be an integer of at least two frames.")
    times = start + np.arange(count, dtype=np.float64) / fps
    sampler = MultiTargetSkinSampler(scene, specs)
    if times[-1] > sampler.duration + 1e-7:
        raise ValueError("Radar interval extends beyond the baked Studio timeline.")
    positions = np.asarray([sampler.sample(float(t))[0] for t in times])
    position, target, up = _world_pose(radar_object)
    forward = np.asarray(target, float) - np.asarray(position, float)
    occluders = StaticOccluders.from_scene(scene, excluded_ids={str(radar_object.id)})
    max_range = float(Derived.compute(component)["max_range_m"])
    result = label_frames(
        radar_position=position, forward=forward, up=up, times_s=times, positions_by_frame=positions,
        target_slices=sampler.target_slices, occluders=occluders, beam_half_deg=_beam_half_deg(component),
        range_m=(MIN_RANGE_M, max_range), key_events=key_events, native_preflight=native_preflight,
        target_kinds={spec["object_id"]: target_kind(scene, spec["object_id"]) for spec in specs})
    result["scene_id"] = str(scene.scene_id)
    result["radar"]["object_id"] = str(radar_object.id)
    return result


__all__ = ["IN_VIEW_SITE_FRACTION", "LABELS", "LABELS_SCHEMA", "label_frames", "label_visibility"]
