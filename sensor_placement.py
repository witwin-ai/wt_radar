"""Bounded sensor-pose selection; invokes existing native preflight unchanged.

All trials happen on a detached Studio scene. No waveform, propagation, site
selection or DSP equations live here, and no live objects are moved by a trial.
"""
import copy
import math
import numpy as np

from witwin_server.tools.base import ToolError


PLACEMENT_OBJECTIVES = (
    'balanced',
    'maximize_range_span',
    'bidirectional_radial_velocity',
    'stronger_doppler',
    'side_view',
    'whole_path_visible',
    'opposite_side',
    'closer_with_full_path_visible',
)


def candidate_positions(aim, height_m, distance_m=None):
    # World Y is up. A bounded ring search around the authored motion avoids
    # inventing a room-origin coordinate before the room has even been built.
    if distance_m is None:
        radii = (1., 1.5)
    else:
        vertical = float(height_m) - float(aim[1])
        horizontal_sq = float(distance_m) ** 2 - vertical ** 2
        if horizontal_sq <= 1e-8:
            raise ToolError(
                'Requested Radar distance is not reachable at the requested height.',
                code='invalid_sensor_placement',
                detail={'distance_m': float(distance_m), 'height_m': float(height_m),
                        'aim_height_m': float(aim[1])},
            )
        radii = (math.sqrt(horizontal_sq),)
    for radius in radii:
        for direction in range(8):
            angle = direction * math.pi / 4
            yield [float(aim[0] + radius * math.cos(angle)), float(height_m),
                   float(aim[2] + radius * math.sin(angle))]


def _static_candidate_constraint(scene, position, *, excluded_ids=()):
    """Conservative room-footprint and furniture-AABB check before native LOS."""
    from .adapter.snapshot import _hierarchy

    matrices, visible = _hierarchy(scene)
    excluded = {str(value) for value in excluded_ids}
    bounds = []
    for object_id, obj in scene.objects.items():
        object_id = str(object_id)
        if object_id in excluded or not visible.get(object_id, True):
            continue
        if obj.get_component('SkinnedMesh') is not None:
            continue
        mesh = obj.get_component('Mesh')
        if mesh is None or mesh.is_empty:
            continue
        vertices = np.asarray(mesh.get_vertices_numpy(), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            continue
        world = (np.column_stack((vertices, np.ones(len(vertices)))) @ matrices[object_id].T)[:, :3]
        lo, hi = world.min(axis=0), world.max(axis=0)
        bounds.append((object_id, str(obj.name), lo, hi))
    point = np.asarray(position, dtype=np.float64)
    floor_candidates = [
        item for item in bounds
        if (item[3][1] - item[2][1]) <= .25
        and item[3][1] <= point[1] + .1
        and (item[3][0] - item[2][0]) * (item[3][2] - item[2][2]) >= 1.
    ]
    if floor_candidates:
        floor = max(
            floor_candidates,
            key=lambda item: (item[3][0] - item[2][0]) * (item[3][2] - item[2][2]),
        )
        if not (floor[2][0] - .05 <= point[0] <= floor[3][0] + .05
                and floor[2][2] - .05 <= point[2] <= floor[3][2] + .05):
            return {
                'code': 'outside_room_footprint',
                'object_id': floor[0],
                'object_name': floor[1],
            }
    for object_id, name, lo, hi in bounds:
        # Treat a 5 cm shell around solid geometry as unavailable. Thin floors
        # remain below a normal Radar height and are therefore not rejected.
        if np.all(point >= lo - .05) and np.all(point <= hi + .05):
            return {
                'code': 'radar_furniture_collision',
                'object_id': object_id,
                'object_name': name,
            }
    return None


def _trajectory_metrics(position, aim, positions_by_frame, times, *, fov_deg, max_doppler_mps):
    """Predict placement-only kinematics from every authored Radar frame.

    This deliberately does not approximate propagation or scattering. Native
    preflight remains the authority for topology/visibility; these values only
    rank poses which already passed that preflight.
    """
    position = np.asarray(position, dtype=np.float64)
    points = np.asarray(positions_by_frame, dtype=np.float64)
    centers = points.mean(axis=1)
    relative = centers - position
    ranges = np.linalg.norm(relative, axis=1)
    if (ranges <= 1e-8).any():
        raise ValueError('Radar candidate intersects the animated target trajectory.')
    los = relative / ranges[:, None]
    if len(times) > 1:
        velocities = np.gradient(centers, np.asarray(times, dtype=np.float64), axis=0)
    else:
        velocities = np.zeros_like(centers)
    radial = np.sum(velocities * los, axis=1)
    tangential = np.linalg.norm(velocities - radial[:, None] * los, axis=1)

    forward = np.asarray(aim, dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    all_relative = points - position[None, None, :]
    all_norm = np.linalg.norm(all_relative, axis=2)
    cosine = np.sum(all_relative * forward[None, None, :], axis=2) / np.maximum(all_norm, 1e-12)
    angles = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    in_fov = angles <= float(fov_deg) / 2.
    peak_radial = float(np.max(np.abs(radial)))
    return {
        'sampled_frame_count': int(len(times)),
        'sampled_site_count': int(points.shape[1]),
        'range_min_m': float(ranges.min()),
        'range_max_m': float(ranges.max()),
        'range_span_m': float(ranges.max() - ranges.min()),
        'mean_range_m': float(ranges.mean()),
        # Positive is receding; negative is approaching.
        'radial_velocity_min_mps': float(radial.min()),
        'radial_velocity_max_mps': float(radial.max()),
        'radial_velocity_rms_mps': float(np.sqrt(np.mean(radial ** 2))),
        'peak_abs_radial_velocity_mps': peak_radial,
        'approaching_peak_mps': float(max(0., -radial.min())),
        'receding_peak_mps': float(max(0., radial.max())),
        'tangential_velocity_rms_mps': float(np.sqrt(np.mean(tangential ** 2))),
        'fov_site_frame_coverage': float(in_fov.mean()),
        'whole_path_in_fov': bool(in_fov.all()),
        'max_off_axis_angle_deg': float(angles.max()),
        'doppler_nyquist_mps': float(max_doppler_mps),
        'doppler_within_nyquist': bool(peak_radial <= float(max_doppler_mps) + 1e-9),
    }


def _objective_score(objective, metrics, *, position, aim, baseline_position):
    native_coverage = float(metrics.get('native_visibility_coverage', 1.))
    fov_coverage = float(metrics['fov_site_frame_coverage'])
    physically_valid = 1. if metrics['doppler_within_nyquist'] else 0.
    coverage = min(native_coverage, fov_coverage)
    if objective == 'maximize_range_span':
        value = metrics['range_span_m']
    elif objective == 'bidirectional_radial_velocity':
        value = min(metrics['approaching_peak_mps'], metrics['receding_peak_mps'])
    elif objective == 'stronger_doppler':
        value = metrics['radial_velocity_rms_mps']
    elif objective == 'side_view':
        value = metrics['tangential_velocity_rms_mps']
    elif objective == 'whole_path_visible':
        value = coverage
    elif objective == 'closer_with_full_path_visible':
        value = -metrics['mean_range_m']
    elif objective == 'opposite_side':
        if baseline_position is None:
            value = 0.
        else:
            current = np.asarray(baseline_position, dtype=np.float64) - np.asarray(aim, dtype=np.float64)
            candidate = np.asarray(position, dtype=np.float64) - np.asarray(aim, dtype=np.float64)
            denom = np.linalg.norm(current) * np.linalg.norm(candidate)
            value = float(-np.dot(current, candidate) / denom) if denom > 1e-9 else 0.
    else:
        value = metrics['range_span_m'] + .25 * metrics['radial_velocity_rms_mps']
    # Physical feasibility is always ranked before the requested measurement
    # preference. Whole-path objectives additionally make full FOV coverage a
    # hard lexicographic preference rather than a soft weighted hint.
    whole_path = 1. if metrics['whole_path_in_fov'] and native_coverage >= 1. - 1e-9 else 0.
    if objective == 'closer_with_full_path_visible':
        return physically_valid, whole_path, coverage, value
    if objective == 'whole_path_visible':
        return physically_valid, whole_path, value, -metrics['mean_range_m']
    return physically_valid, coverage, value, -metrics['mean_range_m']


def choose_sensor_placement(scene, args, *, radar_object_id=None):
    from witwin_server.features.solvers.scene_ref import load_scene_ref, make_scene_ref
    from .library_items import _settings_object
    from .adapter.studio_motion import StudioSkinSampler
    from .adapter.animation import animation_preflight, animation_request, AnimationTopologyError
    from .agent_tools import _look_at_euler

    detached = load_scene_ref(copy.deepcopy(make_scene_ref(scene)))
    sensor = detached.get_object(radar_object_id) if radar_object_id else None
    if sensor is None:
        sensor = _settings_object('Automatic placement probe')
        detached.add_object(sensor)
    radar = sensor.get_component('Radar')
    radar.snapshot_target_id = args['target_object_id']
    radar.t0 = float(args.get('start_s', 0.))
    radar.animation_duration_s = float(args.get('duration_s', 5.))
    radar.animation_fps = float(args.get('fps', 10.))
    radar.fov = float(args.get('fov_deg', 60.))
    radar.view, radar.static_clutter_removal, radar.show_cfar = 'range_doppler', False, False
    sampler = StudioSkinSampler(detached, args['target_object_id'])
    duration_s = float(radar.animation_duration_s)
    fps = float(radar.animation_fps)
    start_s = float(radar.t0)
    frame_count = int(round(duration_s * fps))
    if frame_count < 2 or not np.isclose(frame_count, duration_s * fps,
                                         rtol=0., atol=1e-7):
        raise ToolError('Automatic placement requires duration times FPS to be an integer of at least two frames.',
                        code='invalid_sensor_placement')
    times = start_s + np.arange(frame_count, dtype=np.float64) / fps
    if times[-1] > sampler.duration + 1e-7:
        raise ToolError('Automatic placement interval extends beyond the baked Studio timeline.',
                        code='invalid_sensor_placement')
    positions_by_frame = np.asarray([sampler.positions(float(t)) for t in times])
    aim = positions_by_frame.reshape(-1, 3).mean(axis=0)
    height = float(args.get('height_m', 1.))
    requested_distance = args.get('distance_m')
    distance = float(requested_distance) if requested_distance is not None else None
    objective = str(args.get('placement_objective') or 'balanced')
    if objective not in PLACEMENT_OBJECTIVES:
        raise ToolError(f'Unsupported Radar placement objective: {objective}',
                        code='invalid_sensor_placement')
    if (not np.isfinite(aim).all() or not math.isfinite(height) or not .2 <= height <= 3.
            or (distance is not None and (not math.isfinite(distance) or not .25 <= distance <= 10.))):
        raise ToolError('Automatic placement needs finite target motion and height [0.2,3] m.',
                        code='invalid_sensor_placement')
    from .adapter.common import Derived
    max_doppler_mps = float(Derived.compute(radar)['max_doppler_mps'])
    baseline_position = None
    if radar_object_id:
        live_sensor = scene.get_object(radar_object_id)
        live_transform = live_sensor.get_component('Transform') if live_sensor else None
        if live_transform is not None:
            baseline_position = np.asarray(live_transform.position, dtype=np.float64).tolist()
    failures = []
    valid = []
    ranked = []
    tested_candidate_count = 0
    native_preflight_count = 0
    for position in candidate_positions(aim, height, distance):
        tested_candidate_count += 1
        constraint = _static_candidate_constraint(
            detached,
            position,
            excluded_ids={str(sensor.id), str(args['target_object_id'])},
        )
        if constraint is not None:
            failures.append({'position_m': position, 'reason': constraint['code'], 'detail': constraint})
            continue
        metrics = _trajectory_metrics(
            position, aim, positions_by_frame, times,
            fov_deg=float(radar.fov), max_doppler_mps=max_doppler_mps,
        )
        # Native coverage cannot exceed one.  This optimistic score is therefore
        # a true upper bound on the final score and lets explicit objectives use
        # branch-and-bound without changing which candidate is selected.
        optimistic_metrics = dict(metrics)
        optimistic_metrics['native_visibility_coverage'] = 1.
        optimistic_metrics['occluded_frame_count'] = 0
        candidate = {
            'position_m': position,
            'metrics': metrics,
            'optimistic_score': _objective_score(
                objective, optimistic_metrics, position=position, aim=aim,
                baseline_position=baseline_position,
            ),
        }
        ranked.append(candidate)

        # Preserve the historical deterministic first-valid behavior for the
        # default objective. Explicit objectives first rank the whole bounded
        # set cheaply, below, before invoking expensive native visibility.
        if objective == 'balanced':
            detached.update_transform(
                str(sensor.id), position=position,
                rotation=_look_at_euler(np.asarray(position), aim),
            )
            native_preflight_count += 1
            try:
                evidence = animation_preflight(detached, animation_request(radar))
            except AnimationTopologyError as exc:
                failures.append({'position_m': position, 'reason': str(exc), 'detail': exc.detail})
                continue
            candidate['native_preflight'] = evidence
            valid.append(candidate)
            break

    if objective != 'balanced':
        # Python's stable sort keeps candidate generation order as the
        # deterministic tie-breaker. Evaluate the strongest optimistic bound
        # first and stop once no remaining candidate can beat the best proven
        # native-visible score.
        ranked.sort(key=lambda item: item['optimistic_score'], reverse=True)
        best_score = None
        for candidate in ranked:
            if best_score is not None and candidate['optimistic_score'] <= best_score:
                break
            position = candidate['position_m']
            detached.update_transform(
                str(sensor.id), position=position,
                rotation=_look_at_euler(np.asarray(position), aim),
            )
            native_preflight_count += 1
            try:
                evidence = animation_preflight(detached, animation_request(radar))
            except AnimationTopologyError as exc:
                failures.append({'position_m': position, 'reason': str(exc), 'detail': exc.detail})
                continue
            metrics = dict(candidate['metrics'])
            topology = evidence.get('topology') if isinstance(evidence, dict) else None
            metrics['native_visibility_coverage'] = float(
                topology.get('visibility_coverage', 1.) if isinstance(topology, dict) else 1.
            )
            frame_visibility = topology.get('frames', ()) if isinstance(topology, dict) else ()
            declared_sites = int(topology.get('declared_site_count', 0) or 0) if isinstance(topology, dict) else 0
            metrics['occluded_frame_count'] = sum(
                int(frame.get('reachable_site_count', declared_sites)) < declared_sites
                for frame in frame_visibility
            ) if declared_sites else 0
            candidate['metrics'] = metrics
            candidate['native_preflight'] = evidence
            candidate['score'] = _objective_score(
                objective, metrics, position=position, aim=aim,
                baseline_position=baseline_position,
            )
            valid.append(candidate)
            if best_score is None or candidate['score'] > best_score:
                best_score = candidate['score']

    # Balanced candidates are validated inline and do not need the optimistic
    # score after their first native-visible pose is found.
    for candidate in valid:
        if 'score' not in candidate:
            metrics = dict(candidate['metrics'])
            evidence = candidate['native_preflight']
            topology = evidence.get('topology') if isinstance(evidence, dict) else None
            metrics['native_visibility_coverage'] = float(
                topology.get('visibility_coverage', 1.) if isinstance(topology, dict) else 1.
            )
            frame_visibility = topology.get('frames', ()) if isinstance(topology, dict) else ()
            declared_sites = int(topology.get('declared_site_count', 0) or 0) if isinstance(topology, dict) else 0
            metrics['occluded_frame_count'] = sum(
                int(frame.get('reachable_site_count', declared_sites)) < declared_sites
                for frame in frame_visibility
            ) if declared_sites else 0
            candidate['metrics'] = metrics
            candidate['score'] = _objective_score(
                objective, metrics, position=candidate['position_m'], aim=aim,
                baseline_position=baseline_position,
            )
    if valid:
        selected = max(valid, key=lambda item: item['score'])
        selected_metrics = selected['metrics']
        impossible_reason = None
        if not selected_metrics['doppler_within_nyquist']:
            impossible_reason = 'Every visible candidate exceeds the Radar Doppler Nyquist boundary.'
        elif objective == 'bidirectional_radial_velocity' and (
                selected_metrics['approaching_peak_mps'] <= 1e-4
                or selected_metrics['receding_peak_mps'] <= 1e-4):
            impossible_reason = 'No visible candidate observes both approaching and receding radial motion.'
        elif objective in {'whole_path_visible', 'closer_with_full_path_visible'} and not (
                selected_metrics['whole_path_in_fov']
                and selected_metrics['native_visibility_coverage'] >= 1. - 1e-9):
            impossible_reason = 'No candidate keeps the whole authored path in both FOV and native LOS coverage.'
        elif objective == 'opposite_side' and baseline_position is None:
            impossible_reason = 'Opposite-side placement needs one existing Radar pose as its reference.'
        if impossible_reason:
            raise ToolError(
                impossible_reason + ' Keep the current pose, relax the objective, or choose an explicit pose.',
                code='sensor_placement_objective_unavailable',
                detail={
                    'placement_objective': objective,
                    'best_candidate_position_m': selected['position_m'],
                    'best_candidate_metrics': selected_metrics,
                    'valid_candidate_count': len(valid),
                    'tradeoff_options': ['keep_current_pose', 'relax_objective', 'choose_explicit_pose'],
                },
            )
        baseline_metrics = None
        if baseline_position is not None:
            try:
                baseline_metrics = _trajectory_metrics(
                    baseline_position, aim, positions_by_frame, times,
                    fov_deg=float(radar.fov), max_doppler_mps=max_doppler_mps,
                )
            except ValueError:
                baseline_metrics = None
        improvement = None
        if baseline_metrics is not None:
            improvement = {
                'range_span_delta_m': float(
                    selected_metrics['range_span_m'] - baseline_metrics['range_span_m']),
                'radial_velocity_rms_delta_mps': float(
                    selected_metrics['radial_velocity_rms_mps']
                    - baseline_metrics['radial_velocity_rms_mps']),
                'tangential_velocity_rms_delta_mps': float(
                    selected_metrics['tangential_velocity_rms_mps']
                    - baseline_metrics['tangential_velocity_rms_mps']),
                'fov_coverage_delta': float(
                    selected_metrics['fov_site_frame_coverage']
                    - baseline_metrics['fov_site_frame_coverage']),
                'mean_range_delta_m': float(
                    selected_metrics['mean_range_m'] - baseline_metrics['mean_range_m']),
            }
        return {'position_m': selected['position_m'], 'aim_point_m': aim.tolist(),
                'mode': 'automatic', 'height_m': height,
                'placement_objective': objective,
                'requested_distance_m': distance,
                'actual_distance_m': float(np.linalg.norm(np.asarray(selected['position_m']) - aim)),
                'tested_candidates': tested_candidate_count,
                'valid_candidate_count': len(valid),
                'native_preflight_count': native_preflight_count,
                'predicted_metrics': selected_metrics,
                'baseline_metrics': baseline_metrics,
                'predicted_improvement': improvement,
                'metric_semantics': {
                    'radial_velocity_sign': 'positive=receding, negative=approaching',
                    'visibility': 'FOV prediction ranks poses; native preflight remains authoritative',
                    'trajectory_sampling': 'every requested baked Radar frame',
                },
                'native_preflight': selected['native_preflight'],
                'rejected_candidates': failures}
    raise ToolError('No automatic sensor pose passed native visibility preflight; choose an explicit pose.',
                    code='sensor_placement_unavailable', detail={'candidates': failures})
