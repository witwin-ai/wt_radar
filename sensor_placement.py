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

# Whole-path objectives plan for the union of every declared target over the
# full requested interval: each target's root trajectory must stay this far
# inside the FOV cone, and the pose keeps a tripod footprint clear of furniture
# and room edges instead of hugging one actor.
WHOLE_PATH_OBJECTIVES = frozenset({'whole_path_visible', 'closer_with_full_path_visible'})
FOV_MARGIN_DEG = 5.
# Share of the requested frames whose root must sit inside the FOV margin.
MIN_PATH_FOV_FRACTION = .7
FURNITURE_CLEARANCE_M = .3
ROOM_EDGE_CLEARANCE_M = .3
MIN_SITE_RANGE_M = 1.
PREFERRED_RANGE_M = 3.
_MARGIN_PREFERENCE_CAP_DEG = 10.
FLOOR_GRID_STEP_M = .5
# Native preflight compiles the full-room trace; bound it for whole-path search.
MAX_WHOLE_PATH_NATIVE_PREFLIGHTS = 12


def joint_candidate_positions(root_points, height_m, *, floor=None):
    """Rings around the joint root-trajectory centre plus a grid over the floor.

    Actors standing near a wall leave most ring samples outside the room, so
    the open floor is sampled too; distance to every actor is enforced later.
    """
    points = np.asarray(root_points, dtype=np.float64).reshape(-1, 3)
    lo, hi = points.min(axis=0), points.max(axis=0)
    centre = (lo + hi) / 2.
    seen = set()

    def emit(x, z):
        key = (round(float(x), 2), round(float(z), 2))
        if key in seen:
            return None
        seen.add(key)
        return [float(x), float(height_m), float(z)]

    for radius in (1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.):
        for direction in range(24):
            angle = direction * math.pi / 12
            point = emit(centre[0] + radius * math.cos(angle), centre[2] + radius * math.sin(angle))
            if point is not None:
                yield point
    if floor is None:
        return
    inset = ROOM_EDGE_CLEARANCE_M
    xs = np.arange(float(floor[2][0]) + inset, float(floor[3][0]) - inset + 1e-9, FLOOR_GRID_STEP_M)
    zs = np.arange(float(floor[2][2]) + inset, float(floor[3][2]) - inset + 1e-9, FLOOR_GRID_STEP_M)
    for x in xs:
        for z in zs:
            point = emit(x, z)
            if point is not None:
                yield point


def candidate_positions(aim, height_m, distance_m=None):
    # World Y is up. A bounded ring search around the authored motion avoids
    # inventing a room-origin coordinate before the room has even been built.
    if distance_m is None:
        # A long or laterally wide trajectory may not fit in a 60-degree FOV
        # from the historical 1-1.5 m rings. Include farther in-room rings so
        # whole_path_visible can satisfy the hard all-frame/all-site contract
        # instead of failing at an otherwise fully visible pose.
        radii = (1., 1.5, 2., 2.5, 3.)
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
        # Half-cardinal directions were too coarse for long paths in furnished
        # rooms: a valid sight line can exist between two 45-degree samples.
        # Sixteen deterministic bearings remain bounded while resolving those
        # narrow door/furniture visibility corridors.
        for direction in range(16):
            angle = direction * math.pi / 8
            yield [float(aim[0] + radius * math.cos(angle)), float(height_m),
                   float(aim[2] + radius * math.sin(angle))]


def _angular_centre_aim(position, positions_by_frame):
    """Aim at the centre of observed directions, not the Cartesian centroid.

    For a long trajectory, looking at its mean world point can waste several
    degrees of FOV on the near end. Averaging unit sight directions centres the
    actual angular envelope while remaining independent of mesh names/maps.
    """
    position = np.asarray(position, dtype=np.float64)
    points = np.asarray(positions_by_frame, dtype=np.float64).reshape(-1, 3)
    rays = points - position
    ranges = np.linalg.norm(rays, axis=1)
    if (ranges <= 1e-8).any():
        raise ValueError('Radar candidate intersects the animated target trajectory.')
    direction = np.mean(rays / ranges[:, None], axis=0)
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        return points.mean(axis=0)
    return position + direction / length * float(np.mean(ranges))


def _minimax_aim(position, positions_by_frame, iterations=64):
    """Aim at the centre of the smallest cone holding every target site.

    Averaging directions weights each actor by its site count, so a densely
    sampled person pulls the boresight away from a sparsely sampled cat. The
    Badoiu-Clarkson iteration instead converges to the direction minimising the
    largest off-axis angle of the joint trajectory, deterministically.
    """
    position = np.asarray(position, dtype=np.float64)
    points = np.asarray(positions_by_frame, dtype=np.float64).reshape(-1, 3)
    rays = points - position
    ranges = np.linalg.norm(rays, axis=1)
    if (ranges <= 1e-8).any():
        raise ValueError('Radar candidate intersects the animated target trajectory.')
    units = rays / ranges[:, None]
    direction = units.mean(axis=0)
    if np.linalg.norm(direction) <= 1e-12:
        return points.mean(axis=0)
    direction /= np.linalg.norm(direction)
    for step in range(1, iterations + 1):
        farthest = units[int(np.argmin(units @ direction))]
        direction = direction + (farthest - direction) / (step + 1)
        direction /= np.linalg.norm(direction)
    return position + direction * float(np.mean(ranges))


def _static_bounds(scene, *, excluded_ids=()):
    """World AABBs of static solid meshes (actors and excluded ids skipped)."""
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
    return bounds


def _floor_bounds(bounds, height_m):
    floor_candidates = [
        item for item in bounds
        if (item[3][1] - item[2][1]) <= .25
        and item[3][1] <= float(height_m) + .1
        and (item[3][0] - item[2][0]) * (item[3][2] - item[2][2]) >= 1.
    ]
    if not floor_candidates:
        return None
    return max(
        floor_candidates,
        key=lambda item: (item[3][0] - item[2][0]) * (item[3][2] - item[2][2]),
    )


def _static_candidate_constraint(scene, position, *, excluded_ids=(), bounds=None):
    """Conservative room-footprint and furniture-AABB check before native LOS."""
    if bounds is None:
        bounds = _static_bounds(scene, excluded_ids=excluded_ids)
    point = np.asarray(position, dtype=np.float64)
    floor = _floor_bounds(bounds, point[1])
    if floor is not None:
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


def _clearance(bounds, floor, position):
    """Horizontal tripod clearance to furniture and to the room footprint edge.

    A floor-standing Radar occupies the column from just above the floor to its
    own height, so rugs below that column and ceilings above it do not block it.
    """
    point = np.asarray(position, dtype=np.float64)
    floor_top = float(floor[3][1]) if floor is not None else min(
        (float(item[2][1]) for item in bounds), default=float(point[1]) - 1.)
    nearest = None
    for object_id, name, lo, hi in bounds:
        if floor is not None and object_id == floor[0]:
            continue
        if hi[1] <= floor_top + .1 or lo[1] >= point[1] + .2:
            continue
        dx = max(lo[0] - point[0], 0., point[0] - hi[0])
        dz = max(lo[2] - point[2], 0., point[2] - hi[2])
        distance = float(math.hypot(dx, dz))
        if nearest is None or distance < nearest[0]:
            nearest = (distance, object_id, name)
    room_edge = None
    if floor is not None:
        room_edge = float(min(point[0] - floor[2][0], floor[3][0] - point[0],
                              point[2] - floor[2][2], floor[3][2] - point[2]))
    return {
        'nearest_furniture_m': nearest[0] if nearest else None,
        'nearest_furniture_id': nearest[1] if nearest else None,
        'nearest_furniture_name': nearest[2] if nearest else None,
        'room_edge_m': room_edge,
        'required_furniture_clearance_m': FURNITURE_CLEARANCE_M,
        'required_room_edge_clearance_m': ROOM_EDGE_CLEARANCE_M,
    }


def _whole_path_constraint(clearance):
    if clearance['room_edge_m'] is not None and clearance['room_edge_m'] < ROOM_EDGE_CLEARANCE_M:
        return {'code': 'room_edge_clearance', 'room_edge_m': clearance['room_edge_m']}
    if (clearance['nearest_furniture_m'] is not None
            and clearance['nearest_furniture_m'] < FURNITURE_CLEARANCE_M):
        return {
            'code': 'radar_furniture_clearance',
            'object_id': clearance['nearest_furniture_id'],
            'object_name': clearance['nearest_furniture_name'],
            'distance_m': clearance['nearest_furniture_m'],
        }
    return None


def _trajectory_metrics(position, aim, positions_by_frame, times, *, fov_deg,
                        max_doppler_mps, velocities_by_frame=None):
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
    site_velocities = None
    if velocities_by_frame is not None:
        site_velocities = np.asarray(velocities_by_frame, dtype=np.float64)
        if site_velocities.shape != points.shape or not np.isfinite(site_velocities).all():
            raise ValueError('Target site velocities must be finite and match sampled positions.')
        velocities = site_velocities.mean(axis=1)
    elif len(times) > 1:
        velocities = np.gradient(centers, np.asarray(times, dtype=np.float64), axis=0)
    else:
        velocities = np.zeros_like(centers)
    radial = np.sum(velocities * los, axis=1)
    tangential = np.linalg.norm(velocities - radial[:, None] * los, axis=1)

    forward = np.asarray(aim, dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    all_relative = points - position[None, None, :]
    all_norm = np.linalg.norm(all_relative, axis=2)
    if (all_norm <= 1e-8).any():
        raise ValueError('Radar candidate intersects an animated target site.')
    if site_velocities is None:
        if len(times) > 1:
            site_velocities = np.gradient(
                points, np.asarray(times, dtype=np.float64), axis=0)
        else:
            site_velocities = np.zeros_like(points)
    site_radial = np.sum(
        site_velocities * all_relative / all_norm[:, :, None], axis=2)
    cosine = np.sum(all_relative * forward[None, None, :], axis=2) / np.maximum(all_norm, 1e-12)
    angles = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    in_fov = angles <= float(fov_deg) / 2.
    visible_frames = in_fov.any(axis=1)
    per_frame_site_fraction = in_fov.mean(axis=1)
    continuous_fov_ranks = np.flatnonzero(in_fov.all(axis=0)).tolist()
    centre_peak_radial = float(np.max(np.abs(radial)))
    site_peak_radial = float(np.max(np.abs(site_radial)))
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
        'peak_abs_radial_velocity_mps': site_peak_radial,
        'centre_peak_abs_radial_velocity_mps': centre_peak_radial,
        'approaching_peak_mps': float(max(0., -radial.min())),
        'receding_peak_mps': float(max(0., radial.max())),
        'tangential_velocity_rms_mps': float(np.sqrt(np.mean(tangential ** 2))),
        'fov_site_frame_coverage': float(in_fov.mean()),
        # "Whole path" means the animated target remains observable in every
        # frame. It does not mean every sampled skin site must be visible from
        # one side of an articulated animal; that would reject ordinary
        # self-occlusion even while dozens of sites remain continuously valid.
        'path_frame_coverage': float(visible_frames.mean()),
        'minimum_fov_site_fraction': float(per_frame_site_fraction.min()),
        'fov_continuous_site_count': len(continuous_fov_ranks),
        'fov_continuous_site_ranks': continuous_fov_ranks,
        'whole_path_in_fov': bool(continuous_fov_ranks),
        'max_off_axis_angle_deg': float(angles.max()),
        'doppler_nyquist_mps': float(max_doppler_mps),
        'doppler_within_nyquist': bool(site_peak_radial <= float(max_doppler_mps) + 1e-9),
    }


def _target_coverage(position, aim, positions_by_frame, target_slices, *, fov_deg,
                     margin_deg=FOV_MARGIN_DEG, min_path_fraction=MIN_PATH_FOV_FRACTION):
    """Per-target FOV evidence over the full interval.

    A target's root trajectory (its per-frame site centroid) must stay
    ``margin_deg`` inside the FOV for at least ``min_path_fraction`` of the
    requested frames. Every target is judged on its own path; one visible actor
    never stands in for another one.
    """
    position = np.asarray(position, dtype=np.float64)
    points = np.asarray(positions_by_frame, dtype=np.float64)
    forward = np.asarray(aim, dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    half = float(fov_deg) / 2.

    def off_axis(vectors):
        norms = np.linalg.norm(vectors, axis=-1)
        cosine = np.sum(vectors * forward, axis=-1) / np.maximum(norms, 1e-12)
        return np.degrees(np.arccos(np.clip(cosine, -1., 1.))), norms

    targets = []
    for item in target_slices:
        start, end = int(item['site_start']), int(item['site_end'])
        sites = points[:, start:end]
        site_angles, site_ranges = off_axis(sites - position)
        root = sites.mean(axis=1)
        root_angles, _ = off_axis(root - position)
        continuous = (np.flatnonzero((site_angles <= half).all(axis=0)) + start).tolist()
        root_in_fov = root_angles <= half - float(margin_deg)
        targets.append({
            'object_id': str(item['object_id']),
            'site_start': start,
            'site_end': end,
            'frame_count': int(len(points)),
            'root_frames_in_fov': int(root_in_fov.sum()),
            'path_fov_fraction': float(root_in_fov.mean()),
            'required_path_fov_fraction': float(min_path_fraction),
            'min_root_margin_deg': float(half - root_angles.max()),
            'site_envelope_margin_deg': float(half - site_angles.max()),
            'continuous_fov_site_ranks': continuous,
            'continuous_fov_site_count': len(continuous),
            'min_site_range_m': float(site_ranges.min()),
            'max_site_range_m': float(site_ranges.max()),
            'root_path_length_m': float(np.linalg.norm(np.diff(root, axis=0), axis=1).sum()),
            'root_bounds_m': {'min': root.min(axis=0).tolist(), 'max': root.max(axis=0).tolist()},
            'fov_whole_path': bool(root_in_fov.mean() >= float(min_path_fraction) - 1e-9),
        })
    return {
        'targets': targets,
        'all_targets_whole_path_in_fov': all(item['fov_whole_path'] for item in targets),
        'min_root_margin_deg': min(item['min_root_margin_deg'] for item in targets),
        'min_path_fov_fraction': min(item['path_fov_fraction'] for item in targets),
        'site_envelope_margin_deg': min(item['site_envelope_margin_deg'] for item in targets),
        'min_site_range_m': min(item['min_site_range_m'] for item in targets),
    }


def _apply_native_evidence(metrics, evidence, target_slices=None):
    """Fold one native preflight into a candidate's kinematic metrics."""
    metrics = dict(metrics)
    topology = evidence.get('topology') if isinstance(evidence, dict) else None
    metrics['native_visibility_coverage'] = float(
        topology.get('visibility_coverage', 1.) if isinstance(topology, dict) else 1.
    )
    frame_visibility = topology.get('frames', ()) if isinstance(topology, dict) else ()
    declared_sites = int(topology.get('declared_site_count', 0) or 0) if isinstance(topology, dict) else 0
    active_sites = int(topology.get('active_site_count', 0) or 0) if isinstance(topology, dict) else 0
    if not active_sites and metrics['native_visibility_coverage'] > 0.:
        active_sites = max(1, round(metrics['native_visibility_coverage'] * declared_sites)) if declared_sites else 1
    has_native_ranks = isinstance(topology, dict) and 'active_site_ranks' in topology
    native_ranks = list(topology.get('active_site_ranks', ())) if isinstance(topology, dict) else []
    if not native_ranks and active_sites:
        native_ranks = list(range(active_sites))
    fov_ranks = metrics.get('fov_continuous_site_ranks')
    if fov_ranks is None:
        fov_ranks = (range(int(metrics.get('sampled_site_count', 0)))
                     if metrics.get('whole_path_in_fov') else ())
    joint_ranks = sorted(set(native_ranks) & set(fov_ranks))
    metrics['native_continuous_site_count'] = active_sites
    metrics['joint_continuous_site_count'] = len(joint_ranks)
    metrics['native_whole_path_visible'] = bool(joint_ranks)
    metrics['occluded_frame_count'] = sum(
        int(frame.get('reachable_site_count', declared_sites)) == 0
        for frame in frame_visibility
    ) if frame_visibility else 0
    metrics['partially_occluded_frame_count'] = sum(
        0 < int(frame.get('reachable_site_count', declared_sites)) < declared_sites
        for frame in frame_visibility
    ) if declared_sites else 0

    coverage = metrics.get('coverage')
    if coverage is not None:
        per_target_counts = {
            str(item.get('object_id')): int(item.get('active_site_count', 0) or 0)
            for item in (evidence.get('targets') or ()) if isinstance(item, dict)
        } if isinstance(evidence, dict) else {}
        native_set = set(int(rank) for rank in native_ranks)
        targets = []
        for item in coverage['targets']:
            item = dict(item)
            in_slice = {rank for rank in native_set if item['site_start'] <= rank < item['site_end']}
            # The native solver only simulates sites reachable on every frame;
            # a target without any such site contributes no Radar return.
            if item['object_id'] in per_target_counts:
                native_count = per_target_counts[item['object_id']]
            else:
                native_count = len(in_slice) if has_native_ranks else (
                    item['site_end'] - item['site_start'] if active_sites else 0)
            joint = native_count
            item['native_active_site_count'] = int(native_count)
            item['joint_continuous_site_count'] = int(joint)
            item['whole_path_visible'] = bool(item['fov_whole_path'] and joint > 0)
            targets.append(item)
        coverage = {**coverage, 'targets': targets}
        coverage['targets_without_whole_path'] = [
            item['object_id'] for item in targets if not item['whole_path_visible']]
        coverage['all_targets_whole_path_visible'] = not coverage['targets_without_whole_path']
        metrics['coverage'] = coverage
        metrics['all_targets_whole_path_visible'] = coverage['all_targets_whole_path_visible']
    return metrics


def _objective_score(objective, metrics, *, position, aim, baseline_position):
    native_coverage = float(metrics.get('native_visibility_coverage', 1.))
    fov_coverage = float(metrics.get(
        'path_frame_coverage', metrics['fov_site_frame_coverage']))
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
    native_whole_path = bool(metrics.get(
        'native_whole_path_visible', native_coverage > 0.))
    whole_path = 1. if metrics['whole_path_in_fov'] and native_whole_path else 0.
    joint = metrics.get('all_targets_whole_path_visible')
    if joint is None:
        # Before native preflight only the FOV half of the contract is known.
        joint = metrics.get('all_targets_whole_path_in_fov', metrics['whole_path_in_fov']) and native_whole_path
    joint_whole_path = 1. if joint else 0.
    if objective == 'closer_with_full_path_visible':
        return physically_valid, joint_whole_path, coverage, value
    if objective == 'whole_path_visible':
        if 'coverage' not in metrics:
            return physically_valid, whole_path, value, -metrics['mean_range_m']
        # Every target covered first, then the larger worst per-target path
        # share, then the whole site envelope inside the margin, then a larger
        # root margin (capped so it does not simply push the Radar to the
        # farthest ring), then the preferred range.
        joint_coverage = metrics['coverage']
        path_share = round(float(joint_coverage['min_path_fov_fraction']), 2)
        envelope = 1. if joint_coverage['site_envelope_margin_deg'] >= FOV_MARGIN_DEG else 0.
        margin = round(min(joint_coverage['min_root_margin_deg'], _MARGIN_PREFERENCE_CAP_DEG), 1)
        preferred = -abs(float(metrics['joint_centre_range_m']) - PREFERRED_RANGE_M)
        return physically_valid, joint_whole_path, path_share, envelope, margin, preferred
    return physically_valid, coverage, value, -metrics['mean_range_m']


def _per_target_failure_detail(ranked, positions_by_frame, target_slices, *, fov_deg):
    """Best kinematic pose for each target alone, aimed at that target only.

    It justifies a multiple-Radar suggestion; each such pose still needs its own
    native preflight once a second Radar is actually configured.
    """
    per_target = []
    for item in target_slices:
        own = positions_by_frame[:, item['site_start']:item['site_end']]
        local = [{**item, 'site_start': 0, 'site_end': own.shape[1]}]
        best = None
        for candidate in ranked:
            own_aim = _minimax_aim(candidate['position_m'], own)
            coverage = _target_coverage(candidate['position_m'], own_aim, own, local, fov_deg=fov_deg)
            target = coverage['targets'][0]
            if target['fov_whole_path'] and (best is None or target['min_root_margin_deg'] > best[1]):
                best = (candidate['position_m'], target['min_root_margin_deg'], own_aim.tolist())
        per_target.append({
            'object_id': str(item['object_id']),
            'best_single_target_pose_m': best[0] if best else None,
            'best_single_target_aim_m': best[2] if best else None,
            'best_single_target_root_margin_deg': best[1] if best else None,
            'verification': 'kinematic_fov_only_native_preflight_pending',
        })
    return per_target


def choose_sensor_placement(scene, args, *, radar_object_id=None):
    from witwin_server.features.solvers.scene_ref import load_scene_ref, make_scene_ref
    from .library_items import _settings_object
    from .adapter.studio_motion import MultiTargetSkinSampler
    from .adapter.animation import animation_preflight, animation_request, AnimationTopologyError
    from .adapter.target_contract import encode_target_specs, normalize_target_specs
    from .agent_tools import _configure_directional_pattern, _look_at_euler

    detached = load_scene_ref(copy.deepcopy(make_scene_ref(scene)))
    sensor = detached.get_object(radar_object_id) if radar_object_id else None
    if sensor is None:
        sensor = _settings_object('Automatic placement probe')
        detached.add_object(sensor)
    radar = sensor.get_component('Radar')
    declarations = args.get('targets')
    if declarations is None:
        declarations = [{
            'object_id': args['target_object_id'],
            'site_budget': args.get('target_site_budget', 24),
        }]
    target_specs = normalize_target_specs(detached, declarations)
    target_ids = [spec['object_id'] for spec in target_specs]
    radar.snapshot_target_id = target_ids[0]
    radar.animation_targets_json = encode_target_specs(target_specs)
    radar.t0 = float(args.get('start_s', 0.))
    radar.animation_duration_s = float(args.get('duration_s', 5.))
    radar.animation_fps = float(args.get('fps', 10.))
    radar.target_multipath = bool(args.get('target_multipath', False))
    radar.target_motion_sampling = str(args.get('target_motion_sampling', 'chirp'))
    radar.target_site_budget = int(args.get('target_site_budget', 24))
    radar.fov = float(args.get('fov_deg', 60.))
    _configure_directional_pattern(radar, radar.fov)
    radar.view, radar.static_clutter_removal, radar.show_cfar = 'range_doppler', False, False
    sampler = MultiTargetSkinSampler(detached, target_specs)
    duration_s = float(radar.animation_duration_s)
    fps = float(radar.animation_fps)
    start_s = float(radar.t0)
    from .adapter.timebase import aligned_frame_count
    frame_count = aligned_frame_count(duration_s, fps)
    if frame_count is None:
        raise ToolError('Automatic placement requires duration times FPS to be an integer of at least two frames.',
                        code='invalid_sensor_placement')
    times = start_s + np.arange(frame_count, dtype=np.float64) / fps
    if times[-1] > sampler.duration + 1e-7:
        raise ToolError('Automatic placement interval extends beyond the baked Studio timeline.',
                        code='invalid_sensor_placement')
    sampled_motion = [sampler.sample(float(t)) for t in times]
    positions_by_frame = np.asarray([sample[0] for sample in sampled_motion])
    velocities_by_frame = np.asarray([sample[1] for sample in sampled_motion])
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

    whole_path = objective in WHOLE_PATH_OBJECTIVES
    target_slices = list(getattr(sampler, 'target_slices', ()) or [{
        'object_id': target_ids[0], 'site_start': 0, 'site_end': int(positions_by_frame.shape[1]),
    }])
    root_points = np.concatenate([
        positions_by_frame[:, item['site_start']:item['site_end']].mean(axis=1)
        for item in target_slices
    ])
    joint_centre = (root_points.min(axis=0) + root_points.max(axis=0)) / 2.
    static_bounds = _static_bounds(detached, excluded_ids={str(sensor.id), *target_ids})
    floor = _floor_bounds(static_bounds, height)
    positions = (joint_candidate_positions(root_points, height, floor=floor)
                 if whole_path and distance is None
                 else candidate_positions(aim, height, distance))

    failures = []
    valid = []
    ranked = []
    tested_candidate_count = 0
    native_preflight_count = 0
    for position in positions:
        tested_candidate_count += 1
        constraint = _static_candidate_constraint(detached, position, bounds=static_bounds)
        clearance = _clearance(static_bounds, floor, position) if whole_path else None
        if constraint is None and clearance is not None:
            constraint = _whole_path_constraint(clearance)
        if constraint is not None:
            failures.append({'position_m': position, 'reason': constraint['code'], 'detail': constraint})
            continue
        if distance is not None:
            candidate_aim = aim
        elif whole_path:
            candidate_aim = _minimax_aim(position, positions_by_frame)
        else:
            candidate_aim = _angular_centre_aim(position, positions_by_frame)
        metrics = _trajectory_metrics(
            position, candidate_aim, positions_by_frame, times,
            fov_deg=float(radar.fov), max_doppler_mps=max_doppler_mps,
            velocities_by_frame=velocities_by_frame,
        )
        if whole_path:
            coverage = _target_coverage(
                position, candidate_aim, positions_by_frame, target_slices, fov_deg=float(radar.fov))
            coverage['clearance'] = clearance
            coverage['fov_deg'] = float(radar.fov)
            coverage['required_root_margin_deg'] = FOV_MARGIN_DEG
            coverage['joint_trajectory_bounds_m'] = {
                'min': root_points.min(axis=0).tolist(), 'max': root_points.max(axis=0).tolist()}
            coverage['joint_trajectory_centre_m'] = joint_centre.tolist()
            coverage['aim'] = 'minimax angular centre of every target site over the full interval'
            metrics['coverage'] = coverage
            metrics['all_targets_whole_path_in_fov'] = coverage['all_targets_whole_path_in_fov']
            metrics['joint_centre_range_m'] = float(np.linalg.norm(np.asarray(position) - joint_centre))
            if coverage['min_site_range_m'] < MIN_SITE_RANGE_M:
                failures.append({
                    'position_m': position,
                    'reason': 'too_close_to_target',
                    'detail': {'code': 'too_close_to_target', 'min_site_range_m': coverage['min_site_range_m']},
                })
                continue
        if not metrics['doppler_within_nyquist']:
            failures.append({
                'position_m': position,
                'reason': 'doppler_nyquist_exceeded',
                'detail': {
                    'code': 'doppler_nyquist_exceeded',
                    'peak_abs_radial_velocity_mps': metrics['peak_abs_radial_velocity_mps'],
                    'max_unambiguous_speed_mps': max_doppler_mps,
                },
            })
            continue
        # Native coverage cannot exceed one.  This optimistic score is therefore
        # a true upper bound on the final score and lets explicit objectives use
        # branch-and-bound without changing which candidate is selected.
        optimistic_metrics = dict(metrics)
        optimistic_metrics['native_visibility_coverage'] = 1.
        optimistic_metrics['occluded_frame_count'] = 0
        candidate = {
            'position_m': position,
            'aim_point_m': np.asarray(candidate_aim, dtype=np.float64).tolist(),
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
                rotation=_look_at_euler(
                    np.asarray(position), np.asarray(candidate['aim_point_m'])),
            )
            native_preflight_count += 1
            try:
                evidence = animation_preflight(detached, animation_request(radar), sampled_motion=(
                    times, positions_by_frame, velocities_by_frame))
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
            if whole_path and native_preflight_count >= MAX_WHOLE_PATH_NATIVE_PREFLIGHTS:
                break
            position = candidate['position_m']
            detached.update_transform(
                str(sensor.id), position=position,
                rotation=_look_at_euler(
                    np.asarray(position), np.asarray(candidate['aim_point_m'])),
            )
            native_preflight_count += 1
            try:
                evidence = animation_preflight(detached, animation_request(radar), sampled_motion=(
                    times, positions_by_frame, velocities_by_frame))
            except AnimationTopologyError as exc:
                failures.append({'position_m': position, 'reason': str(exc), 'detail': exc.detail})
                continue
            candidate['metrics'] = _apply_native_evidence(candidate['metrics'], evidence)
            candidate['native_preflight'] = evidence
            candidate['score'] = _objective_score(
                objective, candidate['metrics'], position=position, aim=aim,
                baseline_position=baseline_position,
            )
            valid.append(candidate)
            if best_score is None or candidate['score'] > best_score:
                best_score = candidate['score']
            if objective in WHOLE_PATH_OBJECTIVES and candidate['score'][1] == 1.:
                # Candidates are already sorted by their cheap full-trajectory
                # FOV bound (and by distance for the closer objective). Once
                # native LOS confirms every target's continuous in-FOV site,
                # the requested hard guarantee is satisfied. Exhaustively
                # compiling native propagation for every remaining pose made a
                # 166-frame placement take many minutes without improving
                # correctness.
                break

    # Balanced candidates are validated inline and do not need the optimistic
    # score after their first native-visible pose is found.
    for candidate in valid:
        if 'score' not in candidate:
            candidate['metrics'] = _apply_native_evidence(
                candidate['metrics'], candidate['native_preflight'])
            candidate['score'] = _objective_score(
                objective, candidate['metrics'], position=candidate['position_m'], aim=aim,
                baseline_position=baseline_position,
            )
    if valid:
        selected = max(valid, key=lambda item: item['score'])
        selected_metrics = selected['metrics']
        impossible_reason = None
        extra_detail = {}
        if not selected_metrics['doppler_within_nyquist']:
            impossible_reason = 'Every visible candidate exceeds the Radar Doppler Nyquist boundary.'
        elif objective == 'bidirectional_radial_velocity' and (
                selected_metrics['approaching_peak_mps'] <= 1e-4
                or selected_metrics['receding_peak_mps'] <= 1e-4):
            impossible_reason = 'No visible candidate observes both approaching and receding radial motion.'
        elif whole_path and 'coverage' in selected_metrics and not selected_metrics.get(
                'all_targets_whole_path_visible', False):
            impossible_reason = (
                f'No single Radar pose keeps at least {MIN_PATH_FOV_FRACTION:.0%} of every dynamic '
                'target path inside the FOV margin while native LOS keeps sites of every target.')
            per_target = _per_target_failure_detail(
                ranked, positions_by_frame, target_slices, fov_deg=float(radar.fov))
            extra_detail = {
                'targets_without_whole_path': selected_metrics['coverage'].get(
                    'targets_without_whole_path', []),
                'per_target_coverage': per_target,
                'suggestion': (
                    'multiple_radars'
                    if len(target_specs) > 1 and all(item['best_single_target_pose_m'] for item in per_target)
                    else 'relax_objective'),
            }
        elif objective in WHOLE_PATH_OBJECTIVES and not (
                selected_metrics['whole_path_in_fov']
                and selected_metrics.get('native_whole_path_visible', False)):
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
                    **extra_detail,
                },
            )
        baseline_metrics = None
        if baseline_position is not None:
            try:
                baseline_metrics = _trajectory_metrics(
                baseline_position, aim, positions_by_frame, times,
                fov_deg=float(radar.fov), max_doppler_mps=max_doppler_mps,
                velocities_by_frame=velocities_by_frame,
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
        result = {'position_m': selected['position_m'], 'aim_point_m': selected['aim_point_m'],
                  'mode': 'automatic', 'height_m': height,
                  'targets': target_specs,
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
                      'whole_path_visible': (
                          'every declared target keeps its root trajectory inside the FOV minus '
                          f'{FOV_MARGIN_DEG:g} degrees on at least {MIN_PATH_FOV_FRACTION:.0%} of the '
                          'requested frames and native preflight retains at least one of its own '
                          'skin sites across the entire interval (the solver only simulates such '
                          'sites); site coverage reports additional signal-quality margin'),
                      'trajectory_sampling': 'every requested baked Radar frame',
                  },
                  'native_preflight': selected['native_preflight'],
                  'rejected_candidates': failures}
        if 'coverage' in selected_metrics:
            result['coverage'] = selected_metrics['coverage']
        return result
    raise ToolError('No automatic sensor pose passed native visibility preflight; choose an explicit pose.',
                    code='sensor_placement_unavailable', detail={'candidates': failures})
