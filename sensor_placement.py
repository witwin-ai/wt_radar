"""Bounded sensor-pose selection; invokes existing native preflight unchanged.

All trials happen on a detached Studio scene. No waveform, propagation, site
selection or DSP equations live here, and no live objects are moved by a trial.
"""
import copy
import math
import numpy as np

from witwin_server.tools.base import ToolError


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
    times = np.linspace(radar.t0, radar.t0 + radar.animation_duration_s - 1 / radar.animation_fps, 3)
    aim = np.concatenate([sampler.positions(float(t)) for t in times]).mean(axis=0)
    height = float(args.get('height_m', 1.))
    requested_distance = args.get('distance_m')
    distance = float(requested_distance) if requested_distance is not None else None
    if (not np.isfinite(aim).all() or not math.isfinite(height) or not .2 <= height <= 3.
            or (distance is not None and (not math.isfinite(distance) or not .25 <= distance <= 10.))):
        raise ToolError('Automatic placement needs finite target motion and height [0.2,3] m.',
                        code='invalid_sensor_placement')
    failures = []
    for position in candidate_positions(aim, height, distance):
        detached.update_transform(str(sensor.id), position=position,
                                  rotation=_look_at_euler(np.asarray(position), aim))
        try:
            evidence = animation_preflight(detached, animation_request(radar))
        except AnimationTopologyError as exc:
            failures.append({'position_m': position, 'reason': str(exc), 'detail': exc.detail})
            continue
        # Memory/timing/rig/CUDA failures are not placement problems: surface
        # them immediately, without changing physics settings or reducing FPS.
        return {'position_m': position, 'aim_point_m': aim.tolist(),
                'mode': 'automatic', 'height_m': height,
                'requested_distance_m': distance,
                'actual_distance_m': float(np.linalg.norm(np.asarray(position) - aim)),
                'tested_candidates': len(failures) + 1,
                'native_preflight': evidence, 'rejected_candidates': failures}
    raise ToolError('No automatic sensor pose passed native visibility preflight; choose an explicit pose.',
                    code='sensor_placement_unavailable', detail={'candidates': failures})
