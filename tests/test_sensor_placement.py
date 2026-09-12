import copy
import numpy as np
import pytest
from test_agent_tools import harness, run
from test_snapshot import scene
from test_animation import make_rig
from wt_radar import sensor_placement
from wt_radar.adapter import animation
from witwin_server.tools.base import ToolError


def test_trials_are_detached_and_use_unchanged_native_preflight(scene, monkeypatch):
    original, component = scene
    make_rig(original, component)
    before = copy.deepcopy(original.to_dict())
    tried = []
    def preflight(detached, request):
        assert detached is not original
        sensor = detached.get_object(request['radar_object_id'])
        tried.append(list(sensor.get_component('Transform').position))
        if len(tried) == 1:
            raise animation.AnimationTopologyError('occluded', detail={'code': 'occluded'})
        return {'frame_count': 2, 'site_count': 2, 'radar_device': 'cuda'}
    monkeypatch.setattr(animation, 'animation_preflight', preflight)
    result = sensor_placement.choose_sensor_placement(original, {
        'target_object_id': 'target', 'duration_s': .2, 'fps': 10, 'height_m': 1.,
    }, radar_object_id='radar')
    assert result['tested_candidates'] == 2
    assert result['position_m'][1] == 1.
    assert original.to_dict() == before


def test_requested_automatic_distance_is_preserved_in_world_space(scene, monkeypatch):
    original, component = scene
    make_rig(original, component)
    before = copy.deepcopy(original.to_dict())
    monkeypatch.setattr(animation, 'animation_preflight', lambda *_: {
        'frame_count': 2, 'site_count': 2, 'radar_device': 'cuda',
    })

    result = sensor_placement.choose_sensor_placement(original, {
        'target_object_id': 'target', 'duration_s': .2, 'fps': 10,
        'height_m': 1., 'distance_m': 2.,
    }, radar_object_id='radar')

    assert result['requested_distance_m'] == 2.
    assert result['actual_distance_m'] == pytest.approx(2.)
    assert np.linalg.norm(np.asarray(result['position_m']) - np.asarray(result['aim_point_m'])) == pytest.approx(2.)
    assert original.to_dict() == before


@pytest.mark.parametrize('objective', [
    'maximize_range_span', 'bidirectional_radial_velocity', 'stronger_doppler',
    'side_view', 'whole_path_visible', 'opposite_side',
    'closer_with_full_path_visible',
])
def test_explicit_objectives_rank_all_candidates_and_report_full_trajectory(scene, monkeypatch, objective):
    original, component = scene
    make_rig(original, component)
    before = copy.deepcopy(original.to_dict())
    monkeypatch.setattr(animation, 'animation_preflight', lambda *_: {
        'frame_count': 2, 'site_count': 2, 'radar_device': 'cuda',
        'topology': {'visibility_coverage': 1.},
    })

    result = sensor_placement.choose_sensor_placement(original, {
        'target_object_id': 'target', 'duration_s': .2, 'fps': 10,
        'height_m': 1., 'placement_objective': objective,
    }, radar_object_id='radar')

    assert result['placement_objective'] == objective
    assert result['tested_candidates'] == 16
    assert result['valid_candidate_count'] == 1
    assert result['native_preflight_count'] == 1
    assert result['predicted_metrics']['sampled_frame_count'] == 2
    assert result['predicted_metrics']['sampled_site_count'] == 2
    assert result['metric_semantics']['trajectory_sampling'] == 'every requested baked Radar frame'
    assert isinstance(result['predicted_metrics']['doppler_within_nyquist'], bool)
    assert original.to_dict() == before


def test_explicit_objective_continues_when_the_best_kinematic_pose_is_occluded(scene, monkeypatch):
    original, component = scene
    make_rig(original, component)
    calls = []

    def preflight(*_args):
        calls.append(1)
        if len(calls) == 1:
            raise animation.AnimationTopologyError('occluded', detail={'code': 'occluded'})
        return {
            'frame_count': 2, 'site_count': 2, 'radar_device': 'cuda',
            'topology': {'visibility_coverage': 1.},
        }

    monkeypatch.setattr(animation, 'animation_preflight', preflight)
    result = sensor_placement.choose_sensor_placement(original, {
        'target_object_id': 'target', 'duration_s': .2, 'fps': 10,
        'height_m': 1., 'placement_objective': 'maximize_range_span',
    }, radar_object_id='radar')

    assert result['tested_candidates'] == 16
    assert result['native_preflight_count'] == 2
    assert result['valid_candidate_count'] == 1
    assert result['rejected_candidates'][0]['reason'] == 'occluded'


def test_trajectory_metrics_report_radial_sign_fov_and_nyquist():
    times = np.asarray([0., 1., 2.])
    points = np.asarray([
        [[2., 0., 0.]],
        [[1., 0., 0.]],
        [[2., 0., 0.]],
    ])
    metrics = sensor_placement._trajectory_metrics(
        [0., 0., 0.], [1., 0., 0.], points, times,
        fov_deg=60., max_doppler_mps=.5,
    )
    assert metrics['sampled_frame_count'] == 3
    assert metrics['range_span_m'] == pytest.approx(1.)
    assert metrics['approaching_peak_mps'] > 0
    assert metrics['receding_peak_mps'] > 0
    assert metrics['whole_path_in_fov'] is True
    assert metrics['doppler_within_nyquist'] is False


def test_impossible_bidirectional_objective_reports_metrics_and_tradeoffs(scene, monkeypatch):
    original, component = scene
    make_rig(original, component)
    monkeypatch.setattr(animation, 'animation_preflight', lambda *_: {
        'frame_count': 2, 'site_count': 2, 'radar_device': 'cuda',
        'topology': {'visibility_coverage': 1.},
    })
    one_way = {
        'sampled_frame_count': 2, 'sampled_site_count': 2,
        'range_min_m': 1., 'range_max_m': 2., 'range_span_m': 1., 'mean_range_m': 1.5,
        'radial_velocity_min_mps': -.5, 'radial_velocity_max_mps': -.1,
        'radial_velocity_rms_mps': .3, 'peak_abs_radial_velocity_mps': .5,
        'approaching_peak_mps': .5, 'receding_peak_mps': 0.,
        'tangential_velocity_rms_mps': .2, 'fov_site_frame_coverage': 1.,
        'whole_path_in_fov': True, 'max_off_axis_angle_deg': 10.,
        'doppler_nyquist_mps': 5., 'doppler_within_nyquist': True,
    }
    monkeypatch.setattr(sensor_placement, '_trajectory_metrics', lambda *_a, **_k: dict(one_way))

    with pytest.raises(ToolError) as caught:
        sensor_placement.choose_sensor_placement(original, {
            'target_object_id': 'target', 'duration_s': .2, 'fps': 10,
            'placement_objective': 'bidirectional_radial_velocity',
        }, radar_object_id='radar')

    assert caught.value.code == 'sensor_placement_objective_unavailable'
    assert caught.value.detail['best_candidate_metrics']['receding_peak_mps'] == 0.
    assert caught.value.detail['tradeoff_options'] == [
        'keep_current_pose', 'relax_objective', 'choose_explicit_pose',
    ]


def test_automatic_failure_does_not_leave_a_radar_in_the_live_scene(harness, monkeypatch):
    scene, tools = harness
    before = copy.deepcopy(scene.to_dict())
    def fail(*args, **kwargs):
        raise ToolError('no visible pose', code='sensor_placement_unavailable')
    monkeypatch.setattr(sensor_placement, 'choose_sensor_placement', fail)
    with pytest.raises(ToolError):
        run(tools, 'ensure_sensor', {'scene_id': scene.scene_id, 'operation_id': 'auto-fail',
            'target_object_id': 'catstray', 'placement_mode': 'automatic'})
    assert scene.to_dict() == before


def test_automatic_cannot_overwrite_explicit_coordinates(harness):
    scene, tools = harness
    before = copy.deepcopy(scene.to_dict())
    with pytest.raises(ToolError) as caught:
        run(tools, 'ensure_sensor', {'scene_id': scene.scene_id, 'operation_id': 'auto-conflict',
            'target_object_id': 'catstray', 'placement_mode': 'automatic', 'position_m': [0, 1, 0]})
    assert caught.value.code == 'conflicting_sensor_placement'
    assert scene.to_dict() == before


def test_non_visibility_failures_are_not_retried_or_downgraded(scene, monkeypatch):
    original, component = scene
    make_rig(original, component)
    before = copy.deepcopy(original.to_dict())
    calls = []

    def preflight(detached, request):
        calls.append(request)
        raise RuntimeError('CUDA unavailable')

    monkeypatch.setattr(animation, 'animation_preflight', preflight)
    with pytest.raises(RuntimeError, match='CUDA unavailable'):
        sensor_placement.choose_sensor_placement(original, {
            'target_object_id': 'target', 'duration_s': .2, 'fps': 10,
        }, radar_object_id='radar')
    assert len(calls) == 1
    assert original.to_dict() == before


def test_all_occluded_candidates_stop_at_the_bound_without_mutation(scene, monkeypatch):
    original, component = scene
    make_rig(original, component)
    before = copy.deepcopy(original.to_dict())
    calls = []

    def preflight(detached, request):
        calls.append(request)
        raise animation.AnimationTopologyError('occluded', detail={'code': 'occluded'})

    monkeypatch.setattr(animation, 'animation_preflight', preflight)
    with pytest.raises(ToolError) as caught:
        sensor_placement.choose_sensor_placement(original, {
            'target_object_id': 'target', 'duration_s': .2, 'fps': 10,
        }, radar_object_id='radar')
    assert caught.value.code == 'sensor_placement_unavailable'
    assert len(calls) == 16
    assert original.to_dict() == before


def test_automatic_placement_rejects_candidate_inside_furniture_before_native_preflight(scene, monkeypatch):
    from witwin_server import SceneObject

    original, component = scene
    make_rig(original, component)
    # The first ring candidate is one metre along +X from the trajectory mean.
    obstacle = SceneObject(id='blocking-chair', name='Blocking Chair', mesh_type='Cube')
    obstacle.get_component('Transform').position = [1., 1., -3.05]
    obstacle.get_component('Transform').scale = [.8, .8, .8]
    original.add_object(obstacle)
    calls = []
    monkeypatch.setattr(animation, 'animation_preflight', lambda *_: calls.append(1) or {
        'frame_count': 2, 'site_count': 2, 'radar_device': 'cuda',
    })

    result = sensor_placement.choose_sensor_placement(original, {
        'target_object_id': 'target', 'duration_s': .2, 'fps': 10, 'height_m': 1.,
    }, radar_object_id='radar')

    assert result['tested_candidates'] == 2
    assert result['rejected_candidates'][0]['detail']['code'] == 'radar_furniture_collision'
    assert result['rejected_candidates'][0]['detail']['object_id'] == 'blocking-chair'
    assert len(calls) == 1
