import copy
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
