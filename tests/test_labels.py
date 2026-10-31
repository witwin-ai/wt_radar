"""Per-frame visibility labels (design 7.5): labels, not a visibility gate."""
import numpy as np
import pytest

from test_animation import make_rig
from test_install_rule import add_box
from test_snapshot import scene  # noqa: F401
from wt_radar import labels
from wt_radar.adapter.studio_motion import MultiTargetSkinSampler
from wt_radar.adapter.target_contract import component_target_specs
from wt_radar.los import StaticMesh, StaticOccluders


def cube_mesh(object_id, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    faces = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ])
    return StaticMesh(object_id, object_id, corners, faces)


def body(centre, count=40, radius=.2):
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    offsets = np.column_stack((radius * np.cos(angles), np.linspace(-.3, .3, count), radius * np.sin(angles)))
    return np.asarray(centre, float) + offsets


def test_each_frame_gets_one_of_four_labels_from_beam_range_and_line_of_sight():
    # Radar at the origin looking down -z, 30 degree half beam, 1-8 m range.
    wall = cube_mesh("partition", [-1.5, -1., -3.1], [-0.2, 2., -2.9])
    frames = np.stack([
        body([0., 0., -4.]),       # in view
        body([-.9, 0., -4.]),      # behind the partition: occluded
        body([-4., 0., -2.]),      # 63 degrees off axis: out of FOV
        body([0., 0., -.6]),       # closer than 1 m
        body([0., 0., -9.]),       # beyond the range limit
    ])
    result = labels.label_frames(
        radar_position=[0., 0., 0.], forward=[0., 0., -1.], up=[0., 1., 0.],
        times_s=np.arange(5) / 10., positions_by_frame=frames,
        target_slices=[{"object_id": "person", "site_start": 0, "site_end": 40}],
        occluders=StaticOccluders([wall]), range_m=(1., 8.),
    )
    target = result["targets"][0]
    assert [frame["label"] for frame in target["frames"]] == [
        "in_view", "occluded", "out_of_fov", "out_of_range", "out_of_range"]
    first = target["frames"][0]
    assert first["visible_site_fraction"] == pytest.approx(1.)
    assert first["range_m"] == pytest.approx(4., abs=.01)
    assert abs(first["azimuth_deg"]) < 1. and abs(first["elevation_deg"]) < 1.
    assert target["frames"][1]["visible_site_fraction"] < labels.IN_VIEW_SITE_FRACTION
    assert target["label_counts"] == {"in_view": 1, "occluded": 1, "out_of_fov": 1, "out_of_range": 2}
    assert result["label_method"] == "studio_los_v1"
    assert result["schema"] == labels.LABELS_SCHEMA


def test_key_events_are_the_only_hard_gate():
    frames = np.stack([body([0., 0., -4.]), body([-4., 0., -2.]), body([0., 0., -4.])])
    common = dict(
        radar_position=[0., 0., 0.], forward=[0., 0., -1.], up=[0., 1., 0.],
        times_s=np.arange(3) / 10., positions_by_frame=frames,
        target_slices=[{"object_id": "cat", "site_start": 0, "site_end": 40}],
        occluders=StaticOccluders([]), range_m=(1., 8.),
    )
    passed = labels.label_frames(**common, key_events=[
        {"id": "landing", "object_id": "cat", "t_s": .2}])
    assert passed["gate"] == {"key_events_in_view": True, "failures": []}
    # A frame out of view that is not a key event is a label, never a failure.
    assert passed["targets"][0]["label_counts"]["out_of_fov"] == 1
    assert passed["targets"][0]["frames"][2]["key_event"] == "landing"

    failed = labels.label_frames(**common, key_events=[
        {"id": "takeoff", "object_id": "cat", "t_s": .1}, {"id": "late", "object_id": "cat", "t_s": 5.}])
    assert failed["gate"]["key_events_in_view"] is False
    codes = {item["id"]: item["reason"] for item in failed["gate"]["failures"]}
    assert codes == {"takeoff": "out_of_fov", "late": "outside_measured_interval"}


def test_native_interval_share_is_reported_per_target():
    frames = np.stack([body([0., 0., -4.])] * 2)
    result = labels.label_frames(
        radar_position=[0., 0., 0.], forward=[0., 0., -1.], up=[0., 1., 0.],
        times_s=np.arange(2) / 10., positions_by_frame=frames,
        target_slices=[{"object_id": "cat", "site_start": 0, "site_end": 40}],
        occluders=StaticOccluders([]), range_m=(1., 8.),
        native_preflight={"targets": [{"object_id": "cat", "active_site_count": 30, "declared_site_count": 40}]},
    )
    assert result["targets"][0]["native_active_site_share"] == pytest.approx(.75)


def test_scene_labels_use_the_declared_sites_and_measured_frames(scene):
    original, component = scene
    make_rig(original, component)
    component.animation_targets_json = '[{"object_id": "target", "site_budget": 6}]'
    result = labels.label_visibility(original, original.get_object("radar"), key_events=[
        {"id": "start", "object_id": "target", "t_s": 0.}])
    assert result["frame_times_s"] == pytest.approx([0., .1])
    target = result["targets"][0]
    declared = MultiTargetSkinSampler(original, component_target_specs(component)).target_slices[0]
    assert target["object_id"] == "target"
    assert target["site_count"] == declared["declared_site_count"] > 0
    assert [frame["label"] for frame in target["frames"]] == ["in_view", "in_view"]
    assert result["gate"]["key_events_in_view"] is True
    assert result["radar"]["position_m"] == pytest.approx([0., 1., 0.])

    add_box(original, "screen", [-1., 0., -2.1], [1., 3., -1.9])
    hidden = labels.label_visibility(original, original.get_object("radar"))
    assert {frame["label"] for frame in hidden["targets"][0]["frames"]} == {"occluded"}
