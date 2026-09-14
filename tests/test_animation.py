"""Small real-Studio/native-Radar contracts; no offline experiment dependency."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_snapshot import Context, copy_for_solver, scene  # noqa: F401
from witwin_server import SceneObject
from witwin_server.core.components import SkinnedMeshComponent
from witwin_server.features.timeline import InterpolationType, TimelineManager
from wt_radar.adapter.animation import (
    MODEL, AnimationTopologyError, animation_preflight, animation_request,
    animation_view, export_animation, frame_times, solve_animation,
    solve_native_frame,
)
from wt_radar.adapter.saved_result import SavedRadarResult
from wt_radar.adapter.snapshot import prepare_snapshot, snapshot_request
from wt_radar.adapter.studio_motion import StudioSkinSampler


class AnimationContext(Context):
    def throw_if_cancelled(self):
        pass


def test_hidden_fractional_fps_reports_exact_values_without_rounding():
    with pytest.raises(ValueError, match=r"FPS=29\.998998642.*re-enter exact values"):
        frame_times({"time_s": 7., "duration_s": 12., "fps": 29.998998641967773}, None)


def test_frame_times_accepts_float32_round_trip_for_frame_aligned_duration():
    component = SimpleNamespace(
        num_tx=1,
        num_rx=1,
        chirp_per_frame=1,
        adc_samples=1,
        scene=SimpleNamespace(
            timeline_manager=SimpleNamespace(clip=SimpleNamespace(duration=20.0)),
        ),
    )

    duration = float(np.float32(16.9))
    times = frame_times({"time_s": 0.0, "duration_s": duration, "fps": 10.0}, component)

    assert len(times) == 169
    assert times[-1] == pytest.approx(16.8)


def add_track(scene, oid, field, keys):
    track = scene.timeline_manager.clip.get_or_create_track(oid, "Transform", field)
    for time_s, value in keys:
        track.add_keyframe(time_s, value, interpolation=InterpolationType.LINEAR)
    return track


def make_rig(scene, component):
    """Two separate bone influences with identity bind matrices and actual skinning."""
    target = scene.get_object("target")
    target._components.pop("Mesh", None)
    skin = target.add_component(SkinnedMeshComponent())
    vertices = np.array([
        [-.1, 0, -.25], [0, .1, -.25], [.1, 0, -.25],
        [-.1, 0, .25], [0, .1, .25], [.1, 0, .25],
    ], dtype=np.float32)
    skin.set_mesh_data(vertices, np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32))
    for oid in ("bone0", "bone1"):
        scene.add_object(SceneObject(id=oid, name=oid, mesh_type="Empty"))
        scene.set_parent(oid, "target", keep_local_transform=True)
    indices = np.zeros((6, 4), dtype=np.int64)
    indices[3:, 0] = 1
    weights = np.zeros((6, 4), dtype=np.float32)
    weights[:, 0] = 1
    skin.set_skinning_data(
        ["bone0", "bone1"], indices, weights,
        np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)), root_bone_id="bone0",
    )
    scene.timeline_manager = TimelineManager(scene)
    scene.timeline_manager.clip.duration = 2.0
    component.animation_duration_s = .2
    component.animation_fps = 10
    add_track(scene, "bone0", "position", [(0., [0., 0., 0.]), (2., [0., 0., -2.])])
    return skin, vertices


def test_world_scale_parent_affine_and_bone_velocity_are_applied_once(scene):
    original, component = scene
    _, vertices = make_rig(original, component)
    parent = SceneObject(id="parent", mesh_type="Empty")
    original.add_object(parent)
    original.set_parent("target", "parent", keep_local_transform=True)
    parent.get_component("Transform").scale = [2., 3., 4.]
    parent.get_component("Transform").rotation = [0., .7, 0.]
    target = original.get_object("target").get_component("Transform")
    target.scale = [.65, .65, .65]
    add_track(original, "target", "position", [(0., [0., 1., -3.]), (2., [.3, 1., -3.])])
    sampler = StudioSkinSampler(original, "target")
    p, v = sampler.sample(.5)
    parent_matrix = parent.get_component("Transform").get_transformation_matrix().numpy()
    local = vertices[sampler.vertex_ids].astype(np.float64)
    bone_velocities = np.zeros_like(local)
    bone_velocities[0, 2] = -1.
    local[0, 2] -= .5
    expected_local = .65 * local + [.075, 1., -3.]
    expected = (np.column_stack((expected_local, np.ones(len(local)))) @ parent_matrix.T)[:, :3]
    expected_velocity = (.65 * bone_velocities + [.15, 0., 0.]) @ parent_matrix[:3, :3].T
    np.testing.assert_allclose(p, expected, atol=2e-6, rtol=0)
    np.testing.assert_allclose(v, expected_velocity, atol=2e-3, rtol=0)
    assert original.timeline_manager.current_time == pytest.approx(.5)
    assert not np.allclose(v[0], v[1])  # bone motion is not replaced by root velocity


def test_bone_rotation_velocity_is_not_rigid_root_translation(scene):
    original, component = scene
    _, vertices = make_rig(original, component)
    original.timeline_manager.clip.tracks.clear()
    # Studio stores Euler components in ZXY order, not XYZ.
    add_track(original, "bone0", "rotation", [(0., [0., 0., 0.]), (2., [1., 0., 0.])])
    original.get_object("target").get_component("Transform").scale = [.65, .65, .65]
    sampler = StudioSkinSampler(original, "target")
    p, v = sampler.sample(.5)
    angle = .25
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    rotated = rotation @ vertices[sampler.vertex_ids[0]]
    np.testing.assert_allclose(p[0], .65 * rotated + [0., 1., -3.], atol=1e-6)
    np.testing.assert_allclose(v[0], .65 * .5 * np.array([-rotated[1], rotated[0], 0.]), atol=3e-4)
    np.testing.assert_allclose(v[1], 0., atol=3e-4)


@pytest.mark.parametrize("time_s", [1., 1. - 1e-13, 1. + 1e-13])
def test_knot_uses_right_hand_velocity_and_restores_pose(scene, time_s):
    original, component = scene
    make_rig(original, component)
    original.timeline_manager.clip.tracks.clear()
    add_track(original, "bone0", "position", [
        (0., [0., 0., 0.]), (1., [0., 0., -1.]), (2., [0., 0., -3.]),
    ])
    sampler = StudioSkinSampler(original, "target")
    p, v = sampler.sample(time_s)
    np.testing.assert_allclose(v[0], [0., 0., -2.], atol=2e-3)
    np.testing.assert_allclose(v[1], 0., atol=1e-5)
    np.testing.assert_array_equal(p, sampler.positions(time_s))
    np.testing.assert_array_equal(sampler.sample(2.)[1], np.zeros((2, 3)))


def test_sampler_refuses_non_target_animation_and_non_linear_tracks(scene):
    original, component = scene
    make_rig(original, component)
    track = add_track(original, "floor", "position", [(0., [0., -2., 0.])])
    with pytest.raises(ValueError, match="unsupported track"):
        StudioSkinSampler(original, "target")
    original.timeline_manager.clip.tracks.pop(track.path)
    track = next(iter(original.timeline_manager.clip.tracks.values()))
    track.keyframes[0].interpolation = InterpolationType.STEP
    with pytest.raises(ValueError, match="LINEAR"):
        StudioSkinSampler(original, "target")


def test_sampler_refuses_invalid_skin_weights(scene):
    original, component = scene
    skin, _ = make_rig(original, component)
    data = skin.get_skinning_data()
    weights = np.asarray(data["skin_weights"]).copy()
    weights[0] = 0.  # Studio normalizes nonzero sums at its setter boundary.
    skin.set_skinning_data(data["bone_ids"], data["skin_indices"], weights,
                           data["inverse_bind_matrices"], root_bone_id="bone0")
    with pytest.raises(ValueError, match="skin influences"):
        StudioSkinSampler(original, "target")


@pytest.mark.parametrize("updates", [
    {"duration_s": 0.}, {"duration_s": 31.}, {"fps": 0.}, {"fps": 31.},
    {"time_s": -1.}, {"time_s": float("nan")}, {"duration_s": .15},
    {"duration_s": 2.1},
])
def test_invalid_animation_interval_is_refused(scene, updates):
    original, component = scene
    make_rig(original, component)
    request = {"time_s": 0., "duration_s": .2, "fps": 10., **updates}
    with pytest.raises(ValueError):
        frame_times(request, component)


def test_animation_total_memory_budget_rejects_without_changing_configuration(scene):
    original, component = scene
    make_rig(original, component)
    original.timeline_manager.clip.duration = 30.
    component.num_tx, component.num_rx = 3, 4
    component.chirp_per_frame, component.adc_samples = 128, 256
    before = (component.num_tx, component.num_rx, component.chirp_per_frame, component.adc_samples)
    with pytest.raises(ValueError, match="1536 MiB"):
        frame_times({"time_s": 0., "duration_s": 30., "fps": 30.}, component)
    assert before == (component.num_tx, component.num_rx, component.chirp_per_frame, component.adc_samples)


def test_animation_does_not_silently_freeze_target_mesh_descendants(scene, monkeypatch):
    import wt_radar.adapter.animation as animation
    original, component = scene
    make_rig(original, component)
    accessory = SceneObject(id="collar", mesh_type="Cube")
    original.add_object(accessory)
    original.set_parent("collar", "bone0", keep_local_transform=True)
    request = animation_request(component)

    def unexpected_export(*args, **kwargs):
        pytest.fail("Moving Mesh descendant was accepted and reached static export")

    monkeypatch.setattr(animation, "prepare_snapshot", unexpected_export)
    with pytest.raises(ValueError, match="(?i)(mesh|geometry|descendant)"):
        solve_animation(AnimationContext(), copy_for_solver(original), request)


def test_animation_refuses_radar_parented_under_a_moving_bone(scene, monkeypatch):
    import wt_radar.adapter.animation as animation
    original, component = scene
    make_rig(original, component)
    original.set_parent("radar", "bone0", keep_local_transform=True)
    request = animation_request(component)

    def unexpected_export(*args, **kwargs):
        pytest.fail("Moving radar hierarchy was accepted and reached static export")

    monkeypatch.setattr(animation, "prepare_snapshot", unexpected_export)
    with pytest.raises(ValueError, match="(?i)(radar|sensor|static)"):
        solve_animation(AnimationContext(), copy_for_solver(original), request)


def native_setup(original, component):
    from witwin.radar import Radar
    from witwin.radar.scattering import ScalarRcsResponse
    world, config, meta = prepare_snapshot(copy_for_solver(original), snapshot_request(component))
    radar = Radar(config, device="cuda", position=meta["radar_position_m"],
                  target=meta["radar_target_m"], up=meta["radar_up"])
    response = ScalarRcsResponse.from_rcs(.01, reference_frequency_hz=config.fc, device=radar.device)
    return radar, world, response, meta["world_polarization"]


def test_native_stationary_vs_moving_has_physical_doppler(scene, cuda_ready):
    from witwin.radar.processing import range_profile, range_doppler_map
    original, component = scene
    component.chirp_per_frame = component.num_doppler_bins = 64
    radar, world, response, polarization = native_setup(original, component)
    stationary, zero = solve_native_frame(radar, world, response, 0., [[0., 1., -3.]], [[0., 0., 0.]], polarization)
    moving, active = solve_native_frame(radar, world, response, 0., [[0., 1., -3.]], [[0., 0., -1.]], polarization)
    assert zero["delay_rate_abs_max"] == zero["chirp_change_max"] == 0.
    assert active["delay_rate_abs_max"] > 0. and active["chirp_change_max"] > 0.
    assert zero["nonzero_return"] and active["nonzero_return"]
    assert not torch.equal(stationary.data, moving.data)
    for product, expected_speed in ((stationary, 0.), (moving, 1.)):
        rd = range_doppler_map(range_profile(product, window="rectangular", remove_dc=False), window="hann")
        matrix = rd.data[0, 0].abs().cpu().numpy()
        velocity_bin, _ = np.unravel_index(matrix.argmax(), matrix.shape)
        measured_speed = abs(float(product.axes.velocity_mps[velocity_bin]))
        assert abs(measured_speed - expected_speed) <= product.axes.velocity_bin_mps


def test_native_no_incident_path_is_an_error_not_zero_fill(scene, cuda_ready):
    original, component = scene
    # Closed large slab cuts every LOS from radar to the target.
    wall = SceneObject(id="wall", mesh_type="Cube")
    wall.get_component("Transform").position = [0., 1., -1.5]
    wall.get_component("Transform").scale = [10., 10., .2]
    original.add_object(wall)
    radar, world, response, polarization = native_setup(original, component)
    with pytest.raises(ValueError, match="no inbound leg row"):
        solve_native_frame(radar, world, response, 0., [[0., 1., -3.]], [[0., 0., -1.]], polarization)


def test_native_preflight_blocks_occluded_skin_site_before_synthesis(scene, cuda_ready):
    original, component = scene
    make_rig(original, component)
    wall = SceneObject(id="wall", mesh_type="Cube")
    wall.get_component("Transform").position = [0., 1., -1.5]
    wall.get_component("Transform").scale = [10., 10., .2]
    original.add_object(wall)
    with pytest.raises(AnimationTopologyError) as failure:
        animation_preflight(copy_for_solver(original), animation_request(component))
    assert failure.value.detail["code"] == "sensor_embedded_or_all_sites_occluded"
    assert failure.value.detail["declared_site_count"] == 2
    assert len(failure.value.detail["occluded_sites"]) == 2
    assert "No GPU waveform synthesis" in str(failure.value)


def test_native_preflight_reports_reachable_topology_without_synthesis(scene, cuda_ready):
    original, component = scene
    make_rig(original, component)
    result = animation_preflight(copy_for_solver(original), animation_request(component))
    assert result["topology"]["status"] == "reachable"
    assert result["topology"]["method"] == "native_channel_interval_visibility_intersection_no_synthesis"
    assert result["topology"]["declared_site_count"] == result["site_count"] == 2
    assert result["topology"]["active_site_count"] == 2
    assert result["topology"]["occluded_site_count"] == 0
    assert all(frame["inbound_leg_rows"] > 0 for frame in result["topology"]["frames"])
    assert all(frame["outbound_leg_rows"] > 0 for frame in result["topology"]["frames"])


def test_visible_site_model_preserves_ids_and_does_not_renormalize_rcs(scene, cuda_ready):
    original, component = scene
    make_rig(original, component)
    # The near site ends at z=-2.75 before this slab; the far site at z=-3.25
    # is occluded for the whole interval.
    wall = SceneObject(id="partial-wall", mesh_type="Cube")
    wall.get_component("Transform").position = [0., 1., -2.9]
    wall.get_component("Transform").scale = [1., 1., .1]
    original.add_object(wall)
    request = animation_request(component)
    preflight = animation_preflight(copy_for_solver(original), request)
    topology = preflight["topology"]
    assert topology["declared_site_count"] == 2
    assert topology["active_site_count"] == 1
    assert topology["occluded_site_count"] == 1
    assert topology["occluded_sites"][0]["bone_name"] == "bone0"
    assert topology["rcs_policy"].endswith("no_visible_renormalization")
    assert topology["visibility_coverage"] == pytest.approx(0.5)
    assert topology["visibility_quality"] == "degraded_interval_global_subset"

    result = solve_animation(AnimationContext(), copy_for_solver(original), request)
    assert result.metadata["site_bone_names"] == ["bone1"]
    assert result.metadata["declared_site_bone_names"] == ["bone0", "bone1"]
    assert result.metadata["active_site_ids"] == topology["active_site_ids"]
    assert result.metadata["rcs_per_site_m2"] == pytest.approx(component.snapshot_rcs_m2 / 2)
    assert result.positions_m.shape == result.velocities_mps.shape == (2, 1, 3)


def test_native_animation_propagates_frame_error_without_completed_result(scene, cuda_ready):
    original, component = scene
    make_rig(original, component)
    wall = SceneObject(id="wall", mesh_type="Cube")
    wall.get_component("Transform").position = [0., 1., -1.5]
    wall.get_component("Transform").scale = [10., 10., .2]
    original.add_object(wall)
    request = animation_request(component)
    with pytest.raises(AnimationTopologyError, match="no skin site.*throughout the requested interval") as failure:
        solve_animation(AnimationContext(), copy_for_solver(original), request)
    assert failure.value.detail["code"] == "sensor_embedded_or_all_sites_occluded"


def test_native_animation_refuses_overlapping_cpi(scene, cuda_ready):
    original, component = scene
    make_rig(original, component)
    component.chirp_per_frame = component.num_doppler_bins = 1024
    component.animation_duration_s = .1
    component.animation_fps = 30
    with pytest.raises(ValueError, match="(?i)(CPI|frame spacing)"):
        solve_animation(AnimationContext(), copy_for_solver(original), animation_request(component))


def test_native_animation_refuses_slow_time_velocity_alias(scene, cuda_ready):
    original, component = scene
    make_rig(original, component)
    original.timeline_manager.clip.tracks.clear()
    add_track(original, "bone0", "position", [(0., [0., 0., 0.]), (2., [0., 0., -200.])])
    with pytest.raises((ValueError, RuntimeError), match="(?i)(Nyquist|alias|unambiguous)"):
        solve_animation(AnimationContext(), copy_for_solver(original), animation_request(component))


def test_native_two_frame_animation_export_roundtrip_and_editor_isolation(scene, cuda_ready, tmp_path):
    original, component = scene
    make_rig(original, component)
    component.chirp_per_frame = component.num_doppler_bins = 64
    request = animation_request(component)
    before = original.to_dict()
    result = solve_animation(AnimationContext(), copy_for_solver(original), request)
    assert original.to_dict() == before
    assert result.metadata["model"] == MODEL
    assert result.cube.shape == (2, 1, 1, 64, 128)
    assert result.positions_m.shape == result.velocities_mps.shape == (2, 2, 3)
    assert all(row["delay_rate_abs_max"] > 0 for row in result.metadata["frame_diagnostics"])
    assert result.metadata["components"] == ["los"] and result.metadata["max_depth"] == 0
    np.testing.assert_allclose(result.times_s, [0., .1], atol=0, rtol=0)
    path = tmp_path / "animation.npz"

    def save_result(payload, **kwargs):
        assert kwargs["kind"] == "radar.animation"
        path.write_bytes(payload)
        return str(path)

    assert export_animation(SimpleNamespace(result_ref=save_result), result) == str(path)
    loaded = SavedRadarResult.load(path)
    assert torch.equal(loaded.cube, result.cube)
    np.testing.assert_array_equal(loaded.times_s, result.times_s)
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["positions_m"], result.positions_m)
        np.testing.assert_array_equal(archive["velocities_mps"], result.velocities_mps)
    for frame in range(2):
        payload = animation_view(result, {"frame": frame, "view": "range_profile", "tx": 0, "rx": 0})
        np.testing.assert_array_equal(payload["magnitude"], result.cube[frame, 0, 0, 0].abs().tolist())
        assert "Studio skin motion" in payload["title"]
    with pytest.raises(ValueError, match="Animation frame"):
        animation_view(result, {"frame": 2})
