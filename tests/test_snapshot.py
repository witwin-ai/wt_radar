"""Adapter tests against real Studio, current Radar contracts, and opt-in CUDA."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from witwin_server.api import Scene, SceneObject, make_scene_ref, load_scene_ref
from wt_radar.components.radar import RadarComponent
from wt_radar.adapter.snapshot import (
    MODEL, SnapshotResult, _hierarchy, prepare_snapshot, processing_cube,
    snapshot_request, snapshot_view, solve_snapshot,
)


class Context:
    def log(self, message):
        pass

    def progress(self, *args):
        pass


@pytest.fixture
def scene():
    scene = Scene(scene_id="snapshot-test", name="Snapshot test")
    floor = SceneObject(id="floor", name="Floor", mesh_type="Cube")
    floor.get_component("Transform").position = [0.0, -2.0, 0.0]
    floor.get_component("Transform").scale = [10.0, 0.1, 10.0]
    scene.add_object(floor)
    target = SceneObject(id="target", name="Authored point target", mesh_type="Empty")
    target.get_component("Transform").position = [0.0, 1.0, -3.0]
    scene.add_object(target)
    sensor = SceneObject(id="radar", name="Radar", mesh_type="Empty")
    sensor.get_component("Transform").position = [0.0, 1.0, 0.0]
    component = sensor.add_component(RadarComponent())
    component.snapshot_target_id = "target"
    component.view = "range_profile"
    component.num_tx = 1
    component.num_rx = 1
    component.tx_loc = [[0, 0, 0]]
    component.rx_loc = [[0, 0, 0]]
    component.adc_samples = 128
    component.num_range_bins = 128
    component.chirp_per_frame = 16
    component.num_doppler_bins = 16
    scene.add_object(sensor)
    return scene, component


def copy_for_solver(scene):
    return load_scene_ref(make_scene_ref(scene))


def test_snapshot_contains_unsaved_changes_and_does_not_mutate_editor(scene):
    original, component = scene
    original.get_object("target").get_component("Transform").position = [1, 1, -4]
    before = original.to_dict()
    _, config, meta = prepare_snapshot(copy_for_solver(original), snapshot_request(component))
    assert meta["target_world_point_m"] == [1, 1, -4]
    assert meta["velocity_m_per_s"] == [0, 0, 0]
    assert meta["room_object_ids"] == ["floor"]
    assert meta["model"] == MODEL
    assert config["fc"] == component.fc
    assert original.to_dict() == before


def test_full_parent_affine_scales_local_point_once(scene):
    original, component = scene
    parent = SceneObject(id="parent", mesh_type="Empty")
    original.add_object(parent)
    parent.get_component("Transform").scale = [2.0, 3.0, 4.0]
    parent.get_component("Transform").rotation = [0.0, 0.7, 0.0]
    original.set_parent("target", "parent", keep_local_transform=True)
    target_transform = original.get_object("target").get_component("Transform")
    target_transform.scale = [0.65, 0.65, 0.65]
    component.snapshot_local_point = [0.2, 0.3, 0.4]
    expected_matrix = parent.get_component("Transform").get_transformation_matrix().numpy() @ target_transform.get_transformation_matrix().numpy()
    _, _, meta = prepare_snapshot(copy_for_solver(original), snapshot_request(component))
    np.testing.assert_allclose(meta["target_world_point_m"], (expected_matrix @ [0.2, 0.3, 0.4, 1])[:3], atol=1e-6)


@pytest.mark.parametrize("field,value", [
    ("snapshot_target_id", "missing"), ("snapshot_target_id", "radar"),
    ("snapshot_rcs_m2", 0), ("snapshot_rcs_m2", float("nan")),
    ("enable_thermal", True), ("enable_adc", True), ("pol_enabled", True),
    ("multipath", True), ("max_reflections", 1), ("sampling", "pixel"),
    ("device", "cpu"), ("backend", "slang"), ("snapshot_polarization", [0, 0, 0]),
    ("t0", -1), ("num_doppler_bins", 17),
])
def test_invalid_or_unimplemented_settings_fail(scene, field, value):
    original, component = scene
    setattr(component, field, value)
    with pytest.raises(ValueError):
        prepare_snapshot(copy_for_solver(original), snapshot_request(component))


def test_hidden_parent_hides_target_and_room_geometry(scene):
    original, component = scene
    parent = SceneObject(id="parent", mesh_type="Empty")
    original.add_object(parent)
    original.set_parent("target", "parent", keep_local_transform=True)
    parent.visible = False
    with pytest.raises(ValueError, match="visible"):
        prepare_snapshot(copy_for_solver(original), snapshot_request(component))


def test_parent_cycle_and_unresolved_reference_fail(scene):
    original, component = scene
    transform = original.get_object("target").get_component("Transform")
    transform.parent = transform
    with pytest.raises(ValueError, match="cycle"):
        _hierarchy(original)
    transform.parent = SceneObject(mesh_type="Empty").get_component("Transform")
    with pytest.raises(ValueError, match="Unresolved"):
        _hierarchy(original)


def test_wrong_scene_and_playback_fail(scene):
    original, component = scene
    request = snapshot_request(component)
    request["scene_id"] = "other-scene"
    with pytest.raises(ValueError, match="identity"):
        prepare_snapshot(copy_for_solver(original), request)
    from witwin_server.api import TimelineManager
    original.timeline_manager = TimelineManager(original)
    original.timeline_manager.is_playing = True
    with pytest.raises(ValueError, match="Pause"):
        snapshot_request(component)


def test_timeline_cannot_bypass_unsupported_settings_validation(scene):
    from witwin_server.api import TimelineManager
    original, component = scene
    original.timeline_manager = TimelineManager(original)
    original.timeline_manager.clip.get_or_create_track("radar", "Radar", "multipath").add_keyframe(0.0, True)
    with pytest.raises(ValueError, match="multipath"):
        prepare_snapshot(copy_for_solver(original), snapshot_request(component))
    assert not component.multipath


def test_rd_widget_receives_physical_axes_and_numbers_not_rgb():
    from wt_radar.adapter.replay import numeric_plot
    payload = {"view": "range_doppler", "range_m": [0.0, 1.0, 2.0], "velocity_mps": [-1.0, 0.0, 1.0],
               "magnitude": [[0, 0, 0], [0, 1, 0], [0, 0, 0]]}
    plot = numeric_plot(payload).to_dict()
    assert plot["data"]["values"] == payload["magnitude"]
    assert plot["data"]["x"] == payload["range_m"]
    assert plot["data"]["y"] == payload["velocity_mps"]
    assert plot["data"]["origin"] == "lower"


def test_profile_widget_preserves_tiny_native_amplitudes():
    from wt_radar.adapter.replay import numeric_plot
    payload = {"view": "range_profile", "range_m": [0, 1, 2], "magnitude": [1e-11, 2e-8, 1e-11]}
    plot = numeric_plot(payload).to_dict()
    assert plot["data"]["series"][0]["y"] == payload["magnitude"]
    assert 2e-8 < plot["data"]["ylim"][1] < 3e-8


def test_failed_static_snapshot_preserves_existing_replay(scene, monkeypatch):
    _, component = scene
    monkeypatch.setattr(component, "_scene_is_live", lambda: True)
    component._snapshot_result = True
    component._animation_result = True
    component._solver_result_handle = "old"
    component._solver_run_id = "old-run"
    component.signal_source = {"sourceId": "existing-replay", "mode": "timeline"}
    component.signal_figure.line([0, 1], [2, 3])
    component.snapshot_target_id = ""
    with pytest.raises(ValueError, match="Target ID"):
        asyncio.run(component._simulate_async())
    assert component._solver_result_handle == "old"
    assert component._solver_run_id == "old-run"
    assert component._animation_result
    assert not component._snapshot_result
    assert component.signal_source["sourceId"] == "existing-replay"
    assert component.signal_figure._series
    assert not component.snapshot_figure._series
    assert "failed" in component.snapshot_status
    assert not component._snapshot_running


def test_non_square_array_packing_is_identity_and_spectrum_not_fft_again():
    from witwin.radar import Radar
    from witwin.radar.processing import ProcessingAxes
    from witwin.radar.synthesis.assembly import SynthesisResult
    radar = Radar.from_dict({
        "num_tx": 2, "num_rx": 3, "tx_loc": [[0, 0, 0], [1, 0, 0]],
        "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0]], "adc_samples": 8,
        "chirp_per_frame": 4,
        "fc": 77e9, "slope": 60.012, "adc_start_time": 6, "sample_rate": 4400,
        "idle_time": 7, "ramp_end_time": 65, "power": 15,
    }, device="cpu")
    spec = radar.system_config.waveform_spec()
    cube = torch.arange(192).reshape(1, 2, 3, 4, 8).to(torch.complex64)
    synthesis = SynthesisResult.from_fmcw(
        cube[0].permute(2, 1, 0, 3).reshape(4, 6, 8), spec,
    )
    axes = ProcessingAxes.from_synthesis(synthesis, spec, radar.system_config.sensors.array)
    frame = SimpleNamespace(processing_cube=lambda: __import__(
        "witwin.radar.processing", fromlist=["ProcessingCube"]
    ).ProcessingCube(cube[0], axes))
    sim = SimpleNamespace(
        cube=cube, axis_names=("frame", "tx", "rx", "chirp", "range_bin"),
        kind="fmcw", output_domain="spectrum", times_s=(0.0,), frame=lambda _index: frame,
    )
    processed = processing_cube(sim, radar)
    assert torch.equal(processed.data, cube[0])
    result = SnapshotResult(processed, {})
    payload = snapshot_view(result, {"tx": 1, "rx": 2, "view": "range_profile"})
    assert payload["magnitude"] == cube[0, 1, 2, 0].abs().tolist()


def test_real_gpu_repeat_and_moved_sensor(scene, cuda_ready):
    original, component = scene
    request = snapshot_request(component)
    first = solve_snapshot(Context(), copy_for_solver(original), request)
    repeat = solve_snapshot(Context(), copy_for_solver(original), request)
    assert first.processing.data.device.type == "cuda"
    assert torch.equal(first.processing.data, repeat.processing.data)
    assert first.metadata["components"] == ["los", "reflection"]
    assert first.metadata["max_depth"] == 1
    assert first.metadata["environment_reflection"]["max_depth"] == 1
    assert first.metadata["environment_reflection"]["reflected_path_count"] >= 0
    assert first.metadata["environment_reflection"]["material_slot_count"] > 0
    assert first.metadata["environment_reflection"]["coherent_with_target"] is True
    original.get_object("radar").get_component("Transform").position = [0, 1, -1]
    moved = solve_snapshot(Context(), copy_for_solver(original), snapshot_request(component))
    assert not torch.equal(first.processing.data, moved.processing.data)
    for result, expected_range in ((first, 3.0), (moved, 2.0)):
        payload = snapshot_view(result, {"view": "range_doppler"})
        peak = np.unravel_index(np.asarray(payload["magnitude"]).argmax(), np.shape(payload["magnitude"]))
        assert abs(payload["range_m"][peak[1]] - expected_range) <= result.processing.axes.range_bin_m
        assert abs(payload["velocity_mps"][peak[0]]) < 1e-8


def _floor_material(world_scene):
    (floor,) = [s for s in world_scene.structures if s.metadata["studio_object_id"] == "floor"]
    return floor.material


def _set_material(scene, name, eps_r, sigma_e):
    material = scene.get_object("floor").get_component("Material")
    material.material_name = name
    material.eps_r = [eps_r, 0.0]
    material.sigma_e = sigma_e


def test_itu_room_material_is_evaluated_at_the_radar_carrier(scene):
    # Studio stores the ITU-R P.2040-1 nominal 1 GHz values; at 77 GHz the
    # plasterboard conductivity is sigma = c * f^d = 0.0116 * 77^0.7076.
    original, component = scene
    _set_material(original, "Plasterboard", 2.94, 0.0116)
    world, _, meta = prepare_snapshot(copy_for_solver(original), snapshot_request(component))
    material = _floor_material(world)
    assert material.eps_r == pytest.approx(2.94)
    assert material.sigma_e == pytest.approx(0.0116 * 77.0 ** 0.7076, rel=1e-6)
    assert meta["room_material_frequency_hz"] == component.fc
    assert original.get_object("floor").get_component("Material").sigma_e == pytest.approx(0.0116)


def test_itu_conversion_follows_the_carrier_not_a_fixed_band(scene):
    original, component = scene
    _set_material(original, "Floorboard", 3.66, 0.0044)
    component.fc = 60e9
    world, _, _ = prepare_snapshot(copy_for_solver(original), snapshot_request(component))
    assert _floor_material(world).sigma_e == pytest.approx(0.0044 * 60.0 ** 1.3515, rel=1e-6)


@pytest.mark.parametrize("name,eps_r,sigma_e", [
    ("Custom", 10.0, 0.0),           # hand-authored: no coefficients to scale
    ("Plasterboard", 2.5, 0.02),     # named but edited away from the nominal values
    ("Wet ground", 30.0, 0.15),      # ITU ground models hold only from 1 to 10 GHz
])
def test_materials_without_a_valid_itu_model_keep_their_stored_values(scene, name, eps_r, sigma_e):
    original, component = scene
    _set_material(original, name, eps_r, sigma_e)
    world, _, _ = prepare_snapshot(copy_for_solver(original), snapshot_request(component))
    material = _floor_material(world)
    assert material.eps_r == pytest.approx(eps_r)
    assert material.sigma_e == pytest.approx(sigma_e)
