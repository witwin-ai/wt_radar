"""Static Radar capture (gap G4): the room without moving targets, as N frames."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_animation import AnimationContext, make_rig
from test_install_rule import add_box
from test_install_tools import register
from test_snapshot import copy_for_solver, scene  # noqa: F401
from wt_radar import solver_host
from wt_radar.adapter import static_capture
from wt_radar.adapter.animation import AnimationResult
from wt_radar.adapter.snapshot import snapshot_request, solve_snapshot
from wt_radar.adapter.studio_motion import MultiTargetSkinSampler


def test_room_request_needs_no_target_and_bounds_the_frames(scene):
    original, component = scene
    component.snapshot_target_id = ""
    request = static_capture.static_capture_request(component, frame_count=4)
    assert request["adapter"] == static_capture.STATIC_MODEL
    assert request["mode"] == "room" and request["targets"] == []
    assert request["frame_count"] == 4 and request["duration_s"] == pytest.approx(4 / request["fps"])
    with pytest.raises(ValueError, match="frames"):
        static_capture.static_capture_request(component, frame_count=0)
    with pytest.raises(ValueError, match="mode"):
        static_capture.static_capture_request(component, mode="sweep")


def test_frozen_request_needs_a_time_on_the_timeline(scene):
    original, component = scene
    make_rig(original, component)
    component.animation_targets_json = '[{"object_id": "target", "site_budget": 6}]'
    request = static_capture.static_capture_request(component, mode="frozen", frame_count=3, freeze_time_s=1.)
    assert request["freeze_time_s"] == 1. and request["start_s"] == 1.
    assert [item["object_id"] for item in request["targets"]] == ["target"]
    with pytest.raises(ValueError, match="timeline"):
        static_capture.static_capture_request(component, mode="frozen", freeze_time_s=9.)


def test_solver_host_routes_static_requests(monkeypatch):
    seen = []
    monkeypatch.setattr(static_capture, "solve_static_capture", lambda ctx, scene, config: seen.append(config) or 1)
    assert solver_host.solve(None, None, {"adapter": static_capture.STATIC_MODEL}) == 1
    assert seen


@pytest.mark.gpu
def test_room_capture_is_n_identical_finite_frames_with_the_room_return(scene, cuda_ready):
    original, component = scene
    component.snapshot_target_id = ""
    add_box(original, "back_wall", [-3., -2., -4.2], [3., 3., -4.])
    request = static_capture.static_capture_request(component, frame_count=3)
    result = static_capture.solve_static_capture(AnimationContext(), copy_for_solver(original), request)
    assert isinstance(result, AnimationResult)
    assert tuple(result.cube.shape) == (3, 1, 1, int(component.chirp_per_frame), int(component.adc_samples))
    assert torch.isfinite(result.cube).all() and (result.cube.abs() > 0).any()
    assert torch.equal(result.cube[0], result.cube[2])
    assert result.times_s.tolist() == pytest.approx([0., .1, .2])
    assert result.positions_m.shape == (3, 0, 3)
    meta = result.metadata
    assert meta["capture"] == "static" and meta["mode"] == "room"
    assert meta["frames_identical_by_construction"] is True
    assert meta["environment_reflection"]["reflected_path_count"] > 0
    assert "back_wall" in meta["room_object_ids"]
    # The room-only axes are the producer axes a native simulation publishes.
    component.snapshot_target_id = "target"
    snapshot = solve_snapshot(SimpleNamespace(progress=lambda *a: None, log=lambda *a: None),
                              copy_for_solver(original), snapshot_request(component))
    native = snapshot.processing.axes
    for field in ("range_bin_m", "velocity_bin_mps", "range_bin_count", "doppler_bin_count", "output_domain"):
        assert getattr(result.axes, field) == getattr(native, field)
    assert torch.allclose(result.axes.range_m, native.range_m.cpu())


@pytest.mark.gpu
def test_frozen_capture_holds_the_actor_still_at_its_sites(scene, cuda_ready):
    original, component = scene
    make_rig(original, component)
    component.animation_targets_json = '[{"object_id": "target", "site_budget": 6}]'
    request = static_capture.static_capture_request(component, mode="frozen", frame_count=2, freeze_time_s=1.)
    result = static_capture.solve_static_capture(AnimationContext(), copy_for_solver(original), request)
    assert result.metadata["site_count"] > 0
    assert result.positions_m.shape[0] == 2 and np.all(result.velocities_mps == 0.)
    assert np.allclose(result.positions_m[0], result.positions_m[1])
    # The frozen sites are the active skin sites at t=1 s.
    sampler = MultiTargetSkinSampler(original, request["targets"])
    active = result.metadata["topology_preflight"]["active_site_ranks"]
    assert np.allclose(result.positions_m[0], sampler.sample(1.)[0][active], atol=1e-5)
    assert result.metadata["targets"][0]["active_site_count"] > 0


def test_capture_tool_writes_and_checks_one_new_result(scene, tmp_path, monkeypatch):
    original, component = scene
    component.snapshot_target_id = ""
    tools = register(original, tmp_path)
    tool = tools["capture_static_radar"]
    assert tool.permission_tier == "file_write" and tool.side_effects and tool.requires_confirmation
    submitted = []
    folder = tmp_path / "results" / "radar-animation"

    async def simulate(animation=False, on_submitted=None, request=None):
        submitted.append(request)
        return "done"

    def query(name, params):
        assert name == "animation_export"
        return {"name": name}

    def persist(api, reference):
        from test_install_tools import _write_result
        return str(_write_result(folder / "studio-animation-static.npz", 4))

    monkeypatch.setattr(component, "_simulate_async", simulate)
    monkeypatch.setattr(component, "_query_result", query)
    monkeypatch.setattr("wt_radar.adapter.animation.persist_export", persist)
    result = asyncio.run(tool.run({"scene_id": original.scene_id, "radar_object_id": "radar", "frame_count": 4}))
    assert submitted[0]["adapter"] == static_capture.STATIC_MODEL and submitted[0]["frame_count"] == 4
    assert result["ok"] and result["status"] == "captured"
    assert result["export_path"].endswith("studio-animation-static.npz")
    assert result["checks"]["all_finite"] and result["checks"]["frame_count_matches"]
    assert result["mutates_scene"] is False
