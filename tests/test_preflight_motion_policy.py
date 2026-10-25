"""A site the solve's frozen topology cannot reach is excluded for the whole interval.

A quick person run measured with adaptive motion failed at frame 2 with "site
3000125 has no inbound leg row in the frozen topology": a foot site 0.6 mm
above the floor was reachable at every Studio frame in the chirp visibility
preflight, but not inside the adaptive solve's frozen topology. The native
solve is the final authority on reachability, so the solve drops that site for
the whole interval (the same interval-global rule the preflight applies) and
restarts instead of failing the measurement.
"""
import pytest

from test_animation import AnimationContext, make_rig  # noqa: F401
from test_snapshot import copy_for_solver, scene  # noqa: F401
from wt_radar.adapter import animation as anim


class _GapOnce:
    """Radar proxy whose first stream loses one site's leg row at frame 1."""

    def __init__(self, radar, site_id):
        self._radar = radar
        self._site_id = site_id
        self.stream_calls = 0

    def __getattr__(self, name):
        return getattr(self._radar, name)

    def stream(self, world, targets, **kwargs):
        self.stream_calls += 1
        inner = iter(self._radar.stream(world, targets, **kwargs))
        if self.stream_calls > 1:
            return inner

        def failing():
            yield next(inner)
            raise ValueError(f"site {self._site_id} has no inbound leg row in the frozen topology")

        return failing()


def test_solve_excludes_a_site_its_frozen_topology_loses_and_restarts(scene, cuda_ready, monkeypatch):
    original, component = scene
    make_rig(original, component)
    request = anim.animation_request(component)
    proxies = []
    prepare = anim.prepare_animation

    def prepare_with_gap(*args, **kwargs):
        times, sampler, radar, *rest = prepare(*args, **kwargs)
        proxy = _GapOnce(radar, 3_000_001)
        proxies.append(proxy)
        return (times, sampler, proxy, *rest)

    monkeypatch.setattr(anim, "prepare_animation", prepare_with_gap)
    result = anim.solve_animation(AnimationContext(), copy_for_solver(original), request)

    assert proxies[0].stream_calls == 2
    topology = result.metadata["topology_preflight"]
    assert topology["active_site_ids"] == [3_000_000]
    assert topology["occluded_site_count"] == 1
    assert topology["solve_exclusions"] == [{"site_id": 3_000_001, "site_rank": 1, "frame_index": 1}]
    assert result.metadata["active_site_ids"] == [3_000_000]
    assert len(result.metadata["site_vertex_indices"]) == 1
    assert result.cube.shape[0] == len(result.times_s)


def test_solve_still_fails_when_an_unrelated_native_error_occurs(scene, cuda_ready, monkeypatch):
    original, component = scene
    make_rig(original, component)
    request = anim.animation_request(component)
    prepare = anim.prepare_animation

    class _Broken(_GapOnce):
        def stream(self, world, targets, **kwargs):
            raise ValueError("unrelated native failure")

    def prepare_broken(*args, **kwargs):
        times, sampler, radar, *rest = prepare(*args, **kwargs)
        return (times, sampler, _Broken(radar, 0), *rest)

    monkeypatch.setattr(anim, "prepare_animation", prepare_broken)
    with pytest.raises(Exception, match="unrelated native failure"):
        anim.solve_animation(AnimationContext(), copy_for_solver(original), request)


def test_an_empty_scene_exports_free_space_for_a_line_of_sight_measurement():
    from witwin_server.core.scene import Scene
    from wt_radar.adapter.studio_geometry import export_static_mesh_scene

    exported = export_static_mesh_scene(Scene(scene_id="void"))
    assert exported.mesh_count == 0 and exported.face_count == 0
    assert exported.object_to_structure_id == {}
