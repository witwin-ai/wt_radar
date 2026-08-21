"""Core LinearTrajectory and DynamicScene coverage."""
import numpy as np
import torch

from _helpers import make_config, make_scene


def test_dynamic_scene_round_trip(adapter):
    import witwin.core as wc

    original = make_scene(dynamic=True)
    rebuilt, _ = adapter.to_platform(adapter.to_studio((original, make_config())))

    assert isinstance(rebuilt, wc.DynamicScene)
    assert set(int(key) for key in rebuilt.structure_trajectories) == {2}
    trajectory = rebuilt.structure_trajectories[next(iter(rebuilt.structure_trajectories))]
    np.testing.assert_allclose(trajectory.origin.cpu(), [0.2, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(trajectory.velocity.cpu(), [0.1, 0.0, 0.0], atol=1e-6)
    assert trajectory.reference_time_s == 0.5


def test_rotation_round_trip(adapter):
    import witwin.core as wc

    scene = make_scene()
    trajectory = wc.LinearTrajectory(
        origin=torch.zeros(3),
        velocity=torch.zeros(3),
        rotation=torch.tensor((0.9238795, 0.0, 0.3826834, 0.0)),
        angular_velocity=torch.tensor((0.0, 1.5, 0.0)),
        reference_time_s=0.25,
    )
    dynamic = wc.DynamicScene(scene, structure_trajectories={1: trajectory})
    rebuilt, _ = adapter.to_platform(adapter.to_studio((dynamic, make_config())))
    result = rebuilt.structure_trajectories[next(iter(rebuilt.structure_trajectories))]

    np.testing.assert_allclose(result.rotation.cpu(), trajectory.rotation.cpu(), atol=1e-5)
    np.testing.assert_allclose(result.angular_velocity.cpu(), trajectory.angular_velocity.cpu(), atol=1e-5)
    assert result.reference_time_s == 0.25


def test_positions_at_applies_core_trajectory(adapter):
    from wt_radar.components.scatterer import RadarScattererComponent

    studio = adapter.to_studio((make_scene(dynamic=True), make_config()))
    target = next(obj for obj in studio.objects.values() if obj.name == "car")
    target.add_component(RadarScattererComponent())
    exported = adapter.build_export(studio)
    positions = exported.positions_at(1.5, device="cpu").numpy()
    base = np.asarray(exported.site_positions_m)
    trajectory = exported.scene.structure_trajectories[2]
    translation = trajectory.at(1.5).translation.detach().cpu().numpy()
    np.testing.assert_allclose(positions[0], base[0] + translation, atol=1e-6)


def test_no_motion_component_produces_static_scene(adapter):
    import witwin.core as wc

    rebuilt, _ = adapter.to_platform(adapter.to_studio((make_scene(), make_config())))
    assert isinstance(rebuilt, wc.Scene)
    assert not isinstance(rebuilt, wc.DynamicScene)
