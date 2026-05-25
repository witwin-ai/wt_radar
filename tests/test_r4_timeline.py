"""R4 live: multi-frame timeline (point-cloud sequence + motion source).

A point-cloud sequence file drives ``Timeline.generate`` to produce a frame stack that
matches a direct platform call; the frame slider pushes a frame into the RadarResult view;
the motion source raises cleanly when the scene has no SMPL ``human``. Needs CUDA.
"""
import numpy as np
import pytest
import torch
import witwin.radar as wr

from witwin_server import Scene

from wt_radar.adapter.config_map import ConfigMap

_CONFIG = {
    "num_tx": 3, "num_rx": 4,
    "fc": 77e9, "slope": 60.012, "power": 15.0,
    "adc_samples": 256, "adc_start_time": 6.0, "sample_rate": 4400.0,
    "idle_time": 7.0, "ramp_end_time": 58.0, "chirp_per_frame": 128,
    "frame_per_second": 10.0,
    "num_doppler_bins": 128, "num_range_bins": 256, "num_angle_bins": 64,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
}


def _settings_in_scene():
    settings = ConfigMap.settings_to_studio(wr.RadarConfig.from_dict(_CONFIG))
    studio = Scene(kind="local")
    studio.begin_batch()
    studio.add_object(settings)
    studio.end_batch()
    return settings


def _pointcloud_seq(num_frames=5):
    # One point translating in +x over the sequence.
    return np.stack([np.array([[0.1 * i, 0.0, -3.0]], dtype=np.float32) for i in range(num_frames)], axis=0)


@pytest.mark.gpu
def test_pointcloud_timeline_matches_direct(wtr, cuda_ready, tmp_path):
    positions = _pointcloud_seq()
    path = tmp_path / "seq.npz"
    np.savez(path, positions=positions)

    settings = _settings_in_scene()
    timeline = settings.get_component("RadarTimeline")
    timeline.frame_rate = 10.0
    timeline.source = "pointcloud_sequence"
    timeline.pointcloud_path = str(path)
    timeline.velocity_corrected = False
    timeline.generate()
    frames = timeline._frames
    assert frames is not None and frames.shape[1:] == (3, 4, 128, 256) and frames.shape[0] >= 1

    direct_tl = wr.Timeline(frame_rate=10.0, device="cuda")
    direct_tl.add_pointcloud_sequence(torch.as_tensor(positions, device="cuda"))
    direct_radar = wr.Radar(wr.RadarConfig.from_dict(_CONFIG), backend="dirichlet", device="cuda")
    direct = direct_tl.generate(direct_radar, progress=False, velocity_corrected=False)
    assert frames.shape == direct.shape
    assert torch.allclose(frames, direct, atol=1e-4, rtol=1e-3)


@pytest.mark.gpu
def test_frame_slider_pushes_to_result(wtr, cuda_ready, tmp_path):
    path = tmp_path / "seq.npz"
    np.savez(path, positions=_pointcloud_seq())
    settings = _settings_in_scene()
    timeline = settings.get_component("RadarTimeline")
    timeline.frame_rate = 10.0
    timeline.pointcloud_path = str(path)
    timeline.velocity_corrected = False
    timeline.generate()

    timeline.frame_index = 1
    timeline.show_frame()
    result = settings.get_component("RadarResult")
    assert result._signal is not None
    assert torch.equal(result._signal, timeline._frames[1])


@pytest.mark.gpu
def test_motion_source_requires_human(wtr, cuda_ready, tmp_path):
    path = tmp_path / "motion.npz"
    np.savez(path, pose=np.zeros((1, 72), dtype=np.float32), shape=np.zeros(10, dtype=np.float32),
             root_translation=np.zeros((1, 3), dtype=np.float32))
    settings = _settings_in_scene()
    timeline = settings.get_component("RadarTimeline")
    timeline.source = "motion"
    timeline.motion_path = str(path)
    with pytest.raises(KeyError):
        timeline.generate()
