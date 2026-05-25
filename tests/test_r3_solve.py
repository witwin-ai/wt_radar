"""R3 live solve: tracer + 3 backends + sigproc views + library prefabs.

Round-trips a radar example into the studio scene, runs ``radar.simulate`` on each
available backend, and checks the raw signal shape matches a direct platform run; the
range-doppler / point-cloud / CFAR / MUSIC views produce the expected shapes; the
pytorch-backend signal matches a direct platform call within tolerance; and the library
"Radar (Demo)" prefab round-trips + solves.

Live solve needs CUDA (the mitsuba ray tracer is CUDA-only), so the whole module is
gated on a CUDA device via the ``cuda_ready`` fixture.
"""
import numpy as np
import pytest
import torch
import witwin.radar as wr

from wt_radar.adapter.radar_adapter import RadarAdapter
from wt_radar.adapter.solve import SensorSpec, SigProc, SolveRunner, TracerSpec

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
_SHAPE = (3, 4, 128, 256)
_RESOLUTION = 64


def _scene(device):
    # A static wall + a dynamic (triangle-sampled) box target.
    scene = wr.Scene(device=device)
    scene.add_mesh(name="Wall", geometry=wr.Box(position=(0.0, 0.0, -4.0), size=(2.0, 2.0, 0.1)),
                   material=wr.Material(eps_r=5.0), dynamic=True)
    scene.add_mesh(name="Car", geometry=wr.Box(position=(0.6, 0.0, -2.5), size=(0.6, 0.4, 0.4)),
                   material=wr.Material(eps_r=8.0), dynamic=True)
    return scene


def _backends():
    return ["dirichlet", "slang", "pytorch"]


def _solve_via_adapter(adapter, backend):
    studio = adapter.to_studio((_scene("cpu"), wr.RadarConfig.from_dict(_CONFIG)))
    sensor = studio_settings(studio).get_component("RadarSensor")
    sensor.backend = backend
    sensor.device = "cpu" if backend == "pytorch" else "cuda"
    return SolveRunner.run(
        studio,
        sensor=SensorSpec.from_component(sensor),
        tracer=TracerSpec(resolution=_RESOLUTION),
        motion_sampling="per_chirp", t0=0.0)


def studio_settings(studio):
    return next(o for o in studio.objects.values() if o.get_component("RadarConfig") is not None)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", _backends())
def test_solve_backend_shape(adapter, cuda_ready, backend):
    result = _solve_via_adapter(adapter, backend)
    assert tuple(result.signal.shape) == _SHAPE


@pytest.mark.gpu
def test_sigproc_views(adapter, cuda_ready):
    result = _solve_via_adapter(adapter, "dirichlet")
    rd = SigProc.range_doppler(result.radar, result.signal, tx=0, rx=0, static_clutter_removal=True)
    assert rd.mag_db.shape == (128, 256)
    assert rd.velocities.shape == (128,)
    pc = SigProc.point_cloud(result.radar, result.signal, detector="cfar", static_clutter_removal=True,
                             guard=(2, 4), training=(4, 8), pfa=1e-3, energy_top_k=128)
    assert pc.ndim == 2 and pc.shape[1] == 6
    mask = SigProc.cfar_mask(rd.rd_map, guard=(2, 4), training=(4, 8), pfa=1e-3)
    assert mask.shape == rd.mag_db.shape


@pytest.mark.gpu
def test_pytorch_matches_direct(adapter, cuda_ready):
    # The adapter-rebuilt scene must trace identically to a directly-built one.
    direct_scene = _scene("cuda")
    direct = wr.Radar(wr.RadarConfig.from_dict(_CONFIG), backend="pytorch", device="cpu")
    sig_direct = direct.simulate(direct_scene, resolution=_RESOLUTION, sampling="triangle")
    sig_adapter = _solve_via_adapter(adapter, "pytorch").signal
    assert tuple(sig_adapter.shape) == tuple(sig_direct.shape)
    assert torch.allclose(sig_adapter, sig_direct, atol=1e-4, rtol=1e-3)


@pytest.mark.gpu
def test_music_image(wtr, cuda_ready):
    # MUSIC on a 20x20 UPA (degenerate on small arrays); uses a closure frame (no tracing).
    cfg = dict(_CONFIG, num_tx=20, num_rx=20, chirp_per_frame=8, num_doppler_bins=8,
               tx_loc=[[i, 0, 0] for i in range(20)], rx_loc=[[20, -i, 0] for i in range(20)])
    radar = wr.Radar(wr.RadarConfig.from_dict(cfg), backend="dirichlet", device="cuda")
    points = np.array([[-0.5, 0, -3], [0.5, 0, -3]], dtype=np.float32)

    def location(t):
        pos = torch.tensor(points, dtype=torch.float32, device=radar.device)
        return torch.ones(pos.shape[0], device=radar.device), pos

    frame = radar.mimo(location, t0=0)
    img = SigProc.music_image(radar, frame, num_pixels=48)
    assert img.shape == (48, 48)


@pytest.mark.gpu
def test_library_demo_round_trips_and_solves(wtr, adapter, cuda_ready):
    # The "Radar (Demo)" prefab: a settings object + a moving target that solves.
    from witwin_server import Scene

    from wt_radar.library_items import _settings_object, _target_object

    studio = Scene(kind="local")
    studio.begin_batch()
    studio.add_object(_settings_object())
    studio.add_object(_target_object())
    studio.end_batch()

    scene, config = adapter.to_platform(studio)
    assert len(scene.structures) == 1 and config.num_tx == 3
    sensor = studio_settings(studio).get_component("RadarSensor")
    sensor.backend = "dirichlet"
    result = SolveRunner.run(studio, sensor=SensorSpec.from_component(sensor),
                             tracer=TracerSpec(resolution=_RESOLUTION),
                             motion_sampling="per_chirp", t0=0.0)
    assert tuple(result.signal.shape) == _SHAPE
