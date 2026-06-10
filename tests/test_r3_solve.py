"""R3 live solve: tracer + 3 backends + sigproc views + library prefabs.

Round-trips a radar example into the studio scene, runs ``radar.simulate`` on each
available backend, and checks the raw signal shape matches a direct platform run; the
range-doppler / point-cloud / CFAR / MUSIC views produce the expected shapes; the
pytorch-backend signal matches a direct platform call within tolerance; and the library
"Radar (Demo)" prefab round-trips + solves.

Live solve needs CUDA (the mitsuba ray tracer is CUDA-only), so the whole module is
gated on a CUDA device via the ``cuda_ready`` fixture.
"""
import json
import tomllib
import numpy as np
import pytest
import torch
import witwin.radar as wr

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
    radar = studio_settings(studio).get_component("Radar")
    radar.backend = backend
    radar.device = "cpu" if backend == "pytorch" else "cuda"
    return SolveRunner.run(
        studio,
        sensor=SensorSpec.from_component(radar),
        tracer=TracerSpec(resolution=_RESOLUTION),
        motion_sampling="per_chirp", t0=0.0)


def studio_settings(studio):
    return next(o for o in studio.objects.values() if o.get_component("Radar") is not None)


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
    direct = wr.Radar(wr.RadarConfig.from_dict(_CONFIG), backend="pytorch", device="cuda")
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


def test_sensor_pose_follows_transform(adapter):
    """The SensorSpec position/up come from the owner Transform, not from a separate field."""
    studio = adapter.to_studio((_scene("cpu"), wr.RadarConfig.from_dict(_CONFIG)))
    settings = studio_settings(studio)
    transform = settings.get_component("Transform")
    transform.position = [1.5, -2.0, 0.5]
    transform.rotation = [0.0, 0.0, 0.0]   # identity -> forward is local -Z, up is +Y

    spec = SensorSpec.from_component(settings.get_component("Radar"))

    assert spec.position == pytest.approx([1.5, -2.0, 0.5])
    assert spec.up == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    # target = position + forward; identity rotation -> forward = (0, 0, -1).
    assert spec.target == pytest.approx([1.5, -2.0, -0.5], abs=1e-6)


def test_built_radar_tx_pos_follows_transform(adapter):
    """Moving the Transform shifts the wr.Radar's tx_pos / rx_pos in world coords.

    This catches the case where SensorSpec is right but the value never reaches the
    platform: `build_radar(config, spec)` constructs a real wr.Radar from the spec,
    which recomputes tx_pos = world_from_local_points(tx_loc). The resulting tx_pos
    is what compute_total_path_lengths uses to drive the simulated signal, so this
    is the closest pose-side check we can run on CPU (no CUDA needed).
    """
    studio = adapter.to_studio((_scene("cpu"), wr.RadarConfig.from_dict(_CONFIG)))
    settings = studio_settings(studio)
    radar_comp = settings.get_component("Radar")
    radar_comp.backend = "pytorch"  # CPU-capable; we only inspect pose, not solve
    radar_comp.device = "cpu"

    settings.get_component("Transform").position = [10.0, 0.0, 0.0]
    radar_a = SolveRunner.build_radar(
        wr.RadarConfig.from_dict(_CONFIG), SensorSpec.from_component(radar_comp))

    settings.get_component("Transform").position = [-7.0, 0.0, 0.0]
    radar_b = SolveRunner.build_radar(
        wr.RadarConfig.from_dict(_CONFIG), SensorSpec.from_component(radar_comp))

    assert radar_a.position.tolist() == pytest.approx([10.0, 0.0, 0.0])
    assert radar_b.position.tolist() == pytest.approx([-7.0, 0.0, 0.0])
    # tx_pos[0] is antenna 0 at local (0,0,0) -> equals radar position in world.
    assert radar_a.tx_pos[0].tolist() == pytest.approx([10.0, 0.0, 0.0], abs=1e-4)
    assert radar_b.tx_pos[0].tolist() == pytest.approx([-7.0, 0.0, 0.0], abs=1e-4)


def test_manifest_declares_rfc012_solver():
    manifest = tomllib.loads((__import__("pathlib").Path(__file__).resolve().parents[1] / "witwin.toml").read_text())
    solvers = manifest["contributes"]["solvers"]
    assert solvers[0]["id"] == "witwin.radar.simulate"
    assert solvers[0]["entry"] == "solver_host.py"
    assert solvers[0]["runtime"]["python"] == "${backend:python}"
    assert solvers[0]["lifecycle"] == "warm"
    assert solvers[0]["maxParallel"] == 1


def test_radar_defaults_keep_static_returns_visible(adapter):
    studio = adapter.to_studio((_scene("cpu"), wr.RadarConfig.from_dict(_CONFIG)))
    radar = studio_settings(studio).get_component("Radar")

    assert bool(radar.static_clutter_removal) is False


def test_radar_component_uses_solver_api_for_simulate(adapter, monkeypatch):
    import wt_radar.components.radar as radar_mod

    studio = adapter.to_studio((_scene("cpu"), wr.RadarConfig.from_dict(_CONFIG)))
    radar = studio_settings(studio).get_component("Radar")
    radar.backend = "pytorch"
    radar.device = "cpu"

    class FakeRun:
        status = "succeeded"
        outputs = {"resultHandle": "radar-handle"}
        run_id = "radar-run"
        error = None

    class FakeSolvers:
        def __init__(self):
            self.solve_kwargs = None
            self.queries = []

        def solve(self, solver_id, **kwargs):
            self.solve_kwargs = {"solver_id": solver_id, **kwargs}
            return FakeRun()

        def query(self, solver_id, result_handle, op, params, **kwargs):
            self.queries.append((solver_id, result_handle, op, params, kwargs))
            return {
                "data": {
                    "tx": 0,
                    "rx": 0,
                    "mag_db": np.ones((4, 5), dtype=np.float32),
                    "cfar_rows": [],
                    "cfar_cols": [],
                }
            }

    fake = FakeSolvers()
    notifications = {"success": [], "error": []}
    published_channels = []

    class FakeNotifications:
        @staticmethod
        def success(title, message):
            notifications["success"].append((title, message))

        @staticmethod
        def error(title, message):
            notifications["error"].append((title, message))

    monkeypatch.setattr(radar_mod, "api", type("FakeApi", (), {"solvers": fake})())
    monkeypatch.setattr(radar_mod, "Notifications", FakeNotifications)
    monkeypatch.setattr(radar, "_publish_signal_stream", lambda channels=None: published_channels.append(channels))

    msg = radar.simulate()

    assert msg == "Solve complete"
    assert radar._solver_result_handle == "radar-handle"
    assert radar._solver_run_id == "radar-run"
    assert fake.solve_kwargs["solver_id"] == "witwin.radar.simulate"
    assert fake.solve_kwargs["scene"] is studio
    assert "surface_progress" not in fake.solve_kwargs
    assert fake.solve_kwargs["config"]["sensor"]["backend"] == "pytorch"
    assert fake.solve_kwargs["config"]["tracer"]["resolution"] == radar.resolution
    assert fake.queries[0][2] == "range_doppler"
    assert fake.queries[0][3]["static_clutter_removal"] is False
    assert published_channels == [("rd",)]
    assert notifications["success"] == [("Radar", "Radar solve complete")]
    assert notifications["error"] == []


def test_remote_stream_publish_accepts_numpy_query_payload(adapter, monkeypatch):
    import wt_radar.components.radar as radar_mod

    studio = adapter.to_studio((_scene("cpu"), wr.RadarConfig.from_dict(_CONFIG)))
    radar = studio_settings(studio).get_component("Radar")
    radar._solver_result_handle = "radar-handle"
    radar._solver_run_id = "radar-run"

    class FakeSolvers:
        def query(self, solver_id, result_handle, op, params, **kwargs):
            assert op == "range_doppler"
            return {
                "data": {
                    "tx": 0,
                    "rx": 0,
                    "mag_db": np.ones((2, 3), dtype=np.float32),
                    "cfar_rows": [],
                    "cfar_cols": [],
                }
            }

    class FakeStream:
        def __init__(self):
            self.published = []

        def publish(self, channel_id, payload, metadata=None):
            self.published.append((channel_id, np.asarray(payload), metadata or {}))

    monkeypatch.setattr(radar_mod, "api", type("FakeApi", (), {"solvers": FakeSolvers()})())
    stream = FakeStream()

    radar._publish_rd_stream(stream)

    assert len(stream.published) == 1
    channel_id, payload, metadata = stream.published[0]
    assert channel_id == "rd"
    assert payload.shape == (2, 3)
    assert metadata["shape"] == [2, 3]


def test_show_frame_clears_solver_handle():
    from wt_radar.components.radar import RadarComponent

    radar = RadarComponent()
    radar._frames = torch.zeros((1, 1, 1, 1, 1), dtype=torch.complex64)
    radar._timeline_radar = object()
    radar._solver_result_handle = "stale-handle"
    radar._solver_run_id = "stale-run"
    observed = []
    radar.update_view = lambda: observed.append((radar._solver_result_handle, radar._signal.shape))

    msg = radar.show_frame()

    assert msg == "Frame 0"
    assert radar._solver_result_handle == ""
    assert radar._solver_run_id == ""
    assert observed == [("", torch.Size([1, 1, 1, 1]))]


def test_timeline_frames_register_radar_result_dataset(monkeypatch):
    import wt_radar.components.radar as radar_mod
    from wt_radar.components.radar import RadarComponent

    class FakeDataSources:
        def __init__(self):
            self.timeline_datasets = []

        def register_timeline_dataset(self, descriptor):
            self.timeline_datasets.append(descriptor)
            return descriptor

    fake_data_sources = FakeDataSources()
    monkeypatch.setattr(radar_mod, "api", type("FakeApi", (), {"data_sources": fake_data_sources})())

    radar = RadarComponent()
    radar._signal_source_id = "radar.demo.result"
    radar.frame_rate = 10.0
    radar.frame_index = 1
    radar._frames = torch.zeros((2, 3, 4, 5, 6), dtype=torch.complex64)

    radar._register_signal_timeline_dataset()

    assert fake_data_sources.timeline_datasets
    descriptor = fake_data_sources.timeline_datasets[-1]
    assert descriptor["datasetId"] == "radar.demo.result"
    assert descriptor["kind"] == "radar.result"
    assert descriptor["defaultChannel"] == "rd"
    assert descriptor["timebase"] == {"unit": "seconds", "times": [0.0, 0.1], "fps": 10.0}
    assert {channel["channelId"] for channel in descriptor["channels"]} == {"raw", "rd", "pc"}
    assert len(descriptor["frames"]) == 6
    rd_frame = next(item["frame"] for item in descriptor["frames"] if item["time"] == 0.1 and item["channelId"] == "rd")
    assert rd_frame["sourceId"] == "radar.demo.result"
    assert rd_frame["semantic"] == "radar.range_doppler"
    assert rd_frame["status"] == "pending"
    assert radar.signal_source["mode"] == "timeline"
    assert radar.signal_source["sourceId"] == "radar.demo.result"
    assert radar.signal_source["defaultTime"] == 0.1


@pytest.mark.gpu
def test_library_demo_round_trips_and_solves(wtr, adapter, cuda_ready):
    # The "Radar (Demo)" prefab: a settings object + a plain target that solves.
    from witwin_server import Scene

    from wt_radar.library_items import _settings_object, _target_object

    studio = Scene()
    studio.begin_batch()
    studio.add_object(_settings_object())
    studio.add_object(_target_object())
    studio.end_batch()

    scene, config = adapter.to_platform(studio)
    assert len(scene.structures) == 1 and config.num_tx == 3
    radar = studio_settings(studio).get_component("Radar")
    radar.backend = "dirichlet"
    result = SolveRunner.run(studio, sensor=SensorSpec.from_component(radar),
                             tracer=TracerSpec(resolution=_RESOLUTION),
                             motion_sampling="per_chirp", t0=0.0)
    assert tuple(result.signal.shape) == _SHAPE
