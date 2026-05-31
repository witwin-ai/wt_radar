"""R5 live: pluggable post-processors + multi-radar simulate_group.

A referenced RadarPostProcessor renders its own view after a solve; ``simulate_group`` runs
several named radars on one rebuilt scene. Needs CUDA.
"""
import dataclasses

import numpy as np
import pytest
import witwin.radar as wr

from witwin_server import SceneObject

from wt_radar.adapter.solve import SensorSpec, SolveRunner, TracerSpec

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


def _studio(adapter):
    scene = wr.Scene(device="cpu")
    scene.add_mesh(name="Car", geometry=wr.Box(position=(0.6, 0.0, -2.5), size=(0.6, 0.4, 0.4)),
                   material=wr.Material(eps_r=8.0), dynamic=True)
    return adapter.to_studio((scene, wr.RadarConfig.from_dict(_CONFIG)))


def _settings(studio):
    return next(o for o in studio.objects.values() if o.get_component("Radar") is not None)


def test_post_processor_runs_after_solve(adapter, monkeypatch):
    import wt_radar.components.radar as radar_mod
    from wt_radar.components.post_processing import RangeDopplerProcessorComponent

    studio = _studio(adapter)
    settings = _settings(studio)
    proc_obj = SceneObject(name="RD Processor", mesh_type="Empty")
    proc_obj.add_component(RangeDopplerProcessorComponent())
    studio.add_object(proc_obj)

    radar = settings.get_component("Radar")
    radar.post_processors = [{"object_id": proc_obj.id, "component_type": "RangeDopplerProcessor"}]
    radar.backend = "dirichlet"

    class FakeRun:
        status = "succeeded"
        outputs = {"resultHandle": "radar-handle"}
        run_id = "radar-run"
        error = None

    class FakeSolvers:
        def solve(self, solver_id, **kwargs):
            return FakeRun()

        def query(self, solver_id, result_handle, op, params, **kwargs):
            assert op == "range_doppler"
            return {
                "data": {
                    "tx": 0,
                    "rx": 0,
                    "mag_db": np.ones((4, 5), dtype=np.float32),
                    "cfar_rows": [],
                    "cfar_cols": [],
                }
            }

    monkeypatch.setattr(radar_mod, "api", type("FakeApi", (), {"solvers": FakeSolvers()})())
    radar.simulate()

    proc = proc_obj.get_component("RangeDopplerProcessor")
    assert proc.result_figure._data  # imshow populated the processor's figure


@pytest.mark.gpu
def test_simulate_group(adapter, cuda_ready):
    studio = _studio(adapter)
    base = SensorSpec.from_component(_settings(studio).get_component("Radar"))
    sensors = {
        "front": dataclasses.replace(base, position=[0.0, 0.0, 0.0]),
        "side": dataclasses.replace(base, position=[1.0, 0.0, 0.0]),
    }
    signals = SolveRunner.run_group(studio, sensors=sensors, tracer=TracerSpec(resolution=64),
                                    motion_sampling="per_chirp", t0=0.0)
    assert set(signals) == {"front", "side"}
    assert all(tuple(sig.shape) == _SHAPE for sig in signals.values())
