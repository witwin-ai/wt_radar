"""Transport and UI checks; no solver or signal-processing algorithm is replaced."""
import dataclasses
import json

import numpy as np
import pytest
import torch

from witwin.radar import RadarConfig
from witwin.radar.radar import RadarSystemConfig
from witwin.radar.processing import ProcessingAxes
from witwin.radar.synthesis.assembly import SynthesisResult
from wt_radar.adapter.saved_result import SavedRadarResult
from wt_radar.components.radar import RadarComponent


@pytest.fixture
def payload():
    system = RadarSystemConfig.from_radar_config(RadarConfig.from_dict({
        "num_tx": 1, "num_rx": 1, "tx_loc": [[0, 0, 0]], "rx_loc": [[0, 0, 0]],
        "adc_samples": 8, "num_range_bins": 8, "chirp_per_frame": 4, "num_doppler_bins": 4,
        "fc": 77e9, "slope": 60.012, "adc_start_time": 6, "sample_rate": 4400,
        "idle_time": 7, "ramp_end_time": 65, "frame_per_second": 10,
        "num_angle_bins": 8, "power": 15,
    }))
    sample = torch.complex(torch.arange(32, dtype=torch.float32).reshape(4, 1, 8), torch.ones(4, 1, 8))
    axes = ProcessingAxes.from_synthesis(
        SynthesisResult.from_fmcw(sample, system.waveform_spec()), system.waveform_spec(), system.sensors.array,
    )
    record = dataclasses.asdict(axes)
    for key in ("range_m", "velocity_mps"):
        record[key] = record[key].tolist()
    cube = np.stack([sample[:, 0].numpy()[None, None], 2 * sample[:, 0].numpy()[None, None]])
    return {
        "cube": cube, "times_s": np.array([0.0, 0.1]), "result_schema_version": np.asarray(1),
        "processing_axes_json": np.asarray(json.dumps(record)),
        "producer_metadata_json": np.asarray(json.dumps({"scene_id": "synthetic transport fixture"})),
        "range_m": axes.range_m.numpy(), "velocity_mps": axes.velocity_mps.numpy(),
    }


def save(tmp_path, payload):
    path = tmp_path / "radar_cube.npz"
    np.savez(path, **payload)
    return path


def test_spectrum_passes_official_profile_without_fft_or_filter(tmp_path, payload, monkeypatch):
    result = SavedRadarResult.load(save(tmp_path, payload))
    def forbidden(*args, **kwargs):
        raise AssertionError("A saved range spectrum must not be FFT'd again")
    monkeypatch.setattr(torch.fft, "fft", forbidden)
    monkeypatch.setattr(torch.fft, "ifft", forbidden)
    x, y = result.selected_profile(1, 0, 0, 2)
    assert torch.equal(y, result.cube[1, 0, 0, 2])
    assert np.array_equal(x.numpy(), payload["range_m"])


def test_component_reuses_existing_figure_and_recorded_time(tmp_path, payload):
    component = RadarComponent()
    component.saved_result_path = str(save(tmp_path, payload))
    component.load_saved_result()
    component.saved_frame_index = 1
    component.saved_chirp_index = 2
    component.update_view()
    figure = component.signal_figure.to_dict()
    assert figure["type"] == "line"
    assert figure["data"]["x"] == payload["range_m"].tolist()
    assert figure["data"]["y"] == torch.from_numpy(payload["cube"][1, 0, 0, 2]).abs().tolist()
    assert "0.1000s" in figure["title"]
    assert component._radar is None and component._signal is None
    assert component.signal_source is None and component.signal_stream is None


@pytest.mark.parametrize("change", ["metadata", "shape", "time", "nonfinite", "axis", "real", "beat"])
def test_invalid_result_is_refused(tmp_path, payload, change):
    if change == "metadata":
        del payload["processing_axes_json"]
    elif change == "shape":
        payload["cube"] = payload["cube"][..., :7]
    elif change == "time":
        payload["times_s"] = np.array([0.1, 0.0])
    elif change == "nonfinite":
        payload["cube"][0, 0, 0, 0, 0] = np.nan
    elif change == "axis":
        payload["range_m"] = payload["range_m"] + 1
    elif change == "real":
        payload["cube"] = payload["cube"].real
    else:
        record = json.loads(payload["processing_axes_json"].item())
        record["output_domain"] = "beat"
        payload["processing_axes_json"] = np.asarray(json.dumps(record))
    with pytest.raises(ValueError):
        SavedRadarResult.load(save(tmp_path, payload))


def test_invalid_indices_not_silently_clamped(tmp_path, payload):
    result = SavedRadarResult.load(save(tmp_path, payload))
    for selection in ((2, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 4)):
        with pytest.raises(ValueError):
            result.selected_profile(*selection)


def test_saved_mode_refuses_legacy_dsp(tmp_path, payload):
    component = RadarComponent()
    component.saved_result_path = str(save(tmp_path, payload))
    component.load_saved_result()
    component.view = "range_doppler"
    assert component.signal_figure.to_dict()["type"] == "imshow"
    with pytest.raises(ValueError, match="supports"):
        component.view = "music"
    component.view = "range_profile"
    with pytest.raises(ValueError, match="does not apply"):
        component.static_clutter_removal = True


def test_live_commands_fail_explicitly_on_current_api():
    component = RadarComponent()
    for command in (component.start_stream, component.generate_timeline):
        with pytest.raises(RuntimeError, match="not connected to Radar 0.3"):
            command()


def test_path_change_and_failed_reload_clear_stale_result(tmp_path, payload):
    component = RadarComponent()
    component.saved_result_path = str(save(tmp_path, payload))
    component.load_saved_result()
    component.saved_result_path = str(tmp_path / "missing.npz")
    assert component._saved_result is None
    assert not component.saved_result_loaded
    assert not component.signal_figure._series
    with pytest.raises(FileNotFoundError):
        component.load_saved_result()
    assert "Load failed" in component.saved_result_status
    assert component._saved_result is None
    assert not component.saved_result_loaded


def test_saved_runtime_flags_not_persisted(tmp_path, payload):
    component = RadarComponent()
    component.saved_result_path = str(save(tmp_path, payload))
    component.load_saved_result()
    for name in ("saved_result_loaded", "saved_result_status"):
        assert component._fields_meta[name].transient


def test_configuration_edits_do_not_reinterpret_saved_profile(tmp_path, payload):
    component = RadarComponent()
    component.saved_result_path = str(save(tmp_path, payload))
    component.load_saved_result()
    before = component.signal_figure.to_dict()
    component.slope = 123.0
    component.ramp_end_time = 58.0
    component.update_view()
    after = component.signal_figure.to_dict()
    # A redraw has its own wall-clock timestamp; the saved signal and axes must
    # stay identical, not the rendering event's timestamp.
    assert {k: v for k, v in after.items() if k != "timestamp"} == {
        k: v for k, v in before.items() if k != "timestamp"
    }


@pytest.mark.parametrize("field,value", [
    ("range_bin_m", -1), ("range_bin_m", 1), ("range_origin_m", 100),
    ("slow_time_period_s", float("nan")), ("reference_frequency_hz", -1),
    ("fast_time_name", "sample"), ("tx_loc_half_wavelength", []), ("doppler_sign", -1),
])
def test_contradictory_metadata_is_refused(tmp_path, payload, field, value):
    record = json.loads(payload["processing_axes_json"].item())
    record[field] = value
    payload["processing_axes_json"] = np.asarray(json.dumps(record))
    with pytest.raises(ValueError):
        SavedRadarResult.load(save(tmp_path, payload))


def test_complex_spectrum_keeps_both_components(tmp_path, payload):
    component = RadarComponent()
    component.saved_result_path = str(save(tmp_path, payload))
    component.load_saved_result()
    component.view = "range_spectrum"
    figure = component.signal_figure.to_dict()
    assert len(figure["data"]["series"]) == 2
    assert figure["data"]["series"][0]["y"] == payload["cube"][0, 0, 0, 0].real.tolist()
    assert figure["data"]["series"][1]["y"] == payload["cube"][0, 0, 0, 0].imag.tolist()
