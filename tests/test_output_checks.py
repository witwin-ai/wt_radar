"""Radar output checks (design 7.6): real, finite, targets where they should be."""
import numpy as np
import pytest
import torch

from witwin.radar import Radar
from witwin.radar.processing import ProcessingAxes
from witwin.radar.synthesis.assembly import SynthesisResult
from wt_radar import output_checks


CHIRPS, SAMPLES = 32, 64


@pytest.fixture(scope="module")
def axes():
    radar = Radar.from_dict({
        "num_tx": 1, "num_rx": 1, "tx_loc": [[0, 0, 0]], "rx_loc": [[0, 0, 0]],
        "adc_samples": SAMPLES, "chirp_per_frame": CHIRPS,
        "fc": 77e9, "slope": 60.012, "adc_start_time": 6, "sample_rate": 4400,
        "idle_time": 7, "ramp_end_time": 65, "power": 15,
    }, device="cpu")
    system = radar.system_config
    sample = torch.zeros((CHIRPS, 1, SAMPLES), dtype=torch.complex64)
    return ProcessingAxes.from_synthesis(
        SynthesisResult.from_fmcw(sample, system.waveform_spec()), system.waveform_spec(), system.sensors.array)


def frames(count, *, moving_bin=10, moving_amp=1., static_bin=40, static_amp=.1, doppler_cycles=5):
    """Spectrum-domain frames: one moving return (Doppler ramp) and one static return."""
    chirp = np.arange(CHIRPS)
    cube = np.zeros((count, 1, 1, CHIRPS, SAMPLES), dtype=np.complex64)
    cube[:, 0, 0, :, moving_bin] = moving_amp * np.exp(2j * np.pi * doppler_cycles * chirp / CHIRPS)
    cube[:, 0, 0, :, static_bin] += static_amp
    return torch.from_numpy(cube)


def labels_at(range_m, count, *, label="in_view", speed=1.):
    return {"targets": [{"object_id": "person", "frames": [
        {"frame": index, "label": label, "range_m": float(range_m), "speed_mps": speed}
        for index in range(count)]}]}


def test_clean_run_passes_with_target_present_and_no_warning(axes):
    ranges = axes.range_m.numpy()
    cube = frames(3)
    result = output_checks.radar_output_checks(
        cube, axes, np.arange(3) / 10., labels=labels_at(ranges[10], 3), expected_frame_count=3)
    assert result["passed"], result["failed_checks"]
    assert result["checks"] == {"all_finite": True, "nonzero_return": True, "frame_count_matches": True,
                                "targets_present_when_in_view": True}
    assert result["targets"]["person"]["present_fraction"] == 1.
    assert result["warnings"] == []


def test_strong_static_return_is_a_soft_warning_with_its_range(axes):
    ranges = axes.range_m.numpy()
    cube = frames(2, static_amp=1000.)
    result = output_checks.radar_output_checks(cube, axes, np.arange(2) / 10., labels=labels_at(ranges[10], 2))
    assert result["passed"]
    warning = result["warnings"][0]
    assert warning["code"] == "RADAR_STATIC_DOMINANT"
    assert warning["static_peak_range_m"] == pytest.approx(ranges[40])
    assert warning["frames"] == 2


def test_target_missing_at_its_true_range_fails(axes):
    ranges = axes.range_m.numpy()
    result = output_checks.radar_output_checks(frames(2), axes, np.arange(2) / 10., labels=labels_at(ranges[25], 2))
    assert not result["passed"]
    assert result["failed_checks"] == ["targets_present_when_in_view"]
    assert len(result["targets"]["person"]["missing_frames"]) == 2


def test_frames_not_in_view_or_standing_still_are_not_checked(axes):
    ranges = axes.range_m.numpy()
    occluded = output_checks.radar_output_checks(
        frames(2), axes, np.arange(2) / 10., labels=labels_at(ranges[25], 2, label="occluded"))
    still = output_checks.radar_output_checks(
        frames(2), axes, np.arange(2) / 10., labels=labels_at(ranges[25], 2, speed=0.))
    assert occluded["passed"] and occluded["targets"]["person"]["not_in_view"] == 2
    assert still["passed"] and still["targets"]["person"]["skipped_static"] == 2


def test_nan_and_frame_count_mismatch_fail(axes):
    cube = frames(2)
    cube[1, 0, 0, 3, 5] = float("nan")
    result = output_checks.radar_output_checks(cube, axes, np.arange(2) / 10., expected_frame_count=3)
    assert set(result["failed_checks"]) == {"all_finite", "frame_count_matches", "nonzero_return"}
