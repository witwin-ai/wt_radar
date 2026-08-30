"""Saved-result transport for the current Radar processing API.

No solver, FFT, filtering, axis reconstruction, or antenna/chirp reduction
lives here. The producer serializes ProcessingAxes from its synthesis result;
the viewer restores that record and calls the official range_profile entry.
Older files without that record are intentionally refused, not guessed.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from witwin.radar.processing import ProcessingAxes, ProcessingCube, range_profile
from witwin.radar.synthesis.assembly import BEAT_PHASOR


from .memory_budget import MAX_RESULT_BYTES, require_result_memory
MAX_EXPANDED_BYTES = MAX_RESULT_BYTES + 32 * 1024**2


def _validate_record(record):
    """Reject contradictory metadata; never repair it or infer missing axes."""
    if not isinstance(record, dict):
        raise ValueError("ProcessingAxes metadata must be an object.")
    if record.get("waveform") != "fmcw" or record.get("output_domain") != "spectrum":
        raise ValueError("This first viewer supports producer-declared FMCW range spectra only.")
    if (record.get("fast_time_name"), record.get("slow_time_name")) != ("range_bin", "chirp"):
        raise ValueError("FMCW spectrum axes must be range_bin and chirp.")
    if record.get("phasor") != BEAT_PHASOR or record.get("doppler_sign") != 1:
        raise ValueError("Inconsistent FMCW phasor/Doppler convention.")
    for key in ("num_tx", "num_rx", "range_bin_count", "doppler_bin_count"):
        if type(record.get(key)) is not int or record[key] < 1:
            raise ValueError(f"Invalid positive integer metadata: {key}")
    for key in (
        "range_bin_m", "velocity_bin_mps", "slow_time_period_s", "wavelength_m",
        "reference_frequency_hz", "element_spacing_m", "max_unambiguous_range_m", "max_unambiguous_speed_mps",
    ):
        value = record.get(key)
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid positive finite metadata: {key}")
    for key, count in (("tx_loc_half_wavelength", "num_tx"), ("rx_loc_half_wavelength", "num_rx")):
        values = np.asarray(record.get(key), dtype=np.float64)
        if values.shape != (record[count], 3) or not np.isfinite(values).all():
            raise ValueError(f"Array coordinates disagree with metadata: {key}")


@dataclass
class SavedRadarResult:
    path: Path
    cube: torch.Tensor
    times_s: np.ndarray
    axes: ProcessingAxes
    producer: dict

    @classmethod
    def load(cls, path: str | Path) -> "SavedRadarResult":
        path = Path(path).expanduser()
        if not path.is_absolute() or path.suffix.lower() != ".npz":
            raise ValueError("Choose an absolute path to a radar result .npz file.")
        with ZipFile(path) as archive:
            if sum(member.file_size for member in archive.infolist()) > MAX_EXPANDED_BYTES:
                raise ValueError("Result exceeds the 1568 MiB viewer budget; export smaller batches.")
            try:
                entry = archive.getinfo("cube.npy")
            except KeyError as exc:
                raise ValueError("Missing cube in saved result.") from exc
            with archive.open(entry) as cube_header:
                header_version = np.lib.format.read_magic(cube_header)
                if header_version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(cube_header)
                elif header_version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(cube_header)
                else:
                    raise ValueError("Unsupported cube NPY header version.")
                size = math.prod(shape) * dtype.itemsize
                if len(shape) != 5 or any(dim < 1 for dim in shape) or dtype not in (np.dtype('complex64'), np.dtype('complex128')):
                    raise ValueError("Expected a nonempty complex radar cube.")
                if cube_header.tell() + size != entry.file_size:
                    raise ValueError("Cube header shape disagrees with its stored size.")
                require_result_memory(size)
        with np.load(path, allow_pickle=False) as source:
            required = {"cube", "times_s", "result_schema_version", "processing_axes_json", "producer_metadata_json"}
            if not required.issubset(source.files):
                raise ValueError(
                    "Missing producer ProcessingAxes metadata. Re-export the simulation result; "
                    "the viewer will not infer spectrum/beat domain or physical axes from shape."
                )
            if source["result_schema_version"].shape != () or source["result_schema_version"].item() != 1:
                raise ValueError("Unsupported saved Radar result schema.")
            record = json.loads(source["processing_axes_json"].item())
            producer = json.loads(source["producer_metadata_json"].item())
            cube_array = source["cube"]
            times = source["times_s"].copy()
            if cube_array.dtype not in (np.dtype("complex64"), np.dtype("complex128")):
                raise ValueError("Expected a complex IQ/spectrum cube, not magnitudes.")
            if cube_array.ndim != 5 or cube_array.shape[0] < 1:
                raise ValueError("Expected nonempty [frame, TX, RX, slow, fast] cube.")
            if times.shape != (cube_array.shape[0],) or not np.isfinite(times).all():
                raise ValueError("Frame timestamps are missing, nonfinite, or inconsistent with cube.")
            if not np.all(np.diff(times) > 0):
                raise ValueError("Frame timestamps must be strictly increasing.")
            if not np.isfinite(cube_array).all():
                raise ValueError("Radar cube contains NaN or infinity.")
            _validate_record(record)
            for key in ("range_m", "velocity_mps"):
                values = np.asarray(record[key], dtype=np.float64)
                if values.ndim != 1 or not np.isfinite(values).all() or not np.all(np.diff(values) > 0):
                    raise ValueError(f"Invalid physical axis: {key}")
                if key in source.files and not np.array_equal(values, source[key]):
                    raise ValueError(f"Saved {key} disagrees with ProcessingAxes metadata.")
                record[key] = torch.from_numpy(values.copy())
            if record.get("range_origin_m") != float(record["range_m"][0]):
                raise ValueError("Range origin disagrees with saved coordinates.")
            for key, spacing in (("range_m", "range_bin_m"), ("velocity_mps", "velocity_bin_mps")):
                if not np.allclose(np.diff(record[key].numpy()), record[spacing], rtol=1e-9, atol=1e-12):
                    raise ValueError(f"Saved {key} coordinates disagree with declared bin spacing.")
            for key in ("tx_loc_half_wavelength", "rx_loc_half_wavelength"):
                record[key] = tuple(tuple(row) for row in record[key])
            axes = ProcessingAxes(**record)
            expected = (axes.num_tx, axes.num_rx, axes.doppler_bin_count, axes.range_bin_count)
            if tuple(cube_array.shape[1:]) != expected:
                raise ValueError(f"Cube shape {cube_array.shape[1:]} differs from producer axes {expected}.")
            if not isinstance(producer, dict):
                raise ValueError("Producer metadata must be an object.")
            return cls(path.resolve(), torch.from_numpy(cube_array), times, axes, producer)

    def profile(self, frame: int):
        if not 0 <= frame < len(self.times_s):
            raise ValueError(f"Frame index must be in [0, {len(self.times_s) - 1}].")
        # Explicit official defaults: no second FFT on a spectrum, no filter.
        return range_profile(ProcessingCube(self.cube[frame], self.axes), window="rectangular", remove_dc=False)

    def selected_profile(self, frame: int, tx: int, rx: int, chirp: int):
        for name, value, size in (
            ("TX", tx, self.axes.num_tx), ("RX", rx, self.axes.num_rx),
            ("chirp", chirp, self.axes.doppler_bin_count),
        ):
            if not 0 <= value < size:
                raise ValueError(f"{name} index must be in [0, {size - 1}].")
        profile = self.profile(frame)
        return profile.axes.range_m, profile.data[tx, rx, chirp]


def show_saved_result(component):
    """Use the plugin's existing Figure widget, whose line API accepts metres."""
    result = component._saved_result
    frame = int(component.saved_frame_index)
    tx, rx, chirp = int(component.tx_index), int(component.rx_index), int(component.saved_chirp_index)
    if bool(component.static_clutter_removal) or bool(component.show_cfar):
        raise ValueError("Saved-result replay does not apply static-clutter removal or CFAR. Turn these controls off.")
    view = str(component.view)
    if view == "range_doppler":
        from .snapshot import SnapshotResult, snapshot_view
        from .replay import numeric_plot
        if not 0 <= frame < len(result.times_s):
            raise ValueError("Frame index is outside the recorded result.")
        payload = snapshot_view(SnapshotResult(ProcessingCube(result.cube[frame], result.axes), result.producer),
                                {"view": view, "tx": tx, "rx": rx})
        payload["title"] = f"Range Doppler | frame {frame}/{len(result.times_s)-1} | t={result.times_s[frame]:.4f}s"
        component.signal_figure.set_plot_data(numeric_plot(payload))
        return payload["title"]
    if view not in ("range_profile", "range_spectrum"):
        raise ValueError("Saved-result replay supports Range Profile/Spectrum/Doppler only; no legacy DSP is used.")
    ranges, data = result.selected_profile(frame, tx, rx, chirp)
    fig = component.signal_figure.clear()
    if view == "range_profile":
        fig.line(ranges.tolist(), data.abs().tolist(), label="Magnitude")
        fig.ylabel("Amplitude (linear; not calibrated power)")
    else:
        fig.line(ranges.tolist(), data.real.tolist(), label="Real")
        fig.line(ranges.tolist(), data.imag.tolist(), label="Imaginary")
        fig.ylabel("Complex amplitude")
    fig.xlabel("Range (m)")
    fig.title(
        f"{view.replace('_', ' ').title()} | frame {frame}/{len(result.times_s) - 1} "
        f"t={result.times_s[frame]:.4f}s | TX{tx} RX{rx} chirp{chirp}"
    )
    return f"Displayed frame {frame} at {result.times_s[frame]:.4f}s; official range_profile, no filtering"
