"""
Range FFT post-processing component.
"""
from typing import Dict, Any
import torch

from witwin.components import component, string_field, int_field, bool_field, figure
from witwin.utils.logging import get_logger

from .base import RadarPostProcessor

logger = get_logger("RangeFFT")


@component(name="RangeFFT", display_name="Range FFT", category="Radar/PostProcessing")
class RangeFFTProcessor(RadarPostProcessor):
    """
    Range FFT post-processing component.

    Computes the Range FFT from raw radar signals, supporting both
    MIMO and non-MIMO modes with batch visualization.
    """

    # Output visualization
    fft_plot = figure(plot_type="line", description="Range FFT magnitude")

    # Processing options
    window_type = string_field(
        "none",
        options=["none", "hann", "hamming", "blackman"],
        description="Window function for FFT"
    )

    zero_padding = int_field(
        0, min=0, max=4,
        description="Zero padding factor (2^n multiplier)"
    )

    commit_results = bool_field(
        False,
        description="Commit results to Results panel"
    )

    def _apply_window(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply window function to signal."""
        n_samples = signal.shape[-1]

        if self.window_type == "hann":
            window = torch.hann_window(n_samples, dtype=signal.real.dtype)
        elif self.window_type == "hamming":
            window = torch.hamming_window(n_samples, dtype=signal.real.dtype)
        elif self.window_type == "blackman":
            window = torch.blackman_window(n_samples, dtype=signal.real.dtype)
        else:
            return signal

        return signal * window

    def _compute_range_params(self, params: Dict[str, Any], n_fft: int) -> tuple:
        """Compute range resolution and axis."""
        adc_samples = params.get('adc_samples', n_fft)
        sample_rate = params.get('sample_rate', 4400)
        slope = params.get('slope', 60.012)

        range_resolution = (3e8 * sample_rate * 1e3) / (2 * slope * 1e12 * adc_samples)
        range_axis = [i * range_resolution for i in range(n_fft)]

        return range_resolution, range_axis

    def process(self, signal: torch.Tensor, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Range FFT from raw radar signal."""
        is_mimo = signal.dim() == 3

        # Apply window
        windowed = self._apply_window(signal)

        # Apply zero padding
        if self.zero_padding > 0:
            n_samples = windowed.shape[-1]
            pad_size = n_samples * (2 ** self.zero_padding) - n_samples
            windowed = torch.nn.functional.pad(windowed, (0, pad_size))

        # Compute FFT along last dimension
        fft_result = torch.fft.fft(windowed, dim=-1)
        fft_magnitude = torch.abs(fft_result)

        # Compute range axis
        n_fft = fft_magnitude.shape[-1]
        range_resolution, range_axis = self._compute_range_params(params, n_fft)

        return {
            'fft_magnitude': fft_magnitude,
            'range_axis': range_axis,
            'range_resolution': range_resolution,
            'is_mimo': is_mimo,
        }

    def _on_process_complete(self, result: Dict[str, Any]):
        """Update FFT visualization."""
        fft_magnitude = result['fft_magnitude']
        range_axis = result['range_axis']
        is_mimo = result['is_mimo']

        if is_mimo:
            self._update_mimo_plot(fft_magnitude, range_axis)
        else:
            self._update_single_plot(fft_magnitude, range_axis)

        if self.commit_results:
            self._commit_to_results()

    def _update_single_plot(self, fft_magnitude: torch.Tensor, range_axis: list):
        """Update FFT plot for non-MIMO mode."""
        logger.info(f"Updating plot: shape={fft_magnitude.shape}")
        self.fft_plot.clear()
        self.fft_plot.line(range_axis, fft_magnitude.tolist(), color="#51cf66")
        title = "Range FFT" + (f" ({self.window_type})" if self.window_type != "none" else "")
        self.fft_plot.title(title)
        self.fft_plot.xlabel("Range (m)")
        self.fft_plot.ylabel("Magnitude")

    def _update_mimo_plot(self, fft_magnitude: torch.Tensor, range_axis: list):
        """Update FFT plot for MIMO mode with batch2d visualization."""
        num_tx, num_rx, _ = fft_magnitude.shape

        fft_data_2d = []
        for tx_idx in range(num_tx):
            tx_batch = []
            for rx_idx in range(num_rx):
                fft_mag = fft_magnitude[tx_idx, rx_idx]
                tx_batch.append({
                    'series': [
                        {'x': range_axis, 'y': fft_mag.tolist(), 'label': 'Magnitude', 'color': '#51cf66'},
                    ]
                })
            fft_data_2d.append(tx_batch)

        tx_labels = [f"Tx{i}" for i in range(num_tx)]
        rx_labels = [f"Rx{i}" for i in range(num_rx)]

        first_fft = fft_data_2d[0][0]['series']
        self.fft_plot.clear()
        for s in first_fft:
            self.fft_plot.line(s['x'], s['y'], label=s['label'], color=s['color'])

        title = "Range FFT (MIMO)" + (f" ({self.window_type})" if self.window_type != "none" else "")
        self.fft_plot.title(title)
        self.fft_plot.xlabel("Range (m)")
        self.fft_plot.ylabel("Magnitude")
        self.fft_plot.batch2d(fft_data_2d, tx_labels, rx_labels)

        logger.info(f"Updated MIMO plot: {num_tx}Tx x {num_rx}Rx")

    def _commit_to_results(self):
        """Send FFT result to the Results panel."""
        from witwin import Results
        Results.add_plot(self.fft_plot)
        Results.commit("Range FFT")
