"""
Range-Doppler post-processing component.
"""
from typing import Dict, Any
import numpy as np
import torch

from witwin.components import component, bool_field, figure
from witwin.utils.logging import get_logger

from .base import RadarPostProcessor

logger = get_logger("RangeDoppler")


@component(name="RangeDoppler", display_name="Range-Doppler", category="Radar/PostProcessing")
class RangeDopplerProcessor(RadarPostProcessor):
    """
    Range-Doppler map processing component.

    Computes the 2D Range-Doppler map from radar chirp data.
    Expects input with shape [num_chirps, adc_samples] or MIMO [tx, rx, chirps, samples].
    """

    # Output visualization (heatmap with log/linear toggle via batch)
    rd_map = figure(plot_type="heatmap", description="Range-Doppler map")

    # Processing options
    remove_dc = bool_field(True, description="Remove DC component (mean subtraction)")

    commit_results = bool_field(
        False,
        description="Commit results to Results panel"
    )

    def process(self, signal: torch.Tensor, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute Range-Doppler map from radar signal.

        Args:
            signal: Radar signal tensor
                    - Frame mode: [num_chirps, adc_samples]
                    - MIMO: [num_tx, num_rx, num_chirps, adc_samples]
            params: Radar parameters
        """
        # Convert to numpy for processing
        data = signal.numpy() if isinstance(signal, torch.Tensor) else signal

        # Handle different input shapes
        if data.ndim == 1:
            # Single chirp - can't do Doppler processing
            logger.warning("Single chirp input, cannot compute Range-Doppler map")
            return {'error': 'Need multiple chirps for Range-Doppler'}

        if data.ndim == 2:
            # [num_chirps, adc_samples]
            rd_mag, rd_mag_db, rd_map_complex = self._process_frame(data, params)
            is_mimo = False
        elif data.ndim == 3:
            # MIMO [num_tx * num_rx, num_chirps, adc_samples] or similar
            # Process first virtual antenna for now
            rd_mag, rd_mag_db, rd_map_complex = self._process_frame(data[0], params)
            is_mimo = True
        elif data.ndim == 4:
            # MIMO [num_tx, num_rx, num_chirps, adc_samples]
            # Process first Tx-Rx pair
            rd_mag, rd_mag_db, rd_map_complex = self._process_frame(data[0, 0], params)
            is_mimo = True
        else:
            logger.error(f"Unexpected signal shape: {data.shape}")
            return {'error': f'Unexpected shape: {data.shape}'}

        # Compute axis labels
        num_chirps, num_samples = rd_mag.shape
        range_res = params.get('range_resolution', 1.0)
        doppler_res = params.get('doppler_resolution', 1.0)

        range_axis = [i * range_res for i in range(num_samples)]
        # Doppler axis centered at 0 (after fftshift)
        doppler_axis = [(i - num_chirps // 2) * doppler_res for i in range(num_chirps)]

        return {
            'rd_mag': rd_mag,
            'rd_mag_db': rd_mag_db,
            'rd_map_complex': rd_map_complex,
            'range_axis': range_axis,
            'doppler_axis': doppler_axis,
            'is_mimo': is_mimo,
        }

    def _process_frame(self, data: np.ndarray, params: Dict[str, Any]) -> tuple:
        """
        Process a single frame [num_chirps, adc_samples].

        Returns:
            (magnitude, magnitude_dB, complex_map)
        """
        num_chirps, num_samples = data.shape

        # Copy data for processing
        data_windowed = np.copy(data)

        # Remove DC (mean subtraction along both axes)
        if self.remove_dc:
            data_windowed = data_windowed - np.mean(data_windowed, axis=-1, keepdims=True)
            data_windowed = data_windowed - np.mean(data_windowed, axis=-2, keepdims=True)

        # Apply 2D Hamming window
        range_window = np.hamming(num_samples)
        doppler_window = np.hamming(num_chirps)
        data_windowed = data_windowed * range_window
        data_windowed = data_windowed * doppler_window[:, None]

        # Range FFT (fast-time, along samples axis)
        range_fft = np.fft.fft(data_windowed, axis=-1)

        # Doppler FFT (slow-time, along chirps axis) + fftshift
        rd_map = np.fft.fft(range_fft, axis=-2)
        rd_map = np.fft.fftshift(rd_map, axes=-2)

        # Compute magnitude
        rd_mag = np.abs(rd_map)
        rd_mag_db = 20 * np.log10(rd_mag + 1e-6)

        return rd_mag, rd_mag_db, rd_map

    def _on_process_complete(self, result: Dict[str, Any]):
        """Update Range-Doppler map visualization."""
        if 'error' in result:
            logger.warning(result['error'])
            return

        rd_mag = result['rd_mag']
        rd_mag_db = result['rd_mag_db']
        range_axis = result['range_axis']
        doppler_axis = result['doppler_axis']

        # Create batch data for linear/log toggle
        # Only use first half of range (positive frequencies)
        half_range = len(range_axis) // 2
        rd_mag_half = rd_mag[:, :half_range]
        rd_mag_db_half = rd_mag_db[:, :half_range]
        range_axis_half = range_axis[:half_range]

        batch_data = [
            {
                'heatmap': {
                    'data': rd_mag_db_half.tolist(),
                    'x_axis': range_axis_half,
                    'y_axis': doppler_axis,
                }
            },
            {
                'heatmap': {
                    'data': rd_mag_half.tolist(),
                    'x_axis': range_axis_half,
                    'y_axis': doppler_axis,
                }
            },
        ]
        batch_labels = ["Log (dB)", "Linear"]

        # Set up the figure
        self.rd_map.clear()
        self.rd_map.heatmap(
            rd_mag_db_half.tolist(),
            x_axis=range_axis_half,
            y_axis=doppler_axis,
            colormap="viridis"
        )
        self.rd_map.title("Range-Doppler Map")
        self.rd_map.xlabel("Range (m)")
        self.rd_map.ylabel("Doppler (m/s)")
        self.rd_map.batch(batch_data, batch_labels)

        logger.info(f"Updated Range-Doppler map: {rd_mag.shape}")

        if self.commit_results:
            self._commit_to_results()

    def _commit_to_results(self):
        """Send Range-Doppler map to the Results panel."""
        from witwin import Results
        Results.add_plot(self.rd_map)
        Results.commit("Range-Doppler")
