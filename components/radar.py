"""
Radar Simulation Components.

This module provides radar system components for simulation and visualization.
"""
from typing import Dict, Any
import torch
import numpy as np

from witwin.components import (
    Component, component, float_field, bool_field, plot_field, button,
    GizmoContext, PlotBuilder
)
from witwin.utils.logging import get_logger
from witwin.utils.mitsuba_utils import scene_to_mitsuba

logger = get_logger("Radar")

# Mitsuba imports
import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb')


class SimpleRadar:
    """Simple radar signal processing class."""

    def __init__(self):
        self.c0 = 299792458

        self.num_tx = 3
        self.num_rx = 4
        self.fc = 77e9                   # start frequency (Hz)
        self.slope = 60.012              # frequency slope (MHz/us)
        self.adc_samples = 256
        self.adc_start_time = 6
        self.sample_rate = 4400          # (ksps)
        self.idle_time = 7               # (us), duration between chirps
        self.ramp_end_time = 65          # (us)
        self.loop_per_frame = 128
        self.frame_per_second = 10

        self.num_doppler_bins = self.loop_per_frame
        self.num_range_bins = self.adc_samples
        self.num_angle_bins = 64

        self.range_resolution = (3e8 * self.sample_rate * 1e3) / (2 * self.slope * 1e12 * self.adc_samples)
        self.max_range = (300 * self.sample_rate) / (2 * self.slope * 1e3)
        self.doppler_resolution = 3e8 / (2 * self.fc * 1e9 * (self.idle_time + self.ramp_end_time) * 1e-6 * self.num_doppler_bins * self.num_tx)
        self.max_doppler = 3e8 / (4 * self.fc * 1e9 * (self.idle_time + self.ramp_end_time) * 1e-6 * self.num_tx)

        spacing = self.c0 / self.fc / 2
        self.tx_loc = np.array([[0, 0, 0], [4*spacing, 0, 0], [2*spacing, spacing, 0]])
        self.rx_loc = np.array([[-6*spacing, 0, 0], [-5*spacing, 0, 0], [-4*spacing, 0, 0], [-3*spacing, 0, 0]])

        self._lambda = self.c0 / self.fc

    def dechirp(self, x, xref):
        return xref * torch.conj(x)

    def FSPL(self, distance):
        return 20*torch.log10(distance) + 20*torch.log10(torch.tensor(self.fc)) + 20*torch.log10(torch.tensor(4*np.pi/self._lambda))

    def waveform(self, t, phi=0):
        fc = (self.fc * t + 0.5 * (self.slope * 1e12) * t * t)
        y = torch.exp(2j * torch.pi * fc)
        return y

    def chirp(self, distance):
        t_sample = torch.arange(0, self.adc_samples, dtype=torch.float64) / (self.sample_rate*1e3) + self.adc_start_time*1e-6
        toa = 2 * distance / self.c0

        tx = self.waveform(t_sample)
        rx = self.waveform(t_sample - toa.view(-1, 1))
        rx = tx * torch.conj(rx)
        rx_combined = torch.sum(rx, axis=0)
        return rx_combined

    def fft(self, distance):
        sig = self.chirp(distance)
        return torch.fft.fft(sig)


@component(name="Radar")
class RadarComponent(Component):
    """Radar component for radar objects."""

    # Radar parameters
    range = float_field(1.0, min=0.1, max=100.0, description="Detection range")
    fov = float_field(45.0, min=1.0, max=180.0, description="Field of view (degrees)")

    # Output options
    commit_results = bool_field(False, description="Commit results to Results panel")
    debug = bool_field(False, description="Show Distance/Intensity maps (requires commit_results)")

    # Plot fields for visualization
    signal_real = plot_field(plot_type="line", description="Raw signal (real part)")
    signal_fft = plot_field(plot_type="line", description="FFT magnitude")

    # Resolution for ray tracing
    PIR_resolution = 128

    def __init__(self, **kwargs):
        """Initialize radar component."""
        super().__init__(**kwargs)

    def gen_rays(self, mi_scene):
        """Generate rays from the Mitsuba scene sensor."""
        sensor = mi_scene.sensors()[0]
        film = sensor.film()
        sampler = sensor.sampler()
        film_size = film.crop_size()
        spp = 1
        total_sample_count = dr.prod(film_size) * spp
        if sampler.wavefront_size() != total_sample_count:
            sampler.seed(0, total_sample_count)

        pos = dr.arange(mi.UInt32, total_sample_count)
        pos //= spp
        scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
        pos = mi.Vector2f(mi.Float(pos % int(film_size[0])),
                          mi.Float(pos // int(film_size[0])))
        rays, weights = sensor.sample_ray_differential(
            time=0,
            sample1=sampler.next_1d(),
            sample2=pos * scale,
            sample3=0
        )
        return rays

    def on_draw_gizmos(self, ctx: GizmoContext):
        """Draw radar visualization gizmos."""
        ctx.color = "#ffff00"

        # Yellow sphere at the origin (radar position)
        ctx.draw_sphere(radius=0.05)

        # Cone representing the radar field of view (4 segments = pyramid shape, -Z direction)
        ctx.draw_cone(fov="fov", range="range", segments=4, direction=(0, 0, -1))

    @button(display_name="Render", description="Render radar detection visualization")
    def render(self):
        """Render scene from radar perspective using Mitsuba."""
        # ===== Validation =====
        if not self.owner or not self.scene:
            logger.warning("Not attached to a scene object")
            return "Not attached to scene"

        transform = self.owner.get_component('Transform')
        if not transform:
            logger.warning("No Transform component found")
            return "No Transform component"

        # ===== Build Mitsuba Scene =====
        # Get radar transformation matrix
        radar_transform = transform.get_transformation_matrix().numpy()

        # Build camera data - use radar position and orientation
        camera_data = {
            'world_matrix': radar_transform.flatten('F').tolist(),  # column-major
            'fov': self.fov,
            'viewport_width': self.PIR_resolution,
            'viewport_height': self.PIR_resolution
        }

        # Filter function: exclude radar itself and transmitters/receivers
        def filter_fn(obj):
            if obj.id == self.owner.id:
                return False
            if "transmitter" in obj.id.lower() or "receiver" in obj.id.lower():
                return False
            return True

        logger.info(f"Sensor positioned at {transform.position.tolist()} with rotation {transform.rotation.tolist()}")

        # Build Mitsuba scene
        try:
            mi_scene = scene_to_mitsuba(
                self.scene,
                camera=camera_data,
                include_lights=True,
                filter_fn=filter_fn,
                spp=32,
                width=self.PIR_resolution,
                height=self.PIR_resolution
            )
        except Exception as e:
            logger.error(f"Failed to build Mitsuba scene: {e}")
            return f"Scene build failed: {e}"

        # ===== Generate Rays and Compute Intersections =====
        logger.warning("Rendering radar view...")

        # Generate rays from sensor
        rays = self.gen_rays(mi_scene)

        # Compute ray intersections for distance
        si = mi_scene.ray_intersect(rays)

        # Render for intensity
        intensity_img = mi.render(mi_scene, spp=32)

        # ===== Extract Distance and Intensity =====
        # Extract distance (t values)
        t = si.t
        t = dr.select(t > 9999, 0.0, t)  # Set non-intersecting rays to 0
        distance = np.array(t).reshape(self.PIR_resolution, self.PIR_resolution)

        # Extract intensity (first channel)
        intensity = np.array(intensity_img)[:, :, 0]

        # ===== Radar Signal Processing =====
        radar = SimpleRadar()

        # Flatten distance and filter
        tau_tensor = torch.tensor(distance, dtype=torch.float64).flatten()
        tau_tensor[tau_tensor == 0] = 1e-6
        mask = tau_tensor > 0.1
        tau_filtered = tau_tensor[mask]

        # Compute radar signal (no gradient)
        if len(tau_filtered) > 0:
            sig = radar.chirp(tau_filtered)
        else:
            logger.warning("No valid distance samples found")
            sig = torch.zeros(radar.adc_samples, dtype=torch.complex128)

        # Compute FFT
        fft = torch.fft.fft(sig)
        fft_magnitude = torch.abs(fft)

        # ===== Update plot fields =====
        self.signal_real = PlotBuilder.line(
            sig.real,
            title="Signal (Real)",
            xlabel="Sample",
            ylabel="Amplitude",
            color="orange"
        )

        # Range FFT: convert frequency bins to range (meters)
        num_bins = len(fft_magnitude)
        range_axis = [i * radar.range_resolution for i in range(num_bins)]

        self.signal_fft = PlotBuilder.line(
            range_axis,
            fft_magnitude,
            title="Range FFT",
            xlabel="Range (m)",
            ylabel="Magnitude",
            color="green"
        )

        # ===== Send Results to Frontend (if enabled) =====
        if self.commit_results:
            if hasattr(self.scene, '_server') and self.scene._server and hasattr(self.scene._server, 'results'):
                server = self.scene._server

                # Debug: show Distance/Intensity maps
                if self.debug:
                    server.results.imshow(distance, title="Distance Map")
                    server.results.imshow(intensity, title="Intensity Map")

                # Plot radar signals
                if len(sig) > 0:
                    x = list(range(len(sig)))
                    server.results.plot(x, sig.real, title="Signal Real", xlabel="Sample", ylabel="Amplitude", color="orange")
                    server.results.plot(range_axis, fft_magnitude, title="Range FFT", xlabel="Range (m)", ylabel="Magnitude", color="green")

                server.results.commit("Radar Results")
            else:
                logger.warning("Server or results not available")

        return f"Rendered radar view with Range={self.range:.1f}m, FOV={self.fov:.1f}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert radar component to dictionary for serialization."""
        result = super().to_dict()
        return result

    def from_dict(self, data: Dict[str, Any]):
        """Load radar component from dictionary."""
        super().from_dict(data)
