"""
Radar Simulation Components.

This module provides radar system components for simulation and visualization.
"""
from typing import Dict, Any
import torch
import numpy as np

from witwin.components import (
    Component, component, float_field, int_field, bool_field, figure, button,
    GizmoContext, foldout_group, define_group, list_field, component_field
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

    def __init__(self,
                 fc: float = 77e9,
                 slope: float = 60.012,
                 adc_samples: int = 256,
                 sample_rate: float = 4400,
                 adc_start_time: float = 6,
                 idle_time: float = 7,
                 ramp_end_time: float = 65,
                 loop_per_frame: int = 128,
                 chirp_per_frame: int = 128,
                 frame_per_second: int = 10):
        self.c0 = 299792458

        self.num_tx = 3
        self.num_rx = 4
        self.fc = fc                     # start frequency (Hz)
        self.slope = slope               # frequency slope (MHz/us)
        self.adc_samples = adc_samples
        self.adc_start_time = adc_start_time
        self.sample_rate = sample_rate   # (ksps)
        self.idle_time = idle_time       # (us), duration between chirps
        self.ramp_end_time = ramp_end_time  # (us)
        self.loop_per_frame = loop_per_frame
        self.chirp_per_frame = chirp_per_frame
        self.frame_per_second = frame_per_second

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

        # Pre-compute time samples and TX waveform
        self.t_sample = torch.arange(0, self.adc_samples, dtype=torch.float64) / (self.sample_rate*1e3) + self.adc_start_time*1e-6
        self.tx_waveform = self.waveform(self.t_sample)

    def dechirp(self, x, xref):
        return xref * torch.conj(x)

    def FSPL(self, distance):
        return 20*torch.log10(distance) + 20*torch.log10(torch.tensor(self.fc)) + 20*torch.log10(torch.tensor(4*np.pi/self._lambda))

    def waveform(self, t, phi=0):
        fc = (self.fc * t + 0.5 * (self.slope * 1e12) * t * t)
        y = torch.exp(2j * torch.pi * fc + phi)
        return y

    def chirp(self, distance, phi=0):
        """Compute chirp signal for given distances."""
        toa = distance / self.c0
        rx = self.waveform(self.t_sample - toa, phi)
        rx_combined = torch.sum(rx, axis=-2)
        sig = self.tx_waveform * torch.conj(rx_combined)
        return sig

    def chirp_simple(self, distance):
        """Simple chirp for non-MIMO mode."""
        toa = 2 * distance / self.c0
        tx = self.waveform(self.t_sample)
        rx = self.waveform(self.t_sample - toa.view(-1, 1))
        rx = tx * torch.conj(rx)
        rx_combined = torch.sum(rx, axis=0)
        return rx_combined

    def frameMIMO(self, distances):
        """Compute MIMO frame for all Tx-Rx pairs.

        Args:
            distances: Tensor of distances to targets [num_targets]

        Returns:
            frame: [num_tx, num_rx, adc_samples] complex tensor
        """
        # Compute distances for each Tx-Rx pair
        # distances: [num_targets] -> expand to [num_tx, num_rx, num_targets]
        tx_pos = torch.tensor(self.tx_loc, dtype=torch.float64)  # [num_tx, 3]
        rx_pos = torch.tensor(self.rx_loc, dtype=torch.float64)  # [num_rx, 3]

        # For simplicity, assume targets are at distance along z-axis
        # Total path = dist_to_target + dist_from_target (approx 2*distance for far targets)
        # With antenna offsets, we add phase shifts

        frame = torch.zeros((self.num_tx, self.num_rx, self.adc_samples), dtype=torch.complex128)

        for tx_idx in range(self.num_tx):
            for rx_idx in range(self.num_rx):
                # Phase offset from antenna separation
                antenna_sep = torch.norm(tx_pos[tx_idx] - rx_pos[rx_idx])
                phase_offset = 2 * np.pi * antenna_sep / self._lambda

                # Compute signal for this pair
                toa = 2 * distances / self.c0
                rx = self.waveform(self.t_sample - toa.view(-1, 1), phase_offset)
                rx_combined = torch.sum(rx, axis=0)
                sig = self.tx_waveform * torch.conj(rx_combined)
                frame[tx_idx, rx_idx] = sig

        return frame

    def fft(self, distance):
        sig = self.chirp_simple(distance)
        return torch.fft.fft(sig)


# Frequency units for unit-number fields
FREQUENCY_UNITS = [
    {"label": "Hz", "multiplier": 1},
    {"label": "kHz", "multiplier": 1e3},
    {"label": "MHz", "multiplier": 1e6},
    {"label": "GHz", "multiplier": 1e9},
]


@component(name="Radar")
class RadarComponent(Component):
    """Radar component for radar objects."""

    # Define foldout groups
    define_group(foldout_group("Frequency", display_name="Frequency / Waveform"))
    define_group(foldout_group("ADC", display_name="ADC / Sampling"))
    define_group(foldout_group("Chirp", display_name="Chirp / Frame"))

    # Radar view parameters
    fov = float_field(45.0, min=1.0, max=180.0, description="Field of view (degrees)")

    # Output options
    mimo = bool_field(False, description="Enable MIMO mode (show each Tx-Rx pair)")
    commit_results = bool_field(False, description="Commit results to Results panel")
    debug = bool_field(False, description="Show Distance/Intensity maps (requires commit_results)")

    # ===== Frequency / Waveform Parameters =====
    fc = float_field(
        77e9, min=1e9, max=300e9,
        units=FREQUENCY_UNITS, default_unit="GHz",
        group="Frequency",
        description="Carrier frequency"
    )
    slope = float_field(
        60.012, min=1.0, max=200.0,
        group="Frequency",
        description="Frequency slope (MHz/us)"
    )

    # ===== ADC / Sampling Parameters =====
    adc_samples = int_field(
        256, min=64, max=1024,
        group="ADC",
        description="ADC samples per chirp"
    )
    sample_rate = float_field(
        4400, min=1000, max=20000,
        group="ADC",
        description="Sample rate (ksps)"
    )
    adc_start_time = float_field(
        6.0, min=0.0, max=20.0,
        group="ADC",
        description="ADC start time (us)"
    )

    # ===== Chirp / Frame Parameters =====
    idle_time = float_field(
        7.0, min=1.0, max=50.0,
        group="Chirp",
        description="Idle time between chirps (us)"
    )
    ramp_end_time = float_field(
        65.0, min=10.0, max=200.0,
        group="Chirp",
        description="Ramp end time (us)"
    )
    loop_per_frame = int_field(
        128, min=16, max=512,
        group="Chirp",
        description="Loops per frame"
    )
    chirp_per_frame = int_field(
        128, min=16, max=512,
        group="Chirp",
        description="Chirps per frame"
    )
    frame_per_second = int_field(
        10, min=1, max=60,
        group="Chirp",
        description="Frames per second"
    )

    # Post-processing pipeline (accepts RadarPostProcessor and all subclasses)
    post_processors = list_field(
        component_field(component_type="RadarPostProcessor"),
        description="Post-processing components to run after rendering"
    )

    # Raw signal visualization
    signal_plot = figure(plot_type="line", description="Raw signal (real & imag)")

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

    def get_max_range(self) -> float:
        """Calculate max detection range from radar parameters."""
        return (300 * float(self.sample_rate)) / (2 * float(self.slope) * 1e3)

    def on_draw_gizmos(self, ctx: GizmoContext):
        """Draw radar visualization gizmos."""
        ctx.color = "#ffff00"

        # Yellow sphere at the origin (radar position)
        ctx.draw_sphere(radius=0.05)

        # Cone representing the radar field of view (4 segments = pyramid shape, -Z direction)
        # Range is calculated from radar parameters
        max_range = self.get_max_range()
        ctx.draw_cone(fov="fov", range=max_range, segments=4, direction=(0, 0, -1))

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
        logger.info("Rendering radar view...")

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
        radar = SimpleRadar(
            fc=float(self.fc),
            slope=float(self.slope),
            adc_samples=int(self.adc_samples),
            sample_rate=float(self.sample_rate),
            adc_start_time=float(self.adc_start_time),
            idle_time=float(self.idle_time),
            ramp_end_time=float(self.ramp_end_time),
            loop_per_frame=int(self.loop_per_frame),
            chirp_per_frame=int(self.chirp_per_frame),
            frame_per_second=int(self.frame_per_second),
        )

        # Flatten distance and filter
        tau_tensor = torch.tensor(distance, dtype=torch.float64).flatten()
        tau_tensor[tau_tensor == 0] = 1e-6
        mask = tau_tensor > 0.1
        tau_filtered = tau_tensor[mask]

        no_targets = len(tau_filtered) == 0
        if no_targets:
            logger.warning("No valid distance samples found")

        # ===== Generate Raw Signal =====
        if self.mimo:
            if no_targets:
                raw_signal = torch.zeros((radar.num_tx, radar.num_rx, radar.adc_samples), dtype=torch.complex128)
            else:
                raw_signal = radar.frameMIMO(tau_filtered)  # [num_tx, num_rx, adc_samples]
        else:
            if no_targets:
                raw_signal = torch.zeros(radar.adc_samples, dtype=torch.complex128)
            else:
                raw_signal = radar.chirp_simple(tau_filtered)

        # ===== Update Signal Plot =====
        x_samples = list(range(radar.adc_samples))

        if self.mimo:
            # MIMO mode: batch2d visualization
            num_tx, num_rx = radar.num_tx, radar.num_rx
            signal_data_2d = []

            for tx_idx in range(num_tx):
                tx_batch = []
                for rx_idx in range(num_rx):
                    sig = raw_signal[tx_idx, rx_idx]
                    tx_batch.append({
                        'series': [
                            {'x': x_samples, 'y': sig.real.tolist(), 'label': 'Real', 'color': '#ff9500'},
                            {'x': x_samples, 'y': sig.imag.tolist(), 'label': 'Imag', 'color': '#00aaff'},
                        ]
                    })
                signal_data_2d.append(tx_batch)

            tx_labels = [f"Tx{i}" for i in range(num_tx)]
            rx_labels = [f"Rx{i}" for i in range(num_rx)]

            first_sig = signal_data_2d[0][0]['series']
            self.signal_plot.clear()
            for s in first_sig:
                self.signal_plot.line(s['x'], s['y'], label=s['label'], color=s['color'])
            self.signal_plot.title("Raw Signal (MIMO)")
            self.signal_plot.xlabel("Sample")
            self.signal_plot.ylabel("Amplitude")
            self.signal_plot.batch2d(signal_data_2d, tx_labels, rx_labels)
        else:
            # Non-MIMO mode
            self.signal_plot.clear()
            self.signal_plot.line(x_samples, raw_signal.real.tolist(), label='Real', color='#ff9500')
            self.signal_plot.line(x_samples, raw_signal.imag.tolist(), label='Imag', color='#00aaff')
            self.signal_plot.title("Raw Signal")
            self.signal_plot.xlabel("Sample")
            self.signal_plot.ylabel("Amplitude")

        # ===== Build Signal Metadata =====
        signal_metadata = {
            'adc_samples': int(self.adc_samples),
            'sample_rate': float(self.sample_rate),
            'slope': float(self.slope),
            'fc': float(self.fc),
            'mimo': bool(self.mimo),
            'range_resolution': radar.range_resolution,
            'max_range': radar.max_range,
            'doppler_resolution': radar.doppler_resolution,
            'max_doppler': radar.max_doppler,
            'distances': tau_filtered,
            'distance_map': distance,
            'intensity_map': intensity,
        }

        # ===== Trigger Post-Processors =====
        num_processors = len(self.post_processors) if self.post_processors else 0
        logger.info(f"post_processors list: {self.post_processors}")
        if num_processors > 0:
            logger.info(f"Triggering {num_processors} post-processor(s)...")
            for i, processor_ref in enumerate(self.post_processors):
                logger.info(f"  [{i}] processor_ref = {processor_ref}, type = {type(processor_ref)}")

                # Resolve component reference to actual component instance
                processor = None
                if isinstance(processor_ref, dict) and processor_ref.get('object_id'):
                    obj_id = processor_ref['object_id']
                    comp_type = processor_ref.get('component_type', 'RadarPostProcessor')
                    target_obj = self.scene.get_object(obj_id)
                    if target_obj:
                        # Try to find the component by type or any RadarPostProcessor subclass
                        processor = target_obj.get_component(comp_type)
                        if not processor:
                            # Try all components to find a RadarPostProcessor
                            for comp in target_obj.get_all_components().values():
                                if hasattr(comp, 'set_raw_signal'):
                                    processor = comp
                                    break
                        logger.info(f"  [{i}] Resolved to: {processor}")
                    else:
                        logger.warning(f"  [{i}] Object not found: {obj_id}")
                elif hasattr(processor_ref, 'set_raw_signal'):
                    # Already a component instance
                    processor = processor_ref

                if processor and hasattr(processor, 'set_raw_signal'):
                    logger.info(f"  [{i}] Calling set_raw_signal...")
                    try:
                        processor.set_raw_signal(raw_signal, signal_metadata)
                        logger.info(f"  [{i}] set_raw_signal completed")
                    except Exception as e:
                        import traceback
                        logger.error(f"Post-processor failed: {e}")
                        logger.error(traceback.format_exc())
                else:
                    logger.warning(f"  [{i}] Processor invalid or missing set_raw_signal")

        # ===== Send Debug Results to Frontend (if enabled) =====
        if self.commit_results and self.debug:
            from witwin import Results
            Results.imshow(distance, title="Distance Map")
            Results.imshow(intensity, title="Intensity Map")
            Results.commit("Radar Debug")

        return f"Rendered radar view with MaxRange={radar.max_range:.1f}m, FOV={self.fov:.1f}°, {num_processors} post-processor(s)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert radar component to dictionary for serialization."""
        result = super().to_dict()
        return result

    def from_dict(self, data: Dict[str, Any]):
        """Load radar component from dictionary."""
        super().from_dict(data)
