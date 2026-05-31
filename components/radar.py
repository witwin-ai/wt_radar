"""Unified Radar component — the single scene-level component for a radar simulation.

Replaces the nine separate Radar Settings components (RadarConfig + RadarSensor +
RadarTracer + the four sub-configs + RadarResult + RadarTimeline) with one component
exposing every setting through foldout groups. The marker the adapter keys off is now
``Radar`` (master plan §6.4); optional per-structure ``RadarMotion`` and the
polymorphic ``RadarPostProcessor`` hierarchy stay separate because they live on
other scene objects.

The grouping consolidates the old split: ``Configuration`` holds the FMCW core
(frequency / power + ADC + chirp / frame + FFT bins); ``Antenna`` holds the array
geometry plus the antenna-pattern sub-config; ``Tracer`` is the ray-tracer + sensor
optic (fov); the three ``Noise / Polarization / Receiver`` foldouts are the optional
``RadarConfig`` sub-configs; ``Solve`` runs the solve and ``Post Processing`` drives
the in-component MIMO views + detector/CFAR; ``Timeline`` generates multi-frame sequences.

The sensor **pose** (position + look direction + up) is read from the owner SceneObject's
``Transform`` at solve time — moving / rotating the Radar Settings object in the viewport
moves the simulated radar. The solver backend defaults to ``dirichlet`` on ``cuda`` and
is hidden from the UI; tests / scripts can still override ``backend`` / ``device`` /
``pad_factor`` on the component directly.
"""
from dataclasses import asdict
import json
import threading
import time

import numpy as np

from witwin_server import Notifications
from witwin_server.api import api
from witwin_server.core.components import (
    Component,
    GizmoContext,
    bool_field,
    button,
    component,
    component_field,
    define_group,
    figure,
    float_field,
    foldout_group,
    int_field,
    list_field,
    stream_ref_field,
    string_field,
    vector3_field,
)
from witwin_server.utils.logging import get_logger

from ..adapter.common import (
    FREQUENCY_UNITS,
    SAMPLE_RATE_UNITS,
    SLOPE_UNITS,
    TIME_UNITS,
    Derived,
    num,
)
from ..adapter.solve import SensorSpec, SigProc, TracerSpec

logger = get_logger("Radar")

_CAT = "Simulation/Radar"
_VIEWS = ["raw_signal", "range_doppler", "point_cloud", "music"]

# Foldout group ids (also used in string `show_if`/`hide_if` lookups by field name).
_CONFIG = "Configuration"   # FMCW core: frequency/power + ADC + chirp/frame + FFT bins
_ANTENNA = "Antenna"        # geometry + antenna pattern
_TRACER = "Tracer"
_NOISE = "Noise"
_POLARIZATION = "Polarization"
_RECEIVER = "Receiver"
_SOLVE = "Solve"
_POSTPROC = "PostProcessing"   # signal views + detector/CFAR run on the solved signal
_TIMELINE = "Timeline"

# show_if / hide_if dict conditions reused across multiple fields.
_PIXEL = {"field_name": "sampling", "operator": "eq", "value": "pixel"}
_SEPARABLE = {"field_name": "pattern_kind", "operator": "eq", "value": "separable"}
_MAP = {"field_name": "pattern_kind", "operator": "eq", "value": "map"}
_AGC = "enable_agc"
_POINTCLOUD = {"field_name": "timeline_source", "operator": "eq", "value": "pointcloud_sequence"}
_MOTION_SOURCE = {"field_name": "timeline_source", "operator": "eq", "value": "motion"}


@component(name="Radar", display_name="Radar", category=_CAT)
class RadarComponent(Component):
    """Unified Radar Settings component — FMCW config + sensor + sub-configs + solve + timeline."""

    _signal = None   # runtime MIMO cube (TX, RX, chirps, ADC); never serialized
    _radar = None    # runtime Radar
    _frames = None   # runtime timeline frame stack; never serialized
    _timeline_radar = None
    _solver_result_handle = ""
    _solver_run_id = ""
    _signal_stream_id = ""
    _live_thread = None
    _live_stop_event = None
    _live_pause_event = None
    _live_lock = None
    _live_last_signature = None
    _live_started_at = 0.0
    _live_generation = 0

    define_group(foldout_group(_CONFIG, display_name="Configuration"))
    define_group(foldout_group(_ANTENNA, display_name="Antenna"))
    define_group(foldout_group(_TRACER, display_name="Ray Tracer"))
    define_group(foldout_group(_NOISE, display_name="Noise Model", collapsed=True))
    define_group(foldout_group(_POLARIZATION, display_name="Polarization", collapsed=True))
    define_group(foldout_group(_RECEIVER, display_name="Receiver Chain", collapsed=True))
    define_group(foldout_group(_SOLVE, display_name="Solve"))
    define_group(foldout_group(_POSTPROC, display_name="Post Processing"))
    define_group(foldout_group(_TIMELINE, display_name="Timeline", collapsed=True))

    # --- frequency / power (RadarConfig) ------------------------------------
    fc = float_field(77e9, min=1e9, max=300e9, units=FREQUENCY_UNITS, default_unit="GHz",
                     group=_CONFIG, description="Carrier / start frequency")
    slope = float_field(60.012, min=0.0, units=SLOPE_UNITS, default_unit="MHz/us",
                        group=_CONFIG, description="Chirp frequency slope")
    power = float_field(15.0, group=_CONFIG, description="TX power (dBm)")

    # --- ADC / sampling (RadarConfig) ---------------------------------------
    adc_samples = int_field(256, min=1, group=_CONFIG, description="ADC samples per chirp (fast time)")
    sample_rate = float_field(4400.0, min=0.0, units=SAMPLE_RATE_UNITS, default_unit="ksps",
                              group=_CONFIG, description="ADC sample rate")
    adc_start_time = float_field(6.0, min=0.0, units=TIME_UNITS, default_unit="us",
                                 group=_CONFIG, description="ADC start delay")

    # --- chirp / frame (RadarConfig) ----------------------------------------
    idle_time = float_field(7.0, min=0.0, units=TIME_UNITS, default_unit="us",
                            group=_CONFIG, description="Idle time between chirps")
    ramp_end_time = float_field(58.0, min=0.0, units=TIME_UNITS, default_unit="us",
                                group=_CONFIG, description="Active chirp ramp time")
    chirp_per_frame = int_field(128, min=1, group=_CONFIG, description="Chirps per frame (slow time / Doppler)")
    frame_per_second = float_field(10.0, min=0.0, group=_CONFIG, description="Frame rate (Hz)")

    # --- FFT bins (RadarConfig) ---------------------------------------------
    num_doppler_bins = int_field(128, min=1, group=_CONFIG, description="Doppler FFT bins")
    num_range_bins = int_field(256, min=1, group=_CONFIG, description="Range bins")
    num_angle_bins = int_field(64, min=1, group=_CONFIG, description="Angle FFT bins")

    # --- antenna geometry (RadarConfig; half-wavelength units) --------------
    num_tx = int_field(3, min=1, group=_ANTENNA, description="Number of TX antennas (== len(tx_loc))")
    num_rx = int_field(4, min=1, group=_ANTENNA, description="Number of RX antennas (== len(rx_loc))")
    tx_loc = list_field(vector3_field([0.0, 0.0, 0.0]),
                        default=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
                        group=_ANTENNA, description="TX positions (half-wavelength units)")
    rx_loc = list_field(vector3_field([0.0, 0.0, 0.0]),
                        default=[[-6.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [-3.0, 0.0, 0.0]],
                        group=_ANTENNA, description="RX positions (half-wavelength units)")

    # --- ray tracer (RadarTracer + sensor optic; pose comes from the owner Transform) ---
    fov = float_field(60.0, min=1.0, max=179.0, group=_TRACER,
                      description="Ray-tracing field of view (deg)")
    resolution = int_field(128, min=1, group=_TRACER, description="Ray-tracing film width = height")
    epsilon_r = float_field(5.0, min=1.0, group=_TRACER,
                            description="Default Fresnel permittivity (per-structure eps_r overrides)")
    sampling = string_field("triangle", options=["pixel", "triangle"], enum_toggle=True,
                            group=_TRACER,
                            description="Triangle (mesh facets) or pixel (camera rays) sampling")
    multipath = bool_field(False, show_if=_PIXEL, group=_TRACER,
                           description="Trace reflections (pixel sampling only)")
    max_reflections = int_field(0, min=0, show_if="multipath", group=_TRACER,
                                description="Reflection bounce count")
    ray_batch_size = int_field(65536, min=1, group=_TRACER, description="Rays per batch (multipath)")

    # Hidden solver backend defaults (dirichlet on CUDA). Authoring rarely needs to
    # change these; tests / scripts can still override the fields directly.
    backend = string_field("dirichlet", options=["dirichlet", "slang", "pytorch"], hidden=True)
    pad_factor = int_field(16, min=1, hidden=True)
    device = string_field("cuda", options=["cuda", "cpu"], hidden=True)

    # --- antenna pattern sub-config (RadarAntennaPattern) -------------------
    use_default = bool_field(True, group=_ANTENNA,
                             description="Use the default half-wave dipole pattern")
    pattern_kind = string_field("separable", options=["separable", "map"], enum_toggle=True,
                                hide_if="use_default", group=_ANTENNA,
                                description="Separable per-axis cuts or a 2D gain map")
    x_angles_deg = list_field(float_field(0.0), default=[], hide_if="use_default", group=_ANTENNA,
                              description="Azimuth angles (deg, strictly increasing, >=2)")
    y_angles_deg = list_field(float_field(0.0), default=[], hide_if="use_default", group=_ANTENNA,
                              description="Elevation angles (deg, strictly increasing, >=2)")
    x_values = list_field(float_field(0.0), default=[], show_if=_SEPARABLE, group=_ANTENNA,
                          description="Per-azimuth gain (>=0, len == x_angles)")
    y_values = list_field(float_field(0.0), default=[], show_if=_SEPARABLE, group=_ANTENNA,
                          description="Per-elevation gain (>=0, len == y_angles)")
    values_2d_json = string_field("", widget="textarea", show_if=_MAP, group=_ANTENNA,
                                  description="2D gain map as JSON rows x cols "
                                              "(rows=y_angles, cols=x_angles, >=0)")

    # --- noise model sub-config (RadarNoiseModel) ---------------------------
    enable_thermal = bool_field(False, group=_NOISE,
                                description="Additive complex Gaussian thermal noise")
    thermal_std = float_field(0.0, min=0.0, show_if="enable_thermal", group=_NOISE,
                              description="Thermal noise std (>=0)")
    enable_quantization = bool_field(False, group=_NOISE,
                                     description="ADC-style amplitude quantization")
    quant_bits = int_field(12, min=1, show_if="enable_quantization", group=_NOISE,
                           description="Quantizer bits (>0)")
    quant_full_scale = float_field(1.0, min=0.0, show_if="enable_quantization", group=_NOISE,
                                   description="Quantizer full scale (>0)")
    enable_phase = bool_field(False, group=_NOISE, description="Random-walk phase noise")
    phase_std = float_field(0.0, min=0.0, show_if="enable_phase", group=_NOISE,
                            description="Phase noise std per step (>=0)")
    use_seed = bool_field(False, group=_NOISE, description="Deterministic noise via a fixed seed")
    seed = int_field(0, min=0, show_if="use_seed", group=_NOISE, description="Noise RNG seed (>=0)")

    # --- polarization sub-config (RadarPolarization) ------------------------
    pol_enabled = bool_field(False, group=_POLARIZATION,
                             description="Enable explicit polarization (else unpolarized)")
    pol_tx_uniform = bool_field(True, show_if="pol_enabled", group=_POLARIZATION,
                                description="One shared TX vector (else per-antenna)")
    pol_tx = vector3_field([1.0, 0.0, 0.0], show_if="pol_tx_uniform", group=_POLARIZATION,
                           description="Shared TX polarization vector (non-zero)")
    pol_tx_bank = list_field(vector3_field([1.0, 0.0, 0.0]), default=[],
                             hide_if="pol_tx_uniform", group=_POLARIZATION,
                             description="Per-TX-antenna polarization vectors (len == num_tx)")
    pol_rx_uniform = bool_field(True, show_if="pol_enabled", group=_POLARIZATION,
                                description="One shared RX vector (else per-antenna)")
    pol_rx = vector3_field([1.0, 0.0, 0.0], show_if="pol_rx_uniform", group=_POLARIZATION,
                           description="Shared RX polarization vector (non-zero)")
    pol_rx_bank = list_field(vector3_field([1.0, 0.0, 0.0]), default=[],
                             hide_if="pol_rx_uniform", group=_POLARIZATION,
                             description="Per-RX-antenna polarization vectors (len == num_rx)")
    pol_reflection_flip = bool_field(True, show_if="pol_enabled", group=_POLARIZATION,
                                     description="Flip polarization on reflection")

    # --- receiver chain sub-config (RadarReceiverChain) ---------------------
    enable_lna = bool_field(False, group=_RECEIVER, description="Low-noise amplifier gain")
    lna_gain_db = float_field(0.0, show_if="enable_lna", group=_RECEIVER, description="LNA gain (dB)")
    enable_agc = bool_field(False, group=_RECEIVER, description="Automatic gain control")
    agc_target_rms = float_field(1.0, min=0.0, show_if=_AGC, group=_RECEIVER,
                                 description="AGC target RMS (>0)")
    agc_max_gain_db = float_field(60.0, show_if=_AGC, group=_RECEIVER, description="AGC max gain (dB)")
    agc_min_gain_db = float_field(-60.0, show_if=_AGC, group=_RECEIVER,
                                  description="AGC min gain (dB, <= max)")
    agc_mode = string_field("per_rx", options=["global", "per_rx"], enum_toggle=True, show_if=_AGC,
                            group=_RECEIVER, description="AGC scope")
    enable_adc = bool_field(False, group=_RECEIVER,
                            description="ADC quantization (mutually exclusive with quantization noise)")
    adc_bits = int_field(12, min=1, show_if="enable_adc", group=_RECEIVER, description="ADC bits (>0)")
    adc_full_scale = float_field(1.0, min=0.0, show_if="enable_adc", group=_RECEIVER,
                                 description="ADC full scale (>0)")
    reference_impedance_ohm = float_field(50.0, min=0.0, group=_RECEIVER,
                                          description="Reference impedance (ohm, >0)")

    # --- solve (RadarResult) ------------------------------------------------
    motion_sampling = string_field("per_chirp", options=["per_chirp", "per_frame"], enum_toggle=True,
                                   group=_SOLVE, description="Re-trace per chirp or per frame")
    t0 = float_field(0.0, group=_SOLVE, description="Solve start time (s)")
    stream_max_fps = float_field(2.0, min=0.1, max=30.0, group=_SOLVE,
                                 description="Maximum realtime stream solves per second")
    stream_channels = string_field("raw,rd,pc", group=_SOLVE,
                                   description="Comma-separated live channels: raw, rd, pc")
    stream_history_length = int_field(1, min=1, group=_SOLVE,
                                      description="Stream history frames retained for reconnect")
    stream_on_change_only = bool_field(True, group=_SOLVE,
                                       description="Only re-solve when radar settings or scene transforms change")
    stream_status = string_field("stopped", options=["stopped", "running", "paused", "error"],
                                 enum_toggle=True, readonly=True, group=_SOLVE,
                                 description="Realtime stream producer state")
    post_processors = list_field(component_field(component_type="RadarPostProcessor"), default=[],
                                 group=_SOLVE,
                                 description="Extra post-processor views run after each solve")
    signal_stream = stream_ref_field(default_channel="rd", group=_SOLVE, title="Live Signal")
    signal_figure = figure(title="Radar Signal", group=_SOLVE)

    # --- post processing: signal views + detector/CFAR (RadarResult) --------
    view = string_field("range_doppler", options=_VIEWS, enum_toggle=True, group=_POSTPROC,
                        description="Which signal view to render")
    tx_index = int_field(0, min=0, group=_POSTPROC, description="TX index (raw / range-doppler)")
    rx_index = int_field(0, min=0, group=_POSTPROC, description="RX index (raw / range-doppler)")
    static_clutter_removal = bool_field(False, group=_POSTPROC,
                                        description="Remove static (zero-doppler) clutter")
    show_cfar = bool_field(False, group=_POSTPROC,
                           description="Overlay CFAR detections on the range-doppler map")

    detector = string_field("cfar", options=["cfar", "topk"], enum_toggle=True, group=_POSTPROC,
                            description="Point-cloud detector")
    guard_doppler = int_field(2, min=0, group=_POSTPROC, description="CFAR guard cells (doppler)")
    guard_range = int_field(4, min=0, group=_POSTPROC, description="CFAR guard cells (range)")
    training_doppler = int_field(4, min=0, group=_POSTPROC, description="CFAR training cells (doppler)")
    training_range = int_field(8, min=0, group=_POSTPROC, description="CFAR training cells (range)")
    pfa = float_field(1e-3, min=0.0, group=_POSTPROC, description="CFAR probability of false alarm")
    energy_top_k = int_field(128, min=1, group=_POSTPROC,
                             description="Top-K energy detections (topk detector)")

    # --- timeline (RadarTimeline) -------------------------------------------
    frame_rate = float_field(30.0, min=1.0, group=_TIMELINE, description="Source keyframe rate (Hz)")
    timeline_source = string_field("pointcloud_sequence", options=["pointcloud_sequence", "motion"],
                                   enum_toggle=True, group=_TIMELINE,
                                   description="How keyframes are built")
    pointcloud_path = string_field("", show_if=_POINTCLOUD, group=_TIMELINE,
                                   description=".npy/.npz with (F,N,3) positions [+ (F,N) intensities]")
    motion_path = string_field("", show_if=_MOTION_SOURCE, group=_TIMELINE,
                               description=".npz pose (F,72)/shape/root_translation; needs an SMPL 'human'")
    velocity_corrected = bool_field(True, group=_TIMELINE,
                                    description="Scale displacement to physical velocity per frame")
    frame_index = int_field(0, min=0, group=_TIMELINE,
                            description="Frame to display (steps the RD movie)")

    # --- gizmo (pose comes from the owner Transform; cone direction is local -Z) ---

    def on_draw_gizmos(self, ctx: GizmoContext) -> None:
        """Sphere at the sensor origin + an FOV/range cone along local -Z (Transform rotates it)."""
        ctx.color = "#ffd400"
        ctx.draw_sphere(radius=0.05)
        ctx.draw_cone(fov="fov", range=self._max_range(), segments=4,
                      direction=(0.0, 0.0, -1.0))

    def _max_range(self) -> float:
        return Derived.compute(self)["max_range_m"]

    # --- buttons ------------------------------------------------------------

    @button(display_name="Show Derived Values", group=_ANTENNA)
    def show_derived(self):
        """Report the derived range/doppler resolution + max range/doppler."""
        d = Derived.compute(self)
        msg = (f"range res {d['range_resolution_m']:.4f} m | max range {d['max_range_m']:.2f} m | "
               f"doppler res {d['doppler_resolution_mps']:.4f} m/s | "
               f"max doppler {d['max_doppler_mps']:.2f} m/s")
        Notifications.info("Radar", msg)
        return msg

    @button(display_name="Simulate", group=_SOLVE)
    def simulate(self):
        """Rebuild + solve the live scene, keep the signal, and render the current view."""
        logger.info("=== Simulate clicked ===")
        spec = SensorSpec.from_component(self)
        logger.info(f"Sensor pose:  position={spec.position}  target={spec.target}  up={spec.up}")
        logger.info(f"Sensor:       backend={spec.backend} device={spec.device} fov={spec.fov}")
        run = api.solvers.solve(
            "witwin.radar.simulate",
            scene=self.scene,
            config=self._solver_config(spec),
        )
        if run.status != "succeeded":
            message = (run.error or {}).get("message") or "simulate failed"
            Notifications.error("Radar", message)
            return f"Solve failed: {message}"
        handle = (run.outputs or {}).get("resultHandle")
        if not handle:
            Notifications.error("Radar", "Solver returned no result handle")
            return "Solve failed: no result handle"
        self._solver_result_handle = str(handle)
        self._solver_run_id = run.run_id
        self._radar = None
        self._signal = None
        self.update_view()
        self._publish_signal_stream()
        self._run_post_processors()
        Notifications.success("Radar", "Radar solve complete")
        logger.info("=== Simulate done ===")
        return "Solve complete"

    @button(display_name="Start Stream", group=_SOLVE)
    def start_stream(self):
        """Start realtime solve + stream production."""
        logger.info("=== Radar Start Stream clicked ===")
        if self.scene is None or self.owner is None:
            logger.warning("Radar stream start ignored: component is not attached to a scene")
            return "Radar must be attached to a scene"
        self._ensure_live_state()
        with self._live_lock:
            if self._live_thread is not None and self._live_thread.is_alive():
                if self._live_stop_event.is_set():
                    logger.info("Radar stream start ignored: existing stream is stopping")
                    return "Stream stopping"
                self._live_pause_event.clear()
                self.stream_status = "running"
                self._update_stream_status("active")
                Notifications.info("Radar", "Realtime stream resumed")
                logger.info(f"Radar stream resumed: thread={self._live_thread.name}")
                return "Stream running"
            self._live_generation += 1
            generation = self._live_generation
            self._live_stop_event.clear()
            self._live_pause_event.clear()
            self._live_started_at = time.monotonic()
            self._live_last_signature = None
            self.stream_status = "running"
            self._open_signal_stream(self._stream_channel_descriptors(), status="active")
            self._live_thread = threading.Thread(
                target=self._live_loop,
                args=(generation,),
                name=f"RadarStream-{id(self):x}",
                daemon=True,
            )
            self._live_thread.start()
            logger.info(f"Radar stream thread started: generation={generation} thread={self._live_thread.name}")
        Notifications.info("Radar", "Realtime stream started")
        return "Stream started"

    @button(display_name="Pause Stream", group=_SOLVE)
    def pause_stream(self):
        """Pause realtime solve production without closing the stream."""
        self._ensure_live_state()
        self._live_pause_event.set()
        self.stream_status = "paused"
        self._update_stream_status("paused")
        Notifications.info("Radar", "Realtime stream paused")
        logger.info("Radar stream paused")
        return "Stream paused"

    @button(display_name="Stop Stream", group=_SOLVE)
    def stop_stream(self):
        """Stop realtime production and close the stream."""
        self._ensure_live_state()
        logger.info("Radar stream stop requested")
        with self._live_lock:
            self._live_generation += 1
            self._live_stop_event.set()
            self._live_pause_event.clear()
            self.stream_status = "stopped"
            if self._signal_stream_id:
                try:
                    api.streams.close(self._signal_stream_id, reason="stopped")
                except Exception as exc:  # noqa: BLE001 - stop should be idempotent
                    logger.debug(f"Radar stream close skipped: {exc}")
            self.signal_stream = None
            self._signal_stream_id = ""
        Notifications.info("Radar", "Realtime stream stopped")
        logger.info("Radar stream stopped")
        return "Stream stopped"

    @button(display_name="Update View", group=_POSTPROC)
    def update_view(self):
        """Render the selected view from the last solved signal."""
        if not self._solver_result_handle and self._signal is None:
            logger.info("Update View: no signal yet — run Simulate first")
            return "Simulate first"
        view = str(self.view)
        logger.info(f"Update View: rendering '{view}' (tx={self._tx()} rx={self._rx()})")
        if view == "raw_signal":
            return self._show_raw()
        if view == "range_doppler":
            return self._show_rd()
        if view == "point_cloud":
            return self._show_pc()
        return self._show_music()

    @button(display_name="Generate Frames", group=_TIMELINE)
    def generate_timeline(self):
        """Build the timeline + radar, generate the frame stack, and show the first frame."""
        import torch
        if not torch.cuda.is_available():
            Notifications.warning("Radar", "Timeline generation requires a CUDA device")
            return "CUDA required"
        from ..adapter.timeline_run import TimelineRunner

        radar, frames = TimelineRunner.generate(
            self,
            sensor=SensorSpec.from_component(self),
            tracer=TracerSpec.from_component(self),
            studio_scene=self.scene)
        self._timeline_radar = radar
        self._frames = frames
        self._solver_result_handle = ""
        self._solver_run_id = ""
        self._publish_signal_stream()
        self.show_frame()
        Notifications.success("Radar", f"Timeline: {frames.shape[0]} frames {tuple(frames.shape)}")
        return f"{frames.shape[0]} frames"

    @button(display_name="Show Frame", group=_TIMELINE)
    def show_frame(self):
        """Display the selected timeline frame in the signal view."""
        if self._frames is None:
            return "Generate first"
        index = max(0, min(int(self.frame_index), self._frames.shape[0] - 1))
        self._signal = self._frames[index]
        self._radar = self._timeline_radar
        self._solver_result_handle = ""
        self._solver_run_id = ""
        self.update_view()
        self._publish_signal_stream()
        return f"Frame {index}"

    # --- post-processors ----------------------------------------------------

    def _run_post_processors(self):
        refs = self.post_processors or []
        if not refs:
            return
        scene = self.scene
        for ref in refs:
            proc = self._resolve_processor(scene, ref)
            if proc is not None and bool(getattr(proc, "enabled", True)):
                if self._solver_result_handle:
                    self._run_remote_post_processor(proc)
                else:
                    proc.process(self._radar, self._signal)

    def _run_remote_post_processor(self, proc):
        name = proc.__class__.__name__
        if name == "RangeDopplerProcessorComponent":
            payload = self._query_result("range_doppler", {
                "tx": int(getattr(proc, "tx_index", 0)),
                "rx": int(getattr(proc, "rx_index", 0)),
                "static_clutter_removal": bool(getattr(proc, "static_clutter_removal", False)),
                "show_cfar": False,
            })
            mag_db = np.asarray(payload["mag_db"], dtype=np.float32)
            tx = int(payload.get("tx", getattr(proc, "tx_index", 0)))
            rx = int(payload.get("rx", getattr(proc, "rx_index", 0)))
            proc.result_figure.clear().imshow(mag_db).title(f"RD Tx{tx} Rx{rx}")
            return
        if name == "PointCloudProcessorComponent":
            payload = self._query_result("point_cloud", {
                "detector": "cfar",
                "static_clutter_removal": bool(getattr(proc, "static_clutter_removal", False)),
                "guard": [2, 4],
                "training": [4, 8],
                "pfa": 1e-3,
                "energy_top_k": 128,
            })
            points = payload.get("points")
            pc = np.asarray([] if points is None else points, dtype=np.float32)
            fig = proc.result_figure.clear()
            if pc.size == 0:
                fig.title("Point cloud (no detections)")
            else:
                fig.scatter(pc[:, 0].tolist(), pc[:, 2].tolist(),
                            label="points", color="#30c0ff", size=10)
                fig.title(f"Point cloud ({pc.shape[0]} pts)").xlabel("x (m)").ylabel("z (m)")
            return
        logger.warning(f"Skipping remote Radar post-processor {name}: no query adapter")

    @staticmethod
    def _resolve_processor(scene, ref):
        if hasattr(ref, "process"):
            return ref
        if not isinstance(ref, dict) or not ref.get("object_id"):
            return None
        obj = scene.get_object(ref["object_id"]) if scene is not None else None
        if obj is None:
            return None
        named = obj.get_component(ref.get("component_type", "RadarPostProcessor"))
        if named is not None and hasattr(named, "process"):
            return named
        for comp in obj.get_all_components().values():
            if hasattr(comp, "process"):
                return comp
        return None

    # --- views --------------------------------------------------------------

    def _show_raw(self):
        if self._solver_result_handle:
            payload = self._query_result("raw_signal", {"tx": int(self.tx_index), "rx": int(self.rx_index)})
            x = payload["x"]
            n_tx = int(payload["n_tx"])
            n_rx = int(payload["n_rx"])
            tx = int(payload["tx"])
            rx = int(payload["rx"])
            real = payload["real"]
            imag = payload["imag"]
            batches = payload["batches"]
        else:
            sig = self._signal.detach().cpu().numpy()
            n_tx, n_rx, _, n_adc = sig.shape
            x = list(range(n_adc))
            tx = self._tx()
            rx = self._rx()
            sel = sig[tx, rx, 0]
            real = sel.real.tolist()
            imag = sel.imag.tolist()
            batches = [[self._pair_series(sig[t, r, 0], x) for r in range(n_rx)] for t in range(n_tx)]
        fig = self.signal_figure.clear()
        fig.line(x, real, label="Real", color="#ff9500")
        fig.line(x, imag, label="Imag", color="#00aaff")
        fig.title(f"Raw MIMO (Tx{tx} Rx{rx}, chirp 0)").xlabel("ADC sample").ylabel("Amplitude")
        fig.batch2d(batches, [f"Tx{t}" for t in range(n_tx)], [f"Rx{r}" for r in range(n_rx)])
        logger.info(f"  raw view -> selected Tx{tx} Rx{rx}")
        return "Raw signal updated"

    @staticmethod
    def _pair_series(sample, x):
        return {"series": [
            {"x": x, "y": sample.real.tolist(), "label": "Real", "color": "#ff9500"},
            {"x": x, "y": sample.imag.tolist(), "label": "Imag", "color": "#00aaff"}]}

    def _show_rd(self):
        if self._solver_result_handle:
            payload = self._query_result("range_doppler", {
                "tx": int(self.tx_index),
                "rx": int(self.rx_index),
                "static_clutter_removal": bool(self.static_clutter_removal),
                "show_cfar": bool(self.show_cfar),
                "guard": list(self._guard()),
                "training": list(self._training()),
                "pfa": num(self.pfa),
            })
            mag_db = np.asarray(payload["mag_db"], dtype=np.float32)
            tx = int(payload.get("tx", self.tx_index))
            rx = int(payload.get("rx", self.rx_index))
            rows = payload.get("cfar_rows") or []
            cols = payload.get("cfar_cols") or []
        else:
            rd = SigProc.range_doppler(self._radar, self._signal, tx=self._tx(), rx=self._rx(),
                                       static_clutter_removal=bool(self.static_clutter_removal))
            mag_db = rd.mag_db
            tx = self._tx()
            rx = self._rx()
            rows = []
            cols = []
            if bool(self.show_cfar):
                mask = SigProc.cfar_mask(rd.rd_map, guard=self._guard(),
                                         training=self._training(), pfa=num(self.pfa))
                rows, cols = np.nonzero(mask)
                rows = rows.tolist()
                cols = cols.tolist()
        logger.info(f"  rd-map: shape={mag_db.shape} "
                    f"min={float(mag_db.min()):.2f}dB max={float(mag_db.max()):.2f}dB "
                    f"mean={float(mag_db.mean()):.2f}dB")
        fig = self.signal_figure.clear().imshow(mag_db)
        fig.title(f"Range-Doppler (Tx{tx} Rx{rx}, dB)").xlabel("Range bin").ylabel("Doppler bin")
        if bool(self.show_cfar):
            logger.info(f"  cfar hits: {len(cols)}")
            if cols:
                fig.scatter(cols, rows, label="CFAR", color="#ff3030", size=8)
        return "Range-doppler updated"

    def _show_pc(self):
        if self._solver_result_handle:
            payload = self._query_result("point_cloud", {
                "detector": str(self.detector),
                "static_clutter_removal": bool(self.static_clutter_removal),
                "guard": list(self._guard()),
                "training": list(self._training()),
                "pfa": num(self.pfa),
                "energy_top_k": int(self.energy_top_k),
            })
            points = payload.get("points")
            pc = np.asarray([] if points is None else points, dtype=np.float32)
        else:
            pc = SigProc.point_cloud(
                self._radar, self._signal, detector=str(self.detector),
                static_clutter_removal=bool(self.static_clutter_removal), guard=self._guard(),
                training=self._training(), pfa=num(self.pfa), energy_top_k=int(self.energy_top_k))
        fig = self.signal_figure.clear()
        if pc.shape[0] == 0:
            fig.title("Point cloud (no detections)")
            logger.info("  point cloud: 0 detections")
            return "No detections"
        fig.scatter(pc[:, 0].tolist(), pc[:, 2].tolist(), label="points", color="#30c0ff", size=10)
        fig.title(f"Point cloud ({pc.shape[0]} pts)").xlabel("x (m)").ylabel("z (m)")
        logger.info(f"  point cloud: {pc.shape[0]} detections")
        return f"{pc.shape[0]} points"

    def _stream_owner(self):
        owner = {
            "kind": "component",
            "componentName": "Radar",
            "fieldName": "signal_stream",
        }
        if getattr(self, "_owner", None) is not None:
            owner["objectId"] = self._owner.id
        if self._solver_run_id:
            owner["runId"] = self._solver_run_id
        return owner

    def _stream_id(self):
        if self._signal_stream_id:
            return self._signal_stream_id
        owner_id = getattr(getattr(self, "_owner", None), "id", None) or hex(id(self))[2:]
        self._signal_stream_id = f"radar.{owner_id}.signal"
        return self._signal_stream_id

    def _open_signal_stream(self, channels, *, status="active"):
        stream = api.streams.open(
            self._stream_id(),
            kind="radar.signal",
            owner=self._stream_owner(),
            label="Radar Signal",
            retention={"mode": "ring", "historyLength": int(self.stream_history_length)},
            metadata={"mode": "live" if str(self.stream_status) == "running" else "solve"},
            channels=channels,
        )
        if status != "active":
            stream.update(status=status)
        ref = stream.ref("rd")
        ref["viewport"] = {
            "enabled": True,
            "channelId": "pc",
            "kind": "points",
            "maxFps": min(float(self.stream_max_fps), 10.0),
            "maxPoints": 100000,
            "pointSize": 0.05,
            "color": "#30c0ff",
        }
        self.signal_stream = ref
        return stream

    def _publish_signal_stream(self, channels=None):
        try:
            descriptors = self._stream_channel_descriptors()
            stream = self._open_signal_stream(descriptors)
            stream_id = getattr(stream, "id", getattr(stream, "stream_id", self._stream_id()))
            wanted = set(channels or self._selected_stream_channels(default=("raw", "rd", "pc")))
            logger.info(f"Radar stream publish started: stream={stream_id} channels={sorted(wanted)}")
            if "rd" in wanted:
                self._publish_rd_stream(stream)
            if "pc" in wanted:
                self._publish_pc_stream(stream)
            if "raw" in wanted:
                self._publish_raw_stream(stream)
            logger.info(f"Radar stream publish complete: stream={stream_id}")
        except Exception as exc:  # noqa: BLE001 - stream output must not break solve preview
            logger.warning(f"Radar stream publish skipped: {exc}")

    def _stream_channel_descriptors(self):
        return [
            {"channelId": "raw", "dtype": "complex64", "shape": ["adc"],
             "semantic": "time_series", "label": "Raw IQ"},
            {"channelId": "rd", "dtype": "float32", "shape": ["doppler", "range"],
             "semantic": "heatmap", "label": "Range-Doppler"},
            {"channelId": "pc", "dtype": "float32", "shape": ["points", 6],
             "semantic": "pointcloud_xyz", "label": "Point cloud"},
        ]

    def _selected_stream_channels(self, default=("raw", "rd", "pc")):
        allowed = {"raw", "rd", "pc"}
        selected = {
            part.strip().lower()
            for part in str(self.stream_channels or "").split(",")
            if part.strip()
        }
        selected = selected & allowed
        return tuple(selected or default)

    def _update_stream_status(self, status):
        if not self._signal_stream_id:
            return
        try:
            api.streams.open(
                self._signal_stream_id,
                kind="radar.signal",
                owner=self._stream_owner(),
                channels=self._stream_channel_descriptors(),
            ).update(status=status)
        except Exception as exc:  # noqa: BLE001 - status update is best effort
            logger.debug(f"Radar stream status update skipped: {exc}")

    def _ensure_live_state(self):
        if self._live_lock is None:
            self._live_lock = threading.RLock()
        if self._live_stop_event is None:
            self._live_stop_event = threading.Event()
        if self._live_pause_event is None:
            self._live_pause_event = threading.Event()

    def _live_owner_active(self):
        scene = self.scene
        owner = getattr(self, "owner", None)
        owner_id = getattr(owner, "id", None)
        objects = getattr(scene, "objects", None)
        return bool(scene is not None and owner is not None and owner_id and objects and objects.get(owner_id) is owner)

    def _live_loop(self, generation):
        self._ensure_live_state()
        logger.info(f"Radar live loop entered: generation={generation}")
        while self._live_generation_active(generation) and not self._live_stop_event.is_set():
            if not self._live_owner_active():
                logger.info("Radar live loop stopping: owner is no longer active")
                self.stop_stream()
                break
            if self._live_pause_event.is_set():
                time.sleep(0.05)
                continue
            interval = 1.0 / max(0.1, float(self.stream_max_fps))
            started = time.monotonic()
            signature = self._live_scene_signature()
            should_solve = not bool(self.stream_on_change_only) or signature != self._live_last_signature
            if should_solve:
                logger.info(f"Radar live loop solving: generation={generation} signature_changed={signature != self._live_last_signature}")
                ok = self._solve_once_for_stream(generation)
                if not ok:
                    self.stream_status = "error"
                    self._update_stream_status("error")
                    time.sleep(max(0.25, interval))
                else:
                    self._live_last_signature = signature
            elapsed = time.monotonic() - started
            time.sleep(max(0.01, interval - elapsed))
        logger.info(f"Radar live loop exited: generation={generation}")

    def _live_generation_active(self, generation):
        return generation == self._live_generation

    def _solve_once_for_stream(self, generation):
        try:
            if not self._live_generation_active(generation):
                return True
            if self._live_stop_event is not None and self._live_stop_event.is_set():
                return True
            spec = SensorSpec.from_component(self)
            t0 = self._live_t0()
            logger.info(
                "Radar live solve request: "
                f"backend={spec.backend} device={spec.device} position={spec.position} t0={t0:.6f}"
            )
            logger.info(f"Radar live scene state: {self._live_scene_summary()}")
            run = api.solvers.solve(
                "witwin.radar.simulate",
                scene=self.scene,
                config=self._solver_config(spec, t0_override=t0),
            )
            logger.info(f"Radar live solve returned: status={run.status} run_id={run.run_id}")
            if not self._live_generation_active(generation):
                return True
            if run.status != "succeeded":
                message = (run.error or {}).get("message") or "simulate failed"
                self._mark_stream_error(message)
                return False
            handle = (run.outputs or {}).get("resultHandle")
            if not handle:
                self._mark_stream_error("Solver returned no result handle")
                return False
            self._solver_result_handle = str(handle)
            self._solver_run_id = run.run_id
            self._radar = None
            self._signal = None
            if not self._live_generation_active(generation):
                return True
            if self._live_stop_event is not None and self._live_stop_event.is_set():
                return True
            self.stream_status = "running"
            self._update_stream_status("active")
            self._update_live_preview()
            self._publish_signal_stream(channels=self._selected_stream_channels())
            return True
        except Exception as exc:  # noqa: BLE001 - background thread must survive one failed frame
            logger.warning(f"Radar live solve failed: {exc}")
            self._mark_stream_error(str(exc))
            return False

    def _update_live_preview(self):
        try:
            message = self.update_view()
            logger.info(f"Radar live preview updated: {message}")
        except Exception as exc:  # noqa: BLE001 - preview must not break stream publishing
            logger.warning(f"Radar live preview update skipped: {exc}")

    def _mark_stream_error(self, message):
        if not self._signal_stream_id:
            return
        try:
            api.streams.error(self._signal_stream_id, message)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Radar stream error update skipped: {exc}")

    def _live_t0(self):
        if bool(self.stream_on_change_only):
            return num(self.t0)
        return num(self.t0) + max(0.0, time.monotonic() - float(self._live_started_at or time.monotonic()))

    def _live_scene_signature(self):
        scene = self.scene
        items = [("radar", self._component_signature())]
        if scene is not None:
            for obj_id, obj in sorted(scene.objects.items()):
                transform = obj.get_component("Transform")
                if transform is None:
                    continue
                items.append((
                    obj_id,
                    self._signature_value(getattr(transform, "position", None)),
                    self._signature_value(getattr(transform, "rotation", None)),
                    self._signature_value(getattr(transform, "scale", None)),
                    bool(getattr(obj, "visible", True)),
                ))
        return json.dumps(items, sort_keys=True, separators=(",", ":"))

    def _live_scene_summary(self, limit=8):
        scene = self.scene
        if scene is None:
            return "scene=None"
        objects = getattr(scene, "objects", {}) or {}
        rows = []
        for obj_id, obj in sorted(objects.items()):
            transform = obj.get_component("Transform")
            if transform is None:
                continue
            has_geometry = obj.get_component("PlatformGeometry") is not None or obj.get_component("Mesh") is not None
            if obj is not self.owner and not has_geometry:
                continue
            name = getattr(obj, "name", obj_id)
            role = "radar" if obj is self.owner else "structure"
            pos = self._signature_value(getattr(transform, "position", None))
            rot = self._signature_value(getattr(transform, "rotation", None))
            scale = self._signature_value(getattr(transform, "scale", None))
            rows.append(f"{role}:{name} pos={pos} rot={rot} scale={scale} visible={bool(getattr(obj, 'visible', True))}")
            if len(rows) >= int(limit):
                break
        suffix = "" if len(rows) == len(objects) else f" shown={len(rows)}"
        return f"objects={len(objects)}{suffix} | " + " | ".join(rows)

    def _component_signature(self):
        ignored = {"signal_stream", "signal_figure", "stream_status"}
        values = {}
        for name in sorted(getattr(self, "_fields_meta", {})):
            if name in ignored:
                continue
            values[name] = self._signature_value(getattr(self, name))
        return values

    @staticmethod
    def _signature_value(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, dict):
            return {str(k): RadarComponent._signature_value(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [RadarComponent._signature_value(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return round(float(value), 8)
        if isinstance(value, (int, bool, str)) or value is None:
            return value
        return repr(value)

    def _publish_raw_stream(self, stream):
        if self._solver_result_handle:
            payload = self._query_result("raw_signal", {"tx": int(self.tx_index), "rx": int(self.rx_index)})
            real = np.asarray(payload.get("real") or [], dtype=np.float32)
            imag = np.asarray(payload.get("imag") or [], dtype=np.float32)
            if real.size == 0 or imag.size != real.size:
                return
            iq = np.empty(real.size * 2, dtype=np.float32)
            iq[0::2] = real
            iq[1::2] = imag
            stream.publish("raw", iq, metadata={
                "dtype": "complex64",
                "shape": [int(real.size)],
                "tx": int(payload.get("tx", self.tx_index)),
                "rx": int(payload.get("rx", self.rx_index)),
                "chirp": 0,
            })
            amp = np.sqrt(real * real + imag * imag)
            logger.info(
                "Radar stream raw frame: "
                f"shape={[int(real.size)]} mean_amp={float(amp.mean()):.6e} max_amp={float(amp.max()):.6e}"
            )
            return

        if self._signal is None:
            return
        sig = self._signal.detach().cpu().numpy()
        iq = np.empty(sig.size * 2, dtype=np.float32)
        flat = sig.reshape(-1)
        iq[0::2] = flat.real.astype(np.float32, copy=False)
        iq[1::2] = flat.imag.astype(np.float32, copy=False)
        stream.publish("raw", iq, metadata={"dtype": "complex64", "shape": list(sig.shape)})
        amp = np.abs(sig)
        logger.info(
            "Radar stream raw frame: "
            f"shape={list(sig.shape)} mean_amp={float(amp.mean()):.6e} max_amp={float(amp.max()):.6e}"
        )

    def _publish_rd_stream(self, stream):
        if self._solver_result_handle:
            payload = self._query_result("range_doppler", {
                "tx": int(self.tx_index),
                "rx": int(self.rx_index),
                "static_clutter_removal": bool(self.static_clutter_removal),
                "show_cfar": False,
            })
            mag_db = np.asarray(payload.get("mag_db") or [], dtype=np.float32)
        else:
            if self._signal is None or self._radar is None:
                return
            mag_db = SigProc.range_doppler(
                self._radar,
                self._signal,
                tx=self._tx(),
                rx=self._rx(),
                static_clutter_removal=bool(self.static_clutter_removal),
            ).mag_db.astype(np.float32, copy=False)
        if mag_db.size == 0:
            return
        stream.publish("rd", np.ascontiguousarray(mag_db, dtype=np.float32), metadata={
            "dtype": "float32",
            "shape": list(mag_db.shape),
        })
        logger.info(
            "Radar stream rd frame: "
            f"shape={list(mag_db.shape)} min={float(mag_db.min()):.6e} "
            f"max={float(mag_db.max()):.6e} mean={float(mag_db.mean()):.6e}"
        )

    def _publish_pc_stream(self, stream):
        if self._solver_result_handle:
            payload = self._query_result("point_cloud", {
                "detector": str(self.detector),
                "static_clutter_removal": bool(self.static_clutter_removal),
                "guard": list(self._guard()),
                "training": list(self._training()),
                "pfa": num(self.pfa),
                "energy_top_k": int(self.energy_top_k),
            })
            pc = np.asarray([] if payload.get("points") is None else payload.get("points"), dtype=np.float32)
        else:
            if self._signal is None or self._radar is None:
                return
            pc = np.asarray(SigProc.point_cloud(
                self._radar,
                self._signal,
                detector=str(self.detector),
                static_clutter_removal=bool(self.static_clutter_removal),
                guard=self._guard(),
                training=self._training(),
                pfa=num(self.pfa),
                energy_top_k=int(self.energy_top_k),
            ), dtype=np.float32)
        if pc.ndim == 1:
            pc = pc.reshape((0, 6)) if pc.size == 0 else pc.reshape((1, pc.size))
        stream.publish("pc", np.ascontiguousarray(pc, dtype=np.float32), metadata={
            "dtype": "float32",
            "shape": list(pc.shape),
        })
        if pc.size == 0:
            logger.info(f"Radar stream pc frame: shape={list(pc.shape)} empty")
        else:
            mins = pc[:, :3].min(axis=0).tolist()
            maxs = pc[:, :3].max(axis=0).tolist()
            logger.info(
                "Radar stream pc frame: "
                f"shape={list(pc.shape)} xyz_min={[round(float(v), 4) for v in mins]} "
                f"xyz_max={[round(float(v), 4) for v in maxs]}"
            )

    def _show_music(self):
        try:
            if self._solver_result_handle:
                payload = self._query_result("music", {"num_pixels": 64})
                img = np.asarray(payload["image"], dtype=np.float32)
            else:
                img = SigProc.music_image(self._radar, self._signal, num_pixels=64)
        except Exception as exc:  # noqa: BLE001 - optional view; small UPAs make MUSIC ill-posed
            Notifications.warning("Radar", f"MUSIC needs a larger UPA (e.g. 20x20): {type(exc).__name__}")
            logger.warning(f"  MUSIC unavailable: {type(exc).__name__}")
            return "MUSIC unavailable for this array"
        logger.info(f"  music image: shape={img.shape} "
                    f"min={float(img.min()):.4e} max={float(img.max()):.4e}")
        self.signal_figure.clear().imshow(img).title("MUSIC image (max over range)")
        return "MUSIC updated"

    # --- helpers ------------------------------------------------------------

    def _guard(self):
        return (int(self.guard_doppler), int(self.guard_range))

    def _training(self):
        return (int(self.training_doppler), int(self.training_range))

    def _tx(self):
        if self._signal is None:
            return max(0, int(self.tx_index))
        return max(0, min(int(self.tx_index), self._signal.shape[0] - 1))

    def _rx(self):
        if self._signal is None:
            return max(0, int(self.rx_index))
        return max(0, min(int(self.rx_index), self._signal.shape[1] - 1))

    def _solver_config(self, spec: SensorSpec, *, t0_override=None) -> dict:
        return {
            "sensor": asdict(spec),
            "tracer": asdict(TracerSpec.from_component(self)),
            "motion_sampling": str(self.motion_sampling),
            "t0": num(self.t0) if t0_override is None else num(t0_override),
        }

    def _query_result(self, op: str, params: dict):
        response = api.solvers.query(
            "witwin.radar.simulate",
            self._solver_result_handle,
            op,
            params,
            run_id=self._solver_run_id or None,
        )
        return response.get("data") or {}
