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
import asyncio
import copy
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
    data_source_ref_field,
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
from .radar_ui import present_radar_controls

logger = get_logger("Radar")

_CAT = "Simulation/Radar"
_VIEWS = ["range_profile", "range_spectrum", "raw_signal", "range_doppler", "point_cloud", "music"]

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
_SAVED = "SavedResult"
_SNAPSHOT = "SnapshotTarget"
_ANIMATION = "StudioAnimation"

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

    def to_dict(self):
        return present_radar_controls(super().to_dict())

    @classmethod
    def get_definition(cls):
        return present_radar_controls(super().get_definition())

    _signal = None   # runtime MIMO cube (TX, RX, chirps, ADC); never serialized
    _radar = None    # runtime Radar
    _frames = None   # runtime timeline frame stack; never serialized
    _timeline_radar = None
    _solver_result_handle = ""
    _solver_run_id = ""
    _snapshot_solver_result_handle = ""
    _snapshot_solver_run_id = ""
    _snapshot_solver_pending_run_id = ""
    _signal_stream_id = ""
    _signal_source_id = ""
    _live_thread = None
    _live_stop_event = None
    _live_pause_event = None
    _live_lock = None
    _live_last_signature = None
    _live_last_scene_payload_signature = None
    _live_started_at = 0.0
    _live_generation = 0
    _live_session_id = ""
    _auto_refreshing_view = False
    _saved_result = None
    _snapshot_result = False
    _snapshot_running = False
    _animation_result = False
    _animation_view_generation = 0
    _replay_preparing = False
    _export_running = False
    _POSTPROC_AUTO_REFRESH_FIELDS = frozenset({
        "view",
        "tx_index",
        "rx_index",
        "static_clutter_removal",
        "show_cfar",
        "detector",
        "guard_doppler",
        "guard_range",
        "training_doppler",
        "training_range",
        "pfa",
        "energy_top_k",
    })
    _VIEW_STREAM_CHANNELS = {
        "raw_signal": "raw",
        "range_doppler": "rd",
        "point_cloud": "pc",
    }

    define_group(foldout_group(_CONFIG, display_name="Radar Configuration", collapsed=True))
    define_group(foldout_group(_ANTENNA, display_name="Antenna", collapsed=True))
    define_group(foldout_group(_TRACER, display_name="Legacy Ray Tracer (not snapshot sampling)", collapsed=True))
    define_group(foldout_group(_NOISE, display_name="Legacy Noise (unsupported in snapshots)", collapsed=True))
    define_group(foldout_group(_POLARIZATION, display_name="Legacy Polarization (unsupported in snapshots)", collapsed=True))
    define_group(foldout_group(_RECEIVER, display_name="Legacy Receiver (unsupported in snapshots)", collapsed=True))
    define_group(foldout_group(_SOLVE, display_name="Advanced runtime settings", collapsed=True))
    define_group(foldout_group(
        _SNAPSHOT,
        display_name="Static Snapshot (1-frame diagnostic)",
        collapsed=True,
    ))
    define_group(foldout_group(_ANIMATION, display_name="Simulation & Replay (multi-frame)"))
    define_group(foldout_group(_SAVED, display_name="Load Saved Recording", collapsed=True))
    define_group(foldout_group(_POSTPROC, display_name="Result Display"))
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
                             description="Use Radar's unconfigured antenna pattern (no custom gain map)")
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
                                   group=_SOLVE, hidden=True, description="Legacy only; snapshots freeze all motion")
    t0 = float_field(0.0, group=_ANIMATION,
                     description="Measurement start time (s). Multi-frame simulation continues from here; Snapshot freezes here.")
    stream_max_fps = float_field(30.0, min=0.1, max=30.0, group=_SOLVE,
                                 description="Maximum realtime stream solves per second")
    stream_channels = string_field("rd", group=_SOLVE,
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
    signal_stream = stream_ref_field(
        default_channel="rd",
        group=_SOLVE,
        hide_label=True,
        show_header=False,
        hidden=True,
    )
    signal_source = data_source_ref_field(
        default_channel="rd",
        mode="stream",
        kind="radar.result",
        group=_ANIMATION,
        hide_label=True,
        show_header=False,
        hide_if="saved_result_loaded",
    )
    signal_figure = figure(
        title="Radar Replay",
        group=_ANIMATION,
        hide_if={"field_name": "stream_status", "operator": "in", "value": ["running", "paused"]},
    )

    saved_result_path = string_field("", group=_SAVED,
                                     description="Absolute path to a metadata-bearing radar_cube.npz")
    saved_frame_index = int_field(0, min=0, group=_SAVED, description="Recorded frame index (not a simulation time)")
    saved_chirp_index = int_field(0, min=0, group=_SAVED, description="Recorded chirp to display; no averaging")
    saved_result_loaded = bool_field(False, readonly=True, hidden=True, transient=True, group=_SAVED)
    saved_result_status = string_field("No saved result loaded", readonly=True, transient=True, group=_SAVED)

    snapshot_model = string_field("One explicit RCS point; frozen pose, NO gait Doppler", readonly=True,
                                  display_name="Snapshot diagnostic model", group=_SNAPSHOT, transient=True)
    snapshot_target_id = string_field("", group=_SNAPSHOT,
                                     description="Exact existing target ID. Snapshot uses one local point; Animation samples this object's skin.")
    snapshot_local_point = vector3_field([0.0, 0.0, 0.0], group=_SNAPSHOT,
                                        description="Authored local point in metres before object/parent scale; NOT an automatically sampled skin point.")
    snapshot_rcs_m2 = float_field(0.1, min=0.0, group=_SNAPSHOT,
                                  description="Assumed scalar RCS in square metres; not a calibrated cat value.")
    snapshot_polarization = vector3_field([0.0, 1.0, 0.0], group=_SNAPSHOT,
                                          description="World-space polarization for both propagation legs.")
    snapshot_status = string_field("Not run. Static Snapshot freezes the scene at the measurement start time; LOS only.",
                                   readonly=True, transient=True, group=_SNAPSHOT)
    snapshot_figure = figure(title="Static diagnostic snapshot (one frame)", group=_SNAPSHOT)
    animation_duration_s = float_field(5.0, min=0.01, group=_ANIMATION,
                                       description="Exact baked timeline duration (0..30 seconds), starting at t0.")
    animation_fps = float_field(10.0, min=1.0, group=_ANIMATION,
                                description="Actual independent GPU measurements per second (1..30), not repeated video frames.")
    animation_frame_index = int_field(0, min=0, group=_ANIMATION,
                                      description="Recorded result frame. Does not seek or modify the editor timeline.")
    animation_status = string_field("Not run. Uses existing baked skin animation; LOS only, no extra room bounces.",
                                    readonly=True, transient=True, group=_ANIMATION)
    animation_export_path = string_field("", readonly=True, transient=True, group=_ANIMATION,
                                         description="Exported native complex cube, times, skin positions/velocities and physical axes (.npz).")
    replay_status = string_field("Not prepared. Uses this room's Timeline and numeric widgets.",
                                 readonly=True, transient=True, group=_ANIMATION)

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
        if self._saved_result is not None:
            return  # A review object's Transform is not the recorded sensor pose.
        ctx.color = "#ffd400"
        ctx.draw_sphere(radius=0.05)
        ctx.draw_cone(fov="fov", range=self._max_range(), segments=4,
                      direction=(0.0, 0.0, -1.0))

    def on_value_change(self, field_name, _old_value, _new_value):
        """Refresh solved post-processing previews when view controls change."""
        if field_name == "saved_result_path" and self._saved_result is not None:
            self._saved_result = None
            self.saved_result_loaded = False
            self.saved_result_status = "Result path changed; click Load Saved Result"
            self.signal_figure.clear()
            return
        if self._saved_result is not None:
            if field_name in self._POSTPROC_AUTO_REFRESH_FIELDS | {"saved_frame_index", "saved_chirp_index"}:
                self.update_view()
            return
        if self._animation_result:
            if field_name in self._POSTPROC_AUTO_REFRESH_FIELDS | {"animation_frame_index"}:
                self.update_view()
            elif field_name not in {"animation_status", "animation_export_path", "snapshot_status", "snapshot_figure", "signal_figure", "replay_status"}:
                self.animation_status = "Settings changed. Showing the recorded run; click Simulate & Prepare Replay to recompute."
            return
        if self._snapshot_result:
            if field_name in self._POSTPROC_AUTO_REFRESH_FIELDS:
                self.update_view()
            elif field_name not in {"snapshot_status", "snapshot_figure", "signal_figure"}:
                self.snapshot_status = "Settings changed. Display is the previous diagnostic; run Static Snapshot to recompute."
            return
        if field_name not in self._POSTPROC_AUTO_REFRESH_FIELDS:
            return
        if getattr(self, "_auto_refreshing_view", False):
            return
        stream_status = str(self.stream_status)
        if stream_status in {"running", "paused"}:
            if field_name == "view":
                self._open_signal_stream(
                    self._stream_channel_descriptors(),
                    status="active" if stream_status == "running" else stream_status,
                )
            return
        if not self._has_preview_result():
            return
        self._auto_refreshing_view = True
        try:
            self.update_view()
            self._publish_signal_stream()
        finally:
            self._auto_refreshing_view = False

    def _has_preview_result(self):
        return self._saved_result is not None or bool(getattr(self, "_solver_result_handle", "")) or getattr(self, "_signal", None) is not None

    @button(display_name="Load Saved Result", group=_SAVED)
    def load_saved_result(self):
        """Load exported producer metadata and reuse the existing figure UI."""
        from ..adapter.saved_result import SavedRadarResult
        if self._snapshot_running or self._export_running or self._replay_preparing or str(self.stream_status) in {"running", "paused"}:
            raise ValueError("Stop the running solve/live stream before loading a saved result.")
        self._saved_result = None
        self._last_animation_view_metadata = {}
        self._snapshot_result = False
        self._animation_result = False
        self._animation_view_generation += 1
        self.animation_export_path = ""
        self.saved_result_loaded = False
        self._signal = None
        self._frames = None
        self._radar = None
        self._solver_result_handle = ""
        self._clear_query_cache()
        self.signal_source = None
        self.signal_stream = None
        self.signal_figure.clear()
        self.saved_result_status = "Loading saved result"
        try:
            result = SavedRadarResult.load(str(self.saved_result_path))
        except Exception as exc:
            self.saved_result_status = f"Load failed: {exc}"
            raise
        self.view = "range_profile"
        self.tx_index = 0
        self.rx_index = 0
        self.static_clutter_removal = False
        self.show_cfar = False
        self.saved_frame_index = 0
        self.saved_chirp_index = 0
        self._saved_result = result
        self.saved_result_loaded = True
        self.saved_result_status = (
            f"{len(result.times_s)} frames, {result.times_s[0]:.4f}..{result.times_s[-1]:.4f}s | "
            f"{result.axes.waveform}/{result.axes.output_domain} | "
            f"{result.producer.get('scene_id', 'unknown scene')} | "
            f"{result.producer.get('physical_model', 'unspecified physical model')}"
        )
        return self.update_view()

    @staticmethod
    def _require_legacy_solver():
        # The historical live adapter has not been ported to Radar 0.3. This
        # is an explicit UI boundary, not a compatibility shim or fallback.
        import importlib.util
        if importlib.util.find_spec("witwin.radar.sigproc") is None:
            raise RuntimeError(
                "This plugin's legacy Stream/Generate adapter is not connected to Radar 0.3. "
                "Use Simulate for a frozen-point snapshot or Load Saved Result for replay. No solver or DSP fallback is performed."
            )

    def _max_range(self) -> float:
        return Derived.compute(self)["max_range_m"]

    # --- buttons ------------------------------------------------------------

    @button(display_name="Show Derived Values", group=_ANTENNA)
    def show_derived(self):
        """Report the derived range/doppler resolution + max range/doppler."""
        if self._saved_result is not None:
            axes = self._saved_result.axes
            msg = (
                f"Saved producer axes: range bin {axes.range_bin_m:.6f} m; "
                f"velocity bin {axes.velocity_bin_mps:.6f} m/s. "
                "Current Configuration fields do not reinterpret this saved result."
            )
            Notifications.info("Radar", msg)
            return msg
        d = Derived.compute(self)
        msg = (f"range res {d['range_resolution_m']:.4f} m | max range {d['max_range_m']:.2f} m | "
               f"doppler res {d['doppler_resolution_mps']:.4f} m/s | "
               f"max doppler {d['max_doppler_mps']:.2f} m/s")
        Notifications.info("Radar", msg)
        return msg

    @button(display_name="Run Static Snapshot (1 frame)", group=_SNAPSHOT, operation="solver", cancellable=True,
            progress_surface="component")
    def simulate(self):
        async def _run():
            return await self._simulate_async()

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())
        return _run()

    @button(display_name="Simulate & Prepare Replay", group=_ANIMATION, operation="solver", cancellable=True,
            progress_surface="component")
    def simulate_animation(self):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._simulate_async(animation=True))
        return self._simulate_async(animation=True)

    @button(display_name="Export Animation Result", group=_ANIMATION)
    async def export_animation_result(self):
        if self._export_running or self._replay_preparing:
            return "Export or replay preparation is already running; no duplicate file created."
        self._export_running = True
        task = asyncio.create_task(self._export_animation_result_once())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if task.done() and not task.cancelled():
                task.exception()
            raise
        except Exception as exc:
            if self._scene_is_live():
                self.animation_status = f"Export failed (completed simulation retained): {exc}"
            raise
        finally:
            self._export_running = False

    async def _export_animation_result_once(self):
        if not self._animation_result or self._snapshot_running or not self._scene_is_live():
            raise ValueError("Complete Simulate Animation before exporting.")
        identity = (self._solver_result_handle, self._solver_run_id)
        previous_path = self.animation_export_path
        self.animation_status = "Checking existing export; please wait." if previous_path else "Exporting full native result; please wait."
        payload = await asyncio.to_thread(self._query_result, "animation_export", {})
        if not self._scene_is_live() or not self._animation_result or identity != (self._solver_result_handle, self._solver_run_id):
            return "Previous animation exported; current result has changed."
        from ..adapter.animation import persist_export
        path = await asyncio.to_thread(persist_export, api, payload, existing_path=previous_path)
        if not self._scene_is_live() or not self._animation_result or identity != (self._solver_result_handle, self._solver_run_id):
            return "Previous animation exported to " + path
        self.animation_export_path = path
        reused = path == previous_path
        self.animation_status = ("Already exported; existing NPZ verified (no new copy): " if reused
                                 else "Exported native complex result + motion/axes metadata: ") + path
        Notifications.success("Radar", "Existing NPZ verified; no new copy created." if reused else "Radar NPZ exported.")
        return self.animation_status

    async def _simulate_async(self, animation=False, on_submitted=None):
        """Submit an immutable copy of current Studio state to the native solver."""
        from ..adapter.snapshot import snapshot_request, SUPPORTED_VIEWS
        if self._snapshot_running or self._export_running or self._replay_preparing or str(self.stream_status) in {"running", "paused"}:
            raise ValueError("A Radar solve/stream is already active; wait before starting another simulation.")
        self._snapshot_running = True
        if animation:
            # A new multi-frame measurement replaces the previous measurement.
            # Static snapshots use their own result identity and figure below,
            # so they can never destroy a valid Timeline replay.
            self._snapshot_result = False
            self._animation_result = False
            self._animation_view_generation += 1
            self._saved_result = None
            self.saved_result_loaded = False
            self._solver_result_handle = ""
            self._solver_run_id = ""
            self._solver_pending_run_id = ""
            self._clear_query_cache()
            self._radar = None
            self._signal = None
            self.signal_source = None
            self.signal_stream = None
            self.signal_figure.clear()
            self.snapshot_status = "Animation simulation in progress; not a frozen snapshot."
            self.animation_status = "Running native GPU simulation of baked Studio skin motion"
            self.animation_export_path = ""
            self.animation_frame_index = 0
        else:
            self._snapshot_result = False
            self._snapshot_solver_result_handle = ""
            self._snapshot_solver_run_id = ""
            self._snapshot_solver_pending_run_id = ""
            self.snapshot_figure.clear()
            self.snapshot_status = "Running one-frame frozen-point diagnostic; existing Replay is preserved."
        try:
            if animation:
                from ..adapter.animation import animation_request
                request = animation_request(self)
            else:
                request = snapshot_request(self)
            if str(self.view) not in SUPPORTED_VIEWS or bool(self.static_clutter_removal) or bool(self.show_cfar):
                raise ValueError("Choose Range Profile / Range Spectrum / Range Doppler and disable clutter removal / CFAR.")
            from witwin_server.features.solvers.scene_ref import make_scene_ref
            # No await while copying live values: nested keyframe lists must
            # not alias author state once the worker starts JSON serialization.
            scene_ref = copy.deepcopy(make_scene_ref(self.scene))
            def _submitted(run_id):
                if animation:
                    self._solver_pending_run_id = str(run_id)
                else:
                    self._snapshot_solver_pending_run_id = str(run_id)
                if on_submitted is not None:
                    on_submitted(str(run_id))

            solve_task = asyncio.create_task(asyncio.to_thread(
                api.solvers.solve,
                "witwin.radar.simulate",
                scene=scene_ref,
                config=request,
                on_submitted=_submitted,
            ))
            completion_won_cancel_race = False
            try:
                run = await asyncio.shield(solve_task)
            except asyncio.CancelledError:
                # Cancelling the asyncio waiter does not stop the solver thread.
                # The host publishes this request's immutable run id before it
                # waits for the serialized solver slot. Never infer ownership
                # from the host's global active run, which may belong to a
                # different queued request.
                # If native completion wins the race, keep its result instead of
                # orphaning a valid result handle.
                native_run_id = str(
                    self._solver_pending_run_id if animation
                    else self._snapshot_solver_pending_run_id
                )
                cancel_sent = False
                loop = asyncio.get_running_loop()
                discovery_deadline = loop.time() + 5.0
                while not solve_task.done() and not native_run_id and loop.time() < discovery_deadline:
                    try:
                        native_run_id = str(
                            self._solver_pending_run_id if animation
                            else self._snapshot_solver_pending_run_id
                        )
                    except asyncio.CancelledError:
                        continue
                    except Exception as status_error:  # noqa: BLE001 - retain native task ownership
                        logger.warning(f"Could not discover native Radar run for cancellation: {status_error}")
                        break
                    if not native_run_id and not solve_task.done():
                        await asyncio.sleep(0.01)
                if native_run_id and not solve_task.done():
                    try:
                        cancel_sent = bool(await asyncio.shield(asyncio.to_thread(
                            api.solvers.cancel,
                            native_run_id,
                            "Studio Radar animation request cancelled",
                        )))
                    except asyncio.CancelledError:
                        # A repeated UI/tool cancellation must not abandon the
                        # native run while its first cancellation is in flight.
                        cancel_sent = True
                    except Exception as cancel_error:  # noqa: BLE001 - solve result remains authoritative
                        logger.warning(f"Native Radar cancellation failed: {cancel_error}")
                while not solve_task.done():
                    try:
                        await asyncio.shield(solve_task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if solve_task.cancelled():
                    raise
                run = solve_task.result()
                if getattr(run, "status", None) == "succeeded":
                    completion_won_cancel_race = True
                else:
                    if animation:
                        self._animation_result = False
                    else:
                        self._snapshot_result = False
                    if self._scene_is_live():
                        suffix = " Native solver cancellation was sent." if cancel_sent else ""
                        self.animation_status = "Animation task cancelled; no result displayed." + suffix
                        self.snapshot_status = "Solve task cancelled; no result displayed." + suffix
                    raise asyncio.CancelledError
            if not self._scene_is_live():
                return "Scene closed or replaced during the solve; result not attached to another scene."
            if run.status != "succeeded":
                raise RuntimeError((run.error or {}).get("message") or "Snapshot solve failed")
            handle = (run.outputs or {}).get("resultHandle")
            if not handle:
                raise RuntimeError("Snapshot solver returned no result handle")
            if animation:
                self._solver_result_handle = str(handle)
                self._solver_run_id = run.run_id
                self._solver_pending_run_id = ""
                self._snapshot_result = False
                self._animation_result = True
                count = round(request['duration_s'] * request['fps'])
                self.animation_status = (
                    f"GPU animation complete: {count} real frames, {request['duration_s']:g}s at "
                    f"{request['fps']:g} FPS. Uncalibrated skin sites; LOS only."
                )
                if completion_won_cancel_race:
                    self.animation_status += " Native completion won the cancellation race; result preserved."
                self.snapshot_status = "Animation result active, not a frozen snapshot."
                await self._refresh_animation_view()
                # A multi-frame measurement is not complete from a user's
                # perspective until its replay asset is bound to this scene.
                # Keep the verified native result if local replay publication
                # fails, but make that failure explicit instead of silently
                # leaving a static preview that appears replayable.
                binary_assets = getattr(getattr(api, "server", None), "handlers", {}).get("binary_assets")
                replay_error_message = None
                replay_completion_status = None
                if binary_assets is not None and "view failed" not in self.animation_status:
                    try:
                        replay_completion_status = await self.prepare_synchronized_replay()
                    except Exception as replay_error:  # noqa: BLE001 - native result remains exportable
                        replay_error_message = str(replay_error)
                elif binary_assets is None:
                    replay_error_message = "Studio binary asset service is unavailable."
                if "view failed" not in self.animation_status:
                    metadata = dict(getattr(self, "_last_animation_view_metadata", {}) or {})
                    topology = dict(metadata.get("topology_preflight") or {})
                    declared = int(topology.get("declared_site_count", 0))
                    active = int(topology.get("active_site_count", 0))
                    if declared > 0 and 0 < active <= declared:
                        coverage = 100.0 * active / declared
                        quality = str(topology.get("visibility_quality") or "unknown")
                        self.animation_status = (
                            f"GPU animation complete: {count} real frames, {request['duration_s']:g}s at "
                            f"{request['fps']:g} FPS. Interval-visible sites {active}/{declared} "
                            f"({coverage:.1f}%, {quality}); uncalibrated RCS; LOS only."
                        )
                        if completion_won_cancel_race:
                            self.animation_status += (
                                " Native completion won the cancellation race; result preserved."
                            )
                    if replay_error_message:
                        self.animation_status += f" Replay unavailable: {replay_error_message}"
                        Notifications.warning("Radar", self.animation_status)
                    else:
                        self.animation_status += f" {replay_completion_status or self.replay_status}"
                        Notifications.success("Radar", self.animation_status)
                return self.animation_status
            self._snapshot_solver_result_handle = str(handle)
            self._snapshot_solver_run_id = run.run_id
            self._snapshot_solver_pending_run_id = ""
            self._snapshot_result = True
            from ..adapter.snapshot import show_snapshot
            show_snapshot(self)
            self.snapshot_status = f"GPU snapshot complete, t={request['time_s']:.3f}s. One RCS point; velocity frozen to zero."
            Notifications.success("Radar", "Native GPU snapshot complete (not animated-cat simulation)")
            return self.snapshot_status
        except Exception as exc:
            self._snapshot_result = False
            if self._scene_is_live():
                self.snapshot_figure.clear()
                self.snapshot_status = f"Snapshot failed: {exc}"
                if animation:
                    self._animation_result = False
                    self._solver_result_handle = ""
                    self._solver_run_id = ""
                    self._solver_pending_run_id = ""
                    self._clear_query_cache()
                    self.signal_figure.clear()
                    self.animation_status = f"Animation failed: {exc}"
                else:
                    self._snapshot_solver_result_handle = ""
                    self._snapshot_solver_run_id = ""
                    self._snapshot_solver_pending_run_id = ""
            logger.error("Radar snapshot failed", exc_info=True)
            raise
        finally:
            self._snapshot_running = False

    @button(display_name="Start Stream", group=_SOLVE)
    def start_stream(self):
        """Start solver-side realtime stream production."""
        self._require_legacy_solver()
        logger.info("=== Radar Start Stream clicked ===")
        if self.scene is None or self.owner is None:
            logger.warning("Radar stream start ignored: component is not attached to a scene")
            return "Radar must be attached to a scene"
        self._ensure_live_state()
        self._apply_live_latency_defaults()
        with self._live_lock:
            if self._live_thread is not None and self._live_thread.is_alive():
                if self._live_stop_event.is_set():
                    logger.info("Radar stream start ignored: existing stream is stopping")
                    return "Stream stopping"
                try:
                    api.solvers.call(
                        "witwin.radar.simulate",
                        "live_resume",
                        params={"session_id": self._live_session_id or self._stream_id()},
                        timeout=5,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.stream_status = "error"
                    self._mark_stream_error(str(exc))
                    logger.warning(f"Radar stream resume failed: {exc}")
                    return f"Stream failed: {exc}"
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
            self._clear_query_cache()
            self.stream_status = "running"
            self._open_signal_stream(self._stream_channel_descriptors(), status="active")
            signature = self._live_scene_signature()
            self._live_last_signature = signature
            self._live_last_scene_payload_signature = self._live_scene_payload_signature()
            params = self._live_solver_params(signature)
            logger.info(
                "Radar live start request: "
                f"session={params['session_id']} channels={params['channels']} "
                f"max_fps={params['max_fps']:.2f} scene={self._live_scene_summary()}"
            )
            try:
                response = api.solvers.call(
                    "witwin.radar.simulate",
                    "live_start",
                    params=params,
                    scene=self.scene,
                    timeout=10,
                )
            except Exception as exc:  # noqa: BLE001
                self.stream_status = "error"
                self._mark_stream_error(str(exc))
                logger.warning(f"Radar live start failed: {exc}")
                return f"Stream failed: {exc}"
            self._live_session_id = str((response or {}).get("sessionId") or params["session_id"])
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
        try:
            api.solvers.call(
                "witwin.radar.simulate",
                "live_pause",
                params={"session_id": self._live_session_id or self._stream_id()},
                timeout=5,
            )
        except Exception as exc:  # noqa: BLE001 - local pause still prevents update traffic
            logger.warning(f"Radar stream pause control failed: {exc}")
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
        thread = None
        session_id = ""
        with self._live_lock:
            self._live_generation += 1
            self._live_stop_event.set()
            self._live_pause_event.clear()
            self.stream_status = "stopped"
            self._clear_query_cache()
            thread = self._live_thread
            session_id = self._live_session_id or self._signal_stream_id
            self._live_thread = None
            self._live_session_id = ""
        if session_id:
            try:
                api.solvers.call(
                    "witwin.radar.simulate",
                    "live_stop",
                    params={"session_id": session_id},
                    timeout=5,
                )
            except Exception as exc:  # noqa: BLE001 - stop should still close local stream
                logger.debug(f"Radar live_stop skipped: {exc}")
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._live_lock:
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

    def _scene_is_live(self):
        scene, owner = self.scene, self.owner
        if (scene is None or owner is None or scene.get_object(owner.id) is not owner
                or owner.get_component("Radar") is not self):
            return False
        try:
            return api.server.scenes.get(scene.scene_id) is scene
        except RuntimeError:
            return False

    @button(display_name="Prepare Synchronized Replay", group=_ANIMATION)
    async def prepare_synchronized_replay(self):
        """Prepare numeric data once; frontend follows the existing scene clock."""
        from ..adapter.replay import build_recording, publish_recording
        # _simulate_async marks the operation running until its automatically
        # published Replay is complete. Once the native animation result is
        # attached, Replay preparation is the final phase of that same job.
        if (self._replay_preparing or self._export_running
                or (self._snapshot_running and not self._animation_result)):
            raise ValueError("Wait for the current simulation/replay preparation to finish.")
        if not self._scene_is_live():
            raise ValueError("Open the source scene before preparing replay.")
        if bool(self.static_clutter_removal) or bool(self.show_cfar):
            raise ValueError("Replay does not apply clutter removal or CFAR; turn these controls off.")
        saved = self._saved_result
        identity = (self._solver_result_handle, self._solver_run_id, saved)
        antennas = (int(self.tx_index), int(self.rx_index))
        if saved is None and not self._animation_result:
            raise ValueError("Complete Simulate & Prepare Replay or Load Saved Result first.")
        self._replay_preparing = True
        self.replay_status = "Preparing native numeric RP/RD data (no images)"
        def prepare():
            if saved is not None:
                return build_recording(saved, *antennas)
            payload = self._query_result("animation_replay", {"tx": antennas[0], "rx": antennas[1]})
            data = api.solvers.read_result_ref("witwin.radar.simulate", payload["reference"])
            return payload["recording"], data
        def still_current():
            return (self._scene_is_live() and self._solver_result_handle == identity[0]
                    and self._solver_run_id == identity[1] and self._saved_result is saved
                    and antennas == (int(self.tx_index), int(self.rx_index)))
        task = asyncio.create_task(asyncio.to_thread(prepare))
        try:
            record, data = await asyncio.shield(task)
            if not still_current():
                return "Source changed; old replay was not published."
            if bool(self.static_clutter_removal) or bool(self.show_cfar):
                raise ValueError("Clutter removal or CFAR was enabled during preparation; turn it off before replay.")
            self._animation_view_generation += 1
            self.replay_status = publish_recording(self, api, record, data)
            return self.replay_status
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if task.done() and not task.cancelled():
                task.exception()
            raise
        except Exception as exc:
            if still_current():
                self.replay_status = f"Replay preparation failed: {exc}"
            logger.error("Replay preparation failed", exc_info=True)
            raise
        finally:
            self._replay_preparing = False

    async def _refresh_animation_view(self):
        if not self._scene_is_live():
            return "Scene closed or replaced; animation view not published."
        from ..adapter.animation import animation_view_params, apply_animation_view
        self._animation_view_generation += 1
        generation = self._animation_view_generation
        identity = (self._solver_result_handle, self._solver_run_id)
        params = animation_view_params(self)
        self.signal_figure.clear().title("Loading recorded animation frame")

        def still_current():
            return (self._scene_is_live() and self._animation_result and generation == self._animation_view_generation
                    and identity == (self._solver_result_handle, self._solver_run_id)
                    and params == animation_view_params(self))

        try:
            payload = await asyncio.to_thread(self._query_result, "animation_view", params)
            if still_current():
                return apply_animation_view(self, payload)
            return "Superseded animation view"
        except Exception as exc:
            if still_current():
                self.signal_figure.clear().title("Animation view unavailable: check frame / supported settings")
                self.animation_status = f"Animation view failed: {exc}"
                logger.error("Animation view failed", exc_info=True)
            return "Animation view unavailable"

    def update_view(self):
        """Render the selected view from the last solved signal."""
        if self._animation_result:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self._refresh_animation_view())
            asyncio.create_task(self._refresh_animation_view())
            return "Loading recorded animation frame"
        if self._snapshot_result:
            from ..adapter.snapshot import show_snapshot
            try:
                return show_snapshot(self)
            except Exception:
                self.signal_figure.clear().title("Snapshot view unavailable: check indices / supported settings")
                raise
        if self._saved_result is not None:
            from ..adapter.saved_result import show_saved_result
            try:
                return show_saved_result(self)
            except Exception:
                self.signal_figure.clear().title("Saved result view unavailable: check indices / supported view")
                raise
        if not self._solver_result_handle and self._signal is None:
            logger.info("Update View: no signal yet — run Simulate first")
            return "Simulate first"
        view = str(self.view)
        if view in {"range_profile", "range_spectrum"}:
            return "Load a metadata-bearing saved result first"
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
        self._require_legacy_solver()
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
        self._clear_query_cache()
        self._register_signal_timeline_dataset()
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
        self._clear_query_cache()
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
            payload = self._query_result("range_doppler", self._rd_query_params())
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

    def _source_id(self):
        if self._signal_source_id:
            return self._signal_source_id
        owner_id = getattr(getattr(self, "_owner", None), "id", None) or hex(id(self))[2:]
        self._signal_source_id = f"radar.{owner_id}.result"
        return self._signal_source_id

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
        ref_channel = self._stream_ref_channel()
        ref = stream.ref(ref_channel)
        ref["defaultChannel"] = ref_channel
        ref["channelId"] = ref_channel
        ref["viewport"] = {
            "enabled": True,
            "channelId": "pc",
            "kind": "points",
            "maxFps": min(max(float(self.stream_max_fps), 0.1), 30.0),
            "maxPoints": 100000,
            "pointSize": 0.05,
            "color": "#30c0ff",
        }
        self.signal_stream = ref
        self._register_signal_stream_source(channels, ref_channel, status=status, viewport=ref["viewport"])
        return stream

    def _data_source_ref(self, mode, default_channel, **extra):
        ref = {
            "sourceId": self._source_id(),
            "mode": mode,
            "kind": "radar.result",
            "defaultChannel": default_channel,
            "channelId": default_channel,
            **extra,
        }
        if mode == "stream":
            ref["volatile"] = True
        return ref

    def _register_signal_stream_source(self, channels, default_channel, *, status="active", viewport=None):
        data_sources = getattr(api, "data_sources", None)
        if data_sources is None:
            return
        descriptor = {
            "sourceId": self._source_id(),
            "mode": "stream",
            "kind": "radar.result",
            "streamId": self._stream_id(),
            "label": "Radar Result",
            "owner": self._stream_owner(),
            "retention": {"mode": "ring", "historyLength": int(self.stream_history_length)},
            "metadata": {
                "status": status,
                "radarStreamId": self._stream_id(),
                "view": str(self.view),
            },
            "channels": channels,
            "defaultChannel": default_channel,
            "volatile": True,
        }
        try:
            data_sources.register(descriptor)
            self.signal_source = self._data_source_ref(
                "stream",
                default_channel,
                streamId=self._stream_id(),
                viewport=viewport or {},
            )
        except Exception as exc:  # noqa: BLE001 - data source registration must not break legacy stream preview
            logger.debug(f"Radar data source registration skipped: {exc}")

    def _register_signal_timeline_dataset(self):
        if self._frames is None:
            return
        data_sources = getattr(api, "data_sources", None)
        if data_sources is None:
            return
        try:
            frame_count = int(self._frames.shape[0])
        except Exception:
            return
        if frame_count <= 0:
            return
        times = [index / max(float(self.frame_rate), 1e-6) for index in range(frame_count)]
        dataset_id = self._source_id()
        frames = []
        for index, time_s in enumerate(times):
            frames.extend(self._timeline_frame_refs(index, time_s, dataset_id))
        descriptor = {
            "datasetId": dataset_id,
            "kind": "radar.result",
            "producerId": self._source_id(),
            "dependencyKey": self._live_scene_signature(),
            "label": "Radar Timeline Result",
            "defaultChannel": self._stream_ref_channel(),
            "timebase": {"unit": "seconds", "times": times, "fps": float(self.frame_rate)},
            "channels": self._stream_channel_descriptors(),
            "frames": frames,
            "interpolation": "nearest",
            "retention": "runtime",
            "metadata": {
                "timelineSource": str(self.timeline_source),
                "frameCount": frame_count,
            },
        }
        try:
            data_sources.register_timeline_dataset(descriptor)
            default_channel = self._stream_ref_channel()
            self.signal_source = self._data_source_ref(
                "timeline",
                default_channel,
                datasetId=dataset_id,
                defaultTime=float(getattr(self, "frame_index", 0)) / max(float(self.frame_rate), 1e-6),
                volatile=True,
            )
        except Exception as exc:  # noqa: BLE001 - timeline data sources are additive during migration
            logger.debug(f"Radar timeline data source registration skipped: {exc}")

    def _timeline_frame_refs(self, index, time_s, dataset_id):
        signal_shape = list(getattr(self._frames[index], "shape", []))
        return [
            {
                "time": float(time_s),
                "channelId": "raw",
                "frame": {
                    "sourceId": dataset_id,
                    "channelId": "raw",
                    "time": float(time_s),
                    "dtype": "complex64",
                    "shape": signal_shape,
                    "semantic": "radar.raw",
                    "metadata": {"frameIndex": int(index)},
                    "status": "ready",
                },
            },
            {
                "time": float(time_s),
                "channelId": "rd",
                "frame": {
                    "sourceId": dataset_id,
                    "channelId": "rd",
                    "time": float(time_s),
                    "dtype": "float32",
                    "shape": [int(self.num_doppler_bins), int(self.num_range_bins)],
                    "semantic": "radar.range_doppler",
                    "metadata": {"frameIndex": int(index), "tx": self._tx(), "rx": self._rx()},
                    "status": "pending",
                },
            },
            {
                "time": float(time_s),
                "channelId": "pc",
                "frame": {
                    "sourceId": dataset_id,
                    "channelId": "pc",
                    "time": float(time_s),
                    "dtype": "float32",
                    "shape": [0, 6],
                    "semantic": "pointcloud_xyz",
                    "metadata": {"frameIndex": int(index), "detector": str(self.detector)},
                    "status": "pending",
                },
            },
        ]

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
             "semantic": "time_series", "label": "Raw IQ",
             "plot": {
                 "kind": "line",
                 "x": {"mode": "index", "label": "ADC sample"},
                 "y": {"label": "Amplitude"},
                 "complex": "real_imag",
                 "series": [
                     {"component": "real", "label": "Real", "color": "#ff9500"},
                     {"component": "imag", "label": "Imag", "color": "#00aaff"},
                 ],
             }},
            {"channelId": "rd", "dtype": "float32", "shape": ["doppler", "range"],
             "semantic": "heatmap", "label": "Range-Doppler",
             "plot": {
                 "kind": "heatmap",
                 "x": {"label": "Range bin"},
                 "y": {"label": "Doppler bin"},
                 "colormap": "imshow",
                 "range": {"mode": "full"},
             },
             "presentation": {"colormap": "imshow", "displayRangeMode": "full"}},
            {"channelId": "pc", "dtype": "float32", "shape": ["points", 6],
             "semantic": "pointcloud_xyz", "label": "Point cloud",
             "plot": {
                 "kind": "points",
                 "stride": 6,
                 "axes": ["x", "y", "z"],
                 "colorBy": "intensity",
             }},
        ]

    def _selected_stream_channels(self, default=("raw", "rd", "pc")):
        allowed = ("raw", "rd", "pc")
        selected = {
            part.strip().lower()
            for part in str(self.stream_channels or "").split(",")
            if part.strip()
        }
        selected = tuple(channel for channel in allowed if channel in selected)
        view_channel = self._stream_channel_for_view()
        if view_channel:
            selected_set = set(selected or default)
            selected_set.add(view_channel)
            selected = tuple(channel for channel in allowed if channel in selected_set)
        return tuple(selected or default)

    def _stream_channel_for_view(self):
        return self._VIEW_STREAM_CHANNELS.get(str(self.view))

    def _stream_ref_channel(self):
        return self._stream_channel_for_view() or self._selected_stream_channels(default=("rd",))[0]

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

    def _apply_live_latency_defaults(self):
        selected = self._selected_stream_channels(default=("raw", "rd", "pc"))
        if bool(self.stream_on_change_only) or selected != ("raw", "rd", "pc"):
            return
        logger.info("Radar stream applying latency defaults: stream_on_change_only=True channels=rd")
        self.stream_on_change_only = True
        self.stream_channels = "rd"

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
            interval = 1.0 / min(max(0.1, float(self.stream_max_fps)), 30.0)
            started = time.monotonic()
            signature = self._live_scene_signature()
            scene_payload_signature = self._live_scene_payload_signature()
            if signature != self._live_last_signature:
                include_scene = scene_payload_signature != self._live_last_scene_payload_signature
                logger.info(f"Radar live loop updating solver: generation={generation}")
                ok = self._send_live_update(generation, signature, include_scene=include_scene)
                if not ok:
                    self.stream_status = "error"
                    self._update_stream_status("error")
                    time.sleep(max(0.25, interval))
                else:
                    self._live_last_signature = signature
                    self._live_last_scene_payload_signature = scene_payload_signature
            elapsed = time.monotonic() - started
            time.sleep(max(0.01, interval - elapsed))
        logger.info(f"Radar live loop exited: generation={generation}")

    def _live_generation_active(self, generation):
        return generation == self._live_generation

    def _send_live_update(self, generation, signature, *, include_scene=True):
        try:
            if not self._live_generation_active(generation):
                return True
            if self._live_stop_event is not None and self._live_stop_event.is_set():
                return True
            params = self._live_solver_params(signature)
            logger.info(
                "Radar live update request: "
                f"session={params['session_id']} position={params['config']['sensor']['position']} "
                f"include_scene={bool(include_scene)}"
            )
            kwargs = {"timeout": 5}
            if include_scene:
                kwargs["scene"] = self.scene
            api.solvers.call(
                "witwin.radar.simulate",
                "live_update",
                params=params,
                **kwargs,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - background control thread must survive one failed update
            logger.warning(f"Radar live update failed: {exc}")
            self._mark_stream_error(str(exc))
            return False

    def _solve_once_for_stream(self, generation, signature=None):
        try:
            if not self._live_generation_active(generation):
                return True
            if self._live_stop_event is not None and self._live_stop_event.is_set():
                return True
            spec = SensorSpec.from_component(self)
            t0 = self._live_t0()
            solve_signature = signature if signature is not None else self._live_scene_signature()
            logger.info(
                "Radar live solve request: "
                f"backend={spec.backend} device={spec.device} position={spec.position} t0={t0:.6f}"
            )
            logger.info(f"Radar live scene state: {self._live_scene_summary()}")
            run = api.solvers.solve(
                "witwin.radar.simulate",
                scene=self.scene,
                config=self._solver_config(
                    spec,
                    t0_override=t0,
                    live_session_id=self._stream_id(),
                    live_signature=solve_signature,
                ),
            )
            logger.info(f"Radar live solve returned: status={run.status} run_id={run.run_id}")
            if not self._live_generation_active(generation):
                return True
            if run.status != "succeeded":
                message = (run.error or {}).get("message") or "simulate failed"
                self._mark_stream_error(message)
                return False
            if self._live_scene_signature() != solve_signature:
                logger.info("Radar live solve result is stale; skipping preview and stream publish")
                return True
            handle = (run.outputs or {}).get("resultHandle")
            if not handle:
                self._mark_stream_error("Solver returned no result handle")
                return False
            self._solver_result_handle = str(handle)
            self._solver_run_id = run.run_id
            self._clear_query_cache()
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

    def _live_scene_payload_signature(self):
        """Signature for data that requires resending the studio scene to the solver."""
        scene = self.scene
        items = [("radar_component", self._component_signature())]
        if scene is not None:
            related_ids = self._skinned_mesh_payload_object_ids(scene)
            for obj_id, obj in sorted(scene.objects.items()):
                has_geometry = (
                    obj.get_component("PlatformGeometry") is not None
                    or obj.get_component("Mesh") is not None
                )
                is_related = obj_id in related_ids
                radar_component = obj.get_component("Radar")
                if obj is self.owner and not has_geometry and not is_related:
                    continue

                row = [obj_id]
                if has_geometry or is_related:
                    transform = obj.get_component("Transform")
                    row.extend([
                        self._signature_value(getattr(transform, "position", None) if transform is not None else None),
                        self._signature_value(getattr(transform, "rotation", None) if transform is not None else None),
                        self._signature_value(getattr(transform, "scale", None) if transform is not None else None),
                        bool(getattr(obj, "visible", True)),
                    ])
                if radar_component is not None and radar_component is not self:
                    row.append((
                        "radar_component",
                        self._component_fields_signature(
                            radar_component,
                            ignored={"signal_stream", "signal_figure", "stream_status"},
                        ),
                    ))
                if len(row) > 1:
                    items.append(tuple(row))
        return json.dumps(items, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _skinned_mesh_payload_object_ids(scene):
        ids = set()
        for obj in getattr(scene, "objects", {}).values():
            skinned = obj.get_component("SkinnedMesh")
            bone_ids = getattr(skinned, "bone_ids", None) if skinned is not None else None
            if callable(bone_ids):
                bone_ids = bone_ids()
            ids.update(str(bone_id) for bone_id in (bone_ids or []) if bone_id)
        return ids

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
        return self._component_fields_signature(
            self,
            ignored={"signal_stream", "signal_figure", "stream_status"},
        )

    @staticmethod
    def _component_fields_signature(component, *, ignored=None):
        ignored = set(ignored or ())
        values = {}
        for name in sorted(getattr(component, "_fields_meta", {})):
            if name in ignored:
                continue
            values[name] = RadarComponent._signature_value(getattr(component, name))
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
        sample = np.ascontiguousarray(sig[self._tx(), self._rx(), 0].astype(np.complex64, copy=False))
        iq = np.empty(sample.size * 2, dtype=np.float32)
        iq[0::2] = sample.real.astype(np.float32, copy=False)
        iq[1::2] = sample.imag.astype(np.float32, copy=False)
        stream.publish("raw", iq, metadata={
            "dtype": "complex64",
            "shape": [int(sample.size)],
            "tx": int(self._tx()),
            "rx": int(self._rx()),
            "chirp": 0,
        })
        amp = np.abs(sample)
        logger.info(
            "Radar stream raw frame: "
            f"shape={[int(sample.size)]} mean_amp={float(amp.mean()):.6e} max_amp={float(amp.max()):.6e}"
        )

    def _publish_rd_stream(self, stream):
        if self._solver_result_handle:
            payload = self._query_result("range_doppler", self._rd_query_params(show_cfar=False))
            mag_db_value = payload.get("mag_db")
            mag_db = np.asarray([] if mag_db_value is None else mag_db_value, dtype=np.float32)
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
            "colormap": "imshow",
            "displayRangeMode": "full",
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
            points = payload.get("points")
            pc = np.asarray([] if points is None else points, dtype=np.float32)
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

    def _live_solver_params(self, signature, *, spec=None, t0_override=None):
        spec = spec or SensorSpec.from_component(self)
        session_id = self._stream_id()
        channels = list(self._selected_stream_channels(default=("raw", "rd", "pc")))
        max_fps = min(max(float(self.stream_max_fps), 0.1), 30.0)
        config = self._solver_config(
            spec,
            t0_override=t0_override,
            live_session_id=session_id,
            live_signature=signature,
        )
        return {
            "session_id": session_id,
            "stream_id": session_id,
            "channels": channels,
            "max_fps": max_fps,
            "history_length": int(self.stream_history_length),
            "config": config,
        }

    def _solver_config(self, spec: SensorSpec, *, t0_override=None, live_session_id=None,
                       live_signature=None) -> dict:
        config = {
            "sensor": asdict(spec),
            "tracer": asdict(TracerSpec.from_component(self)),
            "motion_sampling": str(self.motion_sampling),
            "t0": num(self.t0) if t0_override is None else num(t0_override),
        }
        if live_session_id and live_signature:
            config["live"] = {
                "session_id": str(live_session_id),
                "stream_id": str(live_session_id),
                "signature": str(live_signature),
                "scene_payload_signature": str(self._live_scene_payload_signature()),
                "channels": list(self._selected_stream_channels(default=("raw", "rd", "pc"))),
                "max_fps": min(max(float(self.stream_max_fps), 0.1), 30.0),
                "stream_on_change_only": bool(self.stream_on_change_only),
                "motion_sampling": "per_frame",
                "tx": int(self.tx_index),
                "rx": int(self.rx_index),
                "static_clutter_removal": bool(self.static_clutter_removal),
                "show_cfar": bool(self.show_cfar),
                "detector": str(self.detector),
                "guard": list(self._guard()),
                "training": list(self._training()),
                "pfa": num(self.pfa),
                "energy_top_k": int(self.energy_top_k),
            }
        return config

    def _query_result(
        self,
        op: str,
        params: dict,
        *,
        result_handle: str | None = None,
        run_id: str | None = None,
    ):
        """Query one immutable result identity without changing the active Replay."""
        handle = str(result_handle if result_handle is not None else self._solver_result_handle)
        identity_run_id = str(run_id if run_id is not None else self._solver_run_id)
        cache = getattr(self, "_query_cache", None)
        if cache is None:
            cache = {}
            self._query_cache = cache
        cache_key = (
            handle,
            identity_run_id,
            str(op),
            json.dumps(params, sort_keys=True, separators=(",", ":"), default=str),
        )
        if cache_key in cache:
            logger.info(f"Radar solver query cache hit: op={op}")
            return cache[cache_key]
        response = api.solvers.query(
            "witwin.radar.simulate",
            handle,
            op,
            params,
            run_id=identity_run_id or None,
            **({"timeout": 180.0} if op in {"animation_export", "animation_replay", "animation_manifest"} else {}),
        )
        data = response.get("data") or {}
        if len(cache) >= 8:
            cache.pop(next(iter(cache)))
        cache[cache_key] = data
        return data

    def _clear_query_cache(self):
        self._query_cache = {}

    def _rd_query_params(self, *, tx=None, rx=None, static_clutter_removal=None, show_cfar=None):
        show = bool(self.show_cfar) if show_cfar is None else bool(show_cfar)
        params = {
            "tx": int(self.tx_index if tx is None else tx),
            "rx": int(self.rx_index if rx is None else rx),
            "static_clutter_removal": (
                bool(self.static_clutter_removal)
                if static_clutter_removal is None else bool(static_clutter_removal)
            ),
            "show_cfar": show,
        }
        if show:
            params.update({
                "guard": list(self._guard()),
                "training": list(self._training()),
                "pfa": num(self.pfa),
            })
        return params
