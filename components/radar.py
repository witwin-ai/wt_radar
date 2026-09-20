"""Studio controls for Radar 0.4 snapshots, baked animation, and replay.

The sensor pose comes from the owner object's world Transform: local ``-Z`` is
forward and local ``+Y`` is up. Historical Radar 0.3 fields remain serialized
only where needed to open old projects; they are not presented as runnable UI.
"""
import asyncio
import copy
import json

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
from .radar_ui import present_radar_controls

logger = get_logger("Radar")

_CAT = "Simulation/Radar"
_VIEWS = ["range_profile", "range_spectrum", "range_doppler"]

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


@component(name="Radar", display_name="Radar", category=_CAT)
class RadarComponent(Component):
    """Unified Radar Settings component — FMCW config + sensor + sub-configs + solve + timeline."""

    scene_dependency_roles = frozenset({"simulation_sensor"})
    scene_dependency_dataset_kinds = frozenset({"radar.result"})

    def to_dict(self):
        return present_radar_controls(super().to_dict())

    @classmethod
    def get_definition(cls):
        return present_radar_controls(super().get_definition())

    _signal = None   # runtime MIMO cube (TX, RX, chirps, ADC); never serialized
    _radar = None    # runtime Radar
    _solver_result_handle = ""
    _solver_run_id = ""
    _snapshot_solver_result_handle = ""
    _snapshot_solver_run_id = ""
    _snapshot_solver_pending_run_id = ""
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
    # Serialized only so old project files round-trip without data loss. The
    # Radar 0.3 stream UI and execution methods have been removed.
    stream_max_fps = float_field(30.0, min=0.1, max=30.0, group=_SOLVE, hidden=True)
    stream_channels = string_field("rd", group=_SOLVE, hidden=True)
    stream_history_length = int_field(1, min=1, group=_SOLVE, hidden=True)
    stream_on_change_only = bool_field(True, group=_SOLVE, hidden=True)
    stream_status = string_field("stopped", readonly=True, transient=True, group=_SOLVE, hidden=True)
    post_processors = list_field(component_field(component_type="RadarPostProcessor"), default=[],
                                 group=_SOLVE,
                                 description="Extra post-processor views run after each solve")
    signal_source = data_source_ref_field(
        default_channel="rd",
        mode="stream",
        kind="radar.result",
        group=_ANIMATION,
        hide_label=True,
        show_header=False,
        hide_if="saved_result_loaded",
    )
    signal_figure = figure(title="Radar Replay", group=_ANIMATION)

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
    snapshot_status = string_field("Not run. Static Snapshot freezes the scene at the measurement start time and includes one reflection.",
                                   readonly=True, transient=True, group=_SNAPSHOT)
    snapshot_figure = figure(title="Static diagnostic snapshot (one frame)", group=_SNAPSHOT)
    animation_duration_s = float_field(5.0, min=0.01, group=_ANIMATION,
                                       description="Exact baked timeline duration (0..30 seconds), starting at t0.")
    animation_fps = float_field(10.0, min=1.0, group=_ANIMATION,
                                description="Actual independent GPU measurements per second (1..30), not repeated video frames.")
    target_multipath = bool_field(
        False,
        group=_ANIMATION,
        description=(
            "High-cost target-related room multipath (wall-to-target or target-to-wall). "
            "Off matches the original WT Radar default; static one-bounce room background remains enabled."
        ),
    )
    target_motion_sampling = string_field(
        "chirp",
        options=["chirp", "adaptive"],
        enum_toggle=True,
        group=_ANIMATION,
        description=(
            "Chirp evaluates every TX chirp and is the measured default for Studio runs. "
            "Adaptive uses official error-controlled FMCW probes for comparison; "
            "path topology is still rediscovered per Studio frame."
        ),
    )
    target_site_budget = int_field(
        64,
        min=1,
        max=64,
        group=_ANIMATION,
        description=(
            "Maximum moving-target surface sites. Values below the available bone count "
            "use deterministic bind-pose spatial coverage; 64 preserves all available sites."
        ),
    )
    animation_frame_index = int_field(0, min=0, group=_ANIMATION,
                                      description="Recorded result frame. Does not seek or modify the editor timeline.")
    animation_status = string_field("Not run. Uses LOS target returns plus static one-bounce room background.",
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

    # Hidden serialized compatibility for pre-0.4 projects. These fields no
    # longer create controls or drive simulation.
    frame_rate = float_field(30.0, min=1.0, group=_TIMELINE, hidden=True)
    timeline_source = string_field("pointcloud_sequence", group=_TIMELINE, hidden=True)
    pointcloud_path = string_field("", group=_TIMELINE, hidden=True)
    motion_path = string_field("", group=_TIMELINE, hidden=True)
    velocity_corrected = bool_field(True, group=_TIMELINE, hidden=True)
    frame_index = int_field(0, min=0, group=_TIMELINE, hidden=True)

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
        if not self._has_preview_result():
            return
        self._auto_refreshing_view = True
        try:
            self.update_view()
        finally:
            self._auto_refreshing_view = False

    def _has_preview_result(self):
        return self._saved_result is not None or bool(getattr(self, "_solver_result_handle", "")) or getattr(self, "_signal", None) is not None

    @button(display_name="Load Saved Result", group=_SAVED)
    def load_saved_result(self):
        """Load exported producer metadata and reuse the existing figure UI."""
        from ..adapter.saved_result import SavedRadarResult
        if self._snapshot_running or self._export_running or self._replay_preparing:
            raise ValueError("Wait for the running simulation or replay operation before loading a saved result.")
        self._saved_result = None
        self._last_animation_view_metadata = {}
        self._snapshot_result = False
        self._animation_result = False
        self._animation_view_generation += 1
        self.animation_export_path = ""
        self.saved_result_loaded = False
        self._signal = None
        self._radar = None
        self._solver_result_handle = ""
        self._clear_query_cache()
        self.signal_source = None
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
        if self._snapshot_running or self._export_running or self._replay_preparing:
            raise ValueError("A Radar simulation or replay operation is already active; wait before starting another one.")
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
                if request.get("input_fingerprint"):
                    self._last_result_input_fingerprint = str(request["input_fingerprint"])
                count = round(request['duration_s'] * request['fps'])
                target_path_label = (
                    "LOS plus target multipath" if request.get("target_multipath", False)
                    else "LOS-only target"
                )
                target_motion_label = f"{request.get('target_motion_sampling', 'chirp')} motion sampling"
                self.animation_status = (
                    f"GPU animation complete: {count} real frames, {request['duration_s']:g}s at "
                    f"{request['fps']:g} FPS. Uncalibrated skin sites; {target_path_label}; "
                    f"{target_motion_label}; static one-bounce room background."
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
                            f"({coverage:.1f}%, {quality}); uncalibrated RCS; {target_path_label}; "
                            f"{target_motion_label}; static one-bounce room background."
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
        logger.info("Update View: no Radar 0.4 result is attached")
        return "Run Static Snapshot, Simulate & Prepare Replay, or Load Saved Result first"

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
