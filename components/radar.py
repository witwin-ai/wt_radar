"""Unified Radar component — the single scene-level component for a radar simulation.

Replaces the nine separate Radar Settings components (RadarConfig + RadarSensor +
RadarTracer + the four sub-configs + RadarResult + RadarTimeline) with one component
exposing every setting through foldout groups. The marker the adapter keys off is now
``Radar`` (master plan §6.4); optional per-structure ``RadarMotion`` and the
polymorphic ``RadarPostProcessor`` hierarchy stay separate because they live on
other scene objects.

The grouping mirrors the old split so authors find familiar sections:
``Frequency / ADC / Frame / Bins / Antenna`` are the 17 ``RadarConfig`` core fields;
``Tracer`` is the ray-tracer + sensor optic (fov); the four
``Antenna Pattern / Noise / Polarization / Receiver`` foldouts are the optional
``RadarConfig`` sub-configs; ``Solve / View / Detector`` drive the in-component MIMO
view; ``Timeline`` generates multi-frame sequences.

The sensor **pose** (position + look direction + up) is read from the owner SceneObject's
``Transform`` at solve time — moving / rotating the Radar Settings object in the viewport
moves the simulated radar. The solver backend defaults to ``dirichlet`` on ``cuda`` and
is hidden from the UI; tests / scripts can still override ``backend`` / ``device`` /
``pad_factor`` on the component directly.
"""
import numpy as np

from witwin_server import Notifications
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
from ..adapter.solve import SensorSpec, SigProc, SolveRunner, TracerSpec

logger = get_logger("Radar")

_CAT = "Simulation/Radar"
_VIEWS = ["raw_signal", "range_doppler", "point_cloud", "music"]

# Foldout group ids (also used in string `show_if`/`hide_if` lookups by field name).
_FREQUENCY = "Frequency"
_ADC = "ADC"
_FRAME = "Frame"
_BINS = "Bins"
_ANTENNA = "Antenna"
_TRACER = "Tracer"
_PATTERN = "AntennaPattern"
_NOISE = "Noise"
_POLARIZATION = "Polarization"
_RECEIVER = "Receiver"
_SOLVE = "Solve"
_VIEW = "View"
_DETECTOR = "Detector"
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

    define_group(foldout_group(_FREQUENCY, display_name="Frequency / Power"))
    define_group(foldout_group(_ADC, display_name="ADC / Sampling"))
    define_group(foldout_group(_FRAME, display_name="Chirp / Frame"))
    define_group(foldout_group(_BINS, display_name="FFT Bins"))
    define_group(foldout_group(_ANTENNA, display_name="Antenna Geometry"))
    define_group(foldout_group(_TRACER, display_name="Ray Tracer"))
    define_group(foldout_group(_PATTERN, display_name="Antenna Pattern", collapsed=True))
    define_group(foldout_group(_NOISE, display_name="Noise Model", collapsed=True))
    define_group(foldout_group(_POLARIZATION, display_name="Polarization", collapsed=True))
    define_group(foldout_group(_RECEIVER, display_name="Receiver Chain", collapsed=True))
    define_group(foldout_group(_SOLVE, display_name="Solve"))
    define_group(foldout_group(_VIEW, display_name="View"))
    define_group(foldout_group(_DETECTOR, display_name="Detector / CFAR", collapsed=True))
    define_group(foldout_group(_TIMELINE, display_name="Timeline", collapsed=True))

    # --- Frequency / power (RadarConfig) ------------------------------------
    fc = float_field(77e9, min=1e9, max=300e9, units=FREQUENCY_UNITS, default_unit="GHz",
                     group=_FREQUENCY, description="Carrier / start frequency")
    slope = float_field(60.012, min=0.0, units=SLOPE_UNITS, default_unit="MHz/us",
                        group=_FREQUENCY, description="Chirp frequency slope")
    power = float_field(15.0, group=_FREQUENCY, description="TX power (dBm)")

    # --- ADC / sampling (RadarConfig) ---------------------------------------
    adc_samples = int_field(256, min=1, group=_ADC, description="ADC samples per chirp (fast time)")
    sample_rate = float_field(4400.0, min=0.0, units=SAMPLE_RATE_UNITS, default_unit="ksps",
                              group=_ADC, description="ADC sample rate")
    adc_start_time = float_field(6.0, min=0.0, units=TIME_UNITS, default_unit="us",
                                 group=_ADC, description="ADC start delay")

    # --- chirp / frame (RadarConfig) ----------------------------------------
    idle_time = float_field(7.0, min=0.0, units=TIME_UNITS, default_unit="us",
                            group=_FRAME, description="Idle time between chirps")
    ramp_end_time = float_field(58.0, min=0.0, units=TIME_UNITS, default_unit="us",
                                group=_FRAME, description="Active chirp ramp time")
    chirp_per_frame = int_field(128, min=1, group=_FRAME, description="Chirps per frame (slow time / Doppler)")
    frame_per_second = float_field(10.0, min=0.0, group=_FRAME, description="Frame rate (Hz)")

    # --- FFT bins (RadarConfig) ---------------------------------------------
    num_doppler_bins = int_field(128, min=1, group=_BINS, description="Doppler FFT bins")
    num_range_bins = int_field(256, min=1, group=_BINS, description="Range bins")
    num_angle_bins = int_field(64, min=1, group=_BINS, description="Angle FFT bins")

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
    use_default = bool_field(True, group=_PATTERN,
                             description="Use the default half-wave dipole pattern")
    pattern_kind = string_field("separable", options=["separable", "map"], enum_toggle=True,
                                hide_if="use_default", group=_PATTERN,
                                description="Separable per-axis cuts or a 2D gain map")
    x_angles_deg = list_field(float_field(0.0), default=[], hide_if="use_default", group=_PATTERN,
                              description="Azimuth angles (deg, strictly increasing, >=2)")
    y_angles_deg = list_field(float_field(0.0), default=[], hide_if="use_default", group=_PATTERN,
                              description="Elevation angles (deg, strictly increasing, >=2)")
    x_values = list_field(float_field(0.0), default=[], show_if=_SEPARABLE, group=_PATTERN,
                          description="Per-azimuth gain (>=0, len == x_angles)")
    y_values = list_field(float_field(0.0), default=[], show_if=_SEPARABLE, group=_PATTERN,
                          description="Per-elevation gain (>=0, len == y_angles)")
    values_2d_json = string_field("", widget="textarea", show_if=_MAP, group=_PATTERN,
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

    # --- solve / view / detector (RadarResult) ------------------------------
    motion_sampling = string_field("per_chirp", options=["per_chirp", "per_frame"], enum_toggle=True,
                                   group=_SOLVE, description="Re-trace per chirp or per frame")
    t0 = float_field(0.0, group=_SOLVE, description="Solve start time (s)")
    post_processors = list_field(component_field(component_type="RadarPostProcessor"), default=[],
                                 group=_SOLVE,
                                 description="Extra post-processor views run after each solve")
    signal_figure = figure(title="Radar Signal", group=_SOLVE)

    view = string_field("range_doppler", options=_VIEWS, enum_toggle=True, group=_VIEW,
                        description="Which signal view to render")
    tx_index = int_field(0, min=0, group=_VIEW, description="TX index (raw / range-doppler)")
    rx_index = int_field(0, min=0, group=_VIEW, description="RX index (raw / range-doppler)")
    static_clutter_removal = bool_field(True, group=_VIEW,
                                        description="Remove static (zero-doppler) clutter")
    show_cfar = bool_field(False, group=_VIEW,
                           description="Overlay CFAR detections on the range-doppler map")

    detector = string_field("cfar", options=["cfar", "topk"], enum_toggle=True, group=_DETECTOR,
                            description="Point-cloud detector")
    guard_doppler = int_field(2, min=0, group=_DETECTOR, description="CFAR guard cells (doppler)")
    guard_range = int_field(4, min=0, group=_DETECTOR, description="CFAR guard cells (range)")
    training_doppler = int_field(4, min=0, group=_DETECTOR, description="CFAR training cells (doppler)")
    training_range = int_field(8, min=0, group=_DETECTOR, description="CFAR training cells (range)")
    pfa = float_field(1e-3, min=0.0, group=_DETECTOR, description="CFAR probability of false alarm")
    energy_top_k = int_field(128, min=1, group=_DETECTOR,
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
        import torch
        logger.info("=== Simulate clicked ===")
        if not torch.cuda.is_available():
            Notifications.warning("Radar", "Live solve requires a CUDA device (mitsuba ray tracing)")
            return "CUDA required"
        spec = SensorSpec.from_component(self)
        logger.info(f"Sensor pose:  position={spec.position}  target={spec.target}  up={spec.up}")
        logger.info(f"Sensor:       backend={spec.backend} device={spec.device} fov={spec.fov}")
        result = SolveRunner.run(
            self.scene,
            sensor=spec,
            tracer=TracerSpec.from_component(self),
            motion_sampling=str(self.motion_sampling), t0=num(self.t0))
        self._radar = result.radar
        self._signal = result.signal
        # Cheap fingerprints so back-to-back clicks visibly differ in the log even before
        # the figure renders. If these stay constant across clicks the solve is stuck.
        sig_abs = result.signal.detach().cpu().abs()
        logger.info(
            f"Signal:       shape={tuple(result.signal.shape)} "
            f"mean|sig|={sig_abs.mean().item():.6e} max|sig|={sig_abs.max().item():.6e}")
        logger.info(f"Updating view ({self.view}) ...")
        self.update_view()
        self._run_post_processors()
        Notifications.success("Radar", f"Solved: MIMO signal {tuple(result.signal.shape)}")
        logger.info("=== Simulate done ===")
        return "Solve complete"

    @button(display_name="Update View", group=_VIEW)
    def update_view(self):
        """Render the selected view from the last solved signal."""
        if self._signal is None:
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
        self.update_view()
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
                proc.process(self._radar, self._signal)

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
        sig = self._signal.detach().cpu().numpy()
        n_tx, n_rx, _, n_adc = sig.shape
        x = list(range(n_adc))
        sel = sig[self._tx(), self._rx(), 0]
        fig = self.signal_figure.clear()
        fig.line(x, sel.real.tolist(), label="Real", color="#ff9500")
        fig.line(x, sel.imag.tolist(), label="Imag", color="#00aaff")
        fig.title(f"Raw MIMO (Tx{self._tx()} Rx{self._rx()}, chirp 0)").xlabel("ADC sample").ylabel("Amplitude")
        batches = [[self._pair_series(sig[t, r, 0], x) for r in range(n_rx)] for t in range(n_tx)]
        fig.batch2d(batches, [f"Tx{t}" for t in range(n_tx)], [f"Rx{r}" for r in range(n_rx)])
        logger.info(f"  raw view -> selected Tx{self._tx()} Rx{self._rx()}, "
                    f"|real|.max={abs(sel.real).max():.4e} |imag|.max={abs(sel.imag).max():.4e}")
        return "Raw signal updated"

    @staticmethod
    def _pair_series(sample, x):
        return {"series": [
            {"x": x, "y": sample.real.tolist(), "label": "Real", "color": "#ff9500"},
            {"x": x, "y": sample.imag.tolist(), "label": "Imag", "color": "#00aaff"}]}

    def _show_rd(self):
        rd = SigProc.range_doppler(self._radar, self._signal, tx=self._tx(), rx=self._rx(),
                                   static_clutter_removal=bool(self.static_clutter_removal))
        logger.info(f"  rd-map: shape={rd.mag_db.shape} "
                    f"min={float(rd.mag_db.min()):.2f}dB max={float(rd.mag_db.max()):.2f}dB "
                    f"mean={float(rd.mag_db.mean()):.2f}dB")
        fig = self.signal_figure.clear().imshow(rd.mag_db)
        fig.title(f"Range-Doppler (Tx{self._tx()} Rx{self._rx()}, dB)").xlabel("Range bin").ylabel("Doppler bin")
        if bool(self.show_cfar):
            mask = SigProc.cfar_mask(rd.rd_map, guard=self._guard(),
                                     training=self._training(), pfa=num(self.pfa))
            rows, cols = np.nonzero(mask)
            logger.info(f"  cfar hits: {int(cols.size)}")
            if cols.size:
                fig.scatter(cols.tolist(), rows.tolist(), label="CFAR", color="#ff3030", size=8)
        return "Range-doppler updated"

    def _show_pc(self):
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

    def _show_music(self):
        try:
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
        return max(0, min(int(self.tx_index), self._signal.shape[0] - 1))

    def _rx(self):
        return max(0, min(int(self.rx_index), self._signal.shape[1] - 1))
