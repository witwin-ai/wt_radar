"""RadarResult — the Simulate button + in-component signal visualization (master §7.2).

Lives on the Radar Settings object. **Simulate** rebuilds the ``(Scene, RadarConfig)``
pair from the live scene, constructs a ``Radar`` from the RadarSensor pose/backend, runs
the tracer + MIMO generation (``adapter.solve.SolveRunner``), keeps the signal
server-side (runtime state, never persisted), and renders the selected view into its
figure. Radar visualizes in-component only (no Results panel / viewport field): raw MIMO
signal, range-doppler map, point cloud, MUSIC image, and CFAR detections.

Live solve needs CUDA (the mitsuba ray tracer is CUDA-only).
"""
import numpy as np

from witwin_server import Notifications
from witwin_server.components import (
    Component,
    bool_field,
    button,
    component,
    define_group,
    figure,
    float_field,
    foldout_group,
    int_field,
    string_field,
)

from ..adapter.common import num
from ..adapter.solve import SensorSpec, SigProc, SolveRunner, TracerSpec

_CAT = "Simulation/Radar"
_VIEWS = ["raw_signal", "range_doppler", "point_cloud", "music"]


@component(name="RadarResult", category=_CAT)
class RadarResultComponent(Component):
    """Solve + in-component range-doppler / point-cloud / MUSIC / CFAR views."""

    _signal = None   # runtime MIMO cube (TX, RX, chirps, ADC); never serialized
    _radar = None    # runtime Radar

    define_group(foldout_group("Solve"))
    define_group(foldout_group("View"))
    define_group(foldout_group("Detector", display_name="Detector / CFAR"))

    motion_sampling = string_field("per_chirp", options=["per_chirp", "per_frame"], enum_toggle=True,
                                   group="Solve", description="Re-trace per chirp or per frame")
    t0 = float_field(0.0, group="Solve", description="Solve start time (s)")

    view = string_field("range_doppler", options=_VIEWS, enum_toggle=True, group="View",
                        description="Which signal view to render")
    tx_index = int_field(0, min=0, group="View", description="TX index (raw / range-doppler)")
    rx_index = int_field(0, min=0, group="View", description="RX index (raw / range-doppler)")
    static_clutter_removal = bool_field(True, group="View", description="Remove static (zero-doppler) clutter")
    show_cfar = bool_field(False, group="View", description="Overlay CFAR detections on the range-doppler map")

    detector = string_field("cfar", options=["cfar", "topk"], enum_toggle=True, group="Detector",
                            description="Point-cloud detector")
    guard_doppler = int_field(2, min=0, group="Detector", description="CFAR guard cells (doppler)")
    guard_range = int_field(4, min=0, group="Detector", description="CFAR guard cells (range)")
    training_doppler = int_field(4, min=0, group="Detector", description="CFAR training cells (doppler)")
    training_range = int_field(8, min=0, group="Detector", description="CFAR training cells (range)")
    pfa = float_field(1e-3, min=0.0, group="Detector", description="CFAR probability of false alarm")
    energy_top_k = int_field(128, min=1, group="Detector", description="Top-K energy detections (topk detector)")

    signal_figure = figure(title="Radar Signal")

    # --- solve ---------------------------------------------------------------

    @button(display_name="Simulate")
    def simulate(self):
        """Rebuild + solve the live scene, keep the signal, and render the current view."""
        import torch
        if not torch.cuda.is_available():
            Notifications.warning("Radar", "Live solve requires a CUDA device (mitsuba ray tracing)")
            return "CUDA required"
        owner = self.owner
        result = SolveRunner.run(
            self.scene,
            sensor=SensorSpec.from_component(owner.get_component("RadarSensor")),
            tracer=TracerSpec.from_component(owner.get_component("RadarTracer")),
            motion_sampling=str(self.motion_sampling), t0=num(self.t0))
        self._radar = result.radar
        self._signal = result.signal
        self.update_view()
        Notifications.success("Radar", f"Solved: MIMO signal {tuple(result.signal.shape)}")
        return "Solve complete"

    @button(display_name="Update View")
    def update_view(self):
        """Render the selected view from the last solved signal."""
        if self._signal is None:
            return "Simulate first"
        view = str(self.view)
        if view == "raw_signal":
            return self._show_raw()
        if view == "range_doppler":
            return self._show_rd()
        if view == "point_cloud":
            return self._show_pc()
        return self._show_music()

    # --- views ---------------------------------------------------------------

    def _show_raw(self):
        # Raw MIMO signal: the selected pair (real/imag) + a Tx x Rx batch grid (chirp 0).
        sig = self._signal.detach().cpu().numpy()
        n_tx, n_rx, _, n_adc = sig.shape
        x = list(range(n_adc))
        fig = self.signal_figure.clear()
        sel = sig[self._tx(), self._rx(), 0]
        fig.line(x, sel.real.tolist(), label="Real", color="#ff9500")
        fig.line(x, sel.imag.tolist(), label="Imag", color="#00aaff")
        fig.title(f"Raw MIMO (Tx{self._tx()} Rx{self._rx()}, chirp 0)").xlabel("ADC sample").ylabel("Amplitude")
        batches = [[self._pair_series(sig[t, r, 0], x) for r in range(n_rx)] for t in range(n_tx)]
        fig.batch2d(batches, [f"Tx{t}" for t in range(n_tx)], [f"Rx{r}" for r in range(n_rx)])
        return "Raw signal updated"

    @staticmethod
    def _pair_series(sample, x):
        # One batch2d cell: real + imag series for a single Tx-Rx chirp.
        return {"series": [
            {"x": x, "y": sample.real.tolist(), "label": "Real", "color": "#ff9500"},
            {"x": x, "y": sample.imag.tolist(), "label": "Imag", "color": "#00aaff"}]}

    def _show_rd(self):
        # Range-doppler map (dB) + optional CFAR overlay.
        rd = SigProc.range_doppler(self._radar, self._signal, tx=self._tx(), rx=self._rx(),
                                   static_clutter_removal=bool(self.static_clutter_removal))
        fig = self.signal_figure.clear().imshow(rd.mag_db)
        fig.title(f"Range-Doppler (Tx{self._tx()} Rx{self._rx()}, dB)").xlabel("Range bin").ylabel("Doppler bin")
        if bool(self.show_cfar):
            mask = SigProc.cfar_mask(rd.rd_map, guard=self._guard(), training=self._training(), pfa=num(self.pfa))
            rows, cols = np.nonzero(mask)
            if cols.size:
                fig.scatter(cols.tolist(), rows.tolist(), label="CFAR", color="#ff3030", size=8)
        return "Range-doppler updated"

    def _show_pc(self):
        # Filtered point cloud, top-down (x vs z).
        pc = SigProc.point_cloud(
            self._radar, self._signal, detector=str(self.detector),
            static_clutter_removal=bool(self.static_clutter_removal), guard=self._guard(),
            training=self._training(), pfa=num(self.pfa), energy_top_k=int(self.energy_top_k))
        fig = self.signal_figure.clear()
        if pc.shape[0] == 0:
            fig.title("Point cloud (no detections)")
            return "No detections"
        fig.scatter(pc[:, 0].tolist(), pc[:, 2].tolist(), label="points", color="#30c0ff", size=10)
        fig.title(f"Point cloud ({pc.shape[0]} pts)").xlabel("x (m)").ylabel("z (m)")
        return f"{pc.shape[0]} points"

    def _show_music(self):
        # MUSIC 2D image (max over range). Degenerate on small arrays -> warn, don't crash.
        try:
            img = SigProc.music_image(self._radar, self._signal, num_pixels=64)
        except Exception as exc:  # noqa: BLE001 - optional view; small UPAs make MUSIC ill-posed
            Notifications.warning("Radar", f"MUSIC needs a larger UPA (e.g. 20x20): {type(exc).__name__}")
            return "MUSIC unavailable for this array"
        self.signal_figure.clear().imshow(img).title("MUSIC image (max over range)")
        return "MUSIC updated"

    # --- small helpers -------------------------------------------------------

    def _guard(self):
        return (int(self.guard_doppler), int(self.guard_range))

    def _training(self):
        return (int(self.training_doppler), int(self.training_range))

    def _tx(self):
        return max(0, min(int(self.tx_index), self._signal.shape[0] - 1))

    def _rx(self):
        return max(0, min(int(self.rx_index), self._signal.shape[1] - 1))
