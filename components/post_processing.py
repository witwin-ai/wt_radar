"""Pluggable post-processors (carried over from the old plugin, modernized for R5).

A ``RadarResult`` can reference a list of post-processor components that each render an
extra view from the solved signal after every solve. The base ``RadarPostProcessor``
exposes a ``process(radar, signal)`` hook + a figure; subclasses implement a view using
the real ``sigproc`` (range-doppler, point cloud). Authors can add their own processor by
subclassing ``RadarPostProcessor`` in another plugin — ``component_field`` accepts any
subclass.
"""
from typing import Any

from witwin_server.core.components import (
    Component,
    bool_field,
    component,
    figure,
    int_field,
)

from ..adapter.solve import SigProc

_CAT = "Simulation/Radar"


@component(name="RadarPostProcessor", category=_CAT)
class RadarPostProcessorComponent(Component):
    """Base post-processor: subclasses override :meth:`process` to render their view."""

    enabled = bool_field(True, description="Run this processor after each solve")
    result_figure = figure(title="Post-Processed")

    def process(self, radar: Any, signal: Any) -> None:
        """Render this processor's view from the solved ``radar`` + MIMO ``signal``."""
        # Base is a no-op; subclasses render into self.result_figure.
        return None


@component(name="RangeDopplerProcessor", category=_CAT)
class RangeDopplerProcessorComponent(RadarPostProcessorComponent):
    """Render a range-doppler map for one TX-RX pair."""

    tx_index = int_field(0, min=0, description="TX index")
    rx_index = int_field(0, min=0, description="RX index")
    static_clutter_removal = bool_field(True, description="Remove static clutter")

    def process(self, radar: Any, signal: Any) -> None:
        tx = max(0, min(int(self.tx_index), signal.shape[0] - 1))
        rx = max(0, min(int(self.rx_index), signal.shape[1] - 1))
        rd = SigProc.range_doppler(radar, signal, tx=tx, rx=rx,
                                   static_clutter_removal=bool(self.static_clutter_removal))
        self.result_figure.clear().imshow(rd.mag_db).title(f"Range-Doppler (Tx{tx} Rx{rx}, dB)")


@component(name="PointCloudProcessor", category=_CAT)
class PointCloudProcessorComponent(RadarPostProcessorComponent):
    """Render a top-down (x vs z) point cloud."""

    static_clutter_removal = bool_field(True, description="Remove static clutter")

    def process(self, radar: Any, signal: Any) -> None:
        pc = SigProc.point_cloud(radar, signal, detector="cfar",
                                 static_clutter_removal=bool(self.static_clutter_removal),
                                 guard=(2, 4), training=(4, 8), pfa=1e-3, energy_top_k=128)
        fig = self.result_figure.clear()
        if pc.shape[0] == 0:
            fig.title("Point cloud (no detections)")
            return
        fig.scatter(pc[:, 0].tolist(), pc[:, 2].tolist(), label="points", color="#30c0ff", size=10)
        fig.title(f"Point cloud ({pc.shape[0]} pts)").xlabel("x (m)").ylabel("z (m)")
