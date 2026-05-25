"""Multi-frame timeline -> ``witwin.radar.Timeline`` (master §5.2, Phase R4).

A settings-level component that generates a stack of radar frames over time, either from
a point-cloud sequence file or from an SMPL motion file rendered against the scene. Stores
only the source descriptor/path (keyframes are runtime/result state). Generated frames are
displayed by pushing the selected frame into the sibling ``RadarResult`` view (the frame
slider is the RD movie).
"""
from witwin_server import Notifications
from witwin_server.components import (
    Component,
    bool_field,
    button,
    component,
    float_field,
    int_field,
    string_field,
)

from ..adapter.solve import SensorSpec, TracerSpec
from ..adapter.timeline_run import TimelineRunner

_CAT = "Simulation/Radar"
_POINTCLOUD = {"field_name": "source", "operator": "eq", "value": "pointcloud_sequence"}
_MOTION = {"field_name": "source", "operator": "eq", "value": "motion"}


@component(name="RadarTimeline", category=_CAT)
class RadarTimelineComponent(Component):
    """Generate + step through a multi-frame radar sequence."""

    _frames = None   # runtime (num_radar_frames, TX, RX, chirps, ADC); never serialized
    _radar = None

    frame_rate = float_field(30.0, min=1.0, description="Source keyframe rate (Hz)")
    source = string_field("pointcloud_sequence", options=["pointcloud_sequence", "motion"],
                          enum_toggle=True, description="How keyframes are built")
    pointcloud_path = string_field("", show_if=_POINTCLOUD,
                                   description=".npy/.npz with (F,N,3) positions [+ (F,N) intensities]")
    motion_path = string_field("", show_if=_MOTION,
                               description=".npz pose (F,72)/shape/root_translation; needs an SMPL 'human'")
    velocity_corrected = bool_field(True, description="Scale displacement to physical velocity per frame")
    frame_index = int_field(0, min=0, description="Frame to display (steps the RD movie)")

    @button(display_name="Generate Frames")
    def generate(self):
        """Build the timeline + radar, generate the frame stack, and show the first frame."""
        import torch
        if not torch.cuda.is_available():
            Notifications.warning("Radar", "Timeline generation requires a CUDA device")
            return "CUDA required"
        owner = self.owner
        radar, frames = TimelineRunner.generate(
            self,
            sensor=SensorSpec.from_component(owner.get_component("RadarSensor")),
            tracer=TracerSpec.from_component(owner.get_component("RadarTracer")),
            studio_scene=self.scene)
        self._radar = radar
        self._frames = frames
        self.show_frame()
        Notifications.success("Radar", f"Timeline: {frames.shape[0]} frames {tuple(frames.shape)}")
        return f"{frames.shape[0]} frames"

    @button(display_name="Show Frame")
    def show_frame(self):
        """Push the selected frame into the sibling RadarResult view."""
        if self._frames is None:
            return "Generate first"
        result = self.owner.get_component("RadarResult")
        if result is None:
            return "No RadarResult component"
        index = max(0, min(int(self.frame_index), self._frames.shape[0] - 1))
        result._signal = self._frames[index]
        result._radar = self._radar
        result.update_view()
        return f"Frame {index}"
