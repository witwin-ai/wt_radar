"""Current UI hides unavailable actions without changing old project data."""
from wt_radar.components.radar import RadarComponent


def test_current_controls_keep_simulation_replay_and_export():
    for metadata in (RadarComponent.get_definition(), RadarComponent().to_dict()):
        buttons = {button["name"] for button in metadata["buttons"]}
        assert {"simulate", "simulate_animation", "load_saved_result",
                "prepare_synchronized_replay", "export_animation_result"} <= buttons
        assert not buttons & {"start_stream", "pause_stream", "stop_stream", "generate_timeline", "show_frame"}
        groups = {group["name"]: group for group in metadata["groups"]}
        assert not groups.keys() & {"Tracer", "Noise", "Polarization", "Receiver", "Timeline"}
        assert groups["Compatibility"]["collapsed"] is True
        assert all("Legacy" not in group["display_name"] for group in groups.values())
        assert groups["StudioAnimation"]["display_name"] == "Simulation & Replay (multi-frame)"
        assert groups["SnapshotTarget"]["display_name"] == "Static Snapshot (1-frame diagnostic)"
        assert groups["SnapshotTarget"]["collapsed"] is True
        fields = {field["name"]: field for field in metadata["fields"]}
        assert fields["signal_source"]["group"] == "StudioAnimation"
        assert fields["signal_figure"]["group"] == "StudioAnimation"
        assert fields["snapshot_figure"]["group"] == "SnapshotTarget"


def test_old_settings_round_trip_and_remain_available_for_repair():
    component = RadarComponent()
    component.enable_thermal = True
    component.thermal_std = 0.25
    component.frame_rate = 24.0
    saved = component.to_properties_dict()
    metadata = component.to_dict()
    fields = {field["name"]: field for field in metadata["fields"]}
    assert fields["enable_thermal"]["group"] == "Compatibility"
    assert fields["thermal_std"]["group"] == "Compatibility"
    assert fields["frame_rate"]["group"] == "Compatibility"
    assert fields["animation_duration_s"]["group"] == "StudioAnimation"
    assert component.to_properties_dict() == saved
    restored = RadarComponent()
    restored.from_properties_dict(saved["properties"])
    assert restored.enable_thermal is True
    assert restored.thermal_std == 0.25
    assert restored.frame_rate == 24.0
    # Presentation does not mutate the class metadata used by other instances.
    assert RadarComponent._fields_meta["enable_thermal"].group == "Noise"
