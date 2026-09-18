"""Presentation of retained old-project parameters in the current Radar UI."""

COMPATIBILITY_GROUP = "Compatibility"
_OLD_GROUPS = {"Tracer", "Noise", "Polarization", "Receiver", "Timeline"}
_OLD_SOLVE_FIELDS = {
    "stream_max_fps", "stream_channels", "stream_history_length",
    "stream_on_change_only", "stream_status", "post_processors",
}
_OLD_BUTTONS = {"start_stream", "pause_stream", "stop_stream", "generate_timeline", "show_frame"}


def present_radar_controls(metadata):
    """Change UI metadata only; retain every serialized parameter value."""
    result = dict(metadata)
    result["groups"] = [dict(group) for group in metadata["groups"] if group["name"] not in _OLD_GROUPS]
    result["groups"].append({
        "name": COMPATIBILITY_GROUP,
        "type": "foldout",
        "display_name": "Compatibility settings (old projects)",
        "collapsed": True,
        "description": "Retained for old project files. These options are not supported by the current GPU simulation. Keep their defaults for new runs.",
    })
    result["fields"] = []
    for original in metadata["fields"]:
        field = dict(original)
        if field.get("group") in _OLD_GROUPS or field["name"] in _OLD_SOLVE_FIELDS:
            field["group"] = COMPATIBILITY_GROUP
        result["fields"].append(field)
    # The old streaming/frame-generation adapter is not connected to Radar 0.4.
    # Keep the Python methods for compatibility, without advertising UI actions.
    result["buttons"] = [dict(button) for button in metadata["buttons"] if button["name"] not in _OLD_BUTTONS]
    return result
