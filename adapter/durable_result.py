"""Resolve an exported native result against its verified Agent receipt."""
from pathlib import Path
import hashlib

from witwin_server.tools.base import ToolError

from .saved_result import SavedRadarResult


def load_verified_export(project, receipt, *, scene_id, input_fingerprint, motion_fingerprint):
    if receipt.get("input_fingerprint") != input_fingerprint:
        raise ToolError("Scene inputs changed after simulation.", code="stale_result")
    if (receipt.get("status") not in {"verified", "replay_ready", "exported"}
            or (receipt.get("verification") or {}).get("passed") is not True):
        raise ToolError("The saved result has no verified receipt.", code="result_not_verified")
    exports = list((receipt.get("exports") or {}).values())
    if not exports:
        raise ToolError("The native result is unavailable and has no saved export.",
                        code="result_handle_unavailable")
    project = Path(project).resolve()
    # One measurement may have several export operation ids for the same file.
    evidence = exports[-1]
    path = Path(str(evidence.get("path") or "")).resolve()
    if not path.is_relative_to(project / "results" / "radar-animation"):
        raise ToolError("Saved Radar result is outside this Project.", code="export_path_invalid")
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if (path.stat().st_size != evidence.get("size_bytes")
                or digest.hexdigest() != evidence.get("sha256")
                or evidence.get("input_fingerprint") != input_fingerprint
                or evidence.get("run_id") != receipt.get("run_id")
                or evidence.get("result_handle") != receipt.get("result_handle")):
            raise ValueError("Saved file or its receipt identity changed")
        saved = SavedRadarResult.load(path)
        if (saved.producer.get("scene_id") != scene_id
                or saved.producer.get("input_fingerprint") != input_fingerprint
                or saved.producer.get("motion_fingerprint") != motion_fingerprint):
            raise ValueError("Saved producer identity does not match the current scene and motion")
    except (OSError, ValueError, TypeError) as exc:
        raise ToolError(str(exc), code="saved_result_invalid") from exc
    return saved, dict(evidence)
