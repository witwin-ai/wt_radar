"""Agent-facing Radar tools built on the existing Studio/Radar 0.3 bridge.

These tools deliberately stop at orchestration and evidence.  They do not
implement propagation, scattering, waveform generation, or DSP; those remain
owned by ``witwin-radar`` and ``witwin-channel``.
"""
from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import re
import traceback
from typing import Any, Dict, Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from witwin_server.tools.base import ToolError, tool

from .adapter.memory_budget import MAX_RESULT_BYTES, available_memory_bytes
from .adapter.solve import SensorSpec
from .library_items import _settings_object
from .retrieval_tags import (
    CANCEL_TAGS,
    ENSURE_TAGS,
    EXPORT_TAGS,
    INSPECT_TAGS,
    PLAN_TAGS,
    REPLAY_TAGS,
    STATUS_TAGS,
    SUBMIT_TAGS,
    VERIFY_TAGS,
)


SCENE_ID_PROPERTY = {
    "type": "string",
    "minLength": 1,
    "description": "Exact open Studio scene id; never inferred from an object name.",
}

_PIPELINE_RADAR_TAG = "witwin.pipeline.radar"
_RUNTIME_RADAR_FIELDS = {
    "animation_export_path",
    "animation_frame_index",
    "animation_status",
    "replay_status",
    "saved_result_loaded",
    "signal_figure",
    "signal_source",
    "signal_stream",
    "snapshot_status",
    "stream_status",
    # Review-only controls do not alter the native simulation result. They
    # must not stale a valid measurement when preview controls change.
    "view",
    "tx_index",
    "rx_index",
    "static_clutter_removal",
    "show_cfar",
}
_OPERATION_STATE_KEY = "agent_radar_operations_v1"
_OPERATION_RECEIPT_SCHEMA_VERSION = 1
_ACTIVE_JOBS: dict[str, asyncio.Task] = {}
_EXPORT_LOCKS: dict[str, asyncio.Lock] = {}
_PREFLIGHT_CACHE_ATTRIBUTE = "_radar_native_preflight_cache_v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _preflight_cache(ctx: Any) -> dict[str, Any]:
    cache = getattr(ctx, _PREFLIGHT_CACHE_ATTRIBUTE, None)
    if cache is None:
        cache = {}
        setattr(ctx, _PREFLIGHT_CACHE_ATTRIBUTE, cache)
    return cache


def _preflight_cache_key(scene_id: str, radar_object_id: str, fingerprint: str) -> str:
    return f"{scene_id}\0{radar_object_id}\0{fingerprint}"


def _remember_preflight(
    ctx: Any,
    *,
    scene_id: str,
    radar_object_id: str,
    fingerprint: str,
    native_preflight: dict[str, Any],
) -> None:
    _preflight_cache(ctx)[_preflight_cache_key(scene_id, radar_object_id, fingerprint)] = {
        "created_at": _utc_now(),
        "native_preflight": copy.deepcopy(native_preflight),
    }


def _consume_preflight(
    ctx: Any,
    *,
    scene_id: str,
    radar_object_id: str,
    fingerprint: str,
) -> dict[str, Any] | None:
    cached = _preflight_cache(ctx).pop(
        _preflight_cache_key(scene_id, radar_object_id, fingerprint), None,
    )
    return copy.deepcopy(cached["native_preflight"]) if cached is not None else None


def _workspace_state(ctx: Any) -> Any:
    state = getattr(ctx, "workspaceState", None) or getattr(ctx, "workspace_state", None)
    if state is None:
        container = getattr(ctx, "state", None)
        state = getattr(container, "workspace_state", None) or getattr(container, "workspaceState", None)
    return state


def _operation_map(ctx: Any) -> dict[str, Any]:
    state = _workspace_state(ctx)
    if state is None:
        value = getattr(ctx, "_radar_operation_fallback", {}) or {}
    else:
        getter = getattr(state, "get_strict", None) or state.get
        try:
            value = getter(_OPERATION_STATE_KEY, {})
        except Exception as exc:
            raise ToolError(
                "Radar operation state is unreadable; it was not reset or overwritten.",
                code="radar_operation_state_corrupt",
                detail={"path": str(getattr(state, "path", "")), "error": str(exc)},
            ) from exc
    if not isinstance(value, dict):
        raise ToolError(
            "Radar operation state is not a JSON object; it was not reset or overwritten.",
            code="radar_operation_state_corrupt",
            detail={"value_type": type(value).__name__},
        )
    return dict(value)


def _save_operation_map(ctx: Any, value: dict[str, Any]) -> None:
    state = _workspace_state(ctx)
    if state is None:
        setattr(ctx, "_radar_operation_fallback", copy.deepcopy(value))
    else:
        state.set(_OPERATION_STATE_KEY, _jsonable(value))


def _get_operation(ctx: Any, operation_id: str) -> dict[str, Any] | None:
    value = _operation_map(ctx).get(operation_id)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ToolError(
            "Radar operation receipt is not a JSON object.",
            code="radar_operation_state_corrupt",
            detail={"operation_id": operation_id, "value_type": type(value).__name__},
        )
    if value.get("schema_version") != _OPERATION_RECEIPT_SCHEMA_VERSION:
        raise ToolError(
            "Radar operation receipt schema is missing or unsupported.",
            code="radar_operation_state_corrupt",
            detail={
                "operation_id": operation_id,
                "expected_schema_version": _OPERATION_RECEIPT_SCHEMA_VERSION,
                "actual_schema_version": value.get("schema_version"),
            },
        )
    if str(value.get("operation_id") or "") != str(operation_id):
        raise ToolError(
            "Radar operation receipt identity does not match its map key.",
            code="radar_operation_state_corrupt",
            detail={
                "expected_operation_id": operation_id,
                "actual_operation_id": value.get("operation_id"),
            },
        )
    return copy.deepcopy(value)


def _put_operation(ctx: Any, receipt: dict[str, Any]) -> dict[str, Any]:
    receipt = {
        **receipt,
        "schema_version": _OPERATION_RECEIPT_SCHEMA_VERSION,
        "updated_at": _utc_now(),
    }
    operation_id = str(receipt["operation_id"])
    state = _workspace_state(ctx)
    updater = getattr(state, "update_strict", None) if state is not None else None
    if callable(updater):
        def update_map(current):
            if not isinstance(current, dict):
                raise ValueError("Radar operation map is not a JSON object")
            values = dict(current)
            values[operation_id] = _jsonable(receipt)
            return values

        try:
            updater(_OPERATION_STATE_KEY, update_map, {})
        except Exception as exc:
            raise ToolError(
                "Radar operation receipt could not be written atomically.",
                code="radar_operation_state_corrupt",
                detail={"operation_id": operation_id, "error": str(exc)},
            ) from exc
    else:
        values = _operation_map(ctx)
        values[operation_id] = _jsonable(receipt)
        _save_operation_map(ctx, values)
    return copy.deepcopy(receipt)


def _job_key(ctx: Any, operation_id: str) -> str:
    state = _workspace_state(ctx)
    path = str(getattr(state, "path", "memory"))
    return f"{path}:{operation_id}"


def _scene_from_args(ctx: Any, args: Dict[str, Any]):
    requested = str(args.get("scene_id") or "").strip()
    if not requested:
        raise ToolError("scene_id is required", code="scene_context_required")
    current = getattr(ctx, "scene", None)
    if str(getattr(current, "scene_id", "")) == requested:
        return current
    server = getattr(getattr(ctx, "api", None), "server", None)
    getter = getattr(server, "get_scene", None)
    scene = getter(requested) if callable(getter) else None
    if scene is None:
        scenes = getattr(server, "scenes", None)
        scene = scenes.get(requested) if hasattr(scenes, "get") else None
    if scene is None:
        raise ToolError(
            f"scene not found or no longer open: {requested}",
            code="missing_scene",
            detail={"scene_id": requested},
        )
    return scene


def _radar_objects(scene: Any) -> list[Any]:
    return [
        obj for obj in (getattr(scene, "objects", {}) or {}).values()
        if callable(getattr(obj, "get_component", None))
        and obj.get_component("Radar") is not None
    ]


def _select_radar(scene: Any, object_id: str = "") -> Any:
    requested = str(object_id or "").strip()
    if requested:
        obj = scene.get_object(requested)
        if obj is None or obj.get_component("Radar") is None:
            raise ToolError(
                f"Radar object not found in scene: {requested}",
                code="missing_radar",
                detail={"scene_id": str(scene.scene_id), "radar_object_id": requested},
            )
        return obj
    sensors = _radar_objects(scene)
    if not sensors:
        raise ToolError("No Radar Settings object exists in this scene.", code="missing_radar")
    if len(sensors) != 1:
        raise ToolError(
            "More than one Radar exists; specify radar_object_id explicitly.",
            code="ambiguous_radar",
            detail={"radar_object_ids": [str(obj.id) for obj in sensors]},
        )
    return sensors[0]


def _as_vector3(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ToolError(f"{name} must contain three finite numbers", code="invalid_sensor_pose")
    return vector


def _look_at_euler(position: Any, aim_point: Any) -> list[float]:
    origin = _as_vector3(position, "position")
    target = _as_vector3(aim_point, "aim_point")
    forward = target - origin
    length = float(np.linalg.norm(forward))
    if length <= 1e-6:
        raise ToolError("Radar position and aim point must be different.", code="invalid_sensor_pose")
    forward /= length
    reference_up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(forward, reference_up))) > 0.98:
        reference_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, reference_up)
    right /= np.linalg.norm(right)
    up = np.cross(-forward, right)
    rotation = np.column_stack((right, up, -forward))
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ToolError("Could not construct a right-handed Radar pose.", code="invalid_sensor_pose")
    return Rotation.from_matrix(rotation).as_euler("ZXY", degrees=False).tolist()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _persisted_numeric_identity(value: Any) -> Any:
    """Canonicalize authored numbers to the precision Studio persists.

    Component and Timeline numeric fields round-trip through Studio's float32
    storage.  Hashing the pre-save Python float64 spelling made an unchanged
    Radar appear edited after reopening a Scene (for example 77e9 becomes
    76999999488.0).  The solver already consumes these persisted values, so
    float32 is the exact durable-authoring boundary rather than a tolerance.
    """
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, (float, np.floating)):
        canonical = float(np.float32(value))
        return 0.0 if canonical == 0.0 else canonical
    if isinstance(value, dict):
        return {str(key): _persisted_numeric_identity(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_persisted_numeric_identity(item) for item in value]
    return _persisted_numeric_identity(_jsonable(value))


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _call_diagnostics(module_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {"module": module_name, "import_ok": False}
    try:
        module = importlib.import_module(module_name)
        result["import_ok"] = True
        for function_name in ("runtime_diagnostics", "build_info", "capabilities"):
            function = getattr(module, function_name, None)
            if callable(function):
                try:
                    result[function_name] = _jsonable(function())
                except Exception as exc:  # diagnostics must remain reportable
                    result[function_name] = {"ok": False, "error": str(exc)}
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _runtime_evidence() -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "packages": {
            name: _package_version(name)
            for name in ("witwin", "witwin-radar", "witwin-channel", "torch")
        },
        "radar": _call_diagnostics("witwin.radar"),
        "channel": _call_diagnostics("witwin.channel"),
    }
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        evidence["torch"] = {
            "version": str(torch.__version__),
            "cuda_build": str(torch.version.cuda),
            "cuda_available": cuda,
            "device_name": torch.cuda.get_device_name(0) if cuda else None,
            "compute_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
        }
    except Exception as exc:
        evidence["torch"] = {"cuda_available": False, "error": str(exc)}
    return evidence


def _strip_runtime_radar_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_runtime_radar_fields(item)
            for key, item in value.items()
            if key not in _RUNTIME_RADAR_FIELDS
        }
    if isinstance(value, list):
        return [_strip_runtime_radar_fields(item) for item in value]
    return _jsonable(value)


def _update_content_hash(digest: Any, label: str, value: Any, *, dtype: Any) -> None:
    """Hash one solver-consumed numeric buffer with explicit dtype and shape."""
    if value is None:
        array = np.empty((0,), dtype=dtype)
    elif hasattr(value, "detach"):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.ascontiguousarray(array, dtype=dtype)
    header = json.dumps(
        {"label": label, "dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(array.tobytes(order="C"))


def _geometry_content_digest(scene: Any) -> str:
    """Digest the hydrated geometry and skin buffers consumed by the adapter."""
    digest = hashlib.sha256()
    for object_id in sorted((str(value) for value in scene.objects), key=str):
        obj = scene.get_object(object_id)
        skin = obj.get_component("SkinnedMesh") if obj is not None else None
        mesh = skin or (obj.get_component("Mesh") if obj is not None else None)
        if mesh is None:
            continue
        digest.update(json.dumps(
            {"object_id": object_id, "component": "SkinnedMesh" if skin else "Mesh"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"))
        _update_content_hash(digest, "vertices", mesh.get_vertices_numpy(), dtype=np.float32)
        _update_content_hash(digest, "faces", mesh.get_faces_numpy(), dtype=np.uint32)
        if skin is None:
            continue
        skinning = skin.get_skinning_data()
        digest.update(json.dumps(
            {"bone_ids": [str(value) for value in (skinning.get("bone_ids") or [])]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"))
        _update_content_hash(digest, "skin_indices", skinning.get("skin_indices"), dtype=np.int64)
        _update_content_hash(digest, "skin_weights", skinning.get("skin_weights"), dtype=np.float32)
        _update_content_hash(
            digest,
            "inverse_bind_matrices",
            skinning.get("inverse_bind_matrices"),
            dtype=np.float32,
        )
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _scene_input_evidence(scene: Any) -> dict[str, Any]:
    from witwin_server.features.timeline.fingerprint import authored_scene_payload
    payload = _jsonable(authored_scene_payload(scene))
    radar_object_ids: set[str] = set()
    for object_id, obj in (payload.get("objects") or {}).items():
        if any(component.get("type") == "Radar" for component in obj.get("components") or []):
            radar_object_ids.add(str(object_id))
    geometry = _geometry_content_digest(scene)
    payload["solver_geometry_content_sha256"] = geometry
    payload["scene_fingerprint_schema"] = "witwin.radar.authored-scene.v5"
    # Component values are stored by Studio's field system as float32, but
    # Timeline key times/values remain JSON/TOML doubles and the native motion
    # sampler consumes those doubles.  Canonicalizing the entire Scene to
    # float32 would make distinct solver inputs share one identity.
    payload["objects"] = _persisted_numeric_identity(payload.get("objects") or {})
    objects = payload.get("objects") or {}
    non_radar_objects = {
        key: value for key, value in objects.items() if str(key) not in radar_object_ids
    }
    # Radar authoring is represented separately by _radar_measurement_contract.
    # Excluding the settings object here prevents its runtime/review state and
    # quaternion serialization round-trips from contaminating room/motion identity.
    payload["objects"] = non_radar_objects
    return {
        "schema": "witwin.radar.authored-scene.v5",
        "fingerprint": _hash_json(payload),
        "metadata_fingerprint": _hash_json({
            key: value for key, value in payload.items()
            if key not in {"objects", "timeline", "solver_geometry_content_sha256"}
        }),
        "timeline_fingerprint": _hash_json(payload.get("timeline")),
        "non_radar_objects_fingerprint": _hash_json(non_radar_objects),
        "geometry_content_sha256": geometry,
    }


def _scene_input_fingerprint(scene: Any) -> str:
    return str(_scene_input_evidence(scene)["fingerprint"])


def _radar_measurement_contract(obj: Any) -> dict[str, Any]:
    """Durable authored Radar inputs actually consumed by the native adapter."""
    from dataclasses import asdict
    from .adapter.config_map import ConfigMap
    from .adapter.solve import SensorSpec, TracerSpec

    radar = obj.get_component("Radar")
    sensor = asdict(SensorSpec.from_component(radar))
    contract = _persisted_numeric_identity({
        "config": ConfigMap.build_dict(obj),
        "tracer": asdict(TracerSpec.from_component(radar)),
        "target": {
            "object_id": str(radar.snapshot_target_id or ""),
            "local_point_m": _jsonable(radar.snapshot_local_point),
            "rcs_m2": float(radar.snapshot_rcs_m2),
            "polarization": _jsonable(radar.snapshot_polarization),
        },
        "animation": {
            "start_s": float(radar.t0),
            "duration_s": float(radar.animation_duration_s),
            "fps": float(radar.animation_fps),
            "motion_sampling": str(radar.motion_sampling),
        },
    })
    # SensorSpec is derived from the exact float32 world matrix consumed by
    # the solver.  Do not decimal-round it: adjacent authored pose ULPs can
    # change LOS at a visibility boundary and therefore must change identity.
    contract["sensor"] = _jsonable(sensor)
    return contract


def _measurement_input_fingerprint(scene: Any, obj: Any) -> str:
    payload = {
        "schema": "witwin.radar.agent-input.v5",
        "scene_fingerprint": _scene_input_fingerprint(scene),
        "scene_id": str(scene.scene_id),
        "radar_object_id": str(obj.id),
        "radar_contract": _radar_measurement_contract(obj),
        "physics": {"components": ["los"], "max_depth": 0, "device": "cuda"},
        "packages": {
            name: _package_version(name)
            for name in ("witwin", "witwin-radar", "witwin-channel", "torch")
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _timeline_summary(scene: Any) -> dict[str, Any]:
    manager = getattr(scene, "timeline_manager", None)
    if manager is None:
        return {"available": False, "duration_s": 0.0, "playing": False, "recording": False}
    clip = getattr(manager, "clip", None)
    return {
        "available": clip is not None,
        "duration_s": float(getattr(clip, "duration", 0.0) or 0.0),
        "playing": bool(getattr(manager, "is_playing", False)),
        "recording": bool(getattr(manager, "is_recording", False)),
    }


def _recorded_replay_summary(ctx: Any, obj: Any) -> dict[str, Any]:
    """Inspect the saved Figure separately from transient native worker state."""
    from .adapter.replay import MAX_PREVIEW_BYTES, motion_fingerprint

    summary = {
        "status": "not_prepared", "manifest_present": False,
        "payload_available": False, "payload_integrity_verified": False,
        "motion_current": False, "ready_for_timeline": False,
        "scope": "saved numeric RP/RD recording; not a live native solver handle or geometry validation",
    }
    try:
        # Deserialization restores the serialized Figure property, while the
        # imperative Figure builder may still be empty until its first update.
        figure = (obj.get_component("Radar").to_dict().get("values") or {}).get("signal_figure") or {}
        record = (figure.get("data") or {}).get("recording")
        if not isinstance(record, dict):
            return summary
        summary["manifest_present"] = True
        asset = record.get("asset") or {}
        asset_id = str(asset.get("assetId") or "")
        times = np.asarray(record.get("timesS"), dtype=np.float64)
        stride = int(record.get("frameStride", 0))
        expected_bytes = len(times) * stride * 4
        if (
            record.get("schema") != 1 or record.get("sceneId") != obj.scene.scene_id
            or not re.fullmatch(r"[A-Za-z0-9_-]+", asset_id)
            or asset.get("uri") != f"/scenes/{obj.scene.scene_id}/assets/{asset_id}"
            or asset.get("format") != "witwin.radar.replay.float32.v1"
            or times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)
            or not math.isfinite(float(record.get("endTimeS", 0)))
            or float(record.get("endTimeS", 0)) <= times[-1]
            or stride != len(record.get("rangeM", [])) * (1 + len(record.get("velocityMps", [])))
            or expected_bytes <= 0 or expected_bytes > MAX_PREVIEW_BYTES
            or expected_bytes != asset.get("byteLength")
        ):
            return {**summary, "status": "invalid_manifest"}
        summary.update({"asset": copy.deepcopy(asset), "frame_count": len(times),
                        "interval_s": [float(times[0]), float(record["endTimeS"])]})
        handler = getattr(ctx.api.server, "handlers", {}).get("binary_assets")
        if handler is None:
            return {**summary, "status": "service_unavailable"}
        service = handler.service
        payload = service.get_bytes(asset_id)
        path = service.resolve_file(asset_id) if payload is None else None
        if payload is None and (path is None or not path.is_file()):
            return {**summary, "status": "missing_payload"}
        summary["payload_available"] = True
        summary["storage"] = "persistent_file" if path is not None else "memory"
        size = path.stat().st_size if path is not None else len(payload)
        digest = _sha256_file(path) if path is not None and size == expected_bytes else (
            hashlib.sha256(payload).hexdigest() if payload is not None else ""
        )
        summary["payload_integrity_verified"] = size == expected_bytes and asset.get("contentHash") == f"sha256:{digest}"
        if not summary["payload_integrity_verified"]:
            return {**summary, "status": "payload_mismatch"}
        fingerprint = record.get("sceneFingerprint")
        summary["motion_current"] = bool(fingerprint and fingerprint == motion_fingerprint(obj.scene))
        if not summary["motion_current"]:
            return {**summary, "status": "stale_motion" if fingerprint else "unverified_motion"}
        return {**summary, "status": "available", "ready_for_timeline": True}
    except (OSError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        return {**summary, "status": "invalid_manifest", "detail": str(exc)}


def _sensor_summary(obj: Any) -> dict[str, Any]:
    radar = obj.get_component("Radar")
    pose = SensorSpec.from_component(radar)
    target_id = str(getattr(radar, "snapshot_target_id", "") or "")
    result_present = bool(getattr(radar, "_animation_result", False))
    current_fingerprint = None
    if getattr(obj, "scene", None) is not None:
        try:
            current_fingerprint = _measurement_input_fingerprint(obj.scene, obj)
        except Exception:
            current_fingerprint = None
    recorded_fingerprint = str(getattr(radar, "_last_result_input_fingerprint", "") or "") or None
    return {
        "object_id": str(obj.id),
        "name": str(obj.name),
        "tag": str(getattr(obj, "tag", "") or ""),
        "position_m": _jsonable(pose.position),
        "look_at_point_m": _jsonable(pose.target),
        "up": _jsonable(pose.up),
        "fov_deg": float(radar.fov),
        "target_object_id": target_id or None,
        # Read-only current authored measurement identity.  Orchestrators use
        # this to distinguish harmless Radar UI/runtime-state normalization
        # from a real pose/FOV/target/waveform/scene-input change after export.
        "measurement_input_fingerprint": current_fingerprint,
        "animation": {
            "start_s": float(radar.t0),
            "duration_s": float(radar.animation_duration_s),
            "fps": float(radar.animation_fps),
            "view": str(radar.view),
        },
        "result": {
            "running": bool(getattr(radar, "_snapshot_running", False)),
            "present": result_present,
            "current": bool(result_present and recorded_fingerprint and recorded_fingerprint == current_fingerprint),
            "input_fingerprint": recorded_fingerprint,
            "run_id": str(getattr(radar, "_solver_run_id", "") or "") or None,
            "result_handle": str(getattr(radar, "_solver_result_handle", "") or "") or None,
            "export_path": str(getattr(radar, "animation_export_path", "") or "") or None,
            "status": str(getattr(radar, "animation_status", "") or ""),
            "replay_status": str(getattr(radar, "replay_status", "") or ""),
        },
        "physics_contract": {
            "components": ["los"],
            "max_depth": 0,
            "device": "cuda",
            "room_role": "occlusion only; no static clutter or extra bounces",
            "target_model": (
                "interval-visible stable skinned surface sites; uncalibrated equal-RCS "
                "shares are based on all declared sites and are not renormalized"
            ),
        },
    }


def _operation_result(receipt: dict[str, Any], obj: Any | None = None) -> dict[str, Any]:
    status = str(receipt.get("status") or "")
    completion_status = {
        "queued": "pending",
        "running": "pending",
        "simulated": "simulated_unverified",
        "verified": "verified",
        "replay_ready": "verified",
        "exported": "exported",
        "failed": "failed",
        "stale": "failed",
        "interrupted": "failed",
    }.get(status, "pending")
    result = {
        "ok": status not in {"failed", "stale", "interrupted"},
        "status": status,
        "completion_status": completion_status,
        "can_claim_success": completion_status in {"verified", "exported"},
        "operation_id": receipt.get("operation_id"),
        "scene_id": receipt.get("scene_id"),
        "radar_object_id": receipt.get("radar_object_id"),
        "input_fingerprint": receipt.get("input_fingerprint"),
        "created_at": receipt.get("created_at"),
        "updated_at": receipt.get("updated_at"),
        "run_id": receipt.get("run_id"),
        "result_handle": receipt.get("result_handle"),
        "verification": receipt.get("verification"),
        "error": receipt.get("error"),
        "next_step": receipt.get("next_step"),
    }
    if obj is not None:
        result["radar"] = _sensor_summary(obj)
    return result


def _sensor_configuration_state(obj: Any) -> dict[str, Any]:
    radar = obj.get_component("Radar")
    transform = obj.get_component("Transform")
    return {
        "target_object_id": str(radar.snapshot_target_id or ""),
        "start_s": float(radar.t0),
        "duration_s": float(radar.animation_duration_s),
        "fps": float(radar.animation_fps),
        "fov_deg": float(radar.fov),
        "position_m": _jsonable(transform.position),
        "rotation_rad": _jsonable(transform.rotation),
    }


def _configuration_matches(current: dict[str, Any], expected: dict[str, Any]) -> bool:
    if str(current.get("target_object_id") or "") != str(expected.get("target_object_id") or ""):
        return False
    for key in ("start_s", "duration_s", "fps", "fov_deg"):
        if not math.isclose(float(current.get(key, math.nan)), float(expected.get(key, math.nan)), abs_tol=1e-9):
            return False
    for key in ("position_m", "rotation_rad"):
        lhs = np.asarray(current.get(key), dtype=np.float64)
        rhs = np.asarray(expected.get(key), dtype=np.float64)
        if lhs.shape != rhs.shape or not np.allclose(lhs, rhs, rtol=0, atol=1e-8):
            return False
    return True


def _require_live_result_identity(obj: Any, receipt: dict[str, Any]) -> Any:
    radar = obj.get_component("Radar")
    live_run = str(getattr(radar, "_solver_run_id", "") or "")
    live_handle = str(getattr(radar, "_solver_result_handle", "") or "")
    if (
        not bool(getattr(radar, "_animation_result", False))
        or not live_run
        or not live_handle
        or live_run != str(receipt.get("run_id") or "")
        or live_handle != str(receipt.get("result_handle") or "")
    ):
        raise ToolError(
            "The live Radar component no longer owns the submitted native result.",
            code="result_handle_unavailable",
            detail={
                "expected_run_id": receipt.get("run_id"),
                "expected_result_handle": receipt.get("result_handle"),
                "live_run_id": live_run or None,
                "live_result_handle": live_handle or None,
            },
        )
    return radar


def _expected_result_evidence(scene: Any, radar: Any, native_preflight: dict[str, Any]) -> dict[str, Any]:
    measurement = _validate_interval(
        scene, radar, float(radar.t0), float(radar.animation_duration_s), float(radar.animation_fps),
    )
    topology = dict(native_preflight.get("topology") or {})
    declared_site_count = int(topology["declared_site_count"])
    active_site_count = int(topology["active_site_count"])
    occluded_site_count = int(topology["occluded_site_count"])
    active_site_ids = [int(value) for value in topology["active_site_ids"]]
    return {
        "scene_id": str(scene.scene_id),
        "model": str(native_preflight["model"]),
        "frame_count": int(measurement["frame_count"]),
        "cube_shape": list(measurement["cube_shape"]),
        "site_count": active_site_count,
        "declared_site_count": declared_site_count,
        "active_site_count": active_site_count,
        "occluded_site_count": occluded_site_count,
        "active_site_ids": active_site_ids,
        "rcs_policy": str(topology["rcs_policy"]),
        "rcs_per_site_m2": float(radar.snapshot_rcs_m2) / declared_site_count,
        "visibility_coverage": float(topology["visibility_coverage"]),
        "visibility_quality": str(topology["visibility_quality"]),
        "start_s": float(radar.t0),
        "fps": float(radar.animation_fps),
        "versions": {
            name: _package_version(name)
            for name in ("witwin-radar", "witwin-channel", "witwin")
        },
        "device_prefix": "cuda",
        "solver_completion_contract": "atomic_active_sites_no_zero_fill_v2",
    }


def _require_matching_operation(
    ctx: Any,
    operation_id: str,
    *,
    scene_id: str,
    radar_object_id: str,
    input_fingerprint: str | None = None,
) -> dict[str, Any]:
    receipt = _get_operation(ctx, operation_id)
    if receipt is None:
        raise ToolError(
            f"Radar operation not found: {operation_id}",
            code="missing_radar_operation",
            detail={"operation_id": operation_id},
        )
    expected = {"scene_id": str(scene_id), "radar_object_id": str(radar_object_id)}
    if input_fingerprint is not None:
        expected["input_fingerprint"] = str(input_fingerprint)
    conflicts = {
        key: {"stored": receipt.get(key), "requested": value}
        for key, value in expected.items()
        if str(receipt.get(key) or "") != str(value)
    }
    if conflicts:
        raise ToolError(
            "operation_id is already bound to different Radar inputs",
            code="operation_conflict",
            detail={"operation_id": operation_id, "conflicts": conflicts},
        )
    return receipt


async def _run_animation_job(
    ctx: Any,
    *,
    operation_id: str,
    scene: Any,
    obj: Any,
    input_fingerprint: str,
) -> None:
    receipt = _get_operation(ctx, operation_id) or {}
    receipt.update(status="running", next_step="wait_for_simulation")
    _put_operation(ctx, receipt)
    radar = obj.get_component("Radar")
    setattr(radar, "_agent_input_fingerprint", input_fingerprint)
    def _on_submitted(run_id: str) -> None:
        latest = _get_operation(ctx, operation_id) or receipt
        _put_operation(ctx, {
            **latest,
            "status": "running",
            "run_id": str(run_id),
            "next_step": "wait_for_simulation",
        })
    try:
        await radar._simulate_async(animation=True, on_submitted=_on_submitted)
        current = _measurement_input_fingerprint(scene, obj)
        if current != input_fingerprint:
            _put_operation(ctx, {
                **receipt,
                "status": "stale",
                "run_id": str(getattr(radar, "_solver_run_id", "") or "") or None,
                "result_handle": str(getattr(radar, "_solver_result_handle", "") or "") or None,
                "error": {
                    "code": "scene_changed_during_simulation",
                    "message": "The scene changed while the immutable solver copy was running.",
                    "expected_input_fingerprint": input_fingerprint,
                    "current_input_fingerprint": current,
                },
                "next_step": "plan_again",
            })
            return
        run_id = str(getattr(radar, "_solver_run_id", "") or "")
        result_handle = str(getattr(radar, "_solver_result_handle", "") or "")
        if not bool(getattr(radar, "_animation_result", False)) or not run_id or not result_handle:
            raise RuntimeError(
                "Native simulation returned without publishing an animation result, run id, and result handle."
            )
        setattr(radar, "_last_result_input_fingerprint", input_fingerprint)
        _put_operation(ctx, {
            **receipt,
            "status": "simulated",
            "run_id": run_id,
            "result_handle": result_handle,
            "error": None,
            "next_step": "verify_result",
        })
    except asyncio.CancelledError:
        latest = _get_operation(ctx, operation_id) or receipt
        _put_operation(ctx, {
            **latest,
            "status": "interrupted",
            "error": {"code": "task_cancelled", "message": "Radar orchestration task was cancelled."},
            "next_step": "inspect_before_retry",
        })
        raise
    except Exception as exc:
        latest = _get_operation(ctx, operation_id) or receipt
        _put_operation(ctx, {
            **latest,
            "status": "failed",
            "error": {
                "code": "native_simulation_failed",
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "next_step": "inspect_failure",
        })
    finally:
        if getattr(radar, "_agent_input_fingerprint", None) == input_fingerprint:
            delattr(radar, "_agent_input_fingerprint")


def _validate_interval(scene: Any, radar: Any, start_s: float, duration_s: float, fps: float) -> dict[str, Any]:
    values = np.asarray([start_s, duration_s, fps], dtype=np.float64)
    if not np.isfinite(values).all() or start_s < 0 or not 0 < duration_s <= 30 or not 1 <= fps <= 30:
        raise ToolError(
            "Animation requires start >= 0, duration (0,30] seconds, and FPS [1,30].",
            code="invalid_measurement_interval",
        )
    frame_count_float = duration_s * fps
    frame_count = int(round(frame_count_float))
    if frame_count < 2 or not math.isclose(frame_count_float, frame_count, abs_tol=1e-7):
        raise ToolError(
            "duration_s * fps must be an integer of at least two frames.",
            code="invalid_measurement_interval",
            detail={"frame_count": frame_count_float},
        )
    timeline = _timeline_summary(scene)
    if not timeline["available"] or start_s + duration_s > timeline["duration_s"] + 1e-7:
        raise ToolError(
            "Requested measurement extends beyond the baked Studio timeline.",
            code="timeline_interval_unavailable",
            detail={"timeline": timeline, "requested_end_s": start_s + duration_s},
        )
    if timeline["playing"] or timeline["recording"]:
        raise ToolError(
            "Pause timeline playback and recording before Radar preflight or simulation.",
            code="timeline_busy",
            detail={"timeline": timeline},
        )
    result_bytes = (
        frame_count * int(radar.num_tx) * int(radar.num_rx)
        * int(radar.chirp_per_frame) * int(radar.adc_samples) * 8
    )
    needed_bytes = max(512 * 1024**2, 3 * result_bytes)
    if result_bytes > MAX_RESULT_BYTES:
        raise ToolError(
            "Animation exceeds the 1536 MiB raw-cube budget; settings were not changed.",
            code="result_memory_budget_exceeded",
            detail={"result_bytes": result_bytes, "limit_bytes": MAX_RESULT_BYTES},
        )
    available_bytes = int(available_memory_bytes())
    if available_bytes < needed_bytes:
        raise ToolError(
            "Insufficient free host memory for the requested result; settings were not changed.",
            code="insufficient_host_memory",
            detail={"available_bytes": available_bytes, "required_headroom_bytes": needed_bytes},
        )
    return {
        "start_s": start_s,
        "duration_s": duration_s,
        "fps": fps,
        "frame_count": frame_count,
        "cube_shape": [
            frame_count,
            int(radar.num_tx),
            int(radar.num_rx),
            int(radar.chirp_per_frame),
            int(radar.adc_samples),
        ],
        "complex64_result_bytes": result_bytes,
        "required_host_headroom_bytes": needed_bytes,
        "available_host_memory_bytes": available_bytes,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _export_evidence(ctx: Any, path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ToolError("Radar export file is missing or empty.", code="export_verification_failed")
    server = getattr(getattr(ctx, "api", None), "server", None)
    project = Path(getattr(server, "default_scene_dir", ".")).resolve()
    resolved = path.resolve()
    expected_folder = (project / "results" / "radar-animation").resolve()
    if not resolved.is_relative_to(expected_folder):
        raise ToolError(
            "Radar export is outside the project results/radar-animation directory.",
            code="export_path_invalid",
            detail={"path": str(resolved), "expected_folder": str(expected_folder)},
        )
    return {
        "path": str(resolved),
        "project_relative_path": str(resolved.relative_to(project)).replace("\\", "/"),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
        "format": "native_npz",
    }


def _native_manifest_checks(
    manifest: dict[str, Any],
    receipt: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, bool]:
    """Validate native result evidence without performing RF/DSP work."""
    frame_count = int(expected.get("frame_count", 0) or 0)
    required_evidence_frames = max(2, int(math.ceil(frame_count * 0.1)))
    checks = {
        "input_fingerprint_matches": manifest.get("input_fingerprint") == receipt["input_fingerprint"],
        "scene_id_matches": str(manifest.get("scene_id") or "") == str(expected.get("scene_id") or ""),
        "model_matches": str(manifest.get("model") or "") == str(expected.get("model") or ""),
        "frame_count_matches": int(manifest.get("frame_count", -1)) == int(expected.get("frame_count", -2)),
        "cube_shape_matches": list(manifest.get("cube_shape") or []) == list(expected.get("cube_shape") or []),
        "site_count_matches": int(manifest.get("site_count", -1)) == int(expected.get("site_count", -2)),
        "declared_site_count_matches": int(manifest.get("declared_site_count", -1))
        == int(expected.get("declared_site_count", -2)),
        "active_site_count_matches": int(manifest.get("active_site_count", -1))
        == int(expected.get("active_site_count", -2)),
        "occluded_site_count_matches": int(manifest.get("occluded_site_count", -1))
        == int(expected.get("occluded_site_count", -2)),
        "site_partition_is_complete": (
            int(manifest.get("declared_site_count", -1))
            == int(manifest.get("active_site_count", -2))
            + int(manifest.get("occluded_site_count", -3))
        ),
        "active_site_ids_match": [int(value) for value in (manifest.get("active_site_ids") or [])]
        == [int(value) for value in (expected.get("active_site_ids") or [])],
        "rcs_policy_matches": str(manifest.get("rcs_policy") or "")
        == str(expected.get("rcs_policy") or ""),
        "rcs_per_site_matches": bool(np.isclose(
            float(manifest.get("rcs_per_site_m2", float("nan"))),
            float(expected.get("rcs_per_site_m2", float("nan"))),
            rtol=1e-12,
            atol=0.0,
        )),
        "visibility_coverage_matches": bool(np.isclose(
            float(manifest.get("visibility_coverage", float("nan"))),
            float(expected.get("visibility_coverage", float("nan"))),
            rtol=0.0,
            atol=1e-12,
        )),
        "visibility_quality_matches": str(manifest.get("visibility_quality") or "")
        == str(expected.get("visibility_quality") or ""),
        "versions_match": dict(manifest.get("versions") or {}) == dict(expected.get("versions") or {}),
        "all_finite": bool(manifest.get("all_finite")),
        "cuda_device": bool(re.fullmatch(r"cuda(?::\d+)?", str(manifest.get("device") or ""))),
        "timebase_finite": bool(manifest.get("timebase_finite")),
        "motion_arrays_finite": bool(manifest.get("motion_arrays_finite")),
        "no_zero_fill_or_dropped_active_sites": bool(
            manifest.get("no_zero_fill_or_dropped_active_sites")
        ),
        "nonzero_signal_present": bool(manifest.get("signal_nonzero"))
        and float(manifest.get("signal_abs_max") or 0.0) > 0.0
        and int(manifest.get("nonzero_return_frames") or 0) >= required_evidence_frames,
        "target_motion_present": int(manifest.get("moving_site_count") or 0) > 0
        and float(manifest.get("motion_speed_max_mps") or 0.0) >= 0.01
        and float(manifest.get("motion_position_extent_m") or 0.0) >= 0.01,
        "native_motion_coupling_present": int(
            manifest.get("coupled_dynamic_return_frames") or 0
        ) >= required_evidence_frames,
        "solver_completion_contract_matches": (
            str(manifest.get("solver_completion_contract") or "")
            == str(expected.get("solver_completion_contract") or "")
        ),
        "result_handle_present": bool(receipt.get("result_handle")),
        "run_id_present": bool(receipt.get("run_id")),
    }
    try:
        actual_times = np.asarray(manifest.get("times_s"), dtype=np.float64)
        expected_times = float(expected["start_s"]) + (
            np.arange(int(expected["frame_count"]), dtype=np.float64) / float(expected["fps"])
        )
        checks["timebase_matches"] = bool(
            actual_times.shape == expected_times.shape
            and np.isfinite(actual_times).all()
            and np.allclose(actual_times, expected_times, rtol=0, atol=1e-9)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        checks["timebase_matches"] = False
    return checks


def register(ctx: Any) -> None:
    owned_job_keys: set[str] = set()

    def _cancel_owned_jobs() -> None:
        for key in tuple(owned_job_keys):
            task = _ACTIVE_JOBS.get(key)
            if task is not None and not task.done():
                task.cancel()

    tracker = getattr(ctx, "track", None)
    if callable(tracker):
        from witwin_server.plugins.disposable import FunctionDisposable
        tracker(FunctionDisposable(_cancel_owned_jobs))

    @tool(
        name="runtime_diagnostics",
        description=(
            "Report the exact Radar, Channel, Torch and CUDA runtime without changing "
            "the scene or silently substituting a CPU backend."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        permission_tier="read",
        idempotent=True,
        tags=INSPECT_TAGS,
    )
    def _runtime_diagnostics(args: Dict[str, Any]) -> dict[str, Any]:
        runtime = _runtime_evidence()
        blockers = []
        for key in ("radar", "channel"):
            if not bool((runtime.get(key) or {}).get("import_ok")):
                blockers.append({"code": f"{key}_import_failed", "detail": runtime.get(key)})
        if not bool((runtime.get("torch") or {}).get("cuda_available")):
            blockers.append({"code": "cuda_unavailable", "detail": runtime.get("torch")})
        return {
            "ok": not blockers,
            "status": "ready" if not blockers else "blocked",
            "runtime": runtime,
            "blockers": blockers,
        }

    @tool(
        name="inspect_pipeline",
        description=(
            "Inspect Radar Settings, selected animated target, timeline and existing "
            "native result evidence in one exact Studio scene. This is read-only."
        ),
        input_schema={
            "type": "object",
            "required": ["scene_id"],
            "properties": {
                "scene_id": SCENE_ID_PROPERTY,
                "radar_object_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
        permission_tier="read",
        idempotent=True,
        tags=INSPECT_TAGS,
    )
    def _inspect_pipeline(args: Dict[str, Any]) -> dict[str, Any]:
        scene = _scene_from_args(ctx, args)
        sensors = _radar_objects(scene)
        summaries = [
            {**_sensor_summary(obj), "recorded_replay": _recorded_replay_summary(ctx, obj)}
            for obj in sensors
        ]
        requested = str(args.get("radar_object_id") or "").strip()
        selected = next((item for item in summaries if item["object_id"] == requested), None) if requested else (
            summaries[0] if len(summaries) == 1 else None
        )
        blockers: list[dict[str, Any]] = []
        if not sensors:
            blockers.append({"code": "missing_radar", "message": "No Radar Settings object exists."})
        elif selected is None:
            blockers.append({
                "code": "ambiguous_radar" if not requested else "missing_radar",
                "message": "Select one exact Radar object.",
                "radar_object_ids": [item["object_id"] for item in summaries],
            })
        if selected and not selected["target_object_id"]:
            blockers.append({"code": "missing_target", "message": "Radar has no animated target."})
        elif selected and scene.get_object(selected["target_object_id"]) is None:
            blockers.append({
                "code": "missing_target",
                "message": "Radar target no longer exists in this scene.",
                "target_object_id": selected["target_object_id"],
            })
        elif selected:
            target = scene.get_object(selected["target_object_id"])
            if target.get_component("SkinnedMesh") is None:
                blockers.append({
                    "code": "target_not_skinned",
                    "message": "Selected target has no SkinnedMesh and cannot drive the animation adapter.",
                    "target_object_id": selected["target_object_id"],
                })
        timeline = _timeline_summary(scene)
        if not timeline["available"] or timeline["duration_s"] <= 0:
            blockers.append({"code": "timeline_unavailable", "message": "Scene has no baked animation interval."})
        elif timeline["playing"] or timeline["recording"]:
            blockers.append({"code": "timeline_busy", "message": "Pause Timeline playback and recording."})
        return {
            "ok": True,
            "status": "configured" if not blockers else "blocked",
            "scene_id": str(scene.scene_id),
            "timeline": timeline,
            "radars": summaries,
            "selected_radar": selected,
            "blockers": blockers,
            "scene_input_fingerprint": _scene_input_fingerprint(scene),
            "scene_input_evidence": _scene_input_evidence(scene),
        }

    @tool(
        name="ensure_sensor",
        description=(
            "Idempotently create or configure one Studio Radar Settings object at an "
            "explicit world position (fixed mode), or choose a native-preflight-verified "
            "pose near the authored motion (automatic mode), and bind it to "
            "one animated target. This only edits Studio orchestration fields; it does "
            "not alter Radar or Channel algorithms."
        ),
        input_schema={
            "type": "object",
            "required": ["scene_id", "operation_id", "target_object_id"],
            "properties": {
                "scene_id": SCENE_ID_PROPERTY,
                "operation_id": {"type": "string", "minLength": 1},
                "radar_object_id": {"type": "string"},
                "target_object_id": {"type": "string", "minLength": 1},
                "placement_mode": {"type": "string", "enum": ["fixed", "automatic"], "default": "fixed"},
                "height_m": {"type": "number", "minimum": 0.2, "maximum": 3, "default": 1},
                "position_m": {
                    "type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                },
                "aim_point_m": {
                    "type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                },
                "start_s": {"type": "number", "minimum": 0, "default": 0},
                "duration_s": {"type": "number", "exclusiveMinimum": 0, "maximum": 30, "default": 5},
                "fps": {"type": "number", "minimum": 1, "maximum": 30, "default": 10},
                "fov_deg": {"type": "number", "minimum": 1, "maximum": 179, "default": 60},
                "reuse_existing": {"type": "boolean", "default": True},
            },
            "additionalProperties": False,
        },
        side_effects=True,
        requires_confirmation=True,
        permission_tier="scene_write",
        idempotent=True,
        durable_confirmation=True,
        timeout=300.0,
        tags=ENSURE_TAGS,
    )
    def _ensure_sensor(args: Dict[str, Any]) -> dict[str, Any]:
        scene = _scene_from_args(ctx, args)
        operation_id = str(args["operation_id"])
        ensure_payload = {
            key: _jsonable(value)
            for key, value in args.items()
            if key != "__context"
        }
        payload_hash = hashlib.sha256(json.dumps(
            ensure_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        existing_operation = _get_operation(ctx, operation_id)
        if existing_operation is not None:
            if (
                existing_operation.get("kind") != "ensure_sensor"
                or existing_operation.get("payload_hash") != payload_hash
            ):
                raise ToolError(
                    "operation_id is already bound to different Radar inputs",
                    code="operation_conflict",
                    detail={"operation_id": operation_id},
                )
            existing_obj = scene.get_object(str(existing_operation.get("radar_object_id") or ""))
            if existing_obj is None or existing_obj.get_component("Radar") is None:
                raise ToolError(
                    "The Radar created by this operation no longer exists.",
                    code="operation_state_stale",
                    detail={"operation_id": operation_id},
                )
            current_state = _sensor_configuration_state(existing_obj)
            expected_state = dict(existing_operation.get("configured_state") or {})
            if not expected_state or not _configuration_matches(current_state, expected_state):
                raise ToolError(
                    "The Radar configured by this operation has changed since the operation completed.",
                    code="operation_state_stale",
                    detail={
                        "operation_id": operation_id,
                        "expected": expected_state,
                        "current": current_state,
                    },
                )
            return {
                "ok": True,
                "status": "ready",
                "scene_id": str(scene.scene_id),
                "operation_id": operation_id,
                "created": False,
                "reused_operation": True,
                "radar": _sensor_summary(existing_obj),
                "scene_input_fingerprint": _scene_input_fingerprint(scene),
            }
        timeline = _timeline_summary(scene)
        if timeline["playing"] or timeline["recording"]:
            raise ToolError(
                "Pause timeline playback and recording before configuring Radar.",
                code="timeline_busy",
                detail={"timeline": timeline},
            )
        target_id = str(args["target_object_id"]).strip()
        target = scene.get_object(target_id)
        if target is None:
            raise ToolError(
                f"Radar target does not exist in scene: {target_id}",
                code="missing_target",
                detail={"scene_id": str(scene.scene_id), "target_object_id": target_id},
            )
        if target.get_component("SkinnedMesh") is None:
            raise ToolError(
                "Radar animation target must have a SkinnedMesh component.",
                code="target_not_skinned",
                detail={"scene_id": str(scene.scene_id), "target_object_id": target_id},
            )
        automatic = args.get("placement_mode", "fixed") == "automatic"
        if automatic and ("position_m" in args or "aim_point_m" in args):
            raise ToolError("Automatic placement cannot override explicit coordinates; use fixed mode.",
                            code="conflicting_sensor_placement")
        if not automatic and not all(key in args for key in ("position_m", "aim_point_m")):
            raise ToolError("Fixed placement requires position_m and aim_point_m.", code="missing_sensor_pose")
        requested = str(args.get("radar_object_id") or "").strip()
        created = False
        if requested:
            obj = _select_radar(scene, requested)
        else:
            sensors = _radar_objects(scene)
            owned = [obj for obj in sensors if str(getattr(obj, "tag", "")) == _PIPELINE_RADAR_TAG]
            if len(owned) == 1:
                obj = owned[0]
            elif len(owned) > 1 or len(sensors) > 1:
                raise ToolError(
                    "More than one Radar exists; specify radar_object_id explicitly.",
                    code="ambiguous_radar",
                    detail={"radar_object_ids": [str(sensor.id) for sensor in sensors]},
                )
            elif len(sensors) == 1 and bool(args.get("reuse_existing", True)):
                obj = sensors[0]
            elif sensors:
                raise ToolError(
                    "A user Radar already exists and reuse_existing is false.",
                    code="existing_radar_requires_selection",
                    detail={"radar_object_ids": [str(sensor.id) for sensor in sensors]},
                )
            else:
                obj = _settings_object("Agent Radar")
                obj.tag = _PIPELINE_RADAR_TAG
                created = True
        placement = None
        if automatic:
            from .sensor_placement import choose_sensor_placement
            placement = choose_sensor_placement(scene, args, radar_object_id=None if created else str(obj.id))
        position = _as_vector3(placement["position_m"] if placement else args["position_m"], "position_m")
        aim = _as_vector3(placement["aim_point_m"] if placement else args["aim_point_m"], "aim_point_m")
        rotation = _look_at_euler(position, aim)
        if created:
            scene.add_object(obj)
        radar = obj.get_component("Radar")
        transform = obj.get_component("Transform")
        if transform is None or getattr(transform, "parent", None) is not None:
            if created:
                scene.remove_object(str(obj.id))
            raise ToolError(
                "Agent Radar must be a root object because position_m and aim_point_m are world coordinates.",
                code="parented_radar_unsupported",
                detail={"radar_object_id": str(obj.id)},
            )
        previous = {
            "snapshot_target_id": radar.snapshot_target_id,
            "t0": radar.t0,
            "animation_duration_s": radar.animation_duration_s,
            "animation_fps": radar.animation_fps,
            "fov": radar.fov,
            "view": radar.view,
            "static_clutter_removal": radar.static_clutter_removal,
            "show_cfar": radar.show_cfar,
            "position": _jsonable(transform.position),
            "rotation": _jsonable(transform.rotation),
        }
        try:
            radar.snapshot_target_id = target_id
            radar.t0 = float(args.get("start_s", 0.0))
            radar.animation_duration_s = float(args.get("duration_s", 5.0))
            radar.animation_fps = float(args.get("fps", 10.0))
            radar.fov = float(args.get("fov_deg", 60.0))
            # The existing animation adapter requires these review controls.
            radar.view = "range_doppler"
            radar.static_clutter_removal = False
            radar.show_cfar = False
            if not scene.update_transform(str(obj.id), position=position.tolist(), rotation=rotation):
                raise ToolError("Could not update Radar transform.", code="sensor_configuration_failed")
        except Exception:
            if created:
                scene.remove_object(str(obj.id))
            else:
                for key in (
                    "snapshot_target_id", "t0", "animation_duration_s", "animation_fps",
                    "fov", "view", "static_clutter_removal", "show_cfar",
                ):
                    setattr(radar, key, previous[key])
                scene.update_transform(
                    str(obj.id), position=previous["position"], rotation=previous["rotation"],
                )
            raise
        result = {
            "ok": True,
            "status": "ready",
            "scene_id": str(scene.scene_id),
            "operation_id": operation_id,
            "created": created,
            "reused_operation": False,
            "placement": placement or {"mode": "fixed", "position_m": position.tolist(), "aim_point_m": aim.tolist()},
            "radar": _sensor_summary(obj),
            "scene_input_fingerprint": _scene_input_fingerprint(scene),
        }
        _put_operation(ctx, {
            "operation_id": operation_id,
            "kind": "ensure_sensor",
            "payload_hash": payload_hash,
            "scene_id": str(scene.scene_id),
            "radar_object_id": str(obj.id),
            "status": "ready",
            "created_at": _utc_now(),
            "configured_state": _sensor_configuration_state(obj),
            "next_step": "plan_animation_measurement",
        })
        return result

    @tool(
        name="plan_animation_measurement",
        description=(
            "Read-only preflight for a real Radar 0.3 CUDA animation measurement. "
            "It validates the exact target, timeline interval, frame/cube shape, host "
            "memory and current solver settings without changing FPS or duration."
        ),
        input_schema={
            "type": "object",
            "required": ["scene_id"],
            "properties": {
                "scene_id": SCENE_ID_PROPERTY,
                "radar_object_id": {"type": "string"},
                "start_s": {"type": "number", "minimum": 0},
                "duration_s": {"type": "number", "exclusiveMinimum": 0, "maximum": 30},
                "fps": {"type": "number", "minimum": 1, "maximum": 30},
            },
            "additionalProperties": False,
        },
        permission_tier="read",
        idempotent=True,
        tags=PLAN_TAGS,
    )
    def _plan_animation_measurement(args: Dict[str, Any]) -> dict[str, Any]:
        scene = _scene_from_args(ctx, args)
        obj = _select_radar(scene, str(args.get("radar_object_id") or ""))
        radar = obj.get_component("Radar")
        target_id = str(radar.snapshot_target_id or "").strip()
        target = scene.get_object(target_id) if target_id else None
        if target is None or target_id == str(obj.id):
            raise ToolError(
                "Radar must reference one separate existing animated target.",
                code="missing_target",
                detail={"target_object_id": target_id or None},
            )
        from .adapter.snapshot import validate_options
        try:
            validate_options(radar)
        except Exception as exc:
            raise ToolError(str(exc), code="unsupported_radar_settings") from exc
        start_s = float(args.get("start_s", radar.t0))
        duration_s = float(args.get("duration_s", radar.animation_duration_s))
        fps = float(args.get("fps", radar.animation_fps))
        configured = {
            "start_s": float(radar.t0),
            "duration_s": float(radar.animation_duration_s),
            "fps": float(radar.animation_fps),
        }
        requested = {"start_s": start_s, "duration_s": duration_s, "fps": fps}
        # Studio component numerics are stored as float32.  Compare authored
        # values at that storage precision so a value such as 1.4 does not
        # become a false configuration mismatch after round-tripping through
        # the component, while still rejecting a meaningful timing change.
        if any(not math.isclose(requested[key], configured[key], rel_tol=0.0, abs_tol=1e-6)
               for key in configured):
            raise ToolError(
                "Preflight parameters differ from the authored Radar component; call ensure_sensor first.",
                code="radar_configuration_mismatch",
                detail={"configured": configured, "requested": requested},
            )
        measurement = _validate_interval(scene, radar, start_s, duration_s, fps)
        try:
            from witwin_server.features.solvers.scene_ref import load_scene_ref, make_scene_ref
            from .adapter.animation import animation_preflight, animation_request
            request = animation_request(radar)
            solver_scene = load_scene_ref(copy.deepcopy(make_scene_ref(scene)))
            native_preflight = animation_preflight(solver_scene, request)
        except Exception as exc:
            detail = getattr(exc, "detail", None)
            code = "radar_topology_unreachable" if detail else "target_motion_unavailable"
            raise ToolError(str(exc), code=code, detail=detail) from exc
        runtime = _runtime_evidence()
        cuda_ready = bool((runtime.get("torch") or {}).get("cuda_available"))
        if not cuda_ready:
            raise ToolError(
                "Radar animation requires CUDA and no CPU fallback is permitted.",
                code="cuda_unavailable",
                detail={"runtime": runtime},
            )
        input_fingerprint = _measurement_input_fingerprint(scene, obj)
        _remember_preflight(
            ctx,
            scene_id=str(scene.scene_id),
            radar_object_id=str(obj.id),
            fingerprint=input_fingerprint,
            native_preflight=native_preflight,
        )
        return {
            "ok": True,
            "status": "ready",
            "scene_id": str(scene.scene_id),
            "radar": _sensor_summary(obj),
            "target": {
                "object_id": target_id,
                "name": str(target.name),
                "declared_site_count": int(native_preflight["topology"]["declared_site_count"]),
                "active_site_count": int(native_preflight["topology"]["active_site_count"]),
                "occluded_site_count": int(native_preflight["topology"]["occluded_site_count"]),
                "visibility_coverage": float(native_preflight["topology"]["visibility_coverage"]),
                "visibility_quality": str(native_preflight["topology"]["visibility_quality"]),
                "moving_object_ids": native_preflight["moving_object_ids"],
            },
            "measurement": measurement,
            "native_preflight": native_preflight,
            "runtime": runtime,
            "input_fingerprint": input_fingerprint,
            "scene_input_fingerprint": _scene_input_fingerprint(scene),
            "scene_input_evidence": _scene_input_evidence(scene),
            "completion_requirements": {
                "actual_frame_count": measurement["frame_count"],
                "all_frames_finite": True,
                "device": "cuda",
                "result_handle_required": True,
                "no_zero_fill_for_active_sites": True,
                "stable_active_site_ids_for_full_interval": True,
                "occluded_declared_sites_reported": True,
                "no_visible_rcs_renormalization": True,
            },
            "algorithm_unchanged": True,
        }

    @tool(
        name="submit_animation_measurement",
        description=(
            "Submit the already-preflighted Studio animation to the existing native "
            "Radar 0.3 CUDA solver. Returns immediately with an operation receipt; "
            "poll get_simulation and then call verify_result. No fallback is allowed."
        ),
        input_schema={
            "type": "object",
            "required": ["scene_id", "radar_object_id", "operation_id", "expected_input_fingerprint"],
            "properties": {
                "scene_id": SCENE_ID_PROPERTY,
                "radar_object_id": {"type": "string", "minLength": 1},
                "operation_id": {"type": "string", "minLength": 1, "maxLength": 160},
                "expected_input_fingerprint": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
            "additionalProperties": False,
        },
        side_effects=True,
        requires_confirmation=True,
        permission_tier="execute",
        idempotent=True,
        durable_confirmation=True,
        tags=SUBMIT_TAGS,
    )
    async def _submit_animation_measurement(args: Dict[str, Any]) -> dict[str, Any]:
        scene = _scene_from_args(ctx, args)
        obj = _select_radar(scene, str(args["radar_object_id"]))
        operation_id = str(args["operation_id"])
        expected = str(args["expected_input_fingerprint"])
        current = _measurement_input_fingerprint(scene, obj)
        if current != expected:
            raise ToolError(
                "Scene or Radar inputs changed after preflight; simulation was not started.",
                code="stale_preflight",
                detail={"expected_input_fingerprint": expected, "current_input_fingerprint": current},
            )
        existing = _get_operation(ctx, operation_id)
        if existing is not None:
            _require_matching_operation(
                ctx,
                operation_id,
                scene_id=str(scene.scene_id),
                radar_object_id=str(obj.id),
                input_fingerprint=expected,
            )
            return _operation_result(existing, obj)
        radar = obj.get_component("Radar")
        from .adapter.snapshot import validate_options
        try:
            validate_options(radar)
            _validate_interval(
                scene,
                radar,
                float(radar.t0),
                float(radar.animation_duration_s),
                float(radar.animation_fps),
            )
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(str(exc), code="radar_configuration_invalid") from exc
        native_preflight = _consume_preflight(
            ctx,
            scene_id=str(scene.scene_id),
            radar_object_id=str(obj.id),
            fingerprint=expected,
        )
        if native_preflight is None:
            raise ToolError(
                "No matching native topology preflight is available. Call "
                "plan_animation_measurement again before submitting.",
                code="preflight_receipt_missing",
                detail={
                    "scene_id": str(scene.scene_id),
                    "radar_object_id": str(obj.id),
                    "input_fingerprint": expected,
                },
            )
        receipt = _put_operation(ctx, {
            "operation_id": operation_id,
            "scene_id": str(scene.scene_id),
            "radar_object_id": str(obj.id),
            "input_fingerprint": expected,
            "status": "queued",
            "created_at": _utc_now(),
            "run_id": None,
            "result_handle": None,
            "verification": None,
            "expected_evidence": _expected_result_evidence(scene, radar, native_preflight),
            "error": None,
            "next_step": "wait_for_simulation",
        })
        key = _job_key(ctx, operation_id)
        task = asyncio.create_task(_run_animation_job(
            ctx,
            operation_id=operation_id,
            scene=scene,
            obj=obj,
            input_fingerprint=expected,
        ))
        _ACTIVE_JOBS[key] = task
        owned_job_keys.add(key)
        def _finish_job(done: asyncio.Task, job_key: str = key) -> None:
            _ACTIVE_JOBS.pop(job_key, None)
            owned_job_keys.discard(job_key)
            if not done.cancelled():
                # Retrieve any unexpected exception (for example a persistence
                # failure) so asyncio never silently discards orchestration errors.
                done.exception()
        task.add_done_callback(_finish_job)
        return _operation_result(receipt, obj)

    @tool(
        name="get_simulation",
        description=(
            "Observe one submitted Radar animation operation. A missing in-memory task "
            "after restart is reported as interrupted, never guessed to have succeeded."
        ),
        input_schema={
            "type": "object",
            "required": ["scene_id", "radar_object_id", "operation_id"],
            "properties": {
                "scene_id": SCENE_ID_PROPERTY,
                "radar_object_id": {"type": "string", "minLength": 1},
                "operation_id": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        permission_tier="read",
        idempotent=True,
        tags=STATUS_TAGS,
    )
    def _get_simulation(args: Dict[str, Any]) -> dict[str, Any]:
        scene = _scene_from_args(ctx, args)
        obj = _select_radar(scene, str(args["radar_object_id"]))
        operation_id = str(args["operation_id"])
        receipt = _require_matching_operation(
            ctx,
            operation_id,
            scene_id=str(scene.scene_id),
            radar_object_id=str(obj.id),
        )
        if receipt.get("status") in {"queued", "running"}:
            task = _ACTIVE_JOBS.get(_job_key(ctx, operation_id))
            if task is None:
                receipt = _put_operation(ctx, {
                    **receipt,
                    "status": "interrupted",
                    "error": {
                        "code": "worker_state_lost",
                        "message": "The backend no longer owns this in-flight task; inspect before retrying.",
                    },
                    "next_step": "inspect_before_retry",
                })
        return _operation_result(receipt, obj)

    @tool(
        name="cancel_simulation",
        description=(
            "Cancel one owned in-flight Radar animation operation and wait until its "
            "orchestration task has published an interrupted receipt. Completed native "
            "results are never deleted or rewritten."
        ),
        input_schema={
            "type": "object",
            "required": ["scene_id", "radar_object_id", "operation_id"],
            "properties": {
                "scene_id": SCENE_ID_PROPERTY,
                "radar_object_id": {"type": "string", "minLength": 1},
                "operation_id": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        side_effects=True,
        requires_confirmation=True,
        permission_tier="execute",
        idempotent=True,
        tags=CANCEL_TAGS,
    )
    async def _cancel_simulation(args: Dict[str, Any]) -> dict[str, Any]:
        scene = _scene_from_args(ctx, args)
        obj = _select_radar(scene, str(args["radar_object_id"]))
        operation_id = str(args["operation_id"])
        receipt = _require_matching_operation(
            ctx,
            operation_id,
            scene_id=str(scene.scene_id),
            radar_object_id=str(obj.id),
        )
        if receipt.get("status") not in {"queued", "running"}:
            return _operation_result(receipt, obj)
        key = _job_key(ctx, operation_id)
        task = _ACTIVE_JOBS.get(key)
        if task is None:
            receipt = _put_operation(ctx, {
                **receipt,
                "status": "interrupted",
                "error": {
                    "code": "worker_state_lost",
                    "message": "The backend no longer owns this in-flight task.",
                },
                "next_step": "inspect_before_retry",
            })
            return _operation_result(receipt, obj)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        receipt = _require_matching_operation(
            ctx,
            operation_id,
            scene_id=str(scene.scene_id),
            radar_object_id=str(obj.id),
        )
        if receipt.get("status") in {"queued", "running"}:
            receipt = _put_operation(ctx, {
                **receipt,
                "status": "interrupted",
                "error": {"code": "task_cancelled", "message": "Radar orchestration task was cancelled."},
                "next_step": "inspect_before_retry",
            })
        return _operation_result(receipt, obj)

    @tool(
        name="verify_result",
        description=(
            "Verify the completed native result against the preflight fingerprint, "
            "actual frame count, cube shape, finite CUDA data and package provenance. "
            "Only this tool may promote a simulated operation to verified."
        ),
        input_schema={
            "type": "object",
            "required": ["scene_id", "radar_object_id", "operation_id"],
            "properties": {
                "scene_id": SCENE_ID_PROPERTY,
                "radar_object_id": {"type": "string", "minLength": 1},
                "operation_id": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        permission_tier="read",
        idempotent=True,
        tags=VERIFY_TAGS,
        timeout=180.0,
    )
    async def _verify_result(args: Dict[str, Any]) -> dict[str, Any]:
        scene = _scene_from_args(ctx, args)
        obj = _select_radar(scene, str(args["radar_object_id"]))
        operation_id = str(args["operation_id"])
        receipt = _require_matching_operation(
            ctx,
            operation_id,
            scene_id=str(scene.scene_id),
            radar_object_id=str(obj.id),
        )
        previous_status = receipt.get("status")
        if previous_status not in {"simulated", "verified", "replay_ready", "exported"}:
            raise ToolError(
                "Radar operation is not ready for result verification.",
                code="simulation_not_complete",
                detail={"status": receipt.get("status"), "operation_id": operation_id},
            )
        def observe(updated: dict[str, Any]) -> dict[str, Any]:
            # The initial check promotes a completed solve. Rechecking a durable
            # success receipt is read-only, including when today's state is stale.
            if previous_status == "simulated":
                updated = _put_operation(ctx, updated)
            return _operation_result(updated, obj)

        current = _measurement_input_fingerprint(scene, obj)
        if current != receipt["input_fingerprint"]:
            return observe({
                **receipt,
                "status": "stale",
                "error": {
                    "code": "scene_changed_after_simulation",
                    "message": "Current scene inputs no longer match the simulated inputs.",
                    "current_input_fingerprint": current,
                },
                "next_step": "plan_again",
            })
        radar = _require_live_result_identity(obj, receipt)
        try:
            manifest = await asyncio.to_thread(radar._query_result, "animation_manifest", {})
        except Exception as exc:
            raise ToolError(str(exc), code="result_verification_failed") from exc
        expected = dict(receipt.get("expected_evidence") or {})
        checks = _native_manifest_checks(manifest, receipt, expected)
        if not all(checks.values()):
            return observe({
                **receipt,
                "status": "failed",
                "verification": {"passed": False, "checks": checks, "manifest": manifest},
                "error": {
                    "code": "result_evidence_mismatch",
                    "message": "Native result did not satisfy every preflight completion requirement.",
                },
                "next_step": "inspect_failure",
            })
        return observe({
            **receipt,
            # Verification is repeatable, but never trusts an old receipt alone.
            # Keep later replay/export milestones while rechecking the live result.
            "status": "verified" if previous_status == "simulated" else previous_status,
            "verification": {"passed": True, "checks": checks, "manifest": manifest},
            "error": None,
            "next_step": (
                "prepare_replay_or_export" if previous_status == "simulated"
                else receipt.get("next_step", "prepare_replay_or_export")
            ),
        })

    @tool(
        name="prepare_replay",
        description=(
            "Prepare the existing native Range Profile and Range Doppler recording "
            "for synchronized Studio Timeline playback. It reuses official processing "
            "and does not implement alternate DSP."
        ),
        input_schema={
            "type": "object",
            "required": ["scene_id", "radar_object_id", "operation_id"],
            "properties": {
                "scene_id": SCENE_ID_PROPERTY,
                "radar_object_id": {"type": "string", "minLength": 1},
                "operation_id": {"type": "string", "minLength": 1},
                "tx_index": {"type": "integer", "minimum": 0, "default": 0},
                "rx_index": {"type": "integer", "minimum": 0, "default": 0},
            },
            "additionalProperties": False,
        },
        side_effects=True,
        requires_confirmation=True,
        permission_tier="soft_write",
        idempotent=True,
        durable_confirmation=True,
        timeout=180.0,
        tags=REPLAY_TAGS,
    )
    async def _prepare_replay(args: Dict[str, Any]) -> dict[str, Any]:
        scene = _scene_from_args(ctx, args)
        obj = _select_radar(scene, str(args["radar_object_id"]))
        receipt = _require_matching_operation(
            ctx,
            str(args["operation_id"]),
            scene_id=str(scene.scene_id),
            radar_object_id=str(obj.id),
        )
        if receipt.get("status") not in {"verified", "replay_ready", "exported"}:
            raise ToolError("Verify the native result before preparing replay.", code="result_not_verified")
        if _measurement_input_fingerprint(scene, obj) != receipt["input_fingerprint"]:
            raise ToolError("Scene inputs changed after simulation.", code="stale_result")
        radar = _require_live_result_identity(obj, receipt)
        radar.tx_index = int(args.get("tx_index", 0))
        radar.rx_index = int(args.get("rx_index", 0))
        try:
            status = await radar.prepare_synchronized_replay()
        except Exception as exc:
            raise ToolError(str(exc), code="replay_preparation_failed") from exc
        recorded = _recorded_replay_summary(ctx, obj)
        if (
            recorded.get("status") != "available"
            or recorded.get("payload_available") is not True
            or recorded.get("payload_integrity_verified") is not True
            or recorded.get("motion_current") is not True
            or recorded.get("ready_for_timeline") is not True
        ):
            raise ToolError(
                "Synchronized replay was not published as a current, verified numeric asset.",
                code="replay_preparation_unverified",
                detail={
                    "component_status": str(status),
                    "recorded_replay": recorded,
                },
            )
        receipt = _put_operation(ctx, {
            **receipt,
            "status": "replay_ready" if receipt.get("status") != "exported" else "exported",
            "replay": {
                "ready": True,
                "status": str(status),
                "tx_index": radar.tx_index,
                "rx_index": radar.rx_index,
                "recorded_replay": recorded,
            },
            "next_step": "export_result",
        })
        return _operation_result(receipt, obj)

    @tool(
        name="export_result",
        description=(
            "Export one verified native Radar animation NPZ into the current project's "
            "results/radar-animation directory. Repeating the same export operation "
            "returns the existing file and never overwrites user data."
        ),
        input_schema={
            "type": "object",
            "required": ["scene_id", "radar_object_id", "operation_id", "export_operation_id"],
            "properties": {
                "scene_id": SCENE_ID_PROPERTY,
                "radar_object_id": {"type": "string", "minLength": 1},
                "operation_id": {"type": "string", "minLength": 1},
                "export_operation_id": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        side_effects=True,
        requires_confirmation=True,
        permission_tier="file_write",
        idempotent=True,
        durable_confirmation=True,
        timeout=180.0,
        tags=EXPORT_TAGS,
    )
    async def _export_result(args: Dict[str, Any]) -> dict[str, Any]:
        scene = _scene_from_args(ctx, args)
        obj = _select_radar(scene, str(args["radar_object_id"]))
        receipt = _require_matching_operation(
            ctx,
            str(args["operation_id"]),
            scene_id=str(scene.scene_id),
            radar_object_id=str(obj.id),
        )
        if receipt.get("status") not in {"verified", "replay_ready", "exported"}:
            raise ToolError("Verify the native result before export.", code="result_not_verified")
        if _measurement_input_fingerprint(scene, obj) != receipt["input_fingerprint"]:
            raise ToolError("Scene inputs changed after simulation.", code="stale_result")
        _require_live_result_identity(obj, receipt)
        export_id = str(args["export_operation_id"])
        # One native result has one export critical section. Different client
        # export ids must not race the same Radar component or lose journal data.
        lock_key = f"{_job_key(ctx, str(args['operation_id']))}:export"
        lock = _EXPORT_LOCKS.setdefault(lock_key, asyncio.Lock())
        async with lock:
            # Re-read after locking: an identical concurrent request may have
            # already completed and persisted its export receipt.
            receipt = _require_matching_operation(
                ctx,
                str(args["operation_id"]),
                scene_id=str(scene.scene_id),
                radar_object_id=str(obj.id),
            )
            exports = dict(receipt.get("exports") or {})
            prior = exports.get(export_id)
            if isinstance(prior, dict):
                path = Path(str(prior.get("path") or ""))
                if path.is_file() and _sha256_file(path) == prior.get("sha256"):
                    return {**_operation_result(receipt, obj), "export": prior, "reused": True}
                raise ToolError(
                    "The recorded export receipt no longer matches an existing file.",
                    code="export_receipt_stale",
                    detail={"export": prior},
                )
            radar = _require_live_result_identity(obj, receipt)
            current_path = Path(str(getattr(radar, "animation_export_path", "") or ""))
            if current_path.is_file():
                export = _export_evidence(ctx, current_path)
            else:
                try:
                    await radar.export_animation_result()
                except Exception as exc:
                    raise ToolError(str(exc), code="result_export_failed") from exc
                current_path = Path(str(radar.animation_export_path or ""))
                export = _export_evidence(ctx, current_path)
            export = {
                **export,
                "input_fingerprint": receipt.get("input_fingerprint"),
                "run_id": receipt.get("run_id"),
                "result_handle": receipt.get("result_handle"),
            }
            exports[export_id] = export
            receipt = _put_operation(ctx, {
                **receipt,
                "status": "exported",
                "exports": exports,
                "next_step": "complete",
            })
            return {**_operation_result(receipt, obj), "export": export, "reused": False}

    @tool(
        name="describe_pipeline_contract",
        description=(
            "Return the versioned Radar planning and result-action contracts for "
            "orchestrators and plan editors. Does not inspect or edit a scene."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        permission_tier="read",
        idempotent=True,
    )
    def _describe_pipeline_contract(_args: Dict[str, Any]) -> dict[str, Any]:
        actions = {
            "configure": _ensure_sensor,
            "preflight": _plan_animation_measurement,
            "simulate": _submit_animation_measurement,
            "verify": _verify_result,
            "replay": _prepare_replay,
            "export": _export_result,
        }
        return {
            "schema": "witwin.radar.pipeline-capabilities.v1",
            "actions": {
                name: {
                    "tool_name": value.name,
                    "input_schema": copy.deepcopy(value.input_schema),
                    "permission_tier": value.permission_tier,
                    "requires_confirmation": value.requires_confirmation,
                    "idempotent": value.idempotent,
                }
                for name, value in actions.items()
            },
            "dependencies": {
                "simulate": ["configure", "preflight"],
                "verify": ["simulate"],
                "replay": ["verify"],
                "export": ["verify"],
            },
        }

    for value in (
        _runtime_diagnostics,
        _inspect_pipeline,
        _ensure_sensor,
        _plan_animation_measurement,
        _submit_animation_measurement,
        _get_simulation,
        _cancel_simulation,
        _verify_result,
        _prepare_replay,
        _export_result,
        _describe_pipeline_contract,
    ):
        ctx.tools.register(value)
