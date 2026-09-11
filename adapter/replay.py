"""Numeric Studio widgets and recorded scene-time playback; no RF/DSP equations."""
from hashlib import sha256
import json

import numpy as np

MAX_PREVIEW_BYTES = 128 * 1024**2


def motion_fingerprint(scene):
    """Identify authored motion only; this is not a full geometry fingerprint."""
    value = scene.timeline_manager.clip.to_dict()
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _stable_fingerprint(value):
    """Hash persisted provenance without relying on object identity or dict order."""
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def scene_geometry_fingerprint(scene):
    """Fingerprint authored scene geometry independently from baked motion.

    Radar's native input fingerprint remains the authoritative solver guard.  This
    smaller, user-visible digest lets a saved replay say *why* it became stale
    after furniture or room geometry changed, without pretending a Timeline seek
    is a geometry edit.
    """
    scene_data = scene.to_dict()
    scene_data.pop("timeline", None)
    scene_data.pop("timeline_manager", None)
    objects = scene_data.get("objects")
    if isinstance(objects, dict):
        values = objects.values()
    elif isinstance(objects, list):
        values = objects
    else:
        values = []
    geometry = []
    for raw in values:
        item = dict(raw)
        components = item.get("components")
        if isinstance(components, dict):
            # Radar configuration is covered by its own fingerprint below.  Do
            # not make a replay look geometry-stale merely because its status
            # text changed while it was being prepared.
            item["components"] = {
                name: value for name, value in components.items() if name != "Radar"
            }
        geometry.append(item)
    return _stable_fingerprint(geometry)


def radar_pose_config_fingerprint(component):
    owner = getattr(component, "owner", None)
    owner_data = owner.to_dict() if owner is not None else {}
    radar_data = component.to_properties_dict() if hasattr(component, "to_properties_dict") else {}
    return _stable_fingerprint({"owner": owner_data, "radar": radar_data})


def _recording_source_descriptor(component, record, asset_id):
    """Create the persistent DataSource contract stored alongside the Figure."""
    owner = getattr(component, "owner", None)
    source_id = f"radar.{getattr(owner, 'id', 'unknown')}.replay.{asset_id.rsplit('-', 1)[-1][:16]}"
    return {
        "sourceId": source_id,
        "mode": "timeline",
        "kind": "radar.replay",
        "label": "Radar Result & Replay",
        "defaultChannel": "range_doppler",
        "owner": {
            "kind": "component",
            "componentName": "Radar",
            "fieldName": "signal_figure",
            "objectId": getattr(owner, "id", ""),
            "sceneId": component.scene.scene_id,
        },
        "retention": "persistent",
        "timebase": {
            "unit": "seconds",
            "times": list(record["timesS"]),
            "fps": record["fps"],
        },
        "channels": [
            {"channelId": "range_profile", "label": "Range Profile", "dtype": "float32",
             "shape": [len(record["rangeM"])], "semantic": "radar.range_profile"},
            {"channelId": "range_doppler", "label": "Range-Doppler", "dtype": "float32",
             "shape": [len(record["velocityMps"]), len(record["rangeM"])], "semantic": "radar.range_doppler"},
        ],
        "presentation": {
            "replay": True,
            "followTimeline": True,
            "frameCount": len(record["timesS"]),
            "assetId": asset_id,
        },
        "metadata": {
            "resultId": record["resultId"],
            "sceneGeometryFingerprint": record["sceneGeometryFingerprint"],
            "motionFingerprint": record.get("sceneFingerprint"),
            "radarPoseConfigFingerprint": record["radarPoseConfigFingerprint"],
            "stale": False,
        },
    }


def numeric_plot(payload):
    from witwin_server.core.components.core.plot import PlotData
    title = payload.get("title", "Native Radar result")
    ranges = payload["range_m"]
    if payload["view"] == "range_doppler":
        values = np.asarray(payload["magnitude"])
        data = {"values": payload["magnitude"], "x": ranges, "y": payload["velocity_mps"],
                "origin": "lower", "range": {"min": 0., "max": max(float(values.max()), 1e-30),
                                               "label": "Linear amplitude (uncalibrated)"}}
        return PlotData("imshow", data, title=title, xlabel="Range (m)", ylabel="Closing velocity (m/s)")
    if payload["view"] == "range_profile":
        series = [{"x": ranges, "y": payload["magnitude"], "label": "Magnitude"}]
    elif payload["view"] == "range_spectrum":
        series = [{"x": ranges, "y": payload[key], "label": key.title()} for key in ("real", "imag")]
    else:
        raise ValueError("Numeric review supports Range Profile/Spectrum/Doppler only.")
    values = np.asarray([row["y"] for row in series])
    lo, hi = min(0., float(values.min())), max(0., float(values.max()))
    margin = max(hi - lo, 1e-30) * .05
    return PlotData("line", {"series": series, "xlim": [ranges[0], ranges[-1]], "ylim": [lo - margin, hi + margin]},
                    title=title, xlabel="Range (m)", ylabel="Linear amplitude (uncalibrated)")


def build_recording(result, tx=0, rx=0):
    """Pack official RP/RD magnitudes once, for local frontend playback.

    Each little-endian float32 frame contains R profile samples followed by
    D*R RD samples (ascending velocity rows). No images or frame interpolation.
    """
    from witwin.radar.processing import ProcessingCube, range_profile, range_doppler_map
    axes = result.axes
    metadata = result.metadata if hasattr(result, "metadata") else result.producer
    if not 0 <= tx < axes.num_tx or not 0 <= rx < axes.num_rx:
        raise ValueError("Requested antenna is outside the recorded array.")
    times = np.asarray(result.times_s, dtype=np.float64)
    if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Replay requires at least two finite increasing recorded timestamps.")
    nr, nd = len(axes.range_m), len(axes.velocity_mps)
    stride = nr * (1 + nd)
    if len(times) * stride * 4 > MAX_PREVIEW_BYTES:
        raise ValueError("Numeric replay exceeds 128 MiB; no frames or bins were discarded.")
    values = np.empty((len(times), stride), dtype="<f4")
    for index in range(len(times)):
        profile = range_profile(ProcessingCube(result.cube[index], axes), window="rectangular", remove_dc=False)
        rd = range_doppler_map(profile, window="hann")
        values[index, :nr] = profile.data[tx, rx, 0].abs().detach().cpu().numpy()
        values[index, nr:] = rd.data[tx, rx].abs().detach().cpu().numpy().reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite native processing result; no substitute data published.")
    # Last sample is a frame start. Duration describes the measured frame interval.
    end = float(times[-1] + np.diff(times)[-1])
    if "duration_s" in metadata:
        end = float(times[0] + metadata["duration_s"])
    if not np.isfinite(end) or end <= times[-1]:
        raise ValueError("Invalid recorded end time.")
    fps = float(1.0 / np.median(np.diff(times)))
    record = {"schema": 1, "sceneId": metadata.get("scene_id", ""), "timesS": times.tolist(),
              "rangeM": axes.range_m.tolist(), "velocityMps": axes.velocity_mps.tolist(),
              "frameStride": stride, "endTimeS": end, "profileMax": max(float(values[:, :nr].max()), 1e-30),
              "dopplerMax": max(float(values[:, nr:].max()), 1e-30),
              "sourceLabel": f"Native Radar | TX{tx} RX{rx} | {len(times)} recorded frames",
              "fps": fps,
              "resultId": str(metadata.get("result_id") or metadata.get("run_id") or "pending-result"),
              "sceneFingerprint": metadata.get("motion_fingerprint"),
              "fingerprintScope": "authored motion tracks only; geometry is not verified"}
    if not record["sceneFingerprint"]:
        record.pop("sceneFingerprint")
    return record, values.tobytes(order="C")


def publish_recording(component, api, record, data):
    """Use Studio's existing binary data plane; websocket carries only metadata."""
    from witwin_server.core.components.core.plot import PlotData
    if record["sceneId"] != component.scene.scene_id:
        raise ValueError("Recorded scene differs from this room; refusing cat/radar synchronization.")
    fingerprint = record.get("sceneFingerprint")
    if fingerprint and fingerprint != motion_fingerprint(component.scene):
        raise ValueError("The authored motion has changed since simulation; run it again before synchronized replay.")
    if len(data) != len(record["timesS"]) * record["frameStride"] * 4 or len(data) > MAX_PREVIEW_BYTES:
        raise ValueError("Replay payload size disagrees with metadata.")
    handler = api.server.handlers.get("binary_assets")
    if handler is None:
        raise RuntimeError("Studio binary asset service is unavailable.")
    # The Plot manifest is saved with the scene, so its payload must survive a
    # backend restart as well.  A content-addressed id makes re-publication
    # idempotent and avoids leaving a new opaque payload behind on every click.
    content_id = sha256(data).hexdigest()
    ref = handler.service.register_bytes(kind="radar-replay", format="witwin.radar.replay.float32.v1",
                                        data=data, scene_id=component.scene.scene_id,
                                        persistent=True, asset_id=f"radar-replay-{content_id}")
    old_id = getattr(component, "_replay_asset_id", None)
    result_id = record.get("resultId")
    if not result_id or result_id == "pending-result":
        result_id = str(getattr(component, "_solver_run_id", "") or getattr(component, "_solver_result_handle", "") or "saved-result")
    record = {
        **record,
        "resultId": result_id,
        "asset": ref.to_dict(),
        "view": "range_doppler",
        "motionVerifiedAtPreparation": bool(fingerprint),
        "sceneGeometryFingerprint": scene_geometry_fingerprint(component.scene),
        "radarPoseConfigFingerprint": radar_pose_config_fingerprint(component),
        "stale": False,
    }
    descriptor = _recording_source_descriptor(component, record, ref.asset_id)
    record["dataSource"] = descriptor
    previous_source = getattr(component, "signal_source", None)
    registered_source = False
    data_sources = getattr(api, "data_sources", None)
    try:
        if data_sources is not None:
            data_sources.register(descriptor)
            registered_source = True
        component.signal_source = {
            "sourceId": descriptor["sourceId"],
            "mode": "timeline",
            "kind": "radar.replay",
            "defaultChannel": descriptor["defaultChannel"],
            "channelId": descriptor["defaultChannel"],
        }
        component.signal_figure.set_plot_data(PlotData("line", {"recording": record},
                                                      title="Recorded radar — follows this room's Timeline"))
    except Exception:
        component.signal_source = previous_source
        unregister = getattr(data_sources, "unregister", None)
        if registered_source and callable(unregister):
            unregister(descriptor["sourceId"])
        handler.service.clear(ref.asset_id)
        raise
    component._replay_asset_id = ref.asset_id
    if old_id and old_id != ref.asset_id:
        handler.service.clear(old_id)
    return f"Replay ready: {len(record['timesS'])} frames. Play the room Timeline; no data outside the recorded interval."
