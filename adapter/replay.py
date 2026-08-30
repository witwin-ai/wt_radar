"""Numeric Studio widgets and recorded scene-time playback; no RF/DSP equations."""
from hashlib import sha256
import json

import numpy as np

MAX_PREVIEW_BYTES = 128 * 1024**2


def motion_fingerprint(scene):
    """Identify authored motion only; this is not a full geometry fingerprint."""
    value = scene.timeline_manager.clip.to_dict()
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


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
    record = {"schema": 1, "sceneId": metadata.get("scene_id", ""), "timesS": times.tolist(),
              "rangeM": axes.range_m.tolist(), "velocityMps": axes.velocity_mps.tolist(),
              "frameStride": stride, "endTimeS": end, "profileMax": max(float(values[:, :nr].max()), 1e-30),
              "dopplerMax": max(float(values[:, nr:].max()), 1e-30),
              "sourceLabel": f"Native Radar | TX{tx} RX{rx} | {len(times)} recorded frames",
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
    record = {**record, "asset": ref.to_dict(), "view": "range_doppler", "motionVerifiedAtPreparation": bool(fingerprint)}
    try:
        component.signal_figure.set_plot_data(PlotData("line", {"recording": record},
                                                      title="Recorded radar — follows this room's Timeline"))
    except Exception:
        handler.service.clear(ref.asset_id)
        raise
    component._replay_asset_id = ref.asset_id
    if old_id and old_id != ref.asset_id:
        handler.service.clear(old_id)
    return f"Replay ready: {len(record['timesS'])} frames. Play the room Timeline; no data outside the recorded interval."
