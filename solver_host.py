"""Out-of-process Radar solver entry."""
from __future__ import annotations

import sys
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


PLUGIN_PARENT = Path(__file__).resolve().parent.parent
if str(PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_PARENT))

import wt_radar  # noqa: E402,F401  (registers Radar components before Scene.from_dict)
from witwin_server.features.solvers.scene_ref import load_scene_ref  # noqa: E402
from witwin_server.features.solvers.sdk import method, query, serve  # noqa: E402
from wt_radar.adapter.solve import SensorSpec, SigProc, SolveRunner, TracerSpec  # noqa: E402

_LIVE_CACHES: Dict[str, Dict[str, Any]] = {}
_MAX_LIVE_CACHES = 8
_LIVE_SESSIONS: Dict[str, "LiveSession"] = {}
_LIVE_SESSIONS_LOCK = threading.RLock()
_LIVE_FINE_WAIT_WINDOW_S = 0.020 if os.name == "nt" else 0.003
_LIVE_SPIN_ONLY_THRESHOLD_S = 0.050 if os.name == "nt" else 0.0


def _sensor(config: Dict[str, Any]) -> SensorSpec:
    return SensorSpec(**dict(config.get("sensor") or {}))


def _tracer(config: Dict[str, Any]) -> TracerSpec:
    return TracerSpec(**dict(config.get("tracer") or {}))


def _clamp_index(value: Any, size: int) -> int:
    if size <= 0:
        return 0
    return max(0, min(int(value or 0), size - 1))


def _as_numpy(value: Any, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _live_options(params: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    live = dict(config.get("live") or {})
    out = dict(live)
    for key in ("session_id", "stream_id", "channels", "max_fps", "history_length"):
        if params.get(key) is not None:
            out[key] = params.get(key)
    return out


def _live_session_id(params: Dict[str, Any], config: Dict[str, Any]) -> str:
    live = _live_options(params, config)
    return str(live.get("session_id") or live.get("stream_id") or params.get("runId") or "radar.live")


def _live_channels(value: Any) -> tuple[str, ...]:
    allowed = {"raw", "rd", "pc"}
    if isinstance(value, str):
        parts = [part.strip().lower() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        parts = [str(part).strip().lower() for part in value]
    else:
        parts = []
    selected = tuple(part for part in parts if part in allowed)
    return selected or ("raw", "rd", "pc")


def _clamp_fps(value: Any) -> float:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        fps = 10.0
    return min(max(fps, 0.1), 30.0)


def _wait_until(
    stop_event: threading.Event,
    deadline: float,
    *,
    clock=time.monotonic,
    sleep=None,
    fine_window: float = _LIVE_FINE_WAIT_WINDOW_S,
    spin_only_threshold: float = _LIVE_SPIN_ONLY_THRESHOLD_S,
) -> bool:
    """Wait until deadline while keeping the final few ms out of OS timer granularity."""
    fine_window = max(0.0, float(fine_window))
    while not stop_event.is_set():
        remaining = float(deadline) - float(clock())
        if remaining <= 0.0:
            return False
        if remaining > max(fine_window, spin_only_threshold):
            if stop_event.wait(max(0.0, remaining - fine_window)):
                return True
        else:
            if sleep is not None:
                sleep(min(remaining, 0.001))
            elif stop_event.wait(min(remaining, 0.001)):
                return True
    return True


def _next_frame_deadline(
    previous_deadline: float,
    frame_started: float,
    interval: float,
    *,
    now: float | None = None,
) -> float:
    """Advance the live stream cadence without baking in one-off oversleeps."""
    interval = max(0.0, float(interval))
    if previous_deadline <= 0.0:
        deadline = float(frame_started) + interval
    else:
        deadline = float(previous_deadline) + interval
    if now is not None and interval > 0.0:
        behind = float(now) - deadline
        if behind >= 0.0:
            deadline += (int(behind // interval) + 1) * interval
    return deadline


class LiveSession:
    def __init__(self, ctx, session_id: str, scene: Any, config: Dict[str, Any], params: Dict[str, Any]):
        self.ctx = ctx
        self.session_id = str(session_id)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._dirty = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._cache: Dict[str, Any] = {}
        self._state_seq = 0
        self._frame_seq = 0
        self._last_error = ""
        self._started_at = time.monotonic()
        self.scene = scene
        self.config = dict(config or {})
        self.stream_id = self._stream_id(params, self.config)
        self.channels = _live_channels(_live_options(params, self.config).get("channels"))
        self.max_fps = _clamp_fps(_live_options(params, self.config).get("max_fps"))
        self._dirty.set()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"RadarLive-{self.session_id}",
            daemon=True,
        )
        self._thread.start()

    def update(self, ctx, *, scene: Any | None = None, config: Dict[str, Any] | None = None,
               params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        params = params or {}
        with self._lock:
            self.ctx = ctx or self.ctx
            if scene is not None:
                self.scene = scene
            if config is not None:
                self.config = dict(config or {})
            live = _live_options(params, self.config)
            self.stream_id = str(live.get("stream_id") or self.stream_id)
            self.channels = _live_channels(live.get("channels") or self.channels)
            self.max_fps = _clamp_fps(live.get("max_fps") or self.max_fps)
            self._state_seq += 1
            self._dirty.set()
        return self.status()

    def pause(self) -> Dict[str, Any]:
        with self._lock:
            self._state_seq += 1
            self._paused.set()
        return self.status()

    def resume(self) -> Dict[str, Any]:
        with self._lock:
            self._state_seq += 1
            self._paused.clear()
            self._dirty.set()
        return self.status()

    def stop(self, *, timeout: float = 2.0) -> Dict[str, Any]:
        with self._lock:
            self._state_seq += 1
            self._stop.set()
            self._dirty.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        alive = thread is not None and thread.is_alive()
        if not alive:
            self._cache.clear()
        return {"sessionId": self.session_id, "status": "stopping" if alive else "stopped"}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive() and not self._stop.is_set()
            status = "paused" if self._paused.is_set() else ("running" if running else "stopped")
            return {
                "sessionId": self.session_id,
                "streamId": self.stream_id,
                "status": status,
                "frameSeq": self._frame_seq,
                "stateSeq": self._state_seq,
                "lastError": self._last_error,
            }

    def _run(self) -> None:
        next_frame_at = 0.0
        while not self._stop.is_set():
            if self._paused.is_set():
                self._stop.wait(0.05)
                continue
            schedule = self._snapshot()
            interval = 1.0 / schedule["max_fps"]
            if schedule["on_change_only"]:
                if not self._dirty.wait(0.1):
                    continue
                if self._stop.is_set():
                    break
            now = time.monotonic()
            if now < next_frame_at:
                if _wait_until(self._stop, next_frame_at):
                    break
                if self._stop.is_set():
                    break
            if self._stop.is_set():
                break
            self._dirty.clear()
            snapshot = self._snapshot()
            started = time.monotonic()
            try:
                published = self._solve_and_publish(snapshot)
                if published:
                    with self._lock:
                        self._frame_seq += 1
                        self._last_error = ""
            except Exception as exc:  # noqa: BLE001 - live session should report and keep running
                with self._lock:
                    self._last_error = str(exc)
                try:
                    snapshot["ctx"].stream_error(snapshot["stream_id"], str(exc), code="radar_live_error")
                except Exception:  # noqa: BLE001
                    pass
            next_frame_at = _next_frame_deadline(
                next_frame_at,
                started,
                interval,
                now=time.monotonic(),
            )

    def _snapshot(self) -> Dict[str, Any]:
        with self._lock:
            config = dict(self.config or {})
            live = dict(config.get("live") or {})
            on_change_only = bool(live.get("stream_on_change_only", True))
            base_t0 = float(config.get("t0") or 0.0)
            t0 = base_t0 if on_change_only else base_t0 + max(0.0, time.monotonic() - self._started_at)
            return {
                "ctx": self.ctx,
                "scene": self.scene,
                "config": config,
                "live": live,
                "stream_id": self.stream_id,
                "channels": tuple(self.channels),
                "max_fps": self.max_fps,
                "on_change_only": on_change_only,
                "t0": t0,
                "state_seq": self._state_seq,
            }

    def _solve_and_publish(self, snapshot: Dict[str, Any]) -> bool:
        config = snapshot["config"]
        live = snapshot["live"]
        signature = str(live.get("signature") or snapshot["state_seq"])
        result = SolveRunner.run(
            snapshot["scene"],
            sensor=_sensor(config),
            tracer=_tracer(config),
            motion_sampling="per_frame",
            t0=float(snapshot["t0"]),
            live_cache=self._cache,
            cache_key=signature,
            platform_cache_key=str(live.get("scene_payload_signature") or "") or None,
        )
        with self._lock:
            if self._stop.is_set() or self._paused.is_set():
                return False
        self._publish_channels(snapshot, result)
        return True

    def _publish_channels(self, snapshot: Dict[str, Any], result) -> None:
        channels = set(snapshot["channels"])
        live = snapshot["live"]
        stream_id = snapshot["stream_id"]
        ctx = snapshot["ctx"]
        timestamp_us = int(time.time() * 1_000_000)
        sig = _as_numpy(result.signal)
        n_tx = int(sig.shape[0]) if sig.ndim >= 1 else 0
        n_rx = int(sig.shape[1]) if sig.ndim >= 2 else 0
        tx = _clamp_index(live.get("tx"), n_tx)
        rx = _clamp_index(live.get("rx"), n_rx)
        frame_meta = {
            "signature": str(live.get("signature") or snapshot["state_seq"]),
            "stateSeq": int(snapshot["state_seq"]),
            "t0": float(snapshot["t0"]),
        }
        if "raw" in channels and sig.ndim >= 4:
            raw = np.ascontiguousarray(sig[tx, rx, 0].astype(np.complex64, copy=False))
            ctx.stream_publish(stream_id, "raw", raw, metadata={
                "dtype": "complex64",
                "shape": list(raw.shape),
                "tx": int(tx),
                "rx": int(rx),
                "chirp": 0,
                **frame_meta,
            }, timestamp_us=timestamp_us)
        if "rd" in channels:
            rd = SigProc.range_doppler(
                result.radar,
                result.signal,
                tx=tx,
                rx=rx,
                static_clutter_removal=bool(live.get("static_clutter_removal", False)),
            )
            mag_db = np.ascontiguousarray(_as_numpy(rd.mag_db, dtype=np.float32))
            ctx.stream_publish(stream_id, "rd", mag_db, metadata={
                "dtype": "float32",
                "shape": list(mag_db.shape),
                "tx": int(tx),
                "rx": int(rx),
                **frame_meta,
            }, timestamp_us=timestamp_us)
        if "pc" in channels:
            pc = _as_numpy(SigProc.point_cloud(
                result.radar,
                result.signal,
                detector=str(live.get("detector") or "cfar"),
                static_clutter_removal=bool(live.get("static_clutter_removal", False)),
                guard=tuple(int(v) for v in (live.get("guard") or (2, 4))),
                training=tuple(int(v) for v in (live.get("training") or (4, 8))),
                pfa=float(live.get("pfa") or 1e-3),
                energy_top_k=int(live.get("energy_top_k") or 128),
            ), dtype=np.float32)
            if pc.ndim == 1:
                pc = pc.reshape((0, 6)) if pc.size == 0 else pc.reshape((1, pc.size))
            pc = np.ascontiguousarray(pc, dtype=np.float32)
            ctx.stream_publish(stream_id, "pc", pc, metadata={
                "dtype": "float32",
                "shape": list(pc.shape),
                **frame_meta,
            }, timestamp_us=timestamp_us)

    @staticmethod
    def _stream_id(params: Dict[str, Any], config: Dict[str, Any]) -> str:
        live = _live_options(params, config)
        return str(live.get("stream_id") or live.get("session_id") or "radar.live.signal")


def solve(ctx, scene, config):
    config = config or {}
    sensor = _sensor(config)
    tracer = _tracer(config)
    t0 = float(config.get("t0") or 0.0)
    live = dict(config.get("live") or {})
    live_session_id = str(live.get("session_id") or "")
    live_signature = str(live.get("signature") or "")
    scene_payload_signature = str(live.get("scene_payload_signature") or "")
    live_cache = None
    if live_session_id and live_signature:
        live_cache = _LIVE_CACHES.setdefault(live_session_id, {})
        if len(_LIVE_CACHES) > _MAX_LIVE_CACHES:
            for session_id in list(_LIVE_CACHES)[:-_MAX_LIVE_CACHES]:
                _LIVE_CACHES.pop(session_id, None)
    ctx.log(
        "Radar solver: solve received "
        f"backend={sensor.backend} device={sensor.device} t0={t0:.6f} "
        f"live_cache={'on' if live_cache is not None else 'off'}"
    )
    ctx.progress(0.0, "building Radar scene")
    result = SolveRunner.run(
        scene,
        sensor=sensor,
        tracer=tracer,
        motion_sampling=str(config.get("motion_sampling") or "per_chirp"),
        t0=t0,
        live_cache=live_cache,
        cache_key=live_signature or None,
        platform_cache_key=scene_payload_signature or None,
    )
    ctx.log(f"Radar solver: solve complete signal_shape={tuple(result.signal.shape)}")
    ctx.progress(1.0, "Radar solve complete")
    return result


@query("raw_signal")
def raw_signal(ctx, result, params):
    ctx.log("Radar solver query: raw_signal")
    sig = result.signal.detach().cpu().numpy()
    n_tx, n_rx, _, n_adc = sig.shape
    tx = _clamp_index(params.get("tx"), n_tx)
    rx = _clamp_index(params.get("rx"), n_rx)
    x = list(range(n_adc))
    selected = sig[tx, rx, 0]
    batches = [
        [_pair_series(sig[t, r, 0], x) for r in range(n_rx)]
        for t in range(n_tx)
    ]
    return {
        "x": x,
        "n_tx": int(n_tx),
        "n_rx": int(n_rx),
        "tx": int(tx),
        "rx": int(rx),
        "real": selected.real.tolist(),
        "imag": selected.imag.tolist(),
        "batches": batches,
    }


def _pair_series(sample, x):
    return {
        "series": [
            {"x": x, "y": sample.real.tolist(), "label": "Real", "color": "#ff9500"},
            {"x": x, "y": sample.imag.tolist(), "label": "Imag", "color": "#00aaff"},
        ]
    }


@query("range_doppler")
def range_doppler(ctx, result, params):
    ctx.log("Radar solver query: range_doppler")
    sig_shape = result.signal.shape
    tx = _clamp_index(params.get("tx"), int(sig_shape[0]))
    rx = _clamp_index(params.get("rx"), int(sig_shape[1]))
    rd = SigProc.range_doppler(
        result.radar,
        result.signal,
        tx=tx,
        rx=rx,
        static_clutter_removal=bool(params.get("static_clutter_removal", False)),
    )
    rows = []
    cols = []
    if bool(params.get("show_cfar", False)):
        guard = tuple(int(v) for v in (params.get("guard") or (2, 4)))
        training = tuple(int(v) for v in (params.get("training") or (4, 8)))
        mask = SigProc.cfar_mask(rd.rd_map, guard=guard, training=training,
                                 pfa=float(params.get("pfa") or 1e-3))
        rows_arr, cols_arr = np.nonzero(mask)
        rows = rows_arr.tolist()
        cols = cols_arr.tolist()
    return {
        "tx": int(tx),
        "rx": int(rx),
        "mag_db": _as_numpy(rd.mag_db, dtype=np.float32),
        "cfar_rows": rows,
        "cfar_cols": cols,
    }


@query("point_cloud")
def point_cloud(ctx, result, params):
    ctx.log("Radar solver query: point_cloud")
    pc = SigProc.point_cloud(
        result.radar,
        result.signal,
        detector=str(params.get("detector") or "cfar"),
        static_clutter_removal=bool(params.get("static_clutter_removal", False)),
        guard=tuple(int(v) for v in (params.get("guard") or (2, 4))),
        training=tuple(int(v) for v in (params.get("training") or (4, 8))),
        pfa=float(params.get("pfa") or 1e-3),
        energy_top_k=int(params.get("energy_top_k") or 128),
    )
    return {"points": _as_numpy(pc, dtype=np.float32)}


@query("music")
def music(ctx, result, params):
    ctx.log("Radar solver query: music")
    image = SigProc.music_image(result.radar, result.signal,
                                num_pixels=int(params.get("num_pixels") or 64))
    return {"image": _as_numpy(image, dtype=np.float32)}


@method("live_start")
def live_start(ctx, params):
    params = dict(params or {})
    config = dict(params.get("config") or {})
    session_id = _live_session_id(params, config)
    scene = load_scene_ref(params.get("sceneRef"))
    with _LIVE_SESSIONS_LOCK:
        previous = _LIVE_SESSIONS.pop(session_id, None)
    if previous is not None:
        previous.stop(timeout=2.0)
    session = LiveSession(ctx, session_id, scene, config, params)
    with _LIVE_SESSIONS_LOCK:
        _LIVE_SESSIONS[session_id] = session
    session.start()
    ctx.log(
        "Radar live session started "
        f"session={session_id} stream={session.stream_id} channels={list(session.channels)} "
        f"max_fps={session.max_fps:.2f}"
    )
    return {"sessionId": session_id, "streamId": session.stream_id, "status": "started"}


@method("live_update")
def live_update(ctx, params):
    params = dict(params or {})
    session_id = _live_session_id(params, dict(params.get("config") or {}))
    with _LIVE_SESSIONS_LOCK:
        session = _LIVE_SESSIONS.get(session_id)
    if session is None:
        raise RuntimeError(f"unknown radar live session: {session_id}")
    scene = load_scene_ref(params.get("sceneRef")) if params.get("sceneRef") is not None else None
    return session.update(ctx, scene=scene, config=params.get("config"), params=params)


@method("live_pause")
def live_pause(ctx, params):
    params = dict(params or {})
    session_id = str(params.get("session_id") or params.get("sessionId") or "")
    with _LIVE_SESSIONS_LOCK:
        session = _LIVE_SESSIONS.get(session_id)
    if session is None:
        raise RuntimeError(f"unknown radar live session: {session_id}")
    ctx.log(f"Radar live session paused session={session_id}")
    return session.pause()


@method("live_resume")
def live_resume(ctx, params):
    params = dict(params or {})
    session_id = str(params.get("session_id") or params.get("sessionId") or "")
    with _LIVE_SESSIONS_LOCK:
        session = _LIVE_SESSIONS.get(session_id)
    if session is None:
        raise RuntimeError(f"unknown radar live session: {session_id}")
    ctx.log(f"Radar live session resumed session={session_id}")
    return session.resume()


@method("live_stop")
def live_stop(ctx, params):
    params = dict(params or {})
    session_id = str(params.get("session_id") or params.get("sessionId") or "")
    with _LIVE_SESSIONS_LOCK:
        session = _LIVE_SESSIONS.get(session_id)
    if session is None:
        return {"sessionId": session_id, "status": "stopped"}
    ctx.log(f"Radar live session stopping session={session_id}")
    result = session.stop()
    with _LIVE_SESSIONS_LOCK:
        if _LIVE_SESSIONS.get(session_id) is session:
            _LIVE_SESSIONS.pop(session_id, None)
    return result


@method("live_status")
def live_status(ctx, params):
    params = dict(params or {})
    session_id = str(params.get("session_id") or params.get("sessionId") or "")
    with _LIVE_SESSIONS_LOCK:
        session = _LIVE_SESSIONS.get(session_id)
    if session is None:
        return {"sessionId": session_id, "status": "stopped"}
    return session.status()


if __name__ == "__main__":
    serve(solve)
