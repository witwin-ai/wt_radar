"""Out-of-process Radar solver entry."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np


PLUGIN_PARENT = Path(__file__).resolve().parent.parent
if str(PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_PARENT))

import wt_radar  # noqa: E402,F401  (registers Radar components before Scene.from_dict)
from witwin_server.features.solvers.sdk import query, serve  # noqa: E402
from wt_radar.adapter.solve import SensorSpec, SigProc, SolveRunner, TracerSpec  # noqa: E402

_LIVE_CACHES: Dict[str, Dict[str, Any]] = {}
_MAX_LIVE_CACHES = 8


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


def solve(ctx, scene, config):
    config = config or {}
    sensor = _sensor(config)
    tracer = _tracer(config)
    t0 = float(config.get("t0") or 0.0)
    live = dict(config.get("live") or {})
    live_session_id = str(live.get("session_id") or "")
    live_signature = str(live.get("signature") or "")
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


if __name__ == "__main__":
    serve(solve)
