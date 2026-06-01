"""Measure wt-radar solver-side live stream throughput end to end.

The benchmark drives the same solver-host LiveSession used by Start Stream:
Studio Scene -> Radar component config -> SolveRunner -> live post-processing ->
stream_publish. It intentionally runs in-process so the timings isolate radar solve
and publish payload cost from frontend rendering.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np


SMOKE_CONFIG: Dict[str, Any] = {
    "num_tx": 3,
    "num_rx": 4,
    "fc": 77e9,
    "slope": 60.012,
    "power": 15.0,
    "adc_samples": 8,
    "adc_start_time": 6.0,
    "sample_rate": 4400.0,
    "idle_time": 7.0,
    "ramp_end_time": 58.0,
    "chirp_per_frame": 4,
    "frame_per_second": 10.0,
    "num_doppler_bins": 4,
    "num_range_bins": 8,
    "num_angle_bins": 8,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
}


@dataclass(frozen=True)
class PublishRecord:
    perf_time: float
    wall_time: float
    stream_id: str
    channel_id: str
    payload_bytes: int
    shape: tuple[int, ...]
    timestamp_us: int


class BenchmarkContext:
    """Minimal solver SDK context that records live stream publishes."""

    run_id = "radar-live-throughput"

    def __init__(self, *, encode_bytes: bool = False):
        self.encode_bytes = bool(encode_bytes)
        self.records: List[PublishRecord] = []
        self.publish_ms: List[float] = []
        self.logs: List[tuple[str, str]] = []
        self.errors: List[tuple[str, str, Dict[str, Any]]] = []
        self._lock = threading.RLock()

    def stream_publish(self, stream_id: str, channel_id: str, payload=None, **kwargs) -> None:
        started_perf = time.perf_counter()
        wall_time = time.time()
        arr = np.ascontiguousarray(np.asarray(payload))
        if self.encode_bytes:
            raw_bytes = arr.tobytes()
            payload_bytes = len(raw_bytes)
            if getattr(self, "simulate_sdk_base64", False):
                base64.b64encode(raw_bytes).decode("ascii")
        else:
            payload_bytes = int(arr.nbytes)
        timestamp_us = int(kwargs.get("timestamp_us") or wall_time * 1_000_000)
        record = PublishRecord(
            perf_time=started_perf,
            wall_time=wall_time,
            stream_id=str(stream_id),
            channel_id=str(channel_id),
            payload_bytes=payload_bytes,
            shape=tuple(int(v) for v in arr.shape),
            timestamp_us=timestamp_us,
        )
        with self._lock:
            self.records.append(record)
            self.publish_ms.append((time.perf_counter() - started_perf) * 1000.0)

    def stream_error(self, stream_id: str, message: str, **kwargs) -> None:
        with self._lock:
            self.errors.append((str(stream_id), str(message), dict(kwargs)))

    def log(self, message: str, level: str = "info") -> None:
        with self._lock:
            self.logs.append((str(level), str(message)))

    def frame_count(self) -> int:
        with self._lock:
            return len({record.timestamp_us for record in self.records})

    def snapshot_records(self) -> List[PublishRecord]:
        with self._lock:
            return list(self.records)


def bootstrap_paths(platform_root: str | None = None) -> None:
    """Make local studio server/plugins and optional witwin-platform packages importable."""
    studio_root = Path(__file__).resolve().parents[3]
    candidates = [
        studio_root / "server",
        studio_root / "plugins",
    ]
    platform_value = platform_root or os.environ.get("WITWIN_PLATFORM_ROOT")
    platform_path = Path(platform_value) if platform_value else Path("E:/Code/witwin-platform")
    if platform_path.exists():
        candidates.extend([platform_path / "radar", platform_path / "core"])
    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = rank - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def stats(values: Sequence[float]) -> Dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(vals),
        "min": min(vals),
        "mean": mean(vals),
        "p50": percentile(vals, 0.50),
        "p95": percentile(vals, 0.95),
        "max": max(vals),
    }


def _frame_groups(records: Iterable[PublishRecord]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[PublishRecord]] = defaultdict(list)
    for record in records:
        grouped[int(record.timestamp_us)].append(record)
    frames = []
    for timestamp_us, items in grouped.items():
        items_sorted = sorted(items, key=lambda item: item.perf_time)
        frames.append({
            "timestamp_us": timestamp_us,
            "complete_perf": max(item.perf_time for item in items_sorted),
            "channels": [item.channel_id for item in items_sorted],
            "payload_bytes": sum(item.payload_bytes for item in items_sorted),
            "records": items_sorted,
        })
    return sorted(frames, key=lambda item: item["complete_perf"])


def summarize_records(
    records: Sequence[PublishRecord],
    *,
    solve_ms: Sequence[float] | None = None,
    post_ms: Dict[str, Sequence[float]] | None = None,
    warmup_frames: int = 0,
) -> Dict[str, Any]:
    frames = _frame_groups(records)
    measured = frames[int(max(0, warmup_frames)):]
    intervals = [
        (right["complete_perf"] - left["complete_perf"]) * 1000.0
        for left, right in zip(measured, measured[1:])
    ]
    fps = 0.0
    if len(measured) >= 2:
        elapsed = measured[-1]["complete_perf"] - measured[0]["complete_perf"]
        fps = (len(measured) - 1) / elapsed if elapsed > 0 else 0.0

    measured_bytes = [float(frame["payload_bytes"]) for frame in measured]
    bytes_per_frame = mean(measured_bytes) if measured_bytes else 0.0
    solve_values = list(solve_ms or [])[int(max(0, warmup_frames)):]
    post_values = {
        key: list(values)[int(max(0, warmup_frames)):]
        for key, values in (post_ms or {}).items()
    }
    channels = sorted({record.channel_id for frame in measured for record in frame["records"]})

    return {
        "frames_total": len(frames),
        "frames_measured": len(measured),
        "warmup_frames": int(max(0, warmup_frames)),
        "channels": channels,
        "fps": fps,
        "frame_interval_ms": stats(intervals),
        "payload_bytes_per_frame_mean": bytes_per_frame,
        "payload_bytes_per_second": bytes_per_frame * fps,
        "publish_calls": len(records),
        "publish_payload_bytes": sum(record.payload_bytes for record in records),
        "solve_ms": stats(solve_values),
        "post_ms": {key: stats(values) for key, values in post_values.items()},
        "first_frame_channels": frames[0]["channels"] if frames else [],
        "last_frame_channels": frames[-1]["channels"] if frames else [],
    }


def _profile_config(profile: str) -> Dict[str, Any]:
    if profile == "smoke":
        return dict(SMOKE_CONFIG)
    from wt_radar.examples.demo_scene import DEMO_CONFIG

    return dict(DEMO_CONFIG)


def _apply_config_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(config)
    for attr, key in (
        ("adc_samples", "adc_samples"),
        ("chirps", "chirp_per_frame"),
        ("range_bins", "num_range_bins"),
        ("doppler_bins", "num_doppler_bins"),
        ("angle_bins", "num_angle_bins"),
    ):
        value = getattr(args, attr)
        if value is not None:
            out[key] = int(value)
    out["frame_per_second"] = float(args.max_fps)
    return out


def build_studio_scene(args: argparse.Namespace):
    import witwin.radar as wr
    from witwin_server import Scene
    from wt_radar.adapter.config_map import ConfigMap
    from wt_radar.library_items import _target_object

    radar_config = wr.RadarConfig.from_dict(_apply_config_overrides(_profile_config(args.profile), args))
    settings = ConfigMap.settings_to_studio(radar_config)
    settings.name = "Radar Throughput"
    target = _target_object("Radar Throughput Target")

    scene = Scene()
    scene.begin_batch()
    scene.add_object(target)
    scene.add_object(settings)
    scene.end_batch()

    radar = settings.get_component("Radar")
    radar.backend = str(args.backend)
    radar.device = str(args.device)
    radar.resolution = int(args.resolution)
    radar.sampling = str(args.sampling)
    radar.ray_batch_size = int(args.ray_batch_size)
    radar.stream_channels = str(args.channels)
    radar.stream_max_fps = float(args.max_fps)
    radar.stream_history_length = 1
    radar.stream_on_change_only = False
    radar.motion_sampling = "per_frame"
    return scene, radar


def install_timing_hooks(solver_host) -> tuple[Dict[str, Any], Callable[[], None]]:
    from wt_radar.adapter.radar_adapter import RadarAdapter

    timings: Dict[str, Any] = {
        "solve_ms": [],
        "to_platform_ms": [],
        "build_radar_ms": [],
        "wait_remaining_ms": [],
        "wait_ms": [],
        "post_ms": defaultdict(list),
        "active_solve_started": None,
        "active_solve_elapsed_s": None,
        "active_stage": None,
        "active_stage_started": None,
    }
    original_run = solver_host.SolveRunner.run
    original_wait_until = solver_host._wait_until
    original_to_platform = RadarAdapter.to_platform
    original_build_radar = solver_host.SolveRunner.build_radar
    original_rd = solver_host.SigProc.range_doppler
    original_pc = solver_host.SigProc.point_cloud

    def timed_stage(name: str, bucket: str, func, *args, **kwargs):
        started = time.perf_counter()
        timings["active_stage"] = name
        timings["active_stage_started"] = started
        try:
            return func(*args, **kwargs)
        finally:
            timings[bucket].append((time.perf_counter() - started) * 1000.0)
            timings["active_stage"] = None
            timings["active_stage_started"] = None

    def timed_run(*args, **kwargs):
        started = time.perf_counter()
        timings["active_solve_started"] = started
        timings["active_stage"] = "SolveRunner.run"
        timings["active_stage_started"] = started
        try:
            return original_run(*args, **kwargs)
        finally:
            finished = time.perf_counter()
            timings["solve_ms"].append((finished - started) * 1000.0)
            timings["active_solve_elapsed_s"] = finished - started
            timings["active_solve_started"] = None
            timings["active_stage"] = None
            timings["active_stage_started"] = None

    def timed_to_platform(self, *args, **kwargs):
        return timed_stage("RadarAdapter.to_platform", "to_platform_ms", original_to_platform, self, *args, **kwargs)

    def timed_build_radar(*args, **kwargs):
        return timed_stage("SolveRunner.build_radar", "build_radar_ms", original_build_radar, *args, **kwargs)

    def timed_rd(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_rd(*args, **kwargs)
        finally:
            timings["post_ms"]["rd"].append((time.perf_counter() - started) * 1000.0)

    def timed_pc(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_pc(*args, **kwargs)
        finally:
            timings["post_ms"]["pc"].append((time.perf_counter() - started) * 1000.0)

    def timed_wait_until(stop_event, deadline, **kwargs):
        started = time.perf_counter()
        remaining = max(0.0, float(deadline) - time.monotonic())
        try:
            return original_wait_until(stop_event, deadline, **kwargs)
        finally:
            timings["wait_remaining_ms"].append(remaining * 1000.0)
            timings["wait_ms"].append((time.perf_counter() - started) * 1000.0)

    solver_host.SolveRunner.run = staticmethod(timed_run)
    solver_host._wait_until = timed_wait_until
    RadarAdapter.to_platform = timed_to_platform
    solver_host.SolveRunner.build_radar = staticmethod(timed_build_radar)
    solver_host.SigProc.range_doppler = staticmethod(timed_rd)
    solver_host.SigProc.point_cloud = staticmethod(timed_pc)

    def restore() -> None:
        solver_host.SolveRunner.run = staticmethod(original_run)
        solver_host._wait_until = original_wait_until
        RadarAdapter.to_platform = original_to_platform
        solver_host.SolveRunner.build_radar = staticmethod(original_build_radar)
        solver_host.SigProc.range_doppler = staticmethod(original_rd)
        solver_host.SigProc.point_cloud = staticmethod(original_pc)

    return timings, restore


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    bootstrap_paths(args.platform_root)
    import wt_radar.solver_host as solver_host
    from wt_radar.adapter.solve import SensorSpec

    scene, radar = build_studio_scene(args)
    signature = radar._live_scene_signature()
    params = radar._live_solver_params(
        signature,
        spec=SensorSpec.from_component(radar),
        t0_override=float(args.t0),
    )
    params["channels"] = [part.strip() for part in str(args.channels).split(",") if part.strip()]
    params["max_fps"] = float(args.max_fps)
    params["config"]["live"]["channels"] = list(params["channels"])
    params["config"]["live"]["max_fps"] = float(args.max_fps)
    params["config"]["live"]["stream_on_change_only"] = False

    ctx = BenchmarkContext(encode_bytes=bool(args.encode_bytes))
    ctx.simulate_sdk_base64 = bool(args.simulate_sdk_base64)
    timings, restore = install_timing_hooks(solver_host)
    session = solver_host.LiveSession(ctx, str(args.session_id), scene, params["config"], params)
    requested_frames = int(args.frames) + int(args.warmup_frames)
    started = time.perf_counter()
    timeout_reached = False
    try:
        session.start()
        deadline = time.perf_counter() + float(args.timeout)
        while time.perf_counter() < deadline:
            if ctx.frame_count() >= requested_frames:
                break
            if ctx.errors:
                break
            time.sleep(0.02)
        timeout_reached = ctx.frame_count() < requested_frames and not ctx.errors
    finally:
        stop_status = session.stop(timeout=5.0)
        restore()

    elapsed = time.perf_counter() - started
    active_started = timings.get("active_solve_started")
    active_elapsed = (time.perf_counter() - active_started) if active_started else None
    active_stage_started = timings.get("active_stage_started")
    active_stage_elapsed = (time.perf_counter() - active_stage_started) if active_stage_started else None
    summary = summarize_records(
        ctx.snapshot_records(),
        solve_ms=timings["solve_ms"],
        post_ms=dict(timings["post_ms"]),
        warmup_frames=int(args.warmup_frames),
    )
    summary.update({
        "elapsed_wall_s": elapsed,
        "requested_frames": requested_frames,
        "target_max_fps": float(args.max_fps),
        "profile": args.profile,
        "backend": args.backend,
        "device": args.device,
        "resolution": int(args.resolution),
        "sampling": args.sampling,
        "ray_batch_size": int(args.ray_batch_size),
        "encode_bytes": bool(args.encode_bytes),
        "simulate_sdk_base64": bool(args.simulate_sdk_base64),
        "timeout_reached": bool(timeout_reached),
        "active_solve_elapsed_s": active_elapsed,
        "last_completed_solve_elapsed_s": timings.get("active_solve_elapsed_s"),
        "active_stage": timings.get("active_stage"),
        "active_stage_elapsed_s": active_stage_elapsed,
        "to_platform_ms": stats(timings["to_platform_ms"]),
        "build_radar_ms": stats(timings["build_radar_ms"]),
        "wait_remaining_ms": stats(timings["wait_remaining_ms"]),
        "wait_ms": stats(timings["wait_ms"]),
        "publish_ms": stats(ctx.publish_ms),
        "stop_status": stop_status,
        "errors": ctx.errors,
        "logs_tail": ctx.logs[-20:],
    })
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "demo"], default="demo")
    parser.add_argument("--frames", type=int, default=20, help="Measured frames after warmup")
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-fps", type=float, default=30.0)
    parser.add_argument("--channels", default="raw,rd,pc")
    parser.add_argument("--backend", default="dirichlet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--sampling", default="triangle")
    parser.add_argument("--ray-batch-size", type=int, default=65536)
    parser.add_argument("--adc-samples", type=int)
    parser.add_argument("--chirps", type=int)
    parser.add_argument("--range-bins", type=int)
    parser.add_argument("--doppler-bins", type=int)
    parser.add_argument("--angle-bins", type=int)
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--encode-bytes", action="store_true",
                        help="Force payload .tobytes() to approximate binary stream serialization cost")
    parser.add_argument("--simulate-sdk-base64", action="store_true",
                        help="Also base64-encode payload bytes to approximate solver child JSONRPC stream_publish cost")
    parser.add_argument("--session-id", default="radar.throughput.signal")
    parser.add_argument("--platform-root")
    parser.add_argument("--json-out")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_benchmark(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    return 1 if summary.get("errors") or summary.get("timeout_reached") else 0


if __name__ == "__main__":
    raise SystemExit(main())
