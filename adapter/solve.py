"""Scene-driven Radar 0.3 solve and typed processing adapters."""
import time
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from witwin_server.utils.logging import get_logger

from .common import num

logger = get_logger("RadarSolve")


@dataclass
class SensorSpec:
    """Radar pose and runtime device."""

    position: List[float]
    target: Optional[List[float]]
    up: List[float]
    fov: float
    device: str = "cuda"

    @classmethod
    def from_component(cls, radar: Any) -> "SensorSpec":
        position, target, up = _world_pose(radar.owner)
        return cls(position=position, target=target, up=up, fov=num(radar.fov), device="cuda")


@dataclass
class PropagationSpec:
    """Explicit scattering and Channel propagation choices for one solve."""

    target_rcs_m2: float = 1.0
    multipath: bool = False
    max_depth: int = 0
    polarization: Tuple[float, float, float] = (0.0, 1.0, 0.0)

    @classmethod
    def from_component(cls, radar: Any) -> "PropagationSpec":
        return cls(
            target_rcs_m2=num(radar.target_rcs_m2),
            multipath=bool(radar.multipath),
            max_depth=int(radar.max_reflections) if bool(radar.multipath) else 0,
            polarization=tuple(float(value) for value in radar.polarization),
        )

    @property
    def components(self) -> frozenset[str]:
        return frozenset({"los", "reflection"} if self.multipath else {"los"})


@dataclass
class SolveResult:
    """One Studio frame and the typed Radar simulation that produced it."""

    radar: Any
    signal: Any
    simulation: Any
    axes: Any


@dataclass
class RDView:
    mag_db: np.ndarray
    ranges: np.ndarray
    velocities: np.ndarray
    rd_map: Any


def _world_pose(obj: Any) -> tuple:
    from witwin_server.utils.mitsuba_utils import get_world_transform

    matrix = np.asarray(get_world_transform(obj), dtype=np.float64)
    position = matrix[:3, 3].tolist()
    rotation = matrix[:3, :3]
    forward = _normalize(rotation @ np.array([0.0, 0.0, -1.0]), fallback=(0.0, 0.0, -1.0))
    up = _normalize(rotation @ np.array([0.0, 1.0, 0.0]), fallback=(0.0, 1.0, 0.0))
    target = [position[index] + forward[index] for index in range(3)]
    return position, target, list(up)


def _normalize(vector: np.ndarray, *, fallback: tuple) -> tuple:
    norm = float(np.linalg.norm(vector))
    return fallback if norm <= 1e-12 else tuple((vector / norm).tolist())


class SolveRunner:
    """Build and run the current scene-driven Radar pipeline."""

    @staticmethod
    def run(
        studio_scene: Any,
        *,
        sensor: SensorSpec,
        propagation: PropagationSpec,
        t0: float,
        live_cache: Optional[dict] = None,
        cache_key: Optional[str] = None,
        platform_cache_key: Optional[str] = None,
        site_positions: Any = None,
        site_velocities: Any = None,
        site_stable_ids: Any = None,
    ) -> SolveResult:
        import torch
        from witwin.radar.propagation import Kinematics, two_way_duals
        from witwin.radar.scattering import ScalarRcsResponse
        from witwin.radar.simulation import ScatterSitePolicy

        from .radar_adapter import RadarAdapter

        if not torch.cuda.is_available():
            raise RuntimeError("WiTwin Radar 0.3 scene simulation requires a CUDA device")
        sensor = SensorSpec(sensor.position, sensor.target, sensor.up, sensor.fov, "cuda")
        started = time.perf_counter()

        exported = None
        if live_cache is not None and platform_cache_key:
            if live_cache.get("platform_cache_key") == platform_cache_key:
                exported = live_cache.get("platform_export")
        if exported is None:
            exported = RadarAdapter().build_export(studio_scene)

        config_key = SolveRunner._stable_key(exported.config)
        radar = None
        if live_cache is not None and live_cache.get("config_key") == config_key:
            radar = live_cache.get("radar")
            if radar is not None:
                radar.set_pose(position=sensor.position, target=sensor.target, up=sensor.up, fov=sensor.fov)
        if radar is None:
            radar = SolveRunner.build_radar(exported.config, sensor)

        positions = exported.positions_at(t0, device=radar.device) if site_positions is None else site_positions
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32, device=radar.device)
        else:
            positions = positions.to(device=radar.device, dtype=torch.float32)
        positions = positions.contiguous()
        stable_ids = None if site_stable_ids is None else tuple(int(value) for value in site_stable_ids)
        response = ScalarRcsResponse.from_rcs(
            propagation.target_rcs_m2,
            reference_frequency_hz=radar.config.fc,
            device=radar.device,
        )

        def simulate(resolved_positions):
            sites = ScatterSitePolicy.explicit(resolved_positions, stable_ids=stable_ids)
            return radar.simulate(
                exported.scene,
                times=(float(t0),),
                response=response,
                sites=sites,
                components=propagation.components,
                max_depth=propagation.max_depth,
                polarization=propagation.polarization,
                ad_mode="jvp" if site_velocities is not None else "none",
            )

        if site_velocities is None:
            simulation = simulate(positions)
        else:
            velocities = site_velocities
            if not isinstance(velocities, torch.Tensor):
                velocities = torch.tensor(velocities, dtype=torch.float32, device=radar.device)
            else:
                velocities = velocities.to(device=radar.device, dtype=torch.float32)
            kinematics = Kinematics(
                positions_m=positions,
                velocities_m_per_s=velocities.contiguous(),
            )
            with two_way_duals(sites=kinematics) as duals:
                simulation = simulate(duals.sites)
        axes = SolveRunner.processing_axes(radar)
        result = SolveResult(radar=radar, signal=simulation.cube[0], simulation=simulation, axes=axes)

        if live_cache is not None:
            live_cache.clear()
            live_cache.update(
                {
                    "cache_key": cache_key,
                    "platform_cache_key": platform_cache_key,
                    "platform_export": exported,
                    "config_key": config_key,
                    "radar": radar,
                }
            )
        logger.info(
            f"Radar 0.3 solve complete shape={tuple(result.signal.shape)} "
            f"elapsed_ms={(time.perf_counter() - started) * 1000.0:.2f}"
        )
        return result

    @staticmethod
    def run_group(
        studio_scene: Any,
        *,
        sensors: "dict[str, SensorSpec]",
        propagation: PropagationSpec,
        t0: float,
    ) -> "dict[str, Any]":
        """Run each Radar independently; Radar 0.3 intentionally has no group entry point."""
        return {
            name: SolveRunner.run(
                studio_scene,
                sensor=sensor,
                propagation=propagation,
                t0=t0,
            ).signal
            for name, sensor in sensors.items()
        }

    @staticmethod
    def build_radar(config: Any, sensor: SensorSpec) -> Any:
        import witwin.radar as wr

        return wr.Radar(
            config,
            device=sensor.device,
            position=sensor.position,
            target=sensor.target,
            up=sensor.up,
            fov=sensor.fov,
        )

    @staticmethod
    def processing_axes(radar: Any) -> Any:
        from witwin.radar.processing import ProcessingAxes
        from witwin.radar.synthesis import SlowTimeMode

        synthesis = radar._synthesize(
            radar.last_radar_paths,
            slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE,
        )
        return ProcessingAxes.from_synthesis(
            synthesis,
            radar.system_config.waveform_spec(),
            radar.system_config.sensors.array,
        )

    @staticmethod
    def _stable_key(value: Any):
        if is_dataclass(value):
            value = asdict(value)
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, dict):
            return tuple((str(key), SolveRunner._stable_key(item)) for key, item in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(SolveRunner._stable_key(item) for item in value)
        if isinstance(value, (float, np.floating)):
            return round(float(value), 12)
        if isinstance(value, (int, bool, str)) or value is None:
            return value
        if hasattr(value, "__dict__"):
            return SolveRunner._stable_key(
                {str(key): item for key, item in vars(value).items() if not str(key).startswith("_")}
            )
        return repr(value)


class SigProc:
    """Plugin-facing views over the Radar 0.3 typed processing API."""

    @staticmethod
    def _profile(radar: Any, signal: Any, *, static_clutter_removal: bool):
        from witwin.radar.processing import ProcessingCube, range_profile

        axes = SolveRunner.processing_axes(radar)
        data = signal
        if static_clutter_removal:
            data = data - data.mean(dim=-2, keepdim=True)
        # Radar 0.3 may synthesize an already transformed FMCW spectrum. Its typed
        # processing contract requires the default rectangular range window for that
        # domain; Doppler windowing remains valid in the following stage.
        return range_profile(ProcessingCube(data, axes)), axes

    @staticmethod
    def _range_doppler_record(radar: Any, signal: Any, *, static_clutter_removal: bool):
        from witwin.radar.processing import range_doppler_map

        profile, axes = SigProc._profile(radar, signal, static_clutter_removal=static_clutter_removal)
        return range_doppler_map(profile, window="hann"), axes

    @staticmethod
    def range_doppler(
        radar: Any,
        signal: Any,
        *,
        tx: int,
        rx: int,
        static_clutter_removal: bool,
    ) -> RDView:
        import torch

        rd, axes = SigProc._range_doppler_record(
            radar,
            signal,
            static_clutter_removal=static_clutter_removal,
        )
        selected = rd.data[int(tx), int(rx)]
        mag_db = 20.0 * torch.log10(selected.abs() + 1e-12)
        return RDView(
            mag_db.detach().cpu().numpy(),
            axes.range_m.detach().cpu().numpy(),
            axes.velocity_mps.detach().cpu().numpy(),
            selected,
        )

    @staticmethod
    def point_cloud(
        radar: Any,
        signal: Any,
        *,
        detector: str,
        static_clutter_removal: bool,
        guard: Tuple[int, int],
        training: Tuple[int, int],
        pfa: float,
        energy_top_k: int,
    ) -> np.ndarray:
        import torch
        from witwin.radar.processing import ArrayGeometry, Detections, ca_cfar_fast, point_cloud

        rd, axes = SigProc._range_doppler_record(
            radar,
            signal,
            static_clutter_removal=static_clutter_removal,
        )
        combined = rd.data.reshape(-1, *rd.data.shape[-2:]).sum(dim=0).abs()
        if detector == "cfar":
            detections = ca_cfar_fast(combined, guard_cells=guard, training_cells=training, pfa=pfa)
        elif detector == "topk":
            count = min(max(int(energy_top_k), 0), combined.numel())
            mask = torch.zeros_like(combined, dtype=torch.bool)
            if count:
                mask.reshape(-1)[torch.topk(combined.reshape(-1), count).indices] = True
            detections = Detections(mask=mask, threshold=torch.zeros_like(combined))
        else:
            raise ValueError("detector must be 'cfar' or 'topk'")
        cloud = point_cloud(
            detections,
            rd,
            axes,
            ArrayGeometry.from_axes(axes),
            route="phase_comparison",
            max_points=int(energy_top_k),
        )
        if len(cloud) == 0:
            return np.empty((0, 6), dtype=np.float32)
        rows = torch.cat(
            (
                cloud.xyz,
                cloud.velocity_mps.reshape(-1, 1),
                cloud.energy.reshape(-1, 1),
                cloud.range_m.reshape(-1, 1),
            ),
            dim=1,
        )
        return rows.detach().cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def cfar_mask(rd_map: Any, *, guard: Tuple[int, int], training: Tuple[int, int], pfa: float) -> np.ndarray:
        import torch
        from witwin.radar.processing import ca_cfar_fast

        value = rd_map if isinstance(rd_map, torch.Tensor) else torch.as_tensor(rd_map)
        detections = ca_cfar_fast(value.abs(), guard_cells=guard, training_cells=training, pfa=pfa)
        return detections.mask.detach().cpu().numpy()

    @staticmethod
    def music_image(radar: Any, signal: Any, *, num_pixels: int = 64) -> np.ndarray:
        import torch
        from witwin.radar.processing import ArrayGeometry, music_image

        profile, axes = SigProc._profile(radar, signal, static_clutter_removal=False)
        array = ArrayGeometry.from_axes(axes)
        grid = torch.linspace(-torch.pi / 2.0, torch.pi / 2.0, int(num_pixels), device=signal.device)
        range_bins = torch.arange(axes.range_bin_count, device=signal.device)
        signals = max(1, min(7, array.sensor_pair_count - 1))
        smooth = max(1, min(3, axes.num_tx - 1, axes.num_rx - 1))
        image = music_image(
            profile,
            array,
            elevation_rad=grid,
            azimuth_rad=grid,
            range_bins=range_bins,
            num_signals=signals,
            spatial_smooth=smooth,
            num_snapshots=min(8, int(signal.shape[-2])),
        )
        return image.amax(dim=-1).detach().cpu().numpy()
