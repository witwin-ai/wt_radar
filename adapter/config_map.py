"""Sensor config interchange: ``witwin.radar.RadarConfig`` <-> the Radar Settings object.

``settings_to_studio`` builds the single ``Empty`` Radar Settings object and fills the
``RadarConfig`` component from a platform ``RadarConfig`` dataclass. ``build_config``
reads the component back into a dict and hands it to ``RadarConfig.from_dict``, which
runs the platform's own validation (loc counts, finiteness, sub-config rules) — so the
editor never emits an invalid config. Values are stored in the platform's native
non-SI units (see ``common``), so the round trip is exact.

R0 covers the 17 core FMCW fields. The four sub-configs and the sensor pose/backend are
layered on by R1 (``sensor_map`` / ``subconfig_map``).
"""
from typing import Any, Dict, Optional

from witwin_server import SceneObject

from .common import num, vec_list

SETTINGS_NAME = "Radar Settings"
MARKER_COMPONENT = "RadarConfig"


class ConfigMap:
    """``RadarConfig`` <-> the Radar Settings object's ``RadarConfig`` component."""

    @staticmethod
    def settings_to_studio(config: Optional[Any]) -> SceneObject:
        """Build the Radar Settings ``Empty`` object; ``None`` leaves component defaults."""
        from ..components.config import RadarConfigComponent

        obj = SceneObject(name=SETTINGS_NAME, mesh_type="Empty")
        comp = obj.add_component(RadarConfigComponent())
        if config is not None:
            ConfigMap._fill(comp, config)
        return obj

    @staticmethod
    def build_config(settings_obj: SceneObject) -> Any:
        """Read the Radar Settings components back into a validated ``RadarConfig``."""
        from witwin.radar import RadarConfig

        comp = settings_obj.get_component(MARKER_COMPONENT)
        return RadarConfig.from_dict(ConfigMap.build_dict(comp))

    @staticmethod
    def build_dict(comp: Any) -> Dict[str, Any]:
        """``RadarConfig`` component -> the dict accepted by ``RadarConfig.from_dict``."""
        return {
            "num_tx": int(comp.num_tx),
            "num_rx": int(comp.num_rx),
            "fc": num(comp.fc),
            "slope": num(comp.slope),
            "power": num(comp.power),
            "adc_samples": int(comp.adc_samples),
            "adc_start_time": num(comp.adc_start_time),
            "sample_rate": num(comp.sample_rate),
            "idle_time": num(comp.idle_time),
            "ramp_end_time": num(comp.ramp_end_time),
            "chirp_per_frame": int(comp.chirp_per_frame),
            "frame_per_second": num(comp.frame_per_second),
            "num_doppler_bins": int(comp.num_doppler_bins),
            "num_range_bins": int(comp.num_range_bins),
            "num_angle_bins": int(comp.num_angle_bins),
            "tx_loc": vec_list(comp.tx_loc),
            "rx_loc": vec_list(comp.rx_loc),
        }

    @staticmethod
    def _fill(comp: Any, config: Any) -> None:
        # Platform RadarConfig dataclass -> the RadarConfig component (native units).
        comp.fc = float(config.fc)
        comp.slope = float(config.slope)
        comp.power = float(config.power)
        comp.adc_samples = int(config.adc_samples)
        comp.adc_start_time = float(config.adc_start_time)
        comp.sample_rate = float(config.sample_rate)
        comp.idle_time = float(config.idle_time)
        comp.ramp_end_time = float(config.ramp_end_time)
        comp.chirp_per_frame = int(config.chirp_per_frame)
        comp.frame_per_second = float(config.frame_per_second)
        comp.num_doppler_bins = int(config.num_doppler_bins)
        comp.num_range_bins = int(config.num_range_bins)
        comp.num_angle_bins = int(config.num_angle_bins)
        comp.num_tx = int(config.num_tx)
        comp.num_rx = int(config.num_rx)
        comp.tx_loc = [list(loc) for loc in config.tx_loc]
        comp.rx_loc = [list(loc) for loc in config.rx_loc]
