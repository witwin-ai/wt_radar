"""Current ``RadarConfig`` and typed frontend <-> unified Studio Radar component."""
from dataclasses import replace
from typing import Any, Dict, Optional

from witwin_server import SceneObject

from .common import num, vec_list
from .subconfig_map import SubConfigMap

SETTINGS_NAME = "Radar Settings"
MARKER_COMPONENT = "Radar"


class ConfigMap:
    """Map the Radar 0.3 flat configuration plus its typed frontend."""

    @staticmethod
    def settings_to_studio(config: Optional[Any]) -> SceneObject:
        from ..components.radar import RadarComponent

        obj = SceneObject(name=SETTINGS_NAME, mesh_type="Empty")
        radar = obj.add_component(RadarComponent())
        if config is not None:
            ConfigMap._fill(radar, config)
            SubConfigMap.antenna_to_studio(radar, config.antenna_pattern)
            SubConfigMap.frontend_to_studio(radar, config.frontend)
        return obj

    @staticmethod
    def build_config(settings_obj: SceneObject) -> Any:
        from witwin.radar import RadarConfig

        radar = settings_obj.get_component(MARKER_COMPONENT)
        config = RadarConfig.from_dict(ConfigMap.build_dict(settings_obj))
        frontend = SubConfigMap.frontend_build(radar)
        return config if frontend is None else replace(config, frontend=frontend)

    @staticmethod
    def build_dict(settings_obj: SceneObject) -> Dict[str, Any]:
        radar = settings_obj.get_component(MARKER_COMPONENT)
        out = ConfigMap._core_dict(radar)
        out["antenna_pattern"] = SubConfigMap.antenna_build(radar)
        return out

    @staticmethod
    def _core_dict(radar: Any) -> Dict[str, Any]:
        return {
            "num_tx": int(radar.num_tx),
            "num_rx": int(radar.num_rx),
            "fc": num(radar.fc),
            "slope": num(radar.slope),
            "power": num(radar.power),
            "adc_samples": int(radar.adc_samples),
            "adc_start_time": num(radar.adc_start_time),
            "sample_rate": num(radar.sample_rate),
            "idle_time": num(radar.idle_time),
            "ramp_end_time": num(radar.ramp_end_time),
            "chirp_per_frame": int(radar.chirp_per_frame),
            "frame_per_second": num(radar.frame_per_second),
            "num_doppler_bins": int(radar.num_doppler_bins),
            "num_range_bins": int(radar.num_range_bins),
            "num_angle_bins": int(radar.num_angle_bins),
            "tx_loc": vec_list(radar.tx_loc),
            "rx_loc": vec_list(radar.rx_loc),
        }

    @staticmethod
    def _fill(radar: Any, config: Any) -> None:
        for name in (
            "fc",
            "slope",
            "power",
            "adc_samples",
            "adc_start_time",
            "sample_rate",
            "idle_time",
            "ramp_end_time",
            "chirp_per_frame",
            "frame_per_second",
            "num_doppler_bins",
            "num_range_bins",
            "num_angle_bins",
            "num_tx",
            "num_rx",
        ):
            setattr(radar, name, getattr(config, name))
        radar.tx_loc = [list(location) for location in config.tx_loc]
        radar.rx_loc = [list(location) for location in config.rx_loc]
