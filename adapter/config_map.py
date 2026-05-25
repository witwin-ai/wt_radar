"""Sensor config interchange: ``witwin.radar.RadarConfig`` <-> the Radar Settings object.

``settings_to_studio`` builds the single ``Empty`` Radar Settings object — the
``RadarConfig`` core, the sensor pose/backend, and the four sub-config components — and
fills them from a platform ``RadarConfig`` (the sensor pose is editor state, not part of
the pair, so it keeps its defaults). ``build_config`` reads the components back into a
dict, enforces the adc-vs-quantization cross rule, and hands it to
``RadarConfig.from_dict`` which runs the platform's own validation — so the editor never
emits an invalid config. Values are stored in the platform's native non-SI units (see
``common``), so the round trip is exact.
"""
from typing import Any, Dict, Optional

from witwin_server import SceneObject

from .common import num, vec_list
from .subconfig_map import SubConfigMap

SETTINGS_NAME = "Radar Settings"
MARKER_COMPONENT = "RadarConfig"


class ConfigMap:
    """``RadarConfig`` (+ sensor + 4 sub-configs) <-> the Radar Settings object."""

    @staticmethod
    def settings_to_studio(config: Optional[Any]) -> SceneObject:
        """Build the Radar Settings object; ``None`` leaves every component at defaults."""
        from ..components.config import RadarConfigComponent
        from ..components.sensor import RadarSensorComponent
        from ..components.subconfigs import (
            RadarAntennaPatternComponent,
            RadarNoiseModelComponent,
            RadarPolarizationComponent,
            RadarReceiverChainComponent,
        )

        obj = SceneObject(name=SETTINGS_NAME, mesh_type="Empty")
        cfg = obj.add_component(RadarConfigComponent())
        obj.add_component(RadarSensorComponent())
        antenna = obj.add_component(RadarAntennaPatternComponent())
        noise = obj.add_component(RadarNoiseModelComponent())
        polar = obj.add_component(RadarPolarizationComponent())
        chain = obj.add_component(RadarReceiverChainComponent())
        if config is not None:
            ConfigMap._fill(cfg, config)
            SubConfigMap.antenna_to_studio(antenna, config.antenna_pattern)
            SubConfigMap.noise_to_studio(noise, config.noise_model)
            SubConfigMap.pol_to_studio(polar, config.polarization)
            SubConfigMap.chain_to_studio(chain, config.receiver_chain)
        return obj

    @staticmethod
    def build_config(settings_obj: SceneObject) -> Any:
        """Read the Radar Settings components back into a validated ``RadarConfig``."""
        from witwin.radar import RadarConfig

        config_dict = ConfigMap.build_dict(settings_obj)
        ConfigMap._check_cross_rules(config_dict)
        return RadarConfig.from_dict(config_dict)

    @staticmethod
    def build_dict(settings_obj: SceneObject) -> Dict[str, Any]:
        """Radar Settings components -> the dict accepted by ``RadarConfig.from_dict``."""
        comp = settings_obj.get_component(MARKER_COMPONENT)
        out = ConfigMap._core_dict(comp)
        out.update(ConfigMap._subconfig_dict(settings_obj))
        return out

    # --- core fields ---------------------------------------------------------

    @staticmethod
    def _core_dict(comp: Any) -> Dict[str, Any]:
        # The 17 RadarConfig core fields (native units, verbatim).
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

    # --- sub-configs + cross rules -------------------------------------------

    @staticmethod
    def _subconfig_dict(settings_obj: SceneObject) -> Dict[str, Any]:
        # The four optional sub-configs (each None when its component is disabled).
        antenna = settings_obj.get_component("RadarAntennaPattern")
        noise = settings_obj.get_component("RadarNoiseModel")
        polar = settings_obj.get_component("RadarPolarization")
        chain = settings_obj.get_component("RadarReceiverChain")
        return {
            "antenna_pattern": SubConfigMap.antenna_build(antenna) if antenna is not None else None,
            "noise_model": SubConfigMap.noise_build(noise) if noise is not None else None,
            "polarization": SubConfigMap.pol_build(polar) if polar is not None else None,
            "receiver_chain": SubConfigMap.chain_build(chain) if chain is not None else None,
        }

    @staticmethod
    def _check_cross_rules(config_dict: Dict[str, Any]) -> None:
        # Mirror Radar.__init__: ADC quantization can live in exactly one place.
        chain = config_dict.get("receiver_chain")
        noise = config_dict.get("noise_model")
        if (chain is not None and chain.get("adc") is not None
                and noise is not None and noise.get("quantization") is not None):
            raise ValueError(
                "Radar receiver_chain.adc and noise_model.quantization cannot both be "
                "enabled; use one quantizer.")
