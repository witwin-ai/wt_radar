"""Sensor config interchange: ``witwin.radar.Radar`` <-> the unified Radar component.

``settings_to_studio`` builds the single ``Empty`` Radar Settings object carrying ONE
``Radar`` component, then fills its fields from a platform ``Radar`` (the sensor
pose/tracer settings are editor state and keep their defaults, since they are not part
of the pair). ``build_config`` reads the same fields back into a dict, enforces the
adc-vs-quantization cross rule, and hands it to ``Radar.from_dict`` which runs
the platform's own validation — so the editor never emits an invalid config. Values are
stored in the platform's native non-SI units (see ``common``), so the round trip is exact.
"""
from typing import Any, Dict, Optional

from witwin_server import SceneObject

from .common import num, vec_list
from .subconfig_map import SubConfigMap

SETTINGS_NAME = "Radar Settings"
MARKER_COMPONENT = "Radar"


class ConfigMap:
    """Radar 0.4's flat FMCW fields <-> the unified Radar component."""

    @staticmethod
    def settings_to_studio(config: Optional[Any]) -> SceneObject:
        """Build the Radar Settings object; ``None`` leaves every field at defaults."""
        from ..components.radar import RadarComponent

        obj = SceneObject(name=SETTINGS_NAME, mesh_type="Empty")
        radar = obj.add_component(RadarComponent())
        if config is not None:
            ConfigMap._fill(radar, config)
            pattern = getattr(config, "pattern", None)
            if pattern is not None:
                ConfigMap._pattern_to_studio(radar, pattern)
        return obj

    @staticmethod
    def build_config(settings_obj: SceneObject) -> Any:
        """Read the Radar component back into a validated Radar 0.4 object."""
        from witwin.radar import Radar
        from .radar04 import flat_config

        config_dict = ConfigMap.build_dict(settings_obj)
        ConfigMap._check_cross_rules(config_dict)
        return Radar.from_dict(flat_config(config_dict))

    @staticmethod
    def build_dict(settings_obj: SceneObject) -> Dict[str, Any]:
        """The unified Radar component -> the dict accepted by ``RadarConfig.from_dict``."""
        radar = settings_obj.get_component(MARKER_COMPONENT)
        out = ConfigMap._core_dict(radar)
        out.update(ConfigMap._subconfig_dict(radar))
        return out

    # --- core fields ---------------------------------------------------------

    @staticmethod
    def _core_dict(radar: Any) -> Dict[str, Any]:
        # The 17 RadarConfig core fields (native units, verbatim).
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
        # Platform RadarConfig dataclass -> the unified Radar component (native units).
        waveform = config.waveform
        radar.fc = float(config.carrier)
        radar.slope = float(waveform.slope) / 1e12
        radar.power = float(config.power)
        radar.adc_samples = int(waveform.samples_per_chirp)
        radar.adc_start_time = float(waveform.adc_start) * 1e6
        radar.sample_rate = float(waveform.sample_rate) / 1e3
        radar.idle_time = float(waveform.idle) * 1e6
        radar.ramp_end_time = float(waveform.ramp_end) * 1e6
        radar.chirp_per_frame = int(waveform.chirps_per_frame)
        radar.num_doppler_bins = int(waveform.chirps_per_frame)
        radar.num_range_bins = int(waveform.samples_per_chirp)
        radar.num_tx = len(config.tx)
        radar.num_rx = len(config.rx)
        radar.tx_loc = [list(loc) for loc in config.tx]
        radar.rx_loc = [list(loc) for loc in config.rx]

    @staticmethod
    def _pattern_to_studio(radar: Any, pattern: Any) -> None:
        """Map Radar 0.4's typed Pattern without reviving the 0.3 schema."""

        radar.use_default = False
        radar.pattern_kind = "map" if pattern.kind == "map" else "separable"
        radar.x_angles_deg = list(pattern.x_angles)
        radar.y_angles_deg = list(pattern.y_angles)
        if pattern.kind == "map":
            import json
            radar.values_2d_json = json.dumps([list(row) for row in pattern.gain])
        else:
            radar.x_values = list(pattern.x_gain)
            radar.y_values = list(pattern.y_gain)

    # --- sub-configs + cross rules -------------------------------------------

    @staticmethod
    def _subconfig_dict(radar: Any) -> Dict[str, Any]:
        # The four optional sub-configs read off the same Radar component.
        return {
            "antenna_pattern": SubConfigMap.antenna_build(radar),
            "noise_model": SubConfigMap.noise_build(radar),
            "polarization": SubConfigMap.pol_build(radar),
            "receiver_chain": SubConfigMap.chain_build(radar),
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
