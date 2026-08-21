"""Radar antenna-pattern and typed frontend interchange."""
import json
from typing import Any, Dict, Optional

from .common import num


class SubConfigMap:
    """Map the two optional RadarConfig blocks represented by the Studio component."""

    @staticmethod
    def antenna_to_studio(radar: Any, pattern: Optional[dict]) -> None:
        if pattern is None:
            radar.use_default = True
            return
        radar.use_default = False
        radar.pattern_kind = pattern["kind"]
        radar.x_angles_deg = list(pattern["x_angles_deg"])
        radar.y_angles_deg = list(pattern["y_angles_deg"])
        if pattern["kind"] == "separable":
            radar.x_values = list(pattern["x_values"])
            radar.y_values = list(pattern["y_values"])
        else:
            radar.values_2d_json = json.dumps([list(row) for row in pattern["values"]])

    @staticmethod
    def antenna_build(radar: Any) -> Optional[dict]:
        if bool(radar.use_default):
            return None
        kind = str(radar.pattern_kind)
        out: Dict[str, Any] = {
            "kind": kind,
            "x_angles_deg": [num(value) for value in radar.x_angles_deg],
            "y_angles_deg": [num(value) for value in radar.y_angles_deg],
        }
        if kind == "separable":
            out["x_values"] = [num(value) for value in radar.x_values]
            out["y_values"] = [num(value) for value in radar.y_values]
        else:
            out["values"] = json.loads(str(radar.values_2d_json))
        return out

    @staticmethod
    def frontend_to_studio(radar: Any, frontend: Any) -> None:
        if frontend is None:
            return
        radar.reference_impedance_ohm = float(frontend.port.reference_impedance_ohm)
        if frontend.noise is not None:
            radar.enable_thermal = True
            radar.noise_figure_db = float(frontend.noise.noise_figure_db)
            radar.antenna_temperature_k = float(frontend.noise.antenna_temperature_k)
            radar.noise_bandwidth_hz = float(frontend.noise.bandwidth_hz)
            if frontend.noise.phase_noise_dbc_per_hz is not None:
                radar.enable_phase = True
                radar.phase_noise_dbc_per_hz = float(frontend.noise.phase_noise_dbc_per_hz)
                radar.phase_offset_hz = float(frontend.noise.phase_offset_hz)
                radar.phase_sample_rate_hz = float(frontend.noise.phase_sample_rate_hz)
        if frontend.lna is not None:
            radar.enable_lna = True
            radar.lna_gain_db = float(frontend.lna.gain_db)
        if frontend.agc is not None:
            radar.enable_agc = True
            radar.agc_target_rms = float(frontend.agc.target_rms)
            radar.agc_mode = str(frontend.agc.mode)
            radar.agc_min_gain_db = float(frontend.agc.min_gain_db)
            radar.agc_max_gain_db = float(frontend.agc.max_gain_db)
        if frontend.adc is not None:
            radar.enable_adc = True
            radar.adc_bits = int(frontend.adc.bits)
            radar.adc_full_scale = float(frontend.adc.full_scale)
        if int(frontend.seed.seed_base) != 0:
            radar.use_seed = True
            radar.seed = int(frontend.seed.seed_base)

    @staticmethod
    def frontend_build(radar: Any) -> Any:
        from witwin.radar.frontend import AdcSpec, AgcSpec, FrontendSpec, LnaSpec, NoiseSpec, PortSpec, SeedSpec

        noise = None
        if bool(radar.enable_thermal) or bool(radar.enable_phase):
            phase_level = num(radar.phase_noise_dbc_per_hz) if bool(radar.enable_phase) else None
            noise = NoiseSpec(
                noise_figure_db=num(radar.noise_figure_db),
                antenna_temperature_k=num(radar.antenna_temperature_k),
                bandwidth_hz=num(radar.noise_bandwidth_hz),
                phase_noise_dbc_per_hz=phase_level,
                phase_offset_hz=num(radar.phase_offset_hz),
                phase_sample_rate_hz=num(radar.phase_sample_rate_hz),
            )
        lna = LnaSpec(gain_db=num(radar.lna_gain_db)) if bool(radar.enable_lna) else None
        agc = (
            AgcSpec(
                target_rms=num(radar.agc_target_rms),
                mode=str(radar.agc_mode),
                min_gain_db=num(radar.agc_min_gain_db),
                max_gain_db=num(radar.agc_max_gain_db),
            )
            if bool(radar.enable_agc)
            else None
        )
        adc = AdcSpec(bits=int(radar.adc_bits), full_scale=num(radar.adc_full_scale)) if bool(radar.enable_adc) else None
        seed = SeedSpec(int(radar.seed) if bool(radar.use_seed) else 0)
        port = PortSpec(reference_impedance_ohm=num(radar.reference_impedance_ohm))
        if noise is None and lna is None and agc is None and adc is None and seed.seed_base == 0 and port.reference_impedance_ohm == 50.0:
            return None
        return FrontendSpec(port=port, noise=noise, lna=lna, agc=agc, adc=adc, seed=seed)
