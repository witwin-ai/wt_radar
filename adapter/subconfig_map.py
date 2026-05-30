"""Sub-config interchange: the 4 ``RadarConfig`` sub-config dicts <-> the unified Radar component.

Each ``*_to_studio`` reads a validated platform sub-config dict (or ``None``) into the
relevant fields on the unified Radar component; each ``*_build`` reads the same fields
back into the dict shape that ``RadarConfig.from_dict`` validates (or ``None`` when nothing
is enabled). Keeping the dict shapes identical to ``validation.py`` means the platform
validator stays the single source of truth for the round trip.

The polarization fields are namespaced ``pol_*`` on the Radar component (the old
RadarPolarization component used short names that collided once everything moved into
one component).
"""
import json
from typing import Any, Dict, Optional

from .common import num, vec, vec_list


class SubConfigMap:
    """The four ``RadarConfig`` sub-configs <-> their fields on the unified Radar component."""

    # --- antenna pattern -----------------------------------------------------

    @staticmethod
    def antenna_to_studio(radar: Any, pattern: Optional[dict]) -> None:
        # antenna_pattern dict (or None = default dipole) -> antenna-pattern fields.
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
        # antenna-pattern fields -> antenna_pattern dict, or None (use default dipole).
        if bool(radar.use_default):
            return None
        kind = str(radar.pattern_kind)
        out: Dict[str, Any] = {
            "kind": kind,
            "x_angles_deg": [num(a) for a in radar.x_angles_deg],
            "y_angles_deg": [num(a) for a in radar.y_angles_deg],
        }
        if kind == "separable":
            out["x_values"] = [num(v) for v in radar.x_values]
            out["y_values"] = [num(v) for v in radar.y_values]
        else:
            out["values"] = json.loads(str(radar.values_2d_json))
        return out

    # --- noise model ---------------------------------------------------------

    @staticmethod
    def noise_to_studio(radar: Any, noise: Optional[dict]) -> None:
        # noise_model dict (or None) -> noise fields on the Radar component.
        if noise is None:
            return
        if "thermal" in noise:
            radar.enable_thermal = True
            radar.thermal_std = float(noise["thermal"]["std"])
        if "quantization" in noise:
            radar.enable_quantization = True
            radar.quant_bits = int(noise["quantization"]["bits"])
            radar.quant_full_scale = float(noise["quantization"]["full_scale"])
        if "phase" in noise:
            radar.enable_phase = True
            radar.phase_std = float(noise["phase"]["std"])
        if "seed" in noise:
            radar.use_seed = True
            radar.seed = int(noise["seed"])

    @staticmethod
    def noise_build(radar: Any) -> Optional[dict]:
        # noise fields -> noise_model dict, or None when nothing is enabled.
        out: Dict[str, Any] = {}
        if bool(radar.enable_thermal):
            out["thermal"] = {"std": num(radar.thermal_std)}
        if bool(radar.enable_quantization):
            out["quantization"] = {"bits": int(radar.quant_bits),
                                   "full_scale": num(radar.quant_full_scale)}
        if bool(radar.enable_phase):
            out["phase"] = {"std": num(radar.phase_std)}
        if not out:
            return None
        if bool(radar.use_seed):
            out["seed"] = int(radar.seed)
        return out

    # --- polarization --------------------------------------------------------

    @staticmethod
    def pol_to_studio(radar: Any, pol: Optional[dict]) -> None:
        # polarization dict (or None) -> pol_* fields (collapse equal banks to uniform).
        if pol is None:
            return
        radar.pol_enabled = True
        radar.pol_reflection_flip = bool(pol.get("reflection_flip", True))
        SubConfigMap._bank_to_studio(radar, "tx", pol["tx"])
        SubConfigMap._bank_to_studio(radar, "rx", pol["rx"])

    @staticmethod
    def pol_build(radar: Any) -> Optional[dict]:
        # pol_* fields -> polarization dict, or None when disabled.
        if not bool(radar.pol_enabled):
            return None
        return {
            "tx": SubConfigMap._bank_build(radar, "tx"),
            "rx": SubConfigMap._bank_build(radar, "rx"),
            "reflection_flip": bool(radar.pol_reflection_flip),
        }

    @staticmethod
    def _bank_to_studio(radar: Any, side: str, vectors: Any) -> None:
        # A validated per-antenna bank collapses to a single shared vector when all equal.
        banks = vec_list(vectors)
        uniform = bool(banks) and all(v == banks[0] for v in banks)
        setattr(radar, f"pol_{side}_uniform", uniform)
        if uniform:
            setattr(radar, f"pol_{side}", list(banks[0]))
        else:
            setattr(radar, f"pol_{side}_bank", banks)

    @staticmethod
    def _bank_build(radar: Any, side: str) -> Any:
        # Emit a single 3-vector (validator expands to num_tx/num_rx) or an explicit bank.
        if bool(getattr(radar, f"pol_{side}_uniform")):
            return vec(getattr(radar, f"pol_{side}"))
        return vec_list(getattr(radar, f"pol_{side}_bank"))

    # --- receiver chain ------------------------------------------------------

    @staticmethod
    def chain_to_studio(radar: Any, chain: Optional[dict]) -> None:
        # receiver_chain dict (or None) -> receiver-chain fields on the Radar component.
        if chain is None:
            return
        radar.reference_impedance_ohm = float(chain.get("reference_impedance_ohm", 50.0))
        if "lna" in chain:
            radar.enable_lna = True
            radar.lna_gain_db = float(chain["lna"]["gain_db"])
        if "agc" in chain:
            radar.enable_agc = True
            agc = chain["agc"]
            radar.agc_target_rms = float(agc["target_rms"])
            radar.agc_max_gain_db = float(agc["max_gain_db"])
            radar.agc_min_gain_db = float(agc["min_gain_db"])
            radar.agc_mode = str(agc["mode"])
        if "adc" in chain:
            radar.enable_adc = True
            radar.adc_bits = int(chain["adc"]["bits"])
            radar.adc_full_scale = float(chain["adc"]["full_scale"])

    @staticmethod
    def chain_build(radar: Any) -> Optional[dict]:
        # receiver-chain fields -> receiver_chain dict, or None when no block is enabled.
        out: Dict[str, Any] = {}
        if bool(radar.enable_lna):
            out["lna"] = {"gain_db": num(radar.lna_gain_db)}
        if bool(radar.enable_agc):
            out["agc"] = {
                "target_rms": num(radar.agc_target_rms),
                "max_gain_db": num(radar.agc_max_gain_db),
                "min_gain_db": num(radar.agc_min_gain_db),
                "mode": str(radar.agc_mode),
            }
        if bool(radar.enable_adc):
            out["adc"] = {"bits": int(radar.adc_bits), "full_scale": num(radar.adc_full_scale)}
        if not out:
            return None
        out["reference_impedance_ohm"] = num(radar.reference_impedance_ohm)
        return out
