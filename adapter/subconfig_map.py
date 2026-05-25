"""Sub-config interchange: the 4 ``RadarConfig`` sub-config dicts <-> their components.

Each ``*_to_studio`` reads a validated platform sub-config dict (or ``None``) into its
component; each ``*_build`` reads the component back into the dict shape that
``RadarConfig.from_dict`` validates (or ``None`` when nothing is enabled). Keeping the
dict shapes identical to ``validation.py`` means the platform validator stays the single
source of truth for the round trip.
"""
import json
from typing import Any, Dict, Optional

from .common import num, vec, vec_list


class SubConfigMap:
    """The four ``RadarConfig`` sub-configs <-> their Radar Settings components."""

    # --- antenna pattern -----------------------------------------------------

    @staticmethod
    def antenna_to_studio(comp: Any, pattern: Optional[dict]) -> None:
        # antenna_pattern dict (or None = default dipole) -> RadarAntennaPattern.
        if pattern is None:
            comp.use_default = True
            return
        comp.use_default = False
        comp.pattern_kind = pattern["kind"]
        comp.x_angles_deg = list(pattern["x_angles_deg"])
        comp.y_angles_deg = list(pattern["y_angles_deg"])
        if pattern["kind"] == "separable":
            comp.x_values = list(pattern["x_values"])
            comp.y_values = list(pattern["y_values"])
        else:
            comp.values_2d_json = json.dumps([list(row) for row in pattern["values"]])

    @staticmethod
    def antenna_build(comp: Any) -> Optional[dict]:
        # RadarAntennaPattern -> antenna_pattern dict, or None (use default dipole).
        if bool(comp.use_default):
            return None
        kind = str(comp.pattern_kind)
        out: Dict[str, Any] = {
            "kind": kind,
            "x_angles_deg": [num(a) for a in comp.x_angles_deg],
            "y_angles_deg": [num(a) for a in comp.y_angles_deg],
        }
        if kind == "separable":
            out["x_values"] = [num(v) for v in comp.x_values]
            out["y_values"] = [num(v) for v in comp.y_values]
        else:
            out["values"] = json.loads(str(comp.values_2d_json))
        return out

    # --- noise model ---------------------------------------------------------

    @staticmethod
    def noise_to_studio(comp: Any, noise: Optional[dict]) -> None:
        # noise_model dict (or None) -> RadarNoiseModel.
        if noise is None:
            return
        if "thermal" in noise:
            comp.enable_thermal = True
            comp.thermal_std = float(noise["thermal"]["std"])
        if "quantization" in noise:
            comp.enable_quantization = True
            comp.quant_bits = int(noise["quantization"]["bits"])
            comp.quant_full_scale = float(noise["quantization"]["full_scale"])
        if "phase" in noise:
            comp.enable_phase = True
            comp.phase_std = float(noise["phase"]["std"])
        if "seed" in noise:
            comp.use_seed = True
            comp.seed = int(noise["seed"])

    @staticmethod
    def noise_build(comp: Any) -> Optional[dict]:
        # RadarNoiseModel -> noise_model dict, or None when nothing is enabled.
        out: Dict[str, Any] = {}
        if bool(comp.enable_thermal):
            out["thermal"] = {"std": num(comp.thermal_std)}
        if bool(comp.enable_quantization):
            out["quantization"] = {"bits": int(comp.quant_bits), "full_scale": num(comp.quant_full_scale)}
        if bool(comp.enable_phase):
            out["phase"] = {"std": num(comp.phase_std)}
        if not out:
            return None
        if bool(comp.use_seed):
            out["seed"] = int(comp.seed)
        return out

    # --- polarization --------------------------------------------------------

    @staticmethod
    def pol_to_studio(comp: Any, pol: Optional[dict]) -> None:
        # polarization dict (or None) -> RadarPolarization (collapse equal banks to uniform).
        if pol is None:
            return
        comp.enabled = True
        comp.reflection_flip = bool(pol.get("reflection_flip", True))
        SubConfigMap._bank_to_studio(comp, "tx", pol["tx"])
        SubConfigMap._bank_to_studio(comp, "rx", pol["rx"])

    @staticmethod
    def pol_build(comp: Any) -> Optional[dict]:
        # RadarPolarization -> polarization dict, or None when disabled.
        if not bool(comp.enabled):
            return None
        return {
            "tx": SubConfigMap._bank_build(comp, "tx"),
            "rx": SubConfigMap._bank_build(comp, "rx"),
            "reflection_flip": bool(comp.reflection_flip),
        }

    @staticmethod
    def _bank_to_studio(comp: Any, side: str, vectors: Any) -> None:
        # A validated per-antenna bank collapses to a single shared vector when all equal.
        banks = vec_list(vectors)
        uniform = bool(banks) and all(v == banks[0] for v in banks)
        setattr(comp, f"{side}_uniform", uniform)
        if uniform:
            setattr(comp, side, list(banks[0]))
        else:
            setattr(comp, f"{side}_bank", banks)

    @staticmethod
    def _bank_build(comp: Any, side: str) -> Any:
        # Emit a single 3-vector (validator expands to num_tx/num_rx) or an explicit bank.
        if bool(getattr(comp, f"{side}_uniform")):
            return vec(getattr(comp, side))
        return vec_list(getattr(comp, f"{side}_bank"))

    # --- receiver chain ------------------------------------------------------

    @staticmethod
    def chain_to_studio(comp: Any, chain: Optional[dict]) -> None:
        # receiver_chain dict (or None) -> RadarReceiverChain.
        if chain is None:
            return
        comp.reference_impedance_ohm = float(chain.get("reference_impedance_ohm", 50.0))
        if "lna" in chain:
            comp.enable_lna = True
            comp.lna_gain_db = float(chain["lna"]["gain_db"])
        if "agc" in chain:
            comp.enable_agc = True
            agc = chain["agc"]
            comp.agc_target_rms = float(agc["target_rms"])
            comp.agc_max_gain_db = float(agc["max_gain_db"])
            comp.agc_min_gain_db = float(agc["min_gain_db"])
            comp.agc_mode = str(agc["mode"])
        if "adc" in chain:
            comp.enable_adc = True
            comp.adc_bits = int(chain["adc"]["bits"])
            comp.adc_full_scale = float(chain["adc"]["full_scale"])

    @staticmethod
    def chain_build(comp: Any) -> Optional[dict]:
        # RadarReceiverChain -> receiver_chain dict, or None when no block is enabled.
        out: Dict[str, Any] = {}
        if bool(comp.enable_lna):
            out["lna"] = {"gain_db": num(comp.lna_gain_db)}
        if bool(comp.enable_agc):
            out["agc"] = {
                "target_rms": num(comp.agc_target_rms),
                "max_gain_db": num(comp.agc_max_gain_db),
                "min_gain_db": num(comp.agc_min_gain_db),
                "mode": str(comp.agc_mode),
            }
        if bool(comp.enable_adc):
            out["adc"] = {"bits": int(comp.adc_bits), "full_scale": num(comp.adc_full_scale)}
        if not out:
            return None
        out["reference_impedance_ohm"] = num(comp.reference_impedance_ohm)
        return out
