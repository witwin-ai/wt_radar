"""The four optional ``RadarConfig`` sub-configs as components on the Radar Settings object.

Each maps to a validated dict on ``RadarConfig`` (antenna_pattern / noise_model /
polarization / receiver_chain), or to ``None`` when nothing is enabled. The adapter
(``subconfig_map``) reads each component into the dict shape that ``RadarConfig.from_dict``
validates, so the platform's own rules (monotonic antenna angles, >=1 enabled noise/rx
block, per-antenna polarization counts, gain ordering) are the source of truth.
"""
from witwin_server.components import (
    Component,
    bool_field,
    component,
    float_field,
    int_field,
    list_field,
    string_field,
    vector3_field,
)

_CAT = "Simulation/Radar"
_SEPARABLE = {"field_name": "pattern_kind", "operator": "eq", "value": "separable"}
_MAP = {"field_name": "pattern_kind", "operator": "eq", "value": "map"}
_AGC = "enable_agc"


@component(name="RadarAntennaPattern", category=_CAT)
class RadarAntennaPatternComponent(Component):
    """TX/RX antenna gain pattern -> ``antenna_pattern`` (None = default half-wave dipole)."""

    use_default = bool_field(True, description="Use the default half-wave dipole pattern")
    pattern_kind = string_field("separable", options=["separable", "map"], enum_toggle=True,
                                hide_if="use_default", description="Separable per-axis cuts or a 2D gain map")
    x_angles_deg = list_field(float_field(0.0), default=[], hide_if="use_default",
                              description="Azimuth angles (deg, strictly increasing, >=2)")
    y_angles_deg = list_field(float_field(0.0), default=[], hide_if="use_default",
                              description="Elevation angles (deg, strictly increasing, >=2)")
    x_values = list_field(float_field(0.0), default=[], show_if=_SEPARABLE,
                          description="Per-azimuth gain (>=0, len == x_angles)")
    y_values = list_field(float_field(0.0), default=[], show_if=_SEPARABLE,
                          description="Per-elevation gain (>=0, len == y_angles)")
    values_2d_json = string_field("", widget="textarea", show_if=_MAP,
                                  description="2D gain map as JSON rows x cols (rows=y_angles, cols=x_angles, >=0)")


@component(name="RadarNoiseModel", category=_CAT)
class RadarNoiseModelComponent(Component):
    """Sensor noise -> ``noise_model`` (None unless at least one block is enabled)."""

    enable_thermal = bool_field(False, description="Additive complex Gaussian thermal noise")
    thermal_std = float_field(0.0, min=0.0, show_if="enable_thermal", description="Thermal noise std (>=0)")
    enable_quantization = bool_field(False, description="ADC-style amplitude quantization")
    quant_bits = int_field(12, min=1, show_if="enable_quantization", description="Quantizer bits (>0)")
    quant_full_scale = float_field(1.0, min=0.0, show_if="enable_quantization", description="Quantizer full scale (>0)")
    enable_phase = bool_field(False, description="Random-walk phase noise")
    phase_std = float_field(0.0, min=0.0, show_if="enable_phase", description="Phase noise std per step (>=0)")
    use_seed = bool_field(False, description="Deterministic noise via a fixed seed")
    seed = int_field(0, min=0, show_if="use_seed", description="Noise RNG seed (>=0)")


@component(name="RadarPolarization", category=_CAT)
class RadarPolarizationComponent(Component):
    """TX/RX polarization -> ``polarization`` (None when disabled)."""

    enabled = bool_field(False, description="Enable explicit polarization (else unpolarized)")
    tx_uniform = bool_field(True, show_if="enabled", description="One shared TX vector (else per-antenna)")
    tx = vector3_field([1.0, 0.0, 0.0], show_if="tx_uniform", description="Shared TX polarization vector (non-zero)")
    tx_bank = list_field(vector3_field([1.0, 0.0, 0.0]), default=[], hide_if="tx_uniform",
                         description="Per-TX-antenna polarization vectors (len == num_tx)")
    rx_uniform = bool_field(True, show_if="enabled", description="One shared RX vector (else per-antenna)")
    rx = vector3_field([1.0, 0.0, 0.0], show_if="rx_uniform", description="Shared RX polarization vector (non-zero)")
    rx_bank = list_field(vector3_field([1.0, 0.0, 0.0]), default=[], hide_if="rx_uniform",
                         description="Per-RX-antenna polarization vectors (len == num_rx)")
    reflection_flip = bool_field(True, show_if="enabled", description="Flip polarization on reflection")


@component(name="RadarReceiverChain", category=_CAT)
class RadarReceiverChainComponent(Component):
    """Receiver chain (LNA/AGC/ADC) -> ``receiver_chain`` (None unless a block is enabled)."""

    enable_lna = bool_field(False, description="Low-noise amplifier gain")
    lna_gain_db = float_field(0.0, show_if="enable_lna", description="LNA gain (dB)")
    enable_agc = bool_field(False, description="Automatic gain control")
    agc_target_rms = float_field(1.0, min=0.0, show_if=_AGC, description="AGC target RMS (>0)")
    agc_max_gain_db = float_field(60.0, show_if=_AGC, description="AGC max gain (dB)")
    agc_min_gain_db = float_field(-60.0, show_if=_AGC, description="AGC min gain (dB, <= max)")
    agc_mode = string_field("per_rx", options=["global", "per_rx"], enum_toggle=True, show_if=_AGC,
                            description="AGC scope")
    enable_adc = bool_field(False, description="ADC quantization (mutually exclusive with quantization noise)")
    adc_bits = int_field(12, min=1, show_if="enable_adc", description="ADC bits (>0)")
    adc_full_scale = float_field(1.0, min=0.0, show_if="enable_adc", description="ADC full scale (>0)")
    reference_impedance_ohm = float_field(50.0, min=0.0, description="Reference impedance (ohm, >0)")
