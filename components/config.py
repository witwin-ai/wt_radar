"""The FMCW core sensor config -> a ``RadarConfig`` component on a Radar Settings object.

This is the 17-field core of ``witwin.radar.RadarConfig`` (carrier/chirp/ADC/frame +
antenna geometry), laid out with foldout groups and the non-SI unit pickers from
``adapter.common`` so the stored values match the platform dataclass verbatim. The
four optional sub-configs (antenna pattern / noise / polarization / receiver chain)
and the sensor pose/backend live in sibling components added in R1.

The Radar Settings object is recognized by the presence of this component (it is the
adapter's marker component), mirroring how the Maxwell adapter keys off MaxwellDomain.
"""
from witwin_server import Notifications
from witwin_server.components import (
    Component,
    button,
    component,
    define_group,
    float_field,
    foldout_group,
    int_field,
    list_field,
    vector3_field,
)

from ..adapter.common import FREQUENCY_UNITS, SAMPLE_RATE_UNITS, SLOPE_UNITS, TIME_UNITS, Derived

_CAT = "Simulation/Radar"


@component(name="RadarConfig", category=_CAT)
class RadarConfigComponent(Component):
    """FMCW core config -> ``witwin.radar.RadarConfig`` (marks the Radar Settings object)."""

    define_group(foldout_group("Frequency", display_name="Frequency / Power"))
    define_group(foldout_group("ADC", display_name="ADC / Sampling"))
    define_group(foldout_group("Frame", display_name="Chirp / Frame"))
    define_group(foldout_group("Bins", display_name="FFT Bins"))
    define_group(foldout_group("Antenna", display_name="Antenna Geometry"))

    # --- Frequency / power ---------------------------------------------------
    fc = float_field(77e9, min=1e9, max=300e9, units=FREQUENCY_UNITS, default_unit="GHz",
                     group="Frequency", description="Carrier / start frequency")
    slope = float_field(60.012, min=0.0, units=SLOPE_UNITS, default_unit="MHz/us",
                        group="Frequency", description="Chirp frequency slope")
    power = float_field(15.0, group="Frequency", description="TX power (dBm)")

    # --- ADC / sampling -----------------------------------------------------
    adc_samples = int_field(256, min=1, group="ADC", description="ADC samples per chirp (fast time)")
    sample_rate = float_field(4400.0, min=0.0, units=SAMPLE_RATE_UNITS, default_unit="ksps",
                              group="ADC", description="ADC sample rate")
    adc_start_time = float_field(6.0, min=0.0, units=TIME_UNITS, default_unit="us",
                                 group="ADC", description="ADC start delay")

    # --- chirp / frame ------------------------------------------------------
    idle_time = float_field(7.0, min=0.0, units=TIME_UNITS, default_unit="us",
                            group="Frame", description="Idle time between chirps")
    ramp_end_time = float_field(58.0, min=0.0, units=TIME_UNITS, default_unit="us",
                                group="Frame", description="Active chirp ramp time")
    chirp_per_frame = int_field(128, min=1, group="Frame", description="Chirps per frame (slow time / Doppler)")
    frame_per_second = float_field(10.0, min=0.0, group="Frame", description="Frame rate (Hz)")

    # --- FFT bins -----------------------------------------------------------
    num_doppler_bins = int_field(128, min=1, group="Bins", description="Doppler FFT bins")
    num_range_bins = int_field(256, min=1, group="Bins", description="Range bins")
    num_angle_bins = int_field(64, min=1, group="Bins", description="Angle FFT bins")

    # --- antenna geometry (half-wavelength units; counts must match loc lengths) ---
    num_tx = int_field(3, min=1, group="Antenna", description="Number of TX antennas (== len(tx_loc))")
    num_rx = int_field(4, min=1, group="Antenna", description="Number of RX antennas (== len(rx_loc))")
    tx_loc = list_field(vector3_field([0.0, 0.0, 0.0]),
                        default=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
                        group="Antenna", description="TX positions (half-wavelength units)")
    rx_loc = list_field(vector3_field([0.0, 0.0, 0.0]),
                        default=[[-6.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [-3.0, 0.0, 0.0]],
                        group="Antenna", description="RX positions (half-wavelength units)")

    @button(display_name="Show Derived Values")
    def show_derived(self):
        """Report the derived range/doppler resolution + max range/doppler (authoring feedback)."""
        d = Derived.compute(self)
        msg = (f"range res {d['range_resolution_m']:.4f} m | max range {d['max_range_m']:.2f} m | "
               f"doppler res {d['doppler_resolution_mps']:.4f} m/s | max doppler {d['max_doppler_mps']:.2f} m/s")
        Notifications.info("Radar", msg)
        return msg
