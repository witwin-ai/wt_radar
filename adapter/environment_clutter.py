"""Native single-bounce returns from Studio's exported static geometry.

Channel owns reflection discovery and material response.  This adapter only
connects that direct Tx -> environment -> Rx route to Radar synthesis and
keeps antenna leakage out of the returned cube.
"""
from dataclasses import dataclass

import torch


SINGLE_BOUNCE_COMPONENTS = frozenset({"reflection"})


@dataclass(frozen=True)
class EnvironmentClutterResult:
    """One static raw Radar frame and evidence about its reflected paths."""

    cube: torch.Tensor
    reflected_path_count: int
    material_slot_count: int
    max_depth: int = 1


def add_environment_to_fmcw_result(radar, target_cube, environment):
    """Coherently add one static environment return in the result domain.

    ``single_bounce_environment_cube`` returns the native synthesis product in
    the instrument's declared FMCW domain, but before the receive chain.
    Studio's Radar 0.4 adapter currently refuses every legacy frontend option,
    so the configured instrument has no nonlinear receive stage. Refuse a
    future frontend here rather than silently combining two products after a
    nonlinear AGC or ADC.
    """
    if radar.frontend is not None:
        raise NotImplementedError(
            "Static environment returns must be combined before a configured "
            "Radar receive frontend; the Studio Radar 0.4 adapter does not "
            "support that frontend yet."
        )
    if radar.waveform.output not in ("spectrum", "beat"):
        raise ValueError(f"Unsupported FMCW output domain: {radar.waveform.output!r}")
    clutter_cube = environment.cube
    if clutter_cube.shape != target_cube.shape:
        raise ValueError(
            "Environment and target Radar cubes disagree: "
            f"{tuple(clutter_cube.shape)} != {tuple(target_cube.shape)}"
        )
    return target_cube + clutter_cube


def single_bounce_environment_cube(radar, world, *, polarization):
    """Synthesize the static scene's native one-reflection Radar return.

    The returned cube is before the receive frontend.  It can therefore be
    added coherently to the target cube before processing.  No mesh points,
    furniture names, or room-specific scattering amplitudes are invented in
    this plugin: Channel discovers the specular path and evaluates the actual
    exported material.
    """
    from witwin.radar.channel import ChannelPropagationAdapter, compile_scene
    from witwin.radar.paths import (
        ENVIRONMENT_CLUTTER,
        ComponentDeclaration,
        DirectComposer,
        RadarComponentIndex,
    )
    from witwin.radar.sensors import RoundTripPatternStage
    from witwin.radar.simulation import ScatterSitePolicy, bind_radar_world
    from witwin.radar.synthesis import (
        SlowTimeMode,
        SynthesisPathBatch,
        select_component,
    )
    from witwin.radar.synthesis.assembly import assemble_frame_cube
    from witwin.radar.synthesis.fmcw import synthesize_fmcw

    propagation = radar.system_config.propagation
    reference_frequency_hz = propagation.reference_frequency_hz
    compiled = compile_scene(world, reference_frequency_hz=reference_frequency_hz)

    # bind_radar_world is the public owner of array endpoint IDs and powers.
    # The direct environment route does not consume a scatter site; one dummy
    # position is supplied only because the binding contract describes both
    # direct and two-way endpoint sets together.
    dummy_site = radar.tx_pos[:1].detach().clone().contiguous()
    binding = bind_radar_world(
        radar,
        world,
        sites=ScatterSitePolicy.explicit(dummy_site),
        polarization=tuple(polarization),
    )
    adapter = ChannelPropagationAdapter(
        compiled,
        reference_frequency_hz=reference_frequency_hz,
        components=SINGLE_BOUNCE_COMPONENTS,
        max_depth=1,
    )
    frozen = adapter.freeze(binding.transmitters, binding.receivers)
    composer = DirectComposer.freeze(
        frozen,
        radar_source_ids=binding.transmitter_ids,
        radar_sink_ids=binding.receiver_ids,
        reference_frequency_hz=reference_frequency_hz,
    )
    leg = adapter.reevaluate_slots(
        frozen,
        binding.transmitters,
        binding.receivers,
        slot_count=1,
        ad_mode="none",
    )
    paths = composer.compose(leg)

    material_slot_count = len(compiled.materials.material_keys)
    declaration = ComponentDeclaration(
        clutter_material_slots=frozenset(range(material_slot_count)),
        multi_interaction_depth=1,
    )
    index = RadarComponentIndex.from_direct(composer, frozen, declaration)
    reflected_path_count = index.count(ENVIRONMENT_CLUTTER)
    stage = RoundTripPatternStage.freeze(
        radar,
        composer,
        site_ids=(),
        pattern=radar.pattern,
    )
    if leg.departure_target_m is None or leg.arrival_origin_m is None:
        raise RuntimeError(
            "Channel did not publish the interaction points required for the "
            "Radar antenna pattern on direct environment paths."
        )
    rows = composer.row_index
    paths = stage.apply(
        paths,
        tx_pos=radar.tx_pos,
        rx_pos=radar.rx_pos,
        tx_targets_m=leg.departure_target_m.index_select(0, rows).contiguous(),
        rx_targets_m=leg.arrival_origin_m.index_select(0, rows).contiguous(),
    )

    mode = SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
    synthesis_paths = SynthesisPathBatch.from_radar_paths(paths, slow_time_mode=mode)
    clutter_paths = select_component(synthesis_paths, index, ENVIRONMENT_CLUTTER)
    synthesis_cube = synthesize_fmcw(
        clutter_paths,
        radar.system_config.waveform_spec(),
    )
    array = radar.system_config.sensors.array
    cube = assemble_frame_cube(
        synthesis_cube,
        num_tx=array.num_tx,
        num_rx=array.num_rx,
    )
    if cube.device.type != "cuda" or not torch.isfinite(cube).all():
        raise RuntimeError("Native single-bounce environment cube is missing or nonfinite.")
    return EnvironmentClutterResult(
        cube=cube,
        reflected_path_count=reflected_path_count,
        material_slot_count=material_slot_count,
    )


__all__ = [
    "add_environment_to_fmcw_result",
    "EnvironmentClutterResult",
    "SINGLE_BOUNCE_COMPONENTS",
    "single_bounce_environment_cube",
]
