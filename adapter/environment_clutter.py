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
    from witwin.radar.simulation import ScatterSitePolicy, bind_radar_world
    from witwin.radar.synthesis import SlowTimeMode, SynthesisPathBatch, select_component
    from witwin.radar.synthesis.assembly import assemble_frame_cube

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
    leg = adapter.reevaluate(frozen, binding.transmitters, binding.receivers, ad_mode="none")
    paths = composer.compose(leg)

    material_slot_count = len(compiled.materials.material_keys)
    declaration = ComponentDeclaration(
        clutter_material_slots=frozenset(range(material_slot_count)),
        multi_interaction_depth=1,
    )
    index = RadarComponentIndex.from_direct(composer, frozen, declaration)
    reflected_path_count = index.count(ENVIRONMENT_CLUTTER)
    if reflected_path_count != paths.path_count:
        raise RuntimeError(
            "Single-bounce environment discovery returned a path that was not "
            "classified as reflected scene clutter."
        )

    mode = SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
    synthesis_paths = SynthesisPathBatch.from_radar_paths(paths, slow_time_mode=mode)
    clutter_paths = select_component(synthesis_paths, index, ENVIRONMENT_CLUTTER)
    synthesis = radar._synthesize(clutter_paths, slow_time_mode=mode)
    array = radar.system_config.sensors.array
    cube = assemble_frame_cube(
        synthesis.cube,
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
    "EnvironmentClutterResult",
    "SINGLE_BOUNCE_COMPONENTS",
    "single_bounce_environment_cube",
]
