# wt-radar

Round-trips `witwin.radar` solver scenes to and from the witwin-studio editor and runs
the FMCW radar engine, with in-component signal visualization. It is an independent
plugin (its own git repo) built on the frozen core base layer
(`witwin_server.platform_bridge`) — see
[RADAR_PLUGIN_PLAN.md](../../RADAR_PLUGIN_PLAN.md) and the master
[PLATFORM_SCENE_INTEGRATION_PLAN.md](../../PLATFORM_SCENE_INTEGRATION_PLAN.md).

This supersedes the old `wt_radar` (fake `SimpleRadar`); history is preserved in this
same repo.

## What's special about radar

- **The sensor lives outside the scene.** The load/export contract carries a
  **`(Scene, RadarConfig)` pair**, unlike maxwell/channel which carry a single scene.
  `RadarConfig` maps to a singleton **Radar Settings** object.
- **Mesh-first scene input.** Like channel/maxwell, the live solver walks ordinary
  visible Studio `Mesh` objects and exports them directly as radar structures. Users do
  not need radar-specific sidecar components for static targets.
- **Two independent hierarchies.** The studio `Transform` parent tree is for static
  pose; the radar **motion graph** (`scene._structure_motions`, a separate acyclic
  rigid-motion graph) is for time-parameterized dynamics. They are never conflated.
- **Non-SI sensor units on purpose.** `slope` in MHz/us, times in us, `sample_rate` in
  ksps, `power` in dBm, antenna locations in half-wavelength units. The component unit
  pickers keep the stored value in the platform's native unit, so the round trip is
  exact — values are never silently converted to SI.

## Layout

```
adapter/        RadarAdapter (detect/to_studio/to_platform) + config/structure/motion/solve maps
components/      Unified Radar, optional RadarMotion, post-processors
examples/        radar (Scene, RadarConfig) factories for the load demo
library_items.py drag-to-create prefabs (Radar demo, settings, plain mesh target)
tests/           per-phase round-trip + live-solve suites
```

Human/SMPL bodies are owned by the **wt-human** plugin, not radar. A radar scene that
contains one round-trips through the shared base geometry map (`kind="smpl"`) with no
radar-side SMPL code; radar's only human-aware feature is the optional `RadarTimeline`
"motion" source, which renders frames from a human placed by wt-human.

## Phases

| Phase | Scope |
|-------|-------|
| R0 | Scaffolding + structures + `(Scene, RadarConfig)` contract |
| R1 | Full sensor config: antenna pattern / noise / polarization / receiver chain + sensor pose/backend + validation |
| R2 | Dynamics (`RadarMotion` motion graph + parenting/acyclicity); human geometry is wt-human's |
| R3 | Tracer + 3 solver backends + single-frame signal viz (range-doppler / point cloud / MUSIC / CFAR) + library items |
| R4 | Multi-frame timeline (follow-up) |
| R5 | Polish: `simulate_group`, pluggable post-processors, viewport point cloud (follow-up) |

## Running the tests

The round-trip + solve suites need the radar stack importable (`witwin.radar` /
`witwin.core`, which pull in drjit + mitsuba; the package eagerly initializes the mitsuba
CUDA variant) plus `witwin_server` on the path. With those present:

```bash
# from this directory, in an env that has witwin.radar + witwin.core
PYTHONPATH=/path/to/witwin-studio/server python -m pytest tests/ -q
```

If the radar stack can't be imported, the suite skips itself rather than erroring.
The dirichlet/slang backends require a CUDA device; the pytorch backend runs on CPU.
