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
components/      RadarConfig, RadarStructureMeta, RadarMotion, RadarSensor, sub-configs, RadarResult
examples/        radar (Scene, RadarConfig) factories for the load demo
library_items.py drag-to-create prefabs (Radar starter, sensor, moving target, SMPL)
tests/           per-phase round-trip + live-solve suites
```

## Phases

| Phase | Scope |
|-------|-------|
| R0 | Scaffolding + structures + `(Scene, RadarConfig)` contract + `RadarStructureMeta` |
| R1 | Full sensor config: antenna pattern / noise / polarization / receiver chain + sensor pose/backend + validation |
| R2 | SMPL bodies + dynamics (`RadarMotion` motion graph + parenting/acyclicity) |
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
