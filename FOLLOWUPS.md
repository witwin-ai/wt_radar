# wt-radar follow-ups

Deferred work, in rough priority order.

## Base-layer change request (not made here — base is frozen)

- **SMPL bake degradation.** `witwin_server.platform_bridge.geometry_map._bake_local`
  catches only `ImportError`, so an SMPL body whose model `.pkl` files are missing raises
  `FileNotFoundError` instead of degrading to params-only (the `StudioGeometry` docstring
  already promises this degradation). Broadening the catch (e.g.
  `except (ImportError, FileNotFoundError, OSError)`) would make SMPL load/round-trip
  degrade gracefully when model files are absent. This is a base/wt-human concern (radar
  owns no SMPL code), but it affects radar too: round-tripping a wt-human body that lands
  in a radar scene, and the `RadarTimeline` "motion" source, both bake SMPL through the
  base and so need model files present.

## R5 polish not yet implemented

- **Antenna-pattern import helpers** — load measured/CSV patterns into
  `RadarAntennaPattern` (separable cuts or 2D maps) instead of hand-entered tables.
- **Differentiable pytorch-backend hooks** — expose gradient flow through
  `radar.simulate` (the pytorch backend is autograd-friendly) for inverse problems.
- **Viewport point-cloud rendering** — render `process_pc` output as 3D points in the
  viewport (currently an in-component scatter, master §7.2).
- **Multi-radar UI** — `SolveRunner.run_group` + a test cover `Radar.simulate_group`, but
  there is no editor UI for composing multiple sensors yet.
- **Timeline motion source assets** — the SMPL motion timeline source
  (`RadarTimeline.source = "motion"`) needs SMPL model files + a `human` structure; only
  the point-cloud-sequence source is exercised live.

## Platform observation (witwin.radar, external)

- `Scene.update_structure(name, **changes)` cannot rename a structure: the positional
  lookup `name` collides with a `name=` change, so the rename/re-parent remap code in
  `update_structure` is unreachable via the public API. The adapter keys motions by the
  live object name + `RadarMotion.parent` string, so editor renames round-trip correctly
  regardless (see `tests/test_r2_motion.py::test_rename_keeps_motion_refs_consistent`).
