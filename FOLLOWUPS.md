# wt-radar follow-ups

Deferred work, in rough priority order.

## Base-layer change request (not made here — base is frozen)

- **SMPL bake degradation.** `witwin_server.platform_bridge.geometry_map._bake_local`
  catches only `ImportError`, so an SMPL body whose model `.pkl` files are missing raises
  `FileNotFoundError` instead of degrading to params-only (the `StudioGeometry` docstring
  already promises this degradation). Broadening the catch (e.g.
  `except (ImportError, FileNotFoundError, OSError)`) would make SMPL load/round-trip
  degrade gracefully when model files are absent. Until then, the full SMPL
  platform→studio→platform round trip (which bakes the display mesh) is skipped in
  `tests/test_r2_motion_smpl.py::test_smpl_full_round_trip`; the params reconstruction
  (`test_smpl_params_reconstruct`) and motion graph are fully tested.

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
  regardless (see `tests/test_r2_motion_smpl.py::test_rename_keeps_motion_refs_consistent`).
