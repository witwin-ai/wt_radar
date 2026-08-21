# wt-radar follow-ups

- Add measured antenna-pattern CSV import into the current separable/map pattern specs.
- Add viewport rendering for typed Radar point-cloud results.
- Add an editor surface for composing multiple independent Radar sensors.
- Replace the private `_synthesize` call used to derive `ProcessingAxes` when Radar
  exposes the maintained public equivalent.
- Add richer per-object scattering response authoring when Radar exposes stable
  structure-to-site response assignment in the public scene-driven API.
