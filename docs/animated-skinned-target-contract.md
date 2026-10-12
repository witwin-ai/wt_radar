# Animated skinned target contract (Studio -> WT Radar 0.4.1)

Status: implemented and validated

This document defines the integration boundary for an animated human or animal
authored in Studio and measured by WT Radar 0.4.1. It is an adapter contract,
not a Radar physics or signal-processing algorithm.

## Ownership

- The animation producer (Genesis, Motion Matching, or another Studio plugin)
  owns the hydrated `SkinnedMesh`, bone hierarchy, and baked Timeline tracks.
- Studio owns component serialization, linear-blend skinning transforms, and
  Timeline evaluation.
- WT Radar owns deterministic surface-site selection, Timeline sampling,
  world-space site velocity estimation, and conversion to Radar 0.4
  `PointTargets`.
- `witwin-radar >=0.4,<0.5` owns target propagation, waveform synthesis, and
  Radar processing. `witwin-channel >=0.5,<0.6` owns path discovery.

No target-side `RadarScattererComponent` is required. The stable boundary is
the existing Studio `SkinnedMesh` plus baked `Transform` tracks. The Radar
object selects that target through `snapshot_target_id`.

## Producer requirements

The selected target object must provide a hydrated `SkinnedMesh` containing:

- finite bind-pose vertices and triangle faces;
- one or more `bone_ids` resolving to Scene objects with `Transform`;
- four bone indices and four finite, non-negative weights per vertex;
- per-vertex weights summing to one within the Studio tolerance;
- one finite 4x4 inverse-bind matrix per bone; and
- consistent `parent_id` and `Transform.parent` hierarchy representations.

Motion must be baked into the Scene Timeline:

- at least one target or bone `Transform` track is required;
- supported fields are `position`, `rotation`, and `scale`;
- values are finite three-vectors with strictly increasing key times;
- interpolation is `LINEAR`; and
- the requested measurement interval is contained in the clip duration.

An animation plugin may retain source motion metadata, but Radar consumes only
the baked Studio transforms. A controller track must not compete with those
transforms during measurement.

## Consumer behavior

`StudioSkinSampler` validates the complete contract before simulation. It then:

1. chooses one representative bind-pose surface vertex for each influenced
   bone;
2. caps the available set at 64 and applies the configured 1..64 site budget;
3. selects sites deterministically by anatomical priority and bind-pose spatial
   coverage;
4. evaluates Studio's linear-blend skinning in world coordinates at each
   requested time;
5. derives velocity from the same rendered world positions using a finite
   difference of at most 1 ms, with a right-hand derivative at Timeline knots;
   and
6. preserves stable site identity across the entire requested interval.

WT Radar passes the resulting finite world-space positions, velocities, scalar
RCS policy, and stable IDs to the official Radar 0.4 moving-target API. It does
not replace propagation, scattering, waveform, or DSP algorithms.

## Failure semantics

The adapter fails closed before native simulation when the target is missing,
the skin is not hydrated, influences or inverse binds are invalid, a bone or
parent cannot be resolved, tracks are unsupported or non-linear, Timeline
application is stale, sampled geometry is non-finite, or the interval is out of
range. It does not use rigid, identity, zero-return, or dropped-site fallbacks.

Sites without native inbound and outbound paths for the complete interval are
reported by topology preflight. Active sites retain stable IDs; excluded sites
do not transfer their RCS share to visible sites.

## Version and compatibility policy

This contract targets:

- WT Radar plugin on the `codex/r41-development` line;
- `witwin-radar >=0.4,<0.5` (validated with 0.4.1);
- `witwin-channel >=0.5,<0.6`; and
- the Studio `SkinnedMesh` and baked Timeline APIs used by Core 0.4.

Any incompatible change to the skin payload, transform hierarchy, interpolation
semantics, site identity, or Radar moving-target input requires a new contract
revision and conformance tests. A version-number-only check is not sufficient.

## Conformance evidence

The persisted Genesis SMPL scene
`human_genesis_walk_persistent_20261010` was cold-loaded and sampled through
`StudioSkinSampler` with all 24 available SMPL bone sites selected. Samples at
0.0, 0.5, 1.0, 1.5, and 1.95 seconds produced finite world positions and
velocities. This proves the Studio-to-WT-Radar bridge; a native GPU measurement
is still required to claim end-to-end physical acceptance for that specific
human Scene and sensor placement.

