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
the existing Studio `SkinnedMesh` plus baked `Transform` tracks. New projects
declare one or more targets through `animation_targets_json`; the legacy
`snapshot_target_id` remains a supported single-target fallback.

The canonical Agent/tool declaration is:

```json
{
  "targets": [
    {"object_id": "human"},
    {"object_id": "dog"}
  ]
}
```

Target IDs must be unique and each object must own a hydrated `SkinnedMesh`.
When `site_budget` is omitted, typed `SMPLBody` targets resolve from the Radar
component's editable `human_target_site_budget` (720 by default) and registered
`AnimalActor` targets resolve from `animal_target_site_budget` (360 by default).
Untyped skinned meshes use `target_site_budget` (24 by default). An explicit
per-target `site_budget` in the declaration always wins. Every budget remains
editable in the 1..2400 range and is consumed by the backend simulation.

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

`StudioSkinSampler` validates each actor's complete contract. The
`MultiTargetSkinSampler` then presents their stable sites as one combined
native target set. The adapter:

1. chooses one representative bind-pose surface vertex for each influenced
   bone;
2. uses deterministic surface coverage for budgets above the anatomical bone
   anchors, up to 2400 sites per actor;
3. selects sites deterministically by anatomical priority and bind-pose spatial
   coverage;
4. evaluates Studio's linear-blend skinning in world coordinates at each
   requested time;
5. derives velocity from the same rendered world positions using a finite
   difference of at most 1 ms, with a right-hand derivative at Timeline knots;
   and
6. preserves stable site identity across the entire requested interval.

WT Radar concatenates the resulting finite world-space positions, velocities,
scalar RCS policy, and stable IDs and passes them to one official Radar 0.4
moving-target call. Human and dog returns therefore coexist coherently in the
same physical solve; they are not simulated separately or combined as images.
WT Radar does not replace propagation, scattering, waveform, or DSP algorithms.

Automatic placement samples the joint trajectories of every declared target,
aims at their angular envelope, excludes all target meshes from static sensor
collision checks, and accepts a pose only after the combined native topology
preflight passes.

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

- WT Radar plugin on the `codex/r46-development` line;
- `witwin-radar >=0.4,<0.5` (validated with 0.4.1);
- `witwin-channel >=0.5,<0.6`; and
- the Studio `SkinnedMesh` and baked Timeline APIs used by Core 0.4.

Any incompatible change to the skin payload, transform hierarchy, interpolation
semantics, site identity, or Radar moving-target input requires a new contract
revision and conformance tests. A version-number-only check is not sufficient.

## Conformance evidence

Automated conformance tests cover legacy single-target loading, human-and-dog
list normalization, editable typed defaults (720 for `SMPLBody`, 360 for
`AnimalActor`), explicit per-target overrides, combined positions and
velocities, joint automatic placement, and combined native-preflight requests.
A full native GPU measurement is still required to claim end-to-end physical
acceptance for a specific Scene and sensor placement.

