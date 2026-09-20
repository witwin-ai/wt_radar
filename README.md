# WT Radar

Studio integration for the official `witwin-radar 0.4.x` and
`witwin-channel 0.5.x` packages.

WT Radar owns the Studio boundary: it snapshots the open `.wtscene`, samples
baked skin motion, selects explicit target sites, requests native Channel paths,
combines the supported direct environment return, calls official Radar waveform
synthesis, and publishes results for Studio replay and export. It does not modify
or replace Radar, Channel, or Maxwell physics code.

The default animation policy matches the original WT Radar behavior: moving
skin sites use line-of-sight target returns, while static room geometry still
contributes a native one-bounce background. **Target Multipath** is an explicit
high-cost option for wall-to-target and target-to-wall paths; it is off by
default.

Intra-frame target motion evaluates every transmitted chirp by default. The
official Radar 0.4 error-controlled **Adaptive** policy remains available for
comparison, but is not assumed to outperform chirp sampling. Path topology is
still rediscovered at Studio-frame boundaries.

## Supported physical model

The multi-frame animation path combines two coherent contributions:

1. Dynamic target returns from stable skin sites that have native inbound and
   outbound Channel paths for every requested frame.
2. Static room background returns for native one-bounce paths of the form
   `Radar -> wall/floor/furniture -> Radar`.

The environment contribution is adapted in WT Radar because Radar 0.4 does not
publish a direct-environment composer. Path discovery remains owned by the
official Channel package, and FMCW synthesis remains owned by the official Radar
package.

The current model is deliberately limited to one specular environment bounce.
It does not claim diffuse clutter, calibrated animal RCS, target self-occlusion,
or additional room bounces. Missing target paths fail preflight instead of being
zero-filled, and excluded sites do not transfer their RCS to visible sites.

## Studio workflow

1. Open a furnished room containing one skinned target and a baked Motion
   Matching timeline.
2. Add **Library -> Radar -> Radar Settings** to the same scene. The object's
   Transform is the sensor pose; local `-Z` is its forward direction.
3. Set **Snapshot Target ID** to the existing skinned object ID.
4. Set **Measurement start time**, **Animation duration**, and **Animation FPS**.
   The interval must fit the baked timeline, and `duration * FPS` must be an
   integer. The limits are 30 seconds, 30 FPS, and 1536 MiB of raw result data.
5. Click **Simulate & Prepare Replay**. This creates independent native GPU
   measurements for the requested frames and prepares the numeric Studio replay.
6. Inspect Range Profile, Range Spectrum, or Range-Doppler. Changing the result
   frame does not seek or mutate the editor timeline.
7. Click **Export Animation Result** to copy a validated `.npz` into
   `results/radar-animation/` under the project.

**Run Static Snapshot (1 frame)** is a separate single-point diagnostic. The
Library **Radar (Demo)** prefab automatically selects its box target for this
workflow. Historical Start Stream, Generate Frames, `RadarAdapter`, and Radar 0.3
`Scene/RadarConfig/trace` entry points have been removed.

## Result contract

Animation exports contain the complex Radar cube, physical axes, timestamps,
sampled target positions and velocities, active site IDs, topology evidence,
environment-reflection statistics, package versions, and an input fingerprint.
Exports are user-owned and never overwrite an existing file.

The environment evidence includes:

- `model`
- `max_depth` (currently exactly `1`)
- `reflected_path_count`
- `material_slot_count`
- `coherent_with_target`

Studio can load a metadata-bearing exported result without rerunning the solver.
Replay uses the official Radar 0.4 processing API and the existing numeric
LineChart/Heatmap widgets.

## Runtime contract

The plugin manifest and startup checks require:

- `witwin-radar >=0.4,<0.5`
- `witwin-channel >=0.5,<0.6`

An absent or incompatible package fails explicitly before GPU synthesis. The
supported solver request adapters are the explicit-point snapshot and baked
Studio animation models; an unlabelled request does not fall back to Radar 0.3.

## Layout

```text
adapter/
  snapshot.py             one-point frozen diagnostic
  animation.py            baked Studio skin animation orchestration
  studio_geometry.py      Studio mesh/material export
  studio_motion.py        baked skin sampling
  environment_clutter.py  native one-bounce environment request
  direct_environment.py   Channel-path to Radar-input bookkeeping
  radar04.py               official Radar 0.4 construction
  replay.py                Studio numeric replay
components/
  radar.py                 current Studio component and controls
solver_host.py             snapshot/animation dispatch and result queries
agent_tools.py             planning, preflight, execution, evidence
tests/                     current Radar 0.4 contract and regression suite
```

## Tests

Run from this repository using the Studio/Radar environment:

```powershell
C:\Users\WQZ\AppData\Local\anaconda3\envs\witwin2\python.exe -m pytest -q
```

GPU physics checks require CUDA. A skipped CUDA test is not equivalent to a
successful physical acceptance run.

The final acceptance is a real furnished Studio scene with baked target motion.
Every requested frame must be finite and nonzero; the topology must retain active
target sites; environment `reflected_path_count` and `material_slot_count` must be
positive; and the exported NPZ must reopen with the recorded metadata intact.
See [ACCEPTANCE.md](ACCEPTANCE.md) for the latest reproducible GPU run.
