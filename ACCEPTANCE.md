# Radar 0.4 acceptance

Last run: 2026-09-19

- Branch: `codex/radar-result-ui-bridge`
- Scene: `cat_radar_demo_20260831_01`
- Runtime: official `witwin-radar 0.4.0`, `witwin-channel 0.5.0`, `witwin 0.4.0`
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU
- Frames: 40/40 finite and nonzero, Studio time `0.0..3.9 s`
- Cube shape: `[40, 3, 4, 128, 256]`
- Dynamic target sites: 24 anatomically prioritized active sites from 44 available bone sites
- Dynamic target propagation: LOS-only, target multipath disabled, `max_depth=0`
- Dynamic target motion sampling: `chirp` (measured default)
- Static room: 37 meshes/material slots, 1890 triangles
- Environment reflection paths: 60
- Environment model: `radar04_native_single_bounce`, `max_depth=1`
- Solve + fast stored export: 56.156 s (compressed sparse-skin run: 59.735 s; 5.99% lower)
- Adaptive 44-site comparison: 123.547 s, 63.3% slower than 44-site chirp on this scene
- Adaptive/chirp difference: complex relative L2 0.0903%, magnitude relative L2 0.0631%, complex correlation 0.9999996, power ratio 0.9999995
- Target path rows: 288/frame (44-site reference: 528/frame; previous target-multipath mean: 14,080.8/frame)
- Export: `C:\WiTwin\projects\witwin-scratch\results\radar-animation\official-radar-0.4-semantic-24-fast-export-20260919.npz`
- SHA-256: `C3ED8C660E138F4CE521FB3CDFDD5C80F44C86FA2953B6986D8407C955F798E8`
- Saved-result reload: passed with matching shape, times, versions, and environment metadata
- Target motion sampling alone: 13.243 s full-mesh reference versus 1.708 s sparse/memoized, with bit-exact positions and velocities
- Sparse-skin export SHA-256 is identical to the pre-optimization semantic-24 export
- Fast export: 0.134 s and 120.16 MiB versus compressed 3.324 s and 111.63 MiB; all seven saved arrays/metadata fields are bit-exact after reload
- Semantic-24 versus 44-site whole-cube complex relative L2: 2.864%
- Semantic-24 dynamic-only difference after slow-time mean removal: complex relative L2 77.20%, Range-Doppler magnitude relative L2 50.49%, power ratio 0.8498
- Semantic-24 dynamic Range-Doppler peak matched the 44-site reference in 4/40 frames; this records the intentional target-model change
- Repository tests: 278 passed

## Studio Agent natural-language acceptance

The Studio live Agent harness was run against the current checkout through the
real configured provider, Workbench controller, Home Simulation confirmation
action, ProjectServer, and the plugin paths below. Both runs used fresh isolated
test Projects; the user's `witwin-scratch` Project was not mutated.

### Fresh positive measurement

- Prompt: `Record the existing cat motion from 0 to 5 seconds at 10 FPS. Do not change the animation.`
- Evidence: `C:\WiTwin\test-runs\wt-radar-agent-fresh-run-20260919\report.json`
- Plugin under test: `C:\WiTwin\plugins\wt_radar`
- Agent plan: preserve the existing Timeline, measure `0..5 s` at `10 FPS`, and do not export NPZ
- Confirmation path: compiled Home Simulation Run action using the exact pending plan
- Result: pipeline `succeeded`, completion `verified`, native GPU result independently verified
- Frames/cube: 50/50 nonzero frames, `[50, 3, 4, 128, 256]`
- Dynamic target: 24/24 active sites, LOS-only, chirp motion sampling, target multipath disabled
- Static room: 37 material slots and 60 coherent native single-bounce reflection paths
- Physics runtime: `witwin 0.4.0`, `witwin-radar 0.4.0`, `witwin-channel 0.5.0`, CUDA
- Motion preservation: passed; the complete authored Timeline remained unchanged
- Replay: persistent 50-frame payload created and integrity-verified
- Agent turn plus confirmed execution: 116.073 s
- Confirmed pipeline: 97.892 s; GPU submission step, including its synchronous simulation, completed in 79.995 s after preflight

The model also requested Replay although the sentence only asked for a recording.
The native measurement is valid, but this is recorded as an Agent intent-routing
overreach outside the Radar physics implementation.

### Stale-result safety check

- Prompt: `Prepare replay and NPZ from the last successful recording.`
- Evidence: `C:\WiTwin\test-runs\wt-radar-agent-result-reuse-20260919\report.json`
- Calls: `witwin.radar.inspect_pipeline`, then `witwin.radar.prepare_replay`
- Result: the Agent found the saved payload but correctly refused reuse because
  the durable native result fingerprint was stale after Scene inputs changed;
  it did not claim success or export an invalid NPZ.

Reproduction command from this repository:

```powershell
$env:PYTHONPATH='C:\WiTwin\plugins'
C:\Users\WQZ\AppData\Local\anaconda3\envs\witwin2\python.exe tools\accept_scene.py `
  C:\WiTwin\projects\witwin-scratch `
  cat_radar_demo_20260831_01 `
  --output C:\WiTwin\projects\witwin-scratch\results\radar-animation\<new-name>.npz
```

The command uses Studio's normal project-mesh and SkinnedMesh hydration paths,
then invokes the same `animation_request`, `solve_animation`, and
`export_animation` functions used by the component workflow. It returns nonzero
if any frame is nonfinite/zero, the semantic target does not contain exactly 24
active sites, no environment reflection/material is present, target multipath
is enabled, target motion is not chirp-sampled, or the reflection depth is not
exactly one.
