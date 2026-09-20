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
- Solve + compressed export: 71.875 s (44-site reference: 75.64 s; 4.98% lower)
- Adaptive 44-site comparison: 123.547 s, 63.3% slower than 44-site chirp on this scene
- Adaptive/chirp difference: complex relative L2 0.0903%, magnitude relative L2 0.0631%, complex correlation 0.9999996, power ratio 0.9999995
- Target path rows: 288/frame (44-site reference: 528/frame; previous target-multipath mean: 14,080.8/frame)
- Export: `C:\WiTwin\projects\witwin-scratch\results\radar-animation\official-radar-0.4-semantic-24-default-20260919.npz`
- SHA-256: `FB631AD2C25B2463BF82936E52C491A4FAB3E684193C1F2DAA81E49B226BD1BA`
- Saved-result reload: passed with matching shape, times, versions, and environment metadata
- Semantic-24 versus 44-site whole-cube complex relative L2: 2.864%
- Semantic-24 dynamic-only difference after slow-time mean removal: complex relative L2 77.20%, Range-Doppler magnitude relative L2 50.49%, power ratio 0.8498
- Semantic-24 dynamic Range-Doppler peak matched the 44-site reference in 4/40 frames; this records the intentional target-model change
- Repository tests: 278 passed

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
