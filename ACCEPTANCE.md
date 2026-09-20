# Radar 0.4 acceptance

Last run: 2026-09-19

- Branch: `codex/radar-result-ui-bridge`
- Scene: `cat_radar_demo_20260831_01`
- Runtime: official `witwin-radar 0.4.0`, `witwin-channel 0.5.0`, `witwin 0.4.0`
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU
- Frames: 40/40 finite and nonzero, Studio time `0.0..3.9 s`
- Cube shape: `[40, 3, 4, 128, 256]`
- Dynamic target sites: 44 active
- Dynamic target propagation: LOS-only, target multipath disabled, `max_depth=0`
- Static room: 37 meshes/material slots, 1890 triangles
- Environment reflection paths: 60
- Environment model: `radar04_native_single_bounce`, `max_depth=1`
- Solve + compressed export: 75.64 s (previous target-multipath run: 164.73 s; 54.1% lower)
- Target path rows: 528/frame (previous target-multipath mean: 14,080.8/frame)
- Export: `C:\WiTwin\projects\witwin-scratch\results\radar-animation\official-radar-0.4-los-default-20260919.npz`
- SHA-256: `A90D98B10B535415E2E7DCDA4B428C25DE620E3BD6B2357BC611685A6D33853E`
- Saved-result reload: passed with matching shape, times, versions, and environment metadata
- Repository tests: 276 passed

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
if any frame is nonfinite/zero, no target site remains active, no environment
reflection/material is present, or the reflection depth is not exactly one.
