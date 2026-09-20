# Radar 0.4 acceptance

Last run: 2026-09-19

- Branch: `codex/radar-result-ui-bridge`
- Scene: `cat_radar_demo_20260831_01`
- Runtime: official `witwin-radar 0.4.0`, `witwin-channel 0.5.0`, `witwin 0.4.0`
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU
- Frames: 40/40 finite and nonzero, Studio time `0.0..3.9 s`
- Cube shape: `[40, 3, 4, 128, 256]`
- Dynamic target sites: 44 active
- Static room: 37 meshes/material slots, 1890 triangles
- Environment reflection paths: 60
- Environment model: `radar04_native_single_bounce`, `max_depth=1`
- Export: `C:\WiTwin\projects\witwin-scratch\results\radar-animation\official-radar-0.4-cleanup-20260919.npz`
- SHA-256: `747927FB2F7354294B86C69ACD1F049AED87D24498AB5DC16D657FBA9E1EEBBD`
- Saved-result reload: passed with matching shape, times, versions, and environment metadata
- Repository tests: 275 passed

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
