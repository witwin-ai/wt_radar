# wt-radar

WiTwin Studio plugin for the WiTwin Radar 0.3 scene-driven API. The plugin maps
ordinary visible Studio meshes to immutable `witwin.core.Scene` structures and runs
FMCW simulation in the dedicated `witwin_runtime` Conda environment.

## Runtime baseline

- `witwin-radar == 0.3.0`
- `witwin == 0.4.0` (Core)
- `witwin-channel == 0.5.0`
- `rayd == 0.8.0` (`rayd-torch` and `rayd-drjit`)
- PyTorch 2.10 with CUDA

Local plugin checkouts do not install Python requirements. Prepare the solver
environment separately, then add this plugin path explicitly in Studio.

## Data model

- The platform payload is `(witwin.core.Scene | DynamicScene, RadarConfig)`.
- Each exported structure has explicit Core structure/material/assignment/surface IDs.
- Point sites use `RadarScatterer.site_offset` in the target object's local frame.
  Surface mode deterministically samples stable barycentric sites from visible mesh
  geometry and evaluates skinned deformation per frame for velocity-corrected bakes.
- `target_rcs_m2` is the scalar RCS applied to each explicit site. In surface
  mode, `surface_site_count` is therefore part of the physical discretization, not
  only a quality knob; changing it changes the aggregate response. Multipath depth
  and shared endpoint polarization are also authored in the Radar component.
- Motion uses `DynamicScene` and `LinearTrajectory`.
- Signal processing uses the typed `witwin.radar.processing` records for range profile,
  range-Doppler, CFAR, point cloud, and MUSIC.
- Multi-radar execution is implemented as independent Radar 0.3 solves because the
  upstream group entry point was removed.

## Tests

From this plugin directory:

```powershell
$env:PYTHONPATH="E:\Code\witwin-plugins;E:\Code\witwin-studio\server"
conda run -n witwin_runtime python -m pytest tests -q
```

GPU tests require CUDA and the complete Radar/Core/Channel/RayD stack. Hardware tests
skip explicitly when CUDA is unavailable; adapter and component tests still run.

## Live throughput benchmark

```powershell
conda run -n witwin_runtime python tools\live_stream_throughput.py `
  --profile smoke --frames 20 --warmup-frames 2 --max-fps 30 `
  --channels raw,rd,pc --target-rcs-m2 1.0 --encode-bytes
```

Use `--multipath --max-reflections 2` to benchmark reflected paths.
