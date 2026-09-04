# Saved GPU Result review

This milestone connects existing Radar component UI to saved results. It does not
change Channel, Radar synthesis, FFT, filtering, or any signal-processing algorithm.

## Supported

- Load an absolute path to a `radar_cube.npz` exported with schema version 1.
- Select a recorded frame, TX, RX, and chirp. Times come from the file, not UI FPS.
- `range_profile`: one selected complex profile's linear amplitude versus metres.
- `range_spectrum`: the same profile's real/imaginary parts versus metres.
- Existing Figure **Maximize**, **Download**, and **Save to Results** controls.

Processing is exactly `range_profile(ProcessingCube(frame, axes),
window="rectangular", remove_dc=False)`. No antenna/chirp aggregation, dB
normalization, filtering, or extra range FFT is applied. An FMCW spectrum is already
transformed. It is not an ADC-time signal and is not calibrated received power.

## Recorded metadata

The producer must save `cube`, `times_s`, `result_schema_version=1`,
`processing_axes_json` (all fields of its actual `ProcessingAxes`), and
`producer_metadata_json`. The latter describes the source scene, physical model,
and limitations. Files without full producer axes are rejected, rather than
interpreted using current component settings. No pickle or producer script is executed.
The initial reader accepts FMCW spectrum results up to 512 MiB expanded.

## Studio workflow

1. Load this local plugin through Studio's Plugins panel or an explicit
   `--plugin-path C:\WiTwin\plugins\wt_radar` when starting the ProjectServer.
2. Add the existing **Radar Settings** library item, or open the generated
   `C:\WiTwin\projects\radar-ui-review` project.
3. Select its Radar object. Under **Saved GPU Result**, set **Saved Result Path**
   and click **Load Saved Result**.
4. Under **Solve**, maximize **Radar Signal**. Change **Saved Frame Index** and
   **Saved Chirp Index**; double-click numeric fields to type a value. TX/RX and
   the two supported views are under **Post Processing**.

The isolated review project deliberately contains no bedroom geometry: its chart
comes from the real bedroom simulation. Its object's Transform is not the sensor's
recorded pose; no radar gizmo is shown for saved-result objects.

### This Windows machine

Start the isolated review backend (the existing Studio frontend must be running):

```powershell
& C:\WiTwin\plugins\wt_radar\tools\start_saved_result_review.ps1
```

Then open `http://localhost:3000/?server=127.0.0.1&port=8011` and follow the steps
above. The prepared project already points to the latest result; loading a result
starts at frame 0. Stop this review backend with Ctrl+C when finished.

Verified result directory:
`C:\WiTwin\projects\witwin-scratch\results\radar_ui_review\20260827T031216Z`.
It contains `radar_cube.npz`, `manifest.json`,
`range_profiles_tx0_rx0_chirp0.npz`, the corresponding CSV, and two actual Studio
UI screenshots (`radar-ui-range-profile.png`, `radar-ui-range-spectrum.png`).
The selected profiles have shape `[31, 256]`: 31 recorded instants from 0 to 3 s,
spaced by 0.1 s. This is not 30 Hz simulation or a continuous video.

The UI was tested by loading the result and selecting frame 30 (t=3 s), and
switching between both supported views. The exported complex profiles equal the
source cube selection exactly. The current targeted test suite passes 24 tests.

## Export

From a Python environment with Studio and Radar 0.3:

```powershell
python tools/export_saved_profiles.py --input C:\path\radar_cube.npz --output C:\path\range_profiles.npz --tx 0 --rx 0 --chirp 0
```

This writes complex profiles, display magnitudes, physical range axes, and recorded
times to NPZ and CSV. It refuses to overwrite an existing export.

## Explicitly not complete

- The legacy Simulate, Stream, and Generate Frames APIs are not migrated.
- Saved-result RD, point clouds, CFAR, and static-clutter removal are not enabled.
- Frame selection does not drive the scene timeline or produce a video yet.
- The bedroom Cat is still a scalar point-RCS target, not skeletal scattering.
- A spectrum's strongest peak may be multipath, not the direct Cat distance.
- UI configuration edits do not reinterpret saved result axes or rerun simulation.

These limitations are not worked around with replacement algorithms or dependency downgrades.
