# wt-radar

## Studio baked-skin animation (native Radar 0.3)

Agent orchestration supports explicit `fixed` sensor placement and opt-in
`automatic` placement. Automatic mode searches at most 16 poses near the authored
motion, at world Y = `height_m` (default 1 m), and calls the unchanged native
animation preflight over the requested interval for each candidate. Trials use
a detached scene; only an accepted pose is applied. Do not supply coordinates
in automatic mode: conflicting explicit inputs are rejected, never overridden.
This is a bounded placement search, not an RF algorithm change or a guarantee
that every room has a usable pose. Timing, CUDA and configuration errors are
reported immediately rather than treated as placement failures.

The **Studio Animation → Simulate Animation** button submits a detached copy of
the current open scene, including unsaved geometry and the baked timeline.
It uses the existing Studio skinning implementation and passes world-space
surface positions and velocities through native `Kinematics` / `two_way_duals`.
No Radar/Channel propagation or DSP equations are replaced.

1. Open a furnished scene with an existing skinned character and baked Motion
   Matching timeline. Pause playback/recording. Do not create a second character.
2. Add or select **Library → Radar → Radar Settings** in that scene. Its Transform
   is the simulated sensor pose; local -Z is forward. Place it at the intended
   height and orient it toward the target in your own room. TX/RX antenna offsets
   are relative to this pose.
3. Set **Snapshot Target Id** to the existing skin ID (e.g. `catstray`). For
   Animation, Snapshot Local Point is not used: one fixed surface vertex per
   influenced bone is selected from bind geometry. Native Channel discovery is
   run at every requested frame. Only stable site IDs with both inbound and
   outbound paths for the whole interval are active. The assumed RCS is divided
   by all originally declared sites; excluded sites' shares are not moved onto
   visible sites. This is not a calibrated animal RCS or full skin EM.
4. Set **T0**, **Animation Duration S**, and **Animation Fps** precisely. The
   interval must fit the baked timeline; duration × FPS must be an integer.
   Maximum duration/FPS is 30 seconds/30 Hz and raw result memory is capped at
   1536 MiB, with an additional available-memory preflight. Unsupported settings
   fail; they are never silently reduced.
5. Leave the legacy extra-reflection control at zero and keep multipath/noise/CFAR/clutter
   removal off. The native adapter always includes exactly one specular reflection from
   the exported static room geometry. Click **Simulate Animation**. This is distinct from **Simulate**,
   which remains the single-point frozen diagnostic. Legacy Stream/Generate
   Frames are not the Studio animation path.
6. Choose **range_profile**, **range_spectrum** or **range_doppler** and change
   **Animation Frame Index**. Plots carry physical axes and recorded timestamps;
   this control does not seek or rewrite the editor timeline.
7. Click **Export Animation Result**. The validated complex result is copied to
   a unique `.npz` under the project's `results/radar-animation/`, not disposable
   solver cache. It includes native cube/axes, times, sites, velocities and model
   metadata. Existing files are not overwritten. Saved-result replay supports
   Range Profile/Spectrum/Doppler through the native processing API.
8. Click **Prepare Synchronized Replay** after a completed animation or after
   **Load Saved Result**. The existing numeric LineChart/Heatmap widgets display
   the data directly, without matplotlib or per-frame image files. Use the
   widget's **Maximize** to open it beside the room, leave **Follow scene time**
   on, then seek/play the room's **Timeline** within the recorded interval.
   The widget can switch between Range Profile and Range Doppler locally.
   Outside the interval it explicitly shows no recorded measurement. Manual
   widget review does not move the cat. Global amplitude limits are fixed across
   the recording; values are not calibrated animal reflectivity.
9. In Studio's recorded signal widget, **Download Comparison…** opens a
   download-only exporter for the radar-view room reference, Range Profile and
   Range Doppler. Pause the timeline and keep that room's viewport open, then
   export the selected PNG, full MP4, or all PNGs in a ZIP. Comparison is not a
   live preview tab; generation visits the saved measurement times explicitly
   and does not run another simulation. The ordinary RP/RD previews remain live.

Prepared numeric replay assets (128 MiB maximum) persist in the project's hidden
runtime storage; when the scene is saved, its Figure can replay those bytes after
a backend restart without rerunning simulation. Native solver handles and the
component's live status labels are transient, however. Keep the exported NPZ as
the durable user-owned result; use **Load Saved Result** if replay must be prepared
again (for example after runtime cache cleanup). New recordings verify authored
motion tracks at preparation, not full geometry or later edits. A recording from
another scene is rejected; older recordings without a motion fingerprint are
explicitly unverified. Replay does not support CFAR or clutter-removal toggles.

### Strict failure boundaries

Only LINEAR baked target/bone Transform tracks and a static sensor/room are
supported. Other animated geometry, incomplete rigs, inconsistent hierarchy,
nonfinite data, oversized results, overlapping CPIs and Doppler aliasing fail.
Velocities are finite differences of the exact Studio world skin (at most 1 ms,
right-hand at knots), not a second motion interpolation. Native per-CPI synthesis
uses frozen weights with first-order carrier rate, not arbitrary within-CPI
acceleration. Static room occlusion and one native specular room reflection are
enabled; diffuse clutter, additional room bounces and target self-occlusion are
not included.

**Native topology boundary:** Radar 0.3 requires every site submitted for a
frame to have discovered inbound and outbound legs. Before waveform synthesis,
this interface therefore uses native Channel discovery over the exact requested
frame grid and takes the intersection of reachable stable site IDs. It never
invents paths, fills missing returns with zeros, or redistributes an excluded
site's RCS. The result metadata records declared, active and occluded sites plus
per-frame discovery evidence. If no site remains reachable for the whole
interval, preflight fails before GPU waveform synthesis. This is a conservative
measurement-interface policy, not a change to Channel propagation or Radar DSP.

Focused tests:
`python -m pytest -q tests/test_snapshot.py tests/test_saved_result.py tests/test_animation.py tests/test_animation_component.py`

## Radar 0.3 Studio snapshot milestone

**Simulate** now submits the current Studio scene through the existing solver
worker to native Radar 0.3 / Channel 0.5. Select an existing **Snapshot Target ID**,
explicit local point, assumed scalar RCS, and snapshot timeline time. Pause playback
and recording first. The solver samples a copy, never the editor scene.

This is deliberately a **single explicit point, frozen-pose diagnosis**, not a
moving-cat measurement. The target mesh is replaced by that point; no target
self-occlusion, skin scattering or gait Doppler are claimed. Static room meshes
participate both as occluders and as native one-bounce specular reflectors; diffuse
and multi-bounce environment clutter are not included. Other visible unmodelled
SkinnedMeshes are refused. Parent transforms and visibility are respected. RCS is an explicit uncalibrated
assumption. Legacy tracer/noise/polarization/receiver options are not mapped by this
milestone and unsupported settings fail instead of being silently ignored.

Range Profile and Range Spectrum use the official processing API and physical
range coordinates. Range Doppler also uses official DSP. All three views now use
numeric Figure data and the shared native LineChart/Heatmap widgets. Physical
axes, tiny-amplitude limits and scientific-notation labels are supported without
normalization or changes to RF/DSP values.
Stream and Generate Frames remain blocked;
they are not the Studio skeletal timeline adapter.

Run `python -m pytest -q tests/test_snapshot.py tests/test_saved_result.py` in the
Studio/Radar environment. The snapshot tests include actual CUDA repeatability and
a moved-sensor range-peak check; absence of CUDA is a skipped GPU check, not success.

## Current Radar 0.3 saved-result UI milestone

The `codex/radar-result-ui-bridge` branch adds **Saved GPU Result** to the existing
Radar component. It replays metadata-bearing NPZ results through the official
`witwin.radar.processing.range_profile` API and the existing Figure widget.
Use **Load Saved Result**, then **Prepare Synchronized Replay**, to inspect a
previous export in Studio. Saved replay is separate from the snapshot solver
above. Start Stream and Generate Frames still fail explicitly on Radar 0.3.

## Historical legacy adapter documentation (not Radar 0.3 support)

Round-trips `witwin.radar` solver scenes to and from the witwin-studio editor and runs
the FMCW radar engine, with in-component signal visualization. It is an independent
plugin (its own git repo) built on the frozen core base layer
(`witwin_server.platform_bridge`) — see
[RADAR_PLUGIN_PLAN.md](../../RADAR_PLUGIN_PLAN.md) and the master
[PLATFORM_SCENE_INTEGRATION_PLAN.md](../../PLATFORM_SCENE_INTEGRATION_PLAN.md).

This supersedes the old `wt_radar` (fake `SimpleRadar`); history is preserved in this
same repo.

## What's special about radar

- **The sensor lives outside the scene.** The load/export contract carries a
  **`(Scene, RadarConfig)` pair**, unlike maxwell/channel which carry a single scene.
  `RadarConfig` maps to a singleton **Radar Settings** object.
- **Mesh-first scene input.** Like channel/maxwell, the live solver walks ordinary
  visible Studio `Mesh` objects and exports them directly as radar structures. Users do
  not need radar-specific sidecar components for static targets.
- **Two independent hierarchies.** The studio `Transform` parent tree is for static
  pose; the radar **motion graph** (`scene._structure_motions`, a separate acyclic
  rigid-motion graph) is for time-parameterized dynamics. They are never conflated.
- **Non-SI sensor units on purpose.** `slope` in MHz/us, times in us, `sample_rate` in
  ksps, `power` in dBm, antenna locations in half-wavelength units. The component unit
  pickers keep the stored value in the platform's native unit, so the round trip is
  exact — values are never silently converted to SI.

## Layout

```
adapter/        RadarAdapter (detect/to_studio/to_platform) + config/structure/motion/solve maps
components/      Unified Radar, optional RadarMotion, post-processors
examples/        radar (Scene, RadarConfig) factories for the load demo
library_items.py drag-to-create prefabs (Radar demo, settings, plain mesh target)
tests/           per-phase round-trip + live-solve suites
```

Human/SMPL bodies are owned by the **wt-human** plugin, not radar. A radar scene that
contains one round-trips through the shared base geometry map (`kind="smpl"`) with no
radar-side SMPL code; radar's only human-aware feature is the optional `RadarTimeline`
"motion" source, which renders frames from a human placed by wt-human.

## Phases

| Phase | Scope |
|-------|-------|
| R0 | Scaffolding + structures + `(Scene, RadarConfig)` contract |
| R1 | Full sensor config: antenna pattern / noise / polarization / receiver chain + sensor pose/backend + validation |
| R2 | Dynamics (`RadarMotion` motion graph + parenting/acyclicity); human geometry is wt-human's |
| R3 | Tracer + 3 solver backends + single-frame signal viz (range-doppler / point cloud / MUSIC / CFAR) + library items |
| R4 | Multi-frame timeline (follow-up) |
| R5 | Polish: `simulate_group`, pluggable post-processors, viewport point cloud (follow-up) |

## Running the tests

The round-trip + solve suites need the radar stack importable (`witwin.radar` /
`witwin.core`, which pull in drjit + mitsuba; the package eagerly initializes the mitsuba
CUDA variant) plus `witwin_server` on the path. With those present:

```bash
# from this directory, in an env that has witwin.radar + witwin.core
PYTHONPATH=/path/to/witwin-studio/server python -m pytest tests/ -q
```

If the radar stack can't be imported, the suite skips itself rather than erroring.
The dirichlet/slang backends require a CUDA device; the pytorch backend runs on CPU.

On Windows, the dirichlet and slang backends compile through SlangTorch and write a
per-source cache under the external radar package, for example
`E:\Code\witwin-platform\radar\witwin\radar\solvers\.slangtorch_cache`. If pytest appears
to hang before any RayD trace logs, check SlangTorch's lock path first. In this workspace
the non-elevated sandbox could not acquire
`dirichlet.slangb9c103f6b206b8e5.lock`; clearing `.slangtorch_cache` and running the GPU
suite with permission to write/lock the external radar package fixed the apparent hang.

## Measuring live stream throughput

Use `tools/live_stream_throughput.py` from the studio repo root to measure the solver-side
Start Stream path:

```powershell
$env:PYTHONPATH="E:\Code\witwin-studio\server;E:\Code\witwin-studio\plugins;E:\Code\witwin-platform\radar;E:\Code\witwin-platform\core"
conda run -n witwin2 python plugins\wt_radar\tools\live_stream_throughput.py --profile smoke --frames 20 --warmup-frames 2 --max-fps 30 --channels raw,rd,pc --backend dirichlet --device cuda --resolution 16 --encode-bytes
```

The script drives `solver_host.LiveSession` directly and reports frame FPS, inter-frame
latency, solve/post-processing timings, channel payload bytes, and timeout/in-flight solve
state. Use `--profile demo` to test the heavier default demo radar configuration.

To measure the frontend component's actual canvas refresh rate, enable the stream widget
metrics collector in the browser devtools console before starting the stream:

```javascript
localStorage.setItem("witwinStreamWidgetMetrics", "1");
window.__WITWIN_STREAM_WIDGET_METRICS__ = undefined;
```

Reload the app, start the radar stream, then inspect:

```javascript
window.__WITWIN_STREAM_WIDGET_METRICS__.summary()
```

That summary is based on completed `canvas` draws in the stream widgets, not on backend
publish completion.
