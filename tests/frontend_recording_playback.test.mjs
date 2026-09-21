import assert from "node:assert/strict";
import test from "node:test";

import {
  activate,
  recordingFrameAtTime,
  recordingFramePlot,
  validateRecordingManifest,
} from "../frontend/viewport.mjs";

function fixture() {
  return {
    schema: 1,
    sceneId: "main",
    timesS: [0, 0.1, 0.2],
    endTimeS: 0.3,
    rangeM: [0, 1],
    velocityMps: [-1, 1],
    frameStride: 6,
    profileMax: 20,
    dopplerMax: 30,
    sourceLabel: "Native Radar",
    view: "range_doppler",
    asset: {
      uri: "/scenes/main/assets/replay",
      byteLength: 72,
      contentHash: `sha256:${"a".repeat(64)}`,
    },
  };
}

test("scene time selects distinct recorded frames", () => {
  const manifest = validateRecordingManifest(fixture());
  assert.equal(recordingFrameAtTime(manifest, 0), 0);
  assert.equal(recordingFrameAtTime(manifest, 0.099), 0);
  assert.equal(recordingFrameAtTime(manifest, 0.1), 1);
  assert.equal(recordingFrameAtTime(manifest, 0.299), 2);
  assert.equal(recordingFrameAtTime(manifest, 0.3), null);
});

test("range-doppler plot slices the selected frame rather than frame zero", () => {
  const manifest = validateRecordingManifest(fixture());
  const samples = Float32Array.from({ length: 18 }, (_, index) => index);
  const first = recordingFramePlot(manifest, samples, 0);
  const last = recordingFramePlot(manifest, samples, 2);
  assert.deepEqual(Array.from(first.data.values), [2, 3, 4, 5]);
  assert.deepEqual(Array.from(last.data.values), [14, 15, 16, 17]);
  assert.match(last.title, /frame 3\/3/);
});

test("manifest rejects a payload whose byte length cannot hold every frame", () => {
  assert.throws(
    () => validateRecordingManifest({ ...fixture(), asset: { ...fixture().asset, byteLength: 68 } }),
    /shape differs/,
  );
});

test("frontend activation claims only WT Radar recording plots", () => {
  const registrations = [];
  globalThis.__witwin = { react: {} };
  activate({ registerWidgetDecorator: (registration) => registrations.push(registration) });
  assert.equal(registrations.length, 1);
  assert.equal(registrations[0].id, "witwin.radar.recordingPlot");
  assert.equal(registrations[0].match({ value: { data: { recording: { schema: 1 } } } }), true);
  assert.equal(registrations[0].match({ value: { data: {} } }), false);
  delete globalThis.__witwin;
});
