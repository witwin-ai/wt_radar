/** WT Radar's durable numeric replay widget.
 *
 * Studio deliberately has no Radar-specific replay implementation.  This host
 * module owns the recording schema emitted by adapter/replay.py, reads the
 * content-addressed float32 payload through Studio's asset API and follows the
 * source Scene clock without asking the Python backend to redraw every frame.
 *
 * Hand-written ES module (no build step), typed with JSDoc against
 * @witwin/host 2.
 */

export const sdkVersion = "2.0.0";

const MAX_RECORDING_BYTES = 128 * 1024 * 1024;

function finiteIncreasing(values, minimum) {
  return Array.isArray(values) && values.length >= minimum && values.every(
    (value, index) => Number.isFinite(value) && (index === 0 || value > values[index - 1]),
  );
}

export function validateRecordingManifest(raw) {
  if (!raw || raw.schema !== 1 || typeof raw.sceneId !== "string" || !raw.sceneId
      || !finiteIncreasing(raw.timesS, 1) || raw.timesS[0] < 0
      || !finiteIncreasing(raw.rangeM, 2) || !finiteIncreasing(raw.velocityMps, 2)
      || !Number.isFinite(raw.endTimeS) || raw.endTimeS <= raw.timesS.at(-1)
      || typeof raw.sourceLabel !== "string"
      || ![raw.profileMax, raw.dopplerMax].every((value) => Number.isFinite(value) && value >= 0)
      || (raw.view !== undefined && !["range_profile", "range_doppler"].includes(raw.view))) {
    throw new Error("Invalid Radar recording schema, timebase or physical axes.");
  }
  const frameStride = raw.rangeM.length * (1 + raw.velocityMps.length);
  const byteLength = raw.timesS.length * frameStride * 4;
  if (!Number.isSafeInteger(byteLength) || byteLength > MAX_RECORDING_BYTES
      || raw.frameStride !== frameStride || raw.asset?.byteLength !== byteLength) {
    throw new Error("Radar recording shape differs from its binary payload.");
  }
  if (typeof raw.asset?.uri !== "string" || typeof raw.asset?.contentHash !== "string") {
    throw new Error("Radar recording has no durable binary asset.");
  }
  return {
    ...raw,
    asset: { ...raw.asset },
    timesS: [...raw.timesS],
    rangeM: [...raw.rangeM],
    velocityMps: [...raw.velocityMps],
  };
}

/** Hold the last real measurement inside the measured interval. */
export function recordingFrameAtTime(manifest, time) {
  if (!Number.isFinite(time) || time < manifest.timesS[0] || time >= manifest.endTimeS) return null;
  let low = 0;
  let high = manifest.timesS.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (manifest.timesS[middle] <= time) low = middle + 1;
    else high = middle;
  }
  return low - 1;
}

export function recordingFramePlot(manifest, samples, frame, requestedView = manifest.view ?? "range_doppler") {
  if (!Number.isInteger(frame) || frame < 0 || frame >= manifest.timesS.length
      || samples.length !== manifest.timesS.length * manifest.frameStride) {
    throw new Error("Invalid Radar recording frame.");
  }
  const offset = frame * manifest.frameStride;
  const rangeCount = manifest.rangeM.length;
  const title = `${manifest.sourceLabel} | measurement ${manifest.timesS[frame].toFixed(3)} s | frame ${frame + 1}/${manifest.timesS.length}`;
  if (!["range_profile", "range_doppler"].includes(requestedView)) {
    throw new Error("Unsupported Radar replay view.");
  }
  if (requestedView === "range_profile") {
    return {
      type: "line",
      title,
      xlabel: "Range (m)",
      ylabel: "Linear amplitude (uncalibrated)",
      data: {
        x: manifest.rangeM,
        y: samples.subarray(offset, offset + rangeCount),
        xlim: [manifest.rangeM[0], manifest.rangeM.at(-1)],
        ylim: [0, manifest.profileMax > 0 ? manifest.profileMax * 1.05 : 1],
      },
    };
  }
  return {
    type: "imshow",
    title,
    xlabel: "Range (m)",
    ylabel: "Closing velocity (m/s)",
    data: {
      values: samples.subarray(offset + rangeCount, offset + manifest.frameStride),
      shape: [manifest.velocityMps.length, rangeCount],
      x: manifest.rangeM,
      y: manifest.velocityMps,
      origin: "lower",
      range: {
        mode: "explicit",
        min: 0,
        max: manifest.dopplerMax > 0 ? manifest.dopplerMax : 1,
        label: "Linear amplitude (uncalibrated)",
      },
    },
  };
}

function textPlot(title, content) {
  return { type: "text", title, data: { content, format: "plain" } };
}

/**
 * @param {import("@witwin/host").FrontendHost} host
 * @param {any} React the host's React singleton
 */
function createRecordingWidget(host, React) {
  const { useEffect, useMemo, useState } = React;
  return function RadarRecordingWidget(props) {
    const parsed = useMemo(() => {
      try {
        return { manifest: validateRecordingManifest(props.value?.data?.recording) };
      } catch (error) {
        return { error: error instanceof Error ? error.message : String(error) };
      }
    }, [props.value]);
    const manifest = parsed.manifest;
    const [time, setTime] = useState(() => manifest && host.playback
      ? host.playback.getTime(manifest.sceneId) : null);
    const [loaded, setLoaded] = useState();
    const [manualFrame, setManualFrame] = useState(null);
    const [playing, setPlaying] = useState(false);
    const [selectedView, setSelectedView] = useState(() => manifest?.view ?? "range_doppler");

    useEffect(() => {
      if (!manifest || !host.playback) return undefined;
      setTime(host.playback.getTime(manifest.sceneId));
      const subscription = host.playback.subscribe(manifest.sceneId, setTime);
      return () => subscription.dispose();
    }, [manifest?.sceneId]);

    useEffect(() => {
      if (!manifest) return undefined;
      let disposed = false;
      let lease;
      setLoaded(undefined);
      try {
        lease = host.assets.acquireSamples(manifest.asset);
        lease.promise.then(
          (samples) => { if (!disposed) setLoaded({ samples }); },
          (error) => { if (!disposed) setLoaded({ error: error instanceof Error ? error.message : String(error) }); },
        );
      } catch (error) {
        setLoaded({ error: error instanceof Error ? error.message : String(error) });
      }
      return () => { disposed = true; lease?.release(); };
    }, [manifest?.asset?.uri, manifest?.asset?.contentHash, manifest?.asset?.byteLength]);

    useEffect(() => {
      setManualFrame(null);
      setPlaying(false);
    }, [manifest?.asset?.contentHash, manifest?.view]);

    useEffect(() => {
      setSelectedView(manifest?.view ?? "range_doppler");
    }, [manifest?.asset?.contentHash, manifest?.view]);

    useEffect(() => {
      if (!playing || !manifest || manualFrame === null) return undefined;
      if (manualFrame >= manifest.timesS.length - 1) {
        setPlaying(false);
        return undefined;
      }
      const measuredDelay = 1000 * (manifest.timesS[manualFrame + 1] - manifest.timesS[manualFrame]);
      const timer = globalThis.setTimeout(
        () => setManualFrame((frame) => frame === null ? 0 : Math.min(frame + 1, manifest.timesS.length - 1)),
        Math.max(16, Math.min(measuredDelay, 1000)),
      );
      return () => globalThis.clearTimeout(timer);
    }, [playing, manualFrame, manifest]);

    let value;
    if (parsed.error) value = textPlot("Invalid Radar recording", parsed.error);
    else if (loaded?.error) value = textPlot(manifest.sourceLabel, `Radar replay unavailable: ${loaded.error}`);
    else if (!loaded?.samples) value = textPlot(manifest.sourceLabel, "Loading and verifying recorded Radar frames…");
    else {
      const frame = manualFrame ?? recordingFrameAtTime(manifest, time);
      value = frame === null
        ? textPlot(manifest.sourceLabel, `No recorded Radar frame at scene time ${Number(time).toFixed(3)} s.`)
        : recordingFramePlot(manifest, loaded.samples, frame, selectedView);
    }
    const plot = host.ui.renderWidget(
      { ...props.config, widget_type: "plot" },
      { ...props, value },
    );
    if (!manifest || !loaded?.samples) return plot;

    const timelineFrame = recordingFrameAtTime(manifest, time);
    const frame = manualFrame ?? timelineFrame ?? 0;
    const playLabel = playing ? "Pause" : (frame >= manifest.timesS.length - 1 ? "Replay" : "Play");
    const startOrPause = () => {
      if (playing) {
        setPlaying(false);
        return;
      }
      setManualFrame(frame >= manifest.timesS.length - 1 ? 0 : frame);
      setPlaying(true);
    };
    return React.createElement(
      "div",
      { style: { display: "flex", flexDirection: "column", gap: "8px", minWidth: 0 } },
      plot,
      React.createElement(
        "div",
        {
          role: "group",
          "aria-label": "Result Display",
          style: {
            display: "flex",
            alignItems: "center",
            gap: "8px",
            padding: "0 8px",
          },
        },
        React.createElement("span", { style: { whiteSpace: "nowrap" } }, "Result Display"),
        React.createElement(
          "select",
          {
            value: selectedView,
            onChange: (event) => setSelectedView(event.target.value),
            "aria-label": "Radar replay result display",
          },
          React.createElement("option", { value: "range_doppler" }, "Range-Doppler"),
          React.createElement("option", { value: "range_profile" }, "Range Profile"),
        ),
      ),
      React.createElement(
        "div",
        {
          style: {
            display: "flex",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "8px",
            padding: "0 8px 8px",
          },
        },
        React.createElement("button", { type: "button", onClick: startOrPause }, playLabel),
        React.createElement("input", {
          type: "range",
          min: 0,
          max: manifest.timesS.length - 1,
          step: 1,
          value: frame,
          onChange: (event) => {
            setPlaying(false);
            setManualFrame(Number(event.target.value));
          },
          style: { flex: "1 1 112px", minWidth: "88px" },
          "aria-label": "Recorded Radar frame",
        }),
        React.createElement(
          "span",
          { style: { whiteSpace: "nowrap", fontVariantNumeric: "tabular-nums" } },
          `${frame + 1}/${manifest.timesS.length} · ${manifest.timesS[frame].toFixed(3)} s`,
        ),
        React.createElement("button", {
          type: "button",
          onClick: () => {
            setPlaying(false);
            setManualFrame(null);
          },
          disabled: manualFrame === null,
        }, "Follow Timeline"),
      ),
    );
  };
}

/** @type {import("@witwin/host").PluginDefinition} */
const plugin = {
  id: "witwin.radar",
  activate(host) {
    const React = globalThis.__witwin?.react;
    if (!React) throw new Error("WT Radar replay requires the Studio React singleton.");
    host.registerWidgetDecorator({
      id: "witwin.radar.recordingPlot",
      widgetType: "plot",
      priority: 100,
      match: (props) => props.value?.data?.recording?.schema === 1,
      component: createRecordingWidget(host, React),
    });
  },
};

export default plugin;
