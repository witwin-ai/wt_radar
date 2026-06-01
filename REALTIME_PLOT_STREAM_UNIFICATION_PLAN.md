# Realtime Plot / Stream 统一化方案

## 背景

当前 Radar 有两套并行显示路径：

- 静态结果走 `figure()` / `PlotData`，例如 `signal_figure.clear().line(...)`、`imshow(...)`、`scatter(...)`。
- 实时结果走 `stream_ref_field`，前端 `StreamRefWidget` 根据 channel `semantic/dtype` 选择 `StreamHeatmapWidget` 或 `StreamWaveformWidget`。

这导致同一个语义的数据在两套 renderer 中重复实现。典型问题是 Raw MIMO：`simulate` 的 plot 显示 Real/Imag 两条 IQ 曲线，但 stream widget 曾按 `complex64` 幅度显示，看起来和 plot 结果不一致。

目标不是继续给 `signal_stream` 写补丁，而是把 stream 变成 plot 的数据源之一，让静态 plot 和 realtime stream 共享同一套 plot schema、renderer 和交互逻辑。

## 目标

1. `figure()` 和 `stream_ref_field` 最终都进入统一的 plot rendering pipeline。
2. 同一类数据只实现一个 renderer：
   - Raw IQ -> line-series renderer
   - Range-Doppler -> heatmap/imshow renderer
   - Point cloud -> point/scatter renderer
3. 后端只声明数据语义和 plot spec，不指定前端专用组件。
4. stream frame 到达后被转换成同一种 `PlotModel`，再交给 plot renderer。
5. 允许组件逐步迁移，旧 `PlotData` 和旧 stream widgets 在过渡期继续工作。

## 非目标

- 不在第一阶段重写所有 16 种 `PlotData` 图表类型。
- 不把高频 stream payload 存进 React state。
- 不要求所有 stream 都可回放完整历史；实时预览仍默认 latest-only。
- 不把 Radar 专属概念写进 Studio 通用 stream renderer。

## 当前问题

### 1. Renderer 分裂

`PlotData` 和 stream widgets 都能画线、热力图等基础图形，但各自处理：

- dtype / shape 解析
- range / colormap
- axis label / title
- series 命名和颜色
- latest frame scheduling

结果是静态和实时显示很容易漂移。

### 2. Stream descriptor 表达力不足

当前 descriptor 主要是：

```json
{
  "channelId": "rd",
  "dtype": "float32",
  "shape": ["doppler", "range"],
  "semantic": "heatmap",
  "label": "Range-Doppler"
}
```

这不足以表达“这个 channel 应该按 `PlotData.imshow` 的语义显示，使用哪些 axes、哪些 series、哪些 controls”。

### 3. `stream-ref` 选择组件太早

`StreamRefWidget` 当前根据 channel `semantic/dtype` 直接选择 `StreamHeatmapWidget` / `StreamWaveformWidget`。这相当于在 stream 层决定渲染类型，而不是在 plot 层决定。

## 统一数据模型

新增一个前端通用模型，建议命名为 `PlotModel`：

```ts
type PlotModel =
  | LinePlotModel
  | HeatmapPlotModel
  | PointPlotModel
  | TextPlotModel;

interface PlotModelBase {
  kind: 'line' | 'heatmap' | 'points' | 'text';
  title?: string;
  xLabel?: string;
  yLabel?: string;
  zLabel?: string;
  source?: {
    type: 'figure' | 'stream' | 'query';
    streamId?: string;
    channelId?: string;
    seq?: string;
  };
}
```

第一阶段只需要覆盖 Radar 使用的三类：

### LinePlotModel

```ts
interface LinePlotModel extends PlotModelBase {
  kind: 'line';
  x: ArrayLike<number>;
  series: Array<{
    y: ArrayLike<number>;
    label?: string;
    color?: string;
  }>;
  batch?: {
    primaryLabels?: string[];
    secondaryLabels?: string[];
    entries?: LinePlotModel[];
  };
}
```

Raw IQ 的 stream 和 `figure().line()` 都应该生成这个模型。`complex64` 不应该默认转 magnitude，而应该根据 spec 转成 Real/Imag series。

### HeatmapPlotModel

```ts
interface HeatmapPlotModel extends PlotModelBase {
  kind: 'heatmap';
  values: ArrayLike<number>;
  shape: [number, number];
  colormap?: string;
  range?: {
    mode?: 'full' | 'robust' | 'explicit';
    min?: number;
    max?: number;
    dynamicRangeDb?: number;
  };
  overlays?: Array<{
    kind: 'points';
    rows: ArrayLike<number>;
    cols: ArrayLike<number>;
    color?: string;
  }>;
}
```

Range-Doppler 的静态 plot 和 stream 都生成这个模型。

### PointPlotModel

```ts
interface PointPlotModel extends PlotModelBase {
  kind: 'points';
  points: ArrayLike<number>;
  stride: 3 | 6;
  colorBy?: 'intensity' | 'velocity' | 'constant';
  pointSize?: number;
}
```

Point cloud 的 viewport overlay 和 properties panel preview 可以共享这个模型，但 renderer 可以不同：panel 用 2D/3D plot renderer，viewport 用 Three.js bridge。

## 数据源适配层

引入 `plotSources` 层，将不同来源转换成 `PlotModel`：

```text
Figure / PlotData  -> plotDataToModel()
Stream descriptor + frame -> streamFrameToPlotModel()
Solver query result -> queryResultToPlotModel()   可选
```

关键原则：

- Renderer 只认 `PlotModel`。
- Stream widget 不直接画 canvas。
- `PlotData` 不直接绑定 React chart 实现。
- 所有 dtype/shape/metadata 解码都在 adapter 层完成。

## Stream descriptor 扩展

在 channel descriptor 上增加 `plot` 字段：

```json
{
  "channelId": "raw",
  "dtype": "complex64",
  "shape": ["adc"],
  "semantic": "time_series",
  "label": "Raw IQ",
  "plot": {
    "kind": "line",
    "x": { "mode": "index", "label": "ADC sample" },
    "y": { "label": "Amplitude" },
    "complex": "real_imag",
    "series": [
      { "component": "real", "label": "Real", "color": "#ff9500" },
      { "component": "imag", "label": "Imag", "color": "#00aaff" }
    ]
  }
}
```

Range-Doppler：

```json
{
  "channelId": "rd",
  "dtype": "float32",
  "shape": ["doppler", "range"],
  "semantic": "heatmap",
  "label": "Range-Doppler",
  "plot": {
    "kind": "heatmap",
    "x": { "label": "Range bin" },
    "y": { "label": "Doppler bin" },
    "colormap": "imshow",
    "range": { "mode": "full" }
  }
}
```

Point cloud：

```json
{
  "channelId": "pc",
  "dtype": "float32",
  "shape": ["points", 6],
  "semantic": "pointcloud_xyz",
  "label": "Point cloud",
  "plot": {
    "kind": "points",
    "stride": 6,
    "axes": ["x", "y", "z"],
    "colorBy": "intensity"
  }
}
```

Frame metadata can override descriptor defaults for dynamic values:

- selected `tx/rx/chirp`
- actual physical axis values
- explicit `vmin/vmax`
- overlay detections
- frame sequence / solver signature

## Frontend 结构

建议新增目录：

```text
studio/src/features/plots/
  model.ts
  adapters/
    plotDataToModel.ts
    streamFrameToModel.ts
  renderers/
    PlotRenderer.tsx
    LinePlotRenderer.tsx
    HeatmapPlotRenderer.tsx
    PointPlotRenderer.tsx
  controls/
    PlotToolbar.tsx
```

### PlotRenderer

`PlotRenderer` 根据 `model.kind` 分发：

```tsx
<PlotRenderer model={model} live={source.type === 'stream'} />
```

现有 `PlotWidget` 和 `StreamRefWidget` 都使用它：

```text
PlotWidget:
  PlotData -> plotDataToModel -> PlotRenderer

StreamRefWidget:
  latest StreamFrame -> streamFrameToModel -> PlotRenderer
```

### StreamRefWidget 的新职责

保留为 stream data source host，但不再直接选择 `StreamHeatmapWidget` / `StreamWaveformWidget`：

1. 解析 stream id / channel id。
2. 管理 subscribe/unsubscribe。
3. 从 `streamStore` 取 latest frame。
4. 调 `streamFrameToModel(descriptor, frame)`。
5. 渲染 `<PlotRenderer model={model} />`。

旧的 `StreamHeatmapWidget` / `StreamWaveformWidget` 可以先变成 adapter/renderer 内部实现，之后逐步删除。

## 后端 API 变化

### `PlotData`

短期保持兼容，但新增一个标准字段：

```py
PlotData(..., metadata={"plot": {...}})
```

或在 `PlotData.to_dict()` 中输出规范化 plot spec。这样 `figure().line()` 和 stream descriptor 使用同一套元数据字段。

### stream descriptor

`api.streams.open(..., channels=[...])` 支持 channel `plot` 字段。旧字段 `semantic` 继续保留，用于 fallback。

### Radar 发布

Radar 不再发布“前端 widget 指令”，只发布数据和 plot spec：

- raw: complex64 ADC samples + `plot.kind=line`
- rd: float32 2D map + `plot.kind=heatmap`
- pc: float32 Nx6 + `plot.kind=points`

`signal_stream` 不应该知道“这是 radar signal 的特殊组件”。它只是一个 stream-ref field。

## 迁移步骤

### Phase 1: 建立统一模型，不改视觉

1. 新建 `features/plots/model.ts`。
2. 实现 `plotDataToModel()`，覆盖 line/imshow/scatter。
3. 实现 `streamFrameToModel()`，覆盖 raw/rd/pc。
4. 新建 `PlotRenderer`，先复用现有 line/heatmap/point rendering 代码。
5. 加单元测试证明 raw `complex64` -> Real/Imag line model。

验收：

- `simulate` raw plot 和 raw stream preview 进入同一个 `LinePlotRenderer`。
- `simulate` RD plot 和 RD stream preview 进入同一个 `HeatmapPlotRenderer`。

### Phase 2: 改 StreamRefWidget

1. `StreamRefWidget` 不再直接 import `StreamHeatmapWidget` / `StreamWaveformWidget`。
2. 它只负责订阅最新 frame 并转换成 `PlotModel`。
3. channel descriptor 如果没有 `plot` 字段，使用 legacy fallback：
   - `semantic=heatmap` -> heatmap
   - `semantic=time_series` / `dtype=complex64` -> line
   - `semantic=pointcloud_xyz` -> points

验收：

- 旧 stream 不带 `plot` 字段也能显示。
- Radar raw/rd/pc 都能通过统一 renderer 显示。

### Phase 3: 改 Radar descriptor

1. `_stream_channel_descriptors()` 给 raw/rd/pc 增加 `plot` 字段。
2. solver live publish metadata 只提供动态 override，不重复静态 plot spec。
3. 删除 `signal_stream` 上的 radar 特例。

验收：

- 切换 `view` 只改变 selected channel/source，不改变 renderer 逻辑。
- Raw stream 和 Raw plot 在 axis/series/color 上一致。
- RD stream 和 RD plot 在 colormap/range 上一致。

### Phase 4: 清理旧组件

1. 把 `StreamHeatmapWidget` 和 `StreamWaveformWidget` 内部逻辑迁到 `features/plots/renderers`。
2. 删除或降级旧 stream widgets 为兼容 wrapper。
3. 文档说明 stream channel 如何声明 plot spec。

## 测试计划

### 前端单元测试

- `plotDataToModel(line)` 保留 series/color/axis。
- `plotDataToModel(imshow)` 保留 shape/range/colormap。
- `streamFrameToModel(raw complex64)` 输出 Real/Imag series。
- `streamFrameToModel(rd float32)` 输出 heatmap model。
- `StreamRefWidget` 对最新 frame 渲染 `PlotRenderer`，不直接依赖 heatmap/waveform widget。

### 后端单元测试

- Radar raw descriptor 包含 `plot.kind=line` 和 `complex=real_imag`。
- Radar rd descriptor 包含 `plot.kind=heatmap` 和 range/colormap。
- Live publish raw payload 与 `raw_signal` query 的 selected `tx/rx/chirp` 一致。

### 端到端检查

1. Run `Simulate`，记录 Raw plot。
2. Start Stream，切 `view=raw_signal`。
3. 确认 stream preview 与 Raw plot：
   - 两条曲线 Real/Imag。
   - 同样颜色。
   - 同样 x/y label。
   - 同样 tx/rx/chirp metadata。
4. RD 和 point cloud 做同样检查。

## 风险和注意事项

### Canvas 和 React state

高频 stream frame 不能把完整 payload 放进 React state。`StreamRefWidget` 可以把 latest frame record 存 ref，使用 RAF 绘制，或者让 `PlotRenderer` 支持 imperative draw path。

### 大 payload

`PlotModel` 中的 `values` 应保持 typed array，不要 `Array.from()`。序列化只发生在后端静态 `PlotData`，stream payload 已经是二进制。

### 兼容旧 PlotData

`PlotData` 当前字段结构不能一次性移除。需要 adapter 先兼容旧格式，再逐步让 `Figure` 生成更规范的 metadata。

### Viewport point cloud

Properties panel preview 和 viewport overlay 可以共享 `PointPlotModel`，但实际 renderer 可以分离。不要为了统一 preview 而牺牲 viewport 性能。

## 建议的落地顺序

优先做 frontend model/adapter，不先改 Radar 业务逻辑：

1. `features/plots/model.ts`
2. `plotDataToModel.ts`
3. `streamFrameToModel.ts`
4. `PlotRenderer.tsx`
5. 改 `PlotWidget` 使用 `PlotRenderer`
6. 改 `StreamRefWidget` 使用 `PlotRenderer`
7. Radar descriptor 增加 `plot` spec
8. 删除旧 stream widget 特例

这样每一步都有可测试边界，且不会再出现“为 raw signal 单独修一个 widget，但架构仍然分裂”的问题。

