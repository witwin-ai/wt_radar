# wt-radar Realtime Stream 调试记录

## 当前问题现象

- 点击 `Start Stream` 后，`signal_stream` 能创建，前端 stream header 后面的 `#seq` 在正常情况下会增长。
- 拖拽雷达或目标时，console 中会持续出现位置更新 / `Radar live update request` 日志，但图片 stream 后面的 `#seq` 会停止增长。
- 停止拖拽或等待一段时间后，约 6-7 秒才重新拿到新的 stream 帧。
- 前端 heatmap canvas 绘制本身不慢。已采集到的指标中：
  - `drawMs.mean` 约 1ms。
  - `displayLatencyMs.mean` 约 20ms。
  - 因此“帧已经到前端但图片画不出来”不是主要瓶颈。
- 之前尝试在 gizmo 拖拽中高频发送 transient transform 是错误方向：它会触发大量 `live_update`，导致 live solver 持续更新状态，反而让 stream 长时间没有新帧发布。

## 已确认的关键线索

### 前端绘制路径

`StreamRefWidget` 的 header `#seq` 和 heatmap canvas 是两条路径：

- header 订阅 `streamStore`，收到帧后更新 `#seq`。
- heatmap/waveform 子组件通过 `requestAnimationFrame` 绘制 canvas。

通过 `window.__WITWIN_STREAM_WIDGET_METRICS__.summary()` 看到：

- canvas draw time 很低。
- display latency 很低。
- 所以如果 `#seq` 卡住，问题在后端/传输/solver 发布，而不是 canvas 绘制。

### gizmo transform 同步

当前 viewport `TransformControls` 的默认逻辑是：

- 拖拽过程中只更新 Three.js object 本地可视状态。
- 不更新 frontend store。
- 不发送 transform 到后端。
- `mouseUp` 时才调用 `wsService.updateTransform(...)` 同步最终 transform。

这意味着默认情况下，后端 scene 和 live solver 不会实时知道拖拽过程中的连续位置。

### 错误尝试：拖拽中直接 live sync transform

之前尝试过在拖拽中每约 33ms 发送 transient transform。实际日志显示：

- `Radar live loop updating solver` 高频刷屏。
- `Radar live update request` 高频刷屏。
- stream publish 没有跟上，`#seq` 停止。
- 后端出现 websocket drain 相关 `AssertionError`。

结论：不能用“每个 gizmo tick 都推给 server scene + live_update”的方式实现实时雷达。这会让 solver update 和 stream publish 相互饥饿。

## 当前代码层修正

在 `plugins/wt_radar/solver_host.py` 中，live session 原先逻辑是：

- 取 snapshot。
- 开始 solve。
- solve 期间如果有新的 update 导致 `state_seq` 变化。
- solve 完成后检查 `snapshot["state_seq"] != self._state_seq`，如果不同就丢弃这一帧。

在高频 update 时，这会造成持续丢帧：每一帧刚算完都已经 stale，于是永远不 publish。

当前修正为：

- stop / pause 时仍然丢弃完成帧。
- 但如果只是 solve 期间来了新的 update，不再丢弃已经完成的帧。
- 这样最多显示一帧旧 pose，不会让 stream 被持续饿死。

对应回归测试：

```powershell
conda run -n witwin2 python -m pytest plugins\wt_radar\tests\test_r6_realtime_stream.py -q
```

当前结果：`13 passed`。

## 调试方法

### 1. 前端实际绘制指标

在 Studio DevTools console 中启用：

```js
localStorage.setItem("witwinStreamWidgetMetrics", "1")
window.__WITWIN_STREAM_WIDGET_METRICS__?.reset?.()
```

Start Stream 后拖拽 5-10 秒，再运行：

```js
window.__WITWIN_STREAM_WIDGET_METRICS__.summary()
```

重点看：

- `drawMs`: canvas 绘制耗时。
- `displayLatencyMs`: 帧到达前端后到绘制完成的延迟。
- `sampleHashUnique` / `sampleHashChanges`: RD payload 内容是否变化。
- `metadataStateSeqChanges`: solver live session 收到的 scene/config 更新次数。
- `metadataSignatureChanges`: scene signature 是否变化。

判断：

- `drawMs` 高：前端绘制堵。
- `displayLatencyMs` 高：前端主线程/RAF 或 store 到 canvas 链路堵。
- `#seq` 不变：后端没有发布新帧或 websocket 未送达。
- `metadataStateSeqChanges` 很低：后端没有收到新的 scene state。
- `metadataStateSeqChanges` 很高但 `#seq` 卡住：solver/live_update/publish 被更新流饿死。

### 2. 后端日志模式

需要关注以下日志是否成对出现：

```text
Radar live update request: ...
Radar stream publish started: ...
Radar stream publish complete: ...
```

异常模式：

```text
Radar live update request ...
Radar live update request ...
Radar live update request ...
```

但长时间没有 `Radar stream publish complete`，说明 live update 正在压制 frame publish。

### 3. GPU/Dirichlet throughput 脚本

用于隔离 solver/postprocess/publish 模拟耗时：

```powershell
$env:PYTHONPATH='E:\Code\witwin-studio\server;E:\Code\witwin-studio\plugins;E:\Code\witwin-platform\radar;E:\Code\witwin-platform\core'
conda run -n witwin2 python plugins\wt_radar\tools\live_stream_throughput.py --profile demo --frames 20 --warmup-frames 1 --max-fps 30 --channels raw,rd,pc --backend dirichlet --device cuda --resolution 128 --timeout 180 --encode-bytes --simulate-sdk-base64
```

该脚本只能证明 solver-side throughput；它不能替代 Studio 前端端到端测试。

### 4. 聚焦测试命令

wt-radar live stream：

```powershell
conda run -n witwin2 python -m pytest plugins\wt_radar\tests\test_r6_realtime_stream.py -q
```

server stream backpressure：

```powershell
conda run -n witwin2 python -m pytest server\tests\ai_native\test_streams_handler.py -q
```

frontend stream widgets：

```powershell
npm.cmd test -- --watchAll=false --testPathPattern=StreamRefWidget.test.tsx
npm.cmd test -- --watchAll=false --testPathPattern=streamWidgetMetrics.test.ts
```

Studio build：

```powershell
npm.cmd run build
```

## 潜在原因列表

### 1. Scene transform 同步策略不支持真正 live drag

默认 gizmo 只在 mouseUp 同步 transform 到后端。拖拽中的连续位置只存在于前端 Three.js object，后端 radar solver 不知道。

要实现真正 realtime，需要设计专门的 live transform path，不能直接复用常规 `update_transform` 高频广播路径。

### 2. 高频 live_update 会饿死 frame publish

如果 drag 期间每个位置都触发 `live_update`，solver session 的 state 不断改变。旧逻辑会丢弃 solve 期间过期的帧，导致 stream 停止增长。

当前已通过允许“完成帧继续 publish”缓解这一点，但仍需端到端验证。

### 3. Dirichlet 新 pose cache miss / trace 重算成本高

松开后 2-7 秒才看到新结果，可能来自新 pose 下的 Dirichlet trace/cache miss。

需要看 solver log 中：

```text
live cache miss: build_ms=... trace_ms=... signal_ms=...
live cache hit: mode=...
```

如果 `trace_ms` 是秒级，瓶颈在光线追踪/trace 构建，不在前端。

### 4. `raw,rd,pc` 三通道会增加后处理和传输压力

旧默认 `stream_channels` 是 `raw,rd,pc`。RD 预览只看 `rd`，但每个 radar frame 还会做 raw 和 point cloud 发布。

实时预览当前默认改成：

```text
rd
```

这样可以排除 raw/pc 后处理和传输造成的额外压力；需要诊断 raw/pc 时再显式打开。

### 5. WebSocket backpressure / drain 问题

日志出现过 websockets `keepalive_ping` / `drain` 的 `AssertionError`。这说明在高频消息或大 payload 场景下，websocket 写队列可能被打满或进入异常状态。

server 端已经为 `latestOnly=True` 的 stream frame 做了 latest-only coalescing，但 transform/update/control 消息仍可能造成压力。

### 6. Studio 端端到端 FPS 与 solver-side FPS 不同

solver-side benchmark 可以达到 30fps，不代表 Studio component 控件实际 30fps。

端到端需要同时看：

- solver publish fps。
- websocket frame arrival。
- `streamStore` seq。
- canvas draw fps。
- payload hash 变化。

## 2026-05-31 实时渲染优化记录

这次 latency 修复的目标不是“拖拽中每一帧都同步”，而是先保证 gizmo 松开后能很快看到新结果。最终有效的是减少控制消息、缩小重建范围、复用 solver 运行时对象。

### 使用的优化技巧

#### 1. 先区分前端绘制慢还是后端没发帧

不要只看 heatmap 图片是否刷新。`StreamRefWidget` 的 `#seq` 是更直接的信号：

- `#seq` 不变：后端/传输/solver 没有发布新帧。
- `#seq` 变但图片不变：前端绘图、payload materialize、range 或 canvas 路径有问题。

这次确认 `drawMs` 约 1ms、`displayLatencyMs` 约 20ms，所以主要 latency 不在 canvas，而在 solver/update/publish 链路。

#### 2. 实时预览默认只发布需要看的通道

RD 预览只需要 `rd`。实时调试时默认使用：

```text
stream_on_change_only = True
stream_channels = rd
```

避免每次 pose 更新还额外做 raw/point-cloud 后处理和传输。raw/pc 可以作为诊断通道，但不应该是低延迟预览的默认负载。

#### 3. control plane 和 data plane 不要互相饥饿

高频 `live_update` 是控制消息，stream frame 是数据消息。把 gizmo 拖拽中的每个 tick 都变成 scene update，会让 solver session 一直处于“状态刚变”的状态，最终 publish 被饿死。

当前策略：

- gizmo 默认仍然只在 `mouseUp` 同步最终 transform。
- live session 完成一帧后只在 stop/pause 时丢弃；普通 update 到达不再让已完成帧作废。
- 如果将来做 drag-live，需要单独 latest-only pose control channel，而不是复用常规 scene update。

#### 4. pose-only update 不要重发整场景

只移动雷达 sensor 时，geometry、materials、motion graph 没变。`live_update` 现在会比较并传递 `scene_payload_signature`：

- 只有 radar/sensor pose 变化：只发送 solver config，不带 `scene`。
- geometry/structure/config 会影响 platform scene：才重新发送 scene。

这避免了每次松开 gizmo 都经过 sceneRef 序列化、load_scene_ref 和 adapter rebuild。

#### 5. cache miss 也要拆粒度，不要把所有 miss 当作全量重建

旧逻辑里 signature 变了就会：

```text
RadarAdapter.to_platform(...)
wr.Radar(...)
Tracer(...)
trace(...)
sigproc(...)
```

这对 sensor pose-only 变化太重。现在 live cache 拆成几层：

- platform scene/config：同一 solver session 且 `scene_payload_signature` 没变时复用。
- `wr.Radar`：RadarConfig、backend、device、pad_factor 不变时复用。
- sensor pose：通过 `Radar.set_pose(position, target, up, fov)` 更新。
- tracer：scene topology 和 tracer spec 不变时复用。
- trace/path cache：cache key 完全相同时才复用。

因此松开 gizmo 后的常见路径变为：

```text
reuse platform scene/config
reuse wr.Radar
Radar.set_pose(...)
reuse Tracer
trace(...)
RD publish
```

这比重建 `wr.Radar` 和 `to_platform` 快很多，也避免 CUDA/Dirichlet 初始化反复出现在交互路径里。

#### 6. heatmap 的显示 range 要用“显示范围”，不要直接用数据极值

RD map 是 dB heatmap，直接用整张图的 min/max 很容易被一个强 outlier 或直达径拉坏，表现为大面积蓝底和红色十字/横线饱和。

前端 heatmap 现在的默认规则：

- 显式 `min/max`、`vmin/vmax`、`displayMin/displayMax` 优先。
- float 数据没有显式范围时，用稳健高分位作为 `vmax`。
- 默认动态范围约 60dB，避免极小噪声或极大尖峰决定整张图。

这类图像不要把“物理数据范围”和“可视化动态范围”混为一谈。

### 这次踩过的坑

#### 1. “更实时地同步 transform”反而更慢

直觉上高频同步 gizmo 会更实时，但实际会让 control update 压过 stream publish。对重计算型渲染/仿真，实时交互的第一原则是 latest-only 和可丢弃中间状态，而不是逐条可靠执行。

#### 2. 只做 latest-only stream 不够

server -> frontend 的 stream frame 可以 coalesce，但 frontend/server -> solver 的 control update 如果不 coalesce，仍然会把 solver 卡住。control plane 也需要 latest-only 思维。

#### 3. signature 变了不等于所有运行时对象都失效

scene signature 包含 radar pose，pose 变会让 cache key 变。但这不代表 RadarConfig、platform scene、tracer topology 都变了。实时系统里 cache key 要按失效原因拆开，否则每次小改动都会变成全量重建。

#### 4. 端到端延迟不能用单点 benchmark 代替

solver-side throughput 脚本只能说明 solver 在隔离条件下的速度。Studio 真实延迟还包括 sceneRef/control message、solver session 调度、publish、websocket、store、RAF 绘制和 heatmap range。

#### 5. heatmap 看起来“没刷新”可能是 range 错，不是帧没到

这次截图里的红色十字说明数据很可能到了，但显示 range 被强反射/极值拉坏。诊断顺序应该是：

1. 看 `#seq` 是否增长。
2. 看 payload hash 是否变化。
3. 再看 heatmap range/colormap。

不要只凭图片外观判断 stream 没更新。

### 后续真正 drag-live 的架构建议

如果目标是拖拽过程中连续看到雷达响应，建议不要走完整 scene update，而是：

1. 前端 gizmo drag pose 发到独立 latest-only control channel。
2. 后端 live session 只保存最新 pose，不排队执行每个 pose。
3. solver 每一帧开始时读取最新 pose，正在计算的帧允许完成并发布。
4. scene/geometry/config 变化才走重 sceneRef update。
5. stream frame data plane 与 pose control plane 做限速和合并，避免 websocket 写队列互相阻塞。

## 下一步建议

1. 先验证当前 `solver_host.py` 防饥饿修复：拖拽时即使后端收到 update，`#seq` 不应长时间停住。
2. 保持默认 gizmo mouseUp 同步，不再启用高频 transient transform。
3. 把 `stream_channels` 临时设置为 `rd`，测松开后的延迟是否降低。
4. 打开 solver log，确认松开后的 2-7 秒是否对应 `live cache miss trace_ms`。
5. 如果目标是真正 drag-live radar，需要设计新架构：
   - 前端 drag pose 走单独 latest-only control channel。
   - 后端只保留最新 pose，不逐条 live_update。
   - solver frame 边界读取最新 pose。
   - 计算中的帧完成后允许发布，下一帧用最新 pose。
   - 控制消息与 stream frame data plane 分离，避免 websocket control/update 混在高频大 payload 里互相阻塞。
