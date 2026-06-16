# Label Generation Notes

本文件记录空间/时序模型第一版标签生成的约定。`beatmap.json` 中的对象时间、圆心、slider
路径和 spinner 时间窗是原始标注；逐帧的 approach circle 半径、可见性、动作状态和 patch
局部标签是派生监督，不应伪装成原始标注。

## 坐标系

- 原始 osu!standard playfield 坐标为 `512 x 384`。
- 视频像素坐标通过 `package.OsuVideoTransform` 做居中等比缩放。
- Patch 标签先在完整视频坐标系生成，再按 `PatchMeta.x0/y0/x1/y1` 裁剪到局部。

## Approach Circle

若源数据没有直接给出每一帧的缩圈半径，第一版使用近似规则：

```text
preempt_ms = approach-rate-derived preempt time
progress = clamp((object_start_ms - frame_time_ms) / preempt_ms, 0, 1)
current_radius = hit_circle_radius * (1 + 3 * progress)
time_to_hit_ms = object_start_ms - frame_time_ms
radius_change = -3 * hit_circle_radius / preempt_ms
```

该公式只表示训练监督近似：对象越接近 hit time，approach circle 越收缩到
`hit_circle_radius`。如果后续从 osu!lazer 规则或渲染端获得更精确半径，应替换该派生
公式，并保留版本号。

## Slider

Slider 标签在全局视频坐标中生成：

1. 从原始 slider 控制点采样中心线 polyline。
2. 按 hit circle 半径生成 body mask。
3. 单独生成 head/tail/repeat point heatmap。
4. 对中心线局部切线写入无方向方向场 `(cos 2θ, sin 2θ)`。
5. 裁剪到 patch 后保留同一 `object_identity`，用于跨 patch embedding consistency。

第一版支持互不接触、无分叉的 slider 连通分量；交叉、接触和多 head 竞争同一连通分量的样本
应在数据检查或 target 构建阶段标记为超出支持范围。

## Action State

逐帧动作标签从对象时间窗派生：

- `no_op`：当前帧没有需要执行或保持的对象。
- `press`：circle hit time 或 slider head/spinner start 附近的首次按下。
- `hold`：slider/spinner 活跃期间持续按下。
- `release`：slider/spinner 结束后的释放帧。

这些动作标签是因果时序训练目标，只能依赖当前帧时间及历史状态，不能使用未来帧修正过去标签。
