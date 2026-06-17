# Label Generation Notes

本文件记录空间/时序模型第一版标签生成的约定。`beatmap.json` 中的对象时间、圆心、slider
路径和 spinner 时间窗是原始标注；逐帧的 approach circle 半径、可见性、动作状态和 patch
局部标签是派生监督，不应伪装成原始标注。

## 坐标系

- 原始 osu!standard playfield 坐标为 `512 x 384`。
- 视频像素坐标通过 `package.OsuVideoTransform` 做居中等比缩放。
- Patch 标签先在完整视频坐标系生成，再按 `PatchMeta.x0/y0/x1/y1` 裁剪到局部。
- `traning.Lib.training.spatial_targets.build_spatial_loss_targets` 在模型输出的 feature grid
  上直接栅格化标签。右侧/底部 patch 的 padding 区域使用 `PatchMeta.padded_width` 和
  `PatchMeta.padded_height` 对齐，padding 单元保持 background。

## 输入颜色 Cue

`traning.Lib.data.color_cues` 可从 RGB 帧派生 3 个输入通道：有限色号响应、白色数字/内纹响应和
目标相关边缘响应。它们只作为模型输入先验，用来降低背景颜色干扰；不属于原始标注，也不作为
额外监督标签写入 `SpatialLossTargets`。

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

当前版 `SegmentFrameDataset` 会把 `difficulty.approach_preempt_ms` 和
`difficulty.circle_radius_osu_pixels` 写入每帧样本；若历史样本缺少 `approach_preempt_ms`，
target 构建器仅使用 `1000ms` 作为兼容默认值。

`ring_mask` 标记当前帧可见的 approach circle 圆环带宽；`ring_radius` 写入 feature-cell
单位的当前半径，仅在 `ring_mask` 正样本区域参与回归。

## Dense Target Shapes

每个 patch target 与 `SpatialPredictionHead` 输出一一对应：

- `center_heatmap`: `1 x 1 x H x W`，circle、slider head/tail 和 spinner 中心的高斯热图。
- `visible_heatmap`: `1 x 1 x H x W`，当前可见对象结构区域。
- `xy_offset`: `1 x 2 x H x W`，中心单元相对 cell center 的偏移，只在中心热图正样本回归。
- `object_type`: `1 x H x W`，类别表来自 `models.object_heads.OBJECT_TYPE_NAMES`。
- `ring_mask` / `ring_radius`: approach circle 圆环和当前半径。
- `slider_mask` / `slider_direction`: slider body tube 和无方向切线 `(cos 2θ, sin 2θ)`。
- `spinner_mask`: spinner 活跃区域。

## Slider

Slider 标签在全局视频坐标中生成：

1. 从原始 slider 控制点生成第一版中心线 polyline。
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
