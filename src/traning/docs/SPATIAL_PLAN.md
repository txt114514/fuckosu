# Spatial Plan

## 模块定位

源码入口：`src/traning/core/spatial`

空间模块负责在单帧内发现当前可见目标，输出候选点、目标类型、slider 路径和紧凑 embedding。
它不展开长时序，也不决定是否点击；这些由时间和决策模块消费。

## 设计目标

- 保留原始视频像素，避免整体缩放损失圆点边缘、数字、slider 路径和重叠关系。
- 在 8GB VRAM 上通过重叠 patch 串行训练控制显存峰值。
- 对跨 patch 的 circle、approach ring、slider 和 spinner 保持全局一致性。
- 输出足够小的候选缓存，供时序模型训练和推理使用。

## Patch 策略

推荐默认：

```yaml
patch_width: 512
patch_height: 512
overlap_x: 96-128
overlap_y: 96-128
patch_batch_size: 1
backward_per_patch: true
```

训练时每个 patch 独立前向和 backward，loss 按 patch 数归一：

```text
optimizer.zero_grad(set_to_none=True)
for patch in patches:
  forward -> loss / patch_count -> backward
optimizer.step()
```

不允许先保存所有 patch 的计算图再统一 backward。

## 模型结构

首版空间模型由 `traning.lib.models` 组合：

- `SmallLocalEncoder`：高分辨率局部 CNN，使用轻量残差和 stride-8 pyramid。
- `LightweightGlobalEncoder`：低分辨率全帧 encoder，提供全局上下文。
- `GlobalStructureHead`：全局对象性、中心、圆环、slider、spinner 和 context token。
- `GatedSparseFusion`：把全局上下文门控注入 patch 局部特征，并做稀疏跨区域采样。
- `SpatialPredictionHead`：输出 center、visible、type、ring、slider、spinner 和 embedding。

全局分支可冻结，局部分支、融合和空间头逐 patch 反传。

## 输入 Cue 和标签

RGB 帧可附加 3 个确定性 cue：

- osu 色号响应；
- 白色数字/内纹响应；
- 目标相关边缘响应。

这些 cue 只是输入先验，不是原始标注。dense target 由
`build_spatial_loss_targets` 在 feature grid 上生成：

- `center_heatmap`
- `visible_heatmap`
- `xy_offset`
- `object_type`
- `ring_mask` / `ring_radius`
- `slider_mask` / `slider_direction`
- `spinner_mask`

Approach circle 半径使用 difficulty 派生的 `approach_preempt_ms` 和
`circle_radius_osu_pixels` 近似；后续若接入更精确渲染规则，需要保留标签版本号。

## Slider 跨 Patch 约束

支持范围：

- 单条 slider 跨多个 patch；
- 多条互不接触、中心线连通域独立的 slider；
- slider head 和几何尾端位于不同 patch。

暂不支持：

- 两条 slider 中心线交叉；
- 两条 slider 路径接触后形成同一连通域；
- 分叉点上的实例归属判断。

方向场使用无方向切线 `(cos 2θ, sin 2θ)`，避免相邻 patch 对同一中心线给出相反方向时
互相抵消。

## 推理与融合

单帧推理入口是 `run_spatial_frame_inference`：

```text
RGB 归一化和 cue
  -> PatchStream
  -> global/local/fusion/head 前向
  -> detach 到 CPU
  -> SpatialPredictionCanvas 全图融合
  -> Top-K / 局部最大值 / NMS
  -> slider 连通域和 polyline 恢复
```

CPU 侧负责画布融合、候选解码、slider 连通域、端点、branch/continuity 标记和 JSON 输出。
GPU 侧只保留必要的模型前向和 tensor copy。

## 当前实现

- `train-spatial` 已接入首版单帧空间训练。
- `spatial-decode-smoke` 复用单帧推理入口并输出候选 JSON。
- `decode_spatial_candidates` 已提供 Top-K、局部最大值和半径 NMS。
- `decode_slider_paths` 已提供连通域、端点、歧义标记和固定长度 polyline。
- `build-candidate-cache` 已复用空间推理逐帧生成候选缓存。

## 后续计划

- 增加候选局部 refiner，仅对少量候选做 stride-1 或高分辨率复查。
- 增加条件歧义复查，用于低置信、近分候选和 slider head/path 冲突。
- 强化跨 patch embedding 一致性、ring 半径一致性和 slider continuity loss。
- 给 dense target、slider 支持范围和评分版本建立显式版本字段。

