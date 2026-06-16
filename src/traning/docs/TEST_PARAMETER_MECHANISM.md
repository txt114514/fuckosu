# 测试参数寻找、Score 与通过机制

本文说明当前代码实际执行的机制，并把尚未实现的训练方案单独标出。

## 1. 当前状态

| 能力 | 当前状态 |
|---|---|
| 参数按结构、训练、推理三层保存 | 已实现数据模型 |
| trial、课程阶段、rung、预算和 metrics 保存 | 已实现数据模型 |
| 批次中按 `score` 选择最佳 trial | 已实现 |
| 逐帧 `passed / failed` 图集 | 已实现 |
| 点与 slider 的单对象 score 和通过判定 | 已实现 `point-slider-v2` |
| 点击序列模拟、目标命中后失效和重叠目标递进 | 已实现基础 API |
| 点击频率上限 | 已实现配置，默认 `50ms` |
| 错误主责任与标签 | 已实现基础 API 与图集 JSON 字段 |
| TPE 参数生成 | 尚未实现执行器 |
| ASHA 晋级和剪枝 | 尚未实现执行器 |
| 多样本聚合为 trial `score` | 尚未实现 |
| 评估器自动写入逐帧 score 与 `passed` | 尚未实现 |
| 连续通过阈值自动晋级 | 尚未实现 |

因此，当前 `save-annotation-gallery` 是评估结果的消费者，不是评估器。底层已经可以
计算单个点和 slider 的 score，但尚未有评估执行器自动调用它、聚合 trial score 或写入
图集 JSON。

## 2. 当前参数选择机制

评估结果中的每个 trial 必须提供：

```json
{
  "trial_id": "trial_0042",
  "score": 0.91,
  "score_version": "external",
  "parameters": {
    "architecture": {},
    "training": {},
    "inference": {}
  },
  "metrics": {},
  "frames": []
}
```

三类参数的含义：

- `architecture`：通道数、层数、hidden size、stride、patch 规格等。改变后创建新 trial
  并从头训练。
- `training`：learning rate、weight decay、loss 权重、batch size、课程预算等。首版改变
  后也创建新 trial。
- `inference`：点击和松开阈值、cooldown、时间 offset、平滑、NMS 等。固定 checkpoint
  上快速搜索，不重复训练。

当前图集选择规则是：

```text
先按 score 从高到低
score 相同则按 trial_id 字典序从小到大
取第一名
```

等价于代码中的：

```python
min(trials, key=lambda trial: (-trial.score, trial.trial_id))
```

`score` 必须是有限浮点数。`score_version` 标识分数公式；缺省值为 `external`。
同一批次的所有 trial 必须使用相同版本，否则数据校验失败。`metrics` 当前只用于记录
和展示，不参与图集模块的排序。

## 3. 当前 Score 生成机制

当前已实现单个点和 slider 的 `point-slider-v2` 评分函数，详细公式见
[`SCORING_SPEC.md`](SCORING_SPEC.md)。它使用圆半径归一化空间误差、时间误差和
`空间 + 时间 + 空间 * 时间` 计算对象分数；slider 路径先膨胀为 `1.5x` 双向容差走廊
再计算覆盖率。

点击序列使用 `click-sequence-v1` 模拟：每个目标第一次合格命中后立即失效，后续点击
不能刷新该目标分数；重叠目标在已命中的目标消失后继续判定剩余目标；低于
`evaluation.min_click_interval_ms` 的高频点击标记为 `frequency_limited`，不参与命中。
序列结果会把错误归因到 `spatial`、`temporal` 或 `decision`：

- `spatial`：点击位置、slider head/path 或空间融合问题。
- `temporal`：点击偏早、偏晚或提前点击的时序问题。
- `decision`：重复点击、已命中目标后的更高分点击、错误候选选择或点击频率抑制问题。

尚未实现的是将一个 trial 的多个对象、六个子项目和资源指标聚合成最终标量。现阶段
`score` 仍由生成批次评估 JSON 的外部评估器直接写入，图集模块原样使用：

```text
模型输出
→ 外部评估器计算各项 metrics
→ 外部评估器合成一个越大越好的 score
→ 写入 BatchGalleryRequest
→ 图集选择最高 score
```

`ExperimentMetadata.objective_names` 目前预留了三个目标：

```text
quality_score
peak_vram_mb
latency_ms
```

但当前没有把对象分数和这些 trial 级目标合成为单一 `score` 的权重、宏平均、归一化或
资源惩罚公式。不同 `score_version` 的 score 不能直接比较。

计划中的 score 生成顺序是：

1. 使用 `point-slider-v2` 根据位置、时间和 slider 膨胀走廊覆盖生成对象结果。
2. 使用 `click-sequence-v1` 按模型点击流模拟目标消失、重叠递进和点击频率限制。
3. 按六个子项目统计通过率、最长连续通过、错误类型和质量指标。
4. 根据当前课程阶段合成 `quality_score`。
5. 单独记录峰值显存和延迟，不用训练 loss 直接替代实际评估质量。
6. 由固定版本的公式产生供 ASHA 和图集排序使用的标量 `score`。

trial 聚合权重目前尚未确定，因此代码不会擅自生成一个看似精确但不可复现的批次分数。

## 4. 当前通过机制

`point-slider-v2` 已能计算单个对象是否通过：

- 点：距离不超过 `1.0x`，且时间误差不超过 `150ms`。
- slider：起点满足点规则，且参考、预测路径分别膨胀 `1.5x` 后双向覆盖率均为 100%。

`click-sequence-v1` 在对象通过之上增加动作流语义：

- 目标合格命中后从有效目标集合移除，不能被后续更高分点击覆盖。
- 未合格命中的目标保持有效，后续点击仍可判定。
- 重叠目标按谱面时间和 `source_index` 稳定选择首个合格目标；该目标消失后，下一次有效
  点击再判定剩余目标。
- 默认最小点击间隔为 `50ms`，可通过 `evaluation.min_click_interval_ms` 调整。
- 每次点击可输出 `primary_error`、`error_tags`、`spatial_error`、`temporal_error_ms` 和
  `frequency_limited`，供空间、时间和决策参数网分别聚合。

`FrameEvaluation.passed` 仍是批次评估 JSON 的必填布尔值。当前图集模块：

- 不根据 `predicted_osu_xy` 重新计算通过；
- 不根据 `position_error` 等 metrics 修改通过状态；
- 只按已有 `passed` 值归入 `passed/` 或 `failed/`。

训练方案已确定的连续通过门槛如下，但执行器尚未实现：

| 评估项目 | 连续通过 | 最大失败 | 最大样本 |
|---|---:|---:|---:|
| `single_point` | 15 | 2 | 40 |
| `slider`，对应方案中的 `single_slider` | 10 | 2 | 35 |
| `multi_point` | 8 | 3 | 35 |
| `point_slider`，对应方案中的 `multi_slider` | 6 | 3 | 30 |

单个失败会把连续通过计数清零；总失败数未超过上限时仍可继续测试。达到连续通过要求且
失败数未超限，才算通过该子项目。

复杂阶段当前只定义了方案阈值：

```yaml
densify_score_threshold: 0.72
pass_score_threshold: 0.86
```

- score 低于 `0.72`：不向 TPE 反馈为高价值区域。
- score 达到 `0.72`：允许作为高价值观测影响后续采样。
- score 达到 `0.86`：允许进入最终候选集合。

`spinner` 和 `long_sequence` 目前没有独立的连续通过阈值。这两个项目可以进入图集，但在
正式评估器实现前还不能据此自动晋级。

## 5. 计划采用的参数寻找流程

当前训练方案采用以下组合，尚未接入实际执行器：

```text
随机参数作为冷启动
→ TPE 使用历史 trial 生成新参数
→ 低预算训练
→ 固定评估集测试
→ ASHA 在同一 rung 内晋级或剪枝
→ 同一 trial 继承 checkpoint 进入更难课程
→ 难例挖掘改变训练采样分布
→ 少量候选完整训练和模拟排名
```

课程阶段：

1. `basic`：单点、单划，约 `1,000～5,000 step`。
2. `multi_object`：多点、多划，累计约 `5,000～20,000 step`。
3. `complex`：重复点、点划重叠、复杂 slider、高密度场景。
4. `full`：完整训练、推理参数搜索和最终模拟评分。

同一 trial 晋级时继承模型、优化器、scheduler、AMP scaler 和全局 step。TPE 生成的新
参数组合是新 trial，必须从基础阶段开始，不能继承其他 trial 的 checkpoint。

## 6. 图集与测试机制的关系

图集固定支持六个子项目：

```text
single_point / slider / multi_point / point_slider / spinner / long_sequence
```

它只选择批次最高 score 的 trial，并从每个已评估子项目中随机抽取：

- 通过最多 10 帧；
- 不通过最多 10 帧；
- 数量不足时全部输出；
- 未进行的子项目不建立目录。

图集用于人工检查 score 和 passed 判定是否合理，不能替代固定评估集或自动晋级逻辑。
