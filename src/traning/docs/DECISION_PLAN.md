# Decision Plan

## 模块定位

源码入口：`src/traning/core/decision`

决策模块连接空间和时间输出，当前实现重点是候选缓存、时序监督和逐帧决策导出。训练结果
评分、错误归因和参数搜索修改已下沉到 `core/optimization`。

## 候选缓存

当前版本：`spatial-candidate-cache-v1`

离线流程：

```text
视频帧
  -> 空间单帧推理
  -> CPU 全图融合
  -> 候选和 slider path 解码
  -> frames.jsonl + manifest.json
```

每帧记录包含：

- 点候选坐标、类型、score 和 embedding；
- slider polyline、continuity 和歧义标记；
- 候选与 slider path 的关联；
- 低置信、近分、路径异常等 `ambiguity_reasons`。
- `coordinate_transform`、dataset/config/code/score/cache/transform 版本信息。
- 可选 `local_refinement` 和 `ambiguity_review`，记录复查策略、触发原因和前后变化。

这些记录是 temporal 训练和后续在线决策的共同输入。缓存记录还包含可选
`temporal_target`，当前版本为 `beatmap_action_v1`，用于把谱面动作时间窗、候选选择和
目标坐标传给时间模块。

## 单对象评分

点与 slider 使用 `point-slider-v2`：

- 空间距离用 `circle_radius_osu_pixels` 归一化。
- 点通过条件：距离不超过 `1.0x`，时间误差不超过 `150ms`。
- `r < 0.6` 获得最多 `0.05` 中心奖励，`1.0 < r < 1.5` 只保留安慰分。
- 时间误差 `0-20ms` 为 `1.05`，`20-150ms` 分段下降，`>150ms` 进入安慰区，`>=200ms` 为 0。
- 原始分数：`raw = spatial + temporal + spatial * temporal`。

Slider 额外要求：

- 起点满足点规则。
- 参考和预测路径分别膨胀为 `1.5x` 容差走廊。
- 双向覆盖率均为 100%，且最大距离不超过 `1.5x`。
- 总空间系数取 head 和 path 的最弱环节。

## 点击序列模拟

序列模拟版本：`click-sequence-v1`

规则：

- 目标首次合格命中后立即从有效目标集合移除。
- 未合格命中的目标保持有效，后续点击仍可命中。
- 重叠目标按谱面时间、`source_index` 和目标 ID 稳定选择首个合格目标。
- 低于 `evaluation.min_click_interval_ms` 的点击标记为 `frequency_limited`，不参与命中。
- slider 起点和路径都通过后，该 slider 失效，后续预测不再计分。

错误主责任：

| 主责任 | 含义 |
|---|---|
| `spatial` | 位置、路径或空间融合错误 |
| `temporal` | 点击时间偏早、偏晚或提前点击 |
| `decision` | 是否点击、选哪个、重复点击或冷却抑制 |
| `none` | 已命中或无可归因错误 |

## 参数搜索边界

参数分三层管理：

| 层级 | 示例 | 是否重新训练 |
|---|---|---|
| 结构参数 | 通道数、层数、hidden size、stride、patch 大小 | 是 |
| 训练参数 | learning rate、weight decay、loss 权重、课程预算 | 通常是 |
| 推理参数 | press/release 阈值、cooldown、NMS、平滑 | 否 |

完整计划流程：

```text
随机冷启动
  -> TPE 生成新参数
  -> 低预算训练
  -> 固定评估集测试
  -> ASHA 晋级或剪枝
  -> 同一 trial 继承 checkpoint 晋级课程
  -> 难例挖掘调整训练采样
  -> 少量候选完整训练和最终模拟排名
```

课程阶段：

1. `basic`：单点、单划。
2. `multi_object`：多点、多划。
3. `complex`：重复点、点划重叠、复杂 slider、高密度场景。
4. `full`：完整训练、推理参数搜索和最终模拟评分。

## SMET 计划

SMET 属于动态稀疏训练，适合候选 embedding 投影、大型 MLP、GRU/LSTM 大矩阵、
Transformer、候选交互层和动作专家。它节省参数、梯度和优化器状态，但不直接解决
高分辨率激活图显存。

首版不实现 SMET 动态拓扑，只保留配置边界和后续接入位置。

## 当前状态

- 候选缓存生成器已实现。
- `run-decision` 已能加载 `train-temporal` 产出的 `temporal_model.pt`，消费候选缓存并导出
  `decisions.jsonl`。
- 单对象评分和序列模拟底层 API 位于 `traning.lib.metrics`。
- trial 级评分、三类错误归因、ASHA/TPE 参数调整计划、连续通过 gate、难例采样权重和 trial 记录执行器位于 `core/optimization`。
- `core.decision.pipeline.TRAINING_STAGES` 当前真实登记
  `data_input / spatial / candidate_cache / temporal / decision / evaluation`。
  这是单轮训练流水线阶段；完整生命周期阶段另由
  `core.full_flow.stages.FULL_FLOW_STAGES` 登记。
- 完整训练结束后可由 `settings.optimization.enabled` 触发 `analyze_trial_attribution`、
  `plan_next_trial` 和 `execute_optimization_plan`，写出 trial JSONL、`attribution.json`、
  `optimization_plan.json` 和 `next_training_job.json`。
- `python -m traning.main run-job --job <next_training_job.json>` 可消费 job JSON，并调用普通
  `run_training_job_spec` / `run_training` 业务函数执行，不复制训练逻辑。

## 后续计划

- 将 optimization JSONL 记录升级为 SQLite 或实验数据库。
- 将 `quality_score`、`peak_vram_mb`、`latency_ms` 拆成独立目标，再定义可复现排序公式。
