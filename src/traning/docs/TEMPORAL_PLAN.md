# Temporal Plan

## 模块定位

源码入口：`src/traning/core/temporal`

时间模块负责把空间模块产生的每帧候选序列转换为因果动作序列。它不直接读视频，也不做
patch 前向；输入应来自候选缓存或在线空间推理结果。

## 目标

输出当前帧动作：

```text
0: no-op
1: press
2: hold
3: release
```

并在有效动作时输出候选选择、坐标和时间偏移。模型接口必须是流式因果：

```python
output, state = model.step(frame_candidates, state)
```

固定历史输入时，加入未来帧不能改变过去输出。

## 输入契约

推荐训练输入：

```text
[T, K, D]
```

- `T`：时间长度，建议 `128-512`。
- `K`：每帧最大候选数，建议 `16-32`。
- `D`：候选特征维度，建议 `64-128`。

每帧候选来自 `decision` 模块的 `spatial-candidate-cache-v1`，包含：

- 坐标、score、object type；
- embedding；
- slider path id；
- ambiguity reasons；
- frame index 和 timestamp。

## 动作标签

逐帧动作标签从对象时间窗派生：

- `no_op`：当前帧没有需要执行或保持的对象。
- `press`：circle hit time 或 slider head/spinner start 附近首次按下。
- `hold`：slider/spinner 活跃期间持续按下。
- `release`：slider/spinner 结束后的释放帧。

普通点：

```text
no-op -> press -> release -> no-op
```

Slider：

```text
no-op -> press -> hold -> hold -> release
```

重复点：

```text
press -> release -> press -> release
```

坐标损失只在有效动作时计算。

## 首版模型

底层已有 `traning.lib.models.CausalTemporalModel`，首版建议：

```yaml
type: GRU
hidden_size: 128-256
layers: 1-2
```

后续可升级为 causal Transformer、state-space model 或候选间 attention，但必须保持因果。

## 训练计划

1. 从候选缓存读取固定长度窗口，构建 `[T, K, D]`。
2. 候选缓存生成阶段写入 `beatmap_action_v1`：根据 `beatmap.json` 的 hit object 时间窗派生
   `no_op / press / hold / release`，并用 `OsuVideoTransform` 把 osu 坐标转换到视频坐标，
   选择最近候选作为监督。
3. 旧缓存没有 `temporal_target` 时 fallback 到 `top_candidate_proxy`：有候选学
   `press + top candidate`，无候选学 `no_op`。
4. 训练输出 `temporal_model.pt`，供决策导出加载。
5. 先训练基础 `single_point / slider`，再晋级 `multi_point / point_slider`。
6. 引入 `spinner / long_sequence` 后记录长时依赖和 release 稳定性。
7. 用 `click-sequence-v1` 评估动作流，输出空间、时间和决策错误归因。

## 当前状态

- `TemporalCandidateWindowDataset` 可读取 `spatial-candidate-cache-v1` 的 manifest/JSONL，
  按 `sample_key` 分组生成固定长度窗口，保留 frame mask、candidate mask、candidate id、
  padding 和动作监督。
- `run_temporal_training` 已提供 CPU/CUDA 统一 runtime 的首版训练入口，输出
  `summary.json` 和 `temporal_model.pt`，并通过 `train-temporal --cache <candidate-cache-dir>` 接入 CLI。
- `run-decision --cache <candidate-cache-dir> --checkpoint <temporal_model.pt>` 已能导出
  `decisions.jsonl` 和 manifest，包含动作、动作概率、候选选择和预测时间偏移。
- `CausalTemporalModel`、`initial_state` 和 `step` 流式接口已有基础测试。
- trial 级评估聚合位于 `core/optimization/scoring`，复用 `click-sequence-v1` 并输出稳定 score 版本。

## 后续计划

- 改进 `beatmap_action_v1` 的边界帧策略，尤其是 circle release、slider repeat、spinner
  长按和极密集重复点。
- 明确正式候选选择 loss、动作分类 loss、坐标 loss、时间 offset loss 的权重。
- 把 `evaluation.min_click_interval_ms` 纳入时序评估和决策错误归因。
- 增加因果一致性测试：未来窗口变化不能影响过去输出。
