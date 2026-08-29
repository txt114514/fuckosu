# OSU Decision Model

`src/traning` 是唯一活动训练与推理包。V2 已整体迁入并覆盖旧实现；仓库不保留
`src/osu_v2`、旧 `traning.core/conf/state`、外部 evaluator 或独立
`src/visualization` 兼容层。

## 正式链路

```text
RuntimeFrame
  → Perception / CandidateObservation[]
  → Tracking / TrackedObservation[]
  → Per-track BeliefState[]
  → OutcomeDistribution[]
  → CLICK_NOW | WAIT_ONE_STEP
```

最终动作来自 learned outcome 分布和 deterministic optimal-stopping utility。runtime
contract 不包含 GT hit objects、oracle label、训练 target、action logits 或 GT candidate
slot，因此不能通过旁路把真值泄漏进推理。

## 唯一配置和入口

生产配置是 [`configs/traning.yaml`](../../configs/traning.yaml)。加载器拒绝未知键、错误
schema、字符串化数值和跨领域不一致；不会迁移旧配置或在 CUDA 不可用时静默退回 CPU。

```bash
# 仓库完整生命周期
PYTHONPATH=src python src/start/main.py run --config configs/traning.yaml --resume

# 模型侧诊断
PYTHONPATH=src python -m traning.app config-check --config configs/traning.yaml
PYTHONPATH=src python -m traning.app coordinate-audit --config configs/traning.yaml
PYTHONPATH=src python -m traning.app env-check --config configs/traning.yaml --strict

# 直接运行真实 typed dataset 的生产训练
PYTHONPATH=src python -m traning.app train --config configs/traning.yaml --resume
```

训练命令不接受 `module:factory` evaluator。`data/segments.py` 构建 typed dataset bundle，
`training/production_stages.py` 依次运行 Perception、Tracking、Belief、Outcome、Decision、
Evaluation，`training/production.py` 负责参数搜索、恢复和 checkpoint 复验。

## 搜索为何会继续或停止

默认配置为：

```yaml
optimization:
  max_trials: null
```

普通门禁失败会原子提交 observation，然后选择下一组合法、量化且未重复的参数继续执行。
搜索只在以下边界停止：

1. 七个 acceptance gate 全部通过，winning checkpoint 完整复验后返回 `PASSED`；
2. 用户显式设置的正整数预算或有限量化空间耗尽，抛 `SearchExhaustedError`；
3. blocking 数据质量问题、设备错误、损坏制品或训练异常立即失败；
4. 用户/系统中断。

每个 run 的 `search_state.json` 同时绑定 run ID、dataset identity 和完整训练配置摘要；
`--resume` 不会重复已提交 proposal，也不会把另一份数据或配置的历史接进来。

## 坐标和错误分类

训练 target、perception decode、canonical scorer、evaluation event 和 gallery overlay /
production PNG writer 共同消费 `FrameCoordinateTransform`。当前 1484×846 标定为：

```text
video_x = 2.115860914627143 * osu_x
        + 0.0011971920855575358 * osu_y
        + 242.59057485632047

video_y = 0.0003418231662923798 * osu_x
        + 2.1166805757239477 * osu_y
        + 16.12108357719331
```

因此 osu `(79.89, 101.22)` 映射为原帧约 `(411.75, 230.40)`。矩阵、标定身份或原帧尺寸
任一改变都会改变 transform fingerprint；缓存、batch、event 和 checkpoint 指纹不一致时
必须拒绝复用。证据见
[`configs/traning_coordinate_evidence.json`](../../configs/traning_coordinate_evidence.json)。

slider 的 head/中心必须在 playfield 内，但后续 Bézier/Catmull 控制点可合法越界。训练方向、
sequence scoring 和 gallery 都通过共享 affine 几何投影保留控制点，不裁剪、不单独补偏移。

`passed/failed` 由 canonical click event 决定。图片上的 GT/candidate 覆盖层不是实际点击；
`no_op` 留下的未解析目标固定归为 `Decision + unresolved_target`。`long_sequence` 是数据维度，
不是错误模块；事件保留真实来源帧，不能再用整段序列的 AND 结果反写每张图。

## Checkpoint 与遥测

checkpoint 采用 generation-first / manifest-last 原子发布，加载时校验：

- dataset identity；
- 完整配置摘要；
- Perception、Belief、Outcome 模型契约摘要；
- 权重 SHA-256 与 strict state dict；
- 坐标变换指纹。

telemetry 固定为 `metrics.jsonl`、`resources.jsonl`、`evaluation.jsonl` 和
`events.jsonl`。数据流是 Reporter → StateStore → Renderer；renderer 只读 canonical
snapshot/event，不重新计算质量门、passed 或 primary error。

到达 Evaluation 的 trial 会在同 trial 目录发布 `gallery/manifest.json`。manifest-last
提交记录每张 PNG 的 SHA-256、真实 `frame_index`、canonical event IDs、错误域、原帧尺寸
与 transform fingerprint；目录分类只看当前帧事件，不再沿用 legacy 的 sequence 级 AND。

## 导航与验证

- [训练目标与阶段](docs/TRAINING_PLAN.md)
- [生成源码索引](docs/CODEX_INDEX.md)
- [CUDA 环境](docs/ENVIRONMENT.md)
- [旧 frame 105 与坐标偏移诊断](docs/LEGACY_FAILURE_DIAGNOSIS.md)
- [迁移状态](../../V2_MIGRATION_STATUS.md)

```bash
PYTHONPATH=src:. python -m pytest -q src/traning/tests
python project_index/build_index.py
python project_index/build_index.py --check
```
