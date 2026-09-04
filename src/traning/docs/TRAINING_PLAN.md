# traning 当前训练目标与实现状态

本文描述 `src/traning` 的唯一活动实现。V2 已整体迁入并覆盖旧训练包；当前
`traning.core/conf/state/lib` 是权威路径，旧扁平包只作 deprecated wrapper。不存在需要
继续维护的 `src/osu_v2`、外部 evaluator 或独立旧可视化实现。源码定位以
自动生成的 [`CODEX_INDEX.md`](CODEX_INDEX.md) 为准，运行环境约束见
[`ENVIRONMENT.md`](ENVIRONMENT.md)，用户入口见 [`../README.md`](../README.md)。

## 模型目标

正式因果链路固定为：

```text
RuntimeFrame
  -> Perception: CandidateObservation[]
  -> Tracking: TrackedObservation[]
  -> Belief: BeliefState[]
  -> Outcome: OutcomeDistribution[]
  -> Decision: CLICK_NOW | WAIT_ONE_STEP
```

最终动作来自 learned outcome 分布与 deterministic optimal-stopping utility。正式 runtime
不得读取 GT hit objects、训练 target、oracle label、action imitation logits 或 candidate
slot logits；改变未来帧不得改变当前 belief。

## 活动目录分层

```text
src/start/main.py                 # 唯一仓库总启动入口
  -> start/flow.py                # raw → before → split → checks → train → report
  -> traning/
       main.py / __main__.py      # 模型 CLI 公开入口
       conf/                      # 唯一严格配置与版本
       state/                     # typed contracts 与统一 TYPE_REGISTRY
       core/app/                  # 正式 runtime factory 与模型 CLI 编排
       core/data/                 # typed dataset、cache、quality、repository、坐标证据
       core/perception/           # 图像感知、decode、target、loss、runtime
       core/tracking/             # association、track_id、轨迹生命周期
       core/belief/               # 逐 track 因果状态及训练
       core/outcome/              # oracle、反事实 dataset、模型、校准和训练
       core/decision/             # CLICK/WAIT 风险调整效用与 planner
       core/evaluation/           # canonical scoring、sequence、attribution
       core/training/             # 六阶段门禁、参数搜索、恢复、checkpoint
       lib/environment/           # Python/依赖/Torch/CUDA 权威检查
       lib/infrastructure/        # 原子持久化、确定性与基础错误
       lib/telemetry/             # Reporter → Store，四类 JSONL
       lib/visualization/         # 只读 renderer 与 gallery overlay/PNG 原语
       lib/runtime/               # CUDA/AMP/channels-last/显存统一入口
       lib/validation/            # primitive、Tensor 与单次边界检查
```

## Phase 0 → Phase 11 状态

| Phase | 状态 | 当前落点 |
|---|---|---|
| 0 | 完成 | `legacy/legacy_freeze.json` 与 regression golden baseline |
| 1 | 完成 | `state`、`conf`、`lib/infrastructure` |
| 2 | 完成 | `core/data` typed pipeline、cache 完整性、quality gate、repositories |
| 3 | 完成 | `core/perception`，推理只产生 `CandidateObservation` |
| 4 | 完成 | `core/tracking`，slot 重排不改变 track identity |
| 5 | 完成 | `core/belief`，逐 track 因果状态与 future-frame 隔离 |
| 6 | 完成 | `core/outcome/oracle`、反事实 dataset、canonical attribution |
| 7 | 完成 | OutcomeModel、NLL/Brier/ECE/expected-score MAE |
| 8 | 完成 | `core/decision` CLICK/WAIT optimal stopping |
| 9 | 完成 | `core/training` curriculum/ASHA、强恢复状态及 TRAIN hard-example 反馈闭环 |
| 10 | 完成 | `lib/telemetry` JSONL 与只读 renderer |
| 11 | 完成 | `core/app`、checkpoint、坐标证据、唯一 CLI/runtime，旧接口 wrapper 化 |

## 生产训练闭环

`core/data/segments.py` 从 canonical split manifest 构建 `TrainingDatasetBundle`。数据质量只由
`DataQualityIssue.blocks_training` 决定；start 检查与训练复用同一份报告，blocking issue
在任何 runner 执行前终止。

`core/training/production_stages.py` 对每个参数提案依次执行：

```text
Perception -> Tracking -> Belief -> Outcome -> Decision -> Evaluation
```

参数通过统一规格映射到完整训练配置，不允许只调整渲染或某个旁路组件。每个完成的
curriculum/rung job 都先事务发布可恢复 checkpoint，再提交调度状态；只有 FULL 末级 rung
全门禁通过的唯一 trial 才能成为最终结果。checkpoint 同时绑定 dataset identity、模型契约
摘要、权重 SHA-256、完整训练配置摘要和坐标指纹，并在返回成功前重新加载验证。

Evaluation 以事件真实 `frame_index` 生成 production PNG，并在所有图片成功写入后最后原子
提交 `gallery/manifest.json`。`passed/failed` 和 Spatial/Temporal/Decision 目录只由当前帧
canonical events 决定；`long_sequence` 等名称仍是数据维度/样本身份，不参与错误归因。

`core/training/optimization.py` 只提出合法、量化且未执行过的参数组合。普通门禁失败会形成下一轮
观测并继续搜索，不会让程序静默停止。搜索只有三类终态：

1. 全部门禁通过：发布并验证 checkpoint，终态 `PASSED`。
2. 显式正整数预算或有限参数空间耗尽：抛出 `SearchExhaustedError`，终态 `EXHAUSTED`。
3. blocking 数据问题、设备/训练/制品异常：立即失败；这类问题无法通过换参数修复。

`optimization.max_trials: null` 表示不附加人为 trial 截断，不表示伪造无限参数空间。
`core/training/production_schedule.py` 将 proposal、cohort、curriculum stage、ASHA rung、累计预算、
父 checkpoint、反馈制品和最终 history 一起绑定到 run、dataset 与 config identity；状态和外部
制品均带 SHA-256，恢复时先完整校验，避免跨数据/配置续跑或静默重训已提交 job。

Curriculum 现在真正选择累计数据视图：BASIC 使用 atomic single-point/slider，
MULTI_OBJECT 增加 multi-point，COMPLEX 覆盖全部 atomic，FULL 再加入 long-sequence。
ASHA 只比较同 cohort/stage/rung，严格 gate 优先，再按稳定 objective 排名；下一 rung 从配置
读取累计预算，runner 只执行相对父 checkpoint 的新增步数。只有 FULL 末级的唯一 CONTINUE
会令 `TrialAcceptance.schedule` 通过。

Evaluation 将 TRAIN 与 validation 的 canonical events 生成原子 feedback artifact，但只有
TRAIN 失败事件可形成 `(sequence_id, frame_index, destination)` 权重；validation/test 只保留
排除审计，绝不进入训练。有效权重统一为
`min(max_weight, 1 + bonus * sum(route.weight))`：Perception 用确定性加权采样，Outcome 用
归一化加权 loss，Decision 用当前课程中确实存在的难例做加权 gate。新 cohort 从 BASIC
重启时会暂时忽略仅 FULL 可见的帧，避免把“当前课程没有该帧”误判成 Decision 0 分。

## 坐标与分类约束

训练 target、perception 输出解释、oracle/scoring、evaluation 和 gallery overlay 必须共享
`FrameCoordinateTransform`。方程方向、标定身份、原帧尺寸和完整 affine 矩阵共同进入
transform fingerprint；不允许为某张图或某个 renderer 单独增加偏移。

目标中心/head 必须位于 osu playfield；slider 后续控制点用于表达曲线方向，可以合法越出
边界，不能因此把一个实际命中准确的样本误归因到空间模块。错误模块由
`core/evaluation/attribution.py` 的 canonical 最早失败边界统一判定。

## CUDA 与数据加载约束

- 训练代码复用 `traning.lib.runtime` 的 `configure_torch_runtime`、
  `module_to_device`、`tensor_to_device`、`autocast_context`、
  `create_grad_scaler` 和 `collect_memory_snapshot`。
- CUDA step 使用 `optimizer.zero_grad(set_to_none=True)`、AMP、必要时 GradScaler、
  channels-last、TF32/cuDNN benchmark、pinned memory 和 non-blocking copy。
- 不在 step 中保留无用 GPU Tensor 列表，不频繁调用 `torch.cuda.empty_cache()`。
- Dataset 无 UI side effect，model forward 无磁盘 IO，repository 不泄漏裸 SQLite row。

## 唯一运行入口

完整生命周期：

```bash
PYTHONPATH=src python src/start/main.py
PYTHONPATH=src python -m start run --config configs/traning.yaml
```

模型领域诊断或直接训练：

```bash
PYTHONPATH=src python -m traning config-check --config configs/traning.yaml
PYTHONPATH=src python -m traning env-check --config configs/traning.yaml
PYTHONPATH=src python -m traning coordinate-audit --config configs/traning.yaml
PYTHONPATH=src python -m traning train --config configs/traning.yaml
```

上述命令不接受外部 evaluator；生产数据、阶段 runner 和 checkpoint 行为全部由仓库内 typed
实现装配。

## 维护要求

修改 `src/traning/**/*.py` 后运行：

```bash
python project_index/build_index.py
python project_index/build_index.py --check
```

`CODEX_INDEX.md` 是生成文件，不得手工编辑。旧六阶段 plan、readiness 和 gap audit 仅保留为
迁移前历史记录，不可作为活动路径或命令依据。
