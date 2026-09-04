# `traning.state` 类型注册

## 设计原则

`traning.state` 是训练/推理专用数据类型的权威公开面。`TYPE_REGISTRY` 以稳定名称映射到真实 type 对象；它不是字符串清单，也不会为旧路径复制 class。旧 `traning.contracts.*` 只 re-export，因此 `old.Type is canonical.Type`。

跨 `src` 顶层模块共享的几何类型仍由 `package.contracts.geometry` 定义。`traning.state.geometry` 直接复用 `Point2D`、`Size2D` 和矩形契约，并提供规范名称 `Box2D`；不会再定义第二个同义点类。训练专用 batch、prediction、tracking、belief、outcome、decision、loss、telemetry 和 environment 类型留在 state。

## 模块职责

| 模块 | 权威内容 |
|---|---|
| `common.py` | JSON aliases、稳定 id/hash/fingerprint 基础契约 |
| `geometry.py` | 复用 package 几何、resize/circle 公开契约 |
| `data.py` | `VideoFrame`、样本与 batch 契约 |
| `observation.py` | `ObjectType`、`Candidate` 与兼容 observation 名称 |
| `perception.py` | `SpatialPrediction`、`CandidateBatch` |
| `tracking.py` | `TrackState` 与跟踪状态公开契约 |
| `belief.py` | `BeliefState` |
| `outcome.py` | `OutcomePrediction` |
| `decision.py` | `ActionType`、`ActionPrediction` |
| `training.py` | `LossBreakdown` 与训练 batch/loss 协议 |
| `telemetry.py` | `MemoryReport` 与稳定遥测 DTO |
| `environment.py` | `PackageCheck`、`TorchCheck`、唯一 `EnvironmentReport` |
| `registry.py` | `TYPE_REGISTRY` 和注册表一致性入口 |

## 规范名与旧名

旧名仅作为 identity alias：`RuntimeFrame → VideoFrame`、`CandidateObservation → Candidate`、`TrackLifecycle → TrackState`、`OutcomeDistribution → OutcomePrediction`、`DecisionAction → ActionType`、`DecisionResult → ActionPrediction`、`MemorySnapshot → MemoryReport`。这些 alias 兼容已有调用方，但新代码和新文档使用左侧箭头右边的规范名。

不同领域的 loss 分解、manifest 和坐标点不会因字段相似被错误合并。它们可以出现在扩展注册表中，但硬标准名称只指向单一权威对象。架构测试同时扫描所要求名称的 class 定义，并验证 registry value 与 wrapper export 的对象 identity。
