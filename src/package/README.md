# Shared Package

`src/package` 是 `src` 下各顶层模块共同使用的稳定 API 层。

## 放置规则

- 已被两个或更多顶层模块调用的通用 API 放在这里。
- 子模块需要持续扩充独立功能或形成稳定数据/API 契约时迁入这里。
- 需要隔离第三方依赖、文件/数据库实现或管理跨层边界的能力迁入这里。
- 实现按领域拆分为子模块，不把无关能力集中到单个文件。
- 稳定名称必须在 `package/__init__.py` 中显式导出。
- 调用方从 `package` 的公开入口导入，不依赖内部私有名称。
- 原业务模块在迁移后只保留流程编排、配置映射和领域适配。
- 仅服务单个顶层模块且保持简单的局部实现保留在对应模块内部。
- 不因预计未来可能复用而提前建立全局抽象。

## 迁移判断

满足任一条件时，应考虑从子模块迁入 `src/package`：

- API 或数据模型需要被稳定维护和版本化
- 功能开始拥有多个同类操作、策略或后端实现
- 子模块依赖持续增长，需要隔离外部库或基础设施细节
- 调用方不应了解内部文件、数据库、网络或算法实现
- 功能边界已经能独立测试，并不再依赖原模块的业务状态

迁移时先定义公开契约，再移动实现；原路径如需兼容，只保留薄转发层，避免长期维护两份逻辑。

示例：

```python
from package import SharedType, shared_function
```

新增导出时同步维护 `__all__`，并避免在公开入口执行文件 I/O、网络访问或重量级初始化。

## Contracts

`package.contracts` 用于放置跨顶层模块共享的稳定数据契约。它只收纳需要长期维护、
版本化或写入文件/manifest/artifact/cache 的结构体，不收纳单个模块内部的训练中间态。

当前契约分组：

- `geometry`：`Point2D`、`Size2D`、`Rect2D` 和 `CoordinateSpace`。
- `osu`：`OsuHitObject`、`OsuTimingPoint`、`OsuDifficulty` 和谱面对象类型。
- `dataset`：`TrainingItemRef`、`SegmentRef`、`FrameSampleRef` 和数据 split/category/dimension。
- `candidate`：候选缓存帧、空间候选、slider path 候选、时序目标和决策帧记录。
- `experiment`：trial、checkpoint、score version、搜索方法、课程阶段和 trial 状态。
- `evaluation`：`FrameRef`、`PredictionEvent`、`ScoreSummary`、`EvaluationOutcome`、动作和错误域枚举。
- `artifacts`：`ArtifactFileRef` 和 `VersionedArtifactRef`。

每个 contract 都提供 `as_dict()`，支持需要时从 mapping 创建；调用方仍应优先从
`package` 或 `package.contracts` 的公开入口导入。
