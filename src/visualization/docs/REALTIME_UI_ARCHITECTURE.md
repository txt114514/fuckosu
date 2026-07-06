# 实时 UI 架构

本文记录可视化模块的实时状态链路和双 UI 切换规则。

## 状态来源

UI 不推测训练状态，也不扫描日志生成状态。真实状态只从训练流程通过
`visualization.lib.TrainingReporter` 推送：

```text
traning full-flow / ramp / pipeline
  -> update_pipeline_stage / update_metrics / report_score / register_checkpoint
  -> DashboardReporter
  -> TrainingDashboardState
  -> DashboardStateStore
  -> Rich/Plain renderer
```

`DashboardReporter` 会把当前 trial 聚合为 `current_parameter_status`，并写出
`current_parameter_status.json`。该状态包含：

- 当前 trial id 和真实参数表；
- pipeline phase；
- curriculum stage、stage index、训练 step 和预算；
- 当前分数、当前参数最好分、全局最好分；
- 连续通过次数和晋级要求；
- pending/running/passed/failed/warning 测试列表；
- 每项测试最新 score 和 threshold；
- checkpoint、trial_status、prune_reason 和停止原因。

没有 active trial 时显示 `No active trial` / `N/A`，不生成示例参数或固定分数。

## Pipeline Phase

`TrainingDashboardState.pipeline_phase` 是 UI 切换的唯一依据：

```text
startup
data_preparation
pretrain_check
progressive_preparation
training
completed
failed
```

full-flow 使用真实 stage id 映射 phase：

- 数据准备阶段：`SOURCE_CHANGE_CHECK`、`BEFORE_TRAINING`、`DATASET_CONVERSION`、
  `SPLIT_VALIDATION`。
- 预训练检查阶段：`DATA_QUALITY_CHECK`、`ENVIRONMENT_PREFLIGHT`、
  `RESUME_DISCOVERY`、`RESUME_RESTORE`。
- `RAMP_TRAINING` 开始时切换到 `training`，这是 UI A 到 UI B 的真实切换条件。
- `REPORT_GENERATION` 通过后进入 `completed`；失败或中断进入 `failed`。

ramp 独立运行时，preflight 使用 `pretrain_check`，ramp 准备使用
`progressive_preparation`，Level 开始训练时切换到 `training`。

## UI A：启动检查与渐进训练准备

`pipeline_phase` 属于以下值时渲染 UI A：

```text
startup
data_preparation
pretrain_check
progressive_preparation
```

UI A 只显示启动、数据准备、预检、资源和事件状态，不展示正式训练参数面板。

## UI B：正式训练实时 UI

`pipeline_phase` 进入 `training`、`completed` 或 `failed` 后渲染 UI B。

UI B 显示：

- Current Trial；
- Parameters；
- Tests；
- 最佳参数记录；
- 当前参数学习及评分；
- 总体训练进度；
- 本机资源；
- 事件与警告。

Rich renderer 仍只维护一个 `Live` 实例。phase 改变后，下一次 render 从
`view_router.dashboard_view_kind(...)` 选择新的 view，不会启动第二个终端 UI。

## 训练状态接入点

- full-flow：`traning.core.full_flow.orchestrator` 在真实 stage start/finish 时更新
  `pipeline_phase` 和 `PipelineStageState`。
- ramp：`traning.core.training_ramp` 在 preflight、ramp 准备、Level 开始、gate 通过/失败、
  正式训练开始/结束时更新 phase、trial_status、score 和 checkpoint。
- 单轮 pipeline：`traning.core.decision.pipeline` 在 trial 开始、spatial/temporal step、
  evaluation start、score、ASHA prune/promote/continue 和 checkpoint 时更新状态。

所有 UI 数据都来自这些上报点。测试 fixture 只能出现在 `tests/` 中，生产路径不提供
dummy trial、demo score 或固定 passed tests。
