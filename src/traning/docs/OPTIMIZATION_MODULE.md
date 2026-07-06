# Optimization Module

源码入口：`src/traning/core/optimization`

该模块负责训练结果闭环：评分、错误归因和下一轮参数调整。它不直接训练模型、不读视频、不调用
其他 `core` 模块；只消费已经准备好的目标对象、预测点击、运行指标和历史 trial 摘要。

## 目录结构

```text
core/optimization/
  scoring/
    evaluator.py          # trial/sample 级评分聚合
    run_outputs.py        # 从候选缓存和 decisions JSONL 构建固定评估集评分输入
    gallery.py            # 从评分结果生成 BatchGalleryRequest
  attribution/
    analyzer.py           # 空间/时间/决策三类错误归因
  parameter_search/
    planner.py            # ASHA/TPE/课程晋级/难例挖掘参数计划
    curriculum.py         # 连续通过门槛和课程 gate
    hard_examples.py      # 难例采样权重计划
    executor.py           # trial 创建、记录、低预算 job 和 checkpoint 继承
    objectives.py         # quality/VRAM/latency 多目标排序公式
```

## 分层约定

- 本模块内部调用的核心代码保留在 `core/optimization`。
- 不调用 `core/spatial`、`core/temporal`、`core/decision` 等其他 core 模块。
- 需要复用的底层公共 API 使用 `traning.lib.metrics` 和 `traning.state`。
- 只有未来出现跨 core 复用的稳定 API 时，才把对应最小 API 上移到 `lib`。

## 评分模块

入口：

```python
from traning.core.optimization import SampleScoringInput, score_trial
```

评分版本：

```text
point-slider-v2+click-sequence-v1+aggregate-v1
```

聚合逻辑：

- 单对象空间/时间/slider 路径评分复用 `point-slider-v2`。
- 点击序列模拟复用 `click-sequence-v1`。
- sample 分数按目标数量加权。
- miss、frequency limit、unresolved target 会扣分。
- 输出 `TrialScoreReport`，包含 `quality_score`、命中/失败/未解决数量和样本明细。

## 错误归因模块

入口：

```python
from traning.core.optimization import analyze_trial_attribution
```

归因域固定为：

```text
spatial / temporal / decision
```

模块会统计：

- 每类错误数量和占比；
- `early_click`、`late_click`、`spatial_miss`、`frequency_limited` 等 tag；
- 按严重度排序的 hard examples。

未被任何点击解决的 target 归到 `decision`，tag 为 `unresolved_target`。

## 参数搜索修改模块

入口：

```python
from traning.core.optimization import plan_next_trial
```

计划器输入：

- `TrialScoreReport`
- `AttributionSummary`
- 同 rung / 同课程阶段的历史 trial 分数
- 可选显存目标和搜索配置

输出 `OptimizationPlan`：

- `asha_action`：`continue / promote / prune`
- `next_status`：对应 trial 状态
- `next_stage`：课程晋级后的阶段
- `parameter_updates`：按 `training / inference / sampling / search` 分组的参数修改建议
- `hard_example_keys`：难例挖掘采样键
- `priority_domains`：本轮优先优化的错误域
- `objective_score` / `objective_values`：`multi-objective-v1` 综合排序分和各独立目标

## 执行器

入口：

```python
from traning.core.optimization import execute_optimization_plan
```

执行器负责：

- 根据 `OptimizationPlan` 创建下一轮 `TrialMetadata`；
- 合并 `training / inference / sampling` 参数更新；
- 生成低预算 `TrainingJobSpec`；
- 保存 `parent_checkpoint_path`，用于同 trial 晋级或恢复；
- 将完整执行记录追加到 JSONL 或 SQLite trial store；
- 同时输出课程 gate 和 hard example 采样权重。

它不会直接调用 `train-spatial`、`train-temporal` 或其他 core 模块。真正训练仍由外部 runner/CLI
消费 `TrainingJobSpec`，这样优化模块只负责闭环决策和记录，不跨层编排训练实现。

## 结果导出衔接

入口：

```python
from traning.core.optimization import build_batch_gallery_request
```

该函数把 `TrialScoreReport` 转换为 `BatchGalleryRequest`，因此结果导出不再只能依赖手写
外部 JSON。`core/result_export` 仍然只消费 request，不反向调用优化模块。

## 完整训练自动闭环

完整训练入口 `core/decision/pipeline.py::run_full_training_pipeline` 在
`settings.optimization.enabled=true` 时会执行单轮闭环：

```text
trial_score_report.json
  -> analyze_trial_attribution
  -> plan_next_trial
  -> execute_optimization_plan
  -> attribution.json / optimization_plan.json / next_training_job.json
```

默认配置不启用自动优化，避免无限递归训练。启用后默认只生成一个 next job，runner/CLI 通过
`python -m traning.main run-job --job <next_training_job.json>` 消费，实际训练仍调用普通
`run_training_job_spec` / `run_training` 业务函数。停止策略由
`settings.optimization.max_generated_jobs`、`max_trials`、`max_stage`、`dry_run` 和
`job_only` 控制；当前 pipeline 不会默认递归执行子 trial。

trial store 默认保持 JSONL 兼容；需要 SQLite 时设置：

```yaml
optimization:
  trial_store_backend: sqlite
  trial_store_sqlite_path: ../runs/optimization/trials.sqlite
```

多目标排序版本为 `multi-objective-v1`。默认目标包括 `quality_score`、
`peak_vram_mb` 和 `latency_ms`，权重可通过
`optimization.objective_weights` 调整。

统一生命周期入口 `core/full_flow/orchestrator.py::run_full_flow` 通过
`core/training_ramp.py::run_training_ramp` 间接调用上述完整训练 pipeline，因此 full-flow 生成的
level 目录同样包含 `trial_score_report.json`、`attribution.json`、`optimization_plan.json` 和
`next_training_job.json`。full-flow 本身只负责运行阶段、报告和 inheritance 收口，不复制
optimization 的评分或计划逻辑。
