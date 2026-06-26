# Plan Gap Audit

检查日期：2026-06-26

本文对照 `src/traning/docs` 中的六个模块 plan、`TRAINING_PLAN.md`、
`TRAINING_READINESS.md` 和 `OPTIMIZATION_MODULE.md`，只记录“计划里应该做到但当前
还没有真正完成”的项目。

判定规则：

- 训练帧、训练 step 或 smoke 样本只有 1 个也算已打通；低预算不等于没做。
- 已有稳定入口、能产出约定文件、且测试或启动流程能触达的能力，按已实现处理。
- 文档写法已经落后于代码的，放到“文档需更新”，不算实现缺口。
- 质量增强项如果 plan 明确标为“后续/首版不阻塞”，单独列为增强，不影响当前稳定版。

## 真正未完成

当前没有仍被本审计列为 P0/P1/P2 的代码缺口。原审计项已实现并同步到真实入口：

| 原优先级 | 当前实现位置 | 验证位置 |
|---|---|---|
| P0 坐标标定、版本、固定评估、自动闭环 | `traning.lib.coordinates`、`core/decision`、`core/optimization`、`state/versioning.py` | `src/traning/tests/full_checks/test_plan_gap_closure.py`、`test_decision_output_scoring.py`、gallery/export 测试 |
| P0/P1/P2 时序动作、loss 权重、因果测试 | `core/temporal`、`conf/settings.py`、`lib/models/temporal.py` | `test_temporal_targets.py`、`test_temporal_trainer.py`、`test_causal_temporal.py` |
| P1/P2 数据检查、分布、gallery index、artifact migration/smoke | `core/dataset_import`、`core/result_export`、`core/model_export` | `test_data_input_report.py`、`test_result_export_gallery.py`、`test_model_export.py` |
| 本轮新增 full-flow 与真实 resume | `core/full_flow`、`core/training_inheritance`、spatial/temporal trainer | `test_full_flow.py`、`test_training_inheritance.py`、`test_temporal_trainer.py`、`visualization/tests/test_dashboard.py` |

## 文档同步状态

| 文档 | 当前状态 |
|---|---|
| `DECISION_PLAN.md` | 已明确 `core.decision.pipeline.TRAINING_STAGES` 六阶段注册，并区分 `core.full_flow.stages.FULL_FLOW_STAGES`。 |
| `TRAINING_READINESS.md` | 已更新 full-flow、真实 resume、plan/dry-run/execute smoke 和 `traning.main` 主入口。 |
| `OPTIMIZATION_MODULE.md` | 已列出 `scoring/run_outputs.py`，并说明 full-flow 通过 ramp 间接消费 optimization 闭环。 |
| `RESULT_EXPORT_PLAN.md` | 已更新 `predicted_video_xy`、gallery 新位置、manifest/index 和 full-flow artifact 关联。 |
| 根文档 | 已新增 `README.md`、`docs/INDEX.md`、`docs/QUICK_START.md`、`docs/TRAINING_WORKFLOW.md` 和 `docs/codex/*`。 |

## 不算缺口的低预算项

| 项目 | 为什么不算缺口 |
|---|---|
| 空间训练只跑过 1 step 或少量 step | `run_spatial_training`、CLI、loss、runtime 和 summary 已存在；低预算只是训练规模选择。 |
| 候选缓存只生成 1 帧或 20 帧 | `generate_candidate_cache`、manifest、JSONL 和 temporal target 契约已存在；帧数是启动参数。 |
| 时序训练只跑过 1 step 或 20 step | `run_temporal_training` 已输出 `temporal_model.pt` 和 summary；低预算不代表功能缺失。 |
| 决策导出只导出 1 帧或 20 帧 | `run_temporal_decision` 已能从 checkpoint 和 cache 导出 `decisions.jsonl`；帧数由 cache 决定。 |
| gallery 只输出少量 PNG | `save_annotation_gallery` 按 `samples_per_group` 抽样；数量少是配置，不是未实现。 |

## 建议推进顺序

1. 先用 `full-flow --mode plan` 和 `--mode dry-run` 确认当前数据、配置、resume 来源和输出目录。
2. 用 `full-flow --max-levels 1` 做小规模真实训练，确认 gallery、score、artifact 和 inheritance。
3. 通过 host bridge 做 GPU Level A/B，再用 `--auto-launch-full` 放大到目标配置。
