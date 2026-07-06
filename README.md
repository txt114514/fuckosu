# osu video training workspace

This repository prepares osu! gameplay data and trains a video-to-action model.
The stable training CLI is `python -m traning.main`; `python -m traning.cli`
is kept only as a compatibility alias.

## Current Status

Checked on 2026-07-05.

- Git workspace was clean at the time of the check.
- Project indexes are current: `python project_index/build_index.py --check`.
- `src` module entries are importable: `start`, `package`, `before_traning`,
  and `traning`.
- CPU environment check passes. The normal Codex sandbox cannot see CUDA, so
  GPU checks and CUDA training should be run through the host bridge.
- `full-flow --mode plan` passes and writes the expected manifest, state, and
  report files.
- `full-flow --mode dry-run` passes. It runs raw-data checks, split validation,
  startup checks, resume discovery, and report generation without training.
- Current raw-data check found no unmatched beatmap/audio candidates, so
  `before_traning` is skipped unless new raw data appears.
- Current dataset split manifest has 1 train item and 1 validation item:
  `item_000001` has 190 train segments, `item_000002` has 90 validation
  segments, and test split is frozen at 0 items.
- Training startup checks see 190 train segments and about 23,949 estimated
  train frames with no reported data-input issues.

The project is ready for controlled training. The recommended route is
`full-flow`: it coordinates raw-data checks, optional `before_traning`, split
sync, preflight, resume discovery/restore, ramp training, artifact export,
inheritance packaging, and reports. The last documented GPU ramp verification
in `src/traning/docs/TRAINING_READINESS.md` reached Level B on 2026-06-26; GPU
verification was not rerun in this README update.

## 计划完成情况

- `before_traning` 七阶段数据处理流程：谱面导入、校验、难度提取、视频匹配、音画对齐、裁剪和片段生成。
- 训练数据导入与 split manifest 管理。
- 空间模型训练与单帧推理。
- 候选缓存生成。
- 时序模型训练与因果动作预测。
- 决策结果导出。
- 评分、错误归因、gallery 图集和 next job 输出。
- 可视化模块：中文终端训练 UI、Rich/plain/off 进度模式、结构化 `TrainingReporter`、
  full-flow 阶段上报、训练 step/loss/资源状态上报、passed/failed gallery 图集、
  `manifest.json` 和 `index.csv` 导出。
- 模型 artifact 导出。
- checkpoint / inheritance 断点继承与恢复。
- `full-flow` 完整生命周期入口。
- CPU plan / dry-run 检查已通过。
- 文档记录中 GPU ramp 已验证到 Level B。

目前进行到计划的 **受控训练 / ramp-to-full 已打通，准备放大到完整目标 GPU 训练** 这一步。
也就是说，工程闭环已经完成，当前重点不再是补主流程代码，而是执行完整训练并根据真实结果调优。

当前没有未完成的 P0/P1/P2 代码缺口。还没有实现或还没有最终完成验证的内容包括：

- 完整 full 目标 GPU 训练还没有确认跑完。
- 本次 README 更新没有重新跑真实 CUDA 验证，GPU 路径仍需通过 host bridge 检查。
- 模型质量还需要依赖真实长训练继续调参，包括 press/release 时间窗、课程采样、loss 权重和评分门槛。
- 可视化目前是终端 UI 和静态 gallery/index 导出，还没有做独立 Web 仪表盘或交互式浏览器。

已实现的增强项首版：

- 更强的局部模型复查：候选缓存生成阶段会用 `local_consistency_model_v1` 复判候选点、
  slider 端点和低置信区域；可解决的歧义会从候选标记中移除，局部 refiner 可把
  slider head 贴合到路径端点或中心线。
- optimization 记录升级：保留 JSONL 兼容 store，并新增 `SQLiteTrialStore`；
  配置可通过 `optimization.trial_store_backend` 在 `jsonl/sqlite` 之间切换。
- 多目标排序公式：新增 `multi-objective-v1`，把 `quality_score`、`peak_vram_mb`、
  `latency_ms` 拆成独立目标并计算可复现 `objective_score`。
- SMET 动态拓扑：新增动态 top-k 稀疏线性层，并接入时序模型 action/candidate/xy/time heads；
  默认关闭，可通过 `smet.enabled`、`smet.sparsity`、`smet.update_interval`
  和 `smet.min_density` 配置启用。

## Recommended Command

Run the real CUDA path from the container namespace:

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc '
cd /home/dev/workspace
PYTHONPATH=src:. python -m traning.main full-flow \
  --config configs/model_full_small_vram.yaml \
  --device cuda \
  --resume \
  --resume-policy auto \
  --auto-launch-full \
  --progress-ui rich \
  --progress-language zh-CN
'
```

Check CUDA first:

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src:. python -m traning.main env-check --strict --require-cuda'
```

## Non-Training Checks

Preview the full flow on CPU:

```bash
PYTHONPATH=src:. python -m traning.main full-flow --mode plan --config configs/model_small_vram.yaml --device cpu
PYTHONPATH=src:. python -m traning.main full-flow --mode dry-run --config configs/model_small_vram.yaml --device cpu --progress-ui off
```

Inspect top-level module entrypoints:

```bash
PYTHONPATH=src python -m start modules
PYTHONPATH=src python -m start check --config configs/model_small_vram.yaml --device cpu
```

Resume from the latest inheritance package:

```bash
PYTHONPATH=src:. python -m traning.main full-flow --config configs/model_small_vram.yaml --device cuda --resume --resume-policy auto
```

## Outputs

Important outputs are written under `artifacts/training_runs/<run_id>/`:

- `full_flow_state.json`
- `resume_report.json`
- `reports/full_flow_report.md`
- `ramp/`
- `artifacts/`
- `inheritance/`

The dry-run status check used for this README wrote temporary files under
`/tmp/readme_status_check/`.

## Module Snapshot

| Module | Status |
|---|---|
| `src/before_traning` | Seven-stage data preparation pipeline for beatmap import, verification, difficulty export, video match, AV sync, clipping, and segment generation. Current startup check found no new unmatched raw data. |
| `src/package` | Stable shared API layer for contracts, checks, dataset split, coordinates, and slider path helpers. |
| `src/start` | Top-level module registry and startup/self-check orchestration. Current module import check passes. |
| `src/traning` | Main training system. `full-flow` is the recommended lifecycle entry; spatial, temporal, decision, scoring, result export, model export, ramp, and inheritance flows are wired. |
| `src/visualization` | Chinese terminal training UI, report/gallery APIs, and visualization state helpers. |
| `environment` | Environment and CUDA diagnostics. Use host bridge for real GPU checks. |

## Documentation

- [Quick Start](docs/QUICK_START.md)
- [Training Workflow](docs/TRAINING_WORKFLOW.md)
- [Documentation Index](docs/INDEX.md)
- [Project Map](project_index/PROJECT_MAP.md)
- [Training Readiness](src/traning/docs/TRAINING_READINESS.md)
- [CUDA Environment](src/traning/docs/ENVIRONMENT.md)

Engineering docs for Codex and maintainers live in
[docs/codex/INDEX.md](docs/codex/INDEX.md).

## Verification Log

The README status above is based on these commands:

```bash
git status --short
python project_index/build_index.py --check
PYTHONPATH=src python -m start modules
PYTHONPATH=src:. python -m traning.main env-check --strict
PYTHONPATH=src:. python -m traning.main full-flow --mode plan --config configs/model_small_vram.yaml --device cpu --output-root /tmp/readme_status_check --run-id plan_check --skip-full-checks --progress-ui off
PYTHONPATH=src:. python -m traning.main full-flow --mode dry-run --config configs/model_small_vram.yaml --device cpu --output-root /tmp/readme_status_check --run-id dry_run_check --skip-full-checks --progress-ui off
```
