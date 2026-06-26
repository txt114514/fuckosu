# traning 训练启动就绪报告

检查日期：2026-06-26

本文记录 `src/traning` 当前从数据、空间、时间、决策、评分、训练到图像导出的可启动状态。
源码定位仍以 [`CODEX_INDEX.md`](CODEX_INDEX.md) 为准；长期设计见
[`TRAINING_PLAN.md`](TRAINING_PLAN.md) 和六个模块 plan。

## 总结论

主训练链路已经具备开始受控训练的闭环，并提供一条覆盖训练生命周期的统一入口：

```text
full-flow
  -> raw-data/before_traning/split/preflight
  -> inheritance discovery
  -> real checkpoint restore
  -> ramp-to-full controlled levels
  -> final readiness
  -> optional full training launch
  -> score/gallery/artifact/inheritance/report
```

本轮已用真实 `training_package/video_segments` 数据做过最小 smoke，CPU 路径能通过
`python -m start run` 完整启动流程演练。完整启动先调用 before_traning 启动检测包，
只读判断是否存在未匹配的新 `.osz`、视频和已导入待匹配样本；已有匹配或已打包训练集不重新检测、不修改。
随后同步 `training_package/splits/dataset_split_manifest.json`，已有 item 的 split 冻结，
只给新增且已打包完成的 item 做增量分配。之后调用 traning 启动检测包检查配置、设备、数据输入和 core 入口；底层完整训练仍会调用
`start.checks.run_training_startup_checks`，通过模块入口、环境、配置、设备和数据输入自检后，
再写出 `full_training_summary.json`、空间训练 summary、候选缓存、时序 checkpoint、决策
JSONL、score、gallery、next job 和可独立 smoke 的 model artifact。兼容层已删除，训练稳定版只保留当前正式导入路径。普通 Codex sandbox 看不到 CUDA，真实 GPU 训练前还需要按
[`ENVIRONMENT.md`](ENVIRONMENT.md) 走 `host-exec` 做一次 `--require-cuda` 检查。

推荐命令是 `python -m traning.main full-flow`。`python -m traning.cli` 仍保留为兼容别名，
但文档和新入口均以 `traning.main` 为准。

## 模块状态

| 模块 | 思路 | 当前完成度 | 训练前缺口 |
|---|---|---|---|
| 空间 | 保留原始分辨率，用重叠 patch 串行训练；全局 encoder 提供上下文，局部 encoder/fusion/head 产出 dense 空间预测；CPU 侧做全图融合、NMS、slider 连通域和候选解码。 | `train-spatial`、`spatial-decode-smoke`、`run_spatial_frame_inference`、候选/slider 解码、候选缓存、局部 refiner、条件歧义复查和配置化空间一致性 loss 已接入。 | 不阻塞开训。 |
| 时间 | 消费空间候选缓存，按 `sample_key` 组成固定长度因果窗口；动作是 `no_op / press / hold / release`，模型用 GRU 首版保持流式因果接口。 | `TemporalCandidateWindowDataset`、circle release、slider repeat、spinner hold/release、配置化 temporal loss、`train-temporal`、`temporal_model.pt` checkpoint 和因果一致性测试已实现。 | 不阻塞开训。 |
| 决策 | 离线阶段把空间候选、slider path、歧义原因、显式 transform 和时序监督写进 `spatial-candidate-cache-v1`；推理阶段加载 temporal checkpoint，把窗口输出转成逐帧动作决策。 | `build-candidate-cache` 和 `run-decision` 已跑通，能输出带版本信息的 `manifest.json`、`frames.jsonl`、`decisions.jsonl`。 | 不阻塞开训。 |
| 评分 | 单对象用 `point-slider-v2`，点击序列用 `click-sequence-v1`，再由 `core/optimization/scoring` 聚合到 trial 级 `quality_score`。 | `core/optimization/scoring/run_outputs.py` 已从候选缓存和 decisions JSONL 构建真实评分输入，完整训练会写 `trial_score_report.json` 和 `gallery_request.json`。 | 不阻塞开训。 |
| 错误归因 | 在 `core/optimization/attribution` 中按空间、时间、决策三类统计错误和 hard examples。 | 已能输出 domain counts/rates、tag counts、难例列表和难例采样权重。 | 不阻塞开训。后续可把采样权重接入具体训练 runner。 |
| 参数调优 | 在 `core/optimization/parameter_search` 中根据评分、归因和历史 trial 计划下一轮参数。 | 完整训练可在 `settings.optimization.enabled=true` 时自动写出归因、计划、trial JSONL 和 `next_training_job.json`；`run-job` 可消费 job。 | 不阻塞开训。默认不递归执行子 trial。 |
| 训练 | 配置集中在 `conf`，运行时统一走 `traning.lib.runtime`；空间训练按 patch 串行，时序训练消费候选缓存。 | `full-flow` 是推荐入口；`run` 已串接 data-check、空间训练、候选缓存、时序训练和决策导出；分步 CLI 仍保留。 | 不阻塞开训。真实训练需 GPU host bridge；大预算参数按显存逐步放大。 |
| 结果导出 | 训练/评估结果不直接进入训练异常路径，图像导出 best-effort；按最高分 trial 输出 passed/failed 图集、manifest 和 CSV 索引。 | `visualize-label`、`save-annotation-gallery` 已实现；optimization 可直接生成带版本 metadata 的 `BatchGalleryRequest`。 | 不阻塞开训。 |
| 模型导出 | 把 checkpoint、配置和版本信息整理成可迁移 artifact。 | `core/model_export` 已能导出 inference/resume PyTorch artifact、写 manifest、记录 sha256、迁移旧 settings 并执行 CPU artifact smoke。 | 不阻塞开训。 |
| 断点续训 | 保存 `training-checkpoint-v1`，按 committed optimizer step 恢复模型、optimizer、GradScaler、RNG 和训练位置。 | `core/training_inheritance`、spatial/temporal trainer、`FullTrainingRunConfig.resume_stage_checkpoints` 和 full-flow/ramp 均已接线；strict resume smoke 已从 step 2 续到 step 3。 | 不阻塞开训。 |
| 训练放大 | 用固定 gate 从小预算逐级放大到目标配置，失败时保留独立 level 目录和 readiness 报告。 | `core/training_ramp.py` 和 `traning.main ramp-to-full` 已接入真实完整训练流水线、next job dry-run、checkpoint reload、gallery 非空检查和 artifact smoke；`full-flow` 调用该实现。 | 不阻塞开训。默认不自动启动 full，需显式 `--auto-launch-full`。 |
| 完整流程 | 统一编排 raw data、before_traning、split、preflight、resume、ramp、artifact、inheritance 和报告。 | `core/full_flow`、`traning.main full-flow`、`full-flow-status` 已实现 plan/dry-run/status/execute。 | 不阻塞开训。 |

## 本轮检查结果

| 检查 | 命令 | 结果 |
|---|---|---|
| 环境依赖 | `PYTHONPATH=src:. python -m traning.main env-check --strict` | 正式入口为 `traning.main`；`traning.cli` 如存在仅作为兼容别名使用。 |
| 数据输入 | `PYTHONPATH=src:. python -m traning.main data-check --config configs/model_small_vram.yaml --split train` | data check 现在包含细分分布和 slider 拓扑报告。 |
| 端到端 run smoke | `PYTHONPATH=src:. python -m traning.main run --config configs/model_small_vram.yaml --device cpu --spatial-max-steps 1 --temporal-max-steps 1 --patch-limit 1 --cache-max-frames 1 --sequence-length 4 --candidate-slots 8` | 输出 `trial_score_report.json`、`gallery_request.json`、score/gallery 版本信息；启用 optimization 后还会输出 next job。 |
| start 模块入口 | `PYTHONPATH=src python -m start modules` | 通过。列出 `start`、`package`、`before_traning`、`traning` 四个入口且均可 import。 |
| start 训练自检 | `PYTHONPATH=src python -m start check --config configs/model_small_vram.yaml --device cpu` | 通过。模块入口、环境、训练设置、运行设备和数据输入均通过。 |
| start 完整启动 dry-run | `PYTHONPATH=src python -m start run --training-config configs/model_small_vram.yaml --device cpu --dry-run --test-level quick --no-before-match-probe` | 通过。before_traning raw-data 检测显示当前没有未匹配候选；split-sync dry-run 显示将 bootstrap 2 个 item；traning 4 项启动检测通过，quick 检测通过。 |
| full-flow help | `PYTHONPATH=src:. python -m traning.main full-flow --help` | 通过。Typer 能解析 `--mode`、`--resume`、`--from-stage`、`--until-stage`、`--force-stage`、`--skip-stage`。 |
| full-flow plan | `PYTHONPATH=src:. python -m traning.main full-flow --mode plan --config configs/model_small_vram.yaml --device cpu --output-root /tmp/full_flow_verify --run-id plan_check --skip-full-checks --progress-ui off` | 通过。写出 `full_flow_manifest.json`、`full_flow_state.json` 和 `reports/full_flow_report.*`。 |
| full-flow dry-run | `PYTHONPATH=src:. python -m traning.main full-flow --mode dry-run --config configs/model_small_vram.yaml --device cpu --output-root /tmp/full_flow_verify --run-id dry_run_check --skip-full-checks --progress-ui off` | 通过。执行 raw-data、split、traning startup 和 resume discovery，不训练。 |
| full-flow CPU execute smoke | 临时 target：2 spatial step、2 temporal step、120 candidate frames、sequence=2、candidates=2。 | 通过。120 frames、3651 candidates、59 gallery frames、score、next job、artifact 和 inheritance 均生成。 |
| full-flow strict resume smoke | 从 `/tmp/full_flow_verify/execute_tiny_120/inheritance` 恢复到 3 step。 | 通过。`resume_report.json` 记录恢复 optimizer、scaler、rng、training_position、spatial_checkpoint、temporal_checkpoint；最终 spatial/temporal checkpoint 均为 committed step 3。 |
| 受影响测试 | `PYTHONPATH=src:. pytest -q src/traning/tests/full_checks/test_full_flow.py src/traning/tests/full_checks/test_temporal_trainer.py src/traning/tests/full_checks/test_training_inheritance.py src/traning/tests/full_checks/test_full_training_pipeline.py src/visualization/tests/test_dashboard.py` | 13 passed，5 subtests passed，2 warnings。warning 来自 sandbox 内 CUDA 初始化不可用和 opentelemetry import metadata。 |
| 启动/全面测试包 | `PYTHONPATH=src python -m pytest src/start/tests src/before_traning/tests src/traning/tests -q` | 上轮全量为 89 passed，2 skipped，2 warnings；本轮尚未重新跑全量。 |
| package 测试 | `PYTHONPATH=src python -m pytest src/package/tests -q` | 11 passed。 |
| 兼容层删除检查 | `PYTHONPATH=src python -c "import importlib.util, sys, traning; print(importlib.util.find_spec('traning.lib.compat')); print('traning.Lib' in sys.modules)"` | 通过。输出 `None` 和 `False`，确认不再提供 compatibility layer。 |
| ramp Level A GPU | `host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src:. python -m traning.main ramp-to-full --config configs/model_small_vram.yaml --device cuda --max-levels 1 --skip-full-checks --output-root artifacts/training_ramp_verify'` | 通过。run `artifacts/training_ramp_verify/20260626T151919Z`，100 spatial step、100 temporal step、500 candidate frames、15942 candidates、183 gallery frames、CPU artifact smoke finite。 |
| ramp Level B GPU | `host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src:. python -m traning.main ramp-to-full --config configs/model_small_vram.yaml --device cuda --max-levels 2 --skip-full-checks --output-root artifacts/training_ramp_verify --run-id 20260626T151919Z'` | 通过。300 spatial step、300 temporal step、1500 candidate frames、47958 candidates、481 gallery frames、quality_score 0.679333、peak reserved VRAM 0.285 GiB、CPU artifact smoke finite、`run-job --dry-run` 通过。 |

## 可以开始训练的命令顺序

真实 GPU 训练建议在 host bridge 中执行：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src:. python -m traning.main env-check --strict --require-cuda'
```

确认 CUDA 后，推荐用完整流程入口启动。默认不会无限递归；只有显式 `--auto-launch-full`
才会在 ramp gate 后启动目标 full run：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src:. python -m traning.main full-flow --config configs/model_full_small_vram.yaml --device cuda --resume --resume-policy auto --auto-launch-full --progress-ui rich'
```

需要仅执行渐进放大时，仍可使用兼容入口 `ramp-to-full`。
需要从已有继承包恢复时传入 `--resume --resume-policy auto`，或用
`--inherit-from <inheritance-dir> --resume-policy strict` 指定来源。

当前小显存 full 目标配置在 `configs/model_full_small_vram.yaml`，ramp 会在输出目录写入
`resolved_target_config.yaml`、`manifest.json`、`final_readiness.md`、`final_readiness.json` 和
`full_train_command.sh`。`full_train_command.sh` 是 gate 后可执行的完整目标训练命令。

也可以先用 start 入口跑固定小预算端到端训练：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src python -m start run --training-config configs/model_small_vram.yaml --device cuda --test-level quick --spatial-max-steps 100 --temporal-max-steps 100 --patch-limit 4 --cache-max-frames 1000 --sequence-length 64 --candidate-slots 16'
```

`--patch-limit 0` 表示每帧使用全部 patch，`--cache-max-frames 0` 表示不限制候选缓存帧数；
二者适合正式完整训练前确认显存稳定后再使用。

也可以按阶段分步跑。先跑空间训练：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src:. python -m traning.main train-spatial --config configs/model_small_vram.yaml --device cuda --max-steps 100 --patch-limit 4'
```

生成候选缓存：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src:. python -m traning.main build-candidate-cache --config configs/model_small_vram.yaml --device cuda --max-frames 1000 --patch-limit 4 --output runs/candidate_cache_train'
```

训练时序模型：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src:. python -m traning.main train-temporal --config configs/model_small_vram.yaml --cache runs/candidate_cache_train --device cuda --max-steps 1000'
```

导出决策：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src:. python -m traning.main run-decision --config configs/model_small_vram.yaml --cache runs/candidate_cache_train --checkpoint <temporal-run>/temporal_model.pt --output runs/decision_train --device cuda'
```

## 当前可启动判断

可以开始受控训练。已验证到 Level B，且每级都通过真实空间训练、候选缓存、时序训练、
决策导出、固定评分、gallery、next job、job dry-run、artifact 导出和 CPU 反加载推理。
本轮新增的 full-flow CPU execute/resume smoke 也已通过。默认仍不自动启动 full；正式放大到完整目标时使用
`full-flow --auto-launch-full`，或者在已有通过 ramp run 中执行对应的 `full_train_command.sh`。
