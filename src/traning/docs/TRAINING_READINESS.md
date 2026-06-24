# traning 训练启动就绪报告

检查日期：2026-06-24

本文记录 `src/traning` 当前从数据、空间、时间、决策、评分、训练到图像导出的可启动状态。
源码定位仍以 [`CODEX_INDEX.md`](CODEX_INDEX.md) 为准；长期设计见
[`TRAINING_PLAN.md`](TRAINING_PLAN.md) 和六个模块 plan。

## 总结论

主训练链路已经具备开始训练的最低闭环：

```text
data-check
  -> model-smoke
  -> train-spatial
  -> build-candidate-cache
  -> train-temporal
  -> run-decision
  -> save-annotation-gallery
```

本轮已用真实 `training_package/video_segments` 数据做过最小 smoke，CPU 路径能完整启动并写出
空间训练 summary、候选缓存、时序 checkpoint、决策 JSONL 和标注图集。普通 Codex sandbox
看不到 CUDA，真实 GPU 训练前还需要按 [`ENVIRONMENT.md`](ENVIRONMENT.md) 走 `host-exec`
做一次 `--require-cuda` 检查。

## 模块状态

| 模块 | 思路 | 当前完成度 | 训练前缺口 |
|---|---|---|---|
| 空间 | 保留原始分辨率，用重叠 patch 串行训练；全局 encoder 提供上下文，局部 encoder/fusion/head 产出 dense 空间预测；CPU 侧做全图融合、NMS、slider 连通域和候选解码。 | `train-spatial`、`spatial-decode-smoke`、`run_spatial_frame_inference`、候选/slider 解码和候选缓存生成已接入。 | 不阻塞开训。质量增强项是候选 refiner、条件歧义复查、跨 patch 一致性 loss 和更细的 slider 实例归属。 |
| 时间 | 消费空间候选缓存，按 `sample_key` 组成固定长度因果窗口；动作是 `no_op / press / hold / release`，模型用 GRU 首版保持流式因果接口。 | `TemporalCandidateWindowDataset`、`beatmap_action_v1` 标签、`train-temporal`、`temporal_model.pt` checkpoint 已实现。 | 不阻塞开训。后续需加强 circle release、slider repeat、spinner hold、密集重复点边界和因果一致性测试。 |
| 决策 | 离线阶段把空间候选、slider path、歧义原因和时序监督写进 `spatial-candidate-cache-v1`；推理阶段加载 temporal checkpoint，把窗口输出转成逐帧动作决策。 | `build-candidate-cache` 和 `run-decision` 已跑通，能输出 `manifest.json`、`frames.jsonl`、`decisions.jsonl`。 | 不阻塞开训。自动调参闭环由 `core/optimization` 负责。 |
| 评分 | 单对象用 `point-slider-v2`，点击序列用 `click-sequence-v1`，再由 `core/optimization/scoring` 聚合到 trial 级 `quality_score`。 | 底层评分 API、sample/trial 聚合、图集 request 构造和 `point-slider-v2+click-sequence-v1+aggregate-v1` 版本已实现。 | 不阻塞开训。后续重点是把真实评估集预测转换成 `SampleScoringInput`。 |
| 错误归因 | 在 `core/optimization/attribution` 中按空间、时间、决策三类统计错误和 hard examples。 | 已能输出 domain counts/rates、tag counts、难例列表和难例采样权重。 | 不阻塞开训。后续可把采样权重接入具体训练 runner。 |
| 参数调优 | 在 `core/optimization/parameter_search` 中根据评分、归因和历史 trial 计划下一轮参数。 | 已有 ASHA continue/promote/prune、TPE/random 标记、课程 gate、分组参数更新、trial JSONL 记录、低预算 job 和 checkpoint 继承路径。 | 不阻塞开训。训练 runner 消费 `TrainingJobSpec` 后即可自动调度。 |
| 训练 | 配置集中在 `conf`，运行时统一走 `traning.lib.runtime`；空间训练按 patch 串行，时序训练消费候选缓存。 | `data-check`、`model-smoke`、`train-spatial`、`train-temporal` 可启动，测试通过。 | 完整一键 pipeline 注册表还只登记 `dataset_import`；现阶段用 CLI 分步训练。真实训练需 GPU host bridge。 |
| 结果导出 | 训练/评估结果不直接进入训练异常路径，图像导出 best-effort；按最高分 trial 输出 passed/failed 图集和 manifest。 | `visualize-label`、`save-annotation-gallery` 已实现；optimization 可直接生成 `BatchGalleryRequest`。 | 不阻塞开训。HTML/CSV 浏览索引仍是增强项。 |
| 模型导出 | 把 checkpoint、配置和版本信息整理成可迁移 artifact。 | `core/model_export` 已能导出 inference/resume PyTorch artifact、写 manifest、记录 sha256 并校验。 | 不阻塞开训。完整一帧推理 smoke 需要正式空间+时间 checkpoint 组合。 |

## 本轮检查结果

| 检查 | 命令 | 结果 |
|---|---|---|
| 环境依赖 | `PYTHONPATH=src python -m traning.cli env-check --strict` | 通过。Python、ffmpeg、torch、torchvision、opencv、pyav、pydantic、prefect 等必需依赖可导入。sandbox 内 CUDA 不可见。 |
| 数据输入 | `PYTHONPATH=src python -m traning.cli data-check --config configs/model_small_vram.yaml --split train` | 通过。train split：190 个 segment，约 23949 帧，issues=0。 |
| 模型 smoke | `PYTHONPATH=src python -m traning.cli model-smoke --config configs/model_small_vram.yaml` | 通过。CPU 上完成 patch/global/local/fusion/head 前后向。 |
| 空间训练 smoke | `PYTHONPATH=src python -m traning.cli train-spatial --config configs/model_small_vram.yaml --device cpu --max-steps 1 --patch-limit 1` | 通过。写出 `runs/20260624T081034_705227Z__train_spatial`，1 step，1 sample，last_loss=4.622583。 |
| 候选缓存 smoke | `PYTHONPATH=src python -m traning.cli build-candidate-cache --config configs/model_small_vram.yaml --device cpu --max-frames 1 --patch-limit 1 --output /tmp/traning_candidate_cache_smoke` | 通过。输出 1 帧、32 个候选、4 条 slider path。 |
| 时序训练 smoke | `PYTHONPATH=src python -m traning.cli train-temporal --config configs/model_small_vram.yaml --cache /tmp/traning_candidate_cache_smoke --device cpu --max-steps 1 --sequence-length 4 --candidate-slots 8` | 通过。写出 `runs/20260624T081050_921138Z__train_temporal/temporal_model.pt`。 |
| 决策导出 smoke | `PYTHONPATH=src python -m traning.cli run-decision --config configs/model_small_vram.yaml --cache /tmp/traning_candidate_cache_smoke --checkpoint runs/20260624T081050_921138Z__train_temporal/temporal_model.pt --output /tmp/traning_decision_smoke --device cpu` | 通过。输出 `decisions.jsonl` 和 `manifest.json`。 |
| 图像导出 smoke | `PYTHONPATH=src python -m traning.cli save-annotation-gallery --config configs/model_small_vram.yaml --results /tmp/traning_gallery_request.json --output-root /tmp/traning_gallery_cli_smoke --samples-per-group 1` | 通过。输出 2 张 PNG、`manifest.json` 和 `best_parameters.json`。 |
| 单元测试 | `PYTHONPATH=src python -m pytest src/traning/tests -q` | 63 passed，2 skipped，1 warning。warning 来自 sandbox 内 CUDA 初始化不可用。 |

## 可以开始训练的命令顺序

真实 GPU 训练建议在 host bridge 中执行：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src python -m traning.cli env-check --strict --require-cuda'
```

确认 CUDA 后，先跑空间训练：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src python -m traning.cli train-spatial --config configs/model_small_vram.yaml --device cuda --max-steps 100 --patch-limit 4'
```

生成候选缓存：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src python -m traning.cli build-candidate-cache --config configs/model_small_vram.yaml --device cuda --max-frames 1000 --patch-limit 4 --output runs/candidate_cache_train'
```

训练时序模型：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src python -m traning.cli train-temporal --config configs/model_small_vram.yaml --cache runs/candidate_cache_train --device cuda --max-steps 1000'
```

导出决策：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src python -m traning.cli run-decision --config configs/model_small_vram.yaml --cache runs/candidate_cache_train --checkpoint <temporal-run>/temporal_model.pt --output runs/decision_train --device cuda'
```

## 下一步补全清单

1. 先跑一轮小预算 GPU 空间训练，记录显存峰值、loss 和候选缓存质量。
2. 用小预算候选缓存训练 temporal，检查 `press/hold/release` 分布是否合理。
3. 用 optimization 生成首个 `TrialScoreReport`、`AttributionSummary` 和 `TrainingJobSpec`。
4. 用 `build_batch_gallery_request` 把评分结果送进 `save-annotation-gallery`。
5. 用 `export_model_artifact` 打包阶段 checkpoint、配置和 score/candidate-cache 版本。
