# osu! 视频决策模型工作区

仓库以 `src/start/main.py` 作为唯一总启动流程，以 `src/traning` 作为唯一活动训练包。
旧 Spatial → Temporal imitation → argmax Decision 实现和独立 `src/visualization`
包已经退役；V2 已整体迁入 `traning`。旧扁平路径只保留 deprecated re-export，
新代码以 `conf/core/lib/state` 为唯一权威结构。

## 当前模型链路

```text
Image
  → Perception
  → Association / Tracking
  → Temporal Belief
  → Outcome Prediction
  → Decision / Optimal Stopping
```

训练与推理使用 typed contracts，正式动作路径不读取 GT、oracle label 或旧 action
logits。唯一活动配置是 `configs/traning.yaml`；默认 `optimization.max_trials: null`，
普通门禁失败后会继续选择合法且未重复的参数，直到全部门禁通过、搜索空间显式耗尽、
用户中断或发生不可恢复错误。

## 启动

完整流程保持原入口语义：原始数据检查 → 可选 `before_traning` 七阶段转换 → split
同步 → canonical 数据质量与环境检查 → 可恢复生产训练 → 报告。

```bash
PYTHONPATH=src python src/start/main.py run \
  --config configs/traning.yaml \
  --device cuda \
  --resume
```

无参数直接执行同一完整流程：

```bash
PYTHONPATH=src python src/start/main.py
# 等价模块入口
PYTHONPATH=src python -m start
```

只执行模型侧诊断或训练：

```bash
PYTHONPATH=src python -m traning config-check --config configs/traning.yaml
PYTHONPATH=src python -m traning coordinate-audit --config configs/traning.yaml
PYTHONPATH=src python -m traning env-check --config configs/traning.yaml --strict
PYTHONPATH=src python -m traning train --config configs/traning.yaml --resume
```

CUDA 命令必须从主机桥进入正常容器 namespace：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc \
  'cd /home/dev/workspace && bash src/traning/lib/environment/check_gpu.sh'
```

## 主要目录

| 路径 | 职责 |
|---|---|
| `src/before_traning` | 谱面导入、视频匹配、音画对齐、裁剪与 segment 生成 |
| `src/package` | 多个顶层模块共用的坐标、split 与检查公开 API |
| `src/start` | 唯一总启动编排和检查入口 |
| `src/traning` | `conf/core/lib/state` 分层的训练、推理、环境与统一类型实现 |
| `configs/traning.yaml` | 唯一严格生产配置 |

训练 run 写入 `artifacts/training_runs/<run_id>/`。每个已完成 trial 的搜索历史会原子
写入 `search_state.json`；只有全部门禁通过且 checkpoint 摘要、dataset identity、模型
契约和坐标指纹复验成功，流程才返回 `passed`。

每个到达 Evaluation 的 trial 还会生成
`trials/trial-XXXXXX/gallery/manifest.json`。PNG 按当前帧的 canonical event 分入
`passed/none` 或 `failed/<spatial|temporal|decision|mixed>`；序列名只保留为样本目录，
不会再用整段序列的 AND 结果把一张已通过图片拖进 `failed`。

## 验证与文档

```bash
PYTHONPATH=src:. python -m pytest -q src/traning/tests
PYTHONPATH=src:. python -m pytest -q src/start/tests
python project_index/build_index.py --check
```

- [项目导航](project_index/PROJECT_MAP.md)
- [训练目标与阶段](src/traning/docs/TRAINING_PLAN.md)
- [生成的训练代码索引](src/traning/docs/CODEX_INDEX.md)
- [CUDA 环境](src/traning/docs/ENVIRONMENT.md)
- [V2 迁移状态](V2_MIGRATION_STATUS.md)
- [旧准确画面误归因诊断](src/traning/docs/LEGACY_FAILURE_DIAGNOSIS.md)
