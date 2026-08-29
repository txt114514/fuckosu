# Project Map

这是仓库级全局索引。各模块的解释性文档放在对应源码目录内，本文件只维护活动入口，
不登记已退役的兼容路径。

## 模块索引

| 模块 | 源码目录 | 对外说明 | Codex 索引 |
|---|---|---|---|
| 用户文档 | `docs` | [`INDEX.md`](../docs/INDEX.md)、[`QUICK_START.md`](../docs/QUICK_START.md)、[`TRAINING_WORKFLOW.md`](../docs/TRAINING_WORKFLOW.md) | [`codex/INDEX.md`](../docs/codex/INDEX.md) |
| 唯一启动入口 | `src/start` | [`README.md`](../src/start/README.md) | 公开入口：`src/start/main.py` |
| 训练前处理 | `src/before_traning` | [`README.md`](../src/before_traning/docs/README.md) | [`CODEX_INDEX.md`](../src/before_traning/docs/CODEX_INDEX.md) |
| OSU Decision Model | `src/traning` | [`README.md`](../src/traning/README.md)、[`TRAINING_PLAN.md`](../src/traning/docs/TRAINING_PLAN.md)、[`ENVIRONMENT.md`](../src/traning/docs/ENVIRONMENT.md) | [`CODEX_INDEX.md`](../src/traning/docs/CODEX_INDEX.md) |
| 全局共享 API | `src/package` | [`README.md`](../src/package/README.md) | 公开入口：`src/package/__init__.py` |
| 运行环境检查 | `environment` | 环境/CUDA 诊断脚本与 Python 检查 API | 公开入口：`environment/__init__.py` |

`src/traning` 已由 V2 架构整体覆盖，是唯一活动训练实现。仓库不再提供 `src/osu_v2`、
旧 `traning.core/conf/state`、外部 evaluator 或独立 `src/visualization` 兼容层；训练可视化是
`src/traning/visualization` 内部 telemetry 消费者。

## 全局 API 约定

- `src/package` 只存放被 `src` 下多个顶层模块共同调用的稳定 API。
- 调用方应从 `package` 的公开入口导入，不依赖 `_` 开头的名称或内部实现模块。
- 当前跨模块稳定 API 包括 `package.contracts`、`package.checks`、
  `package.coordinates` 和 `package.dataset_split`。
- `package.coordinates` 统一维护 osu、训练帧像素与模型归一化坐标之间的 affine 方程；
  方程、标定身份和训练帧尺寸共同进入指纹，禁止旧尺寸或旧方程产物被静默复用。
- `package.dataset_split` 维护 canonical `DataSplit` 与
  `training_package/splits/dataset_split_manifest.json`，供 start 同步、traning 读取。
- 只服务训练领域的 typed contract、模型、repository、telemetry 和 renderer 保留在
  `src/traning`，不提前迁入全局包。

## 总启动最短阅读路径

1. `src/start/main.py` 是唯一仓库启动入口；直接脚本与 `python -m start` 共用同一流程。
2. `src/start/flow.py` 固定完整生命周期：raw scan → before_traning → canonical split →
   startup checks → typed dataset quality gate → production training → report。
3. `src/start/executor.py` 把启动层检查过的同一份数据质量报告交给
   `traning.training.ProductionTrainer`，避免检查与训练各算一套结论。
4. `src/start/checks/registry.py` 执行严格配置、环境和 canonical 数据质量检查。
5. `PYTHONPATH=src python src/start/main.py` 或 `PYTHONPATH=src python -m start`
   执行默认完整流程。
6. `PYTHONPATH=src python -m start run --config configs/traning.yaml --dry-run`
   演练完整流程但不启动参数搜索。
7. `PYTHONPATH=src python -m start config-check --config configs/traning.yaml`
   校验唯一训练配置。
8. `PYTHONPATH=src python -m start env-check --config configs/traning.yaml`
   检查配置要求的设备与共享坐标标定。
9. `PYTHONPATH=src python -m start coordinate-audit --config configs/traning.yaml`
   复算版本化控制点证据。
10. `PYTHONPATH=src python -m start modules` 查看当前活动模块入口。

## traning 最短阅读路径

1. 先读 `src/traning/docs/CODEX_INDEX.md`，再读
   `src/traning/docs/TRAINING_PLAN.md`；运行 CUDA 或训练 step 前再读
   `src/traning/docs/ENVIRONMENT.md`。
2. `src/traning/config/models.py` 定义唯一严格配置；工程基线是
   `configs/traning.yaml`。
3. `src/traning/contracts` 将 `TrainingSample`、`RuntimeFrame`、训练候选、推理候选、
   belief、outcome 和 decision 从类型上隔离。
4. `src/traning/data/segments.py` 从 canonical split manifest 构建 typed dataset bundle；
   `data/cache`、`data/quality` 和 `data/repositories` 管理制品完整性及 blocking 语义。
5. 正式因果 runtime 位于 `src/traning/app/runtime.py`：Perception → Tracking → Belief →
   Outcome → Decision；最终动作不读取 GT、oracle label 或 imitation logits。
6. `src/traning/training/production_stages.py` 在真实 typed 数据上执行 Perception、Tracking、
   Belief、Outcome、Decision、Evaluation 六阶段门禁。
7. `src/traning/training/production.py` 负责恢复搜索、持续提案和 checkpoint 发布；普通门禁
   失败会继续选择未重复参数，只有全通过、显式耗尽或 blocking/异常才终止。
8. `src/traning/evaluation` 是唯一 canonical scoring/attribution 边界；训练 target、评分和
   gallery 必须共享 `FrameCoordinateTransform`，禁止只在渲染端补偏移。
9. `src/traning/telemetry` 写入 metrics/resources/evaluation/events JSONL；
   `src/traning/visualization` 只消费不可变快照，不修改训练状态。
10. 可直接运行模型 CLI：
    `PYTHONPATH=src python -m traning.app train --config configs/traning.yaml`。
11. 定位符号优先运行 `python project_index/build_index.py --lookup 符号名`。

## before_traning 最短阅读路径

1. 先读 `src/before_traning/docs/CODEX_INDEX.md`，确认阶段、分层和改动影响面。
2. 运行 `python project_index/build_index.py --lookup 符号名` 定位实现。
3. 只按索引命中的源码行和相邻模块扩展阅读。

## 索引维护

- 生成脚本：`project_index/build_index.py`
- 重建命令：`python project_index/build_index.py`
- 校验命令：`python project_index/build_index.py --check`
- 两个模块的 `docs/CODEX_INDEX.md` 都是生成文件，不要手工编辑。
- 模块架构、阶段、配置、状态或文件契约变化时，更新生成脚本中的导航内容并重建。
- 新增、删除或移动模块入口时，更新本文件。
