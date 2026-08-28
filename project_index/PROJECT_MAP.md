# Project Map

这是仓库级全局索引。各模块的解释性文档放在对应源码目录内，本文件只维护模块入口，
避免在仓库外部重复保存模块实现说明。

## 模块索引

| 模块 | 源码目录 | 对外说明 | Codex 索引 |
|---|---|---|---|
| 用户文档 | `docs` | [`INDEX.md`](../docs/INDEX.md)、[`QUICK_START.md`](../docs/QUICK_START.md)、[`TRAINING_WORKFLOW.md`](../docs/TRAINING_WORKFLOW.md) | [`codex/INDEX.md`](../docs/codex/INDEX.md) |
| 启动入口与自检 | `src/start` | [`README.md`](../src/start/README.md) | 公开入口：`src/start/main.py` |
| OSU Decision Model V2 | `src/traning` | [`README.md`](../src/traning/README.md) | 公开入口：`src/traning/app/__init__.py`；CLI：`src/traning/app/cli.py` |
| 全局共享 API | `src/package` | [`README.md`](../src/package/README.md) | 公开入口：`src/package/__init__.py` |
| 运行环境检查 | `environment` | 环境/CUDA 诊断脚本与 Python 检查 API | 公开入口：`environment/__init__.py` |
| 训练前处理 | `src/before_traning` | [`README.md`](../src/before_traning/docs/README.md) | [`CODEX_INDEX.md`](../src/before_traning/docs/CODEX_INDEX.md) |
| 模型训练 | `src/traning` | [`TRAINING_PLAN.md`](../src/traning/docs/TRAINING_PLAN.md)、[`TRAINING_READINESS.md`](../src/traning/docs/TRAINING_READINESS.md)、[`PLAN_GAP_AUDIT.md`](../src/traning/docs/PLAN_GAP_AUDIT.md)、[`OPTIMIZATION_MODULE.md`](../src/traning/docs/OPTIMIZATION_MODULE.md)、[`LIB_STRUCTURE_AUDIT.md`](../src/traning/docs/LIB_STRUCTURE_AUDIT.md)、[`DATASET_IMPORT_PLAN.md`](../src/traning/docs/DATASET_IMPORT_PLAN.md)、[`SPATIAL_PLAN.md`](../src/traning/docs/SPATIAL_PLAN.md)、[`TEMPORAL_PLAN.md`](../src/traning/docs/TEMPORAL_PLAN.md)、[`DECISION_PLAN.md`](../src/traning/docs/DECISION_PLAN.md)、[`RESULT_EXPORT_PLAN.md`](../src/traning/docs/RESULT_EXPORT_PLAN.md)、[`MODEL_EXPORT_PLAN.md`](../src/traning/docs/MODEL_EXPORT_PLAN.md)、[`ENVIRONMENT.md`](../src/traning/docs/ENVIRONMENT.md) | [`CODEX_INDEX.md`](../src/traning/docs/CODEX_INDEX.md) |
| 中文训练可视化 | `src/visualization` | [`README.md`](../src/visualization/docs/README.md)、[`INDEX.md`](../src/visualization/docs/INDEX.md) | 公开入口：`src/visualization/lib/__init__.py` |

## 全局 API 约定

- `src/package` 只存放被 `src` 下多个顶层模块共同调用的稳定 API。
- 调用方应从 `package` 的公开入口导入，不依赖 `_` 开头的名称或内部实现模块。
- 仅服务单个模块的简单局部实现留在对应模块内部。
- 子模块开始持续扩充独立功能、形成稳定契约、隔离第三方依赖或承担跨层边界时，
  即使只有一个调用方，也应迁入 `src/package`；原模块只保留编排和适配。
- 新增全局 API 时按领域建立子模块，并在 `src/package/__init__.py` 显式导出。
- 当前跨模块稳定 API 包括 `package.contracts`、`package.checks`、
  `package.coordinates` 和 `package.dataset_split`；`package.coordinates` 统一维护
  osu、训练帧像素与模型归一化坐标的变换契约及方程指纹；该指纹包含完整变换方程和
  训练帧尺寸，用于阻止旧方程或旧尺寸产物被静默复用；`package.dataset_split` 维护
  `training_package/splits/dataset_split_manifest.json`，供 start 同步、traning 读取。
- 中文训练控制台的稳定 API 位于 `visualization.lib`；训练核心只能使用 reporter、
  gallery API 和状态 DTO，不依赖 Rich、Panel 或 `visualization.core` 内部实现。

## start 最短阅读路径

1. `src/start/modules.py` 固定 `src` 顶层模块入口。
2. `src/start/flow.py` 固定完整启动流程：before_traning 自检和新 raw-data 决策、
   可选 before_traning 更新训练集、traning 自检、渐进/全面检测、完整训练。
3. `src/before_traning/tests/startup_checks/runner.py` 是 before_traning 启动检测入口，
   `before_traning:raw_data` 返回是否存在未匹配新样本。
4. `src/traning/tests/startup_checks/runner.py` 是 traning 启动检测入口。
5. `PYTHONPATH=src python -m start modules` 查看当前入口登记。
6. `PYTHONPATH=src python -m start check --config configs/model_small_vram.yaml --device cpu`
   运行训练启动前自检。
7. `PYTHONPATH=src python -m start run --training-config configs/model_small_vram.yaml --device cpu --dry-run --test-level quick --no-before-match-probe`
   演练完整启动流程。
8. `PYTHONPATH=src:. python -m start v2 config-check --config configs/traning.yaml`
   从统一入口严格校验 V2 配置；旧默认命令与 V2 领域链路保持隔离。
9. `PYTHONPATH=src:. python -m start v2 env-check --config configs/traning.yaml`
   检查 V2 配置要求的 CUDA 与共享 affine 坐标标定。
10. `PYTHONPATH=src:. python -m start v2 coordinate-audit --config configs/traning.yaml`
   复算版本化独立控制点，并公开原始拟合集能否重放。
11. `PYTHONPATH=src:. python -m start v2 train --config configs/traning.yaml --evaluator module:factory`
   运行真实 typed evaluator；普通 trial 失败继续提案，固定数据质量阻断立即失败。

## OSU V2 最短阅读路径

1. 先读 `src/traning/README.md`，确认运行命令、失败语义、坐标指纹和 Phase 0 → 11 边界。
2. `src/traning/config/models.py` 定义单一严格配置；工程基线是 `configs/traning.yaml`。
3. `src/traning/app/cli.py` 提供 `config-check`、`coordinate-audit`、`env-check` 和
   `train`，由
   `src/start/main.py` 显式挂载到 `v2` 命名空间。
4. `src/traning/app/factory.py` 校验模型/config/设备一致性并装配正式 runtime；
   `src/traning/app/runtime.py` 固定 Perception → Tracking → Belief → Outcome → Decision。
5. `src/traning/app/training.py` 把单一 config 接到门禁驱动参数搜索；
   `optimization.max_trials: null` 表示未全通过时不施加额外 trial 截断。
   `src/traning/training/evaluator.py` 把真实阶段 runner 的普通失败映成下一轮观测，
   但固定 blocking DataQuality 会在 runner 构造前终止。
6. Phase 1 → 10 依次位于 `contracts|config|infrastructure`、`data`、`perception`、
   `tracking`、`belief`、`outcome/oracle|dataset`、`outcome/model`、`decision`、
   `training`、`telemetry|visualization`；Phase 0 是冻结代码与 golden baseline，
   Phase 11 由 `app` 完成入口与正式 runtime 收口。
7. 训练 target、canonical scoring 与 gallery 只能通过
   `build_frame_coordinate_transform` 共享同一 affine 方程及尺寸绑定指纹，禁止局部渲染补偿。
   `configs/traning_coordinate_evidence.json` 保存独立控制点和拟合来源可用性声明；当前
   `legacy-control-validated-v1` 只证明控制残差，不能冒充可重放的 passed-sample 拟合。
8. `src/traning/training/checkpoints.py` 事务发布三个模型权重，并同时门禁 dataset identity、
   model-contract SHA-256、权重摘要和坐标指纹。

## before_traning 最短阅读路径

1. 先读 `src/before_traning/docs/CODEX_INDEX.md`，确认阶段、分层和改动影响面。
2. 运行 `python project_index/build_index.py --lookup 符号名` 定位函数或模块。
3. 用 `rg -n "符号名" src/before_traning/docs/CODEX_INDEX.md` 搜索索引。
4. 只按需读取命中的源码行和索引列出的相邻模块。

## traning 最短阅读路径

1. 先读 `src/traning/docs/CODEX_INDEX.md`，确认当前阶段和数据契约。
2. 需要当前整体进度和长期目标时读取 `src/traning/docs/TRAINING_PLAN.md`。
3. 需要确认是否可以开训时读取 `src/traning/docs/TRAINING_READINESS.md`。
4. 处理训练结果评分、错误归因和参数调整时读取 `src/traning/docs/OPTIMIZATION_MODULE.md`。
5. 调整 `src/traning/lib` 或内部复用 API 边界时读取
   `src/traning/docs/LIB_STRUCTURE_AUDIT.md`。
6. 按六个 core 阶段读取对应 plan：`DATASET_IMPORT_PLAN.md`、`SPATIAL_PLAN.md`、
   `TEMPORAL_PLAN.md`、`DECISION_PLAN.md`、`RESULT_EXPORT_PLAN.md`、
   `MODEL_EXPORT_PLAN.md`。
7. 处理 CUDA、显存、AMP、channels-last、GPU bridge 或训练 step 时读取
   `src/traning/docs/ENVIRONMENT.md`。
8. 运行 `python project_index/build_index.py --lookup 符号名` 定位实现。
9. 当前 core 阶段目录：`core/dataset_import`、`core/spatial`、
   `core/temporal`、`core/decision`、`core/result_export`、`core/model_export`。
   训练闭环控制目录是 `core/optimization`，完整生命周期编排目录是
   `core/full_flow`。空间训练入口是 `core/spatial`，候选缓存与单轮阶段编排入口是
   `core/decision`。
10. 批次评估图集入口是 `save-annotation-gallery`；结果契约位于
   `state/gallery_schema.py`，按最高分 trial 和六个数据子项目生成 `passed/failed`
   标注目录。
11. 参数搜索、score、通过机制和 `point-slider-v2` / `click-sequence-v1` 边界见
   `src/traning/docs/DECISION_PLAN.md`。
12. 批次图集、`passed/failed` 和可视化导出边界见
   `src/traning/docs/RESULT_EXPORT_PLAN.md`。

## 索引维护

- 生成脚本：`project_index/build_index.py`
- 重建命令：`python project_index/build_index.py`
- 校验命令：`python project_index/build_index.py --check`
- 两个模块的 `docs/CODEX_INDEX.md` 都是生成文件，不要手工编辑。
- 模块内部架构、阶段、配置、状态或文件契约变化时，更新生成脚本中的导航内容并重建。
- 新增、删除或移动模块导航入口时，更新本文件。
