# traning 当前进度总览

本文是 `src/traning` 的当前状态说明，面向人和 Codex 共同使用。源码导航仍以
[`CODEX_INDEX.md`](CODEX_INDEX.md) 为准；阶段模块计划拆分为六个 plan：

- [`DATASET_IMPORT_PLAN.md`](DATASET_IMPORT_PLAN.md)
- [`SPATIAL_PLAN.md`](SPATIAL_PLAN.md)
- [`TEMPORAL_PLAN.md`](TEMPORAL_PLAN.md)
- [`DECISION_PLAN.md`](DECISION_PLAN.md)
- [`RESULT_EXPORT_PLAN.md`](RESULT_EXPORT_PLAN.md)
- [`MODEL_EXPORT_PLAN.md`](MODEL_EXPORT_PLAN.md)

环境与 CUDA 运行约束见 [`ENVIRONMENT.md`](ENVIRONMENT.md)。
训练启动前的实测检查和下一步命令见 [`TRAINING_READINESS.md`](TRAINING_READINESS.md)。
训练结果评分、归因和参数调整闭环见 [`OPTIMIZATION_MODULE.md`](OPTIMIZATION_MODULE.md)。
`traning` 内部复用层整理见 [`LIB_STRUCTURE_AUDIT.md`](LIB_STRUCTURE_AUDIT.md)。
`src` 总入口与完整训练启动前自检见 [`../../start/README.md`](../../start/README.md)。

## 项目目标

本项目从 osu! 游戏视频中逐帧识别当前应执行的动作，输出：

- 是否操作：`no-op / press / hold / release`
- 操作坐标：osu! playfield 坐标系下的 `x, y`
- 目标类型：`circle / slider / spinner`
- slider 路径、持续时间和必要的重复状态

部署时只能看到当前帧和历史状态，因此完整链路必须保持因果：

```text
当前帧 + 上一时刻状态
  -> 空间识别与候选提取
  -> 时间状态更新
  -> 决策输出动作和坐标
```

## 当前目录分层

```text
main.py
  -> ../start                 # src 顶层入口登记、渐进检查、训练启动前自检
  -> core/dataset_import     # 训练集导入、检查、Dataset/DataLoader
  -> core/spatial            # 空间训练与单帧推理
  -> core/temporal           # 候选缓存窗口、因果时序训练 smoke
  -> core/decision           # 候选缓存、评分、阶段编排
  -> core/optimization       # 训练结果评分、错误归因、参数搜索修改
  -> core/result_export      # 标注预览、评估图集导出
  -> core/model_export       # 模型导出与迁移边界
  -> lib                     # traning 作用域内可复用 API
  -> conf                    # Settings、默认值和配置加载
  -> state                   # run / experiment / checkpoint / gallery schema
```

`core` 保留六个阶段目录，并新增 `optimization` 作为训练闭环控制模块。`traning` 内部公用能力放在 `lib`，配置模型与加载入口放在
`conf`。旧 `traning.Lib`、`traning.data`、`traning.models`、`traning.training`
和旧 core 路径兼容层已删除；新代码必须使用当前正式入口。

## 已实现能力

- `conf.settings`：Pydantic 配置模型、YAML 加载、环境变量覆盖和路径解析。
- `dataset_import`：扫描 `video.mp4` 与 `beatmap.json` 配对，按
  `package.dataset_split` 的持久化 manifest 生成 train/validation/test
  `SegmentFrameDataset` 和 `DataLoader`。
- `spatial`：原分辨率帧、重叠 patch、全局/局部编码、稠密空间头、单帧训练 smoke 和单帧推理。
- `decision`：离线候选缓存 JSONL/manifest，保留候选、embedding、slider polyline 和歧义标记。
- `optimization`：trial/sample 级评分聚合、空间/时间/决策归因、ASHA/TPE 参数调整计划、连续通过 gate、难例采样权重和 trial 记录执行器。
- `result_export`：单帧标注预览、最佳 trial 图集、`passed/failed` 分类和稳定输出目录。
- `state`：trial 参数、checkpoint lineage、课程阶段、gallery request 和 frame evaluation schema。
- `lib.metrics`：`point-slider-v2` 单对象评分和 `click-sequence-v1` 序列模拟底层 API。
- `lib.runtime`：CUDA/AMP/channels-last/TF32/GradScaler/显存预算和 OOM 建议统一入口。
- `start.checks`：完整训练每次启动前执行模块入口、环境、配置、设备和数据输入自检。

## 待实现或待扩展

- `temporal` 已有候选缓存 Dataset/window builder、`beatmap_action_v1` 动作标签、
  temporal checkpoint、训练 CLI 和决策导出 CLI。
- `optimization` 已能创建 trial 记录、生成低预算 job、保存 checkpoint 继承路径、输出连续通过 gate 和难例采样权重。
- `model_export` 已能导出可校验 PyTorch artifact；完整一帧推理 smoke 需要正式空间+时间 checkpoint 组合。
- 候选局部 refiner、条件歧义复查、SMET 动态拓扑属于质量增强层，首版不阻塞主链路。

## 核心技术约束

- 输入保持原始分辨率，默认约 `1484 x 846 / 60 FPS`。
- 训练目标设备是 RTX 4060 Laptop 8GB VRAM，优先用时间换显存。
- 空间训练使用 `512 x 512` 左右重叠 patch，`overlap_x/y` 建议 `96-128`。
- patch 串行前向与逐 patch backward，禁止在训练 step 中长期保留 GPU tensor 列表。
- slider 首版支持跨 patch 连续路径，但不支持两条 slider 交叉、接触后分叉或多 head 竞争同一连通分量。
- 时序模型必须因果，任何训练或评估设计都不能用未来帧修正过去输出。

## 文档阅读顺序

1. [`CODEX_INDEX.md`](CODEX_INDEX.md)：定位源码入口、符号和依赖。
2. 本文：确认当前实现和缺口。
3. 六个模块 plan：按任务所在阶段阅读。
4. [`ENVIRONMENT.md`](ENVIRONMENT.md)：运行 CUDA、GPU、smoke test 或训练前阅读。
5. [`LIB_STRUCTURE_AUDIT.md`](LIB_STRUCTURE_AUDIT.md)：调整 `src/traning/lib` 或公共 API
   边界前阅读。
6. [`../../start/README.md`](../../start/README.md)：调整整体入口或启动自检前阅读。
