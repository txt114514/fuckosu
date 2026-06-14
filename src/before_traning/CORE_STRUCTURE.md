# Core 层职责说明

## 总览

当前调用链收敛为：

```text
main.py
  -> core/beatmap/pipeline.py
    -> Lib/tasks/flows.py
      -> Lib/tasks/tasks.py
        -> core/beatmap|video|audio/*.py
        -> core/* 具体阶段处理器
          -> Lib/beatmap|video|common|tools 通用 API
          -> state 状态与 SQLModel schema
```

默认 direct runner 通过同一注册表直接调用 core 阶段函数。启用 Prefect 时，
`Lib/tasks` 根据注册表循环生成并调用 Prefect task。

## Lib/tasks

`Lib/tasks` 是可复用 task/flow API：

- `tasks.py`: `TaskSpec`、`TaskRegistry`，循环校验注册项并动态生成 Prefect task
- `flows.py`: `TaskPipeline`，循环筛选和执行 direct/Prefect 阶段
- `build_task_pipeline()`: 从任意 task 表构建可调用 Pipeline API

该目录不依赖本工程的 core 阶段或 Settings 类型。

## core/beatmap/pipeline

`core/beatmap/pipeline.py` 保存唯一的项目阶段表 `TRAINING_TASKS`。每个阶段只登记一次：
阶段 key、core 调用、CLI 覆盖 key、默认开关路径和 Prefect 参数。

`TRAINING_PIPELINE` 由 `build_task_pipeline()` 构建，同时提供 `run_direct()`、
`run_prefect()` 和统一的可调用接口。`BEATMAP_TASK_KEYS` 与 `VIDEO_TASK_KEYS`
作为分组表供子流程复用。

## core/beatmap

`core/beatmap` 是谱面业务执行与处理器层。

它固定只保留五个源码文件：

- `importer.py`: 完整导入 processor、兼容 builder 和阶段函数
- `verify.py`: 完整 verify processor、兼容 builder 和阶段函数
- `difficulty.py`: 完整难度 processor、查询 API 和阶段函数
- `pipeline.py`: 通过 `BEATMAP_TASK_KEYS` 分组表选择阶段
- `beatmap.py`: 统一公开上述 API

报错状态回写、执行顺序检查、目录/文件检查不在这五个文件中重复实现，
统一调用 `Lib/common/processing.py` 的 `ProcessingGuard`。

原始 `.osz/.osu` 读取、标准谱面解析、目录约束和 SQLite manifest API
保留在 `Lib/beatmap`。

## core/video 与 core/audio

- `matching/*`: 视频和音频匹配的具体候选收集、选择、移动及回滚
- `av_processing/*`: AV 对齐阶段的前置检查、单项执行、状态与收尾
- `clipping/*`: 固定区域裁剪阶段实施
- `segment.py`: 切分参数映射、分类调度、产物写入和状态推进

这些模块复用 `Lib` 中的 AV 信号算法、裁剪几何、切分规划、ffmpeg、
受控目录与 SQLite 数据集 API。

## Lib

`Lib` 是可复用数据 API 与算法层。

它只保留与具体阶段状态和 Settings 解耦的能力：

- `Lib/beatmap`: `.osz/.osu` 解析、HitObject 模型、标准谱面缓存、包目录与 manifest API
- `Lib/video`: AV 信号算法、裁剪几何、切分规划和片段数据集 SQLite API
- `Lib/common`: 通用批处理、失败定位、处理前置/文件检查、序列命名和 PathSpec
- `Lib/tasks`: 通用 task 注册器和 direct/Prefect flow 执行 API
- `Lib/tools`: ffmpeg/ffprobe 参数与音频提取、裁切、分段、裁剪 API

`Lib` 不依赖 `core`、阶段 Settings 或 `ProcessStatusManager`。原则是：
`Lib` 提供“可复用能力”，`core` 决定“本阶段如何组合能力并推进状态”。
