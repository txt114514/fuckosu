# Project Map

这是给 Codex 和维护者使用的低 token 工程导航。它只保存稳定的架构、调用链和修改影响面；
所有函数的实时签名、行号和关键调用由同目录下的生成索引保存。

## 最短阅读路径

1. 先读本文件，确认改动属于哪个阶段、哪一层。
2. 用 `python project_index/build_index.py --lookup 符号名` 取得最小上下文；
   或用 `rg -n "符号名" project_index/FUNCTION_LOCATIONS.md` 找精确源码位置。
3. 只读 `FUNCTION_INDEX.md` 中对应模块块，确认职责、装饰器、I/O 和关键调用。
4. 打开目标源码及本文件列出的相邻影响文件，不必重新遍历全仓库。

## 工程边界

- Python 源码根：`src/before_traning`
- CLI 入口：`src/before_traning/main.py`
- 默认配置：`src/before_traning/conf/config.yaml`
- 训练包清单：`target_root/.package_manifest.sqlite`
- 可读编号对照表：`target_root/manifest.csv`
- 状态数据库：`target_root/.process_status.sqlite`
- 谱面解析缓存：`target_root/.package_manifest.sqlite` 的 `beatmap_data_record`
- 切分索引数据库：`segment_root/.segment_manifest.sqlite`
- 切分数据集：
  `training_package/video_segments/<内部谱面编号>/<原子类别或 long_sequence>/`
- 外部工具/API：`ffmpeg`、`ffprobe`、`slider`（osu! 滑条真实曲线采样）
- 当前 Python 规模由 `build_index.py` 实时统计；生成索引覆盖全部命名函数、方法和类。

## 分层调用

```text
main.py (Typer)
  -> core/beatmap/pipeline.py 的 TRAINING_PIPELINE
       -> Lib/tasks/flows.py 循环执行
       -> Lib/tasks/tasks.py 循环注册 Prefect task
       -> core/beatmap|video/*.py 阶段入口
            -> core/* 处理器与阶段实施
                 -> Lib/* 通用解析、算法、文件/数据库和媒体 API
                 -> state/ProcessStatusManager 与 SQLModel schema
                 -> 文件系统 / SQLite / ffmpeg
```

层级约束：

- `main.py`：只处理 CLI 覆盖、runner 选择和结果显示。
- `core/beatmap`：固定只保留 `beatmap.py`、`difficulty.py`、`importer.py`、
  `pipeline.py`、`verify.py` 五个源码文件；其中 `pipeline.py` 保存七阶段注册表、
  Pipeline API 和谱面/视频阶段分组表。
- `core/video|audio`：保存具体阶段处理器、状态编排、产物目录和 Settings 映射。
- `Lib/tasks`：可复用 task 注册器与 flow 执行 API；通过循环动态生成 Prefect task，
  同一注册表同时服务 direct 和 Prefect runner。
- `Lib`：只保存可复用解析器、数据模型、算法、受控文件/SQLite API 和媒体 API；
  不得依赖 `core`、具体状态步骤或阶段 Settings。
- `state`：所有阶段共享的持久化处理状态。
- `conf`：Pydantic 模型是新配置真源；`legacy_config.py` 仅维持旧构造器兼容。

## 七阶段修改地图

| 阶段 | 注册 key | Core 入口 | Prefect 注册 | Core 主处理器 | Lib 通用 API | 主要状态 |
|---|---|---|---|---|---|---|
| 导入谱面 | `import_beatmaps` | `core/beatmap/importer.py` | `Lib/tasks` 动态生成 | `BeatmapImportProcessor` | osz、manifest/package、processing API | `osu_imported`, `audio_imported` |
| 导出校验 | `verify_export` | `core/beatmap/verify.py` | `Lib/tasks` 动态生成 | `BeatmapVerifyExporter` | parser、standard cache、processing API | `verify_exported` |
| 导出难度 | `difficulty_export` | `core/beatmap/difficulty.py` | `Lib/tasks` 动态生成 | `BeatmapDifficultyProcessor` | metadata、manifest、processing API | `difficulty_exported` |
| 匹配视频 | `video_match` | `core/video/match.py` | `Lib/tasks` 动态生成 | `core/video/matching/*`, `core/audio/matching/*` | pathspec、AV 信号算法 | `video_matched` |
| AV 对齐 | `av_correspondence` | `core/video/av.py` | `Lib/tasks` 动态生成 | `core/video/av_processing/*` | `Lib/video/av_processing/steps.py`, ffmpeg API | `av_corresponded` |
| 固定裁剪 | `clip` | `core/video/clip.py` | `Lib/tasks` 动态生成 | `core/video/clipping/*` | `Lib/video/clipping/geometry.py`, `crop_video()` | `video_processed` |
| 谱面切分 | `video_segment` | `core/video/segment.py` | `Lib/tasks` 动态生成 | `VideoSegmentationProcessor` | standard cache、planner、segment dataset、`segment_video()` | `video_segmented` |

## 关键领域

### 配置

- `conf/settings.py` 定义嵌套 Pydantic 模型，加载 YAML/JSON，并把配置文件中的相对路径解析为绝对路径。
- `conf/config.yaml` 保存运行默认值。
- `conf/defaults.py` 保存默认 Settings 实例；`conf/artifacts.py` 保存固定训练前产物名。
- `conf/legacy_config.py` 把嵌套 Settings 展平为旧 core 处理器参数；新增/改名配置字段时必须检查这里。
- `conf/field_groups.py` 集中控制参数批量赋值和处理器间转发，尤其是 AV 与音频匹配共享参数。
- `conf/runtime.py` 在导入 Prefect 前设置仓库内 `.prefect`。

### 包目录与文件约束

- `.package_manifest.sqlite` 是允许使用的谱面目录和批处理顺序的唯一真源。
- 内部目录使用稳定编号 `item_000001`；原谱面名称保存在 manifest，不参与路径识别。
- `manifest.csv` 自动生成，只有“编号、谱面名称”两列；仅用于查看，不能作为输入修改。
- `PackageUpdater` 更新 manifest 并创建内部目录。
- `BeatmapFolderStore` 只允许操作 manifest 中启用且存在的源目录；同时提供受登记约束的
  输出目录创建和 `atomic_output_folder()` 原子替换 API。
- `ManifestFolderWalker` 按 manifest sequence 返回内部目录 ID。
- 首次发现旧 `order.txt` 时会自动迁移目录、源视频和状态键，完成后删除 `order.txt`。
- 文件后缀筛选统一经过 `Lib/common/pathspec.py`。
- 通用目录/文件存在检查、前置状态顺序检查、已有产物状态对齐和失败回写统一经过
  `Lib/common/processing.py` 的 `ProcessingGuard`；core/beatmap 不自行重复这些判断。
- `Lib/common/sequence.py` 统一生成 `item_000001`、`segment_000001` 等序列名。

### 状态

- manifest 与处理状态是两张独立 SQLite 数据库；名称/顺序不进入状态表。
- `ProcessStatusManager` 的公开 API 是 `load/save/ensure/is_step_done/mark_step_done/mark_step_pending`。
- 新状态存入 SQLModel/SQLite；首次访问无数据库记录的目录时可迁移旧 `process_status.json`。
- 单项处理失败写入对应步骤的 `detail_json`；不再生成 `*_failed.txt`。
- 异常失败统一记录 `error`、`error_type`、`error_function`、`error_module`；
  traceback 定位由 `Lib/common/failures.py` 负责，并优先选择最深的工程函数帧。
- 文件存在与数据库状态可能不一致；AV、裁剪和匹配 preflight 会做局部纠正。
- 修改状态名时同时检查：
  `conf/settings.py`、`conf/config.yaml`、`state/status_schema.py`、各 preflight、各 `_mark_*` 调用和阶段表。

### 批处理

- `FolderBatchProcessor.run()` 提供标准循环：读取 manifest、显示进度、调用 `process_one`、捕获单项失败并返回整体成功状态。
- verify、difficulty、AV、clip、segment 使用该模板。
- `process_one` 返回值只允许 `"success"` 或 `"skip"`。
- core 层把批处理结果向上传递；注册器生成的 Prefect task 在结果为 `False` 时抛出异常，
  避免失败批次显示为成功。
- `core/beatmap/pipeline.py` 只维护 `TRAINING_TASKS`；`Lib/tasks` 的 direct/Prefect runner
  共用同一注册表和循环。新增阶段时增加一个 `TaskSpec` 并同步状态步骤。

### 训练前产物

- `verify.txt` 是供后续 AV 校正读取的内部训练前产物，文件名由代码固定，不属于用户配置。
- verify 阶段通过 `Lib/beatmap/standard.py` 解析完整标准谱面，并把元数据与 HitObjects
  缓存在 package manifest SQLite；后续阶段优先读取缓存，源 `.osu` 变化时自动刷新。
- 难度值保存在 `.package_manifest.sqlite` 的 `difficulty_value` 字段中，不再生成 `difficulty.txt`。
- 旧 `difficulty.txt` 只用于首次迁移：有效数值导入 manifest 后删除。

### 视频匹配

- `VideoMatchProcessor.run()` 是策略开关。
- `use_audio_match_experiment=True`：对每个视频和谱面音频计算特征相关，贪心一对一匹配并移动文件。
- `False`：按 `osu_YYYY-MM-DD_HH-MM-SS` 文件名时间与 manifest 顺序匹配。
- 两种路径都必须维护 `video_matched`，移动异常时回滚文件和状态。

### AV 对齐

- `AVCoreStepsMixin` 被 AV 裁切和音频匹配共同复用，是改动影响最大的算法模块。
- 粗匹配：低频率能量包络相关。
- 细匹配：低通音乐能量特征，在粗结果附近搜索。
- verify 校正：把 hit object 时间构造成 click train，与视频瞬态特征在小窗口内比较。
- 最终 offset = 原始音频 offset + verify 调整 + 全局 offset。
- 输出 `video_processed.mp4` 后记录完整算法参数和得分。

### 固定区域裁剪

- 裁剪坐标以参考分辨率定义，并按真实视频尺寸等比缩放。
- 输出宽高对齐为偶数以满足 `yuv420p`。
- 裁剪通过同目录临时文件完成，成功后原子替换目标视频。

### 谱面视频切分

- `video_segment` 是完整流程最后一阶段，只读取最终 `video_processed.mp4`，前置状态为 `video_processed`。
- `core/video/segment.py` 保存设置映射、分类调度、状态和产物实施；
  `Lib/video/segmentation/segmentation.py` 只提供显式参数的 `plan_video_segments()` 纯 API。
- `Lib/video/segment_dataset.py` 用 `.segment_manifest.sqlite` 管理片段记录并导出兼容的
  `segments.csv`；旧 CSV 会在首次访问时自动导入 SQLite。
- 复用 SQL 中已缓存的标准谱面数据，并保留 Slider 曲线类型与像素长度；
  使用已安装的 `slider` API 计算真实滑条路径。
- 普通聚组要求非 Spinner 对象同时满足：前后时间间隔不超过本谱面
  `approach_preempt_ms * approach_preempt_ratio`，且命中圆/滑条扫掠路径的最高圆面积
  重合率不低于 `min_circle_overlap_ratio`。默认比例与重合率均为 `0.5`。
  `min_circle_overlap_ratio` 的范围为 `0.0–1.0`，`1.0` 表示 100% 重合。
- `use_priority_merge=true` 时启用高优先级合并：下一个非 Spinner HitObject 只要落入
  当前组末尾的高优先级时间窗口，就无视坐标重合率直接合并，并可继续链式延长当前组。
- 高优先级窗口由 `priority_merge_window_ms` 设置，默认固定为 `200ms` 并默认启用。
- 圆半径按谱面 `[Difficulty] CircleSize` 和 osu! 官方缩放公式计算；Spinner 始终独立。
- 五类输出目录固定为：
  `single_point`、`multi_point`、`slider`、`point_slider`、`spinner`。
- `point_slider` 表示短间隔组中同时包含 Circle 和 Slider；Circle、Slider 均可出现多个，
  例如 `slider|slider|circle|slider` 仍属于 `point_slider`。Circle 与 Slider 判断空间重合时
  使用整条滑条路径，不限于滑条首尾。
- 原子维度中每个 HitObject 只分配给一个训练样本，规划器通过 `source_index`
  检查无遗漏、无重复；视频上下文窗口可以重叠，但重叠窗口中的其他对象不会重复写入标签。
- `build_long_sequences=true` 时额外生成 `long_sequence` 维度。它只组合完整的原子片段，
  不拆分已有高重合组，因此可复用原子维度使用过的 HitObject，但长序列维度内部不使用
  滚动窗口，同一 `source_index` 最多出现一次。
- 长序列以 Spinner 为硬边界，相邻原子片段的间隔不得超过完整 `approach_preempt_ms`；
  默认最多 `12` 个 HitObject、片段总长最多 `10.0` 秒，并且至少来自两个原子片段、
  至少包含两个 Circle 和两个 Slider，确保它是真正的多点多线样本。
- 每段从首个对象前的 `AR 缩圈时长 * approach_preempt_ratio` 开始，默认比例为 `0.5`。
- 每段在末对象结束后增加 `post_context_seconds`，默认固定为 `0.4` 秒。
- 输出位于 `segment_root/<folder_name>/`，根目录同步 `manifest.csv` 编号对照表。
  原子维度仍使用五类目录；组合维度使用 `long_sequence` 目录。每个片段单独建立目录，
  包含 `video.mp4` 和 `beatmap.json`。
- `beatmap.json` 复用 `VerifyOsuParser.hit_object_to_dict()` 生成对象结构，
  时间以片段开始为 0，并记录曲线类型、像素长度和实际聚组参数；
  谱面根目录另有 `segments.csv` 总索引。
- `segments.csv` 和每段 `beatmap.json` 同时记录谱面级训练参数：
  `HPDrainRate`、`CircleSize`、圆半径、`OverallDifficulty`、`ApproachRate`、
  按官方公式计算的 `approach_preempt_ms`、`SliderMultiplier`、`SliderTickRate`
  与 `StackLeniency`。`approach_preempt_ms` 是无模组条件下缩圈/预显持续时间。
- 输出同时记录 `dataset_dimension`、来源原子片段数、普通空间聚组、高优先级合并窗口、
  长序列连续窗口与对象/时长上限。
- 临时目录全部生成成功后由 `BeatmapFolderStore.atomic_output_folder()` 原子替换正式目录；
  初始化时会恢复或清理上次强制中断遗留目录。

## 常见改动影响面

| 改动类型 | 至少检查 |
|---|---|
| 新增 CLI 选项 | `main.py`、对应 Settings 模型、`config.yaml`、flow 参数、core 入口 |
| 新增/重命名配置字段 | `settings.py`、`config.yaml`、`legacy_config.py`、`field_groups.py`、处理器构造函数 |
| 新增流程阶段 | `main.py`、`core/beatmap/pipeline.py` 的 `TRAINING_TASKS`、core 处理器、Lib 通用 API、状态步骤 |
| 修改状态步骤 | Settings、`status_schema.py`、`ProcessStatusManager`、preflight、成功/失败回写 |
| 修改文件名或目录规则 | manifest schema/repository、Settings、core 参数映射、`PackageUpdater`/`BeatmapFolderStore`、状态 detail |
| 修改视频后缀支持 | Settings、pathspec、matching、audio matching、AV source resolve |
| 修改 AV 特征算法 | `av_processing/steps.py`、`audio/matching/steps.py`、Settings AV 参数、状态 detail |
| 修改移动/重命名逻辑 | matching renamer、audio matching wrapup、回滚路径、`video_matched` |
| 修改裁剪坐标 | Clip Settings、geometry、wrapup detail、ffmpeg crop 参数 |
| 修改谱面切分规则 | Segment Settings、`Lib/video/segmentation/planner.py`、`segmentation.py`、`core/video/segment.py` |
| 修改切分输出格式 | `Lib/video/segment_dataset.py`、`state/segment_schema.py`、`core/video/segment.py` |
| 修改批处理约定 | `common/batch.py`、`common/processing.py` 及 verify/difficulty/AV/clip/segment |

## 兼容与入口

- 默认 runner 是 direct；只有 `TRAINING_PREFECT_ENGINE=1` 才进入 Prefect flow。
- `Lib/tasks` 是通用注册与执行 API，不得导入任何 core 阶段；项目阶段只在
  `core/beatmap/pipeline.py` 注入。
- `legacy_config.py` 中的 builder 和 namespace API 仍被多个 core 兼容构造器调用，不能按“旧代码”直接删除。
- 各 core 业务包中的 `main()` 是兼容脚本入口，主流程不从这些入口启动。

## 索引维护

- `PROJECT_MAP.md`：架构、阶段、配置/状态契约变化时人工同步。
- `FUNCTION_INDEX.md`：生成的语义函数索引。
- `FUNCTION_LOCATIONS.md`：生成的极简位置索引。
- `build_index.py`：唯一生成入口，使用 Python AST，不导入项目模块，因此不会触发业务副作用。

单函数/模块的最低 token 查询：

```bash
python project_index/build_index.py --lookup AVCoreStepsMixin._estimate_offset_seconds
python project_index/build_index.py --lookup core/beatmap/pipeline.py
```

每次修改 `src/before_traning/**/*.py` 后执行：

```bash
python project_index/build_index.py
python project_index/build_index.py --check
```

`--check` 可用于提交前验证；源码新增、删除、移动函数或改变签名/调用后，旧索引会返回非零状态。
