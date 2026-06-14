# 训练数据流程框架改造说明

## 保留的主结构

当前主入口已经收敛到这组目录：

```text
before_traning/
  conf/
    config.yaml
    settings.py
  core/
    flows/
      pipeline.py
    tasks/
      importer.py
      verify.py
      difficulty.py
      match.py
      av.py
      clip.py
    beatmap/
      importer.py
      verify.py
      difficulty.py
      pipeline.py
    video/
      match.py
      av.py
      clip.py
      pipeline.py
    audio/
      match.py
  Lib/
    beatmap/
      importer.py
      verify.py
      difficulty.py
      folder_store.py
      order.py
      package.py
      hit_objects.py
      timing_points.py
    video/
      match.py
      av.py
      clip.py
    audio/
      match.py
    common/
      batch.py
      pathspec.py
    media/
      ffmpeg.py
  state/
    process_status.py
  main.py
```

旧的 `Main.py`、`test.py`、`Lib/get_training_data/training_main.py`、`setting_loader.py`、旧 JSON 配置和旧 README 已删除。`Lib/` 的旧分组 `get_training_data`、`function_tools`、`traning_package_manager`、`data_class_manager` 已迁移为按领域命名的核心库。

## 运行方式

```bash
PYTHONPATH=src python src/before_traning/main.py run
PYTHONPATH=src python src/before_traning/main.py verify
PYTHONPATH=src python src/before_traning/main.py match
PYTHONPATH=src python src/before_traning/main.py clip
```

默认使用 direct runner，避免当前容器里 Prefect 临时 API server 的 socket 权限问题。需要 Prefect 引擎时：

```bash
TRAINING_PREFECT_ENGINE=1 PYTHONPATH=src python src/before_traning/main.py run
```

## 配置

默认配置文件：

```text
before_traning/conf/config.yaml
```

主配置模型在 `before_traning/conf/settings.py`，由 `pydantic-settings` 管理。核心运行字段：

```python
target_root: Path
overwrite: bool = False
continue_on_error: bool = False
global_offset_ms: float = 0
```

也可以用 `TRAINING_` 环境变量覆盖 pydantic-settings 支持的字段。

## Workflow

`flows/pipeline.py` 编排这些任务：

```text
import_beatmaps
verify_export
difficulty_export
video_match
av_correspondence
clip
video_segment
```

`core/beatmap/pipeline.py` 只保存一张 `TRAINING_TASKS` 阶段表。`Lib/tasks/tasks.py`
循环生成 Prefect task，`Lib/tasks/flows.py` 用同一注册表执行 direct/Prefect flow；
不再为每个阶段维护独立 task 文件。

当前任务和业务源文件的对应关系：

```text
import_beatmaps      -> core/beatmap/importer.py
verify_export        -> core/beatmap/verify.py
difficulty_export    -> core/beatmap/difficulty.py
video_match          -> core/video/match.py
av_correspondence    -> core/video/av.py
clip                 -> core/video/clip.py
video_segment        -> core/video/segment.py
```

具体业务处理器位于 `core`，`Lib` 只提供可复用 API：

```text
core/beatmap/importer.py          -> Lib/beatmap/osz.py、package.py
core/beatmap/verify.py            -> Lib/beatmap/osu_parser.py、standard.py
core/beatmap/difficulty.py        -> Lib/beatmap/osu_metadata.py、manifest.py
core/video/matching/*             -> Lib/common/pathspec.py
core/audio/matching/*             -> Lib/video/av_processing/steps.py
core/video/av_processing/*        -> Lib/video/av_processing/steps.py、tools/ffmpeg.py
core/video/clipping/*             -> Lib/video/clipping/geometry.py、tools/ffmpeg.py
core/video/segment.py             -> Lib/video/segmentation/*、segment_dataset.py
```

这样具体流程、状态推进和 Settings 映射集中在 `core`；原始文件解析、纯算法、
受控文件/SQLite 操作和 ffmpeg 调用集中在 `Lib`。

## PathSpec 落点

新增公共工具：

```text
before_traning/Lib/common/pathspec.py
```

已接入的核心筛选点：

- `.osz` 导入扫描：`core/beatmap/importer.py`
- 压缩包内 `.osu` 与音频读取：`Lib/beatmap/osz.py`
- 谱面目录内 `.osu` 查找：`Lib/beatmap/folder_store.py`
- 视频源目录按后缀筛选：`core/video/matching/renamer.py`
- 音频匹配候选视频筛选：`core/audio/matching/preflight.py`
- AV 对齐阶段查找谱面文件夹里的源视频：`core/video/av_processing/preflight.py`

音频匹配实施位于 `core/audio/matching`，并复用
`Lib/video/av_processing/steps.py` 中的 AV 信号算法。

## 状态管理

`ProcessStatusManager` 的外部 API 保留，内部使用 SQLModel + SQLite：

```text
target_root/.process_status.sqlite
```

首次读取旧 `process_status.json` 时会迁入 SQLite。这样底层处理器和音频匹配模块不需要改方法名。

## 已实际调用 training_package

已经对当前 `training_package/match-completed_package` 做过 API 调用验证：

- `verify`: 两个谱面已有 `verify.txt`，按状态跳过，结果 success。
- `run --skip-get-files --skip-video-match --skip-av-correspondence --skip-clip`: `verify_export` 和 `difficulty_export` 成功跳过。
- `match --continue-on-error`: `video_match` 识别为所有文件夹已有视频，任务层归为 success。
- `clip --continue-on-error`: `av_correspondence` 和 `clip` 按已有完成状态跳过，结果 success。

本次拆分后又重新验证过：

- `PYTHONPATH=src python -B src/before_traning/main.py --help`
- `PYTHONPATH=src python -B src/before_traning/main.py verify`
- `PYTHONPATH=src python -B src/before_traning/main.py run --skip-get-files --skip-video-match --skip-av-correspondence --skip-clip`
- `PYTHONPATH=src python -B src/before_traning/main.py match --continue-on-error`
- `PYTHONPATH=src python -B src/before_traning/main.py clip --continue-on-error`

## 迁移边界

保留的是核心数据处理算法本身：谱面解析、难度导出、音频匹配、AV 对齐、ffmpeg 裁剪、状态写入。迁移掉的是旧的流程壳、旧配置入口、旧目录分组和双重 video init 校验入口。
