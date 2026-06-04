# 训练数据流程框架改造说明

## 保留的主结构

当前主入口已经收敛到这组目录：

```text
Traning/
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
PYTHONPATH=src python src/Traning/main.py run
PYTHONPATH=src python src/Traning/main.py verify
PYTHONPATH=src python src/Traning/main.py match
PYTHONPATH=src python src/Traning/main.py clip
```

默认使用 direct runner，避免当前容器里 Prefect 临时 API server 的 socket 权限问题。需要 Prefect 引擎时：

```bash
TRAINING_PREFECT_ENGINE=1 PYTHONPATH=src python src/Traning/main.py run
```

## 配置

默认配置文件：

```text
Traning/conf/config.yaml
```

主配置模型在 `Traning/conf/settings.py`，由 `pydantic-settings` 管理。核心运行字段：

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
```

`core/tasks/` 现在只做 Prefect task 包装，不再承载业务细节。`core/` 是唯一直接调用 `Lib/` 核心处理器的业务执行层；direct runner 调用 `core/beatmap` / `core/video`，Prefect runner 调用 `core/tasks`，再由 `core/tasks` 调用对应的执行文件。

当前任务和业务源文件的对应关系：

```text
import_beatmaps      -> core/beatmap/importer.py    -> core/tasks/importer.py
verify_export        -> core/beatmap/verify.py      -> core/tasks/verify.py
difficulty_export    -> core/beatmap/difficulty.py  -> core/tasks/difficulty.py
video_match          -> core/video/match.py         -> core/tasks/match.py
av_correspondence    -> core/video/av.py            -> core/tasks/av.py
clip                 -> core/video/clip.py          -> core/tasks/clip.py
```

对应的 `Lib` 核心处理器：

```text
core/beatmap/importer.py   -> Lib/beatmap/importer.py    -> BeatmapImportProcessor
core/beatmap/verify.py     -> Lib/beatmap/verify.py      -> BeatmapVerifyExporter
core/beatmap/difficulty.py -> Lib/beatmap/difficulty.py  -> BeatmapDifficultyProcessor
core/video/match.py        -> Lib/video/match.py         -> VideoMatchProcessor
core/video/av.py           -> Lib/video/av.py            -> VideoAVProcessor
core/video/clip.py         -> Lib/video/clip.py          -> VideoClipProcessor
core/audio/match.py        -> Lib/audio/match.py         -> AudioMatchProcessor
```

这样每个任务都有独立的 core 执行源文件和独立的 Lib 核心库文件。旧类名仍留在核心文件内部作为兼容实现名，但新流程和 `__all__` 都使用 task 对齐的新类名。

## PathSpec 落点

新增公共工具：

```text
Traning/Lib/common/pathspec.py
```

已接入的核心筛选点：

- `.osz` 导入扫描：`Lib/beatmap/importer.py`
- 解压后 `.osu` 匹配和 `.mp3` 音频校验：`Lib/beatmap/importer.py`
- 谱面目录内 `.osu` 查找：`Lib/beatmap/folder_store.py`
- 视频源目录按后缀筛选：`Lib/video/match.py`
- 音频匹配候选视频筛选：`Lib/audio/match.py`
- AV 对齐阶段查找谱面文件夹里的源视频：`Lib/video/av.py`

音频匹配已迁移到 `Lib/audio/match.py`，并通过 `AudioMatchProcessor` 暴露。

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

- `PYTHONPATH=src python -B src/Traning/main.py --help`
- `PYTHONPATH=src python -B src/Traning/main.py verify`
- `PYTHONPATH=src python -B src/Traning/main.py run --skip-get-files --skip-video-match --skip-av-correspondence --skip-clip`
- `PYTHONPATH=src python -B src/Traning/main.py match --continue-on-error`
- `PYTHONPATH=src python -B src/Traning/main.py clip --continue-on-error`

## 迁移边界

保留的是核心数据处理算法本身：谱面解析、难度导出、音频匹配、AV 对齐、ffmpeg 裁剪、状态写入。迁移掉的是旧的流程壳、旧配置入口、旧目录分组和双重 video init 校验入口。
