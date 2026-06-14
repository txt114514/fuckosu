# Core 层职责说明

## 总览

当前调用链收敛为：

```text
main.py
  -> core/flows/pipeline.py
    -> core/tasks/*.py
      -> core/beatmap|video|audio/*.py
        -> Lib/beatmap|video|audio/*.py
```

默认 direct runner 会跳过 Prefect engine，直接调用 `core/beatmap` 和 `core/video` 的执行函数。启用 Prefect 时，`core/flows` 会调用 `core/tasks`。

## core/tasks

`core/tasks` 是 Prefect 包装层。

它的职责很窄：

- 给每个阶段声明一个 `@task`
- 保留任务名、重试次数等工作流元信息
- 调用对应的 core 执行函数

它不解析谱面、不处理视频、不读写状态细节，也不直接调用 `Lib`。

对应关系：

```text
core/tasks/importer.py    -> core/beatmap/importer.py
core/tasks/verify.py      -> core/beatmap/verify.py
core/tasks/difficulty.py  -> core/beatmap/difficulty.py
core/tasks/match.py       -> core/video/match.py
core/tasks/av.py          -> core/video/av.py
core/tasks/clip.py        -> core/video/clip.py
core/tasks/segment.py     -> core/video/segment.py
```

## core/beatmap

`core/beatmap` 是谱面业务执行层。

它负责把配置对象 `Settings` 转成核心处理器需要的参数，然后调用 `Lib/beatmap`：

- `importer.py`: 从 osu 导出包导入 `.osz/.osu/audio.mp3`
- `verify.py`: 解析谱面 hit objects 并导出 `verify.txt`
- `difficulty.py`: 读取谱面难度并写入 SQLite manifest
- `pipeline.py`: 组合谱面相关阶段，供直接调用或测试使用

它不实现底层解析算法。真正的核心数据处理保留在 `Lib/beatmap`。

## core/flows

`core/flows` 是流程编排层。

它负责决定哪些阶段执行、失败时是否继续、以及 direct runner 和 Prefect runner 的调用顺序：

- `train_pipeline`: Prefect flow，调用 `core/tasks`
- `train_pipeline_direct`: 普通 Python 调用，直接调用 `core/beatmap` 和 `core/video`
- `TemporaryTrainingRunner`: 旧入口兼容包装
- `PIPELINE_STAGES`: 七阶段查找表，统一保存开关位置、direct 入口和 Prefect task

它不直接处理文件内容，也不直接调用 `Lib`。

## Lib

`Lib` 是核心数据处理层。

它保留真正有价值的底层处理器：

- `Lib/beatmap`: 谱面导入、谱面解析、难度导出、包目录管理
- `Lib/video`: 视频匹配、AV 对齐、裁剪和按谱面切分训练片段
- `Lib/audio`: 音频匹配
- `Lib/common`: 批处理、失败定位和 PathSpec 工具
- `Lib/media`: ffmpeg/ffprobe 调用

原则是：`Lib` 负责“怎么处理数据”，`core` 负责“什么时候调用哪个处理器”。
