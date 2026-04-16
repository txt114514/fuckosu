# get_training_data 使用说明

这份文档说明下面这几个模块和类应该怎么用：

- `get_check_data/check_data_main.py`
- `get_check_data/export_verify.py`
- `get_check_data/get_files.py`
- `process_status_manager.py`
- `video_clip/read_video.py`
- `video_clip/AV_correspondence.py`
- `video_clip/clip.py`
- `../data_class_manager/data_osu_original.py`
- `../data_class_manager/data_type_group.py`

如果你只是想快速跑完整流程，优先看“推荐执行顺序”。

## 1. 这套代码在做什么

这套代码主要是在整理训练数据目录，目标是把 `.osz` / `.osu` / 视频文件整理到统一的 `target_root` 目录中，并为每个谱面文件夹生成：

- `verify.txt`
- `difficulty.txt`
- `audio.mp3`
- `process_status.json`

其中：

- `.osz` 导入负责建立 `order.txt` 和对应谱面文件夹
- `verify.txt` 导出负责把 `.osu` 的 HitObjects 转成文本结构
- `difficulty.txt` 导出负责提取 `OverallDifficulty`
- 音频导入会把 `.osu` 对应的 `audio.mp3` 放进谱面目录
- 视频匹配负责把视频按时间顺序移动到对应谱面目录

## 2. 目录和配置

默认配置文件在：

- `src/Traning/Lib/get_training_data/config.json`

当前配置项为：

```json
{
  "check_data": {
    "export_dir": "../../../../osu-lazer/exports",
    "target_root": "../../../../training_package/match-completed_package",
    "keyword": "normal",
    "process_steps": [
      "osu_imported",
      "audio_imported",
      "verify_exported",
      "difficulty_exported",
      "video_matched",
      "av_corresponded",
      "video_processed"
    ]
  },
  "video_shared": {
    "video_suffixes": [".mp4", ".webm", ".mkv", ".avi", ".mov"],
    "audio_filename": "audio.mp3",
    "output_filename": "video_processed.mp4",
    "status_step": "av_corresponded"
  },
  "video_match": {
    "video_root": "../../../../training_package/video_package"
  },
  "av_correspondence": {
    "failed_filename": "av_correspondence_failed.txt",
    "required_steps": ["audio_imported", "video_matched"],
    "sample_rate": 8000,
    "envelope_hz": 100,
    "refine_hz": 1000,
    "refine_search_seconds": 1.5
  }
}
```

含义：

- `export_dir`: `.osz` 导出目录
- `target_root`: 训练数据输出目录
- `keyword`: 从 `.osz` 解包后筛选 `.osu` 文件名时使用的关键字
- `video_shared`: 多个视频相关文件共用的参数
- `video_match`: `read_video.py` 单独使用的参数
- `av_correspondence`: `AV_correspondence.py` 单独使用的参数

注意：

- 这些脚本优先读取 `config.json`
- 如果配置缺失或格式不对，会打印报错并回退到代码里的默认路径

## 3. 推荐执行顺序

推荐按下面顺序操作：

1. 先运行 `CheckDataPipeline`
2. 确认每个谱面目录已经生成 `.osu`
3. 再运行 `VideoPackageRenamer`
4. 然后运行 `AVCorrespondenceProcessor` 做音视频对齐和裁剪
5. 如果还需要额外的固定区域裁剪，再单独使用 `clip.py`

对应关系可以理解成：

1. `OsuOszProcessor` 负责“建目录 + 导入 `.osu`”
2. `VerifyExporter` 负责“生成 `verify.txt`”
3. `DifficultyFileManager` 负责“生成 `difficulty.txt`”
4. `VideoPackageRenamer` 负责“把视频按顺序放进去”
5. `AVCorrespondenceProcessor` 负责“对齐音频并裁出最终视频”

## 4. 核心类怎么用

### 4.1 CheckDataPipeline

文件：

- `get_check_data/check_data_main.py`

作用：

- 整合整个检查数据流程
- 是最推荐直接调用的入口类

它内部会按阶段调用：

- `OsuOszProcessor`
- `VerifyExporter`
- `DifficultyFileManager`

最常见用法：

```python
from Traning.Lib.get_training_data.get_check_data.check_data_main import CheckDataPipeline

pipeline = CheckDataPipeline()
pipeline.run(
    overwrite=False,
    run_get_files=True,
    run_verify_export=True,
    run_difficulty_export=True,
)
```

参数说明：

- `overwrite=False`
  - 已存在的 `verify.txt` / `difficulty.txt` 不强制覆盖
- `run_get_files=True`
  - 是否先从 `.osz` 导入 `.osu`
- `run_verify_export=True`
  - 是否导出 `verify.txt`
- `run_difficulty_export=True`
  - 是否导出 `difficulty.txt`

适合场景：

- 首次建立训练数据
- 想一次性跑完整个导入流程

补充：

- `list_difficulties()` 可以按难度范围筛选已有结果

示例：

```python
items = pipeline.list_difficulties(min_difficulty=4.0, max_difficulty=5.5)
for item in items:
    print(item.folder_name, item.difficulty_value)
```

### 4.2 OsuOszProcessor

文件：

- `get_check_data/get_files.py`

作用：

- 扫描 `export_dir` 下所有 `.osz`
- 解压每个 `.osz`
- 找到文件名中包含 `keyword` 的 `.osu`
- 按 `.osz` 修改时间顺序重建 `order.txt`
- 在 `target_root` 下创建对应文件夹并写入 `.osu`

最常见用法：

```python
from Traning.Lib.get_training_data.get_check_data.get_files import OsuOszProcessor

processor = OsuOszProcessor(
    export_dir="你的 .osz 目录",
    target_root="训练输出目录",
    keyword="normal",
)
processor.run()
```

重要规则：

- `order.txt` 是唯一可信索引
- 只有 `order.txt` 里登记的文件夹才允许被后续逻辑使用
- 文件夹顺序严格按 `.osz` 的时间顺序建立

运行结果：

- 重建 `target_root/order.txt`
- 在 `target_root/<谱面名>/` 下写入目标 `.osu`
- 在同一目录下写入对应的 `audio.mp3`
- 更新每个目录里的 `process_status.json`，把 `osu_imported` 和 `audio_imported` 标记为完成

容易失败的情况：

- `.osz` 不是合法压缩包
- `.osz` 里找不到包含关键字的 `.osu`
- 多个 `.osz` 最终得到相同的 `osu_base_name`

### 4.3 VerifyExporter

文件：

- `get_check_data/export_verify.py`

作用：

- 读取 `.osu`
- 解析 `TimingPoints`、`HitObjects`
- 把 Circle / Slider / Spinner 转成文本，写入 `verify.txt`

最常见用法：

```python
from Traning.Lib.get_training_data.get_check_data.export_verify import VerifyExporter
from Traning.Lib.traning_package_manager.order_walker import OrderFolderWalker
from Traning.Lib.traning_package_manager.files_manager import BeatmapFolderStore

target_root = "训练输出目录"
walker = OrderFolderWalker(target_root=target_root, order_filename="order.txt")
store = BeatmapFolderStore(target_root=target_root, order_filename="order.txt")

exporter = VerifyExporter(walker=walker, store=store)
exporter.run(overwrite=False)
```

输出示例：

```txt
Circle(1234, 1234, 256.0, 192.0)
Slider(1500, 2200, [(256.0, 192.0), (300.0, 220.0)], 2)
Spinner(3000, 4500)
```

注意：

- 目前只支持 `osu!standard`，也就是 `Mode=0`
- 不支持 `mania hold note`
- `Slider` 的结束时间会结合 `TimingPoints` 和 `SliderMultiplier` 自动计算

跳过逻辑：

- 目录不存在
- 没有 `.osu`
- 已经存在 `verify.txt` 且状态里 `verify_exported=True`，并且 `overwrite=False`

### 4.4 DifficultyFileManager

文件：

- `Traning/Lib/traning_package_manager/difficulty_manager.py`

作用：

- 从 `.osu` 的 `[Difficulty]` 段里提取 `OverallDifficulty`
- 写入 `difficulty.txt`

最常见用法：

```python
from Traning.Lib.traning_package_manager.files_manager import BeatmapFolderStore
from Traning.Lib.traning_package_manager.difficulty_manager import DifficultyFileManager

store = BeatmapFolderStore(target_root="训练输出目录", order_filename="order.txt")
manager = DifficultyFileManager(store=store)
manager.run(overwrite=False)
```

输出结果：

```txt
5.3
```

你也可以单独读取：

```python
value = manager.read_difficulty("某个谱面目录名")
print(value)
```

### 4.5 VideoPackageRenamer

文件：

- `process_status_manager.py`
- `video_clip/read_video.py`
- `video_clip/AV_correspondence.py`

注意这个文件名容易误导。

它不是“读取视频帧”的类，而是“把视频文件按时间顺序匹配到谱面目录”的类。

作用：

- 扫描 `video_root` 下的视频文件
- 从视频文件名里提取时间
- 读取 `order.txt`
- 找出当前还没有视频的谱面目录
- 按顺序把视频移动并重命名为：
  - `target_root/<folder_name>/<folder_name>.mp4`
  - 或原视频对应后缀

最常见用法：

```python
from Traning.Lib.get_training_data.video_clip.read_video import VideoPackageRenamer

renamer = VideoPackageRenamer(
    video_root="视频目录",
    target_root="训练输出目录",
)
renamer.run()
```

它要求视频文件名必须严格使用下面这种格式：

```txt
osu_2026-04-16_19-54-54.mkv
```

支持的视频后缀：

- `.mp4`
- `.webm`
- `.mkv`
- `.avi`
- `.mov`

重要限制：

- 视频数量必须和“还没有视频的谱面目录数量”完全一致
- 目标目录必须已经存在
- 如果目录里已经有视频，会自动视为已匹配
- 运行过程中如果中途失败，会尝试回滚

### 4.6 AVCorrespondenceProcessor

文件：

- `video_clip/AV_correspondence.py`

作用：

- 先从视频里抽出音频
- 再和谱面目录中的 `audio.mp3` 做对齐
- 计算视频相对歌曲的起始偏移
- 裁出一个与歌曲音频等长且对齐的 `video_processed.mp4`

最常见用法：

```python
from Traning.Lib.get_training_data.video_clip.AV_correspondence import AVCorrespondenceProcessor

processor = AVCorrespondenceProcessor(target_root="训练输出目录")
processor.run(overwrite=False)
```

输出结果：

- 每个谱面目录新增 `video_processed.mp4`
- `process_status.json` 中的 `av_corresponded` 会被标记完成

注意：

- 目录里必须已经有源视频和 `audio.mp3`
- 源视频文件名默认要求是 `<folder_name>.<视频后缀>`
- 如果视频本身比歌曲开始得更晚，或者时长不够覆盖整首歌，会直接报错

### 4.6 clip.py

文件：

- `video_clip/clip.py`

作用：

- 这是一个独立脚本，不是类
- 使用 OpenCV 按固定矩形裁剪视频

当前脚本写死了这些参数：

- 输入文件：`input.mp4`
- 输出文件：`cropped.mp4`
- 裁剪区域：`left, top, right, bottom = 101, 199, 1848, 1197`

适合场景：

- 你已经有视频，但想统一裁掉边缘区域

使用前通常要先手改这几个变量：

```python
video_path = "input.mp4"
out_path = "cropped.mp4"
left, top, right, bottom = ...
```

## 5. 数据类怎么理解

### 5.1 OsuOriginalTimingPoint

文件：

- `../data_class_manager/data_osu_original.py`

这是一个纯数据类，对应 `.osu` 里的一个 `TimingPoint`。

字段：

- `time`
- `beat_length`
- `meter`
- `sample_set`
- `sample_index`
- `volume`
- `uninherited`
- `effects`

主要用途：

- 被 `VerifyExporter` 用来计算 Slider 实际结束时间

### 5.2 HitObject / Circle / Slider / Spinner

文件：

- `../data_class_manager/data_type_group.py`

这些类是 `verify.txt` 导出时使用的中间对象。

基类：

- `HitObject`
  - `t_start`
  - `t_end`
  - `type`

子类：

- `Circle`
  - 额外字段：`x`, `y`
- `Slider`
  - 额外字段：`path`, `repeats`
- `Spinner`
  - 没有额外字段

示例：

```python
from Traning.Lib.data_class_manager.data_type_group import Circle, Slider, Spinner

circle = Circle(1000, 1000, 256, 192)
slider = Slider(1200, 1800, [(256, 192), (300, 220)], 1)
spinner = Spinner(2000, 3000)
```

## 6. 运行后会生成什么

假设某个谱面目录名叫 `sample_map`，那么最终目录可能是：

```txt
target_root/
├── order.txt
├── verify_failed.txt
├── difficulty_failed.txt
└── sample_map/
    ├── 某个 normal .osu
    ├── audio.mp3
    ├── verify.txt
    ├── difficulty.txt
    ├── process_status.json
    ├── sample_map.mp4
    └── video_processed.mp4
```

其中 `process_status.json` 里会记录这些步骤：

- `osu_imported`
- `audio_imported`
- `verify_exported`
- `difficulty_exported`
- `video_matched`
- `av_corresponded`
- `video_processed`

## 7. 最常见的实际调用方式

### 方式一：一次性跑谱面导入和文本导出

```python
from Traning.Lib.get_training_data.get_check_data.check_data_main import CheckDataPipeline

pipeline = CheckDataPipeline()
pipeline.run()
```

### 方式二：只重建 verify.txt

```python
from Traning.Lib.get_training_data.get_check_data.check_data_main import CheckDataPipeline

pipeline = CheckDataPipeline()
pipeline.run(
    run_get_files=False,
    run_verify_export=True,
    run_difficulty_export=False,
    overwrite=True,
)
```

### 方式三：只匹配视频

```python
from Traning.Lib.get_training_data.video_clip.read_video import VideoPackageRenamer

renamer = VideoPackageRenamer()
renamer.run()
```

## 8. 使用时要特别注意的点

- `order.txt` 非常关键，很多类都会先检查它
- `BeatmapFolderStore` 只允许操作 `order.txt` 中登记过的目录
- `VerifyExporter` 默认只取一个目录中的第一个 `.osu`
- `VideoPackageRenamer` 依赖视频文件名中的中文时间格式
- `clip.py` 不是通用工具类，而是一个需要手动改参数的小脚本

## 9. 简单建议

如果你后面准备继续扩展这套代码，比较建议这样分工理解：

- 想跑主流程，用 `CheckDataPipeline`
- 想单独导入 `.osz`，用 `OsuOszProcessor`
- 想单独生成判定对象文本，用 `VerifyExporter`
- 想单独生成难度，用 `DifficultyFileManager`
- 想给谱面目录配视频，用 `VideoPackageRenamer`
- 想做固定区域裁剪，用 `clip.py`
