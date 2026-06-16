# before_traning

`before_traning` 用于把 osu!standard 谱面包和录屏处理为可训练的视频片段数据集。
流程覆盖谱面导入、谱面校验、难度提取、视频匹配、音画对齐、画面裁剪和按
HitObject 切分。

## 运行

在仓库根目录执行：

```bash
PYTHONPATH=src python src/before_traning/main.py run
```

也可以单独执行常用阶段：

```bash
PYTHONPATH=src python src/before_traning/main.py verify
PYTHONPATH=src python src/before_traning/main.py match
PYTHONPATH=src python src/before_traning/main.py clip
```

默认使用 direct runner。需要 Prefect 引擎时：

```bash
TRAINING_PREFECT_ENGINE=1 PYTHONPATH=src python src/before_traning/main.py run
```

命令和参数以 CLI 帮助为准：

```bash
PYTHONPATH=src python src/before_traning/main.py --help
```

## 配置

默认配置位于 `src/before_traning/conf/config.yaml`，由
`src/before_traning/conf/settings.py` 中的 Pydantic 模型加载。配置文件中的相对路径
会相对仓库解析，也可以通过 `TRAINING_` 前缀的环境变量覆盖受支持字段。

常用运行设置包括：

- `target_root`：训练包目录
- `overwrite`：是否覆盖已有产物
- `continue_on_error`：单项失败后是否继续
- `global_offset_ms`：AV 对齐的全局时间修正
- `parameters.segment.pre_context_jitter_seconds`：片段首目标前置时间的稳定抖动范围
- `parameters.segment.include_audio`：segment `video.mp4` 是否保留音频，默认不保留

## 输入位置

默认输入路径由 `conf/config.yaml` 的 `file_management` 配置：

| 输入 | 配置字段 | 默认仓库内位置 |
|---|---|---|
| osu! 谱面导出包 | `export_dir` | `osu-lazer/exports/` |
| 原始录屏视频 | `video_root` | `training_package/video_package/` |

谱面导入阶段扫描 `export_dir` 下的 `.osz` 文件，从中提取文件名包含配置项
`file_formats.keyword`（默认 `normal`）的 `.osu` 谱面及其音频。当前流程不会直接扫描
一个独立的原始 `.osu` 输入文件夹。

导入后的 `.osu` 和音频位于：

```text
training_package/match-completed_package/
  item_000001/
    <谱面名称>.osu
    audio.mp3
  item_000002/
    <谱面名称>.osu
    audio.mp3
```

该目录由 `target_root` 控制。原始视频支持 `.mp4`、`.webm`、`.mkv`、`.avi` 和
`.mov`；匹配成功后会从 `video_root` 移入对应的 `target_root/item_xxxxxx/` 目录。

## 处理流程

```text
import_beatmaps
  -> verify_export
  -> difficulty_export
  -> video_match
  -> av_correspondence
  -> clip
  -> video_segment
```

默认 direct runner 与 Prefect runner 共用同一阶段注册表，因此阶段顺序、开关和结果语义
保持一致。

## 数据与产物

- `.package_manifest.sqlite`：谱面目录、处理顺序、难度和谱面解析缓存
- `manifest.csv`：编号与原谱面名称的只读对照表
- `.process_status.sqlite`：各谱面的阶段状态和失败详情
- `verify.txt`：供 AV 校正读取的内部谱面时间产物
- `video_processed.mp4`：完成 AV 对齐和固定区域裁剪的视频
- `.segment_manifest.sqlite`：视频片段数据集索引
- `segments.csv`：片段数据集的可读索引
- `beatmap.json`：每个片段的相对时间标签、HitObject 和谱面参数

原子片段按 `single_point`、`multi_point`、`slider`、`point_slider`、`spinner`
分类。启用长序列构建后还会生成 `long_sequence`，它只组合完整原子片段，不拆分
已有高重合组。

片段起点默认不是固定落在首个目标前同一个时间差，而是在
`approach_preempt_ratio` 计算出的基础前置时间上加入确定性抖动。这样同一谱面对象组重跑
时保持可复现，但不同片段不会给模型提供“固定第几毫秒必然点击”的捷径。生成的
segment `video.mp4` 默认使用 `-an` 去掉音频；训练阶段应只依赖画面和 `beatmap.json`
标签。

## 视频与 osu! 坐标转换

`beatmap.json` 中 Circle 的 `x, y` 和 Slider 的 `path` 坐标保持 osu!standard
原始坐标系：

```text
左上角 = (0, 0)
右下角 = (512, 384)
x 向右增加，y 向下增加
```

这些值不是裁剪后视频的像素坐标。训练代码应按下面的方程转换。

### 原视频到裁剪视频

设原视频尺寸为 `W × H`，参考分辨率为 `Wr × Hr`，参考裁剪边界为
`(Lr, Tr, Rr, Br)`。代码首先计算实际裁剪边界：

```text
L = round(Lr × W / Wr)
T = round(Tr × H / Hr)
R = round(Rr × W / Wr)
B = round(Br × H / Hr)
```

其中 `round(v)` 与代码一致，按 `int(v + 0.5)` 取整。随后：

```text
Wc = R - L
Hc = B - T
```

如果 `Wc` 或 `Hc` 是奇数，会减一以满足 `yuv420p` 编码要求。原视频像素
`(x_source, y_source)` 到裁剪视频像素 `(x_crop, y_crop)`：

```text
x_crop = x_source - L
y_crop = y_source - T
```

逆变换：

```text
x_source = x_crop + L
y_source = y_crop + T
```

### osu! 坐标到裁剪视频

训练时建议把 `512 × 384` 的 osu! playfield 等比缩放并居中放入裁剪视频，避免分别
缩放 x/y 造成圆形和滑条几何畸变。设裁剪视频尺寸为 `Wc × Hc`：

```text
s  = min(Wc / 512, Hc / 384)
Ox = (Wc - 512 × s) / 2
Oy = (Hc - 384 × s) / 2

x_crop = Ox + x_osu × s
y_crop = Oy + y_osu × s
```

逆变换：

```text
x_osu = (x_crop - Ox) / s
y_osu = (y_crop - Oy) / s
```

Circle 半径和 Slider 路径上的每个点使用相同缩放：

```text
radius_crop = circle_radius_osu_pixels × s
```

默认参考裁剪参数是：

```text
参考视频：2048 × 1152
裁剪边界：(186, 178) -> (1768, 1080)
```

对于 `1920 × 1080` 原视频，实际输出为 `1484 × 846`，因此：

```text
s = 2.203125
Ox = 178
Oy = 0

x_crop = 178 + 2.203125 × x_osu
y_crop =       2.203125 × y_osu
```

上述 osu! 映射采用“playfield 在裁剪画面内等比居中”的训练约定。如果录制所用的
osu!lazer 布局、skin、窗口比例或 playfield 偏移发生变化，应先在视频中标定实际
playfield 矩形 `(Px, Py, Pw, Ph)`，再使用通用变换：

```text
x_crop = Px + x_osu × Pw / 512
y_crop = Py + y_osu × Ph / 384
```

## 目录职责

- `conf`：配置模型、默认值和兼容映射
- `core`：业务阶段、状态推进和处理器组合
- `Lib`：谱面解析、算法、受控文件/SQLite 与 ffmpeg API
- `state`：manifest、片段索引和处理状态 schema
- `main.py`：Typer CLI 入口

开发者和 Codex 使用的架构、影响面及函数调用索引见
[`CODEX_INDEX.md`](CODEX_INDEX.md)。
