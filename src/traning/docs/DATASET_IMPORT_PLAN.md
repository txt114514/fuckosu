# Dataset Import Plan

## 模块定位

源码入口：`src/traning/core/dataset_import`

训练集导入模块负责把 `before_traning` 产出的片段目录转换为训练可消费的样本流。它只做
数据发现、校验、采样、解码和 batch 组装，不承担模型训练、评分或导出。

## 输入契约

默认数据结构：

```text
training_package/video_segments/
  item_000001/
    segments.csv
    single_point/
      segment_.../
        video.mp4
        beatmap.json
```

每个样本目录必须同时包含：

- `video.mp4`：无音频依赖，训练只读取 RGB 帧。
- `beatmap.json`：对象时间、圆心、slider path、spinner 时间窗和 difficulty 派生参数。

osu! 原始 playfield 坐标为 `512 x 384`。视频像素坐标转换使用
`package.OsuVideoTransform` 和 `traning.lib.coordinates.transform_from_settings_or_sample`。
正式链路优先读取 `coordinate_transform.mode=explicit_rect` 和 `playfield_rect`
（`left/top/width/height`，原始视频像素坐标，裁剪/缩放前）；兼容旧数据时必须显式
声明 `legacy_centered`，并在候选缓存、评分和图集 manifest 中记录 transform source。

## 配置入口

配置由 `traning.conf.Settings` 加载，常用字段：

- `data_input.dataset_root`
- `data_input.split_manifest_path`
- `data_input.train_items`
- `data_input.validation_items`
- `data_input.test_items`
- `data_input.include_items`
- `data_input.exclude_items`
- `data_input.sample_fps`
- `data_input.frame_step`
- `data_input.max_segments`
- `data_input.max_frames_per_segment`
- `data_input.visibility_post_ms`
- `coordinate_transform.version`
- `coordinate_transform.mode`
- `coordinate_transform.playfield_rect`
- `loader.batch_size`
- `loader.num_workers`
- `loader.pin_memory`

配置文件中的相对路径按配置文件所在目录解析，也支持 `OSU_TRAINING_` 环境变量覆盖。

`split_manifest_path` 默认可省略，此时使用 `dataset_root.parent / "splits/dataset_split_manifest.json"`。
该清单由 `package.dataset_split.sync_dataset_split_manifest` 维护，启动流程每次都会尝试同步：

- 已登记 item 的 split 不自动改变。
- 新增 item 按旧配置中的 `train_items` / `validation_items` / `test_items` 优先 bootstrap。
- 其余新增 item 按比例补齐，默认 `train/validation/test = 0.8/0.1/0.1`。
- 默认 `allow_test_growth=false`，因此新增 item 只自动进入 train 或 validation，test 需要显式允许或手动维护。

## 当前实现

- `discover_segments` 发现 `video.mp4` / `beatmap.json` 配对并生成稳定 `SegmentRecord`。
- `discover_data_input` 在 split manifest 存在时优先按 manifest 的 train/validation/test item 列表过滤；
  manifest 缺失时兼容旧配置中的 `train_items` / `validation_items` / `test_items`。
- `inspect_data_input` 输出 `DataInputReport`，包含片段数量、估算帧数、类别、维度、细分
  分布统计和 slider 拓扑问题列表。
- `SegmentFrameDataset` 按 `sample_fps`、`frame_step` 和可选上限建立帧引用。
- `VideoReader` 通过 OpenCV 有限缓存读取 RGB 帧。
- `visible_hit_objects` 按当前 timestamp 和 `visibility_post_ms` 筛选可见对象。
- `collate_frame_samples` 组装图像 batch，同时保留可变长度对象标签和样本元数据。

## 样本输出

每个 frame sample 至少应保留：

- `image`：原分辨率 RGB CHW Tensor，默认归一化到 `[0, 1]`。
- `objects`：当前帧可见对象，保留 `source_index` 以便稳定回溯。
- `timestamp_ms` / `frame_index`：时序训练、候选缓存和图集定位的共同索引。
- `difficulty`：`circle_radius_osu_pixels`、`approach_preempt_ms` 等派生监督所需字段。
- `segment` / `sample_key`：用于图集、候选缓存和错误追踪的稳定身份。

## 与其他模块的边界

- 空间模块负责把对象标签光栅化为 dense target；导入模块只提供原始对象和 difficulty。
- 时间模块负责动作序列语义；导入模块只提供可因果派生的时间字段。
- 结果导出使用 `sample_key + frame_index` 稳定定位帧，不能依赖本次 Dataset 顺序。
- 数据过滤只做明显缺失、交叉/分叉 slider 等契约问题的标记，不在这里做模型质量判断。

## 已接入的扩展

- data check 已报告 slider 自交、接触分叉、退化路径、极短 slider、异常 repeat 和越界。
- `DataInputReport.distribution` 已保存 spinner 数量/持续时间、long sequence 数量、
  slider repeat 分布、对象密度、最小对象间隔、高密度窗口、长 slider 和 segment 覆盖。
- `traning.state.versioning.version_manifest` 固化 dataset、evaluation dataset、配置、
  transform 和 code 版本，写入候选缓存、评分和图集。
- 在启动报告中持续展示 split manifest 的新增 item、各 split 数量和 dry-run/no-op 状态。
