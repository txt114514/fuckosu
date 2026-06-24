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

osu! 原始 playfield 坐标为 `512 x 384`。视频像素坐标转换使用 `package.OsuVideoTransform`，
调用方不要在本模块内复制坐标换算逻辑。

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
- `inspect_data_input` 输出 `DataInputReport`，包含片段数量、估算帧数、类别、维度和问题列表。
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

## 后续计划

- 在 data check 中显式报告 slider 交叉、接触分叉和多 head 竞争同一连通分量。
- 为 `spinner`、`long_sequence` 增补更细的分布统计，便于课程采样。
- 固化数据版本、片段生成版本和坐标转换版本，写入后续 evaluation artifact。
- 在启动报告中持续展示 split manifest 的新增 item、各 split 数量和 dry-run/no-op 状态。
