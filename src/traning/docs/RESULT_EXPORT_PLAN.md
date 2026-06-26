# Result Export Plan

## 模块定位

源码入口：`src/traning/core/result_export`

结果导出模块负责把训练、评估和调参结果转换为可人工检查的图片、JSON 和目录结构。它是
评估结果消费者，不是评估器本身。

## 当前导出能力

- `visualize_click_label`：从 Dataset 选取单帧目标并渲染标注图。
- `save_annotation_gallery`：选择批次最高分 trial，保存通过/失败图集。
- `manifest.json` 保存 request metadata，包括 dataset、evaluation dataset、score、
  candidate cache、transform、configuration、code、trial 和评估配置版本。
- `index.csv` 稳定列出错误类型、segment、beatmap、sample、trial、课程阶段、参数组、
  score、图像路径、predicted/target 坐标、action/ambiguity 和版本字段。
- `OptionalTrainingVisualizer`：可视化失败不抛入训练流程，只返回一次 warning 后静默禁用。
- Gallery 实现位于 `src/visualization/core/gallery/exporter.py`；`traning.lib.visualization.gallery`
  只保留兼容 re-export。
- `full-flow` 最终 artifact 会绑定 gallery request、score report、candidate cache manifest 和
  checkpoint 继承关系，复制到其他目录后仍可按 artifact manifest 相对路径校验。

## 批次输入契约

图集输入是 `BatchGalleryRequest`。每个 trial 必须提供：

```json
{
  "trial_id": "trial_0042",
  "score": 0.91,
  "score_version": "external",
  "parameters": {
    "architecture": {},
    "training": {},
    "inference": {}
  },
  "metrics": {},
  "frames": []
}
```

选择规则：

```text
先按 score 从高到低
score 相同按 trial_id 字典序从小到大
取第一名
```

同一批次所有 trial 必须使用相同 `score_version`。当前默认 `external`，表示 score 由外部
评估器写入，图集模块不重新计算。

## 逐帧输出契约

`FrameEvaluation` 至少需要：

- `sample_key`
- `frame_index`
- `passed`
- `target_source_index`
- 可选 `predicted_osu_xy`
- 可选 `predicted_video_xy`
- 可选 `primary_error`
- 可选 `error_tags`
- 可选 `spatial_error`
- 可选 `temporal_error_ms`
- 可选 `frequency_limited`
- 可选 `metrics`

图集模块不根据 `predicted_osu_xy`、`predicted_video_xy` 或 metrics 重算通过状态。
同一 `sample_key` 下任一待导出帧失败时，该样本组归入 `failed`；全部通过时归入
`passed`。

## 输出目录

默认输出到 `traning_example/`：

```text
output_<次数>__<UTC时间>__<batch>__<best_trial>/
  best_parameters.json
  index.csv
  manifest.json
  passed/<subproject>/<sample-group>/*.png
  failed/<subproject>/<sample-group>/*.png
```

输出次数记录在 `traning_example/.output_counter`，避免覆盖历史图集。UTC 时间和次数进入
目录名、图片文字和 JSON 清单。

## 六个子项目

图集固定支持：

```text
single_point / slider / multi_point / point_slider / spinner / long_sequence
```

每个已评估子项目中：

- 通过最多随机抽取 10 组测试样本；
- 不通过最多随机抽取 10 组测试样本；
- 每个测试样本以 `sample_key` 为组，一个样本组输出到一个独立文件夹；
- 样本组内输出该样本所有参与判定、需要人工检查的打击帧；
- 样本组只有 1 个待打击/待检查帧时，只输出 1 张图；
- 纯 `no-op` 且无错误的帧不进入图集；
- 样本组数量不足时全部输出；
- 未进入课程的子项目不创建目录；
- 随机结果由 `random_seed` 固定，保证可复现。

## 通过阈值计划

连续通过门槛已由 `core/optimization/parameter_search/curriculum.py` 实现，训练方案中的默认值为：

| 子项目 | 连续通过 | 最大失败 | 最大样本 |
|---|---:|---:|---:|
| `single_point` | 15 | 2 | 40 |
| `slider` | 10 | 2 | 35 |
| `multi_point` | 8 | 3 | 35 |
| `point_slider` | 6 | 3 | 30 |

`spinner` 和 `long_sequence` 暂无独立连续通过阈值。

## 当前限制

`save_annotation_gallery` 只消费已经生成的 `BatchGalleryRequest`，不会重新评分；评分和错误归因
仍由 `core/optimization/scoring/run_outputs.py` 与完整训练 pipeline 负责。
