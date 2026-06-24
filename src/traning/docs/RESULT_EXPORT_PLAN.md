# Result Export Plan

## 模块定位

源码入口：`src/traning/core/result_export`

结果导出模块负责把训练、评估和调参结果转换为可人工检查的图片、JSON 和目录结构。它是
评估结果消费者，不是评估器本身。

## 当前导出能力

- `visualize_click_label`：从 Dataset 选取单帧目标并渲染标注图。
- `save_annotation_gallery`：选择批次最高分 trial，保存通过/失败图集。
- `OptionalTrainingVisualizer`：可视化失败不抛入训练流程，只返回一次 warning 后静默禁用。
- `traning.lib.visualization`：渲染、保存、图集抽样、输出身份和可选 ffplay 显示。

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
- 可选 `primary_error`
- 可选 `error_tags`
- 可选 `spatial_error`
- 可选 `temporal_error_ms`
- 可选 `frequency_limited`
- 可选 `metrics`

图集模块不根据 `predicted_osu_xy` 或 metrics 重算通过状态，只按已有 `passed` 分类。

## 输出目录

默认输出到 `traning_example/`：

```text
output_<次数>__<UTC时间>__<batch>__<best_trial>/
  best_parameters.json
  manifest.json
  passed/<subproject>/*.png
  failed/<subproject>/*.png
```

输出次数记录在 `traning_example/.output_counter`，避免覆盖历史图集。UTC 时间和次数进入
目录名、图片文字和 JSON 清单。

## 六个子项目

图集固定支持：

```text
single_point / slider / multi_point / point_slider / spinner / long_sequence
```

每个已评估子项目中：

- 通过最多随机抽取 10 帧；
- 不通过最多随机抽取 10 帧；
- 数量不足时全部输出；
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

## 后续计划

- 接入正式 runner 输出的 score、passed、错误归因和资源指标。
- 在 manifest 中保存数据版本、score 版本、评估集版本和代码版本。
- 增加 HTML 或 CSV 索引，方便按错误类型、子项目和 trial 参数浏览。
- 将导出目录与模型 artifact 关联，支持后续复现实验。
