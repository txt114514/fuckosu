# 点与 Slider 评分公式

版本：`point-slider-v2`

本规格把点、slider 路径和通过判定形式化为可执行公式。实现位于
`src/traning/Lib/metrics/scoring.py`。

## 1. 基本定义

谱面字段 `circle_radius_osu_pixels` 记为圆半径 `x`。直径为 `2x`。所有空间距离都在
osu! 谱面坐标系计算，不使用视频像素距离。

点的欧氏误差：

```text
d = sqrt((pred_x - target_x)^2 + (pred_y - target_y)^2)
r = d / x
```

时间误差取绝对值：

```text
y = abs(predicted_time_ms - target_time_ms)
```

最终原始分数：

```text
raw = spatial + temporal + spatial * temporal
```

为了便于不同样本求平均，另输出：

```text
normalized = raw / 3.2025
```

`3.2025 = 1.05 + 1.05 + 1.05 * 1.05`，所以标准配置下 normalized 位于 `[0, 1]`。

## 2. 点空间系数

### 2.1 通过区

`r <= 1.0` 为位置通过。`r=1.0` 的基准系数为 `1.0`。

`r < 0.6` 按 `0.6` 计算。`0.6 <= r <= 1.0` 使用最多 `0.05` 的凹函数奖励：

```text
q = (1.0 - max(r, 0.6)) / 0.4
spatial = 1.0 + 0.05 * sqrt(q)
```

因此越靠近中心奖励越高，但继续靠近时新增奖励逐渐减少。

### 2.2 安慰区

`1.0 < r < 1.5` 不算通过，只保留最多 `0.05` 的引导分：

```text
q = (1.5 - r) / 0.5
spatial = 0.05 * q^2
```

越靠近 `1.0` 增长越快。`r >= 1.5` 时空间系数为 `0`。

## 3. 时间系数

| 时间误差 | 系数 |
|---|---|
| `0–20ms` | `1.05`，满分加微量奖励 |
| `20–50ms` | 从 `1.05` 线性降到 `1.00`，仍为满分区 |
| `50–100ms` | 从 `1.00` 线性降到 `0.80`，优秀区 |
| `100–150ms` | 从 `0.80` 线性降到 `0.50`，及格区 |
| `>150–200ms` | `0.05 * ((200-y)/50)^2`，安慰区 |
| `>=200ms` | `0` |

点通过条件：

```text
r <= 1.0 and y <= 150ms
```

原始描述中的 `50–100ms` 优秀区和 `80–150ms` 及格区在 `80–100ms` 重叠。本版本按
“优秀优先”处理，因此正式边界为 `50–100ms` 优秀、`100–150ms` 及格。

`150ms` 本身仍属于及格区；超过 `150ms` 后进入安慰区。这里保留明确档位下降，以反映
“通过”和“只有安慰分”的分类差异。

## 4. Slider 路径膨胀

slider 起点使用与点完全相同的位置和时间规则。当前 `beatmap.json` 的 slider `x/y`
可能为空，此时参考起点取采样后路径的 `path[0]`。预测起点未单独提供时也取预测
`path[0]`。

路径不能按零宽中心线直接比较。先以 `1.5x` 为半径分别膨胀参考路径和预测路径，得到
两个容差走廊：

```text
reference_corridor = dilate(reference_path, radius=1.5x)
prediction_corridor = dilate(predicted_path, radius=1.5x)
```

实现不依赖某个图像分辨率进行 mask 光栅化。评分前先以最多 `0.25x` 的步长加密两条
polyline；某点到另一条折线的最短距离不超过 `1.5x`，等价于该点落在对方膨胀走廊内。

双向覆盖指标：

```text
reference_coverage =
    落入 prediction_corridor 的参考路径采样点数 / 参考采样点总数

prediction_precision =
    落入 reference_corridor 的预测路径采样点数 / 预测采样点总数
```

严格路径通过要求：

```text
reference_coverage == 1
prediction_precision == 1
两条有向最大距离都 <= 1.5x
```

这同时阻止两类错误：

- 预测只覆盖 slider 的一部分；
- 预测经过完整 slider 后又延伸到无关区域。

路径通过时系数从 `1.0` 开始，并根据双向最坏距离提供最多 `0.05` 的微量奖励。路径未
通过时只保留：

```text
path_comfort = 0.05 * reference_coverage * prediction_precision
```

slider 总空间系数取最弱环节：

```text
slider_spatial = min(head_spatial, path_coefficient)
```

slider 通过条件：

```text
head position passed
and head time passed
and path passed
```

最终仍使用：

```text
raw = slider_spatial + temporal + slider_spatial * temporal
```

## 5. 点击序列模拟

单对象 score 只回答“这一次预测是否能命中这个目标”。完整评估还需要按时间模拟点击序列：

```text
预测点击流 + 当前未命中目标集合
  -> 应用最小点击间隔
  -> 对仍有效目标逐个判定
  -> 首个合格目标消失
  -> 后续点击只能命中剩余目标
```

序列模拟版本为 `click-sequence-v1`，底层对象公式仍使用 `point-slider-v2`。

### 5.1 目标一次性命中

当某次点击对某个目标同时满足空间和时间通过条件后，该目标被视为模拟点击成功，并从
有效目标集合中移除。之后即使有更靠近中心或时间更精确的点击，也不会提高这个目标的
得分，也不会再次命中这个目标。

如果某次点击没有达到该目标的通过条件，该目标仍保持有效；后续点击仍可命中它。因此
每个目标最多只有一次合格判定，且这次判定按点击时序取最早的合格点击。

### 5.2 重叠目标

多个目标在空间和时间上重叠时，一次点击可能同时满足多个未命中目标。此时按目标谱面
时间从早到晚选择一个目标作为本次命中；若时间相同，再按 `source_index` 和目标 ID
稳定排序。被命中的目标立即消失，下一次有效点击再对剩余的重叠目标判定。

这个规则等价于把已经打击成功的目标从画面里移除，避免同一次或后续点击反复给同一
目标刷更高分。

### 5.3 点击频率限制

评估不禁止模型输出多次疑似点击；即使当前只有一个目标，模型也可以尝试多个候选点。
但为了避免通过高频点击提高命中率，序列模拟器会应用最小点击间隔：

```text
min_click_interval_ms = 50.0
```

该值由 `evaluation.min_click_interval_ms` 配置，默认约 50ms。低于间隔的点击标记为
`frequency_limited`，不参与目标判定，也不会让目标消失。它不刷新冷却时间；下一个距离
最近一次有效点击达到间隔的点击仍可参与判定。

### 5.4 Slider 首次成功后失效

slider 使用与第 4 节相同的起点和路径通过条件。某次预测使 slider 起点与路径都通过后，
该 slider 也从有效目标集合中移除；之后对同一 slider 的点击或路径预测不再计分。

## 6. 错误归因

点击序列模拟同时输出错误主责任，便于后续把评估结果反馈到不同参数网：

| 主责任 | 含义 | 典型标签 |
|---|---|---|
| `spatial` | 位置、路径或空间融合错误 | `spatial_miss`, `head_spatial_miss`, `slider_path_miss` |
| `temporal` | 点击时间偏早或偏晚 | `early_click`, `late_click` |
| `decision` | 是否点击、点哪个、点几次或冷却抑制错误 | `duplicate_after_hit`, `better_score_after_resolution`, `frequency_limited`, `no_active_target` |
| `none` | 本次点击已命中，或没有可归因错误 | 无 |

因此：

- 重复点击同一已命中目标归入 `decision`。
- 已有更高分点击但目标已因较早合格点击失效时，标记
  `duplicate_after_hit + better_score_after_resolution`，归入 `decision`。
- 点击空间偏差归入 `spatial`。
- 点击时间偏离和提前点击归入 `temporal`。
- 低于最小点击间隔的高频点击归入 `decision`，标签为 `frequency_limited`。

评估 JSON 可携带以下字段：

```json
{
  "primary_error": "decision",
  "error_tags": ["duplicate_after_hit", "better_score_after_resolution"],
  "spatial_error": 0.0,
  "temporal_error_ms": -90.0,
  "frequency_limited": false
}
```

## 7. 当前聚合边界

本版本已经实现单个点和单条 slider 的空间、时间、组合分数及通过判定。一个 trial
包含多个样本时如何跨子项目聚合为最终 `score` 尚未确定。点击序列模拟已经能生成
逐目标命中、未命中、点击频率限制和错误归因结果，但尚未接入完整 trial 聚合公式。

在聚合公式确定前，批次 JSON 应写明：

```json
{
  "score": 0.91,
  "score_version": "external"
}
```

未来评估器使用本公式产生 trial score 时，应写入
`"score_version": "point-slider-v2+<aggregation-version>"`，避免不同公式的结果被
错误比较。
