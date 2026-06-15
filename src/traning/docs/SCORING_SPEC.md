# 点与 Slider 评分公式

版本：`point-slider-v2`

本规格把 [`about_score.txt`](about_score.txt) 中的评分想法形式化为可执行公式。实现位于
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

## 5. 当前聚合边界

本版本已经实现单个点和单条 slider 的空间、时间、组合分数及通过判定。一个 trial
包含多个样本时如何跨子项目聚合为最终 `score` 尚未确定。

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
