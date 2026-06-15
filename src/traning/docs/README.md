# osu! Video Training

`src/traning` 负责从 `before_traning` 生成的视频片段数据集训练空间识别与因果时序决策模型。

完整目标和模型路线见 [`TRAINING_PLAN.md`](TRAINING_PLAN.md)。测试参数搜索、
score 与通过边界见
[`TEST_PARAMETER_MECHANISM.md`](TEST_PARAMETER_MECHANISM.md)，点与 slider 的精确公式见
[`SCORING_SPEC.md`](SCORING_SPEC.md)。当前第一个可运行阶段是 `data_input`。

## 目录结构

```text
traning/
  conf/                 配置模型与 config.yaml
  core/                 阶段编排、配置映射和状态推进
    data_input/         数据检查、Dataset 和 DataLoader 组装
    spatial/            空间候选检测阶段
    candidate_cache/    离线候选缓存阶段
    temporal/           因果时序决策阶段
    evaluation/         阶段测试与完整评估
    export/             部署模型导出
  Lib/                  可独立测试的数据、模型、损失和训练 API
  state/                run、experiment 和 checkpoint 元数据
  docs/                 对外说明、目标方案和 Codex 索引
  main.py               Typer CLI
```

`core` 组合训练阶段；`Lib` 不负责业务阶段状态。被训练和部署共同调用的稳定 API 放在
`src/package`。

## 数据输入

默认数据根：

```text
training_package/video_segments/
  item_000001/
    segments.csv
    single_point/
      segment_.../
        video.mp4
        beatmap.json
```

`data_input` 当前提供：

- 扫描并校验 `video.mp4 + beatmap.json` 配对
- 使用 Pydantic 校验标签结构
- 按配置 FPS 和步长建立帧索引
- 使用 OpenCV 解码原分辨率 RGB 帧
- 根据 AR preempt 计算当前帧可见 HitObject
- 生成覆盖完整画面的重叠 patch Tensor 视图
- 使用自定义 collate 保留可变长度标签

## 命令

检查数据集：

```bash
PYTHONPATH=src python src/traning/main.py data-check
```

解码一个样本并显示帧与 patch 信息：

```bash
PYTHONPATH=src python src/traning/main.py data-preview --index 0
```

执行当前已登记的训练阶段：

```bash
PYTHONPATH=src python src/traning/main.py run
```

保存单帧点击标注；`--show` 会通过独立 `ffplay` 进程连接主机 X11：

```bash
PYTHONPATH=src python src/traning/main.py visualize-label \
  --segment-index 0 \
  --object-index 0 \
  --show
```

按一次训练批次的评估结果保存最佳参数图集：

```bash
PYTHONPATH=src python src/traning/main.py save-annotation-gallery \
  --results training_artifacts/batch_0001_evaluation.json
```

结果 JSON 中每个 trial 提供总分、参数和逐帧通过结果。总分越高越好；命令只选择最高
分 trial。帧使用 `sample_key + frame_index` 定位，分类从数据集标签读取；
`dataset_dimension=long_sequence` 优先归入 `long_sequence`，其余使用 `category`：

```json
{
  "batch_id": "batch_0001",
  "random_seed": 2026,
  "trials": [
    {
      "trial_id": "trial_0042",
      "score": 0.91,
      "score_version": "external",
      "parameters": {
        "architecture": {"spatial_channels": 16},
        "training": {"learning_rate": 0.0003},
        "inference": {"click_threshold": 0.72}
      },
      "metrics": {"quality_score": 0.91},
      "frames": [
        {
          "sample_key": "item_000001/segment_000001",
          "frame_index": 120,
          "passed": true,
          "target_source_index": 3,
          "predicted_osu_xy": [256.5, 192.0],
          "metrics": {"position_error": 1.8}
        }
      ]
    }
  ]
}
```

同一批次的所有 trial 必须使用相同 `score_version`，否则拒绝排序。

默认对每个已执行子项目随机保存通过和不通过各最多 10 帧；不足 10 帧时全部保存。
随机结果由 `random_seed` 固定。六个子项目为 `single_point`、`slider`、
`multi_point`、`point_slider`、`spinner`、`long_sequence`。输出结构：

```text
traning_example/
  .output_counter
  output_000001__20260615T123456_123456Z__batch_0001__trial_0042/
  best_parameters.json
  manifest.json
  passed/
    single_point/
    slider/
  failed/
    single_point/
    slider/
```

`output_000001` 是该输出根目录内的持久递增次数，后面的 UTC 时间是本次输出创建时间。
编号和时间同时写入标注图、`best_parameters.json` 与 `manifest.json`。重复执行同一
批次不会覆盖旧图集，而会生成新的编号目录。单帧标注也使用相同规则，例如：

```text
traning_example/output_000002__20260615T123500_654321Z__item_000001__...png
```

只有最佳 trial 实际评估到的子项目才建立目录；对已评估子项目，`passed` 与 `failed`
两侧都会建立同名目录，没有对应样本的一侧保持为空。每帧图片包含标签位置、预测位置、
子项目、通过状态和帧指标。`best_parameters.json` 保存本批次最佳参数，
`manifest.json` 保存抽样结果与无法定位的记录。

图集保存和窗口显示都是训练外的 best-effort 能力。训练进程调用
`OptionalTrainingVisualizer.save_gallery(...)` 时不会收到异常；首次故障只返回一条
warning，后续静默跳过，不向训练进度栏重复输出。

默认配置位于 `src/traning/conf/config.yaml`。环境变量使用
`OSU_TRAINING_` 前缀和双下划线嵌套，例如：

```bash
OSU_TRAINING_DATA_INPUT__MAX_SEGMENTS=10 \
PYTHONPATH=src python src/traning/main.py data-check
```

## 当前边界

当前实现只建立训练数据输入契约，还没有实现空间模型、损失、候选缓存或时序模型。
下一阶段应先定义空间训练 target encoder，再实现高分辨率串行 patch 模型与逐 patch
反向传播。

首版 slider 支持跨多个 patch，但不处理两条 slider 路径交叉或接触分叉。空间模型将
输出局部 path/head/orientation 稠密图，在全局画布融合后按连通域恢复完整路径；
交叉样本在 target 构建或数据检查阶段过滤。

空间阶段默认使用 stride 2 全图 patch 扫描，每个 patch 完成后立即把精简输出转移到
CPU，不保留全部深层特征。全局融合后对候选区域执行局部精修；只有低置信、重叠或
slider 连续性不足的区域才进入第三轮复查。局部精修优先使用 stride 1，训练前显存
dry-run 不通过时固定回退到 stride 2；深层主干不使用 stride 1。

参数优化采用 `TPE/随机生成 + ASHA 阶段剪枝 + 课程学习 + 难例挖掘`。同一 trial 晋级
时继承 checkpoint；结构或训练参数的新组合从基础阶段开始；推理阈值只对固定
checkpoint 快速搜索，不重复训练。当前只完成这些机制的数据模型和图集输入契约，
搜索器、score 合成器和自动通过判定尚未实现。

函数位置和调用关系见 [`CODEX_INDEX.md`](CODEX_INDEX.md)。
