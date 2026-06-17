# osu! Video Training

`src/traning` 用于从 `before_traning` 生成的视频片段数据集训练空间识别与因果时序
决策模型。当前第一个可运行阶段是 `data_input`，后续空间、候选缓存、时序、评估和
导出阶段已经建立目录边界。

完整模型路线见 [`TRAINING_PLAN.md`](TRAINING_PLAN.md)。测试参数搜索、score 与通过
边界见 [`TEST_PARAMETER_MECHANISM.md`](TEST_PARAMETER_MECHANISM.md)。点与 slider 的
可执行评分公式见 [`SCORING_SPEC.md`](SCORING_SPEC.md)。

## 运行

在仓库根目录检查训练环境：

```bash
PYTHONPATH=src python src/traning/main.py env-check
PYTHONPATH=src python -m traning.cli env-check
```

Docker 重建后可把 CUDA 作为硬门禁检查：

```bash
PYTHONPATH=src python -m traning.cli env-check --strict --require-cuda
environment/check_gpu.sh
```

在仓库根目录执行数据集检查。默认 `--split all` 会同时检查全部样本：

```bash
PYTHONPATH=src python src/traning/main.py data-check
```

也可以分别检查训练集和验证集：

```bash
PYTHONPATH=src python src/traning/main.py data-check --split train
PYTHONPATH=src python src/traning/main.py data-check --split validation
```

解码一个样本并显示帧与 patch 信息：

```bash
PYTHONPATH=src python src/traning/main.py data-preview --split train --index 0
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

命令和参数以 CLI 帮助为准：

```bash
PYTHONPATH=src python src/traning/main.py --help
```

## 配置

默认配置位于 `src/traning/conf/config.yaml`，由 `src/traning/conf/settings.py` 中的
Pydantic 模型加载。配置文件中的相对路径会相对配置文件所在目录解析，也可以通过
`OSU_TRAINING_` 前缀和双下划线嵌套覆盖受支持字段。

示例：

```bash
OSU_TRAINING_DATA_INPUT__MAX_SEGMENTS=10 \
PYTHONPATH=src python src/traning/main.py data-check
```

常用运行设置包括：

- `data_input.dataset_root`：训练片段数据集目录
- `data_input.train_items`：训练集 item 白名单
- `data_input.validation_items`：验证集 item 白名单
- `data_input.include_items` / `exclude_items`：全局 item 过滤
- `data_input.sample_fps`：采样 FPS
- `data_input.frame_step`：帧步长
- `data_input.patch_width` / `patch_height`：空间 patch 尺寸
- `data_input.overlap_x` / `overlap_y`：patch 重叠宽度
- `evaluation.min_click_interval_ms`：点击序列评估的最小点击间隔，默认 `50ms`
- `visualization.output_dir`：标注图片和图集输出目录

## 输入位置

默认输入路径：

```text
training_package/video_segments/
  item_000001/
    segments.csv
    single_point/
      segment_.../
        video.mp4
        beatmap.json
```

每个样本目录必须同时包含 `video.mp4` 与 `beatmap.json`。`beatmap.json` 中 Circle 的
`x, y` 和 Slider 的 `path` 坐标保持 osu!standard 原始坐标系；视频与 osu! 坐标转换
使用 `src/package/coordinates.py` 的稳定 API。

训练输入只使用视频帧和 `beatmap.json` 标签。`SegmentFrameDataset` 通过 OpenCV 解码
RGB 帧，不读取音频流；重新生成片段时，`before_traning` 默认也会从 segment `video.mp4`
中移除音频，避免模型或评估流程意外依赖音频。

默认配置把 `item_000001` 作为训练集、`item_000002` 作为验证集：

```yaml
data_input:
  train_items:
    - item_000001
  validation_items:
    - item_000002
```

## 处理流程

```text
data_input
  -> spatial
  -> candidate_cache
  -> temporal
  -> evaluation
  -> export
```

当前 `TRAINING_STAGES` 只登记 `data_input`。该阶段会扫描并校验数据集、加载标签、
建立帧索引、解码原分辨率 RGB 帧、筛选当前帧可见 HitObject，并组装 Dataset 与
DataLoader。`data-check` 可检查 `all`、`train` 或 `validation`；`build_dataset` 和
`build_dataloader` 默认使用 `train` split。

## 数据与产物

- `SegmentFrameDataset`：按片段帧索引返回原分辨率 RGB CHW Tensor 和可变长度标签
- `DataInputReport`：数据集数量、维度、类别和问题报告
- `BatchGalleryRequest`：批次 trial 分数、逐帧通过状态和最佳图集输入契约
- `traning_example/`：默认可视化输出根目录
- `best_parameters.json`：最佳 trial 的参数快照
- `manifest.json`：图集抽样结果与无法定位的记录

`save-annotation-gallery` 的结果 JSON 中每个 trial 提供总分、参数和逐帧通过结果。总分
越高越好；命令只选择最高分 trial。同一批次的所有 trial 必须使用相同
`score_version`，否则拒绝排序。逐帧结果可以携带错误归因字段，用于区分空间参数网、
时间参数网和决策参数网的问题。

示例：

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
          "primary_error": "none",
          "error_tags": [],
          "spatial_error": 1.8,
          "temporal_error_ms": 12.0,
          "frequency_limited": false,
          "metrics": {"position_error": 1.8}
        }
      ]
    }
  ]
}
```

默认对每个已执行子项目随机保存通过和不通过各最多 10 帧；不足 10 帧时全部保存。
随机结果由 `random_seed` 固定。六个子项目为 `single_point`、`slider`、
`multi_point`、`point_slider`、`spinner`、`long_sequence`。

## 当前边界

- 图像保持原分辨率 RGB CHW；默认归一化到 `[0, 1]`。
- 标签时间以片段起点为零；可见对象窗口使用谱面 AR preempt 和可配置尾部时间。
- patch 由 Tensor 视图串行产生，不提前复制全部 patch。
- slider 跨 patch 使用全局 path/head/orientation 稠密图融合。
- 首版不支持 slider 路径交叉或接触分叉；此类样本在 target/data check 阶段过滤。
- 空间主扫描固定 stride 2；stride 1 只用于少量候选的局部精修。
- refiner stride 由训练前显存 dry-run 选择并在 trial 内冻结，主干不使用 stride 1。
- patch 完成后立即 CPU offload 精简输出，不跨 patch 保留深层特征。
- 空间级联为全图扫描、候选精修、条件歧义复查三层。
- 参数搜索采用 TPE/随机生成、ASHA 剪枝、课程晋级和难例挖掘。
- 同一 trial 晋级继承 checkpoint；新参数组合从基础阶段开始；推理参数不触发重训。
- 可视化默认关闭；渲染与 ffplay 显示独立于训练，失败只返回一次告警后静默禁用。
- 单点与 slider 使用 `point-slider-v2`；slider 以 1.5x 双向膨胀走廊计算路径覆盖。
- 点击序列使用 `click-sequence-v1`；目标首次合格命中后失效，重叠目标递进判定，并应用
  可配置的最小点击间隔。
- 错误归因使用 `none/spatial/temporal/decision` 主责任；重复点击和更高分无效点击归入
  决策，空间偏差归入空间，时间偏离和提前点击归入时间。
- trial 聚合 score 仍由外部评估结果提供；搜索器和晋级执行器尚未实现。

## 改动影响面

| 改动 | 至少检查 |
|---|---|
| 数据路径/筛选 | Settings, config, discovery, preflight |
| 标签 schema | annotation, dataset, collate, spatial target encoder |
| 帧采样 | sampling, dataset, temporal/action label semantics |
| patch 规则 | Settings, tiling, spatial model input |
| slider 路径融合 | target encoder, global canvas, skeleton trace, candidate cache |
| 空间级联/offload | spatial model, refiner, review policy, CPU accumulator |
| 搜索/课程协议 | experiment schema, checkpoint lineage, evaluator, sampler |
| 可视化 | visualization settings, gallery schema, renderer, optional service, Docker X11 |
| 批次结构 | dataset, collate, trainer consumers |
| 新训练阶段 | core/pipeline.py, core stage, Lib APIs, state schema |

## 目录职责

- `conf`：配置模型、默认值和 `config.yaml`
- `core`：阶段编排、配置映射和训练入口
- `Lib`：可独立测试的数据、评分、可视化和训练 API
- `state`：run、experiment、checkpoint 和 gallery schema
- `main.py`：Typer CLI 入口

开发者和 Codex 使用的架构导航及函数调用索引见
[`CODEX_INDEX.md`](CODEX_INDEX.md)。
