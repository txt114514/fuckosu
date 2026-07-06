# 可视化模块详细介绍

`src/visualization` 是训练系统的中文可视化与结果导出模块。它和 `traning`
同级存在，负责把训练生命周期、指标、资源、事件和评估图集转换成人可以持续观察的输出。
训练核心只依赖 `visualization.lib` 暴露的稳定 API，不直接依赖 Rich、Panel、终端布局或
`visualization.core` 内部实现。

## 模块目标

- 提供中文训练控制台，展示 full-flow、ramp、空间训练、时序训练、评分、artifact 和 inheritance 阶段状态。
- 记录结构化 dashboard 状态，便于训练中断、排查和后续工具读取。
- 在训练 loop 中持续上报 step、loss、score、数据使用、资源占用和最近事件。
- 导出 passed/failed 评估图集，让用户直接查看预测点、目标点、错误类型和版本信息。
- 保持可视化失败不影响训练主流程：UI 或图集导出异常应被隔离为 warning，而不是中断训练。

## 目录结构

```text
src/visualization/
  conf/                 # DashboardSettings、默认刷新频率、中文消息
  lib/                  # 稳定公共 API，训练代码只允许依赖这一层
  core/                 # 控制器、终端渲染器、Rich 面板和 gallery 实现
  state/                # dashboard_state.json、events.jsonl 等状态持久化
  tests/                # dashboard、panel、gallery 测试
  docs/                 # 当前模块说明
```

## 分层边界

稳定依赖方向：

```text
traning.core
  -> visualization.lib
  -> visualization.state
  -> visualization.core
```

训练代码应从 `visualization.lib` 导入：

```python
from visualization.lib import create_dashboard_reporter, NullReporter
```

可视化内部可以使用 `visualization.core` 的 Rich renderer、plain renderer、gallery exporter
和 panel 组件。`traning` 不应直接导入这些内部实现，否则会把训练逻辑和终端布局耦合在一起。

## 公开 API

主要公共入口：

- `create_dashboard_reporter(...)`：创建 dashboard handle，供 `with` 语句管理生命周期。
- `NullReporter`：关闭 UI 时使用的空 reporter，训练逻辑无需分支判断。
- `TrainingReporter`：训练侧依赖的协议，包含 stage、metrics、score、resource、dataset、event、checkpoint 等上报方法。
- `collect_resource_state()`：采集 CPU、内存、磁盘和可用 GPU 资源状态。
- `export_best_trial_gallery(...)`：导出最佳 trial 的评估图集。

常用 CLI 参数：

```text
--progress-ui auto|rich|plain|off
--progress-language zh-CN
```

`auto` 会根据 TTY 情况选择模式；`off` 只关闭界面，不关闭结构化事件和训练本身。

## Dashboard 状态模型

核心 DTO 位于 `visualization.lib.models`：

- `TrainingDashboardState`：一次训练 run 的完整 dashboard 快照。
- `PipelineStageState`：full-flow 阶段状态，包括开始/结束时间、进度、输出路径和错误原因。
- `CurrentTrainingMetrics`：loss、score、学习率、梯度、最好分数等训练指标。
- `DatasetUsageState`：segment、frame、patch、candidate、sequence 使用情况。
- `ResourceState`：GPU/CPU/内存/磁盘资源状态。
- `TrainingEvent`：结构化事件，保留 severity、message key、阶段、level 和 trial。
- `TrainingStopState`：中断或失败时的停止摘要。
- `BestParameterRecord`：当前最佳参数、分数和 checkpoint 位置。

`DashboardReporter` 会把状态写入运行目录：

```text
dashboard_state.json
events.jsonl
current_parameters.json
current_parameter_status.json
best_parameters.json
best_parameters.yaml
dataset_usage.json
resource_history.jsonl.current.json
stop_state.json
```

这些文件是后续 Web UI、离线分析或恢复排查的稳定输入。

## 数据更新频率

可视化模块里有三类不同的“更新频率”，不能混为一个配置项：

- 结构化状态文件是事件驱动写入，不是固定定时轮询。`DashboardReporter`
  收到 `update_pipeline_stage`、`update_metrics`、`report_score`、`report_resource`、
  `report_dataset_usage`、`emit_event`、`register_checkpoint` 或停止请求时，会立即更新
  `dashboard_state.json`，相关专用文件也会同步写入。
- 当前参数快照通过 `update_metrics(current_parameters=...)` 上报，会写入
  `current_parameters.json`。随后 `report_score(...)` 产生新最高分时，会把当时的参数快照
  固化到 `best_parameters.json` 和 `best_parameters.yaml`。
- 当前参数实时状态由 reporter 根据阶段、metrics、score、checkpoint 和停止事件自动聚合，
  写入 `current_parameter_status.json`，包括通过/警告/失败/运行中的阶段、当前分数、
  分类分数和停止原因。
- 空间训练和时序训练进入真实训练 loop 后，每完成一个训练 step 就上报一次 step、loss、
  数据使用情况；CUDA 训练时也会在每个 step 上报一次资源状态。因此 dashboard 数据刷新速度
  等于当前训练 step 速度，不由 `refresh_per_second` 控制。
- full-flow、ramp、候选缓存、决策导出、最终评估和模型导出等非 step 型阶段，按阶段开始、
  阶段结束、批处理节点、checkpoint 或评分事件上报，不额外启动后台采样线程。

终端画面刷新和数据写入是两套机制：

- Rich UI 使用 `DashboardSettings.refresh_per_second` 控制屏幕重绘，默认值是 `4.0`，
  即最多每秒重绘 4 次。它只影响终端显示，不降低也不提高状态文件写入频率。
- `DashboardSettings.plain_interval_seconds` 默认值是 `5.0`。当前 plain renderer 只在开始和
  结束时打印摘要，不做周期性刷新；该字段保留给后续 plain 周期摘要使用。

样本图片和 Gallery 也有独立节奏：

- 训练样本可视化受 `traning.conf.settings.VisualizationSettings` 控制，`enabled` 默认关闭；
  `every_n_steps` 默认是 `500`。这个值只作用于
  `OptionalTrainingVisualizer.maybe_visualize_step(...)`：调用方传入 `global_step` 时，只有
  `global_step % every_n_steps == 0` 才会真正渲染，否则返回 `skipped`。
- 当前主训练 loop 尚未接入 `maybe_visualize_step(...)` 的自动调用，因此默认 full-flow 训练不会
  每 500 step 自动导出训练样本图。要启用这种周期性样本图，需要在训练 step 上报处显式接入该
  helper，或通过专门的预览/导出入口手动生成。
- 评估 Gallery 不是按时间或 step 定时导出，而是在评分、best trial 或 result export 流程请求
  `export_best_trial_gallery(...)` 时一次性生成。

## 终端 UI

UI 模式：

- `rich`：交互式 TTY 使用 Rich Live，多面板动态刷新。
- `plain`：非 TTY 或降级场景使用纯文本摘要。
- `auto`：TTY 时优先 Rich，非 TTY 时自动 plain。
- `off`：不渲染界面，使用 `NullReporter`，训练继续运行。

Rich 面板包括：

- pipeline 阶段进度；
- 当前学习指标；
- 当前参数摘要和最佳参数；
- 总体进度；
- 资源状态；
- 最近事件。

Rich 初始化失败时应降级 plain 或保持训练继续。停止摘要在状态保存后显示。交互式 TTY
支持 `Q`、`Enter` 和 `Esc` 退出；非 TTY 不等待按键。

## Full-Flow 接入

`traning.core.full_flow` 会把下面阶段统一上报到 `PipelineStageState`：

```text
raw-data
-> before_traning
-> split validation
-> data quality check
-> environment preflight
-> resume discovery / restore
-> ramp training
-> final readiness
-> full training
-> final evaluation
-> model export
-> inheritance finalization
-> report generation
```

进入真实 spatial / temporal 训练后，训练 loop 会继续逐 step 上报：

- global/spatial/temporal step；
- loss 和各子 loss；
- 当前 segment、frame、patch、window 使用情况；
- GPU 显存峰值、CPU/内存/磁盘状态；
- checkpoint、score 和训练事件。

## Gallery 导出

Gallery 用于查看评分后的可视化样本。训练评分生成 `BatchGalleryRequest` 后，
`core/result_export` 通过 `visualization.lib.gallery_api.export_best_trial_gallery`
调用 `visualization.core.gallery.exporter.save_best_trial_gallery`。

导出目录按批次和 trial 命名，内部结构类似：

```text
<output_root>/<sequence>__<batch_id>__<trial_id>/
  best_parameters.json
  manifest.json
  index.csv
  passed/
    single_point/
    multi_point/
    slider/
    point_slider/
    spinner/
    long_sequence/
  failed/
    single_point/
    ...
```

每张图片会绘制：

- 原始帧；
- 目标对象或目标点；
- 预测 osu/video 坐标；
- trial、score、score version、batch、subproject 和 outcome 元数据。

`manifest.json` 保存批次、trial、分数、版本、抽样、样本组、图片路径和导出 issue。
`index.csv` 用稳定列记录错误类型、segment、beatmap、sample、trial、score、图片路径、
预测坐标、score/cache/transform/config 版本，方便后续排序、筛选和人工复盘。

旧的 `traning.lib.visualization.gallery` 只保留兼容转发；新实现位于
`visualization.core.gallery.exporter`，训练业务应通过 `visualization.lib.gallery_api` 调用。

## 设计原则

- 训练逻辑不依赖终端 UI 是否可用。
- 可视化输出只消费训练状态和评估 request，不反向控制训练阶段。
- 图集导出是 best-effort：失败时记录 warning，不把训练主流程变成失败。
- dashboard 状态文件使用原子写入，事件使用 JSONL 追加，便于中断排查。
- 中文显示文案集中在 `visualization.conf.messages`，避免散落在训练模块。

## 测试覆盖

可视化相关测试位于 `src/visualization/tests`：

- `test_dashboard.py`：覆盖 dashboard reporter、TTY 行为和停止摘要。
- `test_panels.py`：覆盖 Rich 面板渲染。
- `test_gallery.py`：覆盖 gallery 输出、manifest 和 index。

训练侧相关测试也会触达可视化边界，例如 full-flow、result export gallery 和 dashboard 集成测试。
