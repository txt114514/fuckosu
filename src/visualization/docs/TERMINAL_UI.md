# 中文终端 UI

本文记录训练可视化终端 UI 的设计约束和使用方式。它是 `src/visualization`
模块的专门 UI 文档，实时状态和双 UI 切换链路见
[REALTIME_UI_ARCHITECTURE.md](REALTIME_UI_ARCHITECTURE.md)，详细模块说明见
[README.md](README.md)。

## 目标

- 训练启动后先创建可视化 UI，再由同一条本地代码执行链自动推进 full-flow、ramp 和训练阶段。
- UI 文案默认中文输出，面向训练观察和排障，不要求用户读取 JSON 文件才能知道当前状态。
- UI 必须显示渐进训练当前 Level、是否通过 gate、当前评分、最好评分、资源状态和最近事件。
- 启动检查/渐进准备使用 UI A；进入真实训练 phase 后自动切换到 UI B。
- UI 失败不能中断训练；Rich 初始化失败时降级 plain，`off` 模式使用 `NullReporter`。

## CLI 参数

```text
--progress-ui auto|rich|plain|off
--progress-language zh-CN
```

模式含义：

- `rich`：在交互式 TTY 中使用 Rich Live 多面板动态刷新。
- `plain`：使用纯文本摘要，适合非交互环境或日志输出。
- `auto`：TTY 时优先 rich，非 TTY 时自动 plain。
- `off`：关闭终端渲染，训练继续运行。

## 小屏紧凑模式

Rich UI 会读取当前终端高度；高度小于等于 `DashboardSettings.compact_terminal_height`
时自动进入紧凑分页模式，避免面板堆叠后下面的参数、测试或事件区不可见。
紧凑模式固定停留在当前页，不自动切换页面；窄终端也会使用紧凑分页。屏幕底部会显示
`dashboard_state.json` 路径，终端无法完整显示时可用 watcher 或普通文件查看完整结构化状态。

紧凑模式仍保留可选按键：

- `1`：概览页。
- `2`：参数页。
- `3`：测试页。
- `4`：评分页。
- `5`：资源页。
- `6`：事件页。
- `Tab` 或空格：下一页。
- `b`：上一页。
- `f`：在紧凑分页和完整堆叠视图之间切换。
- `Ctrl+C`：中断当前训练流程。

启动检查 UI 只有概览、资源和事件三页；正式训练 UI 有概览、参数、测试、评分、资源和事件六页。
非交互终端不会监听按键，仍按 plain/off 模式输出或关闭。

## 多终端面板

`src/start/main.py` 启动时会优先使用容器内的 `tmux` 创建多面板终端，并自动 attach
到该 session。主 pane 运行训练流程，其余 pane 单独运行 `visualization.core.panel_watcher`，
分别显示当前试验、参数、测试、评分、资源和事件。

- 如果当前进程已经在 tmux 内，会直接在当前 tmux session 中分 pane。
- 如果不在 tmux 内，会创建 session 并把当前终端 attach 进去，session 名称形如 `osu_ui_<run_id>`。
- 如果容器没有安装 tmux，训练不会中断，UI 会在事件面板记录“当前容器未安装 tmux，无法自动创建多个终端窗格”。

单面板 watcher 可单独运行：

```bash
PYTHONPATH=src:. python -m visualization.core.panel_watcher \
  artifacts/training_runs/<run_id>/dashboard/dashboard_state.json current
```

真实 CUDA 训练需要在容器 TTY 内运行，例如：

```bash
host-exec docker exec -t -u dev osu_ai_dev bash -lc 'cd /home/dev/workspace && PYTHONPATH=src:. python -m traning.main full-flow --config configs/model_full_small_vram.yaml --device cuda --auto-launch-full --progress-ui rich --progress-language zh-CN --resume --resume-policy auto'
```

## Rich 面板

Rich UI 当前使用 `visualization.core.renderers.rich_renderer.RichDashboardRenderer`，
标题为“统一训练控制台”，主要面板包括：

- 检测流程：full-flow、ramp、spatial、candidate cache、temporal、decision、evaluation 等阶段。
- 最佳参数记录：当前参数评分、历史最高、全局最高、最佳 trial、最佳 checkpoint 和参数摘要。
- 当前参数学习及评分：loss、score、等级、晋升状态、连续通过次数和子项目评分。
- 总体训练进度：Level、global step、spatial/temporal step、缓存帧、候选数量和速度。
- 本机资源：GPU、显存、CPU、进程内存和磁盘状态。
- 最近事件与警告：checkpoint、score、gallery、失败原因和停止事件。

## 反馈链路

训练代码只能依赖 `visualization.lib` 的稳定 reporter API：

```python
from visualization.lib import create_dashboard_reporter
```

关键上报方法：

- `update_pipeline_stage(...)`：阶段 running/passed/failed/skipped。
- `update_metrics(...)`：step、loss、score、Level、晋升状态和当前参数。
- `report_score(...)`：当前评分和新全局最高分。
- `report_resource(...)`：GPU/CPU/内存/磁盘状态。
- `report_dataset_usage(...)`：segment、frame、candidate、sequence 使用情况。
- `emit_event(...)`：最近事件。
- `request_stop(...)`：失败或中断停止摘要。
- `register_checkpoint(...)`：checkpoint 路径。

full-flow 外层创建一个 dashboard reporter；ramp 必须复用同一个 reporter，不能在 full-flow 内部
再用 `progress_ui="off"` 启动 `NullReporter`，否则 Level 状态不会显示在同一个 UI 上。

## 显示覆写

Rich 主视图和单面板 watcher 在最终打印前都会调用
`visualization.core.display_overrides.apply_display_overrides(...)`。该函数递归处理常见 Rich
对象（`Group`、`Panel`、`Table`、`Text` 和字符串），并通过
`visualization.conf.messages` 中的对照表把状态、阶段、字段名和常见错误片段覆写为中文。

因此训练模块和 panel 组件应优先上报稳定 key、状态值或原始错误文本；遗漏的英文显示由最终
打印出口统一处理，不需要在每个模块里重复维护中文化逻辑。

UI 视图由 `TrainingDashboardState.pipeline_phase` 驱动：

- `startup`、`data_preparation`、`pretrain_check`、`progressive_preparation` 渲染启动检查与渐进准备 UI。
- `training`、`completed`、`failed` 渲染正式训练实时 UI。
- full-flow 的 `RAMP_TRAINING` 真实 stage 开始时切换到正式训练 UI，不使用定时或 demo 条件。

## 渐进训练显示要求

ramp 启动后，UI 应自动显示：

- `training_ramp` 阶段进入 running；
- 当前 `level_a` / `level_b` / `level_c*`；
- `current_level`、`current_trial_id`、`completed_levels`、`total_levels`；
- spatial/temporal step 目标和当前进度；
- gate 通过时显示 `Level X 已通过 gate`；
- gate 失败时显示 `Level X 失败`，并写入 `stop_state.json`；
- 所有 Level 通过后显示最终 readiness，通过 `--auto-launch-full` 时继续进入 full training。

## 状态文件

UI 的屏幕刷新和状态写入是两套机制。Rich 刷新频率由
`DashboardSettings.refresh_per_second` 控制，默认每秒最多 4 次；结构化状态则由 reporter
事件驱动写入。

Dashboard 目录会写出：

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

`current_parameter_status.json` 记录当前参数组实时通过/警告/失败/运行中的测试阶段、当前分数、
分类分数和停止原因。这些文件是 Web UI、离线分析和失败恢复排查的稳定输入，但用户不应被迫读取
它们才能知道训练状态；终端 UI 应同步展示关键状态。

## 停止摘要

失败或中断时，训练必须先写入 dashboard 状态和 `stop_state.json`，再渲染最终 UI。交互式 TTY
支持 `Q`、`Enter` 和 `Esc` 退出停止摘要；非 TTY 不等待按键。

## 测试覆盖

相关测试：

- `src/visualization/tests/test_dashboard.py`
- `src/visualization/tests/test_panels.py`
- `src/traning/tests/full_checks/test_full_flow.py`
- `src/traning/tests/full_checks/test_training_ramp.py`

这些测试应覆盖 reporter 状态持久化、TTY 行为、Rich 面板渲染、full-flow 阶段回写和 ramp
Level 通过/失败状态。
