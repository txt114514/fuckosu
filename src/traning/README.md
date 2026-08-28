# OSU Decision Model V2

V2 是与旧训练链路隔离的新实现。正式运行路径固定为：

```text
RuntimeFrame
  -> Perception
  -> Tracking / Association
  -> Per-track Belief
  -> Outcome(current + wait horizon)
  -> Optimal Stopping Decision
```

最终动作只来自 learned Outcome 分布与确定性效用比较，不读取图像真值、离线评分标签、
动作模仿 logits 或候选槽位 logits。

## 快速检查

从工程根目录执行：

```bash
PYTHONPATH=src:. python -m start v2 config-check --config configs/traning.yaml
PYTHONPATH=src:. python -m start v2 coordinate-audit --config configs/traning.yaml
PYTHONPATH=src:. python -m start v2 env-check --config configs/traning.yaml
```

`config-check` 只验证并规范化输出配置，不要求 CUDA。`env-check` 默认是 strict：任何必需
设备或坐标标定失败都会以非零状态退出。只想查看完整诊断而不改变命令退出码时使用：

```bash
PYTHONPATH=src:. python -m start v2 env-check \
  --config configs/traning.yaml --no-strict
```

默认配置要求 CUDA。Codex sandbox 可能看不到容器已有的 GPU；在本工程开发容器中应通过
主机桥验证真实 namespace：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc \
  'cd /home/dev/workspace && PYTHONPATH=src:. python -m start v2 env-check --config configs/traning.yaml'
```

如果明确执行 CPU smoke，需要另建配置并同时设置：

```yaml
runtime:
  device: cpu
  require_cuda: false
  amp: false
```

V2 不会在 CUDA 不可用时静默退回 CPU。`python -m start` 原有默认行为仍属于旧入口；
只有显式的 `v2` 子命令进入这里描述的配置和环境边界。

## 单一配置

工程基线是 [`configs/traning.yaml`](../../configs/traning.yaml)。加载器要求顶层
`schema_version`，拒绝未知键、字符串化数值、错误枚举和不一致的跨模块字段。关键联动包括：

- `outcome.horizons_ms` 必须与 `decision.horizons_ms` 完全一致；
- `data.seed` 必须与 `training.seed` 一致；
- `runtime.require_cuda: true` 必须配合 `device: cuda`；
- AMP 只允许用于 CUDA；
- cache 与 telemetry 目录不得相同；
- `cache.schema_version` 必须等于当前候选缓存制品 schema 2；
- 正式坐标适配必须同时提供原帧尺寸、标定身份和仿射矩阵；
- 工程配置用 `calibration_evidence_path` 绑定版本化控制点证据；
- `optimization.max_trials` 只接受正整数或 `null`。

配置加载的公开 API 是：

```python
from pathlib import Path

from traning.config import load_v2_config

config = load_v2_config(Path("configs/traning.yaml"))
```

加载不会迁移旧 schema，也不会用隐式默认值替代缺失的配置文件。候选缓存
schema 1 没有坐标变换指纹，因此配置加载会直接拒绝，而不是将它默认解释为 schema 2。

## 正式 runtime factory

[`app/factory.py`](app/factory.py) 是模型配置到正式 runtime 的唯一装配边界。调用方先按
checkpoint manifest 校验并加载三个模型的权重，再交给 factory：

```python
from pathlib import Path

from traning.app import (
    assemble_runtime_pipeline,
    build_frame_coordinate_transform,
    require_v2_environment,
)
from traning.config import load_v2_config
from traning.training import load_runtime_checkpoint

config = load_v2_config(Path("configs/traning.yaml"))
require_v2_environment(config)

coordinate_transform = build_frame_coordinate_transform(config)
loaded_runtime_models = load_runtime_checkpoint(
    Path("artifacts/traning/runtime"),
    config,
    coordinate_transform,
    expected_dataset_id="my-versioned-dataset",
)

pipeline = assemble_runtime_pipeline(
    config,
    models=loaded_runtime_models,
)
```

checkpoint manifest 同时记录数据集 identity、producer、三模型契约摘要、发布时完整
训练配置摘要、权重 SHA-256 和坐标指纹。加载时必须给出期望数据集；模型结构、权重、
数据集或坐标任一漂移都会硬失败，而 telemetry 目录等部署侧路径调整不会错误地使相同
网络权重失效。

`assemble_runtime_pipeline` 会拒绝模型与配置漂移、Perception/Belief embedding 维度不一致、
Belief/Outcome embedding 维度不一致以及不可用的配置设备。它返回有状态的
`V2RuntimePipeline`；对每个 `RuntimeFrame` 调用 `step`，得到 frozen
`RuntimeStepResult`，其中包含稳定排序的 candidates、tracks、active beliefs、outcomes
和 CLICK/WAIT decision。

`build_untrained_runtime_for_smoke` 只用于随机权重的结构 smoke，不是部署 checkpoint
加载器，不能把它的输出当成已训练决策结果。

空候选帧仍会把 `frame_id/frame_index/timestamp_ms` 显式传给 tracker。轨迹在当前帧变成
`EXPIRED` 时仍保留在该帧的 tracks 审计输出中，但 belief snapshot 会立即移除它，因此
不会生成 Outcome，也不会进入 Decision。若有状态 step 中途异常，pipeline 会设置
`requires_reset=True` 并拒绝继续消费帧，调用方必须先调用 `reset()`。

## 坐标方程与指纹

V2 不在渲染层补偏移。训练 target、预测评分逆变换和 gallery overlay 都通过
`build_frame_coordinate_transform(config)` 构造的同一个 `FrameCoordinateTransform`，
其底层复用 `package.AffineOsuVideoTransform`。

默认标定只适用于 `1484 x 846` 原帧，方程方向是 osu! playfield 到原帧像素：

```text
video_x = 2.115860914627143 * osu_x
        + 0.0011971920855575358 * osu_y
        + 242.59057485632047

video_y = 0.0003418231662923798 * osu_x
        + 2.1166805757239477 * osu_y
        + 16.12108357719331
```

例如 `(osu_x, osu_y) = (79.89, 101.22)` 映射为约 `(411.75, 230.40)`，而不是旧偏移结果。

这里必须区分“方程有效性”和“拟合来源可重放性”。该矩阵在 legacy 基线提交 `9ed1486`
中首次出现；仓库现有 5 个独立原视频 ROI 控制点全部通过，平均残差约 `0.449 px`、最大
残差约 `1.394 px`。但当时声称使用的完整 passed train/validation 拟合点、inlier 清单和
摘要没有入库，因此不能由当前仓库重新执行原拟合，也不能把这 5 个验证点冒充原拟合集。
V2 将它如实标识为 `legacy-control-validated-v1`，限制和控制点保存在
[`configs/traning_coordinate_evidence.json`](../../configs/traning_coordinate_evidence.json)。

可复核当前配置、矩阵和全部控制点：

```bash
PYTHONPATH=src:. python -m start v2 coordinate-audit \
  --config configs/traning.yaml
```

默认命令在控制点通过时成功，但 JSON 会保留 `fit_reproducible: false` 和 limitation。
需要把“完整原拟合集可重放”作为发布硬门时，加 `--require-refit-provenance`；当前证据会
按设计非零退出。未来只有保存完整观测、样本 identity、split、inlier、算法参数和有序点集
SHA-256 后，才能发布新的 pass-sample 标定身份。

`FrameCoordinateTransform.transform_fingerprint` 是 canonical 变换规格的稳定指纹：它把
完整矩阵、`transform_identity`、变换版本和标定原帧尺寸序列化后计算 SHA-256，并使用
`transform-<16 hex>` 形式。矩阵、身份或尺寸任一变化都会改变指纹；缓存、checkpoint 或
评分产物若记录了不同指纹，调用方必须拒绝复用。错误原帧尺寸和未配置矩阵都是硬失败，
不存在 centered transform 或渲染补偿兜底。

## 训练与持续参数搜索

正式 CLI 从显式的 `module:factory` 边界取得项目数据和真实阶段 runner：

```bash
PYTHONPATH=src:. python -m start v2 train \
  --config configs/traning.yaml \
  --evaluator my_project.v2_training:build_evaluator \
  --run-id my-v2-run
```

factory 接收一个 `V2Config`，返回实现 `TrialEvaluator.evaluate` 的 typed evaluator。
需要使用固定阶段编排时可返回 `OrchestratedTrialEvaluator`，其 runner 依次执行
Perception → Tracking → Belief → Outcome → Decision → Evaluation。普通阶段 `FAILED`
会成为未通过的 `TrialObservation`，继续选择下一组参数；异常表示执行边界损坏并终止。

同一能力也提供 typed Python API：

```python
from pathlib import Path

from traning.app import run_configured_search
from traning.config import load_v2_config

config = load_v2_config(Path("configs/traning.yaml"))
result = run_configured_search(config, evaluator)
```

这里的 `evaluator` 必须实现 `TrialEvaluator.evaluate`，并为每个 proposal 返回身份一致的
`TrialObservation`。一个 trial 只有在 data、perception、tracking、belief、outcome、
decision 和 golden 七个门禁全部通过时才算 `PASSED`。

默认配置：

```yaml
optimization:
  max_trials: null
```

`null` 表示没有人为 trial 数预算。某次试验未全通过不会停止进程；控制器会从规格表中
选择下一组合法、量化且未重复的参数继续求值。它不会产生负数 `score_threshold`，也不会
重复已经执行过的参数组合。搜索只在以下情况下结束：

1. 全部门禁通过，返回对应 `TrialObservation`，终态为 `PASSED`；
2. 用户设置的正整数预算用完，或整个有限量化参数空间用完，抛出
   `SearchExhaustedError`，终态为 `EXHAUSTED`；
3. 固定输入数据存在 blocking quality issue，或 evaluator 抛出设备、训练、制品异常，
   立即失败。固定坏数据无法靠换超参数修复，不能在无预算模式下无限重试。

因此“无预算”不是伪造无限参数空间；它表示在全通过之前不施加额外 trial 截断。参数空间
真正耗尽仍会显式失败，绝不会被报告为成功。evaluator 自身抛出的数据、训练或设备异常也会
原样传播，不会被改写成 `PASSED`。

传入 `TelemetryReporter` 时，每个完成的 trial 会写入 `search.trial.completed`；正常通过
写入 `search.passed`，空间或预算耗尽写入 `search.exhausted`。

## 失败语义

| 边界 | 明确行为 |
|---|---|
| 配置文件不存在、schema 错误或字段漂移 | `config-check` 非零退出，不使用隐式旧配置 |
| strict 环境检查失败 | `env-check` 输出完整 JSON 后非零退出 |
| `--no-strict` 环境检查失败 | 输出中保留 `ok: false`，命令仅用于诊断 |
| runtime 输入帧重复或时间回退 | 在有状态组件前拒绝，已有状态不推进 |
| runtime stateful step 中途失败 | 锁定 pipeline，必须 `reset()` 后才能开始新序列 |
| 普通 trial/stage 未全门禁通过 | 继续提出下一组未重复参数 |
| 固定 DataQuality blocking issue | runner 零调用并立即失败，不做无效参数循环 |
| 显式预算或整个参数空间耗尽 | typed `EXHAUSTED`，不冒充成功 |

## Phase 0 → Phase 11 代码导航

V2 按严格阶段顺序建立，旧 `src/traning` 只作为冻结参考：

| Phase | V2 位置与职责 |
|---|---|
| 0 | legacy freeze 与 golden regression 基线 |
| 1 | `contracts/`、`config/`、`infrastructure/` |
| 2 | `data/`：cache、quality gate、repositories |
| 3 | `perception/`：dense model、decode、training、runtime |
| 4 | `tracking/`：association 与生命周期 |
| 5 | `belief/`：逐 track 因果状态 |
| 6 | `outcome/oracle/`、`outcome/dataset/`、canonical attribution |
| 7 | `outcome/model.py`、training、calibration、metrics |
| 8 | `decision/`：CLICK/WAIT 与 optimal stopping |
| 9 | `training/`：门禁编排、curriculum、ASHA、hard examples、参数搜索 |
| 10 | `telemetry/`、`visualization/`：Reporter → Store → Renderer |
| 11 | `app/` + `training/checkpoints.py` + `data/calibration.py`：正式 CLI/runtime、持续搜索、checkpoint 与坐标证据收口 |

主要公开入口：

- `traning.config`：严格配置；
- `traning.contracts`：canonical typed contracts；
- `traning.app`：环境检查、runtime factory、坐标 factory、持续搜索；
- `python -m start v2 ...`：仓库级 V2 CLI 命名空间。
