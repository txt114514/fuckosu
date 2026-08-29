# traning Codex Index

> 自动生成文件，请勿手工修改。运行 `python project_index/build_index.py` 重建。

面向 Codex 的低 token 工程导航；先按阶段定位，再读取命中的源码。

## 调用分层

```text
main.py -> core/decision/pipeline.py:TRAINING_STAGES
        -> core/dataset_import (训练集导入、preflight、Dataset/DataLoader)
        -> core/spatial (空间训练与单帧推理流程)
        -> core/temporal (候选缓存窗口与时序训练 smoke)
        -> core/decision (候选缓存与决策编排)
        -> core/optimization (评分、错误归因、参数搜索、SQLite 记录和多目标排序)
        -> core/result_export (结果可视化与图集导出)
        -> core/model_export (训练模型导出与迁移边界)
        -> start/checks (完整训练启动前自检)
tests/startup_checks/runner.py -> settings/runtime/data/core startup checks
tests/full_checks/runner.py -> full pytest checks
        -> lib/data | lib/models | lib/training | lib/metrics | lib/runtime | lib/visualization
        -> state (run / experiment / checkpoint metadata)
```

## Core 入口

| key | Core 入口 | 当前状态 |
|---|---|---|
| `dataset_import` | `core/dataset_import` | 训练集导入、检查、Dataset/DataLoader 已实现 |
| `spatial` | `core/spatial` | 空间训练和单帧推理已实现 |
| `temporal` | `core/temporal` | 候选缓存窗口和首版训练 smoke 已实现 |
| `decision` | `core/decision` | 候选缓存和训练阶段编排已实现 |
| `optimization` | `core/optimization` | 评分、归因、参数搜索、SQLite trial store 和多目标排序已实现 |
| `result_export` | `core/result_export` | 结果可视化和图集导出已实现 |
| `model_export` | `core/model_export` | 训练模型导出迁移边界已建立 |

快速查询：`python project_index/build_index.py --lookup 符号名`。

## 符号索引

覆盖 `146` 个 Python 文件、`1197` 个命名函数/方法、`265` 个类。匿名 lambda 不单独列出。

图例：`F` 模块函数，`M` 方法，`N` 嵌套函数，`C` 类；`IO-R/IO-W` 文件读写，`DB` 数据库，`PROCESS` 外部进程。

## `src/traning/app/cli.py`

职责：模型配置、环境、坐标审计和生产训练 CLI；训练数据与阶段 runner 均由仓库内部装配。
工程依赖：`traning.app.environment`, `traning.app.factory`, `traning.config`, `traning.data`, `traning.training`

- `F L26-L31` `load_checked_config(path: Path) -> V2Config`：加载严格配置，保留原始异常供 CLI 显式报告。 调用：`load_v2_config`。
- `F L35-L45` `config_check(config: Path=typer.Option(Path('configs/traning.yaml'), '--config')) -> None` [CLI]：验证 schema、坐标标定和搜索预算并输出规范化配置。 调用：`_json`, `load_checked_config`, `v2_config_to_dict`。
- `F L49-L74` `env_check(config: Path=typer.Option(Path('configs/traning.yaml'), '--config'), strict: bool=typer.Option(True, '--strict/--no-strict')) -> None` [CLI]：检查实际设备与正式坐标标定；strict 模式以失败码退出。 调用：`_json`, `check_v2_environment`, `load_checked_config`。
- `F L78-L122` `coordinate_audit(config: Path=typer.Option(Path('configs/traning.yaml'), '--config'), require_refit_provenance: bool=typer.Option(False, '--require-refit-provenance/--allow-validation-only')) -> None` [CLI]：复算控制点残差，并明确报告原始拟合集能否重放。 调用：`_json`, `audit_affine_calibration`, `build_frame_coordinate_transform`, `load_affine_calibration_evidence`, `load_checked_config`。
- `F L126-L178` `train(config: Path=typer.Option(Path('configs/traning.yaml'), '--config'), output_root: Path=typer.Option(Path('artifacts/training_runs'), '--output-root'), run_id: str | None=typer.Option(None, '--run-id'), resume: bool=typer.Option(True, '--resume/--no-resume'), check_environment: bool=typer.Option(True, '--check-environment/--no-check-environment')) -> None` [CLI]：运行真实数据生产训练；普通门禁失败会持续提出未重复参数。 调用：`ProductionTrainer`, `ProductionTrainer.run`, `_json`, `build_training_datasets`, `load_checked_config`, `require_v2_environment`。
- `F L181-L190` `_json(value: object) -> str`：以稳定键序和 UTF-8 文本编码 CLI JSON。

## `src/traning/app/environment.py`

职责：检查严格配置要求的设备、CUDA 和共享坐标标定。
工程依赖：`package`, `traning.config`, `traning.data`, `traning.infrastructure`

- `C L20-L25` `EnvironmentCheckStatus(str, Enum)` [CLASS]：单项启动检查的非歧义状态。
- `C L29-L50` `EnvironmentCheckResult` [CLASS]：一项无副作用环境检查的 typed 结果。
- `M L36-L50` `EnvironmentCheckResult.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.message.strip`, `self.name.strip`。
- `C L54-L74` `EnvironmentReport` [CLASS]：V2 启动边界消费的完整不可变检查报告。
- `M L59-L66` `EnvironmentReport.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L69-L74` `EnvironmentReport.ok(self) -> bool` [PROPERTY]：没有阻断失败时返回真；可见 warning 不会伪装成完整证据。
- `C L77-L89` `EnvironmentNotReadyError(RuntimeError)` [CLASS]：配置要求的运行设备不可用。
- `M L80-L89` `EnvironmentNotReadyError.__init__(self, report: EnvironmentReport) -> None`：初始化实例依赖、配置和运行状态。 调用：`super.__init__`。
- `F L92-L197` `check_v2_environment(config: V2Config) -> EnvironmentReport`：检查配置一致性与实际 CUDA 可见性，不修改全局 torch 状态。 调用：`EnvironmentCheckResult`, `EnvironmentReport`, `FrameCoordinateTransform`, `audit_affine_calibration`, `load_affine_calibration_evidence`。
- `F L200-L206` `require_v2_environment(config: V2Config) -> EnvironmentReport`：返回通过报告；任一失败则抛出携带完整报告的 typed error。 调用：`EnvironmentNotReadyError`, `check_v2_environment`。

## `src/traning/app/factory.py`

职责：按模型契约、配置和设备装配正式 runtime，并构造唯一坐标变换。
工程依赖：`package`, `traning.app.runtime`, `traning.belief`, `traning.config`, `traning.data`, `traning.decision`, `traning.outcome`, `traning.perception`, `traning.tracking`, `traning.training.checkpoints`

- `F L19-L65` `assemble_runtime_pipeline(config: V2Config, *, models: RuntimeModelBundle) -> V2RuntimePipeline`：用带 checkpoint/坐标身份的模型 bundle 装配唯一 V2 runtime 链路。 调用：`MultiObjectTracker`, `OptimalStoppingPlanner`, `PerTrackBeliefRuntime`, `PerceptionRuntime`, `V2RuntimePipeline`, `build_frame_coordinate_transform`。
- `F L68-L93` `build_untrained_runtime_for_smoke(config: V2Config) -> V2RuntimePipeline`：构造随机权重 smoke runtime；名称显式禁止将其误当成部署 checkpoint。 调用：`DenseOutcomeModel`, `PerTrackBeliefEncoder`, `PerceptionModel`, `RuntimeModelBundle`, `assemble_runtime_pipeline`, `build_frame_coordinate_transform`。
- `F L96-L111` `build_frame_coordinate_transform(config: V2Config) -> FrameCoordinateTransform`：从显式 V2 标定装配训练、评分与 gallery 共用的坐标对象。 调用：`FrameCoordinateTransform`。

## `src/traning/app/runtime.py`

职责：编排 Perception → Tracking → Belief → Outcome → Decision 的有状态因果推理。
工程依赖：`traning.belief`, `traning.contracts`, `traning.data`, `traning.decision`, `traning.outcome`, `traning.perception`, `traning.tracking`

- `C L27-L160` `RuntimeStepResult` [CLASS]：一次完整 runtime step 的不可变、可审计输出。
- `M L40-L160` `RuntimeStepResult.__post_init__(self) -> None`：验证跨层身份、帧上下文和稳定排序没有在编排中漂移。 调用：`self.coordinate_transform_fingerprint.startswith`, `self.frame_id.strip`。
- `C L163-L344` `V2RuntimePipeline` [CLASS]：按唯一正式链路推进感知、跟踪、信念、结果预测和决策。
- `M L171-L212` `V2RuntimePipeline.__init__(self, perception_runtime: PerceptionRuntime, tracker: MultiObjectTracker, belief_runtime: PerTrackBeliefRuntime, outcome_model: DenseOutcomeModel, planner: OptimalStoppingPlanner, coordinate_transform: FrameCoordinateTransform) -> None`：初始化实例依赖、配置和运行状态。
- `M L215-L218` `V2RuntimePipeline.requires_reset(self) -> bool` [PROPERTY]：状态边界异常后是否必须先清空序列状态。
- `M L221-L224` `V2RuntimePipeline.coordinate_transform(self) -> FrameCoordinateTransform` [PROPERTY]：返回 runtime 帧尺寸、训练制品和离线评分共用的坐标身份。
- `M L226-L273` `V2RuntimePipeline.step(self, frame: RuntimeFrame) -> RuntimeStepResult`：处理一帧并仅在完整决策成功后提交外层帧游标。 调用：`RuntimeStepResult`, `self._belief_runtime.snapshot`, `self._belief_runtime.step`, `self._checked_candidates`, `self._perception_runtime.infer`, `self._planner.plan`。
- `M L275-L282` `V2RuntimePipeline.reset(self) -> None`：清空 tracker、belief 与 pipeline 帧游标，开始一个全新序列。 调用：`self._belief_runtime.clear`, `self._tracker.reset`。
- `M L284-L303` `V2RuntimePipeline._validate_frame(self, frame: RuntimeFrame) -> None`：校验 `frame` 对应的数据或结果。
- `M L305-L333` `V2RuntimePipeline._checked_candidates(self, candidates: tuple[CandidateObservation, ...], frame: RuntimeFrame) -> tuple[CandidateObservation, ...]`：执行 `checked candidates` 对应逻辑。
- `M L335-L344` `V2RuntimePipeline._predict_outcomes(self, beliefs: tuple[BeliefState, ...]) -> tuple[OutcomeDistribution, ...]`：执行 `predict outcomes` 对应逻辑。 调用：`self._outcome_model.predict`。

## `src/traning/app/training.py`

职责：把严格配置接入 typed 参数搜索与训练门禁。
工程依赖：`traning.config`, `traning.contracts`, `traning.telemetry`, `traning.training`

- `C L22-L56` `_ReportingEvaluator` [CLASS]：在不改变 evaluator 结果的前提下发布每个已验证 trial。
- `M L28-L56` `_ReportingEvaluator.evaluate(self, parameters: ParameterVector, trial_index: int) -> TrialObservation`：先验证 evaluator identity，再持久化 trial completion 事件。 调用：`TelemetryEvent`, `_timestamp_ms`, `self.evaluator.evaluate`, `self.reporter.publish`。
- `F L59-L71` `initial_parameter_vector(config: V2Config) -> ParameterVector`：从各领域配置集中构造搜索核心唯一允许的初始参数向量。 调用：`ParameterVector`。
- `F L74-L132` `run_configured_search(config: V2Config, evaluator: TrialEvaluator, *, reporter: TelemetryReporter | None=None, history: tuple[TrialObservation, ...]=(), on_trial_completed: TrialCompletedCallback | None=None) -> TrialObservation`：按配置运行或恢复搜索；默认无预算时仅全门禁通过才返回。 调用：`TelemetryEvent`, `_ReportingEvaluator`, `_publish_terminal_search_event`, `_timestamp_ms`, `initial_parameter_vector`, `reporter.publish`。
- `F L135-L151` `_publish_terminal_search_event(reporter: TelemetryReporter, *, event_type: str, trial_count: int) -> None`：把 PASSED/EXHAUSTED 作为明确终态写入 events 通道。 调用：`TelemetryEvent`, `_timestamp_ms`, `reporter.publish`。
- `F L154-L157` `_timestamp_ms() -> float`：返回非负 Unix 毫秒，用于跨进程遥测排序。

## `src/traning/belief/encoder.py`

职责：用逐 track 因果状态编码观测，输出 BeliefState。
工程依赖：`traning.config`, `traning.contracts`

- `C L27-L65` `BeliefTensorOutput` [CLASS]：单步可训练张量输出。
- `M L41-L65` `BeliefTensorOutput.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L69-L79` `_ObservationFeatureSpec` [CLASS]：观测特征的分组规格，避免训练/契约路径产生两套字段顺序。
- `M L75-L79` `_ObservationFeatureSpec.feature_dim(self) -> int` [PROPERTY]：返回按固定字段顺序拼接后的单轨迹观测维度。
- `C L82-L347` `PerTrackBeliefEncoder(nn.Module)` [CLASS]：projection + 独立多层 GRUCell 的逐轨迹 dense baseline。
- `M L85-L112` `PerTrackBeliefEncoder.__init__(self, config: BeliefConfig, appearance_embedding_dim: int) -> None`：初始化实例依赖、配置和运行状态。 调用：`_ObservationFeatureSpec`, `super.__init__`。
- `M L115-L118` `PerTrackBeliefEncoder.flattened_hidden_dim(self) -> int` [PROPERTY]：契约 ``belief_embedding`` 的精确长度。
- `M L121-L124` `PerTrackBeliefEncoder.input_feature_dim(self) -> int` [PROPERTY]：训练侧 ``observation_features`` 的精确特征长度。
- `M L126-L171` `PerTrackBeliefEncoder.forward_step(self, observation_features: torch.Tensor, previous_hidden: torch.Tensor | None=None) -> BeliefTensorOutput`：执行一个可反向传播的批量因果步骤。 调用：`BeliefTensorOutput`, `self._validate_feature_tensor`, `self._validate_hidden_tensor`, `self.position_delta_head`, `self.position_uncertainty_head`, `self.projection`。
- `M L173-L217` `PerTrackBeliefEncoder.step(self, observation: TrackedObservation, previous: BeliefState | None=None) -> BeliefState`：把一个 typed tracking 观测推进为公共 ``BeliefState``。 调用：`BeliefState`, `ObjectTypeDistribution`, `Point2D`, `self._hidden_from_belief`, `self._validate_contract_step`, `self.forward_step`。
- `M L219-L237` `PerTrackBeliefEncoder.observation_features(self, observation: TrackedObservation, previous: BeliefState | None=None) -> torch.Tensor`：把 typed 观测编码为训练与 runtime 共用的单行特征张量。 调用：`self._observation_features`, `self._validate_contract_step`, `self.parameters`。
- `M L239-L251` `PerTrackBeliefEncoder._validate_feature_tensor(self, features: torch.Tensor) -> None`：校验 `feature tensor` 对应的数据或结果。
- `M L253-L262` `PerTrackBeliefEncoder._validate_hidden_tensor(self, hidden: torch.Tensor, *, batch: int) -> None`：校验 `hidden tensor` 对应的数据或结果。
- `M L264-L292` `PerTrackBeliefEncoder._validate_contract_step(self, observation: TrackedObservation, previous: BeliefState | None) -> None`：校验 `contract step` 对应的数据或结果。
- `M L294-L335` `PerTrackBeliefEncoder._observation_features(self, observation: TrackedObservation, previous: BeliefState | None, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor`：执行 `observation features` 对应逻辑。
- `M L337-L347` `PerTrackBeliefEncoder._hidden_from_belief(self, previous: BeliefState | None, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None`：执行 `hidden from belief` 对应逻辑。

## `src/traning/belief/runtime.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts`

- `C L12-L107` `PerTrackBeliefRuntime` [CLASS]：维护显式公共 belief，不保存模型私有 hidden side state。
- `M L15-L21` `PerTrackBeliefRuntime.__init__(self, encoder: PerTrackBeliefEncoder) -> None`：初始化实例依赖、配置和运行状态。
- `M L23-L59` `PerTrackBeliefRuntime.step(self, observations: Sequence[TrackedObservation]) -> tuple[BeliefState, ...]`：按稳定 ID 推进整批观测，全部成功后才提交状态。 调用：`self._checked_observations`, `self.encoder.step`, `staged.get`。
- `M L61-L68` `PerTrackBeliefRuntime.state_for(self, track_id: str) -> BeliefState | None`：读取单条轨迹的不可变 belief。 调用：`self._states.get`。
- `M L70-L73` `PerTrackBeliefRuntime.snapshot(self) -> tuple[BeliefState, ...]`：返回按稳定 track ID 排序的不可变状态快照。
- `M L75-L80` `PerTrackBeliefRuntime.clear(self) -> None`：明确清空所有运行时轨迹状态。
- `M L83-L107` `PerTrackBeliefRuntime._checked_observations(observations: Sequence[TrackedObservation]) -> tuple[TrackedObservation, ...]`：执行 `checked observations` 对应逻辑。

## `src/traning/belief/training.py`

职责：从 typed belief 样本构造批次、损失和运行时可复用特征。
工程依赖：`traning.contracts`

- `C L30-L53` `BeliefTrainingRecord` [CLASS]：一条逐轨迹观测及其仅训练侧可见的状态监督。
- `M L39-L53` `BeliefTrainingRecord.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L57-L93` `BeliefTrainingBatch` [CLASS]：保持轨迹身份的可微分 belief 训练批次。
- `M L67-L93` `BeliefTrainingBatch.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L97-L105` `BeliefLoss` [CLASS]：Belief 位置、可见性、类型和不确定性监督分解。
- `F L108-L157` `collate_belief_records(encoder: PerTrackBeliefEncoder, records: tuple[BeliefTrainingRecord, ...]) -> BeliefTrainingBatch`：通过 encoder 的唯一特征入口拼接逐轨迹训练记录。 调用：`BeliefTrainingBatch`, `encoder.observation_features`。
- `F L160-L196` `compute_belief_loss(output: BeliefTensorOutput, batch: BeliefTrainingBatch) -> BeliefLoss`：监督可部署 belief 字段，并让不确定性拟合真实位置残差。 调用：`BeliefLoss`。
- `F L199-L244` `belief_states_from_output(output: BeliefTensorOutput, batch: BeliefTrainingBatch) -> tuple[BeliefState, ...]`：将 detached 模型输出提交为下一因果步使用的公共 BeliefState。 调用：`BeliefState`, `ObjectTypeDistribution`, `Point2D`。

## `src/traning/config/models.py`

职责：定义唯一严格配置、跨字段约束以及配置摘要。
工程依赖：`package`

- `C L29-L33` `RuntimeDevice(str, Enum)` [CLASS]：V2 允许使用的计算设备。
- `F L36-L41` `_require_int(name: str, value: object, *, minimum: int=0) -> int`：执行 `require int` 对应逻辑。
- `F L44-L60` `_require_real(name: str, value: object, *, minimum: float | None=None, maximum: float | None=None) -> float`：执行 `require real` 对应逻辑。
- `F L63-L66` `_require_bool(name: str, value: object) -> bool`：执行 `require bool` 对应逻辑。
- `F L69-L75` `_require_path(name: str, value: object) -> Path`：执行 `require path` 对应逻辑。
- `F L78-L88` `_require_horizons(name: str, value: object) -> tuple[int, ...]`：执行 `require horizons` 对应逻辑。 调用：`_require_int`。
- `F L91-L117` `_optional_affine_matrix(name: str, value: object) -> AffineMatrix | None`：严格解析 2×3 矩阵，并复用共享 transform 验证可逆性。 调用：`_require_real`。
- `C L121-L145` `PerceptionConfig` [CLASS]：单帧候选感知配置。
- `M L134-L145` `PerceptionConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_bool`, `_require_int`, `_require_real`。
- `C L149-L198` `TrackingConfig` [CLASS]：跨帧候选关联和轨迹生命周期配置。
- `M L161-L198` `TrackingConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_int`, `_require_real`。
- `C L202-L216` `BeliefConfig` [CLASS]：每条轨迹的时序信念模型配置。
- `M L210-L216` `BeliefConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_int`。
- `C L220-L237` `OutcomeConfig` [CLASS]：点击结果分布模型配置。
- `M L228-L237` `OutcomeConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_horizons`, `_require_int`。
- `C L241-L263` `DecisionConfig` [CLASS]：基于未来价值的最优停止决策配置。
- `M L253-L263` `DecisionConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_horizons`, `_require_real`。
- `C L267-L275` `DataLoaderConfig` [CLASS]：typed 样本 DataLoader 的进程与锁页内存配置。
- `M L273-L275` `DataLoaderConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_bool`, `_require_int`。
- `C L279-L314` `DataConfig` [CLASS]：segment 数据发现、item 划分、确定性取帧与加载配置。
- `M L294-L314` `DataConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_int`, `_require_path`, `_require_real`。
- `C L318-L347` `CoordinateConfig` [CLASS]：与 affine 标定绑定的原视频尺寸、方程及可选审计证据。
- `M L327-L347` `CoordinateConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_optional_affine_matrix`, `_require_int`, `_require_path`, `self.transform_identity.strip`。
- `C L351-L365` `CacheConfig` [CLASS]：候选缓存仓库配置。
- `M L357-L365` `CacheConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_int`, `_require_path`。
- `C L369-L384` `RuntimeConfig` [CLASS]：设备与数值运行时配置。
- `M L376-L384` `RuntimeConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_bool`。
- `C L388-L402` `TelemetryConfig` [CLASS]：只读遥测事件存储配置。
- `M L394-L402` `TelemetryConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_int`, `_require_path`。
- `C L406-L422` `TrainingConfig` [CLASS]：V2 各训练阶段共享的优化配置。
- `M L415-L422` `TrainingConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_int`, `_require_real`。
- `C L426-L433` `OptimizationConfig` [CLASS]：严格验收参数搜索的停止预算；None 表示不设 trial 数上限。
- `M L431-L433` `OptimizationConfig.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_int`。
- `C L437-L490` `V2Config` [CLASS]：OSU V2 的单一顶层配置。
- `M L454-L490` `V2Config.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_int`。
- `F L496-L502` `_mapping(name: str, value: object) -> Mapping[str, object]`：执行 `mapping` 对应逻辑。
- `F L505-L517` `_section(parent: Mapping[str, object], name: str, model_type: type[_T]) -> Mapping[str, object]`：执行 `section` 对应逻辑。 调用：`_mapping`, `parent.get`。
- `F L520-L521` `_value(section: Mapping[str, object], name: str, default: _T) -> object | _T`：执行 `value` 对应逻辑。
- `F L524-L529` `_path_value(name: str, value: object) -> Path`：执行 `path value` 对应逻辑。
- `F L532-L535` `_optional_path_value(name: str, value: object) -> Path | None`：执行 `optional path value` 对应逻辑。 调用：`_path_value`。
- `F L538-L543` `_tuple_of_ints(name: str, value: object) -> tuple[int, ...]`：执行 `tuple of ints` 对应逻辑。 调用：`_require_int`。
- `F L546-L551` `_optional_positive_int(name: str, value: object) -> int | None`：严格解析可选正整数，不把布尔值或字符串当作上限。 调用：`_require_int`。
- `F L554-L602` `_parse_perception(root: Mapping[str, object]) -> PerceptionConfig`：解析 `perception` 对应的数据或结果。 调用：`PerceptionConfig`, `_require_bool`, `_require_int`, `_require_real`, `_section`, `_value`。
- `F L605-L656` `_parse_tracking(root: Mapping[str, object]) -> TrackingConfig`：解析 `tracking` 对应的数据或结果。 调用：`TrackingConfig`, `_require_int`, `_require_real`, `_section`, `_value`。
- `F L659-L679` `_parse_belief(root: Mapping[str, object]) -> BeliefConfig`：解析 `belief` 对应的数据或结果。 调用：`BeliefConfig`, `_require_int`, `_section`, `_value`。
- `F L682-L702` `_parse_outcome(root: Mapping[str, object]) -> OutcomeConfig`：解析 `outcome` 对应的数据或结果。 调用：`OutcomeConfig`, `_require_int`, `_section`, `_tuple_of_ints`, `_value`。
- `F L705-L748` `_parse_decision(root: Mapping[str, object]) -> DecisionConfig`：解析 `decision` 对应的数据或结果。 调用：`DecisionConfig`, `_require_real`, `_section`, `_tuple_of_ints`, `_value`。
- `F L751-L812` `_parse_data(root: Mapping[str, object]) -> DataConfig`：解析 `data` 对应的数据或结果。 调用：`DataConfig`, `DataLoaderConfig`, `_optional_positive_int`, `_path_value`, `_require_bool`, `_require_int`。
- `F L815-L848` `_parse_coordinates(root: Mapping[str, object]) -> CoordinateConfig`：解析与原帧尺寸强绑定的可选 affine 标定。 调用：`CoordinateConfig`, `_optional_affine_matrix`, `_optional_path_value`, `_require_int`, `_section`, `_value`。
- `F L851-L863` `_parse_cache(root: Mapping[str, object]) -> CacheConfig`：解析 `cache` 对应的数据或结果。 调用：`CacheConfig`, `_path_value`, `_require_int`, `_section`, `_value`。
- `F L866-L883` `_parse_runtime(root: Mapping[str, object]) -> RuntimeConfig`：解析 `runtime` 对应的数据或结果。 调用：`RuntimeConfig`, `RuntimeDevice`, `_require_bool`, `_section`, `_value`。
- `F L886-L898` `_parse_telemetry(root: Mapping[str, object]) -> TelemetryConfig`：解析 `telemetry` 对应的数据或结果。 调用：`TelemetryConfig`, `_path_value`, `_require_int`, `_section`, `_value`。
- `F L901-L924` `_parse_training(root: Mapping[str, object]) -> TrainingConfig`：解析 `training` 对应的数据或结果。 调用：`TrainingConfig`, `_require_int`, `_require_real`, `_section`, `_value`。
- `F L927-L938` `_parse_optimization(root: Mapping[str, object]) -> OptimizationConfig`：解析显式 trial 上限；null 保持“未全通过就继续”的默认语义。 调用：`OptimizationConfig`, `_require_int`, `_section`, `_value`。
- `F L941-L952` `_load_path(path: Path) -> Mapping[str, object]` [IO-R]：加载 `path` 对应的数据或结果。 调用：`_mapping`。
- `F L955-L995` `load_v2_config(source: Mapping[str, object] | Path) -> V2Config`：从严格映射或 JSON/YAML 文件加载 V2 配置。 调用：`V2Config`, `_load_path`, `_mapping`, `_parse_belief`, `_parse_cache`, `_parse_coordinates`。
- `F L998-L1080` `v2_config_to_dict(config: V2Config) -> dict[str, object]`：把已验证配置转换为可直接交给 JSON 编码器的字典。

## `src/traning/contracts/artifact.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L10-L54` `ArtifactManifest` [CLASS]：制品身份、来源和完整性信息。
- `M L24-L54` `ArtifactManifest.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_identifier`, `require_nonnegative`, `require_sha256`。

## `src/traning/contracts/belief.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L15-L50` `BeliefState` [CLASS]：只由历史观测形成的因果轨迹信念。
- `M L29-L50` `BeliefState.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_finite`, `require_identifier`, `require_nonnegative`, `require_probability`。

## `src/traning/contracts/common.py`

职责：Python 模块；具体职责见下方符号及调用。

- `F L15-L21` `require_identifier(value: str, field_name: str) -> None`：校验稳定标识符，拒绝空白和首尾空格。
- `F L24-L30` `require_finite(value: float, field_name: str) -> None`：拒绝 NaN 和无穷值。
- `F L33-L38` `require_nonnegative(value: float, field_name: str) -> None`：校验有限非负数。 调用：`require_finite`。
- `F L41-L46` `require_probability(value: float, field_name: str) -> None`：校验闭区间 [0, 1] 内的有限概率。 调用：`require_finite`。
- `F L49-L55` `require_probability_sum(values: tuple[float, ...], field_name: str) -> None`：校验归一化概率分布。 调用：`require_probability`。
- `F L58-L64` `require_sha256(value: str) -> None`：校验小写或大写 SHA-256 十六进制摘要。
- `F L67-L80` `require_transform_fingerprint(value: str, field_name: str='transform_fingerprint') -> None`：校验共享坐标 API 生成的稳定变换指纹。

## `src/traning/contracts/data.py`

职责：隔离训练样本、推理帧、训练候选和推理候选，禁止运行时读取 GT-only 字段。
工程依赖：`package`

- `C L27-L67` `GroundTruthObject` [CLASS]：仅训练侧可见的 canonical osu! 目标及其类型几何。
- `M L39-L67` `GroundTruthObject.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_finite`, `require_identifier`, `require_nonnegative`。
- `C L71-L93` `TrainingCandidateRecord` [CLASS]：候选观测与训练专属监督的组合。
- `M L80-L93` `TrainingCandidateRecord.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_finite`, `require_identifier`。
- `C L97-L108` `InferenceCandidateRecord` [CLASS]：推理侧候选记录；类型上不提供任何训练真值。
- `M L103-L108` `InferenceCandidateRecord.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_identifier`。
- `C L112-L139` `RuntimeFrame` [CLASS]：运行时输入帧；只携带图像和运行时身份信息。
- `M L122-L139` `RuntimeFrame.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_identifier`, `require_nonnegative`。
- `C L143-L214` `TrainingSample` [CLASS]：完整训练帧及其仅训练侧可见的监督。
- `M L162-L214` `TrainingSample.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_identifier`, `require_nonnegative`, `require_transform_fingerprint`。
- `C L218-L284` `OutcomeTrainingSample` [CLASS]：反事实 Outcome 模型的训练样本；target_score 是归一化 oracle 分数。
- `M L235-L284` `OutcomeTrainingSample.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_identifier`, `require_nonnegative`, `require_probability`。

## `src/traning/contracts/decision.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L11-L15` `DecisionAction(str, Enum)` [CLASS]：正式运行路径允许的基础动作。
- `C L19-L74` `DecisionResult` [CLASS]：规划器选出的动作及其可审计效用。
- `M L32-L74` `DecisionResult.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_finite`, `require_identifier`, `require_nonnegative`。

## `src/traning/contracts/observation.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L17-L23` `ObjectType(str, Enum)` [CLASS]：候选物体类型。
- `C L27-L39` `ObjectTypeDistribution` [CLASS]：候选物体类型的规范化概率分布。
- `M L35-L39` `ObjectTypeDistribution.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_probability_sum`。
- `C L43-L51` `Point2D` [CLASS]：二维有限坐标。
- `M L49-L51` `Point2D.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_finite`。
- `C L55-L63` `RingAttributes` [CLASS]：ring 分支能在 runtime 直接观测到的概率与像素半径。
- `M L61-L63` `RingAttributes.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_nonnegative`, `require_probability`。
- `C L67-L77` `SliderAttributes` [CLASS]：slider 分支概率、局部方向和可选的已解码路径。
- `M L74-L77` `SliderAttributes.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_probability`。
- `C L81-L87` `SpinnerAttributes` [CLASS]：spinner 分支在当前帧的存在概率。
- `M L86-L87` `SpinnerAttributes.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_probability`。
- `C L91-L132` `CandidateObservation` [CLASS]：单帧感知产生的候选观测，不包含任何真值。
- `M L108-L132` `CandidateObservation.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_finite`, `require_identifier`, `require_nonnegative`, `require_probability`。
- `C L135-L141` `TrackLifecycle(str, Enum)` [CLASS]：轨迹生命周期状态。
- `C L144-L149` `AssociationStatus(str, Enum)` [CLASS]：当前帧的关联结果。
- `C L153-L233` `TrackedObservation` [CLASS]：带稳定轨迹身份、生命周期和关联信息的观测。
- `M L169-L233` `TrackedObservation.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_identifier`, `require_nonnegative`, `require_probability`。

## `src/traning/contracts/outcome.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L18-L25` `OutcomeCategory(IntEnum)` [CLASS]：Outcome 模型固定的五类互斥标签与 tensor channel 顺序。
- `C L29-L58` `OutcomeDistribution` [CLASS]：五类互斥结果及独立过期事件的预测分布。
- `M L43-L58` `OutcomeDistribution.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_identifier`, `require_nonnegative`, `require_probability`, `require_probability_sum`。

## `src/traning/contracts/quality.py`

职责：定义统一 DataQualityIssue 与 blocking 质量门语义。

- `C L9-L14` `DataQualitySeverity(str, Enum)` [CLASS]：质量问题展示严重度；是否阻断由独立字段决定。
- `C L18-L36` `DataQualityIssue` [CLASS]：可定位、可解释的数据质量问题。
- `M L28-L36` `DataQualityIssue.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_identifier`, `self.message.strip`。
- `C L40-L49` `DataQualityReport` [CLASS]：质量门报告；ok 只由 blocks_training 推导。
- `M L46-L49` `DataQualityReport.ok(self) -> bool` [PROPERTY]：仅按 ``blocks_training`` 汇总唯一的质量门结论。

## `src/traning/contracts/telemetry.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L9-L40` `TelemetryEvent` [CLASS]：带 schema 版本的不可变事件快照。
- `M L19-L40` `TelemetryEvent.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_finite`, `require_identifier`, `require_nonnegative`。

## `src/traning/data/cache/cache.py`

职责：以临时文件、fsync、行数和 SHA-256 校验原子发布候选缓存。
工程依赖：`traning.config.versions`, `traning.contracts`, `traning.infrastructure`

- `C L62-L139` `CandidateCacheManifest` [CLASS]：候选缓存清单；缓存身份必须包含坐标变换指纹。
- `M L67-L91` `CandidateCacheManifest.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_transform_fingerprint`。
- `M L94-L97` `CandidateCacheManifest.schema_version(self) -> int` [PROPERTY]：返回通用制品清单中的缓存 schema 版本。
- `M L100-L103` `CandidateCacheManifest.dataset_id(self) -> str` [PROPERTY]：返回生成本缓存的数据集稳定标识。
- `M L106-L109` `CandidateCacheManifest.producer_id(self) -> str` [PROPERTY]：返回生成候选记录的模型或生产者标识。
- `M L112-L115` `CandidateCacheManifest.row_count(self) -> int` [PROPERTY]：返回清单承诺的候选记录总行数。
- `M L118-L121` `CandidateCacheManifest.sha256(self) -> str` [PROPERTY]：返回不可变 records generation 的 SHA-256。
- `M L124-L130` `CandidateCacheManifest.records_filename(self) -> str` [PROPERTY]：返回 manifest 已提交的不可变 records generation。
- `M L133-L139` `CandidateCacheManifest.transform_fingerprint(self) -> str` [PROPERTY]：返回生成本缓存时的完整坐标变换指纹。
- `C L143-L163` `CandidateCacheDataset` [CLASS]：将加载出的候选记录与其完整 manifest 来源永久绑定。
- `M L149-L157` `CandidateCacheDataset.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L160-L163` `CandidateCacheDataset.transform_fingerprint(self) -> str` [PROPERTY]：返回候选像素坐标所绑定的变换指纹。
- `C L166-L255` `CandidateCacheWriter` [CLASS]：先完整发布记录文件，再以摘要和行数生成并发布清单。
- `M L169-L186` `CandidateCacheWriter.__init__(self, cache_dir: Path, *, artifact_id: str, dataset_id: str, producer_id: str, transform_fingerprint: str, split: DataSplit=DataSplit.ALL, schema_version: int=CANDIDATE_CACHE_SCHEMA_VERSION) -> None`：初始化实例依赖、配置和运行状态。
- `M L188-L229` `CandidateCacheWriter.write(self, records: Iterable[InferenceCandidateRecord], *, created_at_ms: float | None=None) -> CandidateCacheManifest`：原子写入全量 JSONL，校验落盘内容后发布对应清单。 调用：`_manifest_to_json`, `atomic_write_json`, `atomic_write_jsonl`, `encoded_records`, `self._build_manifest`, `sha256_file`。
- `N L206-L214` `CandidateCacheWriter.write.encoded_records() -> Iterable[JSONObject]`：逐条校验推理记录并生成严格 JSON object。 调用：`_record_to_json`。
- `M L231-L255` `CandidateCacheWriter._build_manifest(self, *, row_count: int, digest: str, created_at_ms: float, records_filename: str) -> CandidateCacheManifest`：构建 `manifest` 对应的数据或结果。 调用：`ArtifactManifest`, `CandidateCacheManifest`。
- `C L258-L340` `CandidateCacheReader` [CLASS]：仅加载身份、版本、行数和摘要全部符合预期的候选缓存。
- `M L261-L281` `CandidateCacheReader.__init__(self, cache_dir: Path, *, expected_artifact_id: str, expected_dataset_id: str, expected_producer_id: str, expected_transform_fingerprint: str, expected_schema_version: int=CANDIDATE_CACHE_SCHEMA_VERSION) -> None`：初始化实例依赖、配置和运行状态。 调用：`require_transform_fingerprint`。
- `M L283-L305` `CandidateCacheReader.read_manifest(self) -> CandidateCacheManifest`：读取清单并硬校验调用方要求的缓存身份。 调用：`SchemaMismatchError`, `_manifest_from_json`, `read_json_object`。
- `M L307-L338` `CandidateCacheReader.read(self) -> CandidateCacheDataset` [IO-W]：验证完整性后返回不会丢失 manifest 来源的 typed dataset。 调用：`CandidateCacheDataset`, `IntegrityError`, `SchemaMismatchError`, `_decode_record_line`, `self.read_manifest`, `sha256_file`。
- `F L343-L361` `publish_candidate_cache(directory: Path, records: Iterable[InferenceCandidateRecord], *, dataset_id: str, producer_id: str, transform_fingerprint: str, created_at_ms: float) -> ArtifactManifest` [IO-W]：发布候选缓存；坐标指纹是清单身份的必填部分。 调用：`CandidateCacheWriter`, `CandidateCacheWriter.write`。
- `F L364-L382` `load_candidate_cache(directory: Path, *, expected_artifact_id: str, expected_dataset_id: str, expected_producer_id: str, expected_transform_fingerprint: str, expected_schema_version: int=CANDIDATE_CACHE_SCHEMA_VERSION) -> CandidateCacheDataset` [IO-R]：加载经全部身份与完整性校验、且保留 manifest 的缓存。 调用：`CandidateCacheReader`, `CandidateCacheReader.read`。
- `F L385-L410` `_record_to_json(record: InferenceCandidateRecord) -> JSONObject`：执行 `record to json` 对应逻辑。 调用：`_ring_to_json`, `_slider_to_json`, `_spinner_to_json`。
- `F L413-L418` `_ring_to_json(value: RingAttributes | None) -> JSONValue`：执行 `ring to json` 对应逻辑。
- `F L421-L428` `_slider_to_json(value: SliderAttributes | None) -> JSONValue`：执行 `slider to json` 对应逻辑。
- `F L431-L432` `_spinner_to_json(value: SpinnerAttributes | None) -> JSONValue`：执行 `spinner to json` 对应逻辑。
- `F L435-L447` `_decode_record_line(raw_line: bytes, line_number: int) -> InferenceCandidateRecord`：执行 `decode record line` 对应逻辑。 调用：`IntegrityError`, `SchemaMismatchError`, `_record_from_json`。
- `F L450-L463` `_record_from_json(payload: JSONObject, line_number: int) -> InferenceCandidateRecord`：执行 `record from json` 对应逻辑。 调用：`InferenceCandidateRecord`, `SchemaMismatchError`, `_object`, `_observation_from_json`, `_require_fields`, `_string`。
- `F L466-L537` `_observation_from_json(payload: JSONObject, context: str) -> CandidateObservation`：执行 `observation from json` 对应逻辑。 调用：`CandidateObservation`, `ObjectTypeDistribution`, `SchemaMismatchError`, `_array`, `_integer`, `_number`。
- `F L540-L548` `_ring_from_json(value: JSONValue, context: str) -> RingAttributes | None`：执行 `ring from json` 对应逻辑。 调用：`RingAttributes`, `_number`, `_object`, `_require_fields`。
- `F L551-L577` `_slider_from_json(value: JSONValue, context: str) -> SliderAttributes | None`：执行 `slider from json` 对应逻辑。 调用：`Point2D`, `SliderAttributes`, `_array`, `_number`, `_object`, `_require_fields`。
- `F L580-L587` `_spinner_from_json(value: JSONValue, context: str) -> SpinnerAttributes | None`：执行 `spinner from json` 对应逻辑。 调用：`SpinnerAttributes`, `_number`, `_object`, `_require_fields`。
- `F L590-L603` `_manifest_to_json(manifest: CandidateCacheManifest) -> JSONObject`：执行 `manifest to json` 对应逻辑。
- `F L606-L636` `_manifest_from_json(payload: JSONObject) -> CandidateCacheManifest`：执行 `manifest from json` 对应逻辑。 调用：`ArtifactManifest`, `CandidateCacheManifest`, `SchemaMismatchError`, `_array`, `_integer`, `_number`。
- `F L639-L644` `_require_fields(payload: JSONObject, expected: set[str], context: str) -> None`：执行 `require fields` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L647-L650` `_object(value: JSONValue, context: str) -> JSONObject`：执行 `object` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L653-L656` `_array(value: JSONValue, context: str) -> list[JSONValue]`：执行 `array` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L659-L662` `_string(value: JSONValue, context: str) -> str`：执行 `string` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L665-L668` `_integer(value: JSONValue, context: str) -> int`：执行 `integer` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L671-L676` `_number(value: JSONValue, context: str) -> float`：执行 `number` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L679-L685` `_object_without_duplicate_keys(pairs: list[tuple[str, JSONValue]]) -> JSONObject`：执行 `object without duplicate keys` 对应逻辑。 调用：`IntegrityError`。
- `F L688-L689` `_reject_non_finite_constant(value: str) -> JSONValue`：执行 `reject non finite constant` 对应逻辑。 调用：`IntegrityError`。

## `src/traning/data/calibration.py`

职责：读取版本化标定证据并复算共享 affine 方程控制点残差。
工程依赖：`package`, `traning.config`, `traning.infrastructure`

- `F L62-L68` `_require_finite(name: str, value: float) -> None`：拒绝布尔、非数值和非有限坐标。
- `F L71-L75` `_require_text(name: str, value: str) -> None`：要求标识和说明为非空、无首尾空格的字符串。
- `F L78-L86` `_require_sha256(name: str, value: str) -> None`：校验 canonical 小写 SHA-256 文本。
- `C L90-L115` `CalibrationObservation` [CLASS]：一个独立 osu 坐标与原视频像素中心的成对观测。
- `M L100-L115` `CalibrationObservation.__post_init__(self) -> None`：验证标识、坐标范围及可追溯来源。 调用：`_require_finite`, `_require_text`。
- `C L119-L152` `CalibrationFitProvenance` [CLASS]：原始拟合方法、样本数量与样本集合摘要的可用性声明。
- `M L128-L146` `CalibrationFitProvenance.__post_init__(self) -> None`：可复现声明必须同时具备数量和摘要，缺失时必须同时为空。 调用：`_require_sha256`, `_require_text`。
- `M L149-L152` `CalibrationFitProvenance.reproducible(self) -> bool` [PROPERTY]：仅当完整拟合集数量与摘要均可校验时返回真。
- `C L156-L210` `AffineCalibrationEvidence` [CLASS]：绑定矩阵、原帧、拟合来源及独立控制点的严格证据。
- `M L169-L210` `AffineCalibrationEvidence.__post_init__(self) -> None`：验证制品版本、方程、尺寸和控制点集合的不变量。 调用：`_require_finite`, `_require_text`。
- `C L214-L225` `LoadedCalibrationEvidence` [CLASS]：保留证据内容与原始 JSON 字节摘要的加载结果。
- `M L220-L225` `LoadedCalibrationEvidence.__post_init__(self) -> None`：阻止加载边界丢失制品完整性身份。 调用：`_require_sha256`。
- `C L229-L253` `AffineFitResult` [CLASS]：由完整观测集合最小二乘拟合出的方程及残差摘要。
- `M L239-L253` `AffineFitResult.__post_init__(self) -> None`：确保拟合摘要自身有限且能作为后续 manifest 证据。 调用：`_require_finite`, `_require_sha256`。
- `C L257-L310` `CalibrationAuditReport` [CLASS]：配置坐标变换对证据矩阵和独立控制点的完整审计结果。
- `M L274-L299` `CalibrationAuditReport.__post_init__(self) -> None`：校验报告字段，使 CLI 和环境门禁无需重新推导语义。 调用：`_require_finite`, `_require_sha256`, `_require_text`。
- `M L302-L310` `CalibrationAuditReport.ok(self) -> bool` [PROPERTY]：身份、尺寸、方程和全部控制点同时通过时才返回真。
- `F L313-L367` `load_affine_calibration_evidence(path: Path) -> LoadedCalibrationEvidence`：严格加载版本化证据 JSON，并保留原始文件 SHA-256。 调用：`AffineCalibrationEvidence`, `CalibrationFitProvenance`, `LoadedCalibrationEvidence`, `SchemaMismatchError`, `_array`, `_boolean`。
- `F L370-L401` `fit_affine_least_squares(observations: tuple[CalibrationObservation, ...]) -> AffineFitResult`：用全部给定观测确定性拟合 2×3 方程，不执行随机抽样或静默剔除。 调用：`AffineFitResult`, `_matrix_from_array`, `_observation_sha256`, `_validated_observations`。
- `F L404-L449` `audit_affine_calibration(transform: FrameCoordinateTransform, loaded: LoadedCalibrationEvidence) -> CalibrationAuditReport`：以正式共享变换复算每个控制点，返回唯一可消费的门禁报告。 调用：`CalibrationAuditReport`, `_observation_sha256`。
- `F L452-L464` `_validated_observations(observations: tuple[CalibrationObservation, ...]) -> tuple[CalibrationObservation, ...]`：执行 `validated observations` 对应逻辑。
- `F L467-L486` `_observation_sha256(observations: tuple[CalibrationObservation, ...]) -> str`：执行 `observation sha256` 对应逻辑。 调用：`hashlib.sha256`。
- `F L489-L503` `_parse_control(value: object, index: int) -> CalibrationObservation`：解析 `control` 对应的数据或结果。 调用：`CalibrationObservation`, `SchemaMismatchError`, `_number`, `_object`, `_string`。
- `F L506-L521` `_matrix(value: object) -> AffineMatrix`：执行 `matrix` 对应逻辑。 调用：`SchemaMismatchError`, `_array`, `_number`。
- `F L524-L528` `_matrix_from_array(value: np.ndarray) -> AffineMatrix`：执行 `matrix from array` 对应逻辑。
- `F L531-L534` `_object(value: object, name: str) -> dict[str, object]`：执行 `object` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L537-L540` `_array(value: object, name: str) -> list[object]`：执行 `array` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L543-L546` `_string(value: object, name: str) -> str`：执行 `string` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L549-L552` `_optional_string(value: object, name: str) -> str | None`：执行 `optional string` 对应逻辑。 调用：`_string`。
- `F L555-L558` `_integer(value: object, name: str) -> int`：执行 `integer` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L561-L564` `_optional_integer(value: object, name: str) -> int | None`：执行 `optional integer` 对应逻辑。 调用：`_integer`。
- `F L567-L573` `_number(value: object, name: str) -> float`：执行 `number` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L576-L579` `_boolean(value: object, name: str) -> bool`：执行 `boolean` 对应逻辑。 调用：`SchemaMismatchError`。

## `src/traning/data/coordinates.py`

职责：构造绑定原帧尺寸、标定身份和方程指纹的共享训练帧坐标变换。
工程依赖：`package`, `traning.contracts`

- `F L25-L31` `_require_finite(value: float, field_name: str) -> None`：校验坐标是有限实数，并显式拒绝布尔值。
- `F L34-L45` `_require_frame_size(width: int, height: int) -> None`：校验原帧尺寸，避免 ``bool`` 被当成整数接受。
- `C L49-L63` `OsuPoint` [CLASS]：osu!standard playfield 内的有界坐标。
- `M L55-L63` `OsuPoint.__post_init__(self) -> None`：拒绝非有限值和 playfield 边界外的点。 调用：`_require_finite`。
- `C L67-L81` `CanonicalScoringPoint` [CLASS]：由原帧逆变换得到、允许落在 playfield 外的评分坐标。
- `M L78-L81` `CanonicalScoringPoint.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_finite`, `require_transform_fingerprint`。
- `C L85-L104` `FramePixelPoint` [CLASS]：与具体原帧尺寸和坐标变换指纹绑定的有界像素坐标。
- `M L94-L104` `FramePixelPoint.__post_init__(self) -> None`：将尺寸与边界校验放在领域对象边界。 调用：`_require_finite`, `_require_frame_size`, `require_transform_fingerprint`。
- `C L108-L292` `FrameCoordinateTransform` [CLASS]：将一个共享仿射变换与标定原帧的尺寸、身份和指纹绑定。
- `M L121-L163` `FrameCoordinateTransform.__post_init__(self) -> None`：验证变换来源，并用共享规格 API 生成稳定指纹。 调用：`OsuPoint`, `_require_frame_size`, `self._transform_osu_to_frame`, `self.transform.spec`, `self.transform_identity.strip`。
- `M L165-L170` `FrameCoordinateTransform._require_bound_size(self, width: int, height: int) -> None`：要求消费者声明的原帧尺寸与标定尺寸完全一致。 调用：`_require_frame_size`。
- `M L172-L182` `FrameCoordinateTransform._transform_osu_to_frame(self, point: OsuPoint) -> FramePixelPoint`：统一的 osu -> 原帧实现；三个消费者不得自行算系数。 调用：`FramePixelPoint`, `self.transform.osu_to_video`。
- `M L184-L201` `FrameCoordinateTransform.bind_frame_prediction(self, *, x: float, y: float, source_frame_width: int, source_frame_height: int) -> FramePixelPoint`：把 runtime 原帧预测绑定到本变换指纹，供 canonical scoring 使用。 调用：`FramePixelPoint`, `self._require_bound_size`。
- `M L203-L213` `FrameCoordinateTransform.ground_truth_to_training_target(self, point: OsuPoint, *, source_frame_width: int, source_frame_height: int) -> FramePixelPoint`：把 osu GT 转成稠密感知训练使用的原帧像素 target。 调用：`self._require_bound_size`, `self._transform_osu_to_frame`。
- `M L215-L232` `FrameCoordinateTransform.ground_truth_radius_to_training_target(self, radius_osu: float, *, source_frame_width: int, source_frame_height: int) -> float`：把 osu! 半径投影成与同一标定绑定的原帧像素半径。 调用：`_require_finite`, `self._require_bound_size`, `self.transform.osu_radius_to_video`。
- `M L234-L259` `FrameCoordinateTransform.ground_truth_direction_to_training_target(self, start: Point2D, end: Point2D, *, source_frame_width: int, source_frame_height: int) -> tuple[float, float]`：把 osu! 路径首段转换为原帧单位方向，允许控制点越过边界。 调用：`self._require_bound_size`, `self.transform.osu_to_video`。
- `M L261-L280` `FrameCoordinateTransform.prediction_to_canonical_scoring(self, point: FramePixelPoint) -> CanonicalScoringPoint`：把原帧预测逆变换为可落在 playfield 外的评分坐标。 调用：`CanonicalScoringPoint`, `self._require_bound_size`, `self.transform.video_to_osu`。
- `M L282-L292` `FrameCoordinateTransform.target_to_gallery_overlay(self, point: OsuPoint, *, source_frame_width: int, source_frame_height: int) -> FramePixelPoint`：把 osu target 转成 gallery 在原帧上绘制的像素点。 调用：`self._require_bound_size`, `self._transform_osu_to_frame`。

## `src/traning/data/pipeline/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`traning.contracts`, `traning.data.quality`

- `C L13-L30` `DataPipelineResult` [CLASS]：通过质量门后发布给下游的 typed 数据结果。
- `M L20-L30` `DataPipelineResult.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L33-L43` `QualityGateBlockedError(RuntimeError)` [CLASS]：数据质量门阻断训练时的明确失败。
- `M L36-L43` `QualityGateBlockedError.__init__(self, report: DataQualityReport) -> None`：初始化实例依赖、配置和运行状态。 调用：`super.__init__`。
- `F L46-L52` `require_quality(report: DataQualityReport) -> None`：仅按 canonical ``report.ok`` 决定是否阻断。 调用：`QualityGateBlockedError`。
- `C L56-L75` `DataPipeline` [CLASS]：确定性整理样本并执行唯一质量门的编排器。
- `M L61-L63` `DataPipeline.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L65-L75` `DataPipeline.run(self, samples: Sequence[TrainingSample]) -> DataPipelineResult`：运行 pipeline；阻断时抛出携带完整 report 的异常。 调用：`DataPipelineResult`, `DataQualityContext.from_samples`, `require_quality`, `self.quality_gate.evaluate`。
- `F L78-L88` `run_data_pipeline(samples: Sequence[TrainingSample], *, quality_gate: DataQualityGate | None=None) -> DataPipelineResult`：函数式入口；不提供跳过质量门的路径。 调用：`DataPipeline`, `DataQualityGate`, `pipeline.run`。

## `src/traning/data/quality/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`traning.contracts`

- `C L21-L63` `DatasetSummary` [CLASS]：数据集的确定性计数摘要。
- `M L27-L45` `DatasetSummary.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L47-L52` `DatasetSummary.count(self, split: DataSplit) -> int`：返回具体切分的样本数。
- `M L55-L63` `DatasetSummary.from_samples(cls, samples: Sequence[TrainingSample]) -> DatasetSummary`：从 typed 样本生成固定顺序的摘要。 调用：`_checked_samples`。
- `C L67-L93` `DataQualityContext` [CLASS]：供质量规则共享的 typed 数据上下文。
- `M L73-L76` `DataQualityContext.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_checked_samples`。
- `M L79-L93` `DataQualityContext.from_samples(cls, samples: Sequence[TrainingSample]) -> DataQualityContext`：按稳定键排序样本并构建摘要。 调用：`DatasetSummary.from_samples`, `_checked_samples`。
- `C L97-L102` `DataQualityFinding` [CLASS]：规则内部发现；公共 issue 元数据由注册表统一补齐。
- `C L109-L125` `DataQualityRule` [CLASS]：一条可替换的数据质量规则规格。
- `M L117-L125` `DataQualityRule.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.code.strip`。
- `F L128-L135` `_checked_samples(samples: Sequence[TrainingSample]) -> tuple[TrainingSample, ...]`：执行 `checked samples` 对应逻辑。
- `F L138-L151` `_summary_matches(catalog: DataQualityContext) -> tuple[DataQualityFinding, ...]`：执行 `summary matches` 对应逻辑。 调用：`DataQualityFinding`, `DatasetSummary.from_samples`。
- `F L154-L168` `_duplicate_sample_ids(catalog: DataQualityContext) -> tuple[DataQualityFinding, ...]`：执行 `duplicate sample ids` 对应逻辑。 调用：`DataQualityFinding`, `counts.get`。
- `F L171-L176` `_missing_training_split(catalog: DataQualityContext) -> tuple[DataQualityFinding, ...]`：执行 `missing training split` 对应逻辑。 调用：`DataQualityFinding`, `catalog.summary.count`。
- `F L179-L190` `_missing_evaluation_splits(catalog: DataQualityContext) -> tuple[DataQualityFinding, ...]`：执行 `missing evaluation splits` 对应逻辑。 调用：`DataQualityFinding`, `catalog.summary.count`。
- `C L223-L262` `DataQualityGate` [CLASS]：按注册顺序执行规则并生成 canonical report。
- `M L228-L237` `DataQualityGate.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L239-L257` `DataQualityGate.evaluate(self, context: DataQualityContext) -> DataQualityReport`：对 typed 上下文执行全部注册规则。 调用：`DataQualityIssue`, `DataQualityReport`, `rule.evaluate`。
- `M L259-L262` `DataQualityGate.evaluate_samples(self, samples: Sequence[TrainingSample]) -> DataQualityReport`：确定性整理 typed 样本后执行质量门。 调用：`DataQualityContext.from_samples`, `self.evaluate`。

## `src/traning/data/repositories/memory.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts.common`

- `C L12-L42` `InMemoryPreprocessingMetadataRepository` [CLASS]：以稳定标识符维护预处理元数据快照。
- `M L15-L18` `InMemoryPreprocessingMetadataRepository.__init__(self, records: Iterable[PreprocessingMetadata]=()) -> None`：初始化实例依赖、配置和运行状态。 调用：`self.save`。
- `M L20-L24` `InMemoryPreprocessingMetadataRepository.get(self, item_name: str) -> PreprocessingMetadata | None`：按预处理条目标识返回当前快照，缺失时返回 ``None``。 调用：`require_identifier`, `self._records.get`。
- `M L26-L29` `InMemoryPreprocessingMetadataRepository.list_all(self) -> tuple[PreprocessingMetadata, ...]`：按条目标识稳定排序并返回全部预处理元数据。
- `M L31-L36` `InMemoryPreprocessingMetadataRepository.save(self, metadata: PreprocessingMetadata) -> None`：以条目标识新增或替换一份 typed 元数据。
- `M L38-L42` `InMemoryPreprocessingMetadataRepository.delete(self, item_name: str) -> bool`：删除指定预处理条目并返回此前是否存在。 调用：`require_identifier`, `self._records.pop`。
- `C L45-L82` `InMemoryDatasetCatalogRepository` [CLASS]：以内存字典维护数据集目录快照。
- `M L48-L51` `InMemoryDatasetCatalogRepository.__init__(self, entries: Iterable[DatasetCatalogEntry]=()) -> None`：初始化实例依赖、配置和运行状态。 调用：`self.save`。
- `M L53-L57` `InMemoryDatasetCatalogRepository.get(self, folder_name: str) -> DatasetCatalogEntry | None`：按目录名返回当前目录条目，缺失时返回 ``None``。 调用：`require_identifier`, `self._entries.get`。
- `M L59-L69` `InMemoryDatasetCatalogRepository.list_all(self, *, active_only: bool=False) -> tuple[DatasetCatalogEntry, ...]`：按序号和目录名稳定排序，可选择只返回活动条目。 调用：`self._entries.values`。
- `M L71-L76` `InMemoryDatasetCatalogRepository.save(self, entry: DatasetCatalogEntry) -> None`：以目录名新增或替换一份 typed 目录条目。
- `M L78-L82` `InMemoryDatasetCatalogRepository.delete(self, folder_name: str) -> bool`：删除指定目录条目并返回此前是否存在。 调用：`require_identifier`, `self._entries.pop`。

## `src/traning/data/repositories/models.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts.common`

- `F L11-L15` `_require_positive_integer(value: int, field_name: str) -> None`：执行 `require positive integer` 对应逻辑。
- `F L18-L22` `_require_nonnegative_integer(value: int, field_name: str) -> None`：执行 `require nonnegative integer` 对应逻辑。
- `F L25-L27` `_require_optional_text(value: str | None, field_name: str) -> None`：执行 `require optional text` 对应逻辑。 调用：`require_identifier`。
- `C L31-L55` `PreprocessingMetadata` [CLASS]：单个数据项完成视频预处理后形成的坐标链元数据。
- `M L43-L55` `PreprocessingMetadata.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_nonnegative_integer`, `_require_positive_integer`, `require_identifier`。
- `C L59-L89` `DatasetCatalogEntry` [CLASS]：训练数据目录中的一个可寻址数据项。
- `M L71-L89` `DatasetCatalogEntry.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_nonnegative_integer`, `_require_optional_text`, `require_identifier`。
- `C L92-L93` `RepositoryError(RuntimeError)` [CLASS]：repository 无法完成持久化操作时抛出的显式错误。

## `src/traning/data/repositories/protocols.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L11-L24` `PreprocessingMetadataRepository(Protocol)` [CLASS]：预处理元数据的稳定读写契约。
- `M L14-L15` `PreprocessingMetadataRepository.get(self, item_name: str) -> PreprocessingMetadata | None`：按数据项名称读取元数据；不存在时返回 ``None``。
- `M L17-L18` `PreprocessingMetadataRepository.list_all(self) -> tuple[PreprocessingMetadata, ...]`：按数据项名称返回不可变快照。
- `M L20-L21` `PreprocessingMetadataRepository.save(self, metadata: PreprocessingMetadata) -> None`：新增或完整替换一条元数据。
- `M L23-L24` `PreprocessingMetadataRepository.delete(self, item_name: str) -> bool`：删除元数据，并返回删除前是否存在。
- `C L28-L41` `DatasetCatalogRepository(Protocol)` [CLASS]：数据集目录的稳定读写契约。
- `M L31-L32` `DatasetCatalogRepository.get(self, folder_name: str) -> DatasetCatalogEntry | None`：按目录名称读取条目；不存在时返回 ``None``。
- `M L34-L35` `DatasetCatalogRepository.list_all(self, *, active_only: bool=False) -> tuple[DatasetCatalogEntry, ...]`：按 sequence、folder_name 返回不可变目录快照。
- `M L37-L38` `DatasetCatalogRepository.save(self, entry: DatasetCatalogEntry) -> None`：新增或完整替换一个目录条目。
- `M L40-L41` `DatasetCatalogRepository.delete(self, folder_name: str) -> bool`：删除目录条目，并返回删除前是否存在。

## `src/traning/data/repositories/sqlite.py`

职责：在 infrastructure 边界内读取预处理 SQLite，并返回 typed repository 记录。
工程依赖：`traning.contracts.common`

- `C L22-L121` `_SQLiteRepository` [CLASS]：集中管理连接、事务和 sqlite 错误转换。
- `M L25-L30` `_SQLiteRepository.__init__(self, database_path: Path) -> None`：初始化实例依赖、配置和运行状态。
- `M L32-L40` `_SQLiteRepository._connect(self) -> sqlite3.Connection`：执行 `connect` 对应逻辑。 调用：`RepositoryError`。
- `M L42-L48` `_SQLiteRepository._execute_write(self, sql: str, parameters: tuple[object, ...]) -> int`：执行 `execute write` 对应逻辑。 调用：`RepositoryError`, `self._connect`。
- `M L50-L73` `_SQLiteRepository._validate_schema(self, table: str, columns: tuple[str, ...]) -> None`：校验 adapter 所拥有表的版本和精确列集合。 调用：`RepositoryError`, `_required_str`, `self._connect`, `self._fetch_one`。
- `M L75-L93` `_SQLiteRepository._initialize_schema(self, table: str, create_table_sql: str) -> None`：执行 `initialize schema` 对应逻辑。 调用：`RepositoryError`, `self._connect`。
- `M L95-L108` `_SQLiteRepository._fetch_one(self, sql: str, parameters: tuple[object, ...], decoder: Callable[[tuple[object, ...]], _Record]) -> _Record | None`：执行 `fetch one` 对应逻辑。 调用：`RepositoryError`, `self._connect`。
- `M L110-L121` `_SQLiteRepository._fetch_all(self, sql: str, parameters: tuple[object, ...], decoder: Callable[[tuple[object, ...]], _Record]) -> tuple[_Record, ...]`：执行 `fetch all` 对应逻辑。 调用：`RepositoryError`, `self._connect`。
- `C L124-L229` `SQLitePreprocessingMetadataRepository(_SQLiteRepository)` [CLASS]：拥有固定 V2 schema 的预处理元数据 SQLite adapter。
- `M L138-L142` `SQLitePreprocessingMetadataRepository.__init__(self, database_path: Path, *, _initialize: bool=False) -> None`：初始化实例依赖、配置和运行状态。 调用：`self._create_schema`, `self._validate_schema`, `super.__init__`。
- `M L145-L148` `SQLitePreprocessingMetadataRepository.create(cls, database_path: Path) -> SQLitePreprocessingMetadataRepository`：初始化稳定 V2 表并返回 adapter；不迁移或猜测其他 schema。
- `M L150-L163` `SQLitePreprocessingMetadataRepository._create_schema(self) -> None`：执行 `create schema` 对应逻辑。 调用：`self._initialize_schema`。
- `M L165-L175` `SQLitePreprocessingMetadataRepository.get(self, item_name: str) -> PreprocessingMetadata | None`：按条目标识解码一份预处理元数据，缺失时返回 ``None``。 调用：`require_identifier`, `self._fetch_one`。
- `M L177-L186` `SQLitePreprocessingMetadataRepository.list_all(self) -> tuple[PreprocessingMetadata, ...]`：按条目标识稳定排序并解码全部预处理元数据。 调用：`self._fetch_all`。
- `M L188-L218` `SQLitePreprocessingMetadataRepository.save(self, metadata: PreprocessingMetadata) -> None`：在单次事务中新增或更新一份 typed 预处理元数据。 调用：`self._execute_write`。
- `M L220-L229` `SQLitePreprocessingMetadataRepository.delete(self, item_name: str) -> bool`：事务性删除指定预处理条目并返回此前是否存在。 调用：`require_identifier`, `self._execute_write`。
- `C L232-L337` `SQLiteDatasetCatalogRepository(_SQLiteRepository)` [CLASS]：拥有固定 V2 schema 的数据集目录 SQLite adapter。
- `M L246-L250` `SQLiteDatasetCatalogRepository.__init__(self, database_path: Path, *, _initialize: bool=False) -> None`：初始化实例依赖、配置和运行状态。 调用：`self._create_schema`, `self._validate_schema`, `super.__init__`。
- `M L253-L256` `SQLiteDatasetCatalogRepository.create(cls, database_path: Path) -> SQLiteDatasetCatalogRepository`：初始化稳定 V2 表并返回 adapter；不迁移或猜测其他 schema。
- `M L258-L271` `SQLiteDatasetCatalogRepository._create_schema(self) -> None`：执行 `create schema` 对应逻辑。 调用：`self._initialize_schema`。
- `M L273-L281` `SQLiteDatasetCatalogRepository.get(self, folder_name: str) -> DatasetCatalogEntry | None`：按目录名解码一份数据集条目，缺失时返回 ``None``。 调用：`_catalog_select_sql`, `require_identifier`, `self._fetch_one`。
- `M L283-L294` `SQLiteDatasetCatalogRepository.list_all(self, *, active_only: bool=False) -> tuple[DatasetCatalogEntry, ...]`：按序号和目录名列出条目，可限制为活动数据集。 调用：`_catalog_select_sql`, `self._fetch_all`。
- `M L296-L326` `SQLiteDatasetCatalogRepository.save(self, entry: DatasetCatalogEntry) -> None`：在单次事务中新增或更新一份 typed 数据集目录条目。 调用：`self._execute_write`。
- `M L328-L337` `SQLiteDatasetCatalogRepository.delete(self, folder_name: str) -> bool`：事务性删除指定数据集目录并返回此前是否存在。 调用：`require_identifier`, `self._execute_write`。
- `F L340-L344` `_catalog_select_sql() -> str`：执行 `catalog select sql` 对应逻辑。
- `F L347-L349` `_decode_schema_version(row: tuple[object, ...]) -> int`：执行 `decode schema version` 对应逻辑。 调用：`_require_row_length`, `_required_int`。
- `F L352-L356` `_require_row_length(row: tuple[object, ...], expected: int, record_name: str) -> None`：执行 `require row length` 对应逻辑。 调用：`RepositoryError`。
- `F L359-L362` `_required_str(value: object, field_name: str) -> str`：执行 `required str` 对应逻辑。 调用：`RepositoryError`。
- `F L365-L368` `_optional_str(value: object, field_name: str) -> str | None`：执行 `optional str` 对应逻辑。 调用：`_required_str`。
- `F L371-L374` `_required_int(value: object, field_name: str) -> int`：执行 `required int` 对应逻辑。 调用：`RepositoryError`。
- `F L377-L380` `_optional_int(value: object, field_name: str) -> int | None`：执行 `optional int` 对应逻辑。 调用：`_required_int`。
- `F L383-L388` `_optional_float(value: object, field_name: str) -> float | None`：执行 `optional float` 对应逻辑。 调用：`RepositoryError`。
- `F L391-L405` `_decode_preprocessing_metadata(row: tuple[object, ...]) -> PreprocessingMetadata`：执行 `decode preprocessing metadata` 对应逻辑。 调用：`PreprocessingMetadata`, `RepositoryError`, `_require_row_length`, `_required_int`, `_required_str`。
- `F L408-L425` `_decode_catalog_entry(row: tuple[object, ...]) -> DatasetCatalogEntry`：执行 `decode catalog entry` 对应逻辑。 调用：`DatasetCatalogEntry`, `RepositoryError`, `_optional_float`, `_optional_int`, `_optional_str`, `_require_row_length`。

## `src/traning/data/segments.py`

职责：从 canonical split manifest 构建 typed 帧、belief 和 outcome 训练数据集与 DataLoader。
工程依赖：`package`, `package.dataset_split`, `traning.config`, `traning.contracts`, `traning.data.coordinates`, `traning.lib.data.annotation`, `traning.lib.data.discovery`, `traning.lib.data.models`, `traning.lib.data.sampling`, `traning.lib.data.video_reader`

- `C L53-L224` `SegmentTrainingDataset(Sequence[TrainingSample])` [CLASS]：一个具体 split 的随机访问数据集；构造时不解码任何图像。
- `M L56-L83` `SegmentTrainingDataset.__init__(self, records: tuple[SegmentRecord, ...], *, split: DataSplit, sample_fps: float, frame_step: int, max_frames_per_segment: int | None, visibility_post_ms: float, coordinate_transform: FrameCoordinateTransform | None) -> None`：初始化实例依赖、配置和运行状态。 调用：`build_frame_references`, `self._build_sequence_indices`。
- `M L85-L98` `SegmentTrainingDataset._build_sequence_indices(self) -> tuple[tuple[str, tuple[int, ...]], ...]`：直接按 FrameReference 的 record_index 建立序列索引，不解析 sample_id。
- `M L101-L104` `SegmentTrainingDataset.sequence_ids(self) -> tuple[str, ...]` [PROPERTY]：按发现顺序返回稳定 segment 身份。
- `M L107-L112` `SegmentTrainingDataset.transform_fingerprint(self) -> str | None` [PROPERTY]：返回所有样本共享的坐标变换指纹；阻断配置下为 ``None``。
- `M L114-L115` `SegmentTrainingDataset.__len__(self) -> int`：执行 `len` 对应逻辑。
- `M L118` `SegmentTrainingDataset.__getitem__(self, index: int) -> TrainingSample`：执行 `getitem` 对应逻辑。
- `M L121` `SegmentTrainingDataset.__getitem__(self, index: slice) -> tuple[TrainingSample, ...]`：执行 `getitem` 对应逻辑。
- `M L123-L176` `SegmentTrainingDataset.__getitem__(self, index: int | slice) -> TrainingSample | tuple[TrainingSample, ...]`：执行 `getitem` 对应逻辑。 调用：`TrainingSample`, `_ground_truth_object`, `_sample_id`, `self._video_reader`, `self._video_reader.read_frame_at`, `visible_hit_objects`。
- `M L178-L183` `SegmentTrainingDataset._video_reader(self) -> VideoReader`：为当前进程惰性创建有限句柄的 LRU 视频读取器。 调用：`VideoReader`。
- `M L185-L196` `SegmentTrainingDataset.sequence(self, sequence_id: str) -> TrainingSequenceDataset`：返回一个不复制样本的 typed 因果序列视图。 调用：`TrainingSequenceDataset`。
- `M L198-L207` `SegmentTrainingDataset.iter_sequences(self) -> Iterator[TrainingSequenceDataset]`：按稳定发现顺序惰性遍历 segment 序列。 调用：`TrainingSequenceDataset`。
- `M L209-L214` `SegmentTrainingDataset.close(self) -> None`：立即释放当前进程持有的视频句柄。 调用：`self._reader.close`。
- `M L216-L221` `SegmentTrainingDataset.__getstate__(self) -> dict[str, object]`：DataLoader worker 序列化时不传递 OpenCV 句柄。
- `M L223-L224` `SegmentTrainingDataset.__del__(self) -> None`：执行 `del` 对应逻辑。 调用：`self.close`。
- `C L228-L259` `TrainingSequenceDataset(Sequence[TrainingSample])` [CLASS]：一个 segment 的惰性 typed 因果序列视图。
- `M L236-L242` `TrainingSequenceDataset.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.sequence_id.strip`。
- `M L244-L245` `TrainingSequenceDataset.__len__(self) -> int`：执行 `len` 对应逻辑。
- `M L248` `TrainingSequenceDataset.__getitem__(self, index: int) -> TrainingSample`：执行 `getitem` 对应逻辑。
- `M L251` `TrainingSequenceDataset.__getitem__(self, index: slice) -> tuple[TrainingSample, ...]`：执行 `getitem` 对应逻辑。
- `M L253-L259` `TrainingSequenceDataset.__getitem__(self, index: int | slice) -> TrainingSample | tuple[TrainingSample, ...]`：执行 `getitem` 对应逻辑。
- `C L262-L297` `CombinedTrainingDataset(Sequence[TrainingSample])` [CLASS]：三个具体 split 的只读拼接视图，用于 ``DataSplit.ALL``。
- `M L265-L273` `CombinedTrainingDataset.__init__(self, datasets: tuple[SegmentTrainingDataset, ...]) -> None`：初始化实例依赖、配置和运行状态。
- `M L275-L276` `CombinedTrainingDataset.__len__(self) -> int`：执行 `len` 对应逻辑。
- `M L279` `CombinedTrainingDataset.__getitem__(self, index: int) -> TrainingSample`：执行 `getitem` 对应逻辑。
- `M L282` `CombinedTrainingDataset.__getitem__(self, index: slice) -> tuple[TrainingSample, ...]`：执行 `getitem` 对应逻辑。
- `M L284-L291` `CombinedTrainingDataset.__getitem__(self, index: int | slice) -> TrainingSample | tuple[TrainingSample, ...]`：执行 `getitem` 对应逻辑。
- `M L293-L297` `CombinedTrainingDataset.iter_sequences(self) -> Iterator[TrainingSequenceDataset]`：依次遍历 train、validation、test 内的全部因果序列。 调用：`dataset.iter_sequences`。
- `C L301-L383` `TrainingDatasetBundle` [CLASS]：生产数据入口一次返回的数据集、质量、身份与坐标绑定。
- `M L314-L340` `TrainingDatasetBundle.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.dataset_identity.startswith`。
- `M L342-L353` `TrainingDatasetBundle.dataset(self, split: DataSplit) -> SegmentTrainingDataset | CombinedTrainingDataset`：按唯一 DataSplit 返回 typed 数据集；ALL 返回惰性拼接视图。
- `M L356-L359` `TrainingDatasetBundle.train(self) -> SegmentTrainingDataset` [PROPERTY]：返回训练 split。 调用：`self._concrete_dataset`。
- `M L362-L365` `TrainingDatasetBundle.validation(self) -> SegmentTrainingDataset` [PROPERTY]：返回验证 split。 调用：`self._concrete_dataset`。
- `M L368-L371` `TrainingDatasetBundle.test(self) -> SegmentTrainingDataset` [PROPERTY]：返回测试 split。 调用：`self._concrete_dataset`。
- `M L374-L377` `TrainingDatasetBundle.all(self) -> CombinedTrainingDataset` [PROPERTY]：返回固定 split 顺序的惰性拼接视图。 调用：`CombinedTrainingDataset`。
- `M L379-L383` `TrainingDatasetBundle._concrete_dataset(self, split: DataSplit) -> SegmentTrainingDataset`：执行 `concrete dataset` 对应逻辑。
- `F L386-L449` `build_training_datasets(config: V2Config) -> TrainingDatasetBundle`：发现生产 segment 并返回 typed bundle；固定阻断写入 report 而非抛出。 调用：`DataQualityReport`, `SegmentTrainingDataset`, `TrainingDatasetBundle`, `_append_split_quality`, `_assign_records`, `_coordinate_transform`。
- `F L452-L481` `_coordinate_transform(config: V2Config, issues: list[DataQualityIssue]) -> FrameCoordinateTransform | None`：执行 `coordinate transform` 对应逻辑。 调用：`FrameCoordinateTransform`, `_issue`。
- `F L484-L543` `_split_manifest(config: V2Config, issues: list[DataQualityIssue]) -> DatasetSplitManifest | None`：执行 `split manifest` 对应逻辑。 调用：`_issue`。
- `F L546-L603` `_assign_records(records: tuple[SegmentRecord, ...], manifest: DatasetSplitManifest, grouped: dict[DataSplit, list[SegmentRecord]], issues: list[DataQualityIssue]) -> None`：只依据冻结 item 归属分组，禁止同一谱面按 segment 再切分。 调用：`_enum_value`, `_issue`, `manifest.items.get`。
- `F L606-L625` `_validate_records(selected: dict[DataSplit, tuple[SegmentRecord, ...]], config: V2Config, issues: list[DataQualityIssue]) -> None`：校验 `records` 对应的数据或结果。 调用：`_issue`, `_validate_annotation`, `_validate_video_header`。
- `F L628-L656` `_validate_annotation(record: SegmentRecord, issues: list[DataQualityIssue]) -> None`：校验 `annotation` 对应的数据或结果。 调用：`_ground_truth_object`, `_issue`。
- `F L659-L708` `_validate_video_header(record: SegmentRecord, config: V2Config, issues: list[DataQualityIssue]) -> None`：校验 `video header` 对应的数据或结果。 调用：`_issue`, `capture.get`。
- `F L711-L734` `_append_split_quality(datasets: tuple[tuple[DataSplit, SegmentTrainingDataset], ...], issues: list[DataQualityIssue]) -> None`：执行 `append split quality` 对应逻辑。 调用：`_issue`。
- `F L737-L780` `_ground_truth_object(record: SegmentRecord, item: HitObjectAnnotation, object_index: int) -> GroundTruthObject`：把 permissive 标注模型收口成严格 GroundTruthObject。 调用：`GroundTruthObject`, `Point2D`, `_object_id`。
- `F L783-L791` `_object_id(record: SegmentRecord, item: HitObjectAnnotation, object_index: int) -> str`：执行 `object id` 对应逻辑。
- `F L794-L795` `_sample_id(record: SegmentRecord, frame_index: int) -> str`：执行 `sample id` 对应逻辑。
- `F L798-L838` `_dataset_identity(config: V2Config, selected: dict[DataSplit, tuple[SegmentRecord, ...]], transform: FrameCoordinateTransform | None) -> str`：对实际消费的清单、标注和视频内容生成稳定 SHA-256 身份。 调用：`_file_digest`, `_relative_path`, `hasher.update`, `hashlib.sha256`。
- `F L841-L850` `_file_digest(path: Path) -> str` [IO-R IO-W]：执行 `file digest` 对应逻辑。 调用：`hasher.update`, `hashlib.sha256`, `stream.read`。
- `F L853-L857` `_relative_path(path: Path, root: Path) -> str`：执行 `relative path` 对应逻辑。
- `F L860-L876` `_issue(code: str, message: str, *, blocks_training: bool, severity: DataQualitySeverity=DataQualitySeverity.ERROR, sample_id: str | None=None, details: tuple[tuple[str, str | int | float | bool | None], ...]=()) -> DataQualityIssue`：执行 `issue` 对应逻辑。 调用：`DataQualityIssue`。
- `F L879-L880` `_issue_sort_key(issue: DataQualityIssue) -> tuple[str, str, str]`：执行 `issue sort key` 对应逻辑。
- `F L883-L886` `_enum_value(value: object) -> str`：执行 `enum value` 对应逻辑。

## `src/traning/decision/planner.py`

职责：只读取 BeliefState 与 OutcomeDistribution，比较 CLICK 和 WAIT 的风险调整效用。
工程依赖：`traning.config`, `traning.contracts`, `traning.decision.utility`

- `C L17-L161` `OptimalStoppingPlanner` [CLASS]：比较当前 CLICK 与最小正 horizon 的 WAIT 价值。
- `M L24-L33` `OptimalStoppingPlanner.__init__(self, config: DecisionConfig) -> None`：初始化实例依赖、配置和运行状态。
- `M L35-L90` `OptimalStoppingPlanner.plan(self, beliefs: tuple[BeliefState, ...], outcomes: tuple[OutcomeDistribution, ...], timestamp_ms: float) -> DecisionResult`：在当前点击与等待一个最短正 horizon 之间作稳定选择。 调用：`DecisionResult`, `self._utilities_for_horizon`, `self._validate_inputs`, `self._wait_result`。
- `M L92-L135` `OptimalStoppingPlanner._validate_inputs(self, beliefs: tuple[BeliefState, ...], outcomes: tuple[OutcomeDistribution, ...], timestamp_ms: float) -> None`：校验 `inputs` 对应的数据或结果。
- `M L137-L146` `OptimalStoppingPlanner._utilities_for_horizon(self, belief_by_track: dict[str, BeliefState], outcome_by_key: dict[tuple[str, float], OutcomeDistribution], horizon_ms: float) -> tuple[ClickUtility, ...]`：执行 `utilities for horizon` 对应逻辑。 调用：`compute_click_utility`。
- `M L148-L161` `OptimalStoppingPlanner._wait_result(self, timestamp_ms: float, wait_utility: float, confidence: float) -> DecisionResult`：执行 `wait result` 对应逻辑。 调用：`DecisionResult`。

## `src/traning/decision/utility.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.config`, `traning.contracts`

- `C L13-L44` `ClickUtility` [CLASS]：绑定原始 OutcomeDistribution 的单轨迹单 horizon 点击价值。
- `M L22-L44` `ClickUtility.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_finite`。
- `F L47-L74` `compute_click_utility(outcome: OutcomeDistribution, config: DecisionConfig) -> ClickUtility`：按唯一风险惩罚公式计算 CLICK utility，不读取图像或动作 logits。 调用：`ClickUtility`, `_require_finite`。
- `F L77-L81` `_require_finite(value: float, field_name: str) -> None`：执行 `require finite` 对应逻辑。

## `src/traning/evaluation/attribution.py`

职责：用统一规则归因未通过样本，避免准确命中被误分到错误模块。
工程依赖：`traning.contracts.common`

- `C L17-L23` `PrimaryError(str, Enum)` [CLASS]：事件的唯一主要错误域。
- `C L26-L38` `EvaluationTag(str, Enum)` [CLASS]：sequence scorer 的 canonical 标签及未解析目标标签。
- `C L41-L45` `EvaluationCoordinateSpace(str, Enum)` [CLASS]：事件中 click_x/click_y 所属的显式坐标空间。
- `F L51-L55` `_require_identifier(name: str, value: str) -> None`：执行 `require identifier` 对应逻辑。
- `C L59-L182` `SequenceEvaluationEvent` [CLASS]：Phase 9/10 直接消费的单一 typed 归因事件。
- `M L82-L182` `SequenceEvaluationEvent.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_identifier`, `require_transform_fingerprint`。
- `F L185-L193` `_canonical_event_id(parts: tuple[str, ...]) -> str`：对 UTF-8 字节做 length-prefix 编码，避免字段连接歧义。 调用：`hashlib.sha256`。
- `F L196-L306` `build_sequence_evaluation_events(sample_id: str, frame_index: int, result: SequenceScore | FrameSequenceScore) -> tuple[SequenceEvaluationEvent, ...]`：确定性投影 click 评分，并为每个未解析目标追加唯一事件。 调用：`EvaluationTag`, `PrimaryError`, `SequenceEvaluationEvent`, `_canonical_event_id`, `_require_identifier`。

## `src/traning/evaluation/metrics.py`

职责：Python 模块；具体职责见下方符号及调用。

- `F L12-L17` `multiclass_nll(probabilities: Tensor, labels: Tensor) -> Tensor`：返回 batch mean NLL；零目标概率按 dtype 最小正数取有限对数。 调用：`_validate_multiclass`。
- `F L20-L27` `multiclass_brier_score(probabilities: Tensor, labels: Tensor) -> Tensor`：返回各样本所有类别平方误差之和的 batch mean。 调用：`_validate_multiclass`。
- `F L30-L63` `top_label_ece(probabilities: Tensor, labels: Tensor, *, bin_count: int=15) -> Tensor`：计算 top-label ECE。 调用：`_validate_multiclass`。
- `F L66-L76` `expected_score_mae(predicted_scores: Tensor, target_scores: Tensor) -> Tensor`：返回逐样本 expected score 与归一化目标分数的平均绝对误差。 调用：`_validate_binary_pair`。
- `F L79-L96` `expiry_binary_cross_entropy(expiry_probabilities: Tensor, expiry_targets: Tensor) -> Tensor`：返回过期概率的 batch mean binary cross entropy。 调用：`_validate_binary_pair`。
- `F L99-L109` `expiry_brier_score(expiry_probabilities: Tensor, expiry_targets: Tensor) -> Tensor`：返回过期概率的 batch mean Brier score。 调用：`_validate_binary_pair`。
- `F L112-L139` `_validate_multiclass(probabilities: Tensor, labels: Tensor) -> None`：校验 `multiclass` 对应的数据或结果。 调用：`_require_finite`, `_require_tensor`。
- `F L142-L173` `_validate_binary_pair(predictions: Tensor, targets: Tensor, *, prediction_name: str, target_name: str, binary_targets: bool) -> None`：校验 `binary pair` 对应的数据或结果。 调用：`_require_finite`, `_require_tensor`。
- `F L176-L178` `_require_tensor(value: Tensor, name: str) -> None`：执行 `require tensor` 对应逻辑。
- `F L181-L183` `_require_finite(value: Tensor, name: str) -> None`：执行 `require finite` 对应逻辑。

## `src/traning/evaluation/scoring.py`

职责：提供 canonical point 与 slider 空间时间评分。

- `F L15-L21` `_require_number(name: str, value: float, *, positive: bool=False) -> None`：执行 `require number` 对应逻辑。
- `F L24-L28` `_require_point(name: str, point: Point) -> None`：执行 `require point` 对应逻辑。 调用：`_require_number`。
- `F L31-L37` `_require_path(name: str, path: PathPoints, *, allow_empty: bool) -> None`：执行 `require path` 对应逻辑。 调用：`_require_point`。
- `C L41-L106` `ScoreSpec` [CLASS]：连续评分阈值；空间量以 circle radius 为单位，时间量为毫秒。
- `M L59-L93` `ScoreSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_number`。
- `M L96-L99` `ScoreSpec.maximum_coefficient(self) -> float` [PROPERTY]：返回包含空间 bonus 后单个系数的理论上限。
- `M L102-L106` `ScoreSpec.maximum_raw_score(self) -> float` [PROPERTY]：返回空间、时间及交互项组合后的理论原始分上限。
- `C L110-L116` `CombinedScore` [CLASS]：组合后的空间、时间、原始和归一化分数。
- `C L120-L127` `PointScore` [CLASS]：单点连续评分。
- `C L131-L140` `PathScore` [CLASS]：slider 路径的双向走廊评分。
- `C L144-L150` `SliderScore` [CLASS]：slider head 与 path 的联合评分。
- `F L153-L155` `_require_spec(spec: ScoreSpec) -> None`：执行 `require spec` 对应逻辑。
- `F L158-L162` `_interpolate(value: float, start: float, end: float, start_score: float, end_score: float) -> float`：执行 `interpolate` 对应逻辑。
- `F L165-L185` `spatial_coefficient(distance_ratio: float, *, spec: ScoreSpec=ScoreSpec()) -> float`：把非负距离半径比映射为连续空间系数。 调用：`_require_number`, `_require_spec`。
- `F L188-L224` `temporal_coefficient(time_error_ms: float, *, spec: ScoreSpec=ScoreSpec()) -> float`：按绝对时间误差分段插值为连续时间系数。 调用：`_interpolate`, `_require_number`, `_require_spec`。
- `F L227-L238` `combine_coefficients(spatial: float, temporal: float, *, spec: ScoreSpec=ScoreSpec()) -> CombinedScore`：组合空间与时间系数，并按理论最大值归一化。 调用：`CombinedScore`, `_require_number`, `_require_spec`。
- `F L241-L275` `score_point(reference_xy: Point, predicted_xy: Point, *, circle_radius: float, reference_time_ms: float, predicted_time_ms: float, spec: ScoreSpec=ScoreSpec()) -> PointScore`：在同一坐标空间内评分点位置和毫秒级打击时间。 调用：`PointScore`, `_require_number`, `_require_point`, `_require_spec`, `combine_coefficients`, `spatial_coefficient`。
- `F L278-L289` `_point_to_segment_distance(point: Point, start: Point, end: Point) -> float`：执行 `point to segment distance` 对应逻辑。
- `F L292-L298` `_minimum_distance(point: Point, path: PathPoints) -> float`：执行 `minimum distance` 对应逻辑。 调用：`_point_to_segment_distance`。
- `F L301-L315` `_densify_path(path: PathPoints, *, maximum_step: float) -> PathPoints`：执行 `densify path` 对应逻辑。
- `F L318-L326` `_directed_path_statistics(source: PathPoints, target: PathPoints, *, distance_limit: float) -> tuple[float, float]`：执行 `directed path statistics` 对应逻辑。 调用：`_minimum_distance`。
- `F L329-L376` `score_slider_path(reference_path: PathPoints, predicted_path: PathPoints, *, circle_radius: float, spec: ScoreSpec=ScoreSpec()) -> PathScore`：用双向稠密采样评估 slider 中心线膨胀走廊。 调用：`PathScore`, `_densify_path`, `_directed_path_statistics`, `_require_number`, `_require_path`, `_require_spec`。
- `F L379-L421` `score_slider(reference_head_xy: Point | None, predicted_head_xy: Point | None, reference_path: PathPoints, predicted_path: PathPoints, *, circle_radius: float, reference_start_ms: float, predicted_start_ms: float, spec: ScoreSpec=ScoreSpec()) -> SliderScore`：联合评分 slider head、路径与开始时间。 调用：`SliderScore`, `_require_path`, `_require_point`, `_require_spec`, `combine_coefficients`, `score_point`。

## `src/traning/evaluation/sequence.py`

职责：模拟点击序列、频率限制和物件消费，并输出稳定评分结果。
工程依赖：`traning.data.coordinates`

- `F L41-L45` `_finite(name: str, value: float) -> None`：执行 `finite` 对应逻辑。
- `F L48-L55` `_valid_path(name: str, path: PathPoints) -> None`：执行 `valid path` 对应逻辑。 调用：`_finite`。
- `C L59-L70` `SequenceScoreSpec` [CLASS]：序列级点击频率限制与单物件评分规格。
- `M L65-L70` `SequenceScoreSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_finite`。
- `C L74-L110` `TargetObject` [CLASS]：序列 oracle 的 canonical circle/slider 目标。
- `M L86-L110` `TargetObject.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_finite`, `_valid_path`, `self.target_id.strip`。
- `C L114-L126` `PredictedClick` [CLASS]：按时间发布的 canonical osu! 预测点击与可选 slider 路径。
- `M L122-L126` `PredictedClick.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_finite`, `_valid_path`。
- `C L130-L155` `FramePredictedClick` [CLASS]：正式 evaluation 边界接收的原帧像素点击及可选 slider 路径。
- `M L137-L155` `FramePredictedClick.__post_init__(self) -> None`：确保路径点不能逃离帧尺寸与坐标指纹领域对象。 调用：`_finite`。
- `C L159-L166` `TargetResolution` [CLASS]：目标首次不可逆命中的解析记录。
- `C L170-L188` `ClickEvaluation` [CLASS]：单次点击的状态、评分和错误归因。
- `M L185-L188` `ClickEvaluation.frequency_limited(self) -> bool` [PROPERTY]：说明本次点击是否仅因频率限制而未参与匹配。
- `C L192-L215` `SequenceScore` [CLASS]：完整点击序列的评分与未解析目标。
- `M L200-L203` `SequenceScore.hit_count(self) -> int` [PROPERTY]：返回已被首次不可逆解析的目标数量。
- `M L206-L209` `SequenceScore.miss_count(self) -> int` [PROPERTY]：返回 canonical 状态为普通 miss 的点击数量。
- `M L212-L215` `SequenceScore.frequency_limited_count(self) -> int` [PROPERTY]：返回因点击频率限制而跳过的点击数量。
- `C L219-L296` `FrameSequenceScore` [CLASS]：保留原帧点击、尺寸和变换指纹的 sequence score 信封。
- `M L228-L260` `FrameSequenceScore.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.transform_fingerprint.startswith`。
- `M L263-L266` `FrameSequenceScore.clicks(self) -> tuple[ClickEvaluation, ...]` [PROPERTY]：返回 canonical 单击评分，兼容只读统计消费方。
- `M L269-L272` `FrameSequenceScore.resolved_targets(self) -> tuple[TargetResolution, ...]` [PROPERTY]：返回 canonical 已解析目标。
- `M L275-L278` `FrameSequenceScore.unresolved_target_ids(self) -> tuple[str, ...]` [PROPERTY]：返回 canonical 未解析目标。
- `M L281-L284` `FrameSequenceScore.hit_count(self) -> int` [PROPERTY]：返回 canonical 命中数。
- `M L287-L290` `FrameSequenceScore.miss_count(self) -> int` [PROPERTY]：返回 canonical miss 数。
- `M L293-L296` `FrameSequenceScore.frequency_limited_count(self) -> int` [PROPERTY]：返回被频率限制的点击数。
- `F L299-L305` `_target_sort_key(target: TargetObject) -> tuple[float, int, str]`：执行 `target sort key` 对应逻辑。
- `F L308-L334` `_score_target(target: TargetObject, click: PredictedClick, *, circle_radius: float, spec: ScoreSpec) -> PointScore | SliderScore`：执行 `score target` 对应逻辑。 调用：`score_point`, `score_slider`。
- `F L337-L338` `_score_value(score: PointScore | SliderScore) -> float`：执行 `score value` 对应逻辑。
- `F L341-L344` `_spatial_passed(score: PointScore | SliderScore, spec: ScoreSpec) -> bool`：执行 `spatial passed` 对应逻辑。
- `F L347-L349` `_temporal_passed(score: PointScore | SliderScore, spec: ScoreSpec) -> bool`：执行 `temporal passed` 对应逻辑。
- `F L352-L353` `_spatial_error(score: PointScore | SliderScore) -> float`：执行 `spatial error` 对应逻辑。
- `F L356-L357` `_temporal_error_ms(target: TargetObject, click: PredictedClick) -> float`：执行 `temporal error ms` 对应逻辑。
- `F L360-L364` `_spatial_excess(score: PointScore | SliderScore, spec: ScoreSpec) -> float`：执行 `spatial excess` 对应逻辑。
- `F L367-L371` `_temporal_excess(score: PointScore | SliderScore, spec: ScoreSpec) -> float`：执行 `temporal excess` 对应逻辑。
- `F L374-L409` `_error_attribution(target: TargetObject, click: PredictedClick, score: PointScore | SliderScore, *, spec: ScoreSpec) -> tuple[ErrorDomain, tuple[ErrorTag, ...], float, float]`：执行 `error attribution` 对应逻辑。 调用：`_spatial_error`, `_spatial_excess`, `_spatial_passed`, `_temporal_error_ms`, `_temporal_excess`, `_temporal_passed`。
- `F L412-L425` `_best_scored_target(targets: tuple[TargetObject, ...], click: PredictedClick, *, circle_radius: float, spec: ScoreSpec) -> tuple[TargetObject, PointScore | SliderScore] | None`：执行 `best scored target` 对应逻辑。 调用：`_score_target`, `_score_value`。
- `F L428-L572` `score_click_sequence(targets: tuple[TargetObject, ...], clicks: tuple[PredictedClick, ...], *, circle_radius: float, spec: SequenceScoreSpec=SequenceScoreSpec()) -> SequenceScore`：稳定排序点击，每个目标最多解析一次，并保留完整错误归因。 调用：`ClickEvaluation`, `SequenceScore`, `TargetResolution`, `_best_scored_target`, `_error_attribution`, `_finite`。
- `F L575-L631` `score_frame_click_sequence(targets: tuple[TargetObject, ...], clicks: tuple[FramePredictedClick, ...], *, coordinate_transform: FrameCoordinateTransform, circle_radius: float, spec: SequenceScoreSpec=SequenceScoreSpec()) -> FrameSequenceScore`：先用共享指纹逆变换原帧点击，再委托唯一 canonical sequence scorer。 调用：`FrameSequenceScore`, `OsuPoint`, `PredictedClick`, `coordinate_transform.prediction_to_canonical_scoring`, `score_click_sequence`。

## `src/traning/infrastructure/determinism.py`

职责：Python 模块；具体职责见下方符号及调用。

- `F L11-L23` `seed_everything(seed: int) -> None`：设置所有已使用随机源；仅在 CUDA 可用时触碰 CUDA API。

## `src/traning/infrastructure/errors.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L6-L7` `InfrastructureError(RuntimeError)` [CLASS]：OSU V2 基础设施错误的共同基类。
- `C L10-L11` `SchemaMismatchError(InfrastructureError)` [CLASS]：持久化数据的结构与调用方要求的 schema 不一致。
- `C L14-L15` `IntegrityError(InfrastructureError)` [CLASS]：持久化数据不完整、损坏或不满足严格格式要求。
- `C L18-L19` `AtomicWriteError(InfrastructureError)` [CLASS]：原子写入未能完整发布并持久化。

## `src/traning/infrastructure/persistence.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts.common`

- `F L19-L30` `atomic_write_bytes(path: Path, payload: bytes) -> None`：将字节完整落盘后，以同目录原子替换发布。 调用：`_atomic_publish`。
- `N L25-L28` `atomic_write_bytes.write_payload(handle: BinaryIO) -> None` [IO-W]：把已校验字节写入统一原子发布器提供的临时文件。 调用：`handle.write`。
- `F L33-L42` `atomic_write_text(path: Path, text: str, *, encoding: str='utf-8') -> None`：按指定编码原子发布文本。 调用：`AtomicWriteError`, `atomic_write_bytes`。
- `F L45-L49` `atomic_write_json(path: Path, payload: JSONValue) -> None`：以 canonical 紧凑格式原子发布单个 JSON 值。 调用：`_encode_json`, `atomic_write_bytes`。
- `F L52-L68` `atomic_write_jsonl(path: Path, records: Iterable[JSONObject]) -> None`：全量原子发布 JSONL；任何一条失败都不会暴露半成品。 调用：`_atomic_publish`。
- `N L55-L66` `atomic_write_jsonl.write_records(handle: BinaryIO) -> None` [IO-W]：逐行编码严格 JSON object，任一失败即放弃整次发布。 调用：`SchemaMismatchError`, `_encode_json`, `handle.write`。
- `F L71-L83` `sha256_file(path: Path, *, chunk_size: int=1024 * 1024) -> str` [IO-R IO-W]：流式计算文件的十六进制 SHA-256，不把大文件整体读入内存。 调用：`IntegrityError`, `digest.update`, `handle.read`, `hashlib.sha256`。
- `F L86-L102` `read_json_object(path: Path) -> JSONObject` [IO-R IO-W]：读取严格 JSON object，拒绝损坏文本、重复键和非有限浮点数。 调用：`IntegrityError`, `SchemaMismatchError`, `_validate_json_value`, `json.load`。
- `F L105-L138` `_atomic_publish(path: Path, writer: Callable[[BinaryIO], None]) -> None` [IO-W]：执行 write→flush→fsync→replace→目录 fsync 的统一发布协议。 调用：`AtomicWriteError`, `_fsync_directory`。
- `F L141-L149` `_fsync_directory(directory: Path) -> None` [IO-W]：同步目录项，保证 replace 的命名变更也进入持久化边界。 调用：`os.close`。
- `F L152-L169` `_encode_json(payload: JSONValue, *, path: Path, context: str='JSON payload') -> bytes`：执行 `encode json` 对应逻辑。 调用：`AtomicWriteError`, `_validate_json_value`。
- `F L172-L189` `_validate_json_value(value: object, *, context: str) -> None`：校验 `json value` 对应的数据或结果。 调用：`SchemaMismatchError`, `_validate_json_value`。
- `F L192-L200` `_object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]`：执行 `object without duplicate keys` 对应逻辑。 调用：`IntegrityError`。
- `F L203-L204` `_reject_non_finite_constant(value: str) -> object`：执行 `reject non finite constant` 对应逻辑。 调用：`IntegrityError`。

## `src/traning/lib/data/annotation.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L11-L33` `HitObjectAnnotation(BaseModel)` [CLASS]：单个 osu! 物件在 segment 相对时间轴上的标注。
- `M L29-L33` `HitObjectAnnotation._valid_end(cls, value: int, info: ValidationInfo) -> int` [VALIDATOR]：执行 `valid end` 对应逻辑。 调用：`info.data.get`。
- `C L36-L42` `DifficultyAnnotation(BaseModel)` [CLASS]：生成监督目标所需的难度派生参数。
- `C L45-L61` `SourceAnnotation(BaseModel)` [CLASS]：segment 对应谱面与原始裁剪区间的来源信息。
- `M L57-L61` `SourceAnnotation._valid_clip_end(cls, value: int, info: ValidationInfo) -> int` [VALIDATOR]：执行 `valid clip end` 对应逻辑。 调用：`info.data.get`。
- `C L64-L81` `SegmentAnnotation(BaseModel)` [CLASS]：一个视频 segment 的版本化完整训练标注。
- `M L78-L81` `SegmentAnnotation.duration_ms(self) -> int` [PROPERTY]：返回 segment 在源时间轴上的持续毫秒数。
- `F L84-L91` `load_annotation(path: Path) -> SegmentAnnotation` [IO-R]：从 JSON 文件读取并严格校验一个 segment 标注。
- `F L94-L110` `visible_hit_objects(annotation: SegmentAnnotation, timestamp_ms: float, *, visibility_post_ms: float) -> tuple[HitObjectAnnotation, ...]`：返回当前帧应可见的物件，时间均为 segment 内相对毫秒。

## `src/traning/lib/data/color_cues.py`

职责：Python 模块；具体职责见下方符号及调用。

- `F L25-L32` `color_cue_channel_count(mode: ColorCueMode) -> int`：返回指定颜色提示模式追加的确定性通道数量。
- `F L35-L41` `append_color_cues(frame: torch.Tensor, *, mode: ColorCueMode) -> torch.Tensor`：在归一化 CHW RGB 帧后追加确定性的 osu! 视觉提示通道。 调用：`extract_osu_basic_color_cues`。
- `F L44-L62` `extract_osu_basic_color_cues(frame: torch.Tensor) -> torch.Tensor`：返回 ``3xHxW`` 的配色、白色字形和物件边缘响应。 调用：`_object_edge_response`, `_palette_response`, `_white_glyph_response`。
- `F L65-L84` `_palette_response(rgb: torch.Tensor, *, saturation: torch.Tensor, value: torch.Tensor) -> torch.Tensor`：执行 `palette response` 对应逻辑。
- `F L87-L94` `_white_glyph_response(*, saturation: torch.Tensor, value: torch.Tensor) -> torch.Tensor`：执行 `white glyph response` 对应逻辑。
- `F L97-L122` `_object_edge_response(rgb: torch.Tensor, *, object_prior: torch.Tensor) -> torch.Tensor`：执行 `object edge response` 对应逻辑。

## `src/traning/lib/data/coordinates.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data.patch_stream`

- `F L8-L11` `local_to_global(meta: PatchMeta, x: float, y: float) -> tuple[float, float]`：把 patch 局部图像像素平移到完整帧像素。
- `F L14-L17` `global_to_local(meta: PatchMeta, x: float, y: float) -> tuple[float, float]`：把完整帧像素平移到 patch 局部图像像素。
- `F L20-L32` `global_to_patch_indices(metas: tuple[PatchMeta, ...], x: float, y: float) -> tuple[int, ...]`：返回有效图像区包含该完整帧点的所有 patch 索引。
- `F L35-L45` `image_to_feature_grid(x: float, y: float, *, stride: int) -> tuple[float, float]`：按 stride 把图像像素映射到连续特征网格坐标。
- `F L48-L58` `feature_grid_to_image(gx: float, gy: float, *, stride: int) -> tuple[float, float]`：按 stride 把连续特征网格坐标还原为图像像素。

## `src/traning/lib/data/discovery.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data.annotation`, `traning.lib.data.models`, `traning.lib.data.preprocessing_metadata`

- `F L12-L80` `discover_segments(dataset_root: Path, *, dimensions: tuple[str, ...]=(), categories: tuple[str, ...]=(), include_items: tuple[str, ...]=(), exclude_items: tuple[str, ...]=(), max_segments: int | None=None) -> DiscoveryResult`：按稳定顺序发现过滤条件内的视频与标注配对记录。 调用：`DatasetIssue`, `DiscoveryResult`, `SegmentRecord`, `load_annotation`, `load_preprocessing_metadata`。

## `src/traning/lib/data/models.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts.common`, `traning.lib.data.annotation`

- `C L13-L24` `SegmentRecord` [CLASS]：一个已配对且通过标注解析的 segment 数据记录。
- `C L28-L32` `DatasetIssue` [CLASS]：数据发现时可定位但不隐式吞掉的文件问题。
- `C L36-L40` `DiscoveryResult` [CLASS]：数据发现得到的有效记录与全部非致命问题。
- `C L44-L49` `FrameReference` [CLASS]：稳定指向某个 segment 中一个采样帧的轻量引用。

## `src/traning/lib/data/patch_stream.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data.tiling`

- `C L15-L56` `PatchMeta` [CLASS]：一个 CHW patch 在完整帧中的像素位置。
- `M L35-L38` `PatchMeta.width(self) -> int` [PROPERTY]：返回当前 patch 在原始帧内的有效宽度。
- `M L41-L44` `PatchMeta.height(self) -> int` [PROPERTY]：返回当前 patch 在原始帧内的有效高度。
- `M L47-L50` `PatchMeta.padded_width(self) -> int` [PROPERTY]：返回模型实际接收的填充后 patch 宽度。
- `M L53-L56` `PatchMeta.padded_height(self) -> int` [PROPERTY]：返回模型实际接收的填充后 patch 高度。
- `C L59-L187` `PatchStream` [CLASS]：在 CPU 上生成固定尺寸 CHW patch，不耦合任何模型执行。
- `M L62-L83` `PatchStream.__init__(self, *, patch_width: int=512, patch_height: int=512, overlap_x: int=128, overlap_y: int=128, pin_memory: bool=False, padding_value: float=0.0) -> None`：初始化实例依赖、配置和运行状态。
- `M L85-L115` `PatchStream.metas(self, *, frame_width: int, frame_height: int) -> tuple[PatchMeta, ...]`：返回按行优先排列、完整覆盖全帧的确定性 patch 元数据。 调用：`PatchMeta`, `build_patch_windows`, `self._validate_coverage`。
- `M L117-L121` `PatchStream.count(self, frame: torch.Tensor) -> int`：返回 ``iter_patches`` 将为完整帧产生的 patch 数量。 调用：`self._shape`, `self.metas`。
- `M L123-L153` `PatchStream.iter_patches(self, frame: torch.Tensor) -> Iterator[tuple[torch.Tensor, PatchMeta]]`：从 CHW 图像产生 ``(patch, meta)``。 调用：`self._shape`, `self.metas`。
- `M L155-L160` `PatchStream.to_device(self, patch: torch.Tensor, device: torch.device | str) -> torch.Tensor`：在允许时通过非阻塞传输把 patch 搬到目标设备。
- `M L163-L169` `PatchStream._shape(frame: torch.Tensor) -> tuple[int, int, int]`：执行 `shape` 对应逻辑。
- `M L172-L187` `PatchStream._validate_coverage(metas: tuple[PatchMeta, ...], *, frame_width: int, frame_height: int) -> None`：校验 `coverage` 对应的数据或结果。

## `src/traning/lib/data/preprocessing_metadata.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts.common`

- `F L13-L60` `load_preprocessing_metadata(dataset_root: Path, item_name: str) -> JSONObject | None`：读取最近一次成功视频预处理记录；缺失或损坏时返回 ``None``。 调用：`_status_db_for_dataset_root`, `typed_detail.get`。
- `F L63-L71` `_status_db_for_dataset_root(dataset_root: Path) -> Path | None`：执行 `status db for dataset root` 对应逻辑。

## `src/traning/lib/data/sampling.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.lib.data.models`

- `F L10-L36` `build_frame_references(records: tuple[SegmentRecord, ...], *, sample_fps: float, frame_step: int, max_frames_per_segment: int | None) -> tuple[FrameReference, ...]`：按固定采样频率生成 segment 相对时间引用，不读取视频内容。 调用：`FrameReference`。

## `src/traning/lib/data/synthetic_structures.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L11-L18` `SyntheticStructure` [CLASS]：模型与融合冒烟测试使用的小型合成图像及其几何真值。
- `F L21-L26` `_coordinate_grid(width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]`：执行 `coordinate grid` 对应逻辑。
- `F L29-L30` `_image_from_mask(mask: torch.Tensor, *, channels: int=3) -> torch.Tensor`：执行 `image from mask` 对应逻辑。
- `F L33-L51` `make_cross_patch_ring(*, width: int=768, height: int=768, center: tuple[float, float]=(384.0, 384.0), radius: float=210.0, thickness: float=8.0) -> SyntheticStructure`：生成圆周跨越四个典型 patch 的环形结构。 调用：`SyntheticStructure`, `_coordinate_grid`, `_image_from_mask`。
- `F L54-L70` `make_boundary_circle(*, width: int=768, height: int=512, center: tuple[float, float]=(512.0, 256.0), radius: float=48.0) -> SyntheticStructure`：生成圆心位于典型 patch 边界上的实心圆。 调用：`SyntheticStructure`, `_coordinate_grid`, `_image_from_mask`。
- `F L73-L99` `make_cross_patch_slider(*, width: int=1152, height: int=512, start: tuple[float, float]=(120.0, 256.0), end: tuple[float, float]=(1032.0, 256.0), thickness: float=12.0) -> SyntheticStructure`：生成横跨多个 patch 窗口的长直 slider。 调用：`SyntheticStructure`, `_coordinate_grid`, `_image_from_mask`。
- `F L102-L116` `make_spinner(*, width: int=768, height: int=768, center: tuple[float, float]=(384.0, 384.0), radius: float=260.0) -> SyntheticStructure`：生成带高亮外沿的大型 spinner 状圆盘。 调用：`SyntheticStructure`, `_coordinate_grid`。
- `F L119-L130` `make_noise_background(*, width: int=512, height: int=512, seed: int=2026) -> SyntheticStructure`：生成供背景鲁棒性冒烟测试使用的确定性噪声。 调用：`SyntheticStructure`。

## `src/traning/lib/data/tiling.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L12-L30` `PatchWindow` [CLASS]：完整帧坐标系中的半开 patch 窗口。
- `M L21-L24` `PatchWindow.right(self) -> int` [PROPERTY]：返回窗口右侧的半开像素边界。
- `M L27-L30` `PatchWindow.bottom(self) -> int` [PROPERTY]：返回窗口底部的半开像素边界。
- `F L33-L47` `_axis_starts(size: int, patch_size: int, overlap: int) -> tuple[int, ...]`：执行 `axis starts` 对应逻辑。
- `F L50-L72` `build_patch_windows(image_width: int, image_height: int, *, patch_width: int, patch_height: int, overlap_x: int, overlap_y: int) -> tuple[PatchWindow, ...]`：构建行优先窗口；窗口坐标属于完整帧，right/bottom 为半开上界。 调用：`PatchWindow`, `_axis_starts`。
- `F L75-L91` `iter_patches(image: Tensor, windows: tuple[PatchWindow, ...]) -> Iterator[tuple[PatchWindow, Tensor]]`：按给定窗口顺序产生完整帧中的 CHW patch 视图。

## `src/traning/lib/data/video_reader.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L12-L67` `VideoReader` [CLASS]：带 LRU 句柄缓存的随机访问视频读取器。
- `M L15-L19` `VideoReader.__init__(self, max_open_videos: int=4)`：初始化实例依赖、配置和运行状态。
- `M L21-L33` `VideoReader._capture(self, path: Path) -> cv2.VideoCapture`：执行 `capture` 对应逻辑。 调用：`self._captures.pop`, `self._captures.popitem`。
- `M L35-L43` `VideoReader.read_frame(self, path: Path, frame_index: int) -> np.ndarray` [IO-R]：按零起始帧序号解码并返回 RGB 数组。 调用：`capture.read`, `self._capture`。
- `M L45-L57` `VideoReader.read_frame_at(self, path: Path, timestamp_ms: float) -> np.ndarray` [IO-R]：按非负相对毫秒定位并解码一个 RGB 帧。 调用：`capture.read`, `self._capture`。
- `M L59-L64` `VideoReader.close(self) -> None`：立即释放缓存中的全部 OpenCV 视频句柄。 调用：`self._captures.clear`, `self._captures.values`。
- `M L66-L67` `VideoReader.__del__(self) -> None`：执行 `del` 对应逻辑。 调用：`self.close`。

## `src/traning/lib/runtime/memory.py`

职责：统一 CUDA、AMP、GradScaler、channels-last、TF32、显存快照和 OOM 建议。

- `C L17-L24` `MemorySnapshot` [CLASS]：PyTorch CUDA allocator 的峰值与当前 allocated/reserved 快照。
- `C L28-L60` `RuntimeMemoryBudget` [CLASS]：预算检查后得到的主存、显存和 CUDA allocator 上限记录。
- `M L44-L60` `RuntimeMemoryBudget.as_dict(self) -> dict[str, float | str | None]`：返回适合持久化和诊断输出的内存预算字段映射。
- `C L64-L70` `CudaRuntimeConfig` [CLASS]：CUDA 数值性能开关；CPU 设备会忽略 CUDA 专属项。
- `C L74-L96` `CudaRuntimeState` [CLASS]：应用运行时配置后可记录、可比较的 CUDA 实际状态。
- `M L85-L96` `CudaRuntimeState.as_dict(self) -> dict[str, str | bool]`：返回适合日志和诊断输出的强类型字段映射。
- `F L99-L203` `enforce_runtime_memory_budget(*, device: torch.device, max_vram_gib: float, reserve_vram_gib: float, max_ram_gib: float | None, reserve_ram_gib: float, set_cuda_fraction: bool=True) -> RuntimeMemoryBudget`：验证 CPU/CUDA 预算，并为宿主系统保留不可占用的余量。 调用：`RuntimeMemoryBudget`, `_finite`。
- `F L206-L227` `resolve_amp_dtype(device: torch.device, amp_dtype: AmpDType) -> torch.dtype | None`：解析 AMP 精度；非 CUDA 或 float32 返回 ``None`` 表示禁用 autocast。
- `F L231-L240` `autocast_context(device: torch.device, amp_dtype: AmpDType) -> Iterator[None]`：提供统一上下文；AMP 关闭时退化为无操作上下文。 调用：`resolve_amp_dtype`。
- `F L243-L276` `configure_torch_runtime(*, device: torch.device, amp_dtype: AmpDType, runtime: CudaRuntimeConfig=CudaRuntimeConfig()) -> CudaRuntimeState`：应用训练与冒烟测试共用的 CUDA 数值和卷积运行时设置。 调用：`CudaRuntimeState`, `amp_uses_grad_scaler`, `resolve_amp_dtype`。
- `F L279-L284` `amp_uses_grad_scaler(device: torch.device, amp_dtype: AmpDType) -> bool`：仅 float16 CUDA 路径需要 GradScaler；bfloat16 通常无需缩放。 调用：`resolve_amp_dtype`。
- `F L287-L302` `create_grad_scaler(*, device: torch.device, amp_dtype: AmpDType, mode: str='auto') -> torch.amp.GradScaler`：按设备、AMP 精度和显式模式构造统一 GradScaler。 调用：`amp_uses_grad_scaler`。
- `F L305-L316` `module_to_device(module: nn.Module, device: torch.device, *, channels_last: bool) -> nn.Module`：搬运模块，并仅在 CUDA 请求时切为 channels-last 内存格式。
- `F L319-L331` `maybe_compile_module(module: nn.Module, *, enabled: bool, mode: str='default') -> nn.Module`：按配置选择性编译模块，不可用时给出明确失败。
- `F L334-L349` `tensor_to_device(tensor: torch.Tensor, device: torch.device, *, channels_last: bool, non_blocking: bool=True) -> torch.Tensor`：搬运张量；只有四维 CUDA 图像张量应用 channels-last。
- `F L352-L369` `collect_memory_snapshot() -> MemorySnapshot`：采集当前默认 CUDA 设备的 PyTorch allocator 统计。 调用：`MemorySnapshot`。
- `F L372-L407` `format_oom_guidance(*, patch_size: tuple[int, int], global_size: tuple[int, int], batch_size: int, amp_dtype: str, config_path: str | None) -> str`：根据内存快照生成可操作且顺序明确的 CUDA OOM 建议。 调用：`collect_memory_snapshot`。
- `F L410-L411` `_finite(value: float) -> bool`：执行 `finite` 对应逻辑。

## `src/traning/outcome/calibration.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.evaluation.metrics`

- `C L18-L30` `CalibrationEvaluation` [CLASS]：同一 validation logits 校准前后的 NLL。
- `M L24-L30` `CalibrationEvaluation.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L34-L64` `ScalarTemperatureCalibrator` [CLASS]：以严格正标量 ``T`` 执行 ``logits / T``。
- `M L39-L45` `ScalarTemperatureCalibrator.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L47-L51` `ScalarTemperatureCalibrator.transform(self, logits: Tensor) -> Tensor`：校准二维 multiclass logits，保持 dtype 与 device。 调用：`_validate_logits`。
- `M L53-L56` `ScalarTemperatureCalibrator.probabilities(self, logits: Tensor) -> Tensor`：返回温度校准后的类别概率。 调用：`self.transform`。
- `M L58-L64` `ScalarTemperatureCalibrator.evaluate(self, logits: Tensor, labels: Tensor) -> CalibrationEvaluation`：在同一 validation batch 上比较校准前后 NLL。 调用：`CalibrationEvaluation`, `_validate_logits_and_labels`, `multiclass_nll`, `self.probabilities`。
- `F L67-L124` `fit_temperature_calibrator(validation_logits: Tensor, validation_labels: Tensor, *, log_temperature_min: float=-4.0, log_temperature_max: float=4.0, grid_steps: int=257) -> ScalarTemperatureCalibrator`：用固定 log-temperature 网格确定性拟合，并保证 NLL 不劣于 T=1。 调用：`ScalarTemperatureCalibrator`, `_validate_logits_and_labels`。
- `F L127-L134` `evaluate_temperature_calibration(calibrator: ScalarTemperatureCalibrator, logits: Tensor, labels: Tensor) -> CalibrationEvaluation`：函数式 typed 入口，委托 calibrator 评估 validation NLL。 调用：`calibrator.evaluate`。
- `F L137-L148` `_validate_logits_and_labels(logits: Tensor, labels: Tensor) -> None`：校验 `logits and labels` 对应的数据或结果。 调用：`_validate_logits`。
- `F L151-L159` `_validate_logits(logits: Tensor) -> None`：校验 `logits` 对应的数据或结果。

## `src/traning/outcome/dataset/artifact.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts`, `traning.evaluation`, `traning.infrastructure`, `traning.outcome.oracle`

- `C L103-L206` `OutcomeDatasetManifest` [CLASS]：复用 canonical ArtifactManifest 的 Outcome 数据集清单。
- `M L108-L134` `OutcomeDatasetManifest.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_identifier`, `require_transform_fingerprint`。
- `M L137-L143` `OutcomeDatasetManifest.records_filename(self) -> str` [PROPERTY]：返回 manifest 已提交的不可变 records generation 文件名。
- `M L146-L149` `OutcomeDatasetManifest.schema_version(self) -> int` [PROPERTY]：返回 Outcome 数据集制品的 schema 版本。
- `M L152-L155` `OutcomeDatasetManifest.dataset_id(self) -> str` [PROPERTY]：返回反事实样本所归属的数据集稳定标识。
- `M L158-L161` `OutcomeDatasetManifest.split(self) -> DataSplit` [PROPERTY]：返回该制品唯一且具体的数据切分。
- `M L164-L167` `OutcomeDatasetManifest.producer_id(self) -> str` [PROPERTY]：返回生成反事实数据集的生产者标识。
- `M L170-L173` `OutcomeDatasetManifest.row_count(self) -> int` [PROPERTY]：返回 manifest 承诺的 Outcome 样本行数。
- `M L176-L179` `OutcomeDatasetManifest.sha256(self) -> str` [PROPERTY]：返回不可变 records generation 的 SHA-256。
- `M L182-L188` `OutcomeDatasetManifest.oracle_version(self) -> str` [PROPERTY]：返回生成反事实标签时使用的 OutcomeOracle 版本。
- `M L191-L197` `OutcomeDatasetManifest.scoring_version(self) -> str` [PROPERTY]：返回生成标签时使用的 canonical scorer 版本。
- `M L200-L206` `OutcomeDatasetManifest.transform_fingerprint(self) -> str` [PROPERTY]：返回从 runtime 原帧坐标生成 oracle label 时使用的变换指纹。
- `C L209-L393` `OutcomeDatasetArtifactStore` [CLASS]：以 manifest-last 协议发布并严格恢复 typed Outcome 样本。
- `M L212-L215` `OutcomeDatasetArtifactStore.__init__(self, directory: Path) -> None`：初始化实例依赖、配置和运行状态。
- `M L217-L277` `OutcomeDatasetArtifactStore.publish(self, dataset: CounterfactualOutcomeDataset, *, dataset_id: str, producer_id: str, created_at_ms: float | None=None) -> OutcomeDatasetManifest`：完整发布新 generation 后，才原子替换唯一提交点 manifest。 调用：`_manifest_to_json`, `atomic_write_json`, `atomic_write_jsonl`, `encoded_records`, `self._manifest`, `sha256_file`。
- `N L246-L257` `OutcomeDatasetArtifactStore.publish.encoded_records() -> Iterable[JSONObject]`：稳定遍历 typed 样本，并在编码前拒绝重复 sample_id。 调用：`_record_to_json`。
- `M L279-L359` `OutcomeDatasetArtifactStore.load(self, *, expected_dataset_id: str, expected_split: DataSplit, expected_producer_id: str, expected_transform_fingerprint: str, expected_schema_version: int=OUTCOME_DATASET_SCHEMA_VERSION) -> CounterfactualOutcomeDataset` [IO-W]：验证 schema、身份、版本、摘要和行数后恢复 typed samples。 调用：`CounterfactualOutcomeDataset`, `IntegrityError`, `SchemaMismatchError`, `_decode_record_line`, `_manifest_from_json`, `read_json_object`。
- `M L361-L393` `OutcomeDatasetArtifactStore._manifest(self, *, dataset_id: str, split: DataSplit, producer_id: str, row_count: int, digest: str, records_filename: str, oracle_version: str, scoring_version: str, transform_fingerprint: str, created_at_ms: float) -> OutcomeDatasetManifest`：执行 `manifest` 对应逻辑。 调用：`ArtifactManifest`, `OutcomeDatasetManifest`。
- `F L396-L431` `_record_to_json(record: OutcomeTrainingSample) -> JSONObject`：执行 `record to json` 对应逻辑。
- `F L434-L452` `_manifest_to_json(manifest: OutcomeDatasetManifest) -> JSONObject`：执行 `manifest to json` 对应逻辑。
- `F L455-L512` `_manifest_from_json(payload: JSONObject) -> OutcomeDatasetManifest`：执行 `manifest from json` 对应逻辑。 调用：`ArtifactManifest`, `OutcomeDatasetManifest`, `SchemaMismatchError`, `_integer`, `_object`, `_real`。
- `F L515-L528` `_decode_record_line(raw_line: bytes, line_number: int) -> OutcomeTrainingSample`：执行 `decode record line` 对应逻辑。 调用：`IntegrityError`, `SchemaMismatchError`, `_record_from_json`。
- `F L531-L612` `_record_from_json(payload: JSONObject, line_number: int) -> OutcomeTrainingSample`：执行 `record from json` 对应逻辑。 调用：`BeliefState`, `DecisionAction`, `OutcomeCategory`, `OutcomeTrainingSample`, `SchemaMismatchError`, `_boolean`。
- `F L615-L620` `_point(value: object, context: str) -> Point2D`：执行 `point` 对应逻辑。 调用：`Point2D`, `_object`, `_real`, `_require_fields`。
- `F L623-L631` `_type_distribution(value: object, context: str) -> ObjectTypeDistribution`：执行 `type distribution` 对应逻辑。 调用：`ObjectTypeDistribution`, `_object`, `_real`, `_require_fields`。
- `F L634-L641` `_require_fields(payload: JSONObject, expected: frozenset[str], context: str) -> None`：执行 `require fields` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L644-L647` `_object(value: object, context: str) -> JSONObject`：执行 `object` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L650-L653` `_string(value: object, context: str) -> str`：执行 `string` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L656-L659` `_integer(value: object, context: str) -> int`：执行 `integer` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L662-L668` `_real(value: object, context: str) -> float`：执行 `real` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L671-L674` `_boolean(value: object, context: str) -> bool`：执行 `boolean` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L677-L680` `_float_tuple(value: object, context: str) -> tuple[float, ...]`：执行 `float tuple` 对应逻辑。 调用：`SchemaMismatchError`, `_real`。
- `F L683-L687` `_identifier(value: object, name: str) -> None`：执行 `identifier` 对应逻辑。
- `F L690-L698` `_object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]`：执行 `object without duplicate keys` 对应逻辑。 调用：`IntegrityError`。
- `F L701-L702` `_reject_non_finite_constant(value: str) -> object`：执行 `reject non finite constant` 对应逻辑。 调用：`IntegrityError`。

## `src/traning/outcome/dataset/builder.py`

职责：确定性枚举目标与 horizon，用 OutcomeOracle 构造反事实训练样本。
工程依赖：`traning.contracts`, `traning.data`, `traning.outcome.oracle`

- `C L28-L65` `CounterfactualFrame` [CLASS]：同一时刻的 typed beliefs、原帧坐标来源与离线 oracle 状态。
- `M L39-L65` `CounterfactualFrame.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_identifier`, `require_transform_fingerprint`。
- `C L69-L94` `CounterfactualOutcomeDataset` [CLASS]：单一 split 且只使用一个坐标指纹的反事实 Outcome 样本集合。
- `M L76-L94` `CounterfactualOutcomeDataset.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_transform_fingerprint`。
- `C L97-L218` `CounterfactualOutcomeDatasetBuilder` [CLASS]：按 frame、track、horizon 稳定次序生成 CLICK 反事实样本。
- `M L100-L124` `CounterfactualOutcomeDatasetBuilder.__init__(self, oracle: OutcomeOracle, horizons_ms: tuple[float, ...], coordinate_transform: FrameCoordinateTransform) -> None`：初始化实例依赖、配置和运行状态。
- `M L127-L130` `CounterfactualOutcomeDatasetBuilder.horizons_ms(self) -> tuple[float, ...]` [PROPERTY]：返回规范化后的严格递增 horizon。
- `M L132-L218` `CounterfactualOutcomeDatasetBuilder.build(self, frames: Iterable[CounterfactualFrame]) -> CounterfactualOutcomeDataset`：全量校验后，以确定性次序生成 canonical OutcomeTrainingSample。 调用：`CounterfactualOutcomeDataset`, `HypotheticalClick`, `OutcomeTrainingSample`, `Point2D`, `_counterfactual_sample_id`, `_validate_outcome`。
- `F L221-L232` `_counterfactual_sample_id(source_sample_id: str, track_id: str, horizon_index: int, horizon_ms: float) -> str`：用 length-prefix 编码可变标识符，避免分隔符组合产生碰撞。
- `F L235-L243` `_validate_outcome(outcome: OracleOutcome, track_id: str, horizon_ms: float) -> None`：校验 `outcome` 对应的数据或结果。
- `F L246-L250` `_identifier(value: object, name: str) -> None`：执行 `identifier` 对应逻辑。

## `src/traning/outcome/model.py`

职责：根据 belief、动作和 horizon 预测 outcome 分布、期望得分与方差。
工程依赖：`traning.config`, `traning.contracts`

- `C L23-L82` `OutcomeTensorOutput` [CLASS]：批量 Outcome 的 typed 张量输出。
- `M L33-L82` `OutcomeTensorOutput.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.category_probabilities.new_ones`。
- `C L85-L228` `DenseOutcomeModel(nn.Module)` [CLASS]：用 dense MLP 预测 CLICK 在指定 horizon 的结果分布。
- `M L92-L121` `DenseOutcomeModel.__init__(self, config: OutcomeConfig, belief_embedding_dim: int) -> None`：初始化实例依赖、配置和运行状态。 调用：`self.register_buffer`, `super.__init__`。
- `M L123-L157` `DenseOutcomeModel.forward(self, belief_embedding: torch.Tensor, horizon_ms: torch.Tensor) -> OutcomeTensorOutput`：预测 CLICK 条件下五分类结果与独立 expiry 概率。 调用：`OutcomeTensorOutput`, `self._validate_inputs`, `self.category_head`, `self.expiry_head`, `self.expiry_head.squeeze`, `self.score_representatives.to`。
- `M L159-L195` `DenseOutcomeModel.predict(self, belief: BeliefState, horizon_ms: float) -> OutcomeDistribution`：从公共 belief 契约生成单轨迹 canonical OutcomeDistribution。 调用：`OutcomeDistribution`, `self.forward`, `self.parameters`。
- `M L197-L228` `DenseOutcomeModel._validate_inputs(self, belief_embedding: torch.Tensor, horizon_ms: torch.Tensor) -> None`：校验 `inputs` 对应的数据或结果。 调用：`self.parameters`。

## `src/traning/outcome/oracle/oracle.py`

职责：把 canonical 点、slider 和点击序列评分封装为反事实 OutcomeOracle。
工程依赖：`traning.contracts`, `traning.evaluation`

- `C L31-L61` `OracleTarget` [CLASS]：离线 oracle 可见的单个目标真值，不进入 runtime。
- `M L42-L61` `OracleTarget.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_identifier`, `_require_nonnegative`, `_require_point_path`。
- `C L65-L93` `OracleState` [CLASS]：某一时刻离线 oracle 的不可变目标快照。
- `M L73-L93` `OracleState.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_identifier`, `_require_nonnegative`。
- `C L97-L112` `HypotheticalClick` [CLASS]：对指定轨迹和未来 horizon 的离线反事实点击。
- `M L105-L112` `HypotheticalClick.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_identifier`, `_require_nonnegative`, `_require_point_path`。
- `C L116-L171` `OracleOutcome` [CLASS]：离线 canonical score 到五分类 Outcome 标签的完整投影。
- `M L130-L171` `OracleOutcome.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_identifier`, `_require_nonnegative`, `_require_probability`。
- `C L175-L307` `OutcomeOracle` [CLASS]：调用唯一 V2 评分实现生成离线 Outcome 监督。
- `M L181-L184` `OutcomeOracle.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_positive`。
- `M L186-L259` `OutcomeOracle.evaluate(self, state: OracleState, hypothetical_action: HypotheticalClick) -> OracleOutcome`：在 ``state.timestamp + horizon`` 执行反事实点击并生成标签。 调用：`_invalid_outcome`, `_point_tuple`, `_scored_outcome`, `score_slider`, `self._evaluate_point`, `target_by_track.get`。
- `M L261-L279` `OutcomeOracle.evaluate_sequence(self, targets: tuple[TargetObject, ...], clicks: tuple[PredictedClick, ...], *, min_click_interval_ms: float=50.0) -> SequenceScore`：委托 canonical sequence scorer 完成匹配、频率限制和错误归因。 调用：`SequenceScoreSpec`, `score_click_sequence`。
- `M L281-L307` `OutcomeOracle._evaluate_point(self, action: HypotheticalClick, *, target: OracleTarget, execution_time_ms: float, reference_position: tuple[float, float], predicted_position: tuple[float, float]) -> OracleOutcome`：使用 canonical point score 评估 ring 或 slider head。 调用：`_scored_outcome`, `score_point`。
- `F L310-L324` `_invalid_outcome(action: HypotheticalClick, *, target_object_id: str | None, expires: bool) -> OracleOutcome`：执行 `invalid outcome` 对应逻辑。 调用：`OracleOutcome`。
- `F L327-L356` `_scored_outcome(action: HypotheticalClick, *, target: OracleTarget, normalized_score: float, passed: bool, spatial_error: float, time_error_ms: float) -> OracleOutcome`：执行 `scored outcome` 对应逻辑。 调用：`OracleOutcome`, `_require_probability`。
- `F L359-L360` `_point_tuple(point: Point2D) -> tuple[float, float]`：执行 `point tuple` 对应逻辑。
- `F L363-L367` `_require_point_path(path: tuple[Point2D, ...], field_name: str) -> None`：执行 `require point path` 对应逻辑。
- `F L370-L374` `_require_identifier(value: str, field_name: str) -> None`：执行 `require identifier` 对应逻辑。
- `F L377-L381` `_require_nonnegative(value: float, field_name: str) -> None`：执行 `require nonnegative` 对应逻辑。
- `F L384-L387` `_require_positive(value: float, field_name: str) -> None`：执行 `require positive` 对应逻辑。 调用：`_require_nonnegative`。
- `F L390-L393` `_require_probability(value: float, field_name: str) -> None`：执行 `require probability` 对应逻辑。 调用：`_require_nonnegative`。

## `src/traning/outcome/training.py`

职责：训练 OutcomeModel 并计算 NLL、Brier、ECE 和 expected-score MAE。
工程依赖：`traning.contracts`, `traning.contracts.common`, `traning.evaluation.metrics`, `traning.outcome.dataset.builder`, `traning.outcome.model`

- `C L32-L137` `OutcomeBatch` [CLASS]：保留样本血缘且可直接输入 dense Outcome 模型的批次。
- `M L49-L137` `OutcomeBatch.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_transform_fingerprint`。
- `C L141-L161` `OutcomeLossWeights` [CLASS]：主任务权重必须为正；score 只能作为非负辅助项。
- `M L148-L161` `OutcomeLossWeights.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L165-L181` `OutcomeLoss` [CLASS]：单个 Outcome batch 的可审计损失分解。
- `M L173-L181` `OutcomeLoss.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_validate_scalar_tensors`。
- `C L185-L203` `OutcomeEvaluationMetrics` [CLASS]：Outcome batch 的分类、校准、分数和 expiry 指标。
- `M L194-L203` `OutcomeEvaluationMetrics.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_validate_scalar_tensors`。
- `F L206-L271` `collate_outcome_samples(dataset: CounterfactualOutcomeDataset, belief_embedding_dim: int, *, record_indices: tuple[int, ...] | None=None) -> OutcomeBatch`：从单一有指纹数据集组装 CPU float32 batch，禁止混合裸 records。 调用：`OutcomeBatch`。
- `F L274-L296` `compute_outcome_loss(output: OutcomeTensorOutput, batch: OutcomeBatch, weights: OutcomeLossWeights=OutcomeLossWeights()) -> OutcomeLoss`：以分类和 expiry 为主任务，expected score 仅作辅助回归。 调用：`OutcomeLoss`, `_validate_output_batch_alignment`。
- `F L299-L330` `evaluate_outcome_batch(output: OutcomeTensorOutput, batch: OutcomeBatch, *, calibration_bins: int=15) -> OutcomeEvaluationMetrics`：复用 Phase 7 canonical metrics 评估一个 Outcome batch。 调用：`OutcomeEvaluationMetrics`, `_validate_output_batch_alignment`, `expected_score_mae`, `expiry_brier_score`, `multiclass_brier_score`, `multiclass_nll`。
- `F L333-L353` `train_outcome_step(model: DenseOutcomeModel, batch: OutcomeBatch, optimizer: torch.optim.Optimizer, weights: OutcomeLossWeights=OutcomeLossWeights()) -> OutcomeLoss`：执行一个标准 dense Outcome 优化步骤并返回损失分解。 调用：`compute_outcome_loss`, `model.train`, `optimizer.step`, `optimizer.zero_grad`。
- `F L356-L363` `_validate_output_batch_alignment(output: OutcomeTensorOutput, batch: OutcomeBatch) -> None`：校验 `output batch alignment` 对应的数据或结果。
- `F L368-L384` `_validate_scalar_tensors(values: tuple[tuple[str, torch.Tensor], ...]) -> None`：校验 `scalar tensors` 对应的数据或结果。

## `src/traning/perception/decode/decoder.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.config`, `traning.contracts`, `traning.contracts.common`, `traning.perception.models`

- `C L29-L36` `_DecodedCell` [CLASS]：排序和 NMS 期间使用的局部候选，不越过模块边界。
- `F L39-L115` `decode_candidates(output: DensePerceptionOutput, *, frame_id: str, frame_index: int, timestamp_ms: float, frame_width: int, frame_height: int, config: PerceptionConfig, batch_index: int=0) -> tuple[CandidateObservation, ...]`：按复合分数、局部极大和确定性 NMS 解码一个 batch 元素。 调用：`_DecodedCell`, `_build_observation`, `_clamp_offset`, `_deterministic_nms`, `_finite_scalar`, `_validate_decode_request`。
- `F L118-L151` `_validate_decode_request(output: DensePerceptionOutput, batch_index: int, frame_width: int, frame_height: int) -> None`：校验 `decode request` 对应的数据或结果。
- `F L154-L169` `_deterministic_nms(candidates: list[_DecodedCell], *, radius_px: float, limit: int) -> tuple[_DecodedCell, ...]`：执行 `deterministic nms` 对应逻辑。
- `F L172-L259` `_build_observation(output: DensePerceptionOutput, cell: _DecodedCell, *, batch_index: int, frame_id: str, frame_index: int, timestamp_ms: float, scale_x: float, scale_y: float) -> CandidateObservation`：构建 `observation` 对应的数据或结果。 调用：`CandidateObservation`, `ObjectTypeDistribution`, `Point2D`, `RingAttributes`, `SliderAttributes`, `SpinnerAttributes`。
- `F L262-L266` `_finite_scalar(value: Tensor, field_name: str) -> float`：执行 `finite scalar` 对应逻辑。
- `F L269-L270` `_sigmoid_scalar(value: Tensor) -> float`：执行 `sigmoid scalar` 对应逻辑。 调用：`_finite_scalar`。
- `F L273-L274` `_clamp_offset(value: float) -> float`：执行 `clamp offset` 对应逻辑。

## `src/traning/perception/models/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`traning.config`

- `F L18-L22` `_group_count(channels: int) -> int`：执行 `group count` 对应逻辑。
- `F L25-L29` `_require_bchw(name: str, tensor: torch.Tensor) -> None`：执行 `require bchw` 对应逻辑。
- `C L33-L42` `LocalFeatureOutput` [CLASS]：局部编码器的 stride-8 BCHW 特征。
- `M L39-L42` `LocalFeatureOutput.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_bchw`。
- `C L46-L55` `GlobalFeatureOutput` [CLASS]：全局编码器的 stride-16 BCHW 上下文。
- `M L52-L55` `GlobalFeatureOutput.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_bchw`。
- `C L59-L72` `FusedFeatureOutput` [CLASS]：与局部网格对齐的门控融合结果。
- `M L66-L72` `FusedFeatureOutput.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_bchw`。
- `C L76-L118` `DensePerceptionOutput` [CLASS]：stride-8 网格上的完整稠密感知输出。
- `M L95-L118` `DensePerceptionOutput.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_bchw`。
- `C L121-L144` `DepthwiseSeparableConv(nn.Module)` [CLASS]：深度卷积与逐点卷积组成的低成本卷积块。
- `M L124-L139` `DepthwiseSeparableConv.__init__(self, in_channels: int, out_channels: int, *, stride: int=1) -> None`：初始化实例依赖、配置和运行状态。 调用：`_group_count`, `super.__init__`。
- `M L141-L144` `DepthwiseSeparableConv.forward(self, tensor: torch.Tensor) -> torch.Tensor`：执行深度卷积、逐点投影、归一化与激活。 调用：`self.activation`, `self.depthwise`, `self.norm`, `self.pointwise`。
- `C L147-L172` `_SeparableResidualBlock(nn.Module)` [CLASS]：带可学习降采样捷径的深度可分离残差块。
- `M L150-L167` `_SeparableResidualBlock.__init__(self, in_channels: int, out_channels: int, *, stride: int) -> None`：初始化实例依赖、配置和运行状态。 调用：`DepthwiseSeparableConv`, `_group_count`, `super.__init__`。
- `M L169-L172` `_SeparableResidualBlock.forward(self, tensor: torch.Tensor) -> torch.Tensor`：融合主分支与尺寸匹配的捷径并返回残差特征。 调用：`self.activation`, `self.first`, `self.second`, `self.skip`。
- `C L175-L202` `LocalEncoder(nn.Module)` [CLASS]：保存细粒度形状的 stride-8 局部特征编码器。
- `M L178-L195` `LocalEncoder.__init__(self, *, in_channels: int=3, feature_channels: int=48) -> None`：初始化实例依赖、配置和运行状态。 调用：`_SeparableResidualBlock`, `_group_count`, `super.__init__`。
- `M L197-L202` `LocalEncoder.forward(self, frame: torch.Tensor) -> LocalFeatureOutput`：把完整帧编码为保留局部形状的 stride-8 特征。 调用：`LocalFeatureOutput`, `_require_bchw`, `self.stage2`, `self.stage4`, `self.stage8`, `self.stem`。
- `C L205-L229` `_GlobalConvBlock(nn.Module)` [CLASS]：全局支路使用的两层卷积降采样块。
- `M L208-L224` `_GlobalConvBlock.__init__(self, in_channels: int, out_channels: int) -> None`：初始化实例依赖、配置和运行状态。 调用：`_group_count`, `super.__init__`。
- `M L226-L229` `_GlobalConvBlock.forward(self, tensor: torch.Tensor) -> torch.Tensor`：连续执行两层卷积并完成一次二倍全局降采样。 调用：`self.block`。
- `C L232-L259` `GlobalEncoder(nn.Module)` [CLASS]：以完整帧提供 stride-16 的低分辨率全局上下文。
- `M L235-L252` `GlobalEncoder.__init__(self, *, in_channels: int=3, feature_channels: int=64, pretrained: bool=False, frozen: bool=False) -> None`：初始化实例依赖、配置和运行状态。 调用：`_GlobalConvBlock`, `self.requires_grad_`, `super.__init__`。
- `M L254-L259` `GlobalEncoder.forward(self, frame: torch.Tensor) -> GlobalFeatureOutput`：把完整帧编码为 stride-16 的低分辨率全局上下文。 调用：`GlobalFeatureOutput`, `_require_bchw`, `self.stage16`, `self.stage2`, `self.stage4`, `self.stage8`。
- `F L262-L275` `_normalized_grid(*, batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor`：构建与完整局部特征网格对齐的归一化采样中心。
- `F L278-L289` `_safe_channel_normalize(tensor: torch.Tensor, *, eps: float=1e-06) -> torch.Tensor`：逐像素归一化，并把精确零向量映射到确定性的第一坐标轴。 调用：`F.normalize`。
- `C L292-L361` `GatedFusion(nn.Module)` [CLASS]：通过可学习稀疏采样和逐通道门控融合局部与全局特征。
- `M L295-L322` `GatedFusion.__init__(self, *, local_channels: int=48, global_channels: int=64, sampling_points: int=4) -> None`：初始化实例依赖、配置和运行状态。 调用：`_group_count`, `super.__init__`。
- `M L324-L361` `GatedFusion.forward(self, local: LocalFeatureOutput, global_features: GlobalFeatureOutput) -> FusedFeatureOutput`：在局部网格采样全局上下文并以逐通道门控完成融合。 调用：`FusedFeatureOutput`, `_normalized_grid`, `self.gate_predictor`, `self.global_project`, `self.offset_predictor`, `self.offset_predictor.view`。
- `C L364-L415` `SpatialHead(nn.Module)` [CLASS]：从融合特征产生全部 V2 稠密图；每个构建 head 都参与 forward。
- `M L379-L397` `SpatialHead.__init__(self, *, in_channels: int=48, embedding_dim: int=32) -> None`：初始化实例依赖、配置和运行状态。 调用：`_group_count`, `super.__init__`。
- `M L399-L415` `SpatialHead.forward(self, features: FusedFeatureOutput) -> DensePerceptionOutput`：由融合特征一次产生全部稠密感知图与身份向量。 调用：`DensePerceptionOutput`, `_safe_channel_normalize`, `self.heads.items`, `self.identity_head`, `self.trunk`。
- `C L418-L455` `PerceptionModel(nn.Module)` [CLASS]：完整 RGB 帧的统一感知图：local → global → fusion → spatial。
- `M L421-L437` `PerceptionModel.__init__(self, config: PerceptionConfig) -> None`：初始化实例依赖、配置和运行状态。 调用：`GatedFusion`, `GlobalEncoder`, `LocalEncoder`, `SpatialHead`, `super.__init__`。
- `M L439-L455` `PerceptionModel.forward(self, frame: torch.Tensor) -> DensePerceptionOutput`：执行无 GT 的 full-frame local→global→fusion→spatial 推理。 调用：`_require_bchw`, `self.fusion`, `self.global_encoder`, `self.local_encoder`, `self.spatial_head`。

## `src/traning/perception/runtime/runtime.py`

职责：只从 RuntimeFrame 产生 CandidateObservation，不接触训练标签或 oracle。
工程依赖：`traning.config`, `traning.contracts`, `traning.perception.decode`, `traning.perception.models`

- `C L20-L30` `RuntimeTensorFrame` [CLASS]：显式 resize 后的 BCHW RGB tensor 及其原始帧身份。
- `M L26-L30` `RuntimeTensorFrame.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.image.is_floating_point`。
- `C L33-L37` `DensePerceptionModel(Protocol)` [CLASS]：运行时需要的最小模型调用契约。
- `M L36-L37` `DensePerceptionModel.__call__(self, image: Tensor) -> DensePerceptionOutput`：执行单批次稠密推理。
- `F L40-L64` `runtime_frame_to_tensor(frame: RuntimeFrame, config: PerceptionConfig) -> RuntimeTensorFrame`：严格按 raw RGB 解码，并 resize 为配置输入尺寸。 调用：`RuntimeTensorFrame`。
- `F L67-L83` `decode_runtime_output(tensor_frame: RuntimeTensorFrame, output: DensePerceptionOutput, config: PerceptionConfig) -> tuple[CandidateObservation, ...]`：将网络尺寸上的输出直接映回 RuntimeFrame 原始坐标。 调用：`decode_candidates`。
- `C L86-L121` `PerceptionRuntime` [CLASS]：串接严格 RGB 适配、模型调用和 typed candidate 解码。
- `M L89-L109` `PerceptionRuntime.__init__(self, model: DensePerceptionModel, config: PerceptionConfig, *, device: torch.device | str='cpu', amp: bool=False) -> None`：初始化实例依赖、配置和运行状态。
- `M L111-L121` `PerceptionRuntime.infer(self, frame: RuntimeFrame) -> tuple[CandidateObservation, ...]`：执行单帧无监督信息注入的感知推理。 调用：`decode_runtime_output`, `runtime_frame_to_tensor`, `self._model`。

## `src/traning/perception/training/losses.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.perception.models`

- `C L18-L41` `PerceptionLossWeights` [CLASS]：各监督头的非负权重。
- `M L33-L41` `PerceptionLossWeights.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L45-L58` `PerceptionLoss` [CLASS]：保留每个监督分量及其加权总和，便于训练遥测。
- `F L61-L133` `compute_perception_loss(prediction: DensePerceptionOutput, targets: PerceptionTargets, weights: PerceptionLossWeights) -> PerceptionLoss`：计算完整 dense 监督；梯度保持贯穿所有预测 head 和上游融合网络。 调用：`PerceptionLoss`, `_instance_margin_loss`, `_masked_binary_cross_entropy`, `_masked_smooth_l1`, `_type_loss`, `_validate_prediction`。
- `F L136-L172` `_validate_prediction(prediction: DensePerceptionOutput, targets: PerceptionTargets) -> None`：校验 `prediction` 对应的数据或结果。
- `F L175-L179` `_masked_binary_cross_entropy(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor`：执行 `masked binary cross entropy` 对应逻辑。 调用：`_masked_mean`。
- `F L182-L192` `_type_loss(logits: Tensor, type_indices: Tensor) -> Tensor`：执行 `type loss` 对应逻辑。
- `F L195-L197` `_masked_smooth_l1(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor`：执行 `masked smooth l1` 对应逻辑。 调用：`_masked_mean`。
- `F L200-L206` `_masked_mean(values: Tensor, mask: Tensor) -> Tensor`：执行 `masked mean` 对应逻辑。
- `F L209-L244` `_instance_margin_loss(embedding: Tensor, instance_ids: Tensor, *, margin: float) -> Tensor`：跨整个时序 batch 计算 prototype pull 与跨实例 cosine margin。 调用：`F.normalize`。

## `src/traning/perception/training/targets.py`

职责：通过共享 FrameCoordinateTransform 把真实标注映射为训练目标。
工程依赖：`traning.contracts`, `traning.data.coordinates`, `traning.perception.models`

- `C L33-L84` `CoordinateTrainingTarget` [CLASS]：用共享标定投影到原帧像素的单个感知监督目标。
- `M L42-L84` `CoordinateTrainingTarget.__post_init__(self) -> None`：确保 builder 不会通过宽松字典丢失目标身份或坐标来源。 调用：`self.object_id.strip`。
- `F L87-L144` `build_coordinate_training_targets(sample: TrainingSample, coordinate_transform: FrameCoordinateTransform) -> tuple[CoordinateTrainingTarget, ...]`：将样本中 canonical osu! GT 统一映射成稠密感知使用的原帧坐标。 调用：`CoordinateTrainingTarget`, `OsuPoint`, `_slider_direction_to_training_target`, `coordinate_transform.ground_truth_radius_to_training_target`, `coordinate_transform.ground_truth_to_training_target`。
- `F L147-L174` `_slider_direction_to_training_target(coordinate_transform: FrameCoordinateTransform, *, start: Point2D, end: Point2D, source_frame_width: int, source_frame_height: int) -> tuple[float, float]`：用共享 affine 方程转换 slider 首段，同时允许控制点越出游玩区。 调用：`coordinate_transform.ground_truth_direction_to_training_target`。
- `F L177-L300` `rasterize_perception_targets(samples: tuple[TrainingSample, ...], prediction: DensePerceptionOutput, coordinate_transform: FrameCoordinateTransform) -> PerceptionTargets`：把 canonical GT 映射到与 dense 输出完全同构的监督张量。 调用：`PerceptionTargets`, `build_coordinate_training_targets`。
- `C L304-L368` `PerceptionTargets` [CLASS]：与 dense 输出逐头对齐、且不会进入 runtime 的监督张量。
- `M L318-L368` `PerceptionTargets.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。

## `src/traning/telemetry/events.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.config.versions`, `traning.contracts.common`, `traning.contracts.telemetry`, `traning.evaluation.attribution`

- `C L20-L26` `TelemetryChannel(str, Enum)` [CLASS]：正式持久化通道；枚举值就是稳定 JSON ``record_type``。
- `C L30-L34` `ChannelSpec` [CLASS]：把通道与固定文件名集中注册，避免各消费者自行拼接路径。
- `F L46-L54` `_validate_header(schema_version: int, timestamp_ms: float, run_id: str) -> None`：校验所有事件共享的 schema、时间戳和运行标识。 调用：`require_identifier`, `require_nonnegative`。
- `F L57-L63` `_validate_step(step: int) -> None`：拒绝布尔值、非整数和负训练步。
- `F L66-L72` `_validate_count(value: int, field_name: str) -> None`：校验不会被 bool 冒充的非负计数。
- `C L76-L106` `MetricsEvent` [CLASS]：一次训练/评估步的完整模型质量与决策指标快照。
- `M L94-L106` `MetricsEvent.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_validate_count`, `_validate_header`, `_validate_step`, `require_finite`, `require_nonnegative`, `require_probability`。
- `C L110-L130` `ResourceEvent` [CLASS]：一次资源采样，覆盖 GPU、显存和端到端吞吐量。
- `M L122-L130` `ResourceEvent.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_validate_header`, `_validate_step`, `require_nonnegative`, `require_probability`。
- `C L134-L150` `EvaluationEvent` [CLASS]：canonical 评分事件的无损遥测信封。
- `M L147-L150` `EvaluationEvent.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_validate_header`。
- `F L159-L170` `event_channel(event: PublishableTelemetryEvent) -> TelemetryChannel`：返回事件的唯一持久化通道，拒绝联合外的运行时对象。

## `src/traning/telemetry/reporter.py`

职责：把训练指标、资源、评估和事件写入线程安全 telemetry store。
工程依赖：`traning.contracts.common`, `traning.contracts.telemetry`, `traning.evaluation.attribution`, `traning.telemetry.events`, `traning.telemetry.store`

- `C L30-L67` `DashboardMetrics` [CLASS]：训练与推理各层的完整、不可变指标投影。
- `M L45-L67` `DashboardMetrics.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_finite`, `require_nonnegative`, `require_probability`。
- `C L71-L89` `DashboardResources` [CLASS]：GPU、显存与吞吐资源的不可变投影。
- `M L79-L89` `DashboardResources.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_nonnegative`, `require_probability`。
- `C L93-L123` `DashboardSnapshot` [CLASS]：Renderer 唯一允许消费的 versioned dashboard 快照。
- `M L103-L123` `DashboardSnapshot.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_identifier`, `require_nonnegative`。
- `F L126-L164` `project_dashboard(run_id: str, snapshot: StoreSnapshot) -> DashboardSnapshot`：逐字段投影 store 的最新事件，不推导领域判断或复制评估事件。 调用：`DashboardSnapshot`, `_project_metrics`, `_project_resources`, `require_identifier`。
- `F L167-L184` `_project_metrics(event: MetricsEvent | None) -> DashboardMetrics | None`：把完整 MetricsEvent 原值复制到展示契约。 调用：`DashboardMetrics`。
- `F L187-L197` `_project_resources(event: ResourceEvent | None) -> DashboardResources | None`：把完整 ResourceEvent 原值复制到展示契约。 调用：`DashboardResources`。
- `C L201-L227` `TelemetryReporter` [CLASS]：训练/推理线程与唯一 StateStore 之间的极薄 typed 发布边界。
- `M L207-L210` `TelemetryReporter.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_identifier`。
- `M L212-L222` `TelemetryReporter.publish(self, event: PublishableTelemetryEvent) -> None`：校验 run identity 后把 typed event 原对象交给 store。 调用：`self.store.publish`。
- `M L224-L227` `TelemetryReporter.snapshot(self) -> DashboardSnapshot`：从 store 的原子快照即时生成 renderer 输入，不维护旁路缓存。 调用：`project_dashboard`, `self.store.snapshot`。

## `src/traning/telemetry/store.py`

职责：维护带 schema_version 的快照，并原子追加 JSONL 遥测。
工程依赖：`traning.contracts.common`, `traning.contracts.telemetry`, `traning.evaluation.attribution`, `traning.infrastructure.errors`, `traning.infrastructure.persistence`

- `C L111-L122` `StoreSnapshot` [CLASS]：四通道最新值的不可变跨线程视图。
- `C L126-L133` `TelemetryHistory` [CLASS]：一次锁内复制得到的四通道完整历史。
- `C L136-L331` `StateStore` [CLASS]：把 typed events 追加到固定 JSONL，并提供只读 snapshot/history。
- `M L144-L167` `StateStore.__init__(self, directory: Path, *, schema_version: int=TELEMETRY_SCHEMA_VERSION) -> None`：初始化实例依赖、配置和运行状态。 调用：`self._initialize_or_recover`。
- `M L170-L173` `StateStore.directory(self) -> Path` [PROPERTY]：返回调用方明确传入的目录，不做隐式 fallback。
- `M L175-L193` `StateStore.publish(self, event: PublishableTelemetryEvent) -> None`：耐久化一个 typed event，并在成功后原子推进内存状态。 调用：`_copy_contract_event`, `_encode_event`, `event_channel`, `self._append_record`, `self._remember`。
- `M L195-L212` `StateStore.snapshot(self) -> StoreSnapshot`：复制四通道最新状态；不暴露可变 live state。 调用：`StoreSnapshot`, `_copy_contract_event`。
- `M L214-L224` `StateStore.history(self) -> TelemetryHistory`：复制完整历史；canonical evaluation 对象无需复制且保持身份。 调用：`TelemetryHistory`, `_copy_contract_event`。
- `M L226-L276` `StateStore._initialize_or_recover(self) -> None` [IO-W]：创建四个新通道，或严格恢复已有的完整通道集合。 调用：`AtomicWriteError`, `IntegrityError`, `SchemaMismatchError`, `_read_channel`, `atomic_write_jsonl`, `self._directory.is_dir`。
- `M L278-L317` `StateStore._append_record(self, channel: TelemetryChannel, record: JSONObject) -> None` [IO-W]：以单个完整 UTF-8 JSON 行追加，并同步文件内容。 调用：`AtomicWriteError`, `IntegrityError`, `SchemaMismatchError`, `os.close`, `os.write`。
- `M L319-L331` `StateStore._remember(self, event: PublishableTelemetryEvent) -> None`：按封闭联合类型推进唯一通道的 immutable tuple。
- `F L334-L404` `_encode_event(event: PublishableTelemetryEvent) -> JSONObject`：将 typed event 无损投影到其版本化 JSON boundary。
- `F L407-L431` `_read_channel(path: Path, channel: TelemetryChannel, schema_version: int) -> tuple[PublishableTelemetryEvent, ...]` [IO-R]：逐行严格解码一个通道，任何损坏都显式终止恢复。 调用：`IntegrityError`, `_decode_json_object`, `_decode_record`。
- `F L434-L451` `_decode_json_object(line: str, path: Path, line_number: int) -> dict[str, object]`：拒绝重复键、NaN/Infinity 和非 object JSON 根节点。 调用：`IntegrityError`, `SchemaMismatchError`。
- `F L454-L547` `_decode_record(record: dict[str, object], expected_channel: TelemetryChannel, schema_version: int) -> PublishableTelemetryEvent`：按目标通道解码，禁止 record_type 串台。 调用：`EvaluationCoordinateSpace`, `EvaluationEvent`, `EvaluationTag`, `MetricsEvent`, `PrimaryError`, `ResourceEvent`。
- `F L550-L564` `_copy_contract_event(event: TelemetryEvent) -> TelemetryEvent`：深复制 JSON payload，避免可变容器成为跨线程 live state。 调用：`TelemetryEvent`, `_copy_json_value`。
- `F L567-L588` `_copy_json_value(value: object, context: str) -> JSONValue`：校验并深复制 JSON 值，同时拒绝非有限数和非字符串键。 调用：`SchemaMismatchError`, `_copy_json_value`。
- `F L591-L600` `_require_keys(record: dict[str, object], expected: frozenset[str]) -> None`：要求字段集合完全相等，拒绝缺字段和未知字段。 调用：`SchemaMismatchError`。
- `F L603-L605` `_require_exact_value(record: dict[str, object], key: str, expected: object) -> None`：执行 `require exact value` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L608-L612` `_string(record: dict[str, object], key: str) -> str`：执行 `string` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L615-L619` `_integer(record: dict[str, object], key: str) -> int`：执行 `integer` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L622-L623` `_number(record: dict[str, object], key: str) -> float`：执行 `number` 对应逻辑。 调用：`_standalone_number`。
- `F L626-L632` `_standalone_number(value: object, key: str) -> float`：执行 `standalone number` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L635-L639` `_boolean(record: dict[str, object], key: str) -> bool`：执行 `boolean` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L642-L646` `_optional_string(record: dict[str, object], key: str) -> str | None`：执行 `optional string` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L649-L655` `_optional_integer(record: dict[str, object], key: str) -> int | None`：执行 `optional integer` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L658-L664` `_optional_number(record: dict[str, object], key: str) -> float | None`：读取可空有限数值，并继续拒绝 bool 冒充数字。 调用：`_standalone_number`。
- `F L667-L671` `_string_list(record: dict[str, object], key: str) -> tuple[str, ...]`：执行 `string list` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L674-L678` `_object(record: dict[str, object], key: str) -> dict[str, object]`：执行 `object` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L681-L689` `_unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]`：object_pairs_hook：在 JSON decoder 边界拒绝重复键。 调用：`IntegrityError`。
- `F L692-L693` `_reject_constant(value: str) -> object`：执行 `reject constant` 对应逻辑。 调用：`IntegrityError`。

## `src/traning/tests/architecture/test_phase11_source_boundaries.py`

职责：Python 模块；具体职责见下方符号及调用。

- `F L38-L49` `_python_paths(*, include_tests: bool) -> tuple[Path, ...]`：稳定枚举 V2 Python 源码，并排除生成缓存与冻结清单。
- `F L52-L55` `_tree(path: Path) -> ast.Module` [IO-R]：用带文件名的 UTF-8 AST 解析提供可定位的失败信息。
- `F L58-L70` `_imported_names(tree: ast.AST) -> tuple[tuple[str, str], ...]`：返回 ``(module, imported_name)``，统一处理两种 import 语法。
- `F L73-L87` `_is_runtime_or_decision(relative: Path) -> bool`：识别正式 runtime 与整个 deterministic decision 包。
- `F L90-L113` `_chinese_docstring_violations(paths: tuple[Path, ...]) -> tuple[str, ...]`：返回指定源码中缺少中文模块或公开定义说明的位置。 调用：`_tree`。
- `F L116-L120` `test_production_modules_and_public_definitions_have_chinese_docstrings() -> None`：生产模块及所有公开定义必须同时有说明和中文领域语义。 调用：`_chinese_docstring_violations`, `_python_paths`。
- `F L123-L132` `test_test_modules_and_public_definitions_have_chinese_docstrings() -> None`：测试模块与公开定义也必须用中文说明其验证意图。 调用：`_chinese_docstring_violations`, `_python_paths`。
- `F L135-L163` `test_repository_python_modules_have_chinese_docstrings() -> None`：全仓 Python 模块必须说明用途；冻结 legacy 不做机械符号级改写。 调用：`_tree`。
- `F L166-L191` `test_production_has_no_retired_namespace_sparse_or_typing_any_dependency() -> None`：正式生产路径不得接回旧命名空间、稀疏主线或宽泛 Any。 调用：`_imported_names`, `_python_paths`, `_tree`。
- `F L194-L216` `test_runtime_and_decision_cannot_import_training_or_oracle_information() -> None`：正式动作路径只允许消费 runtime contracts，不得获得 GT/oracle/logits。 调用：`_imported_names`, `_is_runtime_or_decision`, `_python_paths`, `_tree`。
- `F L219-L233` `test_tests_do_not_import_retired_parallel_namespace() -> None`：迁入后的测试也不得重新依赖已删除的并行包。 调用：`_imported_names`, `_python_paths`, `_tree`。

## `src/traning/tests/integration/test_phase10_canonical_event_flow.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts`, `traning.evaluation`, `traning.telemetry`, `traning.training`, `traning.visualization`

- `F L26-L61` `test_frame_105_event_identity_is_shared_by_all_consumers(tmp_path) -> None`：未点击但画面很准的目标必须始终归入 Decision。 调用：`EvaluationEvent`, `EvaluationSplitEvent`, `RichDashboardRenderer.render`, `SequenceScore`, `StateStore`, `TelemetryReporter`。

## `src/traning/tests/integration/test_phase11_cli.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.app.cli`, `traning.training`

- `F L20-L36` `test_config_check_loads_the_formal_v2_config() -> None`：工程默认 V2 配置必须能从真实 CLI 边界严格加载。
- `F L39-L53` `test_coordinate_audit_reports_validation_only_provenance() -> None`：默认审计通过控制点，但必须公开原拟合集不可重放。
- `F L56-L72` `test_coordinate_audit_can_require_missing_refit_provenance() -> None`：调用方要求完整拟合复现时，validation-only 证据必须非零退出。
- `F L75-L84` `test_config_check_rejects_missing_file() -> None`：配置不存在时 CLI 必须失败，不能退回隐式默认值。
- `F L87-L102` `test_config_check_rejects_legacy_candidate_cache_schema(tmp_path) -> None` [IO-W]：CLI 不得将无坐标指纹的 cache schema 1 静默迁移到 2。
- `F L105-L117` `test_env_check_can_report_without_hiding_failure(monkeypatch) -> None`：非 strict 模式仍输出 CUDA 真实状态，只是不改变进程退出码。
- `F L120-L146` `test_repository_start_entry_exposes_model_diagnostics_without_namespace() -> None` [PROCESS]：总入口直接暴露当前模型诊断，不再保留 v2 兼容命名空间。 调用：`subprocess.run`。
- `F L149-L194` `test_train_cli_uses_production_trainer_without_external_evaluator(tmp_path: Path, monkeypatch) -> None`：train 命令只装配内建生产服务，不接受 module:factory。 调用：`ParameterVector`。

## `src/traning/tests/integration/test_phase11_configured_search.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.app`, `traning.config`, `traning.contracts`, `traning.data`, `traning.telemetry`, `traning.training`

- `C L31-L59` `_PassesOnThirdTrial` [CLASS]：记录实际执行次数，并仅让第三组参数全门禁通过。
- `M L34-L35` `_PassesOnThirdTrial.__init__(self) -> None`：初始化实例依赖、配置和运行状态。
- `M L37-L59` `_PassesOnThirdTrial.evaluate(self, parameters: ParameterVector, trial_index: int) -> TrialObservation`：返回与 proposal 身份一致的确定性 trial 观测。 调用：`TrialAcceptance`, `TrialObservation`, `self.parameters.append`。
- `C L62-L91` `_ConcreteStageRunner` [CLASS]：用 typed StageResult 表示普通训练失败，而不是抛异常结束搜索。
- `M L65-L67` `_ConcreteStageRunner.__init__(self, trial_index: int) -> None`：初始化实例依赖、配置和运行状态。
- `M L69-L91` `_ConcreteStageRunner.run(self, stage: TrainingStage) -> StageResult`：第一轮早停、第二轮 gate 未过、第三轮完整通过。 调用：`StageResult`, `TrialAcceptance`, `self.calls.append`。
- `F L94-L104` `test_default_config_executes_third_trial_instead_of_stopping_at_two() -> None`：直接固定用户旧运行的提前停止回归。 调用：`V2Config`, `_PassesOnThirdTrial`, `run_configured_search`。
- `F L107-L132` `test_concrete_orchestrated_evaluator_continues_after_stage_failures() -> None`：真实阶段 FAILED 与最终 gate 未过都应继续提案，直到第三轮全通过。 调用：`DataQualityReport`, `OrchestratedTrialEvaluator`, `V2Config`, `run_configured_search`。
- `N L112-L120` `test_concrete_orchestrated_evaluator_continues_after_stage_failures.build_runner(_parameters: ParameterVector, trial_index: int) -> _ConcreteStageRunner`：记录每个 proposal 获得的独立有状态 runner。 调用：`_ConcreteStageRunner`。
- `F L135-L175` `test_blocking_data_quality_stops_search_before_constructing_runner(tmp_path: Path) -> None`：固定坏数据不是参数失败；无预算搜索也必须首轮立即阻断。 调用：`DataQualityIssue`, `DataQualityReport`, `OrchestratedTrialEvaluator`, `StateStore`, `TelemetryReporter`, `V2Config`。
- `N L142-L150` `test_blocking_data_quality_stops_search_before_constructing_runner.build_runner(_parameters: ParameterVector, _trial_index: int) -> _ConcreteStageRunner`：若质量门正确前置，本工厂永远不应被调用。 调用：`_ConcreteStageRunner`。
- `F L178-L187` `test_explicit_trial_budget_is_reported_as_exhausted_not_success() -> None`：用户显式预算为 2 时必须产生 typed EXHAUSTED。 调用：`OptimizationConfig`, `V2Config`, `_PassesOnThirdTrial`, `run_configured_search`。
- `F L190-L201` `test_initial_vector_is_collectively_derived_from_domain_config() -> None`：初始 proposal 不再由零散 job 字典逐字段改写。 调用：`V2Config`, `initial_parameter_vector`。
- `F L204-L221` `test_configured_search_publishes_every_trial_and_explicit_terminal(tmp_path: Path) -> None`：Dashboard 看到真实 trial 历史和 PASSED，而非停留在运行中。 调用：`StateStore`, `TelemetryReporter`, `V2Config`, `_PassesOnThirdTrial`, `run_configured_search`, `store.history`。

## `src/traning/tests/integration/test_phase11_coordinate_flow.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package`, `traning.config`, `traning.contracts`, `traning.data.coordinates`, `traning.evaluation`, `traning.outcome.dataset`, `traning.outcome.oracle`, `traning.perception`, `traning.visualization`

- `F L78-L81` `shared_transform() -> AffineOsuVideoTransform`：使用 ``package`` 公开类构造唯一共享的生产变换对象。
- `F L85-L93` `adapter(shared_transform: AffineOsuVideoTransform) -> FrameCoordinateTransform`：将共享变换显式绑定到标定原帧及其身份。 调用：`FrameCoordinateTransform`。
- `F L96-L165` `test_three_consumers_share_one_transform_and_inverse(adapter: FrameCoordinateTransform, shared_transform: AffineOsuVideoTransform) -> None`：真实训练 target、sequence scoring 和 gallery API 必须共用变换。 调用：`FramePredictedClick`, `GroundTruthObject`, `OsuPoint`, `Point2D`, `TargetObject`, `TrainingSample`。
- `F L168-L207` `test_slider_direction_preserves_legal_out_of_playfield_control_point(adapter: FrameCoordinateTransform) -> None`：slider 控制点越界时必须变换向量，不能裁剪或拒绝真实标注。 调用：`GroundTruthObject`, `Point2D`, `TrainingSample`, `build_coordinate_training_targets`。
- `F L210-L311` `test_pass_sample_is_rasterized_with_decoder_inverse_equation(adapter: FrameCoordinateTransform) -> None`：frame 36 必须通过正式 target 栅格化与 decoder 精确回到同一原帧点。 调用：`DensePerceptionOutput`, `GroundTruthObject`, `OsuPoint`, `PerceptionConfig`, `Point2D`, `TrainingSample`。
- `F L314-L332` `test_pass_sample_control_residuals_stay_within_four_pixels(adapter: FrameCoordinateTransform) -> None`：大量 pass 样本拟合后的五个独立控制点均不得偏移。 调用：`adapter.ground_truth_to_training_target`。
- `F L335-L383` `test_frame_score_event_and_real_png_keep_one_transform_identity(adapter: FrameCoordinateTransform, tmp_path: Path) -> None` [IO-W]：评分、归因、gallery 点位和真实 PNG 必须保留同一标定指纹。 调用：`FramePredictedClick`, `OsuPoint`, `RuntimeFrame`, `TargetObject`, `adapter.ground_truth_to_training_target`, `build_gallery_frame_overlay`。
- `F L386-L412` `test_frame_105_unresolved_stays_decision_with_coordinate_provenance(adapter: FrameCoordinateTransform) -> None`：图上存在准确 GT 但无实际 click 时，frame 105 只能归入 Decision。 调用：`TargetObject`, `build_gallery_frame_overlay`, `build_sequence_evaluation_events`, `score_frame_click_sequence`。
- `F L415-L449` `test_frame_margin_prediction_is_scored_as_spatial_miss(adapter: FrameCoordinateTransform) -> None`：映射到 playfield 外的合法原帧点击应计 miss，而不是中止整段评估。 调用：`FramePixelPoint`, `FramePredictedClick`, `TargetObject`, `score_frame_click_sequence`。
- `F L452-L509` `test_counterfactual_labels_inverse_frame_belief_before_oracle(adapter: FrameCoordinateTransform) -> None`：Outcome dataset 不得把原帧 belief 像素直接与 osu! oracle target 相减。 调用：`BeliefState`, `CounterfactualFrame`, `CounterfactualOutcomeDatasetBuilder`, `CounterfactualOutcomeDatasetBuilder.build`, `ObjectTypeDistribution`, `OracleState`。
- `F L512-L540` `test_fingerprint_binds_matrix_identity_and_source_frame(adapter: FrameCoordinateTransform, shared_transform: AffineOsuVideoTransform) -> None`：指纹可比较，且尺寸或标定身份变化时不得复用。 调用：`FrameCoordinateTransform`。
- `F L543-L579` `test_mismatched_frame_size_is_rejected_by_every_consumer(adapter: FrameCoordinateTransform) -> None`：三个消费者都必须对错误原帧尺寸硬失败。 调用：`FramePixelPoint`, `OsuPoint`, `adapter.ground_truth_to_training_target`, `adapter.prediction_to_canonical_scoring`, `adapter.target_to_gallery_overlay`。
- `F L582-L622` `test_nonfinite_out_of_bounds_and_centered_fallback_are_rejected(shared_transform: AffineOsuVideoTransform) -> None`：非法点和未标定居中变换不得被 clamp 或静默接受。 调用：`FrameCoordinateTransform`, `FramePixelPoint`, `OsuPoint`。

## `src/traning/tests/integration/test_phase11_environment.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.app.environment`, `traning.config`

- `F L21-L41` `test_cpu_environment_with_explicit_coordinates_passes(monkeypatch) -> None`：CPU smoke 不得因 sandbox 看不到 CUDA 而失败。 调用：`CoordinateConfig`, `RuntimeConfig`, `V2Config`, `require_v2_environment`。
- `F L44-L61` `test_required_cuda_unavailable_is_a_typed_failure(monkeypatch) -> None`：设备不可见必须报告真实失败，不能静默退回 CPU。 调用：`CoordinateConfig`, `V2Config`, `check_v2_environment`, `require_v2_environment`。
- `F L64-L72` `test_missing_coordinate_calibration_blocks_formal_environment(monkeypatch) -> None`：正式训练/评分不得使用 centered transform 静默兜底。 调用：`V2Config`, `check_v2_environment`。
- `F L75-L88` `test_formal_config_exposes_validation_only_coordinate_warning(monkeypatch) -> None`：控制点可通过启动门，但缺失的原拟合集必须保持可见 warning。 调用：`check_v2_environment`, `load_v2_config`。

## `src/traning/tests/integration/test_phase11_factory.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.app.factory`, `traning.app.runtime`, `traning.belief`, `traning.config`, `traning.data`, `traning.outcome`, `traning.perception`, `traning.training`

- `F L30-L53` `_cpu_config() -> V2Config`：构造尺寸较小但领域契约完整的 CPU smoke 配置。 调用：`BeliefConfig`, `CoordinateConfig`, `DecisionConfig`, `OutcomeConfig`, `PerceptionConfig`, `RuntimeConfig`。
- `F L56-L70` `test_untrained_builder_is_explicit_and_coordinate_config_is_shared() -> None`：smoke factory 可运行，但正式坐标仍来自同一显式 config。 调用：`OsuPoint`, `_cpu_config`, `build_frame_coordinate_transform`, `build_untrained_runtime_for_smoke`, `coordinates.ground_truth_to_training_target`。
- `F L73-L111` `test_assembly_rejects_model_config_drift() -> None`：checkpoint 模型与启动配置不一致时不得静默装配。 调用：`DenseOutcomeModel`, `PerTrackBeliefEncoder`, `PerceptionConfig`, `PerceptionModel`, `RuntimeModelBundle`, `_cpu_config`。
- `F L114-L131` `test_assembly_rejects_checkpoint_coordinate_drift() -> None`：相同网络结构但由旧坐标系训练的权重 bundle 必须在装配前拒绝。 调用：`DenseOutcomeModel`, `PerTrackBeliefEncoder`, `PerceptionModel`, `RuntimeModelBundle`, `_cpu_config`, `assemble_runtime_pipeline`。
- `F L134-L146` `test_missing_affine_calibration_is_not_centered_fallback() -> None`：默认无矩阵配置必须硬失败，不猜测 playfield 居中矩形。 调用：`RuntimeConfig`, `V2Config`, `build_frame_coordinate_transform`。

## `src/traning/tests/integration/test_phase11_runtime_pipeline.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package`, `traning.app`, `traning.belief`, `traning.config`, `traning.contracts`, `traning.data`, `traning.decision`, `traning.outcome`, `traning.perception`, `traning.tracking`

- `C L39-L98` `_PixelControlledPerceptionModel` [CLASS]：用首像素选择空帧、单目标或双目标稠密预测。
- `M L42-L43` `_PixelControlledPerceptionModel.__init__(self) -> None`：初始化实例依赖、配置和运行状态。
- `M L45-L98` `_PixelControlledPerceptionModel.__call__(self, image: torch.Tensor) -> DensePerceptionOutput`：执行 `call` 对应逻辑。 调用：`DensePerceptionOutput`。
- `C L101-L111` `_FailingOutcomeModel(DenseOutcomeModel)` [CLASS]：在 stateful 边界之后模拟不可恢复的预测异常。
- `M L104-L111` `_FailingOutcomeModel.predict(self, belief: BeliefState, horizon_ms: float) -> OutcomeDistribution`：稳定抛出预测异常，以验证 runtime 的失败锁存边界。
- `F L114-L125` `_frame(index: int, marker: int) -> RuntimeFrame`：构造具有真实 RGB 字节长度的运行时帧。 调用：`RuntimeFrame`。
- `F L128-L150` `_outcome_model(belief_dim: int, *, future_is_better: bool) -> DenseOutcomeModel`：构造仅由 horizon 控制、可精确触发 CLICK 或 WAIT 的 dense 模型。 调用：`DenseOutcomeModel`, `OutcomeConfig`。
- `F L153-L231` `_pipeline(*, future_is_better: bool, max_missed_frames: int=1, fail_predictions: bool=False) -> tuple[V2RuntimePipeline, _PixelControlledPerceptionModel, MultiObjectTracker, PerTrackBeliefRuntime]`：组装完整的正式层级，并返回可观测的有状态组件。 调用：`BeliefConfig`, `DecisionConfig`, `FrameCoordinateTransform`, `MultiObjectTracker`, `OptimalStoppingPlanner`, `OutcomeConfig`。
- `F L234-L269` `test_multi_object_pipeline_is_stable_when_perception_score_order_swaps() -> None`：分数排序交换不得改变目标身份、输出顺序或正式 CLICK 绑定。 调用：`_frame`, `_pipeline`, `pipeline.step`。
- `F L272-L283` `test_future_value_produces_explicit_wait() -> None`：相同正式链路在未来收益更高时必须输出 WAIT 而非当前点击。 调用：`_frame`, `_pipeline`, `pipeline.step`。
- `F L286-L312` `test_empty_frame_context_expires_track_and_excludes_expired_belief() -> None`：空候选仍推进帧；EXPIRED 仅留审计 track，不得进入预测或 Decision。 调用：`_frame`, `_pipeline`, `belief_runtime.snapshot`, `pipeline.step`, `tracker.snapshot`。
- `F L315-L348` `test_frame_validation_precedes_components_and_reset_restarts_identity() -> None`：重复帧在感知前拒绝；reset 后帧游标、track ID 与 belief 一起重启。 调用：`RuntimeFrame`, `_frame`, `_pipeline`, `belief_runtime.snapshot`, `pipeline.reset`, `pipeline.step`。
- `F L351-L373` `test_stateful_failure_latches_pipeline_until_reset() -> None`：有状态边界后的异常不得允许调用方在可能不一致的状态上继续。 调用：`_frame`, `_pipeline`, `belief_runtime.snapshot`, `pipeline.reset`, `pipeline.step`, `tracker.snapshot`。
- `F L376-L404` `test_runtime_source_has_no_training_only_or_shortcut_dependency() -> None` [IO-R]：静态阻止训练信息、旧动作捷径和稀疏实验实现进入正式 app runtime。

## `src/traning/tests/regression/test_legacy_golden_baseline.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.config`, `traning.evaluation`, `traning.perception`

- `F L30-L33` `_load_fixture() -> dict[str, object]` [IO-R]：在 JSON 边界读取固定期望值；生产领域接口不会复用这个宽松类型。
- `F L36-L39` `_logit(probability: float) -> float`：把 legacy 概率反解为新 decoder 消费的有限 logit。
- `F L42-L88` `_build_perception_output() -> DensePerceptionOutput`：构造两个带亚像素偏移的确定性峰值，避免模型权重影响基线。 调用：`DensePerceptionOutput`, `_logit`。
- `F L91-L98` `test_legacy_archive_is_frozen() -> None` [IO-R]：检测 legacy 冻结包被替换或意外重写。 调用：`hashlib.sha256`。
- `F L101-L127` `test_legacy_candidate_geometry_and_recall_match_golden() -> None`：固定空间解码的坐标公式、排序和基础 recall。 调用：`PerceptionConfig`, `_build_perception_output`, `_load_fixture`, `decode_candidates`。
- `F L130-L176` `test_legacy_oracle_matches_golden() -> None`：固定点、slider 头部及路径的连续评分语义。 调用：`_load_fixture`, `score_point`, `score_slider`。
- `F L179-L201` `test_legacy_sequence_score_matches_golden() -> None`：固定频率限制、目标消费和最终序列计数。 调用：`PredictedClick`, `TargetObject`, `_load_fixture`, `score_click_sequence`。

## `src/traning/tests/unit/test_phase10_reporter.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts.telemetry`, `traning.evaluation.attribution`, `traning.telemetry.events`, `traning.telemetry.reporter`, `traning.telemetry.store`

- `F L36-L48` `_canonical_evaluation() -> SequenceEvaluationEvent`：构造 frame 105 同型的零点击未解析 decision 事件。 调用：`SequenceEvaluationEvent`。
- `F L51-L67` `_metrics(run_id: str='run-1', timestamp_ms: float=10.0) -> MetricsEvent`：执行 `metrics` 对应逻辑。 调用：`MetricsEvent`。
- `F L70-L80` `_resources(run_id: str='run-1') -> ResourceEvent`：执行 `resources` 对应逻辑。 调用：`ResourceEvent`。
- `F L83-L112` `test_reporter_publishes_all_required_metrics_and_resources(tmp_path: Path) -> None`：reporter 必须完整投影指标和资源事件的全部字段。 调用：`DashboardMetrics`, `DashboardResources`, `StateStore`, `TelemetryReporter`, `_metrics`, `_resources`。
- `F L115-L130` `test_canonical_evaluation_object_and_semantics_are_preserved(tmp_path: Path) -> None`：评估事件进入 dashboard 后必须保持对象身份与判定语义。 调用：`EvaluationEvent`, `StateStore`, `TelemetryReporter`, `_canonical_evaluation`, `reporter.publish`, `reporter.snapshot`。
- `F L133-L145` `test_reporter_has_no_side_cache_and_projects_store_latest_state(tmp_path: Path) -> None`：reporter 不得维护旁路缓存，并只投影 store 的最新状态。 调用：`DashboardSnapshot`, `StateStore`, `TelemetryReporter`, `_metrics`, `reporter.snapshot`, `store.publish`。
- `F L148-L173` `test_generic_quality_event_is_not_reinterpreted_by_dashboard(tmp_path: Path) -> None`：dashboard 不得擅自重解释通用质量事件。 调用：`EvaluationEvent`, `StateStore`, `TelemetryEvent`, `TelemetryReporter`, `_canonical_evaluation`, `reporter.publish`。
- `F L176-L183` `test_reporter_rejects_cross_run_events_before_publication(tmp_path: Path) -> None`：不同 run 的事件必须在持久化前被 reporter 拒绝。 调用：`StateStore`, `TelemetryReporter`, `_metrics`, `reporter.publish`, `store.history`。
- `F L186-L227` `test_dashboard_contracts_are_frozen_and_cover_required_fields() -> None` [IO-W]：dashboard 契约必须不可变且覆盖规定字段。 调用：`DashboardMetrics`, `DashboardResources`, `DashboardSnapshot`。
- `F L230-L242` `test_reporter_source_never_recomputes_quality_or_evaluation() -> None` [IO-R]：Reporter 不得读 passed/primary_error/blocks_training 发明新语义。

## `src/traning/tests/unit/test_phase10_telemetry_store.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts`, `traning.evaluation`, `traning.infrastructure.errors`, `traning.telemetry.events`, `traning.telemetry.store`

- `F L37-L53` `_metrics(step: int=1) -> MetricsEvent`：执行 `metrics` 对应逻辑。 调用：`MetricsEvent`。
- `F L56-L66` `_resources(step: int=1) -> ResourceEvent`：执行 `resources` 对应逻辑。 调用：`ResourceEvent`。
- `F L69-L81` `_unresolved_evaluation() -> EvaluationEvent`：执行 `unresolved evaluation` 对应逻辑。 调用：`EvaluationEvent`, `SequenceScore`, `build_sequence_evaluation_events`。
- `F L84-L93` `_lifecycle(payload: object | None=None) -> TelemetryEvent`：执行 `lifecycle` 对应逻辑。 调用：`TelemetryEvent`。
- `F L96-L97` `_read_lines(path: Path) -> list[dict[str, object]]` [IO-R]：读取 `lines` 对应的数据或结果。
- `F L100-L129` `test_four_channels_persist_versioned_strict_json_and_recover(tmp_path: Path) -> None` [IO-R]：四类通道必须用版本化严格 JSON 持久化并可恢复。 调用：`StateStore`, `_lifecycle`, `_metrics`, `_read_lines`, `_resources`, `_unresolved_evaluation`。
- `F L132-L157` `test_snapshot_is_frozen_copy_and_keeps_canonical_event_identity(tmp_path: Path) -> None`：快照必须不可变，并保留进程内 canonical 事件对象身份。 调用：`StateStore`, `_lifecycle`, `_unresolved_evaluation`, `store.publish`, `store.snapshot`。
- `F L160-L175` `test_evaluation_disk_roundtrip_preserves_pass_and_error_semantics(tmp_path: Path) -> None`：评估事件磁盘往返不得改变通过状态或错误归因。 调用：`StateStore`, `StateStore.snapshot`, `_unresolved_evaluation`, `store.publish`。
- `F L178-L195` `test_concurrent_publish_keeps_every_complete_record(tmp_path: Path) -> None`：并发发布必须保留每条完整记录且不能发生交错损坏。 调用：`StateStore`, `StateStore.history`, `_metrics`, `_read_lines`, `store.history`, `store.publish`。
- `F L198-L219` `test_publish_calls_fsync_and_rejects_unknown_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None`：发布必须执行 fsync，并拒绝未注册事件类型。 调用：`StateStore`, `_metrics`, `store.publish`。
- `N L209-L213` `test_publish_calls_fsync_and_rejects_unknown_event.observe_fsync(descriptor: int) -> None`：记录 fsync 调用后委托给真实实现。
- `F L231-L251` `test_metrics_reject_bad_values(field: str, value: object, message: str) -> None`：指标契约必须拒绝非法数值与错误类型。 调用：`MetricsEvent`。
- `F L254-L269` `test_store_rejects_partial_channel_set_and_corrupt_json(tmp_path: Path) -> None` [IO-W]：恢复时必须拒绝残缺通道集合与损坏 JSON。 调用：`StateStore`。
- `F L272-L289` `test_store_rejects_wrong_channel_schema_and_truncated_tail(tmp_path: Path) -> None` [IO-W]：恢复时必须拒绝错误通道 schema 与截断尾记录。 调用：`StateStore`。
- `F L292-L311` `test_store_rejects_path_and_schema_mismatch(tmp_path: Path) -> None`：事件通道路径和声明 schema 不一致时必须硬失败。 调用：`StateStore`, `TelemetryEvent`, `store.publish`。
- `F L314-L349` `test_store_binds_first_run_and_recovery_rejects_mixed_runs(tmp_path: Path) -> None` [IO-W]：store 必须绑定首个 run，并拒绝混入其他 run 的历史。 调用：`MetricsEvent`, `StateStore`, `_metrics`, `_read_lines`, `_resources`, `store.history`。

## `src/traning/tests/unit/test_phase10_visualization.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.evaluation.attribution`, `traning.telemetry.reporter`, `traning.visualization`

- `F L35-L47` `_failed_evaluation() -> SequenceEvaluationEvent`：复现 frame 105 的 canonical decision/unresolved 语义。 调用：`SequenceEvaluationEvent`。
- `F L50-L79` `_snapshot(evaluation: SequenceEvaluationEvent | None=None) -> DashboardSnapshot`：构造覆盖 Phase 10 所有必需指标的不可变快照。 调用：`DashboardMetrics`, `DashboardResources`, `DashboardSnapshot`。
- `F L82-L86` `_rich_rows(snapshot: DashboardSnapshot) -> tuple[DashboardMetricRow, ...]`：按 Rich 模型中的固定分区顺序展开指标行。 调用：`RichDashboardRenderer.render`。
- `F L89-L123` `test_rich_and_qt_show_the_complete_metric_registry_in_stable_order() -> None`：Rich 与 Qt 必须按稳定顺序展示完整指标注册表。 调用：`QtDashboardRenderer.render`, `RichDashboardRenderer.render`, `_snapshot`。
- `F L126-L143` `test_evaluation_projection_preserves_identity_pass_and_primary_error() -> None`：评估投影必须保留对象身份、通过状态和主错误。 调用：`QtDashboardRenderer.render`, `RichDashboardRenderer.render`, `_failed_evaluation`, `_snapshot`。
- `F L146-L166` `test_passed_evaluation_is_not_reinterpreted_as_an_error() -> None`：已通过的评估不得被可视化层重新解释为错误。 调用：`RichDashboardRenderer.render`, `SequenceEvaluationEvent`, `_snapshot`。
- `F L169-L186` `test_renderers_are_deterministic_and_models_are_frozen() -> None`：渲染输出必须确定，且视图模型必须保持不可变。 调用：`QtDashboardRenderer.render`, `RichDashboardRenderer.render`, `_failed_evaluation`, `_rich_rows`, `_snapshot`。
- `F L189-L205` `test_missing_optional_telemetry_keeps_all_slots_without_invention() -> None`：缺少可选遥测时必须保留槽位且不能虚构数据。 调用：`DashboardSnapshot`, `QtDashboardRenderer.render`, `RichDashboardRenderer.render`。
- `F L208-L214` `test_renderer_rejects_mapping_instead_of_accepting_mutable_live_state() -> None`：渲染器必须拒绝可变映射，只接受冻结快照。 调用：`QtDashboardRenderer.render`, `RichDashboardRenderer.render`。
- `F L217-L252` `test_renderer_source_has_no_gui_store_io_or_semantic_side_channel() -> None` [IO-R]：Renderer 只能读 reporter snapshot，不得接触 store、scorer 或质量门禁。

## `src/traning/tests/unit/test_phase11_checkpoints.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.app.factory`, `traning.belief`, `traning.config`, `traning.infrastructure`, `traning.outcome`, `traning.perception`, `traning.training`

- `F L34-L57` `_config() -> V2Config`：构造可快速保存与恢复的小型 CPU 模型配置。 调用：`BeliefConfig`, `CoordinateConfig`, `DecisionConfig`, `OutcomeConfig`, `PerceptionConfig`, `RuntimeConfig`。
- `F L60-L78` `_models(config: V2Config) -> RuntimeModelBundle`：构造与配置和坐标指纹一致的三模型 bundle。 调用：`DenseOutcomeModel`, `PerTrackBeliefEncoder`, `PerceptionModel`, `RuntimeModelBundle`, `build_frame_coordinate_transform`。
- `F L81-L122` `test_checkpoint_roundtrip_preserves_models_and_coordinate_identity(tmp_path: Path) -> None`：manifest、权重和坐标身份全通过后才返回可装配模型 bundle。 调用：`_config`, `_models`, `build_frame_coordinate_transform`, `load_runtime_checkpoint`, `publish_runtime_checkpoint`。
- `F L125-L148` `test_checkpoint_rejects_weight_corruption(tmp_path: Path) -> None` [IO-R IO-W]：权重 generation 被修改后必须在 torch 解码前由 SHA-256 拒绝。 调用：`_config`, `_models`, `build_frame_coordinate_transform`, `load_runtime_checkpoint`, `publish_runtime_checkpoint`。
- `F L151-L181` `test_checkpoint_rejects_old_or_changed_coordinate_fingerprint(tmp_path: Path) -> None` [IO-R IO-W]：结构相同的旧坐标权重也不得绕过 manifest 指纹门禁。 调用：`_config`, `_models`, `build_frame_coordinate_transform`, `load_runtime_checkpoint`, `publish_runtime_checkpoint`。
- `F L184-L209` `test_checkpoint_manifest_requires_coordinate_provenance(tmp_path: Path) -> None` [IO-R IO-W]：缺少变换指纹的旧 manifest schema 不能被当作新 checkpoint 加载。 调用：`_config`, `_models`, `build_frame_coordinate_transform`, `load_runtime_checkpoint`, `publish_runtime_checkpoint`。
- `F L212-L233` `test_checkpoint_rejects_wrong_dataset_identity(tmp_path: Path) -> None`：权重结构相同也不能绕过训练数据集身份门禁。 调用：`_config`, `_models`, `build_frame_coordinate_transform`, `load_runtime_checkpoint`, `publish_runtime_checkpoint`。
- `F L236-L272` `test_checkpoint_gates_model_contract_not_deployment_paths(tmp_path: Path) -> None` [IO-W]：部署目录可调整，但任一三模型结构字段变化都必须拒绝。 调用：`_config`, `_models`, `build_frame_coordinate_transform`, `load_runtime_checkpoint`, `publish_runtime_checkpoint`。
- `F L276-L295` `test_checkpoint_rejects_non_finite_timestamp_before_writing(tmp_path: Path, timestamp: float) -> None`：非有限发布时间必须在创建任何 generation 前硬失败。 调用：`_config`, `_models`, `build_frame_coordinate_transform`, `publish_runtime_checkpoint`。

## `src/traning/tests/unit/test_phase11_coordinate_calibration.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.app`, `traning.config`, `traning.data`, `traning.infrastructure`

- `F L26-L41` `_observation(identifier: str, osu_x: float, osu_y: float, matrix: tuple[tuple[float, float, float], tuple[float, float, float]]) -> CalibrationObservation`：按指定精确方程生成无噪声测试观测。 调用：`CalibrationObservation`。
- `F L44-L62` `test_checked_in_coordinate_evidence_validates_but_does_not_claim_refit() -> None`：五个控制点通过不等于原始 passed 拟合集可重放。 调用：`audit_affine_calibration`, `build_frame_coordinate_transform`, `load_affine_calibration_evidence`, `load_v2_config`。
- `F L65-L84` `test_collective_least_squares_fit_is_order_independent() -> None`：未来完整观测集可由统一方程循环拟合，输入顺序不改变矩阵或摘要。 调用：`_observation`, `fit_affine_least_squares`。
- `F L87-L107` `test_coordinate_evidence_rejects_unknown_or_drifting_schema(tmp_path: Path) -> None` [IO-R IO-W]：证据未知字段与方程漂移不能被宽松加载或当成已验证配置。 调用：`audit_affine_calibration`, `build_frame_coordinate_transform`, `load_affine_calibration_evidence`, `load_v2_config`。
- `F L110-L120` `test_affine_fit_rejects_collinear_observations() -> None`：共同样本没有二维覆盖时不得发布不可逆或欠定方程。 调用：`_observation`, `fit_affine_least_squares`。

## `src/traning/tests/unit/test_phase1_config.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.config`, `traning.data.cache`

- `F L22-L30` `test_default_config_round_trips_through_json(tmp_path) -> None` [IO-W]：默认配置经过 JSON 边界后保持同一 typed config。 调用：`V2Config`, `load_v2_config`, `v2_config_to_dict`。
- `F L33-L43` `test_cache_config_uses_current_candidate_artifact_schema() -> None`：配置边界必须与候选缓存制品 schema 保持同一版本。 调用：`CacheConfig`, `V2Config`, `v2_config_to_dict`。
- `F L46-L50` `test_legacy_candidate_cache_schema_is_rejected() -> None`：schema 1 缺少坐标变换指纹，不得被默认升级或接受。 调用：`load_v2_config`。
- `F L53-L57` `test_unknown_top_level_config_key_is_rejected() -> None`：拼错字段不能被静默忽略。 调用：`load_v2_config`。
- `F L60-L69` `test_unknown_nested_config_key_is_rejected() -> None`：嵌套配置也遵循同一个 strict schema。 调用：`load_v2_config`。
- `F L72-L76` `test_unsupported_config_schema_is_rejected() -> None`：版本不兼容时硬失败，不使用旧默认值掩盖问题。 调用：`load_v2_config`。
- `F L79-L83` `test_outcome_category_count_is_the_canonical_five() -> None`：配置不得让模型输出通道与 canonical OutcomeCategory 分叉。 调用：`OutcomeConfig`。
- `F L86-L99` `test_optimization_default_is_unbounded_and_round_trips() -> None`：默认不得复现 legacy max_trials=2 导致的提前终止。 调用：`OptimizationConfig`, `V2Config`, `load_v2_config`, `v2_config_to_dict`。
- `F L103-L107` `test_optimization_trial_limit_is_strict(value: object) -> None`：非法预算不能被静默转换成会提前停止的整数。 调用：`load_v2_config`。
- `F L110-L133` `test_coordinate_affine_matrix_is_versioned_and_round_trips() -> None`：坐标方程必须与原帧尺寸一同进入单一 V2 config。 调用：`CoordinateConfig`, `V2Config`, `load_v2_config`, `v2_config_to_dict`。
- `F L145-L151` `test_coordinate_affine_matrix_rejects_bad_shape_or_values(matrix: object) -> None`：损坏或不可逆的坐标方程不得退回 centered transform。 调用：`load_v2_config`。

## `src/traning/tests/unit/test_phase1_contracts.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package`, `traning.contracts`

- `F L48-L63` `test_data_split_has_one_canonical_vocabulary() -> None`：固定跨数据层唯一允许的 split 值。
- `F L67-L73` `test_runtime_contracts_structurally_exclude_gt(contract_type: type[object]) -> None`：从 dataclass schema 层阻止 runtime 获得任何 GT-only 字段。
- `F L76-L88` `test_runtime_instance_cannot_gain_gt_attribute() -> None`：slots + frozen 使推理对象无法在运行时偷偷补入 GT。 调用：`RuntimeFrame`。
- `F L91-L122` `test_training_sample_requires_coordinate_transform_fingerprint() -> None`：训练 target 必须显式绑定生成它的坐标变换。 调用：`TrainingSample`。
- `F L125-L140` `test_outcome_distribution_rejects_invalid_probability_mass() -> None`：Outcome 五类概率必须形成一个规范化离散分布。 调用：`OutcomeDistribution`。
- `F L143-L154` `test_wait_decision_cannot_select_a_track() -> None`：WAIT 与 CLICK 的目标约束在 contract 构造时确定。 调用：`DecisionResult`。
- `F L157-L168` `test_phase1_core_does_not_import_typing_any() -> None` [IO-R]：在核心长期接口中禁止重新引入宽泛 Any。

## `src/traning/tests/unit/test_phase1_infrastructure.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.infrastructure`

- `F L27-L45` `test_atomic_writers_publish_complete_payloads(tmp_path) -> None` [IO-R]：四种发布入口都只暴露完整最终文件。 调用：`atomic_write_bytes`, `atomic_write_json`, `atomic_write_jsonl`, `atomic_write_text`, `hashlib.sha256`, `read_json_object`。
- `F L48-L68` `test_failed_replace_preserves_previous_file_and_cleans_temp(tmp_path, monkeypatch) -> None` [IO-R IO-W]：发布失败不能破坏旧版本，也不能遗留半成品。 调用：`atomic_write_text`。
- `N L56-L62` `test_failed_replace_preserves_previous_file_and_cleans_temp.fail_replace(source: os.PathLike[str] | str, destination: os.PathLike[str] | str) -> None`：模拟原子替换失败，以验证旧文件仍可恢复。
- `F L71-L77` `test_read_json_object_rejects_non_object_root(tmp_path) -> None` [IO-W]：对象 schema 不接受 JSON array 的 silent coercion。 调用：`read_json_object`。
- `F L80-L87` `test_seed_everything_repeats_python_numpy_and_torch() -> None`：数据处理使用同一 seed 时得到相同随机序列。 调用：`seed_everything`。

## `src/traning/tests/unit/test_phase2_cache.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts`, `traning.data.cache`, `traning.data.cache.cache`, `traning.infrastructure`

- `F L35-L53` `_record(candidate_id: str, *, x: float=32.0) -> InferenceCandidateRecord`：执行 `record` 对应逻辑。 调用：`CandidateObservation`, `InferenceCandidateRecord`, `ObjectTypeDistribution`。
- `F L56-L67` `_records_path(cache_dir: Path) -> Path`：执行 `records path` 对应逻辑。 调用：`read_json_object`。
- `F L70-L101` `test_candidate_cache_round_trip_and_runtime_schema(tmp_path) -> None` [IO-R]：发布后恢复相同 typed record，持久化文本也不含 GT-only 字段。 调用：`_record`, `_records_path`, `load_candidate_cache`, `publish_candidate_cache`。
- `F L104-L147` `test_candidate_cache_rejects_checksum_and_row_count_mismatch(tmp_path) -> None` [IO-R IO-W]：内容篡改和 manifest 行数谎报分别被拒绝。 调用：`_record`, `_records_path`, `atomic_write_json`, `load_candidate_cache`, `publish_candidate_cache`, `read_json_object`。
- `F L150-L194` `test_candidate_cache_rejects_identity_and_gt_field_injection(tmp_path) -> None` [IO-R]：调用方身份不匹配和持久化 GT 注入都硬失败。 调用：`_record`, `_records_path`, `atomic_write_json`, `atomic_write_jsonl`, `load_candidate_cache`, `publish_candidate_cache`。
- `F L197-L241` `test_candidate_cache_rejects_stale_or_legacy_coordinate_identity(tmp_path) -> None`：新坐标标定不得复用旧指纹或 schema v1 缓存。 调用：`_record`, `atomic_write_json`, `load_candidate_cache`, `publish_candidate_cache`, `read_json_object`。
- `F L244-L259` `test_candidate_cache_rejects_malformed_transform_before_publication(tmp_path) -> None`：非共享坐标 API 指纹在落盘前就必须失败。 调用：`_record`, `publish_candidate_cache`。
- `F L262-L302` `test_failed_manifest_commit_keeps_previous_generation_readable(tmp_path, monkeypatch) -> None`：manifest 提交失败时，旧事务仍完整可读。 调用：`_record`, `load_candidate_cache`, `publish_candidate_cache`。
- `N L279-L283` `test_failed_manifest_commit_keeps_previous_generation_readable.fail_manifest(*args: object, **kwargs: object) -> None`：模拟 manifest 提交失败以检查 generation 原子性。 调用：`AtomicWriteError`。

## `src/traning/tests/unit/test_phase2_quality.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts`, `traning.data`

- `F L18-L19` `_finding(_context: DataQualityContext) -> tuple[DataQualityFinding, ...]`：执行 `finding` 对应逻辑。 调用：`DataQualityFinding`。
- `F L22-L39` `test_info_issue_can_block_training() -> None`：UI severity 不能覆盖领域层 blocks_training。 调用：`DataQualityContext.from_samples`, `DataQualityGate`, `DataQualityRule`, `gate.evaluate`, `require_quality`。
- `F L42-L57` `test_error_issue_can_be_nonblocking() -> None`：ERROR 也不会被 pipeline 擅自解释为 blocking。 调用：`DataQualityContext.from_samples`, `DataQualityGate`, `DataQualityRule`, `gate.evaluate`, `require_quality`。
- `F L60-L68` `test_default_gate_blocks_empty_training_split() -> None`：默认门禁不允许空训练集继续进入训练。 调用：`DataQualityContext.from_samples`, `DataQualityGate`, `DataQualityGate.evaluate`。

## `src/traning/tests/unit/test_phase2_repositories.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.data.repositories`

- `F L20-L30` `_metadata(name: str='item-a') -> PreprocessingMetadata`：执行 `metadata` 对应逻辑。 调用：`PreprocessingMetadata`。
- `F L33-L40` `_catalog(folder: str='folder-a') -> DatasetCatalogEntry`：执行 `catalog` 对应逻辑。 调用：`DatasetCatalogEntry`。
- `F L43-L59` `test_memory_repositories_return_typed_deterministic_snapshots() -> None`：内存实现遵循与持久层相同的稳定领域契约。 调用：`InMemoryDatasetCatalogRepository`, `InMemoryPreprocessingMetadataRepository`, `_catalog`, `_metadata`, `catalog_repo.list_all`, `metadata_repo.delete`。
- `F L62-L79` `test_sqlite_repositories_round_trip_without_exposing_rows(tmp_path) -> None`：SQLite table/column 留在 adapter 内，调用方只见 dataclass。 调用：`SQLiteDatasetCatalogRepository`, `SQLiteDatasetCatalogRepository.create`, `SQLiteDatasetCatalogRepository.get`, `SQLitePreprocessingMetadataRepository`, `SQLitePreprocessingMetadataRepository.create`, `SQLitePreprocessingMetadataRepository.get`。
- `F L82-L97` `test_sqlite_repository_rejects_missing_or_wrong_schema(tmp_path) -> None`：adapter 不猜测或迁移未知 preprocessing 表结构。 调用：`SQLitePreprocessingMetadataRepository`, `SQLitePreprocessingMetadataRepository.create`。

## `src/traning/tests/unit/test_phase3_perception.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.config`, `traning.contracts`, `traning.perception`, `traning.perception.models`

- `F L25-L49` `_dense_output(*, height: int=4, width: int=6, embedding_dim: int=3) -> DensePerceptionOutput`：构造可精确控制峰值与坐标的稠密输出。 调用：`DensePerceptionOutput`。
- `F L52-L65` `_set_ring_peak(output: DensePerceptionOutput, *, row: int, column: int, offset_x: float, offset_y: float) -> None`：只在测试中原地设置一个强 ring 峰值。
- `F L68-L98` `_targets_for(output: DensePerceptionOutput) -> PerceptionTargets`：构造两个显式实例 ID，覆盖所有 dense loss 入口。 调用：`PerceptionTargets`。
- `F L101-L125` `test_unfrozen_global_encoder_receives_end_to_end_gradients() -> None`：global_frozen=False 必须真实改变优化图，而不只是保存配置值。 调用：`PerceptionConfig`, `PerceptionLossWeights`, `PerceptionModel`, `_targets_for`, `compute_perception_loss`。
- `F L128-L146` `test_frozen_global_encoder_has_no_grad_but_local_branch_trains() -> None`：冻结只作用于 global，不得意外冻结整个 Perception。 调用：`PerceptionConfig`, `PerceptionModel`。
- `F L149-L187` `test_decode_uses_one_anisotropic_cell_mapping_equation() -> None`：候选点和 ring 半径均由特征网格统一映到原始帧，而非渲染补丁偏移。 调用：`PerceptionConfig`, `_dense_output`, `_set_ring_peak`, `decode_candidates`。
- `F L190-L209` `test_decode_clamps_extreme_edge_offset_to_pixel_domain() -> None`：最后一个 cell 的 +0.5 offset 不得产生等于 frame size 的越界坐标。 调用：`PerceptionConfig`, `_dense_output`, `_set_ring_peak`, `decode_candidates`。
- `F L212-L242` `test_perception_runtime_accepts_only_runtime_frame_fields() -> None`：正式入口从 RuntimeFrame 到 candidates，全程没有训练 label 参数。 调用：`FixedModel`, `PerceptionConfig`, `PerceptionRuntime`, `PerceptionRuntime.infer`, `RuntimeFrame`, `_dense_output`。
- `C L224-L229` `test_perception_runtime_accepts_only_runtime_frame_fields.FixedModel` [CLASS]：返回固定稠密输出的最小感知测试模型。
- `N L227-L229` `test_perception_runtime_accepts_only_runtime_frame_fields.FixedModel.__call__(self, image: torch.Tensor) -> DensePerceptionOutput`：执行 `call` 对应逻辑。
- `F L245-L267` `test_perception_source_has_no_legacy_or_gt_runtime_dependency() -> None` [IO-R]：静态阻止旧接口或 GT-only 名称重新进入正式 Perception 源码。 调用：`source.split`。
- `F L270-L274` `test_pretrained_global_without_weights_is_rejected() -> None`：禁止随机初始化后冒充 pretrained，或随即被错误冻结。 调用：`PerceptionConfig`, `PerceptionModel`。
- `F L277-L296` `test_spatial_head_never_publishes_zero_identity_or_direction_vectors() -> None`：零输出极端情况也必须提供 tracking 可计算 cosine 的单位向量。 调用：`FusedFeatureOutput`, `SpatialHead`。
- `F L299-L339` `test_identity_loss_pulls_same_instance_across_temporal_batch() -> None`：同一 object_id 在相邻帧的 embedding 不同，必须产生跨帧 pull 损失。 调用：`DensePerceptionOutput`, `PerceptionLossWeights`, `PerceptionTargets`, `compute_perception_loss`。

## `src/traning/tests/unit/test_phase4_tracking.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.config`, `traning.contracts`, `traning.tracking.association`, `traning.tracking.tracker`

- `F L26-L49` `_candidate(frame_index: int, candidate_id: str, *, x: float, embedding: tuple[float, ...], object_type: ObjectTypeDistribution=_RING, timestamp_ms: float | None=None, frame_id: str | None=None) -> CandidateObservation`：构造只含 runtime 可观测字段的候选。 调用：`CandidateObservation`。
- `F L52-L62` `_candidate_mapping(observations: tuple[object, ...]) -> dict[str, str]`：把带 candidate 的跟踪输出投影为 candidate→track 映射。
- `F L65-L106` `test_stable_track_ids_replay_and_input_order_invariance() -> None`：相同观测重放或反转候选输入顺序都不得改变稳定身份。 调用：`_candidate`, `_candidate_mapping`, `replay`。
- `N L83-L90` `test_stable_track_ids_replay_and_input_order_invariance.replay(*, reverse: bool) -> tuple[tuple[object, ...], ...]`：按指定输入次序重放帧序列并收集每帧轨迹。 调用：`MultiObjectTracker`, `TrackingConfig`, `tracker.update`。
- `F L109-L131` `test_crossing_targets_follow_embedding_identity() -> None`：目标交叉时，稳定 ID 应跟随 appearance，而不是输入 slot 或当前位置。 调用：`MultiObjectTracker`, `TrackingConfig`, `_candidate`, `_candidate_mapping`, `tracker.update`。
- `F L134-L184` `test_miss_expire_and_frame_gap_use_successful_update_count() -> None`：missed 按成功 update 计数；frame 跳号不放大计数，时间仍按 timestamp。 调用：`MultiObjectTracker`, `TrackingConfig`, `_candidate`, `expire_immediately.update`, `tracker.snapshot`, `tracker.update`。
- `F L187-L215` `test_equal_cost_tie_break_is_stable() -> None`：完全相同成本必须按 track_id、candidate_id 决胜且不受输入顺序影响。 调用：`run`。
- `N L190-L208` `test_equal_cost_tie_break_is_stable.run(*, reverse: bool) -> tuple[tuple[str, str], ...]`：按指定候选次序执行相同成本的稳定匹配场景。 调用：`MultiObjectTracker`, `TrackingConfig`, `_candidate`, `tracker.update`。
- `F L218-L284` `test_invalid_frame_updates_are_transactional() -> None`：重复、乱序或帧身份混杂的失败 update 不得推进任何 tracker 状态。 调用：`MultiObjectTracker`, `TrackingConfig`, `_candidate`, `tracker.snapshot`, `tracker.update`。
- `F L287-L332` `test_tracking_rejects_training_records_and_statically_has_no_gt_dependency() -> None` [IO-R]：正式 tracking 入口只收 runtime candidate，源码不导入 GT 或 legacy。 调用：`MultiObjectTracker`, `TrackingConfig`, `TrainingCandidateRecord`, `_candidate`, `tracker.snapshot`, `tracker.update`。
- `F L368-L380` `test_association_gates_split_tracks(config: TrackingConfig, next_candidate: CandidateObservation) -> None`：任一显式门限拒绝配对时，旧轨迹 missing、候选创建新轨迹。 调用：`MultiObjectTracker`, `_candidate`, `tracker.update`。
- `F L383-L402` `test_invalid_embeddings_fail_without_mutating_state() -> None`：embedding 维数不一致或零范数必须硬失败，不能截断或静默降级。 调用：`MultiObjectTracker`, `TrackingConfig`, `_candidate`, `tracker.snapshot`, `tracker.update`。
- `F L413-L447` `test_one_hot_config_weights_select_exact_cost_component(weights: tuple[float, float, float], expected_total: float) -> None`：三项 one-hot 配置必须分别选择空间、embedding 和类型成本。 调用：`AssociationCostSpec.from_config`, `AssociationCostSpec.from_config.cost`, `TrackAssociationView`, `TrackingConfig`, `_candidate`。

## `src/traning/tests/unit/test_phase5_belief.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.belief`, `traning.config`, `traning.contracts`, `traning.tracking.tracker`

- `F L32-L52` `_candidate(frame_index: int, candidate_id: str, *, x: float, embedding: tuple[float, ...]=(1.0, 0.0), timestamp_ms: float | None=None) -> CandidateObservation`：执行 `candidate` 对应逻辑。 调用：`CandidateObservation`。
- `F L55-L102` `_tracked(track_id: str, frame_index: int, *, lifecycle: TrackLifecycle, age: int, x: float=10.0, embedding: tuple[float, ...]=(1.0, 0.0), timestamp_ms: float | None=None, missed_frames: int=0, time_since_seen_ms: float=0.0) -> TrackedObservation`：执行 `tracked` 对应逻辑。 调用：`TrackedObservation`, `_candidate`。
- `F L105-L109` `_encoder() -> PerTrackBeliefEncoder`：执行 `encoder` 对应逻辑。 调用：`BeliefConfig`, `PerTrackBeliefEncoder`。
- `F L112-L129` `_assert_belief_close(first: BeliefState, second: BeliefState) -> None`：执行 `assert belief close` 对应逻辑。
- `F L132-L184` `test_tensor_heads_have_valid_shapes_and_backward_reaches_every_module() -> None`：完整 dense baseline 的 projection、GRU 与全部 head 都必须参与训练图。 调用：`_encoder`, `encoder.forward_step`。
- `F L187-L241` `test_forward_step_is_causal_and_segmented_equals_continuous() -> None`：未来 suffix 不改变 prefix，传递显式 hidden 的分段递推等于连续递推。 调用：`_encoder`, `run`。
- `N L199-L212` `test_forward_step_is_causal_and_segmented_equals_continuous.run(values: torch.Tensor, hidden: torch.Tensor | None=None) -> tuple[tuple[BeliefTensorOutput, ...], torch.Tensor]`：逐时刻运行 encoder，并返回全部输出及最终隐状态。 调用：`encoder.forward_step`。
- `F L244-L285` `test_runtime_isolates_tracks_from_order_and_other_track_perturbation() -> None`：A/B 输入反序或只扰动 B，都不得改变 A 的 belief。 调用：`_assert_belief_close`, `_encoder`, `run`。
- `N L250-L275` `test_runtime_isolates_tracks_from_order_and_other_track_perturbation.run(*, reverse: bool, perturb_b: bool) -> dict[str, BeliefState]`：在候选重排和单轨扰动条件下运行 belief 状态机。 调用：`PerTrackBeliefRuntime`, `_encoder`, `_tracked`, `runtime.step`。
- `F L288-L300` `test_tracker_to_belief_preserves_stable_identity() -> None`：Tracking 的稳定 track_id 必须原样贯穿公共 BeliefState。 调用：`MultiObjectTracker`, `PerTrackBeliefRuntime`, `TrackingConfig`, `_candidate`, `_encoder`, `runtime.step`。
- `F L303-L347` `test_missing_expired_equal_timestamp_and_clear_replay() -> None`：MISSING 使用 previous；EXPIRED 当帧返回后移除；clear 后可确定性重放。 调用：`PerTrackBeliefRuntime`, `_assert_belief_close`, `_encoder`, `_tracked`, `runtime.clear`, `runtime.snapshot`。
- `F L350-L405` `test_invalid_batches_preserve_snapshot_and_runtime_clock() -> None`：全部校验成功前不得提交 per-track state 或全局 frame clock。 调用：`PerTrackBeliefRuntime`, `_encoder`, `_tracked`, `runtime.snapshot`, `runtime.step`。
- `F L408-L452` `test_belief_model_has_no_action_gt_legacy_or_wide_type_boundary() -> None` [IO-R]：模型结构和源码不得重新引入 action/candidate head、GT、legacy 或 Any。 调用：`_encoder`。

## `src/traning/tests/unit/test_phase6_attribution.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.evaluation`

- `F L26-L47` `test_frame_105_without_clicks_is_only_decision_unresolved() -> None`：无点击帧不得被 overlay/candidate 证据改写为空间错误。 调用：`TargetObject`, `build_sequence_evaluation_events`, `score_click_sequence`。
- `F L50-L80` `test_hit_miss_and_frequency_limited_mapping_is_exact() -> None`：每个 click evaluation 必须一对一保留状态与 canonical 标签。 调用：`PredictedClick`, `TargetObject`, `build_sequence_evaluation_events`, `score_click_sequence`。
- `F L83-L97` `test_event_replay_and_unresolved_order_are_stable() -> None`：相同领域输入必须产生完全相同的顺序和 canonical hash。 调用：`TargetObject`, `build_sequence_evaluation_events`, `score_click_sequence`。
- `F L113-L129` `test_unresolved_event_contract_is_strict(overrides: dict[str, object]) -> None`：未解析目标事件的 invariant 不允许消费者自行降级或改写。 调用：`SequenceEvaluationEvent`, `values.update`。
- `F L132-L156` `test_click_event_contract_rejects_contradictory_pass_and_tags() -> None`：通过 click 必须无错误；失败 click 必须具有错误域和标签。 调用：`SequenceEvaluationEvent`。
- `F L159-L182` `test_attribution_source_has_no_legacy_any_or_rescoring() -> None` [IO-R]：归因层只能投影 SequenceScore，不得读取视觉旁路或重新评分。

## `src/traning/tests/unit/test_phase6_outcome_dataset.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package`, `traning.contracts`, `traning.data`, `traning.evaluation`, `traning.infrastructure`, `traning.outcome.dataset`, `traning.outcome.oracle`

- `F L56-L68` `_belief(track_id: str, timestamp_ms: float, *, x: float) -> BeliefState`：执行 `belief` 对应逻辑。 调用：`BeliefState`, `ObjectTypeDistribution`, `Point2D`。
- `F L71-L102` `_frame(sample_id: str, timestamp_ms: float) -> CounterfactualFrame`：执行 `frame` 对应逻辑。 调用：`CounterfactualFrame`, `OracleState`, `OracleTarget`, `Point2D`, `_belief`。
- `F L105-L112` `_builder(horizons: tuple[float, ...]=(0.0, 125.0, 300.0)) -> CounterfactualOutcomeDatasetBuilder`：执行 `builder` 对应逻辑。 调用：`CounterfactualOutcomeDatasetBuilder`, `OutcomeOracle`。
- `F L115-L126` `_publish(directory: Path, dataset: CounterfactualOutcomeDataset) -> OutcomeDatasetArtifactStore`：执行 `publish` 对应逻辑。 调用：`OutcomeDatasetArtifactStore`, `store.publish`。
- `F L129-L135` `_load(store: OutcomeDatasetArtifactStore) -> CounterfactualOutcomeDataset`：执行 `load` 对应逻辑。 调用：`store.load`。
- `F L138-L141` `_manifest_payload(directory: Path) -> dict[str, object]` [IO-R]：执行 `manifest payload` 对应逻辑。
- `F L144-L149` `_records_path(directory: Path, manifest: dict[str, object]) -> Path`：执行 `records path` 对应逻辑。
- `F L152-L157` `_write_manifest(directory: Path, payload: dict[str, object]) -> None` [IO-W]：写入 `manifest` 对应的数据或结果。
- `F L160-L163` `_rehash_records(directory: Path, manifest: dict[str, object]) -> None` [IO-R]：执行 `rehash records` 对应逻辑。 调用：`_records_path`, `_write_manifest`, `hashlib.sha256`。
- `F L166-L216` `test_builder_order_labels_and_serialized_bytes_are_deterministic(tmp_path: Path) -> None` [IO-R IO-W]：frames/beliefs 反序仍产生相同 records 次序和相同 JSONL 字节。 调用：`_builder`, `_builder.build`, `_frame`, `_load`, `_manifest_payload`, `_publish`。
- `F L219-L229` `test_sample_ids_are_unique_and_encode_changed_horizon() -> None`：相同 index 上 horizon 值变化必须改变 ID，不能只依赖数组位置。 调用：`_builder`, `_builder.build`, `_frame`。
- `F L232-L264` `test_length_prefixed_sample_id_prevents_component_boundary_collision() -> None`：``a:b + c`` 与 ``a + b:c`` 不得生成旧分隔符方案下的同一 sample ID。 调用：`_builder`, `_builder.build`, `one_frame`。
- `N L235-L255` `test_length_prefixed_sample_id_prevents_component_boundary_collision.one_frame(sample_id: str, track_id: str, object_id: str) -> CounterfactualFrame`：构造单轨单目标帧以验证长度前缀 ID 编码。 调用：`CounterfactualFrame`, `OracleState`, `OracleTarget`, `_belief`。
- `F L267-L286` `test_dataset_wrapper_rejects_empty_or_mixed_split_records() -> None` [IO-W]：Typed dataset 必须非空，且 manifest split 不能与 record lineage 分裂。 调用：`CounterfactualOutcomeDataset`, `_builder`, `_builder.build`, `_frame`。
- `F L289-L316` `test_artifact_round_trip_uses_canonical_manifest(tmp_path: Path) -> None` [IO-R]：manifest 必须包装 canonical ArtifactManifest 并记录精确版本和摘要。 调用：`OutcomeDatasetArtifactStore`, `_builder`, `_builder.build`, `_frame`, `_load`, `hashlib.sha256`。
- `F L333-L365` `test_manifest_identity_and_integrity_tampering_is_rejected(tmp_path: Path, case: str, expected_error: type[Exception]) -> None`：摘要、行数、schema、身份及 canonical 版本任一篡改都必须硬失败。 调用：`_builder`, `_builder.build`, `_frame`, `_load`, `_manifest_payload`, `_publish`。
- `F L369-L388` `test_rehashed_gt_or_unknown_record_field_is_rejected(tmp_path: Path, injected_field: str) -> None` [IO-R IO-W]：即使攻击者重算摘要，GT 或未知字段也不能越过 exact-schema 边界。 调用：`_builder`, `_builder.build`, `_frame`, `_load`, `_manifest_payload`, `_publish`。
- `F L391-L406` `test_rehashed_duplicate_sample_id_is_rejected(tmp_path: Path) -> None` [IO-R IO-W]：重复 sample_id 即使行数和摘要一致，也不是合法 typed dataset。 调用：`_builder`, `_builder.build`, `_frame`, `_load`, `_manifest_payload`, `_publish`。
- `F L409-L423` `test_rehashed_record_split_mismatch_is_rejected(tmp_path: Path) -> None` [IO-R IO-W]：record 自带 split 即使合法，也必须与 manifest split 完全一致。 调用：`_builder`, `_builder.build`, `_frame`, `_load`, `_manifest_payload`, `_publish`。
- `F L426-L451` `test_manifest_commit_failure_preserves_old_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None` [IO-R]：新 generation 已写完但 manifest 提交失败时，旧提交仍须完整可读。 调用：`_builder`, `_builder.build`, `_frame`, `_load`, `_publish`, `store.publish`。
- `N L437-L440` `test_manifest_commit_failure_preserves_old_generation.fail_manifest_commit(path: Path, payload: object) -> None`：模拟最终 manifest 原子提交失败。 调用：`AtomicWriteError`。
- `F L454-L487` `test_dataset_source_has_no_legacy_or_any_and_inference_has_no_oracle_fields() -> None` [IO-R]：Dataset 不得依赖 legacy/Any，推理契约在类型上不得暴露 oracle 标签。

## `src/traning/tests/unit/test_phase6_outcome_oracle.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts`, `traning.evaluation`, `traning.outcome.oracle`

- `F L47-L50` `_golden() -> dict[str, object]` [IO-R]：执行 `golden` 对应逻辑。
- `F L53-L67` `_ring_target(*, track_id: str='ring-track', object_id: str='ring-object', start_time_ms: float=1000.0, end_time_ms: float=2000.0) -> OracleTarget`：执行 `ring target` 对应逻辑。 调用：`OracleTarget`, `Point2D`。
- `F L70-L81` `_state(target: OracleTarget, *, timestamp_ms: float=1000.0, resolved: bool=False) -> OracleState`：执行 `state` 对应逻辑。 调用：`OracleState`。
- `F L84-L96` `_click(target: OracleTarget, *, horizon_ms: float=0.0, position: Point2D | None=None, path: tuple[Point2D, ...]=()) -> HypotheticalClick`：执行 `click` 对应逻辑。 调用：`HypotheticalClick`。
- `F L99-L174` `test_v2_point_slider_and_sequence_match_existing_golden() -> None`：V2 canonical scorer 必须逐字段保持冻结 golden。 调用：`PredictedClick`, `TargetObject`, `_golden`, `score_click_sequence`, `score_point`, `score_slider`。
- `F L177-L202` `test_ring_oracle_is_sensitive_to_space_horizon_and_category() -> None`：ring 标签必须由 canonical 空间/时间评分共同决定。 调用：`OutcomeOracle`, `Point2D`, `_click`, `_ring_target`, `_state`, `oracle.evaluate`。
- `F L205-L231` `test_oracle_invalid_states_cover_expired_unknown_resolved_and_spinner() -> None`：不可评分状态必须明确映射为 INVALID，只有过期分支设置 expires。 调用：`HypotheticalClick`, `OracleTarget`, `OutcomeOracle`, `Point2D`, `_click`, `_ring_target`。
- `F L234-L285` `test_slider_head_only_and_full_path_use_canonical_slider_score() -> None`：head-only 合法；完整路径必须与共享 score_slider 数值完全一致。 调用：`OracleTarget`, `OutcomeOracle`, `Point2D`, `_click`, `_state`, `oracle.evaluate`。
- `F L288-L303` `test_oracle_category_is_contract_identity_and_slider_head_matches_path_start() -> None`：Oracle 只重导出 canonical enum，并拒绝相互矛盾的 slider 头与路径。 调用：`OracleTarget`, `Point2D`。
- `F L315-L371` `test_oracle_thresholds_produce_valid_training_sample_categories(monkeypatch: pytest.MonkeyPatch, normalized: float, expected_category: OutcomeCategory) -> None` [IO-W]：Oracle 阈值边界必须直接满足 OutcomeTrainingSample 的同源语义。 调用：`BeliefState`, `CombinedScore`, `ObjectTypeDistribution`, `OutcomeOracle`, `OutcomeOracle.evaluate`, `OutcomeTrainingSample`。
- `F L374-L396` `test_oracle_sequence_is_exact_canonical_delegation() -> None`：Oracle sequence 入口不得改变 canonical 结果或错误归因。 调用：`OutcomeOracle`, `PredictedClick`, `SequenceScoreSpec`, `TargetObject`, `oracle.evaluate_sequence`, `score_click_sequence`。
- `F L418-L437` `test_oracle_outcome_rejects_category_validity_contradictions(kwargs: dict[str, object]) -> None`：OracleOutcome 不允许类别、valid、expires 和 passed 相互矛盾。 调用：`OracleOutcome`, `values.update`。
- `F L440-L468` `test_oracle_outcome_rejects_invalid_category_type_and_negative_error() -> None`：类别枚举与误差范围必须是硬契约。 调用：`OracleOutcome`。
- `F L471-L518` `test_evaluation_and_oracle_ast_use_one_shared_scoring_implementation() -> None` [IO-R]：禁止 legacy/Any 渗透，Oracle 必须调用共享 scorer 而非复制公式。

## `src/traning/tests/unit/test_phase7_outcome_metrics.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.evaluation.metrics`, `traning.outcome.calibration`

- `F L25-L36` `test_hand_computed_multiclass_metrics() -> None`：NLL、Brier 与 ECE 必须匹配可手算二分类 batch。 调用：`multiclass_brier_score`, `multiclass_nll`, `top_label_ece`。
- `F L39-L47` `test_top_label_ece_internal_boundary_enters_right_bin() -> None`：置信度恰为内部边界时必须进入右侧箱。 调用：`top_label_ece`。
- `F L50-L64` `test_hand_computed_score_and_expiry_metrics() -> None`：expected score MAE 与 expiry BCE/Brier 使用独立 typed 边界。 调用：`expected_score_mae`, `expiry_binary_cross_entropy`, `expiry_brier_score`。
- `F L78-L86` `test_multiclass_metrics_reject_invalid_boundary(probabilities: torch.Tensor, labels: torch.Tensor, error: type[Exception]) -> None`：shape、dtype、finite、概率和和 label 越界均硬失败。 调用：`multiclass_nll`。
- `F L89-L103` `test_metrics_reject_device_and_binary_shape_mismatch() -> None`：相关 tensor 不得跨 device，也不得依赖 broadcasting。 调用：`expected_score_mae`, `expiry_binary_cross_entropy`, `expiry_brier_score`, `multiclass_brier_score`。
- `F L106-L114` `test_temperature_one_is_identity_and_requires_positive_scalar() -> None`：T=1 不改变 logits，非正或非有限温度不得构造。 调用：`ScalarTemperatureCalibrator`, `calibrator.transform`。
- `F L117-L130` `test_deterministic_temperature_fit_does_not_worsen_overconfident_nll() -> None`：固定 validation 网格拟合应复现，并缓和含错误样本的过度自信 logits。 调用：`evaluate_temperature_calibration`, `fit_temperature_calibrator`。
- `F L133-L149` `test_calibration_rejects_invalid_logits_labels_and_search_spec() -> None`：校准同样硬拒绝 shape、dtype、device、finite、label 和搜索区间错误。 调用：`fit_temperature_calibrator`。

## `src/traning/tests/unit/test_phase7_outcome_model.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.config`, `traning.contracts`, `traning.outcome.model`

- `F L30-L31` `_config() -> OutcomeConfig` [IO-W]：执行 `config` 对应逻辑。 调用：`OutcomeConfig`。
- `F L34-L46` `_belief() -> BeliefState`：执行 `belief` 对应逻辑。 调用：`BeliefState`, `ObjectTypeDistribution`, `Point2D`。
- `F L49-L52` `_constant_parameters(model: DenseOutcomeModel, value: float) -> None`：执行 `constant parameters` 对应逻辑。
- `F L55-L74` `test_forward_probabilities_expected_score_and_variance_are_exact() -> None`：五分类概率及代表分数矩必须由同一分布推导。 调用：`DenseOutcomeModel`, `_config`。
- `F L77-L99` `test_horizon_feature_deterministically_changes_distribution() -> None` [IO-W]：同一 belief 的不同 horizon 必须有可学习且可证明的分布差异。 调用：`DenseOutcomeModel`, `_config`。
- `F L102-L126` `test_all_trunk_category_and_expiry_parameters_receive_nonzero_gradient() -> None`：所有构建参数都必须进入同一训练 forward。 调用：`DenseOutcomeModel`, `_config`, `_constant_parameters`。
- `F L129-L148` `test_predict_returns_canonical_outcome_distribution() -> None`：runtime 入口只接受 BeliefState+horizon，并保留 track identity。 调用：`DenseOutcomeModel`, `_belief`, `_config`, `model.predict`。
- `F L163-L172` `test_forward_rejects_invalid_shape_dtype_and_values(belief: torch.Tensor, horizon: torch.Tensor, error: type[Exception]) -> None`：forward 必须拒绝错误形状、dtype 与非有限输入。 调用：`DenseOutcomeModel`, `_config`。
- `F L175-L185` `test_predict_rejects_wrong_embedding_and_horizon() -> None`：predict 必须拒绝维度不符的 belief 和非法 horizon。 调用：`DenseOutcomeModel`, `_belief`, `_config`, `model.predict`。
- `F L188-L205` `test_model_ast_has_no_legacy_oracle_gt_smet_or_any() -> None` [IO-R]：runtime 模型不得依赖训练 oracle、GT 或 legacy 稀疏实现。

## `src/traning/tests/unit/test_phase7_outcome_training.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.config`, `traning.contracts`, `traning.outcome.dataset`, `traning.outcome.model`, `traning.outcome.training`

- `F L36-L37` `_config() -> OutcomeConfig` [IO-W]：执行 `config` 对应逻辑。 调用：`OutcomeConfig`。
- `F L40-L52` `_belief(index: int) -> BeliefState`：执行 `belief` 对应逻辑。 调用：`BeliefState`, `ObjectTypeDistribution`, `Point2D`。
- `F L55-L78` `_sample(index: int, category: OutcomeCategory, score: float, *, split: DataSplit=DataSplit.TRAIN) -> OutcomeTrainingSample`：执行 `sample` 对应逻辑。 调用：`OutcomeTrainingSample`, `_belief`。
- `F L81-L88` `_samples() -> tuple[OutcomeTrainingSample, ...]`：执行 `samples` 对应逻辑。 调用：`_sample`。
- `F L91-L101` `_dataset(records: tuple[OutcomeTrainingSample, ...] | None=None) -> CounterfactualOutcomeDataset`：把测试 records 绑定到一个明确坐标指纹，模拟正式 artifact loader。 调用：`CounterfactualOutcomeDataset`, `_samples`。
- `F L104-L121` `test_collate_preserves_lineage_and_tensor_contract() -> None`：批处理拼装必须保留 lineage 并产生规定张量契约。 调用：`_dataset`, `collate_outcome_samples`。
- `F L124-L140` `test_collate_rejects_naked_records_and_wrong_embedding_dim() -> None`：批处理只能接收带指纹 dataset，并拒绝错误 embedding 维度。 调用：`CounterfactualOutcomeDataset`, `_dataset`, `_sample`, `_samples`, `collate_outcome_samples`。
- `F L143-L160` `test_loss_is_finite_and_backward_reaches_all_model_groups() -> None`：训练损失必须有限，且反向传播覆盖所有模型参数组。 调用：`DenseOutcomeModel`, `_config`, `_dataset`, `collate_outcome_samples`, `compute_outcome_loss`。
- `F L163-L181` `test_evaluation_reuses_canonical_metrics_and_returns_finite_values() -> None`：评估必须复用 canonical 指标并返回有限数值。 调用：`DenseOutcomeModel`, `_config`, `_dataset`, `collate_outcome_samples`, `evaluate_outcome_batch`。
- `F L184-L212` `test_real_optimizer_step_clears_grads_and_updates_parameters() -> None`：真实优化步骤必须清除梯度并更新模型参数。 调用：`DenseOutcomeModel`, `RecordingSgd`, `_config`, `_dataset`, `collate_outcome_samples`, `train_outcome_step`。
- `C L187-L196` `test_real_optimizer_step_clears_grads_and_updates_parameters.RecordingSgd(torch.optim.SGD)` [CLASS]：记录 zero_grad 参数的 SGD 测试替身。
- `N L192-L196` `test_real_optimizer_step_clears_grads_and_updates_parameters.RecordingSgd.zero_grad(self, set_to_none: bool=True) -> None`：记录并透传 set_to_none 选项。 调用：`super.zero_grad`。
- `F L222-L229` `test_primary_loss_weights_cannot_be_replaced_by_score(weights: OutcomeLossWeights) -> None`：分数辅助项不得替代分类与过期两个主损失。
- `F L232-L238` `test_invalid_primary_loss_weights_are_rejected() -> None`：任一主损失权重为零时必须拒绝配置。 调用：`OutcomeLossWeights`。
- `F L241-L255` `test_training_module_has_no_runtime_oracle_ground_truth_or_any_dependency() -> None` [IO-R]：训练模块不得依赖 runtime oracle、GT 或宽泛 Any。

## `src/traning/tests/unit/test_phase8_decision_contracts.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.config`, `traning.contracts`

- `F L24-L38` `_outcome(*, track_id: str='track-1', horizon_ms: float=0.0) -> OutcomeDistribution`：执行 `outcome` 对应逻辑。 调用：`OutcomeDistribution`。
- `F L41-L54` `_click(**overrides: object) -> DecisionResult`：执行 `click` 对应逻辑。 调用：`DecisionResult`, `Point2D`, `_outcome`, `values.update`。
- `F L57-L70` `_wait(**overrides: object) -> DecisionResult`：执行 `wait` 对应逻辑。 调用：`DecisionResult`, `values.update`。
- `F L73-L85` `test_decision_config_round_trip_includes_risk_and_wait_cost() -> None` [IO-W]：新增字段必须完整穿过 typed→dict→loader 边界。 调用：`V2Config`, `load_v2_config`, `v2_config_to_dict`。
- `F L88-L92` `test_decision_action_must_be_enum() -> None`：动作字段必须使用 DecisionAction 枚举而非裸字符串。 调用：`_wait`。
- `F L95-L111` `test_click_requires_immediate_horizon_and_complete_audit_fields() -> None`：CLICK 只表达立即执行，且保留目标与 outcome 审计证据。 调用：`_click`, `_outcome`。
- `F L114-L127` `test_wait_requires_positive_horizon_and_no_bound_target_or_outcome() -> None`：WAIT_ONE_STEP 必须选择正 horizon 且不得伪装成延迟 CLICK。 调用：`_outcome`, `_wait`。
- `F L132-L139` `test_horizons_require_click_now_and_positive_wait_step(config_type: type[OutcomeConfig] | type[DecisionConfig], horizons: tuple[int, ...]) -> None`：horizon 集必须包含立即点击点和至少一个正等待步长。
- `F L153-L161` `test_decision_cost_fields_are_strict_finite_nonnegative(field_name: str, value: object, error: type[Exception]) -> None`：决策成本字段必须是有限且非负的严格数值。 调用：`DecisionConfig`。
- `F L164-L170` `test_loader_rejects_unknown_or_invalid_decision_fields() -> None`：配置加载器必须拒绝未知键与错误类型的决策字段。 调用：`load_v2_config`。

## `src/traning/tests/unit/test_phase8_decision_planner.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.config`, `traning.contracts`, `traning.decision.planner`

- `F L25-L36` `_config(**changes: float | tuple[int, ...]) -> DecisionConfig` [IO-W]：执行 `config` 对应逻辑。 调用：`DecisionConfig`。
- `F L39-L51` `_belief(track_id: str, x: float) -> BeliefState`：执行 `belief` 对应逻辑。 调用：`BeliefState`, `ObjectTypeDistribution`, `Point2D`。
- `F L54-L73` `_outcome(track_id: str, horizon_ms: float, value: float, *, success: float=1.0, variance: float=0.0) -> OutcomeDistribution`：执行 `outcome` 对应逻辑。 调用：`OutcomeDistribution`。
- `F L76-L86` `_two_track_inputs() -> tuple[tuple[BeliefState, ...], tuple[OutcomeDistribution, ...]]`：执行 `two track inputs` 对应逻辑。 调用：`_belief`, `_outcome`。
- `F L89-L102` `test_clicks_current_track_with_maximum_utility_and_preserves_binding() -> None`：规划器必须点击当前效用最高轨迹并保留目标绑定。 调用：`OptimalStoppingPlanner`, `OptimalStoppingPlanner.plan`, `_config`, `_two_track_inputs`。
- `F L105-L123` `test_waits_when_future_utility_is_clearly_better() -> None`：未来效用明显更高时规划器必须选择等待。 调用：`OptimalStoppingPlanner`, `OptimalStoppingPlanner.plan`, `_belief`, `_config`, `_outcome`。
- `F L126-L145` `test_risk_penalty_changes_selected_track() -> None`：风险惩罚必须能使规划器从高方差轨迹转向稳健轨迹。 调用：`OptimalStoppingPlanner`, `OptimalStoppingPlanner.plan`, `_belief`, `_config`, `_outcome`。
- `F L148-L159` `test_input_slot_and_order_do_not_change_selection() -> None`：输入槽位与排列顺序不得改变最终决策。 调用：`OptimalStoppingPlanner`, `_config`, `_two_track_inputs`, `planner.plan`。
- `F L162-L175` `test_stable_track_tie_and_click_now_tie_policy() -> None`：轨迹同效用时必须稳定决胜，并优先立即点击。 调用：`OptimalStoppingPlanner`, `OptimalStoppingPlanner.plan`, `_belief`, `_config`, `_outcome`。
- `F L178-L191` `test_low_current_success_forces_wait() -> None`：当前成功概率低于门槛时必须强制等待。 调用：`OptimalStoppingPlanner`, `OptimalStoppingPlanner.plan`, `_belief`, `_config`, `_outcome`。
- `F L194-L202` `test_empty_inputs_return_explicit_costed_wait() -> None`：空输入必须返回携带等待成本的显式 WAIT。 调用：`OptimalStoppingPlanner`, `OptimalStoppingPlanner.plan`, `_config`。
- `F L205-L232` `test_rejects_duplicate_incomplete_unknown_extra_and_nonfinite_inputs() -> None`：规划器必须拒绝重复、不完整、额外轨迹及非有限输入。 调用：`OptimalStoppingPlanner`, `_belief`, `_config`, `_outcome`, `planner.plan`。
- `F L235-L244` `test_rejects_duplicate_track_horizon_outcome() -> None`：同一轨迹与 horizon 的重复 outcome 必须被拒绝。 调用：`OptimalStoppingPlanner`, `_belief`, `_config`, `_outcome`, `planner.plan`。
- `F L247-L273` `test_planner_ast_has_no_forbidden_runtime_inputs_or_shortcuts() -> None` [IO-R]：规划器源码不得读取禁用训练信息或旧捷径字段。

## `src/traning/tests/unit/test_phase8_decision_utility.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.config`, `traning.contracts`, `traning.decision.utility`

- `F L16-L30` `_outcome(**changes: float | str) -> OutcomeDistribution`：执行 `outcome` 对应逻辑。 调用：`OutcomeDistribution`, `values.update`。
- `F L33-L42` `_config(**changes: float) -> DecisionConfig`：执行 `config` 对应逻辑。 调用：`DecisionConfig`, `values.update`。
- `F L45-L54` `test_click_utility_matches_hand_computed_formula() -> None`：唯一 utility 公式和成功概率必须逐项匹配手算结果。 调用：`_config`, `_outcome`, `compute_click_utility`。
- `F L57-L73` `test_risk_and_each_penalty_only_reduce_value_by_its_term() -> None`：risk 与各 penalty 的边际影响必须等于对应概率或方差。 调用：`_config`, `_outcome`, `compute_click_utility`。
- `F L76-L89` `test_compute_is_pure_and_result_is_frozen() -> None` [IO-W]：计算不得修改输入，结果必须保留原 Outcome 对象且不可变。 调用：`_config`, `_outcome`, `compute_click_utility`。
- `F L92-L103` `test_click_utility_rejects_identity_probability_and_finite_contradictions() -> None`：结果 DTO 不允许脱离绑定 Outcome 或伪造成功概率。 调用：`ClickUtility`, `_outcome`。
- `F L106-L112` `test_compute_rejects_wrong_typed_inputs() -> None`：核心入口只接受 canonical OutcomeDistribution 与 DecisionConfig。 调用：`_config`, `_outcome`, `compute_click_utility`。
- `F L115-L138` `test_utility_source_has_no_forbidden_decision_shortcuts() -> None` [IO-R]：Utility 层不得读取图像、GT、oracle、logits、argmax 或 legacy。

## `src/traning/tests/unit/test_phase9_hard_examples.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts`, `traning.evaluation`, `traning.training.hard_examples`

- `F L32-L33` `_event_id(number: int) -> str`：执行 `event id` 对应逻辑。
- `F L36-L52` `_failed_event(number: int, primary_error: PrimaryError, tag: EvaluationTag) -> SequenceEvaluationEvent`：执行 `failed event` 对应逻辑。 调用：`SequenceEvaluationEvent`, `_event_id`。
- `F L55-L67` `_passed_event(number: int) -> SequenceEvaluationEvent`：执行 `passed event` 对应逻辑。 调用：`SequenceEvaluationEvent`, `_event_id`。
- `F L70-L83` `test_frame_105_unresolved_routes_only_to_decision() -> None`：零点击未解析目标必须保持 decision identity，绝不转为空间错误。 调用：`EvaluationSplitEvent`, `SequenceScore`, `build_hard_example_plan`, `build_sequence_evaluation_events`。
- `F L86-L113` `test_registry_routes_three_primary_errors_and_skips_passed() -> None`：注册表必须路由三类主错误，并跳过已通过样本。 调用：`EvaluationSplitEvent`, `_failed_event`, `_passed_event`, `build_hard_example_plan`。
- `F L116-L137` `test_validation_and_test_are_explicitly_excluded_from_weights() -> None`：验证集与测试集事件不得进入训练重加权。 调用：`EvaluationSplitEvent`, `_failed_event`, `build_hard_example_plan`。
- `F L140-L147` `test_all_split_is_rejected() -> None`：无法确定用途的 ALL split 必须被明确拒绝。 调用：`EvaluationSplitEvent`, `_failed_event`。
- `F L150-L160` `test_consumer_views_preserve_same_event_object_and_error_identity() -> None`：各 hard-example 消费视图必须共享同一事件及错误身份。 调用：`EvaluationSplitEvent`, `_failed_event`, `build_hard_example_plan`, `plan.events_for`。
- `F L163-L181` `test_replay_is_stably_sorted_and_duplicate_event_id_is_rejected() -> None`：重放必须稳定排序，并拒绝重复事件 ID。 调用：`EvaluationSplitEvent`, `_failed_event`, `build_hard_example_plan`。
- `F L185-L194` `test_hard_example_weight_must_be_finite_positive(weight: float) -> None`：hard-example 权重必须是有限正数。 调用：`EvaluationSplitEvent`, `HardExampleRoute`, `HardExampleWeight`, `_failed_event`。
- `F L197-L219` `test_hard_example_source_has_no_forbidden_dependency_or_rescoring() -> None` [IO-R]：路由层不得读取视觉旁路、oracle 或重新调用 scorer。

## `src/traning/tests/unit/test_phase9_optimization.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.training.optimization`

- `F L24-L32` `_initial() -> ParameterVector`：执行 `initial` 对应逻辑。 调用：`ParameterVector`。
- `F L35-L44` `_acceptance(*, passed: bool) -> TrialAcceptance`：执行 `acceptance` 对应逻辑。 调用：`TrialAcceptance`。
- `F L47-L59` `_observation(trial_index: int, parameters: ParameterVector, *, objective: float=1.0, passed: bool=False) -> TrialObservation`：执行 `observation` 对应逻辑。 调用：`TrialObservation`, `_acceptance`。
- `F L62-L70` `test_parameter_registry_clamps_legacy_negative_threshold() -> None`：注册表必须把旧负阈值钳制到合法搜索边界。 调用：`threshold.quantize`。
- `F L73-L97` `test_controller_continues_after_high_objective_when_a_gate_fails() -> None`：任一门禁失败时，即使目标分很高也必须继续搜索。 调用：`DeterministicSearchController`, `_initial`, `_observation`, `controller.decide`。
- `F L100-L115` `test_controller_passes_only_when_every_acceptance_gate_passes() -> None`：只有全部接受门禁通过时控制器才能进入 PASSED。 调用：`DeterministicSearchController`, `_initial`, `_observation`, `controller.decide`。
- `F L118-L134` `test_controller_reports_budget_exhaustion_explicitly() -> None`：显式 trial 预算耗尽时必须返回 EXHAUSTED 状态。 调用：`DeterministicSearchController`, `_initial`, `_observation`, `controller.decide`。
- `F L137-L151` `test_same_seed_and_history_produce_same_proposal() -> None`：相同随机种子与历史必须生成完全相同的提案。 调用：`DeterministicSearchController`, `DeterministicSearchController.decide`, `_initial`, `_observation`, `first_controller.decide`。
- `F L154-L168` `test_every_published_proposal_is_in_range_and_quantized() -> None`：每个公开提案都必须位于范围内并符合量化步长。 调用：`DeterministicSearchController`, `_initial`, `_observation`, `controller.decide`, `spec.quantize`, `spec.validate`。
- `F L171-L197` `test_finite_quantized_space_has_a_real_exhausted_terminal_state() -> None`：有限量化空间遍历完后必须产生真实 EXHAUSTED 终态。 调用：`DeterministicSearchController`, `ParameterRegistry`, `ParameterSpec`, `_initial`, `_observation`, `controller.decide`。
- `C L200-L210` `_ThirdTrialPasses` [CLASS]：封装 `ThirdTrialPasses` 相关数据或行为。
- `M L201-L202` `_ThirdTrialPasses.__init__(self) -> None`：初始化实例依赖、配置和运行状态。
- `M L204-L210` `_ThirdTrialPasses.evaluate(self, parameters: ParameterVector, trial_index: int) -> TrialObservation`：在第三次 trial 返回全门禁通过的观测。 调用：`_observation`。
- `C L213-L219` `_NeverPasses` [CLASS]：封装 `NeverPasses` 相关数据或行为。
- `M L214-L219` `_NeverPasses.evaluate(self, parameters: ParameterVector, trial_index: int) -> TrialObservation`：始终返回未完全通过的 trial 观测。 调用：`_observation`。
- `F L222-L231` `test_run_search_keeps_going_until_full_acceptance() -> None`：无预算搜索必须持续到出现全门禁通过的 trial。 调用：`_ThirdTrialPasses`, `_initial`, `run_search`。
- `F L234-L241` `test_run_search_raises_typed_error_on_exhaustion() -> None`：搜索预算耗尽必须抛出携带终态的 typed 错误。 调用：`_NeverPasses`, `_initial`, `run_search`。
- `F L244-L263` `test_optimization_contract_has_no_any_or_legacy_dependency() -> None` [IO-R]：优化契约不得依赖宽泛 Any、legacy 或错误顶层包。

## `src/traning/tests/unit/test_phase9_orchestration.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts`, `traning.training.optimization`, `traning.training.orchestration`, `traning.training.scheduling`

- `F L39-L53` `_quality_report(*, blocked: bool=False) -> DataQualityReport`：执行 `quality report` 对应逻辑。 调用：`DataQualityIssue`, `DataQualityReport`。
- `C L57-L80` `RecordingRunner` [CLASS]：记录阶段调用并可注入失败与最终接受结果的测试 runner。
- `M L68-L80` `RecordingRunner.run(self, stage: TrainingStage) -> StageResult`：记录阶段并返回按测试设置生成的阶段结果。 调用：`StageResult`, `self.calls.append`。
- `F L83-L93` `test_blocking_quality_report_causes_zero_runner_calls() -> None`：数据质量阻断时不得调用任何训练阶段。 调用：`RecordingRunner`, `TrainingOrchestrator`, `TrainingOrchestrator.run`, `_quality_report`。
- `F L96-L105` `test_stages_run_in_fixed_registry_order_and_all_pass() -> None`：所有阶段必须按固定注册表顺序运行并汇总通过。 调用：`RecordingRunner`, `TrainingOrchestrator`, `TrainingOrchestrator.run`, `_quality_report`。
- `F L108-L117` `test_stage_failure_stops_immediately_and_stays_failed() -> None`：阶段失败必须立即短路，且编排结果保持失败。 调用：`RecordingRunner`, `TrainingOrchestrator`, `TrainingOrchestrator.run`, `_quality_report`。
- `F L120-L131` `test_final_acceptance_is_required_after_all_stages() -> None`：阶段全部成功后仍必须满足最终 acceptance 门禁。 调用：`RecordingRunner`, `TrainingOrchestrator`, `TrainingOrchestrator.run`, `TrialAcceptance`, `_quality_report`。
- `F L134-L142` `test_quality_and_acceptance_data_gate_cannot_disagree() -> None`：质量报告与 acceptance 的 data 门禁不得互相矛盾。 调用：`RecordingRunner`, `TrainingOrchestrator`, `TrainingOrchestrator.run`, `TrialAcceptance`, `_quality_report`。
- `F L153-L172` `test_curriculum_advances_only_when_all_gates_pass(stage: CurriculumStage, expected: CurriculumStage) -> None`：课程阶段只有在全部门禁通过时才能推进。 调用：`CurriculumGate`, `decide_curriculum`。
- `F L175-L183` `test_empty_curriculum_gates_hold_and_full_stage_completes() -> None`：空课程门禁必须保持，FULL 通过后必须明确完成。 调用：`CurriculumGate`, `decide_curriculum`。
- `F L186-L192` `_asha() -> AshaScheduler`：执行 `asha` 对应逻辑。 调用：`AshaRung`, `AshaScheduler`。
- `F L195-L217` `test_asha_strict_gate_overrides_high_objective_and_top_fraction_promotes() -> None`：ASHA 严格门禁优先于高分，并只晋升通过者中的头部比例。 调用：`AshaTrial`, `_asha`, `_asha.decide`。
- `F L220-L238` `test_asha_ties_are_stable_and_terminal_rung_continues() -> None`：ASHA 同分决策必须稳定，末级通过者应继续完成流程。 调用：`AshaTrial`, `_asha`, `_asha.decide`。
- `F L241-L252` `test_asha_rejects_non_increasing_budgets_and_mixed_rungs() -> None`：ASHA 必须拒绝非递增预算和混合 rung 输入。 调用：`AshaRung`, `AshaScheduler`, `AshaTrial`, `_asha`, `_asha.decide`。
- `F L255-L286` `test_phase9_modules_do_not_depend_on_sqlite_legacy_ui_or_any() -> None` [IO-R]：Phase 9 模块不得依赖 SQLite、legacy UI 或宽泛 Any。 调用：`identifiers.update`, `type_any_names.update`。

## `src/traning/tests/unit/test_production_search_terminal.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`package`, `traning.config`, `traning.contracts`, `traning.data`, `traning.telemetry`, `traning.training`, `traning.training.production_stages`

- `F L39-L48` `_config(*, max_trials: int | None) -> V2Config`：构造不访问 CUDA、但带完整坐标身份的搜索配置。 调用：`CoordinateConfig`, `OptimizationConfig`, `V2Config`。
- `F L51-L82` `_datasets(config: V2Config) -> TrainingDatasetBundle`：构造不解码视频的空 typed bundle，隔离真实训练开销。 调用：`DataQualityReport`, `FrameCoordinateTransform`, `SegmentTrainingDataset`, `TrainingDatasetBundle`。
- `C L85-L140` `_DeterministicStageRunner` [CLASS]：用可配置通过序号替代昂贵模型训练，同时保留真实阶段协议。
- `M L91-L111` `_DeterministicStageRunner.__init__(self, *, base_config: V2Config, parameters: ParameterVector, trial_index: int, datasets: TrainingDatasetBundle, gates: ProductionGateSpec, run_dir: Path, run_id: str, reporter: TelemetryReporter) -> None`：记录 proposal，并保存生产入口稍后读取的最小状态。 调用：`ProductionTrialMetrics`。
- `M L113-L140` `_DeterministicStageRunner.run(self, stage: TrainingStage) -> StageResult` [IO-W]：普通失败返回 FAILED；指定 trial 则走完整阶段并提交占位 manifest。 调用：`StageResult`, `TrialAcceptance`, `self.stage_results.append`, `trial_checkpoint_directory`。
- `F L143-L144` `_accept_checkpoint(*_args: object, **_kwargs: object) -> None`：让测试只验证发布时机，不重复测试 checkpoint 解码细节。
- `F L147-L185` `test_production_gate_failure_continues_and_publishes_passed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None`：普通 gate 失败必须继续唯一 proposal，并在下一轮通过后发布 PASSED。 调用：`ProductionTrainer`, `ProductionTrainer.run`, `StateStore`, `StateStore.history`, `_config`, `_datasets`。
- `F L188-L226` `test_production_exhaustion_publishes_terminal_and_resume_does_not_repeat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None`：预算耗尽必须发布 EXHAUSTED；恢复只读历史，不重复训练已提交 proposal。 调用：`ProductionTrainer`, `StateStore`, `StateStore.history`, `_config`, `_datasets`, `trainer.run`。

## `src/traning/tests/unit/test_search_resume.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.config`, `traning.infrastructure`, `traning.training`

- `F L20-L21` `_initial() -> ParameterVector`：执行 `initial` 对应逻辑。 调用：`ParameterVector`。
- `F L24-L35` `_observation(trial_index: int, parameters: ParameterVector, *, passed: bool) -> TrialObservation`：执行 `observation` 对应逻辑。 调用：`TrialAcceptance`, `TrialObservation`。
- `C L38-L55` `_PassesAtIndex` [CLASS]：封装 `PassesAtIndex` 相关数据或行为。
- `M L39-L41` `_PassesAtIndex.__init__(self, passing_index: int) -> None`：初始化实例依赖、配置和运行状态。
- `M L43-L55` `_PassesAtIndex.evaluate(self, parameters: ParameterVector, trial_index: int) -> TrialObservation`：记录调用，并只让指定序号的 trial 通过全部门禁。 调用：`_observation`, `self.calls.append`。
- `F L58-L90` `test_run_search_resumes_after_committed_history_without_repeating() -> None`：恢复后首个 evaluator 调用必须从下一个连续 trial 开始。 调用：`_PassesAtIndex`, `_initial`, `run_search`。
- `N L64-L66` `test_run_search_resumes_after_committed_history_without_repeating._commit_then_interrupt(history: tuple[TrialObservation, ...]) -> None`：执行 `commit then interrupt` 对应逻辑。
- `F L93-L112` `test_search_history_store_round_trips_and_rejects_tampering(tmp_path) -> None` [IO-R IO-W]：状态文件必须同时校验历史摘要与 run/data/config 身份。 调用：`SearchHistoryStore`, `V2Config`, `_initial`, `_observation`, `store.load`, `store.persist`。
- `F L115-L136` `test_search_history_store_rejects_cross_run_resume(tmp_path) -> None`：相同路径也不能把另一运行的搜索进度接到当前运行。 调用：`SearchHistoryStore`, `V2Config`, `_initial`, `_observation`, `original.persist`, `other.load`。

## `src/traning/tracking/association/__init__.py`

职责：包导出边界；集中暴露该目录的稳定名称。
工程依赖：`traning.config`, `traning.contracts`

- `F L13-L17` `_require_finite_nonnegative(name: str, value: float) -> None`：执行 `require finite nonnegative` 对应逻辑。
- `F L20-L24` `_require_stable_id(name: str, value: str) -> None`：执行 `require stable id` 对应逻辑。
- `C L28-L50` `AssociationCostWeights` [CLASS]：三类关联成本的分组权重规格。
- `M L35-L44` `AssociationCostWeights.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_finite_nonnegative`。
- `M L47-L50` `AssociationCostWeights.total(self) -> float` [PROPERTY]：返回权重总和，供统一归一化公式使用。
- `C L54-L75` `AssociationCost` [CLASS]：单个轨迹候选配对的可解释成本。
- `M L63-L75` `AssociationCost.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_finite_nonnegative`。
- `C L79-L95` `TrackAssociationView` [CLASS]：关联层可见的最小轨迹投影，不携带训练真值。
- `M L86-L95` `TrackAssociationView.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_stable_id`。
- `C L99-L116` `AssociationMatch` [CLASS]：一个已接受的轨迹候选配对。
- `M L107-L116` `AssociationMatch.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_finite_nonnegative`, `_require_stable_id`。
- `C L120-L153` `AssociationResult` [CLASS]：一次一对一关联的完整确定性结果。
- `M L127-L153` `AssociationResult.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_stable_id`。
- `C L157-L162` `_CandidatePair` [CLASS]：greedy 求解器内部的候选配对。
- `F L165-L176` `_cosine_distance(first: tuple[float, ...], second: tuple[float, ...]) -> float`：执行 `cosine distance` 对应逻辑。
- `F L179-L185` `_type_values(distribution: ObjectTypeDistribution) -> tuple[float, ...]`：执行 `type values` 对应逻辑。
- `F L188-L197` `_object_type_distance(first: ObjectTypeDistribution, second: ObjectTypeDistribution) -> float`：用 total variation distance 比较两个四分类分布。 调用：`_type_values`。
- `C L201-L283` `AssociationCostSpec` [CLASS]：门限与权重的单一关联成本规格。
- `M L210-L222` `AssociationCostSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_finite_nonnegative`。
- `M L225-L243` `AssociationCostSpec.from_config(cls, config: TrackingConfig) -> AssociationCostSpec`：只从 TrackingConfig 读取运行门限。 调用：`AssociationCostWeights`。
- `M L245-L283` `AssociationCostSpec.cost(self, track: TrackAssociationView, candidate: CandidateObservation) -> AssociationCost | None`：计算合法配对成本；超过门限返回 None。 调用：`AssociationCost`, `_cosine_distance`, `_normalized_distance`, `_object_type_distance`。
- `C L287-L362` `GreedyAssociationSolver` [CLASS]：按稳定总序执行一对一 greedy 关联。
- `M L292-L294` `GreedyAssociationSolver.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L296-L362` `GreedyAssociationSolver.solve(self, tracks: Sequence[TrackAssociationView], candidates: Sequence[CandidateObservation]) -> AssociationResult`：生成全部合法配对，再以成本和稳定 ID 决胜。 调用：`AssociationCostSpec.from_config`, `AssociationMatch`, `AssociationResult`, `_CandidatePair`, `_checked_candidates`, `_checked_tracks`。
- `F L365-L376` `_checked_tracks(tracks: Sequence[TrackAssociationView]) -> tuple[TrackAssociationView, ...]`：执行 `checked tracks` 对应逻辑。
- `F L379-L384` `_normalized_distance(distance: float, maximum: float) -> float`：支持零门限：只有精确零距离能通过并贡献零成本。
- `F L387-L398` `_checked_candidates(candidates: Sequence[CandidateObservation]) -> tuple[CandidateObservation, ...]`：执行 `checked candidates` 对应逻辑。
- `F L401-L408` `associate_candidates(tracks: Sequence[TrackAssociationView], candidates: Sequence[CandidateObservation], config: TrackingConfig) -> AssociationResult`：确定性 greedy 关联的函数式入口。 调用：`GreedyAssociationSolver`, `GreedyAssociationSolver.solve`。

## `src/traning/tracking/tracker/tracker.py`

职责：执行确定性 association 和轨迹生命周期管理，维持稳定 track_id。
工程依赖：`traning.config`, `traning.contracts`, `traning.tracking.association`

- `C L24-L31` `_TrackState` [CLASS]：封装 `TrackState` 相关数据或行为。
- `C L34-L208` `MultiObjectTracker` [CLASS]：只使用当前及历史候选更新轨迹，不读取真值或未来帧。
- `M L37-L41` `MultiObjectTracker.__init__(self, config: TrackingConfig) -> None`：初始化实例依赖、配置和运行状态。 调用：`self.reset`。
- `M L43-L151` `MultiObjectTracker.update(self, candidates: Iterable[CandidateObservation], *, frame_id: str | None=None, frame_index: int | None=None, timestamp_ms: float | None=None) -> tuple[TrackedObservation, ...]`：关联一个已处理帧并推进轨迹。 调用：`TrackAssociationView`, `TrackedObservation`, `_index_candidates`, `_resolve_frame_context`, `_validate_association_result`, `associate_candidates`。
- `M L155-L158` `MultiObjectTracker.snapshot(self) -> tuple[TrackedObservation, ...]`：返回当前未过期轨迹的不可变有序观测快照。 调用：`self._ordered_states`。
- `M L160-L166` `MultiObjectTracker.reset(self) -> None`：清空状态并开始新的、从 1 递增的轨迹 ID 空间。
- `M L168-L175` `MultiObjectTracker._validate_monotonic_frame(self, frame_index: int, timestamp_ms: float) -> None`：校验 `monotonic frame` 对应的数据或结果。
- `M L177-L178` `MultiObjectTracker._ordered_states(self) -> tuple[_TrackState, ...]`：执行 `ordered states` 对应逻辑。
- `M L180-L208` `MultiObjectTracker._create_track(self, candidate: CandidateObservation, timestamp_ms: float) -> _TrackState`：执行 `create track` 对应逻辑。 调用：`TrackedObservation`, `_TrackState`。
- `F L211-L259` `_resolve_frame_context(candidates: tuple[CandidateObservation, ...], *, frame_id: str | None, frame_index: int | None, timestamp_ms: float | None) -> tuple[str, int, float]`：解析并定位 `frame context` 对应的数据或结果。
- `F L262-L272` `_index_candidates(candidates: tuple[CandidateObservation, ...]) -> dict[str, CandidateObservation]`：执行 `index candidates` 对应逻辑。
- `F L275-L300` `_validate_association_result(result: AssociationResult, *, track_ids: frozenset[str], candidate_ids: frozenset[str]) -> None`：校验 `association result` 对应的数据或结果。

## `src/traning/training/checkpoints.py`

职责：事务发布模型权重、数据 identity、模型契约摘要和坐标指纹。
工程依赖：`traning.belief`, `traning.config`, `traning.contracts`, `traning.contracts.common`, `traning.data`, `traning.infrastructure`, `traning.outcome`, `traning.perception`

- `C L57-L139` `RuntimeCheckpointManifest` [CLASS]：通用 ArtifactManifest 上的 runtime checkpoint 强约束视图。
- `M L62-L103` `RuntimeCheckpointManifest.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_transform_fingerprint`。
- `M L106-L112` `RuntimeCheckpointManifest.weights_filename(self) -> str` [PROPERTY]：返回 manifest 已提交的不可变权重文件名。
- `M L115-L121` `RuntimeCheckpointManifest.model_contract_sha256(self) -> str` [PROPERTY]：返回决定三个网络结构和输出语义的 canonical 摘要。
- `M L124-L130` `RuntimeCheckpointManifest.training_config_sha256(self) -> str` [PROPERTY]：返回发布权重时完整训练配置的审计摘要。
- `M L133-L139` `RuntimeCheckpointManifest.transform_fingerprint(self) -> str` [PROPERTY]：返回模型训练所绑定的坐标变换指纹。
- `C L143-L165` `RuntimeModelBundle` [CLASS]：已验证三模型及其 checkpoint/坐标身份的唯一 factory 输入。
- `M L152-L165` `RuntimeModelBundle.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_transform_fingerprint`, `self.artifact_id.strip`。
- `F L168-L224` `publish_runtime_checkpoint(directory: Path, config: V2Config, models: RuntimeModelBundle, coordinate_transform: FrameCoordinateTransform, *, dataset_id: str, producer_id: str, created_at_ms: float | None=None) -> RuntimeCheckpointManifest`：先原子发布权重 generation，再以 manifest 作为唯一提交点。 调用：`ArtifactManifest`, `RuntimeCheckpointManifest`, `_cpu_state_dict`, `_manifest_to_json`, `_model_contract_sha256`, `_training_config_sha256`。
- `F L227-L300` `load_runtime_checkpoint(directory: Path, config: V2Config, coordinate_transform: FrameCoordinateTransform, *, expected_dataset_id: str) -> RuntimeModelBundle` [IO-R]：校验 schema/config/坐标/摘要后，以 strict state dict 恢复三个模型。 调用：`DenseOutcomeModel`, `IntegrityError`, `PerTrackBeliefEncoder`, `PerceptionModel`, `RuntimeModelBundle`, `SchemaMismatchError`。
- `F L303-L323` `_validate_checkpoint_context(config: V2Config, models: RuntimeModelBundle, coordinate_transform: FrameCoordinateTransform) -> None`：在触碰磁盘前验证模型 config 和坐标身份完全一致。
- `F L326-L332` `_cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]`：复制为无梯度 CPU tensor，避免 checkpoint 绑定保存时设备。
- `F L335-L345` `_model_contract_sha256(config: V2Config) -> str`：只摘要实际决定三模型权重形状和输出语义的配置。 调用：`_json_sha256`, `v2_config_to_dict`。
- `F L348-L351` `_training_config_sha256(config: V2Config) -> str`：摘要发布时完整配置，用于追溯而不阻止部署侧路径调整。 调用：`_json_sha256`, `v2_config_to_dict`。
- `F L354-L364` `_json_sha256(payload: dict[str, object]) -> str`：计算严格 JSON object 的稳定 SHA-256。 调用：`hashlib.sha256`。
- `F L367-L382` `_manifest_to_json(manifest: RuntimeCheckpointManifest) -> dict[str, object]`：把 typed manifest 投影到唯一严格 JSON 边界。
- `F L385-L412` `_manifest_from_json(payload: dict[str, object]) -> RuntimeCheckpointManifest`：严格恢复 checkpoint manifest，拒绝未知字段和宽松强转。 调用：`ArtifactManifest`, `RuntimeCheckpointManifest`, `SchemaMismatchError`, `_integer`, `_number`, `_string`。
- `F L415-L419` `_string(payload: dict[str, object], key: str) -> str`：执行 `string` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L422-L426` `_integer(payload: dict[str, object], key: str) -> int`：执行 `integer` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L429-L433` `_number(payload: dict[str, object], key: str) -> float`：执行 `number` 对应逻辑。 调用：`SchemaMismatchError`。

## `src/traning/training/evaluator.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts`, `traning.data.pipeline`

- `C L30-L46` `_LazyTrialRunner` [CLASS]：把昂贵 runner 构造延迟到数据质量门实际通过之后。
- `M L38-L46` `_LazyTrialRunner.run(self, stage: TrainingStage) -> StageResult`：首次阶段调用时构造一次 runner，之后保持同一 trial 状态。 调用：`self._runner.run`, `self.factory`。
- `C L50-L98` `OrchestratedTrialEvaluator` [CLASS]：以质量门和真实阶段 runner 求值每个搜索 proposal。
- `M L61-L67` `OrchestratedTrialEvaluator.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L69-L98` `OrchestratedTrialEvaluator.evaluate(self, parameters: ParameterVector, trial_index: int) -> TrialObservation`：完整执行一个 proposal，并把失败门禁保留为可继续搜索的观测。 调用：`TrainingOrchestrator`, `TrainingOrchestrator.run`, `TrialObservation`, `_LazyTrialRunner`, `_acceptance_from_orchestration`, `require_quality`。
- `F L101-L120` `_acceptance_from_orchestration(result: OrchestrationResult) -> TrialAcceptance`：把阶段失败映成 gate，而不把普通未通过误当成程序异常。 调用：`TrialAcceptance`, `stage_status.get`。

## `src/traning/training/hard_examples.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.contracts`, `traning.evaluation.attribution`

- `C L14-L19` `HardExampleDestination(str, Enum)` [CLASS]：hard example 应反馈的模型领域。
- `C L22-L27` `HardExampleConsumer(str, Enum)` [CLASS]：共享同一 canonical event identity 的下游消费者。
- `C L30-L35` `HardExampleExclusionReason(str, Enum)` [CLASS]：未进入训练权重的显式审计原因。
- `F L38-L42` `_require_positive_weight(value: float) -> None`：执行 `require positive weight` 对应逻辑。
- `C L46-L58` `EvaluationSplitEvent` [CLASS]：canonical evaluation event 与唯一数据切分的组合。
- `M L52-L58` `EvaluationSplitEvent.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L62-L76` `HardExampleRouteSpec` [CLASS]：PrimaryError 到训练领域和基础权重的 registry 规格。
- `M L69-L76` `HardExampleRouteSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_positive_weight`。
- `C L100-L123` `HardExampleRoute` [CLASS]：一个 TRAIN hard example 的 canonical 领域路由。
- `M L106-L117` `HardExampleRoute.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L120-L123` `HardExampleRoute.event(self) -> SequenceEvaluationEvent` [PROPERTY]：直接返回原始 canonical event，不复制对象。
- `C L127-L142` `HardExampleWeight` [CLASS]：优化器使用的有限正 hard-example 权重。
- `M L133-L136` `HardExampleWeight.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_require_positive_weight`。
- `M L139-L142` `HardExampleWeight.event(self) -> SequenceEvaluationEvent` [PROPERTY]：保持 route 中 canonical event 的对象身份。
- `C L146-L167` `ExcludedHardExample` [CLASS]：未进入 TRAIN weights 的显式审计记录。
- `M L152-L161` `ExcludedHardExample.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L164-L167` `ExcludedHardExample.event(self) -> SequenceEvaluationEvent` [PROPERTY]：返回被排除的原始 canonical event，不复制也不改写归因。
- `C L171-L216` `HardExamplePlan` [CLASS]：稳定排序的训练权重和排除审计。
- `M L178-L206` `HardExamplePlan.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L208-L216` `HardExamplePlan.events_for(self, consumer: HardExampleConsumer) -> tuple[SequenceEvaluationEvent, ...]`：为 optimizer/telemetry/gallery 返回完全相同的 event 引用。
- `F L219-L277` `build_hard_example_plan(inputs: Sequence[EvaluationSplitEvent]) -> HardExamplePlan`：按 registry 路由失败 TRAIN event，并审计所有其余输入。 调用：`ExcludedHardExample`, `HardExamplePlan`, `HardExampleRoute`, `HardExampleWeight`。

## `src/traning/training/optimization.py`

职责：从有限量化参数规格持续提出未重复参数，并在全门禁通过或明确耗尽时终止。

- `C L16-L20` `ParameterType(str, Enum)` [CLASS]：参数 registry 支持的数值类型。
- `C L24-L105` `ParameterSpec` [CLASS]：单个参数的类型、闭区间和量化步长。
- `M L33-L57` `ParameterSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.name.strip`。
- `M L60-L64` `ParameterSpec.value_count(self) -> int` [PROPERTY]：返回闭区间按当前步长可表示的离散值数量。 调用：`_decimal`。
- `M L66-L76` `ParameterSpec.validate(self, value: float | int) -> None`：校验单值的数值类型、有限性与闭区间范围。
- `M L78-L93` `ParameterSpec.quantize(self, value: float | int) -> float | int`：先 clamp 再按 half-up 量化，绝不发布越界值。 调用：`_decimal`, `quantize`。
- `M L95-L105` `ParameterSpec.value_at(self, index: int) -> float | int`：按离散索引返回规格内精确量化后的参数值。 调用：`_decimal`。
- `C L109-L120` `ParameterVector` [CLASS]：搜索核心唯一允许的强类型参数向量。
- `M L119-L120` `ParameterVector.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`PARAMETER_REGISTRY.validate`。
- `C L124-L169` `ParameterRegistry` [CLASS]：按统一规格循环完成参数校验、clamp 和量化。
- `M L129-L137` `ParameterRegistry.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L140-L143` `ParameterRegistry.space_size(self) -> int` [PROPERTY]：返回所有参数离散取值笛卡尔积的总大小。
- `M L145-L149` `ParameterRegistry.validate(self, vector: ParameterVector) -> None`：按统一规格顺序校验完整 typed 参数向量。 调用：`spec.validate`。
- `M L151-L155` `ParameterRegistry.normalize(self, vector: ParameterVector) -> ParameterVector`：逐字段 clamp、量化并返回新的 canonical 参数向量。 调用：`ParameterVector`, `spec.quantize`。
- `M L157-L169` `ParameterRegistry.vector_at(self, flat_index: int) -> ParameterVector`：以混合进制解码扁平索引，确定性恢复参数向量。 调用：`ParameterVector`, `spec.value_at`。
- `C L196-L216` `TrialAcceptance` [CLASS]：所有阶段门禁的 typed 验收结果。
- `M L207-L210` `TrialAcceptance.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L213-L216` `TrialAcceptance.passed(self) -> bool` [PROPERTY]：仅当 registry 中全部训练与 golden 门禁通过时返回真。
- `C L220-L242` `TrialObservation` [CLASS]：一次已完成试验的参数、目标值与全门禁结果。
- `M L228-L242` `TrialObservation.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L245-L250` `SearchStatus(str, Enum)` [CLASS]：搜索控制器的非歧义状态。
- `C L254-L281` `SearchDecision` [CLASS]：搜索下一步 proposal 或显式终态。
- `M L262-L281` `SearchDecision.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L285-L355` `DeterministicSearchController` [CLASS]：按 seed 遍历未重复量化空间，直到全门禁通过或显式耗尽。
- `M L293-L308` `DeterministicSearchController.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L310-L355` `DeterministicSearchController.decide(self, initial: ParameterVector, history: tuple[TrialObservation, ...]) -> SearchDecision`：纯函数式地依据完整 history 返回下一 proposal 或明确终态。 调用：`SearchDecision`, `self.registry.normalize`, `self.registry.vector_at`。
- `C L358-L364` `TrialEvaluator(Protocol)` [CLASS]：run_search 消费的最小 typed evaluator。
- `M L361-L364` `TrialEvaluator.evaluate(self, parameters: ParameterVector, trial_index: int) -> TrialObservation`：执行一个 proposal 并返回同 index、同参数的 observation。
- `C L371-L378` `SearchExhaustedError(RuntimeError)` [CLASS]：预算或量化空间耗尽且未通过全门禁。
- `M L374-L378` `SearchExhaustedError.__init__(self, decision: SearchDecision) -> None`：初始化实例依赖、配置和运行状态。 调用：`super.__init__`。
- `F L381-L434` `run_search(evaluator: TrialEvaluator, initial: ParameterVector, *, seed: int=0, max_trials: int | None=None, registry: ParameterRegistry=PARAMETER_REGISTRY, history: tuple[TrialObservation, ...]=(), on_trial_completed: TrialCompletedCallback | None=None) -> TrialObservation`：持续或恢复求值；只有全门禁 PASSED 返回，耗尽时抛 typed error。 调用：`DeterministicSearchController`, `SearchExhaustedError`, `controller.decide`, `evaluator.evaluate`。
- `F L437-L438` `_decimal(value: float) -> Decimal`：执行 `decimal` 对应逻辑。

## `src/traning/training/orchestration.py`

职责：按规格表顺序执行 Perception、Tracking、Belief、Outcome、Decision、Evaluation 门禁。
工程依赖：`traning.contracts`, `traning.data.pipeline`, `traning.training.optimization`

- `C L14-L22` `TrainingStage(str, Enum)` [CLASS]：训练流水线的固定阶段。
- `C L35-L39` `ExecutionStatus(str, Enum)` [CLASS]：阶段与整体编排共享的终态。
- `C L43-L68` `StageResult` [CLASS]：单个训练阶段的明确结果。
- `M L51-L68` `StageResult.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.message.strip`。
- `C L71-L75` `StageRunner(Protocol)` [CLASS]：各领域训练实现必须满足的最小运行协议。
- `M L74-L75` `StageRunner.run(self, stage: TrainingStage) -> StageResult`：运行一个阶段并返回对应 typed 结果。
- `C L79-L121` `OrchestrationResult` [CLASS]：质量门、阶段执行与最终 acceptance 的整体审计结果。
- `M L87-L121` `OrchestrationResult.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L125-L177` `TrainingOrchestrator` [CLASS]：按固定 registry 顺序执行训练并在首个失败处停止。
- `M L130-L177` `TrainingOrchestrator.run(self, quality_report: DataQualityReport) -> OrchestrationResult`：先消费唯一质量门，再从 evaluation 取得 canonical acceptance。 调用：`OrchestrationResult`, `require_quality`, `self.runner.run`。

## `src/traning/training/production.py`

职责：生产训练总控；恢复搜索、运行 trial、发布并重新验证全门禁通过的 checkpoint。
工程依赖：`traning.config`, `traning.contracts`, `traning.data`, `traning.telemetry`, `traning.training.checkpoints`, `traning.training.evaluator`, `traning.training.optimization`, `traning.training.orchestration`, `traning.training.search_state`

- `C L44-L192` `ProductionTrainer` [CLASS]：把已检查的真实数据 bundle 接入可恢复的门禁驱动搜索。
- `M L51-L57` `ProductionTrainer.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L59-L192` `ProductionTrainer.run(self, *, run_dir: Path, run_id: str, resume: bool=True, reporter: TelemetryReporter | None=None) -> ProductionTrainingResult` [IO-W]：持续尝试未重复参数，只有全部门禁通过才发布并返回模型。 调用：`OrchestratedTrialEvaluator`, `ProductionTrainingResult`, `SearchHistoryStore`, `StateStore`, `TelemetryReporter`, `_completion_callback`。
- `N L105-L122` `ProductionTrainer.run.runner_factory(parameters: ParameterVector, trial_index: int) -> StageRunner`：为当前 proposal 构造独立且可审计的真实阶段 runner。 调用：`ProductionStageRunner`。
- `N L124-L134` `ProductionTrainer.run.objective_function(_parameters: ParameterVector, trial_index: int, _result: OrchestrationResult) -> float`：从同一 trial runner 中提取跨阶段汇总目标值。 调用：`runners.get`。
- `F L195-L205` `_initial_parameter_vector(config: V2Config) -> ParameterVector`：从唯一配置构造搜索空间的初始 proposal。 调用：`ParameterVector`。
- `F L208-L233` `_completion_callback(history_store: SearchHistoryStore, reporter: TelemetryReporter) -> Callable[[tuple[TrialObservation, ...]], None]`：返回先原子提交历史、再发布搜索事件的 completion callback。
- `N L214-L231` `_completion_callback.complete(history: tuple[TrialObservation, ...]) -> None`：原子保存完整搜索历史，并发布刚完成 trial 的事实事件。 调用：`TelemetryEvent`, `history_store.persist`, `reporter.publish`, `reporter.store.snapshot`。
- `F L236-L262` `_publish_search_terminal(reporter: TelemetryReporter, *, event_type: str, observation: TrialObservation | None, trial_count: int) -> None`：发布通过或耗尽终态，不把普通门禁失败伪装成进程停止。 调用：`TelemetryEvent`, `reporter.publish`, `reporter.store.snapshot`。

## `src/traning/training/production_contracts.py`

职责：Python 模块；具体职责见下方符号及调用。
工程依赖：`traning.config`, `traning.training.optimization`, `traning.training.orchestration`

- `F L14-L18` `_probability(name: str, value: float) -> None`：执行 `probability` 对应逻辑。
- `F L21-L25` `_nonnegative(name: str, value: float) -> None`：执行 `nonnegative` 对应逻辑。
- `C L29-L55` `ProductionGateSpec` [CLASS]：各阶段唯一使用的生产验收阈值。
- `M L41-L55` `ProductionGateSpec.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`_nonnegative`, `_probability`。
- `C L59-L97` `ProductionTrialMetrics` [CLASS]：一个 trial 从所有真实阶段累计得到的指标快照。
- `M L78-L83` `ProductionTrialMetrics.tracking_id_switch_rate(self) -> float` [PROPERTY]：按可连续比较的目标分配数归一化 ID switch。
- `M L86-L97` `ProductionTrialMetrics.objective(self) -> float` [PROPERTY]：越大越好的稳定多阶段搜索目标；门禁仍由布尔验收决定。
- `C L101-L127` `ProductionTrainingResult` [CLASS]：全门禁通过后返回的 winning trial 与已验证 checkpoint。
- `M L111-L127` `ProductionTrainingResult.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。

## `src/traning/training/production_stages.py`

职责：在真实 typed 数据上训练并评估六个生产阶段，统一应用每轮提案参数。
工程依赖：`traning.belief`, `traning.config`, `traning.contracts`, `traning.data`, `traning.decision`, `traning.evaluation`, `traning.infrastructure`, `traning.lib.runtime`, `traning.outcome`, `traning.perception`, `traning.telemetry`, `traning.tracking`, `traning.training.checkpoints`, `traning.training.optimization`, `traning.training.orchestration`

- `F L108-L117` `trial_checkpoint_directory(run_dir: Path, trial_index: int) -> Path`：返回 winning/non-winning trial 都不会互相覆盖的 checkpoint 目录。
- `F L120-L126` `_typed_sample_batch(values: list[TrainingSample]) -> tuple[TrainingSample, ...]`：DataLoader worker 使用的顶层可序列化 typed collate。
- `C L130-L822` `ProductionStageRunner` [CLASS]：一个参数 proposal 的有状态六阶段训练 runner。
- `M L149-L196` `ProductionStageRunner.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`DenseOutcomeModel`, `PerTrackBeliefEncoder`, `PerceptionModel`, `RuntimeModelBundle`, `config_for_parameters`, `configure_torch_runtime`。
- `M L198-L225` `ProductionStageRunner.run(self, stage: TrainingStage) -> StageResult`：从统一注册表选择阶段；普通门禁失败返回 FAILED 供搜索继续。 调用：`StageResult`, `self._publish_stage`, `self.stage_results.append`。
- `M L227-L278` `ProductionStageRunner._run_perception(self) -> StageResult`：用完整 RGB 帧训练所有 Perception head，并在 validation 解码召回。 调用：`PerceptionLossWeights`, `StageResult`, `_image_batch`, `_move_images`, `_sample_batches`, `autocast_context`。
- `M L280-L336` `ProductionStageRunner._evaluate_perception(self) -> tuple[float, float]`：执行 `evaluate perception` 对应逻辑。 调用：`PerceptionLossWeights`, `_image_batch`, `_match_positions`, `_move_images`, `_sample_batches`, `_slice_dense_output`。
- `M L338-L395` `ProductionStageRunner._run_tracking(self) -> StageResult`：在 validation 因果序列上测量真实候选关联的 ID switch。 调用：`MultiObjectTracker`, `StageResult`, `_match_positions`, `build_coordinate_training_targets`, `previous_track_by_object.get`, `self._infer_sample`。
- `M L397-L446` `ProductionStageRunner._run_belief(self) -> StageResult`：以带确定性观测噪声的 GT 轨迹监督 per-track GRU belief。 调用：`StageResult`, `_belief_records`, `autocast_context`, `belief_states_from_output`, `collate_belief_records`, `compute_belief_loss`。
- `M L448-L481` `ProductionStageRunner._evaluate_belief(self) -> float`：执行 `evaluate belief` 对应逻辑。 调用：`_belief_records`, `autocast_context`, `belief_states_from_output`, `collate_belief_records`, `encoder.forward_step`, `self.datasets.validation.iter_sequences`。
- `M L483-L534` `ProductionStageRunner._run_outcome(self) -> StageResult`：由 OutcomeOracle 在线生成反事实标签并训练 dense Outcome 模型。 调用：`StageResult`, `_outcome_batch_to_device`, `_outcome_record_batches`, `autocast_context`, `compute_outcome_loss`, `create_grad_scaler`。
- `M L536-L578` `ProductionStageRunner._evaluate_outcome(self) -> tuple[float, float, float, float]`：执行 `evaluate outcome` 对应逻辑。 调用：`_outcome_batch_to_device`, `_outcome_record_batches`, `evaluate_outcome_batch`。
- `M L580-L635` `ProductionStageRunner._run_decision(self) -> StageResult`：比较 learned planner 与同状态 oracle planner 的 CLICK/WAIT 决策。 调用：`OptimalStoppingPlanner`, `StageResult`, `_counterfactual_frames`, `_oracle_distribution`, `model.predict`, `planner.plan`。
- `M L637-L721` `ProductionStageRunner._run_evaluation(self) -> StageResult`：运行无 GT 泄漏的完整 runtime，再以 canonical scorer 生成 golden gate。 调用：`EvaluationEvent`, `FramePredictedClick`, `StageResult`, `_runtime_frame`, `_sequence_target`, `_timestamp_ms`。
- `M L724-L728` `ProductionStageRunner._coordinate_transform(self) -> FrameCoordinateTransform` [PROPERTY]：执行 `coordinate transform` 对应逻辑。
- `M L730-L740` `ProductionStageRunner._infer_sample(self, sample: TrainingSample) -> tuple[CandidateObservation, ...]`：执行 `infer sample` 对应逻辑。 调用：`_move_images`, `_runtime_frame`, `autocast_context`, `decode_runtime_output`, `runtime_frame_to_tensor`, `self.models.perception_model`。
- `M L742-L765` `ProductionStageRunner._acceptance(self, hit_rate: float) -> TrialAcceptance`：执行 `acceptance` 对应逻辑。 调用：`TrialAcceptance`。
- `M L767-L802` `ProductionStageRunner._publish_stage(self, result: StageResult, elapsed_seconds: float) -> None`：执行 `publish stage` 对应逻辑。 调用：`ResourceEvent`, `TelemetryEvent`, `_timestamp_ms`, `collect_memory_snapshot`, `self.reporter.publish`。
- `M L804-L822` `ProductionStageRunner._publish_metrics(self) -> None`：执行 `publish metrics` 对应逻辑。 调用：`MetricsEvent`, `_timestamp_ms`, `self.reporter.publish`。
- `F L825-L842` `config_for_parameters(config: V2Config, parameters: ParameterVector) -> V2Config` [IO-W]：一次性集体应用 proposal，禁止训练器和渲染器各改一部分参数。
- `F L845-L866` `_sample_batches(dataset: SegmentTrainingDataset, config: V2Config, *, shuffle: bool, seed: int) -> Iterator[tuple[TrainingSample, ...]]`：执行 `sample batches` 对应逻辑。
- `F L869-L877` `_runtime_frame(sample: TrainingSample) -> RuntimeFrame`：执行 `runtime frame` 对应逻辑。 调用：`RuntimeFrame`。
- `F L880-L888` `_image_batch(samples: tuple[TrainingSample, ...], config: V2Config) -> tuple[torch.Tensor, tuple[RuntimeTensorFrame, ...]]`：执行 `image batch` 对应逻辑。 调用：`_runtime_frame`, `runtime_frame_to_tensor`。
- `F L891-L905` `_move_images(images: torch.Tensor, device: torch.device, *, pin_memory: bool, channels_last: bool) -> torch.Tensor`：执行 `move images` 对应逻辑。 调用：`tensor_to_device`。
- `F L908-L924` `_slice_dense_output(output: DensePerceptionOutput, index: int) -> DensePerceptionOutput`：执行 `slice dense output` 对应逻辑。 调用：`DensePerceptionOutput`。
- `F L927-L954` `_match_positions(left: tuple[tuple[str, float, float], ...], right: tuple[tuple[str, float, float], ...], *, maximum_distance: float) -> tuple[tuple[str, str], ...]`：以全局距离排序做确定性一对一匹配，仅供离线指标计算。
- `F L957-L1014` `_belief_records(sample: TrainingSample, states: dict[str, BeliefState], encoder: PerTrackBeliefEncoder, transform: FrameCoordinateTransform, *, noise_px: float) -> tuple[BeliefTrainingRecord, ...]`：执行 `belief records` 对应逻辑。 调用：`BeliefTrainingRecord`, `CandidateObservation`, `Point2D`, `TrackedObservation`, `_deterministic_noise`, `_identity_embedding`。
- `F L1017-L1019` `_one_hot_type(object_type: ObjectType) -> ObjectTypeDistribution`：执行 `one hot type` 对应逻辑。 调用：`ObjectTypeDistribution`。
- `F L1022-L1029` `_identity_embedding(identity: str, dimension: int) -> tuple[float, ...]`：执行 `identity embedding` 对应逻辑。 调用：`hashlib.sha256`。
- `F L1032-L1037` `_deterministic_noise(identity: str, radius: float) -> tuple[float, float]`：执行 `deterministic noise` 对应逻辑。 调用：`hashlib.sha256`。
- `F L1040-L1099` `_counterfactual_frames(dataset: SegmentTrainingDataset, encoder: PerTrackBeliefEncoder, config: V2Config, transform: FrameCoordinateTransform) -> Iterator[tuple[TrainingSample, tuple[BeliefState, ...], tuple[OutcomeTrainingSample, ...]]]`：执行 `counterfactual frames` 对应逻辑。 调用：`CounterfactualFrame`, `CounterfactualOutcomeDatasetBuilder`, `CounterfactualOutcomeDatasetBuilder.build`, `OracleState`, `OutcomeOracle`, `_belief_records`。
- `F L1102-L1120` `_outcome_record_batches(dataset: SegmentTrainingDataset, encoder: PerTrackBeliefEncoder, config: V2Config, transform: FrameCoordinateTransform) -> Iterator[tuple[OutcomeTrainingSample, ...]]`：执行 `outcome record batches` 对应逻辑。 调用：`_counterfactual_frames`。
- `F L1123-L1145` `_outcome_batch_to_device(records: tuple[OutcomeTrainingSample, ...], belief_dim: int, transform_fingerprint: str | None, device: torch.device) -> OutcomeBatch` [IO-W]：执行 `outcome batch to device` 对应逻辑。 调用：`CounterfactualOutcomeDataset`, `collate_outcome_samples`, `tensor_to_device`。
- `F L1148-L1157` `_oracle_target(target: GroundTruthObject) -> OracleTarget`：执行 `oracle target` 对应逻辑。 调用：`OracleTarget`。
- `F L1160-L1174` `_oracle_distribution(sample: OutcomeTrainingSample) -> OutcomeDistribution`：执行 `oracle distribution` 对应逻辑。 调用：`OutcomeDistribution`。
- `F L1177-L1195` `_sequence_target(target: GroundTruthObject) -> TargetObject`：执行 `sequence target` 对应逻辑。 调用：`TargetObject`。
- `F L1198-L1199` `_timestamp_ms() -> float`：执行 `timestamp ms` 对应逻辑。

## `src/traning/training/scheduling.py`

职责：Python 模块；具体职责见下方符号及调用。

- `C L10-L16` `CurriculumStage(str, Enum)` [CLASS]：由简单到完整场景的固定 curriculum。
- `C L28-L42` `CurriculumGate` [CLASS]：一个可审计的 curriculum stage gate。
- `M L34-L42` `CurriculumGate.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.name.strip`。
- `C L45-L50` `CurriculumAction(str, Enum)` [CLASS]：curriculum 的明确调度动作。
- `C L54-L59` `CurriculumDecision` [CLASS]：当前 stage gate 的确定性决策。
- `F L62-L94` `decide_curriculum(current_stage: CurriculumStage, gates: tuple[CurriculumGate, ...]) -> CurriculumDecision`：仅在至少一个 gate 且全部通过时前进。 调用：`CurriculumDecision`。
- `C L98-L119` `AshaRung` [CLASS]：ASHA 的递增资源预算与晋级比例。
- `M L105-L119` `AshaRung.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L123-L147` `AshaTrial` [CLASS]：到达某 rung 的 trial 观测。
- `M L131-L147` `AshaTrial.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.trial_id.strip`。
- `C L150-L155` `AshaAction(str, Enum)` [CLASS]：ASHA 对 trial 的明确动作。
- `C L159-L165` `AshaDecision` [CLASS]：一个 trial 在当前 rung 的确定性动作。
- `C L169-L237` `AshaScheduler` [CLASS]：先执行严格 gate，再按 objective 与 trial_id 稳定排名。
- `M L174-L183` `AshaScheduler.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L185-L237` `AshaScheduler.decide(self, rung_index: int, trials: tuple[AshaTrial, ...]) -> tuple[AshaDecision, ...]`：同 rung gate 失败必剪枝；其余按 top fraction 晋级。 调用：`AshaDecision`。

## `src/traning/training/search_state.py`

职责：按 run、dataset 和 config identity 原子持久化可恢复搜索状态。
工程依赖：`traning.config`, `traning.contracts.common`, `traning.infrastructure`, `traning.training.optimization`

- `C L62-L95` `SearchHistoryState` [CLASS]：与运行、数据和完整配置身份绑定的不可变搜索历史。
- `M L72-L95` `SearchHistoryState.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`require_identifier`, `require_sha256`。
- `C L98-L157` `SearchHistoryStore` [CLASS]：把每个已完成 trial 作为原子恢复点保存到单一状态文件。
- `M L101-L122` `SearchHistoryStore.__init__(self, path: Path, *, run_id: str, dataset_id: str, config: V2Config, initial_parameters: ParameterVector) -> None`：初始化实例依赖、配置和运行状态。 调用：`require_identifier`, `training_config_sha256`。
- `M L124-L144` `SearchHistoryStore.load(self) -> tuple[TrialObservation, ...]`：不存在状态时从零开始；存在时必须通过全部身份与摘要校验。 调用：`SchemaMismatchError`, `_state_from_json`, `read_json_object`, `self.path.exists`。
- `M L146-L157` `SearchHistoryStore.persist(self, history: tuple[TrialObservation, ...]) -> None`：校验完整历史后原子覆盖状态；可直接作为搜索完成回调。 调用：`SearchHistoryState`, `_state_to_json`, `atomic_write_json`。
- `F L160-L166` `training_config_sha256(config: V2Config) -> str`：计算完整 V2 配置的稳定摘要，供恢复状态拒绝跨配置串用。 调用：`_canonical_json_bytes`, `hashlib.sha256`, `v2_config_to_dict`。
- `F L169-L180` `_state_to_json(state: SearchHistoryState) -> dict[str, object]`：执行 `state to json` 对应逻辑。 调用：`_canonical_json_bytes`, `_observation_to_json`, `_parameters_to_json`, `hashlib.sha256`。
- `F L183-L213` `_state_from_json(payload: dict[str, object]) -> SearchHistoryState`：执行 `state from json` 对应逻辑。 调用：`IntegrityError`, `SchemaMismatchError`, `SearchHistoryState`, `_canonical_json_bytes`, `_integer`, `_number`。
- `F L216-L224` `_observation_to_json(observation: TrialObservation) -> dict[str, object]`：执行 `observation to json` 对应逻辑。 调用：`_parameters_to_json`。
- `F L227-L252` `_observation_from_json(payload: object) -> TrialObservation`：执行 `observation from json` 对应逻辑。 调用：`SchemaMismatchError`, `TrialAcceptance`, `TrialObservation`, `_integer`, `_number`, `_parameters_from_json`。
- `F L255-L256` `_parameters_to_json(parameters: ParameterVector) -> dict[str, object]`：执行 `parameters to json` 对应逻辑。
- `F L259-L272` `_parameters_from_json(payload: dict[str, object]) -> ParameterVector`：执行 `parameters from json` 对应逻辑。 调用：`ParameterVector`, `SchemaMismatchError`, `_number`。
- `F L275-L285` `_canonical_json_bytes(payload: object) -> bytes`：执行 `canonical json bytes` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L288-L292` `_string(payload: dict[str, object], key: str) -> str`：执行 `string` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L295-L299` `_integer(payload: dict[str, object], key: str) -> int`：执行 `integer` 对应逻辑。 调用：`SchemaMismatchError`。
- `F L302-L306` `_number(payload: dict[str, object], key: str) -> float`：执行 `number` 对应逻辑。 调用：`SchemaMismatchError`。

## `src/traning/visualization/renderers.py`

职责：只消费不可变 telemetry snapshot 进行终端和 Qt 渲染。
工程依赖：`traning.contracts`, `traning.data.coordinates`, `traning.evaluation.attribution`, `traning.evaluation.sequence`, `traning.infrastructure`, `traning.telemetry.reporter`

- `C L43-L51` `DashboardSection(str, Enum)` [CLASS]：Rich 分区和 Qt 分组共用的稳定领域顺序。
- `C L54-L71` `DashboardMetric(str, Enum)` [CLASS]：Phase 10 必须可视化的完整指标集合。
- `C L74-L80` `QtMetricColumn(str, Enum)` [CLASS]：Qt 指标表的强类型列标识。
- `C L83-L93` `QtEvaluationColumn(str, Enum)` [CLASS]：Qt evaluation 表的强类型列标识。
- `C L97-L105` `_MetricSpec` [CLASS]：集中定义取值、标签和格式，避免 Rich/Qt 各自解释指标。
- `C L109-L139` `DashboardMetricRow` [CLASS]：两个 renderer 共用的不可变指标行。
- `M L120-L139` `DashboardMetricRow.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.label.strip`。
- `C L143-L198` `DashboardEvaluationRow` [CLASS]：直接持有 canonical event，确保 UI 不复制或重新归因。
- `M L148-L150` `DashboardEvaluationRow.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `M L153-L156` `DashboardEvaluationRow.event_id(self) -> str` [PROPERTY]：返回 canonical event identity。
- `M L159-L162` `DashboardEvaluationRow.sample_id(self) -> str` [PROPERTY]：返回 scorer 写入的样本标识。
- `M L165-L168` `DashboardEvaluationRow.frame_index(self) -> int` [PROPERTY]：返回 scorer 写入的帧序号。
- `M L171-L174` `DashboardEvaluationRow.passed(self) -> bool` [PROPERTY]：原样展示 canonical pass，不在 UI 重新计算。
- `M L177-L180` `DashboardEvaluationRow.primary_error(self) -> PrimaryError` [PROPERTY]：原样展示 canonical primary_error，不读取 error tag 猜测。
- `M L183-L186` `DashboardEvaluationRow.error_tags(self) -> tuple[EvaluationTag, ...]` [PROPERTY]：返回 canonical 次级标签。
- `M L189-L192` `DashboardEvaluationRow.target_id(self) -> str | None` [PROPERTY]：返回 scorer 绑定的目标标识。
- `M L195-L198` `DashboardEvaluationRow.click_index(self) -> int | None` [PROPERTY]：返回 scorer 绑定的点击序号。
- `C L202-L227` `GalleryTargetOverlay` [CLASS]：gallery 在原帧上绘制的强类型目标中心与 slider 路径。
- `M L209-L227` `GalleryTargetOverlay.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.target_id.strip`。
- `F L230-L277` `project_gallery_target_overlays(targets: tuple[TargetObject, ...], coordinate_transform: FrameCoordinateTransform) -> tuple[GalleryTargetOverlay, ...]`：用与训练和评分同一的变换投影 gallery 目标，不在 renderer 加偏移。 调用：`GalleryTargetOverlay`, `OsuPoint`, `coordinate_transform.target_to_gallery_overlay`。
- `C L281-L305` `GalleryPredictionOverlay` [CLASS]：原帧预测点及其 scorer 产生的原始 canonical 事件。
- `M L287-L305` `GalleryPredictionOverlay.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。
- `C L309-L355` `GalleryFrameOverlay` [CLASS]：一个原帧的 GT、预测和完整归因事件不可变集合。
- `M L319-L355` `GalleryFrameOverlay.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.transform_fingerprint.startswith`。
- `F L358-L413` `build_gallery_frame_overlay(targets: tuple[TargetObject, ...], score: FrameSequenceScore, events: tuple[SequenceEvaluationEvent, ...], coordinate_transform: FrameCoordinateTransform) -> GalleryFrameOverlay`：把 scorer 原始事件和同一坐标变换组合成可直接渲染的原帧 overlay。 调用：`GalleryFrameOverlay`, `GalleryPredictionOverlay`, `project_gallery_target_overlays`。
- `F L416-L460` `render_gallery_png(frame: RuntimeFrame, overlay: GalleryFrameOverlay, output_path: Path) -> None`：在原始 RGB 帧上绘制 GT 与预测点击，并原子发布真实 PNG 文件。 调用：`_draw_cross`, `atomic_write_bytes`, `image.save`。
- `F L463-L494` `_draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, *, color: tuple[int, int, int], radius: int) -> None`：用确定性整数像素绘制带外框的十字标记。
- `C L498-L519` `RichMetricSection` [CLASS]：Rich 页面中的一个稳定分区。
- `M L505-L519` `RichMetricSection.__post_init__(self) -> None`：完成 dataclass 初始化后的派生字段设置。 调用：`self.title.strip`。
- `C L523-L530` `RichDashboardModel` [CLASS]：不依赖 ``rich`` 包的终端 dashboard 纯 view-model。
- `C L534-L538` `QtMetricTableModel` [CLASS]：不依赖 Qt 运行时的指标表模型。
- `C L542-L546` `QtEvaluationTableModel` [CLASS]：不依赖 Qt 运行时的 canonical evaluation 表模型。
- `C L550-L557` `QtDashboardModel` [CLASS]：Qt 控件层可直接消费的不可变 dashboard 模型。
- `F L722-L729` `_format_metric(value: MetricNumber | None, precision: int) -> str`：仅格式化已经存在的 telemetry 值，不推导或填补业务指标。
- `F L732-L749` `_project_metric_rows(snapshot: DashboardSnapshot) -> tuple[DashboardMetricRow, ...]`：按唯一规格表产生稳定、有完整指标槽位的行。 调用：`DashboardMetricRow`, `_format_metric`。
- `F L752-L759` `_project_evaluations(snapshot: DashboardSnapshot) -> tuple[DashboardEvaluationRow, ...]`：保留 reporter snapshot 内 canonical event 的对象身份。 调用：`DashboardEvaluationRow`。
- `F L762-L766` `_require_snapshot(snapshot: DashboardSnapshot) -> None`：拒绝 mutable mapping/legacy state 等旁路输入。
- `C L769-L792` `RichDashboardRenderer` [CLASS]：把 snapshot 纯投影为终端分区模型；实例本身不保存状态。
- `M L773-L792` `RichDashboardRenderer.render(snapshot: DashboardSnapshot) -> RichDashboardModel`：返回确定性的不可变 Rich view-model。 调用：`RichDashboardModel`, `RichMetricSection`, `_project_evaluations`, `_project_metric_rows`, `_require_snapshot`。
- `C L795-L815` `QtDashboardRenderer` [CLASS]：把 snapshot 纯投影为 Qt 表模型；不导入或调用 Qt。
- `M L799-L815` `QtDashboardRenderer.render(snapshot: DashboardSnapshot) -> QtDashboardModel`：返回确定性的不可变 Qt view-model。 调用：`QtDashboardModel`, `QtEvaluationTableModel`, `QtMetricTableModel`, `_project_evaluations`, `_project_metric_rows`, `_require_snapshot`。
