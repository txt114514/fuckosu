"""以真实 segment 数据执行六个生产训练/评估阶段。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
import hashlib
from itertools import islice
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from package import CurriculumStage

from traning.core.belief import (
    BeliefTrainingRecord,
    PerTrackBeliefEncoder,
    belief_states_from_output,
    collate_belief_records,
    compute_belief_loss,
)
from traning.conf import V2Config
from traning.state import (
    AssociationStatus,
    BeliefState,
    CandidateObservation,
    DataSplit,
    DecisionAction,
    GroundTruthObject,
    ObjectType,
    ObjectTypeDistribution,
    OutcomeDistribution,
    OutcomeTrainingSample,
    Point2D,
    RuntimeFrame,
    TelemetryEvent,
    TrackLifecycle,
    TrackedObservation,
    TrainingSample,
)
from traning.core.data import (
    FrameCoordinateTransform,
    SegmentTrainingDataset,
    TrainingDatasetBundle,
    TrainingSequenceDataset,
)
from traning.core.decision import OptimalStoppingPlanner
from traning.core.evaluation import (
    FramePredictedClick,
    SequenceEvaluationEvent,
    TargetObject,
    build_sequence_evaluation_events,
    score_frame_click_sequence,
)
from traning.lib.infrastructure import seed_everything
from traning.lib.runtime import (
    autocast_context,
    collect_memory_snapshot,
    configure_torch_runtime,
    create_grad_scaler,
    module_to_device,
    tensor_to_device,
)
from traning.core.outcome import (
    CounterfactualFrame,
    CounterfactualOutcomeDataset,
    CounterfactualOutcomeDatasetBuilder,
    DenseOutcomeModel,
    OracleState,
    OracleTarget,
    OutcomeOracle,
    OutcomeBatch,
    collate_outcome_samples,
    compute_outcome_loss,
    evaluate_outcome_batch,
)
from traning.core.perception import (
    DensePerceptionOutput,
    PerceptionLossWeights,
    PerceptionModel,
    RuntimeTensorFrame,
    build_coordinate_training_targets,
    compute_perception_loss,
    decode_runtime_output,
    rasterize_perception_targets,
    runtime_frame_to_tensor,
)
from traning.lib.telemetry import (
    EvaluationEvent,
    MetricsEvent,
    ResourceEvent,
    TELEMETRY_SCHEMA_VERSION,
    TelemetryReporter,
)
from traning.core.tracking import MultiObjectTracker
from traning.core.training.checkpoints import (
    RuntimeModelBundle,
    load_runtime_checkpoint,
    publish_runtime_checkpoint,
)
from traning.core.training.optimization import ParameterVector, TrialAcceptance
from traning.core.training.orchestration import (
    ExecutionStatus,
    StageResult,
    TrainingStage,
)

from .production_contracts import ProductionGateSpec, ProductionTrialMetrics
from .curriculum_data import require_curriculum_dataset
from .hard_example_feedback import HardExampleFeedbackArtifact
from .hard_examples import (
    EvaluationSplitEvent,
    HardExampleDestination,
    HardExamplePlan,
    build_hard_example_plan,
)
from .gallery_artifacts import (
    ProductionGalleryRecord,
    publish_production_gallery_manifest,
    render_production_sequence_gallery,
)
from .production_schedule import ProductionTrialContext


def _typed_sample_batch(values: list[TrainingSample]) -> tuple[TrainingSample, ...]:
    """DataLoader worker 使用的顶层可序列化 typed collate。"""

    result = tuple(values)
    if any(not isinstance(item, TrainingSample) for item in result):
        raise TypeError("DataLoader 只能返回 TrainingSample")
    return result


@dataclass(frozen=True, slots=True)
class _RuntimeSplitEvaluation:
    """一次 TRAIN/VALIDATION runtime scorer 的内部汇总。"""

    split: DataSplit
    target_total: int
    hit_total: int
    events: tuple[SequenceEvaluationEvent, ...]

    def __post_init__(self) -> None:
        if self.split not in (DataSplit.TRAIN, DataSplit.VALIDATION):
            raise ValueError("runtime split summary 只允许 TRAIN/VALIDATION")
        if self.target_total < 0 or self.hit_total < 0:
            raise ValueError("runtime split 计数不得为负数")
        if self.hit_total > self.target_total:
            raise ValueError("hit_total 不得大于 target_total")


@dataclass(slots=True)
class ProductionStageRunner:
    """一个参数 proposal 的有状态六阶段训练 runner。"""

    base_config: V2Config
    context: ProductionTrialContext
    datasets: TrainingDatasetBundle
    gates: ProductionGateSpec
    run_dir: Path
    run_id: str
    reporter: TelemetryReporter
    input_feedback: HardExampleFeedbackArtifact | None = None
    metrics: ProductionTrialMetrics = field(default_factory=ProductionTrialMetrics)
    stage_results: list[StageResult] = field(default_factory=list)
    hard_example_plan: HardExamplePlan | None = field(default=None, init=False)
    feedback_evaluated: bool = field(default=False, init=False)
    config: V2Config = field(init=False)
    models: RuntimeModelBundle = field(init=False)
    train_dataset: SegmentTrainingDataset = field(init=False)
    validation_dataset: SegmentTrainingDataset = field(init=False)
    device: torch.device = field(init=False)
    amp_dtype: str | None = field(init=False)
    channels_last: bool = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.base_config, V2Config):
            raise TypeError("base_config 必须是 V2Config")
        if not isinstance(self.context, ProductionTrialContext):
            raise TypeError("context 必须是 ProductionTrialContext")
        if not isinstance(self.datasets, TrainingDatasetBundle):
            raise TypeError("datasets 必须是 TrainingDatasetBundle")
        if not isinstance(self.gates, ProductionGateSpec):
            raise TypeError("gates 必须是 ProductionGateSpec")
        if not isinstance(self.reporter, TelemetryReporter):
            raise TypeError("reporter 必须是 TelemetryReporter")
        if self.reporter.run_id != self.run_id:
            raise ValueError("reporter.run_id 与 runner.run_id 不一致")
        if self.context.rung_index >= len(self.base_config.optimization.asha_rungs):
            raise ValueError("context.rung_index 超出配置的 ASHA rungs")
        configured_budget = self.base_config.optimization.asha_rungs[
            self.context.rung_index
        ].budget_steps
        if self.context.budget_steps != configured_budget:
            raise ValueError("context.budget_steps 与 ASHA rung 配置不一致")
        if self.input_feedback is not None and not isinstance(
            self.input_feedback,
            HardExampleFeedbackArtifact,
        ):
            raise TypeError("input_feedback 必须是 HardExampleFeedbackArtifact 或 None")
        self.config = config_for_parameters(self.base_config, self.parameters)
        stage_offset = tuple(CurriculumStage).index(self.context.curriculum_stage)
        trial_seed = (
            self.config.training.seed
            + self.trial_index * 1009
            + stage_offset * 37
            + self.context.rung_index
        ) % (2**32)
        seed_everything(trial_seed)
        self.device = torch.device(self.config.runtime.device.value)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("生产配置要求 CUDA，但当前 PyTorch 看不到 CUDA")
        self.amp_dtype = "auto" if self.config.runtime.amp else None
        runtime_state = configure_torch_runtime(
            device=self.device,
            amp_dtype=self.amp_dtype,
        )
        self.channels_last = runtime_state.channels_last

        self.train_dataset = require_curriculum_dataset(
            self.datasets.train,
            self.context.curriculum_stage,
        )
        self.validation_dataset = require_curriculum_dataset(
            self.datasets.validation,
            self.context.curriculum_stage,
        )
        if self.context.parent_checkpoint_path is None:
            perception = PerceptionModel(self.config.perception)
            belief = PerTrackBeliefEncoder(
                self.config.belief,
                appearance_embedding_dim=self.config.perception.embedding_dim,
            )
            outcome = DenseOutcomeModel(
                self.config.outcome,
                belief_embedding_dim=belief.flattened_hidden_dim,
            )
        else:
            inherited = load_runtime_checkpoint(
                self.context.parent_checkpoint_path,
                self.config,
                self._coordinate_transform,
                expected_dataset_id=self.datasets.dataset_identity,
            )
            perception = inherited.perception_model
            belief = inherited.belief_encoder
            outcome = inherited.outcome_model
        module_to_device(perception, self.device, channels_last=self.channels_last)
        module_to_device(belief, self.device, channels_last=False)
        module_to_device(outcome, self.device, channels_last=False)
        transform = self.datasets.transform_fingerprint
        if transform is None:  # pragma: no cover - quality 门已阻断
            raise RuntimeError("训练数据缺少 transform fingerprint")
        self.models = RuntimeModelBundle(
            perception_model=perception,
            belief_encoder=belief,
            outcome_model=outcome,
            artifact_id=(
                f"runtime-{self.run_id}-trial-{self.trial_index:06d}-"
                f"{self.context.curriculum_stage.value}-r{self.context.rung_index:02d}"
            ),
            transform_fingerprint=transform,
        )

    @property
    def parameters(self) -> ParameterVector:
        """返回当前 job 所属唯一 proposal。"""

        return self.context.parameters

    @property
    def trial_index(self) -> int:
        """返回跨 curriculum/rung 保持不变的 proposal index。"""

        return self.context.trial_index

    @property
    def incremental_budget_steps(self) -> int:
        """把 ASHA 累计预算转换为本 job 只需新增的优化步数。"""

        if self.context.rung_index == 0:
            return self.context.budget_steps
        previous = self.config.optimization.asha_rungs[
            self.context.rung_index - 1
        ].budget_steps
        increment = self.context.budget_steps - previous
        if increment < 1:  # pragma: no cover - 配置模型已保证严格递增
            raise RuntimeError("ASHA 增量预算必须为正")
        return increment

    def run(self, stage: TrainingStage) -> StageResult:
        """从统一注册表选择阶段；普通门禁失败返回 FAILED 供搜索继续。"""

        handlers = {
            TrainingStage.PERCEPTION: self._run_perception,
            TrainingStage.TRACKING: self._run_tracking,
            TrainingStage.BELIEF: self._run_belief,
            TrainingStage.OUTCOME: self._run_outcome,
            TrainingStage.DECISION: self._run_decision,
            TrainingStage.EVALUATION: self._run_evaluation,
        }
        if not isinstance(stage, TrainingStage):
            raise TypeError("stage 必须是 TrainingStage")
        started = time.monotonic()
        # OOM/MemoryError 是换参数也未必可修复的运行边界损坏，必须作为异常交给
        # ProductionTrainer 发布 search.failed；普通指标门禁失败才返回 FAILED。
        result = handlers[stage]()
        if result.stage is not stage:
            raise ValueError("阶段 handler 返回了错误 stage")
        self.stage_results.append(result)
        elapsed = max(time.monotonic() - started, 1e-9)
        self._publish_stage(result, elapsed)
        return result

    def _run_perception(self) -> StageResult:
        """用完整 RGB 帧训练所有 Perception head，并在 validation 解码召回。"""

        model = self.models.perception_model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.parameters.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        scaler = create_grad_scaler(device=self.device, amp_dtype=self.amp_dtype)
        loss_weights = PerceptionLossWeights()
        model.train()
        sample_weights = self._sample_weight_vector(HardExampleDestination.PERCEPTION)
        for samples in _training_sample_batches(
            self.train_dataset,
            self.config,
            seed=self.config.training.seed,
            step_budget=self.incremental_budget_steps,
            sample_weights=sample_weights,
        ):
            images, _tensor_frames = _image_batch(samples, self.config)
            images = _move_images(
                images,
                self.device,
                pin_memory=self.config.data.loader.pin_memory,
                channels_last=self.channels_last,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.device, self.amp_dtype):
                prediction = model(images)
                targets = rasterize_perception_targets(
                    samples,
                    prediction,
                    self._coordinate_transform,
                )
                loss = compute_perception_loss(prediction, targets, loss_weights)
            scaler.scale(loss.total).backward()
            scaler.step(optimizer)
            scaler.update()
            self.metrics.training_steps += 1

        validation_loss, recall = self._evaluate_perception()
        self.metrics.perception_loss = validation_loss
        self.metrics.perception_recall = recall
        passed = recall >= self.gates.perception_min_recall
        return StageResult(
            stage=TrainingStage.PERCEPTION,
            status=ExecutionStatus.PASSED if passed else ExecutionStatus.FAILED,
            message=(
                f"validation loss={validation_loss:.6f}, recall={recall:.4f}, "
                f"required={self.gates.perception_min_recall:.4f}"
            ),
        )

    def _evaluate_perception(self) -> tuple[float, float]:
        model = self.models.perception_model
        model.eval()
        loss_total = 0.0
        batch_count = 0
        target_count = 0
        matched_count = 0
        with torch.inference_mode():
            for samples in _sample_batches(
                self.validation_dataset,
                self.config,
                shuffle=False,
                seed=self.config.training.seed,
            ):
                images, tensor_frames = _image_batch(samples, self.config)
                images = _move_images(
                    images,
                    self.device,
                    pin_memory=self.config.data.loader.pin_memory,
                    channels_last=self.channels_last,
                )
                with autocast_context(self.device, self.amp_dtype):
                    prediction = model(images)
                    targets = rasterize_perception_targets(
                        samples,
                        prediction,
                        self._coordinate_transform,
                    )
                    loss = compute_perception_loss(
                        prediction,
                        targets,
                        PerceptionLossWeights(),
                    )
                loss_total += float(loss.total.detach().float().cpu())
                batch_count += 1
                for index, (sample, tensor_frame) in enumerate(
                    zip(samples, tensor_frames, strict=True)
                ):
                    candidates = decode_runtime_output(
                        tensor_frame,
                        _slice_dense_output(prediction, index),
                        self.config.perception,
                    )
                    ground_truth = build_coordinate_training_targets(
                        sample,
                        self._coordinate_transform,
                    )
                    matches = _match_positions(
                        tuple(
                            (item.object_id, item.position.x, item.position.y)
                            for item in ground_truth
                        ),
                        tuple(
                            (item.candidate_id, item.x, item.y) for item in candidates
                        ),
                        maximum_distance=self.config.tracking.max_distance_px,
                    )
                    target_count += len(ground_truth)
                    matched_count += len(matches)
        mean_loss = loss_total / max(batch_count, 1)
        recall = matched_count / target_count if target_count else 0.0
        return mean_loss, recall

    def _run_tracking(self) -> StageResult:
        """在 validation 因果序列上测量真实候选关联的 ID switch。"""

        model = self.models.perception_model
        model.eval()
        switches = 0
        assignments = 0
        with torch.inference_mode():
            for sequence in self.validation_dataset.iter_sequences():
                tracker = MultiObjectTracker(self.config.tracking)
                previous_track_by_object: dict[str, str] = {}
                for sample in sequence:
                    candidates = self._infer_sample(sample)
                    tracks = tracker.update(
                        candidates,
                        frame_id=sample.sample_id,
                        frame_index=sample.frame_index,
                        timestamp_ms=sample.timestamp_ms,
                    )
                    visible_tracks = tuple(
                        item for item in tracks if item.candidate is not None
                    )
                    ground_truth = build_coordinate_training_targets(
                        sample,
                        self._coordinate_transform,
                    )
                    matches = _match_positions(
                        tuple(
                            (item.object_id, item.position.x, item.position.y)
                            for item in ground_truth
                        ),
                        tuple(
                            (
                                item.track_id,
                                item.candidate.x,
                                item.candidate.y,
                            )
                            for item in visible_tracks
                            if item.candidate is not None
                        ),
                        maximum_distance=self.config.tracking.max_distance_px,
                    )
                    for object_id, track_id in matches:
                        previous = previous_track_by_object.get(object_id)
                        if previous is not None:
                            assignments += 1
                            if previous != track_id:
                                switches += 1
                        previous_track_by_object[object_id] = track_id
        self.metrics.tracking_id_switches = switches
        self.metrics.tracking_assignments = assignments
        rate = self.metrics.tracking_id_switch_rate
        passed = assignments > 0 and rate <= self.gates.tracking_max_id_switch_rate
        return StageResult(
            stage=TrainingStage.TRACKING,
            status=ExecutionStatus.PASSED if passed else ExecutionStatus.FAILED,
            message=(
                f"id_switches={switches}, comparable_assignments={assignments}, "
                f"rate={rate:.6f}, required<={self.gates.tracking_max_id_switch_rate:.6f}"
            ),
        )

    def _run_belief(self) -> StageResult:
        """以带确定性观测噪声的 GT 轨迹监督 per-track GRU belief。"""

        encoder = self.models.belief_encoder
        optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=self.parameters.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        scaler = create_grad_scaler(device=self.device, amp_dtype=self.amp_dtype)
        encoder.train()
        for sequence, sample in _training_sequence_frames(
            self.train_dataset,
            seed=self.config.training.seed,
            step_budget=self.incremental_budget_steps,
        ):
            states: dict[str, BeliefState] = {}
            # 为当前采样帧重放同一因果序列的前缀，避免从 future frame 初始化 hidden。
            for causal_sample in sequence:
                sample_is_target = causal_sample.frame_index == sample.frame_index
                records = _belief_records(
                    causal_sample,
                    states,
                    encoder,
                    self._coordinate_transform,
                    noise_px=8.0,
                )
                if not records:
                    if sample_is_target:
                        break
                    continue
                batch = collate_belief_records(encoder, records)
                with autocast_context(self.device, self.amp_dtype):
                    output = encoder.forward_step(
                        batch.observation_features,
                        batch.previous_hidden,
                    )
                    loss = compute_belief_loss(output, batch)
                for state in belief_states_from_output(output, batch):
                    states[state.track_id] = state
                if sample_is_target:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss.total).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    self.metrics.training_steps += 1
                    break

        mae = self._evaluate_belief()
        self.metrics.belief_position_mae_px = mae
        passed = mae <= self.gates.belief_max_position_mae_px
        return StageResult(
            stage=TrainingStage.BELIEF,
            status=ExecutionStatus.PASSED if passed else ExecutionStatus.FAILED,
            message=(
                f"position_mae_px={mae:.6f}, "
                f"required<={self.gates.belief_max_position_mae_px:.6f}"
            ),
        )

    def _evaluate_belief(self) -> float:
        encoder = self.models.belief_encoder
        encoder.eval()
        absolute_error = 0.0
        coordinate_count = 0
        with torch.inference_mode():
            for sequence in self.validation_dataset.iter_sequences():
                states: dict[str, BeliefState] = {}
                for sample in sequence:
                    records = _belief_records(
                        sample,
                        states,
                        encoder,
                        self._coordinate_transform,
                        noise_px=8.0,
                    )
                    if not records:
                        continue
                    batch = collate_belief_records(encoder, records)
                    with autocast_context(self.device, self.amp_dtype):
                        output = encoder.forward_step(
                            batch.observation_features,
                            batch.previous_hidden,
                        )
                    absolute_error += float(
                        (output.position_mean.float() - batch.target_positions.float())
                        .abs()
                        .sum()
                        .cpu()
                    )
                    coordinate_count += 2 * len(records)
                    for state in belief_states_from_output(output, batch):
                        states[state.track_id] = state
        return absolute_error / max(coordinate_count, 1)

    def _run_outcome(self) -> StageResult:
        """由 OutcomeOracle 在线生成反事实标签并训练 dense Outcome 模型。"""

        model = self.models.outcome_model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.parameters.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        scaler = create_grad_scaler(device=self.device, amp_dtype=self.amp_dtype)
        model.train()
        frame_weights = self._feedback_weight_map(HardExampleDestination.OUTCOME)
        for records, record_weights in _training_outcome_batches(
            self.train_dataset,
            self.models.belief_encoder,
            self.config,
            self._coordinate_transform,
            step_budget=self.incremental_budget_steps,
            frame_weights=frame_weights,
        ):
            batch = _outcome_batch_to_device(
                records,
                self.models.belief_encoder.flattened_hidden_dim,
                self.datasets.transform_fingerprint,
                self.device,
            )
            sample_weights = torch.tensor(
                record_weights,
                dtype=batch.belief_embeddings.dtype,
                device=batch.belief_embeddings.device,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.device, self.amp_dtype):
                output = model(batch.belief_embeddings, batch.horizon_ms)
                loss = compute_outcome_loss(
                    output,
                    batch,
                    sample_weights=sample_weights,
                )
            scaler.scale(loss.total).backward()
            scaler.step(optimizer)
            scaler.update()
            self.metrics.training_steps += 1

        nll, brier, ece, score_mae = self._evaluate_outcome()
        self.metrics.outcome_nll = nll
        self.metrics.outcome_brier = brier
        self.metrics.outcome_ece = ece
        self.metrics.expected_score_mae = score_mae
        passed = (
            nll <= self.gates.outcome_max_nll
            and brier <= self.gates.outcome_max_brier
            and ece <= self.gates.outcome_max_ece
        )
        return StageResult(
            stage=TrainingStage.OUTCOME,
            status=ExecutionStatus.PASSED if passed else ExecutionStatus.FAILED,
            message=(
                f"nll={nll:.6f}/{self.gates.outcome_max_nll:.6f}, "
                f"brier={brier:.6f}/{self.gates.outcome_max_brier:.6f}, "
                f"ece={ece:.6f}/{self.gates.outcome_max_ece:.6f}"
            ),
        )

    def _evaluate_outcome(self) -> tuple[float, float, float, float]:
        model = self.models.outcome_model
        model.eval()
        totals = [0.0, 0.0, 0.0, 0.0]
        record_count = 0
        with torch.inference_mode():
            for records, _record_weights in _outcome_record_batches(
                self.validation_dataset,
                self.models.belief_encoder,
                self.config,
                self._coordinate_transform,
            ):
                batch = _outcome_batch_to_device(
                    records,
                    self.models.belief_encoder.flattened_hidden_dim,
                    self.datasets.transform_fingerprint,
                    self.device,
                )
                # canonical metrics 当前保证 float32/64 标量；评估关闭 autocast。
                output = model(batch.belief_embeddings, batch.horizon_ms)
                metrics = evaluate_outcome_batch(
                    output,
                    batch,
                    calibration_bins=self.config.outcome.calibration_bins,
                )
                weight = len(records)
                values = (
                    metrics.multiclass_nll,
                    metrics.multiclass_brier,
                    metrics.calibration_error,
                    metrics.expected_score_mae,
                )
                for index, value in enumerate(values):
                    totals[index] += float(value.cpu()) * weight
                record_count += weight
        if record_count == 0:
            return 1e6, 1e6, 1.0, 1e6
        return (
            totals[0] / record_count,
            totals[1] / record_count,
            totals[2] / record_count,
            totals[3] / record_count,
        )

    def _run_decision(self) -> StageResult:
        """比较 learned planner 与同状态 oracle planner 的 CLICK/WAIT 决策。"""

        planner = OptimalStoppingPlanner(self.config.decision)
        model = self.models.outcome_model
        model.eval()
        agreement, utility, wait_count, click_count, frame_count = (
            self._decision_statistics(
                self.validation_dataset,
                planner,
                model,
                frame_weights=None,
            )
        )
        hard_weights = _applicable_frame_weights(
            self.train_dataset,
            self._feedback_weight_map(HardExampleDestination.DECISION),
        )
        if hard_weights:
            hard_agreement, _hard_utility, _hard_wait, _hard_click, hard_count = (
                self._decision_statistics(
                    self.train_dataset,
                    planner,
                    model,
                    frame_weights=hard_weights,
                )
            )
        else:
            hard_agreement = 1.0
            hard_count = 0
        self.metrics.decision_oracle_agreement = agreement
        self.metrics.decision_hard_agreement = hard_agreement
        self.metrics.decision_utility = utility
        self.metrics.wait_click_ratio = wait_count / max(click_count, 1)
        self.metrics.hard_example_count = len(
            () if self.input_feedback is None else self.input_feedback.frame_weights
        )
        passed = (
            agreement >= self.gates.decision_min_oracle_agreement
            and hard_agreement >= self.gates.decision_min_oracle_agreement
        )
        return StageResult(
            stage=TrainingStage.DECISION,
            status=ExecutionStatus.PASSED if passed else ExecutionStatus.FAILED,
            message=(
                f"oracle_agreement={agreement:.6f}, frames={frame_count}, "
                f"hard_agreement={hard_agreement:.6f}, hard_frames={hard_count}, "
                f"required>={self.gates.decision_min_oracle_agreement:.6f}"
            ),
        )

    def _decision_statistics(
        self,
        dataset: SegmentTrainingDataset,
        planner: OptimalStoppingPlanner,
        model: DenseOutcomeModel,
        *,
        frame_weights: dict[tuple[str, int], float] | None,
    ) -> tuple[float, float, int, int, int]:
        """在完整因果序列上计算普通或 TRAIN-only 难例加权决策指标。"""

        agreement_weight = 0.0
        total_weight = 0.0
        frame_count = 0
        wait_count = 0
        click_count = 0
        utility_total = 0.0
        with torch.inference_mode():
            for sequence_id, sample, beliefs, records in _counterfactual_frames(
                dataset,
                self.models.belief_encoder,
                self.config,
                self._coordinate_transform,
            ):
                if not beliefs:
                    continue
                weight = (
                    1.0
                    if frame_weights is None
                    else frame_weights.get((sequence_id, sample.frame_index), 0.0)
                )
                if weight <= 0.0:
                    continue
                required = {0.0, planner.wait_horizon_ms}
                selected_records = tuple(
                    item for item in records if float(item.horizon_ms) in required
                )
                oracle_outcomes = tuple(
                    _oracle_distribution(item) for item in selected_records
                )
                learned_outcomes = tuple(
                    model.predict(belief, horizon)
                    for belief in beliefs
                    for horizon in (0.0, planner.wait_horizon_ms)
                )
                learned = planner.plan(beliefs, learned_outcomes, sample.timestamp_ms)
                oracle = planner.plan(beliefs, oracle_outcomes, sample.timestamp_ms)
                matched = (
                    learned.action is oracle.action
                    and learned.track_id == oracle.track_id
                )
                agreement_weight += weight * float(matched)
                total_weight += weight
                frame_count += 1
                utility_total += weight * learned.expected_utility
                if learned.action is DecisionAction.WAIT:
                    wait_count += 1
                else:
                    click_count += 1
        agreement = agreement_weight / total_weight if total_weight else 0.0
        utility = utility_total / total_weight if total_weight else 0.0
        return (
            agreement,
            utility,
            wait_count,
            click_count,
            frame_count,
        )

    def _run_evaluation(self) -> StageResult:
        """运行无 GT 泄漏的完整 runtime，再以 canonical scorer 生成 golden gate。"""

        # 延迟导入应用装配层，保持 training 领域模块初始化时不反向依赖 app。
        from traning.core.app.factory import assemble_runtime_pipeline

        pipeline = assemble_runtime_pipeline(self.config, models=self.models)
        train_summary = self._evaluate_runtime_split(
            pipeline,
            self.train_dataset,
            split=DataSplit.TRAIN,
            gallery_directory=None,
            publish_telemetry=False,
            max_sequences=self.context.budget_steps,
        )
        gallery_directory = (
            self.run_dir
            / "trials"
            / f"trial-{self.trial_index:06d}"
            / "jobs"
            / self.context.curriculum_stage.value
            / f"rung-{self.context.rung_index:02d}"
            / "gallery"
        )
        validation_summary = self._evaluate_runtime_split(
            pipeline,
            self.validation_dataset,
            split=DataSplit.VALIDATION,
            gallery_directory=gallery_directory,
            publish_telemetry=True,
            max_sequences=(
                None if self._is_full_terminal_job else self.context.budget_steps
            ),
        )
        plan_inputs = tuple(
            EvaluationSplitEvent(event, DataSplit.TRAIN)
            for event in train_summary.events
        ) + tuple(
            EvaluationSplitEvent(event, DataSplit.VALIDATION)
            for event in validation_summary.events
        )
        self.hard_example_plan = build_hard_example_plan(plan_inputs)
        self.feedback_evaluated = True
        hit_rate = (
            validation_summary.hit_total / validation_summary.target_total
            if validation_summary.target_total
            else 0.0
        )
        self.metrics.golden_hit_rate = hit_rate
        acceptance = self._acceptance(hit_rate)
        self._publish_metrics()
        return StageResult(
            stage=TrainingStage.EVALUATION,
            status=ExecutionStatus.PASSED,
            message=(
                f"golden_hit_rate={hit_rate:.6f}, "
                f"targets={validation_summary.target_total}, "
                f"train_feedback_events={len(train_summary.events)}, "
                f"required>={self.gates.golden_min_hit_rate:.6f}"
            ),
            acceptance=acceptance,
        )

    def _evaluate_runtime_split(
        self,
        pipeline: object,
        dataset: SegmentTrainingDataset,
        *,
        split: DataSplit,
        gallery_directory: Path | None,
        publish_telemetry: bool,
        max_sequences: int | None,
    ) -> _RuntimeSplitEvaluation:
        """用同一 runtime/scorer 评估一个 split，并保持事件真实帧身份。"""

        if split not in (DataSplit.TRAIN, DataSplit.VALIDATION):
            raise ValueError("生产搜索只允许评估 TRAIN 或 VALIDATION")
        target_total = 0
        hit_total = 0
        all_events = []
        gallery_records: list[ProductionGalleryRecord] = []
        sequences = dataset.iter_sequences()
        selected_sequences = (
            sequences if max_sequences is None else islice(sequences, max_sequences)
        )
        for sequence in selected_sequences:
            reset = getattr(pipeline, "reset", None)
            step = getattr(pipeline, "step", None)
            if not callable(reset) or not callable(step):
                raise TypeError("runtime pipeline 必须实现 reset/step")
            reset()
            targets_by_id: dict[str, tuple[GroundTruthObject, int]] = {}
            clicks: list[FramePredictedClick] = []
            runtime_frames: list[RuntimeFrame] = []
            for sample in sequence:
                for target in sample.ground_truth_objects:
                    if target.object_type in (ObjectType.RING, ObjectType.SLIDER):
                        targets_by_id.setdefault(
                            target.object_id,
                            (target, sample.frame_index),
                        )
                runtime_frame = _runtime_frame(sample)
                runtime_frames.append(runtime_frame)
                result = step(runtime_frame)
                if result.decision.action is DecisionAction.CLICK:
                    position = result.decision.target_position
                    if position is None:  # pragma: no cover - DecisionResult 已保证
                        raise RuntimeError("CLICK 决策缺少 target_position")
                    clicks.append(
                        FramePredictedClick(
                            time_ms=result.decision.execute_at_ms,
                            position=self._coordinate_transform.bind_frame_prediction(
                                x=position.x,
                                y=position.y,
                                source_frame_width=sample.width,
                                source_frame_height=sample.height,
                            ),
                            frame_index=sample.frame_index,
                        )
                    )
            targets = tuple(
                _sequence_target(item, frame_index=frame_index)
                for item, frame_index in targets_by_id.values()
            )
            if not targets:
                continue
            circle_radii = tuple(
                item.radius_osu
                for item, _frame_index in targets_by_id.values()
                if item.radius_osu is not None
            )
            circle_radius = float(circle_radii[0]) if circle_radii else 32.0
            score = score_frame_click_sequence(
                targets,
                tuple(clicks),
                coordinate_transform=self._coordinate_transform,
                circle_radius=circle_radius,
            )
            target_total += len(targets)
            hit_total += score.result.hit_count
            events = build_sequence_evaluation_events(
                sequence.sequence_id,
                sequence[-1].frame_index,
                score,
            )
            all_events.extend(events)
            if publish_telemetry:
                for event in events:
                    self.reporter.publish(
                        EvaluationEvent(
                            schema_version=TELEMETRY_SCHEMA_VERSION,
                            timestamp_ms=_timestamp_ms(),
                            run_id=self.run_id,
                            event=event,
                        )
                    )
            if gallery_directory is not None:
                gallery_records.extend(
                    render_production_sequence_gallery(
                        gallery_directory,
                        sequence_id=sequence.sequence_id,
                        frames=tuple(runtime_frames),
                        targets=targets,
                        score=score,
                        events=events,
                        coordinate_transform=self._coordinate_transform,
                    )
                )
        ordered_gallery_records = tuple(
            sorted(
                gallery_records,
                key=lambda record: (record.sequence_id, record.frame_index),
            )
        )
        if gallery_directory is not None:
            publish_production_gallery_manifest(
                gallery_directory,
                run_id=self.run_id,
                dataset_id=self.datasets.dataset_identity,
                trial_index=self.trial_index,
                transform_fingerprint=self._coordinate_transform.transform_fingerprint,
                records=ordered_gallery_records,
            )
        return _RuntimeSplitEvaluation(
            split=split,
            target_total=target_total,
            hit_total=hit_total,
            events=tuple(all_events),
        )

    @property
    def _is_full_terminal_job(self) -> bool:
        """仅 FULL 的最后一个 rung 有资格执行完整 validation。"""

        return (
            self.context.curriculum_stage is CurriculumStage.FULL
            and self.context.rung_index == len(self.config.optimization.asha_rungs) - 1
        )

    def _feedback_weight_map(
        self,
        destination: HardExampleDestination,
    ) -> dict[tuple[str, int], float]:
        """把已校验 artifact 投影为当前领域的 canonical 帧 multiplier。"""

        if self.input_feedback is None:
            return {}
        return {
            item.identity: item.effective_weight
            for item in self.input_feedback.weights_for(destination)
        }

    def _sample_weight_vector(
        self,
        destination: HardExampleDestination,
    ) -> tuple[float, ...] | None:
        """把 sequence/frame feedback 精确映射到当前课程数据集索引。"""

        frame_weights = _applicable_frame_weights(
            self.train_dataset,
            self._feedback_weight_map(destination),
        )
        if not frame_weights:
            return None
        weights = [1.0] * len(self.train_dataset)
        for (sequence_id, frame_index), weight in sorted(frame_weights.items()):
            location = self.train_dataset.resolve_sequence_frame(
                sequence_id,
                frame_index,
            )
            weights[location.dataset_index] = max(
                weights[location.dataset_index],
                weight,
            )
        return tuple(weights)

    def publish_job_checkpoint(self, directory: Path) -> None:
        """为当前 curriculum/rung 发布可供下一增量 job 继承的模型权重。"""

        publish_runtime_checkpoint(
            directory,
            self.config,
            self.models,
            self._coordinate_transform,
            dataset_id=self.datasets.dataset_identity,
            producer_id=(
                f"production-trial-{self.trial_index:06d}-"
                f"{self.context.curriculum_stage.value}-"
                f"r{self.context.rung_index:02d}"
            ),
        )

    @property
    def _coordinate_transform(self) -> FrameCoordinateTransform:
        transform = self.datasets.coordinate_transform
        if transform is None:  # pragma: no cover - quality 门已阻断
            raise RuntimeError("训练数据缺少 coordinate transform")
        return transform

    def _infer_sample(self, sample: TrainingSample) -> tuple[CandidateObservation, ...]:
        tensor_frame = runtime_frame_to_tensor(
            _runtime_frame(sample), self.config.perception
        )
        image = _move_images(
            tensor_frame.image,
            self.device,
            pin_memory=self.config.data.loader.pin_memory,
            channels_last=self.channels_last,
        )
        with autocast_context(self.device, self.amp_dtype):
            output = self.models.perception_model(image)
        return decode_runtime_output(tensor_frame, output, self.config.perception)

    def _acceptance(self, hit_rate: float) -> TrialAcceptance:
        return TrialAcceptance(
            data=self.datasets.quality_report.ok,
            perception=self.metrics.perception_recall
            >= self.gates.perception_min_recall,
            tracking=(
                self.metrics.tracking_assignments > 0
                and self.metrics.tracking_id_switch_rate
                <= self.gates.tracking_max_id_switch_rate
            ),
            belief=(
                self.metrics.belief_position_mae_px
                <= self.gates.belief_max_position_mae_px
            ),
            outcome=(
                self.metrics.outcome_nll <= self.gates.outcome_max_nll
                and self.metrics.outcome_brier <= self.gates.outcome_max_brier
                and self.metrics.outcome_ece <= self.gates.outcome_max_ece
            ),
            decision=(
                self.metrics.decision_oracle_agreement
                >= self.gates.decision_min_oracle_agreement
                and self.metrics.decision_hard_agreement
                >= self.gates.decision_min_oracle_agreement
            ),
            golden=hit_rate >= self.gates.golden_min_hit_rate,
            schedule=True,
        )

    def _publish_stage(self, result: StageResult, elapsed_seconds: float) -> None:
        self.reporter.publish(
            TelemetryEvent(
                schema_version=TELEMETRY_SCHEMA_VERSION,
                event_type="training.stage.completed",
                timestamp_ms=_timestamp_ms(),
                run_id=self.run_id,
                metrics=(("elapsed_seconds", elapsed_seconds),),
                payload=(
                    ("budget_steps", self.context.budget_steps),
                    ("cohort_index", self.context.cohort_index),
                    ("curriculum_stage", self.context.curriculum_stage.value),
                    ("message", result.message),
                    ("rung_index", self.context.rung_index),
                    ("stage", result.stage.value),
                    ("status", result.status.value),
                    ("trial_index", self.trial_index),
                ),
            )
        )
        memory = collect_memory_snapshot()
        if self.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.device)
            total_mb = properties.total_memory / (1024**2)
            used_mb = (memory.current_allocated_gib or 0.0) * 1024.0
        else:
            total_mb = 0.0
            used_mb = 0.0
        self.reporter.publish(
            ResourceEvent(
                schema_version=TELEMETRY_SCHEMA_VERSION,
                timestamp_ms=_timestamp_ms(),
                run_id=self.run_id,
                step=self.trial_index * 6 + len(self.stage_results) - 1,
                gpu_utilization=0.0,
                vram_used_mb=used_mb,
                vram_total_mb=total_mb,
                throughput=1.0 / elapsed_seconds,
            )
        )

    def _publish_metrics(self) -> None:
        self.reporter.publish(
            MetricsEvent(
                schema_version=TELEMETRY_SCHEMA_VERSION,
                timestamp_ms=_timestamp_ms(),
                run_id=self.run_id,
                step=self.metrics.training_steps,
                loss=self.metrics.perception_loss,
                perception_recall=self.metrics.perception_recall,
                tracking_id_switches=self.metrics.tracking_id_switches,
                outcome_nll=self.metrics.outcome_nll,
                outcome_brier=self.metrics.outcome_brier,
                outcome_ece=self.metrics.outcome_ece,
                expected_score_error=self.metrics.expected_score_mae,
                decision_utility=self.metrics.decision_utility,
                wait_click_ratio=self.metrics.wait_click_ratio,
                score=self.metrics.golden_hit_rate,
            )
        )


def config_for_parameters(config: V2Config, parameters: ParameterVector) -> V2Config:
    """一次性集体应用 proposal，禁止训练器和渲染器各改一部分参数。"""

    return replace(
        config,
        perception=replace(
            config.perception,
            score_threshold=parameters.score_threshold,
            max_candidates=parameters.max_candidates,
        ),
        decision=replace(
            config.decision,
            risk_lambda=parameters.risk_lambda,
            wait_cost=parameters.wait_cost,
            min_confidence=parameters.min_confidence,
        ),
        training=replace(config.training, learning_rate=parameters.learning_rate),
    )


def _sample_batches(
    dataset: SegmentTrainingDataset,
    config: V2Config,
    *,
    shuffle: bool,
    seed: int,
    sample_weights: tuple[float, ...] | None = None,
) -> Iterator[tuple[TrainingSample, ...]]:
    generator = torch.Generator().manual_seed(seed)
    sampler: WeightedRandomSampler[int] | None = None
    if sample_weights is not None:
        if len(sample_weights) != len(dataset):
            raise ValueError("sample_weights 长度必须等于 dataset 长度")
        if any(not math.isfinite(value) or value <= 0.0 for value in sample_weights):
            raise ValueError("sample_weights 必须全部是有限正数")
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=config.data.loader.workers,
        pin_memory=False,
        collate_fn=_typed_sample_batch,
        generator=generator,
        drop_last=False,
    )
    for batch in loader:
        if not isinstance(batch, tuple):
            raise TypeError("typed collate 必须返回 tuple")
        yield batch


def _training_sample_batches(
    dataset: SegmentTrainingDataset,
    config: V2Config,
    *,
    seed: int,
    step_budget: int,
    sample_weights: tuple[float, ...] | None,
) -> Iterator[tuple[TrainingSample, ...]]:
    """跨确定性 epoch 重放数据，精确产生 ASHA 本 job 的优化步数。"""

    if step_budget < 1:
        raise ValueError("step_budget 必须为正")
    emitted = 0
    epoch = 0
    while emitted < step_budget:
        progressed = False
        for batch in _sample_batches(
            dataset,
            config,
            shuffle=True,
            seed=seed + epoch,
            sample_weights=sample_weights,
        ):
            progressed = True
            yield batch
            emitted += 1
            if emitted >= step_budget:
                break
        if not progressed:
            raise RuntimeError("curriculum TRAIN dataset 无法产生 perception batch")
        epoch += 1


def _applicable_frame_weights(
    dataset: SegmentTrainingDataset,
    frame_weights: dict[tuple[str, int], float],
) -> dict[tuple[str, int], float]:
    """只保留当前累计课程数据集中真实存在的反馈帧。

    新 cohort 会从 BASIC 重新开始，上一 cohort 的最终反馈可能来自 FULL 的
    ``long_sequence``。这些帧不是 BASIC 的失败，也不能让 BASIC 的 Decision hard gate
    因“零个可评估帧”被误判为 0 分；进入相应累计课程阶段后它们会再次自然生效。
    """

    applicable = {}
    for identity, weight in sorted(frame_weights.items()):
        try:
            dataset.resolve_sequence_frame(*identity)
        except KeyError:
            continue
        applicable[identity] = weight
    return applicable


def _training_sequence_frames(
    dataset: SegmentTrainingDataset,
    *,
    seed: int,
    step_budget: int,
) -> Iterator[tuple[TrainingSequenceDataset, TrainingSample]]:
    """确定性选择含监督目标的序列帧，供 belief 产生精确优化步。"""

    if step_budget < 1:
        raise ValueError("step_budget 必须为正")
    sequences = tuple(dataset.iter_sequences())
    if not sequences:
        raise RuntimeError("curriculum TRAIN dataset 没有 belief sequence")
    emitted = 0
    epoch = 0
    while emitted < step_budget:
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(sequences), generator=generator).tolist()
        progressed = False
        for sequence_index in order:
            sequence = sequences[sequence_index]
            for sample in sequence:
                if not sample.ground_truth_objects:
                    continue
                progressed = True
                yield sequence, sample
                emitted += 1
                if emitted >= step_budget:
                    return
        if not progressed:
            raise RuntimeError("curriculum TRAIN dataset 没有 belief 监督目标")
        epoch += 1


def _runtime_frame(sample: TrainingSample) -> RuntimeFrame:
    return RuntimeFrame(
        frame_id=sample.sample_id,
        frame_index=sample.frame_index,
        timestamp_ms=sample.timestamp_ms,
        width=sample.width,
        height=sample.height,
        image_bytes=sample.image_bytes,
    )


def _image_batch(
    samples: tuple[TrainingSample, ...],
    config: V2Config,
) -> tuple[torch.Tensor, tuple[RuntimeTensorFrame, ...]]:
    frames = tuple(
        runtime_frame_to_tensor(_runtime_frame(sample), config.perception)
        for sample in samples
    )
    return torch.cat(tuple(frame.image for frame in frames), dim=0), frames


def _move_images(
    images: torch.Tensor,
    device: torch.device,
    *,
    pin_memory: bool,
    channels_last: bool,
) -> torch.Tensor:
    if pin_memory and device.type == "cuda" and not images.is_pinned():
        images = images.pin_memory()
    return tensor_to_device(
        images,
        device,
        channels_last=channels_last,
        non_blocking=True,
    )


def _slice_dense_output(
    output: DensePerceptionOutput, index: int
) -> DensePerceptionOutput:
    values = {
        name: getattr(output, name)[index : index + 1]
        for name in (
            "center_logits",
            "visibility_logits",
            "type_logits",
            "xy_offsets",
            "ring_logits",
            "ring_radius",
            "slider_logits",
            "slider_direction",
            "spinner_logits",
            "identity_embedding",
        )
    }
    return DensePerceptionOutput(**values, stride=output.stride)


def _match_positions(
    left: tuple[tuple[str, float, float], ...],
    right: tuple[tuple[str, float, float], ...],
    *,
    maximum_distance: float,
) -> tuple[tuple[str, str], ...]:
    """以全局距离排序做确定性一对一匹配，仅供离线指标计算。"""

    edges = sorted(
        (
            (math.hypot(lx - rx, ly - ry), left_id, right_id)
            for left_id, lx, ly in left
            for right_id, rx, ry in right
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    used_left: set[str] = set()
    used_right: set[str] = set()
    matches: list[tuple[str, str]] = []
    for distance, left_id, right_id in edges:
        if distance > maximum_distance:
            break
        if left_id in used_left or right_id in used_right:
            continue
        used_left.add(left_id)
        used_right.add(right_id)
        matches.append((left_id, right_id))
    return tuple(matches)


def _belief_records(
    sample: TrainingSample,
    states: dict[str, BeliefState],
    encoder: PerTrackBeliefEncoder,
    transform: FrameCoordinateTransform,
    *,
    noise_px: float,
) -> tuple[BeliefTrainingRecord, ...]:
    objects = {item.object_id: item for item in sample.ground_truth_objects}
    targets = build_coordinate_training_targets(sample, transform)
    records: list[BeliefTrainingRecord] = []
    for target in targets:
        ground_truth = objects[target.object_id]
        previous = states.get(target.object_id)
        delta_x, delta_y = _deterministic_noise(
            f"{sample.sample_id}|{target.object_id}",
            noise_px,
        )
        candidate = CandidateObservation(
            frame_id=sample.sample_id,
            frame_index=sample.frame_index,
            timestamp_ms=sample.timestamp_ms,
            candidate_id=f"teacher:{sample.sample_id}:{target.object_id}",
            x=min(max(target.position.x + delta_x, 0.0), sample.width - 1.0),
            y=min(max(target.position.y + delta_y, 0.0), sample.height - 1.0),
            confidence=1.0,
            visibility_probability=1.0,
            object_type_distribution=_one_hot_type(ground_truth.object_type),
            appearance_embedding=_identity_embedding(
                target.object_id,
                encoder.appearance_embedding_dim,
            ),
        )
        created = previous is None
        observation = TrackedObservation(
            track_id=target.object_id,
            frame_id=sample.sample_id,
            frame_index=sample.frame_index,
            timestamp_ms=sample.timestamp_ms,
            lifecycle=TrackLifecycle.NEW if created else TrackLifecycle.ACTIVE,
            association=AssociationStatus.CREATED
            if created
            else AssociationStatus.MATCHED,
            association_confidence=1.0,
            track_age=1 if created else previous.age + 1,
            missed_frames=0,
            time_since_seen_ms=0.0,
            candidate=candidate,
            association_cost=None if created else 0.0,
        )
        records.append(
            BeliefTrainingRecord(
                observation=observation,
                previous=previous,
                target_position=Point2D(target.position.x, target.position.y),
                target_visibility=1.0,
                target_object_type=ground_truth.object_type,
            )
        )
    return tuple(records)


def _one_hot_type(object_type: ObjectType) -> ObjectTypeDistribution:
    values = tuple(1.0 if object_type is item else 0.0 for item in ObjectType)
    return ObjectTypeDistribution(*values)


def _identity_embedding(identity: str, dimension: int) -> tuple[float, ...]:
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    values = [
        ((digest[index % len(digest)] / 255.0) * 2.0 - 1.0)
        for index in range(dimension)
    ]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:  # pragma: no cover - SHA-256 向量不可能全为中点
        values[0] = 1.0
        norm = 1.0
    return tuple(value / norm for value in values)


def _deterministic_noise(identity: str, radius: float) -> tuple[float, float]:
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return (
        radius * ((digest[0] / 255.0) * 2.0 - 1.0),
        radius * ((digest[1] / 255.0) * 2.0 - 1.0),
    )


def _counterfactual_frames(
    dataset: SegmentTrainingDataset,
    encoder: PerTrackBeliefEncoder,
    config: V2Config,
    transform: FrameCoordinateTransform,
) -> Iterator[
    tuple[
        str,
        TrainingSample,
        tuple[BeliefState, ...],
        tuple[OutcomeTrainingSample, ...],
    ]
]:
    encoder.eval()
    with torch.inference_mode():
        for sequence in dataset.iter_sequences():
            states: dict[str, BeliefState] = {}
            for sample in sequence:
                records = _belief_records(
                    sample,
                    states,
                    encoder,
                    transform,
                    noise_px=0.0,
                )
                if not records:
                    continue
                batch = collate_belief_records(encoder, records)
                output = encoder.forward_step(
                    batch.observation_features,
                    batch.previous_hidden,
                )
                beliefs = belief_states_from_output(output, batch)
                for state in beliefs:
                    states[state.track_id] = state
                oracle_targets = tuple(
                    _oracle_target(item) for item in sample.ground_truth_objects
                )
                oracle_state = OracleState(
                    state_id=f"oracle:{sample.sample_id}",
                    timestamp_ms=sample.timestamp_ms,
                    targets=oracle_targets,
                )
                circle_radii = tuple(
                    item.radius_osu
                    for item in sample.ground_truth_objects
                    if item.radius_osu is not None
                )
                oracle = OutcomeOracle(
                    circle_radius=float(circle_radii[0]) if circle_radii else 32.0
                )
                counterfactual = CounterfactualOutcomeDatasetBuilder(
                    oracle,
                    tuple(float(item) for item in config.outcome.horizons_ms),
                    transform,
                ).build(
                    (
                        CounterfactualFrame(
                            sample_id=sample.sample_id,
                            split=sample.split,
                            source_frame_width=sample.width,
                            source_frame_height=sample.height,
                            transform_fingerprint=sample.transform_fingerprint,
                            beliefs=beliefs,
                            oracle_state=oracle_state,
                        ),
                    )
                )
                yield sequence.sequence_id, sample, beliefs, counterfactual.records


def _outcome_record_batches(
    dataset: SegmentTrainingDataset,
    encoder: PerTrackBeliefEncoder,
    config: V2Config,
    transform: FrameCoordinateTransform,
    *,
    frame_weights: dict[tuple[str, int], float] | None = None,
) -> Iterator[tuple[tuple[OutcomeTrainingSample, ...], tuple[float, ...]]]:
    pending: list[OutcomeTrainingSample] = []
    pending_weights: list[float] = []
    for sequence_id, sample, _beliefs, records in _counterfactual_frames(
        dataset,
        encoder,
        config,
        transform,
    ):
        pending.extend(records)
        weight = (
            1.0
            if frame_weights is None
            else frame_weights.get((sequence_id, sample.frame_index), 1.0)
        )
        pending_weights.extend(weight for _record in records)
        while len(pending) >= config.training.batch_size:
            yield (
                tuple(pending[: config.training.batch_size]),
                tuple(pending_weights[: config.training.batch_size]),
            )
            del pending[: config.training.batch_size]
            del pending_weights[: config.training.batch_size]
    if pending:
        yield tuple(pending), tuple(pending_weights)


def _training_outcome_batches(
    dataset: SegmentTrainingDataset,
    encoder: PerTrackBeliefEncoder,
    config: V2Config,
    transform: FrameCoordinateTransform,
    *,
    step_budget: int,
    frame_weights: dict[tuple[str, int], float],
) -> Iterator[tuple[tuple[OutcomeTrainingSample, ...], tuple[float, ...]]]:
    """跨确定性重放精确生成 Outcome 的 ASHA 增量优化步。"""

    if step_budget < 1:
        raise ValueError("step_budget 必须为正")
    emitted = 0
    while emitted < step_budget:
        progressed = False
        for records, weights in _outcome_record_batches(
            dataset,
            encoder,
            config,
            transform,
            frame_weights=frame_weights,
        ):
            progressed = True
            yield records, weights
            emitted += 1
            if emitted >= step_budget:
                return
        if not progressed:
            raise RuntimeError("curriculum TRAIN dataset 无法产生 Outcome batch")


def _outcome_batch_to_device(
    records: tuple[OutcomeTrainingSample, ...],
    belief_dim: int,
    transform_fingerprint: str | None,
    device: torch.device,
) -> OutcomeBatch:
    if transform_fingerprint is None:
        raise RuntimeError("Outcome batch 缺少坐标指纹")
    dataset = CounterfactualOutcomeDataset(
        split=records[0].split,
        records=records,
        transform_fingerprint=transform_fingerprint,
    )
    batch = collate_outcome_samples(dataset, belief_embedding_dim=belief_dim)
    return replace(
        batch,
        belief_embeddings=tensor_to_device(
            batch.belief_embeddings, device, channels_last=False
        ),
        horizon_ms=tensor_to_device(batch.horizon_ms, device, channels_last=False),
        category_targets=tensor_to_device(
            batch.category_targets, device, channels_last=False
        ),
        expiry_targets=tensor_to_device(
            batch.expiry_targets, device, channels_last=False
        ),
        score_targets=tensor_to_device(
            batch.score_targets, device, channels_last=False
        ),
        valid_targets=tensor_to_device(
            batch.valid_targets, device, channels_last=False
        ),
    )


def _oracle_target(target: GroundTruthObject) -> OracleTarget:
    return OracleTarget(
        track_id=target.object_id,
        object_id=target.object_id,
        object_type=target.object_type,
        position=target.position,
        start_time_ms=target.start_time_ms,
        end_time_ms=target.end_time_ms,
        path=target.path,
    )


def _oracle_distribution(sample: OutcomeTrainingSample) -> OutcomeDistribution:
    probabilities = [0.0] * 5
    probabilities[int(sample.target_category)] = 1.0
    return OutcomeDistribution(
        track_id=sample.belief.track_id,
        horizon_ms=sample.horizon_ms,
        p_invalid=probabilities[0],
        p_miss=probabilities[1],
        p_low_score=probabilities[2],
        p_medium_score=probabilities[3],
        p_high_score=probabilities[4],
        p_expire=float(sample.expires),
        expected_score=sample.target_score,
        variance=0.0,
    )


def _sequence_target(
    target: GroundTruthObject,
    *,
    frame_index: int,
) -> TargetObject:
    """把首次可见 GT 目标及其真实来源帧绑定到 sequence scorer。"""

    if target.object_type is ObjectType.RING:
        return TargetObject(
            target_id=target.object_id,
            target_type="circle",
            start_ms=target.start_time_ms,
            end_ms=target.end_time_ms,
            x=target.position.x,
            y=target.position.y,
            frame_index=frame_index,
        )
    if target.object_type is ObjectType.SLIDER:
        return TargetObject(
            target_id=target.object_id,
            target_type="slider",
            start_ms=target.start_time_ms,
            end_ms=target.end_time_ms,
            path=tuple((point.x, point.y) for point in target.path),
            frame_index=frame_index,
        )
    raise ValueError("sequence scorer 只支持 ring/slider")


def _timestamp_ms() -> float:
    return time.time_ns() / 1_000_000.0


__all__ = (
    "ProductionStageRunner",
    "config_for_parameters",
)
