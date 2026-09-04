"""验收 production curriculum、ASHA、恢复和 hard-example 的真实接线。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest
from package import AffineOsuVideoTransform, CurriculumStage

from traning.conf import (
    AshaRungConfig,
    CoordinateConfig,
    OptimizationConfig,
    RuntimeConfig,
    RuntimeDevice,
    V2Config,
)
from traning.state import DataQualityReport, DataSplit
from traning.core.data import (
    FrameCoordinateTransform,
    SegmentTrainingDataset,
    TrainingDatasetBundle,
)
from traning.core.evaluation import EvaluationTag, PrimaryError, SequenceEvaluationEvent
from traning.lib.infrastructure import IntegrityError
from traning.lib.data.annotation import SegmentAnnotation
from traning.lib.data.models import SegmentRecord
from traning.core.training import (
    ExecutionStatus,
    HardExampleDestination,
    HardExamplePlan,
    ProductionGateSpec,
    ProductionTrainer,
    ProductionTrialMetrics,
    SearchExhaustedError,
    StageResult,
    TrainingStage,
    TrialAcceptance,
    build_hard_example_plan,
)
from traning.core.training.hard_example_feedback import HardExampleFeedbackArtifact
from traning.core.training.hard_examples import EvaluationSplitEvent
from traning.core.training.production_schedule import ProductionTrialContext


_AFFINE_MATRIX = (
    (2.0, 0.0, 100.0),
    (0.0, 2.0, 40.0),
)


@dataclass(frozen=True, slots=True)
class _RunnerInvocation:
    """一次 fake production job 的输入快照。"""

    context: ProductionTrialContext
    incremental_budget_steps: int
    feedback: HardExampleFeedbackArtifact | None


class _ScheduledStageRunner:
    """替代昂贵模型但完整实现 production job runner 协议。"""

    invocations: list[_RunnerInvocation] = []
    fail_cohorts: frozenset[int] = frozenset()
    interrupt_key: tuple[int, CurriculumStage, int] | None = None

    def __init__(
        self,
        *,
        base_config: V2Config,
        context: ProductionTrialContext,
        datasets: TrainingDatasetBundle,
        gates: ProductionGateSpec,
        run_dir: Path,
        run_id: str,
        reporter: object,
        input_feedback: HardExampleFeedbackArtifact | None,
    ) -> None:
        """记录生产输入；测试不会读取视频或创建模型。"""

        del datasets, gates, run_id, reporter
        self.base_config = base_config
        self.context = context
        self.run_dir = run_dir
        previous_budget = (
            0
            if context.rung_index == 0
            else base_config.optimization.asha_rungs[
                context.rung_index - 1
            ].budget_steps
        )
        incremental_budget = context.budget_steps - previous_budget
        type(self).invocations.append(
            _RunnerInvocation(context, incremental_budget, input_feedback)
        )
        objective_signal = 0.90 - 0.05 * context.trial_index
        self.metrics = ProductionTrialMetrics(
            perception_recall=objective_signal,
            decision_oracle_agreement=objective_signal,
            golden_hit_rate=objective_signal,
            training_steps=incremental_budget,
        )
        self.stage_results: list[StageResult] = []
        self.hard_example_plan = None
        self.feedback_evaluated = False

    def run(self, stage: TrainingStage) -> StageResult:
        """产生领域门禁与 canonical TRAIN/validation/test 反馈事件。"""

        key = (
            self.context.trial_index,
            self.context.curriculum_stage,
            self.context.rung_index,
        )
        if key == type(self).interrupt_key:
            raise RuntimeError("intentional-job-interruption")
        if stage is TrainingStage.EVALUATION:
            self.hard_example_plan = _hard_example_plan(self.context)
            self.feedback_evaluated = True
            domain_passed = self.context.cohort_index not in type(self).fail_cohorts
            full_terminal = (
                self.context.curriculum_stage is CurriculumStage.FULL
                and self.context.rung_index
                == len(self.base_config.optimization.asha_rungs) - 1
            )
            acceptance = TrialAcceptance(
                data=True,
                perception=True,
                tracking=True,
                belief=True,
                outcome=True,
                decision=True,
                golden=domain_passed,
                schedule=domain_passed and full_terminal,
            )
            result = StageResult(
                stage=stage,
                status=ExecutionStatus.PASSED,
                acceptance=acceptance,
            )
        else:
            result = StageResult(stage=stage, status=ExecutionStatus.PASSED)
        self.stage_results.append(result)
        return result

    def publish_job_checkpoint(self, directory: Path) -> None:
        """发布最小可摘要 job checkpoint，供 parent 链与篡改检查使用。"""

        directory.mkdir(parents=True, exist_ok=False)
        payload = (
            f"trial={self.context.trial_index};"
            f"stage={self.context.curriculum_stage.value};"
            f"rung={self.context.rung_index};"
            f"budget={self.context.budget_steps}"
        )
        (directory / "fake-checkpoint.txt").write_text(payload, encoding="utf-8")


def _config(*, max_trials: int | None) -> V2Config:
    """构造两级 ASHA、两 proposal cohort 的 CPU 测试配置。"""

    return V2Config(
        coordinates=CoordinateConfig(
            transform_identity="production-feedback-schedule-test",
            affine_matrix=_AFFINE_MATRIX,
        ),
        runtime=RuntimeConfig(
            device=RuntimeDevice.CPU,
            require_cuda=False,
            amp=False,
        ),
        optimization=OptimizationConfig(
            max_trials=max_trials,
            cohort_size=2,
            asha_rungs=(
                AshaRungConfig(budget_steps=1, promotion_fraction=0.5),
                AshaRungConfig(budget_steps=3, promotion_fraction=0.5),
            ),
            hard_example_bonus=1.0,
            hard_example_max_weight=4.0,
        ),
    )


def _segment_record(tmp_path: Path, split: DataSplit) -> SegmentRecord:
    """构造含 frame 0 的 atomic/single_point segment，不触发视频读取。"""

    sequence_id = f"{split.value}-sequence"
    directory = tmp_path / split.value
    annotation = SegmentAnnotation(
        schema_version=1,
        segment_id=sequence_id,
        dataset_dimension="atomic",
        category="single_point",
        difficulty={
            "approach_preempt_ms": 600.0,
            "circle_radius_osu_pixels": 32.0,
        },
        source={
            "folder_name": "fixture",
            "osu_filename": "fixture.osu",
            "clip_start_ms": 0,
            "clip_end_ms": 100,
        },
        hit_objects=(),
    )
    return SegmentRecord(
        key=sequence_id,
        item_name="item-1",
        category="single_point",
        dataset_dimension="atomic",
        directory=directory,
        video_path=directory / "unused.mp4",
        annotation_path=directory / "annotation.json",
        annotation=annotation,
    )


def _datasets(config: V2Config, tmp_path: Path) -> TrainingDatasetBundle:
    """构造 curriculum 各阶段均非空、但 fake runner 不解码的 bundle。"""

    transform = FrameCoordinateTransform(
        source_frame_width=config.coordinates.source_width,
        source_frame_height=config.coordinates.source_height,
        transform_identity=config.coordinates.transform_identity,
        transform=AffineOsuVideoTransform(_AFFINE_MATRIX),
    )
    datasets = tuple(
        (
            split,
            SegmentTrainingDataset(
                (_segment_record(tmp_path, split),),
                split=split,
                sample_fps=config.data.sample_fps,
                frame_step=config.data.frame_step,
                max_frames_per_segment=config.data.max_frames_per_segment,
                visibility_post_ms=config.data.visibility_post_ms,
                coordinate_transform=transform,
            ),
        )
        for split in (DataSplit.TRAIN, DataSplit.VALIDATION, DataSplit.TEST)
    )
    return TrainingDatasetBundle(
        datasets=datasets,
        quality_report=DataQualityReport(issues=()),
        dataset_identity=f"dataset-{'1' * 64}",
        transform_fingerprint=transform.transform_fingerprint,
        loader=config.data.loader,
        coordinate_transform=transform,
    )


def _hard_example_plan(context: ProductionTrialContext) -> HardExamplePlan:
    """产生一个 TRAIN 空间难例及两个必须排除的非训练事件。"""

    stage_index = tuple(CurriculumStage).index(context.curriculum_stage)
    base = context.trial_index * 10_000 + stage_index * 100 + context.rung_index * 10
    inputs = tuple(
        EvaluationSplitEvent(
            SequenceEvaluationEvent(
                event_id=f"sequence-event-{base + offset:064x}",
                sample_id="train-sequence",
                frame_index=0,
                passed=False,
                primary_error=PrimaryError.SPATIAL,
                error_tags=(EvaluationTag.SPATIAL_MISS,),
                target_id=f"target-{base + offset}",
                click_index=0,
                click_x=100.0,
                click_y=80.0,
            ),
            split,
        )
        for offset, split in (
            (1, DataSplit.TRAIN),
            (2, DataSplit.VALIDATION),
            (3, DataSplit.TEST),
        )
    )
    return build_hard_example_plan(inputs)


def _install_fake_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """重置并安装 production runner fake。"""

    _ScheduledStageRunner.invocations = []
    _ScheduledStageRunner.fail_cohorts = frozenset()
    _ScheduledStageRunner.interrupt_key = None
    monkeypatch.setattr(
        "traning.core.training.production.ProductionStageRunner",
        _ScheduledStageRunner,
    )
    monkeypatch.setattr(
        "traning.core.training.production.load_runtime_checkpoint",
        _accept_runtime_checkpoint,
        raising=False,
    )


def _accept_runtime_checkpoint(*_args: object, **_kwargs: object) -> None:
    """隔离 runtime 权重解码；本文件只验收调度与制品边界。"""


def _context_key(
    invocation: _RunnerInvocation,
) -> tuple[int, CurriculumStage, int]:
    """返回便于断言的 proposal/stage/rung key。"""

    context = invocation.context
    return context.trial_index, context.curriculum_stage, context.rung_index


def test_only_full_terminal_can_win_and_promotions_use_incremental_parent_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ASHA 只晋级 cohort 头部，且只有 FULL 末 rung 能成为 winner。"""

    _install_fake_runner(monkeypatch)
    config = _config(max_trials=2)
    result = ProductionTrainer(config, _datasets(config, tmp_path)).run(
        run_dir=tmp_path / "scheduled-run",
        run_id="scheduled-run",
    )

    assert result.observation.trial_index == 0
    assert result.observation.acceptance.passed
    invocations = tuple(_ScheduledStageRunner.invocations)
    keys = tuple(_context_key(item) for item in invocations)
    assert keys[:2] == (
        (0, CurriculumStage.BASIC, 0),
        (1, CurriculumStage.BASIC, 0),
    )
    assert (1, CurriculumStage.BASIC, 1) not in keys
    winner_jobs = tuple(item for item in invocations if item.context.trial_index == 0)
    assert tuple(item.context.curriculum_stage for item in winner_jobs) == (
        CurriculumStage.BASIC,
        CurriculumStage.BASIC,
        CurriculumStage.MULTI_OBJECT,
        CurriculumStage.MULTI_OBJECT,
        CurriculumStage.COMPLEX,
        CurriculumStage.COMPLEX,
        CurriculumStage.FULL,
        CurriculumStage.FULL,
    )
    assert tuple(item.context.budget_steps for item in winner_jobs) == (
        1,
        3,
        1,
        3,
        1,
        3,
        1,
        3,
    )
    assert tuple(item.incremental_budget_steps for item in winner_jobs) == (
        1,
        2,
        1,
        2,
        1,
        2,
        1,
        2,
    )
    assert winner_jobs[0].context.parent_checkpoint_path is None
    for previous, current in zip(winner_jobs, winner_jobs[1:], strict=False):
        parent = current.context.parent_checkpoint_path
        assert parent is not None
        assert parent.is_dir()
        assert (parent / "fake-checkpoint.txt").is_file()
        # 父路径必须来自紧邻 job，而不是同 trial 中任意较早 checkpoint。
        assert f"stage={previous.context.curriculum_stage.value}" in (
            parent / "fake-checkpoint.txt"
        ).read_text(encoding="utf-8")
        assert f"rung={previous.context.rung_index}" in (
            parent / "fake-checkpoint.txt"
        ).read_text(encoding="utf-8")
    assert all(
        not (
            item.context.curriculum_stage is CurriculumStage.FULL
            and item.context.rung_index == 1
        )
        for item in invocations[:-1]
    )


def test_pruned_cohort_continues_new_parameters_and_committed_feedback_is_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """普通 prune 后继续新参数；反馈只沿 parent 或已提交 cohort 传播。"""

    _install_fake_runner(monkeypatch)
    _ScheduledStageRunner.fail_cohorts = frozenset({0})
    config = _config(max_trials=3)
    result = ProductionTrainer(config, _datasets(config, tmp_path)).run(
        run_dir=tmp_path / "feedback-run",
        run_id="feedback-run",
    )

    assert result.observation.trial_index == 2
    invocations = tuple(_ScheduledStageRunner.invocations)
    basic_zero = tuple(
        item
        for item in invocations
        if item.context.curriculum_stage is CurriculumStage.BASIC
        and item.context.rung_index == 0
    )
    assert tuple(item.context.trial_index for item in basic_zero) == (0, 1, 2)
    assert len({item.context.parameters for item in basic_zero}) == 3
    # 同 cohort 不能读取尚未形成 cohort 终态的兄弟 proposal 反馈。
    assert basic_zero[0].feedback is None
    assert basic_zero[1].feedback is None
    inherited = basic_zero[2].feedback
    assert inherited is not None
    assert inherited.source_trial_index == 1
    perception_weights = inherited.weights_for(HardExampleDestination.PERCEPTION)
    assert len(perception_weights) == 1
    assert perception_weights[0].identity == ("train-sequence", 0)
    assert perception_weights[0].effective_weight > 1.0
    assert {item.split for item in inherited.excluded} == {
        DataSplit.VALIDATION,
        DataSplit.TEST,
    }
    # 同 proposal 晋级时优先消费直接 parent job 的最新反馈。
    trial_two = tuple(item for item in invocations if item.context.trial_index == 2)
    assert trial_two[1].feedback is not None
    assert trial_two[1].feedback.source_trial_index == 2


def test_resume_skips_committed_jobs_and_tampered_feedback_blocks_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """恢复不重跑已提交 job；被篡改的 parent feedback 必须阻断。"""

    _install_fake_runner(monkeypatch)
    config = _config(max_trials=2)
    run_dir = tmp_path / "resume-run"
    _ScheduledStageRunner.interrupt_key = (0, CurriculumStage.BASIC, 1)
    trainer = ProductionTrainer(config, _datasets(config, tmp_path))

    with pytest.raises(RuntimeError, match="intentional-job-interruption"):
        trainer.run(run_dir=run_dir, run_id="resume-run")

    completed_invocations = tuple(_ScheduledStageRunner.invocations)
    assert tuple(_context_key(item) for item in completed_invocations) == (
        (0, CurriculumStage.BASIC, 0),
        (1, CurriculumStage.BASIC, 0),
        (0, CurriculumStage.BASIC, 1),
    )
    feedback_path: Path | None = None
    payload: dict[str, object] | None = None
    for candidate in run_dir.rglob("*.json"):
        candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
        if (
            isinstance(candidate_payload, dict)
            and candidate_payload.get("source_trial_index") == 0
        ):
            feedback_path = candidate
            payload = candidate_payload
            break
    assert feedback_path is not None
    assert payload is not None
    source_events = payload["source_events"]
    assert isinstance(source_events, list)
    assert isinstance(source_events[0], dict)
    source_events[0]["route_weight"] = 99.0
    feedback_path.write_text(json.dumps(payload), encoding="utf-8")

    _ScheduledStageRunner.interrupt_key = None
    with pytest.raises(IntegrityError, match="SHA-256"):
        trainer.run(run_dir=run_dir, run_id="resume-run", resume=True)

    assert tuple(_ScheduledStageRunner.invocations) == completed_invocations


def test_exhausted_resume_does_not_repeat_any_committed_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全部 proposal 被普通门禁 prune 后，恢复只重放状态而不重训。"""

    _install_fake_runner(monkeypatch)
    _ScheduledStageRunner.fail_cohorts = frozenset({0})
    config = _config(max_trials=2)
    run_dir = tmp_path / "exhausted-run"
    trainer = ProductionTrainer(config, _datasets(config, tmp_path))

    with pytest.raises(SearchExhaustedError):
        trainer.run(run_dir=run_dir, run_id="exhausted-run")
    first_calls = tuple(_ScheduledStageRunner.invocations)
    assert tuple(_context_key(item) for item in first_calls) == (
        (0, CurriculumStage.BASIC, 0),
        (1, CurriculumStage.BASIC, 0),
    )

    with pytest.raises(SearchExhaustedError):
        trainer.run(run_dir=run_dir, run_id="exhausted-run", resume=True)
    assert tuple(_ScheduledStageRunner.invocations) == first_calls
