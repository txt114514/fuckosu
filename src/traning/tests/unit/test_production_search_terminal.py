"""验证生产训练搜索只在明确通过或耗尽时结束。"""

from __future__ import annotations

from pathlib import Path

import pytest
from package import AffineOsuVideoTransform, CurriculumStage

from traning.conf import CoordinateConfig, OptimizationConfig, V2Config
from traning.state import DataQualityReport, DataSplit
from traning.core.data import (
    FrameCoordinateTransform,
    SegmentTrainingDataset,
    TrainingDatasetBundle,
)
from traning.lib.telemetry import StateStore, TelemetryReporter
from traning.core.training import (
    ExecutionStatus,
    HardExampleFeedbackArtifact,
    ParameterVector,
    ProductionGateSpec,
    ProductionTrainer,
    ProductionTrialMetrics,
    SearchExhaustedError,
    StageResult,
    TrainingStage,
    TrialAcceptance,
)
from traning.core.training.production_schedule import ProductionTrialContext


_AFFINE_MATRIX = (
    (2.0, 0.0, 100.0),
    (0.0, 2.0, 40.0),
)


def _config(*, max_trials: int | None) -> V2Config:
    """构造不访问 CUDA、但带完整坐标身份的搜索配置。"""

    return V2Config(
        coordinates=CoordinateConfig(
            transform_identity="production-search-test",
            affine_matrix=_AFFINE_MATRIX,
        ),
        optimization=OptimizationConfig(max_trials=max_trials),
    )


def _datasets(config: V2Config) -> TrainingDatasetBundle:
    """构造不解码视频的空 typed bundle，隔离真实训练开销。"""

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
                (),
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
        dataset_identity=f"dataset-{'0' * 64}",
        transform_fingerprint=transform.transform_fingerprint,
        loader=config.data.loader,
        coordinate_transform=transform,
    )


class _DeterministicStageRunner:
    """用可配置通过序号替代昂贵模型训练，同时保留真实阶段协议。"""

    passing_index: int | None = None
    created_trials: list[tuple[int, ParameterVector]] = []

    def __init__(
        self,
        *,
        base_config: V2Config,
        context: ProductionTrialContext,
        datasets: TrainingDatasetBundle,
        gates: ProductionGateSpec,
        run_dir: Path,
        run_id: str,
        reporter: TelemetryReporter,
        input_feedback: HardExampleFeedbackArtifact | None,
    ) -> None:
        """记录 proposal 首个 job，并保存生产入口读取的最小状态。"""

        del datasets, gates, run_id, reporter, input_feedback
        self.context = context
        self.parameters = context.parameters
        self.trial_index = context.trial_index
        self.run_dir = run_dir
        self.rung_count = len(base_config.optimization.asha_rungs)
        self.metrics = ProductionTrialMetrics(golden_hit_rate=0.1 * context.trial_index)
        self.stage_results: list[StageResult] = []
        self.hard_example_plan = None
        self.feedback_evaluated = False
        if (
            context.curriculum_stage is CurriculumStage.BASIC
            and context.rung_index == 0
        ):
            type(self).created_trials.append((context.trial_index, context.parameters))

    def run(self, stage: TrainingStage) -> StageResult:
        """普通失败返回 FAILED；指定 trial 则走完整阶段并提交占位 manifest。"""

        if self.trial_index != type(self).passing_index:
            result = StageResult(
                stage,
                ExecutionStatus.FAILED,
                "ordinary gate failure",
            )
            self.stage_results.append(result)
            return result
        if stage is TrainingStage.EVALUATION:
            full_terminal = (
                self.context.curriculum_stage is CurriculumStage.FULL
                and self.context.rung_index == self.rung_count - 1
            )
            acceptance = TrialAcceptance(
                data=True,
                perception=True,
                tracking=True,
                belief=True,
                outcome=True,
                decision=True,
                golden=True,
                schedule=full_terminal,
            )
            result = StageResult(
                stage,
                ExecutionStatus.PASSED,
                acceptance=acceptance,
            )
        else:
            result = StageResult(stage, ExecutionStatus.PASSED)
        self.stage_results.append(result)
        return result

    def publish_job_checkpoint(self, directory: Path) -> None:
        """写入最小非空 job artifact，供 production 摘要与 parent 链使用。"""

        directory.mkdir(parents=True, exist_ok=False)
        (directory / "fake-checkpoint.txt").write_text(
            (
                f"trial={self.trial_index};"
                f"stage={self.context.curriculum_stage.value};"
                f"rung={self.context.rung_index}"
            ),
            encoding="utf-8",
        )


def _accept_checkpoint(*_args: object, **_kwargs: object) -> None:
    """让测试只验证发布时机，不重复测试 checkpoint 解码细节。"""


def test_production_gate_failure_continues_and_publishes_passed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """普通 gate 失败必须继续唯一 proposal，并在下一轮通过后发布 PASSED。"""

    config = _config(max_trials=3)
    _DeterministicStageRunner.passing_index = 1
    _DeterministicStageRunner.created_trials = []
    monkeypatch.setattr(
        "traning.core.training.production.ProductionStageRunner",
        _DeterministicStageRunner,
    )
    monkeypatch.setattr(
        "traning.core.training.production.load_runtime_checkpoint",
        _accept_checkpoint,
    )

    result = ProductionTrainer(config, _datasets(config)).run(
        run_dir=tmp_path / "passed-run",
        run_id="passed-run",
    )

    created = _DeterministicStageRunner.created_trials
    assert tuple(index for index, _parameters in created) == (0, 1)
    assert len({parameters for _index, parameters in created}) == 2
    assert result.observation.trial_index == 1
    assert result.observation.acceptance.passed
    all_event_types = tuple(
        event.event_type
        for event in StateStore(tmp_path / "passed-run" / "telemetry").history().events
    )
    assert all_event_types.count("search.job.completed") == 9
    event_types = tuple(
        event_type
        for event_type in all_event_types
        if event_type != "search.job.completed"
    )
    assert event_types == (
        "search.trial.completed",
        "search.trial.completed",
        "search.passed",
    )


def test_production_exhaustion_publishes_terminal_and_resume_does_not_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """预算耗尽必须发布 EXHAUSTED；恢复只读历史，不重复训练已提交 proposal。"""

    config = _config(max_trials=2)
    _DeterministicStageRunner.passing_index = None
    _DeterministicStageRunner.created_trials = []
    monkeypatch.setattr(
        "traning.core.training.production.ProductionStageRunner",
        _DeterministicStageRunner,
    )
    run_dir = tmp_path / "exhausted-run"
    trainer = ProductionTrainer(config, _datasets(config))

    with pytest.raises(SearchExhaustedError) as first_error:
        trainer.run(run_dir=run_dir, run_id="exhausted-run")

    assert first_error.value.decision.trial_count == 2
    created_before_resume = tuple(_DeterministicStageRunner.created_trials)
    assert tuple(index for index, _parameters in created_before_resume) == (0, 1)
    assert len({parameters for _index, parameters in created_before_resume}) == 2
    with pytest.raises(SearchExhaustedError) as resumed_error:
        trainer.run(run_dir=run_dir, run_id="exhausted-run", resume=True)

    assert resumed_error.value.decision.trial_count == 2
    assert tuple(_DeterministicStageRunner.created_trials) == created_before_resume
    # ProductionTrainer 的默认 store 已结束写入作用域；新实例从磁盘恢复完整历史。
    all_event_types = tuple(
        event.event_type for event in StateStore(run_dir / "telemetry").history().events
    )
    assert all_event_types.count("search.job.completed") == 2
    event_types = tuple(
        event_type
        for event_type in all_event_types
        if event_type != "search.job.completed"
    )
    assert event_types == (
        "search.trial.completed",
        "search.trial.completed",
        "search.exhausted",
        "search.exhausted",
    )


def test_production_fatal_error_publishes_failed_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不可恢复的阶段异常必须显式发布 FAILED，不能表现为静默停止。"""

    config = _config(max_trials=None)
    _DeterministicStageRunner.created_trials = []
    monkeypatch.setattr(
        "traning.core.training.production.ProductionStageRunner",
        _DeterministicStageRunner,
    )

    def _raise_fatal(
        _self: _DeterministicStageRunner,
        _stage: TrainingStage,
    ) -> StageResult:
        raise RuntimeError("fatal-stage-boundary")

    monkeypatch.setattr(_DeterministicStageRunner, "run", _raise_fatal)
    run_dir = tmp_path / "failed-run"

    with pytest.raises(RuntimeError, match="fatal-stage-boundary"):
        ProductionTrainer(config, _datasets(config)).run(
            run_dir=run_dir,
            run_id="failed-run",
        )

    events = StateStore(run_dir / "telemetry").history().events
    assert tuple(event.event_type for event in events) == ("search.failed",)
    assert dict(events[0].payload)["error_type"] == "RuntimeError"
