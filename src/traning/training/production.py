"""真实数据、持续参数搜索、断点恢复与 checkpoint 发布的生产入口。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time

from traning.config import V2Config
from traning.contracts import TelemetryEvent
from traning.data import TrainingDatasetBundle, require_quality
from traning.telemetry import StateStore, TelemetryReporter
from traning.training.checkpoints import (
    CHECKPOINT_MANIFEST_FILENAME,
    load_runtime_checkpoint,
)
from traning.training.evaluator import OrchestratedTrialEvaluator
from traning.training.optimization import (
    ParameterVector,
    SearchExhaustedError,
    TrialObservation,
    run_search,
)
from traning.training.orchestration import OrchestrationResult, StageRunner
from traning.training.search_state import SearchHistoryStore

from .production_contracts import (
    ProductionGateSpec,
    ProductionTrainingResult,
)
from .production_stages import (
    ProductionStageRunner,
    config_for_parameters,
    trial_checkpoint_directory,
)


SEARCH_STATE_FILENAME = "search_state.json"
"""每个 run 唯一的原子搜索恢复文件名。"""


@dataclass(frozen=True, slots=True)
class ProductionTrainer:
    """把已检查的真实数据 bundle 接入可恢复的门禁驱动搜索。"""

    config: V2Config
    datasets: TrainingDatasetBundle
    gates: ProductionGateSpec = ProductionGateSpec()

    def __post_init__(self) -> None:
        if not isinstance(self.config, V2Config):
            raise TypeError("config 必须是 V2Config")
        if not isinstance(self.datasets, TrainingDatasetBundle):
            raise TypeError("datasets 必须是 TrainingDatasetBundle")
        if not isinstance(self.gates, ProductionGateSpec):
            raise TypeError("gates 必须是 ProductionGateSpec")

    def run(
        self,
        *,
        run_dir: Path,
        run_id: str,
        resume: bool = True,
        reporter: TelemetryReporter | None = None,
    ) -> ProductionTrainingResult:
        """持续尝试未重复参数，只有全部门禁通过才发布并返回模型。"""

        if not isinstance(run_dir, Path):
            raise TypeError("run_dir 必须是 pathlib.Path")
        if not isinstance(run_id, str) or not run_id or run_id != run_id.strip():
            raise ValueError("run_id 必须非空且无首尾空格")
        if not isinstance(resume, bool):
            raise TypeError("resume 必须是 bool")
        require_quality(self.datasets.quality_report)
        if (
            self.datasets.coordinate_transform is None
        ):  # pragma: no cover - quality 已阻断
            raise RuntimeError("数据 bundle 缺少坐标变换")

        run_dir.mkdir(parents=True, exist_ok=True)
        selected_reporter = reporter or TelemetryReporter(
            run_id,
            StateStore(
                run_dir / "telemetry",
                schema_version=self.config.telemetry.schema_version,
            ),
        )
        if selected_reporter.run_id != run_id:
            raise ValueError("reporter.run_id 与生产 run_id 不一致")

        initial = _initial_parameter_vector(self.config)
        history_store = SearchHistoryStore(
            run_dir / SEARCH_STATE_FILENAME,
            run_id=run_id,
            dataset_id=self.datasets.dataset_identity,
            config=self.config,
            initial_parameters=initial,
        )
        if not resume and history_store.path.exists():
            raise FileExistsError("搜索状态已存在；请启用 resume 或使用新的 run_id")
        history = history_store.load() if resume else ()
        runners: dict[int, ProductionStageRunner] = {}

        def runner_factory(
            parameters: ParameterVector,
            trial_index: int,
        ) -> StageRunner:
            """为当前 proposal 构造独立且可审计的真实阶段 runner。"""

            runner = ProductionStageRunner(
                base_config=self.config,
                parameters=parameters,
                trial_index=trial_index,
                datasets=self.datasets,
                gates=self.gates,
                run_dir=run_dir,
                run_id=run_id,
                reporter=selected_reporter,
            )
            runners[trial_index] = runner
            return runner

        def objective_function(
            _parameters: ParameterVector,
            trial_index: int,
            _result: OrchestrationResult,
        ) -> float:
            """从同一 trial runner 中提取跨阶段汇总目标值。"""

            runner = runners.get(trial_index)
            if runner is None:  # pragma: no cover - lazy runner 在编排前已创建
                raise RuntimeError("trial runner 未注册，无法计算目标值")
            return runner.metrics.objective

        evaluator = OrchestratedTrialEvaluator(
            quality_report=self.datasets.quality_report,
            runner_factory=runner_factory,
            objective_function=objective_function,
        )
        try:
            winner = run_search(
                evaluator,
                initial,
                seed=self.config.training.seed,
                max_trials=self.config.optimization.max_trials,
                history=history,
                on_trial_completed=_completion_callback(
                    history_store,
                    selected_reporter,
                ),
            )
        except SearchExhaustedError as error:
            _publish_search_terminal(
                selected_reporter,
                event_type="search.exhausted",
                observation=error.decision.best_observation,
                trial_count=error.decision.trial_count,
            )
            raise
        except Exception as error:
            _publish_search_failure(selected_reporter, error)
            raise
        try:
            checkpoint_directory = trial_checkpoint_directory(
                run_dir,
                winner.trial_index,
            )
            winning_config = config_for_parameters(self.config, winner.parameters)
            if not (checkpoint_directory / CHECKPOINT_MANIFEST_FILENAME).is_file():
                raise RuntimeError("winning trial 缺少已提交的 runtime checkpoint")

            # 即使 winner 来自恢复历史，也必须重新执行完整摘要/config/坐标校验。
            load_runtime_checkpoint(
                checkpoint_directory,
                winning_config,
                self.datasets.coordinate_transform,
                expected_dataset_id=self.datasets.dataset_identity,
            )
        except Exception as error:
            _publish_search_failure(selected_reporter, error)
            raise
        _publish_search_terminal(
            selected_reporter,
            event_type="search.passed",
            observation=winner,
            trial_count=len(history_store.load()),
        )
        winning_runner = runners.get(winner.trial_index)
        return ProductionTrainingResult(
            observation=winner,
            winning_config=winning_config,
            checkpoint_directory=checkpoint_directory,
            stage_results=(
                () if winning_runner is None else tuple(winning_runner.stage_results)
            ),
            metrics=None if winning_runner is None else winning_runner.metrics,
            resumed=winning_runner is None,
        )


def _initial_parameter_vector(config: V2Config) -> ParameterVector:
    """从唯一配置构造搜索空间的初始 proposal。"""

    return ParameterVector(
        learning_rate=config.training.learning_rate,
        score_threshold=config.perception.score_threshold,
        max_candidates=config.perception.max_candidates,
        risk_lambda=config.decision.risk_lambda,
        wait_cost=config.decision.wait_cost,
        min_confidence=config.decision.min_confidence,
    )


def _completion_callback(
    history_store: SearchHistoryStore,
    reporter: TelemetryReporter,
) -> Callable[[tuple[TrialObservation, ...]], None]:
    """返回先原子提交历史、再发布搜索事件的 completion callback。"""

    def complete(history: tuple[TrialObservation, ...]) -> None:
        """原子保存完整搜索历史，并发布刚完成 trial 的事实事件。"""

        history_store.persist(history)
        latest = history[-1]
        reporter.publish(
            TelemetryEvent(
                schema_version=reporter.store.snapshot().schema_version,
                event_type="search.trial.completed",
                timestamp_ms=time.time_ns() / 1_000_000.0,
                run_id=reporter.run_id,
                metrics=(("objective", latest.objective),),
                payload=(
                    ("acceptance_passed", latest.acceptance.passed),
                    ("trial_index", latest.trial_index),
                ),
            )
        )

    return complete


def _publish_search_terminal(
    reporter: TelemetryReporter,
    *,
    event_type: str,
    observation: TrialObservation | None,
    trial_count: int,
) -> None:
    """发布通过或耗尽终态，不把普通门禁失败伪装成进程停止。"""

    metrics = () if observation is None else (("objective", observation.objective),)
    payload = (
        ("trial_count", trial_count),
        (
            "best_trial_index",
            None if observation is None else observation.trial_index,
        ),
    )
    reporter.publish(
        TelemetryEvent(
            schema_version=reporter.store.snapshot().schema_version,
            event_type=event_type,
            timestamp_ms=time.time_ns() / 1_000_000.0,
            run_id=reporter.run_id,
            metrics=metrics,
            payload=payload,
        )
    )


def _publish_search_failure(
    reporter: TelemetryReporter,
    error: Exception,
) -> None:
    """把不可恢复异常发布为明确 FAILED 终态后保持原异常传播。"""

    reporter.publish(
        TelemetryEvent(
            schema_version=reporter.store.snapshot().schema_version,
            event_type="search.failed",
            timestamp_ms=time.time_ns() / 1_000_000.0,
            run_id=reporter.run_id,
            payload=(
                ("error_type", type(error).__name__),
                ("message", str(error)),
            ),
        )
    )


__all__ = (
    "SEARCH_STATE_FILENAME",
    "ProductionTrainer",
    "ProductionGateSpec",
    "ProductionTrainingResult",
)
