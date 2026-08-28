"""把单一 V2 config 接到严格验收、持续参数搜索入口。"""

from __future__ import annotations

from dataclasses import dataclass
import time

from traning.config import V2Config
from traning.contracts import TelemetryEvent
from traning.telemetry import TELEMETRY_SCHEMA_VERSION, TelemetryReporter
from traning.training import (
    ParameterVector,
    SearchExhaustedError,
    TrialCompletedCallback,
    TrialEvaluator,
    TrialObservation,
    run_search,
)


@dataclass(slots=True)
class _ReportingEvaluator:
    """在不改变 evaluator 结果的前提下发布每个已验证 trial。"""

    evaluator: TrialEvaluator
    reporter: TelemetryReporter

    def evaluate(
        self,
        parameters: ParameterVector,
        trial_index: int,
    ) -> TrialObservation:
        """先验证 evaluator identity，再持久化 trial completion 事件。"""

        observation = self.evaluator.evaluate(parameters, trial_index)
        if not isinstance(observation, TrialObservation):
            raise TypeError("evaluator.evaluate 必须返回 TrialObservation")
        if (
            observation.parameters != parameters
            or observation.trial_index != trial_index
        ):
            raise ValueError("evaluator 返回的 trial identity 与 proposal 不一致")
        self.reporter.publish(
            TelemetryEvent(
                schema_version=TELEMETRY_SCHEMA_VERSION,
                event_type="search.trial.completed",
                timestamp_ms=_timestamp_ms(),
                run_id=self.reporter.run_id,
                metrics=(("objective", observation.objective),),
                payload=(
                    ("acceptance_passed", observation.acceptance.passed),
                    ("trial_index", trial_index),
                ),
            )
        )
        return observation


def initial_parameter_vector(config: V2Config) -> ParameterVector:
    """从各领域配置集中构造搜索核心唯一允许的初始参数向量。"""

    if not isinstance(config, V2Config):
        raise TypeError("config 必须是 V2Config")
    return ParameterVector(
        learning_rate=config.training.learning_rate,
        score_threshold=config.perception.score_threshold,
        max_candidates=config.perception.max_candidates,
        risk_lambda=config.decision.risk_lambda,
        wait_cost=config.decision.wait_cost,
        min_confidence=config.decision.min_confidence,
    )


def run_configured_search(
    config: V2Config,
    evaluator: TrialEvaluator,
    *,
    reporter: TelemetryReporter | None = None,
    history: tuple[TrialObservation, ...] = (),
    on_trial_completed: TrialCompletedCallback | None = None,
) -> TrialObservation:
    """按配置运行或恢复搜索；默认无预算时仅全门禁通过才返回。"""

    if not isinstance(config, V2Config):
        raise TypeError("config 必须是 V2Config")
    if reporter is not None and not isinstance(reporter, TelemetryReporter):
        raise TypeError("reporter 必须是 TelemetryReporter 或 None")
    if not isinstance(history, tuple) or any(
        not isinstance(item, TrialObservation) for item in history
    ):
        raise TypeError("history 必须是 TrialObservation 元组")
    if on_trial_completed is not None and not callable(on_trial_completed):
        raise TypeError("on_trial_completed 必须可调用或为 None")
    selected_evaluator = (
        evaluator if reporter is None else _ReportingEvaluator(evaluator, reporter)
    )
    try:
        result = run_search(
            selected_evaluator,
            initial_parameter_vector(config),
            seed=config.training.seed,
            max_trials=config.optimization.max_trials,
            history=history,
            on_trial_completed=on_trial_completed,
        )
    except SearchExhaustedError as exc:
        if reporter is not None:
            _publish_terminal_search_event(
                reporter,
                event_type="search.exhausted",
                trial_count=exc.decision.trial_count,
            )
        raise
    except Exception as exc:
        if reporter is not None:
            reporter.publish(
                TelemetryEvent(
                    schema_version=TELEMETRY_SCHEMA_VERSION,
                    event_type="search.failed",
                    timestamp_ms=_timestamp_ms(),
                    run_id=reporter.run_id,
                    payload=(("error_type", type(exc).__name__),),
                )
            )
        raise
    if reporter is not None:
        _publish_terminal_search_event(
            reporter,
            event_type="search.passed",
            trial_count=result.trial_index + 1,
        )
    return result


def _publish_terminal_search_event(
    reporter: TelemetryReporter,
    *,
    event_type: str,
    trial_count: int,
) -> None:
    """把 PASSED/EXHAUSTED 作为明确终态写入 events 通道。"""

    reporter.publish(
        TelemetryEvent(
            schema_version=TELEMETRY_SCHEMA_VERSION,
            event_type=event_type,
            timestamp_ms=_timestamp_ms(),
            run_id=reporter.run_id,
            payload=(("trial_count", trial_count),),
        )
    )


def _timestamp_ms() -> float:
    """返回非负 Unix 毫秒，用于跨进程遥测排序。"""

    return time.time_ns() / 1_000_000.0


__all__ = ("initial_parameter_vector", "run_configured_search")
