"""只依赖 typed belief 与 Outcome 分布的确定性最优停止规划器。"""

from __future__ import annotations

import math

from traning.conf import DecisionConfig
from traning.state import (
    BeliefState,
    DecisionAction,
    DecisionResult,
    OutcomeDistribution,
)
from traning.core.decision.utility import ClickUtility, compute_click_utility


class OptimalStoppingPlanner:
    """比较当前 CLICK 与最小正 horizon 的 WAIT 价值。

    每条 belief 轨迹必须恰好提供 horizon=0 和配置中最小正 horizon 两个
    Outcome；为避免调用方误以为更远 horizon 已参与规划，额外 Outcome 会被拒绝。
    """

    def __init__(self, config: DecisionConfig) -> None:
        if not isinstance(config, DecisionConfig):
            raise TypeError("config 必须是 DecisionConfig")
        positive_horizons = tuple(
            float(horizon) for horizon in config.horizons_ms if horizon > 0
        )
        if 0 not in config.horizons_ms or not positive_horizons:
            raise ValueError("decision.horizons_ms 必须包含 0 和至少一个正 horizon")
        self.config = config
        self.wait_horizon_ms = min(positive_horizons)

    def plan(
        self,
        beliefs: tuple[BeliefState, ...],
        outcomes: tuple[OutcomeDistribution, ...],
        timestamp_ms: float,
    ) -> DecisionResult:
        """在当前点击与等待一个最短正 horizon 之间作稳定选择。"""

        self._validate_inputs(beliefs, outcomes, timestamp_ms)
        if not beliefs:
            empty_wait_utility = -self.config.wait_cost
            return self._wait_result(timestamp_ms, empty_wait_utility, 0.0)

        belief_by_track = {belief.track_id: belief for belief in beliefs}
        outcome_by_key = {
            (outcome.track_id, float(outcome.horizon_ms)): outcome
            for outcome in outcomes
        }
        current_utilities = self._utilities_for_horizon(
            belief_by_track, outcome_by_key, 0.0
        )
        future_utilities = self._utilities_for_horizon(
            belief_by_track, outcome_by_key, self.wait_horizon_ms
        )

        # 先按 track_id 排序，使完全相同 utility 的选择不依赖输入 slot/order。
        current_eligible = tuple(
            utility
            for utility in current_utilities
            if utility.success_probability >= self.config.min_confidence
        )
        best_future = max(future_utilities, key=lambda item: item.value)
        wait_utility = best_future.value - self.config.wait_cost
        if not current_eligible:
            return self._wait_result(
                timestamp_ms, wait_utility, best_future.success_probability
            )
        best_current = max(current_eligible, key=lambda item: item.value)

        # CLICK_NOW 与 WAIT 等效时优先立即点击，避免无收益延迟。
        if best_current.value >= wait_utility:
            belief = belief_by_track[best_current.track_id]
            return DecisionResult(
                action=DecisionAction.CLICK,
                track_id=best_current.track_id,
                execute_at_ms=float(timestamp_ms),
                expected_utility=best_current.value,
                wait_utility=wait_utility,
                confidence=best_current.success_probability,
                horizon_ms=0.0,
                target_position=belief.position_mean,
                outcome=best_current.outcome,
            )
        return self._wait_result(
            timestamp_ms, wait_utility, best_future.success_probability
        )

    def _validate_inputs(
        self,
        beliefs: tuple[BeliefState, ...],
        outcomes: tuple[OutcomeDistribution, ...],
        timestamp_ms: float,
    ) -> None:
        if not isinstance(beliefs, tuple):
            raise TypeError("beliefs 必须是 BeliefState tuple")
        if not isinstance(outcomes, tuple):
            raise TypeError("outcomes 必须是 OutcomeDistribution tuple")
        if any(not isinstance(belief, BeliefState) for belief in beliefs):
            raise TypeError("beliefs 只能包含 BeliefState")
        if any(not isinstance(outcome, OutcomeDistribution) for outcome in outcomes):
            raise TypeError("outcomes 只能包含 OutcomeDistribution")
        if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, (int, float)):
            raise TypeError("timestamp_ms 必须是数值")
        if not math.isfinite(float(timestamp_ms)) or timestamp_ms < 0:
            raise ValueError("timestamp_ms 必须是有限非负数")

        belief_tracks = tuple(belief.track_id for belief in beliefs)
        if len(set(belief_tracks)) != len(belief_tracks):
            raise ValueError("belief track_id 必须唯一")
        if any(belief.timestamp_ms != float(timestamp_ms) for belief in beliefs):
            raise ValueError("belief timestamp_ms 必须与规划时间一致")
        known_tracks = set(belief_tracks)
        if any(outcome.track_id not in known_tracks for outcome in outcomes):
            raise ValueError("outcomes 包含未知 track_id")
        outcome_keys = tuple(
            (outcome.track_id, float(outcome.horizon_ms)) for outcome in outcomes
        )
        if len(set(outcome_keys)) != len(outcome_keys):
            raise ValueError("同一 track+horizon 的 Outcome 不得重复")

        required_keys = {
            (track_id, horizon)
            for track_id in known_tracks
            for horizon in (0.0, self.wait_horizon_ms)
        }
        actual_keys = set(outcome_keys)
        missing = required_keys - actual_keys
        if missing:
            raise ValueError("每条 belief 必须提供 current 与 wait horizon Outcome")
        if actual_keys != required_keys:
            raise ValueError("只接受 current 与最小正 horizon Outcome")

    def _utilities_for_horizon(
        self,
        belief_by_track: dict[str, BeliefState],
        outcome_by_key: dict[tuple[str, float], OutcomeDistribution],
        horizon_ms: float,
    ) -> tuple[ClickUtility, ...]:
        return tuple(
            compute_click_utility(outcome_by_key[(track_id, horizon_ms)], self.config)
            for track_id in sorted(belief_by_track)
        )

    def _wait_result(
        self, timestamp_ms: float, wait_utility: float, confidence: float
    ) -> DecisionResult:
        return DecisionResult(
            action=DecisionAction.WAIT,
            track_id=None,
            execute_at_ms=float(timestamp_ms) + self.wait_horizon_ms,
            expected_utility=wait_utility,
            wait_utility=wait_utility,
            confidence=confidence,
            horizon_ms=self.wait_horizon_ms,
            target_position=None,
            outcome=None,
        )


__all__ = ("OptimalStoppingPlanner",)
