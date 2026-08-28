"""确定性规划器的动作和结果契约。"""

from dataclasses import dataclass
from enum import Enum

from .common import require_finite, require_identifier, require_nonnegative
from .observation import Point2D
from .outcome import OutcomeDistribution


class DecisionAction(str, Enum):
    """正式运行路径允许的基础动作。"""

    CLICK = "click"
    WAIT = "wait"


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """规划器选出的动作及其可审计效用。"""

    action: DecisionAction
    track_id: str | None
    execute_at_ms: float
    expected_utility: float
    wait_utility: float
    confidence: float
    horizon_ms: float = 0.0
    target_position: Point2D | None = None
    outcome: OutcomeDistribution | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, DecisionAction):
            raise TypeError("action 必须是 DecisionAction")
        require_nonnegative(self.execute_at_ms, "execute_at_ms")
        require_finite(self.expected_utility, "expected_utility")
        require_finite(self.wait_utility, "wait_utility")
        require_finite(self.confidence, "confidence")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence 必须位于 [0, 1]")
        require_nonnegative(self.horizon_ms, "horizon_ms")
        if self.action is DecisionAction.CLICK:
            if self.horizon_ms != 0.0:
                raise ValueError("CLICK 必须是 horizon_ms=0 的立即动作")
            if self.expected_utility < self.wait_utility:
                raise ValueError("CLICK expected_utility 不得低于 wait_utility")
            if (
                self.track_id is None
                or self.target_position is None
                or self.outcome is None
            ):
                raise ValueError("CLICK 必须包含 track_id、target_position 和 outcome")
            require_identifier(self.track_id, "track_id")
            if not isinstance(self.target_position, Point2D):
                raise TypeError("CLICK target_position 必须是 Point2D")
            if not isinstance(self.outcome, OutcomeDistribution):
                raise TypeError("CLICK outcome 必须是 OutcomeDistribution")
            if self.outcome.track_id != self.track_id:
                raise ValueError(
                    "DecisionResult 与 OutcomeDistribution 的 track_id 必须一致"
                )
            if self.outcome.horizon_ms != 0.0:
                raise ValueError("CLICK outcome 必须对应 horizon_ms=0")
        else:
            if self.horizon_ms <= 0.0:
                raise ValueError("WAIT 必须选择正数 horizon_ms")
            if self.expected_utility != self.wait_utility:
                raise ValueError("WAIT expected_utility 必须等于 wait_utility")
            if (
                self.track_id is not None
                or self.target_position is not None
                or self.outcome is not None
            ):
                raise ValueError("WAIT 不得绑定轨迹、点击位置或点击结果")
