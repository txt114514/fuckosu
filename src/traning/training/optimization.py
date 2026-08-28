"""门禁驱动、可持续且终态显式的确定性参数搜索。"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from functools import reduce
from operator import mul
from typing import Protocol


class ParameterType(str, Enum):
    """参数 registry 支持的数值类型。"""

    FLOAT = "float"
    INTEGER = "integer"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """单个参数的类型、闭区间和量化步长。"""

    name: str
    parameter_type: ParameterType
    minimum: float
    maximum: float
    step: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ValueError("ParameterSpec.name 必须非空且无首尾空格")
        if not isinstance(self.parameter_type, ParameterType):
            raise TypeError("parameter_type 必须是 ParameterType")
        for field_name, value in (
            ("minimum", self.minimum),
            ("maximum", self.maximum),
            ("step", self.step),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} 必须是数值")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} 必须是有限数值")
        if self.minimum > self.maximum or self.step <= 0.0:
            raise ValueError("参数范围必须非空且 step 必须大于 0")
        if self.parameter_type is ParameterType.INTEGER and any(
            not float(value).is_integer()
            for value in (self.minimum, self.maximum, self.step)
        ):
            raise ValueError("integer 参数的 min/max/step 必须都是整数")

    @property
    def value_count(self) -> int:
        """返回闭区间按当前步长可表示的离散值数量。"""

        span = _decimal(self.maximum) - _decimal(self.minimum)
        return int(span // _decimal(self.step)) + 1

    def validate(self, value: float | int) -> None:
        """校验单值的数值类型、有限性与闭区间范围。"""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} 必须是数值")
        if self.parameter_type is ParameterType.INTEGER and not isinstance(value, int):
            raise TypeError(f"{self.name} 必须是整数")
        if not math.isfinite(float(value)):
            raise ValueError(f"{self.name} 必须是有限数值")
        if not self.minimum <= value <= self.maximum:
            raise ValueError(f"{self.name} 必须位于 [{self.minimum}, {self.maximum}]")

    def quantize(self, value: float | int) -> float | int:
        """先 clamp 再按 half-up 量化，绝不发布越界值。"""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} 必须是数值")
        if not math.isfinite(float(value)):
            raise ValueError(f"{self.name} 必须是有限数值")
        clamped = min(self.maximum, max(self.minimum, float(value)))
        ticks = (
            (_decimal(clamped) - _decimal(self.minimum)) / _decimal(self.step)
        ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        index = min(self.value_count - 1, max(0, int(ticks)))
        quantized = _decimal(self.minimum) + index * _decimal(self.step)
        if self.parameter_type is ParameterType.INTEGER:
            return int(quantized)
        return float(quantized)

    def value_at(self, index: int) -> float | int:
        """按离散索引返回规格内精确量化后的参数值。"""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("parameter index 必须是整数")
        if not 0 <= index < self.value_count:
            raise IndexError("parameter index 越界")
        value = _decimal(self.minimum) + index * _decimal(self.step)
        return (
            int(value) if self.parameter_type is ParameterType.INTEGER else float(value)
        )


@dataclass(frozen=True, slots=True)
class ParameterVector:
    """搜索核心唯一允许的强类型参数向量。"""

    learning_rate: float
    score_threshold: float
    max_candidates: int
    risk_lambda: float
    wait_cost: float
    min_confidence: float

    def __post_init__(self) -> None:
        PARAMETER_REGISTRY.validate(self)


@dataclass(frozen=True, slots=True)
class ParameterRegistry:
    """按统一规格循环完成参数校验、clamp 和量化。"""

    specs: tuple[ParameterSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.specs, tuple) or not self.specs:
            raise TypeError("specs 必须是非空 ParameterSpec 元组")
        if any(not isinstance(spec, ParameterSpec) for spec in self.specs):
            raise TypeError("specs 只能包含 ParameterSpec")
        names = tuple(spec.name for spec in self.specs)
        vector_names = tuple(field.name for field in fields(ParameterVector))
        if names != vector_names:
            raise ValueError("registry 字段顺序必须精确覆盖 ParameterVector")

    @property
    def space_size(self) -> int:
        """返回所有参数离散取值笛卡尔积的总大小。"""

        return reduce(mul, (spec.value_count for spec in self.specs), 1)

    def validate(self, vector: ParameterVector) -> None:
        """按统一规格顺序校验完整 typed 参数向量。"""

        for spec in self.specs:
            spec.validate(getattr(vector, spec.name))

    def normalize(self, vector: ParameterVector) -> ParameterVector:
        """逐字段 clamp、量化并返回新的 canonical 参数向量。"""

        values = tuple(spec.quantize(getattr(vector, spec.name)) for spec in self.specs)
        return ParameterVector(*values)  # type: ignore[arg-type]

    def vector_at(self, flat_index: int) -> ParameterVector:
        """以混合进制解码扁平索引，确定性恢复参数向量。"""

        if isinstance(flat_index, bool) or not isinstance(flat_index, int):
            raise TypeError("flat_index 必须是整数")
        if not 0 <= flat_index < self.space_size:
            raise IndexError("flat_index 越界")
        remainder = flat_index
        reversed_values: list[float | int] = []
        for spec in reversed(self.specs):
            remainder, index = divmod(remainder, spec.value_count)
            reversed_values.append(spec.value_at(index))
        return ParameterVector(*reversed(reversed_values))  # type: ignore[arg-type]


PARAMETER_REGISTRY = ParameterRegistry(
    (
        ParameterSpec("learning_rate", ParameterType.FLOAT, 0.00001, 0.1, 0.00001),
        ParameterSpec("score_threshold", ParameterType.FLOAT, 0.0, 1.0, 0.01),
        ParameterSpec("max_candidates", ParameterType.INTEGER, 1, 256, 1),
        ParameterSpec("risk_lambda", ParameterType.FLOAT, 0.0, 10.0, 0.05),
        ParameterSpec("wait_cost", ParameterType.FLOAT, 0.0, 2.0, 0.01),
        ParameterSpec("min_confidence", ParameterType.FLOAT, 0.0, 1.0, 0.01),
    )
)


_GATE_FIELDS = (
    "data",
    "perception",
    "tracking",
    "belief",
    "outcome",
    "decision",
    "golden",
)


@dataclass(frozen=True, slots=True)
class TrialAcceptance:
    """所有阶段门禁的 typed 验收结果。"""

    data: bool
    perception: bool
    tracking: bool
    belief: bool
    outcome: bool
    decision: bool
    golden: bool

    def __post_init__(self) -> None:
        for name in _GATE_FIELDS:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} gate 必须是 bool")

    @property
    def passed(self) -> bool:
        """仅当 registry 中全部训练与 golden 门禁通过时返回真。"""

        return all(getattr(self, name) for name in _GATE_FIELDS)


@dataclass(frozen=True, slots=True)
class TrialObservation:
    """一次已完成试验的参数、目标值与全门禁结果。"""

    trial_index: int
    parameters: ParameterVector
    objective: float
    acceptance: TrialAcceptance

    def __post_init__(self) -> None:
        if isinstance(self.trial_index, bool) or not isinstance(self.trial_index, int):
            raise TypeError("trial_index 必须是整数")
        if self.trial_index < 0:
            raise ValueError("trial_index 不得为负数")
        if not isinstance(self.parameters, ParameterVector):
            raise TypeError("parameters 必须是 ParameterVector")
        if isinstance(self.objective, bool) or not isinstance(
            self.objective, (int, float)
        ):
            raise TypeError("objective 必须是数值")
        if not math.isfinite(float(self.objective)):
            raise ValueError("objective 必须是有限数值")
        if not isinstance(self.acceptance, TrialAcceptance):
            raise TypeError("acceptance 必须是 TrialAcceptance")


class SearchStatus(str, Enum):
    """搜索控制器的非歧义状态。"""

    RUNNING = "running"
    PASSED = "passed"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True, slots=True)
class SearchDecision:
    """搜索下一步 proposal 或显式终态。"""

    status: SearchStatus
    proposal: ParameterVector | None
    best_observation: TrialObservation | None
    trial_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.status, SearchStatus):
            raise TypeError("status 必须是 SearchStatus")
        if isinstance(self.trial_count, bool) or not isinstance(self.trial_count, int):
            raise TypeError("trial_count 必须是整数")
        if self.trial_count < 0:
            raise ValueError("trial_count 不得为负数")
        if self.status is SearchStatus.RUNNING:
            if not isinstance(self.proposal, ParameterVector):
                raise ValueError("RUNNING 必须携带 proposal")
        elif self.proposal is not None:
            raise ValueError("终态不得携带 proposal")
        if self.best_observation is not None and not isinstance(
            self.best_observation, TrialObservation
        ):
            raise TypeError("best_observation 必须是 TrialObservation 或 None")
        if self.status is SearchStatus.PASSED and (
            self.best_observation is None or not self.best_observation.acceptance.passed
        ):
            raise ValueError("PASSED 必须引用全门禁通过的 observation")


@dataclass(frozen=True, slots=True)
class DeterministicSearchController:
    """按 seed 遍历未重复量化空间，直到全门禁通过或显式耗尽。"""

    seed: int
    max_trials: int | None
    registry: ParameterRegistry = PARAMETER_REGISTRY
    _offset: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed 必须是整数")
        if self.max_trials is not None and (
            isinstance(self.max_trials, bool) or not isinstance(self.max_trials, int)
        ):
            raise TypeError("max_trials 必须是整数或 None")
        if self.max_trials is not None and self.max_trials < 1:
            raise ValueError("max_trials 必须至少为 1")
        if not isinstance(self.registry, ParameterRegistry):
            raise TypeError("registry 必须是 ParameterRegistry")
        object.__setattr__(
            self,
            "_offset",
            random.Random(self.seed).randrange(self.registry.space_size),
        )

    def decide(
        self,
        initial: ParameterVector,
        history: tuple[TrialObservation, ...],
    ) -> SearchDecision:
        """纯函数式地依据完整 history 返回下一 proposal 或明确终态。"""

        if not isinstance(initial, ParameterVector):
            raise TypeError("initial 必须是 ParameterVector")
        if not isinstance(history, tuple) or any(
            not isinstance(item, TrialObservation) for item in history
        ):
            raise TypeError("history 必须是 TrialObservation 元组")
        if tuple(item.trial_index for item in history) != tuple(range(len(history))):
            raise ValueError("history.trial_index 必须从 0 连续递增")
        tried = tuple(item.parameters for item in history)
        if len(tried) != len(set(tried)):
            raise ValueError("history 不得包含重复 proposal")
        passed = tuple(item for item in history if item.acceptance.passed)
        if passed:
            winner = max(passed, key=lambda item: (item.objective, -item.trial_index))
            return SearchDecision(SearchStatus.PASSED, None, winner, len(history))
        best = (
            max(history, key=lambda item: (item.objective, -item.trial_index))
            if history
            else None
        )
        if self.max_trials is not None and len(history) >= self.max_trials:
            return SearchDecision(SearchStatus.EXHAUSTED, None, best, len(history))
        tried_set = set(tried)
        if len(tried_set) >= self.registry.space_size:
            return SearchDecision(SearchStatus.EXHAUSTED, None, best, len(history))

        normalized_initial = self.registry.normalize(initial)
        if not history and normalized_initial not in tried_set:
            return SearchDecision(
                SearchStatus.RUNNING, normalized_initial, best, len(history)
            )
        for offset in range(self.registry.space_size):
            index = (self._offset + offset) % self.registry.space_size
            proposal = self.registry.vector_at(index)
            if proposal not in tried_set:
                return SearchDecision(
                    SearchStatus.RUNNING, proposal, best, len(history)
                )
        return SearchDecision(SearchStatus.EXHAUSTED, None, best, len(history))


class TrialEvaluator(Protocol):
    """run_search 消费的最小 typed evaluator。"""

    def evaluate(
        self, parameters: ParameterVector, trial_index: int
    ) -> TrialObservation:
        """执行一个 proposal 并返回同 index、同参数的 observation。"""


TrialCompletedCallback = Callable[[tuple[TrialObservation, ...]], None]
"""每个新 trial 经完整校验后，持久化完整历史的回调。"""


class SearchExhaustedError(RuntimeError):
    """预算或量化空间耗尽且未通过全门禁。"""

    def __init__(self, decision: SearchDecision) -> None:
        if decision.status is not SearchStatus.EXHAUSTED:
            raise ValueError("SearchExhaustedError 只能包装 EXHAUSTED decision")
        self.decision = decision
        super().__init__(f"参数搜索已耗尽：trials={decision.trial_count}")


def run_search(
    evaluator: TrialEvaluator,
    initial: ParameterVector,
    *,
    seed: int = 0,
    max_trials: int | None = None,
    registry: ParameterRegistry = PARAMETER_REGISTRY,
    history: tuple[TrialObservation, ...] = (),
    on_trial_completed: TrialCompletedCallback | None = None,
) -> TrialObservation:
    """持续或恢复求值；只有全门禁 PASSED 返回，耗尽时抛 typed error。

    ``history`` 是上次原子提交后的恢复边界。控制器会重新验证 trial 索引、
    proposal 唯一性与参数范围；新结果也只有通过身份校验后才交给回调持久化，
    因此重启不会跳过一个尚未完整完成的 trial。
    """

    if not isinstance(initial, ParameterVector):
        raise TypeError("initial 必须是 ParameterVector")
    if not callable(getattr(evaluator, "evaluate", None)):
        raise TypeError("evaluator 必须实现 evaluate")
    if not isinstance(history, tuple) or any(
        not isinstance(item, TrialObservation) for item in history
    ):
        raise TypeError("history 必须是 TrialObservation 元组")
    if on_trial_completed is not None and not callable(on_trial_completed):
        raise TypeError("on_trial_completed 必须可调用或为 None")
    controller = DeterministicSearchController(seed, max_trials, registry=registry)
    # 在触发任何昂贵 evaluator 前验证恢复历史；非法或重复历史不得静默重跑。
    controller.decide(initial, history)
    while True:
        decision = controller.decide(initial, history)
        if decision.status is SearchStatus.PASSED:
            if decision.best_observation is None:  # pragma: no cover - DTO 已保证
                raise RuntimeError("PASSED 缺少 observation")
            return decision.best_observation
        if decision.status is SearchStatus.EXHAUSTED:
            raise SearchExhaustedError(decision)
        proposal = decision.proposal
        if proposal is None:  # pragma: no cover - DTO 已保证
            raise RuntimeError("RUNNING 缺少 proposal")
        observation = evaluator.evaluate(proposal, len(history))
        if not isinstance(observation, TrialObservation):
            raise TypeError("evaluator.evaluate 必须返回 TrialObservation")
        if (
            observation.trial_index != len(history)
            or observation.parameters != proposal
        ):
            raise ValueError(
                "evaluator 返回的 trial_index/parameters 与 proposal 不一致"
            )
        history = (*history, observation)
        if on_trial_completed is not None:
            on_trial_completed(history)


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


__all__ = (
    "PARAMETER_REGISTRY",
    "DeterministicSearchController",
    "ParameterRegistry",
    "ParameterSpec",
    "ParameterType",
    "ParameterVector",
    "SearchDecision",
    "SearchExhaustedError",
    "SearchStatus",
    "TrialAcceptance",
    "TrialEvaluator",
    "TrialCompletedCallback",
    "TrialObservation",
    "run_search",
)
