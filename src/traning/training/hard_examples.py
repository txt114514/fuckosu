"""基于 canonical evaluation event 的确定性 hard-example 路由。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from traning.contracts import DataSplit
from traning.evaluation.attribution import PrimaryError, SequenceEvaluationEvent


class HardExampleDestination(str, Enum):
    """hard example 应反馈的模型领域。"""

    PERCEPTION = "perception"
    OUTCOME = "outcome"
    DECISION = "decision"


class HardExampleConsumer(str, Enum):
    """共享同一 canonical event identity 的下游消费者。"""

    OPTIMIZER = "optimizer"
    TELEMETRY = "telemetry"
    GALLERY = "gallery"


class HardExampleExclusionReason(str, Enum):
    """未进入训练权重的显式审计原因。"""

    NON_TRAIN_SPLIT = "non_train_split"
    PASSED = "passed"
    NO_PRIMARY_ERROR = "no_primary_error"


def _require_positive_weight(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("hard-example weight 必须是数值")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError("hard-example weight 必须是有限正数")


@dataclass(frozen=True, slots=True)
class EvaluationSplitEvent:
    """canonical evaluation event 与唯一数据切分的组合。"""

    event: SequenceEvaluationEvent
    split: DataSplit

    def __post_init__(self) -> None:
        if not isinstance(self.event, SequenceEvaluationEvent):
            raise TypeError("event 必须是 SequenceEvaluationEvent")
        if not isinstance(self.split, DataSplit):
            raise TypeError("split 必须是 DataSplit")
        if self.split is DataSplit.ALL:
            raise ValueError("hard-example 输入不得使用 DataSplit.ALL")


@dataclass(frozen=True, slots=True)
class HardExampleRouteSpec:
    """PrimaryError 到训练领域和基础权重的 registry 规格。"""

    primary_error: PrimaryError
    destination: HardExampleDestination
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.primary_error, PrimaryError):
            raise TypeError("primary_error 必须是 PrimaryError")
        if self.primary_error is PrimaryError.NONE:
            raise ValueError("PrimaryError.NONE 不得注册 hard-example 路由")
        if not isinstance(self.destination, HardExampleDestination):
            raise TypeError("destination 必须是 HardExampleDestination")
        _require_positive_weight(self.weight)


HARD_EXAMPLE_ROUTE_REGISTRY: tuple[HardExampleRouteSpec, ...] = (
    HardExampleRouteSpec(
        PrimaryError.SPATIAL,
        HardExampleDestination.PERCEPTION,
        1.0,
    ),
    HardExampleRouteSpec(
        PrimaryError.TEMPORAL,
        HardExampleDestination.OUTCOME,
        1.0,
    ),
    HardExampleRouteSpec(
        PrimaryError.DECISION,
        HardExampleDestination.DECISION,
        1.0,
    ),
)
"""唯一 PrimaryError 路由表；调用方不得重新解释错误标签。"""


@dataclass(frozen=True, slots=True)
class HardExampleRoute:
    """一个 TRAIN hard example 的 canonical 领域路由。"""

    source: EvaluationSplitEvent
    destination: HardExampleDestination

    def __post_init__(self) -> None:
        if not isinstance(self.source, EvaluationSplitEvent):
            raise TypeError("source 必须是 EvaluationSplitEvent")
        if self.source.split is not DataSplit.TRAIN:
            raise ValueError("HardExampleRoute 只能引用 TRAIN event")
        if (
            self.source.event.passed
            or self.source.event.primary_error is PrimaryError.NONE
        ):
            raise ValueError("通过或 NONE event 不得进入 hard-example route")
        if not isinstance(self.destination, HardExampleDestination):
            raise TypeError("destination 必须是 HardExampleDestination")

    @property
    def event(self) -> SequenceEvaluationEvent:
        """直接返回原始 canonical event，不复制对象。"""

        return self.source.event


@dataclass(frozen=True, slots=True)
class HardExampleWeight:
    """优化器使用的有限正 hard-example 权重。"""

    route: HardExampleRoute
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.route, HardExampleRoute):
            raise TypeError("route 必须是 HardExampleRoute")
        _require_positive_weight(self.weight)

    @property
    def event(self) -> SequenceEvaluationEvent:
        """保持 route 中 canonical event 的对象身份。"""

        return self.route.event


@dataclass(frozen=True, slots=True)
class ExcludedHardExample:
    """未进入 TRAIN weights 的显式审计记录。"""

    source: EvaluationSplitEvent
    reason: HardExampleExclusionReason

    def __post_init__(self) -> None:
        if not isinstance(self.source, EvaluationSplitEvent):
            raise TypeError("source 必须是 EvaluationSplitEvent")
        if not isinstance(self.reason, HardExampleExclusionReason):
            raise TypeError("reason 必须是 HardExampleExclusionReason")
        if (
            self.source.split in (DataSplit.VALIDATION, DataSplit.TEST)
            and self.reason is not HardExampleExclusionReason.NON_TRAIN_SPLIT
        ):
            raise ValueError("VALIDATION/TEST 必须明确审计为 NON_TRAIN_SPLIT")

    @property
    def event(self) -> SequenceEvaluationEvent:
        """返回被排除的原始 canonical event，不复制也不改写归因。"""

        return self.source.event


@dataclass(frozen=True, slots=True)
class HardExamplePlan:
    """稳定排序的训练权重和排除审计。"""

    routes: tuple[HardExampleRoute, ...]
    weights: tuple[HardExampleWeight, ...]
    excluded: tuple[ExcludedHardExample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.routes, tuple) or any(
            not isinstance(route, HardExampleRoute) for route in self.routes
        ):
            raise TypeError("routes 必须是 HardExampleRoute 元组")
        if not isinstance(self.weights, tuple) or any(
            not isinstance(weight, HardExampleWeight) for weight in self.weights
        ):
            raise TypeError("weights 必须是 HardExampleWeight 元组")
        if not isinstance(self.excluded, tuple) or any(
            not isinstance(item, ExcludedHardExample) for item in self.excluded
        ):
            raise TypeError("excluded 必须是 ExcludedHardExample 元组")
        if len(self.routes) != len(self.weights):
            raise ValueError("每个 route 必须具有且仅具有一个 weight")
        for route, weight in zip(self.routes, self.weights, strict=True):
            if weight.route is not route:
                raise ValueError("weight 必须按对象身份引用对应 route")
        route_ids = tuple(route.event.event_id for route in self.routes)
        excluded_ids = tuple(item.event.event_id for item in self.excluded)
        if len((*route_ids, *excluded_ids)) != len(set((*route_ids, *excluded_ids))):
            raise ValueError("HardExamplePlan 不得重复 event_id")
        if route_ids != tuple(sorted(route_ids)):
            raise ValueError("routes 必须按 event_id 稳定排序")
        excluded_keys = tuple(
            (item.source.split.value, item.event.event_id) for item in self.excluded
        )
        if excluded_keys != tuple(sorted(excluded_keys)):
            raise ValueError("excluded 必须按 split/event_id 稳定排序")

    def events_for(
        self,
        consumer: HardExampleConsumer,
    ) -> tuple[SequenceEvaluationEvent, ...]:
        """为 optimizer/telemetry/gallery 返回完全相同的 event 引用。"""

        if not isinstance(consumer, HardExampleConsumer):
            raise TypeError("consumer 必须是 HardExampleConsumer")
        return tuple(weight.event for weight in self.weights)


def build_hard_example_plan(
    inputs: Sequence[EvaluationSplitEvent],
) -> HardExamplePlan:
    """按 registry 路由失败 TRAIN event，并审计所有其余输入。"""

    if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
        raise TypeError("inputs 必须是 EvaluationSplitEvent 序列")
    checked = tuple(inputs)
    if any(not isinstance(item, EvaluationSplitEvent) for item in checked):
        raise TypeError("inputs 只能包含 EvaluationSplitEvent")
    event_ids = tuple(item.event.event_id for item in checked)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("hard-example 输入不得重复 event_id")

    registry = {spec.primary_error: spec for spec in HARD_EXAMPLE_ROUTE_REGISTRY}
    routes: list[HardExampleRoute] = []
    weights: list[HardExampleWeight] = []
    excluded: list[ExcludedHardExample] = []
    for item in sorted(checked, key=lambda value: value.event.event_id):
        event = item.event
        if item.split is not DataSplit.TRAIN:
            excluded.append(
                ExcludedHardExample(
                    item,
                    HardExampleExclusionReason.NON_TRAIN_SPLIT,
                )
            )
            continue
        if event.passed:
            excluded.append(
                ExcludedHardExample(item, HardExampleExclusionReason.PASSED)
            )
            continue
        if event.primary_error is PrimaryError.NONE:
            excluded.append(
                ExcludedHardExample(
                    item,
                    HardExampleExclusionReason.NO_PRIMARY_ERROR,
                )
            )
            continue
        spec = registry[event.primary_error]
        route = HardExampleRoute(item, spec.destination)
        routes.append(route)
        weights.append(HardExampleWeight(route, spec.weight))

    ordered = sorted(
        zip(routes, weights, strict=True),
        key=lambda pair: pair[0].event.event_id,
    )
    sorted_routes = tuple(pair[0] for pair in ordered)
    sorted_weights = tuple(pair[1] for pair in ordered)
    sorted_excluded = tuple(
        sorted(
            excluded,
            key=lambda item: (item.source.split.value, item.event.event_id),
        )
    )
    return HardExamplePlan(sorted_routes, sorted_weights, sorted_excluded)


__all__ = (
    "HARD_EXAMPLE_ROUTE_REGISTRY",
    "EvaluationSplitEvent",
    "ExcludedHardExample",
    "HardExampleConsumer",
    "HardExampleDestination",
    "HardExampleExclusionReason",
    "HardExamplePlan",
    "HardExampleRoute",
    "HardExampleRouteSpec",
    "HardExampleWeight",
    "build_hard_example_plan",
)
