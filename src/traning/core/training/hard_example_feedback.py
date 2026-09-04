"""把 TRAIN canonical events 发布为下一 trial 可恢复的帧级反馈制品。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time

from traning.state import DataSplit
from traning.state.common import (
    require_identifier,
    require_sha256,
    require_transform_fingerprint,
)
from traning.core.data import SegmentTrainingDataset
from traning.core.evaluation import PrimaryError
from traning.lib.infrastructure import (
    IntegrityError,
    SchemaMismatchError,
    atomic_write_json,
    read_json_object,
)
from traning.core.training.hard_examples import (
    HARD_EXAMPLE_ROUTE_REGISTRY,
    HardExampleDestination,
    HardExampleExclusionReason,
    HardExamplePlan,
)
from traning.core.training.optimization import PARAMETER_REGISTRY, ParameterVector


HARD_EXAMPLE_FEEDBACK_SCHEMA_VERSION = 1
"""hard-example feedback 原子 JSON 的唯一活动 schema。"""

DEFAULT_HARD_EXAMPLE_BONUS = 1.0
"""route 权重总和转为帧级增益时使用的默认倍率。"""

DEFAULT_HARD_EXAMPLE_MAX_WEIGHT = 4.0
"""多个同帧事件聚合后的默认权重上限。"""

_EVENT_ID_PATTERN = re.compile(r"sequence-event-[0-9a-f]{64}")
_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "dataset_id",
        "config_sha256",
        "transform_fingerprint",
        "source_trial_index",
        "source_parameters",
        "evaluated",
        "bonus",
        "max_weight",
        "created_at_ms",
        "source_events",
        "frame_weights",
        "excluded",
        "payload_sha256",
    }
)
_SOURCE_EVENT_KEYS = frozenset(
    {
        "event_id",
        "sequence_id",
        "frame_index",
        "split",
        "primary_error",
        "destination",
        "route_weight",
    }
)
_FRAME_WEIGHT_KEYS = frozenset(
    {
        "destination",
        "sequence_id",
        "frame_index",
        "dataset_index",
        "event_ids",
        "route_weight_sum",
        "effective_weight",
    }
)
_EXCLUDED_KEYS = frozenset({"event_id", "split", "reason"})
_ROUTE_BY_PRIMARY_ERROR = {
    spec.primary_error: spec.destination for spec in HARD_EXAMPLE_ROUTE_REGISTRY
}
_PARAMETER_KEYS = tuple(spec.name for spec in PARAMETER_REGISTRY.specs)


def _require_event_id(value: str) -> None:
    """校验 canonical event identity，拒绝宽松任意字符串。"""

    if not isinstance(value, str) or _EVENT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("event_id 必须使用 canonical sequence-event SHA-256 格式")


def _require_nonnegative_integer(name: str, value: int) -> None:
    """校验布尔值不会冒充非负整数。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是整数")
    if value < 0:
        raise ValueError(f"{name} 不得为负数")


def _require_finite_positive(name: str, value: float) -> None:
    """校验采样或 loss multiplier 使用的有限正数。"""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} 必须是数值")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")


def _require_dataset_id(value: str) -> None:
    """校验 feedback 绑定到 canonical dataset 内容摘要。"""

    require_identifier(value, "dataset_id")
    if not value.startswith("dataset-"):
        raise ValueError("dataset_id 必须使用 dataset- 前缀")
    require_sha256(value.removeprefix("dataset-"))


@dataclass(frozen=True, slots=True)
class HardExampleSourceEvent:
    """一条只允许来自 TRAIN split 的 canonical feedback 来源。"""

    event_id: str
    sequence_id: str
    frame_index: int
    split: DataSplit
    primary_error: PrimaryError
    destination: HardExampleDestination
    route_weight: float

    def __post_init__(self) -> None:
        _require_event_id(self.event_id)
        require_identifier(self.sequence_id, "sequence_id")
        _require_nonnegative_integer("frame_index", self.frame_index)
        if self.split is not DataSplit.TRAIN:
            raise ValueError("只有 TRAIN event 可以成为 hard-example 权重来源")
        if not isinstance(self.primary_error, PrimaryError):
            raise TypeError("primary_error 必须是 PrimaryError")
        if self.primary_error is PrimaryError.NONE:
            raise ValueError("PrimaryError.NONE 不得成为 hard-example 来源")
        if not isinstance(self.destination, HardExampleDestination):
            raise TypeError("destination 必须是 HardExampleDestination")
        if _ROUTE_BY_PRIMARY_ERROR.get(self.primary_error) is not self.destination:
            raise ValueError("destination 必须服从唯一 hard-example route registry")
        _require_finite_positive("route_weight", self.route_weight)


@dataclass(frozen=True, slots=True)
class HardExampleFrameWeight:
    """以 sequence_id/frame_index 为身份的单领域有效训练权重。"""

    destination: HardExampleDestination
    sequence_id: str
    frame_index: int
    dataset_index: int
    event_ids: tuple[str, ...]
    route_weight_sum: float
    effective_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.destination, HardExampleDestination):
            raise TypeError("destination 必须是 HardExampleDestination")
        require_identifier(self.sequence_id, "sequence_id")
        _require_nonnegative_integer("frame_index", self.frame_index)
        _require_nonnegative_integer("dataset_index", self.dataset_index)
        if (
            not isinstance(self.event_ids, tuple)
            or not self.event_ids
            or any(not isinstance(value, str) for value in self.event_ids)
        ):
            raise TypeError("event_ids 必须是非空字符串元组")
        for event_id in self.event_ids:
            _require_event_id(event_id)
        if self.event_ids != tuple(sorted(set(self.event_ids))):
            raise ValueError("event_ids 必须去重并稳定排序")
        _require_finite_positive("route_weight_sum", self.route_weight_sum)
        _require_finite_positive("effective_weight", self.effective_weight)
        if self.effective_weight <= 1.0:
            raise ValueError("hard-example effective_weight 必须真正高于基线 1.0")

    @property
    def identity(self) -> tuple[str, int]:
        """返回不依赖帧级 sample_id 文本格式的 canonical 帧身份。"""

        return self.sequence_id, self.frame_index

    @property
    def aggregation_key(
        self,
    ) -> tuple[str, int, HardExampleDestination]:
        """返回规范要求的 sequence/frame/destination 聚合键。"""

        return self.sequence_id, self.frame_index, self.destination


@dataclass(frozen=True, slots=True)
class ExcludedFeedbackEvent:
    """未进入权重的 canonical event 及其 split 隔离原因。"""

    event_id: str
    split: DataSplit
    reason: HardExampleExclusionReason

    def __post_init__(self) -> None:
        _require_event_id(self.event_id)
        if not isinstance(self.split, DataSplit) or self.split is DataSplit.ALL:
            raise ValueError("excluded split 必须是具体 DataSplit")
        if not isinstance(self.reason, HardExampleExclusionReason):
            raise TypeError("reason 必须是 HardExampleExclusionReason")
        if self.split in (DataSplit.VALIDATION, DataSplit.TEST):
            if self.reason is not HardExampleExclusionReason.NON_TRAIN_SPLIT:
                raise ValueError("VALIDATION/TEST 必须审计为 NON_TRAIN_SPLIT")
        elif self.reason is HardExampleExclusionReason.NON_TRAIN_SPLIT:
            raise ValueError("TRAIN event 不得伪装成 NON_TRAIN_SPLIT")


@dataclass(frozen=True, slots=True)
class HardExampleFeedbackArtifact:
    """绑定 run/data/config/坐标和 source trial 的不可变反馈制品。"""

    schema_version: int
    run_id: str
    dataset_id: str
    config_sha256: str
    transform_fingerprint: str
    source_trial_index: int
    source_parameters: ParameterVector
    evaluated: bool
    bonus: float
    max_weight: float
    created_at_ms: float
    source_events: tuple[HardExampleSourceEvent, ...]
    frame_weights: tuple[HardExampleFrameWeight, ...]
    excluded: tuple[ExcludedFeedbackEvent, ...]
    payload_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != HARD_EXAMPLE_FEEDBACK_SCHEMA_VERSION:
            raise ValueError("hard-example feedback schema_version 不受当前代码支持")
        require_identifier(self.run_id, "run_id")
        _require_dataset_id(self.dataset_id)
        require_sha256(self.config_sha256)
        if self.config_sha256 != self.config_sha256.lower():
            raise ValueError("config_sha256 必须使用小写十六进制")
        require_transform_fingerprint(self.transform_fingerprint)
        _require_nonnegative_integer("source_trial_index", self.source_trial_index)
        if not isinstance(self.source_parameters, ParameterVector):
            raise TypeError("source_parameters 必须是 ParameterVector")
        if not isinstance(self.evaluated, bool):
            raise TypeError("evaluated 必须是 bool")
        _require_finite_positive("bonus", self.bonus)
        _require_finite_positive("max_weight", self.max_weight)
        if self.max_weight <= 1.0:
            raise ValueError("max_weight 必须大于基线 1.0")
        if (
            isinstance(self.created_at_ms, bool)
            or not isinstance(self.created_at_ms, int | float)
            or not math.isfinite(float(self.created_at_ms))
            or self.created_at_ms < 0.0
        ):
            raise ValueError("created_at_ms 必须是有限非负数")
        self._validate_records()
        require_sha256(self.payload_sha256)
        if self.payload_sha256 != self.payload_sha256.lower():
            raise ValueError("payload_sha256 必须使用小写十六进制")
        actual_sha256 = _json_sha256(_artifact_payload_to_json(self))
        if actual_sha256 != self.payload_sha256:
            raise IntegrityError("hard-example feedback payload SHA-256 不匹配")

    def _validate_records(self) -> None:
        """验证来源、聚合权重和排除审计是一一闭合的集合。"""

        if not isinstance(self.source_events, tuple) or any(
            not isinstance(item, HardExampleSourceEvent) for item in self.source_events
        ):
            raise TypeError("source_events 必须是 HardExampleSourceEvent 元组")
        if not isinstance(self.frame_weights, tuple) or any(
            not isinstance(item, HardExampleFrameWeight) for item in self.frame_weights
        ):
            raise TypeError("frame_weights 必须是 HardExampleFrameWeight 元组")
        if not isinstance(self.excluded, tuple) or any(
            not isinstance(item, ExcludedFeedbackEvent) for item in self.excluded
        ):
            raise TypeError("excluded 必须是 ExcludedFeedbackEvent 元组")
        if not self.evaluated and (
            self.source_events or self.frame_weights or self.excluded
        ):
            raise ValueError("evaluated=false 的反馈必须显式为空")

        source_ids = tuple(item.event_id for item in self.source_events)
        if source_ids != tuple(sorted(set(source_ids))):
            raise ValueError("source_events 必须按 event_id 去重并稳定排序")
        frame_keys = tuple(
            (item.destination.value, item.sequence_id, item.frame_index)
            for item in self.frame_weights
        )
        if frame_keys != tuple(sorted(set(frame_keys))):
            raise ValueError("frame_weights 必须按 destination/sequence/frame 去重排序")
        excluded_keys = tuple(
            (item.split.value, item.event_id) for item in self.excluded
        )
        if excluded_keys != tuple(sorted(set(excluded_keys))):
            raise ValueError("excluded 必须按 split/event_id 去重并稳定排序")
        excluded_ids = tuple(item.event_id for item in self.excluded)
        if set(source_ids).intersection(excluded_ids):
            raise ValueError("同一 event 不得同时进入权重和 excluded")

        grouped: dict[
            tuple[HardExampleDestination, str, int],
            list[HardExampleSourceEvent],
        ] = defaultdict(list)
        for source in self.source_events:
            grouped[
                (source.destination, source.sequence_id, source.frame_index)
            ].append(source)
        if set(grouped) != {
            (item.destination, item.sequence_id, item.frame_index)
            for item in self.frame_weights
        }:
            raise ValueError("每个 source event 必须且只能形成一个帧级权重")
        for frame_weight in self.frame_weights:
            sources = grouped[
                (
                    frame_weight.destination,
                    frame_weight.sequence_id,
                    frame_weight.frame_index,
                )
            ]
            expected_event_ids = tuple(sorted(item.event_id for item in sources))
            expected_route_weight_sum = sum(
                float(item.route_weight) for item in sources
            )
            expected_effective = min(
                float(self.max_weight),
                1.0 + float(self.bonus) * expected_route_weight_sum,
            )
            if frame_weight.event_ids != expected_event_ids:
                raise ValueError("frame weight 的 event_ids 与来源事件不一致")
            if not math.isclose(
                frame_weight.route_weight_sum,
                expected_route_weight_sum,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("frame weight 的 route 权重聚合值不一致")
            if not math.isclose(
                frame_weight.effective_weight,
                expected_effective,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "effective_weight 必须等于 "
                    "min(max_weight, 1 + bonus * sum(route.weight))"
                )

    def weights_for(
        self,
        destination: HardExampleDestination,
    ) -> tuple[HardExampleFrameWeight, ...]:
        """返回某模型领域可直接映射到 sampler/loss 的稳定权重视图。"""

        if not isinstance(destination, HardExampleDestination):
            raise TypeError("destination 必须是 HardExampleDestination")
        return tuple(
            item for item in self.frame_weights if item.destination is destination
        )


def build_hard_example_feedback(
    plan: HardExamplePlan | None,
    train_dataset: SegmentTrainingDataset,
    *,
    run_id: str,
    dataset_id: str,
    config_sha256: str,
    transform_fingerprint: str,
    source_trial_index: int,
    source_parameters: ParameterVector,
    evaluated: bool,
    bonus: float = DEFAULT_HARD_EXAMPLE_BONUS,
    max_weight: float = DEFAULT_HARD_EXAMPLE_MAX_WEIGHT,
    created_at_ms: float | None = None,
) -> HardExampleFeedbackArtifact:
    """解析 TRAIN sequence/frame，或显式发布未执行 evaluation 的空反馈。"""

    _validate_build_context(
        plan,
        train_dataset,
        run_id=run_id,
        dataset_id=dataset_id,
        config_sha256=config_sha256,
        transform_fingerprint=transform_fingerprint,
        source_trial_index=source_trial_index,
        source_parameters=source_parameters,
        evaluated=evaluated,
        bonus=bonus,
        max_weight=max_weight,
    )
    timestamp = time.time_ns() / 1_000_000.0 if created_at_ms is None else created_at_ms
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, int | float)
        or not math.isfinite(float(timestamp))
        or timestamp < 0.0
    ):
        raise ValueError("created_at_ms 必须是有限非负数")

    source_events = tuple(
        sorted(
            (
                HardExampleSourceEvent(
                    event_id=weight.event.event_id,
                    sequence_id=weight.event.sample_id,
                    frame_index=weight.event.frame_index,
                    split=weight.route.source.split,
                    primary_error=weight.event.primary_error,
                    destination=weight.route.destination,
                    route_weight=float(weight.weight),
                )
                for weight in (() if plan is None else plan.weights)
            ),
            key=lambda item: item.event_id,
        )
    )
    grouped: dict[
        tuple[HardExampleDestination, str, int],
        list[HardExampleSourceEvent],
    ] = defaultdict(list)
    for source in source_events:
        grouped[(source.destination, source.sequence_id, source.frame_index)].append(
            source
        )

    frame_weights: list[HardExampleFrameWeight] = []
    for destination, sequence_id, frame_index in sorted(
        grouped,
        key=lambda key: (key[0].value, key[1], key[2]),
    ):
        location = train_dataset.resolve_sequence_frame(sequence_id, frame_index)
        sources = grouped[(destination, sequence_id, frame_index)]
        route_weight_sum = sum(float(item.route_weight) for item in sources)
        frame_weights.append(
            HardExampleFrameWeight(
                destination=destination,
                sequence_id=sequence_id,
                frame_index=frame_index,
                dataset_index=location.dataset_index,
                event_ids=tuple(sorted(item.event_id for item in sources)),
                route_weight_sum=route_weight_sum,
                effective_weight=min(
                    float(max_weight),
                    1.0 + float(bonus) * route_weight_sum,
                ),
            )
        )
    excluded = tuple(
        sorted(
            (
                ExcludedFeedbackEvent(
                    event_id=item.event.event_id,
                    split=item.source.split,
                    reason=item.reason,
                )
                for item in (() if plan is None else plan.excluded)
            ),
            key=lambda item: (item.split.value, item.event_id),
        )
    )
    fields = {
        "schema_version": HARD_EXAMPLE_FEEDBACK_SCHEMA_VERSION,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "config_sha256": config_sha256,
        "transform_fingerprint": transform_fingerprint,
        "source_trial_index": source_trial_index,
        "source_parameters": source_parameters,
        "evaluated": evaluated,
        "bonus": float(bonus),
        "max_weight": float(max_weight),
        "created_at_ms": float(timestamp),
        "source_events": source_events,
        "frame_weights": tuple(frame_weights),
        "excluded": excluded,
    }
    payload_sha256 = _json_sha256(_artifact_payload_from_fields(**fields))
    return HardExampleFeedbackArtifact(
        **fields,
        payload_sha256=payload_sha256,
    )


class HardExampleFeedbackStore:
    """以单个原子 JSON 文件发布和恢复 source trial 的 hard feedback。"""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        dataset_id: str,
        config_sha256: str,
        transform_fingerprint: str,
        train_dataset: SegmentTrainingDataset,
    ) -> None:
        if not isinstance(path, Path):
            raise TypeError("path 必须是 pathlib.Path")
        _validate_store_context(
            train_dataset,
            run_id=run_id,
            dataset_id=dataset_id,
            config_sha256=config_sha256,
            transform_fingerprint=transform_fingerprint,
        )
        self.path = path
        self.run_id = run_id
        self.dataset_id = dataset_id
        self.config_sha256 = config_sha256
        self.transform_fingerprint = transform_fingerprint
        self.train_dataset = train_dataset

    def persist(
        self,
        plan: HardExamplePlan | None,
        *,
        source_trial_index: int,
        source_parameters: ParameterVector,
        evaluated: bool,
        bonus: float = DEFAULT_HARD_EXAMPLE_BONUS,
        max_weight: float = DEFAULT_HARD_EXAMPLE_MAX_WEIGHT,
        created_at_ms: float | None = None,
    ) -> HardExampleFeedbackArtifact:
        """构造完整制品并以 fsync+replace 原子提交。"""

        artifact = build_hard_example_feedback(
            plan,
            self.train_dataset,
            run_id=self.run_id,
            dataset_id=self.dataset_id,
            config_sha256=self.config_sha256,
            transform_fingerprint=self.transform_fingerprint,
            source_trial_index=source_trial_index,
            source_parameters=source_parameters,
            evaluated=evaluated,
            bonus=bonus,
            max_weight=max_weight,
            created_at_ms=created_at_ms,
        )
        atomic_write_json(self.path, _artifact_to_json(artifact))
        return artifact

    def load(
        self,
        *,
        expected_source_trial_index: int,
        expected_source_parameters: ParameterVector,
    ) -> HardExampleFeedbackArtifact:
        """严格验证摘要、运行身份和全部帧定位后返回反馈。"""

        _require_nonnegative_integer(
            "expected_source_trial_index", expected_source_trial_index
        )
        if not isinstance(expected_source_parameters, ParameterVector):
            raise TypeError("expected_source_parameters 必须是 ParameterVector")
        artifact = _artifact_from_json(read_json_object(self.path))
        expected_context = (
            self.run_id,
            self.dataset_id,
            self.config_sha256,
            self.transform_fingerprint,
            expected_source_trial_index,
            expected_source_parameters,
        )
        actual_context = (
            artifact.run_id,
            artifact.dataset_id,
            artifact.config_sha256,
            artifact.transform_fingerprint,
            artifact.source_trial_index,
            artifact.source_parameters,
        )
        if actual_context != expected_context:
            raise SchemaMismatchError(
                "hard-example feedback 与当前 run/data/config/transform/trial 不一致"
            )
        for frame_weight in artifact.frame_weights:
            try:
                location = self.train_dataset.resolve_sequence_frame(
                    frame_weight.sequence_id,
                    frame_weight.frame_index,
                )
            except KeyError as error:
                raise SchemaMismatchError(
                    "hard-example feedback 引用了当前 TRAIN dataset 中不存在的帧"
                ) from error
            if location.dataset_index != frame_weight.dataset_index:
                raise SchemaMismatchError(
                    "hard-example feedback 的 dataset_index 与 sequence/frame 不一致"
                )
        return artifact


def _validate_build_context(
    plan: HardExamplePlan | None,
    train_dataset: SegmentTrainingDataset,
    *,
    run_id: str,
    dataset_id: str,
    config_sha256: str,
    transform_fingerprint: str,
    source_trial_index: int,
    source_parameters: ParameterVector,
    evaluated: bool,
    bonus: float,
    max_weight: float,
) -> None:
    """在构造任何权重前统一验证 plan 与 store 身份。"""

    if not isinstance(evaluated, bool):
        raise TypeError("evaluated 必须是 bool")
    if evaluated and not isinstance(plan, HardExamplePlan):
        raise TypeError("evaluated=true 时 plan 必须是 HardExamplePlan")
    if not evaluated and plan is not None:
        raise ValueError("evaluated=false 时 plan 必须是 None")
    _validate_store_context(
        train_dataset,
        run_id=run_id,
        dataset_id=dataset_id,
        config_sha256=config_sha256,
        transform_fingerprint=transform_fingerprint,
    )
    _require_nonnegative_integer("source_trial_index", source_trial_index)
    if not isinstance(source_parameters, ParameterVector):
        raise TypeError("source_parameters 必须是 ParameterVector")
    _require_finite_positive("bonus", bonus)
    _require_finite_positive("max_weight", max_weight)
    if max_weight <= 1.0:
        raise ValueError("max_weight 必须大于基线 1.0")


def _validate_store_context(
    train_dataset: SegmentTrainingDataset,
    *,
    run_id: str,
    dataset_id: str,
    config_sha256: str,
    transform_fingerprint: str,
) -> None:
    """验证 store 只绑定具体 TRAIN dataset 与唯一坐标身份。"""

    if not isinstance(train_dataset, SegmentTrainingDataset):
        raise TypeError("train_dataset 必须是 SegmentTrainingDataset")
    if train_dataset.split is not DataSplit.TRAIN:
        raise ValueError("hard-example feedback store 只能绑定 TRAIN dataset")
    require_identifier(run_id, "run_id")
    _require_dataset_id(dataset_id)
    require_sha256(config_sha256)
    if config_sha256 != config_sha256.lower():
        raise ValueError("config_sha256 必须使用小写十六进制")
    require_transform_fingerprint(transform_fingerprint)
    if train_dataset.transform_fingerprint != transform_fingerprint:
        raise ValueError("TRAIN dataset 与 feedback store 的坐标指纹不一致")


def _artifact_payload_from_fields(
    *,
    schema_version: int,
    run_id: str,
    dataset_id: str,
    config_sha256: str,
    transform_fingerprint: str,
    source_trial_index: int,
    source_parameters: ParameterVector,
    evaluated: bool,
    bonus: float,
    max_weight: float,
    created_at_ms: float,
    source_events: tuple[HardExampleSourceEvent, ...],
    frame_weights: tuple[HardExampleFrameWeight, ...],
    excluded: tuple[ExcludedFeedbackEvent, ...],
) -> dict[str, object]:
    """把 typed 字段投影成摘要覆盖的 canonical JSON payload。"""

    return {
        "schema_version": schema_version,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "config_sha256": config_sha256,
        "transform_fingerprint": transform_fingerprint,
        "source_trial_index": source_trial_index,
        "source_parameters": _parameters_to_json(source_parameters),
        "evaluated": evaluated,
        "bonus": float(bonus),
        "max_weight": float(max_weight),
        "created_at_ms": float(created_at_ms),
        "source_events": [_source_event_to_json(item) for item in source_events],
        "frame_weights": [_frame_weight_to_json(item) for item in frame_weights],
        "excluded": [_excluded_to_json(item) for item in excluded],
    }


def _artifact_payload_to_json(
    artifact: HardExampleFeedbackArtifact,
) -> dict[str, object]:
    """返回不含自摘要字段的 artifact JSON。"""

    return _artifact_payload_from_fields(
        schema_version=artifact.schema_version,
        run_id=artifact.run_id,
        dataset_id=artifact.dataset_id,
        config_sha256=artifact.config_sha256,
        transform_fingerprint=artifact.transform_fingerprint,
        source_trial_index=artifact.source_trial_index,
        source_parameters=artifact.source_parameters,
        evaluated=artifact.evaluated,
        bonus=artifact.bonus,
        max_weight=artifact.max_weight,
        created_at_ms=artifact.created_at_ms,
        source_events=artifact.source_events,
        frame_weights=artifact.frame_weights,
        excluded=artifact.excluded,
    )


def _artifact_to_json(artifact: HardExampleFeedbackArtifact) -> dict[str, object]:
    """编码带自摘要提交点的完整 artifact JSON。"""

    payload = _artifact_payload_to_json(artifact)
    return {**payload, "payload_sha256": artifact.payload_sha256}


def _source_event_to_json(item: HardExampleSourceEvent) -> dict[str, object]:
    """序列化一个 TRAIN feedback 来源。"""

    return {
        "event_id": item.event_id,
        "sequence_id": item.sequence_id,
        "frame_index": item.frame_index,
        "split": item.split.value,
        "primary_error": item.primary_error.value,
        "destination": item.destination.value,
        "route_weight": float(item.route_weight),
    }


def _frame_weight_to_json(item: HardExampleFrameWeight) -> dict[str, object]:
    """序列化一个可直接消费的帧级 multiplier。"""

    return {
        "destination": item.destination.value,
        "sequence_id": item.sequence_id,
        "frame_index": item.frame_index,
        "dataset_index": item.dataset_index,
        "event_ids": list(item.event_ids),
        "route_weight_sum": float(item.route_weight_sum),
        "effective_weight": float(item.effective_weight),
    }


def _excluded_to_json(item: ExcludedFeedbackEvent) -> dict[str, object]:
    """序列化一条 split 隔离审计。"""

    return {
        "event_id": item.event_id,
        "split": item.split.value,
        "reason": item.reason.value,
    }


def _artifact_from_json(payload: dict[str, object]) -> HardExampleFeedbackArtifact:
    """先验 SHA-256，再把严格字段集合恢复为 typed artifact。"""

    if set(payload) != _ROOT_KEYS:
        raise SchemaMismatchError("hard-example feedback 根字段集合不匹配")
    stored_sha256 = _string(payload, "payload_sha256")
    try:
        require_sha256(stored_sha256)
    except (TypeError, ValueError) as error:
        raise SchemaMismatchError("payload_sha256 格式无效") from error
    if stored_sha256 != stored_sha256.lower():
        raise SchemaMismatchError("payload_sha256 必须使用小写十六进制")
    raw_without_sha = {
        key: value for key, value in payload.items() if key != "payload_sha256"
    }
    if _json_sha256(raw_without_sha) != stored_sha256:
        raise IntegrityError("hard-example feedback payload SHA-256 不匹配")
    try:
        source_events = tuple(
            _source_event_from_json(item)
            for item in _object_list(payload, "source_events")
        )
        frame_weights = tuple(
            _frame_weight_from_json(item)
            for item in _object_list(payload, "frame_weights")
        )
        excluded = tuple(
            _excluded_from_json(item) for item in _object_list(payload, "excluded")
        )
        return HardExampleFeedbackArtifact(
            schema_version=_integer(payload, "schema_version"),
            run_id=_string(payload, "run_id"),
            dataset_id=_string(payload, "dataset_id"),
            config_sha256=_string(payload, "config_sha256"),
            transform_fingerprint=_string(payload, "transform_fingerprint"),
            source_trial_index=_integer(payload, "source_trial_index"),
            source_parameters=_parameters_from_json(
                _object(payload, "source_parameters")
            ),
            evaluated=_boolean(payload, "evaluated"),
            bonus=_number(payload, "bonus"),
            max_weight=_number(payload, "max_weight"),
            created_at_ms=_number(payload, "created_at_ms"),
            source_events=source_events,
            frame_weights=frame_weights,
            excluded=excluded,
            payload_sha256=stored_sha256,
        )
    except IntegrityError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise SchemaMismatchError(
            "hard-example feedback typed schema 不匹配"
        ) from error


def _source_event_from_json(payload: dict[str, object]) -> HardExampleSourceEvent:
    """严格恢复一个 source event。"""

    if set(payload) != _SOURCE_EVENT_KEYS:
        raise SchemaMismatchError("source event 字段集合不匹配")
    return HardExampleSourceEvent(
        event_id=_string(payload, "event_id"),
        sequence_id=_string(payload, "sequence_id"),
        frame_index=_integer(payload, "frame_index"),
        split=DataSplit(_string(payload, "split")),
        primary_error=PrimaryError(_string(payload, "primary_error")),
        destination=HardExampleDestination(_string(payload, "destination")),
        route_weight=_number(payload, "route_weight"),
    )


def _frame_weight_from_json(payload: dict[str, object]) -> HardExampleFrameWeight:
    """严格恢复一个 frame multiplier。"""

    if set(payload) != _FRAME_WEIGHT_KEYS:
        raise SchemaMismatchError("frame weight 字段集合不匹配")
    event_ids = payload["event_ids"]
    if not isinstance(event_ids, list) or any(
        not isinstance(item, str) for item in event_ids
    ):
        raise SchemaMismatchError("frame weight event_ids 必须是字符串数组")
    return HardExampleFrameWeight(
        destination=HardExampleDestination(_string(payload, "destination")),
        sequence_id=_string(payload, "sequence_id"),
        frame_index=_integer(payload, "frame_index"),
        dataset_index=_integer(payload, "dataset_index"),
        event_ids=tuple(event_ids),
        route_weight_sum=_number(payload, "route_weight_sum"),
        effective_weight=_number(payload, "effective_weight"),
    )


def _excluded_from_json(payload: dict[str, object]) -> ExcludedFeedbackEvent:
    """严格恢复一条 excluded split 审计。"""

    if set(payload) != _EXCLUDED_KEYS:
        raise SchemaMismatchError("excluded event 字段集合不匹配")
    return ExcludedFeedbackEvent(
        event_id=_string(payload, "event_id"),
        split=DataSplit(_string(payload, "split")),
        reason=HardExampleExclusionReason(_string(payload, "reason")),
    )


def _object_list(
    payload: dict[str, object],
    key: str,
) -> tuple[dict[str, object], ...]:
    """读取只包含 JSON object 的数组。"""

    value = payload[key]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise SchemaMismatchError(f"{key} 必须是 object 数组")
    return tuple(value)  # type: ignore[return-value]


def _object(payload: dict[str, object], key: str) -> dict[str, object]:
    """读取严格 JSON object 字段。"""

    value = payload[key]
    if not isinstance(value, dict):
        raise SchemaMismatchError(f"{key} 必须是 object")
    return value


def _parameters_to_json(parameters: ParameterVector) -> dict[str, object]:
    """按参数 registry 的唯一字段顺序编码 source proposal。"""

    if not isinstance(parameters, ParameterVector):
        raise TypeError("parameters 必须是 ParameterVector")
    return {name: getattr(parameters, name) for name in _PARAMETER_KEYS}


def _parameters_from_json(payload: dict[str, object]) -> ParameterVector:
    """从严格字段集合恢复 source proposal，不允许缺字段或额外字段。"""

    if tuple(sorted(payload)) != tuple(sorted(_PARAMETER_KEYS)):
        raise SchemaMismatchError("source_parameters 字段集合不匹配")
    return ParameterVector(
        learning_rate=_number(payload, "learning_rate"),
        score_threshold=_number(payload, "score_threshold"),
        max_candidates=_integer(payload, "max_candidates"),
        risk_lambda=_number(payload, "risk_lambda"),
        wait_cost=_number(payload, "wait_cost"),
        min_confidence=_number(payload, "min_confidence"),
    )


def _string(payload: dict[str, object], key: str) -> str:
    """读取严格字符串字段。"""

    value = payload[key]
    if not isinstance(value, str):
        raise SchemaMismatchError(f"{key} 必须是字符串")
    return value


def _boolean(payload: dict[str, object], key: str) -> bool:
    """读取严格布尔字段，不接受整数替代。"""

    value = payload[key]
    if not isinstance(value, bool):
        raise SchemaMismatchError(f"{key} 必须是 bool")
    return value


def _integer(payload: dict[str, object], key: str) -> int:
    """读取严格整数字段，拒绝 JSON bool。"""

    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaMismatchError(f"{key} 必须是整数")
    return value


def _number(payload: dict[str, object], key: str) -> float:
    """读取严格有限数值字段。"""

    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaMismatchError(f"{key} 必须是数值")
    if not math.isfinite(float(value)):
        raise SchemaMismatchError(f"{key} 必须是有限数值")
    return float(value)


def _json_sha256(payload: object) -> str:
    """计算排序、无 NaN 的 UTF-8 canonical JSON 摘要。"""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = (
    "DEFAULT_HARD_EXAMPLE_BONUS",
    "DEFAULT_HARD_EXAMPLE_MAX_WEIGHT",
    "HARD_EXAMPLE_FEEDBACK_SCHEMA_VERSION",
    "ExcludedFeedbackEvent",
    "HardExampleFeedbackArtifact",
    "HardExampleFeedbackStore",
    "HardExampleFrameWeight",
    "HardExampleSourceEvent",
    "build_hard_example_feedback",
)
