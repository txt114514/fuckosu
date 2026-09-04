"""数据质量规则注册表与唯一训练门禁。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from traning.state import (
    DataQualityIssue,
    DataQualityReport,
    DataQualitySeverity,
    DataSplit,
    TrainingSample,
)


_CONCRETE_SPLITS = (DataSplit.TRAIN, DataSplit.VALIDATION, DataSplit.TEST)


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """数据集的确定性计数摘要。"""

    sample_count: int
    split_counts: tuple[tuple[DataSplit, int], ...]

    def __post_init__(self) -> None:
        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count, int
        ):
            raise TypeError("sample_count 必须是整数")
        if self.sample_count < 0:
            raise ValueError("sample_count 不得为负数")
        expected_splits = tuple(split for split, _ in self.split_counts)
        if expected_splits != _CONCRETE_SPLITS:
            raise ValueError("split_counts 必须按 train、validation、test 顺序完整列出")
        for split, count in self.split_counts:
            if not isinstance(split, DataSplit) or split is DataSplit.ALL:
                raise ValueError("split_counts 只能包含具体 DataSplit")
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"{split.value} 样本数必须是整数")
            if count < 0:
                raise ValueError(f"{split.value} 样本数不得为负数")
        if sum(count for _, count in self.split_counts) != self.sample_count:
            raise ValueError("split_counts 总和必须等于 sample_count")

    def count(self, split: DataSplit) -> int:
        """返回具体切分的样本数。"""

        if not isinstance(split, DataSplit) or split is DataSplit.ALL:
            raise ValueError("split 必须是具体 DataSplit")
        return dict(self.split_counts)[split]

    @classmethod
    def from_samples(cls, samples: Sequence[TrainingSample]) -> DatasetSummary:
        """从 typed 样本生成固定顺序的摘要。"""

        checked = _checked_samples(samples)
        counts = tuple(
            (split, sum(sample.split is split for sample in checked))
            for split in _CONCRETE_SPLITS
        )
        return cls(sample_count=len(checked), split_counts=counts)


@dataclass(frozen=True, slots=True)
class DataQualityContext:
    """供质量规则共享的 typed 数据上下文。"""

    samples: tuple[TrainingSample, ...]
    summary: DatasetSummary

    def __post_init__(self) -> None:
        _checked_samples(self.samples)
        if not isinstance(self.summary, DatasetSummary):
            raise TypeError("summary 必须是 DatasetSummary")

    @classmethod
    def from_samples(cls, samples: Sequence[TrainingSample]) -> DataQualityContext:
        """按稳定键排序样本并构建摘要。"""

        checked = _checked_samples(samples)
        ordered = tuple(
            sorted(
                checked,
                key=lambda sample: (
                    _CONCRETE_SPLITS.index(sample.split),
                    sample.sample_id,
                    sample.frame_index,
                ),
            )
        )
        return cls(samples=ordered, summary=DatasetSummary.from_samples(ordered))


@dataclass(frozen=True, slots=True)
class DataQualityFinding:
    """规则内部发现；公共 issue 元数据由注册表统一补齐。"""

    sample_id: str | None
    message: str
    details: tuple[tuple[str, str | int | float | bool | None], ...] = ()


_RuleEvaluator = Callable[[DataQualityContext], tuple[DataQualityFinding, ...]]


@dataclass(frozen=True, slots=True)
class DataQualityRule:
    """一条可替换的数据质量规则规格。"""

    code: str
    severity: DataQualitySeverity
    blocks_training: bool
    evaluate: _RuleEvaluator

    def __post_init__(self) -> None:
        if not self.code or self.code != self.code.strip():
            raise ValueError("质量规则 code 不得为空且不得有首尾空格")
        if not isinstance(self.severity, DataQualitySeverity):
            raise TypeError("质量规则 severity 必须是 DataQualitySeverity")
        if not isinstance(self.blocks_training, bool):
            raise TypeError("质量规则 blocks_training 必须是布尔值")
        if not callable(self.evaluate):
            raise TypeError("质量规则 evaluate 必须可调用")


def _checked_samples(samples: Sequence[TrainingSample]) -> tuple[TrainingSample, ...]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples 必须是 TrainingSample 序列")
    checked = tuple(samples)
    for index, sample in enumerate(checked):
        if not isinstance(sample, TrainingSample):
            raise TypeError(f"samples[{index}] 必须是 TrainingSample")
    return checked


def _summary_matches(catalog: DataQualityContext) -> tuple[DataQualityFinding, ...]:
    actual = DatasetSummary.from_samples(catalog.samples)
    if actual == catalog.summary:
        return ()
    return (
        DataQualityFinding(
            sample_id=None,
            message="数据摘要与 typed 样本不一致",
            details=(
                ("declared_sample_count", catalog.summary.sample_count),
                ("actual_sample_count", actual.sample_count),
            ),
        ),
    )


def _duplicate_sample_ids(
    catalog: DataQualityContext,
) -> tuple[DataQualityFinding, ...]:
    counts: dict[str, int] = {}
    for sample in catalog.samples:
        counts[sample.sample_id] = counts.get(sample.sample_id, 0) + 1
    return tuple(
        DataQualityFinding(
            sample_id=sample_id,
            message="sample_id 在数据集中重复",
            details=(("occurrences", count),),
        )
        for sample_id, count in sorted(counts.items())
        if count > 1
    )


def _missing_training_split(
    catalog: DataQualityContext,
) -> tuple[DataQualityFinding, ...]:
    if catalog.summary.count(DataSplit.TRAIN) > 0:
        return ()
    return (DataQualityFinding(sample_id=None, message="训练切分没有样本"),)


def _missing_evaluation_splits(
    catalog: DataQualityContext,
) -> tuple[DataQualityFinding, ...]:
    return tuple(
        DataQualityFinding(
            sample_id=None,
            message=f"{split.value} 切分没有样本",
            details=(("split", split.value),),
        )
        for split in (DataSplit.VALIDATION, DataSplit.TEST)
        if catalog.summary.count(split) == 0
    )


DEFAULT_QUALITY_RULES: tuple[DataQualityRule, ...] = (
    DataQualityRule(
        code="summary_mismatch",
        severity=DataQualitySeverity.ERROR,
        blocks_training=True,
        evaluate=_summary_matches,
    ),
    DataQualityRule(
        code="duplicate_sample_id",
        severity=DataQualitySeverity.ERROR,
        blocks_training=True,
        evaluate=_duplicate_sample_ids,
    ),
    DataQualityRule(
        code="missing_training_split",
        severity=DataQualitySeverity.ERROR,
        blocks_training=True,
        evaluate=_missing_training_split,
    ),
    DataQualityRule(
        code="missing_evaluation_split",
        severity=DataQualitySeverity.WARNING,
        blocks_training=False,
        evaluate=_missing_evaluation_splits,
    ),
)
"""默认规则表；展示严重度与阻断语义在此显式分离。"""


@dataclass(frozen=True, slots=True)
class DataQualityGate:
    """按注册顺序执行规则并生成 canonical report。"""

    rules: tuple[DataQualityRule, ...] = DEFAULT_QUALITY_RULES

    def __post_init__(self) -> None:
        if not isinstance(self.rules, tuple):
            raise TypeError("rules 必须是 DataQualityRule 元组")
        codes: list[str] = []
        for index, rule in enumerate(self.rules):
            if not isinstance(rule, DataQualityRule):
                raise TypeError(f"rules[{index}] 必须是 DataQualityRule")
            codes.append(rule.code)
        if len(codes) != len(set(codes)):
            raise ValueError("质量规则 code 不得重复")

    def evaluate(self, context: DataQualityContext) -> DataQualityReport:
        """对 typed 上下文执行全部注册规则。"""

        if not isinstance(context, DataQualityContext):
            raise TypeError("context 必须是 DataQualityContext")
        issues: list[DataQualityIssue] = []
        for rule in self.rules:
            issues.extend(
                DataQualityIssue(
                    code=rule.code,
                    severity=rule.severity,
                    blocks_training=rule.blocks_training,
                    sample_id=finding.sample_id,
                    message=finding.message,
                    details=finding.details,
                )
                for finding in rule.evaluate(context)
            )
        return DataQualityReport(issues=tuple(issues))

    def evaluate_samples(self, samples: Sequence[TrainingSample]) -> DataQualityReport:
        """确定性整理 typed 样本后执行质量门。"""

        return self.evaluate(DataQualityContext.from_samples(samples))


__all__ = (
    "DEFAULT_QUALITY_RULES",
    "DataQualityContext",
    "DataQualityFinding",
    "DataQualityGate",
    "DataQualityRule",
    "DatasetSummary",
)
