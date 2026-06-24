from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from traning.core.optimization.scoring import SampleScoreReport


@dataclass(frozen=True)
class SubprojectPassRule:
    consecutive_passes: int
    max_failures: int
    max_samples: int

    def __post_init__(self) -> None:
        if self.consecutive_passes < 1:
            raise ValueError("consecutive_passes must be positive")
        if self.max_failures < 0:
            raise ValueError("max_failures must be nonnegative")
        if self.max_samples < self.consecutive_passes:
            raise ValueError("max_samples must cover the pass window")


DEFAULT_CURRICULUM_RULES: Mapping[str, SubprojectPassRule] = {
    "single_point": SubprojectPassRule(15, 2, 40),
    "slider": SubprojectPassRule(10, 2, 35),
    "multi_point": SubprojectPassRule(8, 3, 35),
    "point_slider": SubprojectPassRule(6, 3, 30),
}


@dataclass(frozen=True)
class SubprojectGateResult:
    subproject: str
    passed: bool
    longest_pass_streak: int
    failure_count: int
    evaluated_count: int
    rule: SubprojectPassRule

    def as_dict(self) -> dict[str, Any]:
        return {
            "subproject": self.subproject,
            "passed": self.passed,
            "longest_pass_streak": self.longest_pass_streak,
            "failure_count": self.failure_count,
            "evaluated_count": self.evaluated_count,
            "rule": {
                "consecutive_passes": self.rule.consecutive_passes,
                "max_failures": self.rule.max_failures,
                "max_samples": self.rule.max_samples,
            },
        }


@dataclass(frozen=True)
class CurriculumGateResult:
    passed: bool
    subprojects: Mapping[str, SubprojectGateResult] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "subprojects": {
                name: result.as_dict()
                for name, result in self.subprojects.items()
            },
        }


def _gate_subproject(
    subproject: str,
    samples: Sequence[SampleScoreReport],
    rule: SubprojectPassRule,
) -> SubprojectGateResult:
    window = tuple(samples[: rule.max_samples])
    streak = 0
    longest = 0
    failures = 0
    for sample in window:
        if sample.passed:
            streak += 1
            longest = max(longest, streak)
        else:
            failures += 1
            streak = 0
    return SubprojectGateResult(
        subproject=subproject,
        passed=(
            longest >= rule.consecutive_passes
            and failures <= rule.max_failures
        ),
        longest_pass_streak=longest,
        failure_count=failures,
        evaluated_count=len(window),
        rule=rule,
    )


def evaluate_curriculum_gate(
    samples: Sequence[SampleScoreReport],
    *,
    rules: Mapping[str, SubprojectPassRule] = DEFAULT_CURRICULUM_RULES,
) -> CurriculumGateResult:
    grouped: dict[str, list[SampleScoreReport]] = defaultdict(list)
    for sample in samples:
        if sample.subproject in rules:
            grouped[sample.subproject].append(sample)

    results = {
        subproject: _gate_subproject(
            subproject,
            grouped.get(subproject, ()),
            rule,
        )
        for subproject, rule in rules.items()
    }
    return CurriculumGateResult(
        passed=all(result.passed for result in results.values()),
        subprojects=results,
    )


__all__ = [
    "CurriculumGateResult",
    "DEFAULT_CURRICULUM_RULES",
    "SubprojectGateResult",
    "SubprojectPassRule",
    "evaluate_curriculum_gate",
]
