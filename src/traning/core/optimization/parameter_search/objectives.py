from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite

from traning.core.optimization.scoring import TrialScoreReport


DEFAULT_OBJECTIVE_WEIGHTS: dict[str, float] = {
    "quality_score": 1.0,
    "peak_vram_mb": -0.00005,
    "latency_ms": -0.001,
}


@dataclass(frozen=True)
class ObjectiveScore:
    values: Mapping[str, float]
    weights: Mapping[str, float] = field(default_factory=lambda: DEFAULT_OBJECTIVE_WEIGHTS)

    @property
    def composite_score(self) -> float:
        return sum(
            float(self.values.get(name, 0.0)) * weight
            for name, weight in self.weights.items()
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "multi-objective-v1",
            "values": dict(self.values),
            "weights": dict(self.weights),
            "composite_score": self.composite_score,
            "sort_key": self.sort_key(),
        }

    def sort_key(self) -> tuple[float, float, float]:
        return (
            self.composite_score,
            float(self.values.get("quality_score", 0.0)),
            -float(self.values.get("latency_ms", 0.0)),
        )


def objective_values_from_report(report: TrialScoreReport) -> dict[str, float]:
    values = {"quality_score": float(report.quality_score)}
    for name in ("peak_vram_mb", "latency_ms"):
        value = report.metrics.get(name)
        if value is not None and isfinite(value):
            values[name] = float(value)
    return values


def score_trial_objectives(
    report: TrialScoreReport,
    *,
    weights: Mapping[str, float] | None = None,
) -> ObjectiveScore:
    return ObjectiveScore(
        values=objective_values_from_report(report),
        weights=dict(weights or DEFAULT_OBJECTIVE_WEIGHTS),
    )


__all__ = [
    "DEFAULT_OBJECTIVE_WEIGHTS",
    "ObjectiveScore",
    "objective_values_from_report",
    "score_trial_objectives",
]
