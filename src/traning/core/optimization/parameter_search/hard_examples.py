from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from traning.core.optimization.attribution import AttributionSummary


@dataclass(frozen=True)
class HardExampleSamplingPlan:
    sample_weights: Mapping[str, float] = field(default_factory=dict)
    reasons: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_weights": dict(self.sample_weights),
            "reasons": {
                key: list(values)
                for key, values in self.reasons.items()
            },
        }


def build_hard_example_sampling_plan(
    attribution: AttributionSummary,
    *,
    base_weight: float = 1.0,
    severity_multiplier: float = 1.5,
    max_examples: int = 128,
) -> HardExampleSamplingPlan:
    if base_weight <= 0:
        raise ValueError("base_weight must be positive")
    if severity_multiplier < 0:
        raise ValueError("severity_multiplier must be nonnegative")
    if max_examples < 0:
        raise ValueError("max_examples must be nonnegative")

    weights: dict[str, float] = {}
    reasons: dict[str, list[str]] = defaultdict(list)
    for example in attribution.hard_examples[:max_examples]:
        weight = base_weight + severity_multiplier * max(0.0, example.severity)
        weights[example.sample_key] = max(
            weights.get(example.sample_key, base_weight),
            weight,
        )
        reason = example.primary_error
        if example.error_tags:
            reason += ":" + ",".join(example.error_tags)
        reasons[example.sample_key].append(reason)
    return HardExampleSamplingPlan(
        sample_weights=weights,
        reasons={key: tuple(values) for key, values in reasons.items()},
    )


__all__ = ["HardExampleSamplingPlan", "build_hard_example_sampling_plan"]
