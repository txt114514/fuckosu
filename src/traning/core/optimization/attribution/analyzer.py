from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from traning.core.optimization.scoring import SampleScoreReport, TrialScoreReport


ATTRIBUTION_DOMAINS = ("spatial", "temporal", "decision")


@dataclass(frozen=True)
class HardExample:
    sample_key: str
    subproject: str
    primary_error: str
    error_tags: tuple[str, ...]
    severity: float
    frame_index: int
    click_index: int | None = None
    target_id: str | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_key": self.sample_key,
            "subproject": self.subproject,
            "primary_error": self.primary_error,
            "error_tags": list(self.error_tags),
            "severity": self.severity,
            "frame_index": self.frame_index,
            "click_index": self.click_index,
            "target_id": self.target_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AttributionSummary:
    domain_counts: Mapping[str, int]
    domain_rates: Mapping[str, float]
    tag_counts: Mapping[str, int]
    primary_domain: str | None
    hard_examples: tuple[HardExample, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain_counts": dict(self.domain_counts),
            "domain_rates": dict(self.domain_rates),
            "tag_counts": dict(self.tag_counts),
            "primary_domain": self.primary_domain,
            "hard_examples": [
                example.as_dict() for example in self.hard_examples
            ],
        }


def _click_severity(click) -> float:
    base = 1.0 if click.status != "hit" else 0.0
    if click.frequency_limited:
        base += 0.75
    if click.spatial_error is not None:
        base += min(1.0, click.spatial_error / 128.0)
    if click.temporal_error_ms is not None:
        base += min(1.0, abs(click.temporal_error_ms) / 200.0)
    if click.score is not None:
        base += 1.0 - click.score.score.normalized
    return base


def _unresolved_example(
    sample: SampleScoreReport,
    target_id: str,
) -> HardExample:
    return HardExample(
        sample_key=sample.sample_key,
        subproject=sample.subproject,
        primary_error="decision",
        error_tags=("unresolved_target",),
        severity=2.0,
        frame_index=sample.frame_index,
        target_id=target_id,
        reason="target remained active after all predicted clicks",
    )


def analyze_trial_attribution(
    report: TrialScoreReport,
    *,
    max_hard_examples: int = 32,
) -> AttributionSummary:
    if max_hard_examples < 0:
        raise ValueError("max_hard_examples must be nonnegative")

    domain_counts: Counter[str] = Counter({domain: 0 for domain in ATTRIBUTION_DOMAINS})
    tag_counts: Counter[str] = Counter()
    hard_examples: list[HardExample] = []

    for sample in report.samples:
        for click in sample.sequence.clicks:
            if click.primary_error in ATTRIBUTION_DOMAINS:
                domain_counts[click.primary_error] += 1
            tag_counts.update(click.error_tags)
            if click.status == "hit" and click.primary_error == "none":
                continue
            severity = _click_severity(click)
            hard_examples.append(
                HardExample(
                    sample_key=sample.sample_key,
                    subproject=sample.subproject,
                    primary_error=click.primary_error,
                    error_tags=tuple(click.error_tags),
                    severity=severity,
                    frame_index=sample.frame_index,
                    click_index=click.click_index,
                    target_id=click.target_id,
                    reason=click.status,
                )
            )
        for target_id in sample.sequence.unresolved_target_ids:
            domain_counts["decision"] += 1
            tag_counts["unresolved_target"] += 1
            hard_examples.append(_unresolved_example(sample, target_id))

    total_errors = sum(domain_counts.values())
    domain_rates = {
        domain: (domain_counts[domain] / total_errors if total_errors else 0.0)
        for domain in ATTRIBUTION_DOMAINS
    }
    primary_domain = (
        max(ATTRIBUTION_DOMAINS, key=lambda domain: (domain_counts[domain], domain))
        if total_errors
        else None
    )
    if primary_domain is not None and domain_counts[primary_domain] == 0:
        primary_domain = None
    ranked = tuple(
        sorted(
            hard_examples,
            key=lambda example: (
                -example.severity,
                example.sample_key,
                example.click_index if example.click_index is not None else -1,
            ),
        )[:max_hard_examples]
    )
    return AttributionSummary(
        domain_counts=dict(domain_counts),
        domain_rates=domain_rates,
        tag_counts=dict(tag_counts),
        primary_domain=primary_domain,
        hard_examples=ranked,
    )


__all__ = [
    "ATTRIBUTION_DOMAINS",
    "AttributionSummary",
    "HardExample",
    "analyze_trial_attribution",
]
