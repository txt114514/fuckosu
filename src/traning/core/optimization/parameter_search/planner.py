from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal

from traning.core.optimization.attribution import ATTRIBUTION_DOMAINS, AttributionSummary
from traning.core.optimization.scoring import TrialScoreReport
from traning.state import CurriculumStage, SearchMethod, TrialStatus


ASHAAction = Literal["continue", "promote", "prune"]


@dataclass(frozen=True)
class ASHAConfig:
    min_trials_per_rung: int = 3
    prune_quantile: float = 0.25
    promotion_quantile: float = 0.75
    prune_quality_floor: float = 0.20
    stage_quality_thresholds: Mapping[CurriculumStage, float] = field(
        default_factory=lambda: {
            CurriculumStage.BASIC: 0.70,
            CurriculumStage.MULTI_OBJECT: 0.76,
            CurriculumStage.COMPLEX: 0.82,
            CurriculumStage.FULL: 0.88,
        }
    )

    def __post_init__(self) -> None:
        if self.min_trials_per_rung < 1:
            raise ValueError("min_trials_per_rung must be positive")
        for value in (
            self.prune_quantile,
            self.promotion_quantile,
            self.prune_quality_floor,
        ):
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError("ASHA thresholds must be in the 0..1 range")
        if self.prune_quantile >= self.promotion_quantile:
            raise ValueError("prune_quantile must be below promotion_quantile")
        object.__setattr__(
            self,
            "stage_quality_thresholds",
            dict(self.stage_quality_thresholds),
        )


@dataclass(frozen=True)
class ParameterSearchConfig:
    search_method: SearchMethod = SearchMethod.TPE
    asha: ASHAConfig = field(default_factory=ASHAConfig)
    target_peak_vram_mb: float | None = None
    max_hard_examples: int = 64

    def __post_init__(self) -> None:
        if self.target_peak_vram_mb is not None and self.target_peak_vram_mb <= 0:
            raise ValueError("target_peak_vram_mb must be positive")
        if self.max_hard_examples < 0:
            raise ValueError("max_hard_examples must be nonnegative")


@dataclass(frozen=True)
class TrialHistoryEntry:
    trial_id: str
    rung: int
    curriculum_stage: CurriculumStage
    quality_score: float
    status: TrialStatus = TrialStatus.COMPLETED
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValueError("trial_id must not be empty")
        if self.rung < 0:
            raise ValueError("rung must be nonnegative")
        if not isfinite(self.quality_score) or not 0 <= self.quality_score <= 1:
            raise ValueError("quality_score must be in the 0..1 range")
        object.__setattr__(self, "metrics", dict(self.metrics))


@dataclass(frozen=True)
class OptimizationPlan:
    trial_id: str
    search_method: SearchMethod
    asha_action: ASHAAction
    next_status: TrialStatus
    current_stage: CurriculumStage
    next_stage: CurriculumStage
    parameter_updates: Mapping[str, Mapping[str, Any]]
    hard_example_keys: tuple[str, ...]
    priority_domains: tuple[str, ...]
    reasons: tuple[str, ...]
    quality_score: float
    score_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "search_method": self.search_method.value,
            "asha_action": self.asha_action,
            "next_status": self.next_status.value,
            "current_stage": self.current_stage.value,
            "next_stage": self.next_stage.value,
            "parameter_updates": {
                section: dict(values)
                for section, values in self.parameter_updates.items()
            },
            "hard_example_keys": list(self.hard_example_keys),
            "priority_domains": list(self.priority_domains),
            "reasons": list(self.reasons),
            "quality_score": self.quality_score,
            "score_version": self.score_version,
        }


_STAGE_ORDER = (
    CurriculumStage.BASIC,
    CurriculumStage.MULTI_OBJECT,
    CurriculumStage.COMPLEX,
    CurriculumStage.FULL,
)


def _next_stage(stage: CurriculumStage) -> CurriculumStage:
    index = _STAGE_ORDER.index(stage)
    return _STAGE_ORDER[min(index + 1, len(_STAGE_ORDER) - 1)]


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _asha_action(
    report: TrialScoreReport,
    history: Sequence[TrialHistoryEntry],
    *,
    current_stage: CurriculumStage,
    rung: int,
    config: ASHAConfig,
) -> tuple[ASHAAction, tuple[str, ...]]:
    reasons: list[str] = []
    if report.quality_score < config.prune_quality_floor:
        return "prune", ("quality below ASHA prune floor",)

    comparable = [
        entry.quality_score
        for entry in history
        if entry.rung == rung
        and entry.curriculum_stage == current_stage
        and entry.status == TrialStatus.COMPLETED
    ]
    if len(comparable) >= config.min_trials_per_rung:
        prune_cutoff = _quantile(comparable, config.prune_quantile)
        promote_cutoff = _quantile(comparable, config.promotion_quantile)
        if report.quality_score < prune_cutoff:
            return "prune", (f"quality below ASHA prune cutoff {prune_cutoff:.4f}",)
        if report.quality_score >= promote_cutoff:
            reasons.append(f"quality reached ASHA promote cutoff {promote_cutoff:.4f}")

    stage_threshold = config.stage_quality_thresholds[current_stage]
    if report.quality_score >= stage_threshold:
        reasons.append(f"quality reached {current_stage.value} threshold {stage_threshold:.4f}")
        return "promote", tuple(reasons)
    return "continue", tuple(reasons or ("quality needs more budget",))


def _priority_domains(attribution: AttributionSummary) -> tuple[str, ...]:
    ranked = sorted(
        ATTRIBUTION_DOMAINS,
        key=lambda domain: (
            -attribution.domain_counts.get(domain, 0),
            domain,
        ),
    )
    return tuple(
        domain
        for domain in ranked
        if attribution.domain_counts.get(domain, 0) > 0
    )


def _hard_example_keys(
    attribution: AttributionSummary,
    *,
    limit: int,
) -> tuple[str, ...]:
    seen: set[str] = set()
    keys: list[str] = []
    for example in attribution.hard_examples:
        key = f"{example.sample_key}:{example.frame_index}:{example.primary_error}"
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
        if len(keys) >= limit:
            break
    return tuple(keys)


def _set_update(
    updates: dict[str, dict[str, Any]],
    section: str,
    name: str,
    value: Any,
) -> None:
    updates.setdefault(section, {})[name] = value


def _apply_domain_updates(
    updates: dict[str, dict[str, Any]],
    attribution: AttributionSummary,
    reasons: list[str],
) -> None:
    tags = attribution.tag_counts
    rates = attribution.domain_rates
    if rates.get("spatial", 0.0) > 0:
        _set_update(updates, "training", "spatial_loss_weight_multiplier", 1.15)
        _set_update(updates, "inference", "score_threshold_delta", -0.03)
        _set_update(updates, "inference", "max_candidates_delta", 4)
        reasons.append("spatial attribution increases spatial loss and candidate recall")
    if rates.get("temporal", 0.0) > 0:
        _set_update(updates, "training", "temporal_loss_weight_multiplier", 1.20)
        _set_update(updates, "training", "sequence_length_multiplier", 1.25)
        if tags.get("early_click", 0) >= tags.get("late_click", 0):
            _set_update(updates, "inference", "timing_bias_ms_delta", 10.0)
        else:
            _set_update(updates, "inference", "timing_bias_ms_delta", -10.0)
        reasons.append("temporal attribution adjusts sequence length and timing bias")
    if rates.get("decision", 0.0) > 0:
        _set_update(updates, "training", "decision_loss_weight_multiplier", 1.15)
        _set_update(updates, "inference", "cooldown_ms_delta", 5.0)
        _set_update(updates, "sampling", "hard_negative_multiplier", 1.20)
        reasons.append("decision attribution increases cooldown and hard negatives")


def _apply_overall_updates(
    updates: dict[str, dict[str, Any]],
    report: TrialScoreReport,
    config: ParameterSearchConfig,
    reasons: list[str],
) -> None:
    if report.quality_score < 0.5:
        _set_update(updates, "training", "budget_steps_multiplier", 1.50)
        _set_update(updates, "search", "exploration_probability", 0.35)
        reasons.append("low overall quality asks for more budget and exploration")
    if config.search_method == SearchMethod.TPE:
        _set_update(updates, "search", "sampler", "tpe")
        _set_update(updates, "search", "candidate_pool", "domain_weighted")
    else:
        _set_update(updates, "search", "sampler", "random")
    peak_vram = report.metrics.get("peak_vram_mb")
    if (
        peak_vram is not None
        and config.target_peak_vram_mb is not None
        and peak_vram > config.target_peak_vram_mb
    ):
        _set_update(updates, "training", "patch_limit_delta", -1)
        _set_update(updates, "training", "candidate_slots_delta", -4)
        reasons.append("peak VRAM exceeded target, reducing patch and candidate pressure")


def plan_next_trial(
    report: TrialScoreReport,
    attribution: AttributionSummary,
    *,
    history: Sequence[TrialHistoryEntry] = (),
    current_stage: CurriculumStage = CurriculumStage.BASIC,
    rung: int = 0,
    config: ParameterSearchConfig = ParameterSearchConfig(),
) -> OptimizationPlan:
    action, asha_reasons = _asha_action(
        report,
        history,
        current_stage=current_stage,
        rung=rung,
        config=config.asha,
    )
    reasons = list(asha_reasons)
    updates: dict[str, dict[str, Any]] = {}
    _apply_domain_updates(updates, attribution, reasons)
    _apply_overall_updates(updates, report, config, reasons)

    if action == "prune":
        next_status = TrialStatus.PRUNED
        next_stage = current_stage
    elif action == "promote":
        next_status = TrialStatus.PROMOTED
        next_stage = _next_stage(current_stage)
    else:
        next_status = TrialStatus.RUNNING
        next_stage = current_stage

    return OptimizationPlan(
        trial_id=report.trial_id,
        search_method=config.search_method,
        asha_action=action,
        next_status=next_status,
        current_stage=current_stage,
        next_stage=next_stage,
        parameter_updates=updates,
        hard_example_keys=_hard_example_keys(
            attribution,
            limit=config.max_hard_examples,
        ),
        priority_domains=_priority_domains(attribution),
        reasons=tuple(dict.fromkeys(reasons)),
        quality_score=report.quality_score,
        score_version=report.score_version,
    )


__all__ = [
    "ASHAAction",
    "ASHAConfig",
    "OptimizationPlan",
    "ParameterSearchConfig",
    "TrialHistoryEntry",
    "plan_next_trial",
]
