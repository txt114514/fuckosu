"""Error attribution for optimization."""

from traning.core.optimization.attribution.analyzer import (
    ATTRIBUTION_DOMAINS,
    AttributionSummary,
    HardExample,
    analyze_trial_attribution,
)

__all__ = [
    "ATTRIBUTION_DOMAINS",
    "AttributionSummary",
    "HardExample",
    "analyze_trial_attribution",
]
