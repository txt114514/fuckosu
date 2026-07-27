"""优化流程错误归因的公开入口。"""

from traning.core.optimization.attribution.analyzer import (
    ATTRIBUTION_DOMAINS,
    AttributionSummary,
    HardExample,
    analyze_trial_attribution,
)
from traning.core.optimization.error_attribution import (
    classify_unresolved_sample_error,
)

__all__ = [
    "ATTRIBUTION_DOMAINS",
    "AttributionSummary",
    "HardExample",
    "analyze_trial_attribution",
    "classify_unresolved_sample_error",
]
