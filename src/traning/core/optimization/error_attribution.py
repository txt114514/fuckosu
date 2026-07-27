"""提供评分图集与优化分析共同消费的未解析目标归因规则。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class UnresolvedSample(Protocol):
    """归因规则所需的最小样本协议，避免反向依赖评分包入口。"""

    metadata: Mapping[str, Any]


def classify_unresolved_sample_error(
    sample: UnresolvedSample,
) -> tuple[str, tuple[str, ...], str]:
    """按最早发生的可观测失败边界归因一个未解析目标。"""

    metadata = sample.metadata
    if metadata.get("transform_status") == "unresolved":
        return (
            "spatial",
            ("unresolved_target", "coordinate_transform_unresolved"),
            "coordinate transform unresolved before target-candidate matching",
        )
    if "candidate_count" in metadata and int(metadata.get("candidate_count") or 0) <= 0:
        return (
            "spatial",
            ("unresolved_target", "candidate_recall_empty"),
            "target frame had no spatial candidates",
        )
    reason = str(metadata.get("candidate_match_unmatched_reason") or "")
    if reason:
        return (
            "spatial",
            ("unresolved_target", "candidate_match_failed", reason),
            f"target-candidate matching failed: {reason}",
        )
    return (
        "decision",
        ("unresolved_target",),
        "target remained active after all predicted clicks",
    )


__all__ = ["UnresolvedSample", "classify_unresolved_sample_error"]
