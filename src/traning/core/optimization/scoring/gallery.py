from __future__ import annotations

from collections.abc import Sequence

from traning.core.optimization.scoring.evaluator import (
    SampleScoreReport,
    TrialScoreReport,
)
from traning.state import (
    BatchGalleryRequest,
    FrameEvaluation,
    TrialGalleryEvaluation,
)


def _metadata_point(value: object) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def _representative_click(sample: SampleScoreReport):
    failures = tuple(
        click
        for click in sample.sequence.clicks
        if click.status != "hit" or click.primary_error != "none"
    )
    if failures:
        return max(
            failures,
            key=lambda click: (
                click.frequency_limited,
                abs(click.temporal_error_ms or 0.0),
                click.spatial_error or 0.0,
            ),
        )
    return sample.sequence.clicks[0] if sample.sequence.clicks else None


def _unresolved_source_index(sample: SampleScoreReport) -> int | None:
    for target_id in sample.sequence.unresolved_target_ids:
        suffix = target_id.rsplit(":", 1)[-1]
        try:
            return int(suffix)
        except ValueError:
            continue
    return None


def _frame_evaluation(sample: SampleScoreReport) -> FrameEvaluation:
    click = _representative_click(sample)
    if click is None:
        return FrameEvaluation(
            sample_key=sample.sample_key,
            frame_index=sample.frame_index,
            passed=sample.passed,
            target_source_index=_unresolved_source_index(sample),
            primary_error=(
                "decision" if sample.unresolved_count else "none"
            ),
            error_tags=(
                ("unresolved_target",) if sample.unresolved_count else ()
            ),
            metrics={"quality_score": sample.quality_score},
        )
    return FrameEvaluation(
        sample_key=sample.sample_key,
        frame_index=sample.frame_index,
        passed=sample.passed,
        target_source_index=click.source_index,
        predicted_osu_xy=(click.click.x, click.click.y),
        predicted_video_xy=_metadata_point(
            sample.metadata.get("predicted_video_xy")
        ),
        primary_error=click.primary_error,
        error_tags=tuple(click.error_tags),
        spatial_error=click.spatial_error,
        temporal_error_ms=click.temporal_error_ms,
        frequency_limited=click.frequency_limited,
        metrics={
            "quality_score": sample.quality_score,
            "object_score": sample.object_score,
        },
    )


def build_batch_gallery_request(
    report: TrialScoreReport,
    *,
    batch_id: str | None = None,
    random_seed: int = 2026,
    metadata: dict[str, object] | None = None,
) -> BatchGalleryRequest:
    """Build the result-export request directly from optimization scoring."""
    return BatchGalleryRequest(
        batch_id=batch_id or f"optimization_{report.trial_id}",
        random_seed=random_seed,
        metadata=dict(metadata or {}),
        trials=(
            TrialGalleryEvaluation(
                trial_id=report.trial_id,
                score=report.quality_score,
                score_version=report.score_version,
                parameters=report.parameters,
                metrics=dict(report.metrics),
                frames=tuple(
                    _frame_evaluation(sample)
                    for sample in report.samples
                ),
            ),
        ),
    )


__all__ = ["build_batch_gallery_request"]
