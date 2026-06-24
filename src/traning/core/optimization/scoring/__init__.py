"""Trial-level scoring for optimization."""

from traning.core.optimization.scoring.evaluator import (
    AGGREGATE_SCORE_VERSION,
    SampleScoreReport,
    SampleScoringInput,
    TrialScoreReport,
    TrialScoreSpec,
    score_sample,
    score_trial,
)
from traning.core.optimization.scoring.gallery import build_batch_gallery_request

__all__ = [
    "AGGREGATE_SCORE_VERSION",
    "SampleScoreReport",
    "SampleScoringInput",
    "TrialScoreReport",
    "TrialScoreSpec",
    "build_batch_gallery_request",
    "score_sample",
    "score_trial",
]
