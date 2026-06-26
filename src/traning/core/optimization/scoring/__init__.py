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
from traning.core.optimization.scoring.run_outputs import (
    DEFAULT_CIRCLE_RADIUS_OSU,
    DecisionOutputScoreResult,
    score_decision_outputs,
)

__all__ = [
    "AGGREGATE_SCORE_VERSION",
    "DEFAULT_CIRCLE_RADIUS_OSU",
    "DecisionOutputScoreResult",
    "SampleScoreReport",
    "SampleScoringInput",
    "TrialScoreReport",
    "TrialScoreSpec",
    "build_batch_gallery_request",
    "score_decision_outputs",
    "score_sample",
    "score_trial",
]
