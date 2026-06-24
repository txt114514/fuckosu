"""Stable data contracts shared across top-level src modules."""

from package.contracts.artifacts import ArtifactFileRef, VersionedArtifactRef
from package.contracts.base import ContractMixin, contract_to_dict
from package.contracts.candidate import (
    CandidateCacheFrameRef,
    DecisionFrameRecord,
    SliderPathCandidateRef,
    SpatialCandidateRef,
    TemporalTargetRef,
)
from package.contracts.dataset import (
    DataSplit,
    DatasetDimension,
    FrameSampleRef,
    SegmentCategory,
    SegmentManifestEntry,
    SegmentRef,
    TrainingItemRef,
)
from package.contracts.evaluation import (
    ErrorDomain,
    EvaluationOutcome,
    FrameRef,
    PredictionAction,
    PredictionEvent,
    ScoreSummary,
)
from package.contracts.experiment import (
    CheckpointRef,
    CurriculumStage,
    ScoreVersionRef,
    SearchMethod,
    TrialParametersRef,
    TrialRef,
    TrialStatus,
)
from package.contracts.geometry import CoordinateSpace, Point2D, Rect2D, Size2D
from package.contracts.osu import (
    OsuDifficulty,
    OsuHitObject,
    OsuObjectType,
    OsuTimingPoint,
)

__all__ = [
    "ArtifactFileRef",
    "CandidateCacheFrameRef",
    "CheckpointRef",
    "ContractMixin",
    "CoordinateSpace",
    "CurriculumStage",
    "DataSplit",
    "DatasetDimension",
    "DecisionFrameRecord",
    "ErrorDomain",
    "EvaluationOutcome",
    "FrameSampleRef",
    "FrameRef",
    "OsuDifficulty",
    "OsuHitObject",
    "OsuObjectType",
    "OsuTimingPoint",
    "Point2D",
    "PredictionAction",
    "PredictionEvent",
    "Rect2D",
    "ScoreSummary",
    "ScoreVersionRef",
    "SearchMethod",
    "SegmentCategory",
    "SegmentManifestEntry",
    "SegmentRef",
    "SliderPathCandidateRef",
    "Size2D",
    "SpatialCandidateRef",
    "TemporalTargetRef",
    "TrainingItemRef",
    "TrialParametersRef",
    "TrialRef",
    "TrialStatus",
    "VersionedArtifactRef",
    "contract_to_dict",
]
