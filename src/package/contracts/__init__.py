"""汇总 ``src`` 顶层模块之间共享的稳定、可持久化数据契约。"""

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
from package.contracts.geometry import (
    Box2D,
    Circle2D,
    CoordinateSpace,
    Point2D,
    Rect2D,
    ResizeMeta,
    Size2D,
)
from package.contracts.osu import (
    OsuDifficulty,
    OsuHitObject,
    OsuObjectType,
    OsuTimingPoint,
)

__all__ = [
    "ArtifactFileRef",
    "Box2D",
    "CandidateCacheFrameRef",
    "CheckpointRef",
    "Circle2D",
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
    "ResizeMeta",
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
