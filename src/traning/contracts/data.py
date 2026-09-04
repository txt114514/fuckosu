"""已弃用兼容转发；新代码必须导入对应的 conf、core、lib 或 state 路径。"""

from traning.state.data import (
    DataSplit,
    GroundTruthObject,
    InferenceCandidateRecord,
    OutcomeTrainingSample,
    RuntimeFrame,
    TrainingCandidateRecord,
    TrainingSample,
)

__deprecated__ = True
__all__ = [
    "DataSplit",
    "GroundTruthObject",
    "InferenceCandidateRecord",
    "OutcomeTrainingSample",
    "RuntimeFrame",
    "TrainingCandidateRecord",
    "TrainingSample",
]
