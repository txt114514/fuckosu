"""离线 Outcome oracle、反事实数据集与后续 learned model 的领域入口。"""

from .calibration import (
    CalibrationEvaluation,
    ScalarTemperatureCalibrator,
    evaluate_temperature_calibration,
    fit_temperature_calibrator,
)
from .dataset import (
    OUTCOME_DATASET_ARTIFACT_TYPE,
    OUTCOME_DATASET_SCHEMA_VERSION,
    CounterfactualFrame,
    CounterfactualOutcomeDataset,
    CounterfactualOutcomeDatasetBuilder,
    OutcomeDatasetArtifactStore,
    OutcomeDatasetManifest,
)
from .oracle import (
    OUTCOME_ORACLE_VERSION,
    HypotheticalClick,
    OracleOutcome,
    OracleState,
    OracleTarget,
    OutcomeCategory,
    OutcomeOracle,
)
from .model import (
    OUTCOME_CATEGORY_COUNT,
    SCORE_REPRESENTATIVES,
    DenseOutcomeModel,
    OutcomeTensorOutput,
)
from .training import (
    OutcomeBatch,
    OutcomeEvaluationMetrics,
    OutcomeLoss,
    OutcomeLossWeights,
    collate_outcome_samples,
    compute_outcome_loss,
    evaluate_outcome_batch,
    train_outcome_step,
)

__all__ = (
    "OUTCOME_DATASET_ARTIFACT_TYPE",
    "OUTCOME_DATASET_SCHEMA_VERSION",
    "OUTCOME_ORACLE_VERSION",
    "OUTCOME_CATEGORY_COUNT",
    "SCORE_REPRESENTATIVES",
    "CalibrationEvaluation",
    "CounterfactualFrame",
    "CounterfactualOutcomeDataset",
    "CounterfactualOutcomeDatasetBuilder",
    "DenseOutcomeModel",
    "HypotheticalClick",
    "OracleOutcome",
    "OracleState",
    "OracleTarget",
    "OutcomeCategory",
    "OutcomeBatch",
    "OutcomeDatasetArtifactStore",
    "OutcomeDatasetManifest",
    "OutcomeEvaluationMetrics",
    "OutcomeLoss",
    "OutcomeLossWeights",
    "OutcomeOracle",
    "OutcomeTensorOutput",
    "ScalarTemperatureCalibrator",
    "collate_outcome_samples",
    "compute_outcome_loss",
    "evaluate_outcome_batch",
    "evaluate_temperature_calibration",
    "fit_temperature_calibrator",
    "train_outcome_step",
)
