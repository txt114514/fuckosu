"""训练模块可依赖的稳定可视化 API、协议和状态模型入口。"""

from visualization.lib.api import create_dashboard_reporter
from visualization.lib.models import (
    BestParameterRecord,
    CurrentTrainingMetrics,
    DatasetUsageState,
    GalleryExportRequest,
    GalleryRenderRequest,
    GallerySelectionRequest,
    PipelinePhase,
    PipelineStageState,
    ResourceState,
    TrainingDashboardState,
    TrainingEvent,
    TrainingInheritanceSummary,
    TrainingStopState,
)
from visualization.lib.protocols import DashboardHandle, TrainingReporter
from visualization.lib.reporter import DashboardReporter, NullReporter
from visualization.lib.resources import collect_resource_state

__all__ = [
    "BestParameterRecord",
    "CurrentTrainingMetrics",
    "DashboardHandle",
    "DashboardReporter",
    "DatasetUsageState",
    "GalleryExportRequest",
    "GalleryRenderRequest",
    "GallerySelectionRequest",
    "NullReporter",
    "PipelinePhase",
    "PipelineStageState",
    "ResourceState",
    "TrainingDashboardState",
    "TrainingEvent",
    "TrainingInheritanceSummary",
    "TrainingReporter",
    "TrainingStopState",
    "collect_resource_state",
    "create_dashboard_reporter",
]
