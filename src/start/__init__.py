"""提供仓库统一启动、自检、样本发现与模块登记入口。"""

from start.checks import (
    StartupCheckReport,
    StartupCheckResult,
    TrainingStartupCheckReport,
    run_startup_checks,
    run_training_startup_checks,
)
from start.modules import (
    START_ENTRY,
    SourceModuleEntry,
    source_module_entries,
    source_module_entry,
)
from start.samples import MatchedSample, MatchedSampleManifest

__all__ = [
    "START_ENTRY",
    "SourceModuleEntry",
    "StartupCheckReport",
    "StartupCheckResult",
    "TrainingStartupCheckReport",
    "MatchedSample",
    "MatchedSampleManifest",
    "run_startup_checks",
    "run_training_startup_checks",
    "source_module_entries",
    "source_module_entry",
]
