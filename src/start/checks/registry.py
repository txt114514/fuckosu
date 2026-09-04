"""登记并执行模块、环境、V2 配置、设备和 canonical 数据质量检查。"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass

import torch

from package.checks import StartupCheckReport, StartupCheckResult
from start.checks.models import TrainingStartupCheckReport
from start.modules import SourceModuleEntry, source_module_entries
from traning.conf import RuntimeDevice, V2Config
from traning.lib.environment import (
    EnvironmentReport as HostEnvironmentReport,
    collect_environment_report,
)
from traning.lib.environment.training import (
    EnvironmentCheckStatus,
    check_v2_environment,
)
from traning.state import DataQualityReport, DataSplit


CheckRunner = Callable[[], StartupCheckResult]


@dataclass(frozen=True, slots=True)
class ProgressiveCheck:
    """带层级的惰性检查；构造注册表不会执行实际探测。"""

    key: str
    level: int
    description: str
    run: CheckRunner


def check_source_module_import(entry: SourceModuleEntry) -> StartupCheckResult:
    """只解析模块 spec，避免检查阶段触发业务副作用。"""

    spec = importlib.util.find_spec(entry.import_name)
    return StartupCheckResult(
        key=f"module:{entry.key}",
        status="passed" if spec is not None else "failed",
        message=(
            f"{entry.import_name} import spec resolved"
            if spec is not None
            else f"{entry.import_name} is not importable"
        ),
        details=entry.as_dict(),
    )


def check_src_module_imports() -> tuple[StartupCheckResult, ...]:
    """检查 start 登记的全部顶层源码模块。"""

    return tuple(
        check_source_module_import(entry)
        for entry in source_module_entries(include_start=True)
    )


def check_environment(
    *,
    report: HostEnvironmentReport | None = None,
    require_cuda: bool = False,
) -> StartupCheckResult:
    """把宿主环境只读报告归约为共享启动检查项。"""

    selected = report or collect_environment_report()
    ready = selected.ready(require_cuda=require_cuda)
    missing = ", ".join(selected.missing_required_packages) or "none"
    message = (
        "required runtime dependencies are available"
        if ready
        else f"environment is not ready; missing={missing}"
    )
    if require_cuda and not selected.torch.cuda_available:
        message = f"{message}; cuda unavailable"
    return StartupCheckResult(
        key="environment",
        status="passed" if ready else "failed",
        message=message,
        details={
            "python": selected.python_version,
            "ffmpeg": selected.ffmpeg_path,
            "nvidia_smi": selected.nvidia_smi_path,
            "torch_available": selected.torch.available,
            "torch_version": selected.torch.version,
            "torch_cuda": selected.torch.torch_cuda,
            "cuda_available": selected.torch.cuda_available,
            "gpu": selected.torch.gpu_name,
            "missing_required_packages": selected.missing_required_packages,
        },
    )


def check_training_settings(config: V2Config) -> StartupCheckResult:
    """确认调用方使用的对象已经通过唯一严格 V2 schema。"""

    if not isinstance(config, V2Config):
        return StartupCheckResult(
            key="training:config",
            status="failed",
            message="training config 不是 V2Config",
        )
    return StartupCheckResult(
        key="training:config",
        status="passed",
        message=f"training config schema {config.schema_version} 已验证",
        details={
            "dataset_root": config.data.dataset_root,
            "split_manifest": config.data.split_manifest,
            "frame_width": config.perception.frame_width,
            "frame_height": config.perception.frame_height,
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "optimization_max_trials": config.optimization.max_trials,
        },
    )


def check_training_runtime(
    config: V2Config,
    *,
    requested_device: RuntimeDevice | None = None,
) -> tuple[StartupCheckResult, ...]:
    """消费 V2 自身环境报告，不复制 CUDA 或坐标门禁语义。"""

    if requested_device is not None and requested_device is not config.runtime.device:
        return (
            StartupCheckResult(
                key="training:runtime",
                status="failed",
                message="requested_device 尚未应用到 V2Config",
                details={
                    "requested_device": requested_device.value,
                    "configured_device": config.runtime.device.value,
                },
            ),
        )
    report = check_v2_environment(config)
    status_map = {
        EnvironmentCheckStatus.PASSED: "passed",
        EnvironmentCheckStatus.WARNING: "warning",
        EnvironmentCheckStatus.FAILED: "failed",
    }
    return tuple(
        StartupCheckResult(
            key=f"training:runtime:{item.name}",
            status=status_map[item.status],
            message=item.message,
            details={
                "configured_device": config.runtime.device.value,
                "cuda_visible": torch.cuda.is_available(),
            },
        )
        for item in report.results
    )


def check_training_data_input(
    quality_report: DataQualityReport | None,
    *,
    split: DataSplit,
    executor_available: bool,
    executor_error: str | None = None,
) -> StartupCheckResult:
    """只按 DataQualityIssue.blocks_training 判断真实数据是否可训练。"""

    if not executor_available or quality_report is None:
        return StartupCheckResult(
            key="training:data_quality",
            status="failed",
            message=executor_error or "production TrainingExecutor 不可用",
            details={"split": split.value, "executor_available": False},
        )
    blocking = tuple(item for item in quality_report.issues if item.blocks_training)
    warnings = tuple(item for item in quality_report.issues if not item.blocks_training)
    return StartupCheckResult(
        key="training:data_quality",
        status="passed" if quality_report.ok else "failed",
        message=(
            "canonical 数据质量门通过"
            if quality_report.ok
            else f"canonical 数据质量门发现 {len(blocking)} 个阻断问题"
        ),
        details={
            "split": split.value,
            "issue_count": len(quality_report.issues),
            "blocking_count": len(blocking),
            "warning_count": len(warnings),
            "blocking_codes": tuple(item.code for item in blocking),
        },
    )


def progressive_startup_checks(
    *,
    require_cuda: bool = False,
) -> tuple[ProgressiveCheck, ...]:
    """返回从模块 spec 到宿主环境的固定渐进检查注册表。"""

    module_checks = tuple(
        ProgressiveCheck(
            key=f"module:{entry.key}",
            level=0,
            description=f"Import {entry.import_name}",
            run=lambda entry=entry: check_source_module_import(entry),
        )
        for entry in source_module_entries(include_start=True)
    )
    return (
        *module_checks,
        ProgressiveCheck(
            key="environment",
            level=1,
            description="Check Python, packages, ffmpeg, torch, and optional CUDA",
            run=lambda: check_environment(require_cuda=require_cuda),
        ),
    )


def run_startup_checks(*, require_cuda: bool = False) -> StartupCheckReport:
    """运行全局轻量启动检查。"""

    return StartupCheckReport(
        scope="src.start",
        results=tuple(
            item.run() for item in progressive_startup_checks(require_cuda=require_cuda)
        ),
    )


def run_training_startup_checks(
    config: V2Config,
    *,
    split: DataSplit,
    requested_device: RuntimeDevice | None = None,
    quality_report: DataQualityReport | None,
    executor_available: bool,
    executor_error: str | None = None,
) -> TrainingStartupCheckReport:
    """组合模块、严格配置、V2 环境和同一份 canonical 数据质量报告。"""

    data_result = check_training_data_input(
        quality_report,
        split=split,
        executor_available=executor_available,
        executor_error=executor_error,
    )
    report = StartupCheckReport(
        scope="traning.production",
        results=(
            *check_src_module_imports(),
            check_training_settings(config),
            *check_training_runtime(config, requested_device=requested_device),
            data_result,
        ),
    )
    return TrainingStartupCheckReport(report=report, data_quality=quality_report)


__all__ = (
    "ProgressiveCheck",
    "check_environment",
    "check_src_module_imports",
    "check_source_module_import",
    "check_training_data_input",
    "check_training_runtime",
    "check_training_settings",
    "progressive_startup_checks",
    "run_startup_checks",
    "run_training_startup_checks",
)
