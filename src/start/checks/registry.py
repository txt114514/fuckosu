from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable

import torch

from environment import EnvironmentReport, collect_environment_report
from start.checks.models import (
    StartupCheckReport,
    StartupCheckResult,
    TrainingStartupCheckReport,
)
from start.modules import SourceModuleEntry, source_module_entries
from traning.conf import DataSplit, Settings
from traning.core.dataset_import import DataInputReport, inspect_data_input


CheckRunner = Callable[[], StartupCheckResult]


@dataclass(frozen=True)
class ProgressiveCheck:
    key: str
    level: int
    description: str
    run: CheckRunner


def check_source_module_import(entry: SourceModuleEntry) -> StartupCheckResult:
    spec = importlib.util.find_spec(entry.import_name)
    if spec is None:
        return StartupCheckResult(
            key=f"module:{entry.key}",
            status="failed",
            message=f"{entry.import_name} is not importable",
            details=entry.as_dict(),
        )
    return StartupCheckResult(
        key=f"module:{entry.key}",
        status="passed",
        message=f"{entry.import_name} import spec resolved",
        details=entry.as_dict(),
    )


def check_src_module_imports() -> tuple[StartupCheckResult, ...]:
    return tuple(
        check_source_module_import(entry)
        for entry in source_module_entries(include_start=True)
    )


def check_environment(
    *,
    report: EnvironmentReport | None = None,
    require_cuda: bool = False,
) -> StartupCheckResult:
    selected = report or collect_environment_report()
    if selected.ready(require_cuda=require_cuda):
        status = "passed"
        message = "required runtime dependencies are available"
    else:
        status = "failed"
        missing = ", ".join(selected.missing_required_packages) or "none"
        message = f"environment is not ready; missing={missing}"
        if require_cuda and not selected.torch.cuda_available:
            message = f"{message}; cuda unavailable"
    return StartupCheckResult(
        key="environment",
        status=status,
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


def check_training_runtime(
    settings: Settings,
    *,
    device: torch.device,
) -> StartupCheckResult:
    if device.type == "cuda" and not torch.cuda.is_available():
        return StartupCheckResult(
            key="training:runtime",
            status="failed",
            message="training requested cuda but torch cannot see CUDA",
            details={
                "requested_device": str(device),
                "configured_device": settings.runtime.device,
            },
        )
    if device.type not in {"cpu", "cuda"}:
        return StartupCheckResult(
            key="training:runtime",
            status="failed",
            message="training device must be cpu or cuda",
            details={"requested_device": str(device)},
        )
    return StartupCheckResult(
        key="training:runtime",
        status="passed",
        message="training runtime device is usable",
        details={
            "requested_device": str(device),
            "configured_device": settings.runtime.device,
            "seed": settings.runtime.seed,
        },
    )


def check_training_settings(settings: Settings) -> StartupCheckResult:
    settings.data_input.validate_tiling()
    settings.tiling.validate_tiling()
    return StartupCheckResult(
        key="training:settings",
        status="passed",
        message="training settings loaded and tiling constraints passed",
        details={
            "dataset_root": settings.data_input.dataset_root,
            "input": {
                "width": settings.input.width,
                "height": settings.input.height,
                "color_cues": settings.input.color_cues,
            },
            "tiling": {
                "patch_width": settings.tiling.patch_width,
                "patch_height": settings.tiling.patch_height,
                "overlap_x": settings.tiling.overlap_x,
                "overlap_y": settings.tiling.overlap_y,
            },
        },
    )


def check_training_data_input(
    settings: Settings,
    *,
    split: DataSplit,
) -> tuple[StartupCheckResult, DataInputReport]:
    report = inspect_data_input(settings, split=split)
    if report.ok:
        status = "passed"
        message = "training data input is available"
    elif settings.data_input.strict:
        status = "failed"
        message = "training data input failed strict validation"
    else:
        status = "warning"
        message = "training data input has issues but strict mode is disabled"
    return (
        StartupCheckResult(
            key="training:data_input",
            status=status,
            message=message,
            details={
                "split": report.split,
                "segments": report.segment_count,
                "estimated_frames": report.frame_count_estimate,
                "issue_count": report.issue_count,
                "issues": report.issues[:10],
            },
        ),
        report,
    )


def progressive_startup_checks(
    *,
    require_cuda: bool = False,
) -> tuple[ProgressiveCheck, ...]:
    return (
        *(
            ProgressiveCheck(
                key=f"module:{entry.key}",
                level=0,
                description=f"Import {entry.import_name}",
                run=lambda entry=entry: check_source_module_import(entry),
            )
            for entry in source_module_entries(include_start=True)
        ),
        ProgressiveCheck(
            key="environment",
            level=1,
            description="Check Python, packages, ffmpeg, torch, and optional CUDA",
            run=lambda: check_environment(require_cuda=require_cuda),
        ),
    )


def run_startup_checks(
    *,
    require_cuda: bool = False,
) -> StartupCheckReport:
    checks = progressive_startup_checks(require_cuda=require_cuda)
    return StartupCheckReport(
        scope="src.start",
        results=tuple(check.run() for check in checks),
    )


def run_training_startup_checks(
    settings: Settings,
    *,
    split: DataSplit,
    device: torch.device,
    require_cuda: bool | None = None,
) -> TrainingStartupCheckReport:
    cuda_required = device.type == "cuda" if require_cuda is None else require_cuda
    data_result, data_report = check_training_data_input(settings, split=split)
    report = StartupCheckReport(
        scope="traning.full_training",
        results=(
            *check_src_module_imports(),
            check_environment(require_cuda=cuda_required),
            check_training_settings(settings),
            check_training_runtime(settings, device=device),
            data_result,
        ),
    )
    return TrainingStartupCheckReport(
        report=report,
        data_input=data_report,
    )


__all__ = [
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
]
