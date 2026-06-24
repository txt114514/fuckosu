from __future__ import annotations

from pathlib import Path

import torch

from package.checks import StartupCheckResult
from traning.conf import DataSplit, Settings, load_settings
from traning.core.dataset_import import inspect_data_input
from traning.core.decision import TRAINING_STAGES, run_full_training_pipeline


def check_settings_load(config_path: Path | None = None) -> tuple[StartupCheckResult, Settings]:
    settings = load_settings(config_path)
    settings.data_input.validate_tiling()
    settings.tiling.validate_tiling()
    return (
        StartupCheckResult(
            key="traning:settings",
            status="passed",
            message="traning settings loaded and tiling constraints passed",
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
        ),
        settings,
    )


def check_runtime_device(
    settings: Settings,
    *,
    device: torch.device,
    require_cuda: bool | None = None,
) -> tuple[StartupCheckResult, None]:
    cuda_required = device.type == "cuda" if require_cuda is None else require_cuda
    if cuda_required and not torch.cuda.is_available():
        return (
            StartupCheckResult(
                key="traning:runtime",
                status="failed",
                message="CUDA is required but torch cannot see CUDA",
                details={
                    "requested_device": str(device),
                    "configured_device": settings.runtime.device,
                },
            ),
            None,
        )
    if device.type not in {"cpu", "cuda"}:
        return (
            StartupCheckResult(
                key="traning:runtime",
                status="failed",
                message="training device must be cpu or cuda",
                details={"requested_device": str(device)},
            ),
            None,
        )
    return (
        StartupCheckResult(
            key="traning:runtime",
            status="passed",
            message="training runtime device is usable",
            details={
                "requested_device": str(device),
                "configured_device": settings.runtime.device,
                "seed": settings.runtime.seed,
                "cuda_available": torch.cuda.is_available(),
            },
        ),
        None,
    )


def check_data_input(
    settings: Settings,
    *,
    split: DataSplit,
) -> tuple[StartupCheckResult, None]:
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
            key="traning:data_input",
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
        None,
    )


def check_core_entrypoints(_settings: Settings | None = None) -> tuple[StartupCheckResult, None]:
    stages = tuple(stage.key for stage in TRAINING_STAGES)
    expected = (
        "data_input",
        "spatial",
        "candidate_cache",
        "temporal",
        "decision",
    )
    missing = tuple(stage for stage in expected if stage not in stages)
    if missing or not callable(run_full_training_pipeline):
        return (
            StartupCheckResult(
                key="traning:core_entrypoints",
                status="failed",
                message="full training core entrypoints are incomplete",
                details={"missing": missing, "stages": stages},
            ),
            None,
        )
    return (
        StartupCheckResult(
            key="traning:core_entrypoints",
            status="passed",
            message="full training core entrypoints are available",
            details={"stages": stages},
        ),
        None,
    )


__all__ = [
    "check_core_entrypoints",
    "check_data_input",
    "check_runtime_device",
    "check_settings_load",
]
