from __future__ import annotations

from pathlib import Path

import torch

from package.checks import StartupCheckReport, StartupCheckResult
from traning.conf import DataSplit, Settings
from traning.tests.startup_checks.items import (
    check_core_entrypoints,
    check_data_input,
    check_runtime_device,
    check_settings_load,
)


def run_startup_checks(
    config_path: Path | None = None,
    *,
    split: DataSplit = "train",
    device: torch.device | None = None,
    require_cuda: bool | None = None,
) -> StartupCheckReport:
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings: Settings | None = None
    results: list[StartupCheckResult] = []
    try:
        result, settings = check_settings_load(config_path)
    except Exception as error:
        result = StartupCheckResult(
            key="traning:settings",
            status="failed",
            message=f"{type(error).__name__}: {error}",
        )
    results.append(result)

    if settings is not None:
        for call in (
            lambda: check_runtime_device(
                settings,
                device=selected_device,
                require_cuda=require_cuda,
            ),
            lambda: check_data_input(settings, split=split),
            lambda: check_core_entrypoints(settings),
        ):
            try:
                result, _ = call()
            except Exception as error:
                result = StartupCheckResult(
                    key="traning:startup_check",
                    status="failed",
                    message=f"{type(error).__name__}: {error}",
                )
            results.append(result)
    else:
        results.extend(
            (
                StartupCheckResult(
                    key="traning:runtime",
                    status="skipped",
                    message="settings did not load",
                ),
                StartupCheckResult(
                    key="traning:data_input",
                    status="skipped",
                    message="settings did not load",
                ),
                StartupCheckResult(
                    key="traning:core_entrypoints",
                    status="skipped",
                    message="settings did not load",
                ),
            )
        )

    return StartupCheckReport(
        scope="traning.startup_checks",
        results=tuple(results),
    )


__all__ = ["run_startup_checks"]
