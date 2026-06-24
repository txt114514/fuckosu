from __future__ import annotations

from pathlib import Path

from before_traning.conf import Settings
from before_traning.tests.startup_checks.items import (
    check_pipeline_tasks,
    check_raw_training_inputs,
    check_segment_planner_contract,
    check_settings_load,
)
from before_traning.tests.startup_checks.samples import DEFAULT_MATCHED_MANIFEST
from package.checks import StartupCheckReport, StartupCheckResult


def run_startup_checks(
    config_path: Path | None = None,
    *,
    matched_manifest_path: Path = DEFAULT_MATCHED_MANIFEST,
    run_match_probe: bool = True,
    min_match_score: float = 0.1,
) -> StartupCheckReport:
    settings: Settings | None = None
    results: list[StartupCheckResult] = []
    try:
        result, settings = check_settings_load(config_path)
    except Exception as error:
        result = StartupCheckResult(
            key="before_traning:settings",
            status="failed",
            message=f"{type(error).__name__}: {error}",
        )
    results.append(result)

    for check in (check_pipeline_tasks, check_segment_planner_contract):
        try:
            result, _ = check(settings)
        except Exception as error:
            result = StartupCheckResult(
                key=f"before_traning:{check.__name__}",
                status="failed",
                message=f"{type(error).__name__}: {error}",
            )
        results.append(result)

    if settings is not None:
        try:
            result, _ = check_raw_training_inputs(
                settings,
                matched_manifest_path=matched_manifest_path,
                run_match_probe=run_match_probe,
                min_match_score=min_match_score,
            )
        except Exception as error:
            result = StartupCheckResult(
                key="before_traning:raw_data",
                status="failed",
                message=f"{type(error).__name__}: {error}",
            )
    else:
        result = StartupCheckResult(
            key="before_traning:raw_data",
            status="skipped",
            message="settings did not load",
        )
    results.append(result)

    return StartupCheckReport(
        scope="before_traning.startup_checks",
        results=tuple(results),
    )


__all__ = ["run_startup_checks"]
