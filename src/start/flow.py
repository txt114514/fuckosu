from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

from before_traning.conf import load_settings as load_before_settings
from before_traning.core.beatmap.pipeline import TRAINING_PIPELINE
from before_traning.tests.full_checks.runner import run_full_checks as run_before_full_checks
from before_traning.tests.startup_checks.runner import (
    run_startup_checks as run_before_startup_checks,
)
from before_traning.tests.startup_checks.samples import (
    DEFAULT_MATCHED_MANIFEST,
    recover_matched_sample_manifest,
)
from package.checks import StartupCheckReport
from package.dataset_split import (
    DatasetSplitSyncResult,
    SplitRatios,
    sync_dataset_split_manifest,
)
from traning.tests.full_checks.runner import run_full_checks as run_traning_full_checks
from traning.tests.startup_checks.runner import (
    run_startup_checks as run_traning_startup_checks,
)
from traning.conf import DataSplit, load_settings as load_training_settings
from traning.core.decision import (
    FullTrainingRunConfig,
    FullTrainingRunResult,
    run_full_training_pipeline,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TestLevel = Literal["none", "quick", "full"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ProgressiveTestReport:
    level: TestLevel
    command: tuple[str, ...] = ()
    returncode: int = 0
    stdout_tail: str = ""
    stderr_tail: str = ""
    status: str = "skipped"
    reports: tuple[StartupCheckReport, ...] = ()

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and all(report.ok for report in self.reports)

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "command": self.command,
            "returncode": self.returncode,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "status": self.status,
            "reports": tuple(report.as_dict() for report in self.reports),
            "ok": self.ok,
        }


@dataclass(frozen=True)
class BeforeTrainingRunReport:
    status: str
    stage_results: Mapping[str, bool] = field(default_factory=dict)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"passed", "skipped"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stage_results": dict(self.stage_results),
            "message": self.message,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class StartupFlowConfig:
    training_config: Path
    before_config: Path | None = None
    split: DataSplit = "train"
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    require_cuda: bool | None = None
    matched_manifest_path: Path = DEFAULT_MATCHED_MANIFEST
    run_before_traning: bool = True
    before_match_probe: bool = True
    before_min_match_score: float = 0.1
    split_manifest_path: Path | None = None
    split_seed: int = 2026
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    allow_test_growth: bool = False
    test_level: TestLevel = "quick"
    dry_run: bool = False
    run_full_training: bool = True
    spatial_max_steps: int = 1
    temporal_max_steps: int = 1
    spatial_learning_rate: float = 1e-4
    temporal_learning_rate: float = 1e-4
    patch_limit: int | None = 1
    cache_max_frames: int | None = 1
    sequence_length: int | None = None
    candidate_slots: int | None = None
    run_dir: Path | None = None


@dataclass(frozen=True)
class StartupFlowResult:
    before_startup: StartupCheckReport
    before_run: BeforeTrainingRunReport
    split_sync: DatasetSplitSyncResult
    training_startup: StartupCheckReport
    tests: ProgressiveTestReport
    full_training: FullTrainingRunResult | None = None
    dry_run: bool = False
    generated_at_utc: str = field(default_factory=_utc_now)

    @property
    def ok(self) -> bool:
        return (
            self.before_run.ok
            and self.before_startup.ok
            and self.training_startup.ok
            and self.tests.ok
            and (self.full_training is not None or self.dry_run)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "dry_run": self.dry_run,
            "generated_at_utc": self.generated_at_utc,
            "before_startup": self.before_startup.as_dict(),
            "before_run": self.before_run.as_dict(),
            "split_sync": self.split_sync.as_dict(),
            "training_startup": self.training_startup.as_dict(),
            "tests": self.tests.as_dict(),
            "full_training": (
                self.full_training.as_summary()
                if self.full_training is not None
                else None
            ),
        }


def run_startup_flow(config: StartupFlowConfig) -> StartupFlowResult:
    before_startup = run_before_startup_checks(
        config.before_config,
        matched_manifest_path=config.matched_manifest_path,
        run_match_probe=config.before_match_probe,
        min_match_score=config.before_min_match_score,
    )
    before_startup.raise_for_errors()
    before_settings = load_before_settings(config.before_config)
    before_run = _maybe_run_before_traning(
        before_settings,
        before_startup=before_startup,
        config=config,
    )
    if not before_run.ok:
        raise RuntimeError(f"before_traning startup step failed: {before_run.message}")

    training_settings = load_training_settings(config.training_config)
    split_sync = _sync_dataset_splits(training_settings, config=config)
    training_startup = run_traning_startup_checks(
        config.training_config,
        split=config.split,
        device=config.device,
        require_cuda=config.require_cuda,
    )
    training_startup.raise_for_errors()

    tests = run_progressive_tests(
        config.test_level,
        startup_reports=(before_startup, training_startup),
    )
    if not tests.ok:
        raise RuntimeError(
            "progressive startup tests failed: "
            f"{' '.join(tests.command)} returned {tests.returncode}"
        )

    full_training = None
    if not config.dry_run and config.run_full_training:
        full_training = run_full_training_pipeline(
            training_settings,
            config=_full_training_config(config),
        )

    return StartupFlowResult(
        before_startup=before_startup,
        before_run=before_run,
        split_sync=split_sync,
        training_startup=training_startup,
        tests=tests,
        full_training=full_training,
        dry_run=config.dry_run,
    )


def run_progressive_tests(
    level: TestLevel,
    *,
    startup_reports: tuple[StartupCheckReport, ...] = (),
) -> ProgressiveTestReport:
    if level == "none":
        return ProgressiveTestReport(level=level, status="skipped")
    if level == "quick":
        reports = startup_reports
        command = tuple(report.scope for report in reports)
    elif level == "full":
        reports = (
            run_before_full_checks(),
            run_traning_full_checks(),
        )
        command = tuple(report.scope for report in reports)
    else:
        raise ValueError(f"unknown test level: {level}")
    returncode = 0 if all(report.ok for report in reports) else 1
    return ProgressiveTestReport(
        level=level,
        command=command,
        returncode=returncode,
        status="passed" if returncode == 0 else "failed",
        reports=reports,
    )


def _maybe_run_before_traning(
    before_settings,
    *,
    before_startup: StartupCheckReport,
    config: StartupFlowConfig,
) -> BeforeTrainingRunReport:
    should_run = _before_should_run(before_startup)
    reason = _before_reason(before_startup)
    if not should_run:
        return BeforeTrainingRunReport(
            status="skipped",
            message=reason,
        )
    if not config.run_before_traning:
        return BeforeTrainingRunReport(
            status="skipped",
            message="before_traning execution disabled by startup config",
        )
    if config.dry_run:
        return BeforeTrainingRunReport(
            status="skipped",
            message="dry-run: before_traning would run",
        )

    results = TRAINING_PIPELINE.run_direct(before_settings)
    if any(not success for success in results.values()):
        failed = ", ".join(stage for stage, success in results.items() if not success)
        return BeforeTrainingRunReport(
            status="failed",
            stage_results=results,
            message=f"failed stages: {failed}",
        )

    manifest = recover_matched_sample_manifest(
        before_settings,
        matched_manifest_path=config.matched_manifest_path,
    )
    manifest.save()
    return BeforeTrainingRunReport(
        status="passed",
        stage_results=results,
        message=f"before_traning completed; matched manifest saved to {manifest.path}",
    )


def _full_training_config(config: StartupFlowConfig) -> FullTrainingRunConfig:
    return FullTrainingRunConfig(
        run_dir=config.run_dir or _run_dir("full_training"),
        split=config.split,
        device=config.device,
        spatial_max_steps=config.spatial_max_steps,
        temporal_max_steps=config.temporal_max_steps,
        spatial_learning_rate=config.spatial_learning_rate,
        temporal_learning_rate=config.temporal_learning_rate,
        patch_limit=config.patch_limit,
        cache_max_frames=config.cache_max_frames,
        sequence_length=config.sequence_length,
        candidate_slots=config.candidate_slots,
    )


def _sync_dataset_splits(
    training_settings,
    *,
    config: StartupFlowConfig,
) -> DatasetSplitSyncResult:
    data_input = training_settings.data_input
    bootstrap = {
        **{item: "train" for item in data_input.train_items},
        **{item: "validation" for item in data_input.validation_items},
        **{item: "test" for item in data_input.test_items},
    }
    return sync_dataset_split_manifest(
        data_input.dataset_root,
        manifest_path=config.split_manifest_path or data_input.split_manifest_path,
        seed=config.split_seed,
        ratios=SplitRatios(
            train=config.train_ratio,
            validation=config.validation_ratio,
            test=config.test_ratio,
        ),
        bootstrap_splits=bootstrap,
        allow_test_growth=config.allow_test_growth,
        dry_run=config.dry_run,
    )


def _before_raw_data_result(report: StartupCheckReport):
    for result in report.results:
        if result.key == "before_traning:raw_data":
            return result
    return None


def _before_should_run(report: StartupCheckReport) -> bool:
    result = _before_raw_data_result(report)
    if result is None:
        return False
    return bool(result.details.get("should_run_before_traning"))


def _before_reason(report: StartupCheckReport) -> str:
    result = _before_raw_data_result(report)
    if result is None:
        return "before_traning raw data check did not run"
    reason = result.details.get("reason")
    return str(reason) if reason else result.message


def _run_dir(kind: str) -> Path:
    timestamp = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%S_%fZ")
    )
    return REPO_ROOT / "runs" / f"{timestamp}__startup_{kind}"


def write_startup_flow_report(
    result: StartupFlowResult,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(result.as_dict()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_ready(item) for item in value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "BeforeTrainingRunReport",
    "ProgressiveTestReport",
    "StartupFlowConfig",
    "StartupFlowResult",
    "run_progressive_tests",
    "run_startup_flow",
    "write_startup_flow_report",
]
