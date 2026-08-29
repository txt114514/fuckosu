"""验证统一启动流程的固定阶段、dry-run 和真实训练结果传播。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from package import DataSplit
from package.checks import StartupCheckReport, StartupCheckResult
from package.dataset_split import (
    DatasetSplitManifest,
    DatasetSplitSyncResult,
    SplitRatios,
)
from start.checks import TrainingStartupCheckReport
from start.flow import (
    StartFlowConfig,
    TrainingExecutionResult,
    run_start_flow,
)
from traning.config import RuntimeConfig, RuntimeDevice, V2Config, load_v2_config
from traning.contracts import DataQualityReport


def _config() -> V2Config:
    config = load_v2_config(Path("configs/traning.yaml"))
    return replace(
        config,
        runtime=RuntimeConfig(
            device=RuntimeDevice.CPU,
            require_cuda=False,
            amp=False,
        ),
    )


def _before_report() -> StartupCheckReport:
    return StartupCheckReport(
        scope="before",
        results=(
            StartupCheckResult(
                key="before_traning:raw_data",
                status="passed",
                message="no new data",
                details={"should_run_before_traning": False, "reason": "no new data"},
            ),
        ),
    )


def _split_sync(path: Path) -> DatasetSplitSyncResult:
    return DatasetSplitSyncResult(
        manifest_path=path,
        created=False,
        changed=False,
        dry_run=True,
        new_items=(),
        manifest=DatasetSplitManifest(
            seed=2026,
            ratios=SplitRatios(),
            items={},
        ),
    )


def test_dry_run_executes_checks_but_never_calls_training(tmp_path: Path) -> None:
    """dry-run 必须保留完整前置证据，同时不触发昂贵训练。"""

    quality = DataQualityReport(issues=())
    executor = Mock()
    executor.inspect.return_value = quality
    checks = TrainingStartupCheckReport(
        report=StartupCheckReport(
            scope="training",
            results=(StartupCheckResult("training:data", "passed", "ok"),),
        ),
        data_quality=quality,
    )
    with (
        patch("start.flow.load_v2_config", return_value=_config()),
        patch("start.flow.run_before_startup_checks", return_value=_before_report()),
        patch("start.flow.load_before_settings", return_value=object()),
        patch("start.flow._sync_dataset_splits", return_value=_split_sync(tmp_path / "split.json")),
        patch("start.flow.run_training_startup_checks", return_value=checks),
    ):
        result = run_start_flow(
            StartFlowConfig(
                dry_run=True,
                run_before_traning=False,
                output_root=tmp_path,
                run_id="dry-run",
            ),
            executor=executor,
        )

    assert result.status == "dry-run-passed"
    assert tuple(item.stage_id for item in result.stages) == (
        "raw_scan",
        "before_conversion",
        "split_sync",
        "checks",
        "training",
        "report",
    )
    executor.run.assert_not_called()
    assert result.report_path.is_file()


def test_passed_requires_explicit_training_result(tmp_path: Path) -> None:
    """非 dry-run 只有 executor 返回 passed 才能形成完整 passed 终态。"""

    quality = DataQualityReport(issues=())
    executor = Mock()
    executor.inspect.return_value = quality
    executor.run.return_value = TrainingExecutionResult(
        status="passed",
        message="all gates passed",
        trial_index=3,
        objective=0.9,
        checkpoint_path=tmp_path / "checkpoint",
    )
    checks = TrainingStartupCheckReport(
        StartupCheckReport(
            "training",
            (StartupCheckResult("training:data", "passed", "ok"),),
        ),
        quality,
    )
    with (
        patch("start.flow.load_v2_config", return_value=_config()),
        patch("start.flow.run_before_startup_checks", return_value=_before_report()),
        patch("start.flow.load_before_settings", return_value=object()),
        patch("start.flow._sync_dataset_splits", return_value=_split_sync(tmp_path / "split.json")),
        patch("start.flow.run_training_startup_checks", return_value=checks),
    ):
        result = run_start_flow(
            StartFlowConfig(
                output_root=tmp_path,
                run_id="execute",
                run_before_traning=False,
            ),
            executor=executor,
        )

    assert result.status == "passed"
    assert result.training is executor.run.return_value
    executor.run.assert_called_once()
    assert executor.run.call_args.kwargs["quality_report"] is quality
    assert executor.run.call_args.kwargs["split"] is DataSplit.TRAIN
