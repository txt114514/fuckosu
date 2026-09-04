"""编排原始数据准备、稳定划分、V2 检查、训练与报告的唯一启动流程。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

from before_traning.conf import load_settings as load_before_settings
from before_traning.core.beatmap.pipeline import TRAINING_PIPELINE
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
from traning.conf import RuntimeDevice, V2Config, load_v2_config
from traning.state import DataQualityReport, DataSplit

from start.checks import TrainingStartupCheckReport, run_training_startup_checks


REPO_ROOT = Path(__file__).resolve().parents[2]
START_FLOW_REPORT_FILENAME = "start_flow_report.json"


def _utc_now() -> str:
    """返回便于排序且显式带 UTC 时区的时间戳。"""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class BeforeTrainingRunReport:
    """记录 before_traning 是否执行及各转换阶段的真实结果。"""

    status: str
    stage_results: tuple[tuple[str, bool], ...] = ()
    message: str = ""

    @property
    def ok(self) -> bool:
        """只有成功或确实无需转换时才允许继续。"""

        return self.status in {"passed", "skipped"}

    def as_dict(self) -> dict[str, Any]:
        """返回 JSON 安全的审计表示。"""

        return {
            "status": self.status,
            "stage_results": dict(self.stage_results),
            "message": self.message,
            "ok": self.ok,
        }


@dataclass(frozen=True, slots=True)
class TrainingExecutionResult:
    """start 与正式 V2 训练服务之间的最小、不可变终态。"""

    status: str
    message: str
    trial_index: int | None = None
    objective: float | None = None
    checkpoint_path: Path | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed"}:
            raise ValueError("training result status 必须是 passed 或 failed")
        if not isinstance(self.message, str) or self.message != self.message.strip():
            raise ValueError("training result message 必须是无首尾空格的字符串")
        if self.trial_index is not None and (
            isinstance(self.trial_index, bool)
            or not isinstance(self.trial_index, int)
            or self.trial_index < 0
        ):
            raise ValueError("trial_index 必须是非负整数或 None")

    @property
    def ok(self) -> bool:
        """仅显式 passed 结果可推进完整流程。"""

        return self.status == "passed"

    def as_dict(self) -> dict[str, Any]:
        """返回不隐去失败原因的 JSON 表示。"""

        return {
            "status": self.status,
            "message": self.message,
            "trial_index": self.trial_index,
            "objective": self.objective,
            "checkpoint_path": self.checkpoint_path,
            "details": dict(self.details),
            "ok": self.ok,
        }


class TrainingExecutor(Protocol):
    """V2 production service 面向启动层的唯一注入协议。

    实现负责从 canonical 数据源构建 backend。start 只先读取质量门，再把同一
    ``DataQualityReport`` 交回训练调用，避免启动检查与正式训练各算一套结论。
    """

    def inspect(
        self,
        config: V2Config,
        *,
        split: DataSplit,
    ) -> DataQualityReport:
        """只读检查训练数据并返回 canonical 质量报告。"""

    def run(
        self,
        config: V2Config,
        *,
        split: DataSplit,
        quality_report: DataQualityReport,
        run_dir: Path,
        run_id: str,
        resume: bool,
    ) -> TrainingExecutionResult:
        """执行完整训练搜索并返回显式终态。"""


@dataclass(frozen=True, slots=True)
class StartStageResult:
    """完整启动生命周期中的一个稳定阶段结果。"""

    stage_id: str
    status: str
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed", "skipped"}:
            raise ValueError("start stage status 必须是 passed、failed 或 skipped")

    @property
    def ok(self) -> bool:
        """跳过只表示该阶段无需执行，不等同于失败。"""

        return self.status != "failed"

    def as_dict(self) -> dict[str, Any]:
        """返回 JSON 安全的阶段快照。"""

        return {
            "stage_id": self.stage_id,
            "status": self.status,
            "message": self.message,
            "details": dict(self.details),
            "ok": self.ok,
        }


@dataclass(frozen=True, slots=True)
class StartFlowConfig:
    """唯一启动流程配置；模型及训练参数全部来自严格 ``V2Config``。"""

    training_config: Path = Path("configs/traning.yaml")
    before_config: Path | None = None
    split: DataSplit = DataSplit.TRAIN
    requested_device: RuntimeDevice | None = None
    matched_manifest_path: Path = DEFAULT_MATCHED_MANIFEST
    run_before_traning: bool = True
    before_match_probe: bool = True
    before_min_match_score: float = 0.1
    split_manifest_path: Path | None = None
    split_seed: int | None = None
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    allow_test_growth: bool = False
    dry_run: bool = False
    output_root: Path = Path("artifacts/training_runs")
    run_id: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    resume: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.training_config, Path):
            raise TypeError("training_config 必须是 pathlib.Path")
        if not isinstance(self.split, DataSplit) or self.split is DataSplit.ALL:
            raise ValueError("start split 必须是 train、validation 或 test")
        if self.requested_device is not None and not isinstance(
            self.requested_device, RuntimeDevice
        ):
            raise TypeError("requested_device 必须是 RuntimeDevice 或 None")
        if not self.run_id or self.run_id != self.run_id.strip():
            raise ValueError("run_id 必须非空且无首尾空格")
        if self.before_min_match_score < 0.0:
            raise ValueError("before_min_match_score 不得为负数")
        if self.split_seed is not None and self.split_seed < 0:
            raise ValueError("split_seed 不得为负数")
        # 由共享 SplitRatios 继续验证有限性、总和及非负约束。
        SplitRatios(
            train=self.train_ratio,
            validation=self.validation_ratio,
            test=self.test_ratio,
        )

    @property
    def run_dir(self) -> Path:
        """返回本次流程唯一的产物目录。"""

        return self.output_root / self.run_id


@dataclass(frozen=True, slots=True)
class StartFlowResult:
    """完整启动证据；passed 不允许由缺失训练结果推断。"""

    status: str
    config_path: Path
    run_id: str
    run_dir: Path
    split: DataSplit
    dry_run: bool
    stages: tuple[StartStageResult, ...]
    report_path: Path
    before_startup: StartupCheckReport | None = None
    before_run: BeforeTrainingRunReport | None = None
    split_sync: DatasetSplitSyncResult | None = None
    training_startup: TrainingStartupCheckReport | None = None
    quality_report: DataQualityReport | None = None
    training: TrainingExecutionResult | None = None
    error: str | None = None
    generated_at_utc: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.status not in {"passed", "dry-run-passed", "failed"}:
            raise ValueError("start flow status 非法")
        if self.status == "passed" and (
            self.dry_run or self.training is None or not self.training.ok
        ):
            raise ValueError("passed 必须包含真实通过的训练结果")
        if self.status == "dry-run-passed" and not self.dry_run:
            raise ValueError("dry-run-passed 仅允许用于 dry_run")
        if self.status != "failed" and any(not stage.ok for stage in self.stages):
            raise ValueError("成功流程不得包含失败阶段")

    @property
    def ok(self) -> bool:
        """返回完整流程是否真实成功或成功完成 dry-run。"""

        return self.status in {"passed", "dry-run-passed"}

    def as_dict(self) -> dict[str, Any]:
        """返回完整且 JSON 安全的启动报告。"""

        return _json_ready(
            {
                "status": self.status,
                "ok": self.ok,
                "config_path": self.config_path,
                "run_id": self.run_id,
                "run_dir": self.run_dir,
                "split": self.split.value,
                "dry_run": self.dry_run,
                "generated_at_utc": self.generated_at_utc,
                "report_path": self.report_path,
                "error": self.error,
                "stages": tuple(stage.as_dict() for stage in self.stages),
                "before_startup": (
                    self.before_startup.as_dict()
                    if self.before_startup is not None
                    else None
                ),
                "before_run": (
                    self.before_run.as_dict() if self.before_run is not None else None
                ),
                "split_sync": (
                    self.split_sync.as_dict() if self.split_sync is not None else None
                ),
                "training_startup": (
                    self.training_startup.as_dict()
                    if self.training_startup is not None
                    else None
                ),
                "quality_report": _quality_report_dict(self.quality_report),
                "training": (
                    self.training.as_dict() if self.training is not None else None
                ),
            }
        )


def run_start_flow(
    flow_config: StartFlowConfig,
    *,
    executor: TrainingExecutor | None,
) -> StartFlowResult:
    """严格按 raw scan → conversion → split → checks → train → report 执行。"""

    stages: list[StartStageResult] = []
    before_startup: StartupCheckReport | None = None
    before_run: BeforeTrainingRunReport | None = None
    split_sync: DatasetSplitSyncResult | None = None
    training_startup: TrainingStartupCheckReport | None = None
    quality_report: DataQualityReport | None = None
    training_result: TrainingExecutionResult | None = None
    report_path = flow_config.run_dir / START_FLOW_REPORT_FILENAME

    try:
        config = load_v2_config(flow_config.training_config)
        config = _effective_training_config(config, flow_config)
    except (OSError, TypeError, ValueError) as exc:
        stages.append(StartStageResult("raw_scan", "failed", f"V2 配置失败：{exc}"))
        return _finish_flow(
            flow_config,
            stages=stages,
            report_path=report_path,
            error=str(exc),
        )

    try:
        # 原始输入检查必须先于任何转换或 split 写入。
        before_startup = run_before_startup_checks(
            flow_config.before_config,
            matched_manifest_path=flow_config.matched_manifest_path,
            run_match_probe=flow_config.before_match_probe,
            min_match_score=flow_config.before_min_match_score,
        )
        if not before_startup.ok:
            before_startup.raise_for_errors()
        stages.append(
            StartStageResult(
                "raw_scan",
                "passed",
                _before_reason(before_startup),
                {"should_convert": _before_should_run(before_startup)},
            )
        )
    except Exception as exc:
        stages.append(StartStageResult("raw_scan", "failed", str(exc)))
        return _finish_flow(
            flow_config,
            stages=stages,
            report_path=report_path,
            before_startup=before_startup,
            error=str(exc),
        )

    try:
        before_settings = load_before_settings(flow_config.before_config)
        before_run = _maybe_run_before_traning(
            before_settings,
            before_startup=before_startup,
            config=flow_config,
        )
        stages.append(
            StartStageResult(
                "before_conversion",
                before_run.status,
                before_run.message,
                {"stage_results": dict(before_run.stage_results)},
            )
        )
        if not before_run.ok:
            raise RuntimeError(before_run.message)
    except Exception as exc:
        if not stages or stages[-1].stage_id != "before_conversion":
            stages.append(StartStageResult("before_conversion", "failed", str(exc)))
        return _finish_flow(
            flow_config,
            stages=stages,
            report_path=report_path,
            before_startup=before_startup,
            before_run=before_run,
            error=str(exc),
        )

    try:
        split_sync = _sync_dataset_splits(config, flow_config=flow_config)
        stages.append(
            StartStageResult(
                "split_sync",
                "passed",
                "数据划分已校验" if not split_sync.changed else "数据划分已同步",
                {
                    "manifest_path": split_sync.manifest_path,
                    "changed": split_sync.changed,
                    "new_items": len(split_sync.new_items),
                },
            )
        )
    except Exception as exc:
        stages.append(StartStageResult("split_sync", "failed", str(exc)))
        return _finish_flow(
            flow_config,
            stages=stages,
            report_path=report_path,
            before_startup=before_startup,
            before_run=before_run,
            error=str(exc),
        )

    inspect_error: str | None = None
    if executor is None:
        inspect_error = "未注入 V2 TrainingExecutor，无法检查数据质量或启动训练"
    else:
        try:
            quality_report = executor.inspect(config, split=flow_config.split)
            if not isinstance(quality_report, DataQualityReport):
                raise TypeError("TrainingExecutor.inspect 必须返回 DataQualityReport")
        except Exception as exc:
            inspect_error = str(exc)

    try:
        training_startup = run_training_startup_checks(
            config,
            split=flow_config.split,
            requested_device=flow_config.requested_device,
            quality_report=quality_report,
            executor_available=executor is not None and inspect_error is None,
            executor_error=inspect_error,
        )
        check_status = "passed" if training_startup.ok else "failed"
        stages.append(
            StartStageResult(
                "checks",
                check_status,
                "V2 启动检查通过" if training_startup.ok else "V2 启动检查失败",
                {"failure_count": len(training_startup.report.failures)},
            )
        )
        if not training_startup.ok:
            training_startup.raise_for_errors()
    except Exception as exc:
        if not stages or stages[-1].stage_id != "checks":
            stages.append(StartStageResult("checks", "failed", str(exc)))
        return _finish_flow(
            flow_config,
            stages=stages,
            report_path=report_path,
            before_startup=before_startup,
            before_run=before_run,
            split_sync=split_sync,
            training_startup=training_startup,
            quality_report=quality_report,
            error=str(exc),
        )

    if flow_config.dry_run:
        stages.append(
            StartStageResult(
                "training",
                "skipped",
                "dry-run 已完成全部只读检查，未调用训练执行器",
            )
        )
    else:
        try:
            if (
                executor is None or quality_report is None
            ):  # pragma: no cover - checks 已阻断
                raise RuntimeError("V2 TrainingExecutor 或 DataQualityReport 缺失")
            training_result = executor.run(
                config,
                split=flow_config.split,
                quality_report=quality_report,
                run_dir=flow_config.run_dir,
                run_id=flow_config.run_id,
                resume=flow_config.resume,
            )
            if not isinstance(training_result, TrainingExecutionResult):
                raise TypeError("TrainingExecutor.run 必须返回 TrainingExecutionResult")
            stages.append(
                StartStageResult(
                    "training",
                    training_result.status,
                    training_result.message,
                    {
                        "trial_index": training_result.trial_index,
                        "objective": training_result.objective,
                        "checkpoint_path": training_result.checkpoint_path,
                    },
                )
            )
            if not training_result.ok:
                raise RuntimeError(training_result.message or "V2 训练未通过")
        except Exception as exc:
            if not stages or stages[-1].stage_id != "training":
                stages.append(StartStageResult("training", "failed", str(exc)))
            return _finish_flow(
                flow_config,
                stages=stages,
                report_path=report_path,
                before_startup=before_startup,
                before_run=before_run,
                split_sync=split_sync,
                training_startup=training_startup,
                quality_report=quality_report,
                training=training_result,
                error=str(exc),
            )

    return _finish_flow(
        flow_config,
        stages=stages,
        report_path=report_path,
        before_startup=before_startup,
        before_run=before_run,
        split_sync=split_sync,
        training_startup=training_startup,
        quality_report=quality_report,
        training=training_result,
    )


def _finish_flow(
    flow_config: StartFlowConfig,
    *,
    stages: list[StartStageResult],
    report_path: Path,
    before_startup: StartupCheckReport | None = None,
    before_run: BeforeTrainingRunReport | None = None,
    split_sync: DatasetSplitSyncResult | None = None,
    training_startup: TrainingStartupCheckReport | None = None,
    quality_report: DataQualityReport | None = None,
    training: TrainingExecutionResult | None = None,
    error: str | None = None,
) -> StartFlowResult:
    """统一计算终态并尽力写报告；报告失败也必须改变最终状态。"""

    failed = error is not None or any(not stage.ok for stage in stages)
    if failed:
        status = "failed"
    elif flow_config.dry_run:
        status = "dry-run-passed"
    elif training is not None and training.ok:
        status = "passed"
    else:
        status = "failed"
        error = error or "训练执行结果缺失"
        stages.append(StartStageResult("training", "failed", error))

    stages.append(
        StartStageResult(
            "report",
            "passed",
            f"启动报告写入 {report_path}",
            {"path": report_path},
        )
    )
    result = StartFlowResult(
        status=status,
        config_path=flow_config.training_config,
        run_id=flow_config.run_id,
        run_dir=flow_config.run_dir,
        split=flow_config.split,
        dry_run=flow_config.dry_run,
        stages=tuple(stages),
        report_path=report_path,
        before_startup=before_startup,
        before_run=before_run,
        split_sync=split_sync,
        training_startup=training_startup,
        quality_report=quality_report,
        training=training,
        error=error,
    )
    try:
        write_start_flow_report(result, report_path)
    except OSError as exc:
        failed_stages = (
            *result.stages[:-1],
            StartStageResult("report", "failed", f"启动报告写入失败：{exc}"),
        )
        return replace(
            result,
            status="failed",
            stages=failed_stages,
            error=f"启动报告写入失败：{exc}",
        )
    return result


def _maybe_run_before_traning(
    before_settings: Any,
    *,
    before_startup: StartupCheckReport,
    config: StartFlowConfig,
) -> BeforeTrainingRunReport:
    """只在扫描明确发现新增原始数据时执行 before_traning。"""

    should_run = _before_should_run(before_startup)
    reason = _before_reason(before_startup)
    if not should_run:
        return BeforeTrainingRunReport(status="skipped", message=reason)
    if not config.run_before_traning:
        return BeforeTrainingRunReport(
            status="skipped",
            message="启动配置已禁用 before_traning 转换",
        )
    if config.dry_run:
        return BeforeTrainingRunReport(
            status="skipped",
            message="dry-run：检测到新数据，但未执行 before_traning 转换",
        )

    raw_results = TRAINING_PIPELINE.run_direct(before_settings)
    results = tuple(
        (str(stage), bool(success)) for stage, success in raw_results.items()
    )
    failed = tuple(stage for stage, success in results if not success)
    if failed:
        return BeforeTrainingRunReport(
            status="failed",
            stage_results=results,
            message=f"before_traning 失败阶段：{', '.join(failed)}",
        )

    manifest = recover_matched_sample_manifest(
        before_settings,
        matched_manifest_path=config.matched_manifest_path,
    )
    # 只有全部转换阶段成功后才能推进 matched manifest。
    manifest.save()
    return BeforeTrainingRunReport(
        status="passed",
        stage_results=results,
        message=f"before_traning 完成；matched manifest 已写入 {manifest.path}",
    )


def _sync_dataset_splits(
    config: V2Config,
    *,
    flow_config: StartFlowConfig,
) -> DatasetSplitSyncResult:
    """使用 V2 数据根目录同步唯一 item 级 split manifest。"""

    dataset_root = config.data.dataset_root
    manifest_path = flow_config.split_manifest_path or config.data.split_manifest
    return sync_dataset_split_manifest(
        dataset_root,
        manifest_path=manifest_path,
        seed=config.data.seed
        if flow_config.split_seed is None
        else flow_config.split_seed,
        ratios=SplitRatios(
            train=flow_config.train_ratio,
            validation=flow_config.validation_ratio,
            test=flow_config.test_ratio,
        ),
        allow_test_growth=flow_config.allow_test_growth,
        dry_run=flow_config.dry_run,
    )


def _effective_training_config(
    config: V2Config,
    flow_config: StartFlowConfig,
) -> V2Config:
    """把启动层允许覆盖的设备、清单和 seed 集体写回唯一 V2Config。"""

    data = replace(
        config.data,
        split_manifest=(flow_config.split_manifest_path or config.data.split_manifest),
        seed=config.data.seed
        if flow_config.split_seed is None
        else flow_config.split_seed,
    )
    runtime = config.runtime
    if flow_config.requested_device is not None:
        is_cuda = flow_config.requested_device is RuntimeDevice.CUDA
        runtime = replace(
            runtime,
            device=flow_config.requested_device,
            require_cuda=is_cuda,
            amp=is_cuda and runtime.amp,
        )
    return replace(config, data=data, runtime=runtime)


def _before_raw_data_result(report: StartupCheckReport) -> Any | None:
    """取得 before_traning 原始数据检查项。"""

    return next(
        (
            result
            for result in report.results
            if result.key == "before_traning:raw_data"
        ),
        None,
    )


def _before_should_run(report: StartupCheckReport) -> bool:
    """读取原始数据检查给出的唯一转换决策。"""

    result = _before_raw_data_result(report)
    return bool(result and result.details.get("should_run_before_traning"))


def _before_reason(report: StartupCheckReport) -> str:
    """返回原始数据决策的人类可读原因。"""

    result = _before_raw_data_result(report)
    if result is None:
        return "before_traning 未提供 raw-data 检查结果"
    return str(result.details.get("reason") or result.message)


def write_start_flow_report(result: StartFlowResult, path: Path) -> None:
    """写出最终审计快照；它不是训练 checkpoint。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _quality_report_dict(report: DataQualityReport | None) -> dict[str, Any] | None:
    """序列化 canonical 数据质量问题，不重新推导另一套 ok 语义。"""

    if report is None:
        return None
    return {
        "ok": report.ok,
        "issues": tuple(
            {
                "code": issue.code,
                "severity": issue.severity.value,
                "blocks_training": issue.blocks_training,
                "sample_id": issue.sample_id,
                "message": issue.message,
                "details": dict(issue.details),
            }
            for issue in report.issues
        ),
    }


def _json_ready(value: Any) -> Any:
    """递归转换报告中的 Path、Enum 和只读映射。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (DataSplit, RuntimeDevice)):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_ready(item) for item in value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


__all__ = (
    "START_FLOW_REPORT_FILENAME",
    "BeforeTrainingRunReport",
    "StartFlowConfig",
    "StartFlowResult",
    "StartStageResult",
    "TrainingExecutionResult",
    "TrainingExecutor",
    "run_start_flow",
    "write_start_flow_report",
)
