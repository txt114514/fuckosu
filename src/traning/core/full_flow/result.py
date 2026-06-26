from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from traning.core.full_flow.stages import FullFlowStageStatus


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass
class FullFlowStageState:
    stage_id: str
    display_name: str
    status: FullFlowStageStatus = "PENDING"
    started_at: str | None = None
    ended_at: str | None = None
    result: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: str | None = None
    artifacts: tuple[str, ...] = ()
    restored: bool = False

    def mark_started(self) -> None:
        self.status = "RUNNING"
        self.started_at = utc_now()
        self.ended_at = None
        self.error = None

    def mark_finished(
        self,
        status: FullFlowStageStatus,
        *,
        result: Mapping[str, Any] | None = None,
        warnings: tuple[str, ...] = (),
        error: str | None = None,
        artifacts: tuple[str, ...] = (),
        restored: bool = False,
    ) -> None:
        self.status = status
        self.ended_at = utc_now()
        self.result = dict(result or {})
        self.warnings = warnings
        self.error = error
        self.artifacts = artifacts
        self.restored = restored

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FullFlowResult:
    run_id: str
    mode: str
    status: str
    output_dir: Path
    manifest_path: Path
    state_path: Path
    report_json_path: Path
    report_markdown_path: Path
    stages: tuple[FullFlowStageState, ...]
    started_at: str
    ended_at: str | None = None
    resume_report_path: Path | None = None
    ramp_manifest_path: Path | None = None
    final_readiness_path: Path | None = None
    stop_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "status": self.status,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "state_path": self.state_path,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "stages": tuple(stage.as_dict() for stage in self.stages),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "resume_report_path": self.resume_report_path,
            "ramp_manifest_path": self.ramp_manifest_path,
            "final_readiness_path": self.final_readiness_path,
            "stop_reason": self.stop_reason,
        }


__all__ = [
    "FullFlowResult",
    "FullFlowStageState",
    "utc_now",
]
