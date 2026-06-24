from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping


CheckStatus = Literal["passed", "warning", "failed", "skipped"]


@dataclass(frozen=True)
class StartupCheckResult:
    key: str
    status: CheckStatus
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status != "failed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "status": self.status,
            "message": self.message,
            "details": json_ready(self.details),
        }


@dataclass(frozen=True)
class StartupCheckReport:
    scope: str
    results: tuple[StartupCheckResult, ...]
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.results)

    @property
    def failures(self) -> tuple[StartupCheckResult, ...]:
        return tuple(result for result in self.results if result.status == "failed")

    @property
    def warnings(self) -> tuple[StartupCheckResult, ...]:
        return tuple(result for result in self.results if result.status == "warning")

    def raise_for_errors(self) -> None:
        if self.ok:
            return
        messages = "; ".join(
            f"{result.key}: {result.message}" for result in self.failures
        )
        raise RuntimeError(f"startup self-check failed: {messages}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "ok": self.ok,
            "generated_at_utc": self.generated_at_utc,
            "results": tuple(result.as_dict() for result in self.results),
        }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(json_ready(item) for item in value)
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


__all__ = [
    "CheckStatus",
    "StartupCheckReport",
    "StartupCheckResult",
    "json_ready",
]
