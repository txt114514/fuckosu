from __future__ import annotations

import json
from typing import Any, Iterable

from sqlmodel import Field, SQLModel, UniqueConstraint


PROCESS_STEPS = (
    "osu_imported",
    "audio_imported",
    "verify_exported",
    "difficulty_exported",
    "video_matched",
    "av_corresponded",
    "video_processed",
)
STATUS_DB_FILENAME = ".process_status.sqlite"


class ProcessStepStatus(SQLModel, table=True):
    __tablename__ = "process_step_status"
    __table_args__ = (
        UniqueConstraint("target_root", "folder_name", "step", name="uq_process_step"),
    )

    id: int | None = Field(default=None, primary_key=True)
    target_root: str = Field(index=True)
    folder_name: str = Field(index=True)
    step: str = Field(index=True)
    done: bool = False
    updated_at: str | None = None
    detail_json: str | None = None


def normalize_process_steps(process_steps: Iterable[str]) -> tuple[str, ...]:
    normalized_steps: list[str] = []
    seen_steps: set[str] = set()

    for index, item in enumerate(process_steps):
        if not isinstance(item, str):
            raise ValueError(f"process_steps[{index}] 必须是字符串")

        step = item.strip()
        if not step:
            raise ValueError(f"process_steps[{index}] 不能为空字符串")
        if step in seen_steps:
            raise ValueError(f"process_steps 中存在重复步骤: {step}")

        seen_steps.add(step)
        normalized_steps.append(step)

    if not normalized_steps:
        raise ValueError("process_steps 不能为空")
    return tuple(normalized_steps)


def default_status(process_steps: Iterable[str]) -> dict[str, Any]:
    return {
        "steps": {
            step: {
                "done": False,
                "updated_at": None,
                "detail": None,
            }
            for step in process_steps
        }
    }


def normalize_status(
    raw_status: dict[str, Any] | None,
    process_steps: Iterable[str],
) -> dict[str, Any]:
    status = default_status(process_steps)
    if raw_status is None:
        return status
    if not isinstance(raw_status, dict):
        raise ValueError("状态结构非法，根节点必须是对象")

    raw_steps = raw_status.get("steps", {}) or {}
    if not isinstance(raw_steps, dict):
        raise ValueError("状态结构非法，steps 必须是对象")

    for step in process_steps:
        raw_step = raw_steps.get(step, {}) or {}
        if not isinstance(raw_step, dict):
            raise ValueError(f"状态结构非法，步骤 {step} 必须是对象")
        status["steps"][step]["done"] = bool(raw_step.get("done", False))
        status["steps"][step]["updated_at"] = raw_step.get("updated_at")
        status["steps"][step]["detail"] = raw_step.get("detail")

    return status


def encode_detail(detail: Any) -> str | None:
    if detail is None:
        return None
    return json.dumps(detail, ensure_ascii=False, default=str)


def decode_detail(detail_json: str | None) -> Any:
    if detail_json is None:
        return None
    try:
        return json.loads(detail_json)
    except json.JSONDecodeError:
        return detail_json
