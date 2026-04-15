from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from Traning.Lib.traning_package_manager.order_walker import OrderFolderWalker


PROCESS_STEPS = (
    "osu_imported",
    "verify_exported",
    "difficulty_exported",
    "video_matched",
    "video_processed",
)


class ProcessStatusManager:
    def __init__(
        self,
        target_root: str,
        order_filename: str = "order.txt",
        status_filename: str = "process_status.json",
    ):
        self.target_root = Path(target_root)
        self.order_filename = order_filename
        self.status_filename = status_filename
        self.walker = OrderFolderWalker(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )

    def _normalize_folder_name(self, folder_name: str) -> str:
        folder_name = folder_name.strip()
        if not folder_name:
            raise ValueError("folder_name 不能为空")
        if Path(folder_name).name != folder_name:
            raise ValueError(f"folder_name 非法，不能包含路径层级: {folder_name}")
        return folder_name

    def _registered_names(self) -> set[str]:
        return set(self.walker.read_folder_names())

    def _assert_registered(self, folder_name: str):
        folder_name = self._normalize_folder_name(folder_name)
        if folder_name not in self._registered_names():
            raise PermissionError(
                f"{folder_name} 未登记在 {self.target_root / self.order_filename} 中，不允许使用"
            )

    def _require_existing_folder(self, folder_name: str) -> Path:
        folder_name = self._normalize_folder_name(folder_name)
        self._assert_registered(folder_name)
        folder_path = self.target_root / folder_name
        if not folder_path.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        return folder_path

    def _default_status(self) -> dict[str, Any]:
        return {
            "steps": {
                step: {
                    "done": False,
                    "updated_at": None,
                    "detail": None,
                }
                for step in PROCESS_STEPS
            }
        }

    def _validate_step(self, step: str):
        if step not in PROCESS_STEPS:
            raise ValueError(f"未知处理步骤: {step}")

    def _normalize_status(self, raw_status: dict[str, Any] | None) -> dict[str, Any]:
        status = self._default_status()
        if raw_status is None:
            return status
        if not isinstance(raw_status, dict):
            raise ValueError("状态文件结构非法，根节点必须是对象")

        raw_steps = raw_status.get("steps", {})
        if raw_steps is None:
            raw_steps = {}
        if not isinstance(raw_steps, dict):
            raise ValueError("状态文件结构非法，steps 必须是对象")

        for step in PROCESS_STEPS:
            raw_step = raw_steps.get(step, {})
            if raw_step is None:
                raw_step = {}
            if not isinstance(raw_step, dict):
                raise ValueError(f"状态文件结构非法，步骤 {step} 必须是对象")

            status["steps"][step]["done"] = bool(raw_step.get("done", False))
            status["steps"][step]["updated_at"] = raw_step.get("updated_at")
            status["steps"][step]["detail"] = raw_step.get("detail")

        return status

    def get_status_path(self, folder_name: str) -> Path:
        folder_path = self._require_existing_folder(folder_name)
        return folder_path / self.status_filename

    def load_status(self, folder_name: str) -> dict[str, Any]:
        status_path = self.get_status_path(folder_name)
        if not status_path.exists():
            return self._default_status()

        import json

        with status_path.open("r", encoding="utf-8") as f:
            raw_status = json.load(f)

        return self._normalize_status(raw_status)

    def save_status(self, folder_name: str, status: dict[str, Any]):
        status_path = self.get_status_path(folder_name)
        normalized = self._normalize_status(status)

        import json

        with status_path.open("w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def ensure_status_file(self, folder_name: str) -> dict[str, Any]:
        status = self.load_status(folder_name)
        self.save_status(folder_name, status)
        return status

    def is_step_done(self, folder_name: str, step: str) -> bool:
        self._validate_step(step)
        status = self.load_status(folder_name)
        return bool(status["steps"][step]["done"])

    def mark_step_done(
        self,
        folder_name: str,
        step: str,
        detail: Any = None,
    ):
        self._validate_step(step)
        status = self.load_status(folder_name)
        status["steps"][step]["done"] = True
        status["steps"][step]["updated_at"] = datetime.now().isoformat(timespec="seconds")
        status["steps"][step]["detail"] = detail
        self.save_status(folder_name, status)

    def mark_step_pending(
        self,
        folder_name: str,
        step: str,
        detail: Any = None,
    ):
        self._validate_step(step)
        status = self.load_status(folder_name)
        status["steps"][step]["done"] = False
        status["steps"][step]["updated_at"] = datetime.now().isoformat(timespec="seconds")
        status["steps"][step]["detail"] = detail
        self.save_status(folder_name, status)

    def get_steps_summary(self, folder_name: str) -> dict[str, bool]:
        status = self.load_status(folder_name)
        return {
            step: bool(status["steps"][step]["done"])
            for step in PROCESS_STEPS
        }
