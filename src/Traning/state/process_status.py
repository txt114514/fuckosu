from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import json
from sqlmodel import Session, SQLModel, create_engine, select

from Traning.Lib.beatmap.order import OrderFolderWalker
from Traning.conf import load_settings
from Traning.Lib.defaults import DEFAULT_SETTINGS as DEFAULTS
from Traning.state.status_schema import (
    PROCESS_STEPS,
    STATUS_DB_FILENAME,
    ProcessStepStatus,
    decode_detail,
    default_status,
    encode_detail,
    normalize_process_steps,
    normalize_status,
)


class ProcessStatusManager:
    def __init__(
        self,
        target_root: str,
        order_filename: str = DEFAULTS.file_management.order_filename,
        status_filename: str = "process_status.json",
        process_steps: Iterable[str] | None = None,
        db_filename: str = STATUS_DB_FILENAME,
    ):
        self.target_root = Path(target_root)
        self.order_filename = order_filename
        self.status_filename = status_filename
        self.db_path = self.target_root / db_filename
        self.process_steps = (
            normalize_process_steps(process_steps)
            if process_steps is not None
            else normalize_process_steps(load_settings().progress.process_steps or PROCESS_STEPS)
        )
        self.walker = OrderFolderWalker(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        SQLModel.metadata.create_all(self.engine)

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
        return default_status(self.process_steps)

    def _validate_step(self, step: str):
        if step not in self.process_steps:
            raise ValueError(f"未知处理步骤: {step}")

    def _normalize_status(self, raw_status: dict[str, Any] | None) -> dict[str, Any]:
        return normalize_status(raw_status, self.process_steps)

    def _select_record(
        self,
        session: Session,
        folder_name: str,
        step: str,
    ) -> ProcessStepStatus | None:
        statement = select(ProcessStepStatus).where(
            ProcessStepStatus.target_root == str(self.target_root),
            ProcessStepStatus.folder_name == folder_name,
            ProcessStepStatus.step == step,
        )
        return session.exec(statement).first()

    def _has_records(self, folder_name: str) -> bool:
        with Session(self.engine) as session:
            statement = select(ProcessStepStatus).where(
                ProcessStepStatus.target_root == str(self.target_root),
                ProcessStepStatus.folder_name == folder_name,
            )
            return session.exec(statement).first() is not None

    def _load_legacy_json(self, folder_name: str) -> dict[str, Any] | None:
        status_path = self.get_status_path(folder_name)
        if not status_path.exists():
            return None
        with status_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def get_status_path(self, folder_name: str) -> Path:
        folder_path = self._require_existing_folder(folder_name)
        return folder_path / self.status_filename

    def load_status(self, folder_name: str) -> dict[str, Any]:
        folder_name = self._normalize_folder_name(folder_name)
        self._require_existing_folder(folder_name)

        if not self._has_records(folder_name):
            legacy_status = self._load_legacy_json(folder_name)
            if legacy_status is not None:
                self.save_status(folder_name, legacy_status)

        status = self._default_status()
        with Session(self.engine) as session:
            statement = select(ProcessStepStatus).where(
                ProcessStepStatus.target_root == str(self.target_root),
                ProcessStepStatus.folder_name == folder_name,
            )
            for row in session.exec(statement):
                if row.step not in status["steps"]:
                    continue
                status["steps"][row.step]["done"] = bool(row.done)
                status["steps"][row.step]["updated_at"] = row.updated_at
                status["steps"][row.step]["detail"] = decode_detail(row.detail_json)
        return status

    def save_status(self, folder_name: str, status: dict[str, Any]):
        folder_name = self._normalize_folder_name(folder_name)
        self._require_existing_folder(folder_name)
        normalized = self._normalize_status(status)

        with Session(self.engine) as session:
            for step, step_status in normalized["steps"].items():
                record = self._select_record(session, folder_name, step)
                if record is None:
                    record = ProcessStepStatus(
                        target_root=str(self.target_root),
                        folder_name=folder_name,
                        step=step,
                    )
                    session.add(record)
                record.done = bool(step_status["done"])
                record.updated_at = step_status["updated_at"]
                record.detail_json = encode_detail(step_status["detail"])
            session.commit()

    def ensure_status_file(self, folder_name: str) -> dict[str, Any]:
        status = self.load_status(folder_name)
        self.save_status(folder_name, status)
        return status

    def is_step_done(self, folder_name: str, step: str) -> bool:
        self._validate_step(step)
        status = self.load_status(folder_name)
        return bool(status["steps"][step]["done"])

    def mark_step_done(self, folder_name: str, step: str, detail: Any = None):
        self._validate_step(step)
        status = self.load_status(folder_name)
        status["steps"][step]["done"] = True
        status["steps"][step]["updated_at"] = datetime.now().isoformat(timespec="seconds")
        status["steps"][step]["detail"] = detail
        self.save_status(folder_name, status)

    def mark_step_pending(self, folder_name: str, step: str, detail: Any = None):
        self._validate_step(step)
        status = self.load_status(folder_name)
        status["steps"][step]["done"] = False
        status["steps"][step]["updated_at"] = datetime.now().isoformat(timespec="seconds")
        status["steps"][step]["detail"] = detail
        self.save_status(folder_name, status)

    def get_steps_summary(self, folder_name: str) -> dict[str, bool]:
        status = self.load_status(folder_name)
        return {step: bool(status["steps"][step]["done"]) for step in self.process_steps}
