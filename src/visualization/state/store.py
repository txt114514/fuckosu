"""管理单次训练的最新仪表盘快照、事件流和命名状态文件。"""

from __future__ import annotations

from pathlib import Path

from visualization.lib.models import TrainingDashboardState, TrainingEvent
from visualization.state.persistence import append_jsonl, atomic_write_json


class DashboardStateStore:
    """将覆盖式快照与追加式事件流隔离到固定文件契约。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.root / "dashboard_state.json"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    def write_state(self, state: TrainingDashboardState) -> None:
        # dashboard_state.json 是观察者轮询的“当前真值”，必须用原子替换发布。
        atomic_write_json(self.state_path, state.as_dict())

    def append_event(self, event: TrainingEvent) -> None:
        # 历史事件不可被最新快照覆盖，使用 JSONL 保持发生顺序和审计能力。
        append_jsonl(self.events_path, event.as_dict())

    def write_named(self, name: str, payload: object) -> Path:
        path = self.root / name
        atomic_write_json(path, payload)
        return path
