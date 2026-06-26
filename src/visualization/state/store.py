from __future__ import annotations

from pathlib import Path

from visualization.lib.models import TrainingDashboardState, TrainingEvent
from visualization.state.persistence import append_jsonl, atomic_write_json


class DashboardStateStore:
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
        atomic_write_json(self.state_path, state.as_dict())

    def append_event(self, event: TrainingEvent) -> None:
        append_jsonl(self.events_path, event.as_dict())

    def write_named(self, name: str, payload: object) -> Path:
        path = self.root / name
        atomic_write_json(path, payload)
        return path
