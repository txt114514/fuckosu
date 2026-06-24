from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from traning.conf import Settings, load_settings
from traning.core.dataset_import import DataInputReport, check_data_input


@dataclass(frozen=True)
class TrainingStage:
    key: str
    run: Callable[[Settings], object]


TRAINING_STAGES = (TrainingStage("data_input", check_data_input),)


def run_pipeline(settings: Settings | None = None) -> dict[str, object]:
    selected = settings or load_settings()
    return {stage.key: stage.run(selected) for stage in TRAINING_STAGES}


__all__ = [
    "DataInputReport",
    "TRAINING_STAGES",
    "TrainingStage",
    "run_pipeline",
]
