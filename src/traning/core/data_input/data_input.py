from __future__ import annotations

from torch.utils.data import DataLoader

from traning.conf import Settings, load_settings
from traning.core.data_input.loader import build_dataloader, build_dataset
from traning.core.data_input.preflight import DataInputReport, inspect_data_input


class DataInputModule:
    def __init__(self, settings: Settings):
        self.settings = settings

    def inspect(self) -> DataInputReport:
        return inspect_data_input(self.settings)

    def dataset(self):
        return build_dataset(self.settings)

    def dataloader(self) -> DataLoader:
        return build_dataloader(self.settings)


def check_data_input(settings: Settings | None = None) -> DataInputReport:
    return DataInputModule(settings or load_settings()).inspect()


__all__ = ["DataInputModule", "check_data_input"]
