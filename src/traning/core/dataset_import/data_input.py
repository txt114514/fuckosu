from __future__ import annotations

from torch.utils.data import DataLoader

from traning.conf import DataSplit, Settings, load_settings
from traning.core.dataset_import.loader import build_dataloader, build_dataset
from traning.core.dataset_import.preflight import DataInputReport, inspect_data_input


class DataInputModule:
    def __init__(self, settings: Settings):
        self.settings = settings

    def inspect(self, *, split: DataSplit = "all") -> DataInputReport:
        return inspect_data_input(self.settings, split=split)

    def dataset(self, *, split: DataSplit = "train"):
        return build_dataset(self.settings, split=split)

    def dataloader(
        self,
        *,
        split: DataSplit = "train",
        shuffle: bool | None = None,
    ) -> DataLoader:
        return build_dataloader(self.settings, split=split, shuffle=shuffle)


def check_data_input(
    settings: Settings | None = None,
    *,
    split: DataSplit = "all",
) -> DataInputReport:
    return DataInputModule(settings or load_settings()).inspect(split=split)


__all__ = ["DataInputModule", "check_data_input"]
