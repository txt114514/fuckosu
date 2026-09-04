"""供测试和纯内存编排使用的 repository 实现。"""

from __future__ import annotations

from collections.abc import Iterable

from traning.state.common import require_identifier

from .models import DatasetCatalogEntry, PreprocessingMetadata


class InMemoryPreprocessingMetadataRepository:
    """以稳定标识符维护预处理元数据快照。"""

    def __init__(self, records: Iterable[PreprocessingMetadata] = ()) -> None:
        self._records: dict[str, PreprocessingMetadata] = {}
        for record in records:
            self.save(record)

    def get(self, item_name: str) -> PreprocessingMetadata | None:
        """按预处理条目标识返回当前快照，缺失时返回 ``None``。"""

        require_identifier(item_name, "item_name")
        return self._records.get(item_name)

    def list_all(self) -> tuple[PreprocessingMetadata, ...]:
        """按条目标识稳定排序并返回全部预处理元数据。"""

        return tuple(self._records[key] for key in sorted(self._records))

    def save(self, metadata: PreprocessingMetadata) -> None:
        """以条目标识新增或替换一份 typed 元数据。"""

        if not isinstance(metadata, PreprocessingMetadata):
            raise TypeError("metadata 必须是 PreprocessingMetadata")
        self._records[metadata.item_name] = metadata

    def delete(self, item_name: str) -> bool:
        """删除指定预处理条目并返回此前是否存在。"""

        require_identifier(item_name, "item_name")
        return self._records.pop(item_name, None) is not None


class InMemoryDatasetCatalogRepository:
    """以内存字典维护数据集目录快照。"""

    def __init__(self, entries: Iterable[DatasetCatalogEntry] = ()) -> None:
        self._entries: dict[str, DatasetCatalogEntry] = {}
        for entry in entries:
            self.save(entry)

    def get(self, folder_name: str) -> DatasetCatalogEntry | None:
        """按目录名返回当前目录条目，缺失时返回 ``None``。"""

        require_identifier(folder_name, "folder_name")
        return self._entries.get(folder_name)

    def list_all(self, *, active_only: bool = False) -> tuple[DatasetCatalogEntry, ...]:
        """按序号和目录名稳定排序，可选择只返回活动条目。"""

        if not isinstance(active_only, bool):
            raise TypeError("active_only 必须是布尔值")
        entries = (
            entry for entry in self._entries.values() if not active_only or entry.active
        )
        return tuple(
            sorted(entries, key=lambda entry: (entry.sequence, entry.folder_name))
        )

    def save(self, entry: DatasetCatalogEntry) -> None:
        """以目录名新增或替换一份 typed 目录条目。"""

        if not isinstance(entry, DatasetCatalogEntry):
            raise TypeError("entry 必须是 DatasetCatalogEntry")
        self._entries[entry.folder_name] = entry

    def delete(self, folder_name: str) -> bool:
        """删除指定目录条目并返回此前是否存在。"""

        require_identifier(folder_name, "folder_name")
        return self._entries.pop(folder_name, None) is not None


__all__ = (
    "InMemoryDatasetCatalogRepository",
    "InMemoryPreprocessingMetadataRepository",
)
