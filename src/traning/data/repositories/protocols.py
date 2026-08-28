"""数据编排依赖的 repository Protocol。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import DatasetCatalogEntry, PreprocessingMetadata


@runtime_checkable
class PreprocessingMetadataRepository(Protocol):
    """预处理元数据的稳定读写契约。"""

    def get(self, item_name: str) -> PreprocessingMetadata | None:
        """按数据项名称读取元数据；不存在时返回 ``None``。"""

    def list_all(self) -> tuple[PreprocessingMetadata, ...]:
        """按数据项名称返回不可变快照。"""

    def save(self, metadata: PreprocessingMetadata) -> None:
        """新增或完整替换一条元数据。"""

    def delete(self, item_name: str) -> bool:
        """删除元数据，并返回删除前是否存在。"""


@runtime_checkable
class DatasetCatalogRepository(Protocol):
    """数据集目录的稳定读写契约。"""

    def get(self, folder_name: str) -> DatasetCatalogEntry | None:
        """按目录名称读取条目；不存在时返回 ``None``。"""

    def list_all(self, *, active_only: bool = False) -> tuple[DatasetCatalogEntry, ...]:
        """按 sequence、folder_name 返回不可变目录快照。"""

    def save(self, entry: DatasetCatalogEntry) -> None:
        """新增或完整替换一个目录条目。"""

    def delete(self, folder_name: str) -> bool:
        """删除目录条目，并返回删除前是否存在。"""


__all__ = ("DatasetCatalogRepository", "PreprocessingMetadataRepository")
