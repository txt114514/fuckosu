"""验证 repository 隐藏 SQLite schema 并只返回领域对象。"""

from __future__ import annotations

import sqlite3

import pytest

from traning.core.data.repositories import (
    DatasetCatalogEntry,
    InMemoryDatasetCatalogRepository,
    InMemoryPreprocessingMetadataRepository,
    PreprocessingMetadata,
    RepositoryError,
    SQLiteDatasetCatalogRepository,
    SQLitePreprocessingMetadataRepository,
)


def _metadata(name: str = "item-a") -> PreprocessingMetadata:
    return PreprocessingMetadata(
        item_name=name,
        source_width=1920,
        source_height=1080,
        crop_left=100,
        crop_top=50,
        crop_width=1280,
        crop_height=720,
        updated_at="2026-08-24T00:00:00Z",
    )


def _catalog(folder: str = "folder-a") -> DatasetCatalogEntry:
    return DatasetCatalogEntry(
        folder_name=folder,
        source_name=f"source-{folder}",
        sequence=1,
        osu_filename="map.osu",
        difficulty_value=5.2,
    )


def test_memory_repositories_return_typed_deterministic_snapshots() -> None:
    """内存实现遵循与持久层相同的稳定领域契约。"""

    metadata_repo = InMemoryPreprocessingMetadataRepository(
        (_metadata("z"), _metadata("a"))
    )
    catalog_repo = InMemoryDatasetCatalogRepository((_catalog("z"), _catalog("a")))
    assert [item.item_name for item in metadata_repo.list_all()] == ["a", "z"]
    assert all(
        isinstance(item, PreprocessingMetadata) for item in metadata_repo.list_all()
    )
    assert all(
        isinstance(item, DatasetCatalogEntry) for item in catalog_repo.list_all()
    )
    assert metadata_repo.get("a") == _metadata("a")
    assert metadata_repo.delete("a") is True
    assert metadata_repo.delete("a") is False


def test_sqlite_repositories_round_trip_without_exposing_rows(tmp_path) -> None:
    """SQLite table/column 留在 adapter 内，调用方只见 dataclass。"""

    database_path = tmp_path / "v2.sqlite3"
    metadata_repo = SQLitePreprocessingMetadataRepository.create(database_path)
    catalog_repo = SQLiteDatasetCatalogRepository.create(database_path)
    metadata_repo.save(_metadata())
    catalog_repo.save(_catalog())
    assert metadata_repo.get("item-a") == _metadata()
    assert catalog_repo.get("folder-a") == _catalog()
    assert isinstance(metadata_repo.list_all(), tuple)
    assert isinstance(catalog_repo.list_all(), tuple)
    assert not isinstance(metadata_repo.get("item-a"), sqlite3.Row)
    assert (
        SQLitePreprocessingMetadataRepository(database_path).get("item-a")
        == _metadata()
    )
    assert SQLiteDatasetCatalogRepository(database_path).get("folder-a") == _catalog()


def test_sqlite_repository_rejects_missing_or_wrong_schema(tmp_path) -> None:
    """adapter 不猜测或迁移未知 preprocessing 表结构。"""

    missing_path = tmp_path / "missing.sqlite3"
    with pytest.raises(RepositoryError):
        SQLitePreprocessingMetadataRepository(missing_path)

    database_path = tmp_path / "wrong.sqlite3"
    SQLitePreprocessingMetadataRepository.create(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE v2_repository_schema SET schema_version = 999 "
            "WHERE table_name = 'v2_preprocessing_metadata'"
        )
    with pytest.raises(RepositoryError, match="版本"):
        SQLitePreprocessingMetadataRepository(database_path)
