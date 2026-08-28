"""隐藏 V2 SQLite 表结构和连接生命周期的 repository adapters。"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

from traning.contracts.common import require_identifier

from .models import DatasetCatalogEntry, PreprocessingMetadata, RepositoryError


_PREPROCESSING_TABLE = "v2_preprocessing_metadata"
_CATALOG_TABLE = "v2_dataset_catalog"
_SCHEMA_TABLE = "v2_repository_schema"
_SCHEMA_VERSION = 1
_Record = TypeVar("_Record")


class _SQLiteRepository:
    """集中管理连接、事务和 sqlite 错误转换。"""

    def __init__(self, database_path: Path) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path 必须是 pathlib.Path")
        if not str(database_path) or "\x00" in str(database_path):
            raise ValueError("database_path 必须是有效的非空路径")
        self._database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self._database_path)
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except sqlite3.Error as exc:
            raise RepositoryError(
                f"无法连接 SQLite 数据库：{self._database_path}"
            ) from exc

    def _execute_write(self, sql: str, parameters: tuple[object, ...]) -> int:
        try:
            with self._connect() as connection:
                cursor = connection.execute(sql, parameters)
                return cursor.rowcount
        except sqlite3.Error as exc:
            raise RepositoryError(f"SQLite 写入失败：{self._database_path}") from exc

    def _validate_schema(self, table: str, columns: tuple[str, ...]) -> None:
        """校验 adapter 所拥有表的版本和精确列集合。"""

        version = self._fetch_one(
            f"SELECT schema_version FROM {_SCHEMA_TABLE} WHERE table_name = ?",
            (table,),
            _decode_schema_version,
        )
        if version is None:
            raise RepositoryError(f"SQLite 未登记 V2 schema：{table}")
        if version != _SCHEMA_VERSION:
            raise RepositoryError(
                f"SQLite schema 版本不兼容：{table}={version}，要求 {_SCHEMA_VERSION}"
            )
        try:
            with self._connect() as connection:
                raw_rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError(f"无法检查 SQLite 表结构：{table}") from exc
        actual = tuple(_required_str(row[1], "column_name") for row in raw_rows)
        if actual != columns:
            raise RepositoryError(
                f"SQLite 表 {table} 的列与 V2 schema 不一致：{actual!r}"
            )

    def _initialize_schema(self, table: str, create_table_sql: str) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_SCHEMA_TABLE} (
                        table_name TEXT PRIMARY KEY NOT NULL,
                        schema_version INTEGER NOT NULL
                    )
                    """
                )
                connection.execute(create_table_sql)
                connection.execute(
                    f"INSERT INTO {_SCHEMA_TABLE} (table_name, schema_version) "
                    "VALUES (?, ?) ON CONFLICT(table_name) DO NOTHING",
                    (table, _SCHEMA_VERSION),
                )
        except sqlite3.Error as exc:
            raise RepositoryError(f"无法初始化 SQLite V2 schema：{table}") from exc

    def _fetch_one(
        self,
        sql: str,
        parameters: tuple[object, ...],
        decoder: Callable[[tuple[object, ...]], _Record],
    ) -> _Record | None:
        try:
            with self._connect() as connection:
                raw = connection.execute(sql, parameters).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"SQLite 查询失败：{self._database_path}") from exc
        if raw is None:
            return None
        return decoder(cast(tuple[object, ...], raw))

    def _fetch_all(
        self,
        sql: str,
        parameters: tuple[object, ...],
        decoder: Callable[[tuple[object, ...]], _Record],
    ) -> tuple[_Record, ...]:
        try:
            with self._connect() as connection:
                rows = connection.execute(sql, parameters).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError(f"SQLite 查询失败：{self._database_path}") from exc
        return tuple(decoder(cast(tuple[object, ...], row)) for row in rows)


class SQLitePreprocessingMetadataRepository(_SQLiteRepository):
    """拥有固定 V2 schema 的预处理元数据 SQLite adapter。"""

    _COLUMNS = (
        "item_name",
        "source_width",
        "source_height",
        "crop_left",
        "crop_top",
        "crop_width",
        "crop_height",
        "updated_at",
    )

    def __init__(self, database_path: Path, *, _initialize: bool = False) -> None:
        super().__init__(database_path)
        if _initialize:
            self._create_schema()
        self._validate_schema(_PREPROCESSING_TABLE, self._COLUMNS)

    @classmethod
    def create(cls, database_path: Path) -> SQLitePreprocessingMetadataRepository:
        """初始化稳定 V2 表并返回 adapter；不迁移或猜测其他 schema。"""

        return cls(database_path, _initialize=True)

    def _create_schema(self) -> None:
        sql = f"""
            CREATE TABLE IF NOT EXISTS {_PREPROCESSING_TABLE} (
                item_name TEXT PRIMARY KEY NOT NULL,
                source_width INTEGER NOT NULL CHECK (source_width > 0),
                source_height INTEGER NOT NULL CHECK (source_height > 0),
                crop_left INTEGER NOT NULL CHECK (crop_left >= 0),
                crop_top INTEGER NOT NULL CHECK (crop_top >= 0),
                crop_width INTEGER NOT NULL CHECK (crop_width > 0),
                crop_height INTEGER NOT NULL CHECK (crop_height > 0),
                updated_at TEXT NOT NULL
            )
        """
        self._initialize_schema(_PREPROCESSING_TABLE, sql)

    def get(self, item_name: str) -> PreprocessingMetadata | None:
        """按条目标识解码一份预处理元数据，缺失时返回 ``None``。"""

        require_identifier(item_name, "item_name")
        return self._fetch_one(
            f"SELECT item_name, source_width, source_height, crop_left, crop_top, "
            f"crop_width, crop_height, updated_at FROM {_PREPROCESSING_TABLE} "
            "WHERE item_name = ?",
            (item_name,),
            _decode_preprocessing_metadata,
        )

    def list_all(self) -> tuple[PreprocessingMetadata, ...]:
        """按条目标识稳定排序并解码全部预处理元数据。"""

        return self._fetch_all(
            f"SELECT item_name, source_width, source_height, crop_left, crop_top, "
            f"crop_width, crop_height, updated_at FROM {_PREPROCESSING_TABLE} "
            "ORDER BY item_name",
            (),
            _decode_preprocessing_metadata,
        )

    def save(self, metadata: PreprocessingMetadata) -> None:
        """在单次事务中新增或更新一份 typed 预处理元数据。"""

        if not isinstance(metadata, PreprocessingMetadata):
            raise TypeError("metadata 必须是 PreprocessingMetadata")
        self._execute_write(
            f"""
                INSERT INTO {_PREPROCESSING_TABLE} (
                    item_name, source_width, source_height, crop_left, crop_top,
                    crop_width, crop_height, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_name) DO UPDATE SET
                    source_width = excluded.source_width,
                    source_height = excluded.source_height,
                    crop_left = excluded.crop_left,
                    crop_top = excluded.crop_top,
                    crop_width = excluded.crop_width,
                    crop_height = excluded.crop_height,
                    updated_at = excluded.updated_at
            """,
            (
                metadata.item_name,
                metadata.source_width,
                metadata.source_height,
                metadata.crop_left,
                metadata.crop_top,
                metadata.crop_width,
                metadata.crop_height,
                metadata.updated_at,
            ),
        )

    def delete(self, item_name: str) -> bool:
        """事务性删除指定预处理条目并返回此前是否存在。"""

        require_identifier(item_name, "item_name")
        return (
            self._execute_write(
                f"DELETE FROM {_PREPROCESSING_TABLE} WHERE item_name = ?", (item_name,)
            )
            > 0
        )


class SQLiteDatasetCatalogRepository(_SQLiteRepository):
    """拥有固定 V2 schema 的数据集目录 SQLite adapter。"""

    _COLUMNS = (
        "folder_name",
        "source_name",
        "sequence",
        "osu_filename",
        "source_osz_name",
        "source_mtime_ns",
        "difficulty_value",
        "active",
    )

    def __init__(self, database_path: Path, *, _initialize: bool = False) -> None:
        super().__init__(database_path)
        if _initialize:
            self._create_schema()
        self._validate_schema(_CATALOG_TABLE, self._COLUMNS)

    @classmethod
    def create(cls, database_path: Path) -> SQLiteDatasetCatalogRepository:
        """初始化稳定 V2 表并返回 adapter；不迁移或猜测其他 schema。"""

        return cls(database_path, _initialize=True)

    def _create_schema(self) -> None:
        sql = f"""
            CREATE TABLE IF NOT EXISTS {_CATALOG_TABLE} (
                folder_name TEXT PRIMARY KEY NOT NULL,
                source_name TEXT NOT NULL UNIQUE,
                sequence INTEGER NOT NULL CHECK (sequence >= 0),
                osu_filename TEXT NOT NULL,
                source_osz_name TEXT,
                source_mtime_ns INTEGER CHECK (source_mtime_ns >= 0),
                difficulty_value REAL CHECK (difficulty_value >= 0),
                active INTEGER NOT NULL CHECK (active IN (0, 1))
            )
        """
        self._initialize_schema(_CATALOG_TABLE, sql)

    def get(self, folder_name: str) -> DatasetCatalogEntry | None:
        """按目录名解码一份数据集条目，缺失时返回 ``None``。"""

        require_identifier(folder_name, "folder_name")
        return self._fetch_one(
            _catalog_select_sql() + " WHERE folder_name = ?",
            (folder_name,),
            _decode_catalog_entry,
        )

    def list_all(self, *, active_only: bool = False) -> tuple[DatasetCatalogEntry, ...]:
        """按序号和目录名列出条目，可限制为活动数据集。"""

        if not isinstance(active_only, bool):
            raise TypeError("active_only 必须是布尔值")
        where = " WHERE active = ?" if active_only else ""
        parameters: tuple[object, ...] = (1,) if active_only else ()
        return self._fetch_all(
            _catalog_select_sql() + where + " ORDER BY sequence, folder_name",
            parameters,
            _decode_catalog_entry,
        )

    def save(self, entry: DatasetCatalogEntry) -> None:
        """在单次事务中新增或更新一份 typed 数据集目录条目。"""

        if not isinstance(entry, DatasetCatalogEntry):
            raise TypeError("entry 必须是 DatasetCatalogEntry")
        self._execute_write(
            f"""
                INSERT INTO {_CATALOG_TABLE} (
                    folder_name, source_name, sequence, osu_filename,
                    source_osz_name, source_mtime_ns, difficulty_value, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(folder_name) DO UPDATE SET
                    source_name = excluded.source_name,
                    sequence = excluded.sequence,
                    osu_filename = excluded.osu_filename,
                    source_osz_name = excluded.source_osz_name,
                    source_mtime_ns = excluded.source_mtime_ns,
                    difficulty_value = excluded.difficulty_value,
                    active = excluded.active
            """,
            (
                entry.folder_name,
                entry.source_name,
                entry.sequence,
                entry.osu_filename,
                entry.source_osz_name,
                entry.source_mtime_ns,
                entry.difficulty_value,
                int(entry.active),
            ),
        )

    def delete(self, folder_name: str) -> bool:
        """事务性删除指定数据集目录并返回此前是否存在。"""

        require_identifier(folder_name, "folder_name")
        return (
            self._execute_write(
                f"DELETE FROM {_CATALOG_TABLE} WHERE folder_name = ?", (folder_name,)
            )
            > 0
        )


def _catalog_select_sql() -> str:
    return (
        "SELECT folder_name, source_name, sequence, osu_filename, source_osz_name, "
        f"source_mtime_ns, difficulty_value, active FROM {_CATALOG_TABLE}"
    )


def _decode_schema_version(row: tuple[object, ...]) -> int:
    _require_row_length(row, 1, "schema version")
    return _required_int(row[0], "schema_version")


def _require_row_length(
    row: tuple[object, ...], expected: int, record_name: str
) -> None:
    if len(row) != expected:
        raise RepositoryError(f"{record_name} 的 SQLite 列数不符合 V2 schema")


def _required_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise RepositoryError(f"SQLite 字段 {field_name} 必须是字符串")
    return value


def _optional_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_str(value, field_name)


def _required_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RepositoryError(f"SQLite 字段 {field_name} 必须是整数")
    return value


def _optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _required_int(value, field_name)


def _optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RepositoryError(f"SQLite 字段 {field_name} 必须是数值")
    return float(value)


def _decode_preprocessing_metadata(row: tuple[object, ...]) -> PreprocessingMetadata:
    _require_row_length(row, 8, "PreprocessingMetadata")
    try:
        return PreprocessingMetadata(
            item_name=_required_str(row[0], "item_name"),
            source_width=_required_int(row[1], "source_width"),
            source_height=_required_int(row[2], "source_height"),
            crop_left=_required_int(row[3], "crop_left"),
            crop_top=_required_int(row[4], "crop_top"),
            crop_width=_required_int(row[5], "crop_width"),
            crop_height=_required_int(row[6], "crop_height"),
            updated_at=_required_str(row[7], "updated_at"),
        )
    except (TypeError, ValueError) as exc:
        raise RepositoryError("SQLite 预处理元数据不满足领域契约") from exc


def _decode_catalog_entry(row: tuple[object, ...]) -> DatasetCatalogEntry:
    _require_row_length(row, 8, "DatasetCatalogEntry")
    active = _required_int(row[7], "active")
    if active not in (0, 1):
        raise RepositoryError("SQLite 字段 active 必须是 0 或 1")
    try:
        return DatasetCatalogEntry(
            folder_name=_required_str(row[0], "folder_name"),
            source_name=_required_str(row[1], "source_name"),
            sequence=_required_int(row[2], "sequence"),
            osu_filename=_required_str(row[3], "osu_filename"),
            source_osz_name=_optional_str(row[4], "source_osz_name"),
            source_mtime_ns=_optional_int(row[5], "source_mtime_ns"),
            difficulty_value=_optional_float(row[6], "difficulty_value"),
            active=bool(active),
        )
    except (TypeError, ValueError) as exc:
        raise RepositoryError("SQLite 数据集目录条目不满足领域契约") from exc


__all__ = (
    "SQLiteDatasetCatalogRepository",
    "SQLitePreprocessingMetadataRepository",
)
