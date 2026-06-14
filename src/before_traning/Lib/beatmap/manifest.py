from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sqlmodel import Session, create_engine, select

from before_traning.Lib.common.sequence import format_sequence_name
from before_traning.state.manifest_schema import (
    MANIFEST_DB_FILENAME,
    BeatmapDataRecord,
    PackageManifestItem,
)


INTERNAL_FOLDER_PREFIX = "item_"
LEGACY_VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
MANIFEST_TABLE_FILENAME = "manifest.csv"
LEGACY_DIFFICULTY_FILENAME = "difficulty.txt"


@dataclass(frozen=True)
class ManifestEntry:
    source_name: str
    osu_filename: str | None = None
    source_osz_name: str | None = None
    source_mtime_ns: int | None = None


class PackageManifest:
    """Small SQLite manifest for stable internal folder IDs and processing order."""

    def __init__(
        self,
        target_root: str,
        manifest_filename: str = MANIFEST_DB_FILENAME,
        legacy_order_filename: str = "order.txt",
        table_filename: str = MANIFEST_TABLE_FILENAME,
    ):
        self.target_root = Path(target_root)
        self.target_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.target_root / manifest_filename
        self.legacy_order_path = self.target_root / legacy_order_filename
        self.table_path = self.target_root / table_filename
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        PackageManifestItem.__table__.create(self.engine, checkfirst=True)
        BeatmapDataRecord.__table__.create(self.engine, checkfirst=True)
        self._ensure_schema()
        self._migrate_legacy_order()
        self._migrate_legacy_difficulty_files()
        self.export_table()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(package_manifest_item)")
            }
            if "difficulty_value" not in columns:
                connection.execute(
                    "ALTER TABLE package_manifest_item "
                    "ADD COLUMN difficulty_value REAL"
                )

    def _all_items(self) -> list[PackageManifestItem]:
        with Session(self.engine) as session:
            statement = select(PackageManifestItem).order_by(PackageManifestItem.sequence)
            return list(session.exec(statement))

    def _normalize_source_name(self, source_name: str) -> str:
        source_name = source_name.strip()
        if not source_name:
            raise ValueError("source_name 不能为空")
        if Path(source_name).name != source_name:
            raise ValueError(f"source_name 非法，不能包含路径层级: {source_name}")
        return source_name

    def _next_folder_number(self, items: list[PackageManifestItem]) -> int:
        numbers: list[int] = []
        for item in items:
            if not item.folder_name.startswith(INTERNAL_FOLDER_PREFIX):
                continue
            suffix = item.folder_name.removeprefix(INTERNAL_FOLDER_PREFIX)
            if suffix.isdigit():
                numbers.append(int(suffix))
        return max(numbers, default=0) + 1

    def _folder_name(self, number: int) -> str:
        return format_sequence_name(INTERNAL_FOLDER_PREFIX, number)

    def _legacy_osu_filename(self, source_name: str) -> str | None:
        folder_path = self.target_root / source_name
        if not folder_path.is_dir():
            return None
        osu_files = sorted(folder_path.glob("*.osu"), key=lambda path: path.name.lower())
        return osu_files[0].name if osu_files else None

    def _rename_legacy_folders(self, mappings: list[tuple[str, str]]) -> None:
        temporary: list[tuple[Path, Path, Path]] = []
        completed: list[tuple[Path, Path]] = []
        renamed_videos: list[tuple[Path, Path]] = []

        for source_name, folder_name in mappings:
            source_path = self.target_root / source_name
            destination_path = self.target_root / folder_name
            if source_path == destination_path or not source_path.exists():
                continue
            if destination_path.exists():
                raise FileExistsError(f"manifest 迁移目标已存在: {destination_path}")
            temp_path = self.target_root / f".__manifest_tmp__{uuid4().hex}"
            source_path.rename(temp_path)
            temporary.append((temp_path, source_path, destination_path))

        try:
            for temp_path, source_path, destination_path in temporary:
                temp_path.rename(destination_path)
                completed.append((destination_path, source_path))
        except Exception:
            for destination_path, source_path in reversed(completed):
                if destination_path.exists():
                    destination_path.rename(source_path)
            for temp_path, source_path, _destination_path in temporary:
                if temp_path.exists():
                    temp_path.rename(source_path)
            raise

        try:
            for source_name, folder_name in mappings:
                folder_path = self.target_root / folder_name
                folder_path.mkdir(parents=True, exist_ok=True)
                for path in list(folder_path.iterdir()):
                    if (
                        path.is_file()
                        and path.stem == source_name
                        and path.suffix.lower() in LEGACY_VIDEO_SUFFIXES
                    ):
                        destination = path.with_name(f"{folder_name}{path.suffix}")
                        if destination.exists():
                            raise FileExistsError(f"manifest 视频迁移目标已存在: {destination}")
                        path.rename(destination)
                        renamed_videos.append((destination, path))
        except Exception:
            for destination, source in reversed(renamed_videos):
                if destination.exists():
                    destination.rename(source)
            self._restore_legacy_folders(mappings)
            raise

    def _restore_legacy_folders(self, mappings: list[tuple[str, str]]) -> None:
        for source_name, folder_name in reversed(mappings):
            folder_path = self.target_root / folder_name
            source_path = self.target_root / source_name
            if not folder_path.exists() or source_path.exists():
                continue
            for path in list(folder_path.iterdir()):
                if (
                    path.is_file()
                    and path.stem == folder_name
                    and path.suffix.lower() in LEGACY_VIDEO_SUFFIXES
                ):
                    path.rename(path.with_name(f"{source_name}{path.suffix}"))
            folder_path.rename(source_path)

    def _migrate_legacy_order(self) -> None:
        if self._all_items() or not self.legacy_order_path.is_file():
            return

        source_names = [
            self._normalize_source_name(line)
            for line in self.legacy_order_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(source_names) != len(set(source_names)):
            raise ValueError(f"{self.legacy_order_path} 中存在重复目录名")
        if not source_names:
            self.legacy_order_path.unlink()
            return

        mappings = [
            (source_name, self._folder_name(sequence))
            for sequence, source_name in enumerate(source_names, start=1)
        ]
        osu_filenames = {
            source_name: self._legacy_osu_filename(source_name)
            for source_name in source_names
        }
        self._rename_legacy_folders(mappings)

        try:
            with Session(self.engine) as session:
                for sequence, (source_name, folder_name) in enumerate(mappings, start=1):
                    session.add(
                        PackageManifestItem(
                            folder_name=folder_name,
                            source_name=source_name,
                            sequence=sequence,
                            osu_filename=osu_filenames[source_name],
                            active=True,
                        )
                    )
                session.commit()
        except Exception:
            self._restore_legacy_folders(mappings)
            raise

        self.legacy_order_path.unlink()

    def _migrate_legacy_difficulty_files(self) -> None:
        files_to_delete: list[Path] = []
        with Session(self.engine) as session:
            items = list(session.exec(select(PackageManifestItem)))
            for item in items:
                difficulty_path = (
                    self.target_root / item.folder_name / LEGACY_DIFFICULTY_FILENAME
                )
                if not difficulty_path.is_file():
                    continue
                if item.difficulty_value is None:
                    try:
                        item.difficulty_value = float(
                            difficulty_path.read_text(encoding="utf-8").strip()
                        )
                    except ValueError:
                        continue
                files_to_delete.append(difficulty_path)
            session.commit()

        for difficulty_path in files_to_delete:
            difficulty_path.unlink(missing_ok=True)

    def export_table(self, destination: Path | None = None) -> Path:
        table_path = destination or self.table_path
        table_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = table_path.with_name(f".{table_path.name}.tmp")
        with temp_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("编号", "谱面名称"))
            for item in self._all_items():
                if not item.active:
                    continue
                writer.writerow((item.folder_name, item.source_name))
        temp_path.replace(table_path)
        return table_path

    def replace(self, entries: list[ManifestEntry]) -> dict[str, str]:
        source_names = [self._normalize_source_name(entry.source_name) for entry in entries]
        if len(source_names) != len(set(source_names)):
            raise ValueError("manifest 输入中存在重复 source_name")

        with Session(self.engine) as session:
            existing = list(session.exec(select(PackageManifestItem)))
            by_source = {item.source_name: item for item in existing}
            next_number = self._next_folder_number(existing)

            for item in existing:
                item.active = False

            result: dict[str, str] = {}
            for sequence, entry in enumerate(entries, start=1):
                source_name = self._normalize_source_name(entry.source_name)
                item = by_source.get(source_name)
                if item is None:
                    item = PackageManifestItem(
                        folder_name=self._folder_name(next_number),
                        source_name=source_name,
                        sequence=sequence,
                    )
                    next_number += 1
                    session.add(item)

                item.sequence = sequence
                item.osu_filename = entry.osu_filename
                item.source_osz_name = entry.source_osz_name
                item.source_mtime_ns = entry.source_mtime_ns
                item.active = True
                result[source_name] = item.folder_name

            session.commit()
        self.export_table()
        return result

    def read_folder_names(self) -> list[str]:
        with Session(self.engine) as session:
            statement = (
                select(PackageManifestItem)
                .where(PackageManifestItem.active == True)  # noqa: E712
                .order_by(PackageManifestItem.sequence)
            )
            return [item.folder_name for item in session.exec(statement)]

    def read_all_folder_names(self) -> list[str]:
        return [item.folder_name for item in self._all_items()]

    def source_name_for(self, folder_name: str) -> str | None:
        with Session(self.engine) as session:
            statement = select(PackageManifestItem).where(
                PackageManifestItem.folder_name == folder_name
            )
            item = session.exec(statement).first()
            return item.source_name if item is not None else None

    def set_difficulty(self, folder_name: str, difficulty_value: float) -> None:
        with Session(self.engine) as session:
            statement = select(PackageManifestItem).where(
                PackageManifestItem.folder_name == folder_name,
                PackageManifestItem.active == True,  # noqa: E712
            )
            item = session.exec(statement).first()
            if item is None:
                raise KeyError(f"manifest 中不存在启用项: {folder_name}")
            item.difficulty_value = float(difficulty_value)
            session.commit()

        legacy_path = self.target_root / folder_name / LEGACY_DIFFICULTY_FILENAME
        legacy_path.unlink(missing_ok=True)

    def difficulty_for(self, folder_name: str) -> float | None:
        with Session(self.engine) as session:
            statement = select(PackageManifestItem).where(
                PackageManifestItem.folder_name == folder_name,
                PackageManifestItem.active == True,  # noqa: E712
            )
            item = session.exec(statement).first()
            return item.difficulty_value if item is not None else None

    def save_beatmap_data(
        self,
        folder_name: str,
        *,
        osu_filename: str,
        source_mtime_ns: int,
        schema_version: int,
        payload: dict[str, object],
    ) -> None:
        if not self.is_active(folder_name):
            raise KeyError(f"manifest 中不存在启用项: {folder_name}")
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with Session(self.engine) as session:
            statement = select(BeatmapDataRecord).where(
                BeatmapDataRecord.folder_name == folder_name
            )
            record = session.exec(statement).first()
            if record is None:
                record = BeatmapDataRecord(
                    folder_name=folder_name,
                    osu_filename=osu_filename,
                    source_mtime_ns=source_mtime_ns,
                    schema_version=schema_version,
                    payload_json=payload_json,
                    updated_at="",
                )
                session.add(record)
            record.osu_filename = osu_filename
            record.source_mtime_ns = source_mtime_ns
            record.schema_version = schema_version
            record.payload_json = payload_json
            record.updated_at = datetime.now().isoformat(timespec="seconds")
            session.commit()

    def beatmap_data_for(
        self,
        folder_name: str,
    ) -> tuple[str, int, int, dict[str, object]] | None:
        with Session(self.engine) as session:
            statement = select(BeatmapDataRecord).where(
                BeatmapDataRecord.folder_name == folder_name
            )
            record = session.exec(statement).first()
            if record is None:
                return None
            payload = json.loads(record.payload_json)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{folder_name} 的谱面缓存根节点必须是对象"
                )
            return (
                record.osu_filename,
                record.source_mtime_ns,
                record.schema_version,
                payload,
            )

    def is_active(self, folder_name: str) -> bool:
        return folder_name in set(self.read_folder_names())


class ManifestFolderWalker:
    def __init__(
        self,
        target_root: str,
        manifest_filename: str = MANIFEST_DB_FILENAME,
    ):
        self.target_root = Path(target_root)
        if not self.target_root.exists():
            raise FileNotFoundError(f"目录不存在: {self.target_root}")
        self.manifest = PackageManifest(
            target_root=str(self.target_root),
            manifest_filename=manifest_filename,
        )

    def read_folder_names(self) -> list[str]:
        return self.manifest.read_folder_names()

    def source_name_for(self, folder_name: str) -> str | None:
        return self.manifest.source_name_for(folder_name)
