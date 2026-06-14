from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from sqlmodel import Session, create_engine, select

from before_traning.state.segment_schema import (
    SEGMENT_DB_FILENAME,
    SegmentDatasetItem,
)


SEGMENT_TABLE_FILENAME = "segments.csv"
SEGMENT_MANIFEST_FILENAME = "manifest.csv"
SEGMENT_BEATMAP_FILENAME = "beatmap.json"
SEGMENT_VIDEO_FILENAME = "video.mp4"
LONG_SEQUENCE_DIRECTORY = "long_sequence"
SEGMENT_SCHEMA_VERSION = 10
SEGMENT_TABLE_FIELDS = (
    "segment_id",
    "dataset_dimension",
    "category",
    "source_plan_count",
    "hp_drain_rate",
    "circle_size",
    "circle_radius",
    "overall_difficulty",
    "approach_rate",
    "approach_preempt_ms",
    "slider_multiplier",
    "slider_tick_rate",
    "stack_leniency",
    "approach_preempt_ratio",
    "overlap_merge_window_ms",
    "min_circle_overlap_ratio",
    "priority_merge_window_ms",
    "use_priority_merge",
    "configured_post_context_seconds",
    "build_long_sequences",
    "long_sequence_continuity_window_ms",
    "long_sequence_max_objects",
    "long_sequence_max_duration_seconds",
    "segment_directory",
    "video_path",
    "beatmap_path",
    "hit_start_ms",
    "hit_end_ms",
    "clip_source_start_ms",
    "clip_source_end_ms",
    "clip_start_seconds",
    "clip_end_seconds",
    "pre_context_seconds",
    "post_context_seconds",
    "object_count",
    "object_indexes",
    "object_types",
    "object_start_times_ms",
    "object_end_times_ms",
    "source_object_start_times_ms",
    "source_object_end_times_ms",
)


class SegmentDatasetManifest:
    def __init__(
        self,
        segment_root: Path,
        output_directories: Iterable[str],
        *,
        db_filename: str = SEGMENT_DB_FILENAME,
    ):
        self.segment_root = Path(segment_root)
        self.segment_root.mkdir(parents=True, exist_ok=True)
        self.output_directories = tuple(output_directories)
        self.db_path = self.segment_root / db_filename
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        SegmentDatasetItem.__table__.create(self.engine, checkfirst=True)

    def _records(self, folder_name: str) -> list[SegmentDatasetItem]:
        with Session(self.engine) as session:
            statement = (
                select(SegmentDatasetItem)
                .where(SegmentDatasetItem.folder_name == folder_name)
                .order_by(SegmentDatasetItem.sequence)
            )
            return list(session.exec(statement))

    def read_rows(self, folder_name: str) -> list[dict[str, str]]:
        return [
            json.loads(record.row_json)
            for record in self._records(folder_name)
        ]

    def replace_folder(
        self,
        folder_name: str,
        rows: list[dict[str, object]],
    ) -> None:
        normalized_rows = [
            {key: str(row[key]) for key in SEGMENT_TABLE_FIELDS}
            for row in rows
        ]
        with Session(self.engine) as session:
            statement = select(SegmentDatasetItem).where(
                SegmentDatasetItem.folder_name == folder_name
            )
            for record in session.exec(statement):
                session.delete(record)
            for sequence, row in enumerate(normalized_rows, start=1):
                session.add(
                    SegmentDatasetItem(
                        folder_name=folder_name,
                        segment_id=row["segment_id"],
                        sequence=sequence,
                        dataset_dimension=row["dataset_dimension"],
                        category=row["category"],
                        video_path=row["video_path"],
                        beatmap_path=row["beatmap_path"],
                        row_json=json.dumps(
                            row,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    )
                )
            session.commit()

    def write_table(
        self,
        output_directory: Path,
        rows: list[dict[str, object]],
    ) -> Path:
        table_path = output_directory / SEGMENT_TABLE_FILENAME
        with table_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=SEGMENT_TABLE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return table_path

    def export_table(self, folder_name: str) -> Path:
        output_directory = self.segment_root / folder_name
        return self.write_table(
            output_directory,
            self.read_rows(folder_name),
        )

    def import_existing_table(self, folder_name: str) -> bool:
        table_path = (
            self.segment_root / folder_name / SEGMENT_TABLE_FILENAME
        )
        if not table_path.is_file():
            return False
        with table_path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != SEGMENT_TABLE_FIELDS:
                return False
            rows = list(reader)
        if not rows:
            return False
        normalized_rows = [
            {key: str(row[key]) for key in SEGMENT_TABLE_FIELDS}
            for row in rows
        ]
        if self.read_rows(folder_name) == normalized_rows:
            return False
        self.replace_folder(folder_name, rows)
        return True

    def output_complete(self, folder_name: str) -> bool:
        output_directory = self.segment_root / folder_name
        if (
            not output_directory.is_dir()
            or not all(
                (output_directory / directory).is_dir()
                for directory in self.output_directories
            )
        ):
            return False
        self.import_existing_table(folder_name)
        rows = self.read_rows(folder_name)
        return bool(rows) and all(
            (output_directory / row["video_path"]).is_file()
            and (output_directory / row["beatmap_path"]).is_file()
            for row in rows
        )


def write_json_file(output_path: Path, payload: dict[str, object]) -> None:
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = [
    "LONG_SEQUENCE_DIRECTORY",
    "SEGMENT_BEATMAP_FILENAME",
    "SEGMENT_MANIFEST_FILENAME",
    "SEGMENT_SCHEMA_VERSION",
    "SEGMENT_TABLE_FIELDS",
    "SEGMENT_TABLE_FILENAME",
    "SEGMENT_VIDEO_FILENAME",
    "SegmentDatasetManifest",
    "write_json_file",
]
