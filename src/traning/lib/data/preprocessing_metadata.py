from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def load_preprocessing_metadata(
    dataset_root: Path,
    item_name: str,
) -> dict[str, Any] | None:
    status_db = _status_db_for_dataset_root(dataset_root)
    if status_db is None:
        return None
    try:
        with sqlite3.connect(status_db) as connection:
            row = connection.execute(
                """
                SELECT detail_json
                FROM process_step_status
                WHERE folder_name = ? AND step = 'video_processed' AND done = 1
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """,
                (item_name,),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None or not row[0]:
        return None
    try:
        detail = json.loads(row[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(detail, dict):
        return None
    return {
        "source": "process_status.video_processed",
        "status_db": str(status_db),
        "source_size": {
            "width": detail.get("video_width"),
            "height": detail.get("video_height"),
        },
        "crop_rect": {
            "left": detail.get("crop_left"),
            "top": detail.get("crop_top"),
            "width": detail.get("crop_width"),
            "height": detail.get("crop_height"),
        },
        "raw": detail,
    }


def _status_db_for_dataset_root(dataset_root: Path) -> Path | None:
    roots = [
        dataset_root.parent / "match-completed_package" / ".process_status.sqlite",
        dataset_root.parent / ".process_status.sqlite",
    ]
    for path in roots:
        if path.is_file():
            return path
    return None


__all__ = ["load_preprocessing_metadata"]
