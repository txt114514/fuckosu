"""从预处理状态库恢复原视频尺寸和裁剪矩形元数据。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

from traning.contracts.common import JSONObject


def load_preprocessing_metadata(
    dataset_root: Path,
    item_name: str,
) -> JSONObject | None:
    """读取最近一次成功视频预处理记录；缺失或损坏时返回 ``None``。"""

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
    typed_detail = cast(JSONObject, detail)
    # 保留 raw 便于审计，同时提升常用尺寸/裁剪字段为稳定的坐标链输入。
    return {
        "source": "process_status.video_processed",
        "status_db": str(status_db),
        "source_size": {
            "width": typed_detail.get("video_width"),
            "height": typed_detail.get("video_height"),
        },
        "crop_rect": {
            "left": typed_detail.get("crop_left"),
            "top": typed_detail.get("crop_top"),
            "width": typed_detail.get("crop_width"),
            "height": typed_detail.get("crop_height"),
        },
        "raw": typed_detail,
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
