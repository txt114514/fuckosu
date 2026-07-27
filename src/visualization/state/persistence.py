"""提供仪表盘快照的原子写入和事件日志的追加持久化。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: Any) -> None:
    """先完整落盘临时文件，再以同目录替换方式发布 JSON 快照。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        # flush 只刷新 Python 缓冲；fsync 保证替换前内容已经交给文件系统。
        os.fsync(handle.fileno())
    # 阅读器只会看到旧快照或完整新快照，不会读到写了一半的 JSON。
    tmp.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    """追加一条已完成换行的事件记录，并在返回前同步到文件系统。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value
