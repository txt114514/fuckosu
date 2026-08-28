"""验证 Phase 1 原子 IO、完整性摘要和确定性 seed。"""

from __future__ import annotations

import hashlib
import json
import os
import random

import numpy as np
import pytest
import torch

from traning.infrastructure import (
    AtomicWriteError,
    SchemaMismatchError,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    read_json_object,
    seed_everything,
    sha256_file,
)


def test_atomic_writers_publish_complete_payloads(tmp_path) -> None:
    """四种发布入口都只暴露完整最终文件。"""

    binary_path = tmp_path / "payload.bin"
    text_path = tmp_path / "payload.txt"
    json_path = tmp_path / "payload.json"
    jsonl_path = tmp_path / "payload.jsonl"
    atomic_write_bytes(binary_path, b"abc\x00def")
    atomic_write_text(text_path, "中文\n")
    atomic_write_json(json_path, {"schema_version": 1, "ok": True})
    atomic_write_jsonl(jsonl_path, ({"row": 1}, {"row": 2}))
    assert binary_path.read_bytes() == b"abc\x00def"
    assert text_path.read_text(encoding="utf-8") == "中文\n"
    assert read_json_object(json_path) == {"schema_version": 1, "ok": True}
    assert [json.loads(line) for line in jsonl_path.read_text().splitlines()] == [
        {"row": 1},
        {"row": 2},
    ]
    assert sha256_file(binary_path) == hashlib.sha256(b"abc\x00def").hexdigest()


def test_failed_replace_preserves_previous_file_and_cleans_temp(
    tmp_path, monkeypatch
) -> None:
    """发布失败不能破坏旧版本，也不能遗留半成品。"""

    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")

    def fail_replace(
        source: os.PathLike[str] | str, destination: os.PathLike[str] | str
    ) -> None:
        """模拟原子替换失败，以验证旧文件仍可恢复。"""

        del source, destination
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises((AtomicWriteError, OSError)):
        atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.iterdir()) == [target]


def test_read_json_object_rejects_non_object_root(tmp_path) -> None:
    """对象 schema 不接受 JSON array 的 silent coercion。"""

    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises((SchemaMismatchError, TypeError, ValueError)):
        read_json_object(path)


def test_seed_everything_repeats_python_numpy_and_torch() -> None:
    """数据处理使用同一 seed 时得到相同随机序列。"""

    seed_everything(20260824)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    seed_everything(20260824)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    assert second == pytest.approx(first)
