"""提供严格 JSON 边界和可持久化的原子文件发布。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import BinaryIO, cast

from traning.contracts.common import JSONObject, JSONScalar, JSONValue

from .errors import AtomicWriteError, IntegrityError, SchemaMismatchError


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """将字节完整落盘后，以同目录原子替换发布。"""

    if not isinstance(payload, bytes):
        raise TypeError("payload 必须是 bytes")

    def write_payload(handle: BinaryIO) -> None:
        """把已校验字节写入统一原子发布器提供的临时文件。"""

        handle.write(payload)

    _atomic_publish(path, write_payload)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """按指定编码原子发布文本。"""

    if not isinstance(text, str):
        raise TypeError("text 必须是 str")
    try:
        payload = text.encode(encoding)
    except (LookupError, UnicodeEncodeError) as exc:
        raise AtomicWriteError(f"无法编码待发布文本：{path}") from exc
    atomic_write_bytes(path, payload)


def atomic_write_json(path: Path, payload: JSONValue) -> None:
    """以 canonical 紧凑格式原子发布单个 JSON 值。"""

    encoded = _encode_json(payload, path=path)
    atomic_write_bytes(path, encoded + b"\n")


def atomic_write_jsonl(path: Path, records: Iterable[JSONObject]) -> None:
    """全量原子发布 JSONL；任何一条失败都不会暴露半成品。"""

    def write_records(handle: BinaryIO) -> None:
        """逐行编码严格 JSON object，任一失败即放弃整次发布。"""

        for line_number, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise SchemaMismatchError(
                    f"JSONL 第 {line_number} 条记录必须是 JSON object"
                )
            handle.write(
                _encode_json(record, path=path, context=f"第 {line_number} 条记录")
            )
            handle.write(b"\n")

    _atomic_publish(path, write_records)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件的十六进制 SHA-256，不把大文件整体读入内存。"""

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise IntegrityError(f"无法读取文件并计算 SHA-256：{path}") from exc
    return digest.hexdigest()


def read_json_object(path: Path) -> JSONObject:
    """读取严格 JSON object，拒绝损坏文本、重复键和非有限浮点数。"""

    try:
        with path.open("r", encoding="utf-8") as handle:
            decoded: object = json.load(
                handle,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_non_finite_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"无法读取完整 JSON：{path}") from exc

    if not isinstance(decoded, dict):
        raise SchemaMismatchError(f"JSON 根节点必须是 object：{path}")
    _validate_json_value(decoded, context=str(path))
    return cast(JSONObject, decoded)


def _atomic_publish(path: Path, writer: Callable[[BinaryIO], None]) -> None:
    """执行 write→flush→fsync→replace→目录 fsync 的统一发布协议。"""

    temporary_path: Path | None = None
    replaced = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        replaced = True
        _fsync_directory(path.parent)
    except Exception as exc:
        cleanup_error: OSError | None = None
        if temporary_path is not None and not replaced:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as unlink_exc:
                cleanup_error = unlink_exc
        if isinstance(exc, AtomicWriteError):
            error = exc
        else:
            error = AtomicWriteError(f"原子发布失败：{path}")
        if cleanup_error is not None:
            error.add_note(f"临时文件清理也失败：{cleanup_error}")
        raise error from exc


def _fsync_directory(directory: Path) -> None:
    """同步目录项，保证 replace 的命名变更也进入持久化边界。"""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _encode_json(
    payload: JSONValue,
    *,
    path: Path,
    context: str = "JSON payload",
) -> bytes:
    _validate_json_value(payload, context=context)
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise AtomicWriteError(f"无法序列化 {context}：{path}") from exc


def _validate_json_value(value: object, *, context: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaMismatchError(f"{context} 含有非有限浮点数")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, context=f"{context}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaMismatchError(f"{context} 含有非字符串 object key")
            _validate_json_value(item, context=f"{context}.{key}")
        return
    raise SchemaMismatchError(f"{context} 含有非 JSON 类型：{type(value).__name__}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"JSON object 含有重复键：{key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> object:
    raise IntegrityError(f"JSON 含有非标准数值常量：{value}")


__all__ = (
    "JSONObject",
    "JSONScalar",
    "JSONValue",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_jsonl",
    "atomic_write_text",
    "read_json_object",
    "sha256_file",
)
