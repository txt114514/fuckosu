from __future__ import annotations


def format_sequence_name(
    prefix: str,
    sequence: int,
    *,
    width: int = 6,
) -> str:
    prefix = prefix.strip()
    if not prefix:
        raise ValueError("sequence prefix 不能为空")
    if sequence <= 0:
        raise ValueError("sequence 必须为正整数")
    if width <= 0:
        raise ValueError("sequence width 必须为正整数")
    return f"{prefix}{sequence:0{width}d}"


__all__ = ["format_sequence_name"]
