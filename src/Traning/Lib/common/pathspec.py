from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pathspec


def suffix_pattern(suffix: str) -> str:
    suffix = suffix.strip().lower()
    if not suffix:
        raise ValueError("suffix 不能为空")
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return f"*{suffix}"


def suffix_patterns(suffixes: Iterable[str]) -> tuple[str, ...]:
    return tuple(suffix_pattern(suffix) for suffix in suffixes)


def gitwildmatch_spec(patterns: Iterable[str]) -> pathspec.PathSpec:
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def suffix_spec(suffixes: Iterable[str]) -> pathspec.PathSpec:
    return gitwildmatch_spec(suffix_patterns(suffixes))


def matches_name(spec: pathspec.PathSpec, path: Path | str) -> bool:
    return spec.match_file(Path(path).name.lower())


def filter_files(paths: Iterable[Path], spec: pathspec.PathSpec) -> list[Path]:
    return [path for path in paths if path.is_file() and matches_name(spec, path)]
