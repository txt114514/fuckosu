from __future__ import annotations

import fcntl
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_NAME_PATTERN = re.compile(r"^output_(\d+)__")


@dataclass(frozen=True)
class OutputIdentity:
    sequence: int
    created_at_utc: str
    timestamp_slug: str

    @property
    def prefix(self) -> str:
        return f"output_{self.sequence:06d}__{self.timestamp_slug}"


def _read_counter(path: Path) -> int:
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (FileNotFoundError, OSError, ValueError):
        return 0


def _existing_max_sequence(output_root: Path) -> int:
    maximum = 0
    for path in output_root.iterdir():
        match = OUTPUT_NAME_PATTERN.match(path.name)
        if match is not None:
            maximum = max(maximum, int(match.group(1)))
    return maximum


def allocate_output_identity(output_root: Path) -> OutputIdentity:
    output_root.mkdir(parents=True, exist_ok=True)
    counter_path = output_root / ".output_counter"
    lock_path = output_root / ".output_counter.lock"
    with lock_path.open("a+", encoding="ascii") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            current = max(
                _read_counter(counter_path),
                _existing_max_sequence(output_root),
            )
            sequence = current + 1
            counter_path.write_text(f"{sequence}\n", encoding="ascii")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    created_at = datetime.now(timezone.utc)
    return OutputIdentity(
        sequence=sequence,
        created_at_utc=created_at.isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z"),
        timestamp_slug=created_at.strftime("%Y%m%dT%H%M%S_%fZ"),
    )


__all__ = ["OutputIdentity", "allocate_output_identity"]
