from __future__ import annotations

import re
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


class DocumentationLinkTests(unittest.TestCase):
    def test_markdown_relative_links_exist(self) -> None:
        checked_roots = (
            REPO_ROOT / "README.md",
            REPO_ROOT / "docs",
            REPO_ROOT / "project_index",
            REPO_ROOT / "src" / "start" / "README.md",
            REPO_ROOT / "src" / "before_traning" / "docs",
            REPO_ROOT / "src" / "traning" / "docs",
            REPO_ROOT / "src" / "visualization" / "docs",
        )
        failures: list[str] = []
        for path in _iter_markdown(checked_roots):
            for target in _local_markdown_targets(path):
                resolved = (path.parent / target).resolve()
                if not resolved.exists():
                    failures.append(f"{path.relative_to(REPO_ROOT)} -> {target}")
        self.assertEqual(failures, [])


def _iter_markdown(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".md":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.md")))
    return tuple(files)


def _local_markdown_targets(path: Path) -> tuple[str, ...]:
    targets: list[str] = []
    for match in MARKDOWN_LINK_RE.finditer(path.read_text(encoding="utf-8")):
        raw = match.group(1).strip()
        if raw.startswith(("http://", "https://", "mailto:", "#")):
            continue
        raw = raw.split("#", 1)[0].strip()
        if not raw:
            continue
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1]
        targets.append(raw)
    return tuple(targets)


if __name__ == "__main__":
    unittest.main()
