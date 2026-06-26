from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from package.coordinates import COORDINATE_TRANSFORM_VERSION


CONFIGURATION_VERSION = "settings-schema-v2"
EVALUATION_DATASET_VERSION = "fixed-eval-split-v1"


@dataclass(frozen=True)
class CodeVersion:
    commit: str
    dirty: bool
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {"commit": self.commit, "dirty": self.dirty, "source": self.source}


def collect_code_version(repo_root: Path | None = None) -> CodeVersion:
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return CodeVersion(commit="unknown", dirty=True, source="git-unavailable")
    return CodeVersion(commit=commit, dirty=bool(status.strip()), source="git")


def dataset_version(settings: Any) -> str:
    data = getattr(settings, "data_input", None)
    payload = {
        "dataset_root": str(getattr(data, "dataset_root", "")),
        "split_manifest_path": str(getattr(data, "split_manifest_path", "")),
        "dimensions": tuple(getattr(data, "dimensions", ())),
        "categories": tuple(getattr(data, "categories", ())),
        "include_items": tuple(getattr(data, "include_items", ())),
        "exclude_items": tuple(getattr(data, "exclude_items", ())),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return f"dataset-{digest}"


def version_manifest(settings: Any) -> dict[str, Any]:
    code = collect_code_version().as_dict()
    return {
        "dataset_version": dataset_version(settings),
        "evaluation_dataset_version": EVALUATION_DATASET_VERSION,
        "configuration_version": CONFIGURATION_VERSION,
        "transform_version": COORDINATE_TRANSFORM_VERSION,
        "code_version": code,
    }


def ensure_compatible_versions(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    override: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    keys = (
        "dataset_version",
        "evaluation_dataset_version",
        "score_version",
        "candidate_cache_version",
        "transform_version",
        "configuration_version",
    )
    mismatches = tuple(
        key for key in keys if left.get(key) is not None and right.get(key) is not None and left.get(key) != right.get(key)
    )
    if mismatches and not override:
        return False, mismatches
    return True, mismatches


__all__ = [
    "CONFIGURATION_VERSION",
    "EVALUATION_DATASET_VERSION",
    "collect_code_version",
    "dataset_version",
    "ensure_compatible_versions",
    "version_manifest",
]
