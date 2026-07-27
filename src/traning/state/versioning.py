"""生成数据、配置、代码与坐标方程版本，并判断产物能否安全复用。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from package.coordinates import (
    COORDINATE_TRANSFORM_VERSION,
    coordinate_transform_fingerprint,
)


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
    """读取 Git 提交和 dirty 状态；Git 不可用时返回显式降级标记。"""

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
    """摘要会改变训练样本成员关系的数据输入选择字段。"""

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
    """生成可写入缓存、检查点和图集的统一版本清单。"""

    code = collect_code_version().as_dict()
    return {
        "dataset_version": dataset_version(settings),
        "evaluation_dataset_version": EVALUATION_DATASET_VERSION,
        "configuration_version": CONFIGURATION_VERSION,
        "transform_version": COORDINATE_TRANSFORM_VERSION,
        "transform_fingerprint": _transform_fingerprint(settings),
        "code_version": code,
    }


def ensure_compatible_versions(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    override: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    """比较两个版本清单，并返回是否允许复用及所有不兼容字段。

    旧清单允许缺少一般的可选版本字段，以保持历史产物可读；坐标变换指纹例外：
    任一侧已有指纹时，另一侧缺失也必须视为失配，否则旧产物可能在新方程下被静默复用。
    ``override`` 只放行调用方继续执行，不会隐藏检测到的失配字段。
    """

    keys = (
        "dataset_version",
        "evaluation_dataset_version",
        "score_version",
        "candidate_cache_version",
        "transform_version",
        "transform_fingerprint",
        "configuration_version",
    )
    mismatches = tuple(
        key
        for key in keys
        if (
            left.get(key) is not None
            and right.get(key) is not None
            and left.get(key) != right.get(key)
        )
        or (
            key == "transform_fingerprint"
            and (left.get(key) is not None or right.get(key) is not None)
            and left.get(key) != right.get(key)
        )
    )
    if mismatches and not override:
        return False, mismatches
    return True, mismatches


def _transform_fingerprint(settings: Any) -> str:
    """为完整坐标方程和训练帧尺寸生成稳定指纹。

    同一仿射系数用于不同宽高的训练帧时，模型归一化坐标代表的像素位置不同，因此宽高
    也是坐标契约的一部分，必须参与摘要，而不能只依赖变换模式的版本常量。
    """

    transform = getattr(settings, "coordinate_transform", None)
    if hasattr(transform, "model_dump"):
        transform_payload = transform.model_dump(mode="json")
    elif isinstance(transform, Mapping):
        transform_payload = dict(transform)
    else:
        transform_payload = {"value": transform}
    input_settings = getattr(settings, "input", None)
    return coordinate_transform_fingerprint(
        {
            "coordinate_transform": transform_payload,
            "training_frame": {
                "width": getattr(input_settings, "width", None),
                "height": getattr(input_settings, "height", None),
            },
        }
    )


__all__ = [
    "CONFIGURATION_VERSION",
    "EVALUATION_DATASET_VERSION",
    "collect_code_version",
    "dataset_version",
    "ensure_compatible_versions",
    "version_manifest",
]
