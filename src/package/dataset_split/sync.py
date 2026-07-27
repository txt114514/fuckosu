"""发现数据 item，并以稳定、可复现的策略同步划分清单。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from package.dataset_split.models import (
    DATASET_SPLIT_SCHEMA_VERSION,
    SPLIT_ORDER,
    DatasetSplit,
    DatasetSplitItem,
    DatasetSplitManifest,
    DatasetSplitSyncResult,
    SplitRatios,
)


DEFAULT_SPLIT_RATIOS = SplitRatios()
DEFAULT_MANIFEST_RELATIVE_PATH = Path("splits") / "dataset_split_manifest.json"


def default_split_manifest_path(dataset_root: Path) -> Path:
    return dataset_root.parent / DEFAULT_MANIFEST_RELATIVE_PATH


def load_split_manifest(path: Path) -> DatasetSplitManifest | None:
    """读取已存在清单；文件不存在返回 ``None``，损坏内容则显式报错。"""

    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"split manifest root must be an object: {path}")
    return DatasetSplitManifest.from_mapping(payload)


def sync_dataset_split_manifest(
    dataset_root: Path,
    *,
    manifest_path: Path | None = None,
    seed: int = 2026,
    ratios: SplitRatios = DEFAULT_SPLIT_RATIOS,
    bootstrap_splits: Mapping[str, DatasetSplit] | None = None,
    allow_test_growth: bool = False,
    dry_run: bool = False,
) -> DatasetSplitSyncResult:
    """增量加入新 item，并保留所有历史 item 的 split 归属。"""

    target_path = manifest_path or default_split_manifest_path(dataset_root)
    existing = load_split_manifest(target_path)
    discovered_counts = discover_dataset_items(dataset_root)
    normalized_ratios = ratios.normalized()
    created = existing is None
    manifest = existing or DatasetSplitManifest(
        seed=seed,
        ratios=normalized_ratios,
        items={},
        allow_test_growth=allow_test_growth,
    )
    manifest = replace(
        manifest,
        ratios=normalized_ratios,
        allow_test_growth=allow_test_growth,
    )

    updated_items = dict(manifest.items)
    bootstrap = dict(bootstrap_splits or {})
    new_items: list[DatasetSplitItem] = []
    # 只给首次发现的 item 分配 split；历史归属永不因比例变化而重排，
    # 从而避免同一谱面跨实验漂移并污染固定评估集。
    for item_name in _stable_new_item_order(
        discovered_counts.keys(),
        known_items=updated_items.keys(),
        seed=manifest.seed,
    ):
        split = bootstrap.get(item_name)
        reason = "bootstrap_config" if split is not None else "incremental_balance"
        if split is None:
            split = _select_split_for_new_item(
                updated_items,
                ratios=manifest.ratios,
                allow_test_growth=manifest.allow_test_growth,
            )
        item = DatasetSplitItem(
            item_name=item_name,
            split=split,
            segment_count=discovered_counts[item_name],
            assigned_at_utc=_utc_now(),
            assignment_reason=reason,
        )
        updated_items[item_name] = item
        new_items.append(item)

    refreshed_items = {
        item_name: (
            replace(item, segment_count=discovered_counts.get(item_name, item.segment_count))
            if item_name in discovered_counts and item.segment_count != discovered_counts[item_name]
            else item
        )
        for item_name, item in updated_items.items()
    }
    refreshed = DatasetSplitManifest(
        schema_version=DATASET_SPLIT_SCHEMA_VERSION,
        seed=manifest.seed,
        unit=manifest.unit,
        ratios=manifest.ratios,
        allow_test_growth=manifest.allow_test_growth,
        items=refreshed_items,
    )
    changed = created or bool(new_items) or refreshed.as_dict() != manifest.as_dict()
    if changed and not dry_run:
        _write_manifest(target_path, refreshed)
    return DatasetSplitSyncResult(
        manifest_path=target_path,
        created=created,
        changed=changed,
        dry_run=dry_run,
        new_items=tuple(new_items),
        manifest=refreshed,
    )


def discover_dataset_items(dataset_root: Path) -> dict[str, int]:
    """按 ``beatmap.json`` 统计 item 的 segment 数，不读取标注内容。"""

    counts: dict[str, int] = {}
    if not dataset_root.is_dir():
        return counts
    for annotation_path in dataset_root.glob("item_*/*/*/beatmap.json"):
        item_name = annotation_path.parents[2].name
        counts[item_name] = counts.get(item_name, 0) + 1
    return dict(sorted(counts.items()))


def _select_split_for_new_item(
    items: Mapping[str, DatasetSplitItem],
    *,
    ratios: SplitRatios,
    allow_test_growth: bool,
) -> DatasetSplit:
    candidates: tuple[DatasetSplit, ...] = (
        SPLIT_ORDER if allow_test_growth else ("train", "validation")
    )
    counts = {split: 0 for split in SPLIT_ORDER}
    for item in items.values():
        counts[item.split] += 1
    total_after = sum(counts.values()) + 1
    ratio_map = ratios.as_dict()
    deficits = {
        split: ratio_map[split] * total_after - counts[split]
        for split in candidates
    }
    # 选择“目标数量 - 当前数量”缺口最大的 split；后续排序项保证平局可复现。
    return max(candidates, key=lambda split: (deficits[split], ratio_map[split], split))


def _stable_new_item_order(
    item_names,
    *,
    known_items,
    seed: int,
) -> tuple[str, ...]:
    known = set(known_items)
    # 先按 seed+名称哈希再按名称排序，结果不受文件系统遍历顺序影响。
    return tuple(
        sorted(
            (item for item in item_names if item not in known),
            key=lambda item: (_stable_hash(seed, item), item),
        )
    )


def _stable_hash(seed: int, item_name: str) -> str:
    payload = f"{seed}:{item_name}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(path: Path, manifest: DatasetSplitManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件必须与目标同目录，replace 才能在同一文件系统内原子提交；
    # 进程若在提交前中断，旧 manifest 仍保持完整。
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(_json_ready(manifest.as_dict()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_ready(item) for item in value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "DEFAULT_SPLIT_RATIOS",
    "default_split_manifest_path",
    "discover_dataset_items",
    "load_split_manifest",
    "sync_dataset_split_manifest",
]
