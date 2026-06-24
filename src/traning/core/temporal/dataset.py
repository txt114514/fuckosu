from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from traning.core.decision import CANDIDATE_CACHE_VERSION
from traning.lib.models import OBJECT_TYPE_NAMES


ACTION_NAMES: tuple[str, ...] = ("no_op", "press", "hold", "release")
NO_OP_ACTION_ID = 0
PRESS_ACTION_ID = 1
HOLD_ACTION_ID = 2
RELEASE_ACTION_ID = 3
IGNORE_CANDIDATE_ID = -100


@dataclass(frozen=True)
class TemporalFeatureSpec:
    candidate_slots: int
    embedding_dim: int
    object_types: tuple[str, ...] = OBJECT_TYPE_NAMES

    def __post_init__(self) -> None:
        if self.candidate_slots <= 0:
            raise ValueError("candidate_slots must be positive")
        if self.embedding_dim < 0:
            raise ValueError("embedding_dim must be nonnegative")
        if not self.object_types:
            raise ValueError("object_types must not be empty")

    @property
    def candidate_feature_dim(self) -> int:
        return 13 + len(self.object_types) + self.embedding_dim

    @property
    def frame_feature_dim(self) -> int:
        return self.candidate_slots * self.candidate_feature_dim


@dataclass(frozen=True)
class TemporalWindow:
    features: torch.Tensor
    candidate_features: torch.Tensor
    candidate_mask: torch.Tensor
    frame_mask: torch.Tensor
    action_target: torch.Tensor
    selected_candidate_target: torch.Tensor
    xy_target: torch.Tensor
    time_offset_target: torch.Tensor
    candidate_ids: tuple[tuple[int | None, ...], ...]
    sample_keys: tuple[str | None, ...]
    frame_indices: tuple[int | None, ...]
    timestamps_ms: tuple[float | None, ...]
    target_strategy: str = "top_candidate_proxy"


class TemporalCandidateWindowDataset(Dataset[TemporalWindow]):
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        sequence_length: int,
        feature_spec: TemporalFeatureSpec,
        stride: int | None = None,
        drop_short: bool = False,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        selected_stride = stride or sequence_length
        if selected_stride <= 0:
            raise ValueError("stride must be positive")
        self.sequence_length = sequence_length
        self.feature_spec = feature_spec
        self.stride = selected_stride
        self.records = tuple(records)
        self._windows = tuple(
            self._build_windows(records=self.records, drop_short=drop_short)
        )
        if not self._windows:
            raise ValueError("temporal dataset must contain at least one window")

    @classmethod
    def from_cache_dir(
        cls,
        cache_dir: Path,
        *,
        sequence_length: int,
        candidate_slots: int,
        embedding_dim: int | None = None,
        stride: int | None = None,
        drop_short: bool = False,
    ) -> TemporalCandidateWindowDataset:
        records = load_candidate_cache_records(cache_dir)
        selected_embedding_dim = (
            _infer_embedding_dim(records) if embedding_dim is None else embedding_dim
        )
        spec = TemporalFeatureSpec(
            candidate_slots=candidate_slots,
            embedding_dim=selected_embedding_dim,
        )
        return cls(
            records,
            sequence_length=sequence_length,
            feature_spec=spec,
            stride=stride,
            drop_short=drop_short,
        )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> TemporalWindow:
        return self._windows[index]

    def _build_windows(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        drop_short: bool,
    ) -> list[TemporalWindow]:
        windows: list[TemporalWindow] = []
        for group in _group_records_by_sample(records):
            if not group:
                continue
            for start in range(0, len(group), self.stride):
                chunk = group[start : start + self.sequence_length]
                if len(chunk) < self.sequence_length and drop_short:
                    continue
                windows.append(
                    _encode_window(
                        chunk,
                        sequence_length=self.sequence_length,
                        spec=self.feature_spec,
                    )
                )
                if start + self.sequence_length >= len(group):
                    break
        return windows


def load_candidate_cache_records(cache_dir: Path) -> tuple[dict[str, Any], ...]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"candidate cache manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != CANDIDATE_CACHE_VERSION:
        raise ValueError(
            "unsupported candidate cache version: "
            f"{manifest.get('version')!r}"
        )
    records_name = manifest.get("records")
    if not isinstance(records_name, str) or not records_name:
        raise ValueError("candidate cache manifest must contain records filename")
    records_path = cache_dir / records_name
    if not records_path.is_file():
        raise FileNotFoundError(f"candidate cache records missing: {records_path}")
    records: list[dict[str, Any]] = []
    lines = records_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("version") != CANDIDATE_CACHE_VERSION:
            raise ValueError(
                f"unsupported record version at line {line_number}: "
                f"{record.get('version')!r}"
            )
        records.append(record)
    if not records:
        raise ValueError("candidate cache records must not be empty")
    return tuple(records)


def _group_records_by_sample(
    records: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    groups: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    current_key: str | None = None
    for record in sorted(records, key=_record_sort_key):
        key = _optional_string(record.get("sample_key"))
        if current and key != current_key:
            groups.append(current)
            current = []
        current.append(record)
        current_key = key
    if current:
        groups.append(current)
    return groups


def _record_sort_key(record: Mapping[str, Any]) -> tuple[str, int, float]:
    sample_key = _optional_string(record.get("sample_key")) or ""
    frame_index = _optional_int(record.get("frame_index"))
    timestamp = _optional_float(record.get("timestamp_ms"))
    return (
        sample_key,
        frame_index if frame_index is not None else -1,
        timestamp if timestamp is not None else -1.0,
    )


def _encode_window(
    records: Sequence[Mapping[str, Any]],
    *,
    sequence_length: int,
    spec: TemporalFeatureSpec,
) -> TemporalWindow:
    candidate_features = torch.zeros(
        sequence_length,
        spec.candidate_slots,
        spec.candidate_feature_dim,
        dtype=torch.float32,
    )
    candidate_mask = torch.zeros(
        sequence_length,
        spec.candidate_slots,
        dtype=torch.bool,
    )
    frame_mask = torch.zeros(sequence_length, dtype=torch.bool)
    action_target = torch.full(
        (sequence_length,),
        NO_OP_ACTION_ID,
        dtype=torch.long,
    )
    selected_candidate = torch.full(
        (sequence_length,),
        IGNORE_CANDIDATE_ID,
        dtype=torch.long,
    )
    xy_target = torch.zeros(sequence_length, 2, dtype=torch.float32)
    time_offset = torch.zeros(sequence_length, 1, dtype=torch.float32)
    candidate_ids: list[tuple[int | None, ...]] = []
    sample_keys: list[str | None] = []
    frame_indices: list[int | None] = []
    timestamps: list[float | None] = []
    window_target_strategy = "top_candidate_proxy"
    for frame_slot, record in enumerate(records[:sequence_length]):
        frame_mask[frame_slot] = True
        sample_keys.append(_optional_string(record.get("sample_key")))
        frame_indices.append(_optional_int(record.get("frame_index")))
        timestamps.append(_optional_float(record.get("timestamp_ms")))
        encoded_candidates = _sorted_candidates(record)[: spec.candidate_slots]
        if encoded_candidates:
            action_target[frame_slot] = PRESS_ACTION_ID
            selected_candidate[frame_slot] = 0
        frame_candidate_ids: list[int | None] = []
        for candidate_slot, candidate in enumerate(encoded_candidates):
            candidate_features[frame_slot, candidate_slot] = _encode_candidate(
                candidate,
                record=record,
                spec=spec,
            )
            candidate_mask[frame_slot, candidate_slot] = True
            frame_candidate_ids.append(_optional_int(candidate.get("candidate_id")))
        while len(frame_candidate_ids) < spec.candidate_slots:
            frame_candidate_ids.append(None)
        candidate_ids.append(tuple(frame_candidate_ids))
        if encoded_candidates:
            top = encoded_candidates[0]
            frame_width = max(_optional_float(record.get("frame_width")) or 1.0, 1.0)
            frame_height = max(_optional_float(record.get("frame_height")) or 1.0, 1.0)
            xy_target[frame_slot, 0] = float(top.get("x", 0.0)) / frame_width
            xy_target[frame_slot, 1] = float(top.get("y", 0.0)) / frame_height
        temporal_target = record.get("temporal_target")
        if isinstance(temporal_target, Mapping):
            target_strategy = str(
                temporal_target.get("target_strategy") or "beatmap_action_v1"
            )
            window_target_strategy = target_strategy
            action_target[frame_slot] = _action_id_from_target(temporal_target)
            selected_candidate[frame_slot] = _selected_candidate_slot(
                temporal_target,
                encoded_candidates,
            )
            frame_width = max(_optional_float(record.get("frame_width")) or 1.0, 1.0)
            frame_height = max(_optional_float(record.get("frame_height")) or 1.0, 1.0)
            target_xy = temporal_target.get("target_video_xy")
            if isinstance(target_xy, Sequence) and len(target_xy) >= 2:
                xy_target[frame_slot, 0] = float(target_xy[0]) / frame_width
                xy_target[frame_slot, 1] = float(target_xy[1]) / frame_height
            time_offset[frame_slot, 0] = float(
                temporal_target.get("time_offset_ms") or 0.0
            )
    while len(sample_keys) < sequence_length:
        sample_keys.append(None)
        frame_indices.append(None)
        timestamps.append(None)
        candidate_ids.append(tuple(None for _ in range(spec.candidate_slots)))
    return TemporalWindow(
        features=candidate_features.flatten(start_dim=1),
        candidate_features=candidate_features,
        candidate_mask=candidate_mask,
        frame_mask=frame_mask,
        action_target=action_target,
        selected_candidate_target=selected_candidate,
        xy_target=xy_target,
        time_offset_target=time_offset,
        candidate_ids=tuple(candidate_ids),
        sample_keys=tuple(sample_keys),
        frame_indices=tuple(frame_indices),
        timestamps_ms=tuple(timestamps),
        target_strategy=window_target_strategy,
    )


def _action_id_from_target(target: Mapping[str, Any]) -> int:
    value = target.get("action_id")
    if isinstance(value, int) and 0 <= value < len(ACTION_NAMES):
        return value
    name = str(target.get("action") or "no_op")
    return ACTION_NAMES.index(name) if name in ACTION_NAMES else NO_OP_ACTION_ID


def _selected_candidate_slot(
    target: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> int:
    selected_id = _optional_int(target.get("selected_candidate_id"))
    if selected_id is None:
        return IGNORE_CANDIDATE_ID
    for index, candidate in enumerate(candidates):
        if _optional_int(candidate.get("candidate_id")) == selected_id:
            return index
    return IGNORE_CANDIDATE_ID


def _encode_candidate(
    candidate: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    spec: TemporalFeatureSpec,
) -> torch.Tensor:
    frame_width = max(_optional_float(record.get("frame_width")) or 1.0, 1.0)
    frame_height = max(_optional_float(record.get("frame_height")) or 1.0, 1.0)
    object_type = _optional_string(candidate.get("object_type")) or "background"
    type_vector = [0.0 for _ in spec.object_types]
    if object_type in spec.object_types:
        type_vector[spec.object_types.index(object_type)] = 1.0
    embedding = _candidate_embedding(candidate, spec.embedding_dim)
    values = [
        1.0,
        float(candidate.get("x", 0.0)) / frame_width,
        float(candidate.get("y", 0.0)) / frame_height,
        _float_field(candidate, "score"),
        _float_field(candidate, "center_score"),
        _float_field(candidate, "visible_score"),
        _float_field(candidate, "type_score"),
        _float_field(candidate, "ring_score"),
        _float_field(candidate, "ring_radius_px") / max(frame_width, frame_height),
        _float_field(candidate, "slider_score"),
        _float_field(candidate, "spinner_score"),
        0.0 if candidate.get("slider_path_id") is None else 1.0,
        1.0 if candidate.get("ambiguous") else 0.0,
    ]
    return torch.tensor(values + type_vector + embedding, dtype=torch.float32)


def _sorted_candidates(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    candidates = record.get("candidates") or ()
    if not isinstance(candidates, Sequence):
        return ()
    return tuple(
        sorted(
            (candidate for candidate in candidates if isinstance(candidate, Mapping)),
            key=lambda candidate: _float_field(candidate, "score"),
            reverse=True,
        )
    )


def _candidate_embedding(
    candidate: Mapping[str, Any],
    embedding_dim: int,
) -> list[float]:
    raw = candidate.get("embedding") or ()
    values = [float(item) for item in raw] if isinstance(raw, Sequence) else []
    if len(values) >= embedding_dim:
        return values[:embedding_dim]
    return values + [0.0] * (embedding_dim - len(values))


def _infer_embedding_dim(records: Sequence[Mapping[str, Any]]) -> int:
    for record in records:
        for candidate in _sorted_candidates(record):
            raw = candidate.get("embedding") or ()
            if isinstance(raw, Sequence):
                return len(raw)
    return 0


def _float_field(candidate: Mapping[str, Any], key: str) -> float:
    value = candidate.get(key)
    if value is None:
        return 0.0
    return float(value)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


__all__ = [
    "ACTION_NAMES",
    "HOLD_ACTION_ID",
    "IGNORE_CANDIDATE_ID",
    "NO_OP_ACTION_ID",
    "PRESS_ACTION_ID",
    "RELEASE_ACTION_ID",
    "TemporalCandidateWindowDataset",
    "TemporalFeatureSpec",
    "TemporalWindow",
    "load_candidate_cache_records",
]
