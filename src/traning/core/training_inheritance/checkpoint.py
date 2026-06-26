from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch


TRAINING_CHECKPOINT_SCHEMA_VERSION = "training-checkpoint-v1"


@dataclass(frozen=True)
class TrainingPosition:
    epoch: int = 0
    next_batch_index: int = 0
    global_step: int = 0
    spatial_step: int = 0
    temporal_step: int = 0
    gradient_accumulation_index: int = 0
    last_committed_step: int = 0
    consistency_boundary: str = "optimizer_step_committed"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> TrainingPosition:
        data = dict(raw or {})
        return cls(
            epoch=int(data.get("epoch", 0)),
            next_batch_index=int(data.get("next_batch_index", 0)),
            global_step=int(data.get("global_step", 0)),
            spatial_step=int(data.get("spatial_step", 0)),
            temporal_step=int(data.get("temporal_step", 0)),
            gradient_accumulation_index=int(
                data.get("gradient_accumulation_index", 0)
            ),
            last_committed_step=int(data.get("last_committed_step", 0)),
            consistency_boundary=str(
                data.get("consistency_boundary", "optimizer_step_committed")
            ),
        )


@dataclass(frozen=True)
class CheckpointRestorePlan:
    checkpoint_path: Path | None
    requested_policy: str
    actual_policy: str
    restored_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    downgrade_reasons: tuple[str, ...] = ()
    start_step: int = 0
    position: TrainingPosition = field(default_factory=TrainingPosition)
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.checkpoint_path is not None and self.actual_policy != "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "requested_policy": self.requested_policy,
            "actual_policy": self.actual_policy,
            "restored_fields": self.restored_fields,
            "missing_fields": self.missing_fields,
            "downgrade_reasons": self.downgrade_reasons,
            "start_step": self.start_step,
            "position": self.position.as_dict(),
        }


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(raw: Mapping[str, Any] | None) -> bool:
    if not raw:
        return False
    restored = False
    if raw.get("python") is not None:
        random.setstate(raw["python"])
        restored = True
    if raw.get("numpy") is not None:
        np.random.set_state(raw["numpy"])
        restored = True
    if raw.get("torch_cpu") is not None:
        torch.set_rng_state(raw["torch_cpu"])
        restored = True
    cuda_state = raw.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
        restored = True
    return restored


def build_training_checkpoint(
    *,
    checkpoint_kind: str,
    run_id: str,
    trial_id: str,
    models: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    position: TrainingPosition,
    score_state: Mapping[str, Any] | None = None,
    grade_state: Mapping[str, Any] | None = None,
    promotion_state: Mapping[str, Any] | None = None,
    dataset_state: Mapping[str, Any] | None = None,
    sampler_state: Mapping[str, Any] | None = None,
    resolved_config: Mapping[str, Any] | None = None,
    dataset_fingerprint: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": checkpoint_kind,
        "consistency_boundary": position.consistency_boundary,
        "run_id": run_id,
        "trial_id": trial_id,
        "models": dict(models),
        "ema": {},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "optimizer_type": type(optimizer).__name__ if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scheduler_type": type(scheduler).__name__ if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "training_position": position.as_dict(),
        "score_state": dict(score_state or {}),
        "grade_state": dict(grade_state or {}),
        "promotion_state": dict(promotion_state or {}),
        "dataset_state": dict(dataset_state or {}),
        "sampler_state": dict(sampler_state or {}),
        "rng_state": capture_rng_state(),
        "resolved_config": dict(resolved_config or {}),
        "dataset_fingerprint": dict(dataset_fingerprint or {}),
        "created_at": datetime.now(UTC).isoformat(),
        "extra": dict(extra or {}),
    }


def atomic_torch_save_checkpoint(
    payload: Mapping[str, Any],
    path: Path,
    *,
    expected_kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    loaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
    validate_training_checkpoint(loaded, expected_kind=expected_kind)
    tmp_path.replace(path)


def load_training_checkpoint(path: Path) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("checkpoint root must be a mapping")
    return raw


def validate_training_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
) -> None:
    if payload.get("schema_version") not in {
        TRAINING_CHECKPOINT_SCHEMA_VERSION,
        None,
    }:
        raise ValueError(f"unsupported checkpoint schema: {payload.get('schema_version')}")
    if expected_kind is not None and payload.get("checkpoint_kind") not in {
        expected_kind,
        None,
    }:
        raise ValueError(
            f"checkpoint kind mismatch: {payload.get('checkpoint_kind')} != {expected_kind}"
        )
    if "model_state" not in payload and "models" not in payload:
        raise ValueError("checkpoint missing model state")
    if payload.get("schema_version") == TRAINING_CHECKPOINT_SCHEMA_VERSION:
        if "training_position" not in payload:
            raise ValueError("checkpoint missing training_position")
        TrainingPosition.from_mapping(payload.get("training_position"))


def restore_module_state(
    module: torch.nn.Module,
    state_dict: Mapping[str, Any],
    *,
    strict: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    target = getattr(module, "_orig_mod", module)
    if strict:
        target.load_state_dict(state_dict, strict=True)
        return tuple(state_dict.keys()), ()
    current = target.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    skipped = tuple(sorted(set(state_dict) - set(compatible)))
    target.load_state_dict({**current, **compatible}, strict=True)
    return tuple(sorted(compatible)), skipped


__all__ = [
    "CheckpointRestorePlan",
    "TRAINING_CHECKPOINT_SCHEMA_VERSION",
    "TrainingPosition",
    "atomic_torch_save_checkpoint",
    "build_training_checkpoint",
    "capture_rng_state",
    "load_training_checkpoint",
    "restore_module_state",
    "restore_rng_state",
    "validate_training_checkpoint",
]
