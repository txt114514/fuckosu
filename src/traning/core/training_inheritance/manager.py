from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import random
import shutil
from typing import Any, Literal, Mapping

import numpy as np
import torch
import yaml

from traning.conf import Settings
from traning.state.versioning import collect_code_version, version_manifest
from traning.core.training_inheritance.checkpoint import (
    load_training_checkpoint,
    validate_training_checkpoint,
)

ResumePolicy = Literal["strict", "auto", "weights-only", "none"]
INHERITANCE_SCHEMA_VERSION = "training-inheritance-v1"


@dataclass(frozen=True)
class InheritancePackage:
    path: Path
    manifest_path: Path
    latest_checkpoint_path: Path | None
    best_checkpoint_path: Path | None
    status: str = "saved"

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "manifest_path": self.manifest_path,
            "latest_checkpoint_path": self.latest_checkpoint_path,
            "best_checkpoint_path": self.best_checkpoint_path,
            "status": self.status,
        }


@dataclass(frozen=True)
class InheritanceLoadResult:
    path: Path | None
    policy: ResumePolicy
    status: str
    compatible: bool
    requested_policy: ResumePolicy | None = None
    loaded_checkpoint_path: Path | None = None
    stage_checkpoint_paths: Mapping[str, Path] = field(default_factory=dict)
    downgrade_reasons: tuple[str, ...] = ()
    restored_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    manifest: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "policy": self.policy,
            "status": self.status,
            "compatible": self.compatible,
            "requested_policy": self.requested_policy or self.policy,
            "loaded_checkpoint_path": self.loaded_checkpoint_path,
            "stage_checkpoint_paths": dict(self.stage_checkpoint_paths),
            "downgrade_reasons": self.downgrade_reasons,
            "restored_fields": self.restored_fields,
            "missing_fields": self.missing_fields,
            "manifest": self.manifest,
        }


def create_inheritance_package(
    *,
    output_dir: Path,
    settings: Settings,
    resolved_config_path: Path | None = None,
    latest_checkpoint_path: Path | None = None,
    best_checkpoint_path: Path | None = None,
    training_state: dict[str, Any] | None = None,
    score_state: dict[str, Any] | None = None,
    promotion_state: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    stage_checkpoints: Mapping[str, Path | None] | None = None,
) -> InheritancePackage:
    inheritance_dir = output_dir / "inheritance"
    inheritance_dir.mkdir(parents=True, exist_ok=True)
    latest_copy = _copy_checkpoint(latest_checkpoint_path, inheritance_dir / "latest_checkpoint.pt")
    best_copy = _copy_checkpoint(best_checkpoint_path, inheritance_dir / "best_checkpoint.pt")
    stage_checkpoint_dir = inheritance_dir / "stage_checkpoints"
    copied_stage_checkpoints: dict[str, str] = {}
    for stage, checkpoint in sorted(dict(stage_checkpoints or {}).items()):
        copied = _copy_checkpoint(
            checkpoint,
            stage_checkpoint_dir / f"{stage}_checkpoint.pt",
        )
        if copied is not None:
            copied_stage_checkpoints[stage] = str(copied.relative_to(inheritance_dir))
    if resolved_config_path and resolved_config_path.exists():
        shutil.copy2(resolved_config_path, inheritance_dir / "resolved_config.yaml")
    else:
        (inheritance_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(settings.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    dataset_fingerprint = _dataset_fingerprint(settings)
    _write_json(inheritance_dir / "dataset_fingerprint.json", dataset_fingerprint)
    _write_json(inheritance_dir / "training_state.json", training_state or {})
    _write_json(inheritance_dir / "score_state.json", score_state or {})
    _write_json(inheritance_dir / "promotion_state.json", promotion_state or {})
    _write_json(inheritance_dir / "rng_state.json", _rng_state())
    manifest = {
        "schema_version": INHERITANCE_SCHEMA_VERSION,
        "code_version": collect_code_version().as_dict(),
        "versions": version_manifest(settings),
        "dataset_fingerprint": dataset_fingerprint,
        "latest_checkpoint": latest_copy.name if latest_copy else None,
        "best_checkpoint": best_copy.name if best_copy else None,
        "stage_checkpoints": copied_stage_checkpoints,
        "artifacts": artifacts or {},
    }
    manifest_path = inheritance_dir / "inheritance_manifest.json"
    _write_json(manifest_path, manifest)
    latest_pointer = output_dir.parent / "latest_inheritance.json"
    _write_json(latest_pointer, {"path": str(inheritance_dir), "manifest_path": str(manifest_path)})
    return InheritancePackage(
        path=inheritance_dir,
        manifest_path=manifest_path,
        latest_checkpoint_path=latest_copy,
        best_checkpoint_path=best_copy,
    )


def load_inheritance_package(
    *,
    inherit_from: Path | str | None,
    current_settings: Settings,
    policy: ResumePolicy,
) -> InheritanceLoadResult:
    if policy == "none" or inherit_from in (None, "", "none"):
        return InheritanceLoadResult(None, policy, "skipped", True)
    path = resolve_inheritance_path(inherit_from)
    if path is None:
        if policy == "strict":
            raise FileNotFoundError("inheritance package not found")
        return InheritanceLoadResult(
            None,
            policy,
            "missing",
            False,
            downgrade_reasons=("inheritance_missing",),
            missing_fields=("manifest",),
        )
    manifest_path = path / "inheritance_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reasons = _compatibility_reasons(manifest, current_settings)
    if reasons and policy == "strict":
        raise ValueError("inheritance incompatible: " + ", ".join(reasons))
    effective_policy: ResumePolicy = policy
    if reasons and policy == "auto":
        effective_policy = "weights-only"
        reasons.append("auto_downgraded_to_weights_only")
    compatible = not reasons
    status = "loaded" if compatible else "downgraded"
    checkpoint_name = manifest.get("latest_checkpoint") or manifest.get("best_checkpoint")
    checkpoint_path = path / checkpoint_name if checkpoint_name else None
    stage_checkpoint_paths = _stage_checkpoint_paths(path, manifest)
    restored_fields: list[str] = []
    missing_fields: list[str] = []
    selected_checkpoint = checkpoint_path if checkpoint_path and checkpoint_path.exists() else None
    if selected_checkpoint is not None:
        try:
            checkpoint = load_training_checkpoint(selected_checkpoint)
            validate_training_checkpoint(checkpoint)
            if checkpoint.get("optimizer") is not None:
                restored_fields.append("optimizer")
            else:
                missing_fields.append("optimizer")
            if checkpoint.get("scaler") is not None:
                restored_fields.append("scaler")
            else:
                missing_fields.append("scaler")
            if checkpoint.get("rng_state") is not None:
                restored_fields.append("rng")
            else:
                missing_fields.append("rng")
            if checkpoint.get("training_position") is not None:
                restored_fields.append("training_position")
            else:
                missing_fields.append("training_position")
        except Exception as error:
            reasons.append(f"checkpoint_invalid:{type(error).__name__}")
            if policy == "strict":
                raise
            selected_checkpoint = None
    for stage, stage_path in tuple(stage_checkpoint_paths.items()):
        try:
            checkpoint = load_training_checkpoint(stage_path)
            validate_training_checkpoint(checkpoint)
        except Exception as error:
            reasons.append(f"{stage}_checkpoint_invalid:{type(error).__name__}")
            if policy == "strict":
                raise
            stage_checkpoint_paths.pop(stage, None)
    for stage in ("spatial", "temporal"):
        if stage in stage_checkpoint_paths:
            restored_fields.append(f"{stage}_checkpoint")
        else:
            missing_fields.append(f"{stage}_checkpoint")
    if effective_policy == "weights-only" and checkpoint_path is None:
        status = "weights_missing"
    if effective_policy == "weights-only":
        restored_fields = [
            field for field in restored_fields if field.endswith("_checkpoint")
        ]
        missing_fields.extend(("optimizer", "scaler", "sampler", "rng"))
    compatible = not reasons
    if not compatible and status == "loaded":
        status = "downgraded"
    return InheritanceLoadResult(
        path=path,
        policy=effective_policy,
        status=status,
        compatible=compatible,
        requested_policy=policy,
        loaded_checkpoint_path=selected_checkpoint,
        stage_checkpoint_paths=stage_checkpoint_paths,
        downgrade_reasons=tuple(reasons),
        restored_fields=tuple(sorted(set(restored_fields))),
        missing_fields=tuple(sorted(set(missing_fields))),
        manifest=manifest,
    )


def resolve_inheritance_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    if str(value) == "latest":
        candidates = sorted(Path("artifacts").glob("**/latest_inheritance.json"))
        candidates.extend(sorted(Path("runs").glob("**/latest_inheritance.json")))
        if not candidates:
            return None
        pointer = json.loads(candidates[-1].read_text(encoding="utf-8"))
        return Path(pointer["path"])
    path = Path(value)
    if path.is_file():
        return path.parent
    return path if path.exists() else None


def _compatibility_reasons(manifest: dict[str, Any], settings: Settings) -> list[str]:
    reasons: list[str] = []
    if manifest.get("schema_version") != INHERITANCE_SCHEMA_VERSION:
        reasons.append("schema_version")
    old = manifest.get("dataset_fingerprint") or {}
    current = _dataset_fingerprint(settings)
    for key in ("dataset_root", "split_manifest_path", "train_items", "sample_fps"):
        if _comparable(old.get(key)) != _comparable(current.get(key)):
            reasons.append(f"dataset_{key}")
    versions = manifest.get("versions") or {}
    if versions.get("configuration_version") != version_manifest(settings).get("configuration_version"):
        reasons.append("configuration_version")
    return reasons


def _stage_checkpoint_paths(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    raw = manifest.get("stage_checkpoints") or {}
    if not isinstance(raw, Mapping):
        return {}
    paths: dict[str, Path] = {}
    for stage, value in raw.items():
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        if path.exists():
            paths[str(stage)] = path
    return paths


def _dataset_fingerprint(settings: Settings) -> dict[str, Any]:
    data = settings.data_input
    return {
        "dataset_root": str(data.dataset_root),
        "split_manifest_path": str(data.split_manifest_path),
        "train_items": tuple(data.train_items),
        "validation_items": tuple(data.validation_items),
        "test_items": tuple(data.test_items),
        "sample_fps": data.sample_fps,
        "frame_step": data.frame_step,
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python_random": repr(random.getstate()),
        "numpy_random": repr(np.random.get_state()),
        "torch_cpu_rng_size": len(torch.get_rng_state()),
        "torch_cuda_rng_available": torch.cuda.is_available(),
    }


def _comparable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_comparable(item) for item in value]
    if isinstance(value, list):
        return [_comparable(item) for item in value]
    return value


def _copy_checkpoint(source: Path | None, destination: Path) -> Path | None:
    if source is None or not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
