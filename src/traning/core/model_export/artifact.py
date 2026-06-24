from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


MODEL_ARTIFACT_VERSION = "model-artifact-v1"
ArtifactKind = Literal["inference", "resume"]


@dataclass(frozen=True)
class ArtifactFile:
    role: str
    path: Path
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path.as_posix(),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ModelArtifactSpec:
    artifact_id: str
    output_dir: Path
    kind: ArtifactKind = "inference"
    settings_path: Path | None = None
    spatial_checkpoint_path: Path | None = None
    temporal_checkpoint_path: Path | None = None
    metadata_path: Path | None = None
    score_version: str | None = None
    candidate_cache_version: str | None = None
    code_version: str | None = None
    extra_files: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id must not be empty")
        if self.kind not in {"inference", "resume"}:
            raise ValueError("artifact kind must be inference or resume")
        object.__setattr__(self, "extra_files", dict(self.extra_files))


@dataclass(frozen=True)
class ModelArtifactResult:
    artifact_dir: Path
    manifest_path: Path
    files: tuple[ArtifactFile, ...]
    kind: ArtifactKind
    version: str = MODEL_ARTIFACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "kind": self.kind,
            "artifact_dir": str(self.artifact_dir),
            "manifest_path": str(self.manifest_path),
            "files": [item.as_dict() for item in self.files],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path, role: str) -> ArtifactFile:
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"artifact input does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return ArtifactFile(
        role=role,
        path=destination,
        sha256=_sha256(destination),
        size_bytes=destination.stat().st_size,
    )


def _copy_optional(
    files: list[ArtifactFile],
    source: Path | None,
    destination: Path,
    role: str,
) -> None:
    if source is not None:
        files.append(_copy_file(source, destination, role))


def _write_readme(path: Path, spec: ModelArtifactSpec) -> ArtifactFile:
    path.write_text(
        "\n".join(
            (
                f"# {spec.artifact_id}",
                "",
                f"Artifact kind: `{spec.kind}`",
                "",
                "This package is a reproducible PyTorch training artifact.",
                "Use `manifest.json` to verify file hashes and schema versions.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return ArtifactFile(
        role="readme",
        path=path,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _manifest_file(item: ArtifactFile, artifact_dir: Path) -> dict[str, Any]:
    data = item.as_dict()
    try:
        data["path"] = item.path.relative_to(artifact_dir).as_posix()
    except ValueError:
        data["path"] = item.path.as_posix()
    return data


def export_model_artifact(spec: ModelArtifactSpec) -> ModelArtifactResult:
    artifact_dir = spec.output_dir / spec.artifact_id
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)

    files: list[ArtifactFile] = []
    _copy_optional(
        files,
        spec.settings_path,
        artifact_dir / "settings.yaml",
        "settings",
    )
    _copy_optional(
        files,
        spec.spatial_checkpoint_path,
        artifact_dir / "spatial_model.pt",
        "spatial_checkpoint",
    )
    _copy_optional(
        files,
        spec.temporal_checkpoint_path,
        artifact_dir / "temporal_model.pt",
        "temporal_checkpoint",
    )
    _copy_optional(
        files,
        spec.metadata_path,
        artifact_dir / "checkpoint_metadata.json",
        "checkpoint_metadata",
    )
    for role, path in sorted(spec.extra_files.items()):
        files.append(
            _copy_file(
                path,
                artifact_dir / "extra" / path.name,
                f"extra:{role}",
            )
        )
    files.append(_write_readme(artifact_dir / "README.md", spec))

    manifest_path = artifact_dir / "manifest.json"
    manifest = {
        "version": MODEL_ARTIFACT_VERSION,
        "artifact_id": spec.artifact_id,
        "kind": spec.kind,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "score_version": spec.score_version,
        "candidate_cache_version": spec.candidate_cache_version,
        "code_version": spec.code_version,
        "files": [_manifest_file(item, artifact_dir) for item in files],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    files.append(
        ArtifactFile(
            role="manifest",
            path=manifest_path,
            sha256=_sha256(manifest_path),
            size_bytes=manifest_path.stat().st_size,
        )
    )
    return ModelArtifactResult(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        files=tuple(files),
        kind=spec.kind,
    )


def validate_model_artifact(manifest_path: Path) -> tuple[str, ...]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("version") != MODEL_ARTIFACT_VERSION:
        raise ValueError("unsupported model artifact version")
    issues: list[str] = []
    artifact_dir = manifest_path.parent
    for item in raw.get("files", ()):
        path = Path(item["path"])
        if not path.is_absolute():
            path = artifact_dir / path
        if not path.exists():
            issues.append(f"missing {item['role']}: {path}")
            continue
        digest = _sha256(path)
        if digest != item.get("sha256"):
            issues.append(f"sha256 mismatch for {item['role']}: {path}")
    return tuple(issues)


__all__ = [
    "ArtifactFile",
    "ArtifactKind",
    "MODEL_ARTIFACT_VERSION",
    "ModelArtifactResult",
    "ModelArtifactSpec",
    "export_model_artifact",
    "validate_model_artifact",
]
