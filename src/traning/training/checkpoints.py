"""V2 runtime 三模型 checkpoint 的事务发布、校验与加载。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
import uuid

import torch

from traning.belief import PerTrackBeliefEncoder
from traning.config import V2Config, v2_config_to_dict
from traning.contracts import ArtifactManifest, DataSplit
from traning.contracts.common import (
    require_identifier,
    require_nonnegative,
    require_transform_fingerprint,
)
from traning.data import FrameCoordinateTransform
from traning.infrastructure import (
    IntegrityError,
    SchemaMismatchError,
    atomic_write_bytes,
    atomic_write_json,
    read_json_object,
    sha256_file,
)
from traning.outcome import DenseOutcomeModel
from traning.perception import PerceptionModel


RUNTIME_CHECKPOINT_SCHEMA_VERSION = 1
RUNTIME_CHECKPOINT_ARTIFACT_TYPE = "traning_runtime_checkpoint"
CHECKPOINT_MANIFEST_FILENAME = "manifest.json"
_MODEL_KEYS = frozenset({"perception", "belief", "outcome"})
_MANIFEST_KEYS = frozenset(
    {
        "artifact_id",
        "artifact_type",
        "schema_version",
        "dataset_id",
        "split",
        "producer_id",
        "row_count",
        "sha256",
        "created_at_ms",
        "metadata",
    }
)


@dataclass(frozen=True, slots=True)
class RuntimeCheckpointManifest:
    """通用 ArtifactManifest 上的 runtime checkpoint 强约束视图。"""

    artifact: ArtifactManifest

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ArtifactManifest):
            raise TypeError("artifact 必须是 ArtifactManifest")
        if self.artifact.artifact_type != RUNTIME_CHECKPOINT_ARTIFACT_TYPE:
            raise ValueError(
                f"artifact_type 必须是 {RUNTIME_CHECKPOINT_ARTIFACT_TYPE!r}"
            )
        if self.artifact.schema_version != RUNTIME_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema_version 必须是 {RUNTIME_CHECKPOINT_SCHEMA_VERSION}"
            )
        if self.artifact.split is not DataSplit.ALL:
            raise ValueError("runtime checkpoint split 必须是 DataSplit.ALL")
        if self.artifact.row_count != len(_MODEL_KEYS):
            raise ValueError("runtime checkpoint 必须精确包含三个模型")
        metadata = dict(self.artifact.metadata)
        expected = {
            "weights_filename",
            "model_contract_sha256",
            "training_config_sha256",
            "transform_fingerprint",
        }
        if set(metadata) != expected:
            raise ValueError("runtime checkpoint metadata 字段集合不匹配")
        weights_filename = metadata["weights_filename"]
        if (
            not isinstance(weights_filename, str)
            or Path(weights_filename).name != weights_filename
            or not weights_filename.startswith("weights.")
            or not weights_filename.endswith(".pt")
        ):
            raise ValueError("weights_filename 必须是安全的不可变 generation 文件名")
        for name in ("model_contract_sha256", "training_config_sha256"):
            value = metadata[name]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} 必须是小写 SHA-256")
        transform_fingerprint = metadata["transform_fingerprint"]
        require_transform_fingerprint(transform_fingerprint)

    @property
    def weights_filename(self) -> str:
        """返回 manifest 已提交的不可变权重文件名。"""

        value = dict(self.artifact.metadata)["weights_filename"]
        if not isinstance(value, str):  # pragma: no cover - 构造期已校验
            raise TypeError("weights_filename 必须是字符串")
        return value

    @property
    def model_contract_sha256(self) -> str:
        """返回决定三个网络结构和输出语义的 canonical 摘要。"""

        value = dict(self.artifact.metadata)["model_contract_sha256"]
        if not isinstance(value, str):  # pragma: no cover - 构造期已校验
            raise TypeError("model_contract_sha256 必须是字符串")
        return value

    @property
    def training_config_sha256(self) -> str:
        """返回发布权重时完整训练配置的审计摘要。"""

        value = dict(self.artifact.metadata)["training_config_sha256"]
        if not isinstance(value, str):  # pragma: no cover - 构造期已校验
            raise TypeError("training_config_sha256 必须是字符串")
        return value

    @property
    def transform_fingerprint(self) -> str:
        """返回模型训练所绑定的坐标变换指纹。"""

        value = dict(self.artifact.metadata)["transform_fingerprint"]
        if not isinstance(value, str):  # pragma: no cover - 构造期已校验
            raise TypeError("transform_fingerprint 必须是字符串")
        return value


@dataclass(frozen=True, slots=True)
class RuntimeModelBundle:
    """已验证三模型及其 checkpoint/坐标身份的唯一 factory 输入。"""

    perception_model: PerceptionModel
    belief_encoder: PerTrackBeliefEncoder
    outcome_model: DenseOutcomeModel
    artifact_id: str
    transform_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.perception_model, PerceptionModel):
            raise TypeError("perception_model 必须是 PerceptionModel")
        if not isinstance(self.belief_encoder, PerTrackBeliefEncoder):
            raise TypeError("belief_encoder 必须是 PerTrackBeliefEncoder")
        if not isinstance(self.outcome_model, DenseOutcomeModel):
            raise TypeError("outcome_model 必须是 DenseOutcomeModel")
        if (
            not isinstance(self.artifact_id, str)
            or not self.artifact_id
            or self.artifact_id != self.artifact_id.strip()
        ):
            raise ValueError("artifact_id 必须非空且无首尾空格")
        require_transform_fingerprint(self.transform_fingerprint)


def publish_runtime_checkpoint(
    directory: Path,
    config: V2Config,
    models: RuntimeModelBundle,
    coordinate_transform: FrameCoordinateTransform,
    *,
    dataset_id: str,
    producer_id: str,
    created_at_ms: float | None = None,
) -> RuntimeCheckpointManifest:
    """先原子发布权重 generation，再以 manifest 作为唯一提交点。"""

    _validate_checkpoint_context(config, models, coordinate_transform)
    if not isinstance(directory, Path):
        raise TypeError("directory 必须是 pathlib.Path")
    require_identifier(dataset_id, "dataset_id")
    require_identifier(producer_id, "producer_id")
    timestamp = time.time_ns() / 1_000_000.0 if created_at_ms is None else created_at_ms
    require_nonnegative(timestamp, "created_at_ms")

    state_payload = {
        "perception": _cpu_state_dict(models.perception_model),
        "belief": _cpu_state_dict(models.belief_encoder),
        "outcome": _cpu_state_dict(models.outcome_model),
    }
    buffer = BytesIO()
    torch.save(state_payload, buffer)
    weights_filename = f"weights.{uuid.uuid4().hex}.pt"
    weights_path = directory / weights_filename
    atomic_write_bytes(weights_path, buffer.getvalue())
    manifest = RuntimeCheckpointManifest(
        ArtifactManifest(
            artifact_id=models.artifact_id,
            artifact_type=RUNTIME_CHECKPOINT_ARTIFACT_TYPE,
            schema_version=RUNTIME_CHECKPOINT_SCHEMA_VERSION,
            dataset_id=dataset_id,
            split=DataSplit.ALL,
            producer_id=producer_id,
            row_count=len(_MODEL_KEYS),
            sha256=sha256_file(weights_path),
            created_at_ms=float(timestamp),
            metadata=(
                ("weights_filename", weights_filename),
                ("model_contract_sha256", _model_contract_sha256(config)),
                ("training_config_sha256", _training_config_sha256(config)),
                (
                    "transform_fingerprint",
                    coordinate_transform.transform_fingerprint,
                ),
            ),
        )
    )
    atomic_write_json(
        directory / CHECKPOINT_MANIFEST_FILENAME,
        _manifest_to_json(manifest),
    )
    return manifest


def load_runtime_checkpoint(
    directory: Path,
    config: V2Config,
    coordinate_transform: FrameCoordinateTransform,
    *,
    expected_dataset_id: str,
) -> RuntimeModelBundle:
    """校验 schema/config/坐标/摘要后，以 strict state dict 恢复三个模型。"""

    if not isinstance(directory, Path):
        raise TypeError("directory 必须是 pathlib.Path")
    if not isinstance(config, V2Config):
        raise TypeError("config 必须是 V2Config")
    if not isinstance(coordinate_transform, FrameCoordinateTransform):
        raise TypeError("coordinate_transform 必须是 FrameCoordinateTransform")
    require_identifier(expected_dataset_id, "expected_dataset_id")
    manifest = _manifest_from_json(
        read_json_object(directory / CHECKPOINT_MANIFEST_FILENAME)
    )
    expected_contract_sha = _model_contract_sha256(config)
    if manifest.model_contract_sha256 != expected_contract_sha:
        raise SchemaMismatchError("runtime checkpoint 模型契约摘要不匹配")
    if manifest.artifact.dataset_id != expected_dataset_id:
        raise SchemaMismatchError("runtime checkpoint dataset identity 不匹配")
    if manifest.transform_fingerprint != coordinate_transform.transform_fingerprint:
        raise SchemaMismatchError("runtime checkpoint 坐标变换指纹不匹配")

    weights_path = directory / manifest.weights_filename
    actual_sha = sha256_file(weights_path)
    if actual_sha != manifest.artifact.sha256:
        raise IntegrityError("runtime checkpoint 权重 SHA-256 不匹配")
    try:
        state_payload = torch.load(
            BytesIO(weights_path.read_bytes()),
            map_location="cpu",
            weights_only=True,
        )
    # torch/pickle 是外部解码边界；摘要通过后仍把所有常规解码异常归一为完整性错误。
    except Exception as exc:
        raise IntegrityError("runtime checkpoint 权重无法安全解码") from exc
    if not isinstance(state_payload, dict) or set(state_payload) != _MODEL_KEYS:
        raise SchemaMismatchError("runtime checkpoint 模型字段集合不匹配")

    perception_model = PerceptionModel(config.perception)
    belief_encoder = PerTrackBeliefEncoder(
        config.belief,
        appearance_embedding_dim=config.perception.embedding_dim,
    )
    outcome_model = DenseOutcomeModel(
        config.outcome,
        belief_embedding_dim=belief_encoder.flattened_hidden_dim,
    )
    model_registry = (
        ("perception", perception_model),
        ("belief", belief_encoder),
        ("outcome", outcome_model),
    )
    try:
        for key, model in model_registry:
            state_dict = state_payload[key]
            if not isinstance(state_dict, dict):
                raise SchemaMismatchError(f"checkpoint {key} state_dict 必须是 object")
            model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise SchemaMismatchError(
            "runtime checkpoint state_dict 与模型结构不一致"
        ) from exc
    return RuntimeModelBundle(
        perception_model=perception_model,
        belief_encoder=belief_encoder,
        outcome_model=outcome_model,
        artifact_id=manifest.artifact.artifact_id,
        transform_fingerprint=manifest.transform_fingerprint,
    )


def _validate_checkpoint_context(
    config: V2Config,
    models: RuntimeModelBundle,
    coordinate_transform: FrameCoordinateTransform,
) -> None:
    """在触碰磁盘前验证模型 config 和坐标身份完全一致。"""

    if not isinstance(config, V2Config):
        raise TypeError("config 必须是 V2Config")
    if not isinstance(models, RuntimeModelBundle):
        raise TypeError("models 必须是 RuntimeModelBundle")
    if not isinstance(coordinate_transform, FrameCoordinateTransform):
        raise TypeError("coordinate_transform 必须是 FrameCoordinateTransform")
    if models.perception_model.config != config.perception:
        raise ValueError("PerceptionModel config 与 V2Config 不一致")
    if models.belief_encoder.config != config.belief:
        raise ValueError("Belief encoder config 与 V2Config 不一致")
    if models.outcome_model.config != config.outcome:
        raise ValueError("Outcome model config 与 V2Config 不一致")
    if models.transform_fingerprint != coordinate_transform.transform_fingerprint:
        raise ValueError("模型 bundle 与坐标变换指纹不一致")


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """复制为无梯度 CPU tensor，避免 checkpoint 绑定保存时设备。"""

    return {
        key: value.detach().to(device="cpu").clone()
        for key, value in model.state_dict().items()
    }


def _model_contract_sha256(config: V2Config) -> str:
    """只摘要实际决定三模型权重形状和输出语义的配置。"""

    serialized = v2_config_to_dict(config)
    contract = {
        "schema_version": serialized["schema_version"],
        "perception": serialized["perception"],
        "belief": serialized["belief"],
        "outcome": serialized["outcome"],
    }
    return _json_sha256(contract)


def _training_config_sha256(config: V2Config) -> str:
    """摘要发布时完整配置，用于追溯而不阻止部署侧路径调整。"""

    return _json_sha256(v2_config_to_dict(config))


def _json_sha256(payload: dict[str, object]) -> str:
    """计算严格 JSON object 的稳定 SHA-256。"""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_to_json(manifest: RuntimeCheckpointManifest) -> dict[str, object]:
    """把 typed manifest 投影到唯一严格 JSON 边界。"""

    artifact = manifest.artifact
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_type": artifact.artifact_type,
        "schema_version": artifact.schema_version,
        "dataset_id": artifact.dataset_id,
        "split": artifact.split.value,
        "producer_id": artifact.producer_id,
        "row_count": artifact.row_count,
        "sha256": artifact.sha256,
        "created_at_ms": artifact.created_at_ms,
        "metadata": {key: value for key, value in artifact.metadata},
    }


def _manifest_from_json(payload: dict[str, object]) -> RuntimeCheckpointManifest:
    """严格恢复 checkpoint manifest，拒绝未知字段和宽松强转。"""

    if set(payload) != _MANIFEST_KEYS:
        raise SchemaMismatchError("runtime checkpoint manifest 字段集合不匹配")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) for key in metadata
    ):
        raise SchemaMismatchError("runtime checkpoint metadata 必须是 JSON object")
    try:
        artifact = ArtifactManifest(
            artifact_id=_string(payload, "artifact_id"),
            artifact_type=_string(payload, "artifact_type"),
            schema_version=_integer(payload, "schema_version"),
            dataset_id=_string(payload, "dataset_id"),
            split=DataSplit(_string(payload, "split")),
            producer_id=_string(payload, "producer_id"),
            row_count=_integer(payload, "row_count"),
            sha256=_string(payload, "sha256"),
            created_at_ms=_number(payload, "created_at_ms"),
            metadata=tuple(sorted(metadata.items())),
        )
        return RuntimeCheckpointManifest(artifact)
    except (TypeError, ValueError) as exc:
        raise SchemaMismatchError(
            "runtime checkpoint manifest typed schema 不匹配"
        ) from exc


def _string(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise SchemaMismatchError(f"{key} 必须是字符串")
    return value


def _integer(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaMismatchError(f"{key} 必须是整数")
    return value


def _number(payload: dict[str, object], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaMismatchError(f"{key} 必须是数值")
    return float(value)


__all__ = (
    "CHECKPOINT_MANIFEST_FILENAME",
    "RUNTIME_CHECKPOINT_ARTIFACT_TYPE",
    "RUNTIME_CHECKPOINT_SCHEMA_VERSION",
    "RuntimeCheckpointManifest",
    "RuntimeModelBundle",
    "load_runtime_checkpoint",
    "publish_runtime_checkpoint",
)
