"""参数搜索历史的严格校验、原子提交与重启恢复边界。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

from traning.conf import V2Config, v2_config_to_dict
from traning.state.common import require_identifier, require_sha256
from traning.lib.infrastructure import (
    IntegrityError,
    SchemaMismatchError,
    atomic_write_json,
    read_json_object,
)
from traning.core.training.optimization import (
    ParameterVector,
    TrialAcceptance,
    TrialObservation,
)


SEARCH_STATE_SCHEMA_VERSION = 2
"""当前唯一支持的可恢复搜索状态版本。"""

_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "dataset_id",
        "config_sha256",
        "initial_parameters",
        "history",
        "history_sha256",
        "updated_at_ms",
    }
)
_PARAMETER_KEYS = (
    "learning_rate",
    "score_threshold",
    "max_candidates",
    "risk_lambda",
    "wait_cost",
    "min_confidence",
)
_ACCEPTANCE_KEYS = (
    "data",
    "perception",
    "tracking",
    "belief",
    "outcome",
    "decision",
    "golden",
    "schedule",
)


@dataclass(frozen=True, slots=True)
class SearchHistoryState:
    """与运行、数据和完整配置身份绑定的不可变搜索历史。"""

    run_id: str
    dataset_id: str
    config_sha256: str
    initial_parameters: ParameterVector
    history: tuple[TrialObservation, ...]
    updated_at_ms: float

    def __post_init__(self) -> None:
        require_identifier(self.run_id, "run_id")
        require_identifier(self.dataset_id, "dataset_id")
        require_sha256(self.config_sha256)
        if not isinstance(self.initial_parameters, ParameterVector):
            raise TypeError("initial_parameters 必须是 ParameterVector")
        if not isinstance(self.history, tuple) or any(
            not isinstance(item, TrialObservation) for item in self.history
        ):
            raise TypeError("history 必须是 TrialObservation 元组")
        if tuple(item.trial_index for item in self.history) != tuple(
            range(len(self.history))
        ):
            raise ValueError("history.trial_index 必须从 0 连续递增")
        parameters = tuple(item.parameters for item in self.history)
        if len(parameters) != len(set(parameters)):
            raise ValueError("history 不得包含重复 proposal")
        if (
            isinstance(self.updated_at_ms, bool)
            or not isinstance(self.updated_at_ms, int | float)
            or not math.isfinite(float(self.updated_at_ms))
            or self.updated_at_ms < 0.0
        ):
            raise ValueError("updated_at_ms 必须是非负数")


class SearchHistoryStore:
    """把每个已完成 trial 作为原子恢复点保存到单一状态文件。"""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        dataset_id: str,
        config: V2Config,
        initial_parameters: ParameterVector,
    ) -> None:
        if not isinstance(path, Path):
            raise TypeError("path 必须是 pathlib.Path")
        if not isinstance(config, V2Config):
            raise TypeError("config 必须是 V2Config")
        if not isinstance(initial_parameters, ParameterVector):
            raise TypeError("initial_parameters 必须是 ParameterVector")
        require_identifier(run_id, "run_id")
        require_identifier(dataset_id, "dataset_id")
        self.path = path
        self.run_id = run_id
        self.dataset_id = dataset_id
        self.config_sha256 = training_config_sha256(config)
        self.initial_parameters = initial_parameters

    def load(self) -> tuple[TrialObservation, ...]:
        """不存在状态时从零开始；存在时必须通过全部身份与摘要校验。"""

        if not self.path.exists():
            return ()
        state = _state_from_json(read_json_object(self.path))
        expected = (
            self.run_id,
            self.dataset_id,
            self.config_sha256,
            self.initial_parameters,
        )
        actual = (
            state.run_id,
            state.dataset_id,
            state.config_sha256,
            state.initial_parameters,
        )
        if actual != expected:
            raise SchemaMismatchError(
                "搜索恢复状态与当前 run/data/config/initial 不一致"
            )
        return state.history

    def persist(self, history: tuple[TrialObservation, ...]) -> None:
        """校验完整历史后原子覆盖状态；可直接作为搜索完成回调。"""

        state = SearchHistoryState(
            run_id=self.run_id,
            dataset_id=self.dataset_id,
            config_sha256=self.config_sha256,
            initial_parameters=self.initial_parameters,
            history=history,
            updated_at_ms=time.time_ns() / 1_000_000.0,
        )
        atomic_write_json(self.path, _state_to_json(state))


def training_config_sha256(config: V2Config) -> str:
    """计算完整 V2 配置的稳定摘要，供恢复状态拒绝跨配置串用。"""

    if not isinstance(config, V2Config):
        raise TypeError("config 必须是 V2Config")
    payload = v2_config_to_dict(config)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _state_to_json(state: SearchHistoryState) -> dict[str, object]:
    history = [_observation_to_json(item) for item in state.history]
    return {
        "schema_version": SEARCH_STATE_SCHEMA_VERSION,
        "run_id": state.run_id,
        "dataset_id": state.dataset_id,
        "config_sha256": state.config_sha256,
        "initial_parameters": _parameters_to_json(state.initial_parameters),
        "history": history,
        "history_sha256": hashlib.sha256(_canonical_json_bytes(history)).hexdigest(),
        "updated_at_ms": float(state.updated_at_ms),
    }


def _state_from_json(payload: dict[str, object]) -> SearchHistoryState:
    if set(payload) != _ROOT_KEYS:
        raise SchemaMismatchError("搜索状态根字段集合不匹配")
    if _integer(payload, "schema_version") != SEARCH_STATE_SCHEMA_VERSION:
        raise SchemaMismatchError(
            f"搜索状态仅支持 schema {SEARCH_STATE_SCHEMA_VERSION}"
        )
    history_payload = payload["history"]
    if not isinstance(history_payload, list):
        raise SchemaMismatchError("history 必须是 JSON array")
    expected_history_sha = _string(payload, "history_sha256")
    require_sha256(expected_history_sha)
    actual_history_sha = hashlib.sha256(
        _canonical_json_bytes(history_payload)
    ).hexdigest()
    if actual_history_sha != expected_history_sha:
        raise IntegrityError("搜索 history SHA-256 不匹配")
    initial_payload = payload["initial_parameters"]
    if not isinstance(initial_payload, dict):
        raise SchemaMismatchError("initial_parameters 必须是 JSON object")
    try:
        return SearchHistoryState(
            run_id=_string(payload, "run_id"),
            dataset_id=_string(payload, "dataset_id"),
            config_sha256=_string(payload, "config_sha256"),
            initial_parameters=_parameters_from_json(initial_payload),
            history=tuple(_observation_from_json(item) for item in history_payload),
            updated_at_ms=_number(payload, "updated_at_ms"),
        )
    except (TypeError, ValueError) as exc:
        raise SchemaMismatchError("搜索状态 typed schema 不匹配") from exc


def _observation_to_json(observation: TrialObservation) -> dict[str, object]:
    return {
        "trial_index": observation.trial_index,
        "parameters": _parameters_to_json(observation.parameters),
        "objective": float(observation.objective),
        "acceptance": {
            key: getattr(observation.acceptance, key) for key in _ACCEPTANCE_KEYS
        },
    }


def _observation_from_json(payload: object) -> TrialObservation:
    if not isinstance(payload, dict) or set(payload) != {
        "trial_index",
        "parameters",
        "objective",
        "acceptance",
    }:
        raise SchemaMismatchError("history item 字段集合不匹配")
    parameters = payload["parameters"]
    acceptance = payload["acceptance"]
    if not isinstance(parameters, dict) or not isinstance(acceptance, dict):
        raise SchemaMismatchError("history parameters/acceptance 必须是 object")
    if tuple(sorted(acceptance)) != tuple(sorted(_ACCEPTANCE_KEYS)):
        raise SchemaMismatchError("acceptance 字段集合不匹配")
    acceptance_values = []
    for key in _ACCEPTANCE_KEYS:
        value = acceptance[key]
        if not isinstance(value, bool):
            raise SchemaMismatchError(f"acceptance.{key} 必须是 bool")
        acceptance_values.append(value)
    return TrialObservation(
        trial_index=_integer(payload, "trial_index"),
        parameters=_parameters_from_json(parameters),
        objective=_number(payload, "objective"),
        acceptance=TrialAcceptance(*acceptance_values),
    )


def _parameters_to_json(parameters: ParameterVector) -> dict[str, object]:
    return {key: getattr(parameters, key) for key in _PARAMETER_KEYS}


def _parameters_from_json(payload: dict[str, object]) -> ParameterVector:
    if tuple(sorted(payload)) != tuple(sorted(_PARAMETER_KEYS)):
        raise SchemaMismatchError("parameters 字段集合不匹配")
    max_candidates = payload["max_candidates"]
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise SchemaMismatchError("max_candidates 必须是整数")
    return ParameterVector(
        learning_rate=_number(payload, "learning_rate"),
        score_threshold=_number(payload, "score_threshold"),
        max_candidates=max_candidates,
        risk_lambda=_number(payload, "risk_lambda"),
        wait_cost=_number(payload, "wait_cost"),
        min_confidence=_number(payload, "min_confidence"),
    )


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SchemaMismatchError("无法计算搜索状态 canonical JSON") from exc


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
    "SEARCH_STATE_SCHEMA_VERSION",
    "SearchHistoryState",
    "SearchHistoryStore",
    "training_config_sha256",
)
