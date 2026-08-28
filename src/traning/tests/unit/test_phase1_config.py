"""验证 V2 配置的严格解析、默认值和可重现序列化。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traning.config import (
    CacheConfig,
    CoordinateConfig,
    OptimizationConfig,
    OutcomeConfig,
    V2Config,
    load_v2_config,
    v2_config_to_dict,
)
from traning.data.cache import CANDIDATE_CACHE_SCHEMA_VERSION


def test_default_config_round_trips_through_json(tmp_path) -> None:
    """默认配置经过 JSON 边界后保持同一 typed config。"""

    original = V2Config()
    payload = v2_config_to_dict(original)
    config_path = tmp_path / "v2.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_v2_config(config_path) == original
    assert load_v2_config(payload) == original


def test_cache_config_uses_current_candidate_artifact_schema() -> None:
    """配置边界必须与候选缓存制品 schema 保持同一版本。"""

    config = V2Config()

    assert CANDIDATE_CACHE_SCHEMA_VERSION == 2
    assert CacheConfig().schema_version == CANDIDATE_CACHE_SCHEMA_VERSION
    assert (
        v2_config_to_dict(config)["cache"]["schema_version"]
        == CANDIDATE_CACHE_SCHEMA_VERSION
    )


def test_legacy_candidate_cache_schema_is_rejected() -> None:
    """schema 1 缺少坐标变换指纹，不得被默认升级或接受。"""

    with pytest.raises(ValueError, match="cache.schema_version 仅支持 2"):
        load_v2_config({"schema_version": 1, "cache": {"schema_version": 1}})


def test_unknown_top_level_config_key_is_rejected() -> None:
    """拼错字段不能被静默忽略。"""

    with pytest.raises((KeyError, ValueError, TypeError)):
        load_v2_config({"schema_version": 1, "unknown_section": {}})


def test_unknown_nested_config_key_is_rejected() -> None:
    """嵌套配置也遵循同一个 strict schema。"""

    with pytest.raises((KeyError, ValueError, TypeError)):
        load_v2_config(
            {
                "schema_version": 1,
                "tracking": {"max_distance_px": 64.0, "typo": True},
            }
        )


def test_unsupported_config_schema_is_rejected() -> None:
    """版本不兼容时硬失败，不使用旧默认值掩盖问题。"""

    with pytest.raises((KeyError, ValueError, TypeError)):
        load_v2_config({"schema_version": 999})


def test_outcome_category_count_is_the_canonical_five() -> None:
    """配置不得让模型输出通道与 canonical OutcomeCategory 分叉。"""

    with pytest.raises(ValueError, match="canonical 五分类"):
        OutcomeConfig(category_count=4)


def test_optimization_default_is_unbounded_and_round_trips() -> None:
    """默认不得复现 legacy max_trials=2 导致的提前终止。"""

    config = V2Config(optimization=OptimizationConfig(max_trials=None))
    payload = v2_config_to_dict(config)

    assert payload["optimization"] == {"max_trials": None}
    assert load_v2_config(payload).optimization.max_trials is None
    assert (
        load_v2_config(
            {"schema_version": 1, "optimization": {"max_trials": 7}}
        ).optimization.max_trials
        == 7
    )


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "2"))
def test_optimization_trial_limit_is_strict(value: object) -> None:
    """非法预算不能被静默转换成会提前停止的整数。"""

    with pytest.raises((TypeError, ValueError)):
        load_v2_config({"schema_version": 1, "optimization": {"max_trials": value}})


def test_coordinate_affine_matrix_is_versioned_and_round_trips() -> None:
    """坐标方程必须与原帧尺寸一同进入单一 V2 config。"""

    matrix = (
        (2.115860914627143, 0.0011971920855575358, 242.59057485632047),
        (0.0003418231662923798, 2.1166805757239477, 16.12108357719331),
    )
    config = V2Config(
        coordinates=CoordinateConfig(
            source_width=1484,
            source_height=846,
            transform_identity="legacy-control-validated-v1",
            affine_matrix=matrix,
            calibration_evidence_path=Path("configs/traning_coordinate_evidence.json"),
        )
    )

    restored = load_v2_config(v2_config_to_dict(config))
    assert restored.coordinates == config.coordinates
    assert restored.coordinates.affine_matrix == matrix
    assert (
        restored.coordinates.calibration_evidence_path
        == config.coordinates.calibration_evidence_path
    )


@pytest.mark.parametrize(
    "matrix",
    (
        [[1.0, 0.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[1.0, 0.0, float("nan")], [0.0, 1.0, 0.0]],
    ),
)
def test_coordinate_affine_matrix_rejects_bad_shape_or_values(
    matrix: object,
) -> None:
    """损坏或不可逆的坐标方程不得退回 centered transform。"""

    with pytest.raises((TypeError, ValueError)):
        load_v2_config({"schema_version": 1, "coordinates": {"affine_matrix": matrix}})
