"""冻结 legacy 的候选几何、评分 oracle 与序列评分行为。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from traning.conf import PerceptionConfig
from traning.core.evaluation import (
    SCORE_VERSION,
    PredictedClick,
    TargetObject,
    score_click_sequence,
    score_point,
    score_slider,
)
from traning.core.perception import (
    DensePerceptionOutput,
    decode_candidates,
)

_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "legacy_golden_v1.json"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _load_fixture() -> dict[str, object]:
    """在 JSON 边界读取固定期望值；生产领域接口不会复用这个宽松类型。"""

    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _logit(probability: float) -> float:
    """把 legacy 概率反解为新 decoder 消费的有限 logit。"""

    return float(torch.logit(torch.tensor(probability)).item())


def _build_perception_output() -> DensePerceptionOutput:
    """构造两个带亚像素偏移的确定性峰值，避免模型权重影响基线。"""

    height, width = 4, 6
    center = torch.full((1, 1, height, width), -20.0, dtype=torch.float32)
    visible = torch.full_like(center, -20.0)
    xy_offset = torch.zeros((1, 2, height, width), dtype=torch.float32)
    object_type_logits = torch.zeros((1, 4, height, width), dtype=torch.float32)
    embedding = torch.ones((1, 3, height, width), dtype=torch.float32)
    peak_specs = (
        (1, 1, 0.9, 0.8, 0.75, 1, 0.25, -0.125, (1.0, 0.0, 0.0)),
        (2, 4, 0.8, 0.9, 0.70, 2, -0.25, 0.375, (0.0, 1.0, 0.0)),
    )
    for (
        row,
        column,
        center_score,
        visible_score,
        type_score,
        type_id,
        dx,
        dy,
        vector,
    ) in peak_specs:
        center[0, 0, row, column] = _logit(center_score)
        visible[0, 0, row, column] = _logit(visible_score)
        probabilities = torch.full((4,), (1.0 - type_score) / 3.0)
        # legacy 两种视觉槽位只用于固定几何；V2 分别通过 ring/slider typed head。
        v2_type_id = 0 if type_id == 1 else 1
        probabilities[v2_type_id] = type_score
        object_type_logits[0, :, row, column] = probabilities.log()
        xy_offset[0, :, row, column] = torch.tensor((dx, dy))
        embedding[0, :, row, column] = torch.tensor(vector)
    scalar_map = torch.zeros_like(center)
    vector_map = torch.ones((1, 2, height, width), dtype=torch.float32)
    return DensePerceptionOutput(
        center_logits=center,
        visibility_logits=visible,
        type_logits=object_type_logits,
        xy_offsets=xy_offset,
        ring_logits=scalar_map.clone(),
        ring_radius=torch.ones_like(center),
        slider_logits=scalar_map.clone(),
        slider_direction=vector_map,
        spinner_logits=scalar_map.clone(),
        identity_embedding=embedding,
    )


def test_legacy_archive_is_frozen() -> None:
    """检测 legacy 冻结包被替换或意外重写。"""

    manifest_path = _REPOSITORY_ROOT / "src/traning/legacy/legacy_freeze.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive_path = _REPOSITORY_ROOT / str(manifest["archive"])
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert digest == manifest["archive_sha256"]


def test_legacy_candidate_geometry_and_recall_match_golden() -> None:
    """固定空间解码的坐标公式、排序和基础 recall。"""

    fixture = _load_fixture()["perception"]
    assert isinstance(fixture, dict)
    candidates = decode_candidates(
        _build_perception_output(),
        frame_id="legacy-golden",
        frame_index=0,
        timestamp_ms=0.0,
        frame_width=192,
        frame_height=128,
        config=PerceptionConfig(
            max_candidates=8,
            score_threshold=0.1,
            nms_radius_px=20.0,
        ),
    )
    expected_geometry = fixture["candidate_geometry"]
    assert isinstance(expected_geometry, list)
    assert len(candidates) / int(fixture["expected_object_count"]) == fixture["recall"]
    assert len(candidates) == len(expected_geometry)
    for candidate, expected in zip(candidates, expected_geometry, strict=True):
        assert isinstance(expected, dict)
        assert candidate.x == pytest.approx(expected["x"])
        assert candidate.y == pytest.approx(expected["y"])
        assert candidate.confidence == pytest.approx(expected["score"], abs=2e-7)


def test_legacy_oracle_matches_golden() -> None:
    """固定点、slider 头部及路径的连续评分语义。"""

    fixture = _load_fixture()
    assert fixture["legacy_score_version"] == SCORE_VERSION
    oracle = fixture["oracle"]
    assert isinstance(oracle, dict)
    expected_point = oracle["point"]
    expected_slider = oracle["slider"]
    assert isinstance(expected_point, dict)
    assert isinstance(expected_slider, dict)

    point = score_point(
        (100.0, 80.0),
        (106.0, 88.0),
        circle_radius=20.0,
        reference_time_ms=1000.0,
        predicted_time_ms=1075.0,
    )
    assert point.distance == pytest.approx(expected_point["distance"])
    assert point.distance_ratio == pytest.approx(expected_point["distance_ratio"])
    assert point.time_error_ms == pytest.approx(expected_point["time_error_ms"])
    assert point.score.spatial == pytest.approx(expected_point["spatial"])
    assert point.score.temporal == pytest.approx(expected_point["temporal"])
    assert point.score.raw == pytest.approx(expected_point["raw"])
    assert point.score.normalized == pytest.approx(expected_point["normalized"])
    assert point.passed is expected_point["passed"]

    slider = score_slider(
        (20.0, 20.0),
        (21.0, 22.0),
        ((20.0, 20.0), (50.0, 20.0), (80.0, 40.0)),
        ((21.0, 22.0), (50.0, 22.0), (79.0, 41.0)),
        circle_radius=16.0,
        reference_start_ms=2000.0,
        predicted_start_ms=2040.0,
    )
    assert slider.head.distance == pytest.approx(expected_slider["head_distance"])
    assert slider.path.reference_coverage == pytest.approx(
        expected_slider["path_reference_coverage"]
    )
    assert slider.path.prediction_precision == pytest.approx(
        expected_slider["path_prediction_precision"]
    )
    assert slider.score.raw == pytest.approx(expected_slider["raw"])
    assert slider.score.normalized == pytest.approx(expected_slider["normalized"])
    assert slider.passed is expected_slider["passed"]


def test_legacy_sequence_score_matches_golden() -> None:
    """固定频率限制、目标消费和最终序列计数。"""

    fixture = _load_fixture()["sequence"]
    assert isinstance(fixture, dict)
    targets = (
        TargetObject("a", "circle", 1000.0, 1000.0, x=100.0, y=80.0, source_index=0),
        TargetObject("b", "circle", 1200.0, 1200.0, x=160.0, y=80.0, source_index=1),
    )
    clicks = (
        PredictedClick(1005.0, 102.0, 82.0),
        PredictedClick(1020.0, 101.0, 81.0),
        PredictedClick(1205.0, 161.0, 79.0),
        PredictedClick(1500.0, 0.0, 0.0),
    )
    result = score_click_sequence(targets, clicks, circle_radius=20.0)
    assert result.hit_count == fixture["hit_count"]
    assert result.miss_count == fixture["miss_count"]
    assert result.frequency_limited_count == fixture["frequency_limited_count"]
    assert list(result.unresolved_target_ids) == fixture["unresolved_target_ids"]
    assert [item.status for item in result.clicks] == fixture["statuses"]
    assert [item.target_id for item in result.clicks] == fixture["target_ids"]
    assert [list(item.error_tags) for item in result.clicks] == fixture["error_tags"]
