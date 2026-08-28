"""确定性的轨迹与候选关联求解器。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from traning.config import TrackingConfig
from traning.contracts import CandidateObservation, ObjectTypeDistribution


def _require_finite_nonnegative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是数值")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} 必须是有限非负数")


def _require_stable_id(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} 必须是字符串")
    if not value or value != value.strip():
        raise ValueError(f"{name} 不得为空且不得有首尾空格")


@dataclass(frozen=True, slots=True)
class AssociationCostWeights:
    """三类关联成本的分组权重规格。"""

    pixel_distance: float = 0.55
    embedding_distance: float = 0.35
    object_type_distance: float = 0.10

    def __post_init__(self) -> None:
        values = (
            ("pixel_distance", self.pixel_distance),
            ("embedding_distance", self.embedding_distance),
            ("object_type_distance", self.object_type_distance),
        )
        for name, value in values:
            _require_finite_nonnegative(name, value)
        if sum(value for _, value in values) <= 0:
            raise ValueError("关联成本权重总和必须大于 0")

    @property
    def total(self) -> float:
        """返回权重总和，供统一归一化公式使用。"""

        return self.pixel_distance + self.embedding_distance + self.object_type_distance


@dataclass(frozen=True, slots=True)
class AssociationCost:
    """单个轨迹候选配对的可解释成本。"""

    pixel_distance: float
    embedding_distance: float
    object_type_distance: float
    total: float
    confidence: float

    def __post_init__(self) -> None:
        for name, value in (
            ("pixel_distance", self.pixel_distance),
            ("embedding_distance", self.embedding_distance),
            ("object_type_distance", self.object_type_distance),
            ("total", self.total),
            ("confidence", self.confidence),
        ):
            _require_finite_nonnegative(name, value)
        if self.object_type_distance > 1:
            raise ValueError("object_type_distance 不得大于 1")
        if self.confidence > 1:
            raise ValueError("confidence 不得大于 1")


@dataclass(frozen=True, slots=True)
class TrackAssociationView:
    """关联层可见的最小轨迹投影，不携带训练真值。"""

    track_id: str
    last_candidate: CandidateObservation
    missed_frames: int

    def __post_init__(self) -> None:
        _require_stable_id("track_id", self.track_id)
        if not isinstance(self.last_candidate, CandidateObservation):
            raise TypeError("last_candidate 必须是 CandidateObservation")
        if isinstance(self.missed_frames, bool) or not isinstance(
            self.missed_frames, int
        ):
            raise TypeError("missed_frames 必须是整数")
        if self.missed_frames < 0:
            raise ValueError("missed_frames 不得为负数")


@dataclass(frozen=True, slots=True)
class AssociationMatch:
    """一个已接受的轨迹候选配对。"""

    track_id: str
    candidate_id: str
    cost: AssociationCost
    confidence: float

    def __post_init__(self) -> None:
        _require_stable_id("track_id", self.track_id)
        _require_stable_id("candidate_id", self.candidate_id)
        if not isinstance(self.cost, AssociationCost):
            raise TypeError("cost 必须是 AssociationCost")
        _require_finite_nonnegative("confidence", self.confidence)
        if self.confidence > 1:
            raise ValueError("confidence 不得大于 1")
        if not math.isclose(self.confidence, self.cost.confidence, abs_tol=1e-12):
            raise ValueError("match confidence 必须与 cost.confidence 一致")


@dataclass(frozen=True, slots=True)
class AssociationResult:
    """一次一对一关联的完整确定性结果。"""

    matches: tuple[AssociationMatch, ...]
    unmatched_track_ids: tuple[str, ...]
    unmatched_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.matches, tuple) or any(
            not isinstance(match, AssociationMatch) for match in self.matches
        ):
            raise TypeError("matches 必须是 AssociationMatch 元组")
        for name, identifiers in (
            ("unmatched_track_ids", self.unmatched_track_ids),
            ("unmatched_candidate_ids", self.unmatched_candidate_ids),
        ):
            if not isinstance(identifiers, tuple):
                raise TypeError(f"{name} 必须是字符串元组")
            for identifier in identifiers:
                _require_stable_id(name, identifier)
            if identifiers != tuple(sorted(identifiers)):
                raise ValueError(f"{name} 必须按稳定 ID 排序")
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{name} 不得含重复 ID")
        matched_tracks = tuple(match.track_id for match in self.matches)
        matched_candidates = tuple(match.candidate_id for match in self.matches)
        if len(matched_tracks) != len(set(matched_tracks)):
            raise ValueError("同一 track_id 最多匹配一次")
        if len(matched_candidates) != len(set(matched_candidates)):
            raise ValueError("同一 candidate_id 最多匹配一次")
        if set(matched_tracks) & set(self.unmatched_track_ids):
            raise ValueError("已匹配轨迹不得同时出现在 unmatched_track_ids")
        if set(matched_candidates) & set(self.unmatched_candidate_ids):
            raise ValueError("已匹配候选不得同时出现在 unmatched_candidate_ids")


@dataclass(frozen=True, slots=True)
class _CandidatePair:
    """greedy 求解器内部的候选配对。"""

    track_id: str
    candidate_id: str
    cost: AssociationCost


def _cosine_distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    if len(first) != len(second):
        raise ValueError("关联 embedding 维度必须一致")
    if not first:
        raise ValueError("关联 embedding 不得为空")
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm <= 0 or second_norm <= 0:
        raise ValueError("关联 embedding 的 L2 范数必须大于 0")
    similarity = sum(left * right for left, right in zip(first, second, strict=True))
    similarity /= first_norm * second_norm
    return (1.0 - min(1.0, max(-1.0, similarity))) / 2.0


def _type_values(distribution: ObjectTypeDistribution) -> tuple[float, ...]:
    return (
        distribution.p_ring,
        distribution.p_slider,
        distribution.p_spinner,
        distribution.p_unknown,
    )


def _object_type_distance(
    first: ObjectTypeDistribution,
    second: ObjectTypeDistribution,
) -> float:
    """用 total variation distance 比较两个四分类分布。"""

    return 0.5 * sum(
        abs(left - right)
        for left, right in zip(_type_values(first), _type_values(second), strict=True)
    )


@dataclass(frozen=True, slots=True)
class AssociationCostSpec:
    """门限与权重的单一关联成本规格。"""

    max_pixel_distance: float
    max_embedding_distance: float
    min_confidence: float
    max_total_cost: float
    weights: AssociationCostWeights = AssociationCostWeights()

    def __post_init__(self) -> None:
        _require_finite_nonnegative("max_pixel_distance", self.max_pixel_distance)
        _require_finite_nonnegative(
            "max_embedding_distance", self.max_embedding_distance
        )
        _require_finite_nonnegative("min_confidence", self.min_confidence)
        _require_finite_nonnegative("max_total_cost", self.max_total_cost)
        if self.min_confidence > 1:
            raise ValueError("min_confidence 不得大于 1")
        if self.max_total_cost > 1:
            raise ValueError("max_total_cost 不得大于 1")
        if not isinstance(self.weights, AssociationCostWeights):
            raise TypeError("weights 必须是 AssociationCostWeights")

    @classmethod
    def from_config(
        cls,
        config: TrackingConfig,
    ) -> AssociationCostSpec:
        """只从 TrackingConfig 读取运行门限。"""

        if not isinstance(config, TrackingConfig):
            raise TypeError("config 必须是 TrackingConfig")
        return cls(
            max_pixel_distance=config.max_distance_px,
            max_embedding_distance=config.max_embedding_distance,
            min_confidence=config.min_association_confidence,
            max_total_cost=config.max_total_cost,
            weights=AssociationCostWeights(
                pixel_distance=config.spatial_weight,
                embedding_distance=config.embedding_weight,
                object_type_distance=config.type_weight,
            ),
        )

    def cost(
        self,
        track: TrackAssociationView,
        candidate: CandidateObservation,
    ) -> AssociationCost | None:
        """计算合法配对成本；超过门限返回 None。"""

        previous = track.last_candidate
        pixel_distance = math.hypot(candidate.x - previous.x, candidate.y - previous.y)
        embedding_distance = _cosine_distance(
            previous.appearance_embedding,
            candidate.appearance_embedding,
        )
        if (
            pixel_distance > self.max_pixel_distance
            or embedding_distance > self.max_embedding_distance
        ):
            return None
        type_distance = _object_type_distance(
            previous.object_type_distribution,
            candidate.object_type_distribution,
        )
        normalized_total = (
            self.weights.pixel_distance
            * _normalized_distance(pixel_distance, self.max_pixel_distance)
            + self.weights.embedding_distance
            * _normalized_distance(embedding_distance, self.max_embedding_distance)
            + self.weights.object_type_distance * type_distance
        ) / self.weights.total
        confidence = max(0.0, 1.0 - normalized_total)
        if normalized_total > self.max_total_cost or confidence < self.min_confidence:
            return None
        return AssociationCost(
            pixel_distance=pixel_distance,
            embedding_distance=embedding_distance,
            object_type_distance=type_distance,
            total=normalized_total,
            confidence=confidence,
        )


@dataclass(frozen=True, slots=True)
class GreedyAssociationSolver:
    """按稳定总序执行一对一 greedy 关联。"""

    config: TrackingConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, TrackingConfig):
            raise TypeError("config 必须是 TrackingConfig")

    def solve(
        self,
        tracks: Sequence[TrackAssociationView],
        candidates: Sequence[CandidateObservation],
    ) -> AssociationResult:
        """生成全部合法配对，再以成本和稳定 ID 决胜。"""

        checked_tracks = _checked_tracks(tracks)
        checked_candidates = _checked_candidates(candidates)
        embedding_dimensions = {
            len(track.last_candidate.appearance_embedding) for track in checked_tracks
        } | {len(candidate.appearance_embedding) for candidate in checked_candidates}
        if len(embedding_dimensions) > 1:
            raise ValueError("所有轨迹和候选的 embedding 维度必须一致")
        spec = AssociationCostSpec.from_config(self.config)
        pairs: list[_CandidatePair] = []
        for track in checked_tracks:
            if track.missed_frames > self.config.max_missed_frames:
                continue
            for candidate in checked_candidates:
                cost = spec.cost(track, candidate)
                if cost is not None:
                    pairs.append(
                        _CandidatePair(
                            track_id=track.track_id,
                            candidate_id=candidate.candidate_id,
                            cost=cost,
                        )
                    )
        pairs.sort(key=lambda pair: (pair.cost.total, pair.track_id, pair.candidate_id))

        matched_tracks: set[str] = set()
        matched_candidates: set[str] = set()
        matches: list[AssociationMatch] = []
        for pair in pairs:
            if (
                pair.track_id in matched_tracks
                or pair.candidate_id in matched_candidates
            ):
                continue
            matched_tracks.add(pair.track_id)
            matched_candidates.add(pair.candidate_id)
            matches.append(
                AssociationMatch(
                    track_id=pair.track_id,
                    candidate_id=pair.candidate_id,
                    cost=pair.cost,
                    confidence=pair.cost.confidence,
                )
            )
        return AssociationResult(
            matches=tuple(matches),
            unmatched_track_ids=tuple(
                sorted(
                    track.track_id
                    for track in checked_tracks
                    if track.track_id not in matched_tracks
                )
            ),
            unmatched_candidate_ids=tuple(
                sorted(
                    candidate.candidate_id
                    for candidate in checked_candidates
                    if candidate.candidate_id not in matched_candidates
                )
            ),
        )


def _checked_tracks(
    tracks: Sequence[TrackAssociationView],
) -> tuple[TrackAssociationView, ...]:
    if isinstance(tracks, (str, bytes)) or not isinstance(tracks, Sequence):
        raise TypeError("tracks 必须是 TrackAssociationView 序列")
    checked = tuple(tracks)
    if any(not isinstance(track, TrackAssociationView) for track in checked):
        raise TypeError("tracks 只能包含 TrackAssociationView")
    identifiers = tuple(track.track_id for track in checked)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("tracks 不得包含重复 track_id")
    return tuple(sorted(checked, key=lambda track: track.track_id))


def _normalized_distance(distance: float, maximum: float) -> float:
    """支持零门限：只有精确零距离能通过并贡献零成本。"""

    if maximum == 0:
        return 0.0
    return distance / maximum


def _checked_candidates(
    candidates: Sequence[CandidateObservation],
) -> tuple[CandidateObservation, ...]:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates 必须是 CandidateObservation 序列")
    checked = tuple(candidates)
    if any(not isinstance(candidate, CandidateObservation) for candidate in checked):
        raise TypeError("candidates 只能包含 CandidateObservation")
    identifiers = tuple(candidate.candidate_id for candidate in checked)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidates 不得包含重复 candidate_id")
    return tuple(sorted(checked, key=lambda candidate: candidate.candidate_id))


def associate_candidates(
    tracks: Sequence[TrackAssociationView],
    candidates: Sequence[CandidateObservation],
    config: TrackingConfig,
) -> AssociationResult:
    """确定性 greedy 关联的函数式入口。"""

    return GreedyAssociationSolver(config=config).solve(tracks, candidates)


__all__ = (
    "AssociationCost",
    "AssociationCostSpec",
    "AssociationCostWeights",
    "AssociationMatch",
    "AssociationResult",
    "GreedyAssociationSolver",
    "TrackAssociationView",
    "associate_candidates",
)
