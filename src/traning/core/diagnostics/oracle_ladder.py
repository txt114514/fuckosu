from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import hypot
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageDraw

from package.coordinates import OSU_PLAYFIELD_HEIGHT, OSU_PLAYFIELD_WIDTH
from traning.conf import Settings
from traning.core.dataset_import import build_dataset
from traning.core.optimization import score_decision_outputs
from traning.core.temporal import TemporalCandidateWindowDataset
from traning.lib.coordinates import transform_from_settings_or_sample


SAMPLER_CONTRACTS: dict[str, dict[str, str]] = {
    "SpatialSampler": {
        "purpose": "spatial smoke or calibration inspection",
        "strategy": "seeded diverse samples; temporal continuity not required",
    },
    "CandidateCacheSampler": {
        "purpose": "candidate generation used by temporal training",
        "strategy": "seeded group selection with contiguous in-group frame blocks",
    },
    "TemporalWindowSampler": {
        "purpose": "causal temporal training windows",
        "strategy": "same-group windows sorted by frame_index/timestamp; no cross-group windows",
    },
    "EvaluationSampler": {
        "purpose": "score comparison across trials",
        "strategy": "persistent fixed_evaluation_manifest.json",
    },
    "GallerySampler": {
        "purpose": "human inspection only",
        "strategy": "independent seeded visual sampler; never changes score membership",
    },
}


@dataclass(frozen=True)
class OracleDiagnosticsResult:
    output_dir: Path
    report_path: Path
    fixed_evaluation_manifest_path: Path
    coordinate_probe_gallery_path: Path | None
    first_error_stage: str
    oracle_gt_score: float
    oracle_target_roundtrip_score: float
    oracle_candidate_score: float
    oracle_temporal_slots_score: float
    actual_score: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "report_path": self.report_path,
            "fixed_evaluation_manifest_path": self.fixed_evaluation_manifest_path,
            "coordinate_probe_gallery_path": self.coordinate_probe_gallery_path,
            "first_error_stage": self.first_error_stage,
            "oracle_gt_score": self.oracle_gt_score,
            "oracle_target_roundtrip_score": self.oracle_target_roundtrip_score,
            "oracle_candidate_score": self.oracle_candidate_score,
            "oracle_temporal_slots_score": self.oracle_temporal_slots_score,
            "actual_score": self.actual_score,
        }


def run_oracle_diagnostics(
    settings: Settings,
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    fixed_seed: int = 2026,
    max_fixed_frames: int = 128,
    probe_limit: int = 12,
) -> OracleDiagnosticsResult:
    selected_output_dir = output_dir or run_dir / "diagnostics" / "oracle_ladder"
    selected_output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "candidate_cache"
    candidate_records = _load_candidate_cache(cache_dir)
    cache_records_path = _candidate_records_path(cache_dir)
    decision_path = run_dir / "decision" / "decisions.jsonl"
    decision_rows = _read_jsonl(decision_path) if decision_path.is_file() else ()

    fixed_manifest = _build_fixed_evaluation_manifest(
        candidate_records,
        seed=fixed_seed,
        max_frames=max_fixed_frames,
    )
    fixed_manifest_path = selected_output_dir / "fixed_evaluation_manifest.json"
    _write_json(fixed_manifest_path, fixed_manifest)

    oracle_gt_path = selected_output_dir / "oracle_gt_decisions.jsonl"
    _write_jsonl(oracle_gt_path, _oracle_decisions(candidate_records, mode="gt"))
    oracle_gt = score_decision_outputs(
        parameter_group_id="oracle_gt",
        candidate_cache_path=cache_records_path,
        decisions_path=oracle_gt_path,
        settings=settings,
    )

    oracle_roundtrip_path = selected_output_dir / "oracle_target_roundtrip_decisions.jsonl"
    _write_jsonl(
        oracle_roundtrip_path,
        _oracle_decisions(candidate_records, mode="target_roundtrip", settings=settings),
    )
    oracle_roundtrip = score_decision_outputs(
        parameter_group_id="oracle_target_roundtrip",
        candidate_cache_path=cache_records_path,
        decisions_path=oracle_roundtrip_path,
        settings=settings,
    )

    oracle_candidate_path = selected_output_dir / "oracle_candidate_decisions.jsonl"
    _write_jsonl(
        oracle_candidate_path,
        _oracle_decisions(candidate_records, mode="matched_candidate"),
    )
    oracle_candidate = score_decision_outputs(
        parameter_group_id="oracle_candidate",
        candidate_cache_path=cache_records_path,
        decisions_path=oracle_candidate_path,
        settings=settings,
    )
    temporal_candidate_slots = _temporal_candidate_slots(run_dir, settings)
    oracle_temporal_slots_path = (
        selected_output_dir / "oracle_temporal_slots_decisions.jsonl"
    )
    _write_jsonl(
        oracle_temporal_slots_path,
        _oracle_decisions(
            candidate_records,
            mode="temporal_slots",
            candidate_slots=temporal_candidate_slots,
        ),
    )
    oracle_temporal_slots = score_decision_outputs(
        parameter_group_id="oracle_temporal_slots",
        candidate_cache_path=cache_records_path,
        decisions_path=oracle_temporal_slots_path,
        settings=settings,
    )

    actual_score = None
    if decision_path.is_file():
        actual_score = score_decision_outputs(
            parameter_group_id=run_dir.name,
            candidate_cache_path=cache_records_path,
            decisions_path=decision_path,
            settings=settings,
        )

    coordinate_probe = _write_coordinate_probe_gallery(
        settings,
        candidate_records,
        decision_rows,
        output_dir=selected_output_dir / "coordinate_probe_gallery",
        probe_limit=probe_limit,
        split=str(_cache_manifest(cache_dir).get("split") or "train"),
    )
    report = {
        "run_dir": str(run_dir),
        "sampler_contracts": SAMPLER_CONTRACTS,
        "fixed_evaluation_manifest": str(fixed_manifest_path),
        "oracle_ladder": {
            "gt_to_evaluator": oracle_gt.as_summary(),
            "target_roundtrip_to_evaluator": oracle_roundtrip.as_summary(),
            "matched_candidate_to_evaluator": oracle_candidate.as_summary(),
            "temporal_slot_candidate_to_evaluator": (
                oracle_temporal_slots.as_summary()
            ),
            "actual_decision_to_evaluator": (
                None if actual_score is None else actual_score.as_summary()
            ),
        },
        "candidate_recall": _candidate_recall(candidate_records),
        "target_assignment": _target_assignment(
            candidate_records,
            candidate_slots=temporal_candidate_slots,
        ),
        "temporal_continuity": _temporal_continuity(settings, cache_dir),
        "decision_diagnostics": (
            None
            if not decision_rows
            else _decision_diagnostics(candidate_records, decision_rows, settings=settings)
        ),
        "loss_audit": {
            "coordinate_loss_mask": "action_frames_only",
            "candidate_loss_mask": "frames_with_selected_candidate_target",
            "source": "traning.core.temporal.trainer._compute_temporal_loss",
        },
        "coordinate_probe": coordinate_probe,
    }
    report["first_error_stage"] = _first_error_stage(report)
    report_path = selected_output_dir / "oracle_ladder_report.json"
    _write_json(report_path, report)
    return OracleDiagnosticsResult(
        output_dir=selected_output_dir,
        report_path=report_path,
        fixed_evaluation_manifest_path=fixed_manifest_path,
        coordinate_probe_gallery_path=(
            None
            if coordinate_probe.get("status") != "saved"
            else selected_output_dir / "coordinate_probe_gallery"
        ),
        first_error_stage=str(report["first_error_stage"]),
        oracle_gt_score=float(oracle_gt.report.quality_score),
        oracle_target_roundtrip_score=float(oracle_roundtrip.report.quality_score),
        oracle_candidate_score=float(oracle_candidate.report.quality_score),
        oracle_temporal_slots_score=float(oracle_temporal_slots.report.quality_score),
        actual_score=(
            None if actual_score is None else float(actual_score.report.quality_score)
        ),
    )


def _load_candidate_cache(cache_dir: Path) -> tuple[dict[str, Any], ...]:
    records_path = _candidate_records_path(cache_dir)
    return tuple(_read_jsonl(records_path))


def _candidate_records_path(cache_dir: Path) -> Path:
    manifest = _cache_manifest(cache_dir)
    records_name = manifest.get("records")
    if not isinstance(records_name, str) or not records_name:
        raise ValueError("candidate cache manifest must contain records filename")
    return cache_dir / records_name


def _cache_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"candidate cache manifest missing: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _oracle_decisions(
    records: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    settings: Settings | None = None,
    candidate_slots: int | None = None,
) -> tuple[dict[str, Any], ...]:
    decisions: list[dict[str, Any]] = []
    for record in records:
        target = _target(record)
        action = str(target.get("action") or "no_op")
        row: dict[str, Any] = {
            "version": "temporal-decision-v1",
            "sample_key": record.get("sample_key"),
            "frame_index": record.get("frame_index"),
            "timestamp_ms": record.get("timestamp_ms"),
            "action": action,
            "action_id": int(target.get("action_id") or 0),
            "action_probability": 1.0,
            "selected_candidate_id": None,
            "selected_candidate_probability": None,
            "predicted_xy_normalized": [0.0, 0.0],
            "time_offset_ms": float(target.get("time_offset_ms") or 0.0),
            "diagnostics": {"oracle_mode": mode},
        }
        if action == "no_op":
            decisions.append(row)
            continue
        if mode == "matched_candidate":
            row["selected_candidate_id"] = target.get("selected_candidate_id")
        elif mode == "temporal_slots":
            selected_id = _safe_int(target.get("selected_candidate_id"))
            row["selected_candidate_id"] = (
                selected_id
                if selected_id is not None
                and _candidate_id_in_top_slots(record, selected_id, candidate_slots)
                else None
            )
        elif mode == "target_roundtrip":
            osu_xy = _roundtrip_target_osu(record, settings=settings)
            if osu_xy is not None:
                row["predicted_xy_normalized"] = [
                    osu_xy[0] / OSU_PLAYFIELD_WIDTH,
                    osu_xy[1] / OSU_PLAYFIELD_HEIGHT,
                ]
        else:
            osu_xy = _point_pair(target.get("target_osu_xy"))
            if osu_xy is not None:
                row["predicted_xy_normalized"] = [
                    osu_xy[0] / OSU_PLAYFIELD_WIDTH,
                    osu_xy[1] / OSU_PLAYFIELD_HEIGHT,
                ]
        decisions.append(row)
    return tuple(decisions)


def _roundtrip_target_osu(
    record: Mapping[str, Any],
    *,
    settings: Settings | None,
) -> tuple[float, float] | None:
    target_video = _point_pair(_target(record).get("target_video_xy"))
    if target_video is None:
        return _point_pair(_target(record).get("target_osu_xy"))
    transform, _ = transform_from_settings_or_sample(
        settings,
        record,
        frame_width=_safe_int(record.get("frame_width")),
        frame_height=_safe_int(record.get("frame_height")),
    )
    return transform.video_to_osu(*target_video)


def _build_fixed_evaluation_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    max_frames: int,
) -> dict[str, Any]:
    selected = _fixed_eval_rows(records, seed=seed, max_frames=max_frames)
    return {
        "version": "fixed-evaluation-manifest-v1",
        "seed": seed,
        "sampler": "EvaluationSampler",
        "strategy": "scene_stratified_group_and_time_coverage",
        "frame_count": len(selected),
        "records": [
            {
                "sample_id": _frame_id(record),
                "group_id": str(record.get("sample_key") or ""),
                "segment_id": _segment_id(record),
                "frame_index": _safe_int(record.get("frame_index")),
                "timestamp": _safe_float(record.get("timestamp_ms")),
                "scene_type": _scene_type(record),
                "target_type": _target_type(record),
            }
            for record in selected
        ],
    }


def _fixed_eval_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    max_frames: int,
) -> tuple[Mapping[str, Any], ...]:
    rng = random.Random(seed)
    by_scene: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_scene[_scene_type(record)][str(record.get("sample_key") or "")].append(record)
    selected: list[Mapping[str, Any]] = []
    scenes = sorted(by_scene)
    rng.shuffle(scenes)
    while len(selected) < max_frames and scenes:
        progressed = False
        for scene in tuple(scenes):
            groups = by_scene[scene]
            group_ids = sorted(groups)
            rng.shuffle(group_ids)
            for group_id in group_ids:
                rows = sorted(groups[group_id], key=_record_time_key)
                if not rows:
                    continue
                index = _coverage_index(rows, len(selected))
                selected.append(rows.pop(index))
                groups[group_id] = rows
                progressed = True
                break
            if len(selected) >= max_frames:
                break
        if not progressed:
            break
    return tuple(selected)


def _coverage_index(rows: Sequence[Mapping[str, Any]], offset: int) -> int:
    if len(rows) <= 2:
        return 0
    choices = (0, len(rows) // 2, len(rows) - 1)
    return choices[offset % len(choices)]


def _candidate_recall(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = _empty_recall_bucket()
    by_scene: dict[str, dict[str, Any]] = defaultdict(_empty_recall_bucket)
    for record in records:
        target = _target(record)
        if str(target.get("action") or "no_op") == "no_op":
            continue
        distances = _candidate_distances(record)
        nearest = min(distances) if distances else None
        for bucket in (totals, by_scene[_scene_type(record)]):
            bucket["target_frames"] += 1
            if nearest is None:
                bucket["no_candidate_frames"] += 1
                continue
            bucket["nearest_distances_px"].append(nearest)
            for radius in (8, 16, 32):
                if nearest <= radius:
                    bucket[f"hits@{radius}px"] += 1
    return _finalize_recall(totals) | {
        "by_scene": {
            scene: _finalize_recall(bucket) for scene, bucket in sorted(by_scene.items())
        }
    }


def _empty_recall_bucket() -> dict[str, Any]:
    return {
        "target_frames": 0,
        "no_candidate_frames": 0,
        "hits@8px": 0,
        "hits@16px": 0,
        "hits@32px": 0,
        "nearest_distances_px": [],
    }


def _finalize_recall(bucket: Mapping[str, Any]) -> dict[str, Any]:
    target_frames = int(bucket["target_frames"])
    distances = [float(value) for value in bucket["nearest_distances_px"]]
    return {
        "target_frames": target_frames,
        "no_candidate_frames": int(bucket["no_candidate_frames"]),
        "recall@8px": _rate(int(bucket["hits@8px"]), target_frames),
        "recall@16px": _rate(int(bucket["hits@16px"]), target_frames),
        "recall@32px": _rate(int(bucket["hits@32px"]), target_frames),
        "nearest_distance_mean_px": _mean(distances),
        "nearest_distance_median_px": _percentile(distances, 50),
        "nearest_distance_p90_px": _percentile(distances, 90),
    }


def _target_assignment(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_slots: int | None,
) -> dict[str, Any]:
    target_frames = 0
    matched = 0
    nearest_within_radius = 0
    selected_not_nearest = 0
    unstable_ids = 0
    selected_in_temporal_slots = 0
    selected_in_raw_top_slots = 0
    examples: list[dict[str, Any]] = []
    for record in records:
        target = _target(record)
        if str(target.get("action") or "no_op") == "no_op":
            continue
        target_frames += 1
        radius = _safe_float(target.get("candidate_match_radius_px")) or 64.0
        selected_id = _safe_int(target.get("selected_candidate_id"))
        candidates = _candidate_rows(record)
        ids = [_safe_int(candidate.get("candidate_id")) for candidate in candidates]
        if len([item for item in ids if item is not None]) != len(set(item for item in ids if item is not None)):
            unstable_ids += 1
        distances = _candidate_distances(record)
        nearest_index = distances.index(min(distances)) if distances else None
        nearest_id = (
            None if nearest_index is None else _safe_int(candidates[nearest_index].get("candidate_id"))
        )
        nearest_distance = None if nearest_index is None else distances[nearest_index]
        if nearest_distance is not None and nearest_distance <= radius:
            nearest_within_radius += 1
        if selected_id is not None:
            matched += 1
            if _candidate_id_in_top_slots(record, selected_id, candidate_slots):
                selected_in_temporal_slots += 1
            if _candidate_id_in_raw_top_slots(record, selected_id, candidate_slots):
                selected_in_raw_top_slots += 1
            if nearest_id is not None and selected_id != nearest_id:
                selected_not_nearest += 1
        if len(examples) < 12:
            examples.append(
                {
                    "sample_id": _frame_id(record),
                    "scene_type": _scene_type(record),
                    "action": target.get("action"),
                    "selected_candidate_id": selected_id,
                    "nearest_candidate_id": nearest_id,
                    "nearest_distance_px": nearest_distance,
                    "selected_in_temporal_slots": (
                        None
                        if selected_id is None
                        else _candidate_id_in_top_slots(record, selected_id, candidate_slots)
                    ),
                    "selected_in_raw_top_slots": (
                        None
                        if selected_id is None
                        else _candidate_id_in_raw_top_slots(record, selected_id, candidate_slots)
                    ),
                    "match_status": target.get("candidate_match_status"),
                    "unmatched_reason": target.get("candidate_match_unmatched_reason"),
                }
            )
    return {
        "target_frames": target_frames,
        "matched_candidate_frames": matched,
        "unmatched_target_frames": target_frames - matched,
        "nearest_within_config_radius_frames": nearest_within_radius,
        "selected_not_nearest_frames": selected_not_nearest,
        "temporal_candidate_slots": candidate_slots,
        "selected_candidate_in_temporal_slots_frames": selected_in_temporal_slots,
        "selected_candidate_in_raw_top_slots_frames": selected_in_raw_top_slots,
        "matched_but_truncated_by_raw_top_slots_frames": matched - selected_in_raw_top_slots,
        "duplicate_or_missing_candidate_id_frames": unstable_ids,
        "match_rate": _rate(matched, target_frames),
        "temporal_slot_match_rate": _rate(
            selected_in_temporal_slots,
            target_frames,
        ),
        "examples": examples,
    }


def _temporal_continuity(settings: Settings, cache_dir: Path) -> dict[str, Any]:
    sequence_length = max(1, int(settings.temporal.history_frames))
    candidate_slots = max(1, int(settings.candidate_cache.max_candidates_per_frame))
    dataset = TemporalCandidateWindowDataset.from_cache_dir(
        cache_dir,
        sequence_length=sequence_length,
        candidate_slots=candidate_slots,
    )
    windows = []
    discontinuous = 0
    for index in range(min(len(dataset), 24)):
        window = dataset[index]
        keys = tuple(key for key in window.sample_keys if key is not None)
        frame_indices = tuple(
            int(value) for value in window.frame_indices if value is not None
        )
        timestamps = tuple(
            float(value) for value in window.timestamps_ms if value is not None
        )
        same_group = len(set(keys)) <= 1
        full_length = len(frame_indices) == sequence_length
        consecutive = all(
            b == a + 1 for a, b in zip(frame_indices, frame_indices[1:])
        )
        timestamp_monotonic = all(
            b >= a for a, b in zip(timestamps, timestamps[1:])
        )
        if not (same_group and full_length and consecutive and timestamp_monotonic):
            discontinuous += 1
        windows.append(
            {
                "group_id": keys[0] if keys else None,
                "sample_keys": keys,
                "frame_indices": frame_indices,
                "timestamps": timestamps,
                "valid_frame_count": len(frame_indices),
                "full_length": full_length,
                "same_group": same_group,
                "consecutive_frame_indices": consecutive,
                "monotonic_timestamps": timestamp_monotonic,
            }
        )
    return {
        "sampler": "TemporalWindowSampler",
        "window_count": len(dataset),
        "sequence_length": sequence_length,
        "sampled_windows": windows,
        "sampled_discontinuous_windows": discontinuous,
        "sampled_windows_continuous": discontinuous == 0,
    }


def _decision_diagnostics(
    records: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    *,
    settings: Settings,
) -> dict[str, Any]:
    cache_by_key = {_frame_key(record): record for record in records}
    tp = fp = fn = tn = 0
    coordinate_errors: list[float] = []
    target_actions = 0
    predicted_actions = 0
    no_op_logits: list[float] = []
    action_logits: list[float] = []
    by_scene: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for decision in decisions:
        record = cache_by_key.get(_frame_key(decision))
        if record is None:
            continue
        target_action = str(_target(record).get("action") or "no_op") != "no_op"
        predicted_action = str(decision.get("action") or "no_op") != "no_op"
        scene = _scene_type(record)
        if target_action:
            target_actions += 1
        if predicted_action:
            predicted_actions += 1
        if target_action and predicted_action:
            tp += 1
            by_scene[scene]["tp"] += 1
        elif not target_action and predicted_action:
            fp += 1
            by_scene[scene]["fp"] += 1
        elif target_action and not predicted_action:
            fn += 1
            by_scene[scene]["fn"] += 1
        else:
            tn += 1
            by_scene[scene]["tn"] += 1
        probabilities = ((decision.get("diagnostics") or {}).get("action_probabilities") or {})
        if isinstance(probabilities, Mapping):
            no_op = _safe_float(probabilities.get("no_op"))
            if no_op is not None:
                no_op_logits.append(no_op)
            action_values = [
                _safe_float(probabilities.get(name))
                for name in ("press", "hold", "release")
            ]
            action_values = [value for value in action_values if value is not None]
            if action_values:
                action_logits.append(max(action_values))
        error = _decision_coordinate_error(record, decision, settings=settings)
        if error is not None:
            coordinate_errors.append(error)
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    frame_count = len(decisions)
    return {
        "target_action_frames": target_actions,
        "predicted_action_frames": predicted_actions,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "mean_coordinate_error": _mean(coordinate_errors),
        "median_coordinate_error": _percentile(coordinate_errors, 50),
        "p90_coordinate_error": _percentile(coordinate_errors, 90),
        "action_no_op_target_ratio": {
            "action": _rate(target_actions, frame_count),
            "no_op": _rate(frame_count - target_actions, frame_count),
        },
        "action_no_op_prediction_ratio": {
            "action": _rate(predicted_actions, frame_count),
            "no_op": _rate(frame_count - predicted_actions, frame_count),
        },
        "action_probability_distribution": {
            "no_op_mean": _mean(no_op_logits),
            "max_action_mean": _mean(action_logits),
        },
        "by_scene": {
            scene: values | {
                "precision": _rate(values["tp"], values["tp"] + values["fp"]),
                "recall": _rate(values["tp"], values["tp"] + values["fn"]),
            }
            for scene, values in sorted(by_scene.items())
        },
    }


def _decision_coordinate_error(
    record: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    settings: Settings,
) -> float | None:
    target_xy = _point_pair(_target(record).get("target_video_xy"))
    if target_xy is None:
        return None
    predicted = _decision_video_xy(record, decision, settings=settings)
    if predicted is None:
        return None
    return hypot(predicted[0] - target_xy[0], predicted[1] - target_xy[1])


def _write_coordinate_probe_gallery(
    settings: Settings,
    records: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    probe_limit: int,
    split: str,
) -> dict[str, Any]:
    selected = _select_probe_records(records, limit=probe_limit)
    if not selected:
        return {"status": "skipped", "reason": "no target records"}
    dataset = build_dataset(settings, split=split)  # type: ignore[arg-type]
    index_by_key = {
        (record.key, reference.frame_index): index
        for index, reference in enumerate(dataset.references)
        for record in (dataset.records[reference.record_index],)
    }
    decision_by_key = {_frame_key(decision): decision for decision in decisions}
    output_dir.mkdir(parents=True, exist_ok=True)
    probes: list[dict[str, Any]] = []
    for probe_index, record in enumerate(selected):
        dataset_index = index_by_key.get(
            (str(record.get("sample_key") or ""), int(record.get("frame_index") or -1))
        )
        if dataset_index is None:
            continue
        sample = dataset[dataset_index]
        image = _sample_image(sample["image"])
        draw = ImageDraw.Draw(image)
        probe = _probe_points(
            record,
            decision_by_key.get(_frame_key(record)),
            settings=settings,
        )
        _draw_probe(draw, probe)
        filename = f"probe_{probe_index:03d}__{_safe_name(_frame_id(record))}.png"
        image.save(output_dir / filename)
        probes.append(probe | {"image": filename})
    manifest = {
        "status": "saved" if probes else "skipped",
        "probe_count": len(probes),
        "probes": probes,
        "error_analysis": _coordinate_error_analysis(probes),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _select_probe_records(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> tuple[Mapping[str, Any], ...]:
    targets = [
        record
        for record in records
        if str(_target(record).get("action") or "no_op") != "no_op"
        and _point_pair(_target(record).get("target_osu_xy")) is not None
    ]
    selected: list[Mapping[str, Any]] = []
    anchors = {
        "center": (OSU_PLAYFIELD_WIDTH / 2.0, OSU_PLAYFIELD_HEIGHT / 2.0),
        "top_left": (0.0, 0.0),
        "top_right": (OSU_PLAYFIELD_WIDTH, 0.0),
        "bottom_left": (0.0, OSU_PLAYFIELD_HEIGHT),
        "bottom_right": (OSU_PLAYFIELD_WIDTH, OSU_PLAYFIELD_HEIGHT),
    }
    for label, point in anchors.items():
        record = min(
            targets,
            key=lambda item: _osu_distance(item, point),
            default=None,
        )
        if record is not None:
            selected.append(record | {"probe_reason": label})
    for scene in ("single_point", "slider", "point_slider", "long_sequence", "dense_hard"):
        record = next((item for item in targets if _scene_type(item) == scene), None)
        if record is not None:
            selected.append(record | {"probe_reason": scene})
    dedup: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in selected:
        dedup.setdefault(_frame_key(record), record)
    return tuple(dedup.values())[:limit]


def _probe_points(
    record: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
    *,
    settings: Settings,
) -> dict[str, Any]:
    target = _target(record)
    transform, transform_spec = transform_from_settings_or_sample(
        settings,
        record,
        frame_width=_safe_int(record.get("frame_width")),
        frame_height=_safe_int(record.get("frame_height")),
    )
    target_osu = _point_pair(target.get("target_osu_xy"))
    target_video = _point_pair(target.get("target_video_xy"))
    gt_projection = None if target_osu is None else transform.osu_to_video(*target_osu)
    decoded_projection = None
    decoded_osu = None
    if target_video is not None:
        decoded_osu = transform.video_to_osu(*target_video)
        decoded_projection = transform.osu_to_video(*decoded_osu)
    nearest_candidate = _nearest_candidate_point(record, target_video)
    prediction = None if decision is None else _decision_video_xy(record, decision, settings=settings)
    return {
        "sample_id": _frame_id(record),
        "probe_reason": record.get("probe_reason"),
        "scene_type": _scene_type(record),
        "action": target.get("action"),
        "transform_source": transform_spec.source,
        "transform_status": transform_spec.transform_status,
        "beatmap_gt_projection": _point_list(gt_projection),
        "dataset_target_point": _point_list(target_video),
        "nearest_candidate_point": _point_list(nearest_candidate),
        "prediction_point": _point_list(prediction),
        "decoded_beatmap_point": _point_list(decoded_osu),
        "decoded_beatmap_reprojection": _point_list(decoded_projection),
        "target_projection_error": _point_error(gt_projection, target_video),
        "nearest_candidate_error": _point_error(nearest_candidate, target_video),
        "prediction_error": _point_error(prediction, target_video),
        "roundtrip_reprojection_error": _point_error(decoded_projection, target_video),
    }


def _draw_probe(draw: ImageDraw.ImageDraw, probe: Mapping[str, Any]) -> None:
    colors = {
        "beatmap_gt_projection": "yellow",
        "dataset_target_point": "lime",
        "nearest_candidate_point": "red",
        "prediction_point": "cyan",
        "decoded_beatmap_reprojection": "magenta",
    }
    for name, color in colors.items():
        point = probe.get(name)
        if not isinstance(point, Sequence) or len(point) < 2:
            continue
        x, y = float(point[0]), float(point[1])
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=3)
        draw.text((x + 8, y + 8), name.replace("_point", ""), fill=color)


def _coordinate_error_analysis(probes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    projection_errors = [
        probe["target_projection_error"]
        for probe in probes
        if isinstance(probe.get("target_projection_error"), Mapping)
    ]
    dx_values = [float(error["dx"]) for error in projection_errors]
    dy_values = [float(error["dy"]) for error in projection_errors]
    distances = [float(error["euclidean"]) for error in projection_errors]
    return {
        "projection_error_mean_dx": _mean(dx_values),
        "projection_error_mean_dy": _mean(dy_values),
        "projection_error_median_px": _percentile(distances, 50),
        "projection_error_p90_px": _percentile(distances, 90),
        "pattern": _error_pattern(dx_values, dy_values, distances),
    }


def _error_pattern(
    dx_values: Sequence[float],
    dy_values: Sequence[float],
    distances: Sequence[float],
) -> str:
    if not distances:
        return "insufficient_probe_data"
    if max(distances) < 1.0:
        return "no_visible_coordinate_projection_error"
    mean_dx = _mean(dx_values) or 0.0
    mean_dy = _mean(dy_values) or 0.0
    centered = [
        hypot(dx - mean_dx, dy - mean_dy)
        for dx, dy in zip(dx_values, dy_values)
    ]
    if centered and (_mean(centered) or 0.0) < max(2.0, (_mean(distances) or 0.0) * 0.25):
        return "fixed_offset_like"
    return "mixed_scale_aspect_crop_or_per_source_error"


def _first_error_stage(report: Mapping[str, Any]) -> str:
    ladder = report["oracle_ladder"]
    if float(ladder["gt_to_evaluator"]["quality_score"]) < 0.99:
        return "Evaluator"
    if float(ladder["target_roundtrip_to_evaluator"]["quality_score"]) < 0.99:
        return "Coordinate target transform"
    if float(ladder["matched_candidate_to_evaluator"]["quality_score"]) < 0.75:
        return "Candidate recall / target assignment"
    if float(ladder["temporal_slot_candidate_to_evaluator"]["quality_score"]) < 0.75:
        return "Temporal candidate slot truncation"
    continuity = report["temporal_continuity"]
    if not bool(continuity.get("sampled_windows_continuous")):
        return "Temporal window sampling"
    decision = report.get("decision_diagnostics")
    if isinstance(decision, Mapping) and float(decision.get("F1") or 0.0) < 0.75:
        return "Temporal/Decision"
    return "No failing stage found by oracle ladder"


def _candidate_distances(record: Mapping[str, Any]) -> list[float]:
    target = _target(record)
    raw = target.get("candidate_distances_px")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [
            float(value)
            for value in raw
            if isinstance(value, int | float)
        ]
    target_xy = _point_pair(target.get("target_video_xy"))
    if target_xy is None:
        return []
    return [
        hypot(float(candidate.get("x", 0.0)) - target_xy[0], float(candidate.get("y", 0.0)) - target_xy[1])
        for candidate in _candidate_rows(record)
    ]


def _nearest_candidate_point(
    record: Mapping[str, Any],
    target_video: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if target_video is None:
        return None
    candidates = _candidate_rows(record)
    if not candidates:
        return None
    candidate = min(
        candidates,
        key=lambda item: hypot(
            float(item.get("x", 0.0)) - target_video[0],
            float(item.get("y", 0.0)) - target_video[1],
        ),
    )
    return (float(candidate.get("x", 0.0)), float(candidate.get("y", 0.0)))


def _decision_video_xy(
    record: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    settings: Settings,
) -> tuple[float, float] | None:
    selected_id = _safe_int(decision.get("selected_candidate_id"))
    if selected_id is not None:
        for candidate in _candidate_rows(record):
            if _safe_int(candidate.get("candidate_id")) == selected_id:
                return (float(candidate.get("x", 0.0)), float(candidate.get("y", 0.0)))
    normalized = _point_pair(decision.get("predicted_xy_normalized"))
    if normalized is None:
        return None
    transform, _ = transform_from_settings_or_sample(
        settings,
        record,
        frame_width=_safe_int(record.get("frame_width")),
        frame_height=_safe_int(record.get("frame_height")),
    )
    return transform.osu_to_video(
        normalized[0] * OSU_PLAYFIELD_WIDTH,
        normalized[1] * OSU_PLAYFIELD_HEIGHT,
    )


def _candidate_rows(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    candidates = record.get("candidates") or ()
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return ()
    return tuple(candidate for candidate in candidates if isinstance(candidate, Mapping))


def _sorted_candidate_rows(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        sorted(
            _candidate_rows(record),
            key=lambda candidate: _safe_float(candidate.get("score")) or 0.0,
            reverse=True,
        )
    )


def _candidate_id_in_top_slots(
    record: Mapping[str, Any],
    candidate_id: int,
    candidate_slots: int | None,
) -> bool:
    return any(
        _safe_int(candidate.get("candidate_id")) == candidate_id
        for candidate in _candidate_rows(record)
    )


def _candidate_id_in_raw_top_slots(
    record: Mapping[str, Any],
    candidate_id: int,
    candidate_slots: int | None,
) -> bool:
    if candidate_slots is None or candidate_slots <= 0:
        return any(
            _safe_int(candidate.get("candidate_id")) == candidate_id
            for candidate in _candidate_rows(record)
        )
    return any(
        _safe_int(candidate.get("candidate_id")) == candidate_id
        for candidate in _sorted_candidate_rows(record)[:candidate_slots]
    )


def _temporal_candidate_slots(run_dir: Path, settings: Settings) -> int:
    summary_path = run_dir / "temporal" / "summary.json"
    if summary_path.is_file():
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
        slots = _safe_int(raw.get("candidate_slots"))
        if slots is not None and slots > 0:
            return slots
    return max(1, int(settings.candidate_cache.max_candidates_per_frame))


def _target(record: Mapping[str, Any]) -> Mapping[str, Any]:
    target = record.get("temporal_target")
    return target if isinstance(target, Mapping) else {}


def _target_type(record: Mapping[str, Any]) -> str:
    target = _target(record)
    return str(target.get("object_type") or target.get("action") or "no_op")


def _scene_type(record: Mapping[str, Any]) -> str:
    sample_key = str(record.get("sample_key") or "")
    if "point_slider" in sample_key:
        return "point_slider"
    if "single_point" in sample_key:
        return "single_point"
    if "slider" in sample_key:
        return "slider"
    if "long_sequence" in sample_key:
        return "long_sequence"
    if len(_candidate_rows(record)) >= 32:
        return "dense_hard"
    return "unknown"


def _segment_id(record: Mapping[str, Any]) -> str:
    sample_key = str(record.get("sample_key") or "")
    return sample_key.rsplit("/", maxsplit=1)[-1]


def _record_time_key(record: Mapping[str, Any]) -> tuple[str, int, float]:
    return (
        str(record.get("sample_key") or ""),
        _safe_int(record.get("frame_index")) or -1,
        _safe_float(record.get("timestamp_ms")) or -1.0,
    )


def _frame_key(record: Mapping[str, Any]) -> tuple[str, int]:
    return str(record.get("sample_key") or ""), _safe_int(record.get("frame_index")) or -1


def _frame_id(record: Mapping[str, Any]) -> str:
    return f"{record.get('sample_key')}:{record.get('frame_index')}"


def _sample_image(image: Any) -> Image.Image:
    array = image.detach().cpu()
    if array.ndim != 3:
        raise ValueError("sample image must be CHW")
    if array.dtype.is_floating_point:
        array = (array.clamp(0, 1) * 255).to(dtype=array.dtype).byte()
    else:
        array = array.byte()
    return Image.fromarray(array.permute(1, 2, 0).numpy(), mode="RGB")


def _osu_distance(record: Mapping[str, Any], point: tuple[float, float]) -> float:
    target_xy = _point_pair(_target(record).get("target_osu_xy"))
    if target_xy is None:
        return float("inf")
    return hypot(target_xy[0] - point[0], target_xy[1] - point[1])


def _point_error(
    point: tuple[float, float] | None,
    reference: tuple[float, float] | None,
) -> dict[str, float] | None:
    if point is None or reference is None:
        return None
    dx = point[0] - reference[0]
    dy = point[1] - reference[1]
    return {"dx": dx, "dy": dy, "euclidean": hypot(dx, dy)}


def _point_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2:
        return None
    x = _safe_float(value[0])
    y = _safe_float(value[1])
    if x is None or y is None:
        return None
    return x, y


def _point_list(value: tuple[float, float] | None) -> list[float] | None:
    return None if value is None else [float(value[0]), float(value[1])]


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)[:120]


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return tuple(rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100) * (len(ordered) - 1))))
    return ordered[index]
