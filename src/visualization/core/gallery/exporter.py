"""按评估分组渲染最佳试验，并以原子目录提交完整结果图集。"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traning.lib.data import SegmentFrameDataset, SegmentRecord
from traning.state.gallery_schema import (
    BatchGalleryRequest,
    EVALUATION_SUBPROJECTS,
    FrameEvaluation,
)
from visualization.conf.messages import display_text
from visualization.core.gallery.manifest import reserve_output_identity_for_commit
from visualization.core.gallery.renderer import (
    render_annotated_frame,
    save_annotated_frame,
)


OUTCOME_DIRECTORIES = {
    True: "passed",
    False: "failed",
}
DIMENSION_SUBPROJECTS = {
    "long_sequence": "long_sequence",
}
ERROR_DOMAIN_NAMES = {
    "none": "无",
    "spatial": "空间",
    "temporal": "时间",
    "decision": "决策",
}
ACTION_NAMES = {
    "no_op": "不操作",
    "press": "按下",
    "hold": "持续按住",
    "release": "释放",
}
MAX_SAFE_NAME_LENGTH = 72


@dataclass
class _SampleFrameGroup:
    sample_key: str
    subproject: str
    passed: bool = True
    frames: list[FrameEvaluation] = field(default_factory=list)

    def add(self, frame: FrameEvaluation) -> None:
        self.frames.append(frame)
        self.passed = self.passed and frame.passed


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    cleaned = cleaned or "unnamed"
    if len(cleaned) <= MAX_SAFE_NAME_LENGTH:
        return cleaned
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]
    prefix_length = MAX_SAFE_NAME_LENGTH - len(digest) - 2
    prefix = cleaned[:prefix_length].rstrip("._-") or "name"
    return f"{prefix}__{digest}"


def _subproject_for_record(record: SegmentRecord) -> str:
    return DIMENSION_SUBPROJECTS.get(
        record.dataset_dimension,
        record.category,
    )


def _frame_lookup(
    dataset: SegmentFrameDataset,
) -> dict[tuple[str, int], tuple[int, str]]:
    return {
        (
            dataset.records[reference.record_index].key,
            reference.frame_index,
        ): (
            dataset_index,
            _subproject_for_record(dataset.records[reference.record_index]),
        )
        for dataset_index, reference in enumerate(dataset.references)
    }


def _metric_lines(metrics: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(f"指标 {name}={value:.6g}" for name, value in sorted(metrics.items()))


def _diagnostic_lines(frame: FrameEvaluation) -> tuple[str, ...]:
    """把机器可读评估字段转换成不会误认真值为预测的图内说明。"""

    lines = ["图例 红色=评分真值目标 紫色=模型点击 黄色/青色=其他真值物件"]
    if frame.action is not None:
        probability = (
            ""
            if frame.action_probability is None
            else f" 概率={frame.action_probability:.6f}"
        )
        lines.append(
            f"模型动作={ACTION_NAMES.get(frame.action, frame.action)}{probability}"
        )
    if frame.predicted_osu_xy is None and frame.predicted_video_xy is None:
        lines.append("模型点击=无")
    if frame.primary_error != "none":
        lines.append(
            f"错误域={ERROR_DOMAIN_NAMES.get(frame.primary_error, frame.primary_error)}"
        )
    if frame.error_tags:
        lines.append(f"错误标签={','.join(frame.error_tags)}")
    if frame.failure_reason:
        lines.append(f"失败原因={frame.failure_reason}")
    return tuple(lines)


def _is_export_frame(frame: FrameEvaluation) -> bool:
    return (
        frame.target_source_index is not None
        or frame.predicted_osu_xy is not None
        or frame.predicted_video_xy is not None
        or frame.primary_error != "none"
    )


def _frame_identity(frame: FrameEvaluation) -> tuple[Any, ...]:
    return (
        frame.frame_index,
        frame.target_source_index,
        frame.predicted_osu_xy,
        frame.predicted_video_xy,
        frame.action,
        frame.action_probability,
        frame.primary_error,
        frame.error_tags,
        frame.failure_reason,
    )


def _frame_order_key(frame: FrameEvaluation) -> tuple[Any, ...]:
    return (
        frame.frame_index,
        frame.target_source_index is None,
        frame.target_source_index if frame.target_source_index is not None else -1,
        str(frame.predicted_video_xy),
        str(frame.predicted_osu_xy),
        str(frame.action),
        frame.primary_error,
        frame.error_tags,
        str(frame.failure_reason),
    )


def _sample_group_key(group: _SampleFrameGroup) -> tuple[str, str]:
    return group.subproject, group.sample_key


def _sorted_sample_groups(
    groups: list[_SampleFrameGroup],
) -> tuple[_SampleFrameGroup, ...]:
    return tuple(sorted(groups, key=_sample_group_key))


def save_best_trial_gallery(
    dataset: SegmentFrameDataset,
    request: BatchGalleryRequest,
    *,
    output_root: Path,
    samples_per_group: int = 10,
) -> tuple[Path, int, tuple[str, ...]]:
    """导出最佳试验图集，返回最终目录、图片数及可恢复的数据问题。"""

    if samples_per_group <= 0:
        raise ValueError("每组样本数必须为正数")

    best_trial = request.best_trial
    # 序号只有在完整目录发布成功后才提交，失败导出不会占用正式输出身份。
    with reserve_output_identity_for_commit(output_root) as reservation:
        output_identity = reservation.identity
        gallery_dir = output_root / (
            f"{output_identity.prefix}__{_safe_name(request.batch_id)}"
            f"__{_safe_name(best_trial.trial_id)}"
        )
        working_dir = gallery_dir.with_name(f".{gallery_dir.name}.tmp")
        if working_dir.exists():
            shutil.rmtree(working_dir)
        if gallery_dir.exists():
            raise FileExistsError(f"gallery output already exists: {gallery_dir}")
        try:
            saved_count, issues = _write_gallery_artifact(
                dataset=dataset,
                request=request,
                output_identity=output_identity,
                gallery_dir=gallery_dir,
                working_dir=working_dir,
                samples_per_group=samples_per_group,
            )
            working_dir.replace(gallery_dir)
            reservation.commit()
        except Exception:
            # 临时目录从不作为有效产物暴露；异常时清理并保留原始错误语义。
            if working_dir.exists():
                shutil.rmtree(working_dir)
            raise
    return gallery_dir, saved_count, tuple(issues)


def _write_gallery_artifact(
    *,
    dataset: SegmentFrameDataset,
    request: BatchGalleryRequest,
    output_identity,
    gallery_dir: Path,
    working_dir: Path,
    samples_per_group: int,
) -> tuple[int, list[str]]:
    """仅在临时目录内构建图片、索引和 manifest，不负责最终发布。"""

    best_trial = request.best_trial
    lookup = _frame_lookup(dataset)
    grouped_by_sample: dict[tuple[str, str], _SampleFrameGroup] = {}
    seen_frame_records: set[tuple[str, str, tuple[Any, ...]]] = set()
    issues: list[str] = []

    for frame in best_trial.frames:
        resolved = lookup.get((frame.sample_key, frame.frame_index))
        if resolved is None:
            issues.append(f"缺少数据集帧 {frame.sample_key}:{frame.frame_index}")
            continue
        _, subproject = resolved
        if subproject not in EVALUATION_SUBPROJECTS:
            issues.append(
                f"不支持的子项目 {subproject!r}，来源 "
                f"{frame.sample_key}:{frame.frame_index}"
            )
            continue
        if not _is_export_frame(frame):
            continue
        seen_key = (frame.sample_key, subproject, _frame_identity(frame))
        if seen_key in seen_frame_records:
            continue
        seen_frame_records.add(seen_key)
        sample_key = (frame.sample_key, subproject)
        group = grouped_by_sample.get(sample_key)
        if group is None:
            group = _SampleFrameGroup(
                sample_key=frame.sample_key,
                subproject=subproject,
            )
            grouped_by_sample[sample_key] = group
        group.add(frame)

    # 使用请求携带的局部随机源，使相同批次的抽样可复现且不污染全局随机状态。
    rng = random.Random(request.random_seed)
    sample_groups: dict[tuple[bool, str], list[_SampleFrameGroup]] = defaultdict(list)
    for group in grouped_by_sample.values():
        if not group.frames:
            continue
        group.frames.sort(key=_frame_order_key)
        sample_groups[(group.passed, group.subproject)].append(group)

    selected: dict[tuple[bool, str], tuple[_SampleFrameGroup, ...]] = {}
    for key, groups in sample_groups.items():
        ordered_groups = _sorted_sample_groups(groups)
        count = min(samples_per_group, len(ordered_groups))
        selected[key] = tuple(rng.sample(ordered_groups, count))

    working_dir.mkdir(parents=True, exist_ok=True)
    reached_subprojects = {subproject for _, subproject in sample_groups}
    for passed in (True, False):
        for subproject in EVALUATION_SUBPROJECTS:
            if subproject in reached_subprojects:
                (working_dir / OUTCOME_DIRECTORIES[passed] / subproject).mkdir(
                    parents=True, exist_ok=True
                )

    saved_frames: list[dict[str, Any]] = []
    saved_sample_groups: list[dict[str, Any]] = []
    for passed in (True, False):
        for subproject in EVALUATION_SUBPROJECTS:
            groups = selected.get((passed, subproject), ())
            if not groups:
                continue
            destination = working_dir / OUTCOME_DIRECTORIES[passed] / subproject
            for group_index, group in enumerate(groups, start=1):
                group_dir = destination / (
                    f"{group_index:02d}__{_safe_name(group.sample_key)}"
                )
                group_dir.mkdir(parents=True, exist_ok=True)
                group_frame_records: list[dict[str, Any]] = []
                for frame_index, frame in enumerate(group.frames, start=1):
                    dataset_index, _ = lookup[(frame.sample_key, frame.frame_index)]
                    sample = dataset[dataset_index]
                    metadata = (
                        f"输出序号={output_identity.sequence:06d}",
                        f"输出时间={output_identity.created_at_utc}",
                        f"批次={request.batch_id}",
                        f"试验={best_trial.trial_id} 评分={best_trial.score:.6g}",
                        f"评分版本={best_trial.score_version}",
                        f"样本组={group_index:02d}",
                        f"子项目={display_text(subproject)}",
                        f"结果={display_text(OUTCOME_DIRECTORIES[passed])}",
                        *_diagnostic_lines(frame),
                        *_metric_lines(frame.metrics),
                    )
                    image = render_annotated_frame(
                        sample,
                        target_source_index=frame.target_source_index,
                        predicted_osu_xy=frame.predicted_osu_xy,
                        predicted_video_xy=frame.predicted_video_xy,
                        metadata_lines=metadata,
                    )
                    filename = f"{frame_index:02d}__frame_{frame.frame_index:06d}.png"
                    output_path = save_annotated_frame(image, group_dir / filename)
                    frame_record = {
                        "sample_key": frame.sample_key,
                        "frame_index": frame.frame_index,
                        "passed": frame.passed,
                        "subproject": subproject,
                        "path": str(output_path.relative_to(working_dir)),
                        "target_source_index": frame.target_source_index,
                        "predicted_osu_xy": frame.predicted_osu_xy,
                        "predicted_video_xy": frame.predicted_video_xy,
                        "action": frame.action,
                        "action_probability": frame.action_probability,
                        "primary_error": frame.primary_error,
                        "error_tags": frame.error_tags,
                        "failure_reason": frame.failure_reason,
                        "frequency_limited": frame.frequency_limited,
                        "metrics": frame.metrics,
                    }
                    saved_frames.append(frame_record)
                    group_frame_records.append(frame_record)
                saved_sample_groups.append(
                    {
                        "sample_key": group.sample_key,
                        "passed": group.passed,
                        "subproject": subproject,
                        "path": str(group_dir.relative_to(working_dir)),
                        "frame_count": len(group_frame_records),
                        "frames": group_frame_records,
                    }
                )

    parameters_path = working_dir / "best_parameters.json"
    parameters_path.write_text(
        json.dumps(
            {
                "batch_id": request.batch_id,
                "output_sequence": output_identity.sequence,
                "output_time_utc": output_identity.created_at_utc,
                "trial_id": best_trial.trial_id,
                "score": best_trial.score,
                "score_version": best_trial.score_version,
                "curriculum_stage": request.metadata.get("curriculum_stage"),
                "batch_identifier": request.metadata.get("batch_id", request.batch_id),
                "metrics": best_trial.metrics,
                "parameters": best_trial.parameters.model_dump(mode="json"),
                "metadata": request.metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    index_path = working_dir / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        # 图集索引同时保留协议版本和方程指纹，便于确认历史图片采用的确切坐标契约。
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "error_type",
                "segment",
                "beatmap",
                "sample",
                "trial",
                "curriculum_stage",
                "parameter_group",
                "score",
                "gallery_image_path",
                "predicted_osu_xy",
                "predicted_video_xy",
                "action_type",
                "ambiguity_reason",
                "failure_reason",
                "score_version",
                "dataset_version",
                "evaluation_dataset_version",
                "candidate_cache_version",
                "transform_version",
                "transform_fingerprint",
                "configuration_version",
            ),
        )
        writer.writeheader()
        metadata = request.metadata
        for frame in saved_frames:
            sample_key = str(frame["sample_key"])
            writer.writerow(
                {
                    "error_type": frame.get("primary_error") or "",
                    "segment": sample_key.rsplit("/", 1)[-1],
                    "beatmap": sample_key.split("/", 1)[0],
                    "sample": sample_key,
                    "trial": best_trial.trial_id,
                    "curriculum_stage": metadata.get("curriculum_stage", ""),
                    "parameter_group": metadata.get("trial_id", best_trial.trial_id),
                    "score": best_trial.score,
                    "gallery_image_path": frame["path"],
                    "predicted_osu_xy": frame.get("predicted_osu_xy"),
                    "predicted_video_xy": frame.get("predicted_video_xy"),
                    "action_type": frame.get("action") or "",
                    "ambiguity_reason": frame.get("ambiguity_reason") or "",
                    "failure_reason": frame.get("failure_reason") or "",
                    "score_version": best_trial.score_version,
                    "dataset_version": metadata.get("dataset_version", ""),
                    "evaluation_dataset_version": metadata.get(
                        "evaluation_dataset_version", ""
                    ),
                    "candidate_cache_version": metadata.get(
                        "candidate_cache_version", ""
                    ),
                    "transform_version": metadata.get("transform_version", ""),
                    # 指纹包含完整变换配置与训练帧尺寸，不能用 transform_version 替代。
                    "transform_fingerprint": metadata.get("transform_fingerprint", ""),
                    "configuration_version": metadata.get("configuration_version", ""),
                }
            )
    manifest_path = working_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch_id": request.batch_id,
                "output_sequence": output_identity.sequence,
                "output_time_utc": output_identity.created_at_utc,
                "selected_trial_id": best_trial.trial_id,
                "selected_trial_score": best_trial.score,
                "score_version": best_trial.score_version,
                "metadata": request.metadata,
                "gallery_request": request.model_dump(mode="json"),
                "score_report_path": request.metadata.get("score_report_path"),
                "candidate_cache_manifest_path": request.metadata.get(
                    "candidate_cache_manifest_path"
                ),
                "spatial_checkpoint_path": request.metadata.get(
                    "spatial_checkpoint_path"
                ),
                "temporal_checkpoint_path": request.metadata.get(
                    "temporal_checkpoint_path"
                ),
                "index_csv": str(index_path.relative_to(working_dir)),
                "random_seed": request.random_seed,
                "samples_per_group": samples_per_group,
                "selected_sample_group_count": len(saved_sample_groups),
                "subprojects": list(EVALUATION_SUBPROJECTS),
                "reached_subprojects": [
                    subproject
                    for subproject in EVALUATION_SUBPROJECTS
                    if subproject in reached_subprojects
                ],
                "saved_frame_count": len(saved_frames),
                "sample_groups": saved_sample_groups,
                "frames": saved_frames,
                "issues": issues,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(saved_frames), issues


__all__ = [
    "OUTCOME_DIRECTORIES",
    "save_best_trial_gallery",
]
