from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter

from loguru import logger

from before_traning.Lib.beatmap.folder_store import BeatmapFolderStore
from before_traning.Lib.beatmap.standard import (
    ParsedStandardBeatmap,
    load_standard_beatmap,
)
from before_traning.Lib.beatmap.osu_parser import VerifyOsuParser
from before_traning.Lib.common.batch import (
    BatchProcessResult,
    FolderBatchProcessor,
)
from before_traning.Lib.common.failures import exception_detail, failure_detail
from before_traning.Lib.common.sequence import format_sequence_name
from before_traning.Lib.tools.ffmpeg import (
    get_media_duration_seconds,
    segment_video,
)
from before_traning.Lib.video.segment_dataset import (
    LONG_SEQUENCE_DIRECTORY,
    SEGMENT_BEATMAP_FILENAME,
    SEGMENT_MANIFEST_FILENAME,
    SEGMENT_SCHEMA_VERSION,
    SEGMENT_TABLE_FILENAME,
    SEGMENT_VIDEO_FILENAME,
    SegmentDatasetManifest,
    write_json_file,
)
from before_traning.Lib.video.segmentation import plan_video_segments
from before_traning.Lib.video.segmentation.planner import (
    SEGMENT_CATEGORIES,
    SegmentPlan,
)
from before_traning.conf import Settings
from before_traning.state.process_status import ProcessStatusManager


SEGMENT_OUTPUT_DIRECTORIES = (
    *SEGMENT_CATEGORIES,
    LONG_SEQUENCE_DIRECTORY,
)


def _overlap_merge_window_ms(
    beatmap: ParsedStandardBeatmap,
    settings: Settings,
) -> int:
    return round(
        beatmap.approach_preempt_ms
        * settings.segment.approach_preempt_ratio
    )


def _output_directory_name(plan: SegmentPlan) -> str:
    if plan.dimension == "atomic":
        return plan.category
    return LONG_SEQUENCE_DIRECTORY


def _segment_directory_name(index: int, plan: SegmentPlan) -> str:
    sequence_name = format_sequence_name("segment_", index)
    return (
        f"{sequence_name}_"
        f"{plan.hit_start_ms:09d}_{plan.hit_end_ms:09d}"
    )


def _beatmap_payload(
    *,
    folder_name: str,
    source_osu_path: Path,
    segment_id: str,
    beatmap: ParsedStandardBeatmap,
    plan: SegmentPlan,
    settings: Settings,
    parser: VerifyOsuParser,
) -> dict[str, object]:
    return {
        "schema_version": SEGMENT_SCHEMA_VERSION,
        "segment_id": segment_id,
        "dataset_dimension": plan.dimension,
        "category": plan.category,
        "source_plan_count": plan.source_plan_count,
        "time_origin": "segment_start",
        "difficulty": {
            "hp_drain_rate": beatmap.hp_drain_rate,
            "circle_size": beatmap.circle_size,
            "circle_radius_osu_pixels": plan.circle_radius,
            "overall_difficulty": beatmap.overall_difficulty,
            "approach_rate": beatmap.approach_rate,
            "approach_preempt_ms": beatmap.approach_preempt_ms,
            "slider_multiplier": beatmap.slider_multiplier,
            "slider_tick_rate": beatmap.slider_tick_rate,
            "stack_leniency": beatmap.stack_leniency,
        },
        "grouping": {
            "approach_preempt_ratio": (
                settings.segment.approach_preempt_ratio
            ),
            "overlap_merge_window_ms": _overlap_merge_window_ms(
                beatmap,
                settings,
            ),
            "min_circle_overlap_ratio": (
                settings.segment.min_circle_overlap_ratio
            ),
            "priority_merge_window_ms": (
                settings.segment.priority_merge_window_ms
            ),
            "use_priority_merge": settings.segment.use_priority_merge,
            "post_context_seconds": settings.segment.post_context_seconds,
            "build_long_sequences": settings.segment.build_long_sequences,
            "long_sequence_continuity_window_ms": round(
                beatmap.approach_preempt_ms
            ),
            "long_sequence_max_objects": (
                settings.segment.long_sequence_max_objects
            ),
            "long_sequence_max_duration_seconds": (
                settings.segment.long_sequence_max_duration_seconds
            ),
        },
        "source": {
            "folder_name": folder_name,
            "osu_filename": source_osu_path.name,
            "clip_start_ms": plan.clip_start_ms,
            "clip_end_ms": plan.clip_end_ms,
        },
        "hit_objects": [
            {
                **parser.hit_object_to_dict(
                    hit_object,
                    time_offset_ms=plan.clip_start_ms,
                ),
                "source_index": source_index,
                "source_start_ms": hit_object.t_start,
                "source_end_ms": hit_object.t_end,
            }
            for source_index, hit_object in zip(
                plan.object_indexes,
                plan.hit_objects,
            )
        ],
    }


def _segment_row(
    *,
    segment_id: str,
    output_directory_name: str,
    directory_name: str,
    beatmap: ParsedStandardBeatmap,
    plan: SegmentPlan,
    settings: Settings,
) -> dict[str, object]:
    local_start_times = [
        value - plan.clip_start_ms
        for value in plan.object_start_times_ms
    ]
    local_end_times = [
        value - plan.clip_start_ms
        for value in plan.object_end_times_ms
    ]
    relative_directory = f"{output_directory_name}/{directory_name}"
    return {
        "segment_id": segment_id,
        "dataset_dimension": plan.dimension,
        "category": plan.category,
        "source_plan_count": plan.source_plan_count,
        "hp_drain_rate": f"{beatmap.hp_drain_rate:.6f}",
        "circle_size": f"{plan.circle_size:.6f}",
        "circle_radius": f"{plan.circle_radius:.6f}",
        "overall_difficulty": f"{beatmap.overall_difficulty:.6f}",
        "approach_rate": f"{beatmap.approach_rate:.6f}",
        "approach_preempt_ms": f"{beatmap.approach_preempt_ms:.6f}",
        "slider_multiplier": f"{beatmap.slider_multiplier:.6f}",
        "slider_tick_rate": f"{beatmap.slider_tick_rate:.6f}",
        "stack_leniency": f"{beatmap.stack_leniency:.6f}",
        "approach_preempt_ratio": (
            f"{settings.segment.approach_preempt_ratio:.6f}"
        ),
        "overlap_merge_window_ms": _overlap_merge_window_ms(
            beatmap,
            settings,
        ),
        "min_circle_overlap_ratio": (
            f"{settings.segment.min_circle_overlap_ratio:.6f}"
        ),
        "priority_merge_window_ms": (
            settings.segment.priority_merge_window_ms
        ),
        "use_priority_merge": settings.segment.use_priority_merge,
        "configured_post_context_seconds": (
            f"{settings.segment.post_context_seconds:.6f}"
        ),
        "build_long_sequences": settings.segment.build_long_sequences,
        "long_sequence_continuity_window_ms": round(
            beatmap.approach_preempt_ms
        ),
        "long_sequence_max_objects": (
            settings.segment.long_sequence_max_objects
        ),
        "long_sequence_max_duration_seconds": (
            f"{settings.segment.long_sequence_max_duration_seconds:.6f}"
        ),
        "segment_directory": relative_directory,
        "video_path": f"{relative_directory}/{SEGMENT_VIDEO_FILENAME}",
        "beatmap_path": f"{relative_directory}/{SEGMENT_BEATMAP_FILENAME}",
        "hit_start_ms": plan.hit_start_ms,
        "hit_end_ms": plan.hit_end_ms,
        "clip_source_start_ms": plan.clip_start_ms,
        "clip_source_end_ms": plan.clip_end_ms,
        "clip_start_seconds": f"{plan.clip_start_seconds:.6f}",
        "clip_end_seconds": f"{plan.clip_end_seconds:.6f}",
        "pre_context_seconds": f"{plan.pre_context_seconds:.6f}",
        "post_context_seconds": f"{plan.post_context_seconds:.6f}",
        "object_count": len(plan.object_types),
        "object_indexes": "|".join(
            str(value) for value in plan.object_indexes
        ),
        "object_types": "|".join(plan.object_types),
        "object_start_times_ms": "|".join(
            str(value) for value in local_start_times
        ),
        "object_end_times_ms": "|".join(
            str(value) for value in local_end_times
        ),
        "source_object_start_times_ms": "|".join(
            str(value) for value in plan.object_start_times_ms
        ),
        "source_object_end_times_ms": "|".join(
            str(value) for value in plan.object_end_times_ms
        ),
    }


class VideoSegmentationProcessor(FolderBatchProcessor):
    def __init__(
        self,
        settings: Settings,
        status_manager: ProcessStatusManager | None = None,
    ):
        super().__init__()
        self.settings = settings
        self.store = BeatmapFolderStore(
            target_root=str(settings.file_management.target_root),
            manifest_filename=settings.file_management.manifest_filename,
        )
        self.walker = self.store.walker
        self.segment_root = Path(settings.file_management.segment_root)
        self.store.recover_atomic_outputs(
            self.segment_root,
            namespace="segment",
        )
        self.walker.manifest.export_table(
            self.segment_root / SEGMENT_MANIFEST_FILENAME
        )
        self.dataset = SegmentDatasetManifest(
            self.segment_root,
            SEGMENT_OUTPUT_DIRECTORIES,
        )
        self.parser = VerifyOsuParser()
        self.status_step = settings.segment.status_step.strip()
        self.required_steps = tuple(
            step.strip()
            for step in settings.segment.required_steps
            if step.strip()
        )
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(settings.file_management.target_root),
            manifest_filename=settings.file_management.manifest_filename,
            process_steps=settings.progress.process_steps,
        )
        registered = set(self.status_manager.process_steps)
        missing = [
            step
            for step in (*self.required_steps, self.status_step)
            if step not in registered
        ]
        if missing:
            raise ValueError(
                f"process_steps 缺少视频切分所需步骤: {', '.join(missing)}"
            )

    def progress_message(
        self,
        index: int,
        total: int,
        folder_name: str,
    ) -> str:
        return f"[切分] {index}/{total} {folder_name}"

    def process_one(
        self,
        folder_name: str,
        overwrite: bool = False,
    ) -> BatchProcessResult:
        if not self.store.folder_exists(folder_name):
            return "skip"

        self.status_manager.ensure_status_file(folder_name)
        output_complete = self.dataset.output_complete(folder_name)
        step_done = self.status_manager.is_step_done(
            folder_name,
            self.status_step,
        )
        if not overwrite and output_complete and step_done:
            return "skip"
        if step_done and not output_complete:
            self.status_manager.mark_step_pending(
                folder_name,
                self.status_step,
                detail=failure_detail(
                    "状态显示已完成视频切分，但输出目录不完整",
                    self.process_one,
                ),
            )

        missing = [
            step
            for step in self.required_steps
            if not self.status_manager.is_step_done(folder_name, step)
        ]
        if missing:
            raise ValueError(
                f"视频切分缺少前置步骤: {', '.join(missing)}"
            )

        source_video_path = self.store.get_file_path(
            folder_name,
            self.settings.file_management.output_filename,
        )
        if not source_video_path.is_file():
            raise FileNotFoundError(
                f"待切分视频不存在: {source_video_path}"
            )
        source_osu_path, beatmap = load_standard_beatmap(
            self.store,
            folder_name,
            parser=self.parser,
        )
        plans = plan_video_segments(
            beatmap,
            video_duration_seconds=get_media_duration_seconds(
                source_video_path
            ),
            approach_preempt_ratio=(
                self.settings.segment.approach_preempt_ratio
            ),
            post_context_seconds=(
                self.settings.segment.post_context_seconds
            ),
            min_circle_overlap_ratio=(
                self.settings.segment.min_circle_overlap_ratio
            ),
            priority_merge_window_ms=(
                self.settings.segment.priority_merge_window_ms
            ),
            use_priority_merge=(
                self.settings.segment.use_priority_merge
            ),
            build_long_sequences=(
                self.settings.segment.build_long_sequences
            ),
            long_sequence_max_objects=(
                self.settings.segment.long_sequence_max_objects
            ),
            long_sequence_max_duration_seconds=(
                self.settings.segment.long_sequence_max_duration_seconds
            ),
        )
        self.status_manager.mark_step_pending(
            folder_name,
            self.status_step,
            detail={
                "stage": "segmenting",
                "source_video_path": str(source_video_path),
                "segment_count": len(plans.all),
                "atomic_segment_count": len(plans.atomic),
                "long_sequence_count": len(plans.long_sequence),
            },
        )

        rows: list[dict[str, object]] = []
        directory_indexes: Counter[str] = Counter()
        with self.store.atomic_output_folder(
            self.segment_root,
            folder_name,
            namespace="segment",
        ) as temporary:
            for directory in SEGMENT_OUTPUT_DIRECTORIES:
                self.store.create_output_directory(temporary, directory)

            for plan in plans.all:
                output_name = _output_directory_name(plan)
                directory_indexes[output_name] += 1
                segment_index = directory_indexes[output_name]
                segment_id = format_sequence_name(
                    f"{output_name}_",
                    segment_index,
                )
                directory_name = _segment_directory_name(
                    segment_index,
                    plan,
                )
                segment_directory = self.store.create_output_directory(
                    temporary,
                    output_name,
                    directory_name,
                )
                segment_video(
                    source_video_path,
                    segment_directory / SEGMENT_VIDEO_FILENAME,
                    start_seconds=plan.clip_start_seconds,
                    end_seconds=plan.clip_end_seconds,
                )
                write_json_file(
                    segment_directory / SEGMENT_BEATMAP_FILENAME,
                    _beatmap_payload(
                        folder_name=folder_name,
                        source_osu_path=source_osu_path,
                        segment_id=segment_id,
                        beatmap=beatmap,
                        plan=plan,
                        settings=self.settings,
                        parser=self.parser,
                    ),
                )
                rows.append(
                    _segment_row(
                        segment_id=segment_id,
                        output_directory_name=output_name,
                        directory_name=directory_name,
                        beatmap=beatmap,
                        plan=plan,
                        settings=self.settings,
                    )
                )
            self.dataset.write_table(temporary, rows)

        self.dataset.replace_folder(folder_name, rows)
        output_directory = self.segment_root / folder_name
        self.status_manager.mark_step_done(
            folder_name,
            self.status_step,
            detail={
                "stage": "done",
                "schema_version": SEGMENT_SCHEMA_VERSION,
                "source_video_path": str(source_video_path),
                "output_directory": str(output_directory),
                "segment_table": str(
                    output_directory / SEGMENT_TABLE_FILENAME
                ),
                "segment_manifest": str(self.dataset.db_path),
                "segment_count": len(plans.all),
                "atomic_segment_count": len(plans.atomic),
                "long_sequence_count": len(plans.long_sequence),
                "atomic_hit_object_count": sum(
                    len(plan.object_types) for plan in plans.atomic
                ),
                "long_sequence_hit_object_count": sum(
                    len(plan.object_types)
                    for plan in plans.long_sequence
                ),
                "category_counts": {
                    directory: directory_indexes[directory]
                    for directory in SEGMENT_OUTPUT_DIRECTORIES
                },
                "atomic_hit_object_assignment": "exactly_once",
                "long_sequence_hit_object_assignment": "at_most_once",
            },
        )
        return "success"

    def handle_failure(self, folder_name: str, error: Exception) -> None:
        if self.store.folder_exists(folder_name):
            self.status_manager.ensure_status_file(folder_name)
            self.status_manager.mark_step_pending(
                folder_name,
                self.status_step,
                detail=exception_detail(error, stage="failed"),
            )


def segment_videos(settings: Settings) -> bool:
    logger.info("开始 video_segment")
    started_at = perf_counter()
    success = VideoSegmentationProcessor(settings).run(
        overwrite=settings.overwrite
    )
    logger.info("完成 video_segment ({:.2f}s)", perf_counter() - started_at)
    return success
