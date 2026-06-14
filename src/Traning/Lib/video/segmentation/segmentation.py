from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from uuid import uuid4

from Traning.Lib.beatmap.folder_store import BeatmapFolderStore
from Traning.Lib.beatmap.hit_objects import HitObject
from Traning.Lib.beatmap.verification.parser import VerifyOsuParser
from Traning.Lib.common.batch import BatchProcessResult, FolderBatchProcessor
from Traning.Lib.common.failures import exception_detail, failure_detail
from Traning.Lib.defaults import DEFAULT_SETTINGS as DEFAULTS
from Traning.Lib.tools.ffmpeg import (
    build_segment_video_args,
    get_media_duration_seconds,
    run_ffmpeg,
)
from Traning.Lib.video.segmentation.planner import (
    SEGMENT_CATEGORIES,
    ParsedStandardBeatmap,
    SegmentPlan,
    build_segment_plans,
    parse_standard_beatmap,
)
from Traning.conf import Settings
from Traning.state.process_status import ProcessStatusManager


SEGMENT_TABLE_FILENAME = "segments.csv"
SEGMENT_MANIFEST_FILENAME = "manifest.csv"
SEGMENT_BEATMAP_FILENAME = "beatmap.json"
SEGMENT_VIDEO_FILENAME = "video.mp4"
SEGMENT_SCHEMA_VERSION = 9
SEGMENT_TABLE_FIELDS = (
    "segment_id",
    "category",
    "hp_drain_rate",
    "circle_size",
    "circle_radius",
    "overall_difficulty",
    "approach_rate",
    "approach_preempt_ms",
    "slider_multiplier",
    "slider_tick_rate",
    "stack_leniency",
    "approach_preempt_ratio",
    "overlap_merge_window_ms",
    "min_circle_overlap_ratio",
    "priority_merge_window_ms",
    "use_priority_merge",
    "post_context_seconds",
    "segment_directory",
    "video_path",
    "beatmap_path",
    "hit_start_ms",
    "hit_end_ms",
    "clip_source_start_ms",
    "clip_source_end_ms",
    "clip_start_seconds",
    "clip_end_seconds",
    "pre_context_seconds",
    "post_context_seconds",
    "object_count",
    "object_indexes",
    "object_types",
    "object_start_times_ms",
    "object_end_times_ms",
    "source_object_start_times_ms",
    "source_object_end_times_ms",
)


class VideoSegmentationProcessor(FolderBatchProcessor):
    def __init__(
        self,
        settings: Settings = DEFAULTS,
        status_manager: ProcessStatusManager | None = None,
    ):
        self.target_root = Path(settings.file_management.target_root)
        self.segment_root = Path(settings.file_management.segment_root)
        self.manifest_filename = settings.file_management.manifest_filename
        self.output_filename = settings.file_management.output_filename
        self.approach_preempt_ratio = (
            settings.segment.approach_preempt_ratio
        )
        self.post_context_seconds = settings.segment.post_context_seconds
        self.min_circle_overlap_ratio = (
            settings.segment.min_circle_overlap_ratio
        )
        self.priority_merge_window_ms = (
            settings.segment.priority_merge_window_ms
        )
        self.use_priority_merge = settings.segment.use_priority_merge
        self.status_step = settings.segment.status_step.strip()
        self.required_steps = tuple(
            step.strip() for step in settings.segment.required_steps if step.strip()
        )
        self.parser = VerifyOsuParser()
        if not self.status_step:
            raise ValueError("segment status_step 不能为空")

        super().__init__()
        self.store = BeatmapFolderStore(
            target_root=str(self.target_root),
            manifest_filename=self.manifest_filename,
        )
        self.walker = self.store.walker
        self._recover_interrupted_outputs()
        self._sync_manifest_table()
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(self.target_root),
            manifest_filename=self.manifest_filename,
            process_steps=settings.progress.process_steps,
        )
        self._ensure_status_steps_registered()

    def _recover_interrupted_outputs(self) -> None:
        self.segment_root.mkdir(parents=True, exist_ok=True)
        backup_prefix = ".__segment_backup__"
        for backup in self.segment_root.glob(f"{backup_prefix}*"):
            folder_name = backup.name.removeprefix(backup_prefix).rsplit("_", 1)[0]
            destination = self.segment_root / folder_name
            if destination.exists():
                self._remove_output_path(backup)
            else:
                backup.rename(destination)

        for temporary in self.segment_root.glob(".__segment_tmp__*"):
            self._remove_output_path(temporary)

    def _sync_manifest_table(self) -> None:
        self.segment_root.mkdir(parents=True, exist_ok=True)
        source_table = self.walker.manifest.table_path
        destination_table = self.segment_root / SEGMENT_MANIFEST_FILENAME
        temporary_table = destination_table.with_name(
            f".{SEGMENT_MANIFEST_FILENAME}.{uuid4().hex}.tmp"
        )
        shutil.copyfile(source_table, temporary_table)
        temporary_table.replace(destination_table)

    def _ensure_status_steps_registered(self) -> None:
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

    def progress_message(self, index: int, total: int, folder_name: str) -> str:
        return f"[切分] {index}/{total} {folder_name}"

    def _output_directory(self, folder_name: str) -> Path:
        return self.segment_root / folder_name

    def _output_complete(self, folder_name: str) -> bool:
        output_directory = self._output_directory(folder_name)
        segment_table = output_directory / SEGMENT_TABLE_FILENAME
        if (
            not output_directory.is_dir()
            or not segment_table.is_file()
            or not all(
                (output_directory / category).is_dir()
                for category in SEGMENT_CATEGORIES
            )
        ):
            return False

        with segment_table.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != SEGMENT_TABLE_FIELDS:
                return False
            rows = list(reader)

        return bool(rows) and all(
            (output_directory / row["video_path"]).is_file()
            and (output_directory / row["beatmap_path"]).is_file()
            for row in rows
        )

    def _ensure_required_steps_done(self, folder_name: str) -> None:
        missing = [
            step
            for step in self.required_steps
            if not self.status_manager.is_step_done(folder_name, step)
        ]
        if missing:
            raise ValueError(f"视频切分缺少前置步骤: {', '.join(missing)}")

    def _segment_directory_name(self, index: int, plan: SegmentPlan) -> str:
        return (
            f"segment_{index:06d}_"
            f"{plan.hit_start_ms:09d}_{plan.hit_end_ms:09d}"
        )

    def _overlap_merge_window_ms(
        self,
        beatmap: ParsedStandardBeatmap,
    ) -> int:
        return round(
            beatmap.approach_preempt_ms * self.approach_preempt_ratio
        )

    def _write_segment(
        self,
        source_video_path: Path,
        output_path: Path,
        plan: SegmentPlan,
    ) -> None:
        run_ffmpeg(
            build_segment_video_args(
                source_video_path,
                output_path,
                trim_start_seconds=plan.clip_start_seconds,
                trim_duration_seconds=plan.duration_seconds,
            )
        )

    def _serialize_hit_object(
        self,
        hit_object: HitObject,
        source_index: int,
        clip_start_ms: int,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            **self.parser.hit_object_to_dict(
                hit_object,
                time_offset_ms=clip_start_ms,
            ),
            "source_index": source_index,
            "source_start_ms": hit_object.t_start,
            "source_end_ms": hit_object.t_end,
        }
        return payload

    def _write_beatmap_data(
        self,
        output_path: Path,
        *,
        folder_name: str,
        source_osu_path: Path,
        segment_id: str,
        beatmap: ParsedStandardBeatmap,
        plan: SegmentPlan,
    ) -> None:
        hit_objects = [
            self._serialize_hit_object(
                hit_object,
                source_index,
                plan.clip_start_ms,
            )
            for source_index, hit_object in zip(
                plan.object_indexes,
                plan.hit_objects,
            )
        ]
        payload = {
            "schema_version": SEGMENT_SCHEMA_VERSION,
            "segment_id": segment_id,
            "category": plan.category,
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
                "approach_preempt_ratio": self.approach_preempt_ratio,
                "overlap_merge_window_ms": (
                    self._overlap_merge_window_ms(beatmap)
                ),
                "min_circle_overlap_ratio": self.min_circle_overlap_ratio,
                "priority_merge_window_ms": self.priority_merge_window_ms,
                "use_priority_merge": self.use_priority_merge,
                "post_context_seconds": self.post_context_seconds,
            },
            "source": {
                "folder_name": folder_name,
                "osu_filename": source_osu_path.name,
                "clip_start_ms": plan.clip_start_ms,
                "clip_end_ms": plan.clip_end_ms,
            },
            "hit_objects": hit_objects,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_segment_table(
        self,
        output_directory: Path,
        rows: list[dict[str, object]],
    ) -> None:
        with (output_directory / SEGMENT_TABLE_FILENAME).open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=SEGMENT_TABLE_FIELDS,
            )
            writer.writeheader()
            writer.writerows(rows)

    def _remove_output_path(self, path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def _build_output(
        self,
        folder_name: str,
        source_video_path: Path,
        source_osu_path: Path,
        beatmap: ParsedStandardBeatmap,
        plans: list[SegmentPlan],
    ) -> tuple[Path, dict[str, int]]:
        self.segment_root.mkdir(parents=True, exist_ok=True)
        temporary = self.segment_root / f".__segment_tmp__{folder_name}_{uuid4().hex}"
        destination = self._output_directory(folder_name)
        backup = self.segment_root / f".__segment_backup__{folder_name}_{uuid4().hex}"
        rows: list[dict[str, object]] = []
        category_indexes: Counter[str] = Counter()

        try:
            for category in SEGMENT_CATEGORIES:
                (temporary / category).mkdir(parents=True, exist_ok=True)

            for plan in plans:
                category_indexes[plan.category] += 1
                segment_index = category_indexes[plan.category]
                segment_id = f"{plan.category}_{segment_index:06d}"
                directory_name = self._segment_directory_name(segment_index, plan)
                segment_directory = temporary / plan.category / directory_name
                segment_directory.mkdir()
                video_path = segment_directory / SEGMENT_VIDEO_FILENAME
                beatmap_path = segment_directory / SEGMENT_BEATMAP_FILENAME
                self._write_segment(source_video_path, video_path, plan)
                self._write_beatmap_data(
                    beatmap_path,
                    folder_name=folder_name,
                    source_osu_path=source_osu_path,
                    segment_id=segment_id,
                    beatmap=beatmap,
                    plan=plan,
                )
                local_start_times = [
                    value - plan.clip_start_ms
                    for value in plan.object_start_times_ms
                ]
                local_end_times = [
                    value - plan.clip_start_ms
                    for value in plan.object_end_times_ms
                ]
                rows.append(
                    {
                        "segment_id": segment_id,
                        "category": plan.category,
                        "hp_drain_rate": f"{beatmap.hp_drain_rate:.6f}",
                        "circle_size": f"{plan.circle_size:.6f}",
                        "circle_radius": f"{plan.circle_radius:.6f}",
                        "overall_difficulty": (
                            f"{beatmap.overall_difficulty:.6f}"
                        ),
                        "approach_rate": f"{beatmap.approach_rate:.6f}",
                        "approach_preempt_ms": (
                            f"{beatmap.approach_preempt_ms:.6f}"
                        ),
                        "slider_multiplier": (
                            f"{beatmap.slider_multiplier:.6f}"
                        ),
                        "slider_tick_rate": (
                            f"{beatmap.slider_tick_rate:.6f}"
                        ),
                        "stack_leniency": f"{beatmap.stack_leniency:.6f}",
                        "approach_preempt_ratio": (
                            f"{self.approach_preempt_ratio:.6f}"
                        ),
                        "overlap_merge_window_ms": (
                            self._overlap_merge_window_ms(beatmap)
                        ),
                        "min_circle_overlap_ratio": (
                            f"{self.min_circle_overlap_ratio:.6f}"
                        ),
                        "priority_merge_window_ms": (
                            self.priority_merge_window_ms
                        ),
                        "use_priority_merge": self.use_priority_merge,
                        "post_context_seconds": (
                            f"{self.post_context_seconds:.6f}"
                        ),
                        "segment_directory": f"{plan.category}/{directory_name}",
                        "video_path": (
                            f"{plan.category}/{directory_name}/"
                            f"{SEGMENT_VIDEO_FILENAME}"
                        ),
                        "beatmap_path": (
                            f"{plan.category}/{directory_name}/"
                            f"{SEGMENT_BEATMAP_FILENAME}"
                        ),
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
                )
            self._write_segment_table(temporary, rows)

            if destination.exists():
                destination.rename(backup)
            temporary.rename(destination)
            if backup.exists():
                self._remove_output_path(backup)
            return destination, {
                category: category_indexes[category]
                for category in SEGMENT_CATEGORIES
            }
        except Exception:
            if destination.exists() and backup.exists():
                self._remove_output_path(destination)
                backup.rename(destination)
            elif backup.exists() and not destination.exists():
                backup.rename(destination)
            raise
        finally:
            if temporary.exists():
                self._remove_output_path(temporary)
            if backup.exists():
                self._remove_output_path(backup)

    def process_one(
        self,
        folder_name: str,
        overwrite: bool = False,
    ) -> BatchProcessResult:
        if not self.store.folder_exists(folder_name):
            return "skip"

        self.status_manager.ensure_status_file(folder_name)
        output_complete = self._output_complete(folder_name)
        step_done = self.status_manager.is_step_done(folder_name, self.status_step)
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

        self._ensure_required_steps_done(folder_name)
        source_video_path = self.store.get_file_path(folder_name, self.output_filename)
        if not source_video_path.is_file():
            raise FileNotFoundError(f"待切分视频不存在: {source_video_path}")

        osu_files = self.store.find_osu_files(folder_name)
        if not osu_files:
            raise FileNotFoundError(f"{folder_name} 中没有 .osu 谱面文件")

        video_duration_seconds = get_media_duration_seconds(source_video_path)
        beatmap = parse_standard_beatmap(osu_files[0])
        overlap_merge_window_ms = self._overlap_merge_window_ms(beatmap)
        plans = build_segment_plans(
            list(beatmap.hit_objects),
            approach_preempt_ratio=self.approach_preempt_ratio,
            circle_size=beatmap.circle_size,
            min_circle_overlap_ratio=self.min_circle_overlap_ratio,
            priority_merge_window_ms=self.priority_merge_window_ms,
            use_priority_merge=self.use_priority_merge,
            approach_preempt_seconds=beatmap.approach_preempt_ms / 1000.0,
            post_context_seconds=self.post_context_seconds,
            video_duration_seconds=video_duration_seconds,
        )
        self.status_manager.mark_step_pending(
            folder_name,
            self.status_step,
            detail={
                "stage": "segmenting",
                "source_video_path": str(source_video_path),
                "segment_count": len(plans),
                "circle_size": beatmap.circle_size,
                "overlap_merge_window_ms": overlap_merge_window_ms,
                "priority_merge_window_ms": self.priority_merge_window_ms,
                "min_circle_overlap_ratio": self.min_circle_overlap_ratio,
                "use_priority_merge": self.use_priority_merge,
                "post_context_seconds": self.post_context_seconds,
            },
        )
        output_directory, counts = self._build_output(
            folder_name,
            source_video_path,
            osu_files[0],
            beatmap,
            plans,
        )
        self.status_manager.mark_step_done(
            folder_name,
            self.status_step,
            detail={
                "stage": "done",
                "schema_version": SEGMENT_SCHEMA_VERSION,
                "source_video_path": str(source_video_path),
                "output_directory": str(output_directory),
                "segment_table": str(output_directory / SEGMENT_TABLE_FILENAME),
                "segment_count": len(plans),
                "hit_object_count": sum(
                    len(plan.object_types) for plan in plans
                ),
                "category_counts": counts,
                "video_windows_may_overlap": True,
                "hit_object_assignment": "exactly_once",
                "approach_preempt_ratio": self.approach_preempt_ratio,
                "overlap_merge_window_ms": overlap_merge_window_ms,
                "priority_merge_window_ms": self.priority_merge_window_ms,
                "use_priority_merge": self.use_priority_merge,
                "hp_drain_rate": beatmap.hp_drain_rate,
                "circle_size": beatmap.circle_size,
                "circle_radius": (
                    plans[0].circle_radius if plans else None
                ),
                "overall_difficulty": beatmap.overall_difficulty,
                "approach_rate": beatmap.approach_rate,
                "approach_preempt_ms": beatmap.approach_preempt_ms,
                "slider_multiplier": beatmap.slider_multiplier,
                "slider_tick_rate": beatmap.slider_tick_rate,
                "stack_leniency": beatmap.stack_leniency,
                "min_circle_overlap_ratio": self.min_circle_overlap_ratio,
                "post_context_seconds": self.post_context_seconds,
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


__all__ = [
    "SEGMENT_BEATMAP_FILENAME",
    "SEGMENT_MANIFEST_FILENAME",
    "SEGMENT_SCHEMA_VERSION",
    "SEGMENT_TABLE_FILENAME",
    "SEGMENT_TABLE_FIELDS",
    "SEGMENT_VIDEO_FILENAME",
    "VideoSegmentationProcessor",
]
