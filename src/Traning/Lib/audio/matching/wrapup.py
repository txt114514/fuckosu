from __future__ import annotations

from pathlib import Path
from typing import Any


class AudioMatchWrapUpMixin:
    def _print_greedy_matches(self, matches: list[dict[str, Any]]):
        print()
        print("[建议匹配] 按最高分做贪心一对一")
        for result in matches:
            suffix = ""
            if "verify_adjustment_ms" in result:
                suffix = (
                    f", audio_offset={result['audio_offset_seconds']}"
                    f", verify_adjustment_ms={result['verify_adjustment_ms']}"
                    f", verify_score={result['verify_score']}"
                )
            print(
                f"  {Path(str(result['video_path'])).name} -> {result['folder_name']}"
                f" score={result['match_score']}"
                f", coarse={result['coarse_match_score']}"
                f", offset={result['offset_seconds']}{suffix}"
            )

    def _apply_matches(
        self,
        matches: list[dict[str, Any]],
        pending_folder_names: list[str],
        candidate_videos: list[Path],
    ):
        if not matches:
            raise ValueError("音频实验匹配未生成任何可应用结果")

        plan: list[tuple[str, Path, Path, dict[str, Any]]] = []
        for result in matches:
            folder_name = str(result["folder_name"])
            source_path = Path(str(result["video_path"]))
            destination_path = self.store.get_file_path(
                folder_name,
                f"{folder_name}{source_path.suffix}",
            )
            plan.append((folder_name, source_path, destination_path, result))

        destination_paths = [destination_path for _, _, destination_path, _ in plan]
        if len(set(destination_paths)) != len(destination_paths):
            raise ValueError("音频实验匹配生成了重复目标文件名")

        source_paths = {source_path for _, source_path, _, _ in plan}
        for _, _, destination_path, _ in plan:
            if destination_path.exists() and destination_path not in source_paths:
                raise FileExistsError(f"目标文件已存在，无法覆盖: {destination_path}")

        temp_plan: list[tuple[Path, Path]] = []
        completed_plan: list[tuple[str, Path, Path]] = []

        for index, (_folder_name, source_path, _destination_path, _result) in enumerate(plan):
            temp_path = source_path.with_name(f".__audio_match_tmp__{index}{source_path.suffix}")
            if temp_path.exists():
                raise FileExistsError(f"临时文件已存在，请先清理: {temp_path}")
            source_path.rename(temp_path)
            temp_plan.append((temp_path, source_path))

        try:
            for (temp_path, original_path), (folder_name, _source_path, destination_path, result) in zip(
                temp_plan,
                plan,
            ):
                temp_path.rename(destination_path)
                completed_plan.append((folder_name, destination_path, original_path))
                detail: dict[str, Any] = {
                    "video_path": str(destination_path),
                    "match_strategy": "audio_experiment",
                    "match_score": result["match_score"],
                    "coarse_match_score": result["coarse_match_score"],
                    "audio_offset_seconds": result["audio_offset_seconds"],
                    "offset_seconds": result["offset_seconds"],
                }
                if "verify_adjustment_ms" in result:
                    detail["verify_adjustment_seconds"] = result["verify_adjustment_seconds"]
                    detail["verify_adjustment_ms"] = result["verify_adjustment_ms"]
                    detail["verify_score"] = result["verify_score"]
                    detail["verify_window_ms"] = result["verify_window_ms"]
                self.status_manager.mark_step_done(
                    folder_name,
                    self.match_status_step,
                    detail=detail,
                )
                print(f"[完成] {original_path.name} -> {destination_path}")
        except Exception:
            for folder_name, destination_path, original_path in reversed(completed_plan):
                if destination_path.exists():
                    destination_path.rename(original_path)
                    self.status_manager.mark_step_pending(
                        folder_name,
                        self.match_status_step,
                        detail={"error": "音频实验匹配移动过程中发生异常，已回滚"},
                    )
            for temp_path, original_path in temp_plan:
                if temp_path.exists():
                    temp_path.rename(original_path)
            raise

        matched_folders = {str(result["folder_name"]) for result in matches}
        unmatched_folders = [
            folder_name for folder_name in pending_folder_names if folder_name not in matched_folders
        ]
        for folder_name in unmatched_folders:
            self.status_manager.mark_step_pending(
                folder_name,
                self.match_status_step,
                detail={
                    "stage": "audio_experiment_unmatched",
                    "error": "音频实验匹配阶段未选中对应视频",
                },
            )

        matched_video_names = {Path(str(result["video_path"])).name for result in matches}
        unmatched_videos = [
            video_path.name
            for video_path in candidate_videos
            if video_path.name not in matched_video_names
        ]
        if unmatched_folders:
            print(f"[提示] 未匹配文件夹 {len(unmatched_folders)} 个: {', '.join(unmatched_folders)}")
        if unmatched_videos:
            print(f"[提示] 未使用视频 {len(unmatched_videos)} 个: {', '.join(unmatched_videos)}")

    def run(
        self,
        *,
        apply_matches: bool = False,
        allow_fallback_videos: bool | None = None,
    ):
        if allow_fallback_videos is None:
            allow_fallback_videos = not apply_matches

        folder_names = self._candidate_folder_names(include_existing_video=not apply_matches)
        if not folder_names:
            if apply_matches:
                raise ValueError("order.txt 对应文件夹都已经存在视频文件，无需继续处理")
            raise ValueError("没有找到带参考音频的谱面文件夹")

        videos = self._candidate_videos(allow_fallback=allow_fallback_videos)
        print(f"[实验] 视频候选 {len(videos)} 个，文件夹候选 {len(folder_names)} 个")

        pair_results = self._score_pairs(videos, folder_names)
        matches = self._select_greedy_matches(pair_results)
        self._print_greedy_matches(matches)

        if apply_matches:
            print()
            print("[实验] 将建议匹配实际应用到视频目录")
            self._apply_matches(matches, folder_names, videos)
