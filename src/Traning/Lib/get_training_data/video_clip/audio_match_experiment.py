from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.function_tools.functions_process_tool import read_config_values
from Traning.Lib.get_training_data.config_loader import (
    AUDIO_MATCH_EXPERIMENT_CONFIG_SPECS,
    ConfigReader,
    build_from_config_or_default,
)
from Traning.Lib.get_training_data.process_status_manager import ProcessStatusManager
from Traning.Lib.get_training_data.video_clip.AV_correspondence import (
    AVCorrespondenceProcessor,
)
from Traning.Lib.traning_package_manager.files_manager import BeatmapFolderStore
from Traning.Lib.traning_package_manager.order_walker import OrderFolderWalker


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_TARGET_ROOT = DEFAULT_REPO_ROOT / "training_package" / "match-completed_package"
DEFAULT_VIDEO_ROOT = DEFAULT_REPO_ROOT / "training_package" / "video_package"
DEFAULT_ORDER_FILENAME = "order.txt"
DEFAULT_AUDIO_FILENAME = "audio.mp3"
DEFAULT_VERIFY_FILENAME = "verify.txt"
DEFAULT_VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".avi", ".mov")
DEFAULT_TOP_K = 3
DEFAULT_MATCH_STATUS_STEP = "video_matched"


def _load_audio_match_experiment_config(config: ConfigReader) -> dict[str, object]:
    return read_config_values(config, AUDIO_MATCH_EXPERIMENT_CONFIG_SPECS)


def build_audio_match_experiment_from_config_or_default(
    config_path: Path | None = None,
) -> "AudioMatchExperiment":
    return build_from_config_or_default(
        AudioMatchExperiment,
        [_load_audio_match_experiment_config],
        config_path=config_path,
        default_builder=AudioMatchExperiment,
    )


class AudioMatchExperiment:
    def __init__(
        self,
        video_root: str = str(DEFAULT_VIDEO_ROOT),
        target_root: str = str(DEFAULT_TARGET_ROOT),
        order_filename: str = DEFAULT_ORDER_FILENAME,
        audio_filename: str = DEFAULT_AUDIO_FILENAME,
        verify_filename: str = DEFAULT_VERIFY_FILENAME,
        video_suffixes: Iterable[str] = DEFAULT_VIDEO_SUFFIXES,
        sample_rate: int = 8000,
        envelope_hz: int = 100,
        refine_hz: int = 1000,
        refine_search_seconds: float = 1.5,
        music_lowpass_hz: int = 1500,
        top_k: int = DEFAULT_TOP_K,
        status_manager: ProcessStatusManager | None = None,
    ):
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        self.video_root = Path(video_root)
        self.target_root = Path(target_root)
        self.order_filename = order_filename
        self.audio_filename = audio_filename
        self.verify_filename = verify_filename
        self.video_suffixes = {suffix.lower() for suffix in video_suffixes}
        self.top_k = top_k
        self.walker = OrderFolderWalker(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )
        self.store = BeatmapFolderStore(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )
        self.aligner = AVCorrespondenceProcessor(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
            audio_filename=self.audio_filename,
            verify_filename=self.verify_filename,
            sample_rate=sample_rate,
            envelope_hz=envelope_hz,
            refine_hz=refine_hz,
            refine_search_seconds=refine_search_seconds,
            music_lowpass_hz=music_lowpass_hz,
            video_suffixes=self.video_suffixes,
        )

    def _folder_has_video(self, folder_name: str) -> bool:
        folder_path = self.store.get_folder_path(folder_name)
        return any(
            child.is_file() and child.suffix.lower() in self.video_suffixes
            for child in folder_path.iterdir()
        )

    def _sync_video_matched_status(self, folder_name: str):
        self.status_manager.ensure_status_file(folder_name)
        has_video = self._folder_has_video(folder_name)
        is_done = self.status_manager.is_step_done(folder_name, DEFAULT_MATCH_STATUS_STEP)
        folder_path = self.store.get_folder_path(folder_name)

        if has_video and not is_done:
            self.status_manager.mark_step_done(
                folder_name,
                DEFAULT_MATCH_STATUS_STEP,
                detail={
                    "folder": str(folder_path),
                    "match_strategy": "existing_file",
                },
            )
            return

        if not has_video and is_done:
            self.status_manager.mark_step_pending(
                folder_name,
                DEFAULT_MATCH_STATUS_STEP,
                detail={"error": "状态显示已匹配视频，但文件夹中未找到视频文件"},
            )

    def _pending_folder_names(self) -> list[str]:
        pending_names: list[str] = []
        for folder_name in self.walker.read_folder_names():
            if not self.store.folder_exists(folder_name):
                raise FileNotFoundError(f"目标文件夹不存在: {self.store.get_folder_path(folder_name)}")

            self._sync_video_matched_status(folder_name)
            if self._folder_has_video(folder_name):
                continue
            if not self.store.file_exists(folder_name, self.audio_filename):
                raise FileNotFoundError(
                    f"{folder_name} 中缺少实验匹配所需音频文件: {self.audio_filename}"
                )
            pending_names.append(folder_name)

        return pending_names

    def _candidate_folder_names(self, *, include_existing_video: bool) -> list[str]:
        pending_names = self._pending_folder_names()
        if pending_names or not include_existing_video:
            return pending_names

        return [
            folder_name
            for folder_name in self.walker.read_folder_names()
            if self.store.folder_exists(folder_name)
            and self.store.file_exists(folder_name, self.audio_filename)
        ]

    def _candidate_videos(self, *, allow_fallback: bool) -> list[Path]:
        if not self.video_root.exists():
            raise FileNotFoundError(f"视频目录不存在: {self.video_root}")

        videos = [
            path
            for path in self.video_root.iterdir()
            if path.is_file() and path.suffix.lower() in self.video_suffixes
        ]
        if videos:
            return sorted(videos, key=lambda path: path.name.lower())

        if not allow_fallback:
            raise ValueError(f"{self.video_root} 中没有可供实验匹配的视频文件")

        # 如果 video_root 里没有待匹配视频，则退回到已导入到谱面目录中的源视频，方便做本地实验。
        fallback_videos: list[Path] = []
        for folder_name in self.walker.read_folder_names():
            for suffix in sorted(self.video_suffixes):
                candidate = self.target_root / folder_name / f"{folder_name}{suffix}"
                if candidate.is_file():
                    fallback_videos.append(candidate)
                    break
        if not fallback_videos:
            raise ValueError("未找到可用于实验的视频文件")
        return fallback_videos

    def _extract_samples(self, source_path: Path, *, from_video: bool) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            self.aligner._extract_audio_to_wav(source_path, wav_path, from_video=from_video)
            return self.aligner._load_wav_samples(wav_path)

    def _build_alignment_features(self, samples: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "coarse": self.aligner._build_feature_series(
                samples,
                self.aligner.envelope_hz,
                mode="energy",
            ),
            "fine": self.aligner._build_music_refine_series(samples),
            "transient": self.aligner._build_feature_series(
                samples,
                self.aligner.refine_hz,
                mode="transient",
            ),
        }

    def _estimate_offset_from_features(
        self,
        video_features: dict[str, np.ndarray],
        song_features: dict[str, np.ndarray],
    ) -> tuple[float, float, float]:
        coarse_start_frame, coarse_score = self.aligner._estimate_best_start_frame(
            video_features["coarse"],
            song_features["coarse"],
        )
        coarse_offset_seconds = coarse_start_frame / float(self.aligner.envelope_hz)

        fine_video = video_features["fine"]
        fine_song = song_features["fine"]
        if fine_video.size < fine_song.size:
            raise ValueError("视频音频时长短于歌曲音频，无法做实验匹配")

        search_margin_frames = max(
            1,
            int(round(self.aligner.refine_search_seconds * self.aligner.refine_hz)),
        )
        coarse_start_frame_fine = int(round(coarse_offset_seconds * self.aligner.refine_hz))
        search_start = max(0, coarse_start_frame_fine - search_margin_frames)
        search_end = min(
            fine_video.size,
            coarse_start_frame_fine + fine_song.size + search_margin_frames,
        )
        if search_end - search_start < fine_song.size:
            search_start = max(0, fine_video.size - fine_song.size)
            search_end = fine_video.size

        fine_start_frame, fine_score = self.aligner._estimate_best_start_frame(
            fine_video[search_start:search_end],
            fine_song,
        )
        refined_offset_seconds = (search_start + fine_start_frame) / float(self.aligner.refine_hz)
        return refined_offset_seconds, fine_score, coarse_score

    def _result_sort_key(self, item: dict[str, Any]) -> tuple[float, float, float, float]:
        verify_score = float(item.get("verify_score", float("-inf")))
        verify_adjustment_ms = abs(float(item.get("verify_adjustment_ms", float("inf"))))
        return (
            float(item["match_score"]),
            verify_score,
            -verify_adjustment_ms,
            float(item["coarse_match_score"]),
        )

    def _score_pairs(
        self,
        videos: list[Path],
        folder_names: list[str],
    ) -> list[dict[str, Any]]:
        folder_features: dict[str, dict[str, np.ndarray]] = {}
        verify_paths: dict[str, Path] = {}
        for folder_name in folder_names:
            audio_path = self.store.get_file_path(folder_name, self.audio_filename)
            folder_features[folder_name] = self._build_alignment_features(
                self._extract_samples(audio_path, from_video=False)
            )
            verify_paths[folder_name] = self.store.get_file_path(folder_name, self.verify_filename)

        pair_results: list[dict[str, Any]] = []
        for video_path in videos:
            video_features = self._build_alignment_features(
                self._extract_samples(video_path, from_video=True)
            )
            video_results: list[dict[str, Any]] = []
            for folder_name in folder_names:
                try:
                    audio_offset_seconds, score, coarse_score = self._estimate_offset_from_features(
                        video_features,
                        folder_features[folder_name],
                    )
                except Exception as e:
                    video_results.append(
                        {
                            "video_path": str(video_path),
                            "video_name": video_path.name,
                            "folder_name": folder_name,
                            "error": str(e),
                        }
                    )
                    continue

                offset_seconds = audio_offset_seconds
                result: dict[str, Any] = {
                    "video_path": str(video_path),
                    "video_name": video_path.name,
                    "folder_name": folder_name,
                    "audio_offset_seconds": round(audio_offset_seconds, 6),
                    "offset_seconds": round(offset_seconds, 6),
                    "match_score": round(score, 6),
                    "coarse_match_score": round(coarse_score, 6),
                }

                verify_adjustment = self.aligner._estimate_verify_adjustment_seconds(
                    video_features["transient"],
                    verify_paths[folder_name],
                    audio_offset_seconds,
                )
                if verify_adjustment is not None:
                    verify_adjustment_seconds, verify_detail = verify_adjustment
                    offset_seconds = audio_offset_seconds + verify_adjustment_seconds
                    result["offset_seconds"] = round(offset_seconds, 6)
                    result.update(
                        {
                            key: round(value, 6)
                            for key, value in verify_detail.items()
                        }
                    )

                video_results.append(result)
                pair_results.append(result)

            scored_results = [item for item in video_results if "match_score" in item]
            scored_results.sort(key=self._result_sort_key, reverse=True)
            print()
            print(f"[视频] {video_path.name}")
            if not scored_results:
                print("  无可用候选，所有对比都失败了")
                for item in video_results[: self.top_k]:
                    print(f"  - {item['folder_name']}: {item['error']}")
                continue

            for rank, result in enumerate(scored_results[: self.top_k], start=1):
                suffix = ""
                if "verify_adjustment_ms" in result:
                    suffix = (
                        f", audio_offset={result['audio_offset_seconds']}"
                        f", verify_adjustment_ms={result['verify_adjustment_ms']}"
                        f", verify_score={result['verify_score']}"
                    )
                print(
                    f"  {rank}. {result['folder_name']}"
                    f" score={result['match_score']}"
                    f", coarse={result['coarse_match_score']}"
                    f", offset={result['offset_seconds']}{suffix}"
                )

        return pair_results

    def _select_greedy_matches(
        self,
        pair_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        used_videos: set[str] = set()
        used_folders: set[str] = set()
        for result in sorted(
            pair_results,
            key=self._result_sort_key,
            reverse=True,
        ):
            video_path = str(result["video_path"])
            folder_name = str(result["folder_name"])
            if video_path in used_videos or folder_name in used_folders:
                continue
            used_videos.add(video_path)
            used_folders.add(folder_name)
            matches.append(result)
        return matches

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
                    DEFAULT_MATCH_STATUS_STEP,
                    detail=detail,
                )
                print(f"[完成] {original_path.name} -> {destination_path}")
        except Exception:
            for folder_name, destination_path, original_path in reversed(completed_plan):
                if destination_path.exists():
                    destination_path.rename(original_path)
                    self.status_manager.mark_step_pending(
                        folder_name,
                        DEFAULT_MATCH_STATUS_STEP,
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
                DEFAULT_MATCH_STATUS_STEP,
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


def main():
    experiment = build_audio_match_experiment_from_config_or_default()
    experiment.run()


if __name__ == "__main__":
    main()
