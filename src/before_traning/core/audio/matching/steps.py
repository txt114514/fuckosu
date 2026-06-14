from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from before_traning.Lib.common.failures import exception_detail, format_failure


class AudioMatchStepsMixin:
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
                            **exception_detail(e),
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
                    print(f"  - {item['folder_name']}: {format_failure(item)}")
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
