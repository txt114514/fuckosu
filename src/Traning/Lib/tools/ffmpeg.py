from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any


__all__ = [
    "build_crop_video_args",
    "build_extract_wav_args",
    "build_trim_video_args",
    "get_audio_stream_start_time",
    "get_video_size",
    "run_ffmpeg",
    "run_ffprobe_json",
]

COMMON_FFMPEG_ARGS = ("-y", "-hide_banner", "-loglevel", "error")
FASTSTART_ARGS = ("-movflags", "+faststart")
AUDIO_AAC_192K_ARGS = ("-c:a", "aac", "-b:a", "192k")
H264_FAST_ARGS = ("-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p")
H264_VERYFAST_ARGS = (
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "18",
    "-pix_fmt",
    "yuv420p",
)


def _command_error_text(
    result: subprocess.CompletedProcess[str],
    unknown_error: str,
) -> str:
    return result.stderr.strip() or result.stdout.strip() or unknown_error


def _run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        check=False,
        capture_output=True,
        text=True,
    )


def run_ffmpeg(args: Sequence[str]):
    result = _run_command(["ffmpeg", *args])
    if result.returncode != 0:
        raise RuntimeError(_command_error_text(result, "未知 ffmpeg 错误"))


def build_extract_wav_args(
    source_path: Path,
    output_path: Path,
    *,
    sample_rate: int,
    from_video: bool,
) -> tuple[str, ...]:
    video_args = ("-vn",) if from_video else ()
    return (
        *COMMON_FFMPEG_ARGS,
        "-i",
        str(source_path),
        *video_args,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    )


def build_trim_video_args(
    source_video_path: Path,
    output_video_path: Path,
    *,
    trim_start_seconds: float,
    trim_duration_seconds: float,
) -> tuple[str, ...]:
    return (
        *COMMON_FFMPEG_ARGS,
        "-i",
        str(source_video_path),
        "-ss",
        f"{trim_start_seconds:.6f}",
        "-t",
        f"{trim_duration_seconds:.6f}",
        *H264_VERYFAST_ARGS,
        *AUDIO_AAC_192K_ARGS,
        *FASTSTART_ARGS,
        str(output_video_path),
    )


def build_crop_video_args(
    source_video_path: Path,
    output_video_path: Path,
    *,
    crop_left: int,
    crop_top: int,
    crop_width: int,
    crop_height: int,
) -> tuple[str, ...]:
    return (
        *COMMON_FFMPEG_ARGS,
        "-i",
        str(source_video_path),
        "-vf",
        f"crop={crop_width}:{crop_height}:{crop_left}:{crop_top}",
        *H264_FAST_ARGS,
        *AUDIO_AAC_192K_ARGS,
        *FASTSTART_ARGS,
        str(output_video_path),
    )


def run_ffprobe_json(
    args: Sequence[str],
    *,
    error_prefix: str,
) -> dict[str, Any]:
    result = _run_command(["ffprobe", *args])
    if result.returncode != 0:
        raise RuntimeError(f"{error_prefix}: {_command_error_text(result, '未知 ffprobe 错误')}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{error_prefix}: ffprobe 输出不是合法 JSON") from e

    if not isinstance(payload, dict):
        raise RuntimeError(f"{error_prefix}: ffprobe JSON 根节点必须是对象")

    return payload


def get_audio_stream_start_time(source_path: Path) -> float:
    payload = run_ffprobe_json(
        [
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=start_time",
            "-of",
            "json",
            str(source_path),
        ],
        error_prefix="读取音频流 start_time 失败",
    )
    streams = payload.get("streams", [])
    if not streams:
        return 0.0

    raw_start_time = streams[0].get("start_time")
    if raw_start_time in (None, "N/A", ""):
        return 0.0
    return float(raw_start_time)


def get_video_size(video_path: Path) -> tuple[int, int]:
    payload = run_ffprobe_json(
        [
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ],
        error_prefix="读取视频尺寸失败",
    )
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"未找到视频流: {video_path}")

    width = int(streams[0]["width"])
    height = int(streams[0]["height"])
    return width, height
