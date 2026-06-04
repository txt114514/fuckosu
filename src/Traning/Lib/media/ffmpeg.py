from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any


__all__ = [
    "get_audio_stream_start_time",
    "get_video_size",
    "run_ffmpeg",
    "run_ffprobe_json",
]


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
