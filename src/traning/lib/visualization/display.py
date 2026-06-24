from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def launch_image_window(
    image_path: Path,
    *,
    title: str,
    ffplay_binary: str = "ffplay",
    display: str | None = None,
    previous_process: subprocess.Popen[bytes] | None = None,
) -> subprocess.Popen[bytes]:
    selected_display = display if display is not None else os.environ.get("DISPLAY")
    if not selected_display:
        raise RuntimeError("DISPLAY is not configured")
    executable = shutil.which(ffplay_binary)
    if executable is None:
        raise RuntimeError(f"ffplay executable was not found: {ffplay_binary}")
    if previous_process is not None and previous_process.poll() is None:
        previous_process.terminate()

    environment = dict(os.environ)
    environment["DISPLAY"] = selected_display
    environment.setdefault("SDL_VIDEODRIVER", "x11")
    process = subprocess.Popen(
        [
            executable,
            "-loglevel",
            "error",
            "-window_title",
            title,
            "-loop",
            "1",
            str(image_path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=environment,
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        return process
    raise RuntimeError(f"ffplay exited immediately with code {return_code}")


__all__ = ["launch_image_window"]
