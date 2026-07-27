"""通过 ffplay 启动非阻塞图像预览窗口并管理上一窗口进程。"""

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
    """启动 ffplay 循环显示单张图片，并确认进程没有立即失败。"""

    selected_display = display if display is not None else os.environ.get("DISPLAY")
    if not selected_display:
        raise RuntimeError("未配置 DISPLAY，无法打开图像窗口")
    executable = shutil.which(ffplay_binary)
    if executable is None:
        raise RuntimeError(f"未找到 ffplay 可执行文件：{ffplay_binary}")
    if previous_process is not None and previous_process.poll() is None:
        # 训练只保留一个预览窗口，避免周期性可视化累积播放器进程。
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
    # 短暂等待只用于捕获参数、DISPLAY 等导致的立即退出；正常窗口随后异步运行。
    try:
        return_code = process.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        return process
    raise RuntimeError(f"ffplay 立即退出，返回码 {return_code}")


__all__ = ["launch_image_window"]
