"""启动独立 PySide6 进程并轮询训练进程发布的原子状态快照。"""

from __future__ import annotations

import argparse
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from visualization.core.gui import MainWindow, load_state_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="osu-ai PySide6 dashboard")
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    _apply_cjk_font(app)
    window = MainWindow()

    def refresh() -> None:
        state = load_state_snapshot(args.state_path)
        # 短暂缺失或无效快照不清空窗口，下一次定时轮询会自行恢复。
        if state is not None:
            window.apply_state(state)

    refresh()
    window.show()
    timer = QtCore.QTimer(window.widget)
    # 将定时器挂到窗口对象，确保窗口销毁时 Qt 同步终止刷新生命周期。
    timer.setInterval(250)
    timer.timeout.connect(refresh)
    timer.start()
    if args.smoke:

        def finish() -> None:
            print(f"window_visible={window.isVisible()}", flush=True)
            app.quit()

        QtCore.QTimer.singleShot(500, finish)
    return int(app.exec())


def _apply_cjk_font(app: QtWidgets.QApplication) -> None:
    preferred = ("Noto Sans CJK SC", "Noto Sans CJK", "WenQuanYi Zen Hei")
    available = set(QtGui.QFontDatabase.families())
    for family in preferred:
        if family in available:
            app.setFont(QtGui.QFont(family, 10))
            return


if __name__ == "__main__":
    raise SystemExit(main())
