from __future__ import annotations

import argparse
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from visualization.core.gui import MainWindow, load_state_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="osu-ai PySide6 dashboard")
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    window = MainWindow()

    def refresh() -> None:
        state = load_state_snapshot(args.state_path)
        if state is not None:
            window.apply_state(state)

    refresh()
    window.show()
    timer = QtCore.QTimer(window.widget)
    timer.setInterval(250)
    timer.timeout.connect(refresh)
    timer.start()
    if args.smoke:

        def finish() -> None:
            print(f"window_visible={window.isVisible()}", flush=True)
            app.quit()

        QtCore.QTimer.singleShot(500, finish)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
