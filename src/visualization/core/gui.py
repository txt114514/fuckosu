from __future__ import annotations

import os
import json
import subprocess
import sys
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from visualization.conf.messages import display_text, render_message
from visualization.lib.models import (
    BestParameterRecord,
    CurrentTrainingMetrics,
    DatasetUsageState,
    PipelinePhase,
    PipelineStageState,
    ResourceState,
    TrainingDashboardState,
    TrainingEvent,
    TrainingStopState,
)
from visualization.lib.reporter import DashboardReporter


_STARTUP_PHASES = {
    PipelinePhase.STARTUP.value,
    PipelinePhase.DATA_PREPARATION.value,
    PipelinePhase.PRETRAIN_CHECK.value,
    PipelinePhase.PROGRESSIVE_PREPARATION.value,
}
_EVENT_LIMIT = 1000


def is_gui_environment_available() -> bool:
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return True
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        import PySide6  # noqa: F401
    except Exception:
        return False
    return True


class GuiDashboardRenderer:
    def __init__(self, reporter: DashboardReporter) -> None:
        self.reporter = reporter
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        env = os.environ.copy()
        src_path = str(Path.cwd() / "src")
        env["PYTHONPATH"] = (
            src_path if not env.get("PYTHONPATH") else f"{src_path}:{env['PYTHONPATH']}"
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "visualization.core.gui_app",
                "--state-path",
                str(self.reporter.store.state_path),
            ],
            cwd=Path.cwd(),
            env=env,
            text=True,
        )

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            with suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=5.0)
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5.0)

    def refresh(self) -> None:
        return None


class VisualizationApplication:
    def __init__(self, reporter: DashboardReporter, *, refresh_ms: int = 250) -> None:
        from PySide6 import QtCore

        self.reporter = reporter
        self.bridge = _create_signal_bridge()
        self.window = MainWindow()
        self.window.apply_state(reporter.snapshot())
        self.timer = QtCore.QTimer(self.window.widget)
        self.timer.setInterval(refresh_ms)
        self.timer.timeout.connect(self.refresh_from_reporter)
        self.bridge.refresh_requested.connect(self.refresh_from_reporter)
        self.bridge.stop_requested.connect(self.stop)
        self.timer.start()

    def request_refresh(self) -> None:
        self.bridge.refresh_requested.emit()

    def request_stop(self) -> None:
        self.bridge.stop_requested.emit()

    def refresh_from_reporter(self) -> None:
        self.window.apply_state(self.reporter.snapshot())

    def stop(self) -> None:
        from PySide6 import QtWidgets

        self.timer.stop()
        self.window.close()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.quit()


def _create_signal_bridge():
    from PySide6 import QtCore

    class SignalBridge(QtCore.QObject):
        refresh_requested = QtCore.Signal()
        stop_requested = QtCore.Signal()

    return SignalBridge()


def load_state_snapshot(path: Path) -> TrainingDashboardState | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _state_from_mapping(raw)


def _state_from_mapping(raw: Mapping[str, Any]) -> TrainingDashboardState:
    data = dict(raw)
    data["pipeline_stages"] = {
        str(key): PipelineStageState(**dict(value))
        for key, value in _mapping(data.get("pipeline_stages")).items()
        if isinstance(value, Mapping)
    }
    data["recent_events"] = [
        TrainingEvent(**dict(value))
        for value in data.get("recent_events", ())
        if isinstance(value, Mapping)
    ]
    data["metrics"] = CurrentTrainingMetrics(**_mapping(data.get("metrics")))
    data["dataset_usage"] = DatasetUsageState(**_mapping(data.get("dataset_usage")))
    data["resources"] = ResourceState(**_mapping(data.get("resources")))
    data["best_parameters"] = BestParameterRecord(
        **_mapping(data.get("best_parameters"))
    )
    stop_state = data.get("stop_state")
    data["stop_state"] = (
        TrainingStopState(**dict(stop_state))
        if isinstance(stop_state, Mapping)
        else None
    )
    return TrainingDashboardState(**data)


class MainWindow:
    def __init__(self) -> None:
        from PySide6 import QtCore, QtWidgets

        self.qt = QtCore.Qt
        self.widget = QtWidgets.QMainWindow()
        self.widget.setWindowTitle("osu-ai 训练可视化")
        self.widget.resize(1280, 820)
        self.stack = QtWidgets.QStackedWidget()
        self.startup_page = StartupPage()
        self.training_page = TrainingPage()
        self.stack.addWidget(self.startup_page.widget)
        self.stack.addWidget(self.training_page.widget)
        self.widget.setCentralWidget(self.stack)
        self._timer_like_children: list[Any] = []

    def show(self) -> None:
        self.widget.show()

    def close(self) -> None:
        self.widget.close()

    def isVisible(self) -> bool:
        return self.widget.isVisible()

    def currentWidget(self):
        return self.stack.currentWidget()

    def apply_state(self, state: TrainingDashboardState) -> None:
        if state.pipeline_phase in _STARTUP_PHASES:
            self.stack.setCurrentWidget(self.startup_page.widget)
            self.startup_page.apply_state(state)
        else:
            self.stack.setCurrentWidget(self.training_page.widget)
            self.training_page.apply_state(state)


class StartupPage:
    def __init__(self) -> None:
        from PySide6 import QtWidgets

        self.widget = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(self.widget)
        self.summary = QtWidgets.QLabel()
        self.summary.setTextInteractionFlags(self.summary.textInteractionFlags())
        root.addWidget(self.summary)
        self.stage_table = QtWidgets.QTableWidget(0, 7)
        self.stage_table.setHorizontalHeaderLabels(
            ["任务", "状态", "进度", "警告", "错误", "消息", "输出"]
        )
        self.stage_table.horizontalHeader().setStretchLastSection(True)
        self.stage_table.setAlternatingRowColors(True)
        root.addWidget(self.stage_table, 3)
        self.events = _plain_text()
        root.addWidget(self.events, 2)

    def apply_state(self, state: TrainingDashboardState) -> None:
        self.summary.setText(
            "\n".join(
                (
                    f"整体状态: {state.status}",
                    f"流程阶段: {state.pipeline_phase}",
                    f"当前任务: {state.phase}",
                    f"进度: {_progress(state.global_step, state.target_global_steps)}",
                )
            )
        )
        stages = tuple(state.pipeline_stages.values())
        self.stage_table.setRowCount(len(stages))
        for row, stage in enumerate(stages):
            values = (
                stage.name,
                stage.status,
                _progress(stage.processed, stage.total),
                str(stage.warning_count or ""),
                stage.error_reason or "",
                stage.message or "",
                stage.output_path or "",
            )
            _set_table_row(self.stage_table, row, values)
        _set_events(self.events, state.recent_events)


class TrainingPage:
    def __init__(self) -> None:
        from PySide6 import QtGui, QtWidgets

        self.widget = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(self.widget)
        top = QtWidgets.QHBoxLayout()
        root.addLayout(top, 1)
        self.trial_labels = _label_grid(
            (
                "trial_id",
                "level",
                "phase",
                "step",
                "loss",
                "score",
                "trial_best",
                "global_best",
                "passes",
                "asha",
                "budget",
            )
        )
        top.addLayout(self.trial_labels["layout"], 2)
        self.resource_labels = _label_grid(
            (
                "gpu",
                "gpu_util",
                "gpu_mem",
                "gpu_allocated",
                "gpu_reserved",
                "gpu_avg",
                "gpu_peak",
                "cpu",
                "ram",
            )
        )
        top.addLayout(self.resource_labels["layout"], 1)

        middle = QtWidgets.QSplitter()
        root.addWidget(middle, 4)
        self.parameters_model = QtGui.QStandardItemModel()
        self.parameters_model.setHorizontalHeaderLabels(["参数", "值"])
        self.parameters = QtWidgets.QTreeView()
        self.parameters.setModel(self.parameters_model)
        self.parameters.setAlternatingRowColors(True)
        self.parameters.setUniformRowHeights(True)
        middle.addWidget(self.parameters)
        self.tests = QtWidgets.QTableWidget(0, 4)
        self.tests.setHorizontalHeaderLabels(["测试", "状态", "得分", "阈值"])
        self.tests.horizontalHeader().setStretchLastSection(True)
        middle.addWidget(self.tests)

        self.events = _plain_text()
        root.addWidget(self.events, 2)

    def apply_state(self, state: TrainingDashboardState) -> None:
        runtime = state.current_parameter_status
        metrics = state.metrics
        resources = state.resources
        _set_label_values(
            self.trial_labels,
            {
                "trial_id": state.current_trial_id or "",
                "level": state.current_level or "",
                "phase": state.phase,
                "step": _progress(
                    runtime.get("stage_step"), runtime.get("stage_budget")
                ),
                "loss": _fmt(metrics.loss),
                "score": _fmt(metrics.score),
                "trial_best": _fmt(metrics.parameter_best_score),
                "global_best": _fmt(metrics.run_global_best_score),
                "passes": _progress(
                    state.consecutive_passes,
                    state.required_passes,
                ),
                "asha": "；".join(
                    str(item)
                    for item in (
                        state.trial_status,
                        state.prune_reason,
                        state.promotion_status,
                    )
                    if item
                ),
                "budget": _progress(
                    runtime.get("budget_used"),
                    runtime.get("budget_total"),
                ),
            },
        )
        _set_label_values(
            self.resource_labels,
            {
                "gpu": resources.gpu_name or "",
                "gpu_util": _percent(resources.gpu_utilization),
                "gpu_mem": _gb_pair(
                    resources.gpu_memory_used_gb, resources.gpu_total_gb
                ),
                "gpu_allocated": _fmt(resources.gpu_allocated_gb),
                "gpu_reserved": _fmt(resources.gpu_reserved_gb),
                "gpu_avg": _percent(resources.gpu_utilization_avg),
                "gpu_peak": _percent(resources.gpu_utilization_max),
                "cpu": _percent(resources.cpu_percent),
                "ram": _fmt(resources.process_memory_gb),
            },
        )
        _set_parameter_model(self.parameters_model, state.current_parameters)
        self.parameters.expandToDepth(1)
        _set_tests(self.tests, runtime)
        _set_events(self.events, state.recent_events)


def _label_grid(names: tuple[str, ...]) -> dict[str, Any]:
    from PySide6 import QtWidgets

    layout = QtWidgets.QGridLayout()
    labels: dict[str, Any] = {"layout": layout}
    for row, name in enumerate(names):
        title = QtWidgets.QLabel(name)
        value = QtWidgets.QLabel("")
        value.setTextInteractionFlags(value.textInteractionFlags())
        layout.addWidget(title, row, 0)
        layout.addWidget(value, row, 1)
        labels[name] = value
    return labels


def _set_label_values(labels: Mapping[str, Any], values: Mapping[str, object]) -> None:
    for key, value in values.items():
        label = labels.get(key)
        if label is not None:
            label.setText(str(value))


def _plain_text():
    from PySide6 import QtWidgets

    widget = QtWidgets.QPlainTextEdit()
    widget.setReadOnly(True)
    widget.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
    return widget


def _set_table_row(table, row: int, values: tuple[object, ...]) -> None:
    from PySide6 import QtWidgets

    for column, value in enumerate(values):
        table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))


def _set_tests(table, runtime: Mapping[str, object]) -> None:
    statuses = _mapping(runtime.get("test_statuses"))
    scores = _mapping(runtime.get("test_scores"))
    thresholds = _mapping(runtime.get("test_thresholds"))
    names = tuple(dict.fromkeys((*statuses.keys(), *scores.keys(), *thresholds.keys())))
    table.setRowCount(len(names))
    for row, name in enumerate(names):
        _set_table_row(
            table,
            row,
            (
                name,
                statuses.get(name, "pending"),
                _fmt(scores.get(name)),
                _fmt(thresholds.get(name)),
            ),
        )


def _set_parameter_model(model, parameters: Mapping[str, object]) -> None:
    from PySide6 import QtGui

    model.removeRows(0, model.rowCount())

    def add(parent, key: str, value: object) -> None:
        key_item = QtGui.QStandardItem(str(key))
        value_item = QtGui.QStandardItem(
            "" if isinstance(value, Mapping) else _fmt(value)
        )
        parent.appendRow([key_item, value_item])
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                add(key_item, str(child_key), child_value)

    for key, value in parameters.items():
        add(model, str(key), value)


def _set_events(widget, events: list[TrainingEvent]) -> None:
    selected = events[-_EVENT_LIMIT:]
    lines = [
        f"{event.timestamp} {display_text(event.severity)} "
        f"{_event_message(event)}"
        for event in selected
    ]
    previous_scroll = widget.verticalScrollBar().value()
    at_bottom = previous_scroll == widget.verticalScrollBar().maximum()
    widget.setPlainText("\n".join(lines))
    if at_bottom:
        widget.verticalScrollBar().setValue(widget.verticalScrollBar().maximum())


def _event_message(event: TrainingEvent) -> str:
    if event.raw_message:
        return display_text(event.raw_message)
    message = render_message(event.message_key, event.message_args)
    if not event.message_args:
        return message
    details = "；".join(
        f"{display_text(key)}={display_text(value)}"
        for key, value in sorted(event.message_args.items())
    )
    return f"{message}（{details}）"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _progress(value: object, total: object) -> str:
    if value is None:
        value = 0
    if total in (None, 0):
        return str(value)
    with suppress(TypeError, ValueError):
        ratio = float(value) / float(total) * 100.0
        return f"{int(value)}/{int(total)} ({ratio:.1f}%)"
    return f"{value}/{total}"


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _percent(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}%"


def _gb_pair(used: float | None, total: float | None) -> str:
    if used is None and total is None:
        return ""
    return f"{_fmt(used)} / {_fmt(total)}"
