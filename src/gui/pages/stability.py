from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QStackedWidget,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ... import stability_measurement
from ..analysis_export import export_stability_results, import_stability_results
from ..plot_widgets import MatplotlibFigureWidget


@dataclass
class StabilityInputModel:
    file_path: str = ""


class StabilityRootPage(QWidget):
    file_path_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self._menu = QListWidget()
        self._stack = QStackedWidget()

        self._model = StabilityInputModel()

        self._page_input = StabilityInputPage(self._model)
        self._page_input.file_path_changed.connect(self.file_path_changed)
        self._page_sensor = QLabel(
            "Sensor configuration\n\n"
            "Note: the current stability pipeline is legacy and assumes a fixed wiring/channel order.\n"
            "We’ll unify it with the config-driven `load_data` mapping in a later step."
        )
        self._page_sensor.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._page_results = StabilityResultsPage(self._model)

        self._stack.addWidget(self._wrap(self._page_input))
        self._stack.addWidget(self._wrap(self._page_sensor))
        self._stack.addWidget(self._wrap(self._page_results))

        for name in ["Input data", "Sensor configuration", "Results"]:
            self._menu.addItem(QListWidgetItem(name))
        self._menu.setCurrentRow(0)
        self._menu.currentRowChanged.connect(self._stack.setCurrentIndex)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._menu)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def set_default_load_dir(self, path: str) -> None:
        self._page_input.set_default_load_dir(path)

    def set_file_path(self, path: str) -> None:
        self._page_input.set_file_path(path)

    def get_sidebar_index(self) -> int:
        return int(self._menu.currentRow())

    def set_sidebar_index(self, idx: int) -> None:
        if 0 <= int(idx) < self._stack.count():
            self._menu.setCurrentRow(int(idx))

    def build_session_snapshot(self) -> dict:
        return {
            "experiment_cfg": {
                "analysis_type": "stability_measurement",
                "file_path": self._model.file_path,
            },
            "gui_state": {
                "sidebar_index": self.get_sidebar_index(),
                "stability_results": self._page_results.build_gui_state(),
            },
        }

    def apply_session_snapshot(self, experiment_cfg: dict, gui_state: dict) -> None:
        cfg = dict(experiment_cfg or {})
        self._page_input.set_file_path(str(cfg.get("file_path") or ""))
        gui_state = dict(gui_state or {})
        self.set_sidebar_index(int(gui_state.get("sidebar_index", 0)))
        self._page_results.apply_gui_state(dict(gui_state.get("stability_results", {}) or {}))

    def hydrate_results_from_manifest(self, analysis_folder: Path, manifest: dict) -> None:
        self._page_results.hydrate_from_manifest(analysis_folder, manifest)

    def export_results(self, analysis_folder: Path) -> dict:
        return self._page_results.export_results(analysis_folder)

    @staticmethod
    def _wrap(widget: QWidget) -> QWidget:
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(widget, 1)
        return container


class StabilityInputPage(QWidget):
    file_path_changed = Signal(str)

    def __init__(self, model: StabilityInputModel) -> None:
        super().__init__()
        self._model = model
        self._default_load_dir = ""
        self._last_emitted_path: str = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title = QLabel("Input data")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        form = QFormLayout()
        self._path = QLineEdit()
        self._path.editingFinished.connect(self._sync_model_from_ui)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        form.addRow("CSV file", self._path)
        form.addRow("", browse)

        layout.addLayout(form)
        layout.addStretch(1)

    def set_default_load_dir(self, path: str) -> None:
        self._default_load_dir = path or ""

    def set_file_path(self, path: str) -> None:
        self._path.setText(path or "")
        self._sync_model_from_ui()

    def _sync_model_from_ui(self) -> None:
        p = (self._path.text() or "").strip()
        self._model.file_path = p
        if p != self._last_emitted_path:
            self._last_emitted_path = p
            self.file_path_changed.emit(p)

    def _browse(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV file",
            self._default_load_dir,
            "CSV (*.csv);;All files (*)",
        )
        if not p:
            return
        self._path.setText(p)
        self._sync_model_from_ui()


class StabilityResultsPage(QWidget):
    def __init__(self, model: StabilityInputModel) -> None:
        super().__init__()
        self._model = model
        self._last_result: Optional[dict] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title = QLabel("Results")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        self._run_btn = QPushButton("Generate results")
        self._run_btn.clicked.connect(self._run)
        layout.addWidget(self._run_btn)

        self._tabs = QTabWidget()
        self._tab_spec_x = MatplotlibFigureWidget()
        self._tab_spec_y = MatplotlibFigureWidget()
        self._tab_std = MatplotlibFigureWidget()
        self._tabs.addTab(self._tab_spec_x, "Spectrogram (X)")
        self._tabs.addTab(self._tab_spec_y, "Spectrogram (Y)")
        self._tabs.addTab(self._tab_std, "Rolling 2x std")
        layout.addWidget(self._tabs, 1)

    def build_gui_state(self) -> dict:
        return {"results_tab_index": int(self._tabs.currentIndex())}

    def apply_gui_state(self, gui_state: dict) -> None:
        idx = int(dict(gui_state or {}).get("results_tab_index", 0))
        if 0 <= idx < self._tabs.count():
            self._tabs.setCurrentIndex(idx)

    def hydrate_from_manifest(self, analysis_folder: Path, manifest: dict) -> None:
        if not manifest:
            return
        data = import_stability_results(Path(analysis_folder), dict(manifest or {}))
        if not data:
            return
        self._last_result = data
        self._plot_spectrogram_tab(self._tab_spec_x, data, axis="x")
        self._plot_spectrogram_tab(self._tab_spec_y, data, axis="y")
        self._plot_rolling_std(self._tab_std, data)

    def export_results(self, analysis_folder: Path) -> dict:
        if not self._last_result:
            return {"analysis_type": "stability_measurement", "tables": [], "figures": []}
        return export_stability_results(
            Path(analysis_folder),
            result=dict(self._last_result or {}),
            meta={"file_path": self._model.file_path},
        )

    def _run(self) -> None:
        if not self._model.file_path:
            QMessageBox.warning(self, "Missing input", "Please select a CSV file first.")
            return
        try:
            res = stability_measurement.process_stability_measurement(self._model.file_path, show_plot=False)
            self._last_result = res
            self._plot_spectrogram_tab(self._tab_spec_x, res, axis="x")
            self._plot_spectrogram_tab(self._tab_spec_y, res, axis="y")
            self._plot_rolling_std(self._tab_std, res)
        except Exception as e:
            QMessageBox.critical(self, "Stability run failed", str(e))

    @staticmethod
    def _plot_spectrogram_tab(widget: MatplotlibFigureWidget, res: dict, axis: str) -> None:
        fig = widget.figure
        fig.clear()

        if axis == "x":
            SFT = res["SFT_x"]
            Sxx = res["Sxx_x"]
            milling_start = res.get("milling_start_time")
            milling_end = res.get("milling_end_time")
            chatter = res.get("chatter_x", {})
            label = "X Acceleration"
        else:
            SFT = res["SFT_y"]
            Sxx = res["Sxx_y"]
            milling_start = res.get("milling_start_time")
            milling_end = res.get("milling_end_time")
            chatter = res.get("chatter_y", {})
            label = "Y Acceleration"

        # Use existing helper which returns a Matplotlib Figure
        fig2 = stability_measurement.plot_spectrogram(
            SFT,
            Sxx,
            res["N"],
            title="Stability Spectrogram",
            signal_label=label,
            show_plot=False,
            milling_start_time=milling_start,
            milling_end_time=milling_end,
        )
        if fig2 is not None:
            ax = fig2.axes[0]
            onset = chatter.get("chatter_onset_time")
            if onset is not None:
                ax.axvline(x=onset, color="r", linestyle="--", linewidth=2)
        widget.replace_figure(fig2 if fig2 is not None else fig)

    @staticmethod
    def _plot_rolling_std(widget: MatplotlibFigureWidget, res: dict) -> None:
        fig = widget.figure
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        tx = np.asarray(res.get("time_centers_std_x", []))
        ty = np.asarray(res.get("time_centers_std_y", []))
        sx = np.asarray(res.get("rolling_std_2x_x", []))
        sy = np.asarray(res.get("rolling_std_2x_y", []))
        if len(tx) and len(sx):
            ax.plot(tx, sx, label="X (2x std)")
        if len(ty) and len(sy):
            ax.plot(ty, sy, label="Y (2x std)")
        ax.set_title("Rolling 2x standard deviation")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s²)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        widget.redraw()

