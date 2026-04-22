from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
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
from ..plot_widgets import MatplotlibFigureWidget


@dataclass
class StabilityInputModel:
    file_path: str = ""


class StabilityRootPage(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self._menu = QListWidget()
        self._stack = QStackedWidget()

        self._model = StabilityInputModel()

        self._page_input = StabilityInputPage(self._model)
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

    @staticmethod
    def _wrap(widget: QWidget) -> QWidget:
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(widget, 1)
        return container


class StabilityInputPage(QWidget):
    def __init__(self, model: StabilityInputModel) -> None:
        super().__init__()
        self._model = model
        self._default_load_dir = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title = QLabel("Input data")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        form = QFormLayout()
        self._path = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        form.addRow("CSV file", self._path)
        form.addRow("", browse)

        layout.addLayout(form)
        layout.addStretch(1)

    def set_default_load_dir(self, path: str) -> None:
        self._default_load_dir = path or ""

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
        self._model.file_path = p


class StabilityResultsPage(QWidget):
    def __init__(self, model: StabilityInputModel) -> None:
        super().__init__()
        self._model = model

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

    def _run(self) -> None:
        if not self._model.file_path:
            QMessageBox.warning(self, "Missing input", "Please select a CSV file first.")
            return
        try:
            res = stability_measurement.process_stability_measurement(self._model.file_path, show_plot=False)
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

