from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from scipy import signal
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTabWidget,
    QStackedWidget,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QHBoxLayout,
    QCheckBox,
    QScrollArea,
    QVBoxLayout,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from ... import frf_estimation, load_data, signal_processing
from ..plot_widgets import MatplotlibFigureWidget


def _required_wiener_mysize(filt_params: Dict[str, Any]) -> int:
    w = filt_params.get("wiener")
    if not isinstance(w, dict) or "mysize" not in w:
        raise ValueError("filtering.params.wiener.mysize is required (integer samples)")
    m = int(w["mysize"])
    if m < 1:
        raise ValueError("wiener.mysize must be >= 1")
    return m


@dataclass
class FrfInputModel:
    data_source: str = "file"  # "file" | "simulation"
    file_paths_x: List[str] = None  # type: ignore[assignment]
    file_paths_y: List[str] = None  # type: ignore[assignment]
    simulation: Dict[str, Any] = None  # type: ignore[assignment]
    channel_mapping: Dict[str, int] = None  # type: ignore[assignment]
    sensitivity_mapping: Dict[str, str] = None  # type: ignore[assignment]


@dataclass
class _DirectionalFrfResult:
    direction: str  # "x" | "y"
    compliance: Tuple[np.ndarray, np.ndarray]  # (f_hz, H_disp_over_force complex)
    accelerance: Tuple[np.ndarray, np.ndarray]  # (f_hz, H_accel_over_force complex)


@dataclass
class _SignalStep:
    name: str
    y: np.ndarray


@dataclass
class _DirectionalProcessing:
    direction: str  # "x" | "y"
    time: np.ndarray
    fs_hz: float

    # Signals (each list is a processing-chain overlay for one axis)
    u_steps: List[_SignalStep]
    resp_steps: List[_SignalStep]  # acceleration response for this direction

    # Window used (for optional overlay/inspection)
    window: Optional[np.ndarray]

    # Frequency-domain diagnostics (Welch grid)
    f_hz: np.ndarray
    psd_u: np.ndarray
    coherence_uy: np.ndarray
    csd_ur: np.ndarray
    psd_r: np.ndarray

    # FFT magnitude (separate plot) for selected steps (raw + last filtered + windowed)
    fft_f_hz: np.ndarray
    fft_u_mag: Dict[str, np.ndarray]  # step-name -> |FFT|
    fft_resp_mag: Dict[str, np.ndarray]
    fft_u_complex: Dict[str, np.ndarray]  # step-name -> FFT complex
    fft_resp_complex: Dict[str, np.ndarray]


@dataclass
class _DirectionalMeasurement:
    direction: str  # "x" | "y"
    filepath: Optional[str]
    raw_channels: Optional[Dict[int, np.ndarray]]
    proc: _DirectionalProcessing
    compliance: Tuple[np.ndarray, np.ndarray]  # (f_hz, H_disp_over_force complex)
    accelerance: Tuple[np.ndarray, np.ndarray]  # (f_hz, H_accel_over_force complex)


class FrfRootPage(QWidget):
    default_sensor_config_save_requested = Signal(dict, dict, dict)
    default_filtering_save_requested = Signal(dict)
    file_paths_changed = Signal(object, object)  # (x_paths, y_paths) where each is List[str]

    def __init__(self) -> None:
        super().__init__()

        self._menu = QListWidget()
        self._stack = QStackedWidget()

        repo_root = Path(__file__).resolve().parents[3]
        cfg_path = repo_root / "config" / "config_before_cabinet_assembly.json"
        base = json_load(cfg_path) if cfg_path.exists() else {}
        self._input_model = FrfInputModel(
            simulation={},
            file_paths_x=[],
            file_paths_y=[],
            channel_mapping=dict(base.get("channel_mapping", {}) or {}),
            sensitivity_mapping=dict(base.get("sensitivity_mapping", {}) or {}),
        )

        self._page_input = FrfInputPage(self._input_model)
        self._page_sensor = FrfSensorConfigPage(self._input_model)
        self._page_filtering = FrfFilteringPage()
        self._page_results = FrfResultsPage(self._input_model)

        self._page_sensor.default_save_requested.connect(self.default_sensor_config_save_requested)
        self._page_input.file_paths_changed.connect(self.file_paths_changed)
        self._page_filtering.default_save_requested.connect(self.default_filtering_save_requested)
        self._page_filtering.filtering_changed.connect(self._page_results.set_filtering_cfg)

        self._stack.addWidget(self._wrap(self._page_input))
        self._stack.addWidget(self._wrap(self._page_sensor))
        self._stack.addWidget(self._wrap(self._page_filtering))
        self._stack.addWidget(self._wrap(self._page_results))

        for name in ["Input data", "Sensor configuration", "Analysis settings", "Results"]:
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

        # Ensure the Results page sees the initial filtering config (defaults).
        self._page_results.set_filtering_cfg(self._page_filtering.build_filtering_cfg())

    def set_default_load_dir(self, path: str) -> None:
        self._page_input.set_default_load_dir(path)

    def set_file_paths(self, x_paths: List[str], y_paths: List[str]) -> None:
        self._page_input.set_file_paths(list(x_paths or []), list(y_paths or []))

    def apply_default_sensor_config(
        self,
        channel_mapping: Dict[str, int],
        sensitivity_mapping: Dict[str, str],
        sensitivities: Dict[str, float],
    ) -> None:
        self._page_sensor.apply_default_sensor_config(
            channel_mapping=channel_mapping,
            sensitivity_mapping=sensitivity_mapping,
            sensitivities=sensitivities,
        )

    def export_results(self, analysis_folder: Path) -> dict:
        return self._page_results.export_results(analysis_folder)

    def apply_default_filtering(self, filtering_cfg: Dict[str, Any]) -> None:
        self._page_filtering.apply_filtering_cfg(dict(filtering_cfg or {}))

    def build_experiment_cfg(self) -> Dict[str, Any]:
        base = self._page_results._build_experiment_cfg()
        f_cfg = self._page_filtering.build_filtering_cfg()
        base["filtering"] = f_cfg
        if "bias_window_size" in f_cfg:
            base["bias_window_size"] = int(f_cfg["bias_window_size"])
        return base

    def apply_experiment_cfg(self, cfg: Dict[str, Any]) -> None:
        cfg = dict(cfg or {})
        filt = dict(cfg.get("filtering") or {})
        if "bias_window_size" in cfg and "bias_window_size" not in filt:
            filt["bias_window_size"] = cfg["bias_window_size"]
        if isinstance(filt, dict):
            self._page_filtering.apply_filtering_cfg(filt)

    @staticmethod
    def _wrap(widget: QWidget) -> QWidget:
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(widget, 1)
        return container


class FrfInputPage(QWidget):
    file_paths_changed = Signal(object, object)  # (x_paths, y_paths)

    def __init__(self, model: FrfInputModel) -> None:
        super().__init__()
        self._model = model
        self._default_load_dir = ""
        self._last_emitted_paths: Tuple[Tuple[str, ...], Tuple[str, ...]] = (tuple(), tuple())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title = QLabel("Input data")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        # Mode toggle
        mode_row = QWidget()
        mode_l = QVBoxLayout(mode_row)
        mode_l.setContentsMargins(0, 0, 0, 0)
        mode_l.addWidget(QLabel("Data source"))

        self._rb_file = QRadioButton("File")
        self._rb_sim = QRadioButton("Simulation")
        self._rb_file.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self._rb_file)
        bg.addButton(self._rb_sim)
        self._rb_file.toggled.connect(self._on_mode_changed)
        self._rb_sim.toggled.connect(self._on_mode_changed)
        mode_l.addWidget(self._rb_file)
        mode_l.addWidget(self._rb_sim)
        layout.addWidget(mode_row)

        # File form
        self._file_box = QWidget()
        file_form = QFormLayout(self._file_box)
        file_form.setContentsMargins(0, 0, 0, 0)

        self._x_list = QListWidget()
        self._x_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._x_browse = QPushButton("Browse…")
        self._x_browse.clicked.connect(lambda: self._browse_into_list(self._x_list))
        self._x_clear = QPushButton("Clear")
        self._x_clear.clicked.connect(lambda: self._clear_list(self._x_list))
        x_row = QWidget()
        x_l = QVBoxLayout(x_row)
        x_l.setContentsMargins(0, 0, 0, 0)
        x_l.addWidget(self._x_list, 1)
        x_btns = QWidget()
        x_btns_l = QHBoxLayout(x_btns)
        x_btns_l.setContentsMargins(0, 0, 0, 0)
        x_btns_l.addWidget(self._x_browse)
        x_btns_l.addWidget(self._x_clear)
        x_l.addWidget(x_btns)
        file_form.addRow("X files", x_row)

        self._y_list = QListWidget()
        self._y_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._y_browse = QPushButton("Browse…")
        self._y_browse.clicked.connect(lambda: self._browse_into_list(self._y_list))
        self._y_clear = QPushButton("Clear")
        self._y_clear.clicked.connect(lambda: self._clear_list(self._y_list))
        y_row = QWidget()
        y_l = QVBoxLayout(y_row)
        y_l.setContentsMargins(0, 0, 0, 0)
        y_l.addWidget(self._y_list, 1)
        y_btns = QWidget()
        y_btns_l = QHBoxLayout(y_btns)
        y_btns_l.setContentsMargins(0, 0, 0, 0)
        y_btns_l.addWidget(self._y_browse)
        y_btns_l.addWidget(self._y_clear)
        y_l.addWidget(y_btns)
        file_form.addRow("Y files", y_row)

        layout.addWidget(self._file_box)

        # Simulation form (minimal; will be expanded)
        self._sim_box = QWidget()
        sim_form = QFormLayout(self._sim_box)
        sim_form.setContentsMargins(0, 0, 0, 0)

        self._sim_mass = QLineEdit("8.0")
        self._sim_stiff = QLineEdit("1e6")
        self._sim_damp = QLineEdit("0.02")
        self._sim_fs = QLineEdit("8000")
        self._sim_T = QLineEdit("0.2")
        self._sim_hammer_force = QLineEdit("50.0")
        self._sim_hammer_bw = QLineEdit("1000.0")

        sim_form.addRow("Mass (kg)", self._sim_mass)
        sim_form.addRow("Stiffness (N/m)", self._sim_stiff)
        sim_form.addRow("Damping ratio (ζ)", self._sim_damp)
        sim_form.addRow("fs (Hz)", self._sim_fs)
        sim_form.addRow("Measurement duration (s)", self._sim_T)
        sim_form.addRow("Hammer force (N)", self._sim_hammer_force)
        sim_form.addRow("Hammer bandwidth (Hz)", self._sim_hammer_bw)

        layout.addWidget(self._sim_box)
        layout.addStretch(1)

        self._sim_box.setEnabled(False)
        self._on_mode_changed()

        # Ensure manual edits are reflected in the model before running.
        # (Previously we only synced on browse + mode toggle, so typed values were ignored.)
        for le in [
            self._sim_mass,
            self._sim_stiff,
            self._sim_damp,
            self._sim_fs,
            self._sim_T,
            self._sim_hammer_force,
            self._sim_hammer_bw,
        ]:
            le.editingFinished.connect(self._sync_model_from_ui)

        self._x_list.itemChanged.connect(self._sync_model_from_ui)
        self._y_list.itemChanged.connect(self._sync_model_from_ui)

    def set_default_load_dir(self, path: str) -> None:
        self._default_load_dir = path or ""

    def set_file_paths(self, x_paths: List[str], y_paths: List[str]) -> None:
        self._set_list_items(self._x_list, x_paths or [])
        self._set_list_items(self._y_list, y_paths or [])
        self._sync_model_from_ui()

    def _browse_into_list(self, list_widget: QListWidget) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV file",
            self._default_load_dir,
            "CSV (*.csv);;All files (*)",
        )
        if not paths:
            return
        current = self._get_list_items(list_widget)
        merged = current + [p for p in paths if isinstance(p, str) and p.strip()]
        self._set_list_items(list_widget, merged)
        self._sync_model_from_ui()

    @staticmethod
    def _clear_list(list_widget: QListWidget) -> None:
        list_widget.clear()

    @staticmethod
    def _get_list_items(list_widget: QListWidget) -> List[str]:
        out: List[str] = []
        for i in range(list_widget.count()):
            it = list_widget.item(i)
            if it is None:
                continue
            s = (it.text() or "").strip()
            if s:
                out.append(s)
        return out

    @staticmethod
    def _set_list_items(list_widget: QListWidget, paths: List[str]) -> None:
        # Deduplicate while preserving order.
        seen: set[str] = set()
        uniq: List[str] = []
        for p in paths:
            if p is None:
                continue
            s = str(p).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        list_widget.blockSignals(True)
        try:
            list_widget.clear()
            for p in uniq:
                it = QListWidgetItem(p)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                list_widget.addItem(it)
        finally:
            list_widget.blockSignals(False)

    def _on_mode_changed(self) -> None:
        self._file_box.setEnabled(self._rb_file.isChecked())
        self._sim_box.setEnabled(self._rb_sim.isChecked())
        self._sync_model_from_ui()

    def _sync_model_from_ui(self) -> None:
        self._model.data_source = "simulation" if self._rb_sim.isChecked() else "file"
        self._model.file_paths_x = self._get_list_items(self._x_list)
        self._model.file_paths_y = self._get_list_items(self._y_list)
        cur_paths = (tuple(self._model.file_paths_x or []), tuple(self._model.file_paths_y or []))
        if cur_paths != self._last_emitted_paths:
            self._last_emitted_paths = cur_paths
            self.file_paths_changed.emit(list(cur_paths[0]), list(cur_paths[1]))
        self._model.simulation = {
            "fs": float(self._sim_fs.text()),
            "T": float(self._sim_T.text()),
            "model": {
                "mass": float(self._sim_mass.text()),
                "stiffness": float(self._sim_stiff.text()),
                "damping": float(self._sim_damp.text()),
            },
            "hammer": {
                "force": float(self._sim_hammer_force.text()),
                "bandwidth_hz": float(self._sim_hammer_bw.text()),
            },
        }


class _ChainList(QWidget):
    """
    Ordered, checkable list for filter chains.

    Each item is checkable; ordering is controlled via Up/Down buttons.
    """

    changed = Signal()

    def __init__(self, title: str, entries: List[Tuple[str, str]]) -> None:
        super().__init__()
        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        for key, label in entries:
            it = QListWidgetItem(label)
            it.setData(Qt.ItemDataRole.UserRole, key)
            it.setFlags(
                it.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            it.setCheckState(Qt.CheckState.Unchecked)
            self._list.addItem(it)

        # Emit change on toggle
        self._list.itemChanged.connect(lambda _it: self._emit_changed())

        self._btn_up = QPushButton("Up")
        self._btn_down = QPushButton("Down")
        self._btn_up.clicked.connect(lambda: self._move(-1))
        self._btn_down.clicked.connect(lambda: self._move(+1))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(title))
        layout.addWidget(self._list, 1)
        layout.addWidget(self._btn_up)
        layout.addWidget(self._btn_down)

        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def _move(self, delta: int) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        new_row = row + int(delta)
        if new_row < 0 or new_row >= self._list.count():
            return
        it = self._list.takeItem(row)
        self._list.insertItem(new_row, it)
        self._list.setCurrentRow(new_row)
        self._emit_changed()

    def _emit_changed(self) -> None:
        # Delay-free: keep it simple, updates are cheap.
        try:
            self.changed.emit()  # type: ignore[attr-defined]
        except Exception:
            pass

    def set_show_all_items(self) -> None:
        """
        Size the list so all items are visible (no internal scrolling).
        """
        if self._list.count() <= 0:
            return
        row_h = self._list.sizeHintForRow(0)
        if row_h <= 0:
            row_h = 22
        total = row_h * self._list.count() + (2 * self._list.frameWidth()) + 6
        self._list.setMinimumHeight(total)
        self._list.setMaximumHeight(total)

    def get_chain(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i in range(self._list.count()):
            it = self._list.item(i)
            key = it.data(Qt.ItemDataRole.UserRole)
            enabled = it.checkState() == Qt.CheckState.Checked
            if isinstance(key, str) and key:
                out.append({"name": key, "enabled": bool(enabled)})
        return out

    def set_chain(self, chain: List[Dict[str, Any]]) -> None:
        chain = list(chain or [])
        desired = [c.get("name") for c in chain if isinstance(c, dict)]

        items: List[QListWidgetItem] = []
        while self._list.count():
            items.append(self._list.takeItem(0))

        by_key: Dict[str, QListWidgetItem] = {}
        for it in items:
            k = it.data(Qt.ItemDataRole.UserRole)
            if isinstance(k, str):
                by_key[k] = it

        for k in desired:
            if isinstance(k, str) and k in by_key:
                self._list.addItem(by_key.pop(k))
        for it in by_key.values():
            self._list.addItem(it)

        enabled_by_key = {c.get("name"): bool(c.get("enabled", False)) for c in chain if isinstance(c, dict)}
        for i in range(self._list.count()):
            it = self._list.item(i)
            k = it.data(Qt.ItemDataRole.UserRole)
            if isinstance(k, str) and k in enabled_by_key:
                it.setCheckState(
                    Qt.CheckState.Checked if enabled_by_key[k] else Qt.CheckState.Unchecked
                )
        self.set_show_all_items()
        self._emit_changed()


class _WidthSyncedScrollArea(QScrollArea):
    """Scroll area whose document widget stays at least as wide as the viewport."""

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        w = self.widget()
        if w is not None:
            w.setMinimumWidth(max(0, self.viewport().width()))


class FrfFilteringPage(QWidget):
    default_save_requested = Signal(dict)
    filtering_changed = Signal(dict)

    def __init__(self) -> None:
        super().__init__()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = _WidthSyncedScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(16)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Analysis settings")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        # Output selection (input is fixed to Force F_x/F_y)
        io_header = QLabel("Output selection")
        io_header.setStyleSheet("font-weight: 600; font-style: italic;")
        layout.addWidget(io_header)
        io_box = QWidget()
        io_form = QFormLayout(io_box)
        io_form.setContentsMargins(0, 0, 0, 0)
        io_form.setVerticalSpacing(8)
        io_form.setHorizontalSpacing(10)

        self._io_output_family = QComboBox()
        for key, label in [
            ("tool", "Tool (x_tool/y_tool)"),
            ("spindle", "Spindle (x_spindle/y_spindle)"),
        ]:
            self._io_output_family.addItem(label, userData=key)
        # Default matches previous behavior (force input -> tool output)
        self._io_output_family.setCurrentIndex(0)
        io_form.addRow(QLabel("Output"), self._io_output_family)
        layout.addWidget(io_box)

        filtering_header = QLabel("Filtering")
        filtering_header.setStyleSheet("font-weight: 600; font-style: italic;")
        layout.addWidget(filtering_header)

        entries = [("wiener", "Wiener"), ("savgol", "Savitzky–Golay")]
        chain_split = QSplitter(Qt.Orientation.Horizontal)
        self._u_chain = _ChainList("Force (u) filter chain", entries)
        self._resp_chain = _ChainList("Response filter chain", entries)
        chain_split.addWidget(self._u_chain)
        chain_split.addWidget(self._resp_chain)
        chain_split.setStretchFactor(0, 1)
        chain_split.setStretchFactor(1, 1)
        layout.addWidget(chain_split)

        self._pipeline_preview = QLabel("")
        layout.addWidget(self._pipeline_preview)

        # Filter parameters
        filt_params = QWidget()
        filt_form = QFormLayout(filt_params)
        filt_form.setContentsMargins(0, 0, 0, 0)
        filt_form.setVerticalSpacing(8)
        filt_form.setHorizontalSpacing(10)

        self._bias_window_size = QLineEdit("1000")
        filt_form.addRow(QLabel("Bias window size (samples)"), self._bias_window_size)

        self._wiener_mysize = QLineEdit("1000")
        filt_form.addRow("Wiener local window / mysize (samples)", self._wiener_mysize)

        self._savgol_wl = QLineEdit("10")
        self._savgol_po = QLineEdit("2")
        filt_form.addRow("SavGol window_length", self._savgol_wl)
        filt_form.addRow("SavGol polyorder", self._savgol_po)

        layout.addWidget(filt_params)

        # Windowing (final step)
        layout.addWidget(QLabel("Windowing (final step)"))
        win_params = QWidget()
        win_form = QFormLayout(win_params)
        win_form.setContentsMargins(0, 0, 0, 0)
        win_form.setVerticalSpacing(8)
        win_form.setHorizontalSpacing(10)

        self._win_in_type = QComboBox()
        self._win_out_type = QComboBox()
        for nm in ["hann", "exponential", "raised-cosine", "signal-based", "constant"]:
            self._win_in_type.addItem(nm, userData=nm)
            self._win_out_type.addItem(nm, userData=nm)
        self._win_in_type.setCurrentIndex(0)
        self._win_out_type.setCurrentIndex(1)
        win_form.addRow(QLabel("Input window type"), self._win_in_type)
        win_form.addRow(QLabel("Output window type"), self._win_out_type)

        self._win_hann_width = QLineEdit("0.05")
        self._win_pre = QLineEdit("0.05")
        self._win_post = QLineEdit("0.1")
        self._win_beta = QLineEdit("0.25")
        self._win_rolloff = QLineEdit("0.005")
        self._win_trigger_pct = QLineEdit("0.1")
        self._win_low_pct = QLineEdit("0.001")
        self._win_sliding = QLineEdit("200")
        self._win_std_mult = QLineEdit("3.0")
        self._win_decay_db = QLineEdit("40.0")

        self._win_rows: Dict[str, Tuple[QLabel, QWidget]] = {}

        def _add_row(key: str, label: str, w: QWidget) -> None:
            lab = QLabel(label)
            win_form.addRow(lab, w)
            self._win_rows[key] = (lab, w)

        _add_row("window_hann_width", "window_hann_width (s)", self._win_hann_width)
        _add_row("window_pre_seconds", "window_pre_seconds (s)", self._win_pre)
        _add_row("window_post_seconds", "window_post_seconds (s)", self._win_post)
        _add_row("window_beta", "window_beta", self._win_beta)
        _add_row("window_rolloff_seconds", "window_rolloff_seconds (s)", self._win_rolloff)
        _add_row("window_trigger_percent", "window_trigger_percent", self._win_trigger_pct)
        _add_row("window_low_percent", "window_low_percent", self._win_low_pct)
        _add_row("window_sliding_size", "window_sliding_size (samples)", self._win_sliding)
        _add_row("window_std_multiplier", "window_std_multiplier", self._win_std_mult)
        _add_row("window_decay_db", "window_decay_db", self._win_decay_db)

        layout.addWidget(win_params)

        # Estimation
        estimation_header = QLabel("Estimation")
        estimation_header.setStyleSheet("font-weight: 600; font-style: italic;")
        layout.addWidget(estimation_header)
        est_box = QWidget()
        est_form = QFormLayout(est_box)
        est_form.setContentsMargins(0, 0, 0, 0)
        est_form.setVerticalSpacing(8)
        est_form.setHorizontalSpacing(10)

        self._estimator_type = QComboBox()
        for nm in ["H1", "H2", "Hv"]:
            self._estimator_type.addItem(nm, userData=nm)
        # Default estimator is Hv (matches src/frf_estimation.get_system_parameters()).
        self._estimator_type.setCurrentIndex(2)
        est_form.addRow(QLabel("Estimator type"), self._estimator_type)

        self._estimator_help = QLabel("")
        self._estimator_help.setWordWrap(True)
        est_form.addRow(QLabel(""), self._estimator_help)
        layout.addWidget(est_box)

        self._btn_save_default = QPushButton("Save as default")
        self._btn_save_default.clicked.connect(self._on_save_default)
        layout.addWidget(self._btn_save_default)

        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # Default order: wiener then savgol (enabled)
        self._u_chain.set_chain([{"name": "wiener", "enabled": True}, {"name": "savgol", "enabled": True}])
        self._resp_chain.set_chain([{"name": "wiener", "enabled": True}, {"name": "savgol", "enabled": True}])
        self._u_chain.set_show_all_items()
        self._resp_chain.set_show_all_items()
        self._refresh_preview()

        for w in [
            self._bias_window_size,
            self._wiener_mysize,
            self._savgol_wl,
            self._savgol_po,
            self._win_hann_width,
            self._win_pre,
            self._win_post,
            self._win_beta,
            self._win_rolloff,
            self._win_trigger_pct,
            self._win_low_pct,
            self._win_sliding,
            self._win_std_mult,
            self._win_decay_db,
        ]:
            w.editingFinished.connect(self._refresh_preview)
        # Live pipeline updates
        self._u_chain.changed.connect(self._refresh_preview)  # type: ignore[attr-defined]
        self._resp_chain.changed.connect(self._refresh_preview)  # type: ignore[attr-defined]
        self._win_in_type.currentIndexChanged.connect(self._on_window_type_changed)
        self._win_out_type.currentIndexChanged.connect(self._on_window_type_changed)
        self._estimator_type.currentIndexChanged.connect(self._on_estimator_changed)
        self._io_output_family.currentIndexChanged.connect(self._refresh_preview)

        self._update_window_param_visibility()
        self._update_estimator_help()

    def _on_window_type_changed(self) -> None:
        self._update_window_param_visibility()
        self._refresh_preview()

    def _on_estimator_changed(self) -> None:
        self._update_estimator_help()
        self._refresh_preview()

    def _update_estimator_help(self) -> None:
        et = str(self._estimator_type.currentData() or "Hv")
        if et == "H1":
            txt = "H₁(ω) = Gᵧᵤ(ω) / Gᵤᵤ(ω)"
        elif et == "H2":
            txt = "H₂(ω) = Gᵧᵧ(ω) / Gᵧᵤ(ω)"
        else:
            txt = "Hᵥ(ω) = (Gᵧᵤ(ω)/|Gᵧᵤ(ω)|) · √(Gᵧᵧ(ω) / Gᵤᵤ(ω))"
        self._estimator_help.setText(txt)

    def _update_window_param_visibility(self) -> None:
        in_type = str(self._win_in_type.currentData() or "hann")
        out_type = str(self._win_out_type.currentData() or "exponential")
        active = {in_type, out_type}

        show: set[str] = set()
        if "hann" in active:
            show |= {"window_hann_width"}
        if "exponential" in active:
            show |= {"window_decay_db"}
        if "raised-cosine" in active:
            show |= {"window_pre_seconds", "window_post_seconds", "window_beta"}
        if "signal-based" in active:
            show |= {
                "window_rolloff_seconds",
                "window_trigger_percent",
                "window_low_percent",
                "window_sliding_size",
                "window_std_multiplier",
            }
        # constant: no params

        for key, (lab, w) in (self._win_rows or {}).items():
            vis = key in show
            lab.setVisible(vis)
            w.setVisible(vis)

    def _refresh_preview(self) -> None:
        def fmt(chain: List[Dict[str, Any]]) -> str:
            parts: List[str] = []
            for c in chain:
                if not isinstance(c, dict):
                    continue
                if not c.get("enabled", False):
                    continue
                nm = c.get("name")
                if isinstance(nm, str) and nm:
                    parts.append(nm)
            return " → ".join(parts) if parts else "(no filters)"

        self._pipeline_preview.setText(
            f"Pipeline: raw → bias → u:[{fmt(self._u_chain.get_chain())}] → resp:[{fmt(self._resp_chain.get_chain())}] → windowing"
        )
        self._emit_filtering_changed()

    def _emit_filtering_changed(self) -> None:
        try:
            self.filtering_changed.emit(self.build_filtering_cfg())
        except Exception:
            pass

    def _on_save_default(self) -> None:
        self.default_save_requested.emit(self.build_filtering_cfg())

    def build_filtering_cfg(self) -> Dict[str, Any]:
        self._refresh_preview()

        def i(txt: str) -> int:
            return int(float((txt or "").strip() or "0") or 0)

        def f(txt: str) -> float:
            return float((txt or "").strip() or "0")

        return {
            "bias_window_size": i(self._bias_window_size.text()),
            "estimator_type": str(self._estimator_type.currentData() or "Hv"),
            "io_selection": {
                "input_family": "force",
                "output_family": str(self._io_output_family.currentData() or "tool"),
            },
            "u_chain": self._u_chain.get_chain(),
            "resp_chain": self._resp_chain.get_chain(),
            "params": {
                "wiener": {"mysize": i(self._wiener_mysize.text())},
                "savgol": {"window_length": i(self._savgol_wl.text()), "polyorder": i(self._savgol_po.text())},
            },
            "windowing": {
                "input_window_type": str(self._win_in_type.currentData() or "hann"),
                "output_window_type": str(self._win_out_type.currentData() or "exponential"),
                "window_hann_width": f(self._win_hann_width.text()),
                "window_pre_seconds": f(self._win_pre.text()),
                "window_post_seconds": f(self._win_post.text()),
                "window_beta": f(self._win_beta.text()),
                "window_rolloff_seconds": f(self._win_rolloff.text()),
                "window_trigger_percent": f(self._win_trigger_pct.text()),
                "window_low_percent": f(self._win_low_pct.text()),
                "window_sliding_size": i(self._win_sliding.text()),
                "window_std_multiplier": f(self._win_std_mult.text()),
                "window_decay_db": f(self._win_decay_db.text()),
            },
        }

    def apply_filtering_cfg(self, cfg: Dict[str, Any]) -> None:
        cfg = dict(cfg or {})
        if "bias_window_size" in cfg:
            self._bias_window_size.setText(str(int(cfg["bias_window_size"])))
        # IO selection
        io_sel = cfg.get("io_selection") or {}
        if isinstance(io_sel, dict):
            def _set_combo(cb: QComboBox, value: str) -> None:
                for idx in range(cb.count()):
                    if cb.itemData(idx) == value:
                        cb.setCurrentIndex(idx)
                        return

            if "output_family" in io_sel:
                _set_combo(self._io_output_family, str(io_sel.get("output_family") or "tool"))

        # Estimator
        if "estimator_type" in cfg:
            desired = str(cfg.get("estimator_type") or "Hv")
            for idx in range(self._estimator_type.count()):
                if self._estimator_type.itemData(idx) == desired:
                    self._estimator_type.setCurrentIndex(idx)
                    break
        self._update_estimator_help()

        if isinstance(cfg.get("u_chain"), list):
            self._u_chain.set_chain(cfg.get("u_chain") or [])
        if isinstance(cfg.get("resp_chain"), list):
            self._resp_chain.set_chain(cfg.get("resp_chain") or [])

        params = cfg.get("params") or {}
        if isinstance(params, dict):
            w = params.get("wiener") or {}
            if isinstance(w, dict) and "mysize" in w:
                self._wiener_mysize.setText(str(w.get("mysize") or ""))
            s = params.get("savgol") or {}
            if isinstance(s, dict):
                if "window_length" in s:
                    self._savgol_wl.setText(str(s.get("window_length") or ""))
                if "polyorder" in s:
                    self._savgol_po.setText(str(s.get("polyorder") or ""))

        win = cfg.get("windowing") or {}
        if isinstance(win, dict):
            _set_combo(self._win_in_type, str(win.get("input_window_type") or "hann"))
            _set_combo(self._win_out_type, str(win.get("output_window_type") or "exponential"))

            mapping = {
                "window_hann_width": self._win_hann_width,
                "window_pre_seconds": self._win_pre,
                "window_post_seconds": self._win_post,
                "window_beta": self._win_beta,
                "window_rolloff_seconds": self._win_rolloff,
                "window_trigger_percent": self._win_trigger_pct,
                "window_low_percent": self._win_low_pct,
                "window_sliding_size": self._win_sliding,
                "window_std_multiplier": self._win_std_mult,
                "window_decay_db": self._win_decay_db,
            }
            for k, le in mapping.items():
                if k in win:
                    le.setText(str(win.get(k)))

        self._update_window_param_visibility()
        self._refresh_preview()
        self._emit_filtering_changed()


class FrfResultsPage(QWidget):
    def __init__(self, model: FrfInputModel) -> None:
        super().__init__()
        self._model = model
        self._filtering_cfg: Dict[str, Any] = {}
        self._x_measurements: List[_DirectionalMeasurement] = []
        self._y_measurements: List[_DirectionalMeasurement] = []
        self._sel_x_idx: int = 0
        self._sel_y_idx: int = 0
        self._last_x: Optional[_DirectionalFrfResult] = None
        self._last_y: Optional[_DirectionalFrfResult] = None
        self._last_proc_x: Optional[_DirectionalProcessing] = None
        self._last_proc_y: Optional[_DirectionalProcessing] = None
        self._last_raw_x: Optional[Dict[int, np.ndarray]] = None
        self._last_raw_y: Optional[Dict[int, np.ndarray]] = None
        self._last_run_meta: Dict[str, Any] = {}

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
        self._tab_selectors: Dict[str, Tuple[QComboBox, QComboBox]] = {}
        self._tab_sel_idx: Dict[str, Tuple[int, int]] = {}
        self._tab_average_checks: Dict[str, QCheckBox] = {}
        self._tab_bode_disp = MatplotlibFigureWidget()
        self._tab_bode_accel = MatplotlibFigureWidget()
        self._tab_time_steps = MatplotlibFigureWidget()
        self._time_steps_mode = "force_x"
        self._time_steps_mode_combo = QComboBox()
        self._time_steps_mode_combo.addItem("Force (X)", userData="force_x")
        self._time_steps_mode_combo.addItem("Response (X)", userData="resp_x")
        self._time_steps_mode_combo.addItem("Force (Y)", userData="force_y")
        self._time_steps_mode_combo.addItem("Response (Y)", userData="resp_y")
        self._time_steps_mode_combo.currentIndexChanged.connect(self._on_time_steps_mode_changed)

        time_steps_container = self._wrap_tab_with_selectors(
            tab_id="time_steps",
            plot_widget=self._tab_time_steps,
            extra_controls=[self._time_steps_mode_combo],
        )

        self._tab_fft_mag = MatplotlibFigureWidget()
        self._tab_coherence = MatplotlibFigureWidget()
        self._tab_force_psd = MatplotlibFigureWidget()
        self._tab_real_frf = MatplotlibFigureWidget()
        self._tab_raw_mosaic = MatplotlibFigureWidget()

        self._tabs.addTab(self._wrap_tab_with_selectors(tab_id="raw_mosaic", plot_widget=self._tab_raw_mosaic), "Raw signals (mosaic)")
        self._tabs.addTab(time_steps_container, "Time signals (processing steps)")
        self._tabs.addTab(self._wrap_tab_with_selectors(tab_id="fft_mag", plot_widget=self._tab_fft_mag, add_average=True), "FFT magnitude")
        self._tabs.addTab(self._wrap_tab_with_selectors(tab_id="coherence", plot_widget=self._tab_coherence, add_average=True), "Coherence")
        self._tabs.addTab(self._wrap_tab_with_selectors(tab_id="force_psd", plot_widget=self._tab_force_psd, add_average=True), "Force PSD")
        self._tabs.addTab(self._wrap_tab_with_selectors(tab_id="real_frf", plot_widget=self._tab_real_frf, add_average=True), "Real(FRF)")
        self._tabs.addTab(self._wrap_tab_with_selectors(tab_id="bode_disp", plot_widget=self._tab_bode_disp, add_average=True), "Bode (displacement/force)")
        self._tabs.addTab(self._wrap_tab_with_selectors(tab_id="bode_accel", plot_widget=self._tab_bode_accel, add_average=True), "Bode (acceleration/force)")
        layout.addWidget(self._tabs, 1)

    def _wrap_tab_with_selectors(
        self,
        *,
        tab_id: str,
        plot_widget: MatplotlibFigureWidget,
        extra_controls: Optional[List[QWidget]] = None,
        add_average: bool = False,
    ) -> QWidget:
        """
        Create a standard tab container with a top selector row:
          X measurement dropdown + Y measurement dropdown (+ optional extra controls),
        then the Matplotlib plot widget below.
        """
        extra_controls = list(extra_controls or [])

        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)

        row = QWidget()
        row_l = QHBoxLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setAlignment(Qt.AlignmentFlag.AlignRight)

        x_cb = QComboBox()
        y_cb = QComboBox()
        x_cb.setMinimumWidth(250)
        y_cb.setMinimumWidth(250)
        x_cb.currentIndexChanged.connect(lambda _idx, tid=tab_id: self._on_measurement_changed(tid, "x"))
        y_cb.currentIndexChanged.connect(lambda _idx, tid=tab_id: self._on_measurement_changed(tid, "y"))

        self._tab_selectors[tab_id] = (x_cb, y_cb)
        self._tab_sel_idx[tab_id] = (0, 0)

        row_l.addWidget(QLabel("X"))
        row_l.addWidget(x_cb)
        row_l.addSpacing(10)
        row_l.addWidget(QLabel("Y"))
        row_l.addWidget(y_cb)

        if add_average:
            avg = QCheckBox("Average")
            avg.stateChanged.connect(lambda _st, tid=tab_id: self._redraw_tab(tid))
            self._tab_average_checks[tab_id] = avg
            row_l.addSpacing(10)
            row_l.addWidget(avg)

        for w in extra_controls:
            row_l.addSpacing(10)
            row_l.addWidget(w)

        lay.addWidget(row)
        lay.addWidget(plot_widget, 1)
        return container

    @staticmethod
    def _disambiguate_labels(paths: List[str]) -> List[str]:
        """
        Turn file paths into display labels. If multiple share the same basename, append parent folder.
        """
        names = [Path(p).name if p else "" for p in paths]
        counts: Dict[str, int] = {}
        for nm in names:
            counts[nm] = counts.get(nm, 0) + 1
        out: List[str] = []
        for p, nm in zip(paths, names):
            if not p:
                out.append("(missing)")
                continue
            if counts.get(nm, 0) <= 1:
                out.append(nm)
            else:
                par = Path(p).parent.name
                out.append(f"{nm}  ({par})")
        return out

    def _populate_measurement_dropdowns(self) -> None:
        x_paths = [str(m.filepath or "") for m in (self._x_measurements or [])]
        y_paths = [str(m.filepath or "") for m in (self._y_measurements or [])]
        x_labels = self._disambiguate_labels(x_paths)
        y_labels = self._disambiguate_labels(y_paths)

        for tab_id, (x_cb, y_cb) in (self._tab_selectors or {}).items():
            x_cb.blockSignals(True)
            y_cb.blockSignals(True)
            try:
                x_cb.clear()
                y_cb.clear()
                for i, lab in enumerate(x_labels):
                    x_cb.addItem(lab, userData=i)
                for i, lab in enumerate(y_labels):
                    y_cb.addItem(lab, userData=i)

                x_cb.setEnabled(len(x_labels) > 0)
                y_cb.setEnabled(len(y_labels) > 0)

                # Default selection per tab: first item if available.
                if len(x_labels) > 0:
                    x_cb.setCurrentIndex(0)
                if len(y_labels) > 0:
                    y_cb.setCurrentIndex(0)

                self._tab_sel_idx[tab_id] = (0, 0)
            finally:
                x_cb.blockSignals(False)
                y_cb.blockSignals(False)

    def _get_selected_indices(self, tab_id: str) -> Tuple[int, int]:
        return self._tab_sel_idx.get(tab_id, (0, 0))

    def _get_selected_measurements(self, tab_id: str) -> Tuple[Optional[_DirectionalMeasurement], Optional[_DirectionalMeasurement]]:
        x_idx, y_idx = self._get_selected_indices(tab_id)
        x_sel = self._x_measurements[x_idx] if 0 <= x_idx < len(self._x_measurements) else None
        y_sel = self._y_measurements[y_idx] if 0 <= y_idx < len(self._y_measurements) else None
        return x_sel, y_sel

    def _on_measurement_changed(self, tab_id: str, which: str) -> None:
        x_cb, y_cb = self._tab_selectors.get(tab_id, (None, None))  # type: ignore[assignment]
        if x_cb is None or y_cb is None:
            return
        x_idx = int(x_cb.currentData() or 0) if which in {"x", "both"} else self._get_selected_indices(tab_id)[0]
        y_idx = int(y_cb.currentData() or 0) if which in {"y", "both"} else self._get_selected_indices(tab_id)[1]
        self._tab_sel_idx[tab_id] = (x_idx, y_idx)
        self._redraw_tab(tab_id)

    def _redraw_tab(self, tab_id: str) -> None:
        x_sel, y_sel = self._get_selected_measurements(tab_id)
        avg_on = bool(self._tab_average_checks.get(tab_id).isChecked()) if tab_id in self._tab_average_checks else False

        if tab_id == "raw_mosaic":
            self._plot_raw_mosaic(
                self._tab_raw_mosaic,
                x_channels=(x_sel.raw_channels if x_sel else None),
                y_channels=(y_sel.raw_channels if y_sel else None),
            )
            return

        if tab_id == "time_steps":
            self._refresh_time_steps_controls()
            self._plot_time_processing_steps(
                self._tab_time_steps,
                x_proc=(x_sel.proc if x_sel else None),
                y_proc=(y_sel.proc if y_sel else None),
                mode=self._time_steps_mode,
            )
            return

        if tab_id == "fft_mag":
            self._plot_fft_magnitude(
                self._tab_fft_mag,
                x_proc=(x_sel.proc if x_sel else None),
                y_proc=(y_sel.proc if y_sel else None),
                avg_on=avg_on,
            )
            return
        if tab_id == "coherence":
            self._plot_coherence(
                self._tab_coherence,
                x_proc=(x_sel.proc if x_sel else None),
                y_proc=(y_sel.proc if y_sel else None),
                avg_on=avg_on,
            )
            return
        if tab_id == "force_psd":
            self._plot_force_psd(
                self._tab_force_psd,
                x_proc=(x_sel.proc if x_sel else None),
                y_proc=(y_sel.proc if y_sel else None),
                avg_on=avg_on,
            )
            return
        if tab_id == "real_frf":
            x_res = ((x_sel.compliance, x_sel.accelerance) if x_sel else None)
            y_res = ((y_sel.compliance, y_sel.accelerance) if y_sel else None)
            self._plot_real_frf(
                self._tab_real_frf,
                x_res=x_res,
                y_res=y_res,
                avg_on=avg_on,
            )
            return
        if tab_id == "bode_disp":
            self._plot_bode(
                self._tab_bode_disp,
                (x_sel.compliance if x_sel else None),
                (y_sel.compliance if y_sel else None),
                y_label="Compliance (m/N)",
                title="Displacement/Force FRF",
                avg_on=avg_on,
                x_all=[m.compliance for m in (self._x_measurements or [])],
                y_all=[m.compliance for m in (self._y_measurements or [])],
            )
            return
        if tab_id == "bode_accel":
            self._plot_bode(
                self._tab_bode_accel,
                (x_sel.accelerance if x_sel else None),
                (y_sel.accelerance if y_sel else None),
                y_label="Accelerance ((m/s²)/N)",
                title="Acceleration/Force FRF",
                avg_on=avg_on,
                x_all=[m.accelerance for m in (self._x_measurements or [])],
                y_all=[m.accelerance for m in (self._y_measurements or [])],
            )
            return

    def set_filtering_cfg(self, cfg: Dict[str, Any]) -> None:
        self._filtering_cfg = dict(cfg or {})

    def _build_experiment_cfg(self) -> Dict[str, Any]:
        # Start from defaults and override from model.
        repo_root = Path(__file__).resolve().parents[3]
        cfg_path = repo_root / "config" / "config_before_cabinet_assembly.json"
        base = json_load(cfg_path) if cfg_path.exists() else {}
        base["analysis_type"] = "frf"
        base["data_source"] = self._model.data_source
        if self._model.channel_mapping:
            base["channel_mapping"] = dict(self._model.channel_mapping)
        if self._model.sensitivity_mapping:
            base["sensitivity_mapping"] = dict(self._model.sensitivity_mapping)
        if self._model.data_source == "simulation":
            base["simulation"] = self._model.simulation or {}
        else:
            # directional file paths
            base["file_paths_x"] = list(self._model.file_paths_x or [])
            base["file_paths_y"] = list(self._model.file_paths_y or [])
        return base

    def _run(self) -> None:
        try:
            exp_cfg = self._build_experiment_cfg()
            repo_root = Path(__file__).resolve().parents[3]
            sens = load_data.load_sensitivities(repo_root / "config" / "sensitivities.json")
            directional = load_data.load_directional_data(exp_cfg, sens)

            self._x_measurements = self._compute_measurement_list(directional.get("x"), direction="x")
            self._y_measurements = self._compute_measurement_list(directional.get("y"), direction="y")

            # Default selection: first measurement per direction (if any).
            self._sel_x_idx = 0
            self._sel_y_idx = 0

            x_sel = self._x_measurements[self._sel_x_idx] if self._x_measurements else None
            y_sel = self._y_measurements[self._sel_y_idx] if self._y_measurements else None

            x_res = ((x_sel.compliance, x_sel.accelerance) if x_sel else None)
            y_res = ((y_sel.compliance, y_sel.accelerance) if y_sel else None)
            x_proc = (x_sel.proc if x_sel else None)
            y_proc = (y_sel.proc if y_sel else None)

            self._last_x = (
                _DirectionalFrfResult(direction="x", compliance=x_res[0], accelerance=x_res[1]) if x_res else None
            )
            self._last_y = (
                _DirectionalFrfResult(direction="y", compliance=y_res[0], accelerance=y_res[1]) if y_res else None
            )
            self._last_proc_x = x_proc
            self._last_proc_y = y_proc

            self._last_raw_x = dict(x_sel.raw_channels or {}) if x_sel and x_sel.raw_channels is not None else None
            self._last_raw_y = dict(y_sel.raw_channels or {}) if y_sel and y_sel.raw_channels is not None else None

            self._last_run_meta = {
                "analysis_type": "frf",
                "data_source": self._model.data_source,
                "file_paths_x": list(self._model.file_paths_x or []),
                "file_paths_y": list(self._model.file_paths_y or []),
                "simulation": dict(self._model.simulation or {}),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

            self._populate_measurement_dropdowns()
            for tid in ["raw_mosaic", "time_steps", "fft_mag", "coherence", "force_psd", "real_frf", "bode_disp", "bode_accel"]:
                self._redraw_tab(tid)
        except Exception as e:
            QMessageBox.critical(self, "FRF run failed", str(e))

    def _compute_measurement_list(self, items: object, *, direction: str) -> List[_DirectionalMeasurement]:
        if not isinstance(items, list):
            return []
        out: List[_DirectionalMeasurement] = []
        for d in items:
            if not isinstance(d, dict):
                continue
            res, proc = self._compute_directional(d, direction=direction)
            if res is None or proc is None:
                continue
            raw_channels = None
            try:
                raw_channels = dict(d.get("channels", {}) or {})
            except Exception:
                raw_channels = None
            out.append(
                _DirectionalMeasurement(
                    direction=direction,
                    filepath=d.get("filepath"),
                    raw_channels=raw_channels,
                    proc=proc,
                    compliance=res[0],
                    accelerance=res[1],
                )
            )
        return out

    def _refresh_time_steps_controls(self) -> None:
        """
        Enable only the dropdown modes that have data available.
        """
        has_x = bool(self._x_measurements)
        has_y = bool(self._y_measurements)

        def set_enabled(user_data: str, enabled: bool) -> None:
            for i in range(self._time_steps_mode_combo.count()):
                if self._time_steps_mode_combo.itemData(i) == user_data:
                    self._time_steps_mode_combo.model().item(i).setEnabled(enabled)  # type: ignore[union-attr]
                    return

        set_enabled("force_x", has_x)
        set_enabled("resp_x", has_x)
        set_enabled("force_y", has_y)
        set_enabled("resp_y", has_y)

        # Pick a sensible default if current selection is disabled
        valid_order = ["force_x", "resp_x", "force_y", "resp_y"]
        if (self._time_steps_mode in ["force_x", "resp_x"] and not has_x) or (
            self._time_steps_mode in ["force_y", "resp_y"] and not has_y
        ):
            for cand in valid_order:
                if (cand in ["force_x", "resp_x"] and has_x) or (cand in ["force_y", "resp_y"] and has_y):
                    self._time_steps_mode = cand
                    break

        # Sync combo selection to mode
        for i in range(self._time_steps_mode_combo.count()):
            if self._time_steps_mode_combo.itemData(i) == self._time_steps_mode:
                self._time_steps_mode_combo.blockSignals(True)
                try:
                    self._time_steps_mode_combo.setCurrentIndex(i)
                finally:
                    self._time_steps_mode_combo.blockSignals(False)
                break

    def _on_time_steps_mode_changed(self) -> None:
        mode = self._time_steps_mode_combo.currentData()
        if not isinstance(mode, str) or not mode:
            return
        self._time_steps_mode = mode
        self._redraw_tab("time_steps")

    def export_results(self, analysis_folder: Path) -> dict:
        """
        Export the most recently computed FRF results into:
          results/figures/*.png
          results/tables/*.csv
          results/logs/export_log.json
        """
        analysis_folder = Path(analysis_folder).resolve()
        fig_dir = analysis_folder / "results" / "figures"
        tab_dir = analysis_folder / "results" / "tables"
        log_dir = analysis_folder / "results" / "logs"
        fig_dir.mkdir(parents=True, exist_ok=True)
        tab_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        exported: Dict[str, Any] = {"figures": [], "tables": [], "log": None, "warnings": []}

        def _write_csv(path: Path, f_hz: np.ndarray, H: np.ndarray) -> None:
            f_hz = np.asarray(f_hz, dtype=float)
            H = np.asarray(H, dtype=complex)
            mag = np.abs(H)
            phase_deg = np.unwrap(np.angle(H)) * 180.0 / np.pi
            with path.open("w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["frequency_hz", "real", "imag", "magnitude", "phase_deg"])
                for i in range(len(f_hz)):
                    w.writerow(
                        [
                            float(f_hz[i]),
                            float(np.real(H[i])),
                            float(np.imag(H[i])),
                            float(mag[i]),
                            float(phase_deg[i]),
                        ]
                    )

        # Always export a log (even if no results were generated yet).
        now = datetime.now(timezone.utc).isoformat()
        log_payload: Dict[str, Any] = {
            "timestamp_utc": now,
            "meta": dict(self._last_run_meta or {}),
            "exported": {"figures": [], "tables": []},
            "warnings": [],
        }

        # Figures (export current tabs, even if no data traces)
        try:
            p1 = fig_dir / "bode_displacement_force.png"
            self._tab_bode_disp.figure.savefig(p1, dpi=200, bbox_inches="tight")
            exported["figures"].append(str(p1))
            log_payload["exported"]["figures"].append(str(p1))
        except Exception as e:
            msg = f"Failed to export bode_displacement_force.png: {e}"
            exported["warnings"].append(msg)
            log_payload["warnings"].append(msg)

        try:
            p2 = fig_dir / "bode_acceleration_force.png"
            self._tab_bode_accel.figure.savefig(p2, dpi=200, bbox_inches="tight")
            exported["figures"].append(str(p2))
            log_payload["exported"]["figures"].append(str(p2))
        except Exception as e:
            msg = f"Failed to export bode_acceleration_force.png: {e}"
            exported["warnings"].append(msg)
            log_payload["warnings"].append(msg)

        # Tables
        def _export_dir(res: _DirectionalFrfResult) -> None:
            d = res.direction.lower()
            p_comp = tab_dir / f"frf_{d}_compliance.csv"
            p_acc = tab_dir / f"frf_{d}_accelerance.csv"
            _write_csv(p_comp, res.compliance[0], res.compliance[1])
            _write_csv(p_acc, res.accelerance[0], res.accelerance[1])
            exported["tables"].extend([str(p_comp), str(p_acc)])
            log_payload["exported"]["tables"].extend([str(p_comp), str(p_acc)])

        if self._last_x is not None:
            try:
                _export_dir(self._last_x)
            except Exception as e:
                msg = f"Failed to export X tables: {e}"
                exported["warnings"].append(msg)
                log_payload["warnings"].append(msg)
        else:
            msg = "No X-direction FRF results available to export (run Generate results first)."
            exported["warnings"].append(msg)
            log_payload["warnings"].append(msg)

        if self._last_y is not None:
            try:
                _export_dir(self._last_y)
            except Exception as e:
                msg = f"Failed to export Y tables: {e}"
                exported["warnings"].append(msg)
                log_payload["warnings"].append(msg)
        else:
            msg = "No Y-direction FRF results available to export (run Generate results first)."
            exported["warnings"].append(msg)
            log_payload["warnings"].append(msg)

        # Log file
        log_path = log_dir / "export_log.json"
        log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
        exported["log"] = str(log_path)

        return exported

    def _compute_directional(
        self,
        data: Optional[Dict[str, Any]],
        direction: str,
    ) -> Tuple[
        Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]],
        Optional[_DirectionalProcessing],
    ]:
        if data is None:
            return None, None

        def _key_for_family(family: str, direction: str) -> str:
            family = (family or "").strip().lower()
            if family == "force":
                return "F_x" if direction == "x" else "F_y"
            if family == "tool":
                return "x_tool" if direction == "x" else "y_tool"
            if family == "spindle":
                return "x_spindle" if direction == "x" else "y_spindle"
            raise ValueError(f"Unknown IO family '{family}' (expected force/tool/spindle)")

        io_sel = (self._filtering_cfg or {}).get("io_selection")
        if isinstance(io_sel, dict):
            in_family = str(io_sel.get("input_family") or "force")
            out_family = str(io_sel.get("output_family") or "tool")
            u_name = _key_for_family(in_family, direction)
            y_name = _key_for_family(out_family, direction)
            if u_name not in data:
                raise ValueError(
                    f"Missing required input signal for {direction}: '{u_name}' (from input_family={in_family})"
                )
            if y_name not in data or data.get(y_name) is None:
                raise ValueError(
                    f"Missing required output signal for {direction}: '{y_name}' (from output_family={out_family})"
                )
        else:
            # Backward-compat: previous heuristic (force -> tool, else spindle fallback).
            u_name = "F_x" if direction == "x" else "F_y"
            y_name = "x_tool" if direction == "x" else "y_tool"

            if u_name not in data or y_name not in data:
                # Try spindle signals as fallback
                y_name = "x_spindle" if direction == "x" else "y_spindle"
            if u_name not in data or y_name not in data or data[y_name] is None:
                raise ValueError(f"Missing required signals for {direction}: need {u_name} and {y_name}")

        time = np.asarray(data["time"])
        u = np.asarray(data[u_name], dtype=float)
        y_accel = np.asarray(data[y_name], dtype=float)  # treated as acceleration

        params = frf_estimation.get_system_parameters()
        filtering = dict(self._filtering_cfg or {})
        if "bias_window_size" in filtering:
            params["bias_window_size"] = int(filtering["bias_window_size"])
        fp = filtering.get("params") if isinstance(filtering.get("params"), dict) else {}
        wn = fp.get("wiener") if isinstance(fp, dict) else None
        if isinstance(wn, dict) and "mysize" in wn:
            params["wiener_mysize"] = int(wn["mysize"])
        params["estimator_type"] = str(filtering.get("estimator_type") or params.get("estimator_type") or "Hv")

        u_proc, y_proc, params = apply_frf_filtering_and_windowing(
            time=time,
            u=u,
            resp=y_accel,
            params=params,
            filtering_cfg=filtering,
        )

        f, H_accel = frf_estimation.compute_frf_from_signals(time, u_proc, y_proc, params)
        H_disp = accel_to_disp_over_force(f, H_accel)

        proc = self._compute_processing_steps(
            time=time, u=u, resp=y_accel, direction=direction, params=params, filtering_cfg=filtering
        )
        return ((f, H_disp), (f, H_accel)), proc

    @staticmethod
    def _enable_legend_toggles(fig) -> None:
        """
        Click legend entries to show/hide corresponding lines (per-figure).
        """
        try:
            canvas = fig.canvas
        except Exception:
            return

        # Keep a mapping on the figure so the callback can find it after redraws.
        fig._legend_pick_map = {}  # type: ignore[attr-defined]
        fig._legend_pick_map_text = {}  # type: ignore[attr-defined]
        fig._legend_pick_map_patch = {}  # type: ignore[attr-defined]

        def _rebuild_map():
            m = {}
            m_text = {}
            m_patch = {}
            for ax in fig.axes:
                leg = ax.get_legend()
                if leg is None:
                    continue
                # Map legend entries to plotted lines by label (robust vs. line ordering differences).
                lines_by_label = {ln.get_label(): ln for ln in ax.get_lines()}
                leg_lines = list(leg.get_lines() or [])
                leg_texts = list(leg.get_texts() or [])
                leg_patches = list(
                    getattr(leg, "legend_handles", []) or getattr(leg, "legendHandles", []) or []
                )
                for leg_line, leg_text in zip(leg_lines, leg_texts):
                    label = leg_text.get_text()
                    orig_line = lines_by_label.get(label)
                    if orig_line is None:
                        continue
                    leg_line.set_picker(True)
                    leg_line.set_pickradius(5)
                    leg_line.set_alpha(1.0 if orig_line.get_visible() else 0.2)
                    m[leg_line] = orig_line
                    # Also allow clicking the legend text (common user behavior).
                    try:
                        leg_text.set_picker(True)
                        m_text[leg_text] = orig_line
                    except Exception:
                        pass

                # Some Matplotlib configurations use patch handles in legends.
                for h in leg_patches:
                    try:
                        lbl = h.get_label()
                    except Exception:
                        continue
                    orig_line = lines_by_label.get(lbl)
                    if orig_line is None:
                        continue
                    try:
                        h.set_picker(True)
                        m_patch[h] = orig_line
                    except Exception:
                        pass
            fig._legend_pick_map = m  # type: ignore[attr-defined]
            fig._legend_pick_map_text = m_text  # type: ignore[attr-defined]
            fig._legend_pick_map_patch = m_patch  # type: ignore[attr-defined]

        def _on_pick(event):
            m = getattr(fig, "_legend_pick_map", {})
            m_text = getattr(fig, "_legend_pick_map_text", {})
            m_patch = getattr(fig, "_legend_pick_map_patch", {})
            artist = getattr(event, "artist", None)

            if artist in m:
                line = m[artist]
                toggle_artist = artist
            elif artist in m_text:
                line = m_text[artist]
                toggle_artist = None  # text alpha handling differs; skip
            elif artist in m_patch:
                line = m_patch[artist]
                toggle_artist = artist
            else:
                return
            before = bool(line.get_visible())
            vis = not before
            line.set_visible(vis)
            if toggle_artist is not None:
                toggle_artist.set_alpha(1.0 if vis else 0.2)
            canvas.draw_idle()

        _rebuild_map()
        # Connect once per plot call; disconnect any previous handler on this figure.
        try:
            old_cid = getattr(fig, "_legend_pick_cid", None)
            if isinstance(old_cid, int):
                canvas.mpl_disconnect(old_cid)
        except Exception:
            pass
        cid = canvas.mpl_connect("pick_event", _on_pick)
        try:
            fig._legend_pick_cid = cid  # type: ignore[attr-defined]
        except Exception:
            pass

    def _plot_raw_mosaic(
        self,
        widget: MatplotlibFigureWidget,
        *,
        x_channels: Optional[Dict[int, np.ndarray]],
        y_channels: Optional[Dict[int, np.ndarray]],
    ) -> None:
        fig = widget.figure
        fig.clear()

        def has_any(ch: Optional[Dict[int, np.ndarray]]) -> bool:
            return bool(ch) and any(isinstance(v, np.ndarray) and len(v) for v in ch.values())

        has_x = has_any(x_channels)
        has_y = has_any(y_channels)
        if not has_x and not has_y:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(
                0.5,
                0.5,
                "No raw channels available (simulation mode or missing channels).",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            widget.redraw()
            return

        n_rows = 8
        n_cols = 2 if (has_x and has_y) else 1

        axes = []
        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c + 1
                axes.append(fig.add_subplot(n_rows, n_cols, idx, sharex=axes[0] if axes else None))

        def plot_col(ch: Dict[int, np.ndarray], col: int, title_prefix: str) -> None:
            for i in range(8):
                ax = axes[i * n_cols + col]
                y = np.asarray(ch.get(i, np.zeros(0)), dtype=float)
                if len(y) == 0:
                    ax.text(0.5, 0.5, f"A{i}: missing", ha="center", va="center", fontsize=8)
                    ax.set_axis_off()
                    continue
                ax.plot(y, linewidth=0.9)
                ax.grid(True, alpha=0.25)
                ax.set_ylabel(f"A{i}", rotation=0, labelpad=14, fontsize=8)
                if i == 0:
                    ax.set_title(f"{title_prefix} raw channels (index plot)", fontsize=10)
                if i != 7:
                    ax.tick_params(labelbottom=False)

        if has_x and x_channels is not None:
            plot_col(x_channels, 0, "X")
        if has_y and y_channels is not None:
            plot_col(y_channels, 1 if (has_x and has_y) else 0, "Y")

        widget.redraw()

    def _plot_time_processing_steps(
        self,
        widget: MatplotlibFigureWidget,
        *,
        x_proc: Optional[_DirectionalProcessing],
        y_proc: Optional[_DirectionalProcessing],
        mode: str,
    ) -> None:
        fig = widget.figure
        fig.clear()

        has_x = x_proc is not None
        has_y = y_proc is not None
        if not has_x and not has_y:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No processing data available.", ha="center", va="center")
            ax.set_axis_off()
            widget.redraw()
            return

        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Time (s)")

        def add_steps(ax, t: np.ndarray, steps: List[_SignalStep], prefix: str) -> None:
            default_visible = {"raw", "bias_corrected", "window", "windowed", "wiener", "savgol"}
            for s in steps:
                line = ax.plot(t, s.y, linewidth=1.0, label=f"{prefix}{s.name}")[0]
                line.set_visible(s.name in default_visible)

        title_map = {
            "force_x": "Force (X) processing steps",
            "resp_x": "Response (X) processing steps",
            "force_y": "Force (Y) processing steps",
            "resp_y": "Response (Y) processing steps",
        }

        if mode == "force_x" and x_proc is not None:
            add_steps(ax, x_proc.time, x_proc.u_steps, prefix="Fx:")
            ax.set_ylabel("Force (N)")
        elif mode == "resp_x" and x_proc is not None:
            add_steps(ax, x_proc.time, x_proc.resp_steps, prefix="x:")
            ax.set_ylabel("Response (m/s²)")
        elif mode == "force_y" and y_proc is not None:
            add_steps(ax, y_proc.time, y_proc.u_steps, prefix="Fy:")
            ax.set_ylabel("Force (N)")
        elif mode == "resp_y" and y_proc is not None:
            add_steps(ax, y_proc.time, y_proc.resp_steps, prefix="y:")
            ax.set_ylabel("Response (m/s²)")
        else:
            if x_proc is not None:
                add_steps(ax, x_proc.time, x_proc.u_steps, prefix="Fx:")
                ax.set_ylabel("Force (N)")
                mode = "force_x"
            elif y_proc is not None:
                add_steps(ax, y_proc.time, y_proc.u_steps, prefix="Fy:")
                ax.set_ylabel("Force (N)")
                mode = "force_y"

        ax.set_title(f"{title_map.get(mode, 'Processing steps')} (click legend to toggle)")
        ax.legend(loc="upper right", fontsize=8)

        self._enable_legend_toggles(fig)
        widget.redraw()

    def _plot_fft_magnitude(
        self,
        widget: MatplotlibFigureWidget,
        *,
        x_proc: Optional[_DirectionalProcessing],
        y_proc: Optional[_DirectionalProcessing],
        avg_on: bool = False,
    ) -> None:
        fig = widget.figure
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("FFT magnitude (click legend to toggle)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("|FFT| (arb.)")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_yscale("log")
        ax.set_xscale("log")

        def add(proc: _DirectionalProcessing, prefix: str) -> None:
            f = proc.fft_f_hz
            for name, mag in proc.fft_u_mag.items():
                ax.plot(f[1:], mag[1:], linewidth=1.0, label=f"{prefix}u:{name}")
            for name, mag in proc.fft_resp_mag.items():
                ax.plot(f[1:], mag[1:], linewidth=1.0, label=f"{prefix}resp:{name}")

        if x_proc is not None:
            add(x_proc, "X:")
        if y_proc is not None:
            add(y_proc, "Y:")

        if avg_on:
            # Average across all loaded measurements (complex FFT bins first, then magnitude).
            def add_avg(direction: str, procs: List[_DirectionalProcessing], prefix: str) -> None:
                if not procs:
                    return
                # Collect common step names present in all procs.
                names_u = set(procs[0].fft_u_complex.keys())
                names_r = set(procs[0].fft_resp_complex.keys())
                for p in procs[1:]:
                    names_u &= set(p.fft_u_complex.keys())
                    names_r &= set(p.fft_resp_complex.keys())
                f0 = procs[0].fft_f_hz
                for nm in sorted(names_u):
                    stack = np.stack([p.fft_u_complex[nm] for p in procs], axis=0)
                    avg = np.mean(stack, axis=0)
                    ax.plot(f0[1:], np.abs(avg)[1:], color="black", linewidth=2.2, alpha=0.9, label=f"{prefix}avg u:{nm}")
                for nm in sorted(names_r):
                    stack = np.stack([p.fft_resp_complex[nm] for p in procs], axis=0)
                    avg = np.mean(stack, axis=0)
                    ax.plot(f0[1:], np.abs(avg)[1:], color="black", linewidth=2.2, alpha=0.9, label=f"{prefix}avg resp:{nm}")

            add_avg("x", [m.proc for m in (getattr(self, "_x_measurements", []) or [])], "X:")
            add_avg("y", [m.proc for m in (getattr(self, "_y_measurements", []) or [])], "Y:")

        ax.legend(loc="best", fontsize=8)
        self._enable_legend_toggles(fig)
        widget.redraw()

    def _plot_coherence(
        self,
        widget: MatplotlibFigureWidget,
        *,
        x_proc: Optional[_DirectionalProcessing],
        y_proc: Optional[_DirectionalProcessing],
        avg_on: bool = False,
    ) -> None:
        fig = widget.figure
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Coherence γ²(f)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("γ²")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xscale("log")
        ax.axhline(0.7, linestyle="--", linewidth=1, alpha=0.6, color="red", label="threshold 0.7")

        if x_proc is not None:
            ax.plot(x_proc.f_hz[1:], x_proc.coherence_uy[1:], linewidth=1.6, label="X")
        if y_proc is not None:
            ax.plot(y_proc.f_hz[1:], y_proc.coherence_uy[1:], linewidth=1.6, label="Y")

        if avg_on:
            def add_avg(procs: List[_DirectionalProcessing], label: str) -> None:
                if not procs:
                    return
                f0 = procs[0].f_hz
                # Average spectra terms on the common grid.
                Puu = np.mean(np.stack([p.psd_u for p in procs], axis=0), axis=0)
                Prr = np.mean(np.stack([p.psd_r for p in procs], axis=0), axis=0)
                Pur = np.mean(np.stack([p.csd_ur for p in procs], axis=0), axis=0)
                coh = (np.abs(Pur) ** 2) / ((Puu * Prr) + 1e-30)
                coh = np.clip(coh, 0.0, 1.0)
                ax.plot(f0[1:], coh[1:], color="black", linewidth=2.4, alpha=0.9, label=f"{label} avg")

            add_avg([m.proc for m in (getattr(self, "_x_measurements", []) or [])], "X")
            add_avg([m.proc for m in (getattr(self, "_y_measurements", []) or [])], "Y")

        ax.legend(loc="best")
        widget.redraw()

    def _plot_force_psd(
        self,
        widget: MatplotlibFigureWidget,
        *,
        x_proc: Optional[_DirectionalProcessing],
        y_proc: Optional[_DirectionalProcessing],
        avg_on: bool = False,
    ) -> None:
        fig = widget.figure
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Force PSD (Welch)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (N²/Hz)")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xscale("log")
        ax.set_yscale("log")

        if x_proc is not None:
            ax.plot(x_proc.f_hz[1:], x_proc.psd_u[1:], linewidth=1.6, label="X")
        if y_proc is not None:
            ax.plot(y_proc.f_hz[1:], y_proc.psd_u[1:], linewidth=1.6, label="Y")

        if avg_on:
            def add_avg(procs: List[_DirectionalProcessing], label: str) -> None:
                if not procs:
                    return
                f0 = procs[0].f_hz
                Puu = np.mean(np.stack([p.psd_u for p in procs], axis=0), axis=0)
                ax.plot(f0[1:], Puu[1:], color="black", linewidth=2.4, alpha=0.9, label=f"{label} avg")

            add_avg([m.proc for m in (getattr(self, "_x_measurements", []) or [])], "X")
            add_avg([m.proc for m in (getattr(self, "_y_measurements", []) or [])], "Y")

        ax.legend(loc="best")
        widget.redraw()

    def _plot_real_frf(
        self,
        widget: MatplotlibFigureWidget,
        *,
        x_res: Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]],
        y_res: Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]],
        avg_on: bool = False,
    ) -> None:
        fig = widget.figure
        fig.clear()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        ax1.set_title("Real(FRF)")
        ax1.set_ylabel("Re{Compliance} (m/N)")
        ax2.set_ylabel("Re{Accelerance} ((m/s²)/N)")
        ax2.set_xlabel("Frequency (Hz)")
        for ax in [ax1, ax2]:
            ax.grid(True, which="both", alpha=0.25)
            ax.set_xscale("log")

        def add(ax, res, label: str):
            f, H = res
            f = np.asarray(f)
            H = np.asarray(H)
            mask = np.isfinite(f) & np.isfinite(np.real(H)) & (f > 0)
            ax.plot(f[mask], np.real(H[mask]), linewidth=1.6, label=label)

        if x_res is not None:
            add(ax1, x_res[0], "X")
            add(ax2, x_res[1], "X")
        if y_res is not None:
            add(ax1, y_res[0], "Y")
            add(ax2, y_res[1], "Y")

        if avg_on:
            def avg_frf(items: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
                if not items:
                    return None
                f0 = np.asarray(items[0][0], dtype=float)
                Hs = [np.asarray(it[1], dtype=complex) for it in items]
                Havg = np.mean(np.stack(Hs, axis=0), axis=0)
                return f0, Havg

            x_comp = avg_frf([m.compliance for m in (getattr(self, "_x_measurements", []) or [])])
            x_acc = avg_frf([m.accelerance for m in (getattr(self, "_x_measurements", []) or [])])
            y_comp = avg_frf([m.compliance for m in (getattr(self, "_y_measurements", []) or [])])
            y_acc = avg_frf([m.accelerance for m in (getattr(self, "_y_measurements", []) or [])])
            if x_comp is not None:
                add(ax1, x_comp, "X avg")
            if x_acc is not None:
                add(ax2, x_acc, "X avg")
            if y_comp is not None:
                add(ax1, y_comp, "Y avg")
            if y_acc is not None:
                add(ax2, y_acc, "Y avg")

        ax1.legend(loc="best")
        ax2.legend(loc="best")
        widget.redraw()

    @staticmethod
    def _plot_bode(
        widget: MatplotlibFigureWidget,
        x: Optional[Tuple[np.ndarray, np.ndarray]],
        y: Optional[Tuple[np.ndarray, np.ndarray]],
        y_label: str,
        title: str,
        avg_on: bool = False,
        x_all: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        y_all: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        fig = widget.figure
        fig.clear()
        ax_mag = fig.add_subplot(2, 1, 1)
        ax_ph = fig.add_subplot(2, 1, 2, sharex=ax_mag)

        def add_trace(res: Tuple[np.ndarray, np.ndarray], label: str) -> None:
            f, H = res
            f = np.asarray(f)
            H = np.asarray(H)
            mask = np.isfinite(f) & np.isfinite(np.real(H)) & np.isfinite(np.imag(H)) & (f > 0)
            f2 = f[mask]
            H2 = H[mask]
            if len(f2) == 0:
                return
            ax_mag.loglog(f2, np.abs(H2), linewidth=1.8, label=label)
            ax_ph.semilogx(f2, np.unwrap(np.angle(H2)) * 180 / np.pi, linewidth=1.2, label=label)

        if x is not None:
            add_trace(x, "X")
        if y is not None:
            add_trace(y, "Y")

        if avg_on:
            def add_avg(items: Optional[List[Tuple[np.ndarray, np.ndarray]]], label: str) -> None:
                if not items:
                    return
                f0 = np.asarray(items[0][0], dtype=float)
                Havg = np.mean(np.stack([np.asarray(it[1], dtype=complex) for it in items], axis=0), axis=0)
                add_trace((f0, Havg), f"{label} avg")

            add_avg(x_all, "X")
            add_avg(y_all, "Y")

        ax_mag.set_title(title)
        ax_mag.set_ylabel(y_label)
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_mag.legend()

        ax_ph.set_xlabel("Frequency (Hz)")
        ax_ph.set_ylabel("Phase (deg)")
        ax_ph.grid(True, which="both", alpha=0.3)
        ax_ph.legend()

        widget.redraw()

    @staticmethod
    def _compute_processing_steps(
        *,
        time: np.ndarray,
        u: np.ndarray,
        resp: np.ndarray,
        direction: str,
        params: Dict[str, Any],
        filtering_cfg: Dict[str, Any] | None = None,
    ) -> _DirectionalProcessing:
        time = np.asarray(time, dtype=float)
        u = np.asarray(u, dtype=float)
        resp = np.asarray(resp, dtype=float)

        if len(time) < 2:
            raise ValueError("Not enough samples to compute processing steps")
        dt = float(time[1] - time[0])
        fs = 1.0 / dt

        filtering_cfg = dict(filtering_cfg or {})
        if "bias_window_size" not in filtering_cfg:
            raise ValueError("filtering configuration must include bias_window_size (samples)")
        bias_n = int(filtering_cfg["bias_window_size"])
        if bias_n <= 0:
            raise ValueError("bias_window_size must be a positive integer")

        u_bias, u_bias_idx, u_bias_actual = signal_processing.remove_bias_window(u, bias_n)
        r_bias, r_bias_idx, r_bias_actual = signal_processing.remove_bias_window(resp, bias_n)

        u_steps: List[_SignalStep] = [_SignalStep("raw", u), _SignalStep("bias_corrected", u_bias)]
        r_steps: List[_SignalStep] = [_SignalStep("raw", resp), _SignalStep("bias_corrected", r_bias)]

        # Apply filter chains (pre-windowing) as configured.
        u_chain = filtering_cfg.get("u_chain") if isinstance(filtering_cfg.get("u_chain"), list) else []
        r_chain = filtering_cfg.get("resp_chain") if isinstance(filtering_cfg.get("resp_chain"), list) else []
        filt_params = filtering_cfg.get("params") if isinstance(filtering_cfg.get("params"), dict) else {}

        def _apply_chain(x_in: np.ndarray, chain: list, is_u: bool) -> Tuple[np.ndarray, List[_SignalStep]]:
            x_cur = np.asarray(x_in, dtype=float)
            steps: List[_SignalStep] = []
            for entry in chain:
                if not isinstance(entry, dict):
                    continue
                nm = entry.get("name")
                if not entry.get("enabled", False):
                    continue
                if nm == "wiener":
                    mysize = _required_wiener_mysize(filt_params)
                    noise_idx = u_bias_idx if is_u else r_bias_idx
                    noise_actual = u_bias_actual if is_u else r_bias_actual
                    noise_pow = signal_processing.calculate_noise_power_from_window(x_cur, noise_idx, noise_actual)
                    x_cur = signal_processing.apply_wiener_filter_legacy(x_cur, noise_pow, mysize)
                    steps.append(_SignalStep("wiener", x_cur))
                elif nm == "savgol":
                    p = filt_params.get("savgol") or {}
                    wl = int((p.get("window_length") or 0) or 0)
                    po = int((p.get("polyorder") or 0) or 0)
                    x_cur = signal_processing.apply_savgol_filter(x_cur, wl, po)
                    steps.append(_SignalStep("savgol", x_cur))
            return x_cur, steps

        u_filtered, u_f_steps = _apply_chain(u_bias, u_chain, is_u=True)
        r_filtered, r_f_steps = _apply_chain(r_bias, r_chain, is_u=False)
        u_steps.extend(u_f_steps)
        r_steps.extend(r_f_steps)

        # Windowing is last
        win_over = filtering_cfg.get("windowing") if isinstance(filtering_cfg.get("windowing"), dict) else {}
        if isinstance(win_over, dict):
            params = dict(params)
            params.update(dict(win_over))

        win_u_type = params.get("input_window_type", "hann")
        win_r_type = params.get("output_window_type", "hann")
        window_u, _meta_u = frf_estimation.create_window(u_filtered, time, params, dt, window_type=win_u_type)
        window_r, _meta_r = frf_estimation.create_window(r_filtered, time, params, dt, window_type=win_r_type)

        def _scale_for_display(x: np.ndarray) -> float:
            try:
                s = float(np.max(np.abs(np.asarray(x, dtype=float))))
            except Exception:
                s = 1.0
            if not np.isfinite(s) or s <= 0:
                return 1.0
            return s

        # Display window at comparable amplitude to the last pre-window (post-filter) signal.
        u_steps.append(_SignalStep("window", window_u * _scale_for_display(u_filtered)))
        r_steps.append(_SignalStep("window", window_r * _scale_for_display(r_filtered)))
        u_win = u_filtered * window_u
        r_win = r_filtered * window_r
        u_steps.append(_SignalStep("windowed", u_win))
        r_steps.append(_SignalStep("windowed", r_win))

        # Welch grid diagnostics (force PSD + coherence between u and response)
        nperseg = int(params.get("welch_nperseg", 2048) or 2048)
        nperseg = max(2, min(nperseg, len(u_bias), len(r_bias)))
        noverlap = int(params.get("welch_noverlap", nperseg // 2) or (nperseg // 2))
        noverlap = max(0, min(noverlap, nperseg - 1))
        welch_window = params.get("welch_window", "hamming")

        f_welch, Puu = signal.welch(u_filtered, fs=fs, nperseg=nperseg, noverlap=noverlap, window=welch_window)
        f2, Prr = signal.welch(r_filtered, fs=fs, nperseg=nperseg, noverlap=noverlap, window=welch_window)
        f3, Pur = signal.csd(u_filtered, r_filtered, fs=fs, nperseg=nperseg, noverlap=noverlap, window=welch_window)
        if not (np.allclose(f_welch, f2) and np.allclose(f_welch, f3)):
            # Align to force grid defensively
            Prr = np.interp(f_welch, f2, Prr)
            Pur = np.interp(f_welch, f3, np.real(Pur)) + 1j * np.interp(f_welch, f3, np.imag(Pur))

        denom = (Puu * Prr) + 1e-30
        coh = np.abs(Pur) ** 2 / denom
        coh = np.clip(coh, 0.0, 1.0)

        # FFT magnitude (separate plot): show for a few key steps
        fft_n = len(time)
        fft_f = np.fft.rfftfreq(fft_n, d=dt)

        def _fft(x: np.ndarray) -> np.ndarray:
            return np.fft.rfft(np.asarray(x, dtype=float), n=fft_n)

        # Always include raw, bias_corrected, windowed, and last filtered if present
        pick_names = ["raw", "bias_corrected", "windowed"]
        last_u = u_steps[-1].name if u_steps else "raw"
        last_r = r_steps[-1].name if r_steps else "raw"
        for nm in [last_u, last_r]:
            if nm not in pick_names:
                pick_names.append(nm)

        fft_u_mag: Dict[str, np.ndarray] = {}
        fft_r_mag: Dict[str, np.ndarray] = {}
        fft_u_complex: Dict[str, np.ndarray] = {}
        fft_r_complex: Dict[str, np.ndarray] = {}
        u_by_name = {s.name: s.y for s in u_steps}
        r_by_name = {s.name: s.y for s in r_steps}
        for nm in pick_names:
            if nm in u_by_name and nm not in fft_u_mag:
                X = _fft(u_by_name[nm])
                fft_u_complex[nm] = X
                fft_u_mag[nm] = np.abs(X)
            if nm in r_by_name and nm not in fft_r_mag:
                Y = _fft(r_by_name[nm])
                fft_r_complex[nm] = Y
                fft_r_mag[nm] = np.abs(Y)

        return _DirectionalProcessing(
            direction=direction,
            time=time,
            fs_hz=float(fs),
            u_steps=u_steps,
            resp_steps=r_steps,
            window=None,  # window already included as a step; keep None to avoid duplication
            f_hz=np.asarray(f_welch, dtype=float),
            psd_u=np.asarray(Puu, dtype=float),
            coherence_uy=np.asarray(coh, dtype=float),
            csd_ur=np.asarray(Pur, dtype=complex),
            psd_r=np.asarray(Prr, dtype=float),
            fft_f_hz=np.asarray(fft_f, dtype=float),
            fft_u_mag=fft_u_mag,
            fft_resp_mag=fft_r_mag,
            fft_u_complex=fft_u_complex,
            fft_resp_complex=fft_r_complex,
        )


def apply_frf_filtering_and_windowing(
    *,
    time: np.ndarray,
    u: np.ndarray,
    resp: np.ndarray,
    params: Dict[str, Any],
    filtering_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply: bias -> (filter chains) -> windowing(last) to both u and resp.

    Returns (u_processed, resp_processed, updated_params) where updated_params includes any
    windowing overrides from filtering_cfg.
    """
    time = np.asarray(time, dtype=float)
    u = np.asarray(u, dtype=float)
    resp = np.asarray(resp, dtype=float)
    params = dict(params or {})
    filtering_cfg = dict(filtering_cfg or {})

    if len(time) < 2:
        return u, resp, params

    dt = float(time[1] - time[0])

    if "bias_window_size" not in filtering_cfg:
        raise ValueError("filtering configuration must include bias_window_size (samples)")
    bias_n = int(filtering_cfg["bias_window_size"])
    if bias_n <= 0:
        raise ValueError("bias_window_size must be a positive integer")
    u_bias, u_bias_idx, u_bias_actual = signal_processing.remove_bias_window(u, bias_n)
    r_bias, r_bias_idx, r_bias_actual = signal_processing.remove_bias_window(resp, bias_n)

    u_chain = filtering_cfg.get("u_chain") if isinstance(filtering_cfg.get("u_chain"), list) else []
    r_chain = filtering_cfg.get("resp_chain") if isinstance(filtering_cfg.get("resp_chain"), list) else []
    filt_params = filtering_cfg.get("params") if isinstance(filtering_cfg.get("params"), dict) else {}

    def _apply_chain(x_in: np.ndarray, chain: list, is_u: bool) -> np.ndarray:
        x_cur = np.asarray(x_in, dtype=float)
        for entry in chain:
            if not isinstance(entry, dict):
                continue
            nm = entry.get("name")
            if not entry.get("enabled", False):
                continue
            if nm == "wiener":
                mysize = _required_wiener_mysize(filt_params)
                noise_idx = u_bias_idx if is_u else r_bias_idx
                noise_actual = u_bias_actual if is_u else r_bias_actual
                noise_pow = signal_processing.calculate_noise_power_from_window(x_cur, noise_idx, noise_actual)
                x_cur = signal_processing.apply_wiener_filter_legacy(x_cur, noise_pow, mysize)
            elif nm == "savgol":
                p = filt_params.get("savgol") or {}
                wl = int((p.get("window_length") or 0) or 0)
                po = int((p.get("polyorder") or 0) or 0)
                x_cur = signal_processing.apply_savgol_filter(x_cur, wl, po)
        return x_cur

    u_filt = _apply_chain(u_bias, u_chain, is_u=True)
    r_filt = _apply_chain(r_bias, r_chain, is_u=False)

    # Windowing overrides from filtering cfg
    win_over = filtering_cfg.get("windowing") if isinstance(filtering_cfg.get("windowing"), dict) else {}
    if isinstance(win_over, dict):
        params.update(dict(win_over))

    win_u_type = params.get("input_window_type", "hann")
    win_r_type = params.get("output_window_type", "hann")
    window_u, _ = frf_estimation.create_window(u_filt, time, params, dt, window_type=win_u_type)
    window_r, _ = frf_estimation.create_window(r_filt, time, params, dt, window_type=win_r_type)
    return u_filt * window_u, r_filt * window_r, params


def json_load(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def accel_to_disp_over_force(f: np.ndarray, H_accel: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    H_accel = np.asarray(H_accel, dtype=complex)
    out = np.zeros_like(H_accel, dtype=complex)
    omega2 = (2 * np.pi * f) ** 2
    mask = omega2 > 0
    out[mask] = H_accel[mask] / (-omega2[mask])
    out[~mask] = 0.0
    return out


class FrfSensorConfigPage(QWidget):
    default_save_requested = Signal(dict, dict, dict)

    def __init__(self, model: FrfInputModel) -> None:
        super().__init__()
        self._model = model
        self._repo_root = Path(__file__).resolve().parents[3]
        self._sens_path = self._repo_root / "config" / "sensitivities.json"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title = QLabel("Sensor configuration")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        tabs = QTabWidget()
        layout.addWidget(tabs, 1)

        self._mapping_table = QTableWidget(0, 3)
        self._mapping_table.setHorizontalHeaderLabels(["Sensor name", "Channel idx", "Sensitivity name"])
        self._mapping_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._mapping_table.customContextMenuRequested.connect(self._mapping_context_menu)
        self._mapping_table.itemChanged.connect(self._sync_mapping_from_table)
        tabs.addTab(self._mapping_table, "Channel + sensitivity mapping")

        self._sens_table = QTableWidget(0, 2)
        self._sens_table.setHorizontalHeaderLabels(["Sensitivity name", "Value"])
        self._sens_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._sens_table.customContextMenuRequested.connect(self._sens_context_menu)
        self._sens_table.itemChanged.connect(self._auto_save_sensitivities)
        tabs.addTab(self._sens_table, "Sensitivities")

        self._btn_save_default = QPushButton("Save as default")
        self._btn_save_default.clicked.connect(self._emit_save_default)
        layout.addWidget(self._btn_save_default)

        self._reload_from_model()
        self._reload_sensitivities()

    def apply_default_sensor_config(
        self,
        channel_mapping: Dict[str, int],
        sensitivity_mapping: Dict[str, str],
        sensitivities: Dict[str, float],
    ) -> None:
        self._model.channel_mapping = dict(channel_mapping or {})
        self._model.sensitivity_mapping = dict(sensitivity_mapping or {})

        self._reload_from_model()

        # Make the sensitivity table consistent with the saved defaults by writing the file once.
        # This also keeps the existing "auto-save sensitivities to config/sensitivities.json" behavior intact.
        if sensitivities is not None:
            payload: Dict[str, float] = {}
            if isinstance(sensitivities, dict):
                for k, v in sensitivities.items():
                    try:
                        payload[str(k)] = float(v)
                    except Exception:
                        continue
            self._sens_path.parent.mkdir(parents=True, exist_ok=True)
            self._sens_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._reload_sensitivities()

    def _reload_from_model(self) -> None:
        self._mapping_table.blockSignals(True)
        try:
            self._mapping_table.setRowCount(0)
            keys = sorted(set(self._model.channel_mapping.keys()) | set(self._model.sensitivity_mapping.keys()))
            for k in keys:
                self._append_mapping_row(
                    sensor_name=k,
                    channel_idx=self._model.channel_mapping.get(k),
                    sens_name=self._model.sensitivity_mapping.get(k),
                )
        finally:
            self._mapping_table.blockSignals(False)

    def _append_mapping_row(self, sensor_name: str = "", channel_idx: Optional[int] = None, sens_name: str = "") -> None:
        r = self._mapping_table.rowCount()
        self._mapping_table.insertRow(r)
        self._mapping_table.setItem(r, 0, QTableWidgetItem(sensor_name))
        self._mapping_table.setItem(r, 1, QTableWidgetItem("" if channel_idx is None else str(channel_idx)))
        self._mapping_table.setItem(r, 2, QTableWidgetItem(sens_name or ""))

    def _mapping_context_menu(self, pos) -> None:
        menu = QMenu(self)
        act_add = menu.addAction("Add row")
        act_del = menu.addAction("Delete selected row(s)")
        act = menu.exec(self._mapping_table.mapToGlobal(pos))
        if act == act_add:
            self._mapping_table.blockSignals(True)
            try:
                self._append_mapping_row()
            finally:
                self._mapping_table.blockSignals(False)
        elif act == act_del:
            rows = sorted({i.row() for i in self._mapping_table.selectedIndexes()}, reverse=True)
            if not rows:
                return
            self._mapping_table.blockSignals(True)
            try:
                for r in rows:
                    self._mapping_table.removeRow(r)
            finally:
                self._mapping_table.blockSignals(False)
            self._sync_mapping_from_table()

    def _sync_mapping_from_table(self) -> None:
        chan: Dict[str, int] = {}
        sens_map: Dict[str, str] = {}
        for r in range(self._mapping_table.rowCount()):
            name_item = self._mapping_table.item(r, 0)
            ch_item = self._mapping_table.item(r, 1)
            s_item = self._mapping_table.item(r, 2)
            if name_item is None:
                continue
            sensor_name = (name_item.text() or "").strip()
            if not sensor_name:
                continue
            if ch_item is not None:
                txt = (ch_item.text() or "").strip()
                if txt:
                    try:
                        chan[sensor_name] = int(txt)
                    except ValueError:
                        # leave it out; the run step will validate anyway
                        pass
            if s_item is not None:
                s_txt = (s_item.text() or "").strip()
                if s_txt:
                    sens_map[sensor_name] = s_txt
        self._model.channel_mapping = chan
        self._model.sensitivity_mapping = sens_map

    def _reload_sensitivities(self) -> None:
        self._sens_table.blockSignals(True)
        try:
            self._sens_table.setRowCount(0)
            if not self._sens_path.exists():
                return
            data = json_load(self._sens_path)
            for k in sorted([kk for kk in data.keys() if not str(kk).startswith("//")]):
                r = self._sens_table.rowCount()
                self._sens_table.insertRow(r)
                self._sens_table.setItem(r, 0, QTableWidgetItem(str(k)))
                self._sens_table.setItem(r, 1, QTableWidgetItem(str(data[k])))
        finally:
            self._sens_table.blockSignals(False)

    def _sens_context_menu(self, pos) -> None:
        menu = QMenu(self)
        act_add = menu.addAction("Add sensitivity")
        act_del = menu.addAction("Delete selected row(s)")
        act = menu.exec(self._sens_table.mapToGlobal(pos))
        if act == act_add:
            self._sens_table.blockSignals(True)
            try:
                r = self._sens_table.rowCount()
                self._sens_table.insertRow(r)
                self._sens_table.setItem(r, 0, QTableWidgetItem("new_sensitivity"))
                self._sens_table.setItem(r, 1, QTableWidgetItem("1.0"))
            finally:
                self._sens_table.blockSignals(False)
            self._auto_save_sensitivities()
        elif act == act_del:
            rows = sorted({i.row() for i in self._sens_table.selectedIndexes()}, reverse=True)
            if not rows:
                return
            self._sens_table.blockSignals(True)
            try:
                for r in rows:
                    self._sens_table.removeRow(r)
            finally:
                self._sens_table.blockSignals(False)
            self._auto_save_sensitivities()

    def _auto_save_sensitivities(self) -> None:
        data = self._collect_sensitivities_from_table()
        self._sens_path.parent.mkdir(parents=True, exist_ok=True)
        self._sens_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _collect_sensitivities_from_table(self) -> Dict[str, float]:
        data: Dict[str, float] = {}
        for r in range(self._sens_table.rowCount()):
            k_item = self._sens_table.item(r, 0)
            v_item = self._sens_table.item(r, 1)
            if k_item is None or v_item is None:
                continue
            k = (k_item.text() or "").strip()
            if not k or k.startswith("//"):
                continue
            v_txt = (v_item.text() or "").strip()
            if not v_txt:
                continue
            try:
                data[k] = float(v_txt)
            except ValueError:
                continue
        return data

    def _emit_save_default(self) -> None:
        self._sync_mapping_from_table()
        self.default_save_requested.emit(
            dict(self._model.channel_mapping or {}),
            dict(self._model.sensitivity_mapping or {}),
            dict(self._collect_sensitivities_from_table()),
        )

