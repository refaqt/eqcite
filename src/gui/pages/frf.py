from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List

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
    QInputDialog,
)

from ... import experiment_config, expression_eval, frf_estimation, load_data, signal_processing
from ..analysis_export import export_frf_results, import_frf_results
from ..plot_widgets import MatplotlibFigureWidget


def _required_wiener_mysize(filt_params: Dict[str, Any]) -> int:
    w = filt_params.get("wiener")
    if not isinstance(w, dict) or "mysize" not in w:
        raise ValueError("filtering.params.wiener.mysize is required (integer samples)")
    m = int(w["mysize"])
    if m < 1:
        raise ValueError("wiener.mysize must be >= 1")
    return m


def _required_trend_estimation(
    filtering_cfg: Dict[str, Any],
    signal_key: str,
) -> Tuple[float, float]:
    trend = filtering_cfg.get("trend_estimation")
    if not isinstance(trend, dict):
        raise ValueError("filtering.trend_estimation is required")
    signal_trend = trend.get(signal_key)
    path = f"filtering.trend_estimation.{signal_key}"
    if not isinstance(signal_trend, dict):
        raise ValueError(f"{path} is required")
    if "impact_start_s" not in signal_trend or "ringdown_end_s" not in signal_trend:
        raise ValueError(f"{path}.impact_start_s and {path}.ringdown_end_s are required")
    impact = float(signal_trend["impact_start_s"])
    ringdown = float(signal_trend["ringdown_end_s"])
    if not np.isfinite(impact) or not np.isfinite(ringdown):
        raise ValueError(f"{path} values must be finite")
    if ringdown <= impact:
        raise ValueError(f"{path}.ringdown_end_s must be greater than impact_start_s")
    return impact, ringdown


def _strip_savgol_from_filtering_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Remove obsolete Savitzky–Golay chain steps and params from saved filtering dicts."""
    out = dict(cfg or {})
    for key in ("u_chain", "resp_chain"):
        ch = out.get(key)
        if isinstance(ch, list):
            out[key] = [
                c
                for c in ch
                if isinstance(c, dict) and str(c.get("name") or "") != "savgol"
            ]
    p = out.get("params")
    if isinstance(p, dict) and "savgol" in p:
        p = dict(p)
        p.pop("savgol", None)
        out["params"] = p
    return out


from .frf_model import FrfInputModel
from .frf_experiment_config import FrfSensorConfigPage
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

    io_input_name: str = ""
    io_output_name: str = ""


@dataclass
class _DirectionalMeasurement:
    direction: str  # "x" | "y"
    filepath: Optional[str]
    raw_channels: Optional[Dict[int, np.ndarray]]
    proc: _DirectionalProcessing
    compliance: Tuple[np.ndarray, np.ndarray]  # (f_hz, H_disp_over_force complex)
    accelerance: Tuple[np.ndarray, np.ndarray]  # (f_hz, H_accel_over_force complex)


class FrfRootPage(QWidget):
    default_experiment_config_save_requested = Signal(dict)
    default_filtering_save_requested = Signal(dict)
    file_paths_changed = Signal(object, object)  # (x_paths, y_paths) where each is List[str]

    def __init__(self, *, compare_entries_shared: Optional[List[_CompareEntry]] = None) -> None:
        super().__init__()

        self._menu = QListWidget()
        self._stack = QStackedWidget()

        repo_root = Path(__file__).resolve().parents[3]
        cfg_path = repo_root / "config" / "config_before_cabinet_assembly.json"
        base = json_load(cfg_path) if cfg_path.exists() else {}
        sens_path = repo_root / "config" / "sensitivities.json"
        try:
            sens = load_data.load_sensitivities(sens_path) if sens_path.exists() else {}
        except Exception:
            sens = {}
        if not experiment_config.uses_new_experiment_schema(base) and (
            base.get("channel_mapping") or base.get("sensitivity_mapping")
        ):
            base = experiment_config.migrate_legacy_experiment_cfg(base, sens)

        self._input_model = FrfInputModel.from_experiment_cfg(base)

        self._page_input = FrfInputPage(self._input_model)
        self._page_sensor = FrfSensorConfigPage(self._input_model)
        self._page_filtering = FrfFilteringPage()
        self._page_results = FrfResultsPage(
            self._input_model,
            sync_experiment_widgets=self._page_sensor.sync_from_widgets,
        )
        self._page_compare = FrfComparePage(
            open_frf_tabs_provider=lambda: [],
            entries_shared=compare_entries_shared,
        )

        self._page_sensor.default_save_requested.connect(self.default_experiment_config_save_requested)
        self._page_input.file_paths_changed.connect(self.file_paths_changed)
        self._page_filtering.default_save_requested.connect(self.default_filtering_save_requested)
        self._page_filtering.filtering_changed.connect(self._page_results.set_filtering_cfg)

        self._stack.addWidget(self._wrap(self._page_input))
        self._stack.addWidget(self._wrap(self._page_sensor))
        self._stack.addWidget(self._wrap(self._page_filtering))
        self._stack.addWidget(self._wrap(self._page_results))
        self._stack.addWidget(self._wrap(self._page_compare))

        for name in ["Input data", "Experiment configuration", "Analysis settings", "Results", "Compare"]:
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

    def set_open_frf_tabs_provider(self, provider: Callable[[], List[object]]) -> None:
        """
        Provide a callable returning the currently-open FRF analysis tabs.

        The provider is injected by the workspace so Compare can pull results from other tabs.
        """
        self._page_compare.set_open_frf_tabs_provider(provider)

    def set_default_load_dir(self, path: str) -> None:
        self._page_input.set_default_load_dir(path)

    def set_file_paths(self, x_paths: List[str], y_paths: List[str]) -> None:
        self._page_input.set_file_paths(list(x_paths or []), list(y_paths or []))

    def apply_default_experiment_config(self, cfg: Dict[str, Any]) -> None:
        self._page_sensor.apply_experiment_config(dict(cfg or {}))

    def apply_default_sensor_config(
        self,
        channel_mapping: Dict[str, int],
        sensitivity_mapping: Dict[str, str],
        sensitivities: Dict[str, float],
    ) -> None:
        """Backward-compatible entry point: migrates legacy triple to new schema."""
        legacy = {
            "channel_mapping": dict(channel_mapping or {}),
            "sensitivity_mapping": dict(sensitivity_mapping or {}),
        }
        migrated = experiment_config.migrate_legacy_experiment_cfg(legacy, dict(sensitivities or {}))
        self.apply_default_experiment_config(migrated)

    def export_results(self, analysis_folder: Path) -> dict:
        return self._page_results.export_results(analysis_folder)

    def get_sidebar_index(self) -> int:
        return int(self._menu.currentRow())

    def set_sidebar_index(self, idx: int) -> None:
        if 0 <= int(idx) < self._stack.count():
            self._menu.setCurrentRow(int(idx))

    def build_session_snapshot(self) -> Dict[str, Any]:
        self._page_sensor.sync_from_widgets()
        return {
            "experiment_cfg": self.build_experiment_cfg(),
            "gui_state": {
                "sidebar_index": self.get_sidebar_index(),
                "experiment_tab_index": self._page_sensor.get_tab_index(),
                "frf_results": self._page_results.build_gui_state(),
            },
        }

    def apply_session_snapshot(self, experiment_cfg: Dict[str, Any], gui_state: Dict[str, Any]) -> None:
        self.apply_experiment_cfg(dict(experiment_cfg or {}))
        gui_state = dict(gui_state or {})
        self.set_sidebar_index(int(gui_state.get("sidebar_index", 0)))
        self._page_sensor.set_tab_index(int(gui_state.get("experiment_tab_index", 0)))
        self._page_results.apply_gui_state(dict(gui_state.get("frf_results", {}) or {}))

    def hydrate_results_from_manifest(self, analysis_folder: Path, manifest: Dict[str, Any]) -> None:
        self._page_results.hydrate_from_manifest(analysis_folder, manifest)

    def apply_default_filtering(self, filtering_cfg: Dict[str, Any]) -> None:
        self._page_filtering.apply_filtering_cfg(dict(filtering_cfg or {}))

    def build_experiment_cfg(self) -> Dict[str, Any]:
        self._page_sensor.sync_from_widgets()
        base = self._page_results._build_experiment_cfg()
        f_cfg = self._page_filtering.build_filtering_cfg()
        base["filtering"] = f_cfg
        return experiment_config.experiment_cfg_for_snapshot(base)

    def apply_experiment_cfg(self, cfg: Dict[str, Any]) -> None:
        cfg = dict(cfg or {})
        self._page_input.apply_from_experiment_cfg(cfg)
        frag = {
            k: cfg[k]
            for k in (
                "sensor_catalog",
                "mapping",
                "expressions",
                "analysis_io",
                "simulation_signal_map",
                "channel_mapping",
                "sensitivity_mapping",
            )
            if k in cfg
        }
        if frag:
            self._page_sensor.apply_experiment_config(frag)
        filt = dict(cfg.get("filtering") or {})
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

    def apply_from_experiment_cfg(self, cfg: Dict[str, Any]) -> None:
        self._model.data_source = str(cfg.get("data_source") or "file")
        self._model.file_paths_x = list(cfg.get("file_paths_x") or [])
        self._model.file_paths_y = list(cfg.get("file_paths_y") or [])
        self._model.simulation = dict(cfg.get("simulation") or {})
        self._rb_sim.setChecked(self._model.data_source == "simulation")
        self._rb_file.setChecked(self._model.data_source != "simulation")
        self._on_mode_changed()
        self._set_list_items(self._x_list, list(self._model.file_paths_x or []))
        self._set_list_items(self._y_list, list(self._model.file_paths_y or []))
        sim = dict(self._model.simulation or {})
        model = dict(sim.get("model") or {})
        hammer = dict(sim.get("hammer") or {})
        self._sim_fs.setText(str(sim.get("fs", self._sim_fs.text())))
        self._sim_T.setText(str(sim.get("T", self._sim_T.text())))
        self._sim_mass.setText(str(model.get("mass", self._sim_mass.text())))
        self._sim_stiff.setText(str(model.get("stiffness", self._sim_stiff.text())))
        self._sim_damp.setText(str(model.get("damping", self._sim_damp.text())))
        self._sim_hammer_force.setText(str(hammer.get("force", self._sim_hammer_force.text())))
        self._sim_hammer_bw.setText(str(hammer.get("bandwidth_hz", self._sim_hammer_bw.text())))
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

        filtering_header = QLabel("Filtering")
        filtering_header.setStyleSheet("font-weight: 600; font-style: italic;")
        layout.addWidget(filtering_header)

        entries = [("wiener", "Wiener")]
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

        self._trend_force_impact_start = QLineEdit("0.0")
        self._trend_force_ringdown_end = QLineEdit("0.1")
        self._trend_resp_impact_start = QLineEdit("0.0")
        self._trend_resp_ringdown_end = QLineEdit("0.1")
        filt_form.addRow(QLabel("Force trend estimation: impact start (s)"), self._trend_force_impact_start)
        filt_form.addRow(QLabel("Force trend estimation: ringdown end (s)"), self._trend_force_ringdown_end)
        filt_form.addRow(QLabel("Response trend estimation: impact start (s)"), self._trend_resp_impact_start)
        filt_form.addRow(QLabel("Response trend estimation: ringdown end (s)"), self._trend_resp_ringdown_end)

        self._wiener_mysize = QLineEdit("1000")
        filt_form.addRow("Wiener local window / mysize (samples)", self._wiener_mysize)

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
        self._win_rolloff = QLineEdit("0.01875")
        self._win_max_start = QLineEdit("-0.03125")
        self._win_max_end = QLineEdit("0.08125")
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
        _add_row("window_rolloff_seconds", "Rolloff time (s)", self._win_rolloff)
        _add_row("window_max_start_seconds", "Window max start (s, vs peak)", self._win_max_start)
        _add_row("window_max_end_seconds", "Window max end (s, vs peak)", self._win_max_end)
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

        # Default: Wiener only (enabled)
        self._u_chain.set_chain([{"name": "wiener", "enabled": True}])
        self._resp_chain.set_chain([{"name": "wiener", "enabled": True}])
        self._u_chain.set_show_all_items()
        self._resp_chain.set_show_all_items()
        self._refresh_preview()

        for w in [
            self._trend_force_impact_start,
            self._trend_force_ringdown_end,
            self._trend_resp_impact_start,
            self._trend_resp_ringdown_end,
            self._wiener_mysize,
            self._win_hann_width,
            self._win_rolloff,
            self._win_max_start,
            self._win_max_end,
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
            show |= {
                "window_rolloff_seconds",
                "window_max_start_seconds",
                "window_max_end_seconds",
            }
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
            f"Pipeline: raw → detrend → u:[{fmt(self._u_chain.get_chain())}] → resp:[{fmt(self._resp_chain.get_chain())}] → windowing"
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
        def i(txt: str) -> int:
            return int(float((txt or "").strip() or "0") or 0)

        def f(txt: str) -> float:
            return float((txt or "").strip() or "0")

        return {
            "estimator_type": str(self._estimator_type.currentData() or "Hv"),
            "trend_estimation": {
                "force": {
                    "impact_start_s": f(self._trend_force_impact_start.text()),
                    "ringdown_end_s": f(self._trend_force_ringdown_end.text()),
                },
                "response": {
                    "impact_start_s": f(self._trend_resp_impact_start.text()),
                    "ringdown_end_s": f(self._trend_resp_ringdown_end.text()),
                },
            },
            "u_chain": self._u_chain.get_chain(),
            "resp_chain": self._resp_chain.get_chain(),
            "params": {
                "wiener": {"mysize": i(self._wiener_mysize.text())},
            },
            "windowing": {
                "input_window_type": str(self._win_in_type.currentData() or "hann"),
                "output_window_type": str(self._win_out_type.currentData() or "exponential"),
                "window_hann_width": f(self._win_hann_width.text()),
                "window_rolloff_seconds": f(self._win_rolloff.text()),
                "window_max_start_seconds": f(self._win_max_start.text()),
                "window_max_end_seconds": f(self._win_max_end.text()),
                "window_trigger_percent": f(self._win_trigger_pct.text()),
                "window_low_percent": f(self._win_low_pct.text()),
                "window_sliding_size": i(self._win_sliding.text()),
                "window_std_multiplier": f(self._win_std_mult.text()),
                "window_decay_db": f(self._win_decay_db.text()),
            },
        }

    def apply_filtering_cfg(self, cfg: Dict[str, Any]) -> None:
        cfg = _strip_savgol_from_filtering_cfg(dict(cfg or {}))

        def _set_combo(cb: QComboBox, value: str) -> None:
            for idx in range(cb.count()):
                if cb.itemData(idx) == value:
                    cb.setCurrentIndex(idx)
                    return

        # Legacy io_selection removed from filtering config
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

        trend = cfg.get("trend_estimation")
        if isinstance(trend, dict):
            force = trend.get("force")
            if isinstance(force, dict):
                if "impact_start_s" in force:
                    self._trend_force_impact_start.setText(str(force.get("impact_start_s")))
                if "ringdown_end_s" in force:
                    self._trend_force_ringdown_end.setText(str(force.get("ringdown_end_s")))
            response = trend.get("response")
            if isinstance(response, dict):
                if "impact_start_s" in response:
                    self._trend_resp_impact_start.setText(str(response.get("impact_start_s")))
                if "ringdown_end_s" in response:
                    self._trend_resp_ringdown_end.setText(str(response.get("ringdown_end_s")))

        params = cfg.get("params") or {}
        if isinstance(params, dict):
            w = params.get("wiener") or {}
            if isinstance(w, dict) and "mysize" in w:
                self._wiener_mysize.setText(str(w.get("mysize") or ""))

        win = cfg.get("windowing") or {}
        if isinstance(win, dict):
            _set_combo(self._win_in_type, str(win.get("input_window_type") or "hann"))
            _set_combo(self._win_out_type, str(win.get("output_window_type") or "exponential"))

            win = dict(win)
            if any(
                k in win
                for k in ("window_pre_seconds", "window_post_seconds", "window_beta")
            ):
                win.update(signal_processing.normalize_raised_cosine_window_params(win))

            mapping = {
                "window_hann_width": self._win_hann_width,
                "window_rolloff_seconds": self._win_rolloff,
                "window_max_start_seconds": self._win_max_start,
                "window_max_end_seconds": self._win_max_end,
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
    def __init__(
        self,
        model: FrfInputModel,
        *,
        sync_experiment_widgets: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._sync_experiment_widgets = sync_experiment_widgets or (lambda: None)
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
        self._last_exp_cfg: Dict[str, Any] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title = QLabel("Results")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        self._run_btn = QPushButton("Generate results")
        self._run_btn.clicked.connect(self._run)
        layout.addWidget(self._run_btn)

        self._io_resolved_label = QLabel("")
        self._io_resolved_label.setWordWrap(True)
        layout.addWidget(self._io_resolved_label)

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
                compliance_avg=True,
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
                compliance_avg=False,
            )
            return

    def set_filtering_cfg(self, cfg: Dict[str, Any]) -> None:
        self._filtering_cfg = _strip_savgol_from_filtering_cfg(dict(cfg or {}))

    def _build_experiment_cfg(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "analysis_type": "frf",
            "data_source": self._model.data_source,
        }
        out.update(self._model.to_experiment_cfg_fragment())
        if self._model.data_source == "simulation":
            out["simulation"] = self._model.simulation or {}
        else:
            out["file_paths_x"] = list(self._model.file_paths_x or [])
            out["file_paths_y"] = list(self._model.file_paths_y or [])
        return out

    def _run(self) -> None:
        try:
            self._sync_experiment_widgets()
            exp_cfg = self._build_experiment_cfg()
            self._last_exp_cfg = dict(exp_cfg)
            experiment_config.validate_analysis_io(exp_cfg)
            directional = load_data.load_directional_data(exp_cfg, {})

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

            self._refresh_resolved_io_label()
            self._populate_measurement_dropdowns()
            for tid in ["raw_mosaic", "time_steps", "fft_mag", "coherence", "force_psd", "real_frf", "bode_disp", "bode_accel"]:
                self._redraw_tab(tid)
        except Exception as e:
            QMessageBox.critical(self, "FRF run failed", str(e))

    def _refresh_resolved_io_label(self) -> None:
        parts: List[str] = []
        if self._last_proc_x is not None:
            parts.append(
                f"X input: {self._last_proc_x.io_input_name or '?'}; "
                f"output: {self._last_proc_x.io_output_name or '?'}"
            )
        if self._last_proc_y is not None:
            parts.append(
                f"Y input: {self._last_proc_y.io_input_name or '?'}; "
                f"output: {self._last_proc_y.io_output_name or '?'}"
            )
        if parts:
            self._io_resolved_label.setText("Resolved I/O — " + " | ".join(parts))
        else:
            self._io_resolved_label.clear()

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

    def build_gui_state(self) -> Dict[str, Any]:
        return {
            "results_tab_index": int(self._tabs.currentIndex()),
            "time_steps_mode": str(self._time_steps_mode),
            "tab_sel_idx": {k: list(v) for k, v in (self._tab_sel_idx or {}).items()},
            "average_checked": {
                k: bool(cb.isChecked()) for k, cb in (self._tab_average_checks or {}).items()
            },
        }

    def apply_gui_state(self, gui_state: Dict[str, Any]) -> None:
        gui_state = dict(gui_state or {})
        tab_sel = dict(gui_state.get("tab_sel_idx", {}) or {})
        for tab_id, (x_cb, y_cb) in (self._tab_selectors or {}).items():
            idxs = tab_sel.get(tab_id, [0, 0])
            x_idx = int(idxs[0]) if len(idxs) > 0 else 0
            y_idx = int(idxs[1]) if len(idxs) > 1 else 0
            self._tab_sel_idx[tab_id] = (x_idx, y_idx)
            if x_cb.count() > 0:
                x_cb.setCurrentIndex(min(max(x_idx, 0), x_cb.count() - 1))
            if y_cb.count() > 0:
                y_cb.setCurrentIndex(min(max(y_idx, 0), y_cb.count() - 1))
        for tab_id, checked in (gui_state.get("average_checked", {}) or {}).items():
            cb = self._tab_average_checks.get(str(tab_id))
            if cb is not None:
                cb.setChecked(bool(checked))
        mode = str(gui_state.get("time_steps_mode") or self._time_steps_mode)
        self._time_steps_mode = mode
        self._refresh_time_steps_controls()
        ridx = int(gui_state.get("results_tab_index", 0))
        if 0 <= ridx < self._tabs.count():
            self._tabs.setCurrentIndex(ridx)

    @staticmethod
    def _measurement_from_hydrated(entry: Dict[str, Any]) -> _DirectionalMeasurement:
        direction = str(entry.get("direction") or "x")
        time_s = np.asarray(entry["time_s"], dtype=float)
        dt = float(time_s[1] - time_s[0]) if len(time_s) > 1 else 1.0
        u_steps = [_SignalStep(str(n), np.asarray(y, dtype=float)) for n, y in entry.get("u_steps", [])]
        r_steps = [_SignalStep(str(n), np.asarray(y, dtype=float)) for n, y in entry.get("resp_steps", [])]
        f_hz, Hc = entry["compliance"]
        _, Ha = entry["accelerance"]
        proc = _DirectionalProcessing(
            direction=direction,
            time=time_s,
            fs_hz=1.0 / dt if dt else 1.0,
            u_steps=u_steps,
            resp_steps=r_steps,
            window=None,
            f_hz=np.asarray(f_hz, dtype=float),
            psd_u=np.asarray(entry.get("psd_u", []), dtype=float),
            coherence_uy=np.asarray(entry.get("coherence", []), dtype=float),
            csd_ur=np.asarray(entry.get("csd_ur", []), dtype=complex),
            psd_r=np.asarray(entry.get("psd_r", []), dtype=float),
            fft_f_hz=np.asarray(entry.get("fft_f_hz", []), dtype=float),
            fft_u_mag=dict(entry.get("fft_u_mag", {}) or {}),
            fft_resp_mag=dict(entry.get("fft_resp_mag", {}) or {}),
            fft_u_complex=dict(entry.get("fft_u_complex", {}) or {}),
            fft_resp_complex=dict(entry.get("fft_resp_complex", {}) or {}),
            io_input_name=str(entry.get("io_input_name") or ""),
            io_output_name=str(entry.get("io_output_name") or ""),
        )
        raw_channels = entry.get("raw_channels")
        if raw_channels is not None and not isinstance(raw_channels, dict):
            raw_channels = None
        return _DirectionalMeasurement(
            direction=direction,
            filepath=entry.get("filepath"),
            raw_channels=dict(raw_channels) if raw_channels else None,
            proc=proc,
            compliance=(np.asarray(f_hz, dtype=float), np.asarray(Hc, dtype=complex)),
            accelerance=(np.asarray(f_hz, dtype=float), np.asarray(Ha, dtype=complex)),
        )

    def hydrate_from_manifest(self, analysis_folder: Path, manifest: Dict[str, Any]) -> None:
        if not manifest:
            return
        bundle = import_frf_results(Path(analysis_folder), dict(manifest or {}))
        self._x_measurements = [self._measurement_from_hydrated(e) for e in bundle.get("x", [])]
        self._y_measurements = [self._measurement_from_hydrated(e) for e in bundle.get("y", [])]
        if self._x_measurements:
            self._last_x = _DirectionalFrfResult(
                direction="x",
                compliance=self._x_measurements[0].compliance,
                accelerance=self._x_measurements[0].accelerance,
            )
            self._last_proc_x = self._x_measurements[0].proc
        if self._y_measurements:
            self._last_y = _DirectionalFrfResult(
                direction="y",
                compliance=self._y_measurements[0].compliance,
                accelerance=self._y_measurements[0].accelerance,
            )
            self._last_proc_y = self._y_measurements[0].proc
        self._refresh_resolved_io_label()
        self._populate_measurement_dropdowns()
        for tid in [
            "raw_mosaic",
            "time_steps",
            "fft_mag",
            "coherence",
            "force_psd",
            "real_frf",
            "bode_disp",
            "bode_accel",
        ]:
            self._redraw_tab(tid)

    def export_results(self, analysis_folder: Path) -> dict:
        meta = dict(self._last_run_meta or {})
        meta["filtering"] = dict(self._filtering_cfg or {})
        meta["estimator_type"] = str(self._filtering_cfg.get("estimator_type") or "Hv")
        return export_frf_results(
            Path(analysis_folder),
            x_measurements=self._x_measurements,
            y_measurements=self._y_measurements,
            meta=meta,
        )

    def list_compliance_sources(self, *, direction: str) -> List[Tuple[str, str, int]]:
        """
        Enumerate selectable compliance sources for Compare.

        Returns a list of (label, kind, index) where:
        - kind == "measurement" and index is the measurement index, or
        - kind == "average" and index is always 0.
        """
        direction = (direction or "x").strip().lower()
        measurements = self._x_measurements if direction == "x" else self._y_measurements
        paths = [str(m.filepath or "") for m in (measurements or [])]
        labels = self._disambiguate_labels(paths)
        out: List[Tuple[str, str, int]] = []
        for i, lab in enumerate(labels):
            out.append((str(lab), "measurement", int(i)))
        if measurements:
            out.append(("Average", "average", 0))
        return out

    def get_compliance_series(self, *, direction: str, kind: str, index: int = 0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return (f_hz, Hc_complex) for compliance (displacement/force).

        - kind="measurement": uses selected measurement index
        - kind="average": uses estimator-consistent average in complex spectra space
        """
        direction = (direction or "x").strip().lower()
        kind = (kind or "measurement").strip().lower()
        measurements = self._x_measurements if direction == "x" else self._y_measurements
        if not measurements:
            return None

        if kind == "average":
            et = str(self._filtering_cfg.get("estimator_type") or "Hv")
            procs = [m.proc for m in measurements]
            f_hz, _H_accel, H_disp = frf_estimation.average_frf_from_processing(procs, estimator_type=et)
            return np.asarray(f_hz, dtype=float), np.asarray(H_disp, dtype=complex)

        idx = int(index)
        if idx < 0 or idx >= len(measurements):
            return None
        f_hz, Hc = measurements[idx].compliance
        return np.asarray(f_hz, dtype=float), np.asarray(Hc, dtype=complex)


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

        exp_cfg = dict(getattr(self, "_last_exp_cfg", {}) or {})
        u_name, y_name = experiment_config.resolve_analysis_io_names(exp_cfg, direction)

        u, y_accel = experiment_config.bind_directional_io_signals(data, u_name, y_name)

        time = np.asarray(data["time"])

        params = frf_estimation.get_system_parameters()
        filtering = dict(self._filtering_cfg or {})
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
        H_disp = frf_estimation.accel_to_disp_over_force(f, H_accel)

        proc = self._compute_processing_steps(
            time=time,
            u=u,
            resp=y_accel,
            direction=direction,
            params=params,
            filtering_cfg=filtering,
            io_input_name=u_name,
            io_output_name=y_name,
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
            default_visible = {"raw", "detrended", "window", "windowed", "wiener"}
            for s in steps:
                line = ax.plot(t, s.y, linewidth=1.0, label=f"{prefix}{s.name}")[0]
                line.set_visible(s.name in default_visible)

        def _pfx(proc: Optional[_DirectionalProcessing], default: str, io: str) -> str:
            if proc is not None and getattr(proc, io, ""):
                return str(getattr(proc, io)) + ":"
            return default

        title_map = {
            "force_x": "Input (X) processing steps",
            "resp_x": "Output (X) processing steps",
            "force_y": "Input (Y) processing steps",
            "resp_y": "Output (Y) processing steps",
        }

        if mode == "force_x" and x_proc is not None:
            add_steps(ax, x_proc.time, x_proc.u_steps, prefix=_pfx(x_proc, "Fx:", "io_input_name"))
            ax.set_ylabel("Force (N)")
        elif mode == "resp_x" and x_proc is not None:
            add_steps(ax, x_proc.time, x_proc.resp_steps, prefix=_pfx(x_proc, "x:", "io_output_name"))
            ax.set_ylabel("Acceleration (m/s²)")
        elif mode == "force_y" and y_proc is not None:
            add_steps(ax, y_proc.time, y_proc.u_steps, prefix=_pfx(y_proc, "Fy:", "io_input_name"))
            ax.set_ylabel("Force (N)")
        elif mode == "resp_y" and y_proc is not None:
            add_steps(ax, y_proc.time, y_proc.resp_steps, prefix=_pfx(y_proc, "y:", "io_output_name"))
            ax.set_ylabel("Acceleration (m/s²)")
        else:
            if x_proc is not None:
                add_steps(ax, x_proc.time, x_proc.u_steps, prefix=_pfx(x_proc, "Fx:", "io_input_name"))
                ax.set_ylabel("Force (N)")
                mode = "force_x"
            elif y_proc is not None:
                add_steps(ax, y_proc.time, y_proc.u_steps, prefix=_pfx(y_proc, "Fy:", "io_input_name"))
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
            et = str(self._filtering_cfg.get("estimator_type") or "Hv")

            def avg_frf(procs: List[_DirectionalProcessing]) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
                if not procs:
                    return None
                f_hz, H_accel, H_disp = frf_estimation.average_frf_from_processing(procs, estimator_type=et)
                return (f_hz, H_disp), (f_hz, H_accel)

            x_avg = avg_frf([m.proc for m in (getattr(self, "_x_measurements", []) or [])])
            y_avg = avg_frf([m.proc for m in (getattr(self, "_y_measurements", []) or [])])
            if x_avg is not None:
                add(ax1, x_avg[0], "X avg")
                add(ax2, x_avg[1], "X avg")
            if y_avg is not None:
                add(ax1, y_avg[0], "Y avg")
                add(ax2, y_avg[1], "Y avg")

        ax1.legend(loc="best")
        ax2.legend(loc="best")
        widget.redraw()

    def _plot_bode(
        self,
        widget: MatplotlibFigureWidget,
        x: Optional[Tuple[np.ndarray, np.ndarray]],
        y: Optional[Tuple[np.ndarray, np.ndarray]],
        y_label: str,
        title: str,
        *,
        avg_on: bool = False,
        compliance_avg: bool = True,
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
            et = str(self._filtering_cfg.get("estimator_type") or "Hv")

            def add_avg(measurements: Optional[List[_DirectionalMeasurement]], label: str) -> None:
                if not measurements:
                    return
                procs = [m.proc for m in measurements]
                f_hz, H_accel, H_disp = frf_estimation.average_frf_from_processing(procs, estimator_type=et)
                H = H_disp if compliance_avg else H_accel
                add_trace((f_hz, H), f"{label} avg")

            add_avg(getattr(self, "_x_measurements", None) or [], "X")
            add_avg(getattr(self, "_y_measurements", None) or [], "Y")

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
        io_input_name: str = "",
        io_output_name: str = "",
    ) -> _DirectionalProcessing:
        time = np.asarray(time, dtype=float)
        u = np.asarray(u, dtype=float)
        resp = np.asarray(resp, dtype=float)

        if len(time) < 2:
            raise ValueError("Not enough samples to compute processing steps")
        dt = float(time[1] - time[0])
        fs = 1.0 / dt

        filtering_cfg = dict(filtering_cfg or {})
        u_impact, u_ringdown = _required_trend_estimation(filtering_cfg, "force")
        r_impact, r_ringdown = _required_trend_estimation(filtering_cfg, "response")
        u_mask = signal_processing.quiet_regions_mask(time, u_impact, u_ringdown)
        r_mask = signal_processing.quiet_regions_mask(time, r_impact, r_ringdown)

        u_detrended, _u_trend = signal_processing.detrend_quiet_regions(time, u, u_impact, u_ringdown)
        r_detrended, _r_trend = signal_processing.detrend_quiet_regions(time, resp, r_impact, r_ringdown)

        u_steps: List[_SignalStep] = [_SignalStep("raw", u), _SignalStep("detrended", u_detrended)]
        r_steps: List[_SignalStep] = [_SignalStep("raw", resp), _SignalStep("detrended", r_detrended)]

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
                    noise_mask = u_mask if is_u else r_mask
                    noise_pow = signal_processing.calculate_noise_power_from_mask(x_cur, noise_mask)
                    x_cur = signal_processing.apply_wiener_filter_legacy(x_cur, noise_pow, mysize)
                    steps.append(_SignalStep("wiener", x_cur))
            return x_cur, steps

        u_filtered, u_f_steps = _apply_chain(u_detrended, u_chain, is_u=True)
        r_filtered, r_f_steps = _apply_chain(r_detrended, r_chain, is_u=False)
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
        nperseg = max(2, min(nperseg, len(u_detrended), len(r_detrended)))
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

        # Always include raw, detrended, windowed, and last filtered if present
        pick_names = ["raw", "detrended", "windowed"]
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
            io_input_name=str(io_input_name or ""),
            io_output_name=str(io_output_name or ""),
        )


@dataclass
class _CompareEntry:
    tab_ref: object
    tab_title: str
    direction: str  # "x" | "y"
    kind: str  # "measurement" | "average"
    index: int
    name: str


class FrfComparePage(QWidget):
    """
    Cross-tab comparison for FRF results.

    Allows overlaying compliance results (displacement/force) from any open FRF analysis tab.
    """

    def __init__(
        self,
        *,
        open_frf_tabs_provider: Callable[[], List[object]],
        entries_shared: Optional[List[_CompareEntry]] = None,
    ) -> None:
        super().__init__()
        self._open_frf_tabs_provider: Callable[[], List[object]] = open_frf_tabs_provider
        self._entries: List[_CompareEntry] = (
            entries_shared if entries_shared is not None else []
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title = QLabel("Compare")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        controls = QWidget()
        c_l = QHBoxLayout(controls)
        c_l.setContentsMargins(0, 0, 0, 0)
        c_l.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._cb_tab = QComboBox()
        self._cb_tab.setMinimumWidth(260)
        self._cb_dir = QComboBox()
        self._cb_dir.addItem("X", userData="x")
        self._cb_dir.addItem("Y", userData="y")
        self._cb_measurement = QComboBox()
        self._cb_measurement.setMinimumWidth(260)
        self._name_edit = QLineEdit("")
        self._name_edit.setPlaceholderText("Entry name (legend)")
        self._name_edit.setMinimumWidth(240)

        self._btn_refresh = QPushButton("Refresh")
        self._btn_add = QPushButton("Add")

        self._btn_refresh.clicked.connect(self._refresh_sources)
        self._cb_tab.currentIndexChanged.connect(self._refresh_measurements)
        self._cb_dir.currentIndexChanged.connect(self._refresh_measurements)
        self._cb_measurement.currentIndexChanged.connect(self._refresh_default_name)
        self._btn_add.clicked.connect(self._on_add)

        c_l.addWidget(QLabel("Tab"))
        c_l.addWidget(self._cb_tab)
        c_l.addSpacing(10)
        c_l.addWidget(QLabel("Dir"))
        c_l.addWidget(self._cb_dir)
        c_l.addSpacing(10)
        c_l.addWidget(QLabel("Measurement"))
        c_l.addWidget(self._cb_measurement)
        c_l.addSpacing(10)
        c_l.addWidget(QLabel("Name"))
        c_l.addWidget(self._name_edit, 1)
        c_l.addSpacing(10)
        c_l.addWidget(self._btn_add)
        c_l.addWidget(self._btn_refresh)

        layout.addWidget(controls)

        mid = QSplitter(Qt.Orientation.Horizontal)
        self._entry_list = QListWidget()
        self._entry_list.setMinimumWidth(280)
        self._entry_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._entry_list.customContextMenuRequested.connect(self._on_entry_context_menu)
        mid.addWidget(self._entry_list)

        self._tabs = QTabWidget()
        self._plot_bode = MatplotlibFigureWidget()
        self._plot_real = MatplotlibFigureWidget()
        self._tabs.addTab(self._plot_bode, "Bode (displacement/force)")
        self._tabs.addTab(self._plot_real, "Real(FRF) (displacement/force)")
        mid.addWidget(self._tabs)
        mid.setStretchFactor(0, 1)
        mid.setStretchFactor(1, 4)
        layout.addWidget(mid, 1)

        export_row = QWidget()
        e_l = QHBoxLayout(export_row)
        e_l.setContentsMargins(0, 0, 0, 0)
        e_l.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._btn_save_csv = QPushButton("Save CSV…")
        self._btn_save_png = QPushButton("Save PNG…")
        self._btn_save_csv.clicked.connect(self._on_save_csv)
        self._btn_save_png.clicked.connect(self._on_save_png)
        e_l.addWidget(self._btn_save_csv)
        e_l.addWidget(self._btn_save_png)
        layout.addWidget(export_row)

        self._refresh_sources()

    def set_open_frf_tabs_provider(self, provider: Callable[[], List[object]]) -> None:
        self._open_frf_tabs_provider = provider
        self._refresh_sources()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        # Keep source tabs fresh and mirror shared compare entries.
        self._refresh_sources()
        self._rebuild_entry_list()
        self._redraw()

    def _open_frf_tabs(self) -> List[object]:
        try:
            tabs = list(self._open_frf_tabs_provider() or [])
        except Exception:
            tabs = []
        return tabs

    @staticmethod
    def _tab_title(tab: object) -> str:
        t = getattr(tab, "title", None)
        if isinstance(t, str) and t.strip():
            return t.strip()
        # fallback
        return "Untitled"

    @staticmethod
    def _tab_results_page(tab: object) -> Optional[object]:
        # AnalysisTab will expose this for FRF tabs.
        fn = getattr(tab, "get_frf_results_page", None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                return None
        return None

    def _refresh_sources(self) -> None:
        tabs = self._open_frf_tabs()
        cur = self._cb_tab.currentData()
        self._cb_tab.blockSignals(True)
        try:
            self._cb_tab.clear()
            for i, tab in enumerate(tabs):
                self._cb_tab.addItem(self._tab_title(tab), userData=tab)
            # Try preserve selection
            if cur is not None:
                for i in range(self._cb_tab.count()):
                    if self._cb_tab.itemData(i) is cur:
                        self._cb_tab.setCurrentIndex(i)
                        break
        finally:
            self._cb_tab.blockSignals(False)
        self._refresh_measurements()

    def _refresh_measurements(self) -> None:
        tab = self._cb_tab.currentData()
        direction = str(self._cb_dir.currentData() or "x")
        rp = self._tab_results_page(tab) if tab is not None else None
        sources: List[Tuple[str, str, int]] = []
        if rp is not None:
            fn = getattr(rp, "list_compliance_sources", None)
            if callable(fn):
                try:
                    sources = list(fn(direction=direction) or [])
                except Exception:
                    sources = []

        self._cb_measurement.blockSignals(True)
        try:
            self._cb_measurement.clear()
            for lab, kind, idx in sources:
                self._cb_measurement.addItem(str(lab), userData=(str(kind), int(idx)))
        finally:
            self._cb_measurement.blockSignals(False)
        self._refresh_default_name()

    def _refresh_default_name(self) -> None:
        # Only set a default if user hasn't typed anything.
        if (self._name_edit.text() or "").strip():
            return
        tab = self._cb_tab.currentData()
        tab_title = self._tab_title(tab) if tab is not None else "Untitled"
        direction = str(self._cb_dir.currentData() or "x").upper()
        meas_lab = str(self._cb_measurement.currentText() or "Measurement")
        self._name_edit.setText(f"{tab_title} - {direction} - {meas_lab}")

    def _on_add(self) -> None:
        tab = self._cb_tab.currentData()
        if tab is None:
            QMessageBox.warning(self, "Compare", "No FRF analysis tab available.")
            return
        rp = self._tab_results_page(tab)
        if rp is None:
            QMessageBox.warning(self, "Compare", "Selected tab has no FRF results page.")
            return
        sel = self._cb_measurement.currentData()
        if not isinstance(sel, tuple) or len(sel) != 2:
            QMessageBox.warning(self, "Compare", "No measurement selected.")
            return
        kind, idx = sel
        direction = str(self._cb_dir.currentData() or "x")
        name = (self._name_edit.text() or "").strip()
        if not name:
            self._name_edit.setFocus()
            QMessageBox.warning(self, "Compare", "Please provide a name for the entry.")
            return
        ent = _CompareEntry(
            tab_ref=tab,
            tab_title=self._tab_title(tab),
            direction=direction,
            kind=str(kind),
            index=int(idx),
            name=name,
        )
        self._entries.append(ent)
        self._name_edit.setText("")  # encourage new name next add
        self._rebuild_entry_list()
        self._redraw()

    def _rebuild_entry_list(self) -> None:
        self._entry_list.clear()
        for ent in self._entries:
            lab = f"{ent.name}  [{ent.tab_title} / {ent.direction.upper()} / {ent.kind}{'' if ent.kind=='average' else f'#{ent.index+1}'}]"
            it = QListWidgetItem(lab)
            it.setData(Qt.ItemDataRole.UserRole, ent)
            self._entry_list.addItem(it)

    def _on_entry_context_menu(self, pos) -> None:
        it = self._entry_list.itemAt(pos)
        if it is None:
            return
        menu = QMenu(self)
        act_edit = menu.addAction("Edit name")
        act_rm = menu.addAction("Remove")
        chosen = menu.exec(self._entry_list.mapToGlobal(pos))
        if chosen == act_edit:
            ent = it.data(Qt.ItemDataRole.UserRole)
            if not isinstance(ent, _CompareEntry):
                return
            new_name, ok = QInputDialog.getText(
                self,
                "Edit compare entry name",
                "Name:",
                text=str(ent.name or ""),
            )
            if not ok:
                return
            new_name = (new_name or "").strip()
            if not new_name:
                QMessageBox.warning(self, "Compare", "Name cannot be empty.")
                return
            ent.name = new_name
            self._rebuild_entry_list()
            self._redraw()
            return
        if chosen == act_rm:
            ent = it.data(Qt.ItemDataRole.UserRole)
            self._entries[:] = [e for e in self._entries if e is not ent]
            self._rebuild_entry_list()
            self._redraw()

    @staticmethod
    def _interp_complex(f_src: np.ndarray, H_src: np.ndarray, f_tgt: np.ndarray) -> np.ndarray:
        f_src = np.asarray(f_src, dtype=float)
        H_src = np.asarray(H_src, dtype=complex)
        f_tgt = np.asarray(f_tgt, dtype=float)
        if len(f_src) == 0 or len(f_tgt) == 0:
            return np.asarray([], dtype=complex)
        r = np.interp(f_tgt, f_src, np.real(H_src))
        im = np.interp(f_tgt, f_src, np.imag(H_src))
        return r + 1j * im

    def _fetch_series(self, ent: _CompareEntry) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        rp = self._tab_results_page(ent.tab_ref)
        if rp is None:
            return None
        fn = getattr(rp, "get_compliance_series", None)
        if not callable(fn):
            return None
        try:
            res = fn(direction=ent.direction, kind=ent.kind, index=ent.index)
        except Exception:
            return None
        if res is None:
            return None
        f_hz, Hc = res
        f_hz = np.asarray(f_hz, dtype=float)
        Hc = np.asarray(Hc, dtype=complex)
        mask = np.isfinite(f_hz) & (f_hz > 0) & np.isfinite(np.real(Hc)) & np.isfinite(np.imag(Hc))
        return f_hz[mask], Hc[mask]

    def _redraw(self) -> None:
        self._plot_compare_bode()
        self._plot_compare_real()

    def _plot_compare_bode(self) -> None:
        fig = self._plot_bode.figure
        fig.clear()
        ax_mag = fig.add_subplot(2, 1, 1)
        ax_ph = fig.add_subplot(2, 1, 2, sharex=ax_mag)
        ax_mag.set_title("Bode (displacement/force)")
        ax_mag.set_ylabel("Magnitude |H| (m/N)")
        ax_ph.set_xlabel("Frequency (Hz)")
        ax_ph.set_ylabel("Phase (deg)")
        for ax in (ax_mag, ax_ph):
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xscale("log")
        ax_mag.set_yscale("log")

        series: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for ent in self._entries:
            s = self._fetch_series(ent)
            if s is None:
                continue
            f_hz, Hc = s
            series.append((ent.name, f_hz, Hc))

        if not series:
            ax_mag.text(0.5, 0.5, "No entries. Add measurements to compare.", ha="center", va="center", transform=ax_mag.transAxes)
            ax_mag.set_axis_off()
            ax_ph.set_axis_off()
            self._plot_bode.redraw()
            return

        f0 = series[0][1]
        for name, f_hz, Hc in series:
            H = Hc if np.allclose(f_hz, f0) else self._interp_complex(f_hz, Hc, f0)
            mag = np.abs(H)
            ph = np.unwrap(np.angle(H)) * 180.0 / np.pi
            ax_mag.loglog(f0, mag, linewidth=1.8, label=name)
            ax_ph.semilogx(f0, ph, linewidth=1.2, label=name)

        ax_mag.legend(loc="best", fontsize=8)
        ax_ph.legend(loc="best", fontsize=8)
        self._plot_bode.redraw()

    def _plot_compare_real(self) -> None:
        fig = self._plot_real.figure
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Real(FRF) (displacement/force)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Re{H} (m/N)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xscale("log")

        series: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for ent in self._entries:
            s = self._fetch_series(ent)
            if s is None:
                continue
            f_hz, Hc = s
            series.append((ent.name, f_hz, Hc))

        if not series:
            ax.text(0.5, 0.5, "No entries. Add measurements to compare.", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self._plot_real.redraw()
            return

        f0 = series[0][1]
        for name, f_hz, Hc in series:
            H = Hc if np.allclose(f_hz, f0) else self._interp_complex(f_hz, Hc, f0)
            ax.semilogx(f0, np.real(H), linewidth=1.6, label=name)
        ax.legend(loc="best")
        self._plot_real.redraw()

    def _on_save_png(self) -> None:
        tab_idx = int(self._tabs.currentIndex())
        widget = self._plot_bode if tab_idx == 0 else self._plot_real
        path, _ = QFileDialog.getSaveFileName(self, "Save plot as PNG", "", "PNG (*.png);;All files (*)")
        if not path:
            return
        try:
            widget.figure.savefig(path, dpi=200, bbox_inches="tight")
        except Exception as e:
            QMessageBox.critical(self, "Save PNG failed", str(e))

    @staticmethod
    def _sanitize_col(name: str) -> str:
        import re

        s = re.sub(r"[^a-zA-Z0-9_]+", "_", (name or "").strip())
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "series"

    def _on_save_csv(self) -> None:
        tab_idx = int(self._tabs.currentIndex())
        default_name = "compare_bode.csv" if tab_idx == 0 else "compare_real_frf.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save plot data as CSV", default_name, "CSV (*.csv);;All files (*)")
        if not path:
            return

        series: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for ent in self._entries:
            s = self._fetch_series(ent)
            if s is None:
                continue
            series.append((ent.name, s[0], s[1]))
        if not series:
            QMessageBox.warning(self, "Save CSV", "No data to export.")
            return

        f0 = series[0][1]
        try:
            with open(path, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                if tab_idx == 0:
                    # Bode: frequency + mag/phase per entry
                    header = ["frequency_hz"]
                    for name, _, _ in series:
                        base = self._sanitize_col(name)
                        header.extend([f"{base}_mag", f"{base}_phase_deg"])
                    w.writerow(header)
                    phases = []
                    mags = []
                    for _name, f_hz, Hc in series:
                        H = Hc if np.allclose(f_hz, f0) else self._interp_complex(f_hz, Hc, f0)
                        mags.append(np.abs(H))
                        phases.append(np.unwrap(np.angle(H)) * 180.0 / np.pi)
                    for i in range(len(f0)):
                        row: List[Any] = [float(f0[i])]
                        for mag, ph in zip(mags, phases):
                            row.append(float(mag[i]))
                            row.append(float(ph[i]))
                        w.writerow(row)
                else:
                    # Real(FRF): frequency + real per entry
                    header = ["frequency_hz"]
                    for name, _, _ in series:
                        base = self._sanitize_col(name)
                        header.append(f"{base}_real")
                    w.writerow(header)
                    for i in range(len(f0)):
                        row: List[Any] = [float(f0[i])]
                        for _name, f_hz, Hc in series:
                            H = Hc if np.allclose(f_hz, f0) else self._interp_complex(f_hz, Hc, f0)
                            row.append(float(np.real(H[i])))
                        w.writerow(row)
        except Exception as e:
            QMessageBox.critical(self, "Save CSV failed", str(e))


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

    u_impact, u_ringdown = _required_trend_estimation(filtering_cfg, "force")
    r_impact, r_ringdown = _required_trend_estimation(filtering_cfg, "response")
    u_mask = signal_processing.quiet_regions_mask(time, u_impact, u_ringdown)
    r_mask = signal_processing.quiet_regions_mask(time, r_impact, r_ringdown)
    u_detrended, _u_trend = signal_processing.detrend_quiet_regions(time, u, u_impact, u_ringdown)
    r_detrended, _r_trend = signal_processing.detrend_quiet_regions(time, resp, r_impact, r_ringdown)

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
                noise_mask = u_mask if is_u else r_mask
                noise_pow = signal_processing.calculate_noise_power_from_mask(x_cur, noise_mask)
                x_cur = signal_processing.apply_wiener_filter_legacy(x_cur, noise_pow, mysize)
        return x_cur

    u_filt = _apply_chain(u_detrended, u_chain, is_u=True)
    r_filt = _apply_chain(r_detrended, r_chain, is_u=False)

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


