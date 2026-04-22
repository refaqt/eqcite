from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
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
    QVBoxLayout,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from ... import frf_estimation, load_data
from ..plot_widgets import MatplotlibFigureWidget


@dataclass
class FrfInputModel:
    data_source: str = "file"  # "file" | "simulation"
    file_path_x: str = ""
    file_path_y: str = ""
    simulation: Dict[str, Any] = None  # type: ignore[assignment]
    channel_mapping: Dict[str, int] = None  # type: ignore[assignment]
    sensitivity_mapping: Dict[str, str] = None  # type: ignore[assignment]


@dataclass
class _DirectionalFrfResult:
    direction: str  # "x" | "y"
    compliance: Tuple[np.ndarray, np.ndarray]  # (f_hz, H_disp_over_force complex)
    accelerance: Tuple[np.ndarray, np.ndarray]  # (f_hz, H_accel_over_force complex)


class FrfRootPage(QWidget):
    default_sensor_config_save_requested = Signal(dict, dict, dict)

    def __init__(self) -> None:
        super().__init__()

        self._menu = QListWidget()
        self._stack = QStackedWidget()

        repo_root = Path(__file__).resolve().parents[3]
        cfg_path = repo_root / "config" / "config_before_cabinet_assembly.json"
        base = json_load(cfg_path) if cfg_path.exists() else {}
        self._input_model = FrfInputModel(
            simulation={},
            channel_mapping=dict(base.get("channel_mapping", {}) or {}),
            sensitivity_mapping=dict(base.get("sensitivity_mapping", {}) or {}),
        )

        self._page_input = FrfInputPage(self._input_model)
        self._page_sensor = FrfSensorConfigPage(self._input_model)
        self._page_results = FrfResultsPage(self._input_model)

        self._page_sensor.default_save_requested.connect(self.default_sensor_config_save_requested)

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

    @staticmethod
    def _wrap(widget: QWidget) -> QWidget:
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(widget)
        lay.addStretch(1)
        return container


class FrfInputPage(QWidget):
    def __init__(self, model: FrfInputModel) -> None:
        super().__init__()
        self._model = model
        self._default_load_dir = ""

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

        self._x_path = QLineEdit()
        self._x_browse = QPushButton("Browse…")
        self._x_browse.clicked.connect(lambda: self._browse_into(self._x_path))
        x_row = QWidget()
        x_l = QVBoxLayout(x_row)
        x_l.setContentsMargins(0, 0, 0, 0)
        x_l.addWidget(self._x_path)
        x_l.addWidget(self._x_browse)
        file_form.addRow("X file", x_row)

        self._y_path = QLineEdit()
        self._y_browse = QPushButton("Browse…")
        self._y_browse.clicked.connect(lambda: self._browse_into(self._y_path))
        y_row = QWidget()
        y_l = QVBoxLayout(y_row)
        y_l.setContentsMargins(0, 0, 0, 0)
        y_l.addWidget(self._y_path)
        y_l.addWidget(self._y_browse)
        file_form.addRow("Y file", y_row)

        layout.addWidget(self._file_box)

        # Simulation form (minimal; will be expanded)
        self._sim_box = QWidget()
        sim_form = QFormLayout(self._sim_box)
        sim_form.setContentsMargins(0, 0, 0, 0)

        self._sim_mass = QLineEdit("8.0")
        self._sim_stiff = QLineEdit("1e6")
        self._sim_damp = QLineEdit("0.0")
        self._sim_fs = QLineEdit("8000")
        self._sim_T = QLineEdit("0.2")
        self._sim_hammer_force = QLineEdit("50.0")
        self._sim_hammer_bw = QLineEdit("1000.0")

        sim_form.addRow("Mass (kg)", self._sim_mass)
        sim_form.addRow("Stiffness (N/m)", self._sim_stiff)
        sim_form.addRow("Damping", self._sim_damp)
        sim_form.addRow("fs (Hz)", self._sim_fs)
        sim_form.addRow("T (s)", self._sim_T)
        sim_form.addRow("Hammer force (N)", self._sim_hammer_force)
        sim_form.addRow("Hammer bandwidth (Hz)", self._sim_hammer_bw)

        layout.addWidget(self._sim_box)
        layout.addStretch(1)

        self._sim_box.setEnabled(False)
        self._on_mode_changed()

    def set_default_load_dir(self, path: str) -> None:
        self._default_load_dir = path or ""

    def _browse_into(self, line_edit: QLineEdit) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV file",
            self._default_load_dir,
            "CSV (*.csv);;All files (*)",
        )
        if not p:
            return
        line_edit.setText(p)
        self._sync_model_from_ui()

    def _on_mode_changed(self) -> None:
        self._file_box.setEnabled(self._rb_file.isChecked())
        self._sim_box.setEnabled(self._rb_sim.isChecked())
        self._sync_model_from_ui()

    def _sync_model_from_ui(self) -> None:
        self._model.data_source = "simulation" if self._rb_sim.isChecked() else "file"
        self._model.file_path_x = self._x_path.text().strip()
        self._model.file_path_y = self._y_path.text().strip()
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


class FrfResultsPage(QWidget):
    def __init__(self, model: FrfInputModel) -> None:
        super().__init__()
        self._model = model
        self._last_x: Optional[_DirectionalFrfResult] = None
        self._last_y: Optional[_DirectionalFrfResult] = None
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
        self._tab_bode_disp = MatplotlibFigureWidget()
        self._tab_bode_accel = MatplotlibFigureWidget()
        self._tabs.addTab(self._tab_bode_disp, "Bode (displacement/force)")
        self._tabs.addTab(self._tab_bode_accel, "Bode (acceleration/force)")
        layout.addWidget(self._tabs, 1)

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
            base["file_paths_x"] = [self._model.file_path_x] if self._model.file_path_x else []
            base["file_paths_y"] = [self._model.file_path_y] if self._model.file_path_y else []
        return base

    def _run(self) -> None:
        try:
            exp_cfg = self._build_experiment_cfg()
            repo_root = Path(__file__).resolve().parents[3]
            sens = load_data.load_sensitivities(repo_root / "config" / "sensitivities.json")
            directional = load_data.load_directional_data(exp_cfg, sens)

            # Compute x and y FRFs if available; then plot combined where possible.
            x_res = self._compute_directional(directional.get("x"), direction="x")
            y_res = self._compute_directional(directional.get("y"), direction="y")

            self._last_x = (
                _DirectionalFrfResult(direction="x", compliance=x_res[0], accelerance=x_res[1]) if x_res else None
            )
            self._last_y = (
                _DirectionalFrfResult(direction="y", compliance=y_res[0], accelerance=y_res[1]) if y_res else None
            )
            self._last_run_meta = {
                "analysis_type": "frf",
                "data_source": self._model.data_source,
                "file_path_x": self._model.file_path_x,
                "file_path_y": self._model.file_path_y,
                "simulation": dict(self._model.simulation or {}),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

            self._plot_bode(
                self._tab_bode_disp,
                x_res[0] if x_res else None,
                y_res[0] if y_res else None,
                y_label="Compliance (m/N)",
                title="Displacement/Force FRF",
            )
            self._plot_bode(
                self._tab_bode_accel,
                x_res[1] if x_res else None,
                y_res[1] if y_res else None,
                y_label="Accelerance ((m/s²)/N)",
                title="Acceleration/Force FRF",
            )
        except Exception as e:
            QMessageBox.critical(self, "FRF run failed", str(e))

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
    ) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        if data is None:
            return None

        # Heuristic: prefer configured names if present, else fall back.
        # For config-driven pipeline, expected: F_x/F_y and x_tool/y_tool exist when sensors available.
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

        f, H_accel = frf_estimation.compute_frf_from_signals(time, u, y_accel, frf_estimation.get_system_parameters())
        H_disp = accel_to_disp_over_force(f, H_accel)
        return (f, H_disp), (f, H_accel)

    @staticmethod
    def _plot_bode(
        widget: MatplotlibFigureWidget,
        x: Optional[Tuple[np.ndarray, np.ndarray]],
        y: Optional[Tuple[np.ndarray, np.ndarray]],
        y_label: str,
        title: str,
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

        ax_mag.set_title(title)
        ax_mag.set_ylabel(y_label)
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_mag.legend()

        ax_ph.set_xlabel("Frequency (Hz)")
        ax_ph.set_ylabel("Phase (deg)")
        ax_ph.grid(True, which="both", alpha=0.3)
        ax_ph.legend()

        widget.redraw()


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

