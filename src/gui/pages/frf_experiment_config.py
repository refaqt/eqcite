"""Experiment configuration tab (Mapping, Sensors, Expressions, Input/Output)."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ... import experiment_config
from .frf_model import FrfInputModel


def _new_channel_combo(selected: int = 0) -> QComboBox:
    cb = QComboBox()
    for i in range(8):
        cb.addItem(str(i), userData=i)
    for idx in range(cb.count()):
        if cb.itemData(idx) == selected:
            cb.setCurrentIndex(idx)
            break
    return cb


def _new_unit_combo(unit: str = "m/s²") -> QComboBox:
    cb = QComboBox()
    for u in ("m/s²", "N"):
        cb.addItem(u, userData=u)
    for idx in range(cb.count()):
        if cb.itemData(idx) == unit:
            cb.setCurrentIndex(idx)
            break
    return cb


def _sensor_combo(ids: List[str], current: str = "") -> QComboBox:
    cb = QComboBox()
    for sid in ids:
        cb.addItem(sid, userData=sid)
    if current:
        for idx in range(cb.count()):
            if cb.itemData(idx) == current:
                cb.setCurrentIndex(idx)
                break
    return cb


class FrfSensorConfigPage(QWidget):
    default_save_requested = Signal(dict)

    def __init__(self, model: FrfInputModel) -> None:
        super().__init__()
        self._model = model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title = QLabel("Experiment configuration")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, 1)

        # --- Mapping ---
        self._mapping_table = QTableWidget(0, 3)
        self._mapping_table.setHorizontalHeaderLabels(["Sensor name", "Channel no.", "Sensor (catalog id)"])
        self._mapping_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._mapping_table.customContextMenuRequested.connect(self._mapping_context_menu)
        self._mapping_table.itemChanged.connect(self._on_mapping_changed)
        self._tabs.addTab(self._mapping_table, "Mapping")

        # --- Sensors catalog ---
        self._catalog_table = QTableWidget(0, 5)
        self._catalog_table.setHorizontalHeaderLabels(
            ["Sensor id", "Model", "Manufacturer", "Sensitivity (per V)", "Output unit"]
        )
        self._catalog_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._catalog_table.customContextMenuRequested.connect(self._catalog_context_menu)
        self._catalog_table.itemChanged.connect(self._on_catalog_changed)
        self._tabs.addTab(self._catalog_table, "Sensors")

        # --- Expressions ---
        self._expr_table = QTableWidget(0, 2)
        self._expr_table.setHorizontalHeaderLabels(["Variable name", "Expression"])
        self._expr_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._expr_table.customContextMenuRequested.connect(self._expr_context_menu)
        self._expr_table.itemChanged.connect(self._on_expr_changed)
        self._tabs.addTab(self._expr_table, "Expressions")

        # --- Input / Output ---
        io_widget = QWidget()
        io_lay = QVBoxLayout(io_widget)
        io_lay.setContentsMargins(8, 8, 8, 8)
        x_box = QWidget()
        xf = QFormLayout(x_box)
        xf.addRow(QLabel("X direction"), QLabel(""))
        self._io_x_in = QComboBox()
        self._io_x_out = QComboBox()
        xf.addRow(QLabel("Input (force)"), self._io_x_in)
        xf.addRow(QLabel("Output (acceleration)"), self._io_x_out)
        io_lay.addWidget(x_box)
        y_box = QWidget()
        yf = QFormLayout(y_box)
        yf.addRow(QLabel("Y direction"), QLabel(""))
        self._io_y_in = QComboBox()
        self._io_y_out = QComboBox()
        yf.addRow(QLabel("Input (force)"), self._io_y_in)
        yf.addRow(QLabel("Output (acceleration)"), self._io_y_out)
        io_lay.addWidget(y_box)
        io_lay.addStretch(1)
        self._tabs.addTab(io_widget, "Input/Output")

        for cb in (self._io_x_in, self._io_x_out, self._io_y_in, self._io_y_out):
            cb.currentIndexChanged.connect(self._on_io_changed)

        self._btn_save_default = QPushButton("Save as default")
        self._btn_save_default.clicked.connect(self._emit_save_default)
        layout.addWidget(self._btn_save_default)

        self._reload_all_from_model()

    def apply_experiment_config(self, cfg: Dict[str, Any]) -> None:
        cfg = dict(cfg or {})
        if cfg.get("sensor_catalog") is not None:
            self._model.sensor_catalog = list(cfg.get("sensor_catalog") or [])
        if cfg.get("mapping") is not None:
            self._model.mapping_rows = list(cfg.get("mapping") or [])
        if cfg.get("expressions") is not None:
            self._model.expressions = list(cfg.get("expressions") or [])
        if cfg.get("analysis_io") is not None:
            self._model.analysis_io = dict(cfg.get("analysis_io") or {})
        if cfg.get("simulation_signal_map") is not None:
            self._model.simulation_signal_map = dict(cfg.get("simulation_signal_map") or {})
        if cfg.get("channel_mapping") is not None:
            self._model.channel_mapping = dict(cfg.get("channel_mapping") or {})
        if cfg.get("sensitivity_mapping") is not None:
            self._model.sensitivity_mapping = dict(cfg.get("sensitivity_mapping") or {})
        self._reload_all_from_model()

    def _sensor_ids(self) -> List[str]:
        out: List[str] = []
        for r in range(self._catalog_table.rowCount()):
            it = self._catalog_table.item(r, 0)
            if it is None:
                continue
            s = (it.text() or "").strip()
            if s:
                out.append(s)
        return out

    def _snapshot_cfg(self) -> Dict[str, Any]:
        self._sync_model_from_widgets()
        return {
            "sensor_catalog": list(self._model.sensor_catalog or []),
            "mapping": list(self._model.mapping_rows or []),
            "expressions": list(self._model.expressions or []),
            "analysis_io": dict(self._model.analysis_io or {}),
            "simulation_signal_map": dict(self._model.simulation_signal_map or {}),
        }

    def _force_and_accel_names(self) -> Tuple[List[str], List[str]]:
        try:
            kinds = experiment_config.build_variable_kinds(self._snapshot_cfg())
        except Exception:
            return [], []
        forces = sorted([n for n, k in kinds.items() if k == "force"])
        accels = sorted([n for n, k in kinds.items() if k == "acceleration"])
        return forces, accels

    def _reload_io_combos(self) -> None:
        forces, accels = self._force_and_accel_names()
        io = dict(self._model.analysis_io or {})
        xb, yb = io.get("x") or {}, io.get("y") or {}
        cur_x_in = str(xb.get("input") or "")
        cur_x_out = str(xb.get("output") or "")
        cur_y_in = str(yb.get("input") or "")
        cur_y_out = str(yb.get("output") or "")

        def refill(cb: QComboBox, names: List[str], cur: str) -> None:
            cb.blockSignals(True)
            try:
                cb.clear()
                for n in names:
                    cb.addItem(n, userData=n)
                cb.addItem("(none)", userData="")
                for idx in range(cb.count()):
                    if str(cb.itemData(idx) or "") == cur:
                        cb.setCurrentIndex(idx)
                        return
                cb.setCurrentIndex(max(0, cb.count() - 1))
            finally:
                cb.blockSignals(False)

        refill(self._io_x_in, forces, cur_x_in)
        refill(self._io_x_out, accels, cur_x_out)
        refill(self._io_y_in, forces, cur_y_in)
        refill(self._io_y_out, accels, cur_y_out)

    def _reload_all_from_model(self) -> None:
        self._reload_catalog_from_model()
        self._reload_mapping_from_model()
        self._reload_expr_from_model()
        self._reload_io_combos()

    def _reload_catalog_from_model(self) -> None:
        self._catalog_table.blockSignals(True)
        try:
            self._catalog_table.setRowCount(0)
            for row in self._model.sensor_catalog or []:
                if not isinstance(row, dict):
                    continue
                r = self._catalog_table.rowCount()
                self._catalog_table.insertRow(r)
                sid = str(row.get("id") or "")
                self._catalog_table.setItem(r, 0, QTableWidgetItem(sid))
                self._catalog_table.setItem(r, 1, QTableWidgetItem(str(row.get("model") or "")))
                self._catalog_table.setItem(r, 2, QTableWidgetItem(str(row.get("manufacturer") or "")))
                self._catalog_table.setItem(r, 3, QTableWidgetItem(str(row.get("sensitivity_vs_per_v") or "")))
                u = str(row.get("unit") or "m/s²")
                self._catalog_table.setCellWidget(r, 4, _new_unit_combo(u))
        finally:
            self._catalog_table.blockSignals(False)

    def _reload_mapping_from_model(self) -> None:
        self._mapping_table.blockSignals(True)
        try:
            self._mapping_table.setRowCount(0)
            ids = self._sensor_ids()
            for row in self._model.mapping_rows or []:
                if not isinstance(row, dict):
                    continue
                r = self._mapping_table.rowCount()
                self._mapping_table.insertRow(r)
                name = str(row.get("name") or "")
                ch = int(row.get("channel", 0))
                sid = str(row.get("sensor_id") or "")
                self._mapping_table.setItem(r, 0, QTableWidgetItem(name))
                self._mapping_table.setCellWidget(r, 1, _new_channel_combo(ch))
                self._mapping_table.setCellWidget(r, 2, _sensor_combo(ids, sid))
        finally:
            self._mapping_table.blockSignals(False)

    def _reload_expr_from_model(self) -> None:
        self._expr_table.blockSignals(True)
        try:
            self._expr_table.setRowCount(0)
            for row in self._model.expressions or []:
                if not isinstance(row, dict):
                    continue
                r = self._expr_table.rowCount()
                self._expr_table.insertRow(r)
                self._expr_table.setItem(r, 0, QTableWidgetItem(str(row.get("name") or "")))
                self._expr_table.setItem(r, 1, QTableWidgetItem(str(row.get("expr") or "")))
        finally:
            self._expr_table.blockSignals(False)

    def _sync_model_from_widgets(self) -> None:
        # Catalog
        catalog: List[Dict[str, Any]] = []
        for r in range(self._catalog_table.rowCount()):
            id_it = self._catalog_table.item(r, 0)
            if id_it is None:
                continue
            sid = (id_it.text() or "").strip()
            if not sid:
                continue
            m_it = self._catalog_table.item(r, 1)
            mf_it = self._catalog_table.item(r, 2)
            s_it = self._catalog_table.item(r, 3)
            u_w = self._catalog_table.cellWidget(r, 4)
            unit = "m/s²"
            if isinstance(u_w, QComboBox):
                unit = str(u_w.currentData() or "m/s²")
            sens_txt = (s_it.text() or "").strip() if s_it is not None else "1"
            try:
                sens = float(sens_txt)
            except ValueError:
                sens = 1.0
            catalog.append(
                {
                    "id": sid,
                    "model": (m_it.text() or "").strip() if m_it else "",
                    "manufacturer": (mf_it.text() or "").strip() if mf_it else "",
                    "sensitivity_vs_per_v": sens,
                    "unit": experiment_config.normalize_output_unit(unit),
                }
            )
        self._model.sensor_catalog = catalog

        # Remap mapping combos after catalog change
        ids = [c["id"] for c in catalog]
        mapping: List[Dict[str, Any]] = []
        for r in range(self._mapping_table.rowCount()):
            n_it = self._mapping_table.item(r, 0)
            if n_it is None:
                continue
            name = (n_it.text() or "").strip()
            if not name:
                continue
            ch_w = self._mapping_table.cellWidget(r, 1)
            sid_w = self._mapping_table.cellWidget(r, 2)
            ch = 0
            if isinstance(ch_w, QComboBox):
                ch = int(ch_w.currentData())
            sid = ""
            if isinstance(sid_w, QComboBox):
                sid = str(sid_w.currentData() or "")
            if sid not in ids:
                sid = ids[0] if ids else ""
            mapping.append({"name": name, "channel": ch, "sensor_id": sid})
        self._model.mapping_rows = mapping

        expressions: List[Dict[str, Any]] = []
        for r in range(self._expr_table.rowCount()):
            a = self._expr_table.item(r, 0)
            b = self._expr_table.item(r, 1)
            if a is None:
                continue
            vn = (a.text() or "").strip()
            ex = (b.text() or "").strip() if b is not None else ""
            if vn:
                expressions.append({"name": vn, "expr": ex})
        self._model.expressions = expressions

        def _io_pair(in_cb: QComboBox, out_cb: QComboBox) -> Dict[str, str]:
            ui = str(in_cb.currentData() or "").strip()
            uo = str(out_cb.currentData() or "").strip()
            if not ui or not uo:
                return {}
            return {"input": ui, "output": uo}

        self._model.analysis_io = {
            "x": _io_pair(self._io_x_in, self._io_x_out),
            "y": _io_pair(self._io_y_in, self._io_y_out),
        }

    def _on_mapping_changed(self) -> None:
        self._sync_model_from_widgets()
        self._reload_io_combos()

    def _on_catalog_changed(self) -> None:
        self._sync_model_from_widgets()
        ids = self._sensor_ids()
        for r in range(self._mapping_table.rowCount()):
            w = self._mapping_table.cellWidget(r, 2)
            cur = ""
            if isinstance(w, QComboBox):
                cur = str(w.currentData() or "")
            self._mapping_table.removeCellWidget(r, 2)
            pick = cur if cur in ids else (ids[0] if ids else "")
            self._mapping_table.setCellWidget(r, 2, _sensor_combo(ids, pick))
        self._reload_io_combos()

    def _on_expr_changed(self) -> None:
        self._sync_model_from_widgets()
        self._reload_io_combos()

    def _on_io_changed(self) -> None:
        self._sync_model_from_widgets()

    def _mapping_context_menu(self, pos) -> None:
        menu = QMenu(self)
        act_add = menu.addAction("Add row")
        act_del = menu.addAction("Delete selected row(s)")
        act = menu.exec(self._mapping_table.mapToGlobal(pos))
        if act == act_add:
            self._mapping_table.blockSignals(True)
            try:
                r = self._mapping_table.rowCount()
                self._mapping_table.insertRow(r)
                self._mapping_table.setItem(r, 0, QTableWidgetItem(""))
                ids = self._sensor_ids()
                self._mapping_table.setCellWidget(r, 1, _new_channel_combo(0))
                self._mapping_table.setCellWidget(r, 2, _sensor_combo(ids, ids[0] if ids else ""))
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
            self._on_mapping_changed()

    def _catalog_context_menu(self, pos) -> None:
        menu = QMenu(self)
        act_add = menu.addAction("Add sensor")
        act_del = menu.addAction("Delete selected row(s)")
        act = menu.exec(self._catalog_table.mapToGlobal(pos))
        if act == act_add:
            self._catalog_table.blockSignals(True)
            try:
                r = self._catalog_table.rowCount()
                self._catalog_table.insertRow(r)
                sid = f"sens_{uuid.uuid4().hex[:8]}"
                self._catalog_table.setItem(r, 0, QTableWidgetItem(sid))
                self._catalog_table.setItem(r, 1, QTableWidgetItem(""))
                self._catalog_table.setItem(r, 2, QTableWidgetItem(""))
                self._catalog_table.setItem(r, 3, QTableWidgetItem("1.0"))
                self._catalog_table.setCellWidget(r, 4, _new_unit_combo("m/s²"))
            finally:
                self._catalog_table.blockSignals(False)
            self._on_catalog_changed()
        elif act == act_del:
            rows = sorted({i.row() for i in self._catalog_table.selectedIndexes()}, reverse=True)
            if not rows:
                return
            self._catalog_table.blockSignals(True)
            try:
                for r in rows:
                    self._catalog_table.removeRow(r)
            finally:
                self._catalog_table.blockSignals(False)
            self._on_catalog_changed()

    def _expr_context_menu(self, pos) -> None:
        menu = QMenu(self)
        act_add = menu.addAction("Add expression")
        act_del = menu.addAction("Delete selected row(s)")
        act = menu.exec(self._expr_table.mapToGlobal(pos))
        if act == act_add:
            self._expr_table.blockSignals(True)
            try:
                r = self._expr_table.rowCount()
                self._expr_table.insertRow(r)
                self._expr_table.setItem(r, 0, QTableWidgetItem(""))
                self._expr_table.setItem(r, 1, QTableWidgetItem(""))
            finally:
                self._expr_table.blockSignals(False)
        elif act == act_del:
            rows = sorted({i.row() for i in self._expr_table.selectedIndexes()}, reverse=True)
            if not rows:
                return
            self._expr_table.blockSignals(True)
            try:
                for r in rows:
                    self._expr_table.removeRow(r)
            finally:
                self._expr_table.blockSignals(False)
            self._on_expr_changed()

    def _emit_save_default(self) -> None:
        self._sync_model_from_widgets()
        for row in self._model.mapping_rows or []:
            experiment_config.assert_valid_mapping_name(str(row.get("name") or ""))
        for row in self._model.expressions or []:
            vn = str(row.get("name") or "").strip()
            if vn:
                experiment_config.assert_valid_mapping_name(vn)
        snap = self._snapshot_cfg()
        try:
            experiment_config.validate_analysis_io(snap)
        except Exception as e:
            QMessageBox.warning(self, "Experiment configuration", str(e))
            return
        self.default_save_requested.emit(snap)
