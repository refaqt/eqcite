from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .pages.frf import FrfRootPage
from .pages.settings import SettingsRootPage
from .pages.stability import StabilityRootPage
from .persistence import (
    AppSettings,
    load_analysis_json,
    load_app_settings,
    resolve_default_load_dir,
    resolve_default_save_dir,
    save_analysis_folder,
    save_app_settings,
    suggest_analysis_folder_name,
)
from .state import AnalysisState


@dataclass
class ExperimentMeta:
    title: str = ""
    description: str = ""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CNC FRF Estimation")
        self.resize(1200, 800)

        self._repo_root = Path(__file__).resolve().parents[2]
        self._app_settings: AppSettings = load_app_settings(self._repo_root)
        self._restoring_settings = False
        self._state = AnalysisState(
            analysis_type="frf",
            sensitivities_path=self._repo_root / "config" / "sensitivities.json",
            experiment_cfg=self._load_default_experiment_cfg(),
        )

        self._meta = ExperimentMeta()

        self._tabs = QTabWidget()
        self._frf_root = FrfRootPage()
        self._stab_root = StabilityRootPage()
        self._settings_root = SettingsRootPage()
        self._tabs.addTab(self._frf_root, "FRF")
        self._tabs.addTab(self._stab_root, "Stability")
        self._tabs.addTab(self._settings_root, "Settings")

        # Bottom metadata fields (persisted later via AnalysisState)
        meta_panel = QWidget()
        meta_layout = QVBoxLayout(meta_panel)
        meta_layout.setContentsMargins(0, 0, 0, 0)

        title_row = QWidget()
        title_layout = QHBoxLayout(title_row)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.addWidget(QLabel("Title"))
        self._title_edit = QLineEdit()
        self._title_edit.setPlaceholderText("Experiment title")
        title_layout.addWidget(self._title_edit, 1)

        desc_row = QWidget()
        desc_layout = QHBoxLayout(desc_row)
        desc_layout.setContentsMargins(0, 0, 0, 0)
        desc_layout.addWidget(QLabel("Description"))
        self._desc_edit = QPlainTextEdit()
        self._desc_edit.setPlaceholderText("Experiment description")
        self._desc_edit.setFixedHeight(80)
        desc_layout.addWidget(self._desc_edit, 1)

        meta_layout.addWidget(title_row)
        meta_layout.addWidget(desc_row)

        # Main splitter: tabs on top, meta panel bottom
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(10, 10, 10, 10)
        central_layout.setSpacing(10)

        vertical_split = QSplitter(Qt.Orientation.Vertical)
        vertical_split.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        vertical_split.addWidget(self._tabs)
        vertical_split.addWidget(meta_panel)
        vertical_split.setStretchFactor(0, 5)
        vertical_split.setStretchFactor(1, 1)
        central_layout.addWidget(vertical_split)

        self.setCentralWidget(central)
        self._build_toolbar()

        self._title_edit.textChanged.connect(self._on_title_changed)
        self._desc_edit.textChanged.connect(self._on_desc_changed)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        self._settings_root.default_save_dir_changed.connect(self._on_default_save_dir_changed)
        self._settings_root.default_load_dir_changed.connect(self._on_default_load_dir_changed)
        self._settings_root.set_default_save_dir(str(self._app_settings.default_save_dir))
        self._settings_root.set_default_load_dir(str(self._app_settings.default_load_dir))
        self._frf_root.set_default_load_dir(str(resolve_default_load_dir(self._repo_root, self._app_settings)))
        self._stab_root.set_default_load_dir(str(resolve_default_load_dir(self._repo_root, self._app_settings)))

        self._frf_root.default_sensor_config_save_requested.connect(self._on_save_default_sensor_config)
        self._frf_root.default_filtering_save_requested.connect(self._on_save_default_frf_filtering)
        self._frf_root.file_paths_changed.connect(self._on_frf_file_paths_changed)
        self._stab_root.file_path_changed.connect(self._on_stability_file_path_changed)
        self._apply_default_sensor_config_if_present()
        self._apply_default_frf_filtering_if_present()
        self._restore_last_file_paths()

    def _restore_last_file_paths(self) -> None:
        self._restoring_settings = True
        try:
            self._frf_root.set_file_paths(
                self._app_settings.last_frf_x_file or "",
                self._app_settings.last_frf_y_file or "",
            )
            self._stab_root.set_file_path(self._app_settings.last_stability_file or "")
        finally:
            self._restoring_settings = False

    def _apply_default_sensor_config_if_present(self) -> None:
        if (
            not self._app_settings.default_channel_mapping
            and not self._app_settings.default_sensitivity_mapping
            and not self._app_settings.default_sensitivities
        ):
            return
        try:
            self._frf_root.apply_default_sensor_config(
                channel_mapping=self._app_settings.default_channel_mapping,
                sensitivity_mapping=self._app_settings.default_sensitivity_mapping,
                sensitivities=self._app_settings.default_sensitivities,
            )
        except Exception as e:
            QMessageBox.critical(self, "Failed to apply default sensor config", str(e))

    def _apply_default_frf_filtering_if_present(self) -> None:
        if not (self._app_settings.default_frf_filtering or {}):
            return
        try:
            self._frf_root.apply_default_filtering(dict(self._app_settings.default_frf_filtering or {}))
        except Exception as e:
            QMessageBox.critical(self, "Failed to apply default filtering", str(e))

    def _on_save_default_sensor_config(
        self,
        channel_mapping: dict,
        sensitivity_mapping: dict,
        sensitivities: dict,
    ) -> None:
        self._app_settings.default_channel_mapping = dict(channel_mapping or {})
        self._app_settings.default_sensitivity_mapping = dict(sensitivity_mapping or {})
        self._app_settings.default_sensitivities = dict(sensitivities or {})
        try:
            save_app_settings(self._repo_root, self._app_settings)
        except Exception as e:
            QMessageBox.critical(self, "Failed to save default sensor config", str(e))
            return
        QMessageBox.information(self, "Defaults saved", "Saved default sensor configuration.")

    def _on_save_default_frf_filtering(self, filtering_cfg: dict) -> None:
        self._app_settings.default_frf_filtering = dict(filtering_cfg or {})
        try:
            save_app_settings(self._repo_root, self._app_settings)
        except Exception as e:
            QMessageBox.critical(self, "Failed to save settings", str(e))
            return
        QMessageBox.information(self, "Defaults saved", "Saved default FRF filtering configuration.")

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        act_load = QAction("Load analysis", self)
        act_load.triggered.connect(self._on_load_analysis)
        tb.addAction(act_load)

        act_save = QAction("Save analysis", self)
        act_save.triggered.connect(self._on_save_analysis)
        tb.addAction(act_save)

        tb.addSeparator()

    def _on_load_analysis(self) -> None:
        start_dir = str(resolve_default_load_dir(self._repo_root, self._app_settings))
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load analysis",
            start_dir,
            "Analysis JSON (analysis.json);;JSON (*.json);;All files (*)",
        )
        if not path_str:
            return
        try:
            st = load_analysis_json(Path(path_str))
        except Exception as e:
            QMessageBox.critical(self, "Load analysis failed", str(e))
            return
        self._apply_state(st)

    def _on_save_analysis(self) -> None:
        self._sync_state_from_ui()
        default_dir = resolve_default_save_dir(self._repo_root, self._app_settings)
        default_dir.mkdir(parents=True, exist_ok=True)
        suggested_name = suggest_analysis_folder_name(self._state)

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select analysis folder (will create a new subfolder)",
            str(default_dir),
        )
        if not folder:
            return
        chosen = Path(folder) / suggested_name
        try:
            analysis_json = save_analysis_folder(self._repo_root, self._state, chosen)
        except Exception as e:
            QMessageBox.critical(self, "Save analysis failed", str(e))
            return
        # Export tab-specific artifacts into results/* so the saved analysis folder is self-contained.
        try:
            if self._tabs.currentWidget() is self._frf_root and self._state.analysis_folder is not None:
                self._frf_root.export_results(self._state.analysis_folder)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(self, "Analysis saved", f"Saved: {analysis_json}")

    def _on_title_changed(self, txt: str) -> None:
        self._meta.title = txt

    def _on_desc_changed(self) -> None:
        self._meta.description = self._desc_edit.toPlainText()

    def _on_tab_changed(self, idx: int) -> None:
        self._state.selected_top_tab = int(idx)

    def _on_default_save_dir_changed(self, path: str) -> None:
        self._app_settings.default_save_dir = Path(path) if path else Path("analyses")
        try:
            save_app_settings(self._repo_root, self._app_settings)
        except Exception as e:
            QMessageBox.critical(self, "Failed to save settings", str(e))

    def _on_default_load_dir_changed(self, path: str) -> None:
        self._app_settings.default_load_dir = Path(path) if path else Path(".")
        self._frf_root.set_default_load_dir(str(resolve_default_load_dir(self._repo_root, self._app_settings)))
        self._stab_root.set_default_load_dir(str(resolve_default_load_dir(self._repo_root, self._app_settings)))
        try:
            save_app_settings(self._repo_root, self._app_settings)
        except Exception as e:
            QMessageBox.critical(self, "Failed to save settings", str(e))

    def _on_frf_file_paths_changed(self, x_path: str, y_path: str) -> None:
        if self._restoring_settings:
            return
        x_path = (x_path or "").strip()
        y_path = (y_path or "").strip()
        if (
            x_path == (self._app_settings.last_frf_x_file or "")
            and y_path == (self._app_settings.last_frf_y_file or "")
        ):
            return
        self._app_settings.last_frf_x_file = x_path
        self._app_settings.last_frf_y_file = y_path
        try:
            save_app_settings(self._repo_root, self._app_settings)
        except Exception as e:
            QMessageBox.critical(self, "Failed to save settings", str(e))

    def _on_stability_file_path_changed(self, path: str) -> None:
        if self._restoring_settings:
            return
        path = (path or "").strip()
        if path == (self._app_settings.last_stability_file or ""):
            return
        self._app_settings.last_stability_file = path
        try:
            save_app_settings(self._repo_root, self._app_settings)
        except Exception as e:
            QMessageBox.critical(self, "Failed to save settings", str(e))

    def _load_default_experiment_cfg(self) -> dict:
        default_cfg_path = self._repo_root / "config" / "config_before_cabinet_assembly.json"
        try:
            return json.loads(default_cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _sync_state_from_ui(self) -> None:
        self._state.experiment_title = self._title_edit.text()
        self._state.experiment_description = self._desc_edit.toPlainText()
        # Keep analysis_type in sync with top-level tab.
        if self._tabs.currentWidget() is self._stab_root:
            self._state.analysis_type = "stability_measurement"
        elif self._tabs.currentWidget() is self._frf_root:
            self._state.analysis_type = "frf"
            self._state.experiment_cfg = self._frf_root.build_experiment_cfg()

    def _apply_state(self, st: AnalysisState) -> None:
        self._state = st
        self._title_edit.setText(st.experiment_title or "")
        self._desc_edit.setPlainText(st.experiment_description or "")
        # Switch to correct top tab (FRF vs Stability)
        if st.analysis_type == "stability_measurement":
            self._tabs.setCurrentWidget(self._stab_root)
        else:
            self._tabs.setCurrentWidget(self._frf_root)
            try:
                self._frf_root.apply_experiment_cfg(dict(st.experiment_cfg or {}))
            except Exception as e:
                QMessageBox.critical(self, "Failed to apply FRF state", str(e))

