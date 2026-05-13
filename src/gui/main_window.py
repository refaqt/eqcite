from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
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

        self._tabs = QTabWidget()
        self._frf_root = FrfRootPage()
        self._stab_root = StabilityRootPage()
        self._settings_root = SettingsRootPage()
        self._tabs.addTab(self._frf_root, "FRF")
        self._tabs.addTab(self._stab_root, "Stability")
        self._tabs.addTab(self._settings_root, "Settings")

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(10, 10, 10, 10)
        central_layout.setSpacing(10)
        central_layout.addWidget(self._tabs, 1)

        self.setCentralWidget(central)
        self._build_toolbar()

        self._tabs.currentChanged.connect(self._on_tab_changed)

        self._settings_root.default_save_dir_changed.connect(self._on_default_save_dir_changed)
        self._settings_root.default_load_dir_changed.connect(self._on_default_load_dir_changed)
        self._settings_root.set_default_save_dir(str(self._app_settings.default_save_dir))
        self._settings_root.set_default_load_dir(str(self._app_settings.default_load_dir))
        self._frf_root.set_default_load_dir(str(resolve_default_load_dir(self._repo_root, self._app_settings)))
        self._stab_root.set_default_load_dir(str(resolve_default_load_dir(self._repo_root, self._app_settings)))

        self._frf_root.default_experiment_config_save_requested.connect(self._on_save_default_experiment_config)
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
                list(self._app_settings.last_frf_x_files or []),
                list(self._app_settings.last_frf_y_files or []),
            )
            self._stab_root.set_file_path(self._app_settings.last_stability_file or "")
        finally:
            self._restoring_settings = False

    def _apply_default_sensor_config_if_present(self) -> None:
        if self._app_settings.default_experiment_config:
            try:
                self._frf_root.apply_default_experiment_config(dict(self._app_settings.default_experiment_config or {}))
            except Exception as e:
                QMessageBox.critical(self, "Failed to apply default experiment config", str(e))
            return
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

    def _on_save_default_experiment_config(self, cfg: dict) -> None:
        self._app_settings.default_experiment_config = dict(cfg or {})
        self._app_settings.default_channel_mapping = {}
        self._app_settings.default_sensitivity_mapping = {}
        self._app_settings.default_sensitivities = {}
        try:
            save_app_settings(self._repo_root, self._app_settings)
        except Exception as e:
            QMessageBox.critical(self, "Failed to save default experiment config", str(e))
            return
        QMessageBox.information(self, "Defaults saved", "Saved default experiment configuration.")

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

    def _on_frf_file_paths_changed(self, x_paths: object, y_paths: object) -> None:
        if self._restoring_settings:
            return

        def _coerce(v: object) -> list[str]:
            if isinstance(v, list):
                out = []
                for x in v:
                    if x is None:
                        continue
                    s = str(x).strip()
                    if s:
                        out.append(s)
                return out
            if v is None:
                return []
            s = str(v).strip()
            return [s] if s else []

        x_list = _coerce(x_paths)
        y_list = _coerce(y_paths)

        if x_list == (self._app_settings.last_frf_x_files or []) and y_list == (self._app_settings.last_frf_y_files or []):
            return

        self._app_settings.last_frf_x_files = x_list
        self._app_settings.last_frf_y_files = y_list
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
        # Keep analysis_type in sync with top-level tab.
        if self._tabs.currentWidget() is self._stab_root:
            self._state.analysis_type = "stability_measurement"
        elif self._tabs.currentWidget() is self._frf_root:
            self._state.analysis_type = "frf"
            self._state.experiment_cfg = self._frf_root.build_experiment_cfg()

    def _apply_state(self, st: AnalysisState) -> None:
        self._state = st
        # Switch to correct top tab (FRF vs Stability)
        if st.analysis_type == "stability_measurement":
            self._tabs.setCurrentWidget(self._stab_root)
        else:
            self._tabs.setCurrentWidget(self._frf_root)
            try:
                self._frf_root.apply_experiment_cfg(dict(st.experiment_cfg or {}))
            except Exception as e:
                QMessageBox.critical(self, "Failed to apply FRF state", str(e))
