from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .analysis_workspace import AnalysisTab, AnalysisWorkspace
from .pages.settings import SettingsRootPage
from .persistence import (
    AppSettings,
    load_analysis_json,
    load_app_settings,
    resolve_default_load_dir,
    resolve_default_save_dir,
    save_analysis_folder,
    save_app_settings,
)
from .state import AnalysisState, AnalysisType


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CNC FRF Estimation")
        self.resize(1200, 800)

        self._repo_root = Path(__file__).resolve().parents[2]
        self._app_settings: AppSettings = load_app_settings(self._repo_root)
        self._restoring_settings = False

        self._tabs = QTabWidget()
        self._frf_workspace = AnalysisWorkspace("frf")
        self._stab_workspace = AnalysisWorkspace("stability_measurement")
        self._settings_root = SettingsRootPage()
        self._settings_tab_index: int | None = None

        self._tabs.addTab(self._frf_workspace, "FRF")
        self._tabs.addTab(self._stab_workspace, "Stability")
        self._settings_tab_index = self._tabs.addTab(self._settings_root, "Settings")

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(10, 10, 10, 10)
        central_layout.setSpacing(10)
        central_layout.addWidget(self._tabs, 1)
        self.setCentralWidget(central)
        self._build_toolbar()

        load_dir = str(resolve_default_load_dir(self._repo_root, self._app_settings))
        self._frf_workspace.set_default_load_dir(load_dir)
        self._stab_workspace.set_default_load_dir(load_dir)
        self._wire_workspace(self._frf_workspace)
        self._wire_workspace(self._stab_workspace)

        self._settings_root.default_save_dir_changed.connect(self._on_default_save_dir_changed)
        self._settings_root.default_load_dir_changed.connect(self._on_default_load_dir_changed)
        self._settings_root.set_default_save_dir(str(self._app_settings.default_save_dir))
        self._settings_root.set_default_load_dir(str(self._app_settings.default_load_dir))

        self._add_analysis_tab(self._frf_workspace, apply_defaults=True)
        self._add_analysis_tab(self._stab_workspace, apply_defaults=False)
        self._restore_last_file_paths(self._stab_workspace.active_tab())
        self._tabs.setCurrentIndex(0)

    def _wire_workspace(self, workspace: AnalysisWorkspace) -> None:
        workspace.file_paths_changed.connect(self._on_frf_file_paths_changed)
        workspace.file_path_changed.connect(self._on_stability_file_path_changed)
        workspace.default_experiment_config_save_requested.connect(self._on_save_default_experiment_config)
        workspace.default_filtering_save_requested.connect(self._on_save_default_frf_filtering)

    def _is_settings_tab(self, idx: int) -> bool:
        return self._settings_tab_index is not None and idx == self._settings_tab_index

    def _workspace_for_type(self, analysis_type: AnalysisType) -> AnalysisWorkspace:
        if analysis_type == "stability_measurement":
            return self._stab_workspace
        return self._frf_workspace

    def _active_workspace(self) -> AnalysisWorkspace | None:
        idx = self._tabs.currentIndex()
        if idx < 0 or self._is_settings_tab(idx):
            return None
        widget = self._tabs.widget(idx)
        if isinstance(widget, AnalysisWorkspace):
            return widget
        return None

    def _active_analysis_tab(self) -> AnalysisTab | None:
        workspace = self._active_workspace()
        if workspace is None:
            return None
        return workspace.active_tab()

    def _add_analysis_tab(
        self,
        workspace: AnalysisWorkspace,
        *,
        apply_defaults: bool = False,
        select: bool = True,
    ) -> AnalysisTab:
        tab = workspace.add_empty_tab(select=select)
        tab.set_default_load_dir(str(resolve_default_load_dir(self._repo_root, self._app_settings)))
        if apply_defaults and workspace.analysis_type == "frf":
            self._apply_frf_defaults_to_tab(tab)
            self._restore_last_file_paths(tab)
        if workspace.analysis_type == "stability_measurement":
            self._restore_last_file_paths(tab)
        return tab

    def _apply_frf_defaults_to_tab(self, tab: AnalysisTab) -> None:
        if self._app_settings.default_experiment_config:
            try:
                tab.apply_default_experiment_config(dict(self._app_settings.default_experiment_config or {}))
            except Exception as e:
                QMessageBox.critical(self, "Failed to apply default experiment config", str(e))
        elif (
            self._app_settings.default_channel_mapping
            or self._app_settings.default_sensitivity_mapping
            or self._app_settings.default_sensitivities
        ):
            try:
                tab.apply_default_sensor_config(
                    channel_mapping=self._app_settings.default_channel_mapping,
                    sensitivity_mapping=self._app_settings.default_sensitivity_mapping,
                    sensitivities=self._app_settings.default_sensitivities,
                )
            except Exception as e:
                QMessageBox.critical(self, "Failed to apply default sensor config", str(e))

        if self._app_settings.default_frf_filtering:
            try:
                tab.apply_default_filtering(dict(self._app_settings.default_frf_filtering or {}))
            except Exception as e:
                QMessageBox.critical(self, "Failed to apply default filtering", str(e))

    def _restore_last_file_paths(self, tab: AnalysisTab | None) -> None:
        if tab is None:
            return
        self._restoring_settings = True
        try:
            if tab.analysis_type == "frf":
                tab.restore_frf_file_paths(
                    list(self._app_settings.last_frf_x_files or []),
                    list(self._app_settings.last_frf_y_files or []),
                )
            else:
                tab.restore_stability_file_path(self._app_settings.last_stability_file or "")
        finally:
            self._restoring_settings = False

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

        act_add = QAction("Add analysis tab", self)
        act_add.triggered.connect(self._on_add_analysis_tab)
        tb.addAction(act_add)

        tb.addSeparator()

    def _on_add_analysis_tab(self) -> None:
        workspace = self._active_workspace()
        if workspace is None:
            QMessageBox.warning(self, "Add analysis tab", "Select the FRF or Stability tab first.")
            return
        apply_defaults = workspace.analysis_type == "frf"
        self._add_analysis_tab(workspace, apply_defaults=apply_defaults, select=True)

    def _on_load_analysis(self) -> None:
        workspace = self._active_workspace()
        if workspace is None:
            QMessageBox.warning(self, "Load analysis", "Select the FRF or Stability tab first.")
            return
        tab = workspace.active_tab()
        if tab is None:
            QMessageBox.warning(self, "Load analysis", "Select an analysis tab first.")
            return

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

        target_workspace = self._workspace_for_type(st.analysis_type)
        if target_workspace is not workspace:
            self._tabs.setCurrentWidget(target_workspace)
            workspace = target_workspace
            tab = workspace.active_tab()
            if tab is None:
                tab = self._add_analysis_tab(workspace, apply_defaults=False, select=True)

        try:
            tab.apply_state(st)
        except Exception as e:
            QMessageBox.critical(self, "Load analysis failed", str(e))
            return
        workspace.set_tab_title(tab, st.title)
        QMessageBox.information(self, "Analysis loaded", f"Loaded: {path_str}")

    def _on_save_analysis(self) -> None:
        workspace = self._active_workspace()
        if workspace is None:
            QMessageBox.warning(self, "Save analysis", "Select the FRF or Stability tab first.")
            return
        tab = workspace.active_tab()
        if tab is None:
            QMessageBox.warning(self, "Save analysis", "Select an analysis tab first.")
            return

        default_dir = resolve_default_save_dir(self._repo_root, self._app_settings)
        default_dir.mkdir(parents=True, exist_ok=True)
        parent = QFileDialog.getExistingDirectory(
            self,
            "Select folder for this analysis",
            str(default_dir),
        )
        if not parent:
            return

        name, ok = QInputDialog.getText(self, "Save analysis", "Analysis name:", text=tab.title)
        if not ok:
            return
        name = (name or "Untitled").strip() or "Untitled"

        state = tab.build_state()
        state.title = name
        folder = Path(parent) / name
        try:
            analysis_json = save_analysis_folder(self._repo_root, state, folder)
            manifest = tab.export_results(folder)
            state.results_manifest = dict(manifest or {})
            state.analysis_folder = folder
            save_analysis_folder(self._repo_root, state, folder)
        except Exception as e:
            QMessageBox.critical(self, "Save analysis failed", str(e))
            return

        tab.set_title(name)
        tab.set_analysis_folder(folder)
        workspace.set_tab_title(tab, name)
        QMessageBox.information(self, "Analysis saved", f"Saved: {analysis_json}")

    def _on_default_save_dir_changed(self, path: str) -> None:
        self._app_settings.default_save_dir = Path(path) if path else Path("analyses")
        try:
            save_app_settings(self._repo_root, self._app_settings)
        except Exception as e:
            QMessageBox.critical(self, "Failed to save settings", str(e))

    def _on_default_load_dir_changed(self, path: str) -> None:
        self._app_settings.default_load_dir = Path(path) if path else Path(".")
        load_dir = str(resolve_default_load_dir(self._repo_root, self._app_settings))
        self._frf_workspace.set_default_load_dir(load_dir)
        self._stab_workspace.set_default_load_dir(load_dir)
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
        if x_list == (self._app_settings.last_frf_x_files or []) and y_list == (
            self._app_settings.last_frf_y_files or []
        ):
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
