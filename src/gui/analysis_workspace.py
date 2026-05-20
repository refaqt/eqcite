from __future__ import annotations

from pathlib import Path
from typing import Optional
from typing import Callable, List

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .pages.frf import FrfRootPage
from .pages.stability import StabilityRootPage
from .state import AnalysisState, AnalysisType


class AnalysisTab(QWidget):
    """One analysis session for a single mode (FRF or stability)."""

    file_paths_changed = Signal(object, object)
    file_path_changed = Signal(str)
    default_experiment_config_save_requested = Signal(dict)
    default_filtering_save_requested = Signal(dict)

    def __init__(
        self,
        analysis_type: AnalysisType,
        parent: QWidget | None = None,
        *,
        shared_compare_entries: Optional[list[object]] = None,
    ) -> None:
        super().__init__(parent)
        self._analysis_type: AnalysisType = analysis_type
        self._title = "Untitled"
        self._analysis_folder: Optional[Path] = None
        self._open_frf_tabs_provider: Optional[Callable[[], List["AnalysisTab"]]] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if analysis_type == "stability_measurement":
            self._stab_root = StabilityRootPage()
            self._frf_root = None
            self._stab_root.file_path_changed.connect(self.file_path_changed)
            layout.addWidget(self._stab_root, 1)
        else:
            self._frf_root = FrfRootPage(compare_entries_shared=shared_compare_entries)
            self._stab_root = None
            self._frf_root.file_paths_changed.connect(self.file_paths_changed)
            self._frf_root.default_experiment_config_save_requested.connect(
                self.default_experiment_config_save_requested
            )
            self._frf_root.default_filtering_save_requested.connect(self.default_filtering_save_requested)
            layout.addWidget(self._frf_root, 1)

    @property
    def title(self) -> str:
        return self._title

    @property
    def analysis_type(self) -> AnalysisType:
        return self._analysis_type

    @property
    def analysis_folder(self) -> Optional[Path]:
        return self._analysis_folder

    def set_title(self, title: str) -> None:
        self._title = (title or "Untitled").strip() or "Untitled"

    def set_analysis_folder(self, folder: Optional[Path]) -> None:
        self._analysis_folder = folder

    def set_default_load_dir(self, path: str) -> None:
        if self._frf_root is not None:
            self._frf_root.set_default_load_dir(path)
        if self._stab_root is not None:
            self._stab_root.set_default_load_dir(path)

    def set_open_frf_tabs_provider(self, provider: Callable[[], List["AnalysisTab"]]) -> None:
        """
        Injected by the workspace: lets FRF Compare enumerate all open FRF analysis tabs.
        """
        self._open_frf_tabs_provider = provider
        if self._frf_root is not None:
            self._frf_root.set_open_frf_tabs_provider(lambda: list(provider() or []))

    def get_frf_results_page(self) -> object | None:
        """
        For cross-tab Compare: return the FRF Results page object (or None).
        """
        if self._frf_root is None:
            return None
        try:
            return getattr(self._frf_root, "_page_results", None)
        except Exception:
            return None

    def apply_default_experiment_config(self, cfg: dict) -> None:
        if self._frf_root is None:
            return
        self._frf_root.apply_default_experiment_config(dict(cfg or {}))

    def apply_default_sensor_config(
        self,
        *,
        channel_mapping: dict,
        sensitivity_mapping: dict,
        sensitivities: dict,
    ) -> None:
        if self._frf_root is None:
            return
        self._frf_root.apply_default_sensor_config(
            channel_mapping=channel_mapping,
            sensitivity_mapping=sensitivity_mapping,
            sensitivities=sensitivities,
        )

    def apply_default_filtering(self, filtering_cfg: dict) -> None:
        if self._frf_root is None:
            return
        self._frf_root.apply_default_filtering(dict(filtering_cfg or {}))

    def restore_frf_file_paths(self, x_paths: list[str], y_paths: list[str]) -> None:
        if self._frf_root is None:
            return
        self._frf_root.set_file_paths(list(x_paths or []), list(y_paths or []))

    def restore_stability_file_path(self, path: str) -> None:
        if self._stab_root is None:
            return
        self._stab_root.set_file_path(path or "")

    def build_state(self) -> AnalysisState:
        if self._analysis_type == "stability_measurement":
            assert self._stab_root is not None
            snap = self._stab_root.build_session_snapshot()
        else:
            assert self._frf_root is not None
            snap = self._frf_root.build_session_snapshot()
        return AnalysisState(
            analysis_type=self._analysis_type,
            title=self._title,
            experiment_cfg=dict(snap.get("experiment_cfg", {}) or {}),
            gui_state=dict(snap.get("gui_state", {}) or {}),
            analysis_folder=self._analysis_folder,
        )

    def apply_state(self, state: AnalysisState) -> None:
        if state.analysis_type != self._analysis_type:
            raise ValueError(
                f"Cannot load {state.analysis_type!r} into a {self._analysis_type!r} analysis tab."
            )
        self.set_title(state.title)
        self.set_analysis_folder(state.analysis_folder)
        manifest = dict(state.results_manifest or {})
        if self._analysis_type == "stability_measurement":
            assert self._stab_root is not None
            self._stab_root.apply_session_snapshot(state.experiment_cfg, state.gui_state)
            if state.analysis_folder is not None:
                self._stab_root.hydrate_results_from_manifest(state.analysis_folder, manifest)
            self._stab_root._page_results.apply_gui_state(
                dict(state.gui_state.get("stability_results", {}) or {})
            )
            return

        assert self._frf_root is not None
        self._frf_root.apply_session_snapshot(state.experiment_cfg, state.gui_state)
        if state.analysis_folder is not None:
            self._frf_root.hydrate_results_from_manifest(state.analysis_folder, manifest)
        self._frf_root._page_results.apply_gui_state(dict(state.gui_state.get("frf_results", {}) or {}))
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
            self._frf_root._page_results._redraw_tab(tid)

    def export_results(self, analysis_folder: Path) -> dict:
        if self._analysis_type == "stability_measurement":
            assert self._stab_root is not None
            return self._stab_root.export_results(analysis_folder)
        assert self._frf_root is not None
        return self._frf_root.export_results(analysis_folder)


class AnalysisWorkspace(QWidget):
    """Nested analysis tabs for one top-level mode (FRF or stability)."""

    file_paths_changed = Signal(object, object)
    file_path_changed = Signal(str)
    default_experiment_config_save_requested = Signal(dict)
    default_filtering_save_requested = Signal(dict)

    def __init__(self, analysis_type: AnalysisType, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._analysis_type = analysis_type
        self._shared_compare_entries: Optional[list[object]] = (
            [] if analysis_type == "frf" else None
        )
        self._tabs = QTabWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tabs, 1)

    @property
    def analysis_type(self) -> AnalysisType:
        return self._analysis_type

    def _wire_tab(self, tab: AnalysisTab) -> None:
        tab.file_paths_changed.connect(self.file_paths_changed)
        tab.file_path_changed.connect(self.file_path_changed)
        tab.default_experiment_config_save_requested.connect(self.default_experiment_config_save_requested)
        tab.default_filtering_save_requested.connect(self.default_filtering_save_requested)

    def add_empty_tab(self, *, title: str = "Untitled", select: bool = True) -> AnalysisTab:
        tab = AnalysisTab(
            self._analysis_type,
            self,
            shared_compare_entries=self._shared_compare_entries,
        )
        self._wire_tab(tab)
        idx = self._tabs.addTab(tab, title)
        tab.set_title(title)
        # For FRF tabs, provide a live view of open FRF tabs in this workspace.
        if self._analysis_type == "frf":
            tab.set_open_frf_tabs_provider(self.open_tabs)
        if select:
            self._tabs.setCurrentIndex(idx)
        return tab

    def active_tab(self) -> AnalysisTab | None:
        widget = self._tabs.currentWidget()
        return widget if isinstance(widget, AnalysisTab) else None

    def open_tabs(self) -> List[AnalysisTab]:
        out: List[AnalysisTab] = []
        for i in range(self._tabs.count()):
            w = self._tabs.widget(i)
            if isinstance(w, AnalysisTab):
                out.append(w)
        return out

    def set_tab_title(self, tab: AnalysisTab, title: str) -> None:
        tab.set_title(title)
        idx = self._tabs.indexOf(tab)
        if idx >= 0:
            self._tabs.setTabText(idx, tab.title)

    def index_of(self, tab: AnalysisTab) -> int:
        return self._tabs.indexOf(tab)

    def set_default_load_dir(self, path: str) -> None:
        for i in range(self._tabs.count()):
            widget = self._tabs.widget(i)
            if isinstance(widget, AnalysisTab):
                widget.set_default_load_dir(path)
