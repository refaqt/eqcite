from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional


AnalysisType = Literal["frf", "stability_measurement"]


def _as_path(p: str | None) -> Optional[Path]:
    if not p:
        return None
    return Path(p)


@dataclass
class AnalysisState:
    analysis_type: AnalysisType = "frf"

    experiment_title: str = ""
    experiment_description: str = ""

    # Snapshot of the experiment JSON config that drives the pipelines.
    experiment_cfg: Dict[str, Any] = field(default_factory=dict)

    # Global sensitivities live in config/sensitivities.json (not duplicated into analysis folder).
    sensitivities_path: Optional[Path] = None

    # GUI-only fields
    selected_top_tab: int = 0
    selected_sidebar_index: Dict[str, int] = field(default_factory=dict)  # {"frf": 0, "stability": 0}

    # Save/load bookkeeping
    analysis_folder: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_type": self.analysis_type,
            "experiment_title": self.experiment_title,
            "experiment_description": self.experiment_description,
            "experiment_cfg": self.experiment_cfg,
            "sensitivities_path": str(self.sensitivities_path) if self.sensitivities_path else None,
            "gui": {
                "selected_top_tab": self.selected_top_tab,
                "selected_sidebar_index": self.selected_sidebar_index,
            },
            "analysis_folder": str(self.analysis_folder) if self.analysis_folder else None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AnalysisState":
        gui = d.get("gui", {}) or {}
        return AnalysisState(
            analysis_type=d.get("analysis_type", "frf"),
            experiment_title=d.get("experiment_title", ""),
            experiment_description=d.get("experiment_description", ""),
            experiment_cfg=d.get("experiment_cfg", {}) or {},
            sensitivities_path=_as_path(d.get("sensitivities_path")),
            selected_top_tab=int(gui.get("selected_top_tab", 0)),
            selected_sidebar_index=dict(gui.get("selected_sidebar_index", {}) or {}),
            analysis_folder=_as_path(d.get("analysis_folder")),
        )

