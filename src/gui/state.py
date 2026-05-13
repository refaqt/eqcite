from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional


AnalysisType = Literal["frf", "stability_measurement"]

ANALYSIS_FORMAT_VERSION = 2


def _as_path(p: str | None) -> Optional[Path]:
    if not p:
        return None
    return Path(p)


@dataclass
class AnalysisState:
  """Serializable snapshot for one analysis tab."""

  analysis_type: AnalysisType = "frf"
  title: str = "Untitled"

  experiment_cfg: Dict[str, Any] = field(default_factory=dict)
  gui_state: Dict[str, Any] = field(default_factory=dict)
  results_manifest: Dict[str, Any] = field(default_factory=dict)

  analysis_folder: Optional[Path] = None

  def to_dict(self) -> Dict[str, Any]:
    return {
      "format_version": ANALYSIS_FORMAT_VERSION,
      "analysis_name": self.title,
      "analysis_type": self.analysis_type,
      "experiment_cfg": self.experiment_cfg,
      "gui": dict(self.gui_state or {}),
      "results_manifest": dict(self.results_manifest or {}),
      "analysis_folder": str(self.analysis_folder) if self.analysis_folder else None,
    }

  @staticmethod
  def from_dict(d: Dict[str, Any]) -> "AnalysisState":
    fmt = d.get("format_version")
    if fmt is None or int(fmt) != ANALYSIS_FORMAT_VERSION:
      raise ValueError(
        f"Unsupported analysis format (expected version {ANALYSIS_FORMAT_VERSION})."
      )

    gui = d.get("gui", {}) or {}
    if not isinstance(gui, dict):
      gui = {}

    manifest = d.get("results_manifest", {}) or {}
    if not isinstance(manifest, dict):
      manifest = {}

    return AnalysisState(
      analysis_type=d.get("analysis_type", "frf"),
      title=str(d.get("analysis_name") or "Untitled"),
      experiment_cfg=dict(d.get("experiment_cfg", {}) or {}),
      gui_state=dict(gui),
      results_manifest=dict(manifest),
      analysis_folder=_as_path(d.get("analysis_folder")),
    )
