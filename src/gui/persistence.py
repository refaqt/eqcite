from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from .state import ANALYSIS_FORMAT_VERSION, AnalysisState


APP_SETTINGS_PATH = Path("config") / "app_settings.json"


@dataclass
class AppSettings:
  default_save_dir: Path = Path("analyses")
  default_load_dir: Path = Path(".")
  default_channel_mapping: Dict[str, int] = field(default_factory=dict)
  default_sensitivity_mapping: Dict[str, str] = field(default_factory=dict)
  default_sensitivities: Dict[str, float] = field(default_factory=dict)
  default_experiment_config: Dict[str, Any] = field(default_factory=dict)
  last_frf_x_files: list[str] = field(default_factory=list)
  last_frf_y_files: list[str] = field(default_factory=list)
  last_stability_file: str = ""
  default_frf_filtering: Dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> Dict[str, Any]:
    return {
      "default_save_dir": str(self.default_save_dir),
      "default_load_dir": str(self.default_load_dir),
      "default_channel_mapping": dict(self.default_channel_mapping or {}),
      "default_sensitivity_mapping": dict(self.default_sensitivity_mapping or {}),
      "default_sensitivities": dict(self.default_sensitivities or {}),
      "default_experiment_config": dict(self.default_experiment_config or {}),
      "last_frf_x_files": list(self.last_frf_x_files or []),
      "last_frf_y_files": list(self.last_frf_y_files or []),
      "last_stability_file": str(self.last_stability_file or ""),
      "default_frf_filtering": dict(self.default_frf_filtering or {}),
    }

  @staticmethod
  def from_dict(d: Dict[str, Any]) -> "AppSettings":
    p = d.get("default_save_dir") or "analyses"
    load_p = d.get("default_load_dir")
    raw_chan = d.get("default_channel_mapping") or {}
    raw_sens_map = d.get("default_sensitivity_mapping") or {}
    raw_sens = d.get("default_sensitivities") or {}
    raw_exp = d.get("default_experiment_config") or {}
    last_frf_x_files = d.get("last_frf_x_files")
    last_frf_y_files = d.get("last_frf_y_files")
    last_frf_x_file = d.get("last_frf_x_file")
    last_frf_y_file = d.get("last_frf_y_file")
    last_stability_file = d.get("last_stability_file") or ""
    default_frf_filtering = d.get("default_frf_filtering") or {}

    chan: Dict[str, int] = {}
    if isinstance(raw_chan, dict):
      for k, v in raw_chan.items():
        try:
          chan[str(k)] = int(v)
        except Exception:
          continue

    sens_map: Dict[str, str] = {}
    if isinstance(raw_sens_map, dict):
      for k, v in raw_sens_map.items():
        if v is None:
          continue
        sens_map[str(k)] = str(v)

    sens: Dict[str, float] = {}
    if isinstance(raw_sens, dict):
      for k, v in raw_sens.items():
        try:
          sens[str(k)] = float(v)
        except Exception:
          continue

    def _coerce_path_list(v_list: Any, v_single: Any) -> list[str]:
      out: list[str] = []
      if isinstance(v_list, list):
        for x in v_list:
          if x is None:
            continue
          s = str(x).strip()
          if s:
            out.append(s)
      if not out and v_single is not None:
        s = str(v_single).strip()
        if s:
          out.append(s)
      seen: set[str] = set()
      uniq: list[str] = []
      for s in out:
        if s in seen:
          continue
        seen.add(s)
        uniq.append(s)
      return uniq

    exp_cfg: Dict[str, Any] = {}
    if isinstance(raw_exp, dict):
      exp_cfg = dict(raw_exp)

    return AppSettings(
      default_save_dir=Path(p),
      default_load_dir=Path(load_p) if load_p else Path(p),
      default_channel_mapping=chan,
      default_sensitivity_mapping=sens_map,
      default_sensitivities=sens,
      default_experiment_config=exp_cfg,
      last_frf_x_files=_coerce_path_list(last_frf_x_files, last_frf_x_file),
      last_frf_y_files=_coerce_path_list(last_frf_y_files, last_frf_y_file),
      last_stability_file=str(last_stability_file or ""),
      default_frf_filtering=dict(default_frf_filtering or {}) if isinstance(default_frf_filtering, dict) else {},
    )


def load_app_settings(repo_root: Path) -> AppSettings:
  path = repo_root / APP_SETTINGS_PATH
  if not path.exists():
    return AppSettings()
  data = json.loads(path.read_text(encoding="utf-8"))
  return AppSettings.from_dict(data)


def save_app_settings(repo_root: Path, settings: AppSettings) -> None:
  path = repo_root / APP_SETTINGS_PATH
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")


def _slugify(s: str) -> str:
  s = (s or "").strip().lower()
  s = re.sub(r"[^a-z0-9]+", "_", s)
  s = re.sub(r"_+", "_", s).strip("_")
  return s or "analysis"


def ensure_analysis_results_layout(folder: Path) -> None:
  folder = folder.resolve()
  (folder / "results" / "figures").mkdir(parents=True, exist_ok=True)
  (folder / "results" / "tables").mkdir(parents=True, exist_ok=True)
  (folder / "results" / "logs").mkdir(parents=True, exist_ok=True)


def save_analysis_folder(
  repo_root: Path,
  state: AnalysisState,
  folder: Path,
) -> Path:
  """Write analysis.json for format version 2."""
  del repo_root
  folder = folder.resolve()
  ensure_analysis_results_layout(folder)
  state.analysis_folder = folder
  analysis_json = folder / "analysis.json"
  analysis_json.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
  return analysis_json


def load_analysis_json(path: Path) -> AnalysisState:
  path = path.resolve()
  data = json.loads(path.read_text(encoding="utf-8"))
  if data.get("format_version") is None:
    raise ValueError(
      f"Legacy analysis format is not supported (expected format_version {ANALYSIS_FORMAT_VERSION})."
    )
  st = AnalysisState.from_dict(data)
  st.analysis_folder = path.parent
  return st


def resolve_default_save_dir(repo_root: Path, settings: AppSettings) -> Path:
  p = settings.default_save_dir
  if not p.is_absolute():
    p = (repo_root / p).resolve()
  return p


def resolve_default_load_dir(repo_root: Path, settings: AppSettings) -> Path:
  p = settings.default_load_dir
  if not p.is_absolute():
    p = (repo_root / p).resolve()
  return p
