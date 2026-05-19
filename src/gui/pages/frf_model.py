"""FRF GUI input model (shared to avoid circular imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FrfInputModel:
    data_source: str = "file"  # "file" | "simulation"
    file_paths_x: List[str] = field(default_factory=list)
    file_paths_y: List[str] = field(default_factory=list)
    simulation: Dict[str, Any] = field(default_factory=dict)

    sensor_catalog: List[Dict[str, Any]] = field(default_factory=list)
    mapping_rows: List[Dict[str, Any]] = field(default_factory=list)
    expressions: List[Dict[str, Any]] = field(default_factory=list)
    analysis_io: Dict[str, Dict[str, str]] = field(default_factory=dict)
    simulation_signal_map: Dict[str, str] = field(default_factory=dict)

    channel_mapping: Dict[str, int] = field(default_factory=dict)
    sensitivity_mapping: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_experiment_cfg(cfg: Dict[str, Any]) -> "FrfInputModel":
        cfg = dict(cfg or {})
        return FrfInputModel(
            data_source=str(cfg.get("data_source") or "file"),
            file_paths_x=list(cfg.get("file_paths_x") or []),
            file_paths_y=list(cfg.get("file_paths_y") or []),
            simulation=dict(cfg.get("simulation") or {}),
            sensor_catalog=list(cfg.get("sensor_catalog") or []),
            mapping_rows=list(cfg.get("mapping") or []),
            expressions=list(cfg.get("expressions") or []),
            analysis_io=dict(cfg.get("analysis_io") or {}),
            simulation_signal_map=dict(cfg.get("simulation_signal_map") or {}),
            channel_mapping=dict(cfg.get("channel_mapping") or {}),
            sensitivity_mapping=dict(cfg.get("sensitivity_mapping") or {}),
        )

    def to_experiment_cfg_fragment(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sensor_catalog": list(self.sensor_catalog or []),
            "mapping": list(self.mapping_rows or []),
            "expressions": list(self.expressions or []),
            "analysis_io": dict(self.analysis_io or {}),
            "simulation_signal_map": dict(self.simulation_signal_map or {}),
        }
        if not (self.sensor_catalog or []):
            out["channel_mapping"] = dict(self.channel_mapping or {})
            out["sensitivity_mapping"] = dict(self.sensitivity_mapping or {})
        return out
