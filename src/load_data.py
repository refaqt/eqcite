#!/usr/bin/env python3
"""
Data loading and experiment variable mapping.

Loads CSV (or simulation), applies the experiment configuration: sensor
catalog + channel mapping + optional expressions, and exposes named
time series for FRF analysis. Legacy channel_mapping + sensitivities.json
path remains for older configs without sensor_catalog.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from . import experiment_config, expression_eval, simulate_system
from .sensor_model import sensors_to_inputs, sensors_to_spindle, sensors_to_tool
from .signal_processing import load_csv_file


def _load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_sensitivities(path: str | Path) -> Dict[str, float]:
    """Load global sensitivities.json as a simple name->value mapping."""
    data = _load_json(path)
    return {k: v for k, v in data.items() if not str(k).startswith("//")}


def _apply_sensitivity(raw_volt: np.ndarray, sensitivity: float) -> np.ndarray:
    return np.asarray(raw_volt, dtype=float) * float(sensitivity)


def _catalog_by_id(sensor_catalog: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in sensor_catalog:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "").strip()
        if sid:
            out[sid] = row
    return out


def build_measurement_variables(
    *,
    time: np.ndarray,
    channels: Mapping[int, np.ndarray],
    experiment_cfg: Mapping[str, Any],
) -> tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Build mapping + expression variables (volts -> physical) and return
    (variables dict, variable_kinds dict).
    """
    sensor_catalog = experiment_cfg.get("sensor_catalog") or []
    if not isinstance(sensor_catalog, list) or not sensor_catalog:
        raise ValueError("experiment_cfg.sensor_catalog is required for the new pipeline")
    catalog_by_id = _catalog_by_id([r for r in sensor_catalog if isinstance(r, dict)])

    mapping = experiment_cfg.get("mapping") or []
    if not isinstance(mapping, list):
        raise ValueError("experiment_cfg.mapping must be a list")

    variables: Dict[str, np.ndarray] = {}
    seen_names: set[str] = set()

    for row in mapping:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        experiment_config.assert_valid_mapping_name(name)
        if name in seen_names:
            raise ValueError(f"Duplicate mapping variable name: {name!r}")
        seen_names.add(name)
        ch = int(row["channel"])
        if ch not in channels:
            raise ValueError(f"Channel {ch} not available for mapping variable {name!r}")
        sid = str(row.get("sensor_id") or "").strip()
        if sid not in catalog_by_id:
            raise ValueError(f"Unknown sensor_id {sid!r} for mapping variable {name!r}")
        sens = float(catalog_by_id[sid]["sensitivity_vs_per_v"])
        variables[name] = _apply_sensitivity(channels[ch], sens)

    expressions = experiment_cfg.get("expressions") or []
    if not isinstance(expressions, list):
        raise ValueError("experiment_cfg.expressions must be a list")

    for row in expressions:
        if not isinstance(row, dict):
            continue
        var_name = str(row.get("name") or "").strip()
        expr = str(row.get("expr") or "").strip()
        if not var_name or not expr:
            continue
        experiment_config.assert_valid_mapping_name(var_name)
        if var_name in seen_names:
            raise ValueError(f"Duplicate variable name (mapping or expression): {var_name!r}")
        seen_names.add(var_name)
        env = {k: variables[k] for k in expression_eval.referenced_mapping_names(expr)}
        variables[var_name] = expression_eval.evaluate_expression(expr, env)

    kinds_map = experiment_config.build_variable_kinds(experiment_cfg)
    kinds_str = {k: str(v) for k, v in kinds_map.items()}
    return variables, kinds_str


def _load_measurement_legacy(
    csv_path: str | Path,
    experiment_cfg: Dict[str, Any],
    sensitivities: Dict[str, float],
) -> Dict[str, Any]:
    time, A0, A1, A2, A3, A4, A5, A6, A7, fs, dt = load_csv_file(str(csv_path))
    channels = {0: A0, 1: A1, 2: A2, 3: A3, 4: A4, 5: A5, 6: A6, 7: A7}

    chan_map = experiment_cfg.get("channel_mapping", {})
    sens_map = experiment_cfg.get("sensitivity_mapping", {})
    sensors_physical: Dict[str, np.ndarray] = {}

    for sensor_name, chan_idx in chan_map.items():
        if chan_idx not in channels:
            raise ValueError(
                f"Channel {chan_idx} for sensor '{sensor_name}' not present in file {csv_path}"
            )
        raw = channels[chan_idx]
        sens_name = sens_map.get(sensor_name)
        if sens_name is None:
            raise ValueError(
                f"No sensitivity mapping found for sensor '{sensor_name}' in experiment config"
            )
        if sens_name not in sensitivities:
            raise ValueError(
                f"Sensitivity '{sens_name}' for sensor '{sensor_name}' not found in sensitivities.json"
            )
        sensors_physical[sensor_name] = _apply_sensitivity(raw, sensitivities[sens_name])

    acc_workbed_x = sensors_physical.get("acc_workbed_x", np.zeros_like(time))
    acc_workbed_y = sensors_physical.get("acc_workbed_y", np.zeros_like(time))

    acc_spindle_top = sensors_physical.get("acc_spindle_top")
    acc_spindle_left = sensors_physical.get("acc_spindle_left")
    acc_spindle_right = sensors_physical.get("acc_spindle_right")

    if (
        acc_spindle_top is not None
        and acc_spindle_left is not None
        and acc_spindle_right is not None
    ):
        x_spindle, y_spindle, z_spindle = sensors_to_spindle(
            acc_spindle_top,
            acc_spindle_left,
            acc_spindle_right,
            acc_workbed_x,
            acc_workbed_y,
        )
    else:
        x_spindle = y_spindle = z_spindle = None

    acc_tool_x = sensors_physical.get("acc_tool_x")
    acc_tool_y = sensors_physical.get("acc_tool_y")
    if acc_tool_x is not None and acc_tool_y is not None:
        x_tool, y_tool = sensors_to_tool(
            acc_tool_x,
            acc_tool_y,
            acc_workbed_x,
            acc_workbed_y,
        )
    else:
        x_tool = y_tool = None

    hammer_x = sensors_physical.get("hammer_x")
    hammer_y = sensors_physical.get("hammer_y")
    inputs = sensors_to_inputs(hammer_x=hammer_x, hammer_y=hammer_y)

    result: Dict[str, Any] = {
        "filepath": str(csv_path),
        "time": time,
        "fs": fs,
        "dt": dt,
        "channels": channels,
        "sensors_physical": sensors_physical,
        "acc_workbed_x": acc_workbed_x,
        "acc_workbed_y": acc_workbed_y,
        "x_spindle": x_spindle,
        "y_spindle": y_spindle,
        "z_spindle": z_spindle,
        "x_tool": x_tool,
        "y_tool": y_tool,
    }
    result.update(inputs)
    return result


def _load_measurement_new_schema(
    csv_path: str | Path,
    experiment_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    time, A0, A1, A2, A3, A4, A5, A6, A7, fs, dt = load_csv_file(str(csv_path))
    channels = {0: A0, 1: A1, 2: A2, 3: A3, 4: A4, 5: A5, 6: A6, 7: A7}
    variables, variable_kinds = build_measurement_variables(
        time=time, channels=channels, experiment_cfg=experiment_cfg
    )
    result: Dict[str, Any] = {
        "filepath": str(csv_path),
        "time": time,
        "fs": fs,
        "dt": dt,
        "channels": channels,
        "sensors_physical": dict(variables),
        "variable_kinds": variable_kinds,
        "acc_workbed_x": np.zeros_like(time),
        "acc_workbed_y": np.zeros_like(time),
        "x_spindle": None,
        "y_spindle": None,
        "z_spindle": None,
        "x_tool": None,
        "y_tool": None,
    }
    result.update(variables)
    return result


def load_measurement_file_with_config(
    csv_path: str | Path,
    experiment_cfg: Dict[str, Any],
    sensitivities: Dict[str, float],
) -> Dict[str, Any]:
    """Load a single CSV using legacy or new experiment schema."""
    if experiment_config.uses_new_experiment_schema(experiment_cfg):
        return _load_measurement_new_schema(csv_path, experiment_cfg)
    return _load_measurement_legacy(csv_path, experiment_cfg, sensitivities)


def load_simulation_with_config(experiment_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulated time-series in the same shape as measurement loading.

    With new experiment schema, ``simulation_signal_map`` maps user-facing
    variable names to keys present in the simulator output dict.
    """
    sim_cfg = experiment_cfg.get("simulation", {})
    sim = simulate_system.simulate_two_axis_system(sim_cfg)
    time = np.asarray(sim["time"])

    canonical: Dict[str, Any] = {
        "time": time,
        "fs": float(sim["fs"]),
        "dt": float(sim["dt"]),
        "F_x": np.asarray(sim["F_x"]),
        "F_y": np.asarray(sim["F_y"]),
        "x_tool": np.asarray(sim.get("x_ddot", sim.get("x"))),
        "y_tool": np.asarray(sim.get("y_ddot", sim.get("y"))),
    }

    if experiment_config.uses_new_experiment_schema(experiment_cfg):
        variables: Dict[str, np.ndarray] = {}
        smap = experiment_cfg.get("simulation_signal_map") or {}
        if not isinstance(smap, dict):
            raise ValueError("simulation_signal_map must be a dict when using new schema")
        for user_key, canon_key in smap.items():
            uk = str(user_key).strip()
            ck = str(canon_key).strip()
            if not uk or not ck:
                continue
            if ck not in canonical:
                raise ValueError(
                    f"simulation_signal_map[{uk!r}] -> unknown canonical key {ck!r}; "
                    f"expected one of {sorted(canonical.keys())}"
                )
            variables[uk] = np.asarray(canonical[ck], dtype=float)

        kinds_only: Dict[str, str] = {}
        for uk, ck in smap.items():
            uk = str(uk).strip()
            ck = str(ck).strip()
            if ck in ("F_x", "F_y"):
                kinds_only[uk] = "force"
            elif ck in ("x_tool", "y_tool"):
                kinds_only[uk] = "acceleration"
            else:
                kinds_only[uk] = "none"

        result: Dict[str, Any] = {
            "filepath": None,
            "time": time,
            "fs": float(sim["fs"]),
            "dt": float(sim["dt"]),
            "channels": {},
            "sensors_physical": dict(variables),
            "variable_kinds": kinds_only,
            "acc_workbed_x": np.zeros_like(time),
            "acc_workbed_y": np.zeros_like(time),
            "x_spindle": None,
            "y_spindle": None,
            "z_spindle": None,
            "x_tool": None,
            "y_tool": None,
        }
        result.update(variables)
        for k, v in canonical.items():
            if k not in ("time", "fs", "dt"):
                result.setdefault(k, v)
        return result

    result = {
        "filepath": None,
        "time": time,
        "fs": float(sim["fs"]),
        "dt": float(sim["dt"]),
        "channels": {},
        "sensors_physical": {},
        "x_tool": canonical["x_tool"],
        "y_tool": canonical["y_tool"],
        "x_spindle": None,
        "y_spindle": None,
        "z_spindle": None,
        "F_x": canonical["F_x"],
        "F_y": canonical["F_y"],
    }
    return result


def load_directional_data(
    experiment_cfg: Dict[str, Any],
    sensitivities: Dict[str, float],
) -> Dict[str, Any]:
    """
    Load or generate data based on ``data_source`` in config.

    Returns:
      dict with optional keys ``x`` / ``y`` mapping to lists of per-file dicts.
    """
    data_source = experiment_cfg.get("data_source", "file")
    if data_source == "simulation":
        sim = load_simulation_with_config(experiment_cfg)
        return {"x": [sim], "y": [sim]}

    if data_source != "file":
        raise ValueError(f"Unknown data_source: {data_source}")

    out: Dict[str, Any] = {}
    file_paths_x = [Path(p) for p in experiment_cfg.get("file_paths_x", [])]
    file_paths_y = [Path(p) for p in experiment_cfg.get("file_paths_y", [])]

    if file_paths_x:
        out["x"] = [
            load_measurement_file_with_config(p, experiment_cfg, sensitivities) for p in file_paths_x
        ]
    if file_paths_y:
        out["y"] = [
            load_measurement_file_with_config(p, experiment_cfg, sensitivities) for p in file_paths_y
        ]

    return out
