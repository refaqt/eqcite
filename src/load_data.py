#!/usr/bin/env python3
"""
Data loading and sensor mapping.

This module loads raw measurement data (CSV) or receives simulated
time-series, applies channel/sensitivity mappings from a JSON
experiment configuration, and uses sensor_model to construct a
unified set of physical variables for downstream FRF or stability
analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from .signal_processing import load_csv_file
from .sensor_model import (
    sensors_to_spindle,
    sensors_to_tool,
    sensors_to_inputs,
)


def _load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open(\"r\", encoding=\"utf-8\") as f:
        return json.load(f)


def load_sensitivities(path: str | Path) -> Dict[str, float]:
    \"\"\"Load global sensitivities.json as a simple name->value mapping.\"\"\"
    data = _load_json(path)
    # Filter out comment-style keys if present
    return {k: v for k, v in data.items() if not k.startswith(\"//\")}


def _apply_sensitivity(
    raw_volt: np.ndarray,
    sensitivity: float,
) -> np.ndarray:
    \"\"\"Convert a voltage signal to physical units using a scalar sensitivity.\"\"\"
    return np.asarray(raw_volt, dtype=float) * float(sensitivity)


def load_measurement_file_with_config(
    csv_path: str | Path,
    experiment_cfg: Dict[str, Any],
    sensitivities: Dict[str, float],
) -> Dict[str, Any]:
    \"\"\"Load a single CSV measurement file using the channel/sensitivity config.

    Returns a dictionary with:
      - 'time', 'fs', 'dt'
      - raw channel arrays by index (A0..A7)
      - sensor-level accelerations (acc_workbed_x, acc_workbed_y, ...)
      - derived variables (x_spindle, y_spindle, z_spindle, x_tool, y_tool)
      - inputs F_x, F_y where available
    \"\"\"
    time, A0, A1, A2, A3, A4, A5, A6, A7, fs, dt = load_csv_file(str(csv_path))

    # Build channel dictionary for convenience (0-based indices)
    channels = {
        0: A0,
        1: A1,
        2: A2,
        3: A3,
        4: A4,
        5: A5,
        6: A6,
        7: A7,
    }

    chan_map = experiment_cfg.get(\"channel_mapping\", {})
    sens_map = experiment_cfg.get(\"sensitivity_mapping\", {})

    # Map channels to physical sensor accelerations / forces
    sensors_physical: Dict[str, np.ndarray] = {}

    for sensor_name, chan_idx in chan_map.items():
        if chan_idx not in channels:
            raise ValueError(
                f\"Channel {chan_idx} for sensor '{sensor_name}' not present in file {csv_path}\"
            )
        raw = channels[chan_idx]
        sens_name = sens_map.get(sensor_name)
        if sens_name is None:
            raise ValueError(
                f\"No sensitivity mapping found for sensor '{sensor_name}' in experiment config\"
            )
        if sens_name not in sensitivities:
            raise ValueError(
                f\"Sensitivity '{sens_name}' for sensor '{sensor_name}' not found in sensitivities.json\"
            )
        sensors_physical[sensor_name] = _apply_sensitivity(raw, sensitivities[sens_name])

    # Workbed accelerations (if present)
    acc_workbed_x = sensors_physical.get(\"acc_workbed_x\", np.zeros_like(time))
    acc_workbed_y = sensors_physical.get(\"acc_workbed_y\", np.zeros_like(time))

    # Spindle accelerations (if present)
    acc_spindle_top = sensors_physical.get(\"acc_spindle_top\")
    acc_spindle_left = sensors_physical.get(\"acc_spindle_left\")
    acc_spindle_right = sensors_physical.get(\"acc_spindle_right\")

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

    # Tool accelerations (if present)
    acc_tool_x = sensors_physical.get(\"acc_tool_x\")
    acc_tool_y = sensors_physical.get(\"acc_tool_y\")
    if acc_tool_x is not None and acc_tool_y is not None:
        x_tool, y_tool = sensors_to_tool(
            acc_tool_x,
            acc_tool_y,
            acc_workbed_x,
            acc_workbed_y,
        )
    else:
        x_tool = y_tool = None

    # Hammer / force inputs (if present)
    hammer_x = sensors_physical.get(\"hammer_x\")
    hammer_y = sensors_physical.get(\"hammer_y\")
    inputs = sensors_to_inputs(hammer_x=hammer_x, hammer_y=hammer_y)

    result: Dict[str, Any] = {
        \"filepath\": str(csv_path),
        \"time\": time,
        \"fs\": fs,
        \"dt\": dt,
        \"channels\": channels,
        \"sensors_physical\": sensors_physical,
        \"acc_workbed_x\": acc_workbed_x,
        \"acc_workbed_y\": acc_workbed_y,
        \"x_spindle\": x_spindle,
        \"y_spindle\": y_spindle,
        \"z_spindle\": z_spindle,
        \"x_tool\": x_tool,
        \"y_tool\": y_tool,
    }
    result.update(inputs)
    return result

