#!/usr/bin/env python3
"""
Program entry point.

Loads configuration, data (measurement or simulation), and runs either
FRF estimation or stability measurement. Designed so the same pipeline
can later be driven by a GUI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from . import load_data
from . import frf_estimation
from . import stability_measurement  # type: ignore


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "example_experiment_frf.json"
DEFAULT_SENS = ROOT / "config" / "sensitivities.json"


def _load_experiment_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_frf_pipeline(exp_cfg_path: Path, sens_path: Path) -> None:
    """Run FRF estimation for the given experiment configuration."""
    exp_cfg = _load_experiment_config(exp_cfg_path)
    sens = load_data.load_sensitivities(sens_path)

    params = frf_estimation.get_system_parameters()

    file_paths_x = [Path(p) for p in exp_cfg.get("file_paths_x", [])]
    file_paths_y = [Path(p) for p in exp_cfg.get("file_paths_y", [])]

    for csv_path in file_paths_x + file_paths_y:
        print(f"\n=== FRF analysis for {csv_path} ===")
        data = load_data.load_measurement_file_with_config(csv_path, exp_cfg, sens)

        time = data["time"]
        results: Dict[str, Any] = {}

        for pair in exp_cfg.get("frf_pairs", []):
            input_name = pair["input"]
            output_name = pair["output"]

            if input_name not in data or output_name not in data:
                print(
                    f"Skipping pair {input_name}->{output_name}: "
                    f"signals not available in loaded data."
                )
                continue

            u = np.asarray(data[input_name])
            y = np.asarray(data[output_name])

            f, H = frf_estimation.compute_frf_from_signals(time, u, y, params)
            results[(input_name, output_name)] = (f, H)
            print(
                f"Computed FRF for {input_name} -> {output_name} "
                f"over {len(f)} frequency bins."
            )

        # Optional: plot Bode for first pair if legacy plotting is available
        if results and frf_estimation.plot_bode_diagram is not None:
            (inp, out), (f, H) = next(iter(results.items()))
            print(f"Plotting Bode diagram for {inp}->{out} using legacy plotting...")
            frf_estimation.plot_bode_diagram(
                f,
                H,
                H,  # placeholder for Gy to keep signature simple
                show_plot=True,
            )


def run_stability_pipeline(exp_cfg_path: Path, sens_path: Path) -> None:
    """Run stability measurement for the given experiment configuration."""
    exp_cfg = _load_experiment_config(exp_cfg_path)
    files = [Path(p) for p in exp_cfg.get("file_paths", [])]
    for csv_path in files:
        print(f"\n=== Stability measurement for {csv_path} ===")
        stability_measurement.process_stability_measurement(str(csv_path), show_plot=True)


def main() -> None:
    """
    Entry point. For now, paths to configuration files are hard-coded.
    """
    cfg = _load_experiment_config(DEFAULT_CONFIG)
    analysis_type = cfg.get("analysis_type", "frf")

    if analysis_type == "frf":
        run_frf_pipeline(DEFAULT_CONFIG, DEFAULT_SENS)
    elif analysis_type == "stability_measurement":
        run_stability_pipeline(DEFAULT_CONFIG, DEFAULT_SENS)
    else:
        raise ValueError(f"Unknown analysis_type in config: {analysis_type}")


if __name__ == "__main__":
    main()

