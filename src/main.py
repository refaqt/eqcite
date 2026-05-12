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
import matplotlib.pyplot as plt

from . import load_data
from . import frf_estimation
from . import stability_measurement  # type: ignore


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config_before_cabinet_assembly.json"
DEFAULT_SENS = ROOT / "config" / "sensitivities.json"


def _load_experiment_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_frf_pipeline(exp_cfg_path: Path, sens_path: Path) -> None:
    """Run FRF estimation for the given experiment configuration."""
    exp_cfg = _load_experiment_config(exp_cfg_path)
    sens = load_data.load_sensitivities(sens_path)

    params = frf_estimation.get_system_parameters()
    if isinstance(exp_cfg, dict):
        if "bias_window_size" in exp_cfg:
            params["bias_window_size"] = int(exp_cfg["bias_window_size"])
        if "wiener_mysize" in exp_cfg:
            params["wiener_mysize"] = int(exp_cfg["wiener_mysize"])
        filt = exp_cfg.get("filtering")
        if isinstance(filt, dict):
            if "bias_window_size" in filt:
                params["bias_window_size"] = int(filt["bias_window_size"])
            fp = filt.get("params")
            if isinstance(fp, dict):
                w = fp.get("wiener")
                if isinstance(w, dict) and "mysize" in w:
                    params["wiener_mysize"] = int(w["mysize"])

    directional = load_data.load_directional_data(exp_cfg, sens)
    data_source = exp_cfg.get("data_source", "file")
    if data_source == "simulation":
        file_paths_x = []
        file_paths_y = []
    else:
        file_paths_x = [Path(p) for p in exp_cfg.get("file_paths_x", [])]
        file_paths_y = [Path(p) for p in exp_cfg.get("file_paths_y", [])]

    pairs_x = exp_cfg.get("frf_pairs_x", exp_cfg.get("frf_pairs", []))
    pairs_y = exp_cfg.get("frf_pairs_y", exp_cfg.get("frf_pairs", []))

    # The current CLI is configured to create one combined plot from one x-file and one y-file.
    x_plot_result: tuple[str, str, np.ndarray, np.ndarray] | None = None  # (input, output, f, H)
    y_plot_result: tuple[str, str, np.ndarray, np.ndarray] | None = None  # (input, output, f, H)

    for csv_path in file_paths_x or [Path("<simulation>")]:
        print(f"\n=== FRF analysis for x file {csv_path} ===")
        data = directional.get("x")
        if data is None:
            data = load_data.load_measurement_file_with_config(csv_path, exp_cfg, sens)

        time = np.asarray(data["time"])
        for pair in pairs_x:
            input_name = pair["input"]
            output_name = pair["output"]

            if input_name not in data or output_name not in data:
                print(
                    f"Skipping x pair {input_name}->{output_name}: "
                    f"signals not available in loaded data."
                )
                continue

            u = np.asarray(data[input_name])
            y = np.asarray(data[output_name])

            f, H = frf_estimation.compute_frf_from_signals(time, u, y, params)
            print(
                f"Computed x FRF for {input_name} -> {output_name} over {len(f)} frequency bins."
            )

            if x_plot_result is None:
                x_plot_result = (input_name, output_name, f, H)
        if x_plot_result is not None:
            break  # analyze only first x file (matches current config intent)

    for csv_path in file_paths_y or [Path("<simulation>")]:
        print(f"\n=== FRF analysis for y file {csv_path} ===")
        data = directional.get("y")
        if data is None:
            data = load_data.load_measurement_file_with_config(csv_path, exp_cfg, sens)

        time = np.asarray(data["time"])
        for pair in pairs_y:
            input_name = pair["input"]
            output_name = pair["output"]

            if input_name not in data or output_name not in data:
                print(
                    f"Skipping y pair {input_name}->{output_name}: "
                    f"signals not available in loaded data."
                )
                continue

            u = np.asarray(data[input_name])
            y = np.asarray(data[output_name])

            f, H = frf_estimation.compute_frf_from_signals(time, u, y, params)
            print(
                f"Computed y FRF for {input_name} -> {output_name} over {len(f)} frequency bins."
            )

            if y_plot_result is None:
                y_plot_result = (input_name, output_name, f, H)
        if y_plot_result is not None:
            break  # analyze only first y file (matches current config intent)

    if frf_estimation.plot_bode_diagram is None:
        return
    if x_plot_result is None or y_plot_result is None:
        print("Skipping Bode plotting: missing x and/or y FRF results.")
        return

    (x_inp, x_out, f_x, H_x) = x_plot_result
    (y_inp, y_out, f_y, H_y) = y_plot_result

    # If the Welch frequency grids differ between x and y files, align y onto x.
    if not np.allclose(f_x, f_y):
        print("Warning: frequency grids differ between x and y FRFs; interpolating y onto x grid.")
        H_y_interp = np.interp(f_x, f_y, np.real(H_y)) + 1j * np.interp(
            f_x, f_y, np.imag(H_y)
        )
        f_plot = f_x
        Hx_plot = H_x
        Hy_plot = H_y_interp
    else:
        f_plot = f_x
        Hx_plot = H_x
        Hy_plot = H_y

    print(f"Plotting combined Bode diagram for {x_inp}->{x_out} (x) and {y_inp}->{y_out} (y)...")
    frf_estimation.plot_bode_diagram(
        f_plot,
        Hx_plot,
        Hy_plot,
        show_plot=True,
    )
    # `plot_bode_diagram()` does not call `plt.show()`, so we do it once here.
    plt.show()


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

