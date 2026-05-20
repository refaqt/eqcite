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

from . import experiment_config, load_data
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
    if not experiment_config.uses_new_experiment_schema(exp_cfg) and (
        exp_cfg.get("channel_mapping") or exp_cfg.get("sensitivity_mapping")
    ):
        sens_all = load_data.load_sensitivities(sens_path)
        exp_cfg = experiment_config.migrate_legacy_experiment_cfg(exp_cfg, sens_all)

    if experiment_config.uses_new_experiment_schema(exp_cfg):
        sens: Dict[str, float] = {}
    else:
        sens = load_data.load_sensitivities(sens_path)

    params = frf_estimation.get_system_parameters()
    filtering_cfg = exp_cfg.get("filtering") if isinstance(exp_cfg.get("filtering"), dict) else None
    if isinstance(exp_cfg, dict):
        if "wiener_mysize" in exp_cfg:
            params["wiener_mysize"] = int(exp_cfg["wiener_mysize"])
        if filtering_cfg is not None:
            fp = filtering_cfg.get("params")
            if isinstance(fp, dict):
                w = fp.get("wiener")
                if isinstance(w, dict) and "mysize" in w:
                    params["wiener_mysize"] = int(w["mysize"])
            if "estimator_type" in filtering_cfg:
                params["estimator_type"] = str(filtering_cfg.get("estimator_type") or "Hv")

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
    if experiment_config.uses_new_experiment_schema(exp_cfg):
        io = exp_cfg.get("analysis_io") or {}
        bx = io.get("x") or {}
        by = io.get("y") or {}
        if bx.get("input") and bx.get("output"):
            pairs_x = [{"input": str(bx["input"]), "output": str(bx["output"])}]
        if by.get("input") and by.get("output"):
            pairs_y = [{"input": str(by["input"]), "output": str(by["output"])}]

    # The current CLI is configured to create one combined plot from one x-file and one y-file.
    x_plot_result: tuple[str, str, np.ndarray, np.ndarray] | None = None  # (input, output, f, H)
    y_plot_result: tuple[str, str, np.ndarray, np.ndarray] | None = None  # (input, output, f, H)

    for csv_path in file_paths_x or [Path("<simulation>")]:
        print(f"\n=== FRF analysis for x file {csv_path} ===")
        if data_source == "simulation":
            x_list = directional.get("x")
            if not isinstance(x_list, list) or not x_list:
                print("No simulation data for x.")
                continue
            data = x_list[0]
        else:
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
            local_params = dict(params)
            if filtering_cfg is not None:
                from .gui.pages.frf import apply_frf_filtering_and_windowing

                u, y, local_params = apply_frf_filtering_and_windowing(
                    time=time,
                    u=u,
                    resp=y,
                    params=local_params,
                    filtering_cfg=filtering_cfg,
                )

            f, H = frf_estimation.compute_frf_from_signals(time, u, y, local_params)
            print(
                f"Computed x FRF for {input_name} -> {output_name} over {len(f)} frequency bins."
            )

            if x_plot_result is None:
                x_plot_result = (input_name, output_name, f, H)
        if x_plot_result is not None:
            break  # analyze only first x file (matches current config intent)

    for csv_path in file_paths_y or [Path("<simulation>")]:
        print(f"\n=== FRF analysis for y file {csv_path} ===")
        if data_source == "simulation":
            y_list = directional.get("y")
            if not isinstance(y_list, list) or not y_list:
                print("No simulation data for y.")
                continue
            data = y_list[0]
        else:
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
            local_params = dict(params)
            if filtering_cfg is not None:
                from .gui.pages.frf import apply_frf_filtering_and_windowing

                u, y, local_params = apply_frf_filtering_and_windowing(
                    time=time,
                    u=u,
                    resp=y,
                    params=local_params,
                    filtering_cfg=filtering_cfg,
                )

            f, H = frf_estimation.compute_frf_from_signals(time, u, y, local_params)
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

