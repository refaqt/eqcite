# eqcite

Python tooling for **frequency response (FRF) estimation** and **stability/chatter measurement** from time-series data, with both a **PySide6 GUI** and a **CLI-style entry point**.

## What this repo contains

- `src/gui/`: desktop GUI for FRF and stability workflows, save/load analysis snapshots, and export results.
- `src/main.py`: script-like entry point that runs FRF or stability pipelines from a default config.
- `src/load_data.py`: measurement/simulation loading plus experiment-schema variable mapping.
- `src/experiment_config.py`: schema helpers for the new FRF experiment model (catalog, mapping, expressions, analysis I/O).
- `src/frf_estimation.py`: FRF computation and plotting utilities.
- `src/stability_measurement.py`: stability/chatter processing.

More details live in `docs/onboarding.md` and `docs/architecture.md`.

## Prerequisites

- Python `3.12+`
- `pip` and `venv`

## Setup

Run from the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the GUI (recommended)

```powershell
python -m src.gui.app
```

The GUI includes:

- top-level tabs for `FRF`, `Stability`, and `Settings`
- nested analysis tabs per mode
- load/save analysis folders (`analysis.json` + exported results)
- FRF experiment configuration editing (sensor catalog, mapping, expressions, direction I/O)

## Run the CLI-style pipeline

```powershell
python -m src.main
```

Current behavior in `src/main.py`:

- uses hard-coded `DEFAULT_CONFIG` (`config/config_before_cabinet_assembly.json`)
- dispatches by `analysis_type` (`frf` or `stability_measurement`)
- FRF run computes from the first available X and Y files and shows a combined Bode plot

## Config model (FRF)

FRF currently supports two schema styles:

1. **New schema** (preferred): uses `sensor_catalog`, `mapping`, `expressions`, and `analysis_io`.
2. **Legacy schema**: uses `channel_mapping` + `sensitivity_mapping` with `config/sensitivities.json`.

If a config does not include `sensor_catalog`, the code can migrate legacy mapping/sensitivity data internally for FRF processing.

Useful examples:

- `config/example_experiment_frf.json`
- `config/example_experiment_stability.json`

## Data sources

- `data_source: "file"`: load measurement CSV files from `file_paths_x` / `file_paths_y` (FRF) or `file_paths` (stability).
- `data_source: "simulation"`: generate synthetic FRF inputs/outputs via `src/simulate_system.py`.

## CSV assumptions

`src/signal_processing.py::load_csv_file()` expects:

1. first line containing sample-rate text like `Samples per sec. = XXXXX/1`
2. CSV data after the header section (loader skips fixed header rows)
3. columns in the shape `Time, A0..A6` (optional `A7`)

## Tests

```powershell
$env:PYTHONPATH="."
python -m unittest discover -s tests -p "test*.py" -v
```

## Troubleshooting

- `FileNotFoundError`: configured input paths are invalid.
- `Could not find sample rate in CSV header`: first CSV line is not in expected format.
- `ValueError: Expected 8 or 9 columns`: CSV payload shape does not match loader expectations.
- FRF pair/input-output selection errors: configured variable names do not exist in loaded signals.

