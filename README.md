# cnc-frf-estimation

Python tooling for **frequency response (FRF) estimation** and **stability / chatter-related measurement** from time-series CSV files.

## What’s inside

- `src/main.py`: main entry point that runs either the FRF pipeline (`analysis_type: "frf"`) or the stability pipeline (`analysis_type: "stability_measurement"`), based on the JSON config it loads.
- `src/load_data.py`: measurement loading + channel/sensitivity mapping (used by the FRF pipeline).
- `src/signal_processing.py`: CSV parsing and shared signal utilities (used by both pipelines).
- `src/frf_estimation.py` + `src/frf-estimation.py` (legacy): FRF computation and plotting utilities.
- `src/stability_measurement.py`: stability pipeline (spectrograms + chatter detection + plotting).

See also:

- `docs/onboarding.md` for repository setup and navigation.
- `docs/architecture.md` for a component map.

## Prerequisites

- Python `3.12+`
- `pip` and `venv` (standard with Python)

## Setup

From the repository root:

1. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

## Run the project (main pipeline)

### Default behavior

The project expects to be run from the **repository root**. The main entry point is:

```powershell
python -m src.main
```

By default, `src/main.py` loads:

- `config/example_experiment_frf.json`

and dispatches based on `analysis_type` inside that JSON.

### Running stability measurement

`src/main.py` does **not** accept command-line arguments; it uses `DEFAULT_CONFIG` hard-coded in `src/main.py`.

To run stability, either:

- edit `src/main.py`’s `DEFAULT_CONFIG` to point at `config/example_experiment_stability.json`, or
- update the JSON file currently referenced by `DEFAULT_CONFIG` so that it sets `analysis_type` to `"stability_measurement"` and provides `file_paths`.

## Config-driven usage

### FRF estimation (`analysis_type: "frf"`)

The FRF pipeline uses the JSON keys below (see `config/example_experiment_frf.json`):

- `analysis_type`: must be `"frf"`
- `file_paths_x`, `file_paths_y`: lists of CSV paths to analyze
- `channel_mapping`: maps **sensor names** to **analog channel indices** (`0`-based, expected to be present in `A0..A7`)
- `sensitivity_mapping`: maps those same **sensor names** to keys in `config/sensitivities.json` (scalars applied as `raw_signal * sensitivity`)
- `frf_pairs`: which FRFs to compute; each pair has:
  - `input`: signal name expected to exist in the loaded data (commonly `F_x` / `F_y`)
  - `output`: signal name expected to exist in the loaded data (commonly `x_tool` / `y_tool`)

What you should expect while it runs:

- It loads each CSV from `file_paths_x + file_paths_y`.
- For each `frf_pair`, it computes `H(f)` (Welch CSD-based FRF) for that input/output pair.
- If the required signals are missing from the loaded data, that pair is skipped with a message.
- It prints how many frequency bins were computed.
- It will plot a Bode diagram for the first computed pair (if the legacy plotting function is available).

### Stability measurement (`analysis_type: "stability_measurement"`)

The stability pipeline is configured by:

- `analysis_type`: must be `"stability_measurement"`
- `file_paths`: list of CSV paths to analyze

Important: in the current code, the stability pipeline is run via `process_stability_measurement(csv_path, show_plot=True)` and does not use `channel_mapping` / `sensitivity_mapping` from the JSON config.

So the stability pipeline expects the CSV channels (`A0..A7`) to already correspond to the sensor wiring assumed by the legacy derived-signal equations (`calculate_derived_signals`).

Filename assumptions used by stability detection:

- Direction (`x` vs `y`) is parsed from the filename by `parse_filename`, which expects a pattern like:
  - `yyyy-mm-dd_hh-mm-ss_X<...>_Y<...>_Z<...>_<direction>.csv`
  - where `<direction>` is `x` or `y` (an optional suffix is allowed)
- Depth-of-cut range (used to compute “cutting depth at chatter start”) is parsed from filenames containing:
  - `Ap_<start>to<end>` (e.g. `Ap_1.0to2.5`)

## CSV format expected by the loaders

Both pipelines rely on `load_csv_file()` from `src/signal_processing.py` (via the legacy wrapper in `src/frf_estimation.py`).

Your CSV should satisfy:

1. The **first line** contains sample rate information in the form:
   - `Samples per sec. = XXXXX/1`
2. The loader skips the next **5 header rows** (`skiprows=5`) and then reads numeric data.
3. The numeric data section is comma-separated and has **8 or 9 columns total**:
   - `Time, A0, A1, A2, A3, A4, A5, A6` (+ optional `A7`)
4. Rows with non-numeric values (or NaN/Inf) are rejected (the loader coerces to numeric and drops rows with NaNs).

## Running from PyCharm

1. Create a new Python run configuration that uses `src.main` as a module (equivalent to `python -m src.main`).
2. Set the **Working directory** to the repository root (`cnc-frf-estimation/`).

This matters if you use relative paths inside experiment JSON.

## Troubleshooting

- `FileNotFoundError`: the CSV path(s) in your JSON config don’t exist at runtime (or the first-line sample-rate string isn’t present in the file you pointed to).
- `ValueError: Expected 8 or 9 columns`: your CSV data rows don’t have `Time + A0..A6 (+ optional A7)` with comma separation.
- `Could not find sample rate in CSV header`: your CSV’s first line doesn’t match the expected `Samples per sec. = .../1` pattern.
- FRF skips FRF pairs: the `input`/`output` names in `frf_pairs` don’t exist in the loaded data for your `channel_mapping`/`sensitivity_mapping` setup.

