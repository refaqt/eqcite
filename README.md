# cnc-frf-estimation

Python tooling for **frequency response (FRF) estimation** and **stability / chatter-related measurement** from time-series CSV files.

## What’s inside

- `src/main.py`: main entry point that runs either the FRF pipeline (`analysis_type: "frf"`) or the stability pipeline (`analysis_type: "stability_measurement"`), based on the JSON config it loads.
- `src/load_data.py`: data loading for `data_source: "file"` **and** `data_source: "simulation"`, plus channel/sensitivity mapping (used by the FRF pipeline).
- `src/signal_processing.py`: CSV parsing and shared signal utilities (used by both pipelines).
- `src/frf_estimation.py`: FRF computation and plotting utilities (interactive tkinter UI removed; driven by `src/main.py`).
- `src/stability_measurement.py`: stability pipeline (spectrograms + chatter detection + plotting).
- `src/simulate_system.py`: simulation helpers (driven by config; importable).

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

- `config/config_before_cabinet_assembly.json`

and dispatches based on `analysis_type` inside that JSON.

### Quickstart: tap test FRF from your own CSV (recommended workflow)

Goal: load your tap-test CSV(s), apply the correct sensor sensitivities, and see the Bode plots.

1. Pick a config file under `config/` to start from
   - Recommended: copy `config/config_before_cabinet_assembly.json` to a new file next to it, e.g. `config/my_tap_test.json`.

2. Edit your config JSON (`config/my_tap_test.json`)
   - Set:
     - `analysis_type` to `"frf"`
     - `data_source` to `"file"`
   - Set your CSV paths:
     - `file_paths_x`: list of **x-direction** tap-test CSVs (the pipeline uses the first file)
     - `file_paths_y`: list of **y-direction** tap-test CSVs (the pipeline uses the first file)
   - Make sure `frf_pairs_x` and `frf_pairs_y` match the signals you expect after loading. Typical defaults:
     - x: `{"input": "F_x", "output": "x_tool"}`
     - y: `{"input": "F_y", "output": "y_tool"}`

3. Verify sensor sensitivities
   - Sensitivity values live in `config/sensitivities.json` (shared across experiments).
   - Your experiment config must map sensors to sensitivity keys via `sensitivity_mapping`.
   - The loader applies: `physical = raw_volts * sensitivity_value`.

4. Verify channel mapping
   - In your experiment config, set `channel_mapping` to map sensor names to analog channel indices (`0`-based for `A0..A7`).
   - If a required channel is missing, the program raises a `ValueError` and stops (so you don’t silently compute wrong FRFs).

5. Point `src/main.py` at your config (currently hard-coded)
   - Open `src/main.py` and set `DEFAULT_CONFIG` to your JSON (e.g. `config/my_tap_test.json`).

6. Run

```powershell
python -m src.main
```

Expected result:
- Console prints `Computed x FRF ...` and `Computed y FRF ...`
- A combined Bode plot window appears (Matplotlib).

### Running stability measurement

`src/main.py` does **not** accept command-line arguments; it uses `DEFAULT_CONFIG` hard-coded in `src/main.py`.

To run stability, either:

- edit `src/main.py`’s `DEFAULT_CONFIG` to point at your stability config JSON, or
- update the JSON file currently referenced by `DEFAULT_CONFIG` so that it sets `analysis_type` to `"stability_measurement"` and provides `file_paths`.

## Config-driven usage

### FRF estimation (`analysis_type: "frf"`)

The FRF pipeline uses the JSON keys below (see `config/example_experiment_frf.json`):

- `analysis_type`: must be `"frf"`
- `file_paths_x`, `file_paths_y`: lists of CSV paths to analyze
- `channel_mapping`: maps **sensor names** to **analog channel indices** (`0`-based, expected to be present in `A0..A7`)
- `sensitivity_mapping`: maps those same **sensor names** to keys in `config/sensitivities.json` (scalars applied as `raw_signal * sensitivity`)
- `frf_pairs_x` (optional): which FRFs to compute from `file_paths_x`; each pair has:
  - `input`: signal name expected to exist in the loaded data (commonly `F_x` / `F_y`)
  - `output`: signal name expected to exist in the loaded data (commonly `x_tool` / `y_tool`)
- `frf_pairs_y` (optional): which FRFs to compute from `file_paths_y`; each pair has:
  - `input`: signal name expected to exist in the loaded data (commonly `F_x` / `F_y`)
  - `output`: signal name expected to exist in the loaded data (commonly `x_tool` / `y_tool`)
- `frf_pairs` (fallback): if `frf_pairs_x`/`frf_pairs_y` are not present, `frf_pairs` is used for both axes; each pair has:
  - `input`: signal name expected to exist in the loaded data (commonly `F_x` / `F_y`)
  - `output`: signal name expected to exist in the loaded data (commonly `x_tool` / `y_tool`)

What you should expect while it runs:

- It loads the first CSV from `file_paths_x` using `frf_pairs_x` (or `frf_pairs` as fallback) and computes the corresponding `H(f)` values.
- It loads the first CSV from `file_paths_y` using `frf_pairs_y` (or `frf_pairs` as fallback) and computes the corresponding `H(f)` values.
- It plots a combined Bode diagram for the x- and y-FRFs and calls `plt.show()` once so the plot windows appear.
- If the required signals are missing from the loaded data, that pair is skipped with a message.
- It prints how many frequency bins were computed.
- It plots a combined Bode diagram using the first computed FRF pair from each axis.

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

## Simulation mode (optional)

For quick development/testing without measurement files:

- Set in your experiment JSON:
  - `data_source: "simulation"`
  - `analysis_type: "frf"`
  - `simulation`: a dict with (example):
    - `fs`, `T`
    - `model: {mass, stiffness, damping}` where `damping` is **damping ratio** ζ (dimensionless; optional; default ζ=0.02)
    - `hammer: {force, bandwidth_hz}`

Then run `python -m src.main` (with `DEFAULT_CONFIG` pointing at that JSON). This produces synthetic `F_x/F_y` and tool outputs and runs the same FRF computation + plotting.

