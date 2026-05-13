# Architecture

High-level map of this repository. Update when layout, entry points, or major modules change.

## Purpose

Python tooling for **frequency response (FRF) estimation** and **stability / chatter-related measurement** from time-series data (CSV), plus utilities for **data acquisition** (e.g. serial devices). Broader product specs live under `docs/specs/`.

## Repository layout

| Path | Role |
|------|------|
| `src/` | Main Python package: pipelines, signal/FRF logic, stability analysis |
| `src/main.py` | CLI-style entry: loads JSON experiment config, runs FRF or stability pipeline |
| `src/frf_estimation.py` | Canonical FRF computation and plotting utilities |
| `src/load_data.py` | CSV/simulation loading: new **experiment schema** builds named variables from catalog + mapping + expressions; legacy path uses `sensitivities.json` + `sensor_model` |
| `src/stability_measurement.py` | Stability measurement / spectrogram-style analysis on CSV cycles |
| `src/stability_lobe_diagram.py` | Stability lobe diagram related logic (see module for current scope) |
| `src/expression_eval.py` | Restricted AST evaluation for experiment expression variables |
| `src/experiment_config.py` | Experiment schema helpers: unit normalization, variable kind classification, legacy migration |
| `src/gui/pages/frf_model.py` | `FrfInputModel` shared between FRF pages |
| `src/gui/pages/frf_experiment_config.py` | **Experiment configuration** Qt page (Mapping / Sensors / Expressions / Input–Output) |
| `src/sensor_model.py` | Legacy sensor geometry transforms (used when old channel-mapping pipeline is active) |
| `src/signal_processing.py` | Shared signal processing |
| `src/simulate_system.py` | Simulation helpers (importable; driven by config) |
| `src/smoke_test.py` | Minimal non-pytest smoke checks for core flows |
| `src/gui/` | Qt (PySide6) desktop GUI (config + results + save/load) |
| `src/legacy/` | Older scripts kept for reference |
| `src/tools/` | Standalone tools (e.g. acquisition UI, G-code helpers, channel fill) |
| `config/` | Example experiment JSON and `sensitivities.json` |
| `docs/` | Architecture, ADRs, skills, mistakes, onboarding, specs |

## Data and configuration flow

1. **Experiment config** (JSON under `config/`, snapshot in `analysis.json`) sets `analysis_type`, `data_source`, file paths or simulation params, and the **FRF experiment schema** when `sensor_catalog` is non-empty: sensor catalog (with `N` vs `m/s²` units), channel mapping rows, optional expressions, per-direction `analysis_io` (force input + acceleration output), and optional `simulation_signal_map` for simulation runs.
2. **Legacy configs** without `sensor_catalog` still use `channel_mapping` + `sensitivity_mapping` + `config/sensitivities.json` and `sensor_model`-derived `F_x` / `x_tool` style signals inside `load_data`.
3. **Data source**:
   - `file`: CSV measurements are loaded through `load_data` using paths from the experiment config.
   - `simulation`: `simulate_system` generates time-series; `load_data` exposes canonical keys and/or user aliases from `simulation_signal_map`.

### Simulation parameter semantics

- `simulation.model.mass`: kg
- `simulation.model.stiffness`: N/m
- `simulation.model.damping`: damping ratio ζ (dimensionless). The simulator derives the damping coefficient internally as \(c = 2\,\zeta\,\sqrt{k m}\).

## GUI architecture (PySide6)

- **Entry point**: `python -m src.gui.app`
- **Core components**:
  - `src/gui/main_window.py`: top-level shell (tabs + toolbar)
  - `src/gui/state.py`: `AnalysisState` (in-memory session state)
  - `src/gui/persistence.py`: save/load `analysis.json` + app settings (`config/app_settings.json`)
  - `src/gui/plot_widgets.py`: Matplotlib embedding for result tabs
- **FRF filtering / analysis** (`src/gui/pages/frf.py` → `FrfFilteringPage`): **`bias_window_size`** (samples) configures sliding min-variance bias removal and the noise-variance segment; per-axis **filter chains** apply **optional Wiener** to force (`u`) and response before windowing; **`filtering.params.wiener.mysize`** configures SciPy’s local Wiener window only. Both are persisted in the experiment snapshot (`experiment_cfg`), with `bias_window_size` also duplicated at the top level of `analysis.json` for readability.
- **FRF results tabs** (`src/gui/pages/frf.py`):
  - `Raw signals (mosaic)`: plots raw channels A0..A7 (per direction when available)
  - `Time signals (processing steps)`: plot with per-direction measurement selection + Force/Response selector, and **clickable legend toggles** for raw/bias/window/windowed/filtered traces
  - `FFT magnitude`: FFT magnitude view; supports optional **Average** overlay computed in complex FFT space
  - `Coherence`: \(\gamma^2(f)\) vs frequency; supports optional **Average** overlay derived from averaged spectra terms
  - `Force PSD`: force excitation PSD vs frequency; supports optional **Average** overlay from averaged PSD
  - `Real(FRF)`: real part of compliance and accelerance; supports optional **Average** overlay from averaged complex FRF
  - `Bode (displacement/force)` and `Bode (acceleration/force)`: magnitude/phase plots; supports optional **Average** overlay from averaged complex FRF

### Multi-measurement FRF behavior

- FRF input supports selecting **multiple CSV files per direction** (X and/or Y). X and Y are treated as **independent** lists (no pairing).
- Each results tab has **X and Y measurement dropdowns** labeled by filename, which control which single measurement is displayed for that direction.
- Frequency-domain tabs include an **Average** checkbox that overlays an additional **black** trace computed by averaging the **complex frequency components first** (then converting to magnitude/phase/real as appropriate).
- **Persistence**:
  - An analysis is saved as a **folder** containing `analysis.json` plus `results/figures`, `results/tables`, and `results/logs`.
  - When saving from the GUI, FRF runs export their current figures + tables into `results/` so the folder is self-contained.
  - Global sensitivities (`config/sensitivities.json`) apply only to the **legacy** loading path; the new experiment schema stores sensitivity on each catalog row.
  - Per-user GUI defaults (folders, **`default_experiment_config`** for the FRF experiment schema, default filtering, last-selected input file paths) are stored in `config/app_settings.json`.

### FRF analysis settings

- In the FRF GUI (`src/gui/pages/frf.py`), the sidebar page **Experiment configuration** defines sensors, mapping, expressions, and **per-direction** analysis I/O (force-only inputs, acceleration-only outputs). The page **Analysis settings** contains:
  - **Filtering**: bias removal, then optional **Wiener** on force and/or response (independent chains), then windowing (applied before FRF estimation).
  - **Raised-cosine windowing** (Analysis settings): peak-relative **rolloff time**, **window max start**, and **window max end** define the plateau and cosine ramps; legacy `window_pre_seconds` / `window_post_seconds` / `window_beta` are migrated on load.
  - **Estimation**: FRF estimator selection (`H1`, `H2`, `Hv`, default `Hv`), passed into `src/frf_estimation.py::compute_frf_from_signals()` via `params['estimator_type']`.

## Embedded / firmware

Project context includes **Arduino + Python** workflows. This clone may not contain `.ino` sources; acquisition here is centered on **Python + pyserial** (e.g. `src/tools/data-acquisition.py` for DataQ DI-2108). Add a firmware subsection here when Arduino sketches live in-repo.

## Dependencies (runtime)

See `requirements.txt` (e.g. NumPy, SciPy, pandas, matplotlib, pyserial, pyFRF).
