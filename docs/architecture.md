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
| `src/load_data.py` | Measurement loading, sensitivities, config-driven column mapping |
| `src/stability_measurement.py` | Stability measurement / spectrogram-style analysis on CSV cycles |
| `src/stability_lobe_diagram.py` | Stability lobe diagram related logic (see module for current scope) |
| `src/sensor_model.py` | Sensor / model helpers |
| `src/signal_processing.py` | Shared signal processing |
| `src/simulate_system.py` | Simulation helpers (importable; driven by config) |
| `src/smoke_test.py` | Minimal non-pytest smoke checks for core flows |
| `src/gui/` | Qt (PySide6) desktop GUI (config + results + save/load) |
| `src/legacy/` | Older scripts kept for reference |
| `src/tools/` | Standalone tools (e.g. acquisition UI, G-code helpers, channel fill) |
| `config/` | Example experiment JSON and `sensitivities.json` |
| `docs/` | Architecture, ADRs, skills, mistakes, onboarding, specs |

## Data and configuration flow

1. **Experiment config** (JSON under `config/`, paths referenced by `src/main.py` defaults) sets `analysis_type` (`frf` or `stability_measurement`), `data_source` (`file` or `simulation`), input file lists (when `file`), simulation model params (when `simulation`), and FRF input/output pairs.
2. **Sensitivities** (`config/sensitivities.json`) scale or map channels when loading measurements.
3. **Data source**:\n+   - `file`: CSV measurements are loaded through `load_data` using paths from the experiment config.\n+   - `simulation`: `simulate_system` generates time-series, and `load_data` normalizes them into the same signals dict shape.

### Simulation parameter semantics

- `simulation.model.mass`: kg
- `simulation.model.stiffness`: N/m
- `simulation.model.damping`: damping ratio ζ (dimensionless). The simulator derives the damping coefficient internally as \(c = 2\,\zeta\,\sqrt{k m}\).

## GUI architecture (PySide6)

- **Entry point**: `python -m src.gui.app`
- **Core components**:
  - `src/gui/main_window.py`: top-level shell (tabs + toolbar + title/description fields)
  - `src/gui/state.py`: `AnalysisState` (in-memory session state)
  - `src/gui/persistence.py`: save/load `analysis.json` + app settings (`config/app_settings.json`)
  - `src/gui/plot_widgets.py`: Matplotlib embedding for result tabs
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
  - Global sensitivities are stored in `config/sensitivities.json` and are not duplicated per analysis folder.
  - Per-user GUI defaults (folders, default FRF sensor configuration, and last-selected input file paths) are stored in `config/app_settings.json`.

### FRF analysis settings

- In the FRF GUI (`src/gui/pages/frf.py`), the sidebar page **Analysis settings** contains:
  - **input/output selection**: choose which normalized signal families are used as FRF input and output (Force `F_x/F_y`, Tool `x_tool/y_tool`, or Spindle `x_spindle/y_spindle`).
  - **Filtering**: filter chains + parameters + windowing configuration (applied before FRF estimation).
  - **Estimation**: FRF estimator selection (`H1`, `H2`, `Hv`, default `Hv`), passed into `src/frf_estimation.py::compute_frf_from_signals()` via `params['estimator_type']`.

## Embedded / firmware

Project context includes **Arduino + Python** workflows. This clone may not contain `.ino` sources; acquisition here is centered on **Python + pyserial** (e.g. `src/tools/data-acquisition.py` for DataQ DI-2108). Add a firmware subsection here when Arduino sketches live in-repo.

## Dependencies (runtime)

See `requirements.txt` (e.g. NumPy, SciPy, pandas, matplotlib, pyserial, pyFRF).
