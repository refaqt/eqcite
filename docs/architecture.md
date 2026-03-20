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
| `src/simulate_system.py` | Simulation helpers |
| `src/legacy/` | Older scripts kept for reference |
| `src/tools/` | Standalone tools (e.g. acquisition UI, G-code helpers, channel fill) |
| `config/` | Example experiment JSON and `sensitivities.json` |
| `docs/` | Architecture, ADRs, skills, mistakes, onboarding, specs |

## Data and configuration flow

1. **Experiment config** (JSON under `config/`, paths referenced by `src/main.py` defaults) sets `analysis_type` (`frf` or `stability_measurement`), input file lists, and FRF input/output pairs.
2. **Sensitivities** (`config/sensitivities.json`) scale or map channels when loading measurements.
3. **CSV measurements** are loaded through `load_data` using paths from the experiment config.

## Embedded / firmware

Project context includes **Arduino + Python** workflows. This clone may not contain `.ino` sources; acquisition here is centered on **Python + pyserial** (e.g. `src/tools/data-acquisition.py` for DataQ DI-2108). Add a firmware subsection here when Arduino sketches live in-repo.

## Dependencies (runtime)

See `requirements.txt` (e.g. NumPy, SciPy, pandas, matplotlib, pyserial, pyFRF).
