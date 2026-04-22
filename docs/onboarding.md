# Onboarding

How to set up, run, and find your way around this repository.

## Prerequisites

- **Python 3.12+** (3.10+ likely works; match your team standard)
- **Git**
- Optional: **Arduino IDE / toolchain** if you work with embedded firmware referenced by your workflow (firmware may live outside this repo)

## Setup

1. Clone the repository.
2. Create and activate a virtual environment (example):

   ```bash
   python -m venv venv
   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the GUI

Run from the repository root:

```bash
python -m src.gui.app
```

- The GUI supports FRF and stability tabs, plus saving/loading analyses.
- App-level settings (like default save folder) are stored in `config/app_settings.json`.

## Running the main pipeline

The package expects to be run **from the repository root** so imports and default paths resolve (`config/`, etc.).

```bash
python -m src.main
```

Defaults are set in `src/main.py` (`DEFAULT_CONFIG`, `DEFAULT_SENS`). Edit `config/example_experiment_frf.json` or `config/example_experiment_stability.json` and point `DEFAULT_CONFIG` if you need a different experiment file.

- **`analysis_type`: `"frf"`** — FRF estimation over configured CSV paths and FRF pairs.
- **`analysis_type`: `"stability_measurement"`** — Stability measurement flow over configured files.

## Tools under `src/tools/`

Run individual scripts as modules or scripts depending on how each file defines `main` (check the file header). Examples include data acquisition (`data-acquisition.py`), G-code helpers, and utilities.

## Where to read next

| Doc | Use |
|-----|-----|
| `docs/architecture.md` | Module map and data flow |
| `docs/skills.md` | Project-specific reusable patterns |
| `docs/specs/` | Detailed specifications |
| `docs/decisions.md` | Past technical decisions (ADRs) |
| `docs/mistakes.md` | What went wrong and how to avoid it |

## IDE notes

There is a nested `.idea` under `src/`; team preference may be to open the **repo root** in the IDE for consistent paths and `python -m src.main`.
