# Architecture decisions (ADRs)

Lightweight log of **non-obvious** technical choices. Add an entry when picking libraries, data formats, APIs, or structure where alternatives existed.

## Format

```markdown
## [Short title] — [Date]
**Context:** [What problem were we solving?]
**Decision:** [What did we decide?]
**Alternatives considered:** [What else was on the table?]
**Consequences:** [What does this make easier or harder?]
```

---

## Wiener noise variance + independent bias vs local window — 2026-05-12
**Context:** The legacy Wiener path used `mean(x²)` on the raw trace while filtering bias-corrected data, which inflated `noise` vs SciPy’s local variance and could zero the output. Users also need separate control of the sliding “quiet segment” length and SciPy’s `mysize`.
**Decision:** Use **`np.var(..., ddof=0)`** for noise power (same scale as `scipy.signal.wiener` `lVar`). Estimate noise on the **bias-corrected** signal. Implement sliding **min-variance** bias removal in `signal_processing.remove_bias_window`. Expose **`bias_window_size`** in the FRF GUI analysis block (and duplicate to top-level `experiment_cfg` for snapshots); require **`filtering.params.wiener.mysize`** when the Wiener step runs. Coerce Wiener **`mysize`** to **odd** ≥ 3. Batch params gain explicit **`wiener_mysize`** independent of **`bias_window_size`**.
**Alternatives considered:** Keeping backward-compatible fallbacks when `mysize` or `bias_window_size` are missing in saved JSON; rejected until explicitly requested.
**Consequences:** Older saved analyses without the new fields may fail at run time until re-saved from the GUI. CLI configs should include `bias_window_size` / `wiener_mysize` (and optional `filtering` mirroring the GUI) for parity.

## Qt (PySide6) desktop GUI + analysis folder format — 2026-04-22
**Context:** The existing workflows were driven by JSON files and CLI-style entry points, but iterating on configurations and intermediate plots was becoming cumbersome. We need a GUI that can later evolve into the primary way to drive FRF and stability analysis.\n\n**Decision:** Implement a desktop GUI using **Qt via PySide6**, with an **analysis folder** persistence format: `analysis.json` snapshot + `results/figures` + `results/tables`. Keep global sensor sensitivities persisted in `config/sensitivities.json`.\n\n**Alternatives considered:**\n- Staying with JSON-only configuration\n- Reintroducing Tkinter for a “no extra dependency” GUI\n- Web UI (e.g. Streamlit)\n\n**Consequences:**\n- Adds a runtime dependency (`PySide6`).\n- Enables a scalable UI (tabs, sidebars, embedded Matplotlib figures) and a durable save/load story that can be GUI-driven later.\n- Stability pipeline remains legacy-mapping-dependent initially; the GUI will surface this and a later refactor can unify mapping via `load_data`.

## Auto-export FRF artifacts on GUI save — 2026-04-22
**Context:** The GUI created analysis folders with `results/figures`, `results/tables`, and `results/logs`, but those folders were often empty because the GUI only saved `analysis.json` and kept plots/results in-memory.

**Decision:** On **Save analysis** in the GUI, export the current FRF figures (PNG), FRF tables (CSV), and an export log (`export_log.json`) into the analysis folder’s `results/` subtree.

**Alternatives considered:**
- Track placeholder files (e.g. `.gitkeep`) to keep directories visible in git
- Only export via explicit “Export…” buttons (separate from save)

**Consequences:**
- Saved analysis folders become self-contained artifacts that can be archived/shared.
- Export depends on having run “Generate results” at least once; when absent, we still write a log with warnings.

## Persist default FRF sensor configuration in app settings — 2026-04-22
**Context:** The FRF GUI allows editing channel mapping and sensitivity mapping, and sensitivity values. Users want a one-click way to persist a preferred configuration and have it restored automatically on the next launch without modifying the shared experiment config templates under `config/`.

**Decision:** Add a “Save as default” action under FRF → **Experiment configuration** that stores the full experiment schema blob in `config/app_settings.json` (via `AppSettings.default_experiment_config`) and auto-applies it on startup. Legacy triples (`default_channel_mapping`, `default_sensitivity_mapping`, `default_sensitivities`) remain readable for migration only.

**Alternatives considered:**
- Overwriting `config/config_before_cabinet_assembly.json` to make it the new “default”
- Storing a separate `config/default_sensor_config.json` file

**Consequences:**
- Defaults become per-user/per-workstation (appropriate for wiring and local calibration), while experiment configs remain shareable templates.
- `app_settings.json` now includes more than folder paths, but remains a single place for user preferences.

## Simulation config `model.damping` means damping ratio ζ — 2026-04-22
**Context:** The simulation config key `simulation.model.damping` was being interpreted as a damping coefficient \(c\) (N·s/m) in `src/simulate_system.py`, but users expect to specify a damping ratio ζ (dimensionless) when configuring a mass-spring-damper model.

**Decision:** Treat `simulation.model.damping` as **damping ratio ζ** and derive the damping coefficient internally as \(c = 2\,\zeta\,\sqrt{k m}\) (equivalently \(2 m \omega_n \zeta\)). Default to ζ=0.02 when the key is omitted.

**Alternatives considered:**
- Add a new key `model.damping_ratio` while keeping `model.damping` as coefficient \(c\) for backward compatibility.
- Keep accepting \(c\) and rename the key to `model.damping_coeff`.

**Consequences:**
- This is a **semantic breaking change** for anyone who previously supplied \(c\) under `model.damping`.
- The simulation configuration matches typical vibration/controls practice (ζ is easier to specify and compare across systems).

## Toggle FRF GUI traces via Matplotlib legend picking — 2026-04-22
**Context:** We want to show raw signals and multiple processing steps (bias/window/filter variants) in the FRF GUI without adding extra plotting dependencies, and still allow users to interactively select/deselect traces.

**Decision:** Use **Matplotlib legend click toggling** (pick events) in the embedded Qt figures (see `src/gui/pages/frf.py`) so clicking a legend entry hides/shows the corresponding trace (with alpha change on the legend item).

**Alternatives considered:**
- Adding a dedicated Qt trace checklist UI next to each plot (more widgets + more state to keep in sync)
- Switching to a different plotting library with built-in trace toggles (larger dependency/architecture change)

**Consequences:**
- Keeps the GUI lightweight and consistent with existing Matplotlib embedding.
- Requires re-wiring pick handlers after re-plotting, but avoids building a parallel UI for visibility state.

## Multi-measurement FRF + complex-domain averaging — 2026-05-12
**Context:** Users often record repeated tap tests per direction (X/Y). We need to compare individual measurements under identical analysis settings and also compute a representative average trace. Averaging must be physically meaningful for complex frequency-domain quantities.

**Decision:** Support selecting **multiple measurement CSVs per direction** in the FRF GUI. In each results tab, let users select the displayed measurement via a **filename dropdown**. For frequency-domain results (FFT magnitude, Coherence, Force PSD, Real(FRF), Bode plots), add an **Average** checkbox that overlays an additional **black** trace computed by averaging **complex frequency components first** and only then computing magnitude/phase/real.\n\nFor coherence/PSD, average the underlying spectra terms on the frequency grid: \n- \u0304P_ur = mean(P_ur), \u0304P_uu = mean(P_uu), \u0304P_rr = mean(P_rr)\n- \u03b3\u00b2 = |\u0304P_ur|^2 / (\u0304P_uu \u0304P_rr)

**Alternatives considered:**\n- Average magnitude/phase directly (incorrect for complex data)\n- Use pyFRF’s multi-tap averaging path everywhere (larger refactor; GUI currently uses a lighter SciPy-based path)\n- Pair X and Y measurements by index (not desired; X/Y should be independent)

**Consequences:**\n- Additional UI state (per-tab measurement selection, per-tab average toggle).\n- Requires retaining complex FFT bins and Welch/CSD spectra terms in the GUI processing structure.\n- Produces averages that respect complex cancellation/phase, aligning with standard FRF practice.

## FRF experiment configuration schema + restricted expressions — 2026-05-12
**Context:** Channel mapping was tied to global `sensitivities.json` keys and hard-coded `sensor_model` transforms (`F_x`, `x_tool`, …). Users need editable sensors, mapping, optional math on mapped channels, and explicit per-direction FRF I/O without maintaining parallel “tool vs spindle” branches in the filtering UI.
**Decision:** Introduce `sensor_catalog`, `mapping`, `expressions`, `analysis_io` (X/Y each with force-only input and acceleration-only output), and optional `simulation_signal_map`. Implement expressions with a **whitelisted `ast`** evaluator (`src/expression_eval.py`) allowing `+ - * /`, unary `+/-`, and `sin`/`cos` only, referencing **mapping names** (v1). Persist GUI defaults as `AppSettings.default_experiment_config`. Migrate legacy JSON in-memory via `experiment_config.migrate_legacy_experiment_cfg`.
**Alternatives considered:** Adding `numexpr`/`asteval`; allowing expressions to reference other expressions (requires dependency ordering); keeping `sensor_model` in the default file path forever.
**Consequences:** `load_data` gains a second code path keyed on non-empty `sensor_catalog`. CLI `main.py` derives `frf_pairs_x/y` from `analysis_io` when the new schema is active. Stability analysis still uses its legacy path until unified.

## Force-only FRF input and acceleration-only output — 2026-05-12
**Context:** Displacement FRF is derived from accelerance via \(1 / -\omega^2\); mixing arbitrary physical types breaks compliance/accelerance interpretation.
**Decision:** Classify variables from catalog units (`N` vs `m/s²`) and expression AST homogeneity; expose only matching names in the Input/Output tab; validate `analysis_io` on save and at FRF run for the new schema.
**Alternatives considered:** Letting users pick any variable and soft-warning only; inferring units from expressions with mixed references.
**Consequences:** Users cannot select an arbitrary voltage trace as “output” without labeling the catalog row as `m/s²` (physical meaning remains user responsibility).
