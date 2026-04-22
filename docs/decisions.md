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

**Decision:** Add a “Save as default” action under FRF → Sensor configuration that stores the mapping and sensitivity values in `config/app_settings.json` (via `AppSettings`) and auto-applies them on startup.

**Alternatives considered:**
- Overwriting `config/config_before_cabinet_assembly.json` to make it the new “default”
- Storing a separate `config/default_sensor_config.json` file

**Consequences:**
- Defaults become per-user/per-workstation (appropriate for wiring and local calibration), while experiment configs remain shareable templates.
- `app_settings.json` now includes more than folder paths, but remains a single place for user preferences.
