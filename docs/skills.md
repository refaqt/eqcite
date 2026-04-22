# Project skills

Reusable **project-specific** patterns for this codebase. Agents and humans should check here before implementing and update here when a pattern changes.

**Scope:** Patterns that only make sense in *this* repository (paths, modules, workflows, conventions). Not general programming tutorials.

**Maintenance:** When a skill is wrong or obsolete, edit or remove it in the same effort as the code change.

---

## [Example — replace with a real skill]
**When to use:** Placeholder until the first real entry is added.
**Pattern:**
```text
# Remove this section when you add the first real skill.
```
**Gotchas:** Delete the example block to avoid noise in search.
**Last used:** —

---

## Run the Qt GUI
**When to use:** You want to configure and run FRF/stability interactively (instead of editing JSONs by hand).
**Pattern:**
```bash
python -m src.gui.app
```
**Gotchas:** Run from the repo root so `config/` paths resolve. If PySide6 isn’t installed yet, install via `pip install -r requirements.txt`.
**Last used:** 2026-04-22

---

## Save/load analysis folder format
**When to use:** You want a portable snapshot of an analysis (inputs + GUI state) plus exported outputs (figures/tables).
**Pattern:**
```text
<analysis-folder>/
  analysis.json
  results/
    figures/
    tables/
    logs/
```
**Gotchas:**
- Sensitivities are global and auto-saved to `config/sensitivities.json` (not duplicated into each analysis folder).
- GUI save exports **FRF** artifacts from the currently computed results:
  - `results/figures/bode_displacement_force.png`
  - `results/figures/bode_acceleration_force.png`
  - `results/tables/frf_x_compliance.csv`, `results/tables/frf_x_accelerance.csv` (and `frf_y_*` if Y was computed)
  - `results/logs/export_log.json`
**Last used:** 2026-04-22

## Entry template

```markdown
## [Skill name]
**When to use:** [One sentence]
**Pattern:**
```[language]
// minimal example
```
**Gotchas:** [Any known edge cases or pitfalls]
**Last used:** [Date or task reference]
```

---

## Simulation config smoke-run
**When to use:** Quickly verify the FRF pipeline works end-to-end without needing access to measurement CSV paths.
**Pattern:**
```python
import json
from pathlib import Path

cfg = json.loads(Path("config/config_before_cabinet_assembly.json").read_text(encoding="utf-8"))
cfg["analysis_type"] = "frf"
cfg["data_source"] = "simulation"
cfg["simulation"] = {
    "fs": 8000,
    "T": 0.2,
    "model": {"mass": 8.0, "stiffness": 1e6},
    "hammer": {"force": 50.0, "bandwidth_hz": 1000.0},
}

tmp = Path("config/_tmp_sim_config.json")
tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

from src.main import run_frf_pipeline
run_frf_pipeline(tmp, Path("config/sensitivities.json"))
```
**Gotchas:** Welch settings must be valid for short signals; `compute_frf_from_signals()` clamps `nperseg/noverlap` for robustness.
**Last used:** 2026-04-22

---

## Matplotlib trace toggles in FRF GUI
**When to use:** You have multiple overlaid traces in a `MatplotlibFigureWidget` (Qt GUI) and want users to select/deselect lines without building extra Qt controls.
**Pattern:**
```python
# After plotting lines and creating legends:
for ax in fig.axes:
    leg = ax.get_legend()
    if leg is None:
        continue
    for leg_line, orig_line in zip(leg.get_lines(), ax.get_lines()):
        leg_line.set_picker(True)
        leg_line.set_pickradius(5)
        pick_map[leg_line] = orig_line

def on_pick(event):
    line = pick_map.get(event.artist)
    if line is None:
        return
    visible = not line.get_visible()
    line.set_visible(visible)
    event.artist.set_alpha(1.0 if visible else 0.2)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("pick_event", on_pick)
```
**Gotchas:**
- Build the legend after plotting lines; otherwise there’s nothing to pick.
- If you clear/redraw the figure, rebuild the pick-map (the old artists are gone).
- Keep labels unique enough (prefix with `X:` / `Y:`) so the legend remains readable.
**Last used:** 2026-04-22 (FRF GUI intermediate plots)
