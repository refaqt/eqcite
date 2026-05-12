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

---

## Persist last-selected GUI file paths
**When to use:** You want the GUI to remember user-selected input file paths across restarts (without tying them to a specific saved analysis folder).
**Pattern:**
```python
# 1) Add fields to AppSettings (src/gui/persistence.py) and persist in config/app_settings.json
# 2) Make the page emit a signal only when the value changes
# 3) In MainWindow, restore under a guard flag so restore doesn't trigger saves

# MainWindow:
self._restoring_settings = True
try:
    self._frf_root.set_file_paths(app_settings.last_frf_x_file, app_settings.last_frf_y_file)
    self._stab_root.set_file_path(app_settings.last_stability_file)
finally:
    self._restoring_settings = False

def _on_frf_file_paths_changed(self, x: str, y: str) -> None:
    if self._restoring_settings:
        return
    app_settings.last_frf_x_file = x
    app_settings.last_frf_y_file = y
    save_app_settings(repo_root, app_settings)
```
**Gotchas:**
- Guard restores to avoid writing settings while populating UI on startup.
- Emit only on changes to avoid frequent writes (e.g. on `editingFinished`).
**Last used:** 2026-04-23

---

## Multi-measurement selectors + complex averaging overlays (FRF GUI)
**When to use:** You’re extending the FRF GUI to handle repeated measurements per direction and want both per-measurement selection and a physically correct “average” in frequency space.
**Pattern:**
```python
# 1) Represent per-direction measurements as a list, keep per-measurement filepath
self._x_measurements: list[_DirectionalMeasurement] = []
self._y_measurements: list[_DirectionalMeasurement] = []

# 2) In each results tab, provide X/Y dropdowns (filename labels)
#    and store per-tab selected indices.
self._tab_selectors: dict[str, tuple[QComboBox, QComboBox]] = {}
self._tab_sel_idx: dict[str, tuple[int, int]] = {}

# 3) For averaging: average complex frequency components first.
H_avg = np.mean(np.stack(H_list, axis=0), axis=0)  # complex mean
mag = np.abs(H_avg)
phase_deg = np.unwrap(np.angle(H_avg)) * 180 / np.pi

# 4) For coherence: average spectra terms, then compute coherence.
Pur_avg = np.mean(np.stack(Pur_list, axis=0), axis=0)  # complex
Puu_avg = np.mean(np.stack(Puu_list, axis=0), axis=0)  # real
Prr_avg = np.mean(np.stack(Prr_list, axis=0), axis=0)  # real
coh = (np.abs(Pur_avg) ** 2) / ((Puu_avg * Prr_avg) + 1e-30)
```
**Gotchas:**
- Don’t average magnitudes/phases directly; that can be wrong when phases differ.\n- Ensure all measurements use the same frequency grid (same sampling + Welch settings). If grids can differ, you must align/interpolate real/imag parts before averaging.\n- For FFT magnitude averaging, store/average the **complex rFFT bins** and only then take `abs()`.
**Last used:** 2026-05-12

---

## Wiener + bias (FRF GUI and shared `signal_processing`)
**When to use:** You touch Wiener filtering, bias removal, or FRF filtering config.
**Pattern:**
- Bias / quiet segment: `remove_bias_window(x, bias_window_size)` → use returned index and length with `calculate_noise_power_from_window` on the **same** bias-corrected array passed to `apply_wiener_filter_legacy`.
- Noise statistic: `calculate_noise_power_from_window` = segment variance (`ddof=0`), not `mean(x**2)`.
- SciPy window: `apply_wiener_filter_legacy(..., mysize)` with `mysize` independent of `bias_window_size` (coerced odd internally).
**Gotchas:** Missing `filtering.params.wiener.mysize` or `bias_window_size` in the GUI filtering dict raises at compute time by design.
**Last used:** 2026-05-12

---

## Scale window overlay in time-signal plots
**When to use:** You plot a window function alongside time-domain processing steps and want it visible at the same amplitude scale as the signal being windowed.
**Pattern:**
```python
# In the processing-step builder (before applying the window):
scale = float(np.max(np.abs(pre_window_signal))) if len(pre_window_signal) else 1.0
if not np.isfinite(scale) or scale <= 0:
    scale = 1.0
steps.append(_SignalStep("window", window * scale))  # display only
windowed = pre_window_signal * window               # computation unchanged
```
**Gotchas:**
- Use `max(abs(.))` so the overlay is visible even for negative-dominant signals.
- Keep the actual windowing math (`signal * window`) unchanged; only scale the displayed `window` trace.
**Last used:** 2026-04-23
