# Mistake log

Log errors, wrong approaches, and **prevention rules** so agents and humans do not repeat them. After each entry, consider copying **Prevention rule** bullets into `.cursorrules` under “Mistake prevention”.

## Format (copy per incident)

```markdown
## [Short description] — [Date]
**What happened:** [One or two sentences]
**Root cause:** [Why did this happen?]
**Fix applied:** [What was done to correct it?]
**Prevention rule:** [One concrete rule to avoid this in the future]
**Affected files:** [List of files involved]
```

---

## PowerShell pip command used invalid `&&` — 2026-03-20
**WARNING (repeated):** This mistake has now happened multiple times. Treat `&&` as invalid in this repository’s PowerShell workflow.
**What happened:** The `venv` dependency install command failed because `&&` is not a valid statement separator in PowerShell in this environment.
**Root cause:** I used shell-style `&&` chaining instead of PowerShell-safe command chaining.
**Fix applied:** Re-ran the installation using PowerShell `;` separators and then verified the venv import smoke test.
**Prevention rule:** In PowerShell, prefer `;` or separate commands; avoid `&&`.
**Affected files:** `requirements.txt`

## Incomplete escaped-quote fix — 2026-03-20
**What happened:** After fixing `src/load_data.py`, the smoke test failed again due to the same escaped-quote artifact in `src/signal_processing.py`.
**Root cause:** The escaped-quote issue was broader than the initially failing file, so fixing only `load_data.py` wasn’t sufficient.
**Fix applied:** Scanned `src/` for remaining `\"` artifacts, corrected the syntax in `src/signal_processing.py`, and reran `import src.main`.
**Prevention rule:** When a SyntaxError points to escaped quoting artifacts, scan all `src/*.py` for the same pattern before rerunning.
**Affected files:** `src/load_data.py`, `src/signal_processing.py`

---
## Stability smoke test used placeholder path — 2026-03-20
**What happened:** I attempted to run `process_stability_measurement()` with a non-existent placeholder filepath (`echo`) and got a cascading file-not-found error.
**Root cause:** Used an invalid test input when launching the smoke test command.
**Fix applied:** Re-ran the stability smoke test using the real default CSV path from `config/config_before_cabinet_assembly.json`.
**Prevention rule:** For pipeline smoke tests, always use a real filepath from the active config (or a known-good small fixture).
**Affected files:** `src/stability_measurement.py`, `src/frf_estimation.py`

---
## Stability smoke test window_duration too large — 2026-03-20
**What happened:** I ran the stability smoke test with `window_duration=1.0` on a file that only contained ~0.25s of data (5000 samples at 20kHz), causing `negative dimensions are not allowed` inside the rolling-window computation.
**Root cause:** Rolling-window code computed an invalid number of windows because the requested window length exceeded the available signal duration.
**Fix applied:** Re-ran the stability smoke test with `window_duration=0.2` (and `show_plot=False`), which completed successfully.
**Prevention rule:** When running stability/rolling-window code, ensure `window_duration` is small enough relative to the data duration (`len(time)/fs`) to keep the computed window count non-negative.
**Affected files:** `src/stability_measurement.py`

---

## Used bash heredoc in PowerShell — 2026-04-22
**What happened:** I tried to run a multi-line Python snippet using bash-style heredoc (`python - <<'PY' ...`) and it failed to parse in PowerShell.
**Root cause:** Assumed bash heredoc syntax would work in the current PowerShell environment.
**Fix applied:** Re-ran the command using `python -c` with an inline script string.
**Prevention rule:** On Windows/PowerShell, avoid bash heredocs; use `python -c`, a temporary `.py` file, or PowerShell-native piping.
**Affected files:** None (command-line only)

## Used `cd /d` (cmd syntax) in PowerShell — 2026-04-22
**What happened:** The GUI launch failed because `cd /d` is `cmd.exe` syntax and PowerShell treated `/d` as an invalid positional parameter.
**Root cause:** Mixed `cmd.exe` directory-change flags into a PowerShell command.
**Fix applied:** Switched to PowerShell-native `Set-Location -Path "..."` (or `cd "..."` without `/d`).
**Prevention rule:** In PowerShell, avoid `cmd.exe`-specific flags like `cd /d`; use `Set-Location` / `cd` with just the path.
**Affected files:** None (command-line only)

## Used bash-style heredoc in `git commit` under PowerShell — 2026-04-23
**What happened:** A `git commit -m "$(cat <<'EOF' ... EOF)"` command failed to parse because the bash heredoc syntax isn’t valid in PowerShell.
**Root cause:** Reused a bash-oriented commit-message pattern in a PowerShell terminal session.
**Fix applied:** Re-ran `git commit` using multiple `-m` flags (PowerShell-safe) to provide subject + body.
**Prevention rule:** In PowerShell terminals, avoid bash heredocs and command substitutions; use PowerShell here-strings or `git commit -m "subject" -m "body"`.
**Affected files:** None (command-line only)

## Qt widget instantiated without QApplication — 2026-04-23
**What happened:** A smoke test tried to instantiate `FrfResultsPage` directly in a Python one-liner, and the process crashed because Qt widgets require an active `QApplication`.
**Root cause:** Assumed importing the module was enough for a lightweight instantiation test; forgot Qt’s requirement for a running application object.
**Fix applied:** Switched the smoke test to create a minimal `QApplication([])` before creating any `QWidget`.
**Prevention rule:** For Qt/PySide6 smoke tests, always create `QApplication` before instantiating any `QWidget` (or limit smoke tests to import-only).
**Affected files:** None (command-line only)
