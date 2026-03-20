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
