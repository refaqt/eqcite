# Portable agent kit

Copy this tree into another repository to reuse the same agent workflow without Cursor User Rules or User Skills.

## Copy into the target repo

| Path | Notes |
| --- | --- |
| `.cursor/rules/` | All `.mdc` files except `repo-profile.mdc` unless you copy and customize it. |
| `.cursor/skills/` | All skill folders with `SKILL.md`. |
| `.cursor/bootstrap/docs/` | Starter stubs for the required `docs/` files. |
| `docs/` | Copy stubs from `.cursor/bootstrap/docs/` on first use, or merge with existing docs. |

### Customize per repository

1. Copy `repo-profile.example.mdc` to `repo-profile.mdc` and edit the stack and execution assumptions.
2. Remove or replace `repo-profile.mdc` in this repo if you are not using the example file.
3. Keep `core.mdc`, `living-docs.mdc`, `planning-and-testing.mdc`, `subagents.mdc`, and scoped rules portable.

## Required `docs/` files

- `docs/architecture.md`
- `docs/decisions.md`
- `docs/mistakes.md`
- `docs/skills.md`
- `docs/onboarding.md`

If any are missing, the agent should create them from `.cursor/bootstrap/docs/` with a brief initial entry.

## Do not copy

- `.cursor/_backup*` (local backups only)
- `~/.cursor/skills-cursor/` (Cursor built-in skills)
- This repo's domain content from `docs/skills.md` unless it applies to the target repo

## After copying

1. Open the target repo root in Cursor.
2. Clear **Settings → Rules → User Rules** and user-level skills once project rules load.
3. Confirm **Settings → Rules** lists the imported project rules and skills.
4. Start a new Agent chat and confirm the agent reads `docs/mistakes.md` and bootstraps missing `docs/` files when needed.

## Context design

- Always-on rules: `core.mdc` and `repo-profile.mdc` only.
- Detailed documentation, planning, subagent, PowerShell, and Python guidance loads when relevant or when matching files are in context.
- Long formats live in `docs/` and `.cursor/skills/`, not in always-on rules.
