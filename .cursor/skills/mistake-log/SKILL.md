---
name: mistake-log
description: >-
  Logs incidents to docs/mistakes.md, escalates repeated failures, and promotes
  prevention rules. Use at task start, after bugs, failed approaches, rejected
  output, or convention misunderstandings in this repository.
---

# Mistake log

## What counts as a mistake

- A bug introduced and fixed in the same session.
- A wrong approach taken and abandoned.
- A misunderstanding of project conventions that caused rework.
- A test failure from an incorrect assumption.
- Any output the user rejected or asked to redo.

## Before starting work

1. Read `docs/mistakes.md`.
2. State which prevention rules apply to the current task.

## After a mistake

1. Append an entry to `docs/mistakes.md` using the format in that file.
2. If the same mistake happens twice, add a bold warning on the relevant section and promote the prevention rule into scoped `.cursor/rules/`.
3. Treat every mistake as a process gap, not a one-off.

## Continuous improvement

- After a long session or several tasks, review `docs/mistakes.md` and `docs/skills.md` for rules or patterns to promote into standing project guidance.
- Prefer making `docs/mistakes.md` shorter over time by preventing repeat categories.

## Entry format

Use the template at the top of `docs/mistakes.md`.
