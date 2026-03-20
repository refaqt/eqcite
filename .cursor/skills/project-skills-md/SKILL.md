---
name: project-skills-md
description: >-
  Maintains and applies project-specific reusable patterns in docs/skills.md.
  Use before starting implementation to avoid reinventing solved patterns; use
  after solving a problem in a novel way in this codebase to record it; use when
  the user mentions skills.md, project skills, reusable patterns, or deprecating
  a documented pattern.
---

# Project skills (`docs/skills.md`)

## Before starting work

1. Read `docs/skills.md` if it exists.
2. If an entry matches the task, follow that pattern instead of inventing a new approach.
3. If nothing applies, proceed with normal implementation.

## After a novel, codebase-specific fix

When you solve something **specific to this repository** (layout, APIs, scripts, conventions here) in a way worth repeating:

1. Add or update an entry in `docs/skills.md` using the format below.
2. Do **not** add generic programming knowledge — only patterns tied to *this* project.

## When updating or deprecating

- Change or remove the entry in `docs/skills.md` in the same change as the code (or immediately after).
- If deprecated, mark clearly (e.g. **Deprecated:** …) or remove after migration, per team preference.

## Entry format (copy per skill)

Each skill in `docs/skills.md` is one `##` heading plus four labeled lines. Paste and fill:

    ## [Skill name]
    **When to use:** [One sentence]
    **Pattern:**
    ```[language]
    // minimal example
    ```
    **Gotchas:** [Any known edge cases or pitfalls]
    **Last used:** [Date or task reference]

(Indented block above is the exact shape to copy; replace brackets and the code fence language.)

## Quality bar

- **Name:** Short, unique, searchable.
- **When to use:** One sentence so the agent can match tasks quickly.
- **Pattern:** Smallest example that still shows the project-specific hook (path, module, function, config key).
- **Gotchas:** Real pitfalls (ordering, assumptions, version pins in *this* repo).
- **Last used:** ISO date or issue/PR reference.

## Additional reference

For the live registry, open [docs/skills.md](../../../docs/skills.md).
