# Skills: procedures kept as files

This directory holds named procedures. The harness appends an **index** of them to the
system prompt (name + one-line description, behind a `[SKYNET SKILLS BEGIN]` marker) and
`skill(action='read', name=...)` returns one body in full. Nothing here is injected when the
directory is empty.

Why files and not memories: a `procedure` memory averages a couple of hundred characters and
shares the recall window with facts, so a procedure with real steps does not fit. A skill file
has no such ceiling and is read only when the work needs it.

## Format

```markdown
---
name: restart-a-unit
description: How to restart a user unit without cutting a reply short.
---

## Steps
...
```

* `name` defaults to the file stem and must be lowercase, digits, dots, dashes, underscores.
* `description` defaults to the first meaningful line of the body and is clipped to 200 chars
  in the index.
* Front matter is optional.
* Bodies are read with an 8,000-character cap; a longer body is returned truncated with a notice.

`README.md`, `_`-prefixed files and dotfiles are documentation, not skills, and are skipped.
A file with neither a description nor a first line of prose is not indexed either — an empty
line in the prompt is worse than a missing one.

## Authoring

Skills are ordinary files in this tree: write them with the usual file tools (`write`, `bash`),
same as `SOUL.md` and `AGENTS.md`. Editing a skill is editing the harness's own text, so it is
worth the same care as any other change to this tree.
