"""Procedures kept as files: the prompt carries an index, the body comes on demand.

A procedure learned in one run had nowhere to live but a memory row: ``procedure``
memories average 255 characters and compete for the same 24-term recall window as
facts. This module gives procedures a file, the way ``SOUL.md`` is a file -- the
index is appended to the system prompt behind a marker, and ``skill(action='read')``
returns the body only when the work actually needs it, so a long procedure costs
nothing until it is used.

Nothing is injected when the directory is empty: an organism that keeps no skills
pays nothing for the feature.

The organism writes these files itself with ``write``/``bash`` inside its own tree,
so authoring a skill needs no tool here; *reading* one does, because the prompt
carries only the index.

A skill file is markdown with optional front matter:

    ---
    name: restart-a-unit
    description: How to restart a user unit without cutting a reply short.
    ---
    ## Steps
    ...

Without front matter the file still counts: the stem becomes the name and the first
meaningful line becomes the description. ``README.md`` and ``_``-prefixed files are
documentation, not skills, and are skipped -- as are dotfiles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SKILLS_DIRNAME = "skills"
SKILL_SUFFIX = ".md"
MAX_DESCRIPTION_CHARS = 200
# The index is paid for on every step, the body only when it is read: the caps are
# deliberately asymmetric.
MAX_INDEX_CHARS = 2400
MAX_BODY_CHARS = 8000
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SKIP_STEMS = frozenset({"readme", "index"})
_FRONT_MATTER_FENCE = "---"


class SkillError(RuntimeError):
    """A skill could not be read: unknown name, malformed file, or a refused path."""


@dataclass(frozen=True)
class Skill:
    """One procedure file: what the index shows and where the body lives."""

    name: str
    description: str
    path: Path


def skills_root(root: str | Path) -> Path:
    """The directory procedures live in, relative to the run root."""
    return Path(root) / SKILLS_DIRNAME


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Return (fields, body). Without front matter the fields are empty.

    Only a file that *starts* with a ``---`` fence is treated as having front matter;
    a ``---`` later in the body is a horizontal rule or a table, not metadata.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER_FENCE:
        return {}, text
    for index in range(1, len(lines)):
        if lines[index].strip() == _FRONT_MATTER_FENCE:
            fields: dict[str, str] = {}
            for line in lines[1:index]:
                key, _, value = line.partition(":")
                if key.strip() and value.strip():
                    fields[key.strip().lower()] = value.strip().strip("\"'")
            return fields, "\n".join(lines[index + 1 :]).lstrip("\n")
    return {}, text


def _first_meaningful_line(body: str) -> str:
    """The first line of prose: markdown headings lose their hashes, quotes stay."""
    for line in body.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped
    return ""


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def load_skill(path: Path) -> Skill | None:
    """Read one candidate file, or ``None`` when it is not a skill we would index."""
    stem = path.stem
    if not stem or stem.startswith((".", "_")) or stem.lower() in SKIP_STEMS:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    fields, body = _split_front_matter(text)
    name = (fields.get("name") or stem).strip().lower()
    if not SKILL_NAME_PATTERN.match(name):
        return None
    description = (fields.get("description") or _first_meaningful_line(body)).strip()
    if not description:
        # A file with no name and no prose is not a procedure; indexing it would only
        # spend prompt budget on an empty line.
        return None
    return Skill(name=name, description=_clip(description, MAX_DESCRIPTION_CHARS), path=path)


def discover(root: str | Path) -> tuple[Skill, ...]:
    """Every skill in the run root, sorted by name. A missing directory is not an error."""
    directory = skills_root(root)
    if not directory.is_dir():
        return ()
    skills: dict[str, Skill] = {}
    for path in sorted(directory.glob(f"*{SKILL_SUFFIX}")):
        if not path.is_file():
            continue
        skill = load_skill(path)
        if skill is not None and skill.name not in skills:
            skills[skill.name] = skill
    return tuple(skills[name] for name in sorted(skills))


def read_skill(root: str | Path, name: str, *, max_chars: int = MAX_BODY_CHARS) -> str:
    """The body of one skill, without its front matter.

    The name must be one this root actually publishes: the lookup is against the
    discovered set, never against a path the caller assembled, so ``../`` in a name
    cannot reach outside the directory.
    """
    wanted = (name or "").strip().lower()
    if not SKILL_NAME_PATTERN.match(wanted):
        raise SkillError(f"not a skill name: {name!r}")
    for skill in discover(root):
        if skill.name == wanted:
            _, body = _split_front_matter(skill.path.read_text(encoding="utf-8"))
            if not body.strip():
                raise SkillError(f"skill {wanted} has no body")
            if len(body) > max_chars:
                return body[:max_chars].rstrip() + f"\n[truncated at {max_chars} chars]"
            return body
    known = ", ".join(skill.name for skill in discover(root)) or "none"
    raise SkillError(f"unknown skill {wanted!r}; known: {known}")


def index_block(root: str | Path) -> str:
    """The prompt block naming every skill, or ``""`` when there are none.

    The index is what makes the bodies reachable; it must stay cheap, so descriptions
    are already clipped per skill and the whole block is clipped again here.
    """
    skills = discover(root)
    if not skills:
        return ""
    lines = [f"SKILLS: {len(skills)} in {skills_root(root)}; read one with skill(action='read', name=...)"]
    lines.extend(f"- {skill.name}: {skill.description}" for skill in skills)
    text = "\n".join(lines)
    if len(text) > MAX_INDEX_CHARS:
        text = text[:MAX_INDEX_CHARS].rstrip() + f"\n[index truncated at {MAX_INDEX_CHARS} chars]"
    return text


class SkillTool:
    """Read the procedures the prompt only names -- and list them in full."""

    name = "skill"
    capability_kind = "read"

    ACTIONS = ("list", "read")
    MAX_LIST_ITEMS = 200

    def __init__(self, root: str | Path | None = None, *, max_body_chars: int = MAX_BODY_CHARS) -> None:
        self.root = Path(root) if root is not None else Path.cwd()
        self.max_body_chars = max_body_chars

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Read the named procedures kept in this project's skills/ directory. The prompt "
                    "carries only an index of them, so action='list' shows every skill with its "
                    "description and action='read' returns one body in full -- do that before "
                    "re-deriving a procedure that may already be written down. Write new skills with "
                    "the ordinary file tools; this tool only reads them."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": list(self.ACTIONS)},
                        "name": {"type": "string", "description": "Skill name; required for action='read'."},
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        action = str(arguments.get("action") or "").strip().lower()
        if action not in self.ACTIONS:
            return {"ok": False, "error": f"action must be one of {', '.join(self.ACTIONS)}"}
        if action == "list":
            skills = discover(self.root)[: self.MAX_LIST_ITEMS]
            return {
                "ok": True,
                "result": {
                    "root": str(skills_root(self.root)),
                    "count": len(skills),
                    "skills": [{"name": skill.name, "description": skill.description} for skill in skills],
                },
            }
        name = str(arguments.get("name") or "").strip()
        if not name:
            return {"ok": False, "error": "name is required for action='read'"}
        try:
            body = read_skill(self.root, name, max_chars=self.max_body_chars)
        except SkillError as error:
            return {"ok": False, "error": str(error)}
        except OSError as error:
            return {"ok": False, "error": f"could not read {name}: {error}"}
        return {"ok": True, "result": {"name": name.strip().lower(), "chars": len(body), "body": body}}
