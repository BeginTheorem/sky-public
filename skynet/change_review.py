"""Deterministic review of a proposed source change, before the gate spends time.

Inspired by the critic and pruning discipline of RRSI (arXiv:2609.24972): a
change that is *degenerate* rather than merely unproven should not reach the
gate. Two failure modes are rejected here, both measured in this repository:

- a no-op edit (comment-only or whitespace-only) that claims a mechanism in its
  hypothesis, and
- a newly added private symbol that nothing references, at module level (the dead
  helper ``_parse_finish_report_with_error`` and the dead flag
  ``use_portfolio_planner``) or as an undecorated class method (the dead
  ``MemoryStore._get_next_level`` / ``_get_level_name`` /
  ``_get_consolidation_prompt`` helpers). Each was promoted and reverted by hand
  before this check existed.

The review is deliberately deterministic: no model call, no benchmark, no
network. It reads the pre-image and post-image of every changed path and scans
the proposal worktree for references. It is a *critic*, not a judge: it only
refuses a change that cannot be a mechanism, and leaves "is this useful?" to
the existing gate. A public new symbol that appears unused is a warning, not a
rejection, so an intentional API addition is never blocked.

The module is a pure reader: it imports only the standard library and never
writes to the worktree, so it is unit-testable on its own.
"""

from __future__ import annotations

import ast
import difflib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Severity = Literal["reject", "warn"]

# A removed token from this list is reported as a warning only: deleting a
# context/output/termination guard without a replacement is worth a second look,
# but a refactor may legitimately move it, so it must never block the gate.
_SAFETY_TOKENS = ("truncat", "max_chars", "watchdog", "deadline")
_COMMENT = re.compile(r"^#")
_SKIP_DIRS = frozenset({
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", ".skynet-improvements",
})
_PRIVATE = re.compile(r"^_[A-Za-z]")
_CONSTANT = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True, slots=True)
class FileChange:
    """One changed path, with its pre-image and post-image as text.

    ``None`` means the file did not exist on that side: a new file has no
    pre-image, a deletion has no post-image. Binary or unreadable content is
    also reported as ``None``; the review then skips the path rather than
    guessing.
    """

    path: str
    before: str | None
    after: str | None


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    path: str
    severity: Severity
    detail: str


def _word(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")


def _changed_lines(before: str, after: str) -> list[str]:
    """The ``+``/``-`` payload lines of a zero-context unified diff."""
    lines: list[str] = []
    for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0):
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith(("+", "-")):
            lines.append(line[1:])
    return lines


def _inert(line: str) -> bool:
    stripped = line.strip()
    return not stripped or bool(_COMMENT.match(stripped))


def _same_ast(before: str, after: str) -> bool:
    """True when two Python sources parse to the same tree (formatting-only).

    ``ast`` ignores indentation, blank lines and comments but keeps string
    literals, so a reformat counts as whitespace-only while a change *inside* a
    string (``"hello world"`` -> ``"helloworld"``) does not -- the earlier
    whole-text whitespace collapse wrongly rejected the latter.
    """
    try:
        return ast.dump(ast.parse(before)) == ast.dump(ast.parse(after))
    except (SyntaxError, ValueError):
        return False


def _degenerate_text(before: str | None, after: str | None, *, is_python: bool) -> str:
    """Name the degeneracy of a pure-text change, or the empty string.

    Byte-identical content is not judged here: a mode-only change (``chmod``)
    has no textual difference and the gate, not this review, owns it.
    """
    if before is None and after is None:
        return ""
    if before == after:
        return ""
    if before is not None and before.strip() == "" and (after is None or after.strip() == ""):
        # Adding an empty file or deleting an empty one is a structural act,
        # not a comment claiming a mechanism; it is not our call to refuse it.
        return ""
    if after is not None and after.strip() == "" and before is None:
        return ""
    if not is_python:
        return ""
    changed = _changed_lines(before or "", after or "")
    if changed and all(_inert(line) for line in changed):
        return "comment-only"
    if before is not None and after is not None and _same_ast(before, after):
        return "whitespace-only"
    return ""


def _top_level_names(text: str) -> dict[str, bool]:
    """Module-level ``name -> decorated`` for the symbols a change may add."""
    names: dict[str, bool] = {}
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return names
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] = bool(node.decorator_list)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names[target.id] = False
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names[node.target.id] = False
    return names


_TEXT_SUFFIXES = frozenset({
    ".py", ".pyi", ".toml", ".cfg", ".ini", ".sh", ".in", ".json", ".md",
    ".txt", ".yml", ".yaml", ".conf", ".example", ".service", ".timer",
})
# A reference may live outside Python: a symbol used only by a template, a
# service unit or a config file is still used. The scan reads every text file so
# "unused" means "nowhere", not "not in a .py"; large files are skipped so a
# pathological tree cannot turn the review into a stall.
_MAX_SCAN_BYTES = 512 * 1024


def _text_sources(worktree: Path) -> dict[str, str]:
    """Every text file under *worktree*, excluding caches, VCS dirs and large files."""
    sources: dict[str, str] = {}
    if not worktree.is_dir():
        return sources
    for root, dirs, files in os.walk(worktree):
        dirs[:] = [name for name in dirs if name not in _SKIP_DIRS]
        for name in files:
            path = Path(root) / name
            if path.suffix not in _TEXT_SUFFIXES:
                continue
            try:
                if path.stat().st_size > _MAX_SCAN_BYTES:
                    continue
                sources[str(path)] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
    return sources


def _private_methods(text: str) -> set[str]:
    """Private, undecorated methods defined in any class body of *text*.

    :func:`_top_level_names` never looks inside a ``class``, so a dead private
    method slipped past the module-level rule: the three ``MemoryStore`` TMT
    helpers below were all added and promoted that way. Decorated methods are
    excluded -- a decorator is a dispatch signal (``@property``, a registry
    hook), so calling such a method "unused" would be a guess, exactly as the
    module-level rule treats a decorated symbol.
    """
    names: set[str] = set()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return names
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if (
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _PRIVATE.match(item.name)
                and not item.decorator_list
            ):
                names.add(item.name)
    return names


def _unused_symbols(change: FileChange, sources: dict[str, str]) -> list[Finding]:
    if change.after is None or not change.path.endswith(".py"):
        return []
    before_names = set(_top_level_names(change.before)) if change.before is not None else set()
    after_names = _top_level_names(change.after)
    findings: list[Finding] = []
    for name, decorated in after_names.items():
        if name in before_names or decorated or name.startswith("__"):
            continue
        references = sum(len(_word(name).findall(text)) for text in sources.values())
        if references > 1:
            continue
        private = bool(_PRIVATE.match(name))
        constant = bool(_CONSTANT.match(name))
        severity: Severity = "reject" if (private or constant) else "warn"
        findings.append(
            Finding(
                code="unused_symbol",
                path=change.path,
                severity=severity,
                detail=f"new module-level name {name!r} has no reference in the proposal worktree",
            )
        )
    before_methods = _private_methods(change.before) if change.before is not None else set()
    for name in sorted(_private_methods(change.after) - before_methods):
        references = sum(len(_word(name).findall(text)) for text in sources.values())
        if references > 1:
            continue
        findings.append(
            Finding(
                code="unused_symbol",
                path=change.path,
                severity="reject",
                detail=f"new private method {name!r} has no reference in the proposal worktree",
            )
        )
    return findings


def _safety_removed(change: FileChange) -> list[Finding]:
    if change.before is None or change.after is None or not change.path.startswith("skynet/") or not change.path.endswith(".py"):
        return []
    return [
        Finding(
            code="safety_mechanism_removed",
            path=change.path,
            severity="warn",
            detail=f"the token {token!r} is removed without a replacement",
        )
        for token in _SAFETY_TOKENS
        if token in change.before and token not in change.after
    ]


def review_change(changes: list[FileChange], *, worktree: Path) -> list[Finding]:
    """All findings for a change set, in path order. Empty means "looks real"."""
    sources = _text_sources(worktree)
    findings: list[Finding] = []
    for change in changes:
        is_python = change.path.endswith(".py")
        degenerate = _degenerate_text(change.before, change.after, is_python=is_python)
        if degenerate:
            findings.append(
                Finding(
                    code="degenerate_change",
                    path=change.path,
                    severity="reject",
                    detail=f"the change is a {degenerate} edit and cannot be a mechanism",
                )
            )
        findings.extend(_unused_symbols(change, sources))
        findings.extend(_safety_removed(change))
    return findings
