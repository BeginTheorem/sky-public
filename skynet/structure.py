"""A bounded structural mirror of the organism's own Python code.

ruff, pyright and pytest do not look *between* functions: duplication,
cyclomatic complexity, dead code and module coupling pass a green gate
unnoticed, and an agent that writes its own code generation after generation
forgets the helpers it already wrote. ``pyscn`` (MIT) closes that gap. This
module runs it and returns a compact, read-only summary the organism can consult
*before* proposing a change.

It is a mirror, never a gate. A finding is information, not a promotion
criterion: scoring "less mess" as an objective would make the organism split
functions or rename clones to please the number. The only hard structural rule
lives in ``tests/test_dependency_contracts.py``, because that one is about
correctness, not tidiness.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ANALYZE_TIMEOUT_SECONDS = 120.0
MAX_SUMMARY_CHARS = 6_000


def pyscn_binary() -> Path | None:
    """Locate pyscn beside the running interpreter, then on PATH.

    The reactor runs as ``<root>/.venv/bin/python -m skynet`` and PATH does not
    include the venv's ``bin``, so the interpreter's own directory is the
    reliable location.
    """
    sibling = Path(sys.executable).parent / "pyscn"
    if sibling.exists():
        return sibling
    found = shutil.which("pyscn")
    return Path(found) if found else None


def analyze(root: Path, *, timeout: float = ANALYZE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Run pyscn over ``<root>/skynet`` and return ``{"available", ...}``.

    Never raises for an environment problem: a missing tool or a non-zero exit
    returns ``available: False`` with a reason, so the organism gets an honest
    "cannot see" instead of a failed run. The subprocess runs in the system
    temporary directory so pyscn can never leave a cache directory inside the
    repository and dirty the worktree the gate inspects.
    """
    binary = pyscn_binary()
    if binary is None:
        return {"available": False, "reason": "pyscn is not installed in this environment"}
    target = root / "skynet"
    if not target.is_dir():
        target = root
    try:
        completed = subprocess.run(
            [str(binary), "analyze", "--json", "--output", "-", str(target)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"pyscn could not run: {exc}"}
    if completed.returncode != 0:
        lines = (completed.stderr or completed.stdout).strip().splitlines()
        tail = lines[-1] if lines else "no output"
        return {"available": False, "reason": f"pyscn exited {completed.returncode}: {tail}"}
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {"available": False, "reason": f"pyscn returned non-JSON output: {exc}"}
    if not isinstance(report, dict):
        return {"available": False, "reason": "pyscn returned a non-object report"}
    return {"available": True, "report": report}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _rel(path: Any, root: Path) -> str:
    try:
        return str(Path(str(path)).resolve().relative_to(root))
    except (ValueError, OSError):
        return Path(str(path)).name


def _location(function: dict[str, Any], root: Path) -> str:
    metrics = _as_dict(function.get("metrics"))
    return f"{_rel(function.get('file_path', ''), root)}:{function.get('start_line')}  {function.get('name')} ({_fmt(metrics.get('complexity'))})"


def _pair(pair: dict[str, Any], root: Path) -> str:
    first = _as_dict(_as_dict(pair.get("clone1")).get("location"))
    second = _as_dict(_as_dict(pair.get("clone2")).get("location"))
    left = f"{_rel(first.get('file_path', ''), root)}:{first.get('start_line')}-{first.get('end_line')}"
    right = f"{_rel(second.get('file_path', ''), root)}:{second.get('start_line')}-{second.get('end_line')}"
    return f"{left} <-> {right}"


def summarize(report: dict[str, Any], root: Path, *, top: int = 8) -> str:
    """Render a compact, bounded text summary of a pyscn report."""
    summary = _as_dict(report.get("summary"))
    lines = [
        f"Structural report for {root.name}/skynet — health {summary.get('health_score')}/100 ({summary.get('grade')})",
        (
            f"avg complexity {_fmt(summary.get('average_complexity'))} | "
            f"functions >=20 {summary.get('high_complexity_count')} | "
            f"dead code {summary.get('dead_code_count')} | "
            f"clone groups {summary.get('clone_groups')} ({_fmt(summary.get('code_duplication_percentage'))}% dup) | "
            f"high-coupling classes {summary.get('high_coupling_classes')} (avg CBO {_fmt(summary.get('average_coupling'))})"
        ),
    ]
    architecture = _as_dict(_as_dict(report.get("system")).get("architecture_analysis"))
    if architecture.get("compliance_score") is not None:
        compliance = round(float(architecture["compliance_score"]) * 100)
        lines.append(f"architecture compliance {compliance}% ({architecture.get('total_violations')} findings)")

    functions = _as_list(_as_dict(report.get("complexity")).get("functions"))
    if functions:
        ranked = sorted(functions, key=lambda item: _as_dict(_as_dict(item).get("metrics")).get("complexity", 0), reverse=True)[:top]
        lines.append("")
        lines.append(f"Most complex functions (top {len(ranked)}):")
        lines.extend(f"  {_location(_as_dict(item), root)}" for item in ranked)

    pairs = _as_list(_as_dict(report.get("clone")).get("clone_pairs"))
    if pairs:
        worst = sorted(pairs, key=lambda item: _as_dict(item).get("similarity", 0), reverse=True)[:top]
        lines.append("")
        lines.append(f"Worst clone pairs (top {len(worst)}):")
        lines.extend(f"  sim {_fmt(_as_dict(item).get('similarity'))}  {_pair(_as_dict(item), root)}" for item in worst)

    text = "\n".join(lines)
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[:MAX_SUMMARY_CHARS] + "\n[report truncated]"
    return text


class StructureTool:
    """Read-only tool exposing the structural mirror to the model."""

    name = "structure"
    capability_kind = "read"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(__file__).resolve().parent.parent

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Read-only structural mirror of this project's own Python: cyclomatic complexity, "
                    "clone groups, dead code, module coupling and architecture findings, computed by pyscn. "
                    "Consult it before proposing a change to see whether you are about to duplicate an "
                    "existing helper or grow an already-large function. It is information, not a gate: no "
                    "finding blocks or downgrades a run."
                ),
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        result = analyze(self.root)
        if not result.get("available"):
            return {"ok": False, "error": str(result.get("reason", "pyscn is unavailable"))}
        return {"ok": True, "result": summarize(result["report"], self.root)}
