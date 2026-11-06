"""The planning pass's read-only senses: reconnaissance without the ability to act.

The plan pass used to run with no tools at all, and a tool call in its reply was
treated as a parse failure -- reasonable while the planner could not do anything with
one, wrong now that Plan is meant to be one big pass that scouts before it proposes.

What it gets is deliberately narrow, and the narrowness is the design:

- the network half is ``webfetch`` plus the ddg sensor (search, then read the page);
- ``read``, ``grep`` and ``structure`` read this tree;
- ``memory`` is the same tool restricted to ``search``: the planner reads memory and
  never writes it, because curation belongs to the episode and the consolidation pass;
- ``bash_readonly`` exists because reconnoitring a real host needs ``git log``, ``ps``,
  ``df``. It runs a command only if every segment of the line starts with an allowed
  command, refuses every redirect and every route to root, and records the refusal.

It is a filter, not a sandbox, and it is not secrecy: the pass runs with the same
privileges as the episode, which can already read the host. What the allowlist buys is
that the pass *cannot write* -- a scout that can edit is not a scout.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Whole commands the pass may run. Read-only by nature, and every one of them is
# inspection: nothing here can create, move, delete or execute a repository change.
READ_ONLY_COMMANDS = frozenset(
    {
        "cat",
        "cut",
        "date",
        "df",
        "du",
        "file",
        "find",
        "free",
        "git",
        "grep",
        "head",
        "jq",
        "ls",
        "ps",
        "rg",
        "sort",
        "stat",
        "tail",
        "uniq",
        "uptime",
        "wc",
        "which",
    }
)

# `git` is a whole toolchain behind one word, so its read-only half is named too:
# `git checkout`, `git clean`, `git commit` and friends never reach the runner.
READ_ONLY_GIT_SUBCOMMANDS = frozenset(
    {
        "blame",
        "branch",
        "cat-file",
        "describe",
        "diff",
        "grep",
        "log",
        "ls-files",
        "ls-tree",
        "rev-parse",
        "shortlog",
        "show",
        "status",
        "tag",
    }
)

# What turns a read into a write or a route to root. Syntax tokens are matched as
# substrings (`>` anywhere is a redirect); whole words are matched on boundaries,
# because `git add -A` contains "dd" and `adapt` contains "apt" -- a substring test
# refuses honest reads and teaches the model nothing. `git` subcommands are handled by
# the allowlist below, which is stricter than any blacklist and names the offender.
FORBIDDEN_SUBSTRINGS = (">",)
FORBIDDEN_WORDS = re.compile(
    r"\b("
    r"sudo|su|doas|pkexec|chmod|chown|install|rm|mv|cp|dd|truncate|mkfs|mount"
    r"|systemctl|apt|apt-get|pip|pip3|npm|make|tee|bash|sh|zsh|eval|exec"
    r")\b"
)

DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_OUTPUT_CHARS = 6_000
_SEGMENT_SEPARATORS = ("&&", "||", ";", "\n", "|")


class PlanBashRefused(RuntimeError):
    """A command the read-only planner shell will not run."""


def refusal_reason(command: str) -> str | None:
    """Why this line cannot run read-only, or ``None`` when it can.

    Every pipeline segment is checked, not just the first word: ``ls | tee x`` writes,
    and ``git log; git commit`` acts. Command substitution and backticks are refused
    outright -- a substitution is a second command the allowlist never sees.
    """
    text = command.strip()
    if not text:
        return "empty command"
    if "`" in text or "$(" in text:
        return "command substitution is not allowed in the read-only planner shell"
    lowered = text.lower()
    for token in FORBIDDEN_SUBSTRINGS:
        if token in lowered:
            return f"refused token {token!r}: the planning pass cannot write or reach root"
    forbidden = FORBIDDEN_WORDS.search(lowered)
    if forbidden is not None:
        return f"refused token {forbidden.group(0)!r}: the planning pass cannot write or reach root"
    segments = [text]
    for separator in _SEGMENT_SEPARATORS:
        segments = [part for chunk in segments for part in chunk.split(separator)]
    for segment in segments:
        words = segment.strip().split()
        if not words:
            continue
        head = os.path.basename(words[0]).lower()
        if head == "git":
            subcommand = next((word for word in words[1:] if not word.startswith("-")), "")
            if subcommand not in READ_ONLY_GIT_SUBCOMMANDS:
                return f"git {subcommand or '(none)'} is not a read-only git subcommand"
            continue
        if head not in READ_ONLY_COMMANDS:
            return f"{words[0]!r} is not in the read-only allowlist"
    return None


class PlanBashTool:
    """The planning pass's shell: inspection only, and every refusal is named."""

    name = "bash_readonly"
    capability_kind = "read"

    def __init__(self, root: str | Path | None = None, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.root = Path(root) if root is not None else Path.cwd()
        self.timeout_seconds = timeout_seconds

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Run one read-only shell command for reconnaissance: git log/show/diff/status/blame, "
                    "cat/head/tail/wc/grep/rg/find/ls/stat/jq, and host readings (ps, df, du, free, uptime). "
                    "Writes are impossible by construction: redirects, command substitution, root escalation "
                    "and any command outside the allowlist are refused with the reason. Use it to read this "
                    "host and this repository, then answer with the JSON object."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "One read-only shell line."}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        command = str(arguments.get("command") or "")
        reason = refusal_reason(command)
        if reason is not None:
            return {"ok": False, "error": reason}
        started = time.monotonic()
        try:
            # shell=True is required to run a shell line at all; the allowlist and
            # `refusal_reason` above are what decide whether the line may run, and every
            # segment is checked before this point.
            completed = subprocess.run(  # noqa: S602
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"command exceeded {self.timeout_seconds:.0f}s"}
        except OSError as exc:
            return {"ok": False, "error": f"command could not run: {exc}"}
        output = (completed.stdout or "") + (completed.stderr or "")
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n[truncated at {MAX_OUTPUT_CHARS} chars]"
        return {
            "ok": completed.returncode == 0,
            "result": {"command": command, "exit_code": completed.returncode, "output": output, "seconds": round(time.monotonic() - started, 3)},
        }


def plan_toolset(
    available: Mapping[str, Any],
    *,
    memory_tool: Any,
    root: str | Path,
    sensor_servers: Sequence[str] = ("ddg",),
) -> dict[str, Any]:
    """The tools the planning pass may call, keyed as the model sees them.

    Reads from the live inventory instead of a second hardcoded list: the own tools are
    picked by name, the sensors by their server, and ``memory_tool`` is handed in
    already restricted to ``search``. The guard is the invariant, not a formality -- a
    tool that declares ``capability_kind == 'write'`` may not ride along, so it fails
    here rather than in production. The check is written against ``write`` and not
    against ``read`` on purpose: the tree's vocabulary is exactly those two, a tool
    without a kind is read by house convention (``react.py``), and every other object in
    a test double -- a ``Mock`` above all -- would otherwise answer "not read" and turn a
    legitimate inventory into a construction error.
    """
    wanted = ("webfetch", "read", "grep", "structure")
    chosen: dict[str, Any] = {name: available[name] for name in wanted if name in available}
    for name, tool in available.items():
        server = getattr(getattr(tool, "client", None), "server_name", "")
        if server and server in sensor_servers:
            chosen[name] = tool
    chosen["memory"] = memory_tool
    chosen["bash_readonly"] = PlanBashTool(root)
    writable = sorted(name for name, tool in chosen.items() if str(getattr(tool, "capability_kind", "read")) == "write")
    if writable:
        raise ValueError(f"the planning pass may not carry a write-capable tool; got {writable}")
    return chosen


def plan_tool_schemas(tools: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The request-side view of the plan toolset."""
    return [tool.schema for tool in tools.values()]


__all__ = ["PlanBashRefused", "PlanBashTool", "plan_tool_schemas", "plan_toolset", "refusal_reason"]
