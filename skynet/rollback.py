"""The organism's own rollback request.

A self-modifying system needs a way to undo its own bad change without waiting
for the owner. The request is written durably first, then executed by the
Supervisor on the next start or by the external startup script if the process
never comes back - so it works even when the promoted code is the reason the
process is broken.
"""

from __future__ import annotations

import contextlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from .checkpoints import CheckpointManager
from .provider import Tool
from .time import utc_now

log = logging.getLogger("skynet.rollback")

ROLLBACK_REQUEST_FILENAME = "rollback-request.json"


def request_path(root: str | Path) -> Path:
    return Path(root) / "state" / ROLLBACK_REQUEST_FILENAME


def _git(root: Path, args: list[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


class RollbackRequestTool(Tool):
    """Record a durable request to roll the tree back to an earlier commit.

    Marked dangerous on purpose: rolling back discards every commit after the
    target, including work the organism may have promoted since.
    """

    name = "request_rollback"
    capability_kind = "write"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "DANGEROUS. Request a rollback of the running code to an earlier commit, "
                    "executed by the supervisor on the next start. Use it only when a promotion "
                    "provably broke the organism and you cannot fix it forward. Rolling back "
                    "discards every later commit. Requires a concrete reason."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "minLength": 10, "maxLength": 1000},
                        "commit": {
                            "type": "string",
                            "description": "Optional hexadecimal commit to roll back to; defaults to the commit before HEAD.",
                        },
                    },
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        del idempotency_key
        reason = str(arguments.get("reason", "")).strip()
        if len(reason) < 10:
            return {"ok": False, "error": "reason must be at least 10 characters"}
        target = str(arguments.get("commit", "")).strip()
        checkpoint = CheckpointManager(self.root)
        try:
            if not target:
                target = _git(self.root, ["rev-parse", "HEAD~1"])
            if not checkpoint.is_ancestor(target):
                return {
                    "ok": False,
                    "error": f"commit {target} is not an ancestor of HEAD; rolling back would discard later work",
                }
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            return {"ok": False, "error": f"cannot resolve the rollback target: {exc}"}

        path = request_path(self.root)
        if path.exists():
            return {"ok": False, "error": "a rollback request is already pending; it is executed on the next start"}
        payload = {
            "rollback_commit": target,
            "reason": reason[:1000],
            "requested_at": utc_now(),
            "dangerous": True,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            import os

            os.replace(temporary, path)
        except OSError as exc:
            return {"ok": False, "error": f"could not persist the rollback request: {exc}"}
        return {
            "ok": True,
            "state": "queued",
            "rollback_commit": target,
            "note": "the supervisor executes this on the next start; it is not applied mid-run",
        }


def consume_request(root: str | Path) -> dict[str, Any] | None:
    """Execute and remove a pending rollback request; returns what happened.

    The ancestry check is repeated here: a stale request whose target is no
    longer an ancestor must never discard later commits.
    """
    base = Path(root).resolve()
    path = request_path(base)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    commit = str(payload.get("rollback_commit", ""))
    reason = str(payload.get("reason", ""))
    result: dict[str, Any] = {"rollback_commit": commit, "reason": reason, "applied": False}
    if (base / ".git").exists():
        checkpoint = CheckpointManager(base)
        if commit and checkpoint.is_ancestor(commit):
            try:
                checkpoint._git(["reset", "--hard", commit])
                result["applied"] = True
            except Exception as exc:
                result["error"] = str(exc)[:500]
        else:
            result["error"] = "rollback commit is missing or no longer an ancestor of HEAD"
    else:
        result["error"] = "not a git worktree"
    with contextlib.suppress(OSError):
        path.unlink()
    log.warning("executed rollback request: %s", result)
    return result
