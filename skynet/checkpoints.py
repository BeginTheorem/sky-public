from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .time import utc_now


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Checkpoint:
    commit: str
    created_at: str
    health: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {"commit": self.commit, "created_at": self.created_at, "health": self.health}


class CheckpointManager:
    """Explicit Git checkpoint boundary for the future self-improvement loop."""

    def __init__(self, root: str | Path, metadata_path: str | Path | None = None) -> None:
        self.root = Path(root)
        self.metadata_path = Path(metadata_path) if metadata_path else self.root / "state" / "checkpoint.json"

    def _git(self, args: Sequence[str]) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise CheckpointError(result.stderr.strip() or f"git {' '.join(args)} failed")
        return result.stdout.strip()

    def current_commit(self) -> str:
        return self._git(["rev-parse", "HEAD"])

    def is_ancestor(self, ancestor: str, descendant: str = "HEAD") -> bool:
        """Report whether `ancestor` is reachable from `descendant`.

        A rollback must never `reset --hard` to a commit that is not an ancestor
        of HEAD: a stale rollback_commit from an older promotion would silently
        discard every later commit, including the running one.
        """
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=self.root, capture_output=True, text=True, check=False,
        )
        return result.returncode == 0

    def is_clean(self, *, allow_untracked: bool = False) -> bool:
        output = self._git(["status", "--porcelain", "--untracked-files=all"])
        ignored = {
            self.metadata_path.resolve(),
            (self.metadata_path.parent / "runtime.jsonl").resolve(),
            (self.root / ".deployed-commit").resolve(),
            (self.root / ".deployment-refresh").resolve(),
            (self.root / "state" / "self-improvement-proposals.json").resolve(),
            (self.root / "state" / "reboot-request.json").resolve(),
            (self.root / "state" / "reboot-guard.json").resolve(),
            (self.root / "config" / "skynet.env").resolve(),
        }
        for line in output.splitlines():
            relative = line[3:].strip()
            path = (self.root / relative).resolve() if relative else None
            if path in ignored:
                continue
            if path is not None and any(part.lower() in {"__pycache__", ".pytest_cache", ".pytest-cache", ".mypy_cache", ".ruff_cache"} for part in path.relative_to(self.root).parts):
                continue
            # Only runtime-generated untracked files may cross a checkpoint.
            # Source and test files must be tracked before promotion.
            if (
                allow_untracked
                and line.startswith("?? ")
                and path is not None
                and path.parent == self.metadata_path.parent
                and path.name in {"provider-fallback.json", "telegram-offset.json", "telegram-log-cursor.json", "money-boost.json"}
            ):
                continue
            return False
        return True

    def create(self, health: dict[str, object], *, require_clean: bool = True, allow_untracked: bool = False) -> Checkpoint:
        if require_clean and not self.is_clean(allow_untracked=allow_untracked):
            raise CheckpointError("cannot create checkpoint with a dirty worktree")
        checkpoint = Checkpoint(self.current_commit(), utc_now(), health)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.metadata_path.name}.", dir=self.metadata_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(checkpoint.as_dict(), indent=2, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.metadata_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return checkpoint

    def load(self) -> Checkpoint:
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            return Checkpoint(str(data["commit"]), str(data["created_at"]), dict(data["health"]))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"invalid checkpoint metadata: {exc}") from exc

    def rollback(self) -> str:
        """Restore the recorded checkpoint."""
        checkpoint = self.load()
        self._git(["reset", "--hard", checkpoint.commit])
        return checkpoint.commit
