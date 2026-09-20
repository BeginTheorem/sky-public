from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .time import parse_timestamp, utc_datetime_now, utc_now


class RecoveryError(RuntimeError):
    pass


COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


@dataclass(slots=True)
class RebootGuard:
    """Durable post-reboot health window; never rolls back without a supplied action."""

    state_path: Path
    health_window_cycles: int = 3
    max_window_age_seconds: float = 86_400.0

    @property
    def path(self) -> Path:
        return self.state_path / "reboot-guard.json"

    def _quarantine(self, reason: str) -> Path | None:
        """Move an unreadable or stale guard aside instead of wedging startup.

        ``observe`` runs before every cycle, so a guard that keeps raising would
        turn into a permanent cycle_error loop: the organism would never run
        ReAct again. Quarantining the file lets the next cycle proceed, while
        the original bytes stay on disk for later inspection.
        """
        target_dir = self.state_path / "quarantine"
        stamp = utc_now().replace(":", "").replace("-", "") + "-" + uuid4().hex[:8]
        target = target_dir / f"reboot-guard-{stamp}.json"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            self.path.replace(target)
        except OSError:
            return None
        with contextlib.suppress(OSError):
            (target_dir / f"reboot-guard-{stamp}.reason").write_text(reason + "\n", encoding="utf-8")
        return target

    def _age_exceeded(self, guard: dict[str, object]) -> bool:
        started_at = guard.get("started_at")
        if not isinstance(started_at, str) or not started_at:
            return False
        try:
            started = parse_timestamp(started_at)
        except ValueError:
            return True
        return (utc_datetime_now() - started).total_seconds() > self.max_window_age_seconds


    def begin(self, request_path: Path) -> dict[str, object] | None:
        if not request_path.exists():
            return None
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            if not isinstance(request, dict) or not request.get("commit") or not request.get("rollback_commit"):
                raise ValueError("missing commit or rollback_commit")
            for key in ("commit", "rollback_commit"):
                if not isinstance(request[key], str) or not COMMIT_RE.fullmatch(request[key]):
                    raise ValueError(f"{key} must be a hexadecimal commit id")
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            # Quarantine the corrupt request so startup does not crash-loop.
            stamp = utc_now().replace(":", "").replace("-", "")
            with contextlib.suppress(OSError):
                request_path.replace(request_path.with_name(f"{request_path.name}.corrupt-{stamp}"))
            raise RecoveryError(f"invalid reboot request: {exc}") from exc
        guard = {
            "commit": str(request["commit"]),
            "rollback_commit": str(request["rollback_commit"]),
            "proposal_id": str(request.get("proposal_id", "")),
            "started_at": utc_now(),
            "healthy_cycles": 0,
            "failed": False,
            "release": request.get("health", {}).get("release") if isinstance(request.get("health"), dict) else None,
        }
        self.state_path.mkdir(parents=True, exist_ok=True)
        self._atomic_write(guard)
        request_path.unlink()
        return guard

    def observe(self, health: dict[str, object], rollback: Callable[[str], None] | None = None) -> dict[str, object]:
        if not self.path.exists():
            return {"active": False, "ok": True, "changed": False}
        try:
            guard = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(guard, dict):
                raise TypeError("reboot guard must be a JSON object")
            if not isinstance(guard.get("healthy_cycles", 0), int) or isinstance(guard.get("healthy_cycles", 0), bool):
                raise TypeError("healthy_cycles must be an integer")
            for key in ("commit", "rollback_commit"):
                if not isinstance(guard.get(key), str) or not guard[key]:
                    raise ValueError(f"{key} must be a non-empty string")
                if not COMMIT_RE.fullmatch(guard[key]):
                    raise ValueError(f"{key} must be a hexadecimal commit id")
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            self._quarantine(f"invalid reboot guard: {exc}")
            return {"active": False, "ok": True, "changed": True, "quarantined": True, "reason": f"invalid reboot guard: {exc}"}
        if guard.get("active") is False or guard.get("rolled_back"):
            return {"active": False, "ok": not bool(guard.get("failed")), "completed": bool(guard.get("completed_at")), "rolled_back": bool(guard.get("rolled_back")), "proposal_id": guard.get("proposal_id"), "changed": False}
        if self._age_exceeded(guard):
            # A window nobody finished cannot be trusted either way; fail open,
            # keep the bytes for inspection, and let reconciliation decide the
            # proposal from git history.
            self._quarantine(f"stale reboot window older than {self.max_window_age_seconds:g}s")
            return {"active": False, "ok": True, "changed": True, "quarantined": True, "reason": "stale reboot window", "commit": guard.get("commit"), "proposal_id": guard.get("proposal_id")}
        if health.get("ok"):
            guard["healthy_cycles"] = int(guard.get("healthy_cycles", 0)) + 1
            if guard["healthy_cycles"] >= self.health_window_cycles:
                guard["completed_at"] = utc_now()
                guard["active"] = False
                self._atomic_write(guard)
                return {"active": False, "ok": True, "completed": True, "commit": guard.get("commit"), "proposal_id": guard.get("proposal_id"), "changed": True}
            self._atomic_write(guard)
            return {"active": True, "ok": True, "healthy_cycles": guard["healthy_cycles"], "changed": True}
        guard["failed"] = True
        guard["failure"] = health
        commit = str(guard["rollback_commit"])
        # Persist the failed health observation before external rollback work.
        self._atomic_write(guard)
        if rollback is None:
            raise RecoveryError("post-reboot health failed and no rollback action is configured")
        try:
            rollback(commit)
        except Exception as exc:
            # A rollback that itself fails must not be recorded as a success:
            # claiming rolled_back while the broken promoted code keeps running
            # would be a false durable record. Keep the window open, persist the
            # failure and re-raise. The window is still bounded - an open window
            # older than max_window_age_seconds is quarantined above - so this
            # cannot loop forever, and the retry is visible to the operator.
            guard["rollback_error"] = str(exc)[:1000]
            guard["rollback_attempts"] = int(guard.get("rollback_attempts", 0)) + 1
            self._atomic_write(guard)
            raise
        guard["rolled_back"] = True
        self._atomic_write(guard)
        return {"active": False, "ok": False, "rolled_back": True, "commit": commit, "proposal_id": guard.get("proposal_id"), "changed": True}

    def _atomic_write(self, value: dict[str, object]) -> None:
        self.state_path.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.state_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.state_path, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
