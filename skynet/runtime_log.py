from __future__ import annotations

import json
import os
import re
from pathlib import Path
from threading import Lock
from typing import Any

from .time import utc_now

_VERBOSE_MAX_BYTES = 50_000_000
_VERBOSE_FILE_MODE = 0o600
_VERBOSE_TRUTHY = {"1", "true", "yes", "on"}
_verbose_lock = Lock()

# Obvious credentials must not survive into the operator-readable dump. The
# patterns cover the provider keys this harness sees in practice.
_SECRET_PATTERNS = (
    (re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-]{6,}"), "[REDACTED]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+"), "Bearer [REDACTED]"),
    (re.compile(r'(?i)"api_key"\s*:\s*"[^"]*"'), '"api_key":"[REDACTED]"'),
    (re.compile(r'(?i)"token"\s*:\s*"[^"]*"'), '"token":"[REDACTED]"'),
)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in _VERBOSE_TRUTHY


def _state_path(base_path: str | Path | None) -> Path:
    if base_path is not None:
        return Path(base_path)
    return Path(os.getenv("SKYNET_STATE", "state/skynet.sqlite3"))


def _redact(text: str) -> str:
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def verbose_enabled(base_path: str | Path | None = None) -> bool:
    """Whether full provider dumps are requested.

    The env var is the runtime override; ``verbose.json`` next to the state
    database is the hot-reload toggle written by ``skynet verbose on``. Both
    are read lazily so a running service picks up the file without a restart.
    """
    if _truthy(os.getenv("SKYNET_VERBOSE_PROVIDER")):
        return True
    try:
        data = json.loads(_state_path(base_path).with_name("verbose.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return bool(data.get("enabled", False)) if isinstance(data, dict) else False


def _compact(path: Path, limit: int) -> None:
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError:
        return
    kept: list[bytes] = []
    total = 0
    for line in reversed(lines):
        if total + len(line) > limit:
            break
        kept.append(line)
        total += len(line)
    # A single oversized record must not blank the file entirely.
    if not kept and lines:
        kept = [lines[-1]]
    temporary = path.with_name(f"{path.name}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, _VERBOSE_FILE_MODE)
    try:
        os.write(descriptor, b"".join(reversed(kept)))
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def verbose_write(
    kind: str,
    payload: Any,
    *,
    base_path: str | Path,
    run_id: str | None = None,
    max_bytes: int | None = None,
) -> None:
    """Append one full provider dump, redacted, to ``verbose.jsonl``.

    Disabled unless :func:`verbose_enabled` is true. The file is a byte ring:
    when it grows past the cap it is rewritten with the newest records that
    fit. Logging failures never propagate to the provider path.
    """
    try:
        if not verbose_enabled(base_path):
            return
        path = Path(base_path).with_name("verbose.jsonl")
        record = {"timestamp": utc_now(), "kind": kind, "run_id": run_id, "payload": payload}
        line = _redact(json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":")) + "\n")
        limit = _VERBOSE_MAX_BYTES if max_bytes is None else max(0, int(max_bytes))
        with _verbose_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            existed = path.exists()
            descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, _VERBOSE_FILE_MODE)
            try:
                os.write(descriptor, line.encode("utf-8"))
            finally:
                os.close(descriptor)
            if not existed:
                os.chmod(path, _VERBOSE_FILE_MODE)
            if limit and path.stat().st_size > limit:
                _compact(path, limit)
    except Exception:
        # Verbose diagnostics must not prevent a provider/tool action.
        return


class RuntimeLog:
    """Append-only JSONL journal for operational and model-facing analysis."""

    def __init__(self, path: str | Path, *, max_bytes: int = 25_000_000, backups: int = 3, kinds: str | None = None) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self.max_bytes = max(0, max_bytes)
        self.backups = max(0, backups)
        # An optional allowlist of kinds (SKYNET_RUNTIME_LOG_KINDS). The event log
        # is the durable source of truth; this file is a projection, and on a busy
        # day it is 25 MB of mostly tool chatter.
        self.kinds = {item.strip() for item in (kinds or "").split(",") if item.strip()}

    def _rotate_locked(self, incoming_size: int) -> None:
        if not self.max_bytes or not self.path.exists() or self.path.stat().st_size + incoming_size <= self.max_bytes:
            return
        for index in range(self.backups, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index - 1}") if index > 1 else self.path
            target = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                source.replace(target)

    def write(self, kind: str, payload: Any, *, run_id: str | None = None) -> None:
        if self.kinds and kind not in self.kinds:
            return
        record = {
            "timestamp": utc_now(),
            "kind": kind,
            "run_id": run_id,
            "payload": payload,
        }
        line = json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":")) + "\n"
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_locked(len(line.encode("utf-8")))
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line)
                    stream.flush()
            except OSError:
                # Operational logging must not prevent a provider/tool action.
                return
