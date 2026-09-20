from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KeySlot:
    index: int
    key: str


def _coerce_index(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class KeyRotator:
    """Thread-safe round-robin key pool with temporary cooldowns."""

    PERMANENT_BACKOFF_SECONDS = 1800.0

    def __init__(
        self,
        keys: list[str],
        cooldown_seconds: float = 60.0,
        permanent_backoff_seconds: float = PERMANENT_BACKOFF_SECONDS,
    ) -> None:
        self._keys = [key for key in keys if key]
        self._cooldown = max(0.0, cooldown_seconds)
        self._permanent_backoff = max(0.0, permanent_backoff_seconds)
        self._cycle = itertools.cycle(range(len(self._keys))) if self._keys else iter(())
        self._cooled_until: dict[int, float] = {}
        self._permanent: set[int] = set()
        self._lock = threading.RLock()

    @property
    def size(self) -> int:
        return len(self._keys)

    @property
    def total(self) -> int:
        return len(self._keys)

    def next_available(self) -> KeySlot | None:
        with self._lock:
            for _ in range(len(self._keys)):
                index = next(self._cycle)
                if index not in self._permanent and self._cooled_until.get(index, 0.0) <= time.monotonic():
                    return KeySlot(index, self._keys[index])
        return None

    def mark_failed(self, index: int, *, permanent: bool = False, cooldown_seconds: float | None = None) -> None:
        with self._lock:
            if permanent:
                self._permanent.add(index)
            else:
                self._cooled_until[index] = time.monotonic() + (self._cooldown if cooldown_seconds is None else max(0.0, cooldown_seconds))

    def earliest_available_in(self) -> float:
        """Seconds until the next key becomes available; 0.0 when one already is.

        When every key is permanent (or the pool is empty) no cooldown can ever
        expire, so the permanent backoff is returned instead of a zero that
        would make the caller retry immediately.
        """
        now = time.monotonic()
        with self._lock:
            remaining = [
                max(0.0, self._cooled_until.get(index, 0.0) - now)
                for index in range(len(self._keys))
                if index not in self._permanent
            ]
        return min(remaining) if remaining else self._permanent_backoff

    def to_dict(self) -> dict[str, object]:
        """Serialize durable key state as durations, never monotonic stamps."""
        now = time.monotonic()
        with self._lock:
            remaining = {
                str(index): round(max(0.0, until - now), 3)
                for index, until in self._cooled_until.items()
                if until - now > 0
            }
            return {
                "cooldown_seconds": self._cooldown,
                "permanent": sorted(self._permanent),
                "remaining": remaining,
            }

    def restore(self, state: object) -> None:
        """Restore key state from ``to_dict`` output; ignore malformed input."""
        if not isinstance(state, dict):
            return
        permanent = state.get("permanent")
        if isinstance(permanent, (list, tuple, set, frozenset)):
            for item in permanent:
                index = _coerce_index(item)
                if index is not None and 0 <= index < len(self._keys):
                    self._permanent.add(index)
        remaining = state.get("remaining")
        if isinstance(remaining, dict):
            now = time.monotonic()
            for key, value in remaining.items():
                index = _coerce_index(key)
                if index is None or not 0 <= index < len(self._keys):
                    continue
                try:
                    seconds = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if seconds <= 0:
                    continue
                self._cooled_until[index] = now + seconds

    def status(self) -> list[dict[str, object]]:
        now = time.monotonic()
        with self._lock:
            return [{
                "index": index,
                "masked": f"{key[:8]}...{key[-4:]}",
                "available": index not in self._permanent and self._cooled_until.get(index, 0.0) <= now,
                "permanent": index in self._permanent,
                "cooldown_remaining": max(0.0, round(self._cooled_until.get(index, 0.0) - now, 1)),
            } for index, key in enumerate(self._keys)]
