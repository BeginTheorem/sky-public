"""Provider diagnostics must survive a worker thread and never abort the ladder.

The memory loop and the planner call the provider chain on worker threads, and
the fallback provider logs through ``StateStore.record_provider_event``. The
canonical SQLite connection belongs to the reactor thread, so a durable write
from a worker used to raise "SQLite objects created in a thread can only be used
in that same thread" and replace the real provider failure with that error.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from skynet.providers.fallback import FallbackProvider
from skynet.store import StateStore


class _Stub:
    name = "stub"

    def complete(self, messages: Any, *, max_tokens: int, tools: Any = ()) -> Any:
        raise NotImplementedError


class _ThreadedRecordingTests(unittest.TestCase):
    def test_durable_provider_event_from_worker_thread_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            try:
                errors: list[BaseException] = []

                def worker() -> None:
                    try:
                        store.record_provider_event("fallback_failure", {"provider": "openrouter", "category": "network"})
                    except BaseException as exc:
                        errors.append(exc)

                thread = threading.Thread(target=worker)
                thread.start()
                thread.join()
                self.assertEqual(errors, [], "a durable provider event must not raise off the owner thread")
                rows = store.connection.execute("SELECT payload FROM event_log WHERE kind='fallback_failure'").fetchall()
                self.assertEqual(len(rows), 1)
            finally:
                store.close()

    def test_non_durable_provider_event_from_worker_thread_stays_in_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            try:
                errors: list[BaseException] = []

                def worker() -> None:
                    try:
                        store.record_provider_event("fallback_attempt", {"provider": "openrouter", "attempt": 1})
                    except BaseException as exc:
                        errors.append(exc)

                thread = threading.Thread(target=worker)
                thread.start()
                thread.join()
                self.assertEqual(errors, [])
                rows = store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='fallback_attempt'").fetchall()
                self.assertEqual(rows[0][0], 0)
            finally:
                store.close()


class _LadderResilienceTests(unittest.TestCase):
    def test_event_logger_failure_never_aborts_the_ladder(self) -> None:
        seen: list[str] = []

        def boom(event: str, payload: dict[str, object]) -> None:
            seen.append(event)
            raise RuntimeError("sink down")

        provider = FallbackProvider([_Stub()], event_logger=boom)
        provider._log("fallback_failure", {"provider": "stub"})  # must not raise
        self.assertEqual(seen, ["fallback_failure"])


if __name__ == "__main__":
    unittest.main()
