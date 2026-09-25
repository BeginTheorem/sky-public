"""Proof that the run-count transcript retention path behaves as designed.

``prune_run_history`` (``skynet/store.py``) is wired into the checkpoint
(``skynet/reactor.py``) and has never fired on the live ledger -- ~150 runs at
~24/day against a 200-run window -- so its first live firing would also be its
first test. This seeds a synthetic store past the window (aged runs, fresh runs,
and a folded tool result whose address lives in three holders) and drives the
real checkpoint path, asserting the exact deletion set rather than a count.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from helpers import FakeProvider, FixtureTool

from skynet.models import Budget, RunRecord, RunStatus
from skynet.reactor import Reactor, ReactorConfig
from skynet.store import StateStore
from skynet.time import utc_now

KEEP_RUNS = 200
SEEDED_RUNS = 210


class RetentionProofTests(unittest.TestCase):
    def _seed_runs(self, store: StateStore, total: int) -> None:
        """One finished run each, with an address a pruner could orphan."""
        for index in range(total):
            run_id = f"run-{index:03d}"
            key = f"{run_id}:0:call-1"
            store.create_run(RunRecord(run_id, 1, RunStatus.COMPLETED, utc_now(), Budget()))
            store.record_effect(key, "bash", "hash", {"ok": True, "blob": "x" * 200}, "applied")
            store.append_event("tool_result", {"call_id": "call-1", "tool_name": "bash", "result": {"effect_key": key}}, run_id)
            folded = json.dumps({"ok": True, "truncated": True, "preview": "x", "effect_key": key, "full_result_in": "capability_effects.idempotency_key"})
            store.append_transcript("react_history", {"messages": [{"role": "tool", "tool_call_id": "call-1", "content": folded}]}, run_id)
            store.snapshot_episode(run_id)
        store.connection.commit()

    @staticmethod
    def _dangling(store: StateStore) -> list[tuple[str, str]]:
        """Effect addresses kept in any surviving holder that no longer resolve."""
        resolved = {row[0] for row in store.connection.execute("SELECT idempotency_key FROM capability_effects")}
        dangling: list[tuple[str, str]] = []
        for row in store.connection.execute("SELECT payload FROM event_log WHERE kind='tool_result'"):
            key = (json.loads(row["payload"]).get("result") or {}).get("effect_key")
            if key and key not in resolved:
                dangling.append(("event_log", key))
        for row in store.connection.execute("SELECT payload FROM transcript WHERE kind='react_history'"):
            for message in json.loads(row["payload"]).get("messages", []):
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                try:
                    folded = json.loads(content)
                except ValueError:
                    continue
                key = folded.get("effect_key") if isinstance(folded, dict) else None
                if key and key not in resolved:
                    dangling.append(("transcript", key))
        for row in store.connection.execute("SELECT payload FROM episode_snapshots"):
            for event in json.loads(row["payload"]).get("events", []):
                if isinstance(event, dict) and event.get("kind") == "tool_result":
                    key = (event.get("payload", {}).get("result") or {}).get("effect_key")
                    if key and key not in resolved:
                        dangling.append(("snapshot", key))
        return dangling

    def test_transcript_retention_deletes_exactly_the_aged_runs(self) -> None:
        # The checkpoint runs inside its own episode, so the run the tick
        # creates holds one slot of the window: the expectation is not
        # SEEDED_RUNS - KEEP_RUNS but SEEDED_RUNS - KEEP_RUNS + 1. Stating it
        # from the store's own run order avoids hard-coding that off-by-one.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            self._seed_runs(store, SEEDED_RUNS)
            before_transcript = store.connection.execute("SELECT COUNT(*) FROM transcript").fetchone()[0]
            before_snapshots = store.connection.execute("SELECT COUNT(*) FROM episode_snapshots").fetchone()[0]
            before_bytes = store.connection.execute("PRAGMA page_count").fetchone()[0] * store.connection.execute("PRAGMA page_size").fetchone()[0]
            store.close()

            reactor = Reactor(FakeProvider(), {"fixture_tool": FixtureTool()}, ReactorConfig(state_path=path, transcript_retention_runs=KEEP_RUNS))
            self.assertEqual(reactor.tick("retention-proof"), RunStatus.COMPLETED)
            connection = reactor.store.connection

            kept = {str(row[0]) for row in connection.execute("SELECT run_id FROM runs ORDER BY rowid DESC LIMIT ?", (KEEP_RUNS,)).fetchall()}
            aged = tuple(f"run-{index:03d}" for index in range(SEEDED_RUNS) if f"run-{index:03d}" not in kept)
            fresh = tuple(f"run-{index:03d}" for index in range(SEEDED_RUNS) if f"run-{index:03d}" in kept)
            self.assertTrue(aged, "the seed must exceed the window")
            placeholders = ",".join("?" * len(aged))
            fresh_placeholders = ",".join("?" * len(fresh))

            after_transcript = connection.execute("SELECT COUNT(*) FROM transcript").fetchone()[0]
            after_snapshots = connection.execute("SELECT COUNT(*) FROM episode_snapshots").fetchone()[0]
            after_bytes = connection.execute("PRAGMA page_count").fetchone()[0] * connection.execute("PRAGMA page_size").fetchone()[0]
            print(
                f"MEASURED aged={len(aged)} fresh={len(fresh)} "
                f"transcript {before_transcript}->{after_transcript} "
                f"snapshots {before_snapshots}->{after_snapshots} bytes {before_bytes}->{after_bytes}"
            )

            pruned = json.loads(connection.execute("SELECT payload FROM event_log WHERE kind='history_pruned' ORDER BY sequence DESC LIMIT 1").fetchone()[0])
            self.assertEqual(pruned, {"transcript": len(aged), "episodes": len(aged), "keep_runs": KEEP_RUNS})

            # The exact deletion set: every aged run's history is gone ...
            self.assertEqual(connection.execute(f"SELECT COUNT(*) FROM transcript WHERE run_id IN ({placeholders})", aged).fetchone()[0], 0)
            self.assertEqual(connection.execute(f"SELECT COUNT(*) FROM episode_snapshots WHERE run_id IN ({placeholders})", aged).fetchone()[0], 0)
            # ... and no fresh run was touched: its history survives and still resolves.
            self.assertEqual(connection.execute(f"SELECT COUNT(*) FROM transcript WHERE run_id IN ({fresh_placeholders})", fresh).fetchone()[0], len(fresh))
            self.assertEqual(connection.execute(f"SELECT COUNT(*) FROM episode_snapshots WHERE run_id IN ({fresh_placeholders})", fresh).fetchone()[0], len(fresh))
            for run_id in fresh:
                self.assertIsNotNone(reactor.store.effect(f"{run_id}:0:call-1"), f"{run_id} still holds its folded tool result")
            # Retention bounds history only: the audit trail is not pruned by run count.
            self.assertEqual(connection.execute(f"SELECT COUNT(*) FROM event_log WHERE run_id IN ({placeholders})", aged).fetchone()[0], len(aged))
            self.assertEqual(connection.execute(f"SELECT COUNT(*) FROM capability_effects WHERE idempotency_key IN ({placeholders})", tuple(f"{run_id}:0:call-1" for run_id in aged)).fetchone()[0], len(aged))
            # No surviving holder points at a removed row, and the file is intact.
            self.assertEqual(self._dangling(reactor.store), [])
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            # The surviving history is exactly the kept window: the checkpoint's
            # own run and the fresh seed, with nothing aged left behind.
            surviving = {str(row[0]) for row in connection.execute("SELECT DISTINCT run_id FROM transcript WHERE run_id IS NOT NULL").fetchall()}
            self.assertEqual(surviving, kept)
            surviving_snapshots = {str(row[0]) for row in connection.execute("SELECT run_id FROM episode_snapshots").fetchall()}
            self.assertTrue(surviving_snapshots <= kept)
            reactor.close()
