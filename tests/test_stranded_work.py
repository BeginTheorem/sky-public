"""Stranded work: pending rows nothing can select, and the ledger that has to agree.

Three behaviours, one gap. A task created for a question the ledger already calls resolved
is born unselectable, because `planner_candidates` gives a terminal fingerprint novelty 0.0
and `PortfolioPlanner.rank()` drops it before it is ever a candidate. An outstanding outward
look in that state does not merely idle: the valve's liveness check reads the same row as
work in flight and stays shut behind it. And a row like that, exempted from the status
repair because it belongs to a safety-net template, was never retired either.

Measured on the live ledger 2026-10-18/19: of three non-terminal tasks, two had
`attempts=0` and were 10.0 and 21.4 hours old; the older one was the outward look, the valve
recorded 90 suppressions over 21 hours, and no outward work ran in that time.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from skynet.store import StateStore


def _stamp(hours_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat(timespec="microseconds").replace("+00:00", "Z")


class ReviveHypothesisTests(unittest.TestCase):
    def test_a_resolved_question_reads_terminal_and_reviving_it_makes_it_live(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.register_hypothesis(
                workstream_id="recovery",
                fingerprint="f" * 64,
                structural_fingerprint="s" * 64,
                problem="p",
                expected_behavior="b",
            )
            self.assertFalse(store.fingerprint_is_terminal("f" * 64, "s" * 64, ttl_days=7))
            store.mark_hypothesis("f" * 64, "completed")
            self.assertTrue(store.fingerprint_is_terminal("f" * 64, "s" * 64, ttl_days=7))

            self.assertEqual(store.revive_hypothesis("f" * 64), 1)
            self.assertFalse(
                store.fingerprint_is_terminal("f" * 64, "s" * 64, ttl_days=7),
                "a revived question must read live again, or the instance minted for it is unselectable",
            )
            status = store.connection.execute(
                "SELECT status FROM hypotheses WHERE fingerprint=?", ("f" * 64,)
            ).fetchone()[0]
            self.assertEqual(status, "ready")
            events = [
                row[0]
                for row in store.connection.execute("SELECT kind FROM event_log WHERE kind='hypothesis_revived'")
            ]
            self.assertEqual(events, ["hypothesis_revived"])
            store.close()

    def test_reviving_an_already_live_question_changes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.register_hypothesis(
                workstream_id="recovery",
                fingerprint="f" * 64,
                structural_fingerprint="s" * 64,
                problem="p",
                expected_behavior="b",
            )
            self.assertEqual(store.revive_hypothesis("f" * 64), 0)
            self.assertEqual(
                store.connection.execute(
                    "SELECT COUNT(*) FROM event_log WHERE kind='hypothesis_revived'"
                ).fetchone()[0],
                0,
            )
            store.close()

    def test_a_question_resolved_outside_the_window_is_not_terminal(self) -> None:
        """The TTL that frees the planner frees this check the same way."""
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.register_hypothesis(
                workstream_id="recovery",
                fingerprint="f" * 64,
                structural_fingerprint="s" * 64,
                problem="p",
                expected_behavior="b",
            )
            store.mark_hypothesis("f" * 64, "completed")
            store.connection.execute(
                "UPDATE hypotheses SET updated_at=? WHERE fingerprint=?", (_stamp(48), "f" * 64)
            )
            self.assertFalse(store.fingerprint_is_terminal("f" * 64, "s" * 64, ttl_days=1))
            self.assertTrue(store.fingerprint_is_terminal("f" * 64, "s" * 64, ttl_days=30))
            store.close()


class RetireStrandedTasksTests(unittest.TestCase):
    def _make_task(
        self,
        store: StateStore,
        *,
        fingerprint: str = "f" * 64,
        structural: str = "s" * 64,
        age_hours: float = 8.0,
        attempts: int = 0,
        terminal: bool = True,
        title: str = "stranded",
    ) -> str:
        goal_id = store.add_goal("g", priority=1.0)
        task_id = store.add_task(
            title,
            str(goal_id),
            expected_new_fact="fact",
            hypothesis_fingerprint=fingerprint,
            structural_fingerprint=structural,
            area="recovery",
        )
        if terminal:
            store.register_hypothesis(
                workstream_id="recovery",
                fingerprint=fingerprint,
                structural_fingerprint=structural,
                problem="p",
                expected_behavior="b",
            )
            store.mark_hypothesis(fingerprint, "completed")
        store.connection.execute(
            "UPDATE tasks SET attempts=?, created_at=?, updated_at=? WHERE task_id=?",
            (attempts, _stamp(age_hours), _stamp(age_hours), task_id),
        )
        return task_id

    def test_an_old_unattempted_unselectable_task_is_retired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            task_id = self._make_task(store)
            retired = store.retire_stranded_tasks(older_than_hours=6.0)
            self.assertEqual(retired, [task_id])
            status = store.connection.execute(
                "SELECT status FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()[0]
            self.assertEqual(status, "cancelled")
            payload = json.loads(
                store.connection.execute(
                    "SELECT payload FROM event_log WHERE kind='task_retired_stranded'"
                ).fetchone()[0]
            )
            self.assertEqual(payload["count"], 1)
            self.assertEqual(payload["tasks"][0]["task_id"], task_id)
            self.assertEqual(payload["older_than_hours"], 6.0)
            self.assertEqual(store.retire_stranded_tasks(older_than_hours=6.0), [], "idempotent")
            store.close()

    def test_young_work_is_left_alone(self) -> None:
        """A freshly minted instance must survive its own cycle, or creation churns."""
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            self._make_task(store, age_hours=1.0)
            self.assertEqual(store.retire_stranded_tasks(older_than_hours=6.0), [])
            store.close()

    def test_work_that_was_attempted_is_the_organisms_own_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            self._make_task(store, attempts=1, age_hours=48.0)
            self.assertEqual(store.retire_stranded_tasks(older_than_hours=6.0), [])
            store.close()

    def test_a_question_that_is_still_live_keeps_its_pending_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            self._make_task(store, terminal=False, age_hours=48.0)
            self.assertEqual(store.retire_stranded_tasks(older_than_hours=6.0), [])
            store.close()

    def test_an_excluded_fingerprint_is_never_retired_here(self) -> None:
        """The safety nets are the valve's business: it opens them instead of retiring them."""
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            self._make_task(store, fingerprint="keep" * 16, structural="keep" * 16, age_hours=48.0)
            self.assertEqual(
                store.retire_stranded_tasks(older_than_hours=6.0, exclude_fingerprints=("keep" * 16,)),
                [],
            )
            store.close()

    def test_the_window_is_operator_controlled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            self._make_task(store, age_hours=3.0)
            self.assertEqual(store.retire_stranded_tasks(older_than_hours=6.0), [])
            self.assertEqual(len(store.retire_stranded_tasks(older_than_hours=1.0)), 1)
            store.close()

    def test_a_retired_row_keeps_its_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            task_id = self._make_task(store, title="Validate something")
            store.retire_stranded_tasks(older_than_hours=6.0)
            row = store.connection.execute(
                "SELECT title, area, attempts FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            self.assertEqual(row["title"], "Validate something")
            self.assertEqual(row["area"], "recovery")
            self.assertEqual(row["attempts"], 0)
            store.close()


if __name__ == "__main__":
    unittest.main()
