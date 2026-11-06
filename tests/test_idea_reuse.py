"""An archived idea becomes work once -- never the same work twice.

Measured on the live ledger (2026-10-16): the probe idea was materialized six
times into the same defect that no longer existed, because the cell rule keeps
the best-evidenced idea live and every task carried its own id as its
idempotency key -- a key that de-duplicates nothing. These tests pin the three
outcomes of ``materialize_idea`` and the planner sampling past spent ideas.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from skynet.planner import PortfolioPlanner
from skynet.store import StateStore


def _idea(cell: str, *, quality: float, title: str) -> dict[str, object]:
    """A minimal, well-formed archive row for a behavioural descriptor cell."""
    return {
        "cell_key": cell,
        "subsystem": "general",
        "change_type": "workflow",
        "evidence_source": "own-repo",
        "title": title,
        "quality": quality,
    }


class IdeaReuseTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = StateStore(Path(self._tmp.name) / "state.sqlite3")
        self.goal_id = self.store.add_goal("portfolio", priority=4.0)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def archive(self, cell: str, *, quality: float = 1.0, title: str = "an idea") -> str:
        idea_id = self.store.archive_idea(_idea(cell, quality=quality, title=title))
        self.assertIsNotNone(idea_id)
        assert idea_id is not None
        return str(idea_id)

    def complete(self, task_id: str) -> None:
        self.store.apply_task_updates([{"task_id": task_id, "status": "completed"}], None)

    def scalar(self, sql: str, *args: object) -> object:
        return self.store.connection.execute(sql, args).fetchone()[0]

    def event(self, kind: str) -> dict[str, object]:
        row = self.store.connection.execute(
            "SELECT payload FROM event_log WHERE kind=? ORDER BY sequence DESC LIMIT 1", (kind,)
        ).fetchone()
        payload = json.loads(row["payload"]) if row is not None else {}
        return dict(payload)

    def test_a_live_task_comes_back_instead_of_a_twin(self) -> None:
        idea = self.archive("general|workflow|own-repo")
        first = self.store.materialize_idea(idea, self.goal_id)
        self.assertIsNotNone(first)
        second = self.store.materialize_idea(idea, self.goal_id)
        self.assertEqual(second, first, "the second caller is handed the live task")
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM tasks"), 1, "no twin task")
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM event_log WHERE kind='idea_materialized'"), 1, "no twin event")

    def test_the_key_names_the_idea_not_the_task_itself(self) -> None:
        idea = self.archive("general|workflow|own-repo")
        task_id = self.store.materialize_idea(idea, self.goal_id)
        key = self.scalar("SELECT idempotency_key FROM tasks WHERE task_id=?", task_id)
        self.assertEqual(key, f"idea:{idea}", "the key is the idea's, so a duplicate cannot slip in under a new id")

    def test_a_legacy_caller_keeps_its_own_id_as_key(self) -> None:
        task_id = self.store.add_task("plain task", self.goal_id)
        self.assertEqual(self.scalar("SELECT idempotency_key FROM tasks WHERE task_id=?", task_id), task_id)

    def test_a_spent_idea_is_retired_and_costs_no_second_task(self) -> None:
        idea = self.archive("general|workflow|own-repo")
        task_id = self.store.materialize_idea(idea, self.goal_id)
        assert task_id is not None
        self.complete(task_id)
        self.assertIsNone(self.store.materialize_idea(idea, self.goal_id), "a finished idea is not re-opened")
        self.assertEqual(self.scalar("SELECT status FROM idea_archive WHERE idea_id=?", idea), "superseded")
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM tasks"), 1, "the same work was not created again")
        payload = self.event("idea_superseded")
        self.assertEqual(payload["reason"], "spent")
        self.assertEqual(payload["task_status"], "completed")
        self.assertEqual(self.store.live_ideas(), [], "a spent idea stops occupying its cell")

    def test_the_planner_continues_past_a_spent_candidate(self) -> None:
        """The loop's whole point: a spent candidate costs a retirement, not the cycle.

        The draw order is pinned here (spent first) because the real sampler is
        weighted and could legitimately hand the fresh candidate over first --
        which would prove nothing about continuing.
        """
        spent = self.archive("general|workflow|own-repo", title="the defect that no longer exists")
        done = self.store.materialize_idea(spent, self.goal_id)
        assert done is not None
        self.complete(done)
        self.archive("general|eval-improvement|own-repo", title="a genuinely new idea")
        drawn = self.store.sample_archive_parents

        def spent_first(*, k: int = 2, lam: float = 10.0, alpha0: float = 0.5) -> list[dict[str, object]]:
            rows = drawn(k=k, lam=lam, alpha0=alpha0)
            rows.sort(key=lambda row: str(row["idea_id"]) != spent)
            return rows

        with mock.patch.object(self.store, "sample_archive_parents", spent_first):
            work = PortfolioPlanner(self.store)._select_from_archive(record=False)
        self.assertIsNotNone(work, "the fresh candidate becomes work")
        assert work is not None
        self.assertEqual(work["task"]["title"], "a genuinely new idea")
        self.assertEqual(self.scalar("SELECT status FROM idea_archive WHERE idea_id=?", spent), "superseded")
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM tasks"), 2, "one task for the finished idea, one for the fresh one")

    def test_an_exhausted_archive_says_so(self) -> None:
        idea = self.archive("general|workflow|own-repo")
        task_id = self.store.materialize_idea(idea, self.goal_id)
        assert task_id is not None
        self.complete(task_id)
        self.assertIsNone(PortfolioPlanner(self.store)._select_from_archive(record=False))
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM tasks"), 1)


if __name__ == "__main__":
    unittest.main()
