"""Result orderings that terminate in a non-unique key are not repeatable.

A query that orders by a column two rows can share leaves the returned order to
the row's physical position, so two ledgers holding the same logical corpus in
different insertion orders answer differently. Both sites below were measured
that way, read-only, on the live ledger.

This is the repeatability failure Lin & Yang describe for indexer-assigned ids
(their Lucene tie-breaking note, arXiv:1807.05798v2): the fix is to terminate
the key in the stable external id -- ``task_id``, ``goal_id`` -- never in a
value the corpus is allowed to duplicate.

CENSUS (generation 226) -- every tracked site in ``skynet/`` that returns an
ordered result, found with these patterns::

    grep -rn "ORDER BY" skynet/ scripts/ --include=*.py
    grep -rn "sorted(" skynet/ --include=*.py

and probed by rebuilding the live ledger twice, once with every table's rows
inserted forward and once reversed, then reading each site through the shipped
code path. Verdicts are the two-ledger probe, not the key text:

  site (file:line)                     ordering                     terminator  probe
  skynet/reactor.py:2257               updated_at DESC               NONE        ORDER-DEPENDENT (fixed here)
  skynet/store.py:2055                 priority DESC, created_at     created_at  stable
  skynet/store.py:2056                 deadline IS NULL, deadline    NONE        stable (no deadlines live)
  skynet/store.py:2089                 g.priority DESC, t.created_at created_at  stable
  skynet/store.py:2158                 quality DESC                  NONE        stable (updated_at distinct)
  skynet/store.py:2165                 quality DESC                  NONE        stable (updated_at distinct)
  skynet/planner.py:301                priority DESC                 NONE        ORDER-DEPENDENT (fixed here)
  skynet/store.py:3092                 confidence DESC, updated_at   NONE        stable (updated_at distinct)
  skynet/store.py:635 / :725 / :747    created_at / last_seen_at     NONE        stable (no live ties)
  skynet/store.py:2027                 finished_at DESC              NONE        stable (no live ties)
  skynet/reporting.py:84               started_at DESC, rowid DESC   rowid       stable
  skynet/store.py:1869                 rowid                         rowid       stable
  skynet/store.py:3002                 created_at                    NONE        stable (no live ties)
  skynet/judge_health.py:126           finished_at DESC              NONE        stable (no live ties)
  skynet/store.py:3078                 created_at, decision_id DESC  decision_id stable
  skynet/autonomous_planner.py:270     p.created_at DESC             NONE        stable (no live ties)
  skynet/store.py:2726                 p.created_at                  NONE        stable (no live ties)
  skynet/reporting.py:131              created_at DESC, sequence     sequence    stable

Three sites probed order-dependent; the two whose output feeds a decision are
fixed here. ``planner.py:301`` and ``reactor.py:2257`` each changed their answer
while the logical corpus was identical, and both are no-ops on the live corpus
in the observed order (the same goal, ``eaa65414``, and the same 40 rows).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from skynet.models import ModelTurn
from skynet.planner import PortfolioPlanner
from skynet.reactor import Reactor, ReactorConfig
from skynet.store import StateStore

TIE = "2026-09-25T12:34:30.519284Z"


def _idea(cell: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cell_key": cell,
        "subsystem": "memory",
        "change_type": "workflow",
        "evidence_source": "own-repo",
        "title": "ordering candidate",
        "quality": 0.5,
        "novelty": 0.5,
    }
    payload.update(extra)
    return payload


class _WindowProvider:
    """Answers the planner with no proposals and keeps the payload it was given."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def complete(self, messages, *, max_tokens, tools=()):
        self.payloads.append(json.loads(messages[1]["content"]))
        return ModelTurn(text=json.dumps({"proposals": []}))

    def reset(self) -> None:
        return None


class PlannerTaskWindowOrderTests(unittest.TestCase):
    """The planner prompt's task window must not depend on physical row order.

    ``Reactor._run_autonomous_planning`` reads tasks with ``ORDER BY updated_at
    DESC LIMIT 100`` into ``AutonomousPlanner._bounded_payload``'s 40-row
    ``PAYLOAD_TASK_WINDOW``. ``updated_at`` is not unique: the live ledger (184
    tasks) holds a five-row group sharing ``2026-09-25T12:34:30.519284Z``, and
    on two ledgers holding that corpus in opposite physical orders ranks 18, 19,
    21 and 22 of the window swap -- the four rows the planner is shown differ
    while the corpus does not. The stake is membership, not only order: when
    such a group straddles the 40-row cut the two builds carry different rows,
    and a planner shown a different window can re-propose work the missing row
    had already answered.
    """

    @staticmethod
    def _stamp(rank: int) -> str:
        """A descending, unique timestamp per rank, so rank == order position."""
        return f"2026-09-24T{23 - rank // 60:02d}:{59 - rank % 60:02d}:00.000000Z"

    def _build(self, path: Path, order: str) -> None:
        store = StateStore(path)
        goal_id = store.add_goal("window goal", priority=1.0)
        ranks = list(range(120))
        if order == "reverse":
            ranks.reverse()
        for rank in ranks:
            task_id = f"task-{rank:03d}"
            created = self._stamp(rank)
            # Every task is terminal, so the portfolio is empty and the
            # autonomous planner is the only path left that can run.
            store.connection.execute(
                "INSERT INTO tasks(task_id, goal_id, title, status, attempts, idempotency_key,"
                " created_at, updated_at, hypothesis_fingerprint, structural_fingerprint,"
                " expected_new_fact, area) VALUES (?,?,?,'blocked',0,?,?,?,?,?,'x','engineering')",
                (task_id, goal_id, task_id, task_id, created, created, f"fp-{task_id}", f"sf-{task_id}"),
            )
        # One five-row group sharing an updated_at, straddling rank 40.
        for rank in range(38, 43):
            store.connection.execute("UPDATE tasks SET updated_at=? WHERE task_id=?", (TIE, f"task-{rank:03d}"))
        store.connection.commit()
        store.close()

    def _window(self, order: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            self._build(path, order)
            provider = _WindowProvider()
            reactor = Reactor(provider, {}, ReactorConfig(state_path=path))
            reactor.tick("timer")
            reactor.close()
            self.assertTrue(provider.payloads, "the autonomous planner was never called")
            return [task["task_id"] for task in provider.payloads[0]["tasks"]]

    def test_the_planner_window_does_not_depend_on_physical_row_order(self) -> None:
        self.assertEqual(self._window("forward"), self._window("reverse"))


class ArchiveGoalOrderTests(unittest.TestCase):
    """The goal a materialized idea attaches to must not depend on row order.

    ``PortfolioPlanner._select_from_archive`` picks the active goal with
    ``ORDER BY priority DESC LIMIT 1``. Goals may share a priority -- the live
    ledger holds two at 4.0 -- so the tie was decided by whichever row the
    planner read first. Measured read-only on that ledger, the query returns
    ``eaa65414`` under the forward order and ``dcc0e38a`` under a reversed one,
    so the same archive parent could be materialized against a different goal.
    """

    GOALS = (
        ("goal-one", "first equal-priority goal", "2026-09-20T00:00:00.000000Z"),
        ("goal-two", "second equal-priority goal", "2026-09-21T00:00:00.000000Z"),
    )

    def _chosen_goal(self, order: str) -> str:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            rows = self.GOALS if order == "forward" else tuple(reversed(self.GOALS))
            for goal_id, title, created in rows:
                store.connection.execute(
                    "INSERT INTO goals(goal_id,title,status,priority,constraints,next_action,outcome,created_at,updated_at)"
                    " VALUES (?,?,'active',4.0,'{}','','{}',?,?)",
                    (goal_id, title, created, created),
                )
            store.archive_idea(_idea("memory|workflow|own-repo"))
            store.connection.commit()
            work = PortfolioPlanner(store)._select_from_archive(record=False)
            self.assertIsNotNone(work)
            task = cast(dict, cast(dict, work)["task"])
            chosen = store.connection.execute(
                "SELECT goal_id FROM tasks WHERE task_id=?", (task["task_id"],)
            ).fetchone()[0]
            store.close()
            return str(chosen)

    def test_an_archive_task_attaches_to_the_same_goal_in_both_row_orders(self) -> None:
        self.assertEqual(self._chosen_goal("forward"), self._chosen_goal("reverse"))


if __name__ == "__main__":
    unittest.main()
