"""The planner's searched recall query must be readable after the fact.

`Reactor._memory_query` (the injected-run assembler) has had a durable row since
it was made visible; `Reactor._planner_memory_query` -- the query the PLANNING
decision is actually made from -- had none, so a part-order question about it
could only be answered by rebuilding the query from the StartEnvelope, and the
rebuild's own choices (front cap, ledger copy) change the answer. These tests pin
the writer, the ablation it carries, and the shipped order.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from helpers import FakeProvider

from skynet.memory_store import MemoryStore
from skynet.reactor import Reactor, ReactorConfig
from skynet.store import PROTECTED_EVENT_KINDS


class PlannerRecallAuditTests(unittest.TestCase):
    def _reactor(self, directory: str) -> Reactor:
        root = Path(directory)
        return Reactor(
            FakeProvider(),
            {},
            ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root),
        )

    def _last_row(self, reactor: Reactor):
        row = reactor.store.connection.execute(
            "SELECT payload FROM event_log WHERE kind='planner_memory_recall' ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def test_the_planner_recall_query_is_recorded_with_its_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory)
            state = reactor.store.state()
            state.next_plan = {
                "previous_outcome": {
                    "report": "the gate rejected a patch because the anchor was stale " * 20,
                    "status": "COMPLETED",
                },
                "initial_prompt": "retry with a multi-line anchor",
            }
            goals = [{"title": "Advance the SkyNet roadmap"}]
            query = reactor._planner_memory_query(state, goals)
            reactor._record_planner_recall(query, [{"memory_id": "m1"}], state, goals)
            payload = self._last_row(reactor)
            self.assertIsNotNone(payload)
            assert payload is not None
            # The row is the query that was searched, not a paraphrase of it.
            self.assertEqual(payload["query"], query)
            self.assertEqual(payload["query_chars"], len(query))
            self.assertEqual(payload["hits"], 1)
            effective = len(MemoryStore._normalize_terms(query))
            self.assertEqual(payload["query_terms_effective"], effective)
            self.assertGreater(effective, 0)
            # The ablation is measured against the real query, so it must equal
            # the term difference the planner-recall analyses compute by hand.
            blanked = reactor._planner_query_without_previous_outcome(state, goals)
            self.assertEqual(len(blanked), 1)
            expected = len(
                set(MemoryStore._normalize_terms(query))
                - set(MemoryStore._normalize_terms(blanked[0]))
            )
            self.assertEqual(payload["previous_outcome_terms"], expected)
            self.assertEqual(payload["goal_title_terms"], 3)
            reactor.close()

    def test_a_short_previous_outcome_is_alive_in_the_window(self) -> None:
        """A short report must not be silently dead: the ablation has to move.

        The part sits first and only the LAST 24 terms survive, so whether it
        contributes anything depends on how long the parts after it are. Pinning
        both regimes keeps the writer honest about that.
        """
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory)
            state = reactor.store.state()
            state.next_plan = {
                "previous_outcome": {"report": "a distinctive prior marker", "status": "COMPLETED"},
                "initial_prompt": "short prompt",
            }
            goals = [{"title": "a goal"}]
            query = reactor._planner_memory_query(state, goals)
            reactor._record_planner_recall(query, [], state, goals)
            payload = self._last_row(reactor)
            assert payload is not None
            self.assertGreater(payload["previous_outcome_terms"], 0)
            reactor.close()

    def test_an_envelope_without_a_previous_outcome_reports_no_contribution(self) -> None:
        """No previous outcome must read as 0, not as "the part contributed everything"."""
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory)
            state = reactor.store.state()
            state.next_plan = {"initial_prompt": "continue the bounded work", "next": "retry"}
            goals = [{"title": "Advance the SkyNet roadmap"}]
            query = reactor._planner_memory_query(state, goals)
            self.assertEqual(reactor._planner_query_without_previous_outcome(state, goals), [])
            reactor._record_planner_recall(query, [], state, goals)
            payload = self._last_row(reactor)
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertEqual(payload["previous_outcome_terms"], 0)
            self.assertEqual(payload["hits"], 0)
            reactor.close()

    def test_the_row_survives_retention_like_the_other_planner_post_mortems(self) -> None:
        self.assertIn("planner_memory_recall", PROTECTED_EVENT_KINDS)

    def test_the_shipped_part_order_is_pinned(self) -> None:
        """A positional change must update this audit instead of landing silently.

        The measured position of the previous-outcome part is what generations
        191-223 argued about; pinning it here means a reorder cannot be promoted
        without a test that says so.
        """
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory)
            state = reactor.store.state()
            state.next_plan = {
                "previous_outcome": {"report": "a unique previous marker", "status": "COMPLETED"},
                "initial_prompt": "an initial prompt marker",
            }
            goals = [{"title": "a goal marker"}]
            query = reactor._planner_memory_query(state, goals)
            self.assertTrue(query.startswith("a unique previous marker"))
            self.assertTrue(query.rstrip().endswith("a goal marker"))
            self.assertIn("an initial prompt marker", query)
            reactor.close()


if __name__ == "__main__":
    unittest.main()
