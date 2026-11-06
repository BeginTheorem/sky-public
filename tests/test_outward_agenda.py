"""Owner-directed outward agenda: add life, not only harness polish.

The organism's active goals are harness repairs at priority 4.0, so a curated
outward seed at priority 1.0 loses on criticality every cycle: on the live
ledger ~96% of tool calls read its own database and code. These tests pin the
three levers that rebalance that — a standing curated agenda, a ranking bias
toward outward work, and the planner's world/owner slot — without touching the
procedural criterion or the single-planner invariant.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from skynet import handoff
from skynet.handoff import ROADMAP_AREA, ROADMAP_GOAL_TITLE, ROADMAP_TASKS, seed, seedable_items
from skynet.models import ModelTurn
from skynet.planner import PortfolioPlanner
from skynet.planner_contract import PLANNER_INSTRUCTION
from skynet.reactor import Reactor, ReactorConfig
from skynet.store import StateStore


class _StubProvider:
    """No turn is requested by these tests; the provider only has to exist."""

    def complete(self, messages, *, max_tokens, tools=()):
        return ModelTurn(text="{}", usage_tokens=1)


class OutwardAgendaTests(unittest.TestCase):
    def _reactor(self, directory: str) -> Reactor:
        root = Path(directory)
        return Reactor(
            _StubProvider(),
            {},
            ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root),
        )

    def test_the_curated_agenda_carries_market_work(self) -> None:
        """Owner decision 2026-10-21: the standing agenda is the market.

        It used to be general world work (a paper, a dataset, a science thread); the owner
        moved it to the Moscow Exchange and the strategies a private investor can run there,
        because what the organism does with its cycles is decided by this list.
        """
        titles = [title for title, _ in ROADMAP_TASKS]
        facts = " ".join(fact for _, fact in ROADMAP_TASKS)
        self.assertTrue(any("MOEX" in title for title in titles))
        self.assertTrue(any("strateg" in title.lower() for title in titles))
        self.assertTrue(any("retail" in title.lower() or "private investor" in title.lower() for title in titles))
        self.assertNotIn("SkyNet bottleneck", " ".join(titles))
        # Every acceptance is an acceptance criterion, and a market one at that: a stored
        # artifact, a point-in-time cut-off, and the okama/ISS data path named.
        self.assertTrue(all("Acceptance is" in fact for _, fact in ROADMAP_TASKS))
        self.assertIn("okama", facts)
        self.assertIn("artifacts/", facts)
        self.assertIn("cut-off", facts)

    def test_seedable_items_shrinks_to_empty_after_a_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            self.assertEqual(len(seedable_items(store)), len(ROADMAP_TASKS))
            seed(store, root=directory)
            self.assertEqual(seedable_items(store), [])
            store.close()

    def test_top_up_seeds_outward_work_when_the_agenda_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory)
            store = reactor.store
            # Without the roadmap goal there is nothing to top up; the empty
            # goal table is the fallback arm's job, not the standing one.
            self.assertFalse(reactor._ensure_roadmap_seed(top_up=True))
            goal_id = store.add_goal(ROADMAP_GOAL_TITLE, priority=1.0)
            first_title, first_fact = ROADMAP_TASKS[0]
            hypothesis, structural = handoff._roadmap_fingerprints(first_title, first_fact)
            store.add_task(
                first_title,
                goal_id,
                expected_new_fact=first_fact,
                hypothesis_fingerprint=hypothesis,
                structural_fingerprint=structural,
                area=ROADMAP_AREA,
            )
            store.connection.execute(
                "UPDATE tasks SET status='completed' WHERE area=?", (ROADMAP_AREA,)
            )
            store.connection.commit()
            # One curated item is done and the others were never worked, so the agenda
            # hands out its next round -- one at a time, not all remaining items at
            # once; the entries still waiting stay due for later turns of the wheel.
            self.assertTrue(reactor._ensure_roadmap_seed(top_up=True))
            self.assertEqual(
                store.connection.execute(
                    "SELECT COUNT(*) FROM tasks WHERE area=? AND status='pending'", (ROADMAP_AREA,)
                ).fetchone()[0],
                1,
            )
            # With a live outward item the standing arm is event-quiet.
            self.assertFalse(reactor._ensure_roadmap_seed(top_up=True))
            reactor.close()

    def test_a_blocked_outward_item_does_not_freeze_the_agenda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory)
            store = reactor.store
            goal_id = store.add_goal(ROADMAP_GOAL_TITLE, priority=1.0)
            first_title, first_fact = ROADMAP_TASKS[0]
            hypothesis, structural = handoff._roadmap_fingerprints(first_title, first_fact)
            store.add_task(
                first_title,
                goal_id,
                expected_new_fact=first_fact,
                hypothesis_fingerprint=hypothesis,
                structural_fingerprint=structural,
                area=ROADMAP_AREA,
            )
            store.connection.execute("UPDATE tasks SET status='blocked' WHERE area=?", (ROADMAP_AREA,))
            store.connection.commit()
            # A block is terminal until recovery and stays visible; the agenda
            # continues rather than freezing on it.
            self.assertTrue(reactor._ensure_roadmap_seed(top_up=True))
            reactor.close()

    def test_outward_bonus_lifts_roadmap_into_contention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            harness_goal = store.add_goal("repair the harness", priority=4.0)
            store.add_task("fix a harness defect", harness_goal, area="engineering")
            roadmap_goal = store.add_goal(ROADMAP_GOAL_TITLE, priority=1.0)
            store.add_task("surf the open web", roadmap_goal, area=ROADMAP_AREA)
            # On criticality alone the 4.0 harness goal wins every cycle.
            unbiased = PortfolioPlanner(store, outward_weight=0.0).rank()
            self.assertEqual(unbiased[0].workstream_id, "engineering")
            # The owner-directed bias brings the outward seed back into play.
            biased = PortfolioPlanner(store, outward_weight=0.40).rank()
            self.assertEqual(biased[0].workstream_id, ROADMAP_AREA)
            store.close()

    def test_the_planner_reserves_slots_for_the_market(self) -> None:
        """The market slot is a directive in the prompt, not a hope: two of three proposals."""
        self.assertIn("Reserve at least two of the three slots for the market", PLANNER_INSTRUCTION)
        self.assertIn("point-in-time dataset", PLANNER_INSTRUCTION)
        self.assertIn("MOEX", PLANNER_INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
