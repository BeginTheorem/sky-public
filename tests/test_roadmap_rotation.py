"""The curated outward agenda rotates instead of draining once.

Measured 2026-10-21: the seven curated items were seeded on 2026-10-18, every round
completed inside that day, and ``seedable_items()`` offered nothing for the next three
days -- the outward agenda was empty while the planner fell back on harness work, and
nothing reported it. These tests pin the rotation that replaces the one-shot pool:

* an entry comes round again once the wheel has turned;
* the re-asked round is **selectable** and survives the cycle's own status repair -- it carries
  its own fingerprint, because a round over the finished one's fingerprint is flipped to terminal
  within a tick (`repair_status_consistency` aligns every ready hypothesis that has a completed
  task) and then cancelled as work nothing can select;
* the wheel paces the agenda (no burst of seven re-asked items at once), and one entry
  never has two live rounds;
* a stopped agenda says so, once, on its own.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from skynet import handoff
from skynet.handoff import (
    ROADMAP_AREA,
    ROADMAP_GOAL_TITLE,
    ROADMAP_ROTATION_HOURS,
    ROADMAP_TASKS,
    seed,
    seedable_items,
)
from skynet.models import ModelTurn
from skynet.planner import PortfolioPlanner
from skynet.reactor import EXTERNAL_SEEK_EXPECTED_FACT, EXTERNAL_SEEK_TASK_TITLE, Reactor, ReactorConfig
from skynet.store import StateStore
from skynet.time import utc_datetime_now

TURN_HOURS = ROADMAP_ROTATION_HOURS / len(ROADMAP_TASKS)


class _StubProvider:
    """These tests never request a turn; the provider only has to exist."""

    def complete(self, messages, *, max_tokens, tools=()):
        return ModelTurn(text="{}", usage_tokens=1)


class RoadmapRotationTests(unittest.TestCase):
    def _reactor(self, directory: str) -> Reactor:
        root = Path(directory)
        return Reactor(
            _StubProvider(),
            {},
            ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root),
        )

    def _age_agenda(self, store: StateStore, *, hours: float) -> None:
        """Move every curated round ``hours`` into the past."""
        stamps = (utc_datetime_now() - timedelta(hours=hours)).isoformat(timespec="microseconds").replace("+00:00", "Z")
        store.connection.execute(
            "UPDATE tasks SET created_at=?, updated_at=? WHERE area=?", (stamps, stamps, ROADMAP_AREA)
        )
        store.connection.commit()

    def _finish_every_round(self, store: StateStore) -> None:
        """Complete every curated round the way the cycle does, hypotheses included."""
        store.connection.execute("UPDATE tasks SET status='completed' WHERE area=?", (ROADMAP_AREA,))
        store.connection.commit()
        store.repair_status_consistency()

    def _count(self, store: StateStore, sql: str, *params: object) -> int:
        return int(store.connection.execute(sql, params).fetchone()[0])

    def _alerts(self, store: StateStore, kind: str = "roadmap_agenda_starved") -> int:
        return self._count(store, "SELECT COUNT(*) FROM alerts WHERE kind=?", kind)

    def test_a_finished_round_comes_round_again_and_is_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            seed(store, root=directory)
            self._finish_every_round(store)
            title, fact = ROADMAP_TASKS[0]
            first, first_structural = handoff._roadmap_fingerprints(title, fact)
            # The round is done and the ledger agrees: this is exactly the state that used
            # to make every later round unselectable.
            self.assertTrue(store.fingerprint_is_terminal(first, first_structural, ttl_days=30.0))
            # Too soon: the wheel has not come round yet, so nothing is offered.
            self.assertEqual(seedable_items(store), [])
            self._age_agenda(store, hours=TURN_HOURS + 1.0)
            # The wheel is open again and offers exactly one round -- the least recently worked
            # entry. All seven rounds share this timestamp, so the item order decides which.
            self.assertEqual([item[0] for item in seedable_items(store)], [title])
            summary = seed(store, root=directory, limit=1)
            self.assertEqual((summary["tasks_created"], summary["tasks_rotated"]), (1, 1))
            # The new round is its own identity, so the per-cycle repair -- which marks
            # completed every ready hypothesis whose fingerprint has a finished task --
            # cannot flip it back to terminal and then cancel it as unselectable work.
            store.repair_status_consistency()
            self.assertEqual(
                self._count(store, "SELECT COUNT(*) FROM tasks WHERE area=? AND status='pending'", ROADMAP_AREA),
                1,
                "the rotated round survives the cycle's own status repair",
            )
            second, second_structural = handoff._roadmap_fingerprints(title, fact, round_index=1)
            self.assertNotEqual((first, first_structural), (second, second_structural))
            self.assertFalse(store.fingerprint_is_terminal(second, second_structural, ttl_days=30.0))
            # And the round that finished stays finished: it keeps its own fingerprint.
            self.assertTrue(store.fingerprint_is_terminal(first, first_structural, ttl_days=30.0))
            ranked = PortfolioPlanner(store, outward_weight=0.40).rank()
            self.assertIn(ROADMAP_AREA, [item.workstream_id for item in ranked])
            store.close()

    def test_the_wheel_is_closed_an_hour_after_a_round(self) -> None:
        """The boundary the first version of the gate never sat on.

        The interval is one turn -- about seven hours at the default -- and the gate compared
        it against a difference in *seconds*, so any agenda more than seven seconds old read as
        "the wheel has come round" and the whole agenda was re-asked on every cycle. Measured
        live on 2026-10-21/22: 65 rounds in 14.1 hours, 4.6 an hour, 55.8M tokens, and every
        item re-asked eight to twelve times until she refused to work. One hour is far inside a
        turn and far outside seven seconds, which is exactly the gap the old fixtures skipped.
        """
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            self.assertAlmostEqual(handoff._wheel_interval_seconds(ROADMAP_ROTATION_HOURS), TURN_HOURS * 3600.0, places=3)
            seed(store, root=directory)
            self._finish_every_round(store)
            self._age_agenda(store, hours=1.0)
            self.assertEqual(seedable_items(store), [], "one hour is not a turn")
            self.assertEqual(seed(store, root=directory)["tasks_created"], 0, "and the wheel must hold")
            store.close()

    def test_only_one_rotated_round_per_call_whatever_the_limit(self) -> None:
        """``limit=None`` is the startup seed and the empty-goal fallback; the wheel still paces."""
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            seed(store, root=directory)
            self._finish_every_round(store)
            self._age_agenda(store, hours=TURN_HOURS + 1.0)
            # The whole agenda is due by the gate, and one call may still take only one round:
            # without this, seven rounds came out sharing one microsecond of created_at.
            summary = seed(store, root=directory)
            self.assertEqual((summary["tasks_created"], summary["tasks_rotated"]), (1, 1))
            self.assertEqual(
                self._count(store, "SELECT COUNT(*) FROM tasks WHERE area=? AND status='pending'", ROADMAP_AREA), 1
            )
            self.assertEqual(seedable_items(store), [], "the next round waits for the next turn")
            store.close()

    def test_a_fresh_store_still_seeds_the_whole_agenda_at_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            self.assertEqual(len(seedable_items(store)), len(ROADMAP_TASKS))
            summary = seed(store, root=directory)
            self.assertEqual(summary["tasks_created"], len(ROADMAP_TASKS))
            self.assertEqual(summary["tasks_rotated"], 0, "the first pass is not a rotation")
            store.close()

    def test_a_live_round_blocks_a_second_one_of_the_same_item(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            seed(store, root=directory)
            self._age_agenda(store, hours=ROADMAP_ROTATION_HOURS * 2)
            summary = seed(store, root=directory)
            self.assertEqual(summary["tasks_created"], 0, "every entry already has a live round")
            self.assertEqual(
                self._count(store, "SELECT COUNT(*) FROM tasks WHERE area=?", ROADMAP_AREA), len(ROADMAP_TASKS)
            )
            store.close()

    def test_the_standing_arm_hands_out_one_round_at_a_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory)
            seed(reactor.store, root=directory)
            self._finish_every_round(reactor.store)
            self._age_agenda(reactor.store, hours=TURN_HOURS + 1.0)
            self.assertTrue(reactor._ensure_roadmap_seed(top_up=True))
            live = self._count(
                reactor.store, "SELECT COUNT(*) FROM tasks WHERE area=? AND status='pending'", ROADMAP_AREA
            )
            self.assertEqual(live, 1, "one round leaves at a time")
            # Quiet while that round is out, and quiet without a rotation due.
            self.assertFalse(reactor._ensure_roadmap_seed(top_up=True))
            reactor.close()

    def test_a_stopped_agenda_reports_itself_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory)
            store = reactor.store
            seed(store, root=directory)
            self._finish_every_round(store)
            # A round handed out moments ago is the healthy rest between rounds.
            reactor._alert_if_roadmap_agenda_starved()
            reactor._alert_if_roadmap_agenda_starved()
            self.assertEqual(self._alerts(store), 0)
            # Three days with nothing handed out is the measured stall.
            self._age_agenda(store, hours=72.0)
            reactor._alert_if_roadmap_agenda_starved()
            reactor._alert_if_roadmap_agenda_starved()
            self.assertEqual(self._alerts(store), 1, "one alert per window, not one per cycle")
            reactor.close()

    def test_a_parked_agenda_does_not_nag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory)
            store = reactor.store
            seed(store, root=directory)
            self._finish_every_round(store)
            self._age_agenda(store, hours=72.0)
            store.connection.execute("UPDATE goals SET status='completed' WHERE title=?", (ROADMAP_GOAL_TITLE,))
            store.connection.commit()
            reactor._alert_if_roadmap_agenda_starved()
            self.assertEqual(self._alerts(store), 0, "an agenda whose goal is closed is not starving")
            reactor.close()


class OutwardLookSubjectTests(unittest.TestCase):
    def test_the_outward_look_asks_about_the_market(self) -> None:
        """The valve's subject is the market and lives outside the repository.

        Measured before the first repointing: 0% of her tokens went outward and 97% inward,
        while this family fired 13-20 times a day -- it was reading arXiv about her own
        planner, memory and supervisor. The owner then moved the organism's subject to the
        Moscow Exchange, and the valve's question moved with it.
        """
        for text in (EXTERNAL_SEEK_TASK_TITLE, EXTERNAL_SEEK_EXPECTED_FACT):
            self.assertNotIn("SkyNet bottleneck", text)
            self.assertNotIn("improvement hypothesis", text)
        self.assertIn("MOEX market", EXTERNAL_SEEK_TASK_TITLE)
        self.assertIn("private investor", EXTERNAL_SEEK_TASK_TITLE)
        self.assertIn("outside this repository", EXTERNAL_SEEK_EXPECTED_FACT)
        # The market is the subject and the harness is explicitly excluded -- the old
        # acceptance asked for "a bounded, testable improvement hypothesis", so every
        # round of the valve closed as a memory entry about her own machinery.
        self.assertIn("harness", EXTERNAL_SEEK_EXPECTED_FACT.lower())


if __name__ == "__main__":
    unittest.main()
