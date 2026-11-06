"""A stalled search is pushed toward never-tried descriptor cells.

RRSI reserves proposal capacity for unexplored components when progress stalls
inside the noise band (arXiv:2609.24972). SkyNet expresses the same nudge as a
directive to its single planner: under a stall the payload carries the explicit
untried cells and a dedicated instruction, the planner scores a materialized
task against its true archive cell, and the reactor records that it directed.

The directive is deliberately not a filter: the harness never fabricates a
proposal, so a reply that ignores it is still accepted. What changes is what the
planner is told and how scarcity is scored -- not who selects work.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

from helpers import FakeProvider

from skynet.autonomous_planner import AutonomousPlanner
from skynet.idea_archive import cell_key
from skynet.planner import PortfolioPlanner
from skynet.planner_contract import PLANNER_STALL_INSTRUCTION
from skynet.reactor import Reactor, ReactorConfig
from skynet.store import StateStore


def _archive(store: StateStore, subsystem: str, change_type: str, evidence: str) -> str:
    idea_id = store.archive_idea({
        "subsystem": subsystem,
        "change_type": change_type,
        "evidence_source": evidence,
        "cell_key": cell_key(subsystem, change_type, evidence),
        "title": f"{subsystem} {change_type} {evidence}",
        "quality": 1.0,
        "novelty": 1.0,
    })
    if idea_id is None:  # pragma: no cover - an empty cell always admits the first idea
        raise RuntimeError("an empty archive cell must admit the first idea")
    return idea_id


def _store(directory: str) -> StateStore:
    return StateStore(Path(directory) / "state.sqlite3")


class StoreCellMappingTests(unittest.TestCase):
    def test_a_materialized_idea_is_keyed_to_its_cell_by_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            try:
                idea_id = _archive(store, "memory", "new-tool", "paper")
                goal = store.add_goal("g", priority=1.0)
                task_id = store.materialize_idea(idea_id, goal)
                self.assertIsNotNone(task_id)
                assert task_id is not None
                self.assertEqual(store.idea_cells_by_task(), {task_id: "memory|new-tool|paper"})
            finally:
                store.close()


class CellForTests(unittest.TestCase):
    def test_a_materialized_task_is_scored_against_its_true_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            try:
                planner = PortfolioPlanner(store)
                row = {"task_id": "t1", "area": "memory"}
                self.assertEqual(planner._cell_for(row, {"t1": "memory|new-tool|paper"}), "memory|new-tool|paper")
            finally:
                store.close()

    def test_an_unmapped_task_falls_back_to_the_area_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            try:
                planner = PortfolioPlanner(store)
                self.assertEqual(planner._cell_for({"task_id": "t2", "area": "memory"}, {}), "memory|workflow|own-repo")
            finally:
                store.close()


class PlannerPayloadTests(unittest.TestCase):
    def _planner(self, store: StateStore) -> AutonomousPlanner:
        return AutonomousPlanner(FakeProvider(), store, max_input_chars=700_000, timeout_seconds=1000.0)

    def test_stall_adds_the_untried_cells_and_the_stall_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            try:
                payload = self._planner(store)._bounded_payload(
                    [], [], [], {}, exploration={"stalled": True, "seconds_since_progress": 7200.0}
                )
                exploration = payload["exploration"]
                self.assertTrue(exploration["stalled"])
                self.assertEqual(exploration["instruction"], PLANNER_STALL_INSTRUCTION)
                self.assertEqual(len(exploration["untried_cells"]), 24)
                self.assertIn("general|new-tool|own-repo", exploration["untried_cells"])
            finally:
                store.close()

    def test_no_exploration_block_without_a_stall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            try:
                payload = self._planner(store)._bounded_payload([], [], [], {})
                self.assertNotIn("exploration", payload)
            finally:
                store.close()

    def test_a_filled_cell_is_not_offered_as_untried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            try:
                planner = self._planner(store)
                _archive(store, "memory", "new-tool", "paper")
                empty = planner._empty_cells(200)
                self.assertNotIn("memory|new-tool|paper", empty)
                self.assertIn("memory|new-tool|web", empty)
            finally:
                store.close()


class ReactorDirectiveTests(unittest.TestCase):
    def _reactor(self, directory: str, *, stall_alert_seconds: float) -> Reactor:
        return Reactor(Mock(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3", stall_alert_seconds=stall_alert_seconds))

    @staticmethod
    def _completed_run(reactor: Reactor, when: datetime) -> None:
        stamp = when.isoformat().replace("+00:00", "Z")
        reactor.store.connection.execute(
            "INSERT INTO runs (run_id, attempt, status, started_at, finished_at, budget) VALUES ('r', 1, 'completed', ?, ?, '{}')",
            (stamp, stamp),
        )
        reactor.store.connection.commit()

    def test_the_directive_fires_only_when_the_stall_window_is_crossed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, stall_alert_seconds=3600.0)
            try:
                self.assertIsNone(reactor._planner_exploration())
                self._completed_run(reactor, datetime.now(UTC) - timedelta(hours=2))
                directive = reactor._planner_exploration()
                self.assertTrue(dict(directive or {}).get("stalled"))
            finally:
                reactor.store.close()

    def test_a_disabled_stall_window_never_directs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, stall_alert_seconds=0.0)
            try:
                self._completed_run(reactor, datetime.now(UTC) - timedelta(days=1))
                self.assertIsNone(reactor._planner_exploration())
            finally:
                reactor.store.close()


if __name__ == "__main__":
    unittest.main()
