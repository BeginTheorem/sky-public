"""The roadmap handoff that gives the organism something to work on."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from skynet import handoff
from skynet.handoff import (
    CANONICAL_MEMORIES,
    ROADMAP_AREA,
    ROADMAP_GOAL_PRIORITY,
    ROADMAP_GOAL_TITLE,
    ROADMAP_TASKS,
    seed,
)
from skynet.planner import PortfolioPlanner
from skynet.store import StateStore


class HandoffTests(unittest.TestCase):
    def test_seed_creates_the_roadmap_goal_tasks_and_searchable_memories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.add_goal("genesis", priority=1.0)
            summary = seed(store, root=directory)
            self.assertTrue(summary["goal_created"])
            self.assertEqual(summary["tasks_created"], len(ROADMAP_TASKS))
            goal = store.connection.execute(
                "SELECT title, status, priority FROM goals WHERE title=?", (ROADMAP_GOAL_TITLE,)
            ).fetchone()
            self.assertEqual((goal["status"], goal["priority"]), ("active", 1.0))
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM tasks WHERE area=?", (ROADMAP_AREA,)).fetchone()[0],
                len(ROADMAP_TASKS),
            )
            # Every task carries its acceptance criterion as the new fact, so a
            # report alone cannot close it.
            missing = store.connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE area=? AND (expected_new_fact IS NULL OR expected_new_fact='')",
                (ROADMAP_AREA,),
            ).fetchone()[0]
            self.assertEqual(missing, 0)
            # The canonical set is seeded as ordinary searchable rows, never
            # pinned: the one-shot pin injection stays free for the organism's
            # own pins and the model reaches these facts through memory.search.
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM memories WHERE kind='procedure'").fetchone()[0],
                len(CANONICAL_MEMORIES),
            )
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM memories WHERE pinned=1").fetchone()[0],
                0,
            )
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='roadmap_seeded'").fetchone()[0],
                1,
            )
            store.close()

    def test_seed_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            first = seed(store, root=directory)
            second = seed(store, root=directory)
            self.assertFalse(second["goal_created"])
            self.assertEqual(second["tasks_created"], 0)
            self.assertEqual(second["tasks_skipped"], len(ROADMAP_TASKS))
            self.assertEqual(second["goal_id"], first["goal_id"])
            self.assertEqual(second["memories_upserted"], 0)
            # A re-run converges the goal priority instead of leaving a stale one.
            store.connection.execute("UPDATE goals SET priority=4.0 WHERE title=?", (ROADMAP_GOAL_TITLE,))
            seed(store, root=directory)
            self.assertEqual(
                store.connection.execute("SELECT priority FROM goals WHERE title=?", (ROADMAP_GOAL_TITLE,)).fetchone()[0],
                1.0,
            )
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM goals WHERE title=?", (ROADMAP_GOAL_TITLE,)).fetchone()[0],
                1,
            )
            # Re-running adds nothing twice: the task count is stable.
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM tasks WHERE area=?", (ROADMAP_AREA,)).fetchone()[0],
                len(ROADMAP_TASKS),
            )
            store.close()

    def test_seed_rehomes_pending_items_after_the_last_goal_closes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            seed(store, root=directory)
            dead_goal = store.connection.execute(
                "SELECT goal_id FROM goals WHERE title=?", (ROADMAP_GOAL_TITLE,)
            ).fetchone()["goal_id"]
            survivor = store.connection.execute(
                "SELECT task_id FROM tasks WHERE area=? AND status='pending' ORDER BY title LIMIT 1",
                (ROADMAP_AREA,),
            ).fetchone()["task_id"]
            # Finish every curated item but one, then close the goal: the survivor
            # is now invisible to the scheduler (task_work needs an active goal),
            # present in the table but unselectable -- the live stall state.
            store.connection.execute(
                "UPDATE tasks SET status='completed' WHERE area=? AND task_id!=?",
                (ROADMAP_AREA, survivor),
            )
            store.connection.execute("UPDATE goals SET status='completed' WHERE goal_id=?", (dead_goal,))
            store.connection.commit()
            summary = seed(store, root=directory)
            self.assertTrue(summary["goal_created"])
            self.assertEqual(summary["tasks_rehomed"], 1)
            new_goal = store.connection.execute(
                "SELECT goal_id FROM goals WHERE title=? AND status='active'", (ROADMAP_GOAL_TITLE,)
            ).fetchone()
            self.assertIsNotNone(new_goal)
            assert new_goal is not None
            self.assertNotEqual(new_goal["goal_id"], dead_goal)
            self.assertEqual(
                store.connection.execute("SELECT goal_id FROM tasks WHERE task_id=?", (survivor,)).fetchone()["goal_id"],
                new_goal["goal_id"],
            )
            # The re-homed item is selectable again, so the organism is not idle.
            self.assertIsNotNone(PortfolioPlanner(store).select())
            store.close()

    def test_seed_never_points_the_organism_at_a_document(self) -> None:
        source = inspect.getsource(handoff)
        self.assertNotIn("Architecture", source)
        for text in (*CANONICAL_MEMORIES, *(fact for _, fact in ROADMAP_TASKS)):
            self.assertNotIn("Architecture", text)
            self.assertNotIn("section", text)
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            seed(store, root=directory)
            row = store.connection.execute(
                "SELECT constraints FROM goals WHERE title=?", (ROADMAP_GOAL_TITLE,)
            ).fetchone()
            self.assertIn("genuinely open item", row["constraints"])
            self.assertNotIn("Architecture", row["constraints"])
            memories = store.connection.execute("SELECT content FROM memories").fetchall()
            for memory in memories:
                self.assertNotIn("Architecture", memory["content"])
            store.close()

    def test_scaffolding_does_not_outrank_self_set_goals(self) -> None:
        self.assertEqual(ROADMAP_GOAL_PRIORITY, 1.0)
        self.assertNotEqual(ROADMAP_GOAL_PRIORITY, 4.0)

    def test_the_planner_picks_up_roadmap_work_after_the_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.add_goal("genesis", priority=1.0)
            self.assertIsNone(PortfolioPlanner(store).select())
            seed(store, root=directory)
            ranked = PortfolioPlanner(store).rank()
            self.assertEqual([item.workstream_id for item in ranked], [ROADMAP_AREA])
            selected = PortfolioPlanner(store).select()
            self.assertIsNotNone(selected)
            assert selected is not None
            self.assertEqual(selected["task"]["area"], ROADMAP_AREA)
            store.close()
