"""Split from the former monolithic CoreTests suite."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import UTC
from pathlib import Path
from typing import cast
from unittest.mock import Mock

from helpers import FakeProvider

from skynet.autonomous_planner import AutonomousPlanner
from skynet.models import ModelTurn
from skynet.planner import PortfolioPlanner, hypothesis_fingerprint, structural_fingerprint
from skynet.reactor import Reactor, ReactorConfig
from skynet.store import StateStore


def _iso(seconds: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(seconds, tz=UTC).isoformat()

class CoreTests(unittest.TestCase):
    def test_planner_bounding_preserves_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            planner = AutonomousPlanner(FakeProvider(), store, max_input_chars=1000, timeout_seconds=1000.0)
            payload = planner._bounded_payload(
                [{"goal_id": "g", "title": "goal"}],
                [{"task_id": "t", "title": "x"}],
                [{"content": "memory " * 100}],
                {"report": "previous " * 100},
            )
            encoded = json.dumps(payload, ensure_ascii=False)
            self.assertLessEqual(len(encoded), 1000)
            self.assertIsInstance(json.loads(encoded), dict)
            store.close()
    def test_autonomous_planner_persists_and_selects_valid_proposal(self) -> None:
        class PlanningProvider:
            def __init__(self) -> None:
                self.tools_seen = None

            def complete(self, messages, *, max_tokens, tools=()):
                self.tools_seen = tools
                goal_id = json.loads(messages[1]["content"])["goals"][0]["goal_id"]
                return ModelTurn(text=json.dumps({"proposals": [{
                    "goal_id": goal_id,
                    "title": "Validate planner recovery transition",
                    "problem": "Recovery transitions need an observable invariant",
                    "hypothesis": "A recovery transition can be verified by a focused state test",
                    "expected_new_fact": "The recovery transition persists the next lifecycle state",
                    "validation": "Run the focused recovery state test",
                    "scope": ["skynet/reactor.py", "tests/test_core.py"],
                    "kind": "validation",
                }]}), usage_tokens=3)

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("Improve recovery", priority=1.0)
            old_task = store.add_task("Old blocked recovery attempt", goal_id)
            store.connection.execute("UPDATE tasks SET status='blocked' WHERE task_id=?", (old_task,))
            store.connection.commit()
            provider = PlanningProvider()
            planner = AutonomousPlanner(provider, store, timeout_seconds=1000.0)
            goals, _ = store.active_work()
            created = planner.generate(generation=0, goals=goals, tasks=[], memories=[], previous={}, trigger="test")
            self.assertEqual(len(created), 1)
            self.assertEqual(provider.tools_seen, ())
            selected = PortfolioPlanner(store).select()
            self.assertIsNotNone(selected)
            self.assertEqual(cast(dict, selected)["task"]["task_id"], created[0]["task_id"])
            self.assertEqual(store.connection.execute("SELECT status FROM planner_attempts").fetchone()[0], "completed")
            store.close()
    def test_planner_tables_exist_and_provider_failure_is_durable(self) -> None:
        class FailingPlannerProvider:
            def complete(self, messages, *, max_tokens, tools=()):
                raise RuntimeError("planner unavailable")

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("Keep improving", priority=1.0)
            store.add_task("Blocked seed", goal_id)
            store.connection.execute("UPDATE tasks SET status='blocked'")
            store.connection.commit()
            tables = {row["name"] for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({"planner_attempts", "planner_proposals"} <= tables)
            planner = AutonomousPlanner(FailingPlannerProvider(), store, timeout_seconds=1000.0)
            goals, _ = store.active_work()
            self.assertEqual(planner.generate(generation=0, goals=goals, tasks=[], memories=[], previous={}, trigger="test"), [])
            self.assertEqual(store.connection.execute("SELECT status FROM planner_attempts").fetchone()[0], "provider_error")
            store.close()
    def test_planner_fallback_is_recreated_after_it_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            goal_id = store.add_goal("continuous improvement", priority=1.0)
            blocked = store.add_task("exhausted task", goal_id)
            store.connection.execute("UPDATE tasks SET status='blocked' WHERE task_id=?", (blocked,))
            store.connection.commit()
            store.close()

            reactor = Reactor(Mock(), {}, ReactorConfig(state_path=path))
            goals, _ = reactor.store.active_work()
            first_id = reactor._create_planner_fallback(goals, generation=1)
            self.assertIsNotNone(first_id)
            # A pending fallback is reused, never duplicated.
            self.assertEqual(reactor._create_planner_fallback(goals, generation=2), first_id)
            # Once it finishes, an empty portfolio must yield a fresh fallback
            # instead of parking the organism in sleep.
            reactor.store.connection.execute("UPDATE tasks SET status='completed' WHERE task_id=?", (first_id,))
            reactor.store.connection.commit()
            second_id = reactor._create_planner_fallback(goals, generation=3)
            self.assertIsNotNone(second_id)
            self.assertNotEqual(second_id, first_id)
            pending = reactor.store.connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE goal_id=? AND title LIKE 'Diagnose one concrete SkyNet bottleneck%' AND status='pending'",
                (goal_id,),
            ).fetchone()[0]
            self.assertEqual(pending, 1)
            reactor.close()
    def test_planner_limits_candidates_and_records_novelty_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            for index in range(6):
                goal_id = store.add_goal(f"portfolio {index}", priority=4 - index * 0.1)
                store.add_task(
                    f"task {index}",
                    goal_id,
                    expected_new_fact=f"fact {index}",
                    hypothesis_fingerprint=f"fp-{index}",
                    structural_fingerprint=f"sf-{index}",
                    area=f"area-{index}",
                )
            selected = PortfolioPlanner(store).select()
            self.assertEqual(cast(dict, selected)["kind"], "task")
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM planner_decisions").fetchone()[0], 1)
            decision = store.connection.execute("SELECT candidates FROM planner_decisions").fetchone()[0]
            self.assertEqual(len(json.loads(decision)), 4)
            store.close()
    def test_planner_deduplicates_exact_and_structural_hypotheses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("portfolio", priority=4)
            second_goal_id = store.add_goal("portfolio second", priority=3)
            third_goal_id = store.add_goal("portfolio third", priority=2)
            # Distinct areas: a workstream is an area of work, not a goal (a
            # goal is created only at genesis, so a goal-keyed portfolio was
            # unreachable by construction).
            store.add_task("first wording", goal_id, expected_new_fact="fact", hypothesis_fingerprint="same", structural_fingerprint="shape", area="area-a")
            store.add_task("second wording", second_goal_id, expected_new_fact="fact", hypothesis_fingerprint="same-2", structural_fingerprint="shape", area="area-b")
            store.add_task("third wording", third_goal_id, expected_new_fact="fact", hypothesis_fingerprint="different", structural_fingerprint="different-shape", area="area-c")
            decision = store.connection.execute("SELECT candidates FROM planner_decisions").fetchone()
            self.assertIsNone(decision)
            selected = PortfolioPlanner(store).select()
            self.assertEqual(cast(dict, selected)["task"]["title"], "first wording")
            candidates = json.loads(store.connection.execute("SELECT candidates FROM planner_decisions ORDER BY rowid DESC LIMIT 1").fetchone()[0])
            self.assertEqual([item["task_id"] for item in candidates], [store.connection.execute("SELECT task_id FROM tasks WHERE title='first wording'").fetchone()[0], store.connection.execute("SELECT task_id FROM tasks WHERE title='third wording'").fetchone()[0]])
            store.close()
    def test_task_update_closes_hypothesis_for_future_planning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("portfolio", priority=4)
            task_id = store.add_task("bounded task", goal_id, hypothesis_fingerprint="task-fp", structural_fingerprint="task-shape")
            store.apply_task_updates([{"task_id": task_id, "status": "completed"}], run_id="run-1")
            status = store.connection.execute("SELECT status FROM hypotheses WHERE fingerprint='task-fp'").fetchone()[0]
            self.assertEqual(status, "completed")
            self.assertIsNone(PortfolioPlanner(store).select())
            store.close()
    def test_autonomous_planner_provider_timeout_is_durable(self) -> None:
        release = threading.Event()

        class HangingPlanner:
            def complete(self, messages, *, max_tokens, tools=()):
                release.wait(5)
                return ModelTurn(text="{}")

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.add_goal("portfolio", priority=1.0)
            goals, _ = store.active_work()
            planner = AutonomousPlanner(HangingPlanner(), store, timeout_seconds=0.05)
            created = planner.generate(generation=0, goals=goals, tasks=[], memories=[], previous={}, trigger="timeout-test")
            self.assertEqual(created, [])
            status = store.connection.execute("SELECT status FROM planner_attempts").fetchone()[0]
            self.assertEqual(status, "provider_error")
            release.set()
            store.close()

class FingerprintPlannerProvider:
    """Returns one deterministic proposal so dedup liveness is observable."""

    def __init__(self, goal_id: str, *, problem: str, expected_new_fact: str, scope: list[str], kind: str = "engineering") -> None:
        self.goal_id = goal_id
        self.problem = problem
        self.expected_new_fact = expected_new_fact
        self.scope = scope
        self.kind = kind

    @property
    def fingerprints(self) -> tuple[str, str]:
        return (
            hypothesis_fingerprint(area=self.kind, problem=self.problem, expected_behavior=self.expected_new_fact, files=self.scope),
            structural_fingerprint(area=self.kind, target=self.problem, behavior_kind=self.kind),
        )

    def complete(self, messages, *, max_tokens, tools=()):
        return ModelTurn(text=json.dumps({"proposals": [{
            "goal_id": self.goal_id,
            "title": "bounded dedup probe",
            "problem": self.problem,
            "hypothesis": "the dedup gate only blocks live or terminal work",
            "expected_new_fact": self.expected_new_fact,
            "validation": "assert the planner accepts the proposal",
            "scope": list(self.scope),
            "kind": self.kind,
        }]}), usage_tokens=1)

class DedupLivenessTests(unittest.TestCase):
    def test_cancelled_task_fingerprint_is_proposable_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("dedup", priority=1.0)
            provider = FingerprintPlannerProvider(
                goal_id,
                problem="A cancelled task must not occupy its fingerprint forever",
                expected_new_fact="The fingerprint becomes proposable after the task is cancelled",
                scope=["skynet/autonomous_planner.py"],
            )
            hypothesis_fp, _ = provider.fingerprints
            goals, _ = store.active_work()
            first = AutonomousPlanner(provider, store, timeout_seconds=1000.0).generate(generation=0, goals=goals, tasks=[], memories=[], previous={}, trigger="first")
            self.assertEqual(len(first), 1)
            duplicate = AutonomousPlanner(provider, store, timeout_seconds=1000.0).generate(generation=1, goals=goals, tasks=[], memories=[], previous={}, trigger="second")
            self.assertEqual(duplicate, [])
            store.reset_dispatcher(reason="dedup-test-cancel")
            self.assertEqual(store.connection.execute("SELECT status FROM tasks WHERE task_id=?", (first[0]["task_id"],)).fetchone()[0], "cancelled")
            self.assertEqual(store.connection.execute("SELECT status FROM hypotheses WHERE fingerprint=?", (hypothesis_fp,)).fetchone()[0], "ready")
            third = AutonomousPlanner(provider, store, timeout_seconds=1000.0).generate(generation=2, goals=goals, tasks=[], memories=[], previous={}, trigger="third")
            self.assertEqual(len(third), 1)
            self.assertNotEqual(third[0]["task_id"], first[0]["task_id"])
            recorded = {row[0]: row[1] for row in store.connection.execute("SELECT status, COUNT(*) FROM planner_proposals WHERE hypothesis_fingerprint=? GROUP BY status", (hypothesis_fp,))}
            self.assertEqual(recorded, {"accepted": 2, "deduplicated": 1})
            store.close()

    def test_pending_task_fingerprint_is_still_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("dedup", priority=1.0)
            provider = FingerprintPlannerProvider(
                goal_id,
                problem="Pending work must stay deduplicated",
                expected_new_fact="A pending task still blocks an identical proposal",
                scope=["skynet/autonomous_planner.py"],
            )
            hypothesis_fp, structural_fp = provider.fingerprints
            store.add_task("live task", goal_id, expected_new_fact="A pending task still blocks an identical proposal", hypothesis_fingerprint=hypothesis_fp, structural_fingerprint=structural_fp)
            goals, _ = store.active_work()
            created = AutonomousPlanner(provider, store, timeout_seconds=1000.0).generate(generation=0, goals=goals, tasks=[], memories=[], previous={}, trigger="pending-guard")
            self.assertEqual(created, [])
            self.assertEqual(store.connection.execute("SELECT status FROM planner_proposals WHERE hypothesis_fingerprint=?", (hypothesis_fp,)).fetchone()[0], "deduplicated")
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 1)
            store.close()

    def test_completed_task_fingerprint_is_still_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("dedup", priority=1.0)
            provider = FingerprintPlannerProvider(
                goal_id,
                problem="Completed work must stay deduplicated",
                expected_new_fact="A completed task still blocks an identical proposal",
                scope=["skynet/autonomous_planner.py"],
            )
            hypothesis_fp, structural_fp = provider.fingerprints
            task_id = store.add_task("finished task", goal_id, expected_new_fact="A completed task still blocks an identical proposal", hypothesis_fingerprint=hypothesis_fp, structural_fingerprint=structural_fp)
            store.apply_task_updates([{"task_id": task_id, "status": "completed"}], run_id="run-1")
            self.assertEqual(store.connection.execute("SELECT status FROM hypotheses WHERE fingerprint=?", (hypothesis_fp,)).fetchone()[0], "completed")
            goals, _ = store.active_work()
            created = AutonomousPlanner(provider, store, timeout_seconds=1000.0).generate(generation=0, goals=goals, tasks=[], memories=[], previous={}, trigger="completed-guard")
            self.assertEqual(created, [])
            self.assertEqual(store.connection.execute("SELECT status FROM planner_proposals WHERE hypothesis_fingerprint=?", (hypothesis_fp,)).fetchone()[0], "deduplicated")
            store.close()

    def test_pending_inbox_is_a_notification_not_a_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            store.add_goal("continuous improvement", priority=1.0)
            store.add_inbox_event("evt-1", "user_message", {"text": "operator note"})
            store.close()

            reactor = Reactor(Mock(), {}, ReactorConfig(state_path=path))
            notifications = reactor._inbox_notifications(reactor.store.pending_inbox())
            self.assertEqual(len(notifications), 1)
            self.assertEqual(notifications[0]["kind"], "inbox_notification")
            self.assertEqual(notifications[0]["event_id"], "evt-1")
            self.assertEqual(notifications[0]["text"], "operator note")
            # A notification creates no work and consumes nothing.
            self.assertEqual(reactor.store.connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 0)
            self.assertEqual([event["event_id"] for event in reactor.store.pending_inbox()], ["evt-1"])
            reactor.close()

    def test_planner_penalizes_repeated_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("portfolio", priority=1.0)
            stale = store.add_task("stale", goal_id, expected_new_fact="fact-a", hypothesis_fingerprint="fp-stale", structural_fingerprint="sf-stale")
            fresh = store.add_task("fresh", goal_id, expected_new_fact="fact-b", hypothesis_fingerprint="fp-fresh", structural_fingerprint="sf-fresh")
            store.connection.execute("UPDATE tasks SET attempts=3 WHERE task_id=?", (stale,))
            store.connection.commit()
            selected = PortfolioPlanner(store).select()
            self.assertIsNotNone(selected)
            self.assertEqual(cast(dict, selected)["task"]["task_id"], fresh)
            store.close()

class GoalProposalProvider:
    """Returns one bounded goal proposal so the harness-owned goal path is testable."""

    def __init__(self, *, title: str, problem: str = "a durable gap", expected_behavior: str = "the gap is closed", priority: float = 0.7) -> None:
        self.title = title
        self.problem = problem
        self.expected_behavior = expected_behavior
        self.priority = priority

    def complete(self, messages, *, max_tokens, tools=()):
        return ModelTurn(text=json.dumps({
            "proposals": [],
            "goal_proposals": [{
                "title": self.title,
                "problem": self.problem,
                "expected_behavior": self.expected_behavior,
                "validation": "assert the new goal is active and bounded",
                "priority": self.priority,
            }],
        }), usage_tokens=1)

class InvalidGoalProposalProvider:
    def complete(self, messages, *, max_tokens, tools=()):
        return ModelTurn(text=json.dumps({
            "proposals": [],
            "goal_proposals": [{"title": "", "problem": "x", "expected_behavior": "y", "validation": "z"}],
        }), usage_tokens=1)

class PortfolioAreaTests(unittest.TestCase):
    def _store(self, directory: str) -> StateStore:
        return StateStore(Path(directory) / "state.sqlite3")

    def test_workstream_is_area_not_goal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            goal_a = store.add_goal("goal a", priority=1.0)
            goal_b = store.add_goal("goal b", priority=1.0)
            # Same goal, two areas: two genuinely different workstreams.
            store.add_task("engineering work", goal_a, expected_new_fact="fact", hypothesis_fingerprint="fp-a1", structural_fingerprint="sf-a1", area="engineering")
            store.add_task("research work", goal_a, expected_new_fact="fact", hypothesis_fingerprint="fp-a2", structural_fingerprint="sf-a2", area="research")
            # Different goals, same area: one workstream.
            store.add_task("more engineering", goal_b, expected_new_fact="fact", hypothesis_fingerprint="fp-b1", structural_fingerprint="sf-b1", area="engineering")
            ranked = PortfolioPlanner(store).rank()
            self.assertEqual(sorted(item.workstream_id for item in ranked), ["engineering", "research"])
            self.assertEqual(len(ranked), 2)
            store.close()

    def test_terminal_hypothesis_suppresses_until_the_ttl_expires(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            goal_id = store.add_goal("portfolio", priority=4.0)
            store.add_task("bounded work", goal_id, expected_new_fact="fact", hypothesis_fingerprint="fp-ttl", structural_fingerprint="sf-ttl", area="engineering")
            self.assertEqual(len(PortfolioPlanner(store).rank()), 1)
            store.mark_hypothesis("fp-ttl", "completed", {"run_id": "run-1"})
            self.assertEqual(PortfolioPlanner(store).rank(), [])
            # Beyond the TTL the fingerprint is free again: otherwise the
            # hypothesis space is consumed monotonically until the planner
            # always finds nothing.
            store.connection.execute(
                "UPDATE hypotheses SET updated_at=? WHERE fingerprint='fp-ttl'",
                (_iso(0.0),),
            )
            self.assertEqual(len(PortfolioPlanner(store, hypothesis_ttl_days=30.0).rank()), 1)
            store.close()

    def test_area_success_rate_feeds_expected_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            goal_id = store.add_goal("portfolio", priority=4.0)
            done_id = store.add_task("already solved", goal_id, expected_new_fact="fact", hypothesis_fingerprint="fp-done", structural_fingerprint="sf-done", area="engineering")
            store.apply_task_updates([{"task_id": done_id, "status": "completed"}], run_id="run-1")
            store.mark_hypothesis("fp-done", "completed", {"run_id": "run-1"})
            store.add_task("next engineering step", goal_id, expected_new_fact="fact", hypothesis_fingerprint="fp-next", structural_fingerprint="sf-next", area="engineering")
            ranked = PortfolioPlanner(store).rank()
            self.assertEqual(len(ranked), 1)
            self.assertGreater(ranked[0].expected_value, 0.0)
            store.close()

    def test_epsilon_zero_always_takes_the_top_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            goal_id = store.add_goal("portfolio", priority=4.0)
            for index in range(3):
                store.add_task(f"task {index}", goal_id, expected_new_fact="fact", hypothesis_fingerprint=f"fp-{index}", structural_fingerprint=f"sf-{index}", area=f"area-{index}")
            selected = PortfolioPlanner(store, epsilon=0.0).select()
            top = PortfolioPlanner(store, epsilon=0.0).rank()[0]
            self.assertEqual(cast(dict, selected)["task"]["task_id"], top.task_id)
            store.close()

    def test_epsilon_greedy_explores_and_journals_the_draw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            goal_id = store.add_goal("portfolio", priority=4.0)
            for index in range(4):
                store.add_task(f"task {index}", goal_id, expected_new_fact="fact", hypothesis_fingerprint=f"fp-{index}", structural_fingerprint=f"sf-{index}", area=f"area-{index}")
            ranked = PortfolioPlanner(store).rank()
            selected = PortfolioPlanner(store, epsilon=1.0).select()
            self.assertNotEqual(cast(dict, selected)["task"]["task_id"], ranked[0].task_id)
            payload = json.loads(
                store.connection.execute("SELECT payload FROM event_log WHERE kind='planner_decision' ORDER BY sequence DESC LIMIT 1").fetchone()[0]
            )
            self.assertEqual(payload["reason"], "epsilon_greedy_explore")
            exploration = payload["exploration"]
            self.assertIsNotNone(exploration["seed"])
            self.assertGreaterEqual(exploration["chosen_rank"], 1)
            store.close()

    def test_rng_is_deterministic_and_durable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            first = store.next_random()
            seed = store.rng_seed()
            self.assertIsNotNone(seed)
            store.close()
            reopened = StateStore(path)
            self.assertEqual(reopened.rng_seed(), seed)
            # The draw counter advanced, so the sequence continues instead of
            # repeating the same decision forever.
            self.assertNotEqual(reopened.next_random(), first)
            self.assertTrue(0.0 <= first < 1.0)
            reopened.close()

    def test_goal_proposal_creates_a_bounded_goal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            store.add_goal("genesis goal", priority=1.0)
            goals, _ = store.active_work()
            provider = GoalProposalProvider(title="Own the provider health surface", problem="lockouts are invisible", expected_behavior="a lockout is escalated")
            created = AutonomousPlanner(provider, store, timeout_seconds=1000.0).generate(generation=0, goals=goals, tasks=[], memories=[], previous={}, trigger="goal")
            self.assertEqual(created, [])
            row = store.connection.execute("SELECT title, status, priority, constraints FROM goals WHERE title=?", ("Own the provider health surface",)).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual((row["status"], row["priority"]), ("active", 0.7))
            self.assertEqual(json.loads(row["constraints"])["source"], "autonomous_planner")
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='goal_proposal_accepted'").fetchone()[0],
                1,
            )
            store.close()

    def test_goal_proposal_rejected_at_the_active_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            for index in range(2):
                store.add_goal(f"goal {index}", priority=1.0)
            goals, _ = store.active_work()
            provider = GoalProposalProvider(title="a goal that must be refused")
            AutonomousPlanner(provider, store, timeout_seconds=1000.0, max_active_goals=2).generate(generation=0, goals=goals, tasks=[], memories=[], previous={}, trigger="goal")
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM goals WHERE title='a goal that must be refused'").fetchone()[0], 0)
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='goal_proposal_rejected'").fetchone()[0],
                1,
            )
            store.close()

    def test_goal_proposal_duplicate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            store.add_goal("Existing goal title", priority=1.0)
            goals, _ = store.active_work()
            provider = GoalProposalProvider(title="Existing goal title")
            AutonomousPlanner(provider, store, timeout_seconds=1000.0).generate(generation=0, goals=goals, tasks=[], memories=[], previous={}, trigger="goal")
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0], 1)
            store.close()

    def test_schema_invalid_goal_proposal_never_reaches_goal_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            store.add_goal("genesis goal", priority=1.0)
            goals, _ = store.active_work()
            AutonomousPlanner(InvalidGoalProposalProvider(), store, timeout_seconds=1000.0).generate(generation=0, goals=goals, tasks=[], memories=[], previous={}, trigger="goal")
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0], 1)
            status = store.connection.execute("SELECT status FROM planner_attempts ORDER BY started_at DESC LIMIT 1").fetchone()[0]
            self.assertEqual(status, "invalid_response")
            store.close()

    def test_validate_goal_rejects_malformed_payloads(self) -> None:
        validate = AutonomousPlanner._validate_goal
        self.assertIsNone(validate(None))
        self.assertIsNone(validate({"title": "", "problem": "p", "expected_behavior": "e", "validation": "v"}))
        self.assertIsNone(validate({"title": "t", "problem": "", "expected_behavior": "e", "validation": "v"}))
        self.assertIsNone(validate({"title": "t", "problem": "p", "expected_behavior": "e"}))
        self.assertIsNone(validate({"title": "t" * 301, "problem": "p", "expected_behavior": "e", "validation": "v"}))
        accepted = validate({"title": "  bounded goal  ", "problem": "p", "expected_behavior": "e", "validation": "v"})
        assert accepted is not None
        self.assertEqual(accepted["title"], "bounded goal")
        self.assertEqual(accepted["priority"], 0.5)
