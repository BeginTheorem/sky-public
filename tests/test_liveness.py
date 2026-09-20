"""Liveness accounting: real attempts, model-side give-up and provider neutrality."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import UTC
from pathlib import Path
from typing import cast

from helpers import FixtureTool

from skynet.dialogue import AcknowledgeInboxTool
from skynet.models import ModelTurn, RunStatus, ToolCall
from skynet.planner import PortfolioPlanner
from skynet.reactor import Reactor, ReactorConfig
from skynet.time import parse_timestamp, utc_datetime_now


def _iso(seconds: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(seconds, tz=UTC).isoformat()

FINISH_JSON = json.dumps({
    "status": "COMPLETED",
    "summary": "finished the bounded episode",
    "evidence": ["fixture tool result"],
    "actions": [],
    "changes": [],
    "tests": [],
    "blocker": "",
    "next_hypothesis": "",
})
MEMORY_JSON = json.dumps({
    "memory_candidates": [],
    "next_plan": {},
    "initial_prompt": "",
    "goal_updates": [],
    "task_updates": [],
    "evaluation": {},
})

def _is_memory_request(messages) -> bool:
    return "memory_candidates" in messages[1]["content"]

def _classify(messages) -> str:
    if any("Finish phase" in str(message.get("content", "")) for message in messages):
        return "finish"
    if _is_memory_request(messages):
        return "memory"
    if "Generate at most 3 bounded proposals" in messages[1]["content"]:
        return "planner"
    return "react"

class InboxProvider:
    """Empty planner, verified ReAct completion; used to drive inbox accounting."""

    def complete(self, messages, *, max_tokens, tools=()):
        kind = _classify(messages)
        if kind == "memory":
            return ModelTurn(text=MEMORY_JSON, usage_tokens=1)
        if kind == "planner":
            return ModelTurn(text=json.dumps({"proposals": []}), usage_tokens=1)
        if kind == "finish":
            return ModelTurn(text=FINISH_JSON, usage_tokens=1)
        if messages[-1].get("role") == "tool":
            return ModelTurn(text=FINISH_JSON, usage_tokens=1)
        return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], usage_tokens=1)

class GoalClosingProvider(InboxProvider):
    """Completes the selected goal through the Memory Loop on the first run.

    Lets a pending operator message be observed after the last goal closes.
    """

    def __init__(self) -> None:
        self.goal_id: str | None = None

    def complete(self, messages, *, max_tokens, tools=()):
        if _is_memory_request(messages):
            goal_updates = (
                [{"goal_id": self.goal_id, "status": "completed", "outcome": "the memory loop closed the only goal"}]
                if self.goal_id
                else []
            )
            return ModelTurn(text=json.dumps({**json.loads(MEMORY_JSON), "goal_updates": goal_updates}), usage_tokens=1)
        return super().complete(messages, max_tokens=max_tokens, tools=tools)

class ProseProvider:
    """Always answers the ReAct phase with prose, so the Finish Report is invalid."""

    def complete(self, messages, *, max_tokens, tools=()):
        if _is_memory_request(messages):
            return ModelTurn(text=MEMORY_JSON, usage_tokens=1)
        return ModelTurn(text="I inspected the state but did not produce a report.", usage_tokens=1)

class DownProvider:
    """Every provider call fails, as during a full provider outage."""

    def complete(self, messages, *, max_tokens, tools=()):
        raise RuntimeError("all providers failed")

class RecoveringProvider:
    """Fails once, then produces a verified COMPLETED run."""

    def __init__(self) -> None:
        self.react_calls = 0

    def complete(self, messages, *, max_tokens, tools=()):
        if _is_memory_request(messages):
            return ModelTurn(text=MEMORY_JSON, usage_tokens=1)
        self.react_calls += 1
        if self.react_calls == 1:
            return ModelTurn(text="not ready yet", usage_tokens=1)
        if self.react_calls == 2:
            return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], usage_tokens=1)
        return ModelTurn(text=FINISH_JSON, usage_tokens=1)

class RetryableBlockedProvider:
    """Always reports a retryable environment blocker, so the task stays pending."""

    def complete(self, messages, *, max_tokens, tools=()):
        if _is_memory_request(messages):
            return ModelTurn(text=MEMORY_JSON, usage_tokens=1)
        return ModelTurn(
            text=json.dumps({
                "status": "BLOCKED",
                "summary": "the worktree is dirty and the proposal gate refuses",
                "evidence": [],
                "actions": [],
                "changes": [],
                "tests": [],
                "blocker": "environment blocker: dirty worktree",
                "next_hypothesis": "",
            }),
            usage_tokens=1,
        )

class LivenessTests(unittest.TestCase):
    def _reactor(self, directory: str, provider, *, tools=None, **kwargs) -> Reactor:
        root = Path(directory)
        config = ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root, **kwargs)
        return Reactor(provider, tools or {}, config)

    def test_attempts_increment_on_each_selected_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            task_id = reactor.store.add_task("bounded diagnostic", goal_id)
            self.assertEqual(reactor.tick("test"), RunStatus.NEEDS_RECOVERY)
            self.assertEqual(self._task_row(reactor, task_id)["attempts"], 1)
            self.assertEqual(reactor.tick("test"), RunStatus.NEEDS_RECOVERY)
            self.assertEqual(self._task_row(reactor, task_id)["attempts"], 2)
            reactor.close()

    def test_giveup_after_three_model_failures_retires_the_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            task_id = reactor.store.add_task("bounded diagnostic", goal_id)
            for _ in range(3):
                self.assertEqual(reactor.tick("test"), RunStatus.NEEDS_RECOVERY)
            row = self._task_row(reactor, task_id)
            self.assertEqual((row["status"], row["consecutive_model_failures"]), ("blocked", 3))
            hypothesis = reactor.store.connection.execute(
                "SELECT h.status FROM hypotheses h JOIN tasks t ON t.hypothesis_fingerprint=h.fingerprint WHERE t.task_id=?",
                (task_id,),
            ).fetchone()
            self.assertEqual(hypothesis["status"], "exhausted")
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='task_gave_up'").fetchone()[0],
                1,
            )
            self.assertIsNone(PortfolioPlanner(reactor.store).select())
            reactor.close()

    def test_only_a_fatal_provider_outage_exempts_the_giveup_budget(self) -> None:
        """A budget failure after provider retries is still a model-side failure.

        A run that only retried a provider along the way must not be treated as
        an outage; otherwise the give-up budget never moves and the task keeps
        winning the ranking.
        """
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider())
            for run_id, failure, expected in (
                ("run-budget", "time budget", False),
                ("run-output", "output budget", False),
                ("run-fatal", "all providers failed: ollama[x]: exhausted", True),
            ):
                reactor.store.append_event("provider_failure", {"error": "openrouter timed out"}, run_id)
                reactor.store.append_event(
                    "run_result_committed", {"status": "needs_recovery", "failure": failure}, run_id
                )
                self.assertEqual(reactor._run_had_provider_failure(run_id), expected, run_id)
            # A run with no recorded result at all is not exempt either.
            self.assertFalse(reactor._run_had_provider_failure("run-missing"))
            reactor.close()

    def test_provider_outage_does_not_consume_the_giveup_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, DownProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            task_id = reactor.store.add_task("bounded diagnostic", goal_id)
            for _ in range(4):
                self.assertEqual(reactor.tick("test"), RunStatus.NEEDS_RECOVERY)
            row = self._task_row(reactor, task_id)
            self.assertEqual(row["status"], "pending")
            self.assertEqual(row["consecutive_model_failures"], 0)
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='task_gave_up'").fetchone()[0],
                0,
            )
            reactor.close()

    def test_completed_run_resets_the_model_failure_streak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = RecoveringProvider()
            reactor = self._reactor(directory, provider, tools={"fixture_tool": FixtureTool()})
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            task_id = reactor.store.add_task("bounded diagnostic", goal_id)
            self.assertEqual(reactor.tick("test"), RunStatus.NEEDS_RECOVERY)
            self.assertEqual(self._task_row(reactor, task_id)["consecutive_model_failures"], 1)
            self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
            row = self._task_row(reactor, task_id)
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["consecutive_model_failures"], 0)
            reactor.close()

    def test_unrelated_run_does_not_consume_inbox_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, InboxProvider(), tools={"fixture_tool": FixtureTool()})
            goal_id = reactor.store.add_goal("inbox", priority=1.0)
            reactor.store.add_task("regular planner work", goal_id, expected_new_fact="a verified fact")
            reactor.store.add_inbox_event("evt-1", "user_message", {"text": "operator note"})
            self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
            self.assertEqual(
                [event["event_id"] for event in reactor.store.pending_inbox()],
                ["evt-1"],
            )
            reactor.close()

    def test_inbox_is_a_notification_until_acknowledged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, InboxProvider(), tools={"fixture_tool": FixtureTool()})
            reactor.store.add_goal("inbox", priority=1.0)
            reactor.store.add_inbox_event("evt-1", "user_message", {"text": "operator note"})
            # A tick only notifies: it neither consumes the message nor creates a task.
            self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
            self.assertEqual([event["event_id"] for event in reactor.store.pending_inbox()], ["evt-1"])
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='inbox_task_created'").fetchone()[0],
                0,
            )
            # The organism closes it itself.
            tool = AcknowledgeInboxTool(reactor.store)
            result = tool.execute({"event_id": "evt-1", "decision": "ignored"}, idempotency_key="k")
            self.assertEqual(result["state"], "closed")
            self.assertEqual(reactor.store.pending_inbox(), [])
            reactor.close()

    def test_roadmap_seed_refills_an_empty_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, InboxProvider())
            reactor.store.connection.execute("DELETE FROM tasks")
            reactor.store.connection.execute("DELETE FROM goals")
            reactor.store.connection.commit()
            self.assertTrue(reactor._ensure_roadmap_seed())
            pending = reactor.store.connection.execute("SELECT COUNT(*) FROM tasks WHERE status='pending'").fetchone()[0]
            self.assertGreater(pending, 0)
            reactor.close()

    def test_operator_message_is_not_auto_consumed_when_the_last_goal_closes(self) -> None:
        """Closing the last goal must not swallow a pending operator message.

        The message is a notification the organism decides about; it stays
        pending across the goal's closure until acknowledge_inbox is called.
        """
        with tempfile.TemporaryDirectory() as directory:
            provider = GoalClosingProvider()
            reactor = self._reactor(directory, provider, tools={"fixture_tool": FixtureTool()})
            goal_id = reactor.store.add_goal("the last goal", priority=1.0)
            provider.goal_id = goal_id
            reactor.store.add_task("bounded diagnostic", goal_id)
            reactor.store.add_inbox_event("evt-operator", "user_message", {"text": "operator note"})
            self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
            self.assertEqual(reactor.store.connection.execute("SELECT status FROM goals").fetchone()[0], "completed")
            self.assertEqual([event["event_id"] for event in reactor.store.pending_inbox()], ["evt-operator"])
            reactor.close()

    def test_no_active_goal_event_is_deduplicated_by_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider())
            # A blocked goal makes the store non-empty (no genesis) while
            # leaving no active goal to plan against.
            goal_id = reactor.store.add_goal("already spent", priority=1.0)
            reactor.store.connection.execute("UPDATE goals SET status='blocked' WHERE goal_id=?", (goal_id,))
            reactor.store.connection.commit()
            reactor.tick("test")
            reactor.tick("test")
            reactor.tick("test")
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='no_active_goal'").fetchone()[0],
                1,
            )
            reactor.close()

    def test_provider_request_event_records_the_tool_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider(), tools={"fixture_tool": FixtureTool()})
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic", goal_id)
            reactor.tick("test")
            row = reactor.store.connection.execute(
                "SELECT payload FROM event_log WHERE kind='provider_request' ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            self.assertIsNotNone(row)
            payload = json.loads(row[0])
            # fixture_tool plus the always-injected ask_user/send_message/memory/
            # acknowledge_inbox tools.
            self.assertEqual(payload["tool_count"], 5)
            self.assertIsInstance(payload["request_chars"], int)
            self.assertTrue(payload["model"])
            reactor.close()

    def test_restart_failure_escalates_at_the_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider(), restart_failure_limit=2)

            def failing() -> None:
                raise RuntimeError("systemctl unavailable")

            reactor.set_self_improvement_restart(failing)
            reactor._request_self_improvement_restart({"proposal_id": "proposal-x", "commit": "abc"})
            reactor._request_self_improvement_restart({"proposal_id": "proposal-x", "commit": "abc"})
            self.assertEqual(reactor.store.count_restart_failures("proposal-x"), 2)
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='restart_escalated'").fetchone()[0],
                1,
            )
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM outbox WHERE kind='restart_escalation'").fetchone()[0],
                1,
            )
            reactor.close()

    def test_retryable_blocked_task_is_retired_after_the_cap(self) -> None:
        """A retryable environment blocker is retired after the cap.

        The task must not return to pending forever when it is the only
        candidate; it is retired like any other repeated failure so replacement
        planning can run.
        """
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, RetryableBlockedProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            task_id = reactor.store.add_task("blocked by the environment", goal_id)
            for _ in range(3):
                self.assertEqual(reactor.tick("test"), RunStatus.BLOCKED)
            row = self._task_row(reactor, task_id)
            self.assertEqual((row["status"], row["consecutive_model_failures"]), ("blocked", 3))
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='task_gave_up'").fetchone()[0],
                1,
            )
            reactor.close()

    def test_retryable_blocked_below_the_cap_stays_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, RetryableBlockedProvider(), task_giveup_failures=5)
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            task_id = reactor.store.add_task("blocked by the environment", goal_id)
            self.assertEqual(reactor.tick("test"), RunStatus.BLOCKED)
            row = self._task_row(reactor, task_id)
            self.assertEqual((row["status"], row["consecutive_model_failures"]), ("pending", 1))
            reactor.close()

    def test_provider_lockout_escalates_once_and_backs_off(self) -> None:
        """Three consecutive all-provider failures escalate and stop the churn."""
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, DownProvider(), provider_lockout_seconds=900.0)
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic", goal_id)
            for _ in range(3):
                self.assertEqual(reactor.tick("test"), RunStatus.NEEDS_RECOVERY)
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='provider_lockout'").fetchone()[0],
                1,
            )
            alerts = reactor.store.pending_alerts()
            self.assertEqual([alert["kind"] for alert in alerts], ["provider_lockout"])
            self.assertEqual(alerts[0]["severity"], "critical")
            wake = reactor.store.state().next_wake_at
            remaining = (parse_timestamp(cast(str, wake)) - utc_datetime_now()).total_seconds()
            self.assertGreater(remaining, 800.0)
            reactor.close()

    def test_provider_lockout_alert_is_deduplicated_across_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, DownProvider(), provider_lockout_seconds=0.0)
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic", goal_id)
            for _ in range(6):
                reactor.tick("test")
            self.assertEqual(reactor.store.connection.execute("SELECT COUNT(*) FROM alerts").fetchone()[0], 1)
            self.assertGreaterEqual(
                reactor.store.connection.execute("SELECT occurrences FROM alerts").fetchone()[0], 2
            )
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM outbox WHERE kind='alert'").fetchone()[0],
                1,
            )
            reactor.close()

    def test_consecutive_provider_failure_streak_resets_on_any_other_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider())
            rows = (
                ("needs_recovery", "all providers failed: ollama[x]: exhausted", _iso(1.0)),
                ("needs_recovery", "all providers failed: ollama[x]: exhausted", _iso(2.0)),
                ("completed", "", _iso(3.0)),
                ("needs_recovery", "all providers failed: ollama[x]: exhausted", _iso(4.0)),
            )
            for status, failure, created_at in rows:
                reactor.store.connection.execute(
                    "INSERT INTO run_results(run_id, status, report, steps, usage_tokens, failure, created_at) "
                    "VALUES (lower(hex(randomblob(4))), ?, '{}', 1, 1, ?, ?)",
                    (status, failure, created_at),
                )
            self.assertEqual(reactor._consecutive_provider_failures(), 1)
            reactor.close()

    def test_start_envelope_carries_runtime_facts_and_pinned_memories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic", goal_id)
            reactor.store.consolidate("run-1", [
                {"kind": "procedure", "content": "the working test runner is .venv/bin/python -m pytest -q", "confidence": 0.95},
            ])
            memory_id = reactor.store.connection.execute("SELECT memory_id FROM memories").fetchone()[0]
            reactor.store.set_memory_pinned(memory_id)
            reactor.tick("test")
            payload = json.loads(
                reactor.store.connection.execute(
                    "SELECT payload FROM event_log WHERE kind='run_started' ORDER BY sequence DESC LIMIT 1"
                ).fetchone()[0]
            )
            observations = {item["kind"]: item for item in payload["observations"]}
            self.assertIn("runtime_facts", observations)
            facts = observations["runtime_facts"]
            self.assertIn("test_command", facts)
            # ask_user, send_message_to_user, memory and acknowledge_inbox are
            # always injected, so the runtime surface is never empty even with no
            # operator tools.
            self.assertEqual(facts["tool_count"], 4)
            # The schema is injected so the model does not have to guess table
            # names.
            self.assertIn("event_log", facts["tables"])
            self.assertIn("memories", facts["tables"])
            self.assertTrue(facts["scratch_directory"].endswith("skynet-scratch"))
            self.assertIn("pinned_memories", observations)
            self.assertEqual(len(observations["pinned_memories"]["items"]), 1)
            reactor.close()

    def test_start_envelope_reports_a_dirty_worktree_and_the_reboot_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "tracked.py").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            (root / "tracked.py").write_text("two\n", encoding="utf-8")
            # The guard lives next to the state database (state_path's parent).
            (root / "reboot-guard.json").write_text(
                json.dumps({"proposal_id": "p1", "commit": "abc1234", "completed_at": _iso(0.0), "healthy_cycles": 3}),
                encoding="utf-8",
            )
            reactor = self._reactor(directory, ProseProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic", goal_id)
            reactor.tick("test")
            payload = json.loads(
                reactor.store.connection.execute(
                    "SELECT payload FROM event_log WHERE kind='run_started' ORDER BY sequence DESC LIMIT 1"
                ).fetchone()[0]
            )
            observations = {item["kind"]: item for item in payload["observations"]}
            self.assertTrue(observations["dirty_worktree"]["dirty"])
            self.assertIn("tracked.py", observations["dirty_worktree"]["files"])
            # The organism learns how its own last promotion ended.
            self.assertEqual(observations["reboot_outcome"]["outcome"], "accepted")
            self.assertEqual(observations["reboot_outcome"]["proposal_id"], "p1")
            reactor.close()

    def test_planner_memory_query_is_derived_from_the_situation(self) -> None:
        # The query must reflect the actual goal and the previous outcome.
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider())
            state = reactor.store.state()
            state.next_plan = {"previous_outcome": {"summary": "the gate rejected a patch"}, "next": "retry with a multi-line anchor"}
            query = reactor._planner_memory_query(state, [{"title": "Advance the SkyNet roadmap"}])
            self.assertIn("Advance the SkyNet roadmap", query)
            self.assertIn("the gate rejected a patch", query)
            self.assertIn("multi-line anchor", query)
            self.assertNotEqual(query, "autonomous planning next bounded work")
            reactor.close()

    def test_memory_query_is_built_from_the_task_envelope(self) -> None:
        # `selected_work` is task_work()'s envelope, so the title must be read
        # from the nested task, not the top-level work dict.
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider())
            work = {
                "kind": "task",
                "task": {"task_id": "t1", "title": "Harden the provider ladder", "expected_new_fact": "a cooldown is persisted"},
                "goal": {"goal_id": "g1", "title": "Advance the roadmap"},
            }
            query = reactor._memory_query(work, [], {})
            self.assertIn("Harden the provider ladder", query)
            self.assertIn("a cooldown is persisted", query)
            # The goal title is the fallback when a task carries no title.
            self.assertIn("Advance the roadmap", reactor._memory_query({"kind": "goal", "goal": {"title": "Advance the roadmap"}}, [], {}))
            reactor.close()

    def test_memory_retrieval_is_recorded_every_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic about the planner", goal_id)
            reactor.tick("test")
            row = reactor.store.connection.execute(
                "SELECT payload FROM event_log WHERE kind='memory_retrieval' ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            self.assertIsNotNone(row)
            payload = json.loads(row[0])
            self.assertIn("hits", payload)
            self.assertIn("query_terms", payload)
            self.assertIn("pinned", payload)
            reactor.close()

    @staticmethod
    def _task_row(reactor: Reactor, task_id: str):
        return reactor.store.connection.execute(
            "SELECT status, attempts, consecutive_model_failures FROM tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()

if __name__ == "__main__":
    unittest.main()
