"""Liveness accounting: real attempts, model-side give-up and provider neutrality."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from helpers import FixtureTool

from skynet.dialogue import AcknowledgeInboxTool
from skynet.metrics import format_report, memory_health
from skynet.models import AgentRunResult, ModelTurn, RunStatus, ToolCall
from skynet.planner import PortfolioPlanner
from skynet.providers.errors import ProviderError
from skynet.providers.fallback import is_chain_wide_cooldown_abort
from skynet.reactor import Reactor, ReactorConfig
from skynet.time import parse_timestamp, utc_datetime_now

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

    Used to reproduce the loss of the last active goal while an operator
    message is still pending.
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

        On the live server a task kept being retried because its runs had some
        provider retries along the way, so `_run_had_provider_failure` reported
        an outage and the give-up budget never moved: attempts grew to three
        while the task stayed pending and kept winning the ranking.
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

    def test_the_unreachable_chain_predicate_separates_the_two_live_classes(self) -> None:
        """The predicate is defined by the recorded strings, not by a guess.

        Read off the live ledger (174 committed runs, 2026-09-25): 5 runs died
        with the chain-wide cooldown text and 12 with a ladder that had struck at
        least one provider first. The two sets are disjoint, and the shared
        "all providers failed" substring is what made them indistinguishable.
        """
        zero_strike = [
            "all providers failed after 0 attempts: every provider is in cooldown (nemotron=21.0s)",
            "all providers failed after 0 attempts: every provider is in cooldown (openrouter=29.9s)",
            "all providers failed after 0 attempts: every provider is in cooldown (nemotron=17.2s)",
            "all providers failed after 0 attempts: every provider is in cooldown (openrouter=12.3s)",
            "all providers failed after 0 attempts: every provider is in cooldown (openrouter=23.8s)",
        ]
        not_this_class = [
            "all providers failed after 2 attempts: nemotron[server]: nemotron request failed: HTTP 500",
            "all providers failed after 1 attempts: openrouter[network]: openrouter SSE stream ended before [DONE] | blocked_until={'openrouter': 30.0}",
            "all providers failed after 0 attempts",  # the count alone is not the class
            "time budget",
            "",
        ]
        for failure in zero_strike:
            self.assertTrue(is_chain_wide_cooldown_abort(failure), failure)
        for failure in not_this_class:
            self.assertFalse(is_chain_wide_cooldown_abort(failure), failure)

    def test_a_zero_strike_cooldown_abort_is_recorded_per_run_and_task(self) -> None:
        """The class the shipped exemption cannot name must be countable.

        `_run_had_provider_failure` matches the shared "all providers failed"
        substring, so it reports every one of these episodes as an outage and
        never charges the model-side give-up budget. That exemption is right;
        what was missing is a run-scoped row that NAMES the unreachable-chain
        case, because the chain's own `fallback_all_cooling` event is written
        with run_id NULL and cannot be attributed to the task that paid for it.
        """
        class CooldownProvider:
            """Every call is refused with the chain's own declared horizon."""

            def complete(self, messages, *, max_tokens, tools=()):
                if _is_memory_request(messages):
                    return ModelTurn(text=MEMORY_JSON, usage_tokens=1)
                raise ProviderError(
                    "all providers failed after 0 attempts: every provider is in cooldown (nemotron=17.2s)",
                    category="unavailable",
                    retryable=True,
                    cooldown_seconds=17.2,
                )

        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, CooldownProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            task_id = reactor.store.add_task("bounded diagnostic", goal_id)
            self.assertEqual(reactor.tick("test"), RunStatus.NEEDS_RECOVERY)
            rows = reactor.store.connection.execute(
                "SELECT run_id, payload FROM event_log WHERE kind='provider_unreachable'"
            ).fetchall()
            self.assertEqual(len(rows), 1)
            payload = json.loads(rows[0]["payload"])
            self.assertEqual(payload["task_id"], task_id)
            self.assertTrue(payload["attempt_charged"])
            self.assertFalse(payload["model_failure_charged"])
            # The measured counter deltas: this class costs the task one attempt
            # (the run did start) and no model-side failure.
            row = self._task_row(reactor, task_id)
            self.assertEqual((row["attempts"], row["consecutive_model_failures"]), (1, 0))
            self.assertEqual(row["status"], "pending")
            reactor.close()

    def test_a_provider_strike_is_not_recorded_as_an_unreachable_chain(self) -> None:
        """A ladder that struck a provider failed differently; keep it distinct."""
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, DownProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic", goal_id)
            self.assertEqual(reactor.tick("test"), RunStatus.NEEDS_RECOVERY)
            self.assertEqual(
                reactor.store.connection.execute(
                    "SELECT COUNT(*) FROM event_log WHERE kind='provider_unreachable'"
                ).fetchone()[0],
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

    def test_owner_message_is_not_auto_consumed_when_the_last_goal_closes(self) -> None:
        """Closing the last goal must not swallow a pending owner message.

        The message is a notification the organism decides about; it stays
        pending across the goal's closure until acknowledge_inbox is called.
        """
        with tempfile.TemporaryDirectory() as directory:
            provider = GoalClosingProvider()
            reactor = self._reactor(directory, provider, tools={"fixture_tool": FixtureTool()})
            goal_id = reactor.store.add_goal("the last goal", priority=1.0)
            provider.goal_id = goal_id
            reactor.store.add_task("bounded diagnostic", goal_id)
            reactor.store.add_inbox_event("evt-owner", "user_message", {"text": "operator note"})
            self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
            self.assertEqual(reactor.store.connection.execute("SELECT status FROM goals").fetchone()[0], "completed")
            self.assertEqual([event["event_id"] for event in reactor.store.pending_inbox()], ["evt-owner"])
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
            # acknowledge_inbox/read_inbox/record_plan tools.
            self.assertEqual(payload["tool_count"], 7)
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
        """A retryable environment blocker used to be retried with no limit.

        The task returned to pending on every cycle and, when it was the only
        candidate, was selected forever. It must now be retired like any other
        repeated failure so replacement planning can run.
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
                ("needs_recovery", "all providers failed: ollama[x]: exhausted", "2026-09-18T00:00:01Z"),
                ("needs_recovery", "all providers failed: ollama[x]: exhausted", "2026-09-18T00:00:02Z"),
                ("completed", "", "2026-09-18T00:00:03Z"),
                ("needs_recovery", "all providers failed: ollama[x]: exhausted", "2026-09-18T00:00:04Z"),
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
            # ask_user, send_message_to_user, memory, acknowledge_inbox,
            # read_inbox and record_plan are always injected, so the runtime
            # surface is never empty even with no operator tools.
            self.assertEqual(facts["tool_count"], 6)
            # The model used to query tables that do not exist (`events` instead
            # of `event_log`); the schema is injected so it stops guessing.
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
                json.dumps({"proposal_id": "p1", "commit": "abc1234", "completed_at": "2026-09-18T00:00:00Z", "healthy_cycles": 3}),
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

    def test_start_envelope_reports_an_open_window_as_in_progress_and_live(self) -> None:
        """A promotion mid-window must be visible without re-checking git by hand.

        Measured on the live event log: 81 promotions carried
        restart_after_checkpoint, only 2 ever reached a terminal
        reboot_observation row, yet 80 of 108 run envelopes carried the
        "verify the promoted self-improvement after reboot" hint. The window
        was durable but unreadable: ``begin`` wrote no ``active`` key, so
        ``_reboot_outcome``'s ``in_progress`` branch could not fire and no
        envelope ever saw ``in_progress`` (18 of 18 read ``accepted``).
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "tracked.py").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            (root / "reboot-guard.json").write_text(
                json.dumps({
                    "proposal_id": "p1", "commit": head, "rollback_commit": head,
                    "healthy_cycles": 1, "active": True, "failed": False,
                    "started_at": "2026-09-18T00:00:00Z",
                }),
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
            outcome = observations["reboot_outcome"]
            self.assertEqual(outcome["outcome"], "in_progress")
            self.assertEqual(outcome["proposal_id"], "p1")
            self.assertEqual(outcome["commit"], head)
            self.assertEqual(outcome["head"], head)
            self.assertTrue(outcome["live"])
            self.assertNotIn("stale", outcome)
            reactor.close()

    def test_recovery_hint_yields_to_the_envelope_it_would_ask_about(self) -> None:
        """A resolved window must not also be handed to the next run as a hint.

        Measured on the live event log: 81 of 109 run_started
        envelopes carried "verify the promoted self-improvement after reboot",
        and each of those runs was exactly the run the reboot window covered --
        it paid a `git rev-parse HEAD` re-verification although the envelope
        already carried outcome=in_progress, live=true. The hint is now emitted
        only when the window cannot answer for itself.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "tracked.py").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            (root / "reboot-guard.json").write_text(
                json.dumps({
                    "proposal_id": "p1", "commit": head, "rollback_commit": head,
                    "healthy_cycles": 1, "active": True, "failed": False,
                    "started_at": "2026-09-18T00:00:00Z",
                }),
                encoding="utf-8",
            )
            reactor = self._reactor(directory, ProseProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic", goal_id)
            with patch.object(reactor.runner, "run", return_value=AgentRunResult(
                status=RunStatus.COMPLETED,
                report=json.dumps({"status": "COMPLETED", "summary": "promoted"}),
                steps=1,
                usage_tokens=1,
                control_action={"type": "restart_after_checkpoint", "proposal_id": "p1", "commit": head},
            )):
                reactor.tick("test")
            # The block is written into the durable plan the NEXT start envelope
            # serializes, so read it from the store, not from this run's envelope.
            block = reactor.store.state().next_plan["self_improvement_recovery"]
            self.assertTrue(block["resolved"])
            self.assertNotIn("next", block, "the envelope already answers the question, so no hint is emitted")
            self.assertNotIn("verify the promoted self-improvement after reboot", json.dumps(reactor.store.state().next_plan))
            # The observation and the decision to skip the hint travel together:
            # the same envelope that now carries the answer carries the outcome.
            payload = json.loads(
                reactor.store.connection.execute(
                    "SELECT payload FROM event_log WHERE kind='run_started' ORDER BY sequence DESC LIMIT 1"
                ).fetchone()[0]
            )
            outcome = {item["kind"]: item for item in payload["observations"]}["reboot_outcome"]
            self.assertEqual((outcome["outcome"], outcome["live"]), ("in_progress", True))
            reactor.close()

    def test_recovery_hint_survives_when_no_window_can_answer_it(self) -> None:
        """No guard row means nothing to read, so the re-verification stays."""
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic", goal_id)
            with patch.object(reactor.runner, "run", return_value=AgentRunResult(
                status=RunStatus.COMPLETED,
                report=json.dumps({"status": "COMPLETED", "summary": "promoted"}),
                steps=1,
                usage_tokens=1,
                control_action={"type": "restart_after_checkpoint", "proposal_id": "p1", "commit": "abc1234"},
            )):
                reactor.tick("test")
            plan = reactor.store.state().next_plan
            block = plan["self_improvement_recovery"]
            self.assertFalse(block["resolved"])
            self.assertEqual(block["next"], "verify the promoted self-improvement after reboot")
            self.assertEqual(block["checks"], ["service starts", "database opens", "provider is reachable", "health window completes"])
            payload = json.loads(
                reactor.store.connection.execute(
                    "SELECT payload FROM event_log WHERE kind='run_started' ORDER BY sequence DESC LIMIT 1"
                ).fetchone()[0]
            )
            self.assertNotIn("reboot_outcome", {item["kind"] for item in payload["observations"]})
            reactor.close()

    def test_resolved_recovery_block_carries_the_verdict_instead_of_asserting_it(self) -> None:
        """The plan the next run reads must carry the values it would re-derive.

        Measured: the block said ``resolved: true`` and claimed
        "outcome=in_progress, live=true (head == promoted commit)" as prose, but
        carried neither the observed outcome, nor head, nor how far the window
        had run -- so the next run could only take the claim on trust.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "tracked.py").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            (root / "reboot-guard.json").write_text(
                json.dumps({
                    "proposal_id": "p1", "commit": head, "rollback_commit": head,
                    "healthy_cycles": 1, "window_cycles": 3, "active": True,
                    "failed": False, "started_at": "2026-09-18T00:00:00Z",
                }),
                encoding="utf-8",
            )
            reactor = self._reactor(directory, ProseProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic", goal_id)
            with patch.object(reactor.runner, "run", return_value=AgentRunResult(
                status=RunStatus.COMPLETED,
                report=json.dumps({"status": "COMPLETED", "summary": "promoted"}),
                steps=1,
                usage_tokens=1,
                control_action={"type": "restart_after_checkpoint", "proposal_id": "p1", "commit": head},
            )):
                reactor.tick("test")
            block = reactor.store.state().next_plan["self_improvement_recovery"]
            print("resolved block:", json.dumps(block, sort_keys=True))
            self.assertTrue(block["resolved"])
            self.assertTrue(block["live"])
            self.assertEqual(block["outcome"], "in_progress")
            self.assertEqual(block["head"], head)
            self.assertEqual(block["commit"], head)
            self.assertEqual(block["healthy_cycles"], 1)
            self.assertEqual(block["window_cycles"], 3)
            self.assertNotIn("next", block)
            self.assertNotIn("checks", block)
            reactor.close()

    def test_recovery_block_never_asserts_a_verdict_it_did_not_observe(self) -> None:
        """An unresolved block used to carry resolved_by claiming live=true.

        Read: with ``resolved: false`` the block still read
        "reboot_outcome in the next start envelope: outcome=in_progress,
        live=true (head == promoted commit)" -- the exact claim the branch had
        just failed to establish. A reader that trusts ``resolved_by`` would
        believe the promotion was verified when it was not.
        """
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic", goal_id)
            with patch.object(reactor.runner, "run", return_value=AgentRunResult(
                status=RunStatus.COMPLETED,
                report=json.dumps({"status": "COMPLETED", "summary": "promoted"}),
                steps=1,
                usage_tokens=1,
                control_action={"type": "restart_after_checkpoint", "proposal_id": "p1", "commit": "abc1234"},
            )):
                reactor.tick("test")
            block = reactor.store.state().next_plan["self_improvement_recovery"]
            print("unresolved block:", json.dumps(block, sort_keys=True))
            self.assertFalse(block["resolved"])
            self.assertNotIn("resolved_by", block)
            self.assertIn("no reboot window is readable", block["unresolved_because"])
            # The historical hint and its four checks survive unchanged.
            self.assertEqual(block["next"], "verify the promoted self-improvement after reboot")
            self.assertEqual(block["checks"], ["service starts", "database opens", "provider is reachable", "health window completes"])
            reactor.close()

    def test_unresolved_recovery_block_names_a_window_on_a_commit_head_is_not_on(self) -> None:
        """The three ways a window can fail to answer are distinguished."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "tracked.py").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            stale = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            (root / "tracked.py").write_text("two\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "later"], cwd=root, check=True)
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            (root / "reboot-guard.json").write_text(
                json.dumps({
                    "proposal_id": "p1", "commit": stale, "rollback_commit": stale,
                    "healthy_cycles": 1, "active": True, "failed": False,
                    "started_at": "2026-09-18T00:00:00Z",
                }),
                encoding="utf-8",
            )
            reactor = self._reactor(directory, ProseProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic", goal_id)
            with patch.object(reactor.runner, "run", return_value=AgentRunResult(
                status=RunStatus.COMPLETED,
                report=json.dumps({"status": "COMPLETED", "summary": "promoted"}),
                steps=1,
                usage_tokens=1,
                control_action={"type": "restart_after_checkpoint", "proposal_id": "p1", "commit": stale},
            )):
                reactor.tick("test")
            block = reactor.store.state().next_plan["self_improvement_recovery"]
            print("stale-window block:", json.dumps(block, sort_keys=True))
            self.assertFalse(block["resolved"])
            self.assertFalse(block["live"])
            self.assertIn(stale, block["unresolved_because"])
            self.assertIn(head, block["unresolved_because"])
            # A guard row written before ``begin`` recorded the window size has no
            # ``window_cycles`` key; the block must omit it rather than invent one.
            self.assertNotIn("window_cycles", block)
            self.assertEqual(block["next"], "verify the promoted self-improvement after reboot")
            reactor.close()

    def test_recovery_block_reads_the_verdict_of_the_window_it_names(self) -> None:
        """The block's verdict must describe the window the block names.

        Read (event_log 17949): run b59054c5 started while the
        window in front of it was terminal, promoted d275d4b8, and wrote
        ``resolved: false`` plus the "verify the promoted self-improvement after
        reboot" imperative for it -- although d275d4b8 was live, as the very next
        start envelope's reboot_outcome (outcome=in_progress, head=d275d4b8,
        live=true) shows. The verdict was computed from the wake-time
        observation, which describes the *previous* window, and then attached to
        a different commit. ``reboot_outcome`` is therefore re-read for the
        promoted commit, so a promotion that is live is not handed back as an
        unverified hint.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "tracked.py").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            old_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            # The window in front of this run is terminal: its envelope reads
            # outcome=accepted, which is exactly why the verdict used to be false.
            (root / "reboot-guard.json").write_text(
                json.dumps({
                    "proposal_id": "p1", "commit": old_head, "rollback_commit": old_head,
                    "healthy_cycles": 3, "window_cycles": 3, "active": False,
                    "failed": False, "completed_at": "2026-09-22T20:50:24Z",
                }),
                encoding="utf-8",
            )
            reactor = self._reactor(directory, ProseProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic", goal_id)
            promoted: dict[str, str] = {}

            def promotion(*_args, **_kwargs):
                """What a real promotion does: commit, then open a fresh window."""
                (root / "tracked.py").write_text("two\n", encoding="utf-8")
                subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
                subprocess.run(["git", "commit", "-qm", "promoted"], cwd=root, check=True)
                new_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
                promoted["head"] = new_head
                (root / "reboot-guard.json").write_text(
                    json.dumps({
                        "proposal_id": "p2", "commit": new_head, "rollback_commit": old_head,
                        "healthy_cycles": 1, "window_cycles": 3, "active": True,
                        "failed": False, "release": None,
                        "started_at": "2026-09-22T21:49:44Z",
                    }),
                    encoding="utf-8",
                )
                return AgentRunResult(
                    status=RunStatus.COMPLETED,
                    report=json.dumps({"status": "COMPLETED", "summary": "promoted"}),
                    steps=1,
                    usage_tokens=1,
                    control_action={"type": "restart_after_checkpoint", "proposal_id": "p2", "commit": new_head},
                )

            with patch.object(reactor.runner, "run", side_effect=promotion):
                reactor.tick("test")
            block = reactor.store.state().next_plan["self_improvement_recovery"]
            print("block after a live promotion:", json.dumps(block, sort_keys=True))
            self.assertTrue(block["resolved"], "the promotion is live, so the block must not ask for re-verification")
            self.assertTrue(block["live"])
            self.assertEqual(block["commit"], promoted["head"])
            self.assertEqual(block["head"], promoted["head"])
            self.assertEqual(block["outcome"], "in_progress")
            self.assertNotIn("next", block)
            self.assertNotIn("checks", block)
            self.assertNotIn("unresolved_because", block)
            reactor.close()

    def test_resolved_recovery_block_omits_a_progress_count_it_did_not_read(self) -> None:
        """Progress counters belong to the window they were read from.

        Measured on the live plan ledger: all 11 resolved
        ``self_improvement_recovery`` blocks carried the *previous* window's
        ``healthy_cycles``. The shape is ordinary -- the promotion commits, and
        its own guard row is written only by the next startup, so the row
        readable at block-emission time names the commit promoted *before* this
        one (event_log 23247: block commit 661c05d4, the row it read named
        76e07b58 with healthy_cycles=2, and the block reported 2 of 3). The
        committed window had run zero cycles. ``live`` is a fact about HEAD and
        stays true, but "N of M healthy cycles" is a claim about the named
        window's progress.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "tracked.py").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            previous = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            # The row in front of this run is OPEN and names the PREVIOUS
            # promotion, two-thirds of the way through its own window.
            (root / "reboot-guard.json").write_text(
                json.dumps({
                    "proposal_id": "p1", "commit": previous, "rollback_commit": previous,
                    "healthy_cycles": 2, "window_cycles": 3, "active": True,
                    "failed": False, "started_at": "2026-09-24T14:30:00Z",
                }),
                encoding="utf-8",
            )
            reactor = self._reactor(directory, ProseProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic", goal_id)
            promoted: dict[str, str] = {}

            def promotion(*_args, **_kwargs):
                """A promotion commits; its own guard row waits for the restart."""
                (root / "tracked.py").write_text("two\n", encoding="utf-8")
                subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
                subprocess.run(["git", "commit", "-qm", "promoted"], cwd=root, check=True)
                promoted["head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
                return AgentRunResult(
                    status=RunStatus.COMPLETED,
                    report=json.dumps({"status": "COMPLETED", "summary": "promoted"}),
                    steps=1,
                    usage_tokens=1,
                    control_action={"type": "restart_after_checkpoint", "proposal_id": "p2", "commit": promoted["head"]},
                )

            with patch.object(reactor.runner, "run", side_effect=promotion):
                reactor.tick("test")
            block = reactor.store.state().next_plan["self_improvement_recovery"]
            print("block after promoting past an open window:", json.dumps(block, sort_keys=True))
            self.assertTrue(block["resolved"])
            self.assertTrue(block["live"])
            self.assertEqual(block["commit"], promoted["head"])
            self.assertEqual(block["head"], promoted["head"])
            self.assertNotEqual(block["commit"], previous)
            # 2 of 3 belongs to the window that named ``previous``, not to this one.
            self.assertNotIn("healthy_cycles", block)
            self.assertNotIn("window_cycles", block)
            reactor.close()

    def test_recovery_block_never_blames_a_window_it_did_not_read(self) -> None:
        """A window that names a different commit must not be reported as this one's.

        Measured on the live event log: all 8 ``unresolved_because`` strings ever
        written are the same misattribution, "the reboot window already ended
        (outcome=accepted)" -- including event_log 22361, where the row that
        answered named the *previous* promotion (e3fb1093) while the window for
        the promotion under test (324d958a) had not opened yet. The flag that
        gates the honest message was ``bool(verdict)``, i.e. True for any row at
        all, so the branch "no reboot window names the promoted commit" was
        unreachable and the block asserted a fact about a window it never read.
        That false reason is what the planner turned into a queued
        re-verification task.

        This test reproduces the live shape exactly: a terminal window in front
        of the run naming the base commit, and a promotion whose own window has
        not been opened yet.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "tracked.py").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            old_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            # The window in front of this run already ended, and it names the
            # PREVIOUS promotion's commit.
            (root / "reboot-guard.json").write_text(
                json.dumps({
                    "proposal_id": "p1", "commit": old_head, "rollback_commit": old_head,
                    "healthy_cycles": 3, "window_cycles": 3, "active": False,
                    "failed": False, "completed_at": "2026-09-24T11:00:00Z",
                }),
                encoding="utf-8",
            )
            reactor = self._reactor(directory, ProseProvider())
            goal_id = reactor.store.add_goal("liveness", priority=1.0)
            reactor.store.add_task("bounded diagnostic", goal_id)
            promoted: dict[str, str] = {}

            def promotion(*_args, **_kwargs):
                """A promotion commits, but its window opens only at the restart."""
                (root / "tracked.py").write_text("two\n", encoding="utf-8")
                subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
                subprocess.run(["git", "commit", "-qm", "promoted"], cwd=root, check=True)
                promoted["head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
                return AgentRunResult(
                    status=RunStatus.COMPLETED,
                    report=json.dumps({"status": "COMPLETED", "summary": "promoted"}),
                    steps=1,
                    usage_tokens=1,
                    control_action={"type": "restart_after_checkpoint", "proposal_id": "p2", "commit": promoted["head"]},
                )

            with patch.object(reactor.runner, "run", side_effect=promotion):
                reactor.tick("test")
            block = reactor.store.state().next_plan["self_improvement_recovery"]
            print("block with no window for the promotion:", json.dumps(block, sort_keys=True))
            self.assertFalse(block["resolved"])
            self.assertEqual(block["commit"], promoted["head"])
            self.assertIn(promoted["head"], block["unresolved_because"])
            self.assertNotIn("already ended", block["unresolved_because"])
            self.assertNotIn(old_head, block["unresolved_because"])
            # The historical hint and its four checks survive unchanged.
            self.assertEqual(block["next"], "verify the promoted self-improvement after reboot")
            self.assertEqual(block["checks"], ["service starts", "database opens", "provider is reachable", "health window completes"])
            reactor.close()

    def test_start_envelope_names_a_window_the_tree_has_moved_past(self) -> None:
        """An open window naming a commit HEAD is not on is reported, not hidden.

        This is the observable the deferred-restart confirmation tax was paid
        for: without it, the only way to learn whether the promoted commit is
        the running one was to re-run ``git rev-parse HEAD`` by hand in a later
        run.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "tracked.py").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            stale = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            (root / "tracked.py").write_text("two\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "later"], cwd=root, check=True)
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            self.assertNotEqual(head, stale)
            (root / "reboot-guard.json").write_text(
                json.dumps({
                    "proposal_id": "p1", "commit": stale, "rollback_commit": stale,
                    "healthy_cycles": 1, "active": True, "failed": False,
                    "started_at": "2026-09-18T00:00:00Z",
                }),
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
            outcome = observations["reboot_outcome"]
            self.assertEqual(outcome["outcome"], "in_progress")
            self.assertEqual(outcome["commit"], stale)
            self.assertEqual(outcome["head"], head)
            self.assertFalse(outcome["live"])
            self.assertTrue(outcome["stale"])
            reactor.close()

    def test_planner_memory_query_is_derived_from_the_situation(self) -> None:
        # The planner used to search a fixed phrase that recalled nothing about
        # the actual goal or the previous outcome.
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

    def test_planner_memory_query_reads_the_outcome_key_the_reactor_writes(self) -> None:
        # The reactor records a run's outcome as {"report": ..., "status": ...} --
        # the shape `previous_outcome` is written with after every run -- so a
        # reader that asked only for "summary" took nothing from the previous
        # run: measured on the live ledger, previous_outcome["report"] is
        # non-empty on 180/180 run_started envelopes and ["summary"] on 0/180.
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider())
            state = reactor.store.state()
            state.next_plan = {
                "previous_outcome": {"report": "the gate rejected a patch", "status": "COMPLETED"},
                "initial_prompt": "retry with a multi-line anchor",
            }
            query = reactor._planner_memory_query(state, [{"title": "Advance the SkyNet roadmap"}])
            self.assertIn("the gate rejected a patch", query)
            self.assertIn("retry with a multi-line anchor", query)
            self.assertIn("Advance the SkyNet roadmap", query)
            reactor.close()

    def test_planner_memory_query_still_reads_an_older_summary_only_envelope(self) -> None:
        # An envelope written by an older writer carries "summary" and no
        # "report"; repairing the reader must not make that shape dead again.
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider())
            state = reactor.store.state()
            state.next_plan = {
                "previous_outcome": {"summary": "the gate rejected a patch"},
                "initial_prompt": "retry with a multi-line anchor",
            }
            query = reactor._planner_memory_query(state, [{"title": "Advance the SkyNet roadmap"}])
            self.assertIn("the gate rejected a patch", query)
            reactor.close()

    def test_memory_query_is_built_from_the_task_envelope(self) -> None:
        # `selected_work` is task_work()'s envelope; reading `selected_work["title"]`
        # returned "" for every task and made recall silently empty.
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
            # Every injected memory in this fixture has no resolvable provenance,
            # and the payload must say so instead of leaving it to be inferred.
            self.assertEqual(payload["injected_total"], payload["hits"] + payload["pinned"])
            self.assertEqual(payload["injected_resolvable"], 0)
            self.assertEqual(payload["injected_unresolvable"], payload["injected_total"])
            # The same counts reach the digest, so the gap is readable without
            # re-deriving it from the raw payloads.
            health = memory_health(reactor.store.connection, "1970-01-01T00:00:00Z")
            self.assertEqual(health["retrieval_injected"], payload["injected_total"])
            self.assertEqual(health["retrieval_unresolvable_injected"], payload["injected_unresolvable"])
            self.assertIn("unresolvable_injected=", format_report({"memory": health}))
            reactor.close()

    def test_memory_provenance_splits_resolvable_from_unknown(self) -> None:
        """A NULL or unknown source_run is provenance that cannot be checked.

        Retrieval injects on match or pin alone, so the split has to be counted
        explicitly: a future trust policy needs this number to move, and an
        unreadable provenance must not be counted as a verified one.
        """
        with tempfile.TemporaryDirectory() as directory:
            reactor = self._reactor(directory, ProseProvider())
            store = reactor.store
            store.connection.execute(
                "INSERT INTO runs(run_id, attempt, status, started_at, budget) VALUES('run-known',1,'running','2026-01-01T00:00:00Z','{}')"
            )
            known = store.remember_memory("a memory from a known run", kind="fact", source_run="run-known")
            unknown = store.remember_memory("a memory from an unknown run", kind="fact", source_run="ghost-run")
            orphan = store.remember_memory("a memory with no provenance at all", kind="fact", source_run=None)
            rows = [
                dict(row)
                for row in store.connection.execute(
                    "SELECT memory_id, source_run FROM memories WHERE memory_id IN (?,?,?)", (known, unknown, orphan)
                ).fetchall()
            ]
            self.assertEqual(len(rows), 3)
            self.assertEqual(store.memory_provenance(rows), {"total": 3, "resolvable": 1, "unresolvable": 2})
            self.assertEqual(store.memory_provenance([]), {"total": 0, "resolvable": 0, "unresolvable": 0})
            reactor.close()

    @staticmethod
    def _task_row(reactor: Reactor, task_id: str):
        return reactor.store.connection.execute(
            "SELECT status, attempts, consecutive_model_failures FROM tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()


if __name__ == "__main__":
    unittest.main()
