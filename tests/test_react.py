"""ReAct runner behaviour tests."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

from helpers import FakeProvider, FixtureTool, commit_all, git_repo

from skynet import react
from skynet.models import Budget, ModelTurn, RunStatus, StartEnvelope, ToolCall
from skynet.provider import Tool
from skynet.providers.errors import ProviderError
from skynet.react import ReActConfig, ReActRunner
from skynet.reactor import Reactor, WatchdogTimeout
from skynet.store import StateStore

FINISH_OK = json.dumps({
    "status": "COMPLETED",
    "summary": "done",
    "evidence": ["tool"],
    "actions": [],
    "changes": [],
    "tests": [],
    "blocker": "",
    "next_hypothesis": "",
})


class CoreTests(unittest.TestCase):
    def test_environment_blocker_is_retryable_not_exhausted(self) -> None:
        report = json.dumps({
            "status": "BLOCKED",
            "summary": "proposal could not start",
            "blocker": "self-improvement requires no modified tracked files in the main worktree",
            "next_hypothesis": "retry after the worktree is clean",
        })
        self.assertTrue(Reactor._is_retryable_environment_blocker(report))
        self.assertFalse(Reactor._is_retryable_environment_blocker(json.dumps({
            "status": "BLOCKED",
            "summary": "the hypothesis was disproved",
            "blocker": "the proposed behavior is unsafe",
            "next_hypothesis": "choose a different bounded hypothesis",
        })))
    def test_react_start_envelope_has_explicit_data_boundaries(self) -> None:
        class CaptureProvider:
            def __init__(self) -> None:
                self.messages = []

            def complete(self, messages, *, max_tokens, tools=()):
                self.messages.append(messages)
                return ModelTurn(text="Status: BLOCKED\nNo evidence.", usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            provider = CaptureProvider()
            runner = ReActRunner(provider, StateStore(Path(directory) / "state.sqlite3"), {})
            runner.run(StartEnvelope("test", observations=[{"text": "context"}]), "system")
            content = provider.messages[0][1]["content"]
            self.assertTrue(content.startswith("START ENVELOPE\n"))
            self.assertTrue(content.endswith("\nEND START ENVELOPE"))
    def test_hard_denial_is_recorded_as_its_own_event(self) -> None:
        # A permission decision must be durable, not only inside the tool_result
        # payload: the metrics allowlist and PROTECTED_EVENT_KINDS both name
        # 'policy_denied', but nothing ever wrote it, so a hard denial was
        # invisible to every reporting surface.
        class DenyTool:
            name = "deny_tool"
            schema = {"type": "function", "function": {"name": "deny_tool", "description": "d", "parameters": {"type": "object", "additionalProperties": False}}}  # noqa: RUF012 - Tool protocol reads schema as an instance property

            def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
                return {"ok": False, "policy_denied": True, "policy_warning": True, "protected_path": "state/", "error": "policy denied"}

        class OneShotProvider:
            calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                if self.calls == 1:
                    return ModelTurn(tool_calls=[ToolCall("deny_tool", {})], usage_tokens=1)
                return ModelTurn(text=FINISH_OK, usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            runner = ReActRunner(OneShotProvider(), store, {"deny_tool": DenyTool()})
            runner.run(StartEnvelope("timer", run_id="run-denial"), "system")
            rows = store.connection.execute("SELECT payload FROM event_log WHERE kind='policy_denied'").fetchall()
            self.assertEqual(len(rows), 1)
            payload = json.loads(rows[0][0])
            self.assertEqual(payload["tool"], "deny_tool")
            self.assertEqual(payload["protected_path"], "state/")
            # A hard denial must not be shadowed by a soft warning on the same
            # result.
            soft = store.connection.execute("SELECT count(*) FROM event_log WHERE kind='policy_soft_denied'").fetchone()[0]
            self.assertEqual(soft, 0)
            store.close()
    def test_real_bash_hard_denial_reaches_the_event_log(self) -> None:
        # The producer/consumer link, not a fake: the real BashTool returns
        # policy_denied and the real runner records it.
        from skynet.policy import ExecutionPolicy
        from skynet.tools import BashTool

        class BashDenyProvider:
            def __init__(self, cwd: str) -> None:
                self.cwd = cwd
                self.calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                if self.calls == 1:
                    return ModelTurn(tool_calls=[ToolCall("bash", {"command": "rm -rf .git/", "cwd": self.cwd})], usage_tokens=1)
                return ModelTurn(text=FINISH_OK, usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("keep", encoding="utf-8")
            tool = BashTool(ExecutionPolicy(workspace=root))
            store = StateStore(root / "state.sqlite3")
            runner = ReActRunner(BashDenyProvider(str(root)), store, {"bash": tool})
            runner.run(StartEnvelope("timer", run_id="run-bash-denial"), "system")
            rows = store.connection.execute("SELECT payload FROM event_log WHERE kind='policy_denied'").fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0][0])["tool"], "bash")
            self.assertTrue((root / ".git" / "config").exists())
            store.close()
    def test_runtime_log_contains_events_and_full_provider_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.append_event("test_event", {"value": 1}, "run-1")
            store.runtime_log.write("provider_request", {"messages": [{"role": "user", "content": "input"}]}, run_id="run-1")
            store.runtime_log.write("provider_response", {"text": "output", "tool_calls": []}, run_id="run-1")
            records = [json.loads(line) for line in (Path(directory) / "runtime.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["kind"] for record in records], ["event", "provider_request", "provider_response"])
            self.assertEqual(records[0]["payload"]["event_kind"], "test_event")
            self.assertEqual(records[1]["payload"]["messages"][0]["content"], "input")
            self.assertEqual(records[2]["payload"]["text"], "output")
            store.close()
    def test_react_allows_cumulative_usage_across_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            class MultiStepProvider:
                calls = 0

                def complete(self, messages, *, max_tokens, tools=()):
                    self.calls += 1
                    if self.calls == 1:
                        return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], usage_tokens=150_000)
                    return ModelTurn(text=FINISH_OK, usage_tokens=150_000)

            store = StateStore(Path(directory) / "state.sqlite3")
            provider = MultiStepProvider()
            result = ReActRunner(
                provider,
                store,
                {"fixture_tool": FixtureTool()},
                ReActConfig(max_tokens=200_000, output_tokens=8_192, provider_retries=0),
            ).run(StartEnvelope("test"), "system")
            self.assertEqual(result.status, RunStatus.COMPLETED)
            self.assertEqual(provider.calls, 2)
            store.close()
    def test_react_uses_per_request_output_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            class CapturingProvider:
                max_tokens = None

                def complete(self, messages, *, max_tokens, tools=()):
                    self.max_tokens = max_tokens
                    return ModelTurn(text=FINISH_OK, usage_tokens=3)

            provider = CapturingProvider()
            store = StateStore(Path(directory) / "state.sqlite3")
            result_runner = ReActRunner(
                provider,
                store,
                {},
                ReActConfig(max_tokens=200_000, output_tokens=100, provider_retries=0),
            )
            result = result_runner.run(StartEnvelope("test"), "system")
            self.assertEqual(result.status, RunStatus.COMPLETED)
            self.assertIsNotNone(provider.max_tokens)
            self.assertGreater(provider.max_tokens or 0, 0)
            self.assertEqual(provider.max_tokens, 100)
            store.close()
    def test_react_stops_before_tool_when_provider_exhausts_time_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            class SlowProvider:
                def complete(self, messages, *, max_tokens, tools=()):
                    time.sleep(0.02)
                    return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], usage_tokens=1)

            class CountingTool(FixtureTool):
                calls = 0

                def execute(self, arguments, *, idempotency_key):
                    self.calls += 1
                    return super().execute(arguments, idempotency_key=idempotency_key)

            tool = CountingTool()
            store = StateStore(Path(directory) / "state.sqlite3")
            result = ReActRunner(
                SlowProvider(),
                store,
                {"fixture_tool": tool},
                ReActConfig(timeout_seconds=0.01, provider_retries=0),
            ).run(StartEnvelope("test"), "system")
            self.assertEqual(result.status, RunStatus.NEEDS_RECOVERY)
            self.assertEqual(tool.calls, 0)
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='budget_exhausted'").fetchone()[0],
                1,
            )
            store.close()
    def test_react_stops_before_provider_when_the_request_would_exceed_context(self) -> None:
        class CountingProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                return ModelTurn(text="unexpected")

        cases = (
            (ReActConfig(max_tokens=80, context_finish_reserve=10), "system " + "x" * 300, []),
            (ReActConfig(max_tokens=200_000, max_compactions=0), "system", [{"large": "x" * 600_000}]),
            (ReActConfig(max_tokens=100, context_finish_reserve=10, max_compactions=0), "system", [{"large": "x" * 400}]),
        )
        for config, system, observations in cases:
            with tempfile.TemporaryDirectory() as directory:
                provider = CountingProvider()
                store = StateStore(Path(directory) / "state.sqlite3")
                runner = ReActRunner(provider, store, {}, config)
                result = runner.run(StartEnvelope("test", observations=observations), system)
                self.assertEqual(result.status, RunStatus.NEEDS_RECOVERY)
                self.assertEqual(provider.calls, 0)
                self.assertTrue(result.report.startswith("Finish Report (harness)"))
                store.close()

    def test_repeated_identical_model_responses_trigger_the_doom_loop_guard(self) -> None:
        # The guard existed but had no test. A model that keeps emitting the same
        # prose without acting must be cut off, not allowed to burn the budget.
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")

            class RepeatingProvider:
                def __init__(self) -> None:
                    self.calls = 0

                def complete(self, messages, *, max_tokens, tools=()):
                    self.calls += 1
                    if not tools:
                        return ModelTurn(text=FINISH_OK, usage_tokens=1)
                    # The same call with the same arguments, forever.
                    return ModelTurn(tool_calls=[ToolCall("fixture_tool", {"n": 1})], usage_tokens=1)

            provider = RepeatingProvider()
            runner = ReActRunner(
                provider, store, {"fixture_tool": FixtureTool()}, ReActConfig(max_repeated_responses=2, max_steps=20)
            )
            result = runner.run(StartEnvelope("test"), "system")
            # The guard cuts the loop and asks for a final report, so the run can
            # still complete - what matters is that it stopped repeating.
            self.assertEqual(result.failure, "doom loop")
            kinds = [row[0] for row in store.connection.execute("SELECT kind FROM event_log")]
            self.assertIn("doom_loop", kinds)
            payload = json.loads(
                store.connection.execute("SELECT payload FROM event_log WHERE kind='doom_loop'").fetchone()[0]
            )
            self.assertEqual(payload["kind"], "model_response")
            # The guard fires early: it does not wait for the step budget.
            self.assertLess(provider.calls, 20)
            store.close()

    def test_a_changing_response_is_not_a_doom_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")

            class VaryingProvider:
                def __init__(self) -> None:
                    self.calls = 0

                def complete(self, messages, *, max_tokens, tools=()):
                    self.calls += 1
                    if not tools or self.calls >= 4:
                        return ModelTurn(text=FINISH_OK, usage_tokens=1)
                    return ModelTurn(tool_calls=[ToolCall("fixture_tool", {"n": self.calls})], usage_tokens=1)

            provider = VaryingProvider()
            runner = ReActRunner(
                provider, store, {"fixture_tool": FixtureTool()}, ReActConfig(max_repeated_responses=2, max_steps=20)
            )
            result = runner.run(StartEnvelope("test"), "system")
            self.assertNotEqual(result.failure, "doom loop")
            kinds = [row[0] for row in store.connection.execute("SELECT kind FROM event_log")]
            self.assertNotIn("doom_loop", kinds)
            store.close()

    def test_react_compacts_once_before_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            provider = FakeProvider()
            runner = ReActRunner(provider, store, {}, ReActConfig(max_tokens=180, context_finish_reserve=20, max_compactions=1))
            start = StartEnvelope("test", observations=[{"payload": "x" * 140}])
            result = runner.run(start, "system")
            self.assertEqual(result.status, RunStatus.NEEDS_RECOVERY)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='context_compacted'").fetchone()[0], 0)
            self.assertEqual(provider.calls, 0)
            self.assertTrue(result.report.startswith("Finish Report (harness)"))
            store.close()
    def test_react_does_not_treat_session_usage_as_context_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            class BudgetProvider:
                calls = 0

                def complete(self, messages, *, max_tokens, tools=()):
                    self.calls += 1
                    if self.calls == 1:
                        return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], usage_tokens=80)
                    return ModelTurn(text=FINISH_OK, usage_tokens=10)

            store = StateStore(Path(directory) / "state.sqlite3")
            provider = BudgetProvider()
            runner = ReActRunner(
                provider,
                store,
                {"fixture_tool": FixtureTool()},
                ReActConfig(max_tokens=1000, output_tokens=25, max_steps=5),
            )
            result = runner.run(StartEnvelope("test"), "system")
            self.assertEqual(result.status, RunStatus.COMPLETED)
            self.assertEqual(json.loads(result.report)["status"], "COMPLETED")
            self.assertEqual(provider.calls, 2)
            finish = store.connection.execute("SELECT payload FROM event_log WHERE kind='finish_report'").fetchone()
            self.assertIsNotNone(finish)
            self.assertEqual(json.loads(json.loads(finish[0])["text"])["status"], "COMPLETED")
            store.close()
    def test_react_uses_protective_finish_threshold_before_next_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            class Provider:
                calls = 0

                def complete(self, messages, *, max_tokens, tools=()):
                    self.calls += 1
                    return ModelTurn(text=FINISH_OK, usage_tokens=1)

            provider = Provider()
            store = StateStore(Path(directory) / "state.sqlite3")
            runner = ReActRunner(
                provider,
                store,
                {"fixture_tool": FixtureTool()},
                ReActConfig(max_tokens=1000, finish_threshold_ratio=0.01),
            )
            result = runner.run(StartEnvelope("test"), "system")
            self.assertEqual(json.loads(result.report)["status"], "COMPLETED")
            self.assertEqual(provider.calls, 1)
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='react_phase' AND payload LIKE '%protective context threshold%'").fetchone()[0],
                1,
            )
            store.close()
    def test_provider_failure_retries_then_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            class FlakyProvider:
                calls = 0

                def complete(self, messages, *, max_tokens, tools=()):
                    self.calls += 1
                    if self.calls < 3:
                        raise OSError("temporary gateway failure")
                    return ModelTurn(text='{"status":"completed","summary":"recovered"}')

            store = StateStore(Path(directory) / "state.sqlite3")
            provider = FlakyProvider()
            result = ReActRunner(
                provider,
                store,
                {},
                ReActConfig(provider_retries=2, provider_retry_backoff_seconds=0),
            ).run(StartEnvelope("test"), "system")
            self.assertEqual(result.status, RunStatus.COMPLETED)
            self.assertEqual(provider.calls, 3)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='provider_retry'").fetchone()[0], 2)
            store.close()
    def test_provider_failure_becomes_recovery_after_bounded_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            class DeadProvider:
                calls = 0

                def complete(self, messages, *, max_tokens, tools=()):
                    self.calls += 1
                    raise TimeoutError("gateway timeout")

            store = StateStore(Path(directory) / "state.sqlite3")
            provider = DeadProvider()
            result = ReActRunner(
                provider,
                store,
                {},
                ReActConfig(provider_retries=1, provider_retry_backoff_seconds=0),
            ).run(StartEnvelope("test"), "system")
            self.assertEqual(result.status, RunStatus.NEEDS_RECOVERY)
            self.assertEqual(provider.calls, 2)
            self.assertEqual(store.connection.execute("SELECT kind FROM event_log ORDER BY sequence DESC LIMIT 1").fetchone()[0], "provider_failure")
            store.close()
    def test_read_tool_exception_retries_but_write_tool_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            class FlakyTool:
                name = "flaky"
                capability_kind = "read"
                timeout_seconds = 1.0
                calls = 0

                @property
                def schema(self):
                    return {"type": "function", "function": {"name": self.name, "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}

                def execute(self, arguments, *, idempotency_key):
                    self.calls += 1
                    if self.calls == 1:
                        raise OSError("temporary read failure")
                    return {"ok": True}

            class ToolProvider:
                def complete(self, messages, *, max_tokens, tools=()):
                    if len(messages) == 2:
                        return ModelTurn(tool_calls=[ToolCall("flaky", {})])
                    return ModelTurn(text='{"status":"completed"}')

            store = StateStore(Path(directory) / "state.sqlite3")
            tool = FlakyTool()
            result = ReActRunner(
                ToolProvider(), store, cast(dict[str, Tool], {"flaky": tool}), ReActConfig(tool_retry_backoff_seconds=0)
            ).run(StartEnvelope("test"), "system")
            self.assertEqual(result.status, RunStatus.COMPLETED)
            self.assertEqual(tool.calls, 2)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='tool_retry'").fetchone()[0], 1)
            store.close()
    def test_compaction_never_starts_the_tail_with_a_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = ReActRunner(MagicMock(), StateStore(Path(directory) / "state.sqlite3"), {}, ReActConfig(compaction_keep_messages=5))
            messages = [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a1"},
                {"role": "tool", "tool_call_id": "c1", "content": "r1"},
                {"role": "assistant", "content": "a2"},
                {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            ]
            for keep in (1, 2, 3, 4):
                start = runner._tail_start(messages, keep)
                self.assertNotEqual(messages[start].get("role"), "tool")
    def test_forced_finish_respects_blocked_status(self) -> None:
        class BlockedFinishProvider:
            def complete(self, messages, *, max_tokens, tools=()):
                return ModelTurn(text=json.dumps({
                    "status": "BLOCKED", "summary": "cannot proceed", "evidence": [],
                    "actions": [], "changes": [], "tests": [], "blocker": "no access", "next_hypothesis": "",
                }), usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            from skynet.models import Budget, RunRecord
            from skynet.time import utc_now
            store = StateStore(Path(directory) / "state.sqlite3")
            runner = ReActRunner(BlockedFinishProvider(), store, {}, ReActConfig())
            with store.transaction():
                store.create_run(RunRecord("finish-run", 1, RunStatus.RUNNING, utc_now(), Budget()))
            messages: list[dict[str, object]] = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
            status, report, _usage = runner._finish(messages, "finish-run", reason="test")
            self.assertEqual(status, RunStatus.BLOCKED)
            self.assertIn("no access", report)
    def test_react_does_not_abort_on_cumulative_output(self) -> None:
        class VerboseProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                if not tools:
                    return ModelTurn(text=json.dumps({
                        "status": "COMPLETED", "summary": "bounded finish", "evidence": ["tool"],
                        "actions": [], "changes": [], "tests": [], "blocker": "", "next_hypothesis": "",
                    }), completion_tokens=10, usage_tokens=10)
                self.calls += 1
                # Arguments vary: an identical call repeated is a doom loop, which
                # is a different test.
                return ModelTurn(
                    tool_calls=[ToolCall("fixture_tool", {"n": self.calls})],
                    completion_tokens=4_000, usage_tokens=4_000,
                )

        with tempfile.TemporaryDirectory() as directory:
            from skynet.models import Budget
            store = StateStore(Path(directory) / "state.sqlite3")
            runner = ReActRunner(VerboseProvider(), store, {"fixture_tool": FixtureTool()}, ReActConfig(max_steps=6))
            result = runner.run(StartEnvelope("test", budget=Budget(steps=6, output_tokens=100)), "system")
            self.assertEqual(result.steps, 6)
            self.assertEqual(result.failure, "step budget")
            self.assertIsNotNone(store.connection.execute("SELECT 1 FROM event_log WHERE kind='budget_exhausted'").fetchone())
            store.close()

    def test_episode_does_not_duplicate_provider_retries(self) -> None:
        class AlwaysFails:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                self.calls += 1
                raise ProviderError("down", category="server", retryable=True)

        provider = AlwaysFails()
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            result = ReActRunner(provider, store, {}, ReActConfig()).run(StartEnvelope("test"), "system")
            self.assertEqual(provider.calls, 1)
            self.assertEqual(result.status, RunStatus.NEEDS_RECOVERY)
            self.assertEqual(result.failure, "down")
            store.close()

    def test_react_time_budget_counts_only_model_time(self) -> None:
        class SlowProvider:
            def complete(self, messages, *, max_tokens, tools=()):
                if not tools:
                    return ModelTurn(text=json.dumps({
                        "status": "COMPLETED", "summary": "bounded finish", "evidence": ["tool"],
                        "actions": [], "changes": [], "tests": [], "blocker": "", "next_hypothesis": "",
                    }), usage_tokens=1)
                time.sleep(0.05)
                return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], usage_tokens=1)

        class SlowTool(FixtureTool):
            calls = 0

            def execute(self, arguments, *, idempotency_key) -> dict[str, object]:
                type(self).calls += 1
                time.sleep(0.3)
                return super().execute(arguments, idempotency_key=idempotency_key)

        SlowTool.calls = 0
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            result = ReActRunner(
                SlowProvider(),
                store,
                {"fixture_tool": SlowTool()},
                ReActConfig(timeout_seconds=0.12, provider_retries=0),
            ).run(StartEnvelope("test"), "system")
            self.assertEqual(result.failure, "time budget")
            self.assertGreaterEqual(SlowTool.calls, 1)
            store.close()

    def test_react_completion_time_ignores_failed_attempts(self) -> None:
        class FlakyProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                if self.calls == 1:
                    time.sleep(0.15)
                    raise ProviderError("network wobble", category="network", retryable=True)
                return ModelTurn(text="ok", usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            runner = ReActRunner(FlakyProvider(), store, {}, ReActConfig(provider_retries=1))
            turn, seconds = runner._complete_with_retry([{"role": "user", "content": "x"}], max_tokens=10, tools=[], run_id="run", step=0)
            self.assertEqual(turn.text, "ok")
            self.assertLess(seconds, 0.05)
            store.close()

    def test_compact_messages_preserves_start_and_bounded_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            runner = ReActRunner(MagicMock(), store, {}, ReActConfig(compaction_keep_messages=8))
            messages: list[dict[str, object]] = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
            for index in range(4):
                messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{index}", "type": "function", "function": {"name": "fixture_tool", "arguments": "{}"}}]})
                messages.append({"role": "tool", "tool_call_id": f"c{index}", "content": "x" * 200})
            compacted = runner._compact_messages(messages)
            self.assertLess(len(compacted), len(messages))
            self.assertEqual(compacted[:2], messages[:2])
            self.assertEqual(compacted[2]["role"], "system")
            self.assertIn("Context compacted", cast(str, compacted[2]["content"]))
            self.assertNotEqual(compacted[3].get("role"), "tool")
            store.close()

    def test_compact_messages_tolerates_a_tiny_keep_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            runner = ReActRunner(MagicMock(), store, {}, ReActConfig(compaction_keep_messages=2))
            messages: list[dict[str, object]] = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
            messages.append(cast(dict[str, object], {"role": "assistant", "content": "a"}))
            messages.append(cast(dict[str, object], {"role": "tool", "tool_call_id": "c", "content": "r"}))
            self.assertLessEqual(len(runner._compact_messages(messages)), len(messages))
            store.close()

    def test_multi_pass_compaction_stays_bounded_and_keeps_system(self) -> None:
        # With max_compactions=3 the runner may compact repeatedly; every pass
        # must keep the transcript bounded and never drop the system prompt or
        # split the newest tool exchange from its assistant call.
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            runner = ReActRunner(MagicMock(), store, {}, ReActConfig())
            self.assertEqual(runner.config.max_compactions, 3)
            self.assertEqual(runner.config.compaction_keep_messages, 4)
            self.assertEqual(runner.config.tool_result_max_chars, 8_000)
            messages: list[dict[str, object]] = [{"role": "system", "content": "identity"}, {"role": "user", "content": "start"}]
            for index in range(10):
                messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{index}", "type": "function", "function": {"name": "fixture_tool", "arguments": "{}"}}]})
                messages.append({"role": "tool", "tool_call_id": f"c{index}", "content": "x" * 400})
            compacted = messages
            sizes = [len(messages)]
            for _ in range(runner.config.max_compactions):
                compacted = runner._compact_messages(compacted)
                sizes.append(len(compacted))
                self.assertEqual(compacted[:2], messages[:2], "system prompt and Start envelope survive every pass")
            self.assertLess(sizes[-1], sizes[0])
            self.assertEqual(sizes, sorted(sizes, reverse=True), "compaction never grows the transcript")
            self.assertLessEqual(len(compacted), runner.config.compaction_keep_messages + 1)
            self.assertEqual(compacted[-1]["tool_call_id"], "c9")
            self.assertEqual(compacted[-2].get("role"), "assistant")
            store.close()

    def test_react_request_metadata_records_tool_count_request_chars_and_model(self) -> None:
        class ModelProvider:
            model = "test-model-v1"

            def complete(self, messages, *, max_tokens, tools=()):
                return ModelTurn(text=FINISH_OK, usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            runner = ReActRunner(ModelProvider(), store, {"fixture_tool": FixtureTool()}, ReActConfig(provider_retries=0))
            runner.run(StartEnvelope("test"), "system")
            row = store.connection.execute(
                "SELECT payload FROM transcript WHERE kind='provider_request' ORDER BY sequence LIMIT 1"
            ).fetchone()
            self.assertIsNotNone(row)
            meta = json.loads(row[0])
            self.assertEqual(meta["tool_count"], 1)
            self.assertEqual(meta["model"], "test-model-v1")
            self.assertGreater(meta["request_chars"], 0)
            store.close()

    def test_react_compacts_context_and_continues(self) -> None:
        class LargeResultTool:
            name = "large_tool"
            schema = {"type": "function", "function": {"name": name, "description": "big", "parameters": {"type": "object", "additionalProperties": False}}}  # noqa: RUF012 - Tool protocol reads schema as an instance property

            def execute(self, arguments, *, idempotency_key):
                return {"ok": True, "payload": "x" * 900}

        class CompactingProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                if self.calls <= 4:
                    return ModelTurn(tool_calls=[ToolCall("large_tool", {})], completion_tokens=1, usage_tokens=1)
                return ModelTurn(text=json.dumps({"status": "COMPLETED", "summary": "done", "evidence": ["tool"], "actions": [], "changes": [], "tests": [], "blocker": "", "next_hypothesis": ""}), completion_tokens=1, usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            provider = CompactingProvider()
            config = ReActConfig(max_tokens=400, context_finish_reserve=100, finish_threshold_ratio=2.0, context_warning_ratio=2.0, context_warning_ratio_high=2.0, compaction_keep_messages=2)
            runner = ReActRunner(provider, store, {"large_tool": LargeResultTool()}, config)
            result = runner.run(StartEnvelope("test", budget=Budget(tokens=400, steps=8, seconds=60)), "system")
            compactions = store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='context_compacted'").fetchone()[0]
            self.assertGreaterEqual(compactions, 1)
            self.assertLessEqual(compactions, config.max_compactions)
            # Repeated compaction keeps the run alive long enough to finish
            # instead of dying at the first context threshold.
            self.assertEqual(result.status, RunStatus.COMPLETED)
            store.close()

    def test_react_stores_bounded_transcripts_for_large_tool_results(self) -> None:
        class HugeTool:
            name = "huge_tool"
            schema = {"type": "function", "function": {"name": name, "description": "huge", "parameters": {"type": "object", "additionalProperties": False}}}  # noqa: RUF012 - Tool protocol reads schema as an instance property

            def execute(self, arguments, *, idempotency_key):
                return {"ok": True, "stdout": "x" * 300_000}

        class ToolLoopProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                if self.calls <= 6:
                    return ModelTurn(tool_calls=[ToolCall("huge_tool", {})], completion_tokens=1, usage_tokens=1)
                return ModelTurn(text=json.dumps({"status": "COMPLETED", "summary": "done", "evidence": ["tool"], "actions": [], "changes": [], "tests": [], "blocker": "", "next_hypothesis": ""}), completion_tokens=1, usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            config = ReActConfig(max_tokens=2000, context_finish_reserve=100, finish_threshold_ratio=2.0, context_warning_ratio=2.0, context_warning_ratio_high=2.0)
            runner = ReActRunner(ToolLoopProvider(), store, {"huge_tool": HugeTool()}, config)
            result = runner.run(StartEnvelope("test", budget=Budget(tokens=2000, steps=12, seconds=60)), "system")
            self.assertEqual(result.status, RunStatus.NEEDS_RECOVERY)
            run_id = store.connection.execute("SELECT run_id FROM transcript WHERE run_id IS NOT NULL ORDER BY sequence DESC LIMIT 1").fetchone()[0]

            history_rows = store.connection.execute("SELECT payload FROM transcript WHERE run_id=? AND kind='react_history'", (run_id,)).fetchall()
            self.assertEqual(len(history_rows), 1)

            requests = store.connection.execute("SELECT payload FROM transcript WHERE run_id=? AND kind='provider_request'", (run_id,)).fetchall()
            self.assertTrue(requests)
            for (payload,) in requests:
                self.assertNotIn("messages", json.loads(payload))
                self.assertLess(len(payload), 2000)

            largest = store.connection.execute("SELECT MAX(LENGTH(payload)) FROM transcript WHERE run_id=?", (run_id,)).fetchone()[0]
            self.assertLess(largest, 250_000)

            snapshot = store.snapshot_episode(run_id)
            self.assertNotIn("react_history", [row["kind"] for row in snapshot["transcript"]])
            store.close()

    def test_react_exhausts_step_budget_durably(self) -> None:
        class LoopingProvider:
            def complete(self, messages, *, max_tokens, tools=()):
                return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], completion_tokens=1, usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            from skynet.models import Budget
            store = StateStore(Path(directory) / "state.sqlite3")
            runner = ReActRunner(LoopingProvider(), store, {"fixture_tool": FixtureTool()}, ReActConfig())
            result = runner.run(StartEnvelope("test", budget=Budget(steps=2)), "system")
            self.assertEqual(result.status, RunStatus.NEEDS_RECOVERY)
            self.assertEqual(result.failure, "step budget")
            self.assertEqual(result.steps, 2)
            store.close()

    def test_react_defers_promoted_restart_until_accounting(self) -> None:
        class DeferredRestartTool:
            name = "promote_self_improvement"
            schema = {"type": "function", "function": {"name": name, "description": "promote", "parameters": {"type": "object", "additionalProperties": False}}}  # noqa: RUF012 - Tool protocol reads schema as an instance property

            def execute(self, arguments, *, idempotency_key):
                return {"ok": True, "control_action": {"type": "restart_after_checkpoint", "commit": "abc1234"}}

        class PromotingProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                if self.calls == 1:
                    return ModelTurn(tool_calls=[ToolCall("promote_self_improvement", {})], completion_tokens=1, usage_tokens=1)
                return ModelTurn(text=json.dumps({"status": "COMPLETED", "summary": "promoted", "evidence": ["promote"], "actions": [], "changes": [], "tests": [], "blocker": "", "next_hypothesis": ""}), completion_tokens=1, usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            runner = ReActRunner(PromotingProvider(), store, {"promote_self_improvement": DeferredRestartTool()}, ReActConfig())
            result = runner.run(StartEnvelope("test"), "system")
            self.assertEqual(result.failure, "deferred restart")
            self.assertEqual(result.control_action["type"], "restart_after_checkpoint")
            self.assertEqual(result.status, RunStatus.COMPLETED)
            store.close()

    def test_blocked_finish_with_promotion_is_completed(self) -> None:
        class DeferredRestartTool:
            name = "promote_self_improvement"
            schema = {"type": "function", "function": {"name": name, "description": "promote", "parameters": {"type": "object", "additionalProperties": False}}}  # noqa: RUF012 - Tool protocol reads schema as an instance property

            def execute(self, arguments, *, idempotency_key):
                return {"ok": True, "control_action": {"type": "restart_after_checkpoint", "commit": "abc1234"}}

        class CautiousProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                if self.calls == 1:
                    return ModelTurn(tool_calls=[ToolCall("promote_self_improvement", {})], completion_tokens=1, usage_tokens=1)
                return ModelTurn(text=json.dumps({"status": "BLOCKED", "summary": "promoted but unsure about the next step", "evidence": ["promote"], "actions": [], "changes": [], "tests": [], "blocker": "uncertain about the next bounded task", "next_hypothesis": ""}), completion_tokens=1, usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            runner = ReActRunner(CautiousProvider(), store, {"promote_self_improvement": DeferredRestartTool()}, ReActConfig())
            result = runner.run(StartEnvelope("test"), "system")
            self.assertEqual(result.failure, "deferred restart")
            self.assertEqual(result.status, RunStatus.COMPLETED)
            self.assertEqual(result.control_action["type"], "restart_after_checkpoint")
            store.close()
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            runner = ReActRunner(MagicMock(), store, {}, ReActConfig(tool_result_max_chars=50))
            result = runner._bounded_result({"ok": True, "output": "x" * 500})
            self.assertTrue(result["truncated"])
            self.assertTrue(result["ok"])
            self.assertLessEqual(len(cast(str, result["preview"])), 50)
            self.assertEqual(runner._bounded_result({"ok": True}), {"ok": True})
            store.close()

    def test_tail_start_never_begins_with_a_tool_result(self) -> None:
        import random

        rng = random.Random(2026)
        for _ in range(200):
            messages: list[dict[str, object]] = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
            for index in range(rng.randrange(0, 12)):
                messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{index}"}]})
                if rng.random() < 0.9:
                    messages.append({"role": "tool", "tool_call_id": f"c{index}", "content": "r"})
            for keep in (0, 1, 2, 5, 8, 100):
                start = ReActRunner._tail_start(messages, keep)
                self.assertGreaterEqual(start, 2)
                self.assertLessEqual(start, len(messages))
                if start < len(messages):
                    self.assertNotEqual(messages[start].get("role"), "tool")

    def test_request_token_estimate_is_monotonic_and_bounded(self) -> None:
        self.assertGreaterEqual(ReActRunner._request_tokens([], []), 1)
        base = [{"role": "user", "content": "hello"}]
        previous = ReActRunner._request_tokens(base, [])
        for size in (10, 100, 1000):
            grown = [*base, {"role": "tool", "content": "y" * size}]
            current = ReActRunner._request_tokens(grown, [])
            self.assertGreaterEqual(current, previous)
            previous = current

    def test_prose_finish_report_is_recoverable_not_completed(self) -> None:
        class ProseProvider:
            def complete(self, messages, *, max_tokens, tools=()):
                return ModelTurn(text="I have finished thinking about it.", usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            result = ReActRunner(ProseProvider(), store, {}, ReActConfig(provider_retries=0)).run(StartEnvelope("test"), "system")
            self.assertEqual(result.status, RunStatus.NEEDS_RECOVERY)
            self.assertIn("Finish Report invalid", result.report)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='finish_invalid'").fetchone()[0], 1)
            store.close()


class _RegisteringTool:
    name = "pg_tool"
    capability_kind = "write"
    timeout_seconds = 1.0

    def __init__(self, pgid: int | None = None, exc: BaseException | None = None) -> None:
        self.pgid = pgid
        self.exc = exc

    @property
    def schema(self) -> dict[str, object]:
        return {"type": "function", "function": {"name": self.name, "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        if self.pgid is not None:
            react.register_process_group(self.pgid)
        if self.exc is not None:
            raise self.exc
        return {"ok": True}


class _BashLikeTool:
    name = "bash"
    capability_kind = "write"
    timeout_seconds = 1.0

    def __init__(self, cwd: Path, command: str) -> None:
        self.cwd = cwd
        self.command = command

    @property
    def schema(self) -> dict[str, object]:
        return {"type": "function", "function": {"name": self.name, "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        completed = subprocess.run(["/bin/bash", "-lc", self.command], cwd=str(self.cwd), capture_output=True, text=True, check=False)
        return {"ok": completed.returncode == 0, "stdout": completed.stdout, "stderr": completed.stderr}


class _ToolThenFinishProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, max_tokens, tools=()):
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(tool_calls=[ToolCall("bash", {})], usage_tokens=1)
        return ModelTurn(text=FINISH_OK, usage_tokens=1)


class ProcessGroupTests(unittest.TestCase):
    def _runner(self, directory: str) -> ReActRunner:
        return ReActRunner(MagicMock(), StateStore(Path(directory) / "state.sqlite3"), {}, ReActConfig())

    @staticmethod
    def _wait_dead(process: subprocess.Popen[bytes], timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)

    def test_registered_group_is_killed_when_a_baseexception_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = self._runner(directory)
            process = subprocess.Popen(["sleep", "60"], start_new_session=True)
            try:
                with self.assertRaises(WatchdogTimeout):
                    runner._execute_tool_with_retry(_RegisteringTool(os.getpgid(process.pid), WatchdogTimeout()), {}, "key", "run", 0)
                self._wait_dead(process)
                self.assertIsNotNone(process.poll(), "registered process group was not killed")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                runner.store.close()

    def test_no_registration_is_a_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = self._runner(directory)
            process = subprocess.Popen(["sleep", "60"], start_new_session=True)
            try:
                with self.assertRaises(WatchdogTimeout):
                    runner._execute_tool_with_retry(_RegisteringTool(None, WatchdogTimeout()), {}, "key", "run", 0)
                self.assertIsNone(process.poll(), "unregistered process must not be touched")
            finally:
                process.kill()
                process.wait()
                runner.store.close()

    def test_successful_call_kills_nothing_even_when_registered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = self._runner(directory)
            process = subprocess.Popen(["sleep", "60"], start_new_session=True)
            try:
                result = runner._execute_tool_with_retry(_RegisteringTool(os.getpgid(process.pid), None), {}, "key", "run", 0)
                self.assertEqual(result, {"ok": True})
                self.assertIsNone(process.poll(), "a successful call must not kill a live child")
            finally:
                process.kill()
                process.wait()
                runner.store.close()

    def test_original_exception_is_never_masked_by_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = self._runner(directory)
            try:
                # A bogus pgid exercises the swallowed-error cleanup path.
                with self.assertRaises(WatchdogTimeout):
                    runner._execute_tool_with_retry(_RegisteringTool(2**31 - 1, WatchdogTimeout()), {}, "key", "run", 0)
            finally:
                runner.store.close()

    def test_registry_is_a_noop_outside_a_tracked_call(self) -> None:
        react.register_process_group(4242)
        with react.tracked_process_group() as groups:
            react.register_process_group(4242)
            self.assertEqual(groups, {4242})


class WorktreeWarningTests(unittest.TestCase):
    def _run_bash(self, root: Path, command: str) -> tuple[StateStore, ReActRunner]:
        store = StateStore(root / "state" / "state.sqlite3")
        runner = ReActRunner(
            _ToolThenFinishProvider(),
            store,
            {"bash": _BashLikeTool(root, command)},
            ReActConfig(),
            worktree_root=root,
        )
        runner.run(StartEnvelope("test"), "system")
        return store, runner

    def _tool_result(self, store: StateStore) -> dict[str, object]:
        row = store.connection.execute("SELECT payload FROM event_log WHERE kind='tool_result'").fetchone()
        self.assertIsNotNone(row)
        return json.loads(row[0])["result"]

    def test_modified_tracked_file_emits_event_and_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "tracked.txt").write_text("clean\n", encoding="utf-8")
            commit_all(root, "seed")
            store, _runner = self._run_bash(root, "printf 'dirty\\n' > tracked.txt")
            try:
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='worktree_dirty_after_bash'").fetchone()[0], 1)
                result = self._tool_result(store)
                self.assertIn("worktree_warning", result)
                self.assertIn("tracked.txt", str(result["worktree_warning"]))
            finally:
                store.close()

    def test_clean_repo_emits_neither_event_nor_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "tracked.txt").write_text("clean\n", encoding="utf-8")
            commit_all(root, "seed")
            store, _runner = self._run_bash(root, "true")
            try:
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='worktree_dirty_after_bash'").fetchone()[0], 0)
                self.assertNotIn("worktree_warning", self._tool_result(store))
            finally:
                store.close()

    def test_git_failure_is_silent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, _runner = self._run_bash(root, "true")
            try:
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='worktree_dirty_after_bash'").fetchone()[0], 0)
                self.assertNotIn("worktree_warning", self._tool_result(store))
            finally:
                store.close()
