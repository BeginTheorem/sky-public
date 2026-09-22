"""Split from the former monolithic CoreTests suite."""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import Mock, patch

from helpers import FakeProvider, FixtureTool, commit_all, git_repo

from skynet import metrics
from skynet.heartbeat import wake as heartbeat_wake
from skynet.models import AgentRunResult, Budget, LifecycleState, ModelTurn, RunStatus, ToolCall
from skynet.outbox import deliver_to_jsonl
from skynet.planner import PortfolioPlanner
from skynet.reactor import SYSTEM, SYSTEM_MEMORY, SYSTEM_REACT, Reactor, ReactorConfig, TurnShapeRecorder, mcp_senses_line
from skynet.store import StateStore


class CoreTests(unittest.TestCase):
    def test_system_requires_goal_genesis_and_identity(self) -> None:
        self.assertIn("harness owns lifecycle", SYSTEM)
        self.assertIn("evidence-backed bottleneck", SYSTEM)
        self.assertIn("Initial, Common, and Finish", SYSTEM_REACT)
        self.assertIn("propose_self_improvement", SYSTEM_REACT)
        self.assertIn("ANTI-LOOP PROTOCOL", SYSTEM_REACT)
        self.assertIn("Never repeat a read-only command", SYSTEM_REACT)
        self.assertIn("status BLOCKED", SYSTEM_REACT)
        self.assertIn("Memory Loop", SYSTEM_MEMORY)
        self.assertIn("JSON contract", SYSTEM_MEMORY)
    def test_react_runner_bounds_follow_the_start_envelope_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(
                FakeProvider(),
                {},
                ReactorConfig(
                    state_path=root / "state.sqlite3",
                    budget=Budget(steps=7, tokens=1_234, seconds=12.5, output_tokens=321),
                ),
            )
            config = reactor.runner.config
            self.assertEqual((config.max_steps, config.max_tokens, config.timeout_seconds, config.output_tokens), (7, 1_234, 12.5, 321))
            reactor.close()

    def test_planner_output_budget_comes_from_its_own_setting(self) -> None:
        # A min(4096, ...) clamp at the planner seam meant raising
        # SKYNET_OUTPUT_TOKENS never reached the planner. The ReAct cap stays
        # small here while the planner keeps its own, larger ceiling.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(
                FakeProvider(),
                {},
                ReactorConfig(
                    state_path=root / "state.sqlite3",
                    budget=Budget(steps=7, tokens=1_234, seconds=12.5, output_tokens=100),
                    planner_output_tokens=16_384,
                ),
            )
            self.assertEqual(reactor.runner.config.output_tokens, 100)
            self.assertEqual(reactor.autonomous_planner.output_tokens, 16_384)
            reactor.close()

    def test_soul_loading_does_not_depend_on_prompt_substring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(
                FakeProvider(),
                {},
                ReactorConfig(state_path=root / "state.sqlite3", system_prompt="custom prompt mentioning SOUL.md"),
            )
            self.assertIn("[SKYNET SOUL BEGIN]", reactor.config.system_prompt)
            self.assertIn("# SOUL.md", reactor.config.system_prompt)
            reactor.close()

    def test_system_prompt_carries_workspace_and_senses_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=root / "state.sqlite3"))
            prompt = reactor.config.system_prompt
            self.assertIn("[SKYNET WORKSPACE BEGIN]", prompt)
            self.assertIn("WORKSPACE: ", prompt)
            self.assertIn("SENSES: none configured", prompt)
            reactor.close()

    def test_mcp_senses_line_groups_tools_by_server(self) -> None:
        arxiv_client = Mock()
        arxiv_client.server_name = "arxiv"
        search = Mock()
        search.client = arxiv_client
        search.remote_name = "search_papers"
        abstract = Mock()
        abstract.client = arxiv_client
        abstract.remote_name = "get_abstract"
        web = Mock()
        web.client = Mock(server_name="ddg")
        web.remote_name = "search"
        tools = cast(Any, {"search_papers": search, "get_abstract": abstract, "search": web, "bash": Mock(spec=[])})
        self.assertEqual(mcp_senses_line(tools), "SENSES: arxiv (get_abstract, search_papers); ddg (search)")

    def test_system_has_no_hardcoded_workspace_or_senses(self) -> None:
        self.assertNotIn("/home/", SYSTEM)
        self.assertIn("SENSES line", SYSTEM)
        self.assertIn("WORKSPACE line", SYSTEM)
        for leaked in ("search_papers", "search_code", "browser_", "playwright", "arxiv-mcp"):
            self.assertNotIn(leaked, SYSTEM)
    def test_blocked_goal_does_not_trigger_bootstrap_genesis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            from skynet.store import StateStore
            store = StateStore(Path(directory) / "state.sqlite3")
            store.add_goal("already blocked", priority=1.0)
            store.connection.execute("UPDATE goals SET status='blocked'")
            store.connection.commit()
            store.close()
            reactor = Reactor(Mock(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            self.assertIsNone(reactor.tick("test"))
            goals = reactor.store.connection.execute("SELECT title, status FROM goals ORDER BY created_at").fetchall()
            self.assertEqual([(row[0], row[1]) for row in goals], [("already blocked", "blocked")])
            reactor.store.close()
    def test_autonomous_scheduler_plans_after_exhausted_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("blocked autonomous work", priority=1.0)
            task_id = store.add_task("blocked task", goal_id)
            store.connection.execute("UPDATE tasks SET status='blocked' WHERE task_id=?", (task_id,))
            store.connection.commit()
            store.close()

            reactor = Reactor(Mock(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            reactor.tick("timer")
            fallback = reactor.store.connection.execute(
                "SELECT title, status FROM tasks WHERE title LIKE 'Diagnose one concrete SkyNet bottleneck%'"
            ).fetchone()
            self.assertIsNotNone(fallback)
            self.assertIn(fallback["status"], {"pending", "blocked"})
            reactor.store.close()
    def test_sleep_without_work_schedules_next_wake(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            goal_id = store.add_goal("blocked autonomous work", priority=1.0)
            task_id = store.add_task("blocked task", goal_id)
            store.connection.execute("UPDATE tasks SET status='blocked' WHERE task_id=?", (task_id,))
            store.connection.commit()
            store.close()

            reactor = Reactor(Mock(), {}, ReactorConfig(state_path=path, wake_interval_seconds=60.0))
            before = datetime.now(UTC)
            reactor._sleep_without_work(reactor.store.state(), "test idle")
            next_wake_value = reactor.store.state().next_wake_at
            self.assertIsNotNone(next_wake_value)
            next_wake = datetime.fromisoformat(str(next_wake_value).replace("Z", "+00:00"))
            self.assertGreaterEqual(next_wake, before + timedelta(seconds=59))
            self.assertLessEqual(next_wake, datetime.now(UTC) + timedelta(seconds=61))
            self.assertEqual(reactor.store.state().lifecycle.value, "sleep")
            reactor.store.close()
    def test_exhausted_seed_tasks_do_not_create_numbered_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            goal_id = store.add_goal("continuous improvement", priority=1.0)
            task_id = store.add_task("blocked task", goal_id)
            store.connection.execute("UPDATE tasks SET status='blocked' WHERE task_id=?", (task_id,))
            store.connection.commit()
            store.close()

            reactor = Reactor(Mock(), {}, ReactorConfig(state_path=path))
            self.assertEqual(reactor.store.active_work()[1], [])
            reactor.store.close()
    def test_default_capabilities_include_shell_and_research_tools(self) -> None:
        from skynet.tools import default_tools

        tools = default_tools()
        self.assertEqual(set(tools), {"bash", "webfetch", "read", "grep", "db", "structure"})
        description = cast(str, tools["bash"].schema["function"]["description"])
        self.assertIn("system bash", description)
        self.assertIn("webfetch", tools)
    def test_reactor_persists_checkpoint_and_tool_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(
                FakeProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=Path(directory) / "state.sqlite3"),
            )
            self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
            state = reactor.store.state()
            self.assertEqual(state.generation, 1)
            self.assertEqual(state.retry_count, 0)
            self.assertIsNone(state.active_run_id)
            # Ordered explicitly: an index on event_log(kind, sequence) can make
            # the planner return rows in index order instead of insertion order.
            events = reactor.store.connection.execute("SELECT kind FROM event_log ORDER BY sequence").fetchall()
            self.assertEqual([row[0] for row in events], ["planner_decision", "memory_retrieval", "run_started", "decision_record", "react_phase", "react_phase", "tool_call", "tool_result", "finish_report", "run_result_committed", "provider_request", "memory_loop_started", "scheduler_decision", "memory_loop_finished", "memory_event", "memory_consolidated", "run_finished", "plan_observation", "task_progress", "checkpoint", "metrics_snapshot"])
            response = reactor.store.connection.execute("SELECT payload FROM event_log WHERE kind='finish_report'").fetchone()[0]
            self.assertIn('"text":', response)
            self.assertEqual(reactor.store.connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 1)
            self.assertIn("initial_prompt", reactor.store.state().next_plan)
            self.assertEqual(reactor.store.connection.execute("SELECT COUNT(*) FROM capability_effects").fetchone()[0], 1)
            self.assertEqual(reactor.store.connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0], 2)
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM evaluations WHERE status='metrics'").fetchone()[0],
                1,
            )
            self.assertEqual(reactor.store.connection.execute("SELECT COUNT(*) FROM run_results").fetchone()[0], 1)
            self.assertEqual(reactor.store.connection.execute("SELECT status FROM tasks").fetchone()[0], "completed")
            decision = json.loads(reactor.store.connection.execute("SELECT payload FROM event_log WHERE kind='decision_record'").fetchone()[0])
            self.assertIn("intention", decision)
            self.assertEqual(reactor.store.connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0], 1)
            self.assertEqual(reactor.store.connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 1)
            self.assertEqual(len(reactor.store.pending_outbox()), 1)
            self.assertEqual(deliver_to_jsonl(reactor.store, Path(directory) / "outbox.jsonl"), 1)
            self.assertEqual(len(reactor.store.pending_outbox()), 0)
            reactor.close()
    def test_mid_run_progress_reaches_the_owner_outbox_as_agent_message(self) -> None:
        from skynet.outbox import render_outbox_message

        class SlowProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                if self.calls == 1:
                    return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], usage_tokens=1, model_seconds=4000.0)
                if self.calls == 2:
                    return ModelTurn(text=json.dumps({
                        "status": "COMPLETED",
                        "summary": "bounded episode with a mid-run note",
                        "evidence": ["fixture tool result"],
                        "actions": [],
                        "changes": [],
                        "tests": [],
                        "blocker": "",
                        "next_hypothesis": "",
                    }), usage_tokens=1)
                return ModelTurn(text=json.dumps({
                    "memory_candidates": [],
                    "next_plan": {"next": "continue"},
                    "initial_prompt": "continue",
                    "goal_updates": [],
                    "task_updates": [],
                    "evaluation": {},
                }), usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(
                SlowProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(
                    state_path=root / "state.sqlite3",
                    budget=Budget(steps=100, tokens=200_000, seconds=7200.0, output_tokens=8192),
                    run_progress_seconds=3600.0,
                ),
            )
            self.assertEqual(reactor.tick("timer"), RunStatus.COMPLETED)
            pending = reactor.store.pending_outbox()
            progress = [message for message in pending if message["kind"] == "agent_message"]
            finish = [message for message in pending if message["kind"] == "agent_response"]
            self.assertEqual(len(progress), 1, "exactly one mid-run note")
            self.assertEqual(len(finish), 1, "the finish report is still delivered untouched")
            message = progress[0]["payload"]["message"]
            self.assertIn("Прогон продолжается", message)
            self.assertLessEqual(len(message), 300)
            # The bot's drain renders this kind through the shared renderer.
            self.assertIn("Прогон продолжается", render_outbox_message("agent_message", progress[0]["payload"]))
            # The note is durable, not only in the queue.
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='run_progress_reported'").fetchone()[0],
                1,
            )
            reactor.close()
    def test_path_like_claim_in_a_clean_worktree_does_not_reopen_the_task(self) -> None:
        class PrototypeProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                if self.calls == 1:
                    return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], usage_tokens=1)
                return ModelTurn(text=json.dumps({
                    "status": "COMPLETED",
                    "summary": "prototyped a fix in a scratch directory",
                    "evidence": ["fixture tool result"],
                    "actions": [],
                    "changes": ["skynet/foo.py: bounded change"],
                    "tests": [],
                    "blocker": "",
                    "next_hypothesis": "",
                }), usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(
                PrototypeProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root),
            )
            self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
            # The claim is not verifiable here (the root is not even a git
            # repository, so nothing is dirty), so the finished task stays
            # completed and no uncaptured-changes signal is raised.
            self.assertEqual(reactor.store.connection.execute("SELECT status FROM tasks").fetchone()[0], "completed")
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='uncaptured_changes'").fetchone()[0],
                0,
            )
            reactor.close()

    def test_dirty_worktree_with_uncaptured_changes_reopens_the_task(self) -> None:
        class PrototypeProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                if self.calls == 1:
                    return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], usage_tokens=1)
                return ModelTurn(text=json.dumps({
                    "status": "COMPLETED",
                    "summary": "edited the repository directly instead of proposing",
                    "evidence": ["fixture tool result"],
                    "actions": [],
                    "changes": ["skynet/foo.py: bounded change"],
                    "tests": [],
                    "blocker": "",
                    "next_hypothesis": "",
                }), usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            commit_all(root, "base")
            # A direct edit through bash leaves the main worktree dirty; the
            # work is real and not captured, so the task must stay in the queue.
            (root / "module.py").write_text("value = 2\n", encoding="utf-8")
            reactor = Reactor(
                PrototypeProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root),
            )
            self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
            self.assertEqual(reactor.store.connection.execute("SELECT status FROM tasks").fetchone()[0], "pending")
            row = reactor.store.connection.execute("SELECT payload FROM event_log WHERE kind='uncaptured_changes'").fetchone()
            self.assertIsNotNone(row)
            self.assertIs(json.loads(row[0])["dirty_worktree"], True)
            reactor.close()

    def test_prose_denial_in_changes_does_not_reopen_the_task(self) -> None:
        class DenialProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                if self.calls == 1:
                    return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], usage_tokens=1)
                return ModelTurn(text=json.dumps({
                    "status": "COMPLETED",
                    "summary": "read-only investigation, nothing durable changed",
                    "evidence": ["fixture tool result"],
                    "actions": [],
                    "changes": ["No durable code change: propose_self_improvement was not called this episode."],
                    "tests": [],
                    "blocker": "",
                    "next_hypothesis": "",
                }), usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(
                DenialProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root),
            )
            self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
            self.assertEqual(reactor.store.connection.execute("SELECT status FROM tasks").fetchone()[0], "completed")
            self.assertEqual(reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='uncaptured_changes'").fetchone()[0], 0)
            reactor.close()

    def test_captured_changes_complete_the_task(self) -> None:
        class ProposalTool(FixtureTool):
            name = "propose_self_improvement"
            schema = {"type": "function", "function": {"name": name, "description": "fixture", "parameters": {"type": "object", "additionalProperties": False}}}  # noqa: RUF012 - Tool protocol reads schema as an instance property

        class CapturedProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                if self.calls == 1:
                    return ModelTurn(tool_calls=[ToolCall("propose_self_improvement", {})], usage_tokens=1)
                return ModelTurn(text=json.dumps({
                    "status": "COMPLETED",
                    "summary": "proposal registered and validated",
                    "evidence": ["proposal tool result"],
                    "actions": [],
                    "changes": ["skynet/foo.py: bounded change"],
                    "tests": [],
                    "blocker": "",
                    "next_hypothesis": "",
                }), usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(
                CapturedProvider(),
                {"propose_self_improvement": ProposalTool()},
                ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root),
            )
            self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
            self.assertEqual(reactor.store.connection.execute("SELECT status FROM tasks").fetchone()[0], "completed")
            self.assertEqual(reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='uncaptured_changes'").fetchone()[0], 0)
            reactor.close()

    def test_reactor_value_estimate_is_zero_without_tool_evidence(self) -> None:
        class CompletedProvider(FakeProvider):
            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                return ModelTurn(text=json.dumps({"status": "completed", "summary": "finished"}), usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory, patch.object(Reactor, "_success_criteria", return_value=[
            {"criterion": "required verified progress", "kind": "verified_progress", "required": True},
            {"criterion": "optional verified progress", "kind": "verified_progress", "required": False},
        ]):
            reactor = Reactor(
                CompletedProvider(),
                {},
                ReactorConfig(state_path=Path(directory) / "state.sqlite3"),
            )
            self.assertEqual(reactor.tick("optional-evaluation-test"), RunStatus.COMPLETED)
            evaluation = json.loads(reactor.store.connection.execute("SELECT evaluation FROM evaluations").fetchone()[0])
            self.assertEqual(evaluation["success_criteria_results"][0]["required"], True)
            self.assertEqual(evaluation["success_criteria_results"][1]["required"], False)
            self.assertEqual(evaluation["success_criteria_results"][1]["passed"], False)
            self.assertEqual(evaluation["value_estimate"], 0.0)
            reactor.close()
    def test_reactor_value_estimate_reflects_required_criteria(self) -> None:
        class CompletedProvider(FakeProvider):
            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                return ModelTurn(text=json.dumps({"status": "completed", "summary": "finished"}), usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory, patch.object(Reactor, "_success_criteria", return_value=[
            {"criterion": "required marker", "kind": "report_contains", "text": "missing", "required": True},
        ]):
            reactor = Reactor(
                CompletedProvider(),
                {},
                ReactorConfig(state_path=Path(directory) / "state.sqlite3"),
            )
            self.assertEqual(reactor.tick("evaluation-test"), RunStatus.COMPLETED)
            evaluation = json.loads(reactor.store.connection.execute("SELECT evaluation FROM evaluations").fetchone()[0])
            self.assertEqual(evaluation["success_criteria_results"][0]["passed"], False)
            self.assertEqual(evaluation["value_estimate"], 0.0)
            reactor.close()
    def test_reactor_persists_evaluation_record(self) -> None:
        class EvaluatingProvider(FakeProvider):
            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                if self.calls == 1:
                    return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], usage_tokens=1)
                return ModelTurn(text=json.dumps({
                    "status": "completed",
                    "summary": "created baseline",
                }), usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(
                EvaluatingProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=Path(directory) / "state.sqlite3"),
            )
            self.assertEqual(reactor.tick("artifact-test"), RunStatus.COMPLETED)
            evaluation = reactor.store.connection.execute("SELECT evaluation FROM evaluations").fetchone()[0]
            self.assertIn("harness_technical_status", evaluation)
            self.assertIn("success_criteria_results", evaluation)
            reactor.close()
    def test_reactor_exception_clears_active_run_for_recovery(self) -> None:
        class BrokenProvider:
            def complete(self, messages, *, max_tokens, tools=()):
                raise RuntimeError("provider exploded")

        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(BrokenProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            self.assertEqual(reactor.tick("failure-test"), RunStatus.NEEDS_RECOVERY)
            self.assertIsNone(reactor.store.state().active_run_id)
            self.assertEqual(reactor.store.connection.execute("SELECT status FROM runs").fetchone()[0], "needs_recovery")
            reactor.close()
    def test_reactor_failure_between_run_and_consolidation_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {"fixture_tool": FixtureTool()}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            original = reactor.store.consolidate_versioned
            def fail_once(*args, **kwargs):
                reactor.store.consolidate_versioned = original
                raise RuntimeError("consolidation failure")
            reactor.store.consolidate_versioned = fail_once
            with self.assertRaises(RuntimeError):
                reactor.tick("chaos")
            self.assertIsNone(reactor.store.state().active_run_id)
            reactor.close()
    def test_empty_state_bootstraps_autonomous_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {"fixture_tool": FixtureTool()}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            reactor.tick("startup")
            goal = reactor.store.connection.execute("SELECT title FROM goals LIMIT 1").fetchone()
            self.assertIsNotNone(goal)
            self.assertEqual(reactor.store.connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0], 1)
            reactor.close()
    def test_tick_prunes_history_beyond_the_retention_window(self) -> None:
        from skynet.models import Budget, RunRecord
        from skynet.time import utc_now

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            reactor = Reactor(
                FakeProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=path, transcript_retention_runs=1),
            )
            reactor.store.create_run(RunRecord("old-run", 1, RunStatus.COMPLETED, utc_now(), Budget()))
            reactor.store.append_transcript("react_history", {"messages": ["old"]}, "old-run")
            reactor.store.snapshot_episode("old-run")

            self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
            self.assertEqual(reactor.store.connection.execute("SELECT COUNT(*) FROM transcript WHERE run_id='old-run'").fetchone()[0], 0)
            self.assertEqual(reactor.store.connection.execute("SELECT COUNT(*) FROM episode_snapshots WHERE run_id='old-run'").fetchone()[0], 0)
            pruned = reactor.store.connection.execute("SELECT payload FROM event_log WHERE kind='history_pruned'").fetchone()
            self.assertIsNotNone(pruned)
            reactor.close()

    def test_tick_includes_durable_work_without_stealing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {"fixture_tool": FixtureTool()}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            goal_id = reactor.store.add_goal("test autonomy", priority=1)
            reactor.store.add_task("inspect state", goal_id=goal_id)
            reactor.store.add_inbox_event("event-1", "user_message", {"text": "continue"})
            reactor.tick("event")
            # The unrelated run observes the operator message but must not
            # consume it; only the inbox task it belongs to may do that.
            self.assertEqual([event["event_id"] for event in reactor.store.pending_inbox()], ["event-1"])
            self.assertEqual(reactor.store.active_work()[0][0]["goal_id"], goal_id)
            selected = PortfolioPlanner(reactor.store).select()
            self.assertIsNone(selected)
            started = reactor.store.connection.execute(
                "SELECT payload FROM event_log WHERE kind='run_started' ORDER BY sequence LIMIT 1"
            ).fetchone()
            self.assertIn("event-1", started["payload"])
            reactor.close()
    def test_reactor_sleeps_when_pending_task_fingerprint_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            goal_id = store.add_goal("portfolio", priority=4)
            store.add_task("already investigated", goal_id, expected_new_fact="fact", hypothesis_fingerprint="done", structural_fingerprint="done-shape")
            store.mark_hypothesis("done", "completed", {"evidence": "existing"})
            store.close()
            reactor = Reactor(Mock(), {}, ReactorConfig(state_path=path))
            reactor.tick("timer")
            self.assertEqual(
                reactor.store.connection.execute(
                    "SELECT COUNT(*) FROM tasks WHERE title LIKE 'Diagnose one concrete SkyNet bottleneck%'"
                ).fetchone()[0],
                1,
            )
            reactor.close()
    def test_lifecycle_transition_is_durable_and_journaled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                state = store.state()
                store.transition(state, LifecycleState.SLEEP, reason="test transition", record_event=True)
            self.assertEqual(store.state().lifecycle, LifecycleState.SLEEP)
            row = store.connection.execute(
                "SELECT payload FROM event_log WHERE kind='lifecycle_transition'"
            ).fetchone()
            self.assertIn('"to": "sleep"', row[0])
            store.close()
    def test_next_start_includes_relevant_memory_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            class CaptureProvider(FakeProvider):
                start_messages: ClassVar[list[str]] = []

                def complete(self, messages, *, max_tokens, tools=()):
                    if not self.start_messages:
                        self.start_messages.append(messages[1]["content"])
                    return super().complete(messages, max_tokens=max_tokens, tools=tools)

            provider = CaptureProvider()
            reactor = Reactor(provider, {"fixture_tool": FixtureTool()}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            reactor.store.consolidate("seed", [{"kind": "fact", "content": "gateway recovery requires bounded retry", "confidence": 0.9}])
            reactor.store.add_inbox_event("memory-event", "user_message", {"text": "gateway recovery"})
            reactor.tick("memory-test")
            start_content = provider.start_messages[0]
            start = json.loads(start_content.removeprefix("START ENVELOPE\n").removesuffix("\nEND START ENVELOPE"))
            memory = [item for item in start["observations"] if item["kind"] == "memory_context"]
            self.assertEqual(len(memory), 1)
            self.assertIn("gateway recovery", memory[0]["items"][0]["content"])
            reactor.close()
    def test_goal_and_task_updates_emit_progress_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                goal_id = store.add_goal("measure progress")
                task_id = store.add_task("collect evidence", goal_id=goal_id)
                store.increment_task_attempts(task_id)
                store.apply_goal_updates([{"goal_id": goal_id, "status": "completed", "outcome": {"evidence": "ok"}}], run_id="run-1")
                store.apply_task_updates([{"task_id": task_id, "status": "completed"}], run_id="run-1")
            kinds = [row[0] for row in store.connection.execute("SELECT kind FROM event_log ORDER BY sequence")]
            self.assertEqual(kinds, ["goal_progress", "task_progress"])
            task_event = store.connection.execute("SELECT payload FROM event_log WHERE kind='task_progress'").fetchone()
            self.assertEqual(json.loads(task_event[0])["attempts"], 1)
            store.close()
    def test_criteria_evaluation_requires_successful_tool_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                store.append_event("tool_result", {"call": {"tool_name": "clock"}, "result": {"ok": True}}, "run-1")
            results = store.evaluate_criteria("run-1", [{"criterion": "verified progress", "kind": "verified_progress"}], "done", RunStatus.COMPLETED)
            self.assertEqual([item["passed"] for item in results], [True])
            self.assertEqual(results[0]["kind"], "verified_progress")
            store.close()
    def test_verified_progress_does_not_accept_report_without_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            results = store.evaluate_criteria(
                "run-1",
                [{"criterion": "verified work", "kind": "verified_progress", "required": True}],
                "I completed the work.",
                RunStatus.COMPLETED,
            )
            self.assertFalse(results[0]["passed"])
            self.assertTrue(results[0]["verified"])
            self.assertEqual(results[0]["evidence"], [])
            store.close()
    def test_heartbeat_does_not_create_a_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {"fixture_tool": FixtureTool()}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            state = reactor.store.state()
            state.active_run_id = "already-running"
            reactor.store.set_state(state)
            self.assertEqual(heartbeat_wake(reactor, "timer"), RunStatus.RUNNING)
            reactor.close()
    def test_reactor_registers_self_improvement_for_git_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "README.md").write_text("one\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "__init__.py").write_text("\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            reactor = Reactor(
                FakeProvider(),
                {},
                ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root),
            )
            try:
                self.assertIn("propose_self_improvement", reactor.runner.tools)
                # Promotion is folded into propose_self_improvement; the separate
                # promote tool was removed.
                self.assertNotIn("promote_self_improvement", reactor.runner.tools)
            finally:
                reactor.close()

    def test_process_lock_is_exclusive_and_releasable(self) -> None:
        from skynet.lock import ProcessLock

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skynet.lock"
            first = ProcessLock(path)
            first.acquire()
            second = ProcessLock(path)
            with self.assertRaisesRegex(RuntimeError, "owns the lock"):
                second.acquire()
            first.release()
            second.acquire()
            second.release()

    def test_lifecycle_tick_does_not_leak_threads_or_descriptors(self) -> None:
        from helpers import count_open_fds, wait_for_fd_count, wait_for_threads_to_finish

        baseline_threads = {thread.name for thread in threading.enumerate()}
        baseline_fds = count_open_fds()
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {"fixture_tool": FixtureTool()}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            reactor.tick("test")
            reactor.close()
        lingering = {thread.name for thread in threading.enumerate()} - baseline_threads
        self.assertEqual(wait_for_threads_to_finish(lingering), set())
        self.assertLessEqual(wait_for_fd_count(baseline_fds), baseline_fds)

    def test_daily_maintenance_runs_with_metrics_disabled_once_per_day(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(
                FakeProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=root / "state.sqlite3", metrics_snapshot_enabled=False),
            )
            stale = (datetime.now(UTC) - timedelta(days=30)).isoformat().replace("+00:00", "Z")

            def insert_stale(memory_id: str) -> None:
                reactor.store.connection.execute(
                    "INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (memory_id, "fact", f"stale gateway fact {memory_id}", 0.9, "seed", stale),
                )
                reactor.store.connection.commit()

            def decay_events() -> int:
                return reactor.store.connection.execute(
                    "SELECT COUNT(*) FROM event_log WHERE kind='memory_confidence_decayed'"
                ).fetchone()[0]

            insert_stale("stale-1")
            reactor.tick("test")
            self.assertEqual(decay_events(), 1)
            # The metrics snapshot is still gated by its own flag.
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM evaluations WHERE status='metrics'").fetchone()[0],
                0,
            )
            marker = json.loads((root / "maintenance-marker.json").read_text(encoding="utf-8"))
            self.assertEqual(marker["marker"], metrics.daily_marker())
            # A same-day second tick must not run maintenance again: a freshly
            # seeded stale memory keeps its confidence because the pass is skipped.
            insert_stale("stale-2")
            reactor.tick("test")
            self.assertEqual(decay_events(), 1)
            self.assertEqual(
                reactor.store.connection.execute("SELECT confidence FROM memories WHERE memory_id='stale-2'").fetchone()[0],
                0.9,
            )
            reactor.close()

    def test_daily_maintenance_compacts_legacy_snapshots_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(
                FakeProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=root / "state.sqlite3", metrics_snapshot_enabled=False),
            )
            store = reactor.store
            store.append_transcript("provider_request", {"messages": []}, "run-legacy")
            rebuilt = store._snapshot_transcript("run-legacy")
            events = [{"sequence": 1, "kind": "tool_result", "payload": {"ok": True}}]
            legacy = json.dumps({"events": events, "transcript": rebuilt}, ensure_ascii=False, sort_keys=True)
            store.connection.execute(
                "INSERT INTO episode_snapshots(snapshot_id, run_id, first_sequence, last_sequence, payload_hash, payload, created_at) "
                "VALUES ('legacy', 'run-legacy', 1, 1, 'old', ?, '2026-01-01T00:00:00Z')",
                (legacy,),
            )
            store.connection.commit()

            def compact_events() -> int:
                return store.connection.execute(
                    "SELECT COUNT(*) FROM event_log WHERE kind='history_compacted'"
                ).fetchone()[0]

            reactor._run_daily_maintenance()
            stored = json.loads(
                store.connection.execute(
                    "SELECT payload FROM episode_snapshots WHERE run_id='run-legacy'"
                ).fetchone()[0]
            )
            self.assertEqual(stored["transcript"], {"rebuilt_from": "transcript", "rows_at_freeze": len(rebuilt)})
            self.assertEqual(stored["events"], events, "events are not rebuildable and must be untouched")
            # The read path rebuilds the same projection the legacy row froze.
            self.assertEqual(store.snapshot_episode("run-legacy")["transcript"], rebuilt)
            self.assertEqual(compact_events(), 1)

            reactor._run_daily_maintenance()
            self.assertEqual(compact_events(), 1, "a second maintenance pass rewrites nothing and logs nothing")
            self.assertEqual(store.snapshot_episode("run-legacy")["transcript"], rebuilt)
            reactor.close()

    def test_memory_injection_respects_configured_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            class CaptureProvider(FakeProvider):
                def __init__(self) -> None:
                    super().__init__()
                    self.start_messages: list[str] = []

                def complete(self, messages, *, max_tokens, tools=()):
                    if not self.start_messages:
                        self.start_messages.append(messages[1]["content"])
                    return super().complete(messages, max_tokens=max_tokens, tools=tools)

            provider = CaptureProvider()
            reactor = Reactor(
                provider,
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=Path(directory) / "state.sqlite3", memory_inject_limit=3),
            )
            reactor.store.consolidate("seed", [
                {"kind": "fact", "content": f"gateway recovery variant {index}", "confidence": 0.9}
                for index in range(5)
            ])
            reactor.store.add_inbox_event("memory-event", "user_message", {"text": "gateway recovery"})
            reactor.tick("memory-test")
            start_content = provider.start_messages[0]
            start = json.loads(start_content.removeprefix("START ENVELOPE\n").removesuffix("\nEND START ENVELOPE"))
            memory = [item for item in start["observations"] if item["kind"] == "memory_context"]
            self.assertEqual(len(memory), 1)
            self.assertEqual(len(memory[0]["items"]), 3)
            reactor.close()

    def test_degraded_memory_loop_appends_a_durable_event(self) -> None:
        class DegradedMemoryProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, *, max_tokens, tools=()):
                self.calls += 1
                if self.calls == 1:
                    return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], usage_tokens=1)
                if self.calls == 2:
                    return ModelTurn(
                        text=json.dumps({"status": "COMPLETED", "summary": "finished the bounded episode", "evidence": ["fixture tool result"], "actions": [], "changes": [], "tests": [], "blocker": "", "next_hypothesis": ""}),
                        usage_tokens=1,
                    )
                return ModelTurn(text="not valid json", usage_tokens=1)

        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(
                DegradedMemoryProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=Path(directory) / "state.sqlite3"),
            )
            self.assertEqual(reactor.tick("degraded-memory-test"), RunStatus.COMPLETED)
            degraded = reactor.store.connection.execute(
                "SELECT run_id, payload FROM event_log WHERE kind='memory_degraded'"
            ).fetchone()
            self.assertIsNotNone(degraded)
            started = reactor.store.connection.execute("SELECT run_id FROM event_log WHERE kind='run_started' LIMIT 1").fetchone()
            self.assertEqual(degraded["run_id"], started["run_id"])
            self.assertTrue(json.loads(degraded["payload"])["error"])
            finished = json.loads(
                reactor.store.connection.execute("SELECT payload FROM event_log WHERE kind='memory_loop_finished'").fetchone()["payload"]
            )
            self.assertTrue(finished["degraded"])
            reactor.close()

    def test_healthy_memory_loop_has_no_degraded_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(
                FakeProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=Path(directory) / "state.sqlite3"),
            )
            self.assertEqual(reactor.tick("healthy-memory-test"), RunStatus.COMPLETED)
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='memory_degraded'").fetchone()[0],
                0,
            )
            finished = json.loads(
                reactor.store.connection.execute("SELECT payload FROM event_log WHERE kind='memory_loop_finished'").fetchone()["payload"]
            )
            self.assertFalse(finished["degraded"])
            reactor.close()

    def test_configured_memory_loop_timeout_reaches_the_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(
                FakeProvider(),
                {},
                ReactorConfig(state_path=Path(directory) / "state.sqlite3", memory_loop_timeout_seconds=42.5),
            )
            self.assertEqual(reactor.memory_loop.timeout_seconds, 42.5)
            self.assertEqual(reactor.memory_loop.budget.seconds, 42.5)
            reactor.close()

    def test_memory_tool_is_registered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(
                FakeProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=Path(directory) / "state.sqlite3"),
            )
            self.assertIn("memory", reactor.runner.tools)
            reactor.close()

    def test_turn_shape_recorder_captures_scalars_not_reasoning_text(self) -> None:
        class ShapeProvider:
            name = "shape"
            accepts_timeout_override = True

            def __init__(self) -> None:
                self.turn = ModelTurn(text="abc", reasoning_content="private thoughts", completion_tokens=9, finish_reason="length")

            def complete(self, messages, *, max_tokens, tools=()):
                return self.turn

        inner = ShapeProvider()
        recorder = TurnShapeRecorder(inner)
        self.assertIsNone(recorder.last_shape)
        self.assertIs(recorder.complete([], max_tokens=1), inner.turn, "the wrapper is a pass-through")
        self.assertEqual(
            recorder.last_shape,
            {"text_chars": 3, "reasoning_chars": len("private thoughts"), "completion_tokens": 9, "finish_reason": "length"},
        )
        self.assertNotIn("private thoughts", str(recorder.last_shape), "reasoning text is never captured")
        # Unknown attributes keep the wrapped provider's capability contract.
        self.assertTrue(recorder.accepts_timeout_override)
        recorder.reset()
        self.assertIsNone(recorder.last_shape)

    def test_deferred_restart_is_not_persisted_as_a_provider_failure(self) -> None:
        # `run_results.failure` is the fault ledger metrics read verbatim; a
        # deferred restart is a control label on a COMPLETED run, not a fault.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(FakeProvider(), {"fixture_tool": FixtureTool()}, ReactorConfig(state_path=root / "state.sqlite3"))
            crafted = AgentRunResult(
                status=RunStatus.COMPLETED,
                report=json.dumps({"status": "COMPLETED", "summary": "promoted"}),
                steps=2,
                usage_tokens=5,
                failure="deferred restart",
                control_action={"type": "restart_after_checkpoint", "commit": "abc1234"},
            )
            with patch.object(Reactor, "_success_criteria", return_value=[]), patch.object(reactor.runner, "run", return_value=crafted):
                self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
            failure = reactor.store.connection.execute("SELECT failure FROM run_results").fetchone()[0]
            self.assertEqual(failure, "", "a COMPLETED deferred restart must not enter the fault ledger")
            committed = json.loads(
                reactor.store.connection.execute("SELECT payload FROM event_log WHERE kind='run_result_committed'").fetchone()[0]
            )
            self.assertEqual(committed["failure"], "deferred restart", "the event keeps the control label")
            reactor.close()

    def test_unverified_action_claims_need_an_instrument_phrase_and_no_ledger_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            report = json.dumps({"summary": "sent the update", "actions": ["Sent the update via send_message_to_user."]})
            findings = reactor._unverified_action_claims("run-1", report)
            self.assertEqual([finding["tool"] for finding in findings], ["send_message_to_user"])
            self.assertEqual(findings[0]["run_id"], "run-1")
            # A ledger row for the tool clears the finding.
            reactor.store.record_effect("run-1:0:call-send", "send_message_to_user", "h", {"ok": True}, "applied")
            self.assertEqual(reactor._unverified_action_claims("run-1", report), [])
            # A bare mention is not a claim; only the instrument phrase counts.
            bare = json.dumps({"summary": "s", "actions": ["Consider send_message_to_user."]})
            self.assertEqual(reactor._unverified_action_claims("run-2", bare), [])
            # A malformed or non-report payload is not a claim.
            self.assertEqual(reactor._unverified_action_claims("run-2", "not json"), [])
            self.assertEqual(reactor._unverified_action_claims("run-2", json.dumps([1, 2])), [])
            reactor.close()
