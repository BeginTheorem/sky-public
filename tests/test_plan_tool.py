"""The A1-lite plan artifact: record_plan, ordering, and cross-cycle carry."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from helpers import FixtureTool

from skynet.models import ModelTurn, RunStatus, ToolCall
from skynet.plan_tool import (
    RecordPlanTool,
    latest_recorded_plan,
    plan_preceded_first_mutation,
)
from skynet.reactor import PLAN_INSTRUCTION, Reactor, ReactorConfig
from skynet.store import StateStore


class _PlanProvider:
    """Records a plan on the first ReAct step, then finishes; memory loop third."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, max_tokens, tools=()):
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(
                tool_calls=[
                    ToolCall(
                        "record_plan",
                        {"plan": "Inspect the invariant, then propose a bounded fix.", "steps": ["inspect", "propose"]},
                    )
                ],
                usage_tokens=1,
            )
        if self.calls == 2:
            return ModelTurn(
                text=json.dumps(
                    {
                        "status": "COMPLETED",
                        "summary": "recorded a plan and inspected",
                        "evidence": ["record_plan result"],
                        "actions": [],
                        "changes": [],
                        "tests": [],
                        "blocker": "",
                        "next_hypothesis": "",
                    }
                ),
                usage_tokens=1,
            )
        return ModelTurn(
            text=json.dumps(
                {
                    "memory_candidates": [{"kind": "fact", "content": "plan carried", "confidence": 0.9}],
                    "next_plan": {"next": "continue from the recorded plan"},
                    "initial_prompt": "continue",
                    "goal_updates": [],
                    "task_updates": [],
                    "evaluation": {},
                }
            ),
            usage_tokens=1,
        )


class _NoPlanProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.system_prompts: list[str] = []

    def complete(self, messages, *, max_tokens, tools=()):
        if not self.system_prompts:
            self.system_prompts.append(messages[0]["content"])
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], usage_tokens=1)
        if self.calls == 2:
            return ModelTurn(
                text=json.dumps(
                    {
                        "status": "COMPLETED",
                        "summary": "read-only investigation",
                        "evidence": ["fixture tool result"],
                        "actions": [],
                        "changes": [],
                        "tests": [],
                        "blocker": "",
                        "next_hypothesis": "",
                    }
                ),
                usage_tokens=1,
            )
        return ModelTurn(
            text=json.dumps(
                {
                    "memory_candidates": [],
                    "next_plan": {"next": "continue"},
                    "initial_prompt": "continue",
                    "goal_updates": [],
                    "task_updates": [],
                    "evaluation": {},
                }
            ),
            usage_tokens=1,
        )


class RecordPlanToolTests(unittest.TestCase):
    def _store(self, directory: str) -> StateStore:
        return StateStore(Path(directory) / "state.sqlite3")

    def test_schema_is_openai_style_and_requires_a_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            tool = RecordPlanTool(store)
            self.assertEqual(tool.name, "record_plan")
            self.assertEqual(tool.capability_kind, "write")
            function = tool.schema["function"]
            self.assertEqual(function["name"], "record_plan")
            self.assertEqual(function["parameters"]["required"], ["plan"])
            self.assertIn("plan", function["parameters"]["properties"])
            self.assertIn("steps", function["parameters"]["properties"])
            store.close()

    def test_records_a_durable_event_with_run_id_and_step_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            tool = RecordPlanTool(store)
            result = tool.execute(
                {"plan": "first inspect, then change", "steps": ["inspect", "change"]},
                idempotency_key="run-123:4:call-9",
            )
            self.assertTrue(result["ok"], result)
            self.assertTrue(result["recorded"])
            row = store.connection.execute("SELECT run_id, payload FROM event_log WHERE kind='plan_recorded'").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["run_id"], "run-123")
            payload = json.loads(row["payload"])
            self.assertEqual(payload["run_id"], "run-123")
            self.assertEqual(payload["plan"], "first inspect, then change")
            self.assertEqual(payload["steps"], ["inspect", "change"])
            self.assertEqual(payload["step_index"], 4)
            self.assertFalse(payload["truncated"])
            store.close()

    def test_length_bound_is_enforced_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            tool = RecordPlanTool(store)
            long_plan = "x" * (tool.MAX_PLAN_CHARS + 500)
            many_steps = [f"step {index} " + "y" * (tool.MAX_STEP_CHARS + 50) for index in range(20)]
            result = tool.execute(
                {"plan": long_plan, "steps": many_steps},
                idempotency_key="run-bound:1:call",
            )
            self.assertTrue(result["ok"], result)
            self.assertTrue(result["truncated"])
            payload = json.loads(
                store.connection.execute("SELECT payload FROM event_log WHERE kind='plan_recorded'").fetchone()["payload"]
            )
            self.assertEqual(len(payload["plan"]), tool.MAX_PLAN_CHARS)
            self.assertEqual(len(payload["steps"]), tool.MAX_STEPS)
            self.assertTrue(all(len(step) <= tool.MAX_STEP_CHARS for step in payload["steps"]))
            store.close()

    def test_empty_plan_and_missing_run_are_structured_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            tool = RecordPlanTool(store)
            self.assertFalse(tool.execute({"plan": "   "}, idempotency_key="run:1:call")["ok"])
            # No run in the key and no active run: refuse instead of writing a
            # plan that cannot be attributed to a run.
            self.assertFalse(tool.execute({"plan": "valid"}, idempotency_key="opaque")["ok"])
            store.close()

    def test_falls_back_to_active_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            state = store.state()
            state.active_run_id = "active-run"
            store.set_state(state)
            tool = RecordPlanTool(store)
            result = tool.execute({"plan": "use the active run"}, idempotency_key="opaque")
            self.assertTrue(result["ok"], result)
            row = store.connection.execute("SELECT run_id FROM event_log WHERE kind='plan_recorded'").fetchone()
            self.assertEqual(row["run_id"], "active-run")
            store.close()


class PlanOrderingTests(unittest.TestCase):
    def _store(self, directory: str) -> StateStore:
        return StateStore(Path(directory) / "state.sqlite3")

    def test_plan_before_mutation_is_true(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            store.append_event("plan_recorded", {"plan": "p"}, "r1")
            store.append_event("tool_call", {"tool_name": "propose_self_improvement", "arguments": {}}, "r1")
            self.assertIs(plan_preceded_first_mutation(store, "r1"), True)
            store.close()

    def test_mutation_before_plan_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            store.append_event("tool_call", {"tool_name": "memory", "arguments": {"action": "remember"}}, "r1")
            store.append_event("plan_recorded", {"plan": "p"}, "r1")
            self.assertIs(plan_preceded_first_mutation(store, "r1"), False)
            store.close()

    def test_memory_read_action_is_not_a_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            store.append_event("tool_call", {"tool_name": "memory", "arguments": {"action": "search"}}, "r1")
            self.assertIsNone(plan_preceded_first_mutation(store, "r1"))
            store.close()

    def test_run_without_mutations_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            store.append_event("tool_call", {"tool_name": "read", "arguments": {}}, "r1")
            store.append_event("tool_call", {"tool_name": "bash", "arguments": {"command": "ls"}}, "r1")
            self.assertIsNone(plan_preceded_first_mutation(store, "r1"))
            store.close()

    def test_rollback_and_promote_count_as_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            store.append_event("tool_call", {"tool_name": "request_rollback", "arguments": {}}, "r1")
            store.append_event("plan_recorded", {"plan": "p"}, "r1")
            self.assertIs(plan_preceded_first_mutation(store, "r1"), False)
            store.close()

    def test_latest_recorded_plan_returns_the_last_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            store.append_event("plan_recorded", {"plan": "first"}, "r1")
            store.append_event("plan_recorded", {"plan": "second"}, "r1")
            latest = latest_recorded_plan(store, "r1")
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest["plan"], "second")
            store.close()


class PlanCrossCycleTests(unittest.TestCase):
    def test_plan_survives_into_next_envelope_without_clobbering_next_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(
                _PlanProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root),
            )
            try:
                self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
                state = reactor.store.state()
                recorded = state.next_plan["recorded_plan"]
                self.assertTrue(str(recorded["plan"]).startswith("Inspect the invariant"))
                self.assertEqual(recorded["steps"], ["inspect", "propose"])
                # The MemoryLoop's own write is intact: a distinct key was merged
                # in, not a replacement of the whole next_plan dict.
                self.assertEqual(state.next_plan["planner_hints"], {"next": "continue from the recorded plan"})
                self.assertIn("initial_prompt", state.next_plan)
                # The ordering answer for this run is a research-only None: a
                # plan was recorded but no mutation followed.
                observation = reactor.store.connection.execute(
                    "SELECT payload FROM event_log WHERE kind='plan_observation'"
                ).fetchone()
                self.assertIsNotNone(observation)
                payload = json.loads(observation["payload"])
                self.assertTrue(payload["planned"])
                self.assertIsNone(payload["preceded_first_mutation"])
            finally:
                reactor.close()

    def test_run_without_a_plan_is_not_failed_or_downgraded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(
                _NoPlanProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root),
            )
            try:
                self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
                self.assertEqual(reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='plan_recorded'").fetchone()[0], 0)
                observation = reactor.store.connection.execute(
                    "SELECT payload FROM event_log WHERE kind='plan_observation'"
                ).fetchone()
                self.assertIsNotNone(observation)
                payload = json.loads(observation["payload"])
                self.assertFalse(payload["planned"])
                self.assertIsNone(payload["preceded_first_mutation"])
                self.assertEqual(reactor.store.state().lifecycle.value, "sleep")
            finally:
                reactor.close()

    def test_prompt_instruction_is_present_in_the_built_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = _NoPlanProvider()
            reactor = Reactor(
                provider,
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root),
            )
            try:
                reactor.tick("test")
                self.assertIn(PLAN_INSTRUCTION, reactor.config.system_prompt)
                self.assertIn("record_plan", reactor.config.system_prompt)
                self.assertIn("Read-only investigation needs no plan", reactor.config.system_prompt)
                # The instruction is in the actual system message the model saw,
                # not only in the config string.
                self.assertTrue(provider.system_prompts)
                self.assertIn("record_plan", provider.system_prompts[0])
                self.assertIn("record_plan", reactor.runner.tools)
            finally:
                reactor.close()
