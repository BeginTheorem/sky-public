"""One big Plan: the pass scouts with its read-only tools before it proposes.

The shipped contract was "no tools, and a tool call in the reply is a parse failure".
The owner asked for the opposite shape -- a large planning pass that calls tools and
scouts -- so these tests pin both halves: with tools the pass really executes them and
keeps going, and without tools the old contract is byte-for-byte what it was.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from skynet.autonomous_planner import MAX_TOOL_ROUNDS, AutonomousPlanner
from skynet.model_contracts import MAX_MODEL_REASKS
from skynet.models import ModelTurn, ToolCall
from skynet.store import StateStore


class _ScriptedProvider:
    """Answers with the scripted turns and records what it was asked for."""

    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = list(turns)
        self.calls = 0
        self.requests: list[dict[str, Any]] = []

    def complete(self, messages: Any, *, max_tokens: int, tools: Any = ()) -> ModelTurn:
        self.requests.append({"messages": list(messages), "tools": list(tools)})
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        return turn


class _RecordingTool:
    """A read-only tool that records its calls."""

    name = "bash_readonly"
    capability_kind = "read"

    def __init__(self, *, explode: bool = False) -> None:
        self.explode = explode
        self.calls: list[dict[str, Any]] = []

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": "read-only shell", "parameters": {"type": "object", "properties": {}}}}

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        self.calls.append({"arguments": dict(arguments), "idempotency_key": idempotency_key})
        if self.explode:
            raise RuntimeError("tool went away")
        return {"ok": True, "result": {"command": arguments.get("command"), "output": "hello"}}


def scout_turn() -> ModelTurn:
    return ModelTurn(tool_calls=[ToolCall("bash_readonly", {"command": "ls"})], usage_tokens=5)


def answer_turn() -> ModelTurn:
    return ModelTurn(text=json.dumps({"proposals": []}), usage_tokens=3, prompt_tokens=100, completion_tokens=3)


class PlanScoutingTest(unittest.TestCase):
    def planner(self, directory: str, provider: _ScriptedProvider) -> AutonomousPlanner:
        store = StateStore(Path(directory) / "state.sqlite3")
        return AutonomousPlanner(provider, store, timeout_seconds=30.0)

    def test_the_pass_runs_the_tool_then_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = _ScriptedProvider([scout_turn(), answer_turn()])
            planner = self.planner(directory, provider)
            tool = _RecordingTool()
            proposals = planner.generate(generation=1, goals=[], tasks=[], memories=[], previous={}, trigger="test", tools={"bash_readonly": tool})
            self.assertEqual(proposals, [])
            self.assertEqual(planner.last_status, "no_proposals")
            self.assertEqual(len(tool.calls), 1, "the pass executed what it asked for")
            self.assertEqual(tool.calls[0]["arguments"], {"command": "ls"})
            # The second request carries the tool result back to the model.
            second = provider.requests[1]["messages"]
            self.assertTrue(any(message.get("role") == "tool" for message in second), "the result was fed back")
            self.assertTrue(any(message.get("role") == "assistant" and message.get("tool_calls") for message in second))
            events = [row["kind"] for row in planner.store.connection.execute("SELECT kind FROM event_log ORDER BY sequence")]
            self.assertIn("planner_tool_call", events)
            planner.store.close()

    def test_scouting_is_bounded_by_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = _ScriptedProvider([scout_turn()])  # always asks for a tool
            planner = self.planner(directory, provider)
            tool = _RecordingTool()
            planner.generate(generation=1, goals=[], tasks=[], memories=[], previous={}, trigger="test", tools={"bash_readonly": tool})
            self.assertEqual(planner.last_status, "invalid_response", "a model that only scouts never decides")
            self.assertEqual(len(tool.calls), MAX_TOOL_ROUNDS, "one execution per round, and the rounds are bounded")
            # Two independent budgets bound the pass: reconnaissance rounds and parse
            # re-asks. A model that only ever asks for tools spends both and still stops --
            # the worst case is what the pass's run row states as its step budget.
            self.assertEqual(provider.calls, 1 + MAX_TOOL_ROUNDS + MAX_MODEL_REASKS)
            planner.store.close()

    def test_a_tool_fault_is_data_not_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = _ScriptedProvider([scout_turn(), answer_turn()])
            planner = self.planner(directory, provider)
            proposals = planner.generate(generation=1, goals=[], tasks=[], memories=[], previous={}, trigger="test", tools={"bash_readonly": _RecordingTool(explode=True)})
            self.assertEqual(proposals, [])
            self.assertEqual(planner.last_status, "no_proposals", "the pass still answered")
            payload = json.loads(cast(str, provider.requests[1]["messages"][-1]["content"]))
            self.assertFalse(payload["ok"])
            self.assertIn("tool raised", payload["error"])
            planner.store.close()

    def test_without_tools_a_call_is_still_a_parse_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = _ScriptedProvider([scout_turn()])
            planner = self.planner(directory, provider)
            proposals = planner.generate(generation=1, goals=[], tasks=[], memories=[], previous={}, trigger="test")
            self.assertEqual(proposals, [])
            self.assertEqual(planner.last_status, "invalid_response", "the shipped contract is untouched without tools")
            self.assertEqual(len(provider.requests[0]["tools"]), 0, "no schemas are sent to a pass with no tools")
            events = [row["kind"] for row in planner.store.connection.execute("SELECT kind FROM event_log ORDER BY sequence")]
            self.assertIn("planner_retry", events)
            planner.store.close()

    def test_the_forcing_re_ask_is_not_offered_the_schemas(self) -> None:
        """Once the reconnaissance rounds are spent the pass is forced to answer, so the
        re-ask carries no schemas and must not invite a call it cannot execute."""

        with tempfile.TemporaryDirectory() as directory:
            provider = _ScriptedProvider([scout_turn()] * (MAX_TOOL_ROUNDS + 1) + [ModelTurn(text="not json")])
            planner = self.planner(directory, provider)
            planner.generate(generation=1, goals=[], tasks=[], memories=[], previous={}, trigger="test", tools={"bash_readonly": _RecordingTool()})
            self.assertEqual(planner.last_status, "invalid_response")
            reasks = provider.requests[1 + MAX_TOOL_ROUNDS :]
            self.assertEqual(len(reasks), MAX_MODEL_REASKS)
            for request in reasks:
                self.assertEqual(request["tools"], [], "a forcing re-ask carries no schemas")
                self.assertNotIn("call a read-only tool first", str(request["messages"][-1]["content"]))
            planner.store.close()

    def test_the_request_carries_the_read_only_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = _ScriptedProvider([answer_turn()])
            planner = self.planner(directory, provider)
            tool = _RecordingTool()
            planner.generate(generation=1, goals=[], tasks=[], memories=[], previous={}, trigger="test", tools={"bash_readonly": tool})
            sent = provider.requests[0]["tools"]
            self.assertEqual([schema["function"]["name"] for schema in sent], ["bash_readonly"], "the pass is told what it may call")
            planner.store.close()


if __name__ == "__main__":
    unittest.main()
