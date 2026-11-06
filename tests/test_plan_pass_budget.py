"""The plan pass's token allowance, and the guard that enforces it.

Measured defect, 2026-10-21: the guard was a frozen per-step constant
(``floor(274_688 / 8) * 0.95 = 32_619``) -- the pass's booking divided by the
eight calls it may make -- while a real per-call cost on the live ledger was
33-38k (every call re-sends its whole prompt, so per-call cost is roughly
constant, not a share of the worst case). Every two-step pass therefore died at
66-77k of a 238k allowance: 29 of 32 passes that day, 103 of 111 the day before,
each retried minutes later and killed again. The guard is now sized at the
booking itself and the booking is read from the planner, so the two cannot drift.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from skynet.autonomous_planner import AutonomousPlanner, plan_pass_token_ceiling
from skynet.models import ModelTurn, ToolCall
from skynet.store import StateStore


class _ScriptedProvider:
    """Answers with the scripted turns, in order, and repeats the last one."""

    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = list(turns)
        self.calls = 0

    def complete(self, messages: Any, *, max_tokens: int, tools: Any = ()) -> ModelTurn:
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        return turn


class _ReadOnlyTool:
    name = "bash_readonly"
    capability_kind = "read"

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": "read-only shell", "parameters": {"type": "object", "properties": {}}}}

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        return {"ok": True, "result": {"output": "hello"}}


def scout_turn(prompt_tokens: int, completion_tokens: int = 1_000) -> ModelTurn:
    return ModelTurn(tool_calls=[ToolCall("bash_readonly", {"command": "ls"})], prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def answer_turn(prompt_tokens: int = 1_000, completion_tokens: int = 1_000) -> ModelTurn:
    return ModelTurn(text=json.dumps({"proposals": []}), prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class PlanPassBudgetTest(unittest.TestCase):
    def planner(self, directory: str, provider: Any, **kwargs: Any) -> AutonomousPlanner:
        store = StateStore(Path(directory) / "state.sqlite3")
        return AutonomousPlanner(provider, store, timeout_seconds=30.0, **kwargs)

    def generate(self, planner: AutonomousPlanner) -> list[dict[str, Any]]:
        return planner.generate(
            generation=1,
            goals=[],
            tasks=[],
            memories=[],
            previous={},
            trigger="test",
            tools={"bash_readonly": _ReadOnlyTool()},
        )

    def attempts(self, planner: AutonomousPlanner) -> list[tuple[str, str]]:
        return [(row[0], row[1]) for row in planner.store.connection.execute("SELECT status, failure FROM planner_attempts ORDER BY rowid")]

    def test_the_allowance_is_derived_from_the_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            planner = self.planner(directory, _ScriptedProvider([answer_turn()]), max_input_chars=600_000)
            self.assertEqual(planner.max_pass_tokens, plan_pass_token_ceiling(600_000, planner.output_tokens))
            planner.resize_window(300_000)
            self.assertEqual(planner.max_input_chars, 300_000)
            self.assertEqual(planner.max_pass_tokens, plan_pass_token_ceiling(300_000, planner.output_tokens))
            self.assertLess(planner.max_pass_tokens, plan_pass_token_ceiling(600_000, planner.output_tokens), "a smaller window books a smaller pass")
            planner.store.close()

    def test_a_two_step_pass_past_the_old_frozen_allowance_still_decides(self) -> None:
        """The regression: 67k over two calls must not abort a 238k pass."""
        with tempfile.TemporaryDirectory() as directory:
            provider = _ScriptedProvider([scout_turn(33_000), answer_turn(33_000)])
            planner = self.planner(directory, provider, max_input_chars=600_000)
            self.assertGreater(66_000, 32_619 * 2, "the old frozen guard would have fired here")
            self.assertLess(68_000, planner.max_pass_tokens, "and the booked allowance is what the guard reads now")
            self.assertEqual(self.generate(planner), [])
            self.assertEqual(planner.last_status, "no_proposals", "the pass reached its answer instead of being aborted")
            self.assertNotIn("plan_budget_mid_pass_exceeded", [status for status, _ in self.attempts(planner)])
            planner.store.close()

    def test_the_guard_still_stops_a_pass_that_outspends_its_allowance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = _ScriptedProvider([scout_turn(200_000), scout_turn(200_000)])
            planner = self.planner(directory, provider, max_input_chars=600_000)
            allowance = planner.max_pass_tokens
            self.assertEqual(self.generate(planner), [])
            attempts = self.attempts(planner)
            self.assertEqual([status for status, _ in attempts], ["plan_budget_mid_pass_exceeded"])
            self.assertIn(str(allowance), attempts[0][1], "the abort names the allowance it read, not a frozen number")
            planner.store.close()
