"""The closing turn has to be readable, and the model was reaching for a tool that did not exist.

Measured on the live ledger 2026-10-19: every one of the 39 recorded calls to `finish` died at
`unknown tool: finish` -- the model reaches for the closing call the ReAct pattern teaches it --
while 15 of 69 closing turns came back as prose the parser could not read. The one bounded repair
turn saved most of those (23 repairs, 7 of them failing), and two runs ended NEEDS_RECOVERY with
their executed evidence carried forward.

The report's contract now travels as the tool's arguments, which needs no prose to reconstruct.
An unusable report comes back as the parser's own words inside the same episode, because a
specific error the model can act on is cheaper than re-sending the transcript for a blind resample
(Huang et al. 2023, arXiv:2310.01798v2).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from skynet.finish_tool import FINISH_TOOL_NAME, FinishTool
from skynet.model_contracts import FINISH_REPORT_SCHEMA
from skynet.models import Budget, ModelTurn, RunStatus, StartEnvelope, ToolCall
from skynet.react import ReActConfig, ReActRunner
from skynet.store import StateStore

REPORT: dict[str, object] = {
    "status": "COMPLETED",
    "summary": "closed by the finish call",
    "evidence": ["the tool ran and this line is its trace"],
    "actions": [],
    "changes": [],
    "tests": [],
    "blocker": "",
    "next_hypothesis": "",
}


class _GrowingTool:
    name = "grow"

    @property
    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": "big", "parameters": {"type": "object", "additionalProperties": False}},
        }

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        return {"ok": True, "payload": "x" * 3000}


class _ScriptedProvider:
    """Replays one scripted turn per call; the last scripted turn repeats."""

    # The runner may only force a tool on a provider that advertises the capability.
    accepts_tool_choice = True

    def __init__(self, *turns: ModelTurn) -> None:
        self.turns = list(turns)
        self.calls = 0
        self.offered: list[list[object]] = []
        self.choices: list[dict[str, object] | None] = []

    def complete(self, messages, *, max_tokens, tools=(), tool_choice=None):
        self.calls += 1
        self.offered.append(list(tools))
        self.choices.append(tool_choice)
        turn = self.turns[min(self.calls - 1, len(self.turns) - 1)]
        return ModelTurn(
            text=turn.text,
            tool_calls=turn.tool_calls,
            usage_tokens=1,
            completion_tokens=1,
            prompt_tokens=1,
        )


def _finish_call(report: dict[str, object]) -> ModelTurn:
    return ModelTurn(tool_calls=[ToolCall(FINISH_TOOL_NAME, dict(report))], usage_tokens=1)


def _run(provider, config: ReActConfig, *, steps: int = 6):
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(Path(directory) / "state.sqlite3")
        tools = {FINISH_TOOL_NAME: FinishTool(), "grow": _GrowingTool()}
        runner = ReActRunner(provider, store, tools, config)
        result = runner.run(
            StartEnvelope("test", budget=Budget(tokens=config.max_tokens, steps=steps, seconds=600)),
            "system",
        )
        events = [
            (str(row[0]), json.loads(row[1] or "{}"))
            for row in store.connection.execute("SELECT kind, payload FROM event_log")
        ]
        store.close()
        return result, events


def _offered_names(provider: _ScriptedProvider) -> list[str]:
    return [
        str(schema["function"]["name"])
        for turn_tools in provider.offered
        for schema in turn_tools
        if isinstance(schema, dict) and "function" in schema
    ]


class FinishToolTests(unittest.TestCase):
    def test_the_tool_is_the_contract_itself(self) -> None:
        tool = FinishTool()
        self.assertEqual(tool.name, FINISH_TOOL_NAME)
        parameters = tool.schema["function"]["parameters"]
        self.assertIs(parameters, FINISH_REPORT_SCHEMA, "the tool must travel the parser's own schema, not a copy that can drift")
        self.assertEqual(
            set(parameters["required"]),
            {"status", "summary", "evidence", "actions", "changes", "tests", "blocker", "next_hypothesis"},
        )
        self.assertEqual(tool.capability_kind, "read", "closing is not an effect: it must not enter the effect-once ledger")

    def test_a_finish_call_closes_the_episode(self) -> None:
        provider = _ScriptedProvider(_finish_call(REPORT))
        result, events = _run(provider, ReActConfig(max_tokens=100_000))
        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertIn("closed by the finish call", result.report)
        self.assertEqual(
            provider.choices,
            [None],
            "a normal step must never be forced: the model chooses its own tool there",
        )
        self.assertFalse(
            [kind for kind, payload in events if kind == "tool_result" and "unknown tool" in json.dumps(payload)],
            "the closing tool must exist: 39 live calls died at 'unknown tool: finish'",
        )
        self.assertTrue(
            [payload for kind, payload in events if kind == "finish_report" and payload.get("via") == FINISH_TOOL_NAME],
            "the report must be recorded as a finish call",
        )

    def test_an_unusable_report_comes_back_inside_the_episode(self) -> None:
        """COMPLETED without evidence is refused with the parser's words, not with a repair turn."""
        provider = _ScriptedProvider(
            _finish_call({**REPORT, "evidence": []}),
            _finish_call(REPORT),
        )
        result, events = _run(provider, ReActConfig(max_tokens=100_000))
        self.assertEqual(result.status, RunStatus.COMPLETED, "the model fixed its own report in the same episode")
        # The refusal travels as the call's result, which is where a tool error belongs: the model
        # reads it in the next step of the same episode.
        refusals = [
            payload.get("result") or {}
            for kind, payload in events
            if kind == "tool_result" and (payload.get("result") or {}).get("failure_class") == "invalid_finish_report"
        ]
        self.assertEqual(len(refusals), 1, "the refusal must reach the model as a tool result")
        self.assertIn("COMPLETED Finish Report requires evidence", str(refusals[0].get("error")))
        self.assertFalse(
            [kind for kind, _ in events if kind == "finish_repair"],
            "a report the model can correct must not cost the bounded repair turn",
        )

    def test_the_closing_turn_carries_only_this_tool_and_closes_on_it(self) -> None:
        """The closing request offers the report's tool and no execution tool, and reads the call.

        Pinned on the closing turn itself rather than on a budget that has to trip first: the
        property under test is what that turn carries and where the report comes from.
        """
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            provider = _ScriptedProvider(_finish_call(REPORT))
            runner = ReActRunner(provider, store, {FINISH_TOOL_NAME: FinishTool()}, ReActConfig(max_tokens=100_000))
            # The closing turn is the subject of this test, so it is called directly.
            status, report, _usage = runner._finish(
                [{"role": "user", "content": "close the episode"}], "run-closing", reason="test"
            )
            offered = provider.offered[0]
            store.close()
        self.assertEqual(
            [str(schema["function"]["name"]) for schema in offered if isinstance(schema, dict)],
            [FINISH_TOOL_NAME],
            "the closing turn carries the report's tool and no execution tool",
        )
        self.assertEqual(status, RunStatus.COMPLETED)
        self.assertIn("closed by the finish call", report, "the report must come from the call, not from prose")
        self.assertEqual(
            provider.choices,
            [{"type": "function", "function": {"name": FINISH_TOOL_NAME}}],
            "the closing turn asks for the report's tool by name instead of hoping for it",
        )

    def test_prose_is_still_read(self) -> None:
        """The old path stays: a provider that returns the contract as text still closes."""
        provider = _ScriptedProvider(ModelTurn(text=json.dumps(REPORT), usage_tokens=1, completion_tokens=1))
        result, _ = _run(provider, ReActConfig(max_tokens=100_000))
        self.assertEqual(result.status, RunStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
