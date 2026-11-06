"""Compaction has to be reachable, and it was not.

The path existed, was documented, and had a test whose fixture set `finish_threshold_ratio=2.0`
-- a value production never uses. In production the reserve clears at `request_limit - reserve`,
the finish fires at `finish_threshold_ratio` of the same limit, the shipped reserve is
`limit // 8`, and those two thresholds are the same number (87,500 on a 100k window; 525,000
against 550,000 on a 600k one) with the finish tested first and returning. Measured on the live
ledger 2026-10-19: **zero** `context_compacted` events in the whole epoch, and 14 runs closed as
`protective context threshold` -- 26.7M tokens, no deliverable.

These tests pin both halves: that a ratio below the finish threshold resurrects a full window,
and that the old ordering is still recognisable as the deadlock it was.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from skynet.models import Budget, ModelTurn, RunStatus, StartEnvelope, ToolCall
from skynet.react import ReActConfig, ReActRunner
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


class _GrowingTool:
    """A tool whose result keeps the transcript growing: the pressure compaction exists for."""

    name = "grow"

    @property
    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": "big", "parameters": {"type": "object", "additionalProperties": False}},
        }

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        return {"ok": True, "payload": "x" * 3000}


class _ToolCallingProvider:
    """Calls the tool for the first `tool_steps` turns, then returns a Finish Report."""

    def __init__(self, tool_steps: int) -> None:
        self.tool_steps = tool_steps
        self.calls = 0

    def complete(self, messages, *, max_tokens, tools=()):
        self.calls += 1
        if self.calls <= self.tool_steps:
            return ModelTurn(tool_calls=[ToolCall("grow", {})], completion_tokens=1, usage_tokens=1)
        return ModelTurn(text=FINISH_OK, completion_tokens=1, usage_tokens=1)


def _run(config: ReActConfig, *, tool_steps: int = 12, steps: int = 10) -> tuple[object, int, list[str]]:
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(Path(directory) / "state.sqlite3")
        runner = ReActRunner(_ToolCallingProvider(tool_steps), store, {"grow": _GrowingTool()}, config)
        result = runner.run(StartEnvelope("test", budget=Budget(tokens=config.max_tokens, steps=steps, seconds=600)), "system")
        compactions = store.connection.execute(
            "SELECT COUNT(*) FROM event_log WHERE kind='context_compacted'"
        ).fetchone()[0]
        kinds = [
            str(row[0])
            for row in store.connection.execute("SELECT kind FROM event_log WHERE kind IN ('context_compacted','context_limit')")
        ]
        store.close()
        return result, int(compactions), kinds


class ReachableCompactionTests(unittest.TestCase):
    def test_a_ratio_below_the_finish_threshold_resurrects_the_window(self) -> None:
        config = ReActConfig(
            max_tokens=400,
            context_finish_reserve=100,
            compaction_keep_messages=2,
            compaction_threshold_ratio=0.75,
        )
        result, compactions, _ = _run(config, tool_steps=3)
        self.assertGreaterEqual(compactions, 1, "the window filled and the transcript was not compacted")
        self.assertEqual(getattr(result, "status", None), RunStatus.COMPLETED, "the episode continued to its own finish")

    def test_the_shipped_numbers_were_a_deadlock_by_arithmetic(self) -> None:
        """Why the old order was fatal, as arithmetic rather than as a live run.

        The reserve clears at ``limit - reserve``; the finish fires at ``finish_threshold_ratio``
        of the same limit; the shipped reserve is ``limit // 8``. So the two numbers coincide
        exactly, and a finish check tested first can only ever return. On the live ledger this
        produced zero compactions and 14 runs closed at the ceiling.
        """
        for limit, reserve_config in ((100_000, 50_000), (150_000, 50_000), (600_000, 50_000)):
            reserve = min(reserve_config, max(0, limit // 8))
            finish_at = limit * ReActConfig().finish_threshold_ratio
            compaction_needed = limit - reserve
            self.assertLessEqual(
                finish_at,
                compaction_needed,
                f"window {limit}: the finish threshold must not sit above the point compaction needs",
            )
        # Only while the operator's reserve is the binding one do the two coincide exactly; on
        # a 600k window the 50k cap widens the gap to 25,000 in the finish's favour, which is
        # equally fatal to a compaction check tested second.
        for limit in (100_000, 150_000, 200_000):
            reserve = max(0, limit // 8)
            self.assertEqual(limit * ReActConfig().finish_threshold_ratio, limit - reserve, f"window {limit}")

    def test_testing_compaction_first_is_what_opens_it(self) -> None:
        """Even at the finish ratio, ordering compaction first makes it reachable.

        This is the difference between the shipped code and the fix: same threshold, and the
        transcript still gets compacted instead of the episode being closed.
        """
        config = ReActConfig(
            max_tokens=400,
            context_finish_reserve=100,
            compaction_keep_messages=2,
            compaction_threshold_ratio=0.875,
        )
        _, compactions, _ = _run(config, tool_steps=3)
        self.assertGreaterEqual(compactions, 1)

    def test_compaction_is_bounded_by_max_compactions(self) -> None:
        config = ReActConfig(
            max_tokens=400,
            context_finish_reserve=100,
            compaction_keep_messages=2,
            compaction_threshold_ratio=0.4,
            max_compactions=1,
        )
        result, compactions, _ = _run(config, tool_steps=50, steps=20)
        self.assertEqual(compactions, 1, "the budget is spent exactly once and the loop then ends")
        self.assertIsNotNone(getattr(result, "status", None), "the run terminates instead of compacting forever")

    def test_compaction_can_be_switched_off(self) -> None:
        config = ReActConfig(
            max_tokens=400,
            context_finish_reserve=100,
            compaction_keep_messages=2,
            compaction_threshold_ratio=0.0,
        )
        result, compactions, _ = _run(config)
        self.assertEqual(compactions, 0)
        self.assertEqual(getattr(result, "failure", ""), "protective context threshold")

    def test_the_reserve_alone_still_closes_the_episode_when_nothing_can_shrink(self) -> None:
        """Between the compaction trigger and the ceiling the run is not left hanging."""
        config = ReActConfig(
            max_tokens=400,
            context_finish_reserve=100,
            compaction_keep_messages=2,
            compaction_threshold_ratio=0.2,
            max_compactions=0,
        )
        result, compactions, _ = _run(config)
        self.assertEqual(compactions, 0)
        self.assertIn(getattr(result, "failure", ""), {"protective context threshold", "context budget"})


if __name__ == "__main__":
    unittest.main()
