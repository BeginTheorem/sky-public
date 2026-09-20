"""Deterministic user-facing render of finished runs."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC
from pathlib import Path

from skynet.models import Budget, RunRecord, RunStatus
from skynet.planner import PlannerCandidate
from skynet.reporting import recent_run_pairs, render_pairs, render_recent
from skynet.store import StateStore


def _iso(seconds: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(seconds, tz=UTC).isoformat()

def _candidate(workstream_id: str, task_id: str, title: str, score: float) -> PlannerCandidate:
    return PlannerCandidate(
        workstream_id=workstream_id,
        title=title,
        task_id=task_id,
        score=score,
        criticality=0.5,
        novelty=0.5,
        repetition_penalty=0.0,
        reason="highest novelty-adjusted criticality",
    )

def _seed_run(
    store: StateStore,
    *,
    run_id: str,
    started_at: str,
    finished_at: str,
    task_id: str,
    title: str,
    summary: str,
    status: RunStatus = RunStatus.COMPLETED,
    with_planner: bool = True,
    draw: float | None = None,
    selected_index: int = 0,
) -> None:
    if with_planner:
        candidates = [
            _candidate("area-a", task_id, title, 0.9),
            _candidate("area-b", "other-task", "other task", 0.4),
        ]
        store.record_planner_decision(
            candidates,
            candidates[selected_index],
            reason="epsilon_greedy_explore" if selected_index else candidates[0].reason,
            draw=draw,
        )
    store.create_run(
        RunRecord(run_id, 1, status, started_at, Budget(), finished_at=finished_at)
    )
    store.append_event(
        "run_started",
        {"observations": [{"kind": "durable_state", "generation": 3}]},
        run_id,
    )
    store.append_event(
        "decision_record",
        {"selected_work": {"kind": "task", "task": {"task_id": task_id, "title": title}}},
        run_id,
    )
    for tool_name in ("bash", "bash", "read"):
        store.append_event("tool_call", {"call_id": f"{run_id}-{tool_name}", "tool_name": tool_name, "arguments": {}}, run_id)
    store.append_event(
        "finish_report",
        {
            "text": json.dumps(
                {
                    "status": status.value.upper(),
                    "summary": summary,
                    "evidence": ["first fact", "second fact", "third fact"],
                    "changes": ["skynet/reporting.py", "tests/test_reporting.py"],
                    "tests": ["pytest tests/test_reporting.py -q"],
                    "blocker": "",
                    "next_hypothesis": "",
                }
            ),
            "usage_tokens": 84500,
            "step": 12,
        },
        run_id,
    )
    store.commit_run_result(run_id, status, summary, 12, 84500, "")
    store.record_evaluation(
        run_id,
        {"success_criteria_results": [{"passed": True}, {"passed": True}], "value_estimate": 1.0},
        status,
        summary,
    )

class ReportingTests(unittest.TestCase):
    def test_recent_run_pairs_are_newest_first_with_both_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("reporting", priority=1.0)
            first_task = store.add_task("Fix the reporting pipeline", goal_id)
            second_task = store.add_task("Second task", goal_id)

            _seed_run(
                store,
                run_id="run-1",
                started_at=_iso(50520.0),
                finished_at=_iso(50820.0),
                task_id=first_task,
                title="Fix the reporting pipeline",
                summary="Replaced the prose wall with structured blocks.",
                draw=0.01,
                selected_index=1,
            )
            _seed_run(
                store,
                run_id="run-2",
                started_at=_iso(54000.0),
                finished_at=_iso(54300.0),
                task_id=second_task,
                title="Second task",
                summary="Second run finished cleanly.",
            )

            pairs = recent_run_pairs(store, 5)

            self.assertEqual([pair["scheduler"]["run_id"] for pair in pairs], ["run-2", "run-1"])
            self.assertTrue(pairs[0]["scheduler"]["found"])
            self.assertEqual(pairs[0]["scheduler"]["task_title"], "Second task")
            self.assertEqual(pairs[0]["scheduler"]["candidate_count"], 2)
            self.assertFalse(pairs[0]["scheduler"]["exploration"])
            self.assertTrue(pairs[1]["scheduler"]["exploration"])
            self.assertEqual(pairs[1]["scheduler"]["generation"], 3)
            self.assertEqual(pairs[0]["report"]["status"], "COMPLETED")
            self.assertEqual(pairs[0]["report"]["steps"], 12)
            self.assertEqual(pairs[0]["report"]["tokens"], 84500)
            self.assertEqual(pairs[0]["report"]["tools"], {"bash": 2, "read": 1})
            self.assertEqual(pairs[0]["report"]["criteria_passed"], 2)
            self.assertEqual(pairs[0]["report"]["criteria_total"], 2)
            self.assertEqual(pairs[0]["report"]["evidence"], 3)
            self.assertEqual(pairs[0]["report"]["changes"], 2)
            self.assertEqual(pairs[0]["report"]["summary"], "Second run finished cleanly.")
            store.close()

    def test_render_recent_flattens_two_blocks_per_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("reporting", priority=1.0)
            task_id = store.add_task("Fix the reporting pipeline", goal_id)
            _seed_run(
                store,
                run_id="run-1",
                started_at=_iso(50520.0),
                finished_at=_iso(50820.0),
                task_id=task_id,
                title="Fix the reporting pipeline",
                summary="Replaced the prose wall with structured blocks.",
            )

            rendered = render_recent(store, 5)

            self.assertEqual(len(rendered), 2)
            scheduler, report = rendered
            self.assertTrue(scheduler.startswith("SCHEDULER"))
            self.assertIn("Fix the reporting pipeline", scheduler)
            self.assertIn("highest novelty-adjusted criticality", scheduler)
            self.assertTrue(report.startswith("REPORT"))
            self.assertIn("status: COMPLETED", report)
            self.assertIn("Replaced the prose wall with structured blocks.", report)
            self.assertIn("tools: bash x2, read x1", report)
            self.assertIn("criteria: 2/2", report)
            store.close()

    def test_render_pairs_preserves_order(self) -> None:
        pairs = [
            {"scheduler": {"found": False, "started_at": _iso(54000.0)}, "report": {"status": "BLOCKED", "finished_at": _iso(54300.0)}},
            {"scheduler": {"found": False, "started_at": _iso(50400.0)}, "report": {"status": "FAILED", "finished_at": _iso(50700.0)}},
        ]
        rendered = render_pairs(pairs)
        self.assertEqual(len(rendered), 4)
        self.assertIn("15:00", rendered[0])
        self.assertIn("BLOCKED", rendered[1])
        self.assertIn("14:00", rendered[2])
        self.assertIn("FAILED", rendered[3])

    def test_run_without_planner_decision_renders_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.create_run(
                RunRecord(
                    "lonely-run",
                    1,
                    RunStatus.NEEDS_RECOVERY,
                    _iso(50520.0),
                    Budget(),
                    finished_at=_iso(50820.0),
                )
            )
            store.append_event("run_started", {"observations": []}, "lonely-run")

            pairs = recent_run_pairs(store, 5)
            self.assertEqual(len(pairs), 1)
            self.assertFalse(pairs[0]["scheduler"]["found"])
            rendered = render_recent(store, 5)
            self.assertIn("no planner decision", rendered[0])
            self.assertIn("status: NEEDS_RECOVERY", rendered[1])
            store.close()

if __name__ == "__main__":
    unittest.main()
