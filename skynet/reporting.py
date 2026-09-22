"""Deterministic owner-facing rendering of finished runs.

The model already writes a long English report; the owner needs a compact
summary built from the structured rows the harness owns. Two blocks per run:
SCHEDULER (what the planner chose) and REPORT (what the run produced). Nothing
here reasons about the text: every field is read from `runs`, `run_results`,
`evaluations`, `tasks` or the `event_log` (`run_started`, `planner_decision`,
`decision_record`, `finish_report`, `tool_call`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from .time import parse_timestamp

SUMMARY_LIMIT = 400
FIELD_LIMIT = 300
BLOCK_LIMIT = 1200
DEFAULT_LIMIT = 5


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _plain(value: Any) -> str:
    """Collapse whitespace and strip the markdown scaffolding a report may carry.

    Legacy reports (written before the summary was constrained) arrive as a wall
    of headers and bold reasoning; the owner reads this in a chat, so the
    decoration is removed rather than shown.
    """
    text = str(value or "")
    text = re.sub(r"`+", "", text)
    text = re.sub(r"[*_]{1,2}", "", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    return " ".join(text.split())


def _oneline(value: Any, limit: int = FIELD_LIMIT) -> str:
    text = _plain(value)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _truncate(text: str, limit: int = BLOCK_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _format_utc(value: Any) -> str:
    if not value:
        return "unknown"
    try:
        parsed = parse_timestamp(str(value))
    except (TypeError, ValueError):
        return str(value)[:19]
    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def _finished_runs(store: Any, limit: int) -> list[Any]:
    # A finished run is any non-running row; `finished_at` can still be null on
    # legacy rows, so status is the reliable discriminator.
    return store.connection.execute(
        "SELECT run_id, attempt, status, started_at, finished_at, budget FROM runs "
        "WHERE status != 'running' ORDER BY started_at DESC, rowid DESC LIMIT ?",
        (max(0, int(limit)),),
    ).fetchall()


def _run_started_event(store: Any, run_id: str) -> Any:
    return store.connection.execute(
        "SELECT sequence, payload, created_at FROM event_log "
        "WHERE run_id=? AND kind='run_started' ORDER BY sequence DESC LIMIT 1",
        (run_id,),
    ).fetchone()


def _run_generation(store: Any, run_id: str) -> int | None:
    event = _run_started_event(store, run_id)
    if event is None:
        return None
    payload = _as_dict(_parse_json(event["payload"]))
    for observation in payload.get("observations", []):
        if isinstance(observation, dict) and observation.get("kind") == "durable_state":
            generation = observation.get("generation")
            return int(generation) if isinstance(generation, (int, float)) else None
    return None


def _planner_event(store: Any, run_id: str, started_at: str) -> Any:
    """Return the planner decision that selected this run.

    `planner_decision` carries no run id, but the planner records it immediately
    before `run_started`, so the newest decision before that event is the one
    that picked the work. A timestamp fallback covers legacy runs that lack a
    `run_started` row.
    """
    event = _run_started_event(store, run_id)
    if event is not None:
        decision = store.connection.execute(
            "SELECT payload, created_at FROM event_log "
            "WHERE kind='planner_decision' AND sequence < ? ORDER BY sequence DESC LIMIT 1",
            (event["sequence"],),
        ).fetchone()
        if decision is not None:
            return decision
    if not started_at:
        return None
    return store.connection.execute(
        "SELECT payload, created_at FROM event_log "
        "WHERE kind='planner_decision' AND created_at <= ? "
        "ORDER BY created_at DESC, sequence DESC LIMIT 1",
        (started_at,),
    ).fetchone()


def _task_title(store: Any, task_id: Any) -> str:
    if not task_id:
        return ""
    row = store.connection.execute("SELECT title FROM tasks WHERE task_id=?", (str(task_id),)).fetchone()
    return str(row["title"]) if row is not None else ""


def _decision_record_title(store: Any, run_id: str) -> str:
    """Title of the work actually attempted, from the run's decision record.

    The planner decision names the ranked pick; when no novel work existed the
    run may have executed an inbox or fallback task, which only `decision_record`
    records.
    """
    row = store.connection.execute(
        "SELECT payload FROM event_log WHERE run_id=? AND kind='decision_record' ORDER BY sequence DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if row is None:
        return ""
    selected = _as_dict(_parse_json(row["payload"])).get("selected_work")
    if not isinstance(selected, dict):
        return ""
    raw_task = selected.get("task")
    raw_goal = selected.get("goal")
    task: dict[str, Any] = raw_task if isinstance(raw_task, dict) else {}
    goal: dict[str, Any] = raw_goal if isinstance(raw_goal, dict) else {}
    return str(task.get("title") or goal.get("title") or "")


def _scheduler_block(store: Any, run_row: Any) -> dict[str, Any]:
    run_id = str(run_row["run_id"])
    started_at = str(run_row["started_at"] or "")
    generation = _run_generation(store, run_id)
    decision = _planner_event(store, run_id, started_at)
    if decision is None:
        return {
            "run_id": run_id,
            "started_at": started_at,
            "generation": generation,
            "found": False,
            "task_id": None,
            "workstream_id": None,
            "task_title": "",
            "reason": "",
            "candidate_count": 0,
            "exploration": False,
            "draw": None,
        }
    payload = _as_dict(_parse_json(decision["payload"]))
    raw_candidates = payload.get("candidates")
    candidates = raw_candidates if isinstance(raw_candidates, list) else []
    selected_task_id = payload.get("selected_task_id")
    selected_workstream_id = payload.get("selected_workstream_id")
    exploration = _as_dict(payload.get("exploration"))
    # The planner records a draw whenever epsilon > 0, but only an exploration
    # pick moves off rank 0; chosen_rank is what distinguishes the two.
    chosen_rank = exploration.get("chosen_rank")
    explore = bool(exploration) and chosen_rank not in (None, 0)
    title = _task_title(store, selected_task_id) or _decision_record_title(store, run_id)
    return {
        "run_id": run_id,
        "started_at": started_at,
        "generation": generation,
        "found": True,
        "task_id": str(selected_task_id) if selected_task_id else None,
        "workstream_id": str(selected_workstream_id) if selected_workstream_id else None,
        "task_title": title,
        "reason": str(payload.get("reason") or ""),
        "candidate_count": len(candidates),
        "exploration": explore,
        "draw": exploration.get("draw"),
    }


def _finish_report_fields(text: Any) -> dict[str, Any]:
    data = _parse_json(text)
    if isinstance(data, dict):
        return {
            "summary": str(data.get("summary") or ""),
            "evidence": data.get("evidence") if isinstance(data.get("evidence"), list) else [],
            "changes": data.get("changes") if isinstance(data.get("changes"), list) else [],
            "tests": data.get("tests") if isinstance(data.get("tests"), list) else [],
            "blocker": str(data.get("blocker") or ""),
        }
    prose = str(text or "").strip()
    return {"summary": prose, "evidence": [], "changes": [], "tests": [], "blocker": ""}


def _report_block(store: Any, run_row: Any) -> dict[str, Any]:
    run_id = str(run_row["run_id"])
    status = str(run_row["status"] or "unknown").upper()
    started_at = str(run_row["started_at"] or "")
    finished_at = str(run_row["finished_at"] or "") or started_at
    result = store.connection.execute(
        "SELECT steps, usage_tokens, report, failure FROM run_results WHERE run_id=?", (run_id,)
    ).fetchone()
    finish = store.connection.execute(
        "SELECT payload FROM event_log WHERE run_id=? AND kind='finish_report' ORDER BY sequence DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    finish_payload = _as_dict(_parse_json(finish["payload"])) if finish is not None else {}
    parsed = _finish_report_fields(finish_payload.get("text"))
    evaluation = store.connection.execute(
        "SELECT status, summary, evaluation FROM evaluations "
        "WHERE run_id=? AND status != 'metrics' ORDER BY evaluation_id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    evaluation_data = _as_dict(_parse_json(evaluation["evaluation"])) if evaluation is not None else {}
    criteria = evaluation_data.get("success_criteria_results")
    criteria = criteria if isinstance(criteria, list) else []
    passed = sum(1 for item in criteria if isinstance(item, dict) and item.get("passed") is True)

    summary = parsed["summary"]
    if not summary and parsed["evidence"]:
        summary = str(parsed["evidence"][0])
    if not summary and evaluation is not None:
        evaluation_summary = _parse_json(evaluation["summary"])
        if isinstance(evaluation_summary, dict):
            summary = str(evaluation_summary.get("summary") or "")
        elif evaluation_summary is None:
            summary = str(evaluation["summary"] or "")

    steps = int(result["steps"]) if result is not None and result["steps"] is not None else finish_payload.get("step")
    tokens = int(result["usage_tokens"]) if result is not None and result["usage_tokens"] is not None else finish_payload.get("usage_tokens")

    counts: dict[str, int] = {}
    for row in store.connection.execute(
        "SELECT payload FROM event_log WHERE run_id=? AND kind='tool_call'", (run_id,)
    ).fetchall():
        tool_payload = _as_dict(_parse_json(row["payload"]))
        name = str(tool_payload.get("tool_name") or "unknown")
        counts[name] = counts.get(name, 0) + 1

    return {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "steps": steps,
        "tokens": tokens,
        "tools": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        "criteria_passed": passed,
        "criteria_total": len(criteria),
        "evidence": len(parsed["evidence"]),
        "changes": len(parsed["changes"]),
        "tests": len(parsed["tests"]),
        # `failure` is a *control* label on a completed run, not a fault. The
        # deferred-restart path stores "deferred restart" there while the status
        # stays COMPLETED (51 of 66 rows on the live ledger, 0 non-completed
        # rows), so using it as the blocker fallback printed
        # "blocker: deferred restart" beneath "status: COMPLETED" in 16 of the
        # newest 25 owner-facing reports. A completed run has no blocker unless
        # the model authored one; every other status keeps the fallback.
        "blocker": parsed["blocker"]
        or (str(result["failure"]) if status != "COMPLETED" and result is not None and result["failure"] else ""),
        "summary": summary,
    }


def recent_run_pairs(store: Any, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    """Newest finished runs as `{"scheduler": ..., "report": ...}` blocks."""
    return [
        {"scheduler": _scheduler_block(store, run_row), "report": _report_block(store, run_row)}
        for run_row in _finished_runs(store, limit)
    ]


def _render_scheduler(block: dict[str, Any]) -> str:
    head = f"SCHEDULER  {_format_utc(block.get('started_at'))}"
    generation = block.get("generation")
    if generation is not None:
        head += f"  gen {generation}"
    if not block.get("found"):
        return _truncate(head + "\nno planner decision")
    title = _oneline(block.get("task_title") or block.get("workstream_id") or "(unselected)")
    reason = _oneline(block.get("reason") or "(none)")
    exploration = "yes" if block.get("exploration") else "no"
    return _truncate(
        f"{head}\n"
        f"task: {title}\n"
        f"reason: {reason}\n"
        f"candidates: {int(block.get('candidate_count') or 0)}   exploration: {exploration}"
    )


def _render_report(block: dict[str, Any]) -> str:
    head = f"REPORT  {_format_utc(block.get('finished_at'))}  status: {block.get('status') or 'UNKNOWN'}"
    lines = [head]
    facts: list[str] = []
    if block.get("steps") is not None:
        facts.append(f"steps: {block['steps']}")
    if block.get("tokens") is not None:
        facts.append(f"tokens: {block['tokens']}")
    tools = block.get("tools") if isinstance(block.get("tools"), dict) else {}
    if tools:
        facts.append("tools: " + ", ".join(f"{name} x{count}" for name, count in tools.items()))
    if facts:
        lines.append("   ".join(facts))
    tally = [
        f"criteria: {int(block.get('criteria_passed') or 0)}/{int(block.get('criteria_total') or 0)}",
        f"evidence: {int(block.get('evidence') or 0)}",
        f"changes: {int(block.get('changes') or 0)}",
    ]
    if block.get("tests"):
        tally.append(f"tests: {int(block['tests'])}")
    lines.append("   ".join(tally))
    lines.append("summary: " + (_oneline(block.get("summary"), SUMMARY_LIMIT) or "(no summary)"))
    if block.get("blocker"):
        lines.append("blocker: " + _oneline(block["blocker"]))
    return _truncate("\n".join(lines))


def render_pairs(pairs: Sequence[dict[str, Any]]) -> list[str]:
    """Flatten pairs to `[scheduler, report, scheduler, report, ...]`."""
    rendered: list[str] = []
    for pair in pairs:
        rendered.append(_render_scheduler(_as_dict(pair.get("scheduler"))))
        rendered.append(_render_report(_as_dict(pair.get("report"))))
    return rendered


def render_recent(store: Any, limit: int = DEFAULT_LIMIT) -> list[str]:
    return render_pairs(recent_run_pairs(store, limit))
