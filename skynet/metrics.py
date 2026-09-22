"""Experiment metrics computed by pure SQL over the durable state.

Every number the project claims about itself must be reproducible from
``state/skynet.sqlite3`` plus the filesystem. This module is the only reader of
``evaluations`` and ``planner_decisions`` in production code: before it existed
both tables were written every cycle and never looked at, which is why a 66%
``needs_recovery`` rate and a memory retrieval that returned zero rows survived
159 generations unnoticed.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

from .time import utc_datetime_now, utc_now

DEVIATION_KINDS = (
    "doom_loop",
    "budget_exhausted",
    "emergency_finish",
    "watchdog_timeout",
    "uncaptured_changes",
    "finish_invalid",
    "finish_failure",
    "restart_failed",
    "restart_escalated",
    "cycle_error",
    "provider_lockout",
    # Chain-level provider deviations: a single provider strike and the whole
    # chain going into cooldown are both ways a cycle goes wrong. The reload and
    # supervisor lifecycle kinds are deliberately excluded: they are ordinary
    # operational events, not deviations.
    "fallback_failure",
    "fallback_all_cooling",
    "worktree_dirty_after_bash",
    "livelock_suspected",
    "memory_loop_failed",
    "memory_degraded",
    "policy_denied",
    "policy_soft_denied",
    "gate_protected_warned",
    "gate_protected_soft_denied",
    "registry_corrupted",
    "mcp_server_unavailable",
)

MEMORY_RETRIEVAL_KIND = "memory_retrieval"


def _rows(connection: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in connection.execute(sql, params).fetchall()]
    except sqlite3.Error:
        return []


def _one(connection: sqlite3.Connection, sql: str, params: tuple[Any, ...] = (), default: Any = 0) -> Any:
    rows = _rows(connection, sql, params)
    if not rows:
        return default
    value = next(iter(rows[0].values()), default)
    return default if value is None else value


def outcome_mix(connection: sqlite3.Connection, since: str) -> dict[str, Any]:
    """Run outcomes, steps and token spend inside the window."""
    rows = _rows(
        connection,
        "SELECT status, COUNT(*) AS n, COALESCE(SUM(steps), 0) AS steps, "
        "COALESCE(SUM(usage_tokens), 0) AS tokens FROM run_results WHERE created_at >= ? GROUP BY status",
        (since,),
    )
    by_status = {str(row["status"]): int(row["n"]) for row in rows}
    total = sum(by_status.values())
    completed = by_status.get("completed", 0)
    tokens = {str(row["status"]): int(row["tokens"]) for row in rows}
    steps = {str(row["status"]): int(row["steps"]) for row in rows}
    # Only completed runs pay for the work they produced. The previous formula
    # divided the all-status token total by `completed`, so tokens burned by
    # failed and needs_recovery runs inflated the per-completed number (a 66%
    # needs_recovery rate made one completed run look ~3x more expensive).
    completed_tokens = tokens.get("completed", 0)
    wasted_tokens = sum(value for status, value in tokens.items() if status != "completed")
    return {
        "total_runs": total,
        "by_status": by_status,
        "completion_rate": round(completed / total, 4) if total else None,
        "tokens_total": sum(tokens.values()),
        "tokens_by_status": tokens,
        "steps_by_status": steps,
        "tokens_per_completed_run": int(completed_tokens / completed) if completed else None,
        "tokens_wasted": wasted_tokens,
    }


def deviations(connection: sqlite3.Connection, since: str) -> dict[str, int]:
    """Counters for the ways a cycle can go wrong."""
    placeholders = ",".join("?" for _ in DEVIATION_KINDS)
    rows = _rows(
        connection,
        f"SELECT kind, COUNT(*) AS n FROM event_log WHERE created_at >= ? AND kind IN ({placeholders}) "
        "GROUP BY kind ORDER BY n DESC",
        (since, *DEVIATION_KINDS),
    )
    return {str(row["kind"]): int(row["n"]) for row in rows}


def provider_failures(connection: sqlite3.Connection, since: str) -> dict[str, Any]:
    """Why the model was unreachable, and whether the brake engaged."""
    rows = _rows(
        connection,
        "SELECT payload FROM event_log WHERE created_at >= ? AND kind='provider_error_classified'",
        (since,),
    )
    categories: dict[str, int] = {}
    zero_cooldowns = 0
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except (TypeError, json.JSONDecodeError):
            continue
        category = str(payload.get("category", "unknown"))
        categories[category] = categories.get(category, 0) + 1
        if payload.get("retryable") and not float(payload.get("cooldown_seconds") or 0.0):
            zero_cooldowns += 1
    failure_reasons = _rows(
        connection,
        "SELECT failure, COUNT(*) AS n FROM run_results WHERE created_at >= ? AND failure IS NOT NULL "
        "AND failure != '' GROUP BY failure ORDER BY n DESC LIMIT 5",
        (since,),
    )
    total = sum(categories.values())
    degraded_categories = {name: n for name, n in categories.items() if name in {"unavailable", "timeout", "server"}}
    degraded_count = sum(degraded_categories.values())
    # The threshold is deliberately conservative: a handful of transient
    # strikes is normal, but a majority of availability/timeout/server errors
    # means the chain is the bottleneck, not the idea under test.
    degraded = total >= 5 and degraded_count / total >= 0.5
    return {
        "classified": categories,
        "classified_total": total,
        "degraded": degraded,
        "degraded_categories": degraded_categories,
        "recommended_action": (
            "chain is degraded: prefer a healthy provider or extend the cooldown before trusting run outcomes"
            if degraded
            else ""
        ),
        "retryable_without_cooldown": zero_cooldowns,
        "top_failures": [{"failure": str(row["failure"])[:160], "n": int(row["n"])} for row in failure_reasons],
    }


def livelock_streaks(connection: sqlite3.Connection, since: str, *, threshold: int = 3) -> list[dict[str, Any]]:
    """Longest consecutive run of selections per task inside the window.

    Grouping by ``selected_task_id`` with ``COUNT(*)`` measured the total number
    of selections, so a task picked 90 times with work interleaved looked like a
    90-in-a-row livelock. The streak is the maximum run of adjacent decisions
    for the same task in insertion order, which is what the Reactor guard acts
    on.
    """
    rows = _rows(
        connection,
        "SELECT selected_task_id FROM planner_decisions "
        "WHERE created_at >= ? AND selected_task_id IS NOT NULL "
        "ORDER BY created_at, rowid",
        (since,),
    )
    longest: dict[str, int] = {}
    previous_id: str | None = None
    current = 0
    for row in rows:
        task_id = str(row["selected_task_id"])
        if task_id == previous_id:
            current += 1
        else:
            previous_id = task_id
            current = 1
        if current > longest.get(task_id, 0):
            longest[task_id] = current
    ranked = [
        {"task_id": task_id, "streak": streak}
        for task_id, streak in longest.items()
        if streak >= threshold
    ]
    ranked.sort(key=lambda item: item["streak"], reverse=True)
    return ranked[:10]


def fingerprint_pressure(connection: sqlite3.Connection) -> dict[str, Any]:
    """How much of the hypothesis space is already consumed."""
    hypotheses = {
        str(row["status"]): int(row["n"])
        for row in _rows(connection, "SELECT status, COUNT(*) AS n FROM hypotheses GROUP BY status")
    }
    proposals = {
        str(row["status"]): int(row["n"])
        for row in _rows(connection, "SELECT status, COUNT(*) AS n FROM planner_proposals GROUP BY status")
    }
    tasks = {
        str(row["status"]): int(row["n"])
        for row in _rows(connection, "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status")
    }
    total = sum(hypotheses.values())
    terminal = sum(value for key, value in hypotheses.items() if key in {"completed", "rejected", "exhausted"})
    return {
        "hypotheses": hypotheses,
        "hypothesis_space_consumed": round(terminal / total, 4) if total else None,
        "proposals": proposals,
        "tasks": tasks,
    }


def diversity_health(connection: sqlite3.Connection, since: str) -> dict[str, Any]:
    """Diversity signals that survive paraphrase and duplication.

    Raw output entropy is not enough: template collapse keeps output entropy high
    while input dependence falls to zero, and entropy correlates with final
    performance negatively where mutual information correlates positively. Every
    value here is computed from stored rows, with no new model calls.
    """
    from .idea_archive import (
        CELLS_TOTAL,
        effective_modes,
        lexical_uniqueness,
        structural_disorder,
    )

    rows = _rows(
        connection,
        "SELECT cell_key, evidence_source, title, problem_description, quality, created_at FROM idea_archive",
    )
    counts = Counter(str(r["cell_key"]) for r in rows)
    recent = [
        str(r["title"]) + " " + str(r["problem_description"])
        for r in rows
        if str(r.get("created_at", "")) >= since
    ]
    external = sum(1 for r in rows if str(r["evidence_source"]) in {"paper", "web"})

    snr_rows = _rows(
        connection,
        """
        SELECT t.area AS area,
               AVG(CASE WHEN e.status='passed' THEN 1.0 ELSE 0.0 END) AS mean,
               COUNT(*) AS n
        FROM evaluations e
        JOIN runs r ON r.run_id=e.run_id
        JOIN tasks t ON t.task_id=r.task_id
        WHERE e.created_at >= ?
        GROUP BY t.area HAVING n >= 3
        """,
        (since,),
    )
    outcome_variance = {
        str(r["area"]): round(float(r["mean"]) * (1.0 - float(r["mean"])), 6) for r in snr_rows
    }

    kind_rows = _rows(
        connection,
        """
        SELECT t.area AS area, COUNT(DISTINCT rr.run_id) AS n
        FROM run_results rr
        JOIN runs r ON r.run_id=rr.run_id
        JOIN tasks t ON t.task_id=r.task_id
        WHERE rr.created_at >= ?
        GROUP BY t.area
        """,
        (since,),
    )
    kinds = len(kind_rows)

    texts = recent or [str(r["title"]) + " " + str(r["problem_description"]) for r in rows][-40:]
    return {
        "cells_total": CELLS_TOTAL,
        "cells_filled": len(counts),
        "coverage_ratio": round(len(counts) / CELLS_TOTAL, 4) if CELLS_TOTAL else 0.0,
        "effective_modes": round(effective_modes(counts.values()), 4),
        "ideas": len(rows),
        "ideas_recent": len(recent),
        "external_evidence_ratio": round(external / len(rows), 4) if rows else None,
        "lexical_uniqueness": lexical_uniqueness(texts),
        "structural_disorder": structural_disorder(texts),
        "outcome_variance_by_area": outcome_variance,
        "task_kinds_observed": kinds,
        "chance_retrieval_accuracy": round(1.0 / kinds, 4) if kinds else None,
    }


def memory_health(connection: sqlite3.Connection, since: str) -> dict[str, Any]:
    """Whether recall works at all, and whether the vocabulary stayed canonical."""
    total = int(_one(connection, "SELECT COUNT(*) FROM memories"))
    kinds = {
        str(row["kind"]): int(row["n"])
        for row in _rows(connection, "SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind ORDER BY n DESC")
    }
    retrieval = _rows(
        connection,
        "SELECT payload FROM event_log WHERE created_at >= ? AND kind=? ORDER BY sequence DESC LIMIT 200",
        (since, MEMORY_RETRIEVAL_KIND),
    )
    hits: list[int] = []
    for row in retrieval:
        try:
            payload = json.loads(row["payload"])
        except (TypeError, json.JSONDecodeError):
            continue
        hits.append(int(payload.get("hits", 0) or 0))
    return {
        "memories": total,
        "distinct_kinds": len(kinds),
        "kinds": kinds,
        "pinned": int(_one(connection, "SELECT COUNT(*) FROM memories WHERE pinned=1")),
        "retrieval_samples": len(hits),
        "retrieval_median_hits": _median(hits),
        "retrieval_zero_hit_runs": sum(1 for item in hits if item == 0),
    }


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return round((ordered[middle - 1] + ordered[middle]) / 2, 2)


def delivery_health(connection: sqlite3.Connection) -> dict[str, Any]:
    """The owner channel: an outbox nobody drains is a silent organism."""
    outbox = {
        str(row["delivery_state"]): int(row["n"])
        for row in _rows(connection, "SELECT delivery_state, COUNT(*) AS n FROM outbox GROUP BY delivery_state")
    }
    alerts = {
        str(row["severity"]): int(row["n"])
        for row in _rows(
            connection,
            "SELECT severity, COUNT(*) AS n FROM alerts WHERE delivered_at IS NULL GROUP BY severity",
        )
    }
    return {
        "outbox": outbox,
        "outbox_pending": outbox.get("pending", 0) + outbox.get("delivering", 0),
        "outbox_dead": outbox.get("dead", 0),
        "alerts_pending": sum(alerts.values()),
        "alerts_pending_by_severity": alerts,
        "inbox_pending": int(_one(connection, "SELECT COUNT(*) FROM inbox WHERE consumed_at IS NULL")),
    }


def proposal_outcomes(registry_path: Path | str) -> dict[str, int]:
    """Promotion outcomes live in the registry JSON, not in SQL."""
    try:
        data = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    counts: dict[str, int] = {}
    for record in data.values():
        if not isinstance(record, dict):
            continue
        status = str(record.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def resource_usage(state_dir: Path | str) -> dict[str, Any]:
    """Sizes that decide whether the experiment can keep running."""
    directory = Path(state_dir)
    database = directory / "skynet.sqlite3"
    if not database.exists():
        candidates = sorted(directory.glob("*.sqlite3"))
        database = candidates[0] if candidates else database
    sizes = {"db_bytes": database.stat().st_size if database.exists() else 0}
    total = 0
    for suffix in ("-wal", "-shm"):
        sibling = Path(str(database) + suffix)
        if sibling.exists():
            sizes[f"db{suffix}_bytes"] = sibling.stat().st_size
            total += sizes[f"db{suffix}_bytes"]
    total += sizes["db_bytes"]
    try:
        directory_bytes = sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())
    except OSError:
        directory_bytes = total
    sizes["state_dir_bytes"] = directory_bytes
    usage = shutil.disk_usage(directory if directory.exists() else Path("."))
    sizes["disk_free_bytes"] = usage.free
    sizes["disk_total_bytes"] = usage.total
    return sizes


def snapshot(
    store: Any,
    *,
    since_days: float = 1.0,
    registry_path: Path | str | None = None,
    state_dir: Path | str | None = None,
) -> dict[str, Any]:
    """One reproducible picture of the experiment."""
    connection = store.connection
    since = (utc_datetime_now() - timedelta(days=max(0.0, since_days))).isoformat().replace("+00:00", "Z")
    try:
        state = store.state()
    except Exception:
        state = None
    root = Path(state_dir) if state_dir is not None else Path(_state_dir_from(connection))
    data: dict[str, Any] = {
        "generated_at": utc_now(),
        "window_days": round(since_days, 3),
        "since": since,
        "outcomes": outcome_mix(connection, since),
        "deviations": deviations(connection, since),
        "providers": provider_failures(connection, since),
        "livelock": livelock_streaks(connection, since),
        "planning": fingerprint_pressure(connection),
        "diversity": diversity_health(connection, since),
        "memory": memory_health(connection, since),
        "delivery": delivery_health(connection),
    }
    if registry_path is not None:
        data["proposals"] = proposal_outcomes(registry_path)
    data["resources"] = resource_usage(root)
    if state is not None:
        generation = int(getattr(state, "generation", 0) or 0)
        epoch_span = max(1, int(os.getenv("SKYNET_CRITERION_EPOCH_GENERATIONS", "100") or 100))
        data["agent"] = {
            "lifecycle": getattr(state.lifecycle, "value", str(state.lifecycle)),
            "generation": generation,
            "criterion_epoch": generation // epoch_span,
            "criterion_epoch_generations": epoch_span,
            "retry_count": int(getattr(state, "retry_count", 0) or 0),
            "next_wake_at": getattr(state, "next_wake_at", None),
            "active_run_id": getattr(state, "active_run_id", None),
        }
    return data


def _state_dir_from(connection: sqlite3.Connection) -> str:
    """Derive the state directory from the open database filename."""
    try:
        filename = connection.execute("PRAGMA database_list").fetchone()[2]
    except (sqlite3.Error, IndexError, TypeError):
        return "state"
    if not filename:
        return "state"
    return str(Path(filename).parent)


def format_report(data: dict[str, Any], *, max_chars: int = 3800) -> str:
    """Render a snapshot as compact text for the owner channel."""
    lines: list[str] = []
    outcomes = data.get("outcomes", {})
    agent = data.get("agent", {})
    lines.append(f"SkyNet metrics — last {data.get('window_days', 1)}d @ {data.get('generated_at', '')}")
    if agent:
        lines.append(f"agent: {agent.get('lifecycle')} gen={agent.get('generation')} epoch={agent.get('criterion_epoch')} retry={agent.get('retry_count')}")
    rate = outcomes.get("completion_rate")
    lines.append(
        "runs: total={total} completed={rate} {by_status}".format(
            total=outcomes.get("total_runs", 0),
            rate=f"{rate:.0%}" if isinstance(rate, float) else "n/a",
            by_status=outcomes.get("by_status", {}),
        )
    )
    lines.append(
        "tokens: total={total} per_completed={per} wasted={wasted}".format(
            total=outcomes.get("tokens_total", 0),
            per=outcomes.get("tokens_per_completed_run"),
            wasted=outcomes.get("tokens_wasted", 0),
        )
    )
    providers = data.get("providers", {})
    if providers.get("classified"):
        lines.append(f"provider errors: {providers['classified']} zero_cooldown={providers.get('retryable_without_cooldown')}")
    if providers.get("degraded"):
        lines.append(f"provider health: DEGRADED {providers.get('degraded_categories')} — {providers.get('recommended_action')}")
    lines.extend(f"  {failure['n']}x {failure['failure']}" for failure in providers.get("top_failures", [])[:3])
    if data.get("deviations"):
        lines.append(f"deviations: {data['deviations']}")
    if data.get("livelock"):
        lines.append(f"livelock: {data['livelock'][:5]}")
    planning = data.get("planning", {})
    lines.append(f"hypotheses: {planning.get('hypotheses')} consumed={planning.get('hypothesis_space_consumed')}")
    lines.append(f"tasks: {planning.get('tasks')}")
    diversity = data.get("diversity", {})
    lines.append(
        "diversity: cells={filled}/{total} effective_modes={modes} external_evidence={external}".format(
            filled=diversity.get("cells_filled", 0),
            total=diversity.get("cells_total", 0),
            modes=diversity.get("effective_modes"),
            external=diversity.get("external_evidence_ratio"),
        )
    )
    memory = data.get("memory", {})
    lines.append(
        "memory: total={total} kinds={kinds} pinned={pinned} median_hits={hits} zero_hit_runs={zero}".format(
            total=memory.get("memories", 0),
            kinds=memory.get("distinct_kinds", 0),
            pinned=memory.get("pinned", 0),
            hits=memory.get("retrieval_median_hits"),
            zero=memory.get("retrieval_zero_hit_runs", 0),
        )
    )
    delivery = data.get("delivery", {})
    lines.append(
        "delivery: outbox_pending={pending} dead={dead} alerts_pending={alerts} inbox_pending={inbox}".format(
            pending=delivery.get("outbox_pending", 0),
            dead=delivery.get("outbox_dead", 0),
            alerts=delivery.get("alerts_pending", 0),
            inbox=delivery.get("inbox_pending", 0),
        )
    )
    if data.get("proposals"):
        lines.append(f"proposals: {data['proposals']}")
    resources = data.get("resources", {})
    if resources:
        lines.append(
            "resources: db={db}MiB state={state}MiB disk_free={free}GiB".format(
                db=round(resources.get("db_bytes", 0) / 1_048_576, 1),
                state=round(resources.get("state_dir_bytes", 0) / 1_048_576, 1),
                free=round(resources.get("disk_free_bytes", 0) / 1_073_741_824, 1),
            )
        )
    text = "\n".join(lines)
    return text[:max_chars]


def daily_marker(day: str | None = None) -> str:
    """The evaluation summary used to make the daily snapshot idempotent."""
    return f"experiment_metrics:{day or utc_now()[:10]}"


def record_daily_snapshot(
    store: Any,
    *,
    since_days: float = 1.0,
    registry_path: Path | str | None = None,
    state_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    """Write one snapshot per UTC day into ``evaluations``; return it if written."""
    marker = daily_marker()
    existing = _one(
        store.connection,
        "SELECT COUNT(*) FROM evaluations WHERE summary=?",
        (marker,),
    )
    if int(existing or 0):
        return None
    data = snapshot(store, since_days=since_days, registry_path=registry_path, state_dir=state_dir)
    store.record_metrics_snapshot(marker, data)
    return data
