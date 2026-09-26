from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

# ``memory_provenance`` resolves a batch of source_run values with one lookup per
# chunk; SQLite's default variable limit is 999, so the chunk stays well below it.
PROVENANCE_LOOKUP_CHUNK = 400

from .memory_store import MemoryStore, normalize_memory_kind
from .models import AgentState, LifecycleState, RunRecord, RunStatus
from .runtime_log import RuntimeLog
from .time import utc_datetime_now, utc_now

# A ledger row whose result carries one of these markers records a refusal
# decision, not an application: `skynet/tools.py` returns before executing the
# command for BOTH the soft denylist (`policy_warning`, tools.py:234) and the
# resurrection hard denylist (`policy_denied`, tools.py:190). Such a row must
# therefore not count as a prior application. The hard marker was missing from
# both effect-identity readers after commit 0ee707191708c0bf2f7fe4cca1367ce77902a6
# fixed only the soft one; measured on the live ledger (generation
# 92), 10 rows carry `policy_denied` and all 10 are the first and only row of
# their identity, so the omission is latent rather than active. Excluding both
# changes 0 of the 42 announced identities and 0 same_run_repeat decisions.
_REFUSAL_RESULT_MARKERS = ("policy_warning", "policy_denied")
_REFUSAL_RESULT_PREDICATE = " AND ".join(
    f"json_extract({{alias}}.result, '$.{marker}') IS NULL" for marker in _REFUSAL_RESULT_MARKERS
)
# The durable label that separates a refusal from an application. A response
# body is not an application: a row carrying this status never ran the tool, so
# no reader may count it as work. The label is derived from the same markers the
# SQL predicate above reads, and both are kept: the label is what the writer
# stamps from now on, the marker predicate is what the 63 refusal rows already
# on disk (measured, generation 93) are still recognised by.
REFUSED_EFFECT_STATUS = "refused"

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS agent_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    lifecycle TEXT NOT NULL,
    generation INTEGER NOT NULL,
    next_wake_at TEXT,
    active_run_id TEXT,
    next_plan TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    budget TEXT NOT NULL,
    heartbeat_at TEXT,
    last_phase TEXT,
    last_progress_at TEXT,
    steps INTEGER NOT NULL DEFAULT 0,
    usage_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS run_results (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    report TEXT NOT NULL,
    steps INTEGER NOT NULL,
    usage_tokens INTEGER NOT NULL,
    failure TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS event_log (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transcript (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inbox (
    event_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    consumed_at TEXT
);
CREATE TABLE IF NOT EXISTS goals (
    goal_id TEXT PRIMARY KEY, title TEXT NOT NULL, status TEXT NOT NULL,
    priority REAL NOT NULL DEFAULT 0, constraints TEXT NOT NULL DEFAULT '{}',
    next_action TEXT NOT NULL DEFAULT '', outcome TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY, goal_id TEXT, title TEXT NOT NULL,
    status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
    consecutive_model_failures INTEGER NOT NULL DEFAULT 0,
    deadline TEXT, idempotency_key TEXT UNIQUE, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, area TEXT NOT NULL DEFAULT 'general'
);
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL,
    confidence REAL NOT NULL, source_run TEXT, updated_at TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    superseded_by TEXT,
    valid_from TEXT,
    valid_to TEXT,
    evidence TEXT,
    decayed_at TEXT,
    UNIQUE(kind, content)
);
CREATE TABLE IF NOT EXISTS affect_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    valence REAL NOT NULL DEFAULT 0.5,
    draws INTEGER NOT NULL DEFAULT 0,
    source_run TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS rng_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    seed INTEGER NOT NULL,
    draws INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS user_questions (
    question_id TEXT PRIMARY KEY,
    run_id TEXT,
    task_id TEXT,
    goal_id TEXT,
    question TEXT NOT NULL,
    options TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    answered_at TEXT,
    answer TEXT,
    source TEXT
);
CREATE INDEX IF NOT EXISTS idx_user_questions_open ON user_questions(status, created_at);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    dedup_key TEXT NOT NULL UNIQUE,
    payload TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'warning',
    occurrences INTEGER NOT NULL DEFAULT 1,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    delivered_at TEXT,
    channel TEXT
);
CREATE TABLE IF NOT EXISTS memory_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS memory_consolidations (
    episode_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    input_version INTEGER NOT NULL,
    output_version INTEGER NOT NULL,
    memory_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episode_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    first_sequence INTEGER,
    last_sequence INTEGER,
    payload_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_reconciliations (
    run_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    status TEXT NOT NULL,
    result TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(run_id, call_id)
);
CREATE TABLE IF NOT EXISTS outbox (
    message_id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
    delivery_state TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL,
    delivered_at TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    claimed_at TEXT
);
CREATE TABLE IF NOT EXISTS capability_effects (
    idempotency_key TEXT PRIMARY KEY, capability TEXT NOT NULL,
    arguments_hash TEXT NOT NULL, result TEXT NOT NULL, status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    summary TEXT NOT NULL,
    evaluation TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY, workstream_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL UNIQUE, structural_fingerprint TEXT NOT NULL,
    problem TEXT NOT NULL, expected_behavior TEXT NOT NULL,
    status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
    last_outcome TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS planner_decisions (
    decision_id TEXT PRIMARY KEY, generation INTEGER NOT NULL,
    selected_workstream_id TEXT, selected_task_id TEXT,
    candidates TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS planner_resets (
    reset_id TEXT PRIMARY KEY, reset_key TEXT NOT NULL UNIQUE,
    reason TEXT NOT NULL, proposal_id TEXT, commit_hash TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS planner_attempts (
    attempt_id TEXT PRIMARY KEY, generation INTEGER NOT NULL, trigger TEXT NOT NULL,
    status TEXT NOT NULL, strategy TEXT NOT NULL DEFAULT '', failure TEXT NOT NULL DEFAULT '',
    proposal_count INTEGER NOT NULL DEFAULT 0, started_at TEXT NOT NULL, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS planner_proposals (
    proposal_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL, goal_id TEXT NOT NULL,
    title TEXT NOT NULL, problem TEXT NOT NULL, hypothesis TEXT NOT NULL,
    expected_new_fact TEXT NOT NULL, validation TEXT NOT NULL, scope TEXT NOT NULL,
    kind TEXT NOT NULL, hypothesis_fingerprint TEXT NOT NULL, structural_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL, rejection_reason TEXT NOT NULL DEFAULT '', created_task_id TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS idea_archive (
    idea_id TEXT PRIMARY KEY,
    parent_id TEXT,
    lineage_depth INTEGER NOT NULL DEFAULT 0,
    subsystem TEXT NOT NULL DEFAULT 'general',
    change_type TEXT NOT NULL DEFAULT 'workflow',
    evidence_source TEXT NOT NULL DEFAULT 'own-repo',
    cell_key TEXT NOT NULL,
    title TEXT NOT NULL,
    problem_description TEXT NOT NULL DEFAULT '',
    hypothesis TEXT NOT NULL DEFAULT '',
    expected_new_fact TEXT NOT NULL DEFAULT '',
    validation TEXT NOT NULL DEFAULT '',
    inspiration_ref TEXT NOT NULL DEFAULT '',
    quality REAL NOT NULL DEFAULT 0.0,
    novelty REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'active',
    superseded_by TEXT,
    children INTEGER NOT NULL DEFAULT 0,
    task_id TEXT,
    proposal_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_idea_cell ON idea_archive(cell_key, quality DESC);
CREATE INDEX IF NOT EXISTS idx_idea_status ON idea_archive(status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_idea_source ON idea_archive(evidence_source, status);

CREATE INDEX IF NOT EXISTS idx_event_log_run ON event_log(run_id);
CREATE INDEX IF NOT EXISTS idx_event_log_kind ON event_log(kind, sequence DESC);
CREATE INDEX IF NOT EXISTS idx_transcript_run ON transcript(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_stale ON runs(status, heartbeat_at);
CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id);
-- effect_reapplied_runs (below) looks up a call identity by (capability,
-- arguments_hash) and orders the distinct runs by created_at; without this
-- index that is a full SCAN of a monotonically growing table on the tool-call
-- path (measured 2026-09-20 on the live ledger, 2101 rows: 1.328 ms/call,
-- SCAN + two temp B-trees; with the index 0.014 ms/call, ~97x, covering SEARCH).
CREATE INDEX IF NOT EXISTS idx_capability_effects_identity ON capability_effects(capability, arguments_hash, created_at, idempotency_key);
CREATE INDEX IF NOT EXISTS idx_outbox_pending ON outbox(delivery_state, created_at);
CREATE INDEX IF NOT EXISTS idx_inbox_pending ON inbox(consumed_at);
CREATE INDEX IF NOT EXISTS idx_hypotheses_structural ON hypotheses(structural_fingerprint);
CREATE INDEX IF NOT EXISTS idx_proposals_fingerprint ON planner_proposals(hypothesis_fingerprint);
CREATE INDEX IF NOT EXISTS idx_alerts_pending ON alerts(delivered_at, severity);
CREATE INDEX IF NOT EXISTS idx_planner_decisions_created ON planner_decisions(created_at);
"""

# Indexes on columns that only exist after an `ALTER TABLE ... ADD COLUMN` must
# be created after the migration ran: on an upgraded database the base SCHEMA
# executes first and `CREATE TABLE IF NOT EXISTS` will not add the new column,
# so creating these indexes inside SCHEMA would abort the store at startup.
POST_MIGRATION_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_tasks_area ON tasks(area, status);
CREATE INDEX IF NOT EXISTS idx_memories_pinned ON memories(pinned, updated_at DESC);
"""

SCHEMA_VERSION = 13
# v13 adds `affect_state`: the one persisted valence channel, stored as
# `(value, draws)`. Like v10 it is a brand-new table, so `executescript(SCHEMA)`
# creates it with every column on an upgraded database and there is no ALTER to
# catch up on.
MIGRATION_NAMES = {3: "durable-boundaries", 4: "measured-liveness", 5: "planning-rng", 6: "owner-dialogue", 7: "dead-persistence", 8: "drop-dead-checkpoints", 9: "memory-validity", 10: "idea-archive", 11: "memory-decay-clock", 12: "run-progress", 13: "affect-valence"}

# `evaluations.run_id` is NOT NULL; metrics snapshots belong to no run, so they
# share one sentinel instead of inventing a fake run identity.
METRICS_RUN_ID = "metrics"

# A lease left in `delivering` by a crashed process is recovered exactly once,
# when this process first opens the store. Doing it on every writable open
# reset a live in-flight lease as soon as the reactor and telegram processes
# opened the same file, so the same message could be delivered twice.
#
# The flag is per-process, and the reactor and the telegram bot are separate
# processes: each starts with it unset, so the first writable open in *each* of
# them ran the unconditional UPDATE. Measured (generation 51) on a
# scratch store: the reactor claims a message (`delivering`, `claimed_at` set),
# a second process opens the same file and the row is back to `pending` with
# `claimed_at` untouched, and that process immediately re-claims it -- exactly
# the double delivery the docstring forbids. Startup recovery therefore
# recovers only leases older than the claim window, which is what actually
# distinguishes a dead sender from a live one.
_INFLIGHT_LEASES_RECOVERED = False

# How long a claimed outbox row may stay `delivering` before another consumer
# may take it. Shared by `claim` and by startup recovery so the two can never
# disagree about which leases are live.
OUTBOX_LEASE_SECONDS = 300.0

# The run-count window that bounds how long a protected pointer's *evidence*
# lives. It mirrors the reactor's ``transcript_retention_runs`` default and the
# ``SKYNET_TRANSCRIPT_RETENTION_RUNS`` the CLI reads; production callers pass the
# live setting explicitly so this value only covers direct StateStore use.
DEFAULT_TRANSCRIPT_RETENTION_RUNS = int(os.getenv("SKYNET_TRANSCRIPT_RETENTION_RUNS", "200"))

# Event kinds that survive every retention window: the post-mortem of a failed
# experiment is exactly the set of things that went wrong.
PROTECTED_EVENT_KINDS = frozenset({
    "doom_loop", "budget_exhausted", "emergency_finish", "watchdog_timeout",
    "owner_stop",
    "uncaptured_changes", "finish_invalid", "finish_failure", "restart_failed",
    # The pointer to an episode whose report had to be reconstructed from its own
    # tool calls: pruning it erases which runs lost their closing turn.
    "finish_evidence_carried",
    # A Finish Report action contradicted by the run's own capability ledger:
    # the post-mortem must survive retention exactly like uncaptured_changes.
    "report_claim_unverified",
    "restart_requested", "restart_escalated", "restart_skipped", "cycle_error", "provider_lockout",
    "provider_failure", "provider_error_classified", "worktree_dirty_after_bash",
    # The post-mortem of an episode discarded before it could try anything. It is
    # run-scoped where `fallback_all_cooling` is not (that event is written by the
    # chain with run_id NULL), so it is the only durable row naming which run and
    # which task paid for the chain-wide cooldown.
    "provider_unreachable",
    "livelock_suspected", "memory_loop_failed", "memory_degraded", "policy_denied", "policy_soft_denied",
    # The episode the memory loop could not capture. Retention must not erase
    # the only pointer back to it: pruning this row deletes the work it names.
    "memory_capture_pending",
    # The end of that pointer's life. `pending_memory_captures` excludes a
    # pointer once it carries a disposition, so without this row surviving
    # retention the bound would be invisible: a re-attempt that failed again
    # would look identical to one that was never made.
    "memory_capture_released",
    "policy_override_confirmed", "registry_corrupted", "mcp_server_unavailable",
    "tool_name_collision", "outbox_dead_letter", "outbox_backlog_suppressed",
    "alert_raised", "task_gave_up", "success_criteria_failed", "goal_proposal_rejected",
    # A planning attempt that crashed and one that genuinely found no work are the
    # same empty portfolio; only these kinds tell them apart, so retention must not
    # drop the evidence that the planner failed.
    "planner_invalid_response", "planner_provider_error", "planner_output_truncated", "planner_fallback_capped",
    # The backoff walk and the fallback creation/expiry sequence are the durable
    # trail of a stall: together they show how long the safety net was capped
    # and how the process kept sleeping. Retention must not erase the evidence
    # of the deadlock before the fix is judged.
    "planner_backoff", "planner_fallback_created",
    # The shape of the planner's own model turn: the only durable record that
    # separates an empty reply from one whose tokens went to the reasoning
    # channel, so retention must not erase the planner post-mortem.
    "planner_turn",
    # The query the PLANNER searched, with the ablation that says whether its
    # previous-outcome part reaches the recall window at all. It is an instrument
    # of the planner contour like `planner_turn`: generations 191-223 could not
    # answer a part-order question about this query without rebuilding it from
    # the StartEnvelope, and the rebuild's own front cap and ledger copy change
    # the verdict, so retention must not erase what was actually searched.
    "planner_memory_recall",
    # The watchdog's only durable trace that the gate-protected planner contour
    # degraded; the alert row is deduped, so this event is the post-mortem.
    "judge_health_degraded",
    # A protected-path proposal is only warned, not rejected; the warning and
    # the acknowledgement of the identical resubmission are the durable trace
    # of the warn-once mechanism. The retired soft-denial/approval kinds stay
    # listed so history is never pruned.
    # A maintenance/retention pass that raised. Each pass catches its own
    # exception so the cycle cannot die, and until now the only trace was a
    # journald warning (``deploy/skynet.service.in`` sends stdout there):
    # measured on the live store, 0 durable rows name any retention failure, so
    # a pass that failed or silently released rows was invisible to retention
    # itself. The row is an instrument of the maintenance contour and must
    # outlive the window it reports on.
    "maintenance_failed",
    "gate_protected_warned", "gate_protected_warning_acknowledged",
    "gate_protected_soft_denied", "gate_protected_approved",
    # The A1-lite plan artifact is the measurement substrate for whether an
    # explicit plan improves a run; retention must not erase the evidence
    # before the owner can judge the experiment.
    "plan_recorded", "plan_observation",
    # The append-only memory audit is the only durable trace of a memory change
    # made outside a checkpoint; retention must not erase the observation.
    "memory_audit_external_change", "memory_audit_anomaly",
    # A chain that no longer verifies is the durable trace of a rewrite of the
    # audit log itself; that observation is the whole point, so retention must
    # not erase it.
    "memory_audit_chain_broken",
    "memory_kinds_normalized", "improvement_proposals_reconciled",
    "improvement_environment_blocks_resolved",
    "recovery_reconciliation", "run_failure", "run_interrupted", "metrics_snapshot",
    "effect_reapplied",
    "provider_error", "provider_retry", "fallback_failure", "fallback_skipped",
    "provider_output_truncated",
    # Chain-level diagnostics that used to be runtime-only: a dead or fully
    # cooling provider chain is exactly the post-mortem a failed experiment
    # depends on, so retention must not drop it.
    "fallback_all_cooling", "provider_chain_reloaded",
    # Supervisor lifecycle: whether the process actually started, stopped and
    # observed a reboot window is the durable frame around every cycle.
    "supervisor_start", "supervisor_stop", "reboot_observation",
    # The idea archive is the organism's lineage: losing it would erase which
    # stepping stones were tried and why, so it survives every retention window.
    "idea_archived", "idea_superseded", "idea_archive_rejected", "idea_materialized",
    "external_seek_created", "external_seek_suppressed", "external_senses_unavailable", "idea_learnability_rejected",
})

# Provider-chain diagnostics that are low-volume and diagnostically load-bearing:
# the reactor's provider hook routes these through ``append_event`` so they
# survive retention, while the high-volume attempt/selection/request kinds stay
# in the advanced runtime projection only. ``append_event`` already mirrors a row
# into that projection as kind ``event``, so the promoted kinds must NOT also be
# written with ``runtime_log.write`` (that would duplicate every row).
DURABLE_PROVIDER_EVENT_KINDS = frozenset({
    "fallback_failure",
    # The complement of ``fallback_failure``: a retry that answered. It shares
    # the failure event's durability so `retry success rate` -- the metric the
    # retry-amplification paper requires to reconstruct a retry storm
    # (arXiv:2608.25403, sec. 8.1) -- is answerable from ``event_log`` alone
    # instead of only from the rotating runtime projection.
    "fallback_retry_recovered",
    "fallback_all_cooling",
    # The chain-wide cooldown the fallback chose to wait out, and the near miss
    # it refused to wait out. ``fallback_all_cooling`` says an episode was
    # discarded; these two say whether the wait that exists to preserve it
    # actually fired, which is the counterfactual the ablation needs.
    "fallback_cooldown_wait",
    "fallback_cooldown_wait_skipped",
    "provider_chain_reloaded",
    # A reply cut off at the output ceiling is the post-mortem of an
    # output-ceiling incident. It was the only record of the stall
    # and it lived in the rotating runtime log, so neither retention nor any
    # event-log reader could find it.
    "provider_output_truncated",
})


def _model_evidence(value: Any, *, limit: int = 2000) -> str | None:
    """Normalize a model-authored ``evidence`` field into a storable string.

    The memory-loop response schema declares ``evidence`` on every candidate
    (``skynet/model_contracts.py``: an array of strings), but this consolidation
    path only ever read it for a *supersede* note and never wrote it to the new
    row: the SQL statement behind it has no ``evidence`` column at all. Measured
    on the live store: of 44 belief-kind rows written after commit
    3c8f3fa, 39 carried no evidence, so a claim and its support were separable at
    the moment of writing. The ``remember_memory`` path already stores evidence,
    which is why the loss stayed invisible to ``MemoryTool``.

    A model-authored field is untrusted: a list, a tuple or a bare string is
    accepted, anything else yields ``None`` so the row is written unattributed
    rather than with a fabricated citation. Empty text yields ``None`` too, so a
    candidate that cites nothing keeps ``evidence IS NULL`` and stays visible to
    the provenance counts in ``memory_audit.provenance_counts``.
    """
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (list, tuple)):
        text = "; ".join(str(part).strip() for part in value if str(part).strip())
    else:
        return None
    if not text:
        return None
    return text[:limit]


def _bounded_model_float(value: Any, *, default: float) -> float:
    """Normalize numeric fields coming from model-authored JSON."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


_ARCHIVE_EPSILON = 0.1


def _admits_over_incumbent(
    candidate_quality: float,
    candidate_novelty: float,
    incumbent_quality: float,
    incumbent_novelty: float,
    *,
    epsilon: float = _ARCHIVE_EPSILON,
) -> bool:
    """Exclusive epsilon-dominance over (quality, novelty), Cully & Demiris.

    arXiv:1708.09251 sec. 3.1.2 "The Archive": managing a collection by
    replacing a member only when the newcomer is Pareto-superior "is very
    difficult to reach, as the new individual should be both better and more
    diverse than the previous one. This prevents most new individuals from
    being added to the collection, which limits the quality of the produced
    collections." Their softened rule admits a slightly worse candidate when it
    is strictly more novel: x1 dominates x2 iff
        N(x1) >= (1-eps) * N(x2)  and  Q(x1) >= (1-eps) * Q(x2)
        and (N(x1)-N(x2)) * Q(x2) > -(Q(x1)-Q(x2)) * N(x2)
    With novelty constant (as the archive call site currently passes), the
    third condition collapses to the quality-only comparison, so this is a
    strict relaxation of the previous rule: it can only ever add admissions,
    never remove one, and never rotates a cell on an exact tie.
    """
    if incumbent_quality <= 0.0 and incumbent_novelty <= 0.0:
        # A degenerate incumbent holds the cell without contributing anything;
        # keeping it would make the cell permanently unreachable.
        return True
    if candidate_quality < (1.0 - epsilon) * incumbent_quality:
        return False
    if candidate_novelty < (1.0 - epsilon) * incumbent_novelty:
        return False
    improvement = (candidate_novelty - incumbent_novelty) * incumbent_quality + (
        candidate_quality - incumbent_quality
    ) * incumbent_novelty
    if improvement > 0.0:
        return True
    # Equal novelty (including the no-signal case where both are zero) leaves
    # the weighted term at zero, so a strictly better quality still wins the
    # cell. This keeps the previous quality-only behaviour intact and means the
    # relaxation can only ever ADD admissions, never remove one.
    return candidate_quality > incumbent_quality


class EventRepository:
    """Append/read event-log rows for one connection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def append(self, kind: str, payload: dict[str, Any], run_id: str | None = None) -> int:
        cursor = self.connection.execute(
            "INSERT INTO event_log(run_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
            (run_id, kind, json.dumps(payload, ensure_ascii=False), utc_now()),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return an event sequence")
        return int(cursor.lastrowid)

    def recent(self, run_id: str, limit: int | None = 20) -> list[dict[str, Any]]:
        query = "SELECT sequence, kind, payload, created_at FROM event_log WHERE run_id=? ORDER BY sequence DESC"
        params: tuple[Any, ...] = (run_id,)
        if limit is not None:
            query += " LIMIT ?"
            params += (limit,)
        rows = self.connection.execute(query, params).fetchall()
        events = []
        for row in reversed(rows):
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                payload = {"raw": row["payload"]}
            events.append({"sequence": row["sequence"], "kind": row["kind"], "payload": payload, "created_at": row["created_at"]})
        return events


class RunRepository:
    """Run lifecycle rows for one connection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(self, run: RunRecord) -> None:
        self.connection.execute(
            "INSERT INTO runs(run_id, attempt, status, started_at, finished_at, budget, heartbeat_at, last_phase, last_progress_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run.run_id, run.attempt, run.status.value, run.started_at, run.finished_at, json.dumps({"steps": run.budget.steps, "tokens": run.budget.tokens, "seconds": run.budget.seconds}), utc_now(), "initial", utc_now()),
        )

    def touch(self, run_id: str, phase: str, *, progress: bool = False, steps: int | None = None, usage_tokens: int | None = None) -> None:
        now = utc_now()
        self.connection.execute(
            "UPDATE runs SET heartbeat_at=?, last_phase=?, "
            "last_progress_at=CASE WHEN ? THEN ? ELSE last_progress_at END, "
            "steps=COALESCE(?, steps), usage_tokens=COALESCE(?, usage_tokens) "
            "WHERE run_id=? AND status='running'",
            (now, phase, progress, now, steps, usage_tokens, run_id),
        )

    def finish(self, run_id: str, status: RunStatus) -> None:
        self.connection.execute("UPDATE runs SET status=?, finished_at=? WHERE run_id=?", (status.value, utc_now(), run_id))

    def result(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM run_results WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None


class OutboxRepository:
    """Outbox queue rows for one connection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def add(self, kind: str, payload: dict[str, Any], message_id: str | None = None) -> str:
        message_id = message_id or str(uuid4())
        self.connection.execute("INSERT OR IGNORE INTO outbox(message_id, kind, payload, delivery_state, created_at) VALUES (?, ?, ?, 'pending', ?)", (message_id, kind, json.dumps(payload), utc_now()))
        return message_id

    def pending(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM outbox WHERE delivery_state='pending' ORDER BY created_at LIMIT ?", (limit,)).fetchall()
        return [{"message_id": row["message_id"], "kind": row["kind"], "payload": json.loads(row["payload"])} for row in rows]

    def claim(self, limit: int = 50, lease_seconds: float = OUTBOX_LEASE_SECONDS) -> list[dict[str, Any]]:
        expired_before = (utc_datetime_now() - timedelta(seconds=max(1.0, lease_seconds))).isoformat().replace("+00:00", "Z")
        rows = self.connection.execute("SELECT * FROM outbox WHERE delivery_state='pending' OR (delivery_state='delivering' AND claimed_at < ?) ORDER BY created_at LIMIT ?", (expired_before, limit)).fetchall()
        messages = []
        for row in rows:
            updated = self.connection.execute("UPDATE outbox SET delivery_state='delivering', attempts=attempts+1, claimed_at=? WHERE message_id=? AND (delivery_state='pending' OR (delivery_state='delivering' AND claimed_at < ?))", (utc_now(), row["message_id"], expired_before)).rowcount
            if updated:
                messages.append({"message_id": row["message_id"], "kind": row["kind"], "payload": json.loads(row["payload"])})
        return messages

    def mark_delivered(self, message_id: str) -> None:
        self.connection.execute("UPDATE outbox SET delivery_state='delivered', delivered_at=?, claimed_at=NULL WHERE message_id=?", (utc_now(), message_id))

    def mark_failed(self, message_id: str, error: str, *, max_attempts: int = 5) -> str:
        """Return the message to the queue, or park it as dead after the cap.

        Without the cap a permanently undeliverable message was retried on every
        cycle forever, growing `attempts` without any observable consequence.
        """
        row = self.connection.execute(
            "SELECT attempts FROM outbox WHERE message_id=?", (message_id,)
        ).fetchone()
        attempts = int(row["attempts"]) if row is not None else 0
        state = "dead" if attempts >= max(1, max_attempts) else "pending"
        self.connection.execute(
            "UPDATE outbox SET delivery_state=?, last_error=?, claimed_at=NULL WHERE message_id=?",
            (state, error[:1000], message_id),
        )
        return state


class AlertRepository:
    """Durable, de-duplicated escalation records.

    The table owns history and de-duplication; delivery stays with the outbox so
    there is still exactly one delivery path.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def raise_alert(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        severity: str = "warning",
        dedup_key: str | None = None,
        dedup_window_seconds: float = 21_600.0,
    ) -> tuple[str, bool, int]:
        """Record an alert; collapse repeats inside the dedup window.

        Returns ``(alert_id, is_new_or_rearmed, occurrences)``. A collapsed
        alert does not re-enter the delivery queue, it only counts up.
        """
        key = str(dedup_key or kind)
        encoded = json.dumps(payload, ensure_ascii=False)
        now = utc_now()
        cutoff = (utc_datetime_now() - timedelta(seconds=max(0.0, dedup_window_seconds))).isoformat().replace("+00:00", "Z")
        row = self.connection.execute(
            "SELECT alert_id, delivered_at, occurrences FROM alerts WHERE dedup_key=?", (key,)
        ).fetchone()
        if row is not None:
            delivered_at = row["delivered_at"]
            if delivered_at is None or str(delivered_at) >= cutoff:
                occurrences = int(row["occurrences"]) + 1
                self.connection.execute(
                    "UPDATE alerts SET occurrences=?, last_seen_at=?, severity=?, payload=? WHERE alert_id=?",
                    (occurrences, now, severity, encoded, row["alert_id"]),
                )
                return str(row["alert_id"]), False, occurrences
            self.connection.execute(
                "UPDATE alerts SET occurrences=1, first_seen_at=?, last_seen_at=?, delivered_at=NULL, "
                "channel=NULL, severity=?, payload=? WHERE alert_id=?",
                (now, now, severity, encoded, row["alert_id"]),
            )
            return str(row["alert_id"]), True, 1
        alert_id = str(uuid4())
        self.connection.execute(
            "INSERT INTO alerts(alert_id, kind, dedup_key, payload, severity, occurrences, first_seen_at, last_seen_at) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
            (alert_id, kind, key, encoded, severity, now, now),
        )
        return alert_id, True, 1

    def pending(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM alerts WHERE delivered_at IS NULL ORDER BY "
            "CASE severity WHEN 'critical' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END, last_seen_at LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item["payload"])
            except (TypeError, json.JSONDecodeError):
                item["payload"] = {"raw": item["payload"]}
            result.append(item)
        return result

    def mark_delivered(self, alert_id: str, channel: str) -> None:
        self.connection.execute(
            "UPDATE alerts SET delivered_at=?, channel=? WHERE alert_id=?", (utc_now(), channel, alert_id)
        )

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT alert_id, kind, severity, occurrences, first_seen_at, last_seen_at, delivered_at, channel "
            "FROM alerts ORDER BY last_seen_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


class StateStore:
    """SQLite canonical state boundary; JSONL is only a best-effort projection."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self.runtime_log = RuntimeLog(self.path.parent / "runtime.jsonl", kinds=os.getenv("SKYNET_RUNTIME_LOG_KINDS"))
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, isolation_level=None)
        else:
            self.connection = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        # The connection belongs to this thread. The provider chain also runs on
        # the memory-loop and planner worker threads, so a durable diagnostic
        # from there must not touch this object (see record_provider_event).
        self._connection_thread = threading.get_ident()
        # The watchdog thread and the telegram process open the same file; a
        # short busy timeout turns a transient write lock into a wait instead of
        # an immediate "database is locked" failure. synchronous=NORMAL is the
        # documented WAL trade-off: durability across process crashes is kept,
        # only a power loss can drop the last transaction.
        self.connection.execute("PRAGMA busy_timeout=5000")
        if not read_only:
            self.connection.execute("PRAGMA synchronous=NORMAL")
        self.events = EventRepository(self.connection)
        self.runs = RunRepository(self.connection)
        self.outbox = OutboxRepository(self.connection)
        self.alerts = AlertRepository(self.connection)
        if not read_only:
            self.connection.executescript(SCHEMA)
            self._migrate_schema()
            self._ensure_state()
            self._recover_inflight_leases_once()
        self.memory_store = MemoryStore(self.connection) if not read_only else None
        if self.memory_store is not None:
            normalized = self.memory_store.normalize_legacy_kinds()
            if normalized:
                self.append_event("memory_kinds_normalized", {"rows": normalized})

    def _migrate_schema(self) -> None:
        migration_checksum = hashlib.sha256(SCHEMA.encode()).hexdigest()
        self.connection.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL, applied_at TEXT NOT NULL)")
        migration = self.connection.execute("SELECT checksum FROM schema_migrations WHERE version=?", (SCHEMA_VERSION,)).fetchone()
        if migration and migration[0] != migration_checksum:
            # Additive schema changes are applied by `executescript(SCHEMA)`;
            # only refresh the recorded checksum for an existing database.
            self.connection.execute("UPDATE schema_migrations SET checksum=?, applied_at=? WHERE version=?", (migration_checksum, utc_now(), SCHEMA_VERSION))
        state_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(agent_state)")}
        if "retry_count" not in state_columns:
            self.connection.execute("ALTER TABLE agent_state ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0")
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(outbox)")}
        if "attempts" not in columns:
            self.connection.execute("ALTER TABLE outbox ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
        if "last_error" not in columns:
            self.connection.execute("ALTER TABLE outbox ADD COLUMN last_error TEXT")
        run_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(runs)")}
        if "heartbeat_at" not in run_columns:
            self.connection.execute("ALTER TABLE runs ADD COLUMN heartbeat_at TEXT")
        if "last_phase" not in run_columns:
            self.connection.execute("ALTER TABLE runs ADD COLUMN last_phase TEXT")
        if "last_progress_at" not in run_columns:
            self.connection.execute("ALTER TABLE runs ADD COLUMN last_progress_at TEXT")
        if "steps" not in run_columns:
            # A run that dies mid-episode keeps only its final zeros: the
            # in-memory counters never reached a row. Persist the running totals
            # so an interrupted run's cost survives the process.
            self.connection.execute("ALTER TABLE runs ADD COLUMN steps INTEGER NOT NULL DEFAULT 0")
        if "usage_tokens" not in run_columns:
            self.connection.execute("ALTER TABLE runs ADD COLUMN usage_tokens INTEGER NOT NULL DEFAULT 0")
        if "claimed_at" not in columns:
            self.connection.execute("ALTER TABLE outbox ADD COLUMN claimed_at TEXT")
        reset_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(planner_resets)")}
        if "previous_context" in reset_columns:
            self.connection.execute("ALTER TABLE planner_resets DROP COLUMN previous_context")
        # The checkpoints table was write-only: nothing ever read a stored
        # checkpoint back, so it is dropped as dead persistence.
        self.connection.execute("DROP TABLE IF EXISTS checkpoints")
        # Relations were never requested by the Memory Loop schema, so the table
        # stayed empty and the read path always returned nothing; drop it.
        self.connection.execute("DROP TABLE IF EXISTS relations")
        self.connection.execute("INSERT OR IGNORE INTO memory_meta(id, version) VALUES (1, 0)")
        task_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(tasks)")}
        if "hypothesis_fingerprint" not in task_columns:
            self.connection.execute("ALTER TABLE tasks ADD COLUMN hypothesis_fingerprint TEXT")
        if "structural_fingerprint" not in task_columns:
            self.connection.execute("ALTER TABLE tasks ADD COLUMN structural_fingerprint TEXT")
        if "expected_new_fact" not in task_columns:
            self.connection.execute("ALTER TABLE tasks ADD COLUMN expected_new_fact TEXT NOT NULL DEFAULT ''")
        if "consecutive_model_failures" not in task_columns:
            self.connection.execute("ALTER TABLE tasks ADD COLUMN consecutive_model_failures INTEGER NOT NULL DEFAULT 0")
        if "area" not in task_columns:
            self.connection.execute("ALTER TABLE tasks ADD COLUMN area TEXT NOT NULL DEFAULT 'general'")
        memory_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(memories)")}
        if "pinned" not in memory_columns:
            self.connection.execute("ALTER TABLE memories ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        if "decayed_at" not in memory_columns:
            # Decay needs its own clock: reusing updated_at for idempotency
            # rewrote the column the recall 'recency' axis ranks on, so the pass
            # that punished a stale memory also promoted it in search results.
            self.connection.execute("ALTER TABLE memories ADD COLUMN decayed_at TEXT")
        for column, definition in (
            ("status", "TEXT NOT NULL DEFAULT 'active'"),
            ("superseded_by", "TEXT"),
            ("valid_from", "TEXT"),
            ("valid_to", "TEXT"),
            ("evidence", "TEXT"),
        ):
            if column not in memory_columns:
                self.connection.execute(f"ALTER TABLE memories ADD COLUMN {column} {definition}")
        self.connection.execute("UPDATE agent_state SET lifecycle='sleep' WHERE lifecycle='idle'")
        self.connection.execute("UPDATE tasks SET status='pending', updated_at=? WHERE status='needs_recovery'", (utc_now(),))
        self.connection.execute(
            "UPDATE planner_attempts SET status='failed', failure='process restarted during planning', finished_at=? WHERE status='started'",
            (utc_now(),),
        )
        # v10: `idea_archive` is a brand-new table, so `executescript(SCHEMA)`
        # above already created it with all columns; unlike an ALTER-added
        # column there is nothing to catch up on an upgraded database.
        self._backfill_task_fingerprints()
        self.connection.executescript(POST_MIGRATION_INDEXES)
        self.connection.execute("INSERT OR IGNORE INTO schema_migrations(version, name, checksum, applied_at) VALUES (?, ?, ?, ?)", (SCHEMA_VERSION, MIGRATION_NAMES[SCHEMA_VERSION], migration_checksum, utc_now()))

    def _backfill_task_fingerprints(self) -> None:
        """Make legacy tasks visible to planner novelty and repetition checks."""
        from .planner import hypothesis_fingerprint as make_hypothesis_fingerprint
        from .planner import structural_fingerprint as make_structural_fingerprint

        rows = self.connection.execute(
            "SELECT task_id, goal_id, title, status, expected_new_fact FROM tasks "
            "WHERE hypothesis_fingerprint IS NULL OR structural_fingerprint IS NULL"
        ).fetchall()
        for row in rows:
            title = str(row["title"])
            expected = str(row["expected_new_fact"] or title)
            hypothesis = make_hypothesis_fingerprint(area=title, problem=title, expected_behavior=expected)
            structural = make_structural_fingerprint(area=title, target=title, behavior_kind="task")
            self.connection.execute(
                "UPDATE tasks SET hypothesis_fingerprint=?, structural_fingerprint=?, expected_new_fact=? WHERE task_id=?",
                (hypothesis, structural, expected, row["task_id"]),
            )
            hypothesis_status = {
                "completed": "completed",
                "blocked": "rejected",
                "cancelled": "rejected",
            }.get(str(row["status"]), "ready")
            self.register_hypothesis(
                workstream_id=str(row["goal_id"] or "unscoped"),
                fingerprint=hypothesis,
                structural_fingerprint=structural,
                problem=title,
                expected_behavior=expected,
                status=hypothesis_status,
            )
            if hypothesis_status != "ready":
                self.connection.execute(
                    "UPDATE hypotheses SET status=?, updated_at=? WHERE fingerprint=?",
                    (hypothesis_status, utc_now(), hypothesis),
                )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _ensure_state(self) -> None:
        self.connection.execute(
            """INSERT OR IGNORE INTO agent_state
               (id, lifecycle, generation, next_wake_at, active_run_id, next_plan, retry_count)
               VALUES (1, ?, 0, NULL, NULL, '{}', 0)""",
            (LifecycleState.BOOT.value,),
        )

    def recover_inflight_outbox(self, *, stale_after_seconds: float | None = None) -> int:
        """Return `delivering` outbox leases to `pending` after a crash.

        Explicit on purpose: this is startup recovery, not a side effect of
        opening the database. A second process opening the same file while a
        send is in flight must not steal the lease (that double-delivers).

        ``stale_after_seconds=None`` keeps the original "recover every
        `delivering` row" behaviour for a deliberate operator call. The
        automatic startup path passes the lease window instead, because a
        process cannot tell a crashed sender from a live one by a per-process
        flag and must not reset a lease another process may still be holding.
        An expired lease stays recoverable either way: ``claim`` already treats
        `delivering` rows older than the window as claimable, so nothing is
        stranded behind this predicate.
        """
        if stale_after_seconds is None:
            return max(0, self.connection.execute(
                "UPDATE outbox SET delivery_state='pending' WHERE delivery_state='delivering'"
            ).rowcount)
        cutoff = (utc_datetime_now() - timedelta(seconds=max(1.0, stale_after_seconds))).isoformat().replace("+00:00", "Z")
        return max(0, self.connection.execute(
            "UPDATE outbox SET delivery_state='pending' "
            "WHERE delivery_state='delivering' AND (claimed_at IS NULL OR claimed_at < ?)",
            (cutoff,),
        ).rowcount)

    def _recover_inflight_leases_once(self) -> None:
        """Run stale-lease recovery on the first writable open of this process.

        Stale-only, because the per-process flag does not make this open the
        first one on the *file*: the telegram bot starting during a reactor send
        used to reset that send's live lease and deliver the message twice.
        """
        global _INFLIGHT_LEASES_RECOVERED
        if _INFLIGHT_LEASES_RECOVERED:
            return
        self.recover_inflight_outbox(stale_after_seconds=OUTBOX_LEASE_SECONDS)
        _INFLIGHT_LEASES_RECOVERED = True

    def state(self) -> AgentState:
        row = self.connection.execute("SELECT * FROM agent_state WHERE id = 1").fetchone()
        if row is None:
            raise RuntimeError("agent_state row is missing; state database is not initialized")
        try:
            next_plan = json.loads(row["next_plan"])
        except (TypeError, json.JSONDecodeError):
            next_plan = {}
        try:
            lifecycle = LifecycleState(row["lifecycle"])
        except ValueError as exc:
            if row["lifecycle"] == "idle":
                # Legacy persisted value; writable startup migrates it to sleep.
                lifecycle = LifecycleState.SLEEP
            else:
                raise RuntimeError(f"unknown persisted lifecycle: {row['lifecycle']}") from exc
        return AgentState(
            lifecycle=lifecycle,
            generation=row["generation"],
            next_wake_at=row["next_wake_at"],
            active_run_id=row["active_run_id"],
            next_plan=next_plan if isinstance(next_plan, dict) else {},
            retry_count=row["retry_count"],
        )

    def set_state(self, state: AgentState) -> None:
        self.connection.execute(
            """UPDATE agent_state SET lifecycle=?, generation=?, next_wake_at=?,
               active_run_id=?, next_plan=?, retry_count=? WHERE id=1""",
            (
                state.lifecycle.value,
                state.generation,
                state.next_wake_at,
                state.active_run_id,
                json.dumps(state.next_plan),
                state.retry_count,
            ),
        )

    def reset_short_memory(self, *, reason: str = "manual_cli_reset", allow_interrupted_run: bool = False) -> dict[str, Any]:
        """Clear cross-cycle working context while preserving durable history."""
        return self._reset_work_context(cancel_tasks=False, reason=reason, allow_interrupted_run=allow_interrupted_run)

    def reset_dispatcher(self, *, reason: str = "manual_dispatcher_reset") -> dict[str, Any]:
        """Cancel queued work and clear scheduler context without deleting history."""
        return self._reset_work_context(cancel_tasks=True, reason=reason, allow_interrupted_run=False)

    def _reset_work_context(self, *, cancel_tasks: bool, reason: str, allow_interrupted_run: bool) -> dict[str, Any]:
        with self.transaction():
            state = self.state()
            interrupted_run_id = state.active_run_id
            if interrupted_run_id is not None and not allow_interrupted_run:
                raise RuntimeError("cannot reset while an active run exists; stop the service first")
            if interrupted_run_id is not None:
                self.finish_run(interrupted_run_id, RunStatus.INTERRUPTED)
                self.append_event("run_interrupted", {"reason": reason, "run_id": interrupted_run_id}, interrupted_run_id)
                state.active_run_id = None
            cancelled_task_ids: list[str] = []
            if cancel_tasks:
                cancelled_task_ids = [str(row[0]) for row in self.connection.execute("SELECT task_id FROM tasks WHERE status IN ('pending','running')")]
                if cancelled_task_ids:
                    self.connection.execute("UPDATE tasks SET status='cancelled', updated_at=? WHERE status IN ('pending','running')", (utc_now(),))
            state.next_plan = {}
            state.retry_count = 0
            state.next_wake_at = (utc_datetime_now() + timedelta(seconds=60)).isoformat().replace("+00:00", "Z")
            state.lifecycle = LifecycleState.SLEEP
            state.generation += 1
            self.set_state(state)
            details = {
                "reason": reason,
                "generation": state.generation,
                "interrupted_run_id": interrupted_run_id,
                "cancelled_task_ids": cancelled_task_ids,
                "preserved": ["goals", "tasks", "memories", "event_log", "transcript", "episode_snapshots", "hypotheses"],
            }
            self.append_event("dispatcher_reset" if cancel_tasks else "short_memory_reset", details)
            return details

    def reset_planning_context(self, *, reason: str, proposal_id: str | None = None, commit: str | None = None) -> dict[str, Any]:
        """Clear ephemeral planning context while preserving the portfolio and evidence."""
        from uuid import uuid4
        reset_key = "|".join((reason, proposal_id or "", commit or ""))
        with self.transaction():
            existing = self.connection.execute("SELECT reset_id FROM planner_resets WHERE reset_key=?", (reset_key,)).fetchone()
            if existing:
                return {"reset_id": existing[0], "idempotent": True, "reset_key": reset_key}
            state = self.state()
            if state.active_run_id is not None:
                raise RuntimeError("cannot reset planning context while an active run exists")
            reset_id = str(uuid4())
            self.connection.execute(
                "INSERT INTO planner_resets(reset_id, reset_key, reason, proposal_id, commit_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (reset_id, reset_key, reason, proposal_id, commit, utc_now()),
            )
            state.next_plan = {}
            state.retry_count = 0
            state.next_wake_at = (utc_datetime_now() + timedelta(seconds=60)).isoformat().replace("+00:00", "Z")
            state.generation += 1
            state.lifecycle = LifecycleState.SLEEP
            self.set_state(state)
            self.append_event("planner_reset", {"reset_id": reset_id, "reason": reason, "proposal_id": proposal_id, "commit": commit, "preserved": ["goals", "tasks", "memories", "event_log", "hypotheses"]})
            return {"reset_id": reset_id, "idempotent": False, "reset_key": reset_key, "generation": state.generation}

    def register_hypothesis(self, *, workstream_id: str, fingerprint: str, structural_fingerprint: str, problem: str, expected_behavior: str, status: str = "ready") -> bool:
        now = utc_now()
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO hypotheses(hypothesis_id, workstream_id, fingerprint, structural_fingerprint, problem, expected_behavior, status, attempts, last_outcome, created_at, updated_at) VALUES (lower(hex(randomblob(16))), ?, ?, ?, ?, ?, ?, 0, '{}', ?, ?)",
            (workstream_id, fingerprint, structural_fingerprint, problem, expected_behavior, status, now, now),
        )
        return cursor.rowcount > 0

    def mark_hypothesis(self, fingerprint: str, status: str, outcome: dict[str, Any] | None = None) -> None:
        self.connection.execute(
            "UPDATE hypotheses SET status=?, attempts=attempts+1, last_outcome=?, updated_at=? WHERE fingerprint=?",
            (status, json.dumps(outcome or {}, ensure_ascii=False), utc_now(), fingerprint),
        )

    def start_planner_attempt(self, generation: int, trigger: str, strategy: str = "") -> str:
        from uuid import uuid4
        attempt_id = str(uuid4())
        self.connection.execute(
            "INSERT INTO planner_attempts(attempt_id,generation,trigger,status,strategy,started_at) VALUES (?,?,?,?,?,?)",
            (attempt_id, generation, trigger, "started", strategy, utc_now()),
        )
        self.append_event("planner_generation_started", {"attempt_id": attempt_id, "trigger": trigger})
        return attempt_id

    def finish_planner_attempt(self, attempt_id: str, status: str, *, failure: str = "", proposal_count: int = 0) -> None:
        self.connection.execute(
            "UPDATE planner_attempts SET status=?, failure=?, proposal_count=?, finished_at=? WHERE attempt_id=?",
            (status, failure, proposal_count, utc_now(), attempt_id),
        )
        self.append_event("planner_generation_finished", {"attempt_id": attempt_id, "status": status, "failure": failure, "proposal_count": proposal_count})

    def record_planner_proposal(self, proposal: dict[str, Any], attempt_id: str, status: str, *, reason: str = "", task_id: str | None = None) -> str:
        from uuid import uuid4
        proposal_id = str(uuid4())
        self.connection.execute(
            "INSERT INTO planner_proposals(proposal_id,attempt_id,goal_id,title,problem,hypothesis,expected_new_fact,validation,scope,kind,hypothesis_fingerprint,structural_fingerprint,status,rejection_reason,created_task_id,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (proposal_id, attempt_id, proposal["goal_id"], proposal["title"], proposal["problem"], proposal["hypothesis"], proposal["expected_new_fact"], proposal["validation"], json.dumps(proposal.get("scope", [])), proposal["kind"], proposal["hypothesis_fingerprint"], proposal["structural_fingerprint"], status, reason, task_id, utc_now()),
        )
        self.append_event("planner_proposal_accepted" if status == "accepted" else "planner_proposal_rejected", {"proposal_id": proposal_id, "attempt_id": attempt_id, "status": status, "reason": reason})
        return proposal_id

    def transition(self, state: AgentState, target: LifecycleState, *, run_id: str | None = None, reason: str = "", record_event: bool = False) -> AgentState:
        previous = state.lifecycle
        state.lifecycle = target
        self.set_state(state)
        if record_event and previous != target:
            self.append_event(
                "lifecycle_transition",
                {"from": previous.value, "to": target.value, "reason": reason},
                run_id,
            )
        return state

    def create_run(self, run: RunRecord) -> None:
        self.runs.create(run)

    def touch_run(self, run_id: str, phase: str, *, progress: bool = False, steps: int | None = None, usage_tokens: int | None = None) -> None:
        self.runs.touch(run_id, phase, progress=progress, steps=steps, usage_tokens=usage_tokens)

    def run_usage(self, run_id: str) -> dict[str, int]:
        """The run's durable running totals, so a killed episode keeps its cost.

        The ReAct loop's counters live in memory and only reach `run_results` on
        a Finish Report; an interrupted run used to be recorded as steps=0,
        tokens=0 even after dozens of tool calls.
        """
        row = self.connection.execute("SELECT steps, usage_tokens FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return {"steps": 0, "usage_tokens": 0}
        return {"steps": int(row["steps"] or 0), "usage_tokens": int(row["usage_tokens"] or 0)}

    def run_executed_tool_names(self, run_id: str) -> list[str]:
        """The tools this run actually executed, read from its own durable rows.

        The ReAct Loop cannot answer this for a killed episode: its message
        history dies with the process, and ``ReActRunner._succeeded_tool_names``
        reads exactly that history. The event log is the durable equivalent, so
        the same predicate is applied here to the stored events -- a
        ``tool_result`` whose ``result.ok`` is not False, named by the
        ``tool_call`` that preceded it. The name is resolved through the
        request's ``call_id`` rather than the result's own copy, so a result
        cannot rename the tool that produced it. A refused call (unknown tool
        name, policy denial) reports ``ok`` False and is therefore not evidence.

        Measured on the live ledger (generation 238): of 13,996 ``tool_result``
        events, 13,009 carry ``result.ok`` true and 0 lack the key, so the
        ``is not False`` and ``is True`` readings coincide here. A stronger
        check is available because the same work is written twice by two
        different code paths -- this reader from ``event_log``, and
        ``run_effect_capabilities`` from ``capability_effects``. Over all 221
        runs they agree exactly (221 agree, 0 differ), and on the five
        ``interrupted`` runs both count the same 142 calls. A reader that will
        size a run's stored evidence is therefore cross-checked against a ledger
        it does not read.
        """
        rows = self.connection.execute(
            "SELECT kind, payload FROM event_log WHERE run_id=? AND kind IN ('tool_call','tool_result') ORDER BY sequence",
            (run_id,),
        ).fetchall()
        requested: dict[str, str] = {}
        names: list[str] = []
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if row["kind"] == "tool_call":
                call_id = str(payload.get("call_id") or "")
                name = str(payload.get("tool_name") or "")
                if call_id and name:
                    requested[call_id] = name
                continue
            result = payload.get("result")
            if not isinstance(result, dict) or result.get("ok") is False:
                continue
            raw_call = payload.get("call")
            call: dict[str, Any] = raw_call if isinstance(raw_call, dict) else {}
            name = requested.get(str(call.get("call_id") or "")) or str(call.get("tool_name") or "")
            if name:
                names.append(name)
        return names

    def stale_active_run(self, before: str) -> str | None:
        """Return the active run id when its heartbeat is older than `before`.

        Uses a dedicated read-only connection so it is safe to call from the
        stale-run watchdog thread while the main thread owns the main
        connection.
        """
        connection = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True, timeout=5.0)
        try:
            connection.row_factory = sqlite3.Row
            row = connection.execute("SELECT active_run_id FROM agent_state WHERE id = 1").fetchone()
            active_run_id = str(row["active_run_id"]) if row and row["active_run_id"] else None
            if active_run_id is None:
                return None
            stale = connection.execute(
                "SELECT 1 FROM runs WHERE run_id=? AND status='running' AND heartbeat_at < ?",
                (active_run_id, before),
            ).fetchone()
            return active_run_id if stale else None
        finally:
            connection.close()

    def finish_run(self, run_id: str, status: RunStatus) -> None:
        self.runs.finish(run_id, status)

    def commit_run_result(self, run_id: str, status: RunStatus, report: str, steps: int, usage_tokens: int, failure: str) -> dict[str, Any]:
        """Persist the model result before any consolidation or scheduling work."""
        self.connection.execute(
            "INSERT INTO run_results(run_id, status, report, steps, usage_tokens, failure, created_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, report=excluded.report, "
            "steps=excluded.steps, usage_tokens=excluded.usage_tokens, failure=excluded.failure",
            (run_id, status.value, report, steps, usage_tokens, failure, utc_now()),
        )
        row = self.connection.execute(
            "SELECT run_id, status, report, steps, usage_tokens, failure, created_at FROM run_results WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise AssertionError("run result row missing after upsert")
        return dict(row)

    def update_run_result_status(self, run_id: str, status: RunStatus, failure: str) -> None:
        self.connection.execute(
            "UPDATE run_results SET status=?, failure=? WHERE run_id=?",
            (status.value, failure, run_id),
        )

    def run_result(self, run_id: str) -> dict[str, Any] | None:
        return self.runs.result(run_id)

    def run_effect_capabilities(self, run_id: str) -> dict[str, int]:
        """How many calls this run actually ran, per capability.

        `capability_effects.idempotency_key` is `{run_id}:{step}:{call_id}`, so
        the run prefix isolates one run's calls. This is the evidence a Finish
        Report claim can be checked against: a tool that always records an
        effect and has a zero count here did not run.

        Counting *rows* was not the same question. `skynet/tools.py` returns a
        policy refusal before the command runs, and until this reader excluded
        them a refusal row satisfied the check for the tool it named. The
        exclusion is two-fold because the ledger holds rows from both eras:
        `status` names the refusals written since the label exists, and the
        marker predicate names those already on disk. Measured on the live
        ledger (generation 93): of the 8 consumers of this table this
        is the only one that counted a refusal as an application, and 0 of the
        63 refusal rows belong to a claim tool, so the repair changes no verdict
        recorded so far - it closes the hole instead of altering history.
        """
        rows = self.connection.execute(
            "SELECT capability, count(*) AS n FROM capability_effects AS p "
            "WHERE p.idempotency_key LIKE ? || ':%' AND p.status != ? AND "
            + _REFUSAL_RESULT_PREDICATE.format(alias="p")
            + " GROUP BY capability",
            (run_id, REFUSED_EFFECT_STATUS),
        ).fetchall()
        return {str(row["capability"]): int(row["n"]) for row in rows}

    @staticmethod
    def effect_status(result: Mapping[str, Any]) -> str:
        """The ledger `status` for one finished tool call.

        The single place that decides whether a result is an application or a
        refusal, so the writer (`react.py` records every result) and the reader
        above cannot drift apart. A refusal never executed the command, so it is
        labelled `REFUSED_EFFECT_STATUS`; anything else is `applied` when the
        tool reported success and `failed` when it ran and reported a failure -
        a failing command is work, a refused one is not.
        """
        if any(result.get(marker) for marker in _REFUSAL_RESULT_MARKERS):
            return REFUSED_EFFECT_STATUS
        return "applied" if result.get("ok") else "failed"

    def recent_run_events(self, run_id: str, limit: int | None = 20) -> list[dict[str, Any]]:
        """Return the bounded, non-reasoning episode needed for recovery."""
        return self.events.recent(run_id, limit)

    def episode_for_memory(self, run_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        allowed = {"run_started", "tool_call", "tool_result", "provider_failure", "provider_error_classified", "provider_retry", "tool_failure", "tool_retry", "context_warning", "context_compacted", "context_limit", "finish_report", "run_finished", "doom_loop"}
        return [event for event in self.recent_run_events(run_id, limit) if event["kind"] in allowed]

    def _snapshot_transcript(self, run_id: str) -> list[dict[str, Any]]:
        """The transcript projection a snapshot exposes, rebuilt from the table.

        ``react_history`` is the latest-state row the memory loop reads directly
        from the transcript table (``reactor.py:408``), so embedding it again
        would duplicate the run.
        """
        return [row for row in self.recent_transcript(run_id) if row["kind"] != "react_history"]

    def snapshot_episode(self, run_id: str) -> dict[str, Any]:
        """Freeze one run's episode; the events are stored, the transcript is not.

        Measured (generation 46, ``dbstat`` on a read-only copy of the
        live ledger): the ``transcript`` part was 3.179 MB of the 13.700 MB of
        snapshot payload (23.2%), and the read-time rebuild reproduced all 43
        stored copies exactly (0 mismatches, both directions). The copy also had
        no lifetime of its own: ``prune_run_history`` deletes ``transcript`` and
        ``episode_snapshots`` in one statement under one ``keep_runs`` window, and
        no production reader touches it -- only ``snapshot["events"]``
        (``reactor.py:396``) and ``snapshot["snapshot_id"]`` (``reactor.py:424``)
        are read. It was a frozen duplicate of a projection, so the row now keeps
        a pointer to its source instead of the payload and the projection is
        rebuilt on read.
        """
        existing = self.connection.execute("SELECT snapshot_id, first_sequence, last_sequence, payload_hash, payload FROM episode_snapshots WHERE run_id=?", (run_id,)).fetchone()
        if existing:
            snapshot = json.loads(existing[4])
            if isinstance(snapshot, list):
                snapshot = {"events": snapshot}
            # A legacy row carries its own frozen transcript list; only a pointer
            # (or a missing part) is rebuilt, so old bytes keep their meaning.
            if not isinstance(snapshot.get("transcript"), list):
                snapshot["transcript"] = self._snapshot_transcript(run_id)
            return {"snapshot_id": existing[0], "run_id": run_id, "first_sequence": existing[1], "last_sequence": existing[2], "payload_hash": existing[3], **snapshot}
        events = self.episode_for_memory(run_id, limit=None)
        transcript = self._snapshot_transcript(run_id)
        from uuid import uuid4
        # Only the events are stored: the transcript part is a rebuildable
        # projection of the transcript table with an identical retention window,
        # so it is replaced by a pointer to its source.
        payload = json.dumps(
            {"events": events, "transcript": {"rebuilt_from": "transcript", "rows_at_freeze": len(transcript)}},
            ensure_ascii=False,
            sort_keys=True,
        )
        sequences = [int(event["sequence"]) for event in events]
        # payload_hash stays the digest of the stored payload, so a row is still
        # verifiable against its own bytes.
        snapshot = {"snapshot_id": str(uuid4()), "run_id": run_id, "first_sequence": min(sequences) if sequences else None, "last_sequence": max(sequences) if sequences else None, "payload_hash": hashlib.sha256(payload.encode()).hexdigest(), "events": events, "transcript": transcript}
        self.connection.execute("INSERT INTO episode_snapshots(snapshot_id, run_id, first_sequence, last_sequence, payload_hash, payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (snapshot["snapshot_id"], run_id, snapshot["first_sequence"], snapshot["last_sequence"], snapshot["payload_hash"], payload, utc_now()))
        return snapshot

    def compact_episode_snapshots(self) -> dict[str, int]:
        """Rewrite legacy snapshot rows that still embed a frozen transcript.

        ``snapshot_episode`` writes a pointer for new rows, but rows written
        before that decision keep a verbatim copy of a projection the
        ``transcript`` table already owns, and the read path returns early for
        an existing row -- so the copies are forward-only leftovers. Measured
        on the live ledger: 44 of 51 rows still carried the list
        form, 14.74 MB of the 16.78 MB of snapshot payload bytes, and all 44
        rebuilt byte-identically from ``transcript`` (0 mismatches). This
        rewrites exactly those rows to the pointer form, which is the eviction
        rule MemGPT applies to its FIFO queue (arXiv:2310.08560 sec. 2.2): the
        evicted copy becomes an address into the retained store instead of a
        second copy of it.

        Only the transcript part is dropped: ``events`` is not rebuildable and
        is never touched, and a row whose rebuild does not reproduce the stored
        list is left alone (old bytes keep their meaning). ``payload_hash`` is
        recomputed from the new bytes, so the row stays verifiable against
        itself.
        """
        rewritten = 0
        reclaimed = 0
        rows = self.connection.execute(
            "SELECT run_id, payload FROM episode_snapshots WHERE json_type(payload, '$.transcript') = 'array'"
        ).fetchall()
        for row in rows:
            run_id = row["run_id"]
            snapshot = json.loads(row["payload"])
            stored = snapshot.get("transcript")
            rebuilt = self._snapshot_transcript(run_id)
            if stored != rebuilt:
                continue
            compacted = dict(snapshot)
            compacted["transcript"] = {"rebuilt_from": "transcript", "rows_at_freeze": len(rebuilt)}
            payload = json.dumps(compacted, ensure_ascii=False, sort_keys=True)
            self.connection.execute(
                "UPDATE episode_snapshots SET payload=?, payload_hash=? WHERE run_id=?",
                (payload, hashlib.sha256(payload.encode()).hexdigest(), run_id),
            )
            rewritten += 1
            reclaimed += len(row["payload"]) - len(payload)
        return {"snapshots": rewritten, "bytes": max(0, reclaimed)}

    def append_event(self, kind: str, payload: dict[str, Any], run_id: str | None = None) -> int:
        sequence = self.events.append(kind, payload, run_id)
        self.runtime_log.write("event", {"sequence": sequence, "event_kind": kind, "payload": payload}, run_id=run_id)
        return sequence

    def record_provider_event(self, kind: str, payload: dict[str, Any]) -> None:
        """Route a provider-chain diagnostic to the durable event log.

        Kinds in ``DURABLE_PROVIDER_EVENT_KINDS`` go through ``append_event`` so
        they survive retention and are visible in ``skynet logs``; every other
        provider kind (attempts, selections, requests, cooldown skips) stays in
        the advanced runtime projection only, because writing it durably would
        flood ``event_log``. ``append_event`` mirrors the promoted row back into
        the runtime projection, so no separate ``runtime_log.write`` is needed.
        """
        if kind in DURABLE_PROVIDER_EVENT_KINDS:
            if threading.get_ident() == self._connection_thread:
                self.append_event(kind, payload)
            else:
                self._append_event_off_thread(kind, payload)
        else:
            self.runtime_log.write(kind, payload)

    def _append_event_off_thread(self, kind: str, payload: dict[str, Any]) -> None:
        """Insert one durable event from a thread that does not own the connection.

        The memory loop and the planner call the provider chain on worker
        threads, so a fallback during those phases logged from there and hit
        "SQLite objects created in a thread can only be used in that same
        thread"; the real provider failure was replaced by that error and the
        durable row was lost. WAL plus the busy timeout lets a short-lived second
        connection wait its turn instead, and the runtime projection is mirrored
        with the same shape ``append_event`` uses. A diagnostic must never abort
        the provider ladder, so a write failure is swallowed.
        """
        sequence: int | None = None
        try:
            connection = sqlite3.connect(self.path, isolation_level=None, timeout=5.0)
            try:
                connection.execute("PRAGMA busy_timeout=5000")
                cursor = connection.execute(
                    "INSERT INTO event_log(run_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
                    (None, kind, json.dumps(payload, ensure_ascii=False), utc_now()),
                )
                sequence = cursor.lastrowid
            finally:
                connection.close()
        except sqlite3.Error:
            pass
        self.runtime_log.write("event", {"sequence": sequence, "event_kind": kind, "payload": payload}, run_id=None)

    def append_transcript(self, kind: str, payload: dict[str, Any], run_id: str | None = None) -> int:
        cursor = self.connection.execute(
            "INSERT INTO transcript(run_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
            (run_id, kind, json.dumps(payload, ensure_ascii=False), utc_now()),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return a transcript sequence")
        return int(cursor.lastrowid)

    def record_pending_memory_capture(self, run_id: str, *, error: str, transcript_rows: int, events: int) -> int:
        """Persist the episode's memory work instead of discarding it.

        A transient provider fault inside the post-episode Memory Loop (a
        timeout, a network error, an unparseable reply) makes ``consolidate``
        return a *degraded* result with zero candidates. The episode is over and
        the loop is never retried, so every durable observation the episode
        produced is lost -- `event_log` rows 16199-16205 on run ``e7e36432``
        are that exact sequence: one provider fault, then
        ``memory_loop_finished {memory_candidates: 0, degraded: true}`` on a run
        that still committed as completed. This row is the recoverable pointer:
        it names the run, the frozen episode snapshot and the transcript, so a
        later pass (or the owner) can re-derive the memories from evidence that
        still exists. It does not retry anything and it never fails the run.

        Idempotent by run id plus kind: a repeated call updates the existing row
        rather than appending a second pending record for the same episode.
        """
        existing = self.connection.execute(
            "SELECT sequence FROM event_log WHERE kind='memory_capture_pending' AND run_id=? LIMIT 1",
            (run_id,),
        ).fetchone()
        snapshot = self.connection.execute(
            "SELECT snapshot_id FROM episode_snapshots WHERE run_id=? LIMIT 1",
            (run_id,),
        ).fetchone()
        payload: dict[str, Any] = {
            "run_id": run_id,
            "error": str(error)[:1000],
            "snapshot_id": str(snapshot["snapshot_id"]) if snapshot is not None else None,
            "episode_events": int(events),
            "transcript_rows": int(transcript_rows),
        }
        if existing is not None:
            self.connection.execute(
                "UPDATE event_log SET payload=? WHERE sequence=?",
                (json.dumps(payload, ensure_ascii=False), int(existing["sequence"])),
            )
            return int(existing["sequence"])
        return self.append_event("memory_capture_pending", payload, run_id)

    def pending_memory_captures(self, limit: int = 25) -> list[dict[str, Any]]:
        """Unresolved pending captures, oldest first.

        A pointer whose payload carries a ``disposition`` has been reconciled
        and is deliberately excluded: the disposition is what gives the pointer
        a bounded lifetime. Without it ``memory_capture_pending`` was permanent
        -- it is in ``PROTECTED_EVENT_KINDS`` and no code path ever resolved
        ``snapshot_id`` or the pending ``run_id`` (measured generation 177,
        read-only on the live ledger: 27 pointers, 0 naming any disposition, and
        the evidence under them held past the run-count window with no reader).
        """
        rows = self.connection.execute(
            "SELECT sequence, run_id, payload, created_at FROM event_log "
            "WHERE kind='memory_capture_pending' AND run_id IS NOT NULL "
            "AND json_type(payload, '$.disposition') IS NULL "
            "ORDER BY sequence LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [
            {
                "sequence": int(row["sequence"]),
                "run_id": str(row["run_id"]),
                "payload": json.loads(row["payload"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def pending_capture_evidence(self, run_id: str) -> dict[str, int]:
        """How much of the evidence a pending capture names still exists.

        The pointer's whole purpose is that the frozen episode and the
        transcript outlive the degraded loop; a resolution pass must be able to
        tell "the evidence is here, re-derive it" from "the window took it".
        """
        return {
            "snapshots": int(self.connection.execute("SELECT COUNT(*) FROM episode_snapshots WHERE run_id=?", (run_id,)).fetchone()[0]),
            "transcript_rows": int(self.connection.execute("SELECT COUNT(*) FROM transcript WHERE run_id=?", (run_id,)).fetchone()[0]),
        }

    def _pending_capture_expiry(self, run_id: str, keep_runs: int) -> tuple[str, str, int]:
        """When the evidence a protected pointer names leaves the run window.

        NIST SP 800-92 section 5.4 makes disposal happen "when the required data
        retention period has ended", so the end of the period must be *written*
        when the pointer's life ends, not inferred by whoever looks later. The
        period that governs a ``memory_capture_pending`` pointer is
        ``prune_run_history``'s run-count window, so the bound is the run in
        which this pointer's run leaves that window, dated at the inter-run
        interval measured from the ledger itself.
        """
        window = max(1, int(keep_runs))
        row = self.connection.execute("SELECT rowid FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            # Its run row is already gone: the window has taken it, so the bound
            # is now and the record says why.
            return utc_now(), "run_window_elapsed", window
        rank = int(
            self.connection.execute("SELECT COUNT(*) FROM runs WHERE rowid > ?", (int(row["rowid"]),)).fetchone()[0]
        )
        remaining = max(0, window - rank)
        if remaining <= 0:
            return utc_now(), "run_window_elapsed", window
        pace = self.connection.execute(
            "SELECT AVG(delta) FROM ("
            "SELECT julianday(started_at) - julianday(LAG(started_at) OVER (ORDER BY rowid)) AS delta FROM runs)"
        ).fetchone()[0]
        if not pace or float(pace) <= 0.0:
            # A ledger with one run has no inter-run interval to measure, so the
            # date the window ends cannot be derived from it. The bound is still
            # written, labelled as what it is rather than passed off as measured.
            return utc_now(), "run_window_pace_unmeasured", window
        expires = (utc_datetime_now() + timedelta(days=remaining * float(pace))).isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z")
        return expires, "run_window_end", window

    def dispose_pending_memory_capture(
        self, run_id: str, *, disposition: str, detail: str = "", memories: int = 0, keep_runs: int | None = None
    ) -> bool:
        """End a pending capture's life: record the outcome once, durably.

        ``recaptured`` means the evidence was re-derived into memories;
        ``unrecoverable`` means a re-attempt ran and degraded again (one attempt
        only, so the pointer cannot spin); ``expired`` means the evidence had
        already left the run-count window before any re-attempt. The pointer row
        itself stays -- it is in ``PROTECTED_EVENT_KINDS`` and remains the audit
        record of the episode whose Memory Loop was lost -- but it is no longer
        *pending*, so ``pending_memory_captures`` stops returning it and the
        attempt never repeats. Idempotent by run id: a second call for an
        already-disposed pointer writes nothing.

        Each disposition also writes ``expires_at`` (with the ``expiry_basis``
        and ``expiry_keeps`` that produced it): NIST SP 800-92 section 5.4 ends
        a retention period with a scheduled act, not with whoever reads next, so
        the terminal record carries the bound itself. Measured live before this
        change (generation 180, read-only copy, 165 runs): 2 terminal pointers,
        0 naming an expiry, so the invariant "terminal-dispositioned protected
        pointers with no written bound = 0" read 2.
        """
        bounded = str(disposition)[:40]
        row = self.connection.execute(
            "SELECT sequence, payload FROM event_log WHERE kind='memory_capture_pending' AND run_id=? LIMIT 1",
            (run_id,),
        ).fetchone()
        if row is None:
            return False
        payload = json.loads(row["payload"])
        if payload.get("disposition"):
            return False
        expires_at, basis, window = self._pending_capture_expiry(
            run_id, DEFAULT_TRANSCRIPT_RETENTION_RUNS if keep_runs is None else keep_runs
        )
        payload.update(
            {
                "disposition": bounded,
                "disposed_at": utc_now(),
                # The bound NIST 800-92 section 5.4 requires: the record of the
                # loss survives, and the moment its payload is scheduled to be
                # gone is written down instead of implied by a later reader.
                "expires_at": expires_at,
                "expiry_basis": basis,
                "expiry_keeps": window,
                "disposal_detail": str(detail)[:500],
                "memories_recovered": int(memories),
            }
        )
        self.connection.execute(
            "UPDATE event_log SET payload=? WHERE sequence=?",
            (json.dumps(payload, ensure_ascii=False), int(row["sequence"])),
        )
        self.append_event(
            "memory_capture_released",
            {
                "run_id": run_id,
                "disposition": bounded,
                "detail": str(detail)[:500],
                "memories": int(memories),
                "expires_at": expires_at,
                "expiry_basis": basis,
            },
            run_id,
        )
        return True

    def prune_event_log(self, retention_days: int, *, protected_kinds: frozenset[str] = PROTECTED_EVENT_KINDS) -> int:
        """Drop routine events older than the window; never the safety record.

        `event_log` grew without bound (6751 rows in 159 generations, ~42 per
        cycle). The protected set keeps every kind a post-mortem or an audit
        depends on: deviations, escalations, provider classification, policy
        decisions, restarts and reconciliations.
        """
        if retention_days <= 0:
            return 0
        cutoff = (utc_datetime_now() - timedelta(days=retention_days)).isoformat().replace("+00:00", "Z")
        placeholders = ",".join("?" for _ in protected_kinds)
        return max(0, self.connection.execute(
            f"DELETE FROM event_log WHERE created_at < ? AND kind NOT IN ({placeholders})",
            (cutoff, *protected_kinds),
        ).rowcount)

    def prune_capability_effects(self, retention_days: int) -> int:
        """Drop effect-idempotency rows older than the window.

        ``capability_effects`` was the one monotone table that had no retention
        statement until this window was added. Re-measured (generation
        39) on a read-only copy of the live ledger: 2281 rows / 6.12 MB of table
        pages, 13.69% of a 46.9 MB database, plus 729 KB of its own indexes
        (15.25% inclusive), growing ~126 rows/hour (~3,000 rows/day over an 18.1 h
        span) with 60.5% of the rows holding 95% of the result bytes (``bash``
        alone is 58.7%). At that rate this 30-day window settles near 90k rows /
        ~284 MB including indexes, so the window is what bounds the table -- the
        old ``~7 rows/hour`` figure understated it roughly eighteenfold. Nothing
        reaches back that far: ``effect`` is looked up by an
        exact ``{run_id}:{step}:{call_id}`` key belonging to the run being
        executed, and runs are reconciled within the reboot window (1 day); the
        only cross-run reader, ``effect_reapplied_runs``, is a diagnostic pointer
        whose one-time ``effect_reapplied`` event survives the window, so past the
        window the worst case is one benign re-announcement.

        A row is kept while a surviving event still names it. ``event_log`` and
        this table are pruned by two *independent* windows
        (``event_retention_days`` / ``effect_retention_days``), and the age-only
        DELETE broke the invariant this table exists to support: measured
        (generation 43) on scratch copies of the live ledger, with the
        effects aged past a 30-day cutoff, the plain DELETE removed 2448 rows and
        left 2 dangling pointers -- every folded ``tool_result`` event carrying an
        ``effect_key`` pointed at a row that no longer existed. Reachable through
        configuration alone (``SKYNET_EVENT_RETENTION_DAYS=60`` with the default
        effect window): ``prune_event_log(60)`` removed 0 events while
        ``prune_capability_effects(30)`` removed 2448 rows. The guard pins exactly
        the pointed subset and drops the rest (same probe: 2446 removed, 0
        dangling), and the pin is bounded -- 91 folded events over 19.4 h (~112/day)
        is ~3.4k rows across a 30-day window, far below the ~90k rows this window
        already tolerates. Rows whose pointer has itself been pruned are dropped
        as before, so retention still bounds the table.

        The folded result also lands in the ReAct transcript, which this guard did
        not read. ``_bounded_tool_content`` (``react.py:937-950``) puts the same
        ``effect_key`` into the ``tool`` message ``_record_history`` persists as
        ``react_history`` (``react.py:441-445``, ``react.py:806-811``,
        ``store.py:1030``). The path differs from the event log: ``content`` is
        itself a JSON-encoded string, so the key is ``$.messages[*].content``
        parsed again to ``$.effect_key``, not a nested object. Transcript retention
        is by run count (``transcript_retention_runs``, default 200) and independent
        of the 30-day effect window, so an aged row whose only surviving pointer was
        a retained transcript was still deleted -- the dangling pointer this guard
        exists to prevent, through a second door. Reproduced on a scratch store with
        a retained ``react_history`` row: the plain DELETE removed the row and left
        the transcript pointing at nothing.

        ``episode_snapshots`` is the third holder and was the last unguarded one.
        ``snapshot_episode`` stores the run's ``tool_result`` events verbatim in
        ``payload.events`` (``store.py:1035``), so the address survives there in
        the original nested shape (``$.payload.result.effect_key``, not the
        double-encoded ``content`` shape the transcript copy uses). Snapshots are
        pruned by run count, never by age, which makes this reachable through
        configuration alone: with a 1-day event window and the default 30-day
        effect window the other two holders release the pin while the snapshot
        keeps it (measured on a scratch copy of the live ledger -- 4327 rows
        removed, 1 dangling snapshot address). The third subquery closes it:
        same probe, 4326 removed, 0 dangling, 51 pinned.

        The guard is therefore a second ``NOT IN`` subquery over ``transcript``. It
        is one SQL statement rather than a Python-collected key set because the pin
        set has no fixed size -- one retained run can fold hundreds of results and a
        200-run window can exceed SQLite's bound-variable limit -- while
        ``json_each`` takes no parameters and is materialised once (``EXPLAIN QUERY
        PLAN`` reports ``LIST SUBQUERY``, so the cost is independent of the number of
        aged effect rows scanned). It stays bounded by construction: the subquery
        reads only rows still present in ``transcript``, so a checkpoint that runs
        ``prune_run_history`` first (``reactor.py:612``) releases the pin on the next
        daily pass. ``json_valid``/``IS NOT NULL`` keep ``NULL`` out of the subquery,
        which ``NOT IN`` needs to delete anything at all.
        """
        if retention_days <= 0:
            return 0
        cutoff = (utc_datetime_now() - timedelta(days=retention_days)).isoformat().replace("+00:00", "Z")
        return max(0, self.connection.execute(
            "DELETE FROM capability_effects WHERE created_at < ? AND idempotency_key NOT IN ("
            "SELECT json_extract(payload, '$.result.effect_key') FROM event_log "
            "WHERE kind = 'tool_result' AND json_extract(payload, '$.result.effect_key') IS NOT NULL) "
            "AND idempotency_key NOT IN ("
            "SELECT json_extract(json_extract(message.value, '$.content'), '$.effect_key') "
            "FROM transcript, json_each(transcript.payload, '$.messages') AS message "
            "WHERE transcript.kind = 'react_history' "
            "AND json_valid(json_extract(message.value, '$.content')) "
            "AND json_extract(json_extract(message.value, '$.content'), '$.effect_key') IS NOT NULL) "
            "AND idempotency_key NOT IN ("
            "SELECT json_extract(event.value, '$.payload.result.effect_key') "
            "FROM episode_snapshots, json_each(episode_snapshots.payload, '$.events') AS event "
            "WHERE json_valid(event.value) "
            "AND json_type(event.value) = 'object' "
            "AND json_type(event.value, '$.payload') = 'object' "
            "AND json_type(event.value, '$.payload.result') = 'object' "
            "AND json_extract(event.value, '$.kind') = 'tool_result' "
            "AND json_extract(event.value, '$.payload.result.effect_key') IS NOT NULL) "
            "AND idempotency_key NOT IN ("
            "SELECT node.value "
            "FROM episode_snapshots, json_tree(episode_snapshots.payload, '$.events') AS node "
            "WHERE node.key = 'effect_key' AND node.type = 'text')",
            (cutoff,),
        ).rowcount)

    def prune_run_history(self, keep_runs: int) -> dict[str, int]:
        """Drop transcript and episode rows for runs older than the newest `keep_runs`.

        Audit data (event_log, checkpoints, memories, goals, tasks) is never
        pruned; only the per-run conversation history and episode snapshots.

        ``memory_capture_pending`` is in ``PROTECTED_EVENT_KINDS``, so the row is
        deliberately kept past every retention window -- but the evidence it
        names was not. The row carries the ``run_id`` and ``snapshot_id`` of a
        degraded episode whose only other record is the transcript and the
        frozen snapshot, so a run-count prune that dropped both made the one
        protected pointer to that work point at nothing. Measured (generation
        172, read-only on a copy of the live ledger, 158 runs / 25 pending
        captures): at the live first-fire boundary (``keep_runs=47`` on today's
        ledger deletes exactly the 111 runs the ``N=311`` window will delete) 1
        of the 25 already dangled -- the pointer resolved while its
        ``snapshot_id`` matched 0 rows and its ``run_id`` had 0 transcript rows --
        and at saturation all 25 do, ~15.8% of runs. The other two retention
        paths were clean at their first fire (``prune_event_log`` removed
        24,054 -> 2,221 rows and ``prune_capability_effects`` 9,011 -> 149, both
        with 0 dangling addresses across all six holder classes), so only this
        one needs the guard.

        The guard generation 172 added for that was a tautology and held 0 rows.
        A DELETE's ``run_id NOT IN (window)`` already excludes every address
        inside the window, so an extra ``AND run_id NOT IN (pending AND
        inside-window)`` can only subtract rows the first conjunct has already
        removed. Measured (generation 179, read-only copy of the live ledger,
        165 runs): sweeping EVERY ``keep_runs`` 1..165, the conjunct changes the
        deletion set for 0 of the 165 values, and the pin set never exceeds the
        28 pending pointers. The guard restored in generation 173 is the one
        that fires: an unbounded pin was the defect, not a dangling address.

        That bound was too generous: the pin held every pending capture forever,
        not only the ones inside the window. Measured (generation 173, read-only
        on a copy of the live ledger, 159 runs / 25 pending captures / 190 MB)
        it holds 1,459 transcript rows (6.38 MB) and 25 snapshots (5.34 MB) --
        11.72 MB, 12.7% of the 92.4 MB of evidence -- and nothing ever removes
        the row, because ``memory_capture_pending`` is in
        ``PROTECTED_EVENT_KINDS`` and no code path resolves ``snapshot_id`` or the
        pending ``run_id``: the reactor, CLI, owner digest and deviations digest
        only record or count the kind, and ``memory_capture_pending_failed`` is
        itself unprotected. 33 of 153 completed memory loops were degraded, 25
        left a pending row, arriving at ~5.14/day, so the pin grew ~2.4 MB/day
        (~0.9 GB/year) while the run-count window it exists to serve keeps 200
        runs. A second, unbounded retention class inside ``transcript`` is
        exactly what this method exists to bound.

        The pin therefore ends with the window: the pointer is held only while
        its run is still inside ``keep_runs``. Once the run leaves, the pointer
        row survives -- it is protected, and its run id, error and counts remain
        the audit record -- but the evidence under it is released, the same way
        ``prune_capability_effects`` releases its rows when the transcript that
        named them is pruned. The bound is structural: at most ``keep_runs``
        captures are pinned, the window this method already honours.

        That rule is the log-management requirement the pointer exists to keep.
        NIST SP 800-92 section 3.2 defines disposal as "removing all entries from
        a log that precede a certain date and time" and section 5.4 makes the act
        happen "when the required data retention period has ended", not when a
        reader next asks -- so the payload a protected pointer names must be
        cleared on schedule while the record of the loss stays readable. Measured
        (generation 179, read-only on a copy of the live ledger, 165 runs / 27
        open pointers, 12,949 transcript rows / 159 snapshots): the open pointers
        pin 1,627 transcript rows (7,474,435 B) and 27 snapshots (6,321,504 B) =
        13,795,939 B, oldest pointer rank 54 of the 200-run window and 2.03 days
        old, so 0 are outside it yet and each is released as its run leaves. The
        lifetime is 200 runs, measured at 31.7 runs/day over the live 5.20-day
        history, i.e. ~6.3 days; generation 172's predicted release at a 15.8%
        pending rate was 11,720,808 B, and today's 16.4% (27/165) gives
        13,795,939 B. What actually ends the pin early is the resolution pass
        ``Reactor._reconcile_pending_captures``: 1 of 28 pointers has ever been
        dispositioned, so the class is bounded in design and, at this window, not
        yet by practice.
        """
        if keep_runs <= 0:
            return {"transcript": 0, "episodes": 0, "owed": 0}
        keep = "SELECT run_id FROM runs ORDER BY rowid DESC LIMIT ?"
        # A run whose report is still owed keeps its history while the event
        # window still holds its audit rows. The 30-day event window is what
        # governs exactly those rows, so the hold releases itself: once every
        # unprotected event row of the run is gone the subquery drops it and the
        # next sweep reclaims the history. Bounding the hold by a second clock
        # rather than by a new constant is what keeps it from becoming the
        # unbounded pin generation 173 removed. Measured (generation 208,
        # read-only on the live ledger, 192 runs / 31 owed): 26 of the 31 are
        # offside the first-fire window and would lose 1,187 transcript rows and
        # 21 snapshots there; the transcript is the only copy of 690,880 B of
        # the owed runs' 766,392 B of provider_response text, and 0 of the 23
        # owed runs hold all their tool results in event_log. Saturation is ~32
        # runs / 10.9 MB at the measured 32.9 runs/day and 16.1% owed rate, and
        # simulating the event window released all 26.
        protected = ", ".join("?" for _ in sorted(PROTECTED_EVENT_KINDS))
        owed = (
            f"SELECT run_id FROM runs WHERE status != '{RunStatus.COMPLETED.value}' AND run_id IN ("
            f"SELECT DISTINCT run_id FROM event_log WHERE run_id IS NOT NULL AND kind NOT IN ({protected}))"
        )
        params = (keep_runs, *sorted(PROTECTED_EVENT_KINDS))
        transcript = self.connection.execute(
            f"DELETE FROM transcript WHERE run_id IS NOT NULL AND run_id NOT IN ({keep}) AND run_id NOT IN ({owed})",
            params,
        ).rowcount
        episodes = self.connection.execute(
            f"DELETE FROM episode_snapshots WHERE run_id NOT IN ({keep}) AND run_id NOT IN ({owed})",
            params,
        ).rowcount
        # Counted after the delete, so the number names the evidence that is
        # actually still there rather than what the guard intended to hold.
        skipped = self.connection.execute(
            f"SELECT COUNT(DISTINCT run_id) FROM transcript "
            f"WHERE run_id IS NOT NULL AND run_id NOT IN ({keep}) AND run_id IN ({owed})",
            params,
        ).fetchone()[0]
        return {"transcript": max(0, transcript), "episodes": max(0, episodes), "owed": max(0, int(skipped))}

    def set_transcript(self, kind: str, payload: dict[str, Any], run_id: str) -> int:
        """Replace the previous row of this kind for the run.

        Some transcripts describe the latest state of a run rather than an
        append-only stream (the message history for memory consolidation), so
        keeping every intermediate copy would grow the database quadratically.
        """
        self.connection.execute("DELETE FROM transcript WHERE run_id=? AND kind=?", (run_id, kind))
        return self.append_transcript(kind, payload, run_id)

    def recent_transcript(self, run_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT sequence, kind, payload, created_at FROM transcript WHERE run_id=? ORDER BY sequence"
        params: tuple[Any, ...] = (run_id,)
        if limit is not None:
            query = "SELECT sequence, kind, payload, created_at FROM (" + query + " DESC LIMIT ?) ORDER BY sequence"
            params = (run_id, limit)
        rows = self.connection.execute(query, params).fetchall()
        return [{"sequence": row["sequence"], "kind": row["kind"], "payload": json.loads(row["payload"]), "created_at": row["created_at"]} for row in rows]

    def react_history_for_memory(self, run_id: str) -> list[dict[str, Any]]:
        """Return the latest ReAct message snapshot without reasoning traces."""
        row = self.connection.execute(
            "SELECT payload FROM transcript WHERE run_id=? AND kind='react_history' ORDER BY sequence DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row is None:
            return []
        payload = json.loads(row[0])
        history = payload.get("messages", []) if isinstance(payload, dict) else []
        if not isinstance(history, list):
            return []
        return [
            {key: value for key, value in message.items() if key != "reasoning_content"}
            for message in history
            if isinstance(message, dict)
        ]

    def add_inbox_event(self, event_id: str, kind: str, payload: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO inbox(event_id, kind, payload, consumed_at) VALUES (?, ?, ?, NULL)",
            (event_id, kind, json.dumps(payload)),
        )

    def pending_inbox(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM inbox WHERE consumed_at IS NULL ORDER BY rowid LIMIT ?", (limit,)
        ).fetchall()
        return [{"event_id": r["event_id"], "kind": r["kind"], "payload": json.loads(r["payload"])} for r in rows]

    def pending_owner_messages(self, limit: int = 10) -> list[dict[str, Any]]:
        """Pending owner messages (kind ``user_message``), oldest first.

        A dedicated query rather than ``pending_inbox`` filtering: ``limit`` must
        bound owner messages, not every pending kind, or a backlog of answers and
        other notifications would crowd the requested messages out of the window.
        Read-only by construction -- it never touches ``consumed_at``.
        """
        try:
            capped = max(1, int(limit))
        except (TypeError, ValueError):
            capped = 10
        rows = self.connection.execute(
            "SELECT * FROM inbox WHERE consumed_at IS NULL AND kind='user_message' ORDER BY rowid LIMIT ?",
            (capped,),
        ).fetchall()
        return [{"event_id": r["event_id"], "kind": r["kind"], "payload": json.loads(r["payload"])} for r in rows]

    def consume_inbox(self, event_id: str) -> None:
        self.connection.execute("UPDATE inbox SET consumed_at=? WHERE event_id=? AND consumed_at IS NULL", (utc_now(), event_id))

    def count_restart_failures(self, proposal_id: str) -> int:
        """Count durable restart failures recorded for one promoted proposal.

        The counter must survive the next_plan overwrite that every run performs,
        so it is derived from the event log instead of scheduler state.
        """
        if not proposal_id:
            return 0
        rows = self.connection.execute(
            "SELECT payload FROM event_log WHERE kind='restart_failed' ORDER BY sequence DESC LIMIT 100"
        ).fetchall()
        count = 0
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and str(payload.get("proposal_id")) == proposal_id:
                count += 1
        return count

    def record_restart_failure(self, *, proposal_id: str, commit: str, error: str, limit: int, source: str) -> int:
        """Record one restart failure and escalate exactly once at the limit.

        Escalation is durable and idempotent: the Nth failure (N == limit)
        emits the escalation event and one outbox alert; later failures keep
        counting but do not spam the operator channel.
        """
        with self.transaction():
            self.append_event(
                "restart_failed",
                {"error": error[:1000], "proposal_id": proposal_id, "commit": commit, "source": source},
            )
            failures = self.count_restart_failures(proposal_id)
            if limit > 0 and failures == limit:
                self.append_event(
                    "restart_escalated",
                    {"proposal_id": proposal_id, "failures": failures, "commit": commit, "source": source},
                )
                self.add_outbox(
                    "restart_escalation",
                    {
                        "proposal_id": proposal_id,
                        "commit": commit,
                        "failures": failures,
                        "message": "self-improvement promotion succeeded but the process restart keeps failing; the running code is stale",
                    },
                )
        return failures

    def next_random(self) -> float:
        """A deterministic, durable draw in [0, 1) for seeded exploration.

        The seed is created once and never changes, so replaying the same number
        of draws reproduces the same decisions. Every draw is journalled through
        planner_decisions, which is what makes an exploration run auditable.
        """
        row = self.connection.execute("SELECT seed, draws FROM rng_state WHERE id=1").fetchone()
        if row is None:
            seed = random.SystemRandom().randrange(1, 2**31)
            self.connection.execute("INSERT OR IGNORE INTO rng_state(id, seed, draws) VALUES (1, ?, 0)", (seed,))
            draws = 0
        else:
            seed, draws = int(row["seed"]), int(row["draws"])
        digest = hashlib.sha256(f"{seed}:{draws}".encode()).digest()
        self.connection.execute("UPDATE rng_state SET draws=draws+1 WHERE id=1")
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    def rng_seed(self) -> int | None:
        row = self.connection.execute("SELECT seed FROM rng_state WHERE id=1").fetchone()
        return int(row["seed"]) if row else None

    def affect_valence(self) -> tuple[float, int]:
        """The single persisted valence channel, as ``(value, draws)``.

        One channel, not seven: a taxonomy of moods is worth building only after
        one channel has been shown to change a decision. With no row the channel
        is inert (``VALENCE_PRIOR``, zero draws), so selection is unchanged.
        """
        from .planner import VALENCE_PRIOR

        row = self.connection.execute("SELECT valence, draws FROM affect_state WHERE id=1").fetchone()
        if row is None:
            return (VALENCE_PRIOR, 0)
        return (max(0.0, min(float(row["valence"]), 1.0)), max(0, int(row["draws"])))

    def charge_affect_valence(self, *, window: int) -> dict[str, Any]:
        """Charge the channel with the progress rate of the newest ``window`` runs.

        The value IS the window's measured rate -- it is computed here from the
        same rows the draw count comes from, so the reading, the confidence and
        the reported counts cannot be supplied independently and then disagree.
        Stored as ``(value, draws)`` and never as a bare value: the count is both
        the confidence the tilt uses and the selection bias, so a channel with one
        observation cannot swing a decision.

        Idempotent by construction: the charge is a function of the current
        window, not a running total, so charging an unchanged window twice leaves
        ``affect_state`` identical. With no finished runs the channel is set to
        the inert prior rather than inventing a reading from nothing.
        """
        from .planner import VALENCE_PRIOR

        window_rows = self.valence_window(window=window)
        stats = window_rows["stats"]
        rows = window_rows["rows"]
        draws = len(rows)
        value = VALENCE_PRIOR if not rows else max(0.0, min(float(stats["progress"]), 1.0))
        source_run = str(rows[0]["run_id"]) if rows else None
        self.connection.execute(
            "INSERT INTO affect_state(id, valence, draws, source_run, updated_at) VALUES (1, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET valence=excluded.valence, draws=excluded.draws, "
            "source_run=excluded.source_run, updated_at=excluded.updated_at",
            (value, draws, source_run, utc_now()),
        )
        return {
            "valence": value,
            "draws": draws,
            "window": int(window_rows["window"]),
            "source_run": source_run,
            **stats,
        }

    def valence_window(self, *, window: int) -> dict[str, Any]:
        """The one definition of the valence window: the newest terminal runs.

        Both the charging write and the observational reader go through here. A
        duplicated window predicate would let the charge and the reported rate
        disagree about which runs they were looking at, which is exactly the
        class of defect this channel exists to be measured against.
        """
        bounded_window = max(1, int(window))
        rows = self.connection.execute(
            "SELECT status, run_id, finished_at FROM runs WHERE finished_at IS NOT NULL ORDER BY finished_at DESC LIMIT ?",
            (bounded_window,),
        ).fetchall()
        statuses = [str(row["status"]) for row in rows]
        completed = sum(1 for status in statuses if status == "completed")
        return {
            "window": bounded_window,
            "rows": rows,
            "stats": {
                "labelled_cases": len(statuses),
                "completed": completed,
                "progress": (completed / len(statuses)) if statuses else 0.0,
                "statuses": statuses,
                "latest_finished_at": rows[0]["finished_at"] if rows else None,
            },
        }

    def add_goal(self, title: str, priority: float = 0.0, goal_id: str | None = None, *, constraints: dict[str, Any] | None = None) -> str:
        from uuid import uuid4
        goal_id = goal_id or str(uuid4())
        now = utc_now()
        self.connection.execute(
            "INSERT INTO goals(goal_id, title, status, priority, constraints, next_action, outcome, created_at, updated_at) VALUES (?, ?, 'active', ?, ?, '', '{}', ?, ?)",
            (goal_id, title, priority, json.dumps(constraints or {}, ensure_ascii=False), now, now),
        )
        return goal_id

    def active_work(self, limit: int = 20) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        goals = self.connection.execute("SELECT * FROM goals WHERE status='active' ORDER BY priority DESC, created_at LIMIT ?", (limit,)).fetchall()
        tasks = self.connection.execute("SELECT * FROM tasks WHERE status IN ('pending','running') ORDER BY deadline IS NULL, deadline LIMIT ?", (limit,)).fetchall()
        return ([dict(r) for r in goals], [dict(r) for r in tasks])

    def planner_candidates(self, limit: int = 16, *, terminal_ttl_days: float = 30.0) -> list[dict[str, Any]]:
        """Rankable work, grouped by area rather than by goal.

        `workstream_id` used to be `goal_id`, but goals are created only at
        genesis, so the advertised four-workstream portfolio was unreachable by
        construction. `area` gives one goal several genuinely different
        directions to compare.

        `fingerprint_state` replaces the old novelty/repetition pair, which were
        computed from the same EXISTS predicate and therefore carried one bit of
        information in two columns. A terminal hypothesis stops suppressing
        novelty after `terminal_ttl_days`, otherwise the hypothesis space is
        consumed monotonically until the planner always finds nothing.
        """
        cutoff = (utc_datetime_now() - timedelta(days=max(0.0, terminal_ttl_days))).isoformat().replace("+00:00", "Z")
        rows = self.connection.execute(
            """SELECT g.goal_id, g.title, g.priority, t.task_id, t.area,
                      t.hypothesis_fingerprint, t.structural_fingerprint, t.attempts,
                      CASE WHEN t.hypothesis_fingerprint IS NULL THEN 'new'
                           WHEN EXISTS (SELECT 1 FROM hypotheses h
                                        WHERE (h.fingerprint=t.hypothesis_fingerprint OR h.structural_fingerprint=t.structural_fingerprint)
                                          AND h.status IN ('completed','rejected','exhausted')
                                          AND h.updated_at >= ?) THEN 'terminal'
                           ELSE 'live' END AS fingerprint_state,
                      (SELECT CASE WHEN COUNT(*)=0 THEN 0.0 ELSE
                          CAST(SUM(CASE WHEN h2.status='completed' THEN 1 ELSE 0 END) AS REAL)/COUNT(*) END
                       FROM tasks t2 JOIN hypotheses h2 ON h2.fingerprint=t2.hypothesis_fingerprint
                       WHERE t2.area=t.area AND h2.status IN ('completed','rejected','exhausted')) AS area_success_rate
               FROM goals g JOIN tasks t ON t.goal_id=g.goal_id
                 WHERE g.status='active' AND t.status IN ('pending','running')
               ORDER BY g.priority DESC, t.created_at LIMIT ?""", (cutoff, limit)
        ).fetchall()
        return [dict(row) for row in rows]

    def record_planner_decision(self, candidates: list[Any], selected: Any | None, *, reason: str | None = None, draw: float | None = None, threshold: float | None = None, base_threshold: float | None = None) -> None:
        from uuid import uuid4
        state = self.state()
        payload = [
            {
                "workstream_id": item.workstream_id,
                "task_id": item.task_id,
                "score": round(item.score, 6),
                "novelty": item.novelty,
                "repetition_penalty": item.repetition_penalty,
                "expected_value": getattr(item, "expected_value", 0.0),
            }
            for item in candidates
        ]
        chosen_reason = reason or (selected.reason if selected else "no novel work")
        details: dict[str, Any] = {
            "selected_workstream_id": selected.workstream_id if selected else None,
            "selected_task_id": selected.task_id if selected else None,
            "candidates": payload,
            "reason": chosen_reason,
        }
        if draw is not None:
            details["exploration"] = {
                "seed": self.rng_seed(),
                "draw": round(draw, 6),
                "chosen_rank": candidates.index(selected) if selected in candidates else None,
            }
            # The threshold the draw was decided against is part of the record:
            # without it a later replay has to re-derive why the draw flared,
            # and a channel that influenced selection would be invisible.
            if threshold is not None:
                details["exploration"]["threshold"] = round(float(threshold), 6)
                if base_threshold is not None:
                    # The untilted arm, so the counterfactual is decidable from the
                    # record alone: a replay can compare the rank the draw produced
                    # against the rank the same draw would have produced with the
                    # channel absent, without knowing the config of that day. Legacy
                    # rows written before the valence channel existed carry neither
                    # key and keep their shape: the writer never invents it.
                    details["exploration"]["base_threshold"] = round(float(base_threshold), 6)
                if hasattr(self, "affect_valence"):
                    value, draws = self.affect_valence()
                    if draws:
                        details["exploration"]["valence"] = {"value": round(value, 6), "draws": draws}
        self.connection.execute("INSERT INTO planner_decisions(decision_id, generation, selected_workstream_id, selected_task_id, candidates, reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (str(uuid4()), state.generation, details["selected_workstream_id"], details["selected_task_id"], json.dumps(payload), chosen_reason, utc_now()))
        self.append_event("planner_decision", details)

    def task_work(self, task_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT t.*, g.title AS goal_title, g.priority AS goal_priority FROM tasks t LEFT JOIN goals g ON g.goal_id=t.goal_id WHERE t.task_id=? AND t.status IN ('pending','running') AND (t.goal_id IS NULL OR g.status='active')", (task_id,)).fetchone()
        return {"kind": "task", "task": dict(row)} if row else None

    def add_task(self, title: str, goal_id: str | None = None, deadline: str | None = None, task_id: str | None = None, *, expected_new_fact: str = "", hypothesis_fingerprint: str | None = None, structural_fingerprint: str | None = None, area: str = "general") -> str:
        from uuid import uuid4

        from .planner import hypothesis_fingerprint as make_hypothesis_fingerprint
        from .planner import structural_fingerprint as make_structural_fingerprint
        task_id = task_id or str(uuid4())
        now = utc_now()
        expected_new_fact = expected_new_fact or title
        hypothesis_fingerprint = hypothesis_fingerprint or make_hypothesis_fingerprint(area=title, problem=title, expected_behavior=expected_new_fact)
        structural_fingerprint = structural_fingerprint or make_structural_fingerprint(area=title, target=title, behavior_kind="task")
        self.connection.execute("INSERT INTO tasks(task_id, goal_id, title, status, attempts, deadline, idempotency_key, created_at, updated_at, hypothesis_fingerprint, structural_fingerprint, expected_new_fact, area) VALUES (?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?, ?, ?, ?)", (task_id, goal_id, title, deadline, task_id, now, now, hypothesis_fingerprint, structural_fingerprint, expected_new_fact, area or "general"))
        self.register_hypothesis(
            workstream_id=goal_id or "unscoped",
            fingerprint=hypothesis_fingerprint,
            structural_fingerprint=structural_fingerprint,
            problem=title,
            expected_behavior=expected_new_fact,
        )
        return task_id

    def best_in_cell(self, cell: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM idea_archive WHERE cell_key=? AND status='active' ORDER BY quality DESC LIMIT 1",
            (str(cell),),
        ).fetchone()
        return dict(row) if row is not None else None

    def archived_ideas(self, *, status: str = "active", limit: int = 500) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM idea_archive WHERE status=? ORDER BY quality DESC LIMIT ?",
            (str(status), int(limit)),
        ).fetchall()
        return [dict(row) for row in rows]

    def idea_cells(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT cell_key, COUNT(*) AS n FROM idea_archive WHERE status='active' GROUP BY cell_key"
        ).fetchall()
        return {str(row["cell_key"]): int(row["n"]) for row in rows}

    def archive_idea(self, idea: Mapping[str, Any]) -> str | None:
        """Admit one idea under the MAP-Elites cell rule.

        A cell keeps exactly one active occupant, so the archive stays bounded
        and cell coverage cannot be inflated with duplicates. Admission is
        exclusive epsilon-dominance over quality AND novelty
        (`_admits_over_incumbent`, arXiv:1708.09251 sec. 3.1.2): a candidate
        that ties on quality but is more novel takes the cell, instead of being
        discarded anonymously because quality alone did not strictly improve.
        """
        from uuid import uuid4

        cell = str(idea["cell_key"])
        quality = float(idea.get("quality", 0.0) or 0.0)
        incumbent = self.best_in_cell(cell)
        idea_id = str(uuid4())
        now = utc_now()
        novelty = float(idea.get("novelty", 0.0) or 0.0)
        if incumbent is not None:
            incumbent_quality = float(incumbent["quality"])
            incumbent_novelty = float(incumbent["novelty"])
            if not _admits_over_incumbent(quality, novelty, incumbent_quality, incumbent_novelty):
                self.append_event("idea_archive_rejected", {
                    "cell_key": cell,
                    "reason": "incumbent is not strictly dominated",
                    "incumbent_quality": incumbent_quality,
                    "candidate_quality": quality,
                    "incumbent_novelty": incumbent_novelty,
                    "candidate_novelty": novelty,
                    "epsilon": _ARCHIVE_EPSILON,
                })
                return None
            self.connection.execute(
                "UPDATE idea_archive SET status='superseded', superseded_by=?, updated_at=? WHERE idea_id=?",
                (idea_id, now, incumbent["idea_id"]),
            )
            self.append_event("idea_superseded", {
                "cell_key": cell, "superseded": incumbent["idea_id"], "by": idea_id,
            })
        parent = idea.get("parent_id")
        depth = 0
        if parent:
            row = self.connection.execute(
                "SELECT lineage_depth FROM idea_archive WHERE idea_id=?", (str(parent),)
            ).fetchone()
            depth = int(row["lineage_depth"]) + 1 if row is not None else 1
        self.connection.execute(
            """INSERT INTO idea_archive(idea_id, parent_id, lineage_depth, subsystem, change_type,
                   evidence_source, cell_key, title, problem_description, hypothesis,
                   expected_new_fact, validation, inspiration_ref, quality, novelty, status,
                   children, task_id, proposal_id, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'active',0,?,?,?,?)""",
            (
                idea_id, parent, depth, idea["subsystem"], idea["change_type"], idea["evidence_source"],
                cell, idea["title"], idea.get("problem_description", ""), idea.get("hypothesis", ""),
                idea.get("expected_new_fact", ""), idea.get("validation", ""), idea.get("inspiration_ref", ""),
                quality, novelty,
                idea.get("task_id"), idea.get("proposal_id"), now, now,
            ),
        )
        self.append_event("idea_archived", {
            "idea_id": idea_id, "cell_key": cell, "quality": round(quality, 6),
            "evidence_source": idea["evidence_source"], "lineage_depth": depth,
            "inspiration_ref": str(idea.get("inspiration_ref", ""))[:300],
        })
        return idea_id

    def sample_archive_parents(self, *, k: int = 2, lam: float = 10.0, alpha0: float = 0.5) -> list[dict[str, Any]]:
        """Sample up to k parents without replacement, DGM-weighted.

        w = sigmoid(lam*(quality-alpha0)) / (1+children). Saturated ideas
        (quality >= 1.0) are excluded; every other archived idea keeps a
        non-zero chance, so no improvement path becomes unreachable. Draws come
        from the journalled RNG, so a replay reproduces the same lineage.
        """
        from .idea_archive import parent_weight

        eligible = [item for item in self.archived_ideas(status="active") if float(item["quality"]) < 1.0]
        if not eligible:
            return []
        weights = [
            parent_weight(float(item["quality"]), int(item["children"]), lam=lam, alpha0=alpha0)
            for item in eligible
        ]
        picked: list[dict[str, Any]] = []
        for _ in range(max(1, min(int(k), len(eligible)))):
            total = sum(weights)
            if total <= 0.0:
                break
            draw = self.next_random() * total
            accumulated = 0.0
            index = len(eligible) - 1
            for i, weight in enumerate(weights):
                accumulated += weight
                if draw < accumulated:
                    index = i
                    break
            picked.append(eligible.pop(index))
            weights.pop(index)
        for item in picked:
            self.bump_idea_children(str(item["idea_id"]))
        return picked

    def bump_idea_children(self, idea_id: str) -> None:
        self.connection.execute(
            "UPDATE idea_archive SET children=children+1, updated_at=? WHERE idea_id=?",
            (utc_now(), str(idea_id)),
        )

    def record_idea_outcome(self, idea_id: str, *, quality: float, status: str) -> None:
        """Close the loop: an archived idea's measured quality is what ranks it."""
        bounded = max(0.0, min(float(quality), 1.0))
        self.connection.execute(
            "UPDATE idea_archive SET quality=?, status=?, updated_at=? WHERE idea_id=?",
            (bounded, str(status), utc_now(), str(idea_id)),
        )

    def materialize_idea(self, idea_id: str, goal_id: str) -> str:
        """Turn an archived stepping stone into an executable task.

        DGM formulates the child as a problem description rather than a diff, so
        an archived idea can be re-opened long after it was first written.
        """
        from .planner import hypothesis_fingerprint, structural_fingerprint

        row = self.connection.execute("SELECT * FROM idea_archive WHERE idea_id=?", (str(idea_id),)).fetchone()
        if row is None:
            raise KeyError(idea_id)
        idea = dict(row)
        title = str(idea["title"])[:300]
        fact = str(idea["expected_new_fact"] or title)
        task_id = self.add_task(
            title,
            goal_id,
            expected_new_fact=fact,
            hypothesis_fingerprint=hypothesis_fingerprint(
                area=str(idea["subsystem"]),
                problem=str(idea["problem_description"] or title),
                expected_behavior=fact,
                selector=[str(idea["cell_key"])],
            ),
            structural_fingerprint=structural_fingerprint(
                area=str(idea["subsystem"]),
                target=str(idea["problem_description"] or title),
                behavior_kind=str(idea["change_type"]),
            ),
            area=str(idea["subsystem"]),
        )
        self.connection.execute(
            "UPDATE idea_archive SET task_id=?, status='materialized', updated_at=? WHERE idea_id=?",
            (task_id, utc_now(), str(idea_id)),
        )
        self.append_event("idea_materialized", {
            "idea_id": str(idea_id), "task_id": task_id, "cell_key": str(idea["cell_key"]),
        })
        return task_id

    def apply_goal_updates(self, updates: list[dict[str, Any]], run_id: str | None = None) -> None:
        for item in updates:
            goal_id = item.get("goal_id")
            if not isinstance(goal_id, str):
                continue
            status = str(item.get("status", "active"))
            if status not in {"active", "blocked", "completed", "cancelled"}:
                continue
            updated = self.connection.execute("UPDATE goals SET status=?, next_action=?, outcome=?, updated_at=? WHERE goal_id=?", (status, str(item.get("next_action", "")), json.dumps(item.get("outcome", {})), utc_now(), goal_id)).rowcount
            if updated and run_id:
                self.append_event("goal_progress", {"goal_id": goal_id, "status": status, "next_action": str(item.get("next_action", "")), "outcome": item.get("outcome", {})}, run_id)

    def apply_task_updates(self, updates: list[dict[str, Any]], run_id: str | None = None) -> None:
        for item in updates:
            task_id = item.get("task_id")
            if not isinstance(task_id, str):
                continue
            # `status` is not required by the memory-response schema. A missing
            # status is an outcome-only update, so leave the current status
            # untouched instead of defaulting to "pending" and silently
            # reopening a task the run just completed.
            raw_status = item.get("status")
            if raw_status is None:
                continue
            status = str(raw_status)
            if status in {"pending", "running", "completed", "blocked", "cancelled"}:
                updated = self.connection.execute(
                    "UPDATE tasks SET status=?, updated_at=? WHERE task_id=?",
                    (status, utc_now(), task_id),
                ).rowcount
                if updated and run_id:
                    attempts = self.connection.execute("SELECT attempts FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0]
                    self.append_event("task_progress", {"task_id": task_id, "status": status, "attempts": attempts}, run_id)
                if updated and status in {"completed", "blocked", "cancelled"}:
                    fingerprint_row = self.connection.execute("SELECT hypothesis_fingerprint FROM tasks WHERE task_id=?", (task_id,)).fetchone()
                    if fingerprint_row and fingerprint_row[0]:
                        hypothesis_status = "completed" if status == "completed" else "exhausted"
                        self.mark_hypothesis(str(fingerprint_row[0]), hypothesis_status, {"task_id": task_id, "status": status, "source": "task_update"})

    def increment_task_attempts(self, task_id: str) -> int:
        """Count one started run against the task and return the new total.

        ``attempts`` is the number of runs the planner actually started for the
        task, not the number of status transitions. A NEEDS_RECOVERY loop keeps
        the task pending, so a transition-based counter stayed frozen at zero
        and the planner's attempt penalty never engaged.
        """
        self.connection.execute("UPDATE tasks SET attempts=attempts+1, updated_at=? WHERE task_id=?", (utc_now(), task_id))
        row = self.connection.execute("SELECT attempts FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return int(row[0]) if row else 0

    def record_task_failure(self, task_id: str) -> int:
        """Count one model-side failure and return the consecutive total.

        Provider outages are transient and must not consume a task's give-up
        budget, so the caller only invokes this for failures produced by the
        model or harness (invalid Finish Report, doom loop, exhausted budget).
        """
        self.connection.execute(
            "UPDATE tasks SET consecutive_model_failures=consecutive_model_failures+1, updated_at=? WHERE task_id=?",
            (utc_now(), task_id),
        )
        row = self.connection.execute("SELECT consecutive_model_failures FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return int(row[0]) if row else 0

    def reset_task_failures(self, task_id: str) -> None:
        self.connection.execute("UPDATE tasks SET consecutive_model_failures=0 WHERE task_id=?", (task_id,))

    def repair_status_consistency(self, *, exclude_fingerprints: Iterable[str] = ()) -> dict[str, int]:
        """Remove task rows that can never be selected again.

        A pending task whose hypothesis is already terminal has zero novelty, so
        the planner skips it forever while it still clutters the queue. A
        cancelled task's ``ready`` hypothesis is intentionally left alone: the
        autonomous planner's dedup treats only terminal hypotheses as occupied,
        so a ready hypothesis with no live task stays proposable (P0-7).

        ``exclude_fingerprints`` keeps the stable safety-net tasks out of the
        cancel branch: they stay pending with a terminal fingerprint by design
        and are selected directly. The caller supplies them so this module does
        not have to import reactor constants (circular import).

        That exclusion is scoped to the NEWEST live instance of each family.
        Excluding the fingerprint outright exempted every stacked copy forever:
        measured on the live ledger (2026-09-25, read-only copy, 6 pending
        instances of the external-seek template, oldest 2026-09-21), all six read
        ``fingerprint_state='terminal'`` by ``planner_candidates`` and none was
        selectable by ``PortfolioPlanner.rank()``, so the repair kept rows the
        planner is already paying to skip. The safety net needs one live
        instance to stay reachable; the older copies are stale work. A
        ``running`` instance is never retired by this path.

        Idempotent: a second call repairs nothing.
        """
        now = utc_now()
        exclusions = tuple(str(item) for item in exclude_fingerprints if item)
        cancel_sql = """UPDATE tasks SET status='cancelled', updated_at=?
                   WHERE status IN ('pending', 'running')
                     AND hypothesis_fingerprint IN (
                         SELECT fingerprint FROM hypotheses WHERE status IN ('completed', 'rejected', 'exhausted')
                     )"""
        cancel_params: tuple[Any, ...] = (now,)
        if exclusions:
            placeholders = ",".join("?" for _ in exclusions)
            cancel_sql += (
                f" AND (hypothesis_fingerprint NOT IN ({placeholders})"
                "        OR (status <> 'running' AND created_at < ("
                "              SELECT MAX(t2.created_at) FROM tasks t2"
                "               WHERE t2.hypothesis_fingerprint = tasks.hypothesis_fingerprint"
                "                 AND t2.status IN ('pending', 'running'))))"
            )
            cancel_params = (now, *exclusions)
        with self.transaction():
            aligned_hypotheses = self.connection.execute(
                """UPDATE hypotheses SET status='completed', updated_at=?
                   WHERE status='ready'
                     AND fingerprint IN (
                         SELECT hypothesis_fingerprint FROM tasks
                         WHERE status='completed' AND hypothesis_fingerprint IS NOT NULL
                     )""",
                (now,),
            ).rowcount
            cancelled_tasks = self.connection.execute(cancel_sql, cancel_params).rowcount
            if aligned_hypotheses or cancelled_tasks:
                self.append_event("status_repair", {"hypotheses": aligned_hypotheses, "tasks": cancelled_tasks})
        return {"hypotheses": aligned_hypotheses, "tasks": cancelled_tasks}

    def backfill_missing_run_results(self) -> int:
        """Give every finished run a durable result row so recovery state is complete.

        Runs interrupted before a result was committed otherwise leave a gap in
        the run history that no later reader can fill.
        """
        with self.transaction():
            rows = self.connection.execute(
                "SELECT r.run_id, r.status FROM runs r LEFT JOIN run_results rr ON rr.run_id = r.run_id "
                "WHERE rr.run_id IS NULL AND r.status <> 'running'"
            ).fetchall()
            for row in rows:
                self.connection.execute(
                    "INSERT OR IGNORE INTO run_results(run_id, status, report, steps, usage_tokens, failure, created_at) VALUES (?, ?, ?, 0, 0, ?, ?)",
                    (row["run_id"], row["status"], "Run ended without a committed result (harness recovery backfill).", "missing_result_backfilled", utc_now()),
                )
        if rows:
            self.append_event("run_results_backfilled", {"count": len(rows)})
        return len(rows)

    def consolidate(self, run_id: str, candidates: list[dict[str, Any]]) -> int:
        return int(self._consolidate_candidates(run_id, candidates)["added"])

    def _consolidate_candidates(self, run_id: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply candidates and report what landed for the operator log."""
        added = 0
        superseded = 0
        memory_ids: list[str] = []
        for item in candidates:
            if isinstance(item, str):
                item = {"content": item}
            if not isinstance(item, dict):
                continue
            content = item.get("content") or item.get("text")
            if not isinstance(content, str) or not content.strip():
                continue
            kind = normalize_memory_kind(str(item.get("kind", "observation")))
            confidence = _bounded_model_float(item.get("confidence", 0.5), default=0.5)
            # The candidate's own citation belongs on the row it supports, not
            # only on the supersede note it may also carry.
            row_evidence = _model_evidence(item.get("evidence"))
            self.connection.execute("INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at,evidence) VALUES (lower(hex(randomblob(16))),?,?,?,?,?,?) ON CONFLICT(kind,content) DO UPDATE SET confidence=max(confidence, excluded.confidence), source_run=excluded.source_run, updated_at=excluded.updated_at, evidence=COALESCE(excluded.evidence, memories.evidence)", (kind, content.strip(), max(0.0, min(confidence, 1.0)), run_id, utc_now(), row_evidence))
            added += 1
            memory_id = self.connection.execute("SELECT memory_id FROM memories WHERE kind=? AND content=?", (kind, content.strip())).fetchone()[0]
            memory_ids.append(str(memory_id))
            if self.memory_store is not None:
                self.memory_store.index_memory(memory_id, kind, content.strip())
            supersedes_id = item.get("supersedes_memory_id")
            if isinstance(supersedes_id, str) and supersedes_id:
                exists = self.connection.execute("SELECT 1 FROM memories WHERE memory_id=?", (supersedes_id,)).fetchone()
                if exists is not None:
                    note = row_evidence or content.strip()
                    self.supersede_memory(supersedes_id, str(memory_id), note)
                    superseded += 1
        return {"added": added, "superseded": superseded, "memory_ids": memory_ids}

    def consolidate_versioned(self, episode_id: str, run_id: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply one episode's memory changes exactly once in the current transaction."""
        existing = self.connection.execute(
            "SELECT input_version, output_version, memory_count FROM memory_consolidations WHERE episode_id=?",
            (episode_id,),
        ).fetchone()
        if existing:
            return {"replayed": True, "input_version": existing[0], "output_version": existing[1], "memory_count": existing[2]}
        input_version = int(self.connection.execute("SELECT version FROM memory_meta WHERE id=1").fetchone()[0])
        details = self._consolidate_candidates(run_id, candidates)
        count = int(details["added"])
        output_version = input_version + 1
        self.connection.execute("UPDATE memory_meta SET version=? WHERE id=1", (output_version,))
        self.connection.execute(
            "INSERT INTO memory_consolidations(episode_id, run_id, input_version, output_version, memory_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (episode_id, run_id, input_version, output_version, count, utc_now()),
        )
        # A compact, bounded digest so the owner/operator can audit a
        # consolidation from the event log without opening the database.
        self.append_event(
            "memory_event",
            {"run_id": run_id, "added": count, "superseded": details["superseded"], "memory_ids": details["memory_ids"][:20]},
            run_id,
        )
        return {"replayed": False, "input_version": input_version, "output_version": output_version, "memory_count": count}

    def remember_memory(
        self,
        content: str,
        *,
        kind: str = "observation",
        confidence: float = 0.6,
        pinned: bool = False,
        source_run: str | None = None,
        evidence: str | None = None,
    ) -> str:
        """Store one explicit memory and return its id.

        Upserts on `(kind, content)` like `consolidate`: a repeat raises the
        confidence to the maximum instead of fragmenting the row. The memory is
        born `active` with `valid_from` set, and is indexed for search.
        """
        normalized_kind = normalize_memory_kind(str(kind))
        bounded = max(0.0, min(_bounded_model_float(confidence, default=0.6), 1.0))
        now = utc_now()
        self.connection.execute(
            "INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at,pinned,status,valid_from,evidence) "
            "VALUES (lower(hex(randomblob(16))),?,?,?,?,?,?, 'active', ?, ?) "
            "ON CONFLICT(kind,content) DO UPDATE SET confidence=max(confidence, excluded.confidence), "
            "source_run=excluded.source_run, updated_at=excluded.updated_at, "
            "pinned=max(pinned, excluded.pinned), status='active', "
            "valid_from=COALESCE(memories.valid_from, excluded.valid_from), "
            "evidence=COALESCE(excluded.evidence, memories.evidence)",
            (normalized_kind, content.strip(), bounded, source_run, now, 1 if pinned else 0, now, evidence),
        )
        memory_id = self.connection.execute(
            "SELECT memory_id FROM memories WHERE kind=? AND content=?", (normalized_kind, content.strip())
        ).fetchone()[0]
        if self.memory_store is not None:
            self.memory_store.index_memory(str(memory_id), normalized_kind, content.strip())
        return str(memory_id)

    def forget_memory(self, memory_id: str) -> bool:
        """Delete one memory row and its search projection."""
        deleted = self.connection.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,)).rowcount
        if deleted and self.memory_store is not None:
            self.memory_store.drop_index([memory_id])
        return bool(deleted)

    def correct_memory(
        self,
        memory_id: str,
        *,
        content: str,
        evidence: str,
        confidence: float | None = None,
        source_run: str | None = None,
    ) -> str:
        """Replace a memory with a corrected version, keeping the audit trail.

        The corrected memory is inserted as a fresh `active` row; the old row is
        marked `superseded`, linked via `superseded_by` and closed with
        `valid_to`. Raises `ValueError` if `memory_id` is unknown.

        `source_run` attributes the successor to the run that made the
        correction. When the caller passes none, the predecessor's attribution is
        inherited: a correction must not launder away the only provenance the
        fact had (measured: 8 of 33 live supersede pairs had a NULL
        successor whose predecessor carried a run). An unknown run still yields
        NULL rather than an invented id.
        """
        row = self.connection.execute(
            "SELECT kind, confidence, source_run FROM memories WHERE memory_id=?", (memory_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown memory_id: {memory_id}")
        base_confidence = row["confidence"] if confidence is None else confidence
        if source_run is None:
            source_run = row["source_run"]
        new_id = self.remember_memory(
            content, kind=str(row["kind"]), confidence=base_confidence, evidence=evidence, source_run=source_run
        )
        self.supersede_memory(memory_id, new_id, evidence)
        return new_id

    def supersede_memory(self, old_id: str, new_id: str, evidence: str) -> bool:
        """Mark `old_id` superseded by `new_id` and close its validity window.

        A memory cannot supersede itself: when a correction repeats the original
        content, the upsert in `remember_memory` yields the same row, and marking
        it superseded would silently delete the corrected fact from search.
        """
        if old_id == new_id:
            return False
        rowcount = self.connection.execute(
            "UPDATE memories SET status='superseded', superseded_by=?, valid_to=?, "
            "evidence=COALESCE(?, evidence) WHERE memory_id=?",
            (new_id, utc_now(), evidence, old_id),
        ).rowcount
        return bool(rowcount)

    def classify_recovery(self, run_id: str) -> dict[str, Any]:
        """Classify the last action so recovery does not blindly replay unknown effects."""
        events = self.recent_run_events(run_id, limit=None)
        committed_result = self.run_result(run_id)
        calls = {str(event["payload"].get("call_id")): event for event in events if event["kind"] == "tool_call" and event["payload"].get("call_id")}
        results = {
            str(event["payload"].get("call", {}).get("call_id"))
            for event in events
            if event["kind"] == "tool_result" and event["payload"].get("call", {}).get("call_id")
        }
        unknown = [call_id for call_id in calls if call_id not in results]
        reconciled = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT call_id FROM recovery_reconciliations WHERE run_id=? AND status IN ('confirmed', 'not_applied')",
                (run_id,),
            )
        }
        unknown = [call_id for call_id in unknown if call_id not in reconciled]
        if unknown:
            return {"status": "unknown_outcome", "call_ids": unknown}
        if committed_result and not any(event["kind"] == "run_finished" for event in events):
            return {"status": "result_committed", "run_result": committed_result}
        if events and events[-1]["kind"] == "run_finished":
            return {"status": "finished"}
        return {"status": "retryable", "reason": "no_unresolved_tool_call"}

    def reconcile_tool_call(self, run_id: str, call_id: str, status: str, result: dict[str, Any]) -> dict[str, Any]:
        if status not in {"confirmed", "not_applied", "unknown"}:
            raise ValueError(f"invalid reconciliation status: {status}")
        existing = self.connection.execute(
            "SELECT status, result FROM recovery_reconciliations WHERE run_id=? AND call_id=?",
            (run_id, call_id),
        ).fetchone()
        if existing and existing[0] in {"confirmed", "not_applied"}:
            preserved_result = json.loads(existing[1])
            return {"run_id": run_id, "call_id": call_id, "status": existing[0], "result": preserved_result}
        self.connection.execute("INSERT OR REPLACE INTO recovery_reconciliations(run_id, call_id, status, result, created_at) VALUES (?, ?, ?, ?, ?)", (run_id, call_id, status, json.dumps(result), utc_now()))
        self.append_event("tool_reconciled", {"call_id": call_id, "status": status, "result": result}, run_id)
        return {"run_id": run_id, "call_id": call_id, "status": status, "result": result}

    def effect_reapplied_runs(self, capability: str, arguments_hash: str, *, exclude_run_id: str) -> list[str]:
        """Other runs that already applied this exact call, oldest first.

        ``capability_effects.idempotency_key`` is ``{run_id}:{step}:{call_id}``
        (skynet/react.py), so the ledger deduplicates one *slot* and not one
        *call*. A run retried after a crash carries a fresh run_id and re-applies
        work the previous attempt already applied, and the same is true inside a
        single run: measured with the real runner, an identical
        run_id, step and call_id executes the tool once, while the identical
        run_id and step with a regenerated call_id executes it again - and a
        resumed model turn regenerates the call_id, because ToolCall.call_id is a
        fresh uuid4. The live ledger already holds 5 identities applied twice
        inside one run_id. That is the at-least-once-across-crashes
        cell of the RESUME CONTRACT's ``effect exactly-once`` property
        (arXiv:2608.03836v3), where LangGraph 1.2.9 re-executes durably recorded
        work after a real SIGKILL. The contract needs the *identity* of the call,
        not its slot, so this reads the stable pair (capability, arguments_hash)
        the ledger already stores. It reports the fact and does not act on it: on
        the live database (re-measured, generation 39: 2281 rows) 12
        call identities repeat across runs and 8 of those 12 hold different results - ``db(sql="SELECT * FROM
        agent_state")`` returned 9972 bytes in one run and 12114 in the next - so
        serving a recorded result for a repeated pair would answer a live question
        with stale state. The false example this docstring used to carry is worth
        keeping as a warning: ``PRAGMA table_info(inbox)`` was claimed to differ
        between runs, and all 10 recorded results are row_count 4 / 271 bytes,
        identical. Because the returned list holds *every* previous run, a
        non-empty list is not a first repeat: it grows with each occurrence, and
        ``if previous_runs`` announced one identity once per repeat. The caller
        uses a non-empty list paired with ``effect_reapplied_announced`` as the
        one-time signal, so an identity that had already repeated before the
        signal existed is still announced exactly once. The event it writes is a
        pointer rather than a tally: its ``previous_run_count`` is the number of
        other runs at announcement time, which is the transition value 1 for the
        common case and can exceed 1 for a late-announced identity; a reader that
        needs the current count calls this method again with the payload's
        ``arguments_hash``.
        """
        # A policy warning is a refusal, not an application: ``skynet/tools.py``
        # returns the soft-denial result before running the command, so such a
        # run must not be reported as a run that already applied the call.
        # Measured on the live ledger, excluding them leaves all 40
        # announced identities and their ``previous_run_ids`` unchanged and
        # removes the warning-only false positives.
        rows = self.connection.execute(
            "SELECT DISTINCT substr(p.idempotency_key, 1, instr(p.idempotency_key, ':') - 1) FROM capability_effects AS p "
            "WHERE p.capability=? AND p.arguments_hash=? "
            "AND substr(p.idempotency_key, 1, instr(p.idempotency_key, ':') - 1) != ? "
            "AND " + _REFUSAL_RESULT_PREDICATE.format(alias="p") + " "
            "ORDER BY p.created_at",
            (capability, arguments_hash, exclude_run_id),
        ).fetchall()
        return [str(row[0]) for row in rows if row[0]]

    def effect_identity_prior_count(self, capability: str, arguments_hash: str, *, exclude_effect_key: str) -> int:
        """How many OTHER ledger rows already applied this call identity.

        A refusal is not an application: ``skynet/tools.py`` returns the
        soft-denial result before running the command, so a warning row must not
        count. Counting it made the one live ``same_run_repeat=true`` event a
        false positive (run 92e56e65, whose "prior application" was the warning
        row), and on the live ledger it inflated the in-run repeat
        census from the 2 identities that really executed twice to 16. The hard
        denylist (``policy_denied``) never executes either and is excluded by the
        same predicate (see ``_REFUSAL_RESULT_MARKERS``).

        ``effect_reapplied_runs`` excludes the current run on purpose (it names
        the other runs a reader can compare against), so a call re-issued
        *inside* one run with a regenerated ``ToolCall.call_id`` was invisible to
        every reader: measured on the live ledger, 15 identities hold
        two rows each inside a single run_id (14 bash, 1 read), all 15 pairs hold
        different results, and no durable event names any of them. This count is
        run-agnostic: it sees both cells of the RESUME CONTRACT's
        ``effect exactly-once`` violation (arXiv:2608.03836v3) - across crashes
        and inside one run. It reports the fact and never serves the recorded
        result, because differing results are the norm (15 of 15 in-run pairs).
        """
        row = self.connection.execute(
            "SELECT count(*) FROM capability_effects AS p WHERE p.capability=? AND p.arguments_hash=? "
            "AND p.idempotency_key!=? AND " + _REFUSAL_RESULT_PREDICATE.format(alias="p"),
            (capability, arguments_hash, exclude_effect_key),
        ).fetchone()
        return int(row[0]) if row else 0

    def effect_reapplied_announced(self, capability: str, arguments_hash: str) -> bool:
        """Whether the one-time ``effect_reapplied`` pointer already exists.

        ``len(previous_runs) == 1`` was the old announcement signal, which ties
        "once per identity" to "exactly one prior run": an identity that had
        already repeated twice before the signal existed never announced at all.
        The event is a protected kind (``PROTECTED_EVENT_KINDS``), so the row
        survives every retention window and is the durable record of the
        announcement; the caller pairs this check with a non-empty
        ``effect_reapplied_runs`` to keep the event once-only.
        """
        row = self.connection.execute(
            "SELECT 1 FROM event_log WHERE kind='effect_reapplied' "
            "AND json_extract(payload, '$.tool')=? AND json_extract(payload, '$.arguments_hash')=? LIMIT 1",
            (capability, arguments_hash),
        ).fetchone()
        return row is not None

    def record_effect(self, key: str, capability: str, arguments_hash: str, result: dict[str, Any], status: str) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO capability_effects(idempotency_key, capability, arguments_hash, result, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (key, capability, arguments_hash, json.dumps(result), status, utc_now()),
        )

    def record_evaluation(self, run_id: str, evaluation: dict[str, Any], status: RunStatus, summary: str) -> None:
        self.connection.execute(
            "INSERT INTO evaluations(run_id, status, summary, evaluation, created_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, status.value, summary, json.dumps(evaluation), utc_now()),
        )

    def record_metrics_snapshot(self, marker: str, data: dict[str, Any]) -> None:
        """Persist one experiment-metrics snapshot, idempotent per UTC day.

        ``evaluations`` was written every cycle and never read; this gives it a
        reader and a durable daily baseline instead of a separate table.
        """
        self.connection.execute(
            "INSERT INTO evaluations(run_id, status, summary, evaluation, created_at) VALUES (?, 'metrics', ?, ?, ?)",
            (METRICS_RUN_ID, marker, json.dumps(data, ensure_ascii=False), utc_now()),
        )

    def evaluate_criteria(self, run_id: str, criteria: list[dict[str, Any]], report: str, status: RunStatus) -> list[dict[str, Any]]:
        """Evaluate success criteria against durable tool evidence.

        The harness uses a single criterion kind (`verified_progress`): a run
        made verified progress only when at least one tool call succeeded.
        """
        events = self.recent_run_events(run_id, limit=None)
        evidence = [
            event["sequence"]
            for event in events
            if event["kind"] == "tool_result" and event["payload"].get("result", {}).get("ok") is True
        ]
        passed = bool(evidence)
        return [
            {
                "criterion": str(criterion.get("criterion", "")).strip(),
                "kind": "verified_progress",
                "passed": passed,
                "required": bool(criterion.get("required", True)),
                "verified": True,
                "evidence": evidence,
            }
            for criterion in criteria
        ]

    def effect(self, key: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT result FROM capability_effects WHERE idempotency_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def add_outbox(self, kind: str, payload: dict[str, Any], message_id: str | None = None) -> str:
        return self.outbox.add(kind, payload, message_id)

    def pending_outbox(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.outbox.pending(limit)

    def claim_outbox(self, limit: int = 50, lease_seconds: float = 300.0) -> list[dict[str, Any]]:
        with self.transaction():
            return self.outbox.claim(limit, lease_seconds)

    def mark_outbox_delivered(self, message_id: str) -> None:
        self.outbox.mark_delivered(message_id)

    def mark_outbox_failed(self, message_id: str, error: str, *, max_attempts: int = 5) -> str:
        return self.outbox.mark_failed(message_id, error, max_attempts=max_attempts)

    def raise_alert(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        severity: str = "warning",
        dedup_key: str | None = None,
        dedup_window_seconds: float = 21_600.0,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Escalate a condition to the owner channel exactly once per window.

        Transaction-agnostic on purpose: callers inside a durable accounting
        transaction keep their atomicity, and a collapsed repeat only bumps a
        counter instead of spamming the delivery queue.
        """
        alert_id, fresh, occurrences = self.alerts.raise_alert(
            kind,
            payload,
            severity=severity,
            dedup_key=dedup_key,
            dedup_window_seconds=dedup_window_seconds,
        )
        if fresh:
            self.append_event(
                "alert_raised",
                {"alert_id": alert_id, "kind": kind, "severity": severity, "payload": payload},
                run_id,
            )
            self.add_outbox(
                "alert",
                {"alert_id": alert_id, "kind": kind, "severity": severity, "occurrences": occurrences, "payload": payload},
            )
        return {"alert_id": alert_id, "new": fresh, "occurrences": occurrences}

    def ask_question(
        self,
        question: str,
        *,
        options: list[str] | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        goal_id: str | None = None,
        ttl_seconds: float = 86_400.0,
        question_key: str | None = None,
    ) -> str:
        """Persist a question to the owner and queue it for delivery.

        Durable before anything else: a question that only exists in a prompt is
        lost the moment the run is interrupted. Delivery rides the existing
        outbox lease path, so there is still exactly one transport.

        `question_key` is the CALL identity (`{run_id}:{step}:{call_id}`), and it
        makes the whole call replayable. `react.py` writes `capability_effects`
        only after the tool returns, so a crash in that window leaves the owner's
        question queued with no cached result; the resumed run re-issues the
        identical call and, with a fresh uuid4 as the key, asked the owner the
        same question twice - reproduced with the real tool on a fresh store:
        2 `user_questions` rows and 2 `user_question` outbox rows unpatched, 1
        patched. This is the same crash-duplication cell as the owner message
        (arXiv:2608.01710v1), where the row identity must be the call's durable
        identity rather than a fresh issuance.

        The call identity is NOT the row id. `telegram_bot.handle_answer` routes
        `/answer <8-char prefix>` and reports an ambiguous prefix as an error, and
        every call in one run shares the `{run_id}:` prefix - measured: two calls
        from run 40c694f7 both render as `40c694f7` - so reusing the key as the id
        would make the owner unable to answer either question. The id is derived
        from the key instead (`ask-{sha256(key)[:16]}`), which is stable across
        the replay and distinct per call.
        """
        from uuid import uuid4

        question_id = f"ask-{hashlib.sha256(question_key.encode()).hexdigest()[:16]}" if question_key else str(uuid4())
        now_dt = utc_datetime_now()
        inserted = self.connection.execute(
            "INSERT OR IGNORE INTO user_questions(question_id, run_id, task_id, goal_id, question, options, status, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)",
            (
                question_id,
                run_id,
                task_id,
                goal_id,
                question,
                json.dumps(options or [], ensure_ascii=False),
                utc_now(),
                (now_dt + timedelta(seconds=max(1.0, ttl_seconds))).isoformat().replace("+00:00", "Z"),
            ),
        ).rowcount
        if not inserted:
            # A replay of the same call: the row is already durable and queued, so
            # this is not a new question and must not announce or queue one.
            return question_id
        self.append_event(
            "question_asked",
            {"question_id": question_id, "question": question[:500], "options": options or []},
            run_id,
        )
        self.add_outbox(
            "user_question",
            {"question_id": question_id, "question": question, "options": options or [], "expires_at": (now_dt + timedelta(seconds=max(1.0, ttl_seconds))).isoformat().replace("+00:00", "Z")},
            message_id=question_key,
        )
        return question_id

    def answer_question(self, question_id: str, answer: str, *, source: str = "owner") -> bool:
        """Record an owner answer; returns False when the question is unknown or closed.

        The answer is also consolidated into a durable `user_dialogue` memory: the
        run that asked the question is usually long gone, so the answer only
        reaches a later wake through the memory index. A refused (already closed)
        answer writes nothing.
        """
        row = self.connection.execute(
            "SELECT question, status FROM user_questions WHERE question_id=?", (question_id,)
        ).fetchone()
        if row is None or str(row["status"]) != "open":
            return False
        self.connection.execute(
            "UPDATE user_questions SET status='answered', answered_at=?, answer=?, source=? WHERE question_id=?",
            (utc_now(), answer, source, question_id),
        )
        self.append_event("question_answered", {"question_id": question_id, "source": source, "answer": answer[:500]})
        self.consolidate(
            f"owner_dialogue:{question_id}",
            [{
                "kind": "user_dialogue",
                "content": f"Owner answered the question '{str(row['question']).strip()}' with: {answer.strip()} (source={source})",
                "confidence": 0.9,
            }],
        )
        return True

    def answer_for(self, question_id: str) -> dict[str, Any] | None:
        """The recorded answer, if the question has been answered."""
        row = self.connection.execute(
            "SELECT question_id, question, answer, answered_at, source FROM user_questions "
            "WHERE question_id=? AND status='answered'",
            (question_id,),
        ).fetchone()
        return dict(row) if row else None

    def open_questions(self, limit: int = 10, *, expire: bool = True) -> list[dict[str, Any]]:
        """Open questions, expiring the ones past their deadline.

        An expired question is not an error: the organism is expected to proceed
        on its own assumption, which is why the envelope says so explicitly.
        """
        now = utc_now()
        if expire:
            self.connection.execute(
                "UPDATE user_questions SET status='expired' WHERE status='open' AND expires_at <= ?", (now,)
            )
        rows = self.connection.execute(
            "SELECT question_id, question, options, status, created_at, expires_at, answered_at, answer "
            "FROM user_questions WHERE status='open' ORDER BY created_at LIMIT ?",
            (max(1, limit),),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["options"] = json.loads(item["options"])
            except (TypeError, json.JSONDecodeError):
                item["options"] = []
            result.append(item)
        return result

    def pending_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.alerts.pending(limit)

    def mark_alert_delivered(self, alert_id: str, channel: str) -> None:
        self.alerts.mark_delivered(alert_id, channel)

    def decay_memory_confidence(
        self, *, factor: float = 0.95, floor: float = 0.05, stale_days: float = 7.0, drop_below: float = 0.1
    ) -> dict[str, int]:
        """Fade memories nothing has reinforced, and drop the weakest.

        Retention is importance-weighted (Ebbinghaus-inspired): a stale memory
        keeps the fraction ``factor + (1 - factor) * confidence`` of its
        confidence, so a well-evidenced memory fades slower than a weak guess
        instead of every row losing the same share. Pinned memories never fade,
        the result is floored at `floor`, and unpinned memories below
        `drop_below` are removed.
        """
        now = utc_now()
        cutoff = (utc_datetime_now() - timedelta(days=max(0.0, stale_days))).isoformat().replace("+00:00", "Z")
        # Stamping `decayed_at` makes the fade idempotent: a second maintenance
        # pass inside the same stale window sees a fresh decay clock and does not
        # apply the retention factor again. `updated_at` is deliberately left
        # alone -- it is the column the recall 'recency' axis ranks on, so
        # writing it here promoted the very memories this pass punished.
        faded = self.connection.execute(
            "UPDATE memories SET confidence = MAX(?, confidence * (? + (1.0 - ?) * confidence)), decayed_at = ? "
            "WHERE pinned=0 AND updated_at < ? AND COALESCE(decayed_at, '') < ?",
            (floor, factor, factor, now, cutoff, cutoff),
        ).rowcount
        dropped_rows = self.connection.execute(
            "SELECT memory_id FROM memories WHERE pinned=0 AND confidence < ?", (drop_below,)
        ).fetchall()
        dropped = 0
        if dropped_rows:
            ids = [str(row["memory_id"]) for row in dropped_rows]
            placeholders = ",".join("?" for _ in ids)
            self.connection.execute(f"DELETE FROM memories WHERE memory_id IN ({placeholders})", ids)
            self.memory_store.drop_index(ids) if self.memory_store is not None else None
            dropped = len(ids)
        return {"faded": max(0, faded), "dropped": dropped}

    def reconcile_delivered_alerts(self) -> int:
        """Close pending alerts whose outbox message already left the queue.

        Two consumers used to race for the same lease: when the jsonl drain won,
        the alert reached a file but `mark_alert_delivered` was never called, so
        `alerts_pending` counted a delivered alert forever. This reconciles that
        residue without re-sending anything; a dead-lettered message is left
        pending on purpose, because it was never delivered.
        """
        rows = self.connection.execute(
            "SELECT a.alert_id FROM alerts a WHERE a.delivered_at IS NULL AND EXISTS ("
            "  SELECT 1 FROM outbox o WHERE o.kind='alert' AND o.delivery_state='delivered' "
            "  AND o.payload LIKE '%' || a.alert_id || '%')"
        ).fetchall()
        for row in rows:
            self.mark_alert_delivered(str(row["alert_id"]), "reconciled")
        return len(rows)

    def selected_task_streak(self, task_id: str, limit: int = 8) -> int:
        """Count consecutive planner decisions that selected the same task."""
        rows = self.connection.execute(
            "SELECT selected_task_id FROM planner_decisions ORDER BY created_at DESC, decision_id DESC LIMIT ?",
            (max(1, limit),),
        ).fetchall()
        streak = 0
        for row in rows:
            if row["selected_task_id"] != task_id:
                break
            streak += 1
        return streak

    def pinned_memories(self, limit: int = 32) -> list[dict[str, Any]]:
        """Always-injected memories, independent of any retrieval query."""
        rows = self.connection.execute(
            "SELECT memory_id, kind, content, confidence, source_run, updated_at FROM memories "
            "WHERE pinned=1 AND status='active' ORDER BY confidence DESC, updated_at DESC LIMIT ?",
            (max(1, limit),),
        ).fetchall()
        return [dict(row) for row in rows]

    def memory_provenance(self, memories: Iterable[Mapping[str, Any]]) -> dict[str, int]:
        """Split a memory set into provenance that resolves and provenance that does not.

        ``source_run`` is the only provenance a memory stores, and the retrieval
        path does not consult it: a row reaches the prompt because it matches the
        task query or because it is pinned. Counting the split turns
        retrieval-time provenance into a metric instead of an inference, so a
        future trust policy has an observable to move. A NULL or unknown
        ``source_run`` counts as unresolvable, which is the conservative reading:
        it is provenance that cannot be checked, not provenance that was checked
        and passed.
        """
        rows = [row for row in memories if isinstance(row, Mapping)]
        candidates = sorted({str(row.get("source_run")) for row in rows if row.get("source_run") is not None})
        known: set[str] = set()
        for index in range(0, len(candidates), PROVENANCE_LOOKUP_CHUNK):
            chunk = candidates[index : index + PROVENANCE_LOOKUP_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            known.update(
                str(row["run_id"])
                for row in self.connection.execute(
                    f"SELECT run_id FROM runs WHERE run_id IN ({placeholders})", chunk
                ).fetchall()
            )
        resolvable = sum(1 for row in rows if str(row.get("source_run")) in known)
        return {"total": len(rows), "resolvable": resolvable, "unresolvable": len(rows) - resolvable}

    def set_memory_pinned(self, memory_id: str, pinned: bool = True) -> bool:
        return bool(self.connection.execute(
            "UPDATE memories SET pinned=? WHERE memory_id=?", (1 if pinned else 0, memory_id)
        ).rowcount)

    def search_memories(self, query: str, limit: int = 20, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        if self.memory_store is None:
            return []
        return self.memory_store.search(query, limit, include_inactive=include_inactive)

    def close(self) -> None:
        self.connection.close()
