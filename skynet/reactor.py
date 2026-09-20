from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock, current_thread, main_thread
from typing import Any

from . import metrics
from .autonomous_planner import AutonomousPlanner
from .dialogue import AcknowledgeInboxTool, AskUserTool, SendMessageToUserTool
from .memory import MemoryLoop, MemoryResult
from .memory_tool import MemoryTool
from .model_contracts import SYSTEM, SYSTEM_MEMORY, SYSTEM_REACT
from .models import AgentState, Budget, LifecycleState, RunRecord, RunStatus, StartEnvelope
from .planner import PortfolioPlanner
from .provider import LLMProvider, Tool
from .react import ReActConfig, ReActRunner
from .rollback import RollbackRequestTool
from .self_improvement import PromoteSelfImprovementTool, SelfImprovementManager, SelfImprovementTool
from .store import StateStore
from .time import utc_datetime_now, utc_now

log = logging.getLogger("skynet.reactor")

SOUL_BLOCK_MARKER = "\n\n[SKYNET SOUL BEGIN]\n"

REACT_LOOP_ROLE_SUFFIX = "You are the bounded Agent Run. The harness owns lifecycle and durable state."

HEALTHY_LIFECYCLES = frozenset({
    LifecycleState.BOOT, LifecycleState.RECOVER, LifecycleState.PLAN, LifecycleState.SLEEP,
    LifecycleState.START, LifecycleState.REACT, LifecycleState.CONSOLIDATE, LifecycleState.CHECKPOINT,
})


class WatchdogTimeout(BaseException):
    """Raised in the main thread to interrupt a run that made no durable progress.

    Inherits from BaseException on purpose: it must fly through the ordinary
    ``except Exception`` blocks inside tools and providers and be handled only
    by the Reactor lifecycle boundary.
    """

BOOTSTRAP_GOAL_TITLE = "Improve SkyNet's ability to understand, validate, and improve itself."

BOOTSTRAP_TASK_TITLE = "Choose one concrete, non-repeated engineering bottleneck in SkyNet. Establish a measurable hypothesis, inspect only the relevant code and tests, and either validate a small improvement or record a justified blocker. Do not repeat unchanged administrative bootstrap checks, host reconnaissance, or a previous proposal."
FALLBACK_TASK_TITLE = "Diagnose one concrete SkyNet bottleneck with a focused invariant or regression check, and record one verified fact or justified blocker."
EXTERNAL_SEEK_TASK_TITLE = (
    "Consult one external source (arXiv, GitHub, or the open web) about a concrete SkyNet bottleneck, "
    "distil it into one bounded, testable improvement hypothesis, and record the citation."
)

# An owner message is a notification the organism may act on, never an order and
# never an auto-created task: see `_inbox_notifications` and `acknowledge_inbox`.

@dataclass(slots=True)
class ReactorConfig:
    state_path: Path | str = Path("state/skynet.sqlite3")
    system_prompt: str = SYSTEM
    budget: Budget = field(default_factory=Budget)
    memory_input_tokens: int = 50_000
    wake_interval_seconds: float = 60.0
    self_improvement_root: Path | str | None = None
    watchdog_timeout_seconds: float = 1800.0
    transcript_retention_runs: int = 200
    task_giveup_failures: int = 3
    restart_failure_limit: int = 3
    reboot_window_max_age_seconds: float = 86_400.0
    # A provider lockout is an external condition that used to burn cycles
    # silently: after this many consecutive all-provider failures the organism
    # escalates once and backs off for a long interval instead of retrying.
    provider_lockout_threshold: int = 3
    provider_lockout_seconds: float = 1800.0
    livelock_streak_limit: int = 3
    pinned_memory_limit: int = 32
    memory_inject_limit: int = 3
    metrics_snapshot_enabled: bool = True
    # Exploration is required to evaluate the ranking at all: without it the
    # planner always takes the top candidate and a bad score function is
    # indistinguishable from a good one.
    planner_epsilon: float = 0.1
    hypothesis_ttl_days: float = 30.0
    # Quality-diversity: an empty descriptor cell outranks a crowded one. Small
    # on purpose, so it can never dominate a genuinely more critical task.
    cell_scarcity_weight: float = 0.20
    # Opening the boundary: when internal work is exhausted the organism looks
    # outward instead of parking in sleep. A closed system either cycles or
    # thermalizes; a flux across the boundary is what keeps it far from
    # equilibrium. These knobs own that flux.
    external_seek_enabled: bool = True
    external_seek_every_generations: int = 4
    external_seek_cooldown_seconds: float = 3600.0
    # A criterion epoch fixes what "good" means for a span of generations. The
    # boundary is only recorded for now: it gives a later, non-stationary
    # utility an anchor without changing the criterion itself.
    criterion_epoch_generations: int = 100
    archive_parent_k: int = 2
    archive_parent_lambda: float = 10.0
    archive_parent_alpha0: float = 0.5
    max_active_goals: int = 8
    # Retention and GC run once per UTC day, next to the metrics snapshot.
    event_retention_days: int = 30


class Reactor:
    """Single-writer lifecycle owner. It is deliberately not started on import."""

    def __init__(self, provider: LLMProvider, tools: Mapping[str, Tool], config: ReactorConfig | None = None) -> None:
        self.config = config or ReactorConfig()
        self.store = StateStore(self.config.state_path)
        self.provider = provider
        set_event_logger = getattr(provider, "set_event_logger", None)
        if set_event_logger is not None:
            set_event_logger(lambda kind, payload: self.store.runtime_log.write(kind, payload))
        runtime_tools = dict(tools)
        improvement_root = self.config.self_improvement_root or Path.cwd()
        if (Path(improvement_root) / ".git").exists():
            improvement_manager = SelfImprovementManager(improvement_root)
            runtime_tools["propose_self_improvement"] = SelfImprovementTool(improvement_manager)
            runtime_tools["promote_self_improvement"] = PromoteSelfImprovementTool(improvement_manager, self._self_improvement_health)
            runtime_tools["request_rollback"] = RollbackRequestTool(improvement_root)
        # The owner dialogue is always available: it is how the organism asks a
        # question it cannot answer and how it reports without waiting.
        runtime_tools["ask_user"] = AskUserTool(self.store)
        runtime_tools["send_message_to_user"] = SendMessageToUserTool(self.store)
        runtime_tools["acknowledge_inbox"] = AcknowledgeInboxTool(self.store)
        runtime_tools["memory"] = MemoryTool(self.store)
        self.runner = ReActRunner(
            provider,
            self.store,
            runtime_tools,
            ReActConfig(
                max_steps=self.config.budget.steps,
                max_tokens=self.config.budget.tokens,
                output_tokens=self.config.budget.output_tokens,
                timeout_seconds=self.config.budget.seconds,
                # Scale the finish reserve and the per-tool-result cap with the
                # widened 500k window; both are operator-tunable.
                context_finish_reserve=int(os.getenv("SKYNET_CONTEXT_FINISH_RESERVE", "50000")),
                tool_result_max_chars=int(os.getenv("SKYNET_TOOL_RESULT_MAX_CHARS", "16000")),
            ),
            worktree_root=improvement_root,
        )
        self.memory_loop = MemoryLoop(provider, budget=Budget(steps=1, tokens=self.config.memory_input_tokens, seconds=180.0, output_tokens=self.config.budget.output_tokens), timeout_seconds=180.0)
        self.autonomous_planner = AutonomousPlanner(
            provider,
            self.store,
            output_tokens=min(4096, self.config.budget.output_tokens),
            timeout_seconds=self.config.budget.seconds,
            max_active_goals=self.config.max_active_goals,
            hypothesis_ttl_days=self.config.hypothesis_ttl_days,
        )
        self._self_improvement_restart: Any = None
        self._run_lock = Lock()
        soul_path = Path(__file__).resolve().parent.parent / "SOUL.md"
        if soul_path.exists() and SOUL_BLOCK_MARKER not in self.config.system_prompt:
            self.config.system_prompt += SOUL_BLOCK_MARKER + soul_path.read_text(encoding="utf-8") + "\n[SKYNET SOUL END]"

    def tick(self, wake_cause: str = "timer") -> RunStatus | None:
        if not self._run_lock.acquire(blocking=False):
            return None
        try:
            state = self.store.state()
            if state.active_run_id is not None:
                return RunStatus.RUNNING
            goals, tasks = self.store.active_work()
            # Genesis is for a genuinely empty database only. Do not resurrect
            # a blocked/cancelled goal after an intentional memory or goal reset.
            has_any_goals = bool(self.store.connection.execute("SELECT 1 FROM goals LIMIT 1").fetchone())
            if not goals and not tasks and not has_any_goals:
                with self.store.transaction():
                    bootstrap_id = self.store.add_goal(
                        BOOTSTRAP_GOAL_TITLE,
                        priority=1.0,
                    )
                    self.store.add_task(BOOTSTRAP_TASK_TITLE, bootstrap_id, area="bootstrap")
                goals, tasks = self.store.active_work()
            inbox_events = self.store.pending_inbox()
            # An empty active-goal set is no longer a reason to sleep: the
            # operator inbox and the autonomous planner still hold wake
            # authority. The gap is recorded once per generation instead.
            if not goals:
                self._record_no_active_goal(state, inbox_events)
            # Selectability, not mere task existence, determines whether the
            # LLM planner must be called. This prevents exhausted pending rows
            # from blocking autonomous replacement planning.
            selected_work = self._select_work()
            # An owner message is a notification, not work: it never creates a
            # task and never preempts. When nothing is selectable the fixed pool
            # is the fallback, then the autonomous planner, then the bounded
            # bootstrap task.
            if selected_work is None:
                generated_tasks = self._run_autonomous_planning(state, goals, wake_cause)
                goals, tasks = self.store.active_work()
                selected_work = self.store.task_work(generated_tasks[0]) if generated_tasks else self._select_work()
                if selected_work is None:
                    fallback_id = self._create_planner_fallback(goals, state.generation)
                    if fallback_id:
                        selected_work = self.store.task_work(fallback_id)
                # Last resort before sleeping: re-seed the curated roadmap so an
                # organism whose last goal closed still has bootstrap work.
                if selected_work is None and not goals and self._ensure_roadmap_seed():
                    goals, _ = self.store.active_work()
                    selected_work = self._select_work()
            # The boundary valve is cadence-driven, not exhaustion-driven: the
            # bounded fallback above always produces something, so waiting for an
            # empty portfolio meant external_seek could never fire.
            # Genesis is never preempted: the first cycle bootstraps, and
            # generation zero satisfies every modulus.
            if state.generation > 0 and self._external_seek_due(state) and self._external_senses():
                external_work = self._seek_external_evidence(state, goals, reason="periodic outward look")
                if external_work is not None:
                    selected_work = external_work
            if selected_work is None:
                # Internal exhaustion is a signal to look outward, not to stop.
                selected_work = self._seek_external_evidence(state, goals, reason="internal exhaustion")
            if selected_work is None:
                return self._sleep_without_work(state, "autonomous planner and bounded fallback produced no ready task")
            memory_query = self._memory_query(selected_work, inbox_events, state.next_plan)
            memory_context = self.store.search_memories(memory_query, limit=self.config.memory_inject_limit) if memory_query else []
            pinned_memories = self.store.pinned_memories(limit=self.config.pinned_memory_limit)
            start = StartEnvelope(
                wake_cause=wake_cause,
                observations=[{"kind": "durable_state", "generation": state.generation}],
                active_goals=goals,
                pending_work=[selected_work],
                previous_outcome=state.next_plan.get("previous_outcome", {}) if isinstance(state.next_plan, dict) else {},
                next_plan=state.next_plan,
                intention=self._intention(selected_work, state.next_plan),
                success_criteria=self._success_criteria(selected_work),
                budget=self.config.budget,
            )
            start.observations.extend(self._inbox_notifications(inbox_events))
            # Retrieval used to be silently empty (every term was AND-joined), so
            # the hit count is now recorded on every cycle: a regression in recall
            # must be visible in the metrics instead of inferred.
            self.store.append_event(
                "memory_retrieval",
                {
                    "query_chars": len(memory_query),
                    "query_terms": len(memory_query.split()),
                    "hits": len(memory_context),
                    "pinned": len(pinned_memories),
                    "top_kinds": [item.get("kind") for item in memory_context[:3]],
                },
                start.run_id,
            )
            start.observations.append({"kind": "runtime_facts", **self._runtime_facts()})
            worktree_state = self._worktree_state()
            if worktree_state.get("dirty"):
                start.observations.append({"kind": "dirty_worktree", **worktree_state})
            reboot_outcome = self._reboot_outcome()
            if reboot_outcome:
                start.observations.append({"kind": "reboot_outcome", **reboot_outcome})
            open_questions = self.store.open_questions(limit=10)
            if open_questions:
                # An unanswered question is not a blocker: the organism proceeds
                # on its own assumption, but it must know what it asked.
                start.observations.append({
                    "kind": "user_questions",
                    "items": open_questions,
                    "note": "answer may never arrive; proceed on an explicit assumption and record it",
                })
            if pinned_memories:
                start.observations.append({"kind": "pinned_memories", "items": pinned_memories})
            if memory_context:
                start.observations.append({"kind": "memory_context", "query": memory_query, "items": memory_context})
            run = RunRecord(start.run_id, 1, RunStatus.RUNNING, utc_now(), self.config.budget)
            deferred_restart: dict[str, Any] = {}
            with self.store.transaction():
                self.store.transition(state, LifecycleState.START, run_id=start.run_id, reason="run selected")
                state.active_run_id = start.run_id
                self.store.set_state(state)
                self.store.create_run(run)
                selected_attempt_task = selected_work.get("task") if selected_work.get("kind") == "task" else None
                if selected_attempt_task and selected_attempt_task.get("task_id"):
                    self.store.increment_task_attempts(str(selected_attempt_task["task_id"]))
                self.store.append_event("run_started", start.as_dict(), start.run_id)
                self.store.append_event(
                    "decision_record",
                    {"selected_work": selected_work, "intention": start.intention, "success_criteria": start.success_criteria, "wake_cause": wake_cause},
                    start.run_id,
                )
                self.store.transition(state, LifecycleState.REACT, run_id=start.run_id, reason="react started")

            result = self.runner.run(start, self.config.system_prompt + "\n\n" + SYSTEM_REACT + "\n" + REACT_LOOP_ROLE_SUFFIX)
            with self._defer_watchdog_signal(), self.store.transaction():
                self.store.commit_run_result(
                    start.run_id,
                    result.status,
                    result.report,
                    result.steps,
                    result.usage_tokens,
                    result.failure,
                )
                self.store.append_event(
                    "run_result_committed",
                    {
                        "status": result.status.value,
                        "steps": result.steps,
                        "usage_tokens": result.usage_tokens,
                        "failure": result.failure,
                    },
                    start.run_id,
                )
                self._record_provider_request_telemetry(start.run_id)
            memory_result = MemoryResult(memory_candidates=[], next_plan={})
            snapshot = self.store.snapshot_episode(start.run_id)
            episode = snapshot["events"]
            self.store.append_event("memory_loop_started", {"episode_events": len(episode)}, start.run_id)
            memory_succeeded = False
            if result.status in {RunStatus.COMPLETED, RunStatus.BLOCKED, RunStatus.FAILED, RunStatus.NEEDS_RECOVERY}:
                try:
                    memory_result = self.memory_loop.consolidate(
                        episode,
                        result,
                        self.config.system_prompt + "\n\n" + SYSTEM_MEMORY,
                        runtime_log=lambda kind, payload, **_: self.store.runtime_log.write(
                            kind, payload, run_id=start.run_id
                        ),
                        chat_history=self.store.react_history_for_memory(start.run_id),
                        short_memory=memory_context,
                    )
                    memory_succeeded = True
                except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    self.store.append_event("memory_loop_failed", {"error": str(exc)[:1000]}, start.run_id)
            with self._defer_watchdog_signal(), self.store.transaction():
                    state = self.store.state()
                    self.store.transition(state, LifecycleState.CONSOLIDATE, run_id=start.run_id, reason="react finished")
                    self.store.append_event(
                        "scheduler_decision",
                        {"selected_work": selected_work, "has_inbox": bool(inbox_events)},
                        start.run_id,
                    )
                    if memory_succeeded:
                        self.store.append_event("memory_loop_finished", {"memory_candidates": len(memory_result.memory_candidates)}, start.run_id)
                    consolidation = self.store.consolidate_versioned(snapshot["snapshot_id"], start.run_id, memory_result.memory_candidates)
                    memory_count = consolidation["memory_count"]
                    self.store.append_event("memory_consolidated", consolidation, start.run_id)
                    criteria_results = self.store.evaluate_criteria(start.run_id, start.success_criteria, result.report, result.status)
                    required_results = [item for item in criteria_results if item.get("required", True)]
                    effective_status = result.status
                    if result.status == RunStatus.COMPLETED and required_results and not all(item.get("passed") is True for item in required_results):
                        effective_status = RunStatus.BLOCKED
                        self.store.update_run_result_status(start.run_id, effective_status, "required success criteria failed")
                        self.store.append_event("success_criteria_failed", {"criteria": criteria_results}, start.run_id)
                    self.store.append_event(
                        "run_finished",
                        {
                            "status": effective_status.value,
                            "report": result.report,
                            "steps": result.steps,
                            "usage_tokens": result.usage_tokens,
                            "failure": result.failure,
                        },
                        start.run_id,
                    )
                    retryable_blocker = effective_status == RunStatus.BLOCKED and self._is_retryable_environment_blocker(result.report)
                    provider_failed = self._run_had_provider_failure(start.run_id)
                    selected_task = selected_work.get("task") if selected_work and selected_work.get("kind") == "task" else None
                    gave_up = False
                    if selected_task:
                        task_status = {
                            RunStatus.COMPLETED: "completed",
                            RunStatus.BLOCKED: "pending" if retryable_blocker else "blocked",
                            RunStatus.FAILED: "blocked",
                            RunStatus.NEEDS_RECOVERY: "pending",
                            RunStatus.INTERRUPTED: "pending",
                        }.get(effective_status, "pending")
                        if task_status == "completed" and self._claims_uncaptured_changes(start.run_id, result.report):
                            # A run can report file changes it prototyped outside the
                            # proposal workflow. Keep the task in the queue instead of
                            # treating the discarded prototype as completed work.
                            self.store.append_event(
                                "uncaptured_changes",
                                {"task_id": selected_task["task_id"], "report": result.report[:1000]},
                                start.run_id,
                            )
                            task_status = "pending"
                        task_id = str(selected_task["task_id"])
                        if effective_status == RunStatus.COMPLETED:
                            self.store.reset_task_failures(task_id)
                        elif effective_status in {RunStatus.NEEDS_RECOVERY, RunStatus.BLOCKED} and not provider_failed:
                            # A provider outage is transient; a model-side or
                            # retryable-environment failure repeated on the same
                            # bounded task is not. Retryable BLOCKED used to be
                            # returned to pending with no cap, so a single task
                            # with a permanent environment blocker was selected
                            # forever while it was the only candidate. After the
                            # give-up limit the task is retired so replacement
                            # planning can run.
                            failures = self.store.record_task_failure(task_id)
                            if failures >= self.config.task_giveup_failures:
                                task_status = "blocked"
                                gave_up = True
                                self.store.append_event(
                                    "task_gave_up",
                                    {"task_id": task_id, "consecutive_failures": failures, "failure": result.failure[:500], "report": result.report[:500]},
                                    start.run_id,
                                )
                        self.store.apply_task_updates(
                            [{"task_id": selected_task["task_id"], "status": task_status}],
                            run_id=start.run_id,
                        )
                    if effective_status == RunStatus.BLOCKED:
                        blocked_task = selected_task
                        self.store.append_event(
                            "planner_task_retryable" if (retryable_blocker and not gave_up) else "planner_task_exhausted",
                            {
                                "task_id": blocked_task.get("task_id") if blocked_task else None,
                                "reason": "environment blocker; retry the same bounded task after the next wake" if (retryable_blocker and not gave_up) else "run did not complete; autonomous replacement planning follows",
                                "status": effective_status.value,
                            },
                            start.run_id,
                        )
                    if selected_task and selected_task.get("hypothesis_fingerprint"):
                        if effective_status == RunStatus.COMPLETED:
                            hypothesis_status = "completed"
                        elif retryable_blocker and not gave_up:
                            hypothesis_status = "ready"
                        elif effective_status == RunStatus.BLOCKED or gave_up:
                            hypothesis_status = "exhausted"
                        else:
                            hypothesis_status = "ready"
                        self.store.mark_hypothesis(
                            str(selected_task["hypothesis_fingerprint"]),
                            hypothesis_status,
                            {"run_id": start.run_id, "status": effective_status.value, "report": result.report[:1000]},
                        )
                    self._apply_memory_updates(memory_result, selected_work, start.run_id, effective_status)
                    value_estimate = (
                        sum(1.0 for item in required_results if item.get("passed") is True) / len(required_results)
                        if effective_status == RunStatus.COMPLETED and required_results
                        else (1.0 if effective_status == RunStatus.COMPLETED else 0.0)
                    )
                    evaluation = {
                        "success_criteria_results": criteria_results,
                        "value_estimate": value_estimate,
                        "source": "harness_technical_status",
                    }
                    evaluation["memory_loop_evaluation"] = memory_result.evaluation
                    self.store.record_evaluation(start.run_id, evaluation, effective_status, result.report)
                    self._record_idea_outcome(selected_work, value_estimate)
                    self._archive_run_self_improvements(start.run_id, selected_work, value_estimate)
                    self.store.add_outbox(
                        "agent_response",
                        {
                            "run_id": start.run_id,
                            "status": effective_status.value,
                            "report": result.report,
                        },
                    )
                    state.next_plan = {
                        "initial_prompt": result.report if effective_status == RunStatus.BLOCKED else (memory_result.initial_prompt or result.report),
                        "previous_outcome": {"report": result.report, "status": effective_status.value},
                        "planner_hints": memory_result.next_plan if effective_status != RunStatus.BLOCKED else {},
                    }
                    if effective_status == RunStatus.BLOCKED and not retryable_blocker:
                        state.next_plan.update({
                            "blocked_run": True,
                            "next": "Work on the replacement bounded task; do not revisit the blocked task unless its state changes.",
                            "avoid": ["Do not repeat the blocked task or stale blocker report."],
                        })
                    if result.control_action.get("type") == "restart_after_checkpoint":
                        deferred_restart = dict(result.control_action)
                        state.next_plan["self_improvement_recovery"] = {
                            "proposal_id": deferred_restart.get("proposal_id"),
                            "commit": deferred_restart.get("commit"),
                            "next": "verify the promoted self-improvement after reboot",
                            "checks": ["service starts", "database opens", "provider is reachable", "health window completes"],
                        }
                    state.retry_count = 0 if effective_status.value == "completed" else state.retry_count + 1
                    delay = self.config.wake_interval_seconds if effective_status.value == "completed" else min(300, 2 ** min(state.retry_count, 8))
                    provider_streak = self._consecutive_provider_failures()
                    if provider_streak >= self.config.provider_lockout_threshold:
                        # Escalate once and stop burning cycles against a chain
                        # that has nothing available.
                        self.store.append_event(
                            "provider_lockout",
                            {"consecutive_failures": provider_streak, "backoff_seconds": self.config.provider_lockout_seconds},
                            start.run_id,
                        )
                        self.store.raise_alert(
                            "provider_lockout",
                            {
                                "consecutive_failures": provider_streak,
                                "last_failure": str(result.failure)[:300],
                                "providers": self._provider_status_summary(),
                            },
                            severity="critical",
                            dedup_key="provider_lockout",
                            run_id=start.run_id,
                        )
                        delay = max(delay, self.config.provider_lockout_seconds)
                    cosmetic = self._cosmetic_streak()
                    if cosmetic["streak"] >= 3:
                        self.store.append_event("rust_polishing_suspected", cosmetic, start.run_id)
                        self.store.raise_alert(
                            "rust_polishing_suspected",
                            {**cosmetic, "note": "the last proposals were cosmetic; prefer a roadmap item with a measurable outcome"},
                            severity="warning",
                            dedup_key="rust_polishing_suspected",
                            run_id=start.run_id,
                        )
                    if selected_task and selected_task.get("task_id") and not provider_failed:
                        # A provider outage makes the planner re-select the same
                        # pending task; that is an availability problem with its
                        # own escalation, not evidence of a planning loop.
                        streak = self.store.selected_task_streak(
                            str(selected_task["task_id"]), limit=self.config.livelock_streak_limit + 2
                        )
                        if streak >= self.config.livelock_streak_limit:
                            self.store.append_event(
                                "livelock_suspected",
                                {"task_id": str(selected_task["task_id"]), "streak": streak},
                                start.run_id,
                            )
                            self.store.raise_alert(
                                "livelock_suspected",
                                {
                                    "task_id": str(selected_task["task_id"]),
                                    "streak": streak,
                                    "title": str(selected_task.get("title", ""))[:200],
                                },
                                severity="warning",
                                dedup_key=f"livelock:{selected_task['task_id']}",
                                run_id=start.run_id,
                            )
                    state.next_wake_at = (utc_datetime_now() + timedelta(seconds=delay)).isoformat().replace("+00:00", "Z")
                    state.generation += 1
                    epoch = max(1, self.config.criterion_epoch_generations)
                    if state.generation % epoch == 0:
                        self.store.append_event(
                            "criterion_epoch_started",
                            {"epoch": state.generation // epoch, "generation": state.generation, "span_generations": epoch},
                            start.run_id,
                        )
                    state.active_run_id = None
                    self.store.transition(state, LifecycleState.CHECKPOINT, run_id=start.run_id, reason="checkpoint started")
                    self.store.finish_run(start.run_id, effective_status)
                    self.store.set_state(state)
                    pruned = self.store.prune_run_history(self.config.transcript_retention_runs)
                    self.store.append_event(
                        "checkpoint",
                        {"generation": state.generation, "memory_count": memory_count},
                        start.run_id,
                    )
                    if pruned["transcript"] or pruned["episodes"]:
                        self.store.append_event("history_pruned", {**pruned, "keep_runs": self.config.transcript_retention_runs}, start.run_id)
                    self.store.transition(state, LifecycleState.SLEEP, run_id=start.run_id, reason="checkpoint complete")
            if self.config.metrics_snapshot_enabled:
                self._record_daily_metrics()
            # Retention and decay are maintenance, not metrics: gating them on
            # the snapshot flag silently disabled them when snapshots were off.
            self._maybe_run_daily_maintenance()
            if deferred_restart:
                self._request_self_improvement_restart(deferred_restart)
            return result.status
        except WatchdogTimeout:
            interrupted = self.interrupt_stale_run(reason="watchdog_timeout", force=True)
            return interrupted if interrupted is not None else RunStatus.INTERRUPTED
        except Exception as exc:
            # The recovery write must not be torn by the very signal that caused
            # the failure, so it runs with the interrupt signals deferred.
            with self._defer_watchdog_signal(), self.store.transaction():
                state = self.store.state()
                failed_run_id = state.active_run_id
                if failed_run_id is not None:
                    self.store.finish_run(failed_run_id, RunStatus.INTERRUPTED)
                    self.store.append_event(
                        "run_interrupted",
                        {"reason": "reactor_exception", "error": str(exc)[:1000]},
                        failed_run_id,
                    )
                self.store.transition(state, LifecycleState.FAILED, run_id=failed_run_id, reason="reactor exception")
                state.retry_count += 1
                state.active_run_id = None
                state.next_plan = {
                    **state.next_plan,
                    "recovery": {"error": str(exc)[:1000], "retry_count": state.retry_count},
                }
                state.next_wake_at = (utc_datetime_now() + timedelta(seconds=min(300, 2 ** min(state.retry_count, 8)))).isoformat().replace("+00:00", "Z")
                self.store.set_state(state)
                self.store.append_event("run_failure", {"error": str(exc)[:1000], "retry_count": state.retry_count}, failed_run_id)
            raise
        finally:
            self._run_lock.release()

    def _self_improvement_health(self) -> dict[str, object]:
        state = self.store.state()
        return {
            "ok": state.lifecycle in HEALTHY_LIFECYCLES,
            "lifecycle": state.lifecycle.value,
        }

    def _apply_memory_updates(self, memory_result: MemoryResult, selected_work: dict[str, Any], run_id: str, result_status: RunStatus) -> None:
        task = selected_work.get("task") if selected_work.get("kind") == "task" else None
        goal = selected_work.get("goal") if selected_work.get("kind") == "goal" else None
        task_id = task.get("task_id") if isinstance(task, dict) else None
        goal_id = task.get("goal_id") if isinstance(task, dict) else (goal.get("goal_id") if isinstance(goal, dict) else None)
        safe_goal_updates = []
        for update in memory_result.goal_updates:
            if not isinstance(update, dict) or update.get("goal_id") != goal_id:
                continue
            if update.get("status") == "completed" and result_status != RunStatus.COMPLETED:
                continue
            safe_goal_updates.append(update)
        safe_task_updates = []
        for update in memory_result.task_updates:
            if not isinstance(update, dict) or update.get("task_id") != task_id:
                continue
            if update.get("status") == "completed" and result_status != RunStatus.COMPLETED:
                continue
            safe_task_updates.append(update)
        self.store.apply_goal_updates(safe_goal_updates, run_id=run_id)
        self.store.apply_task_updates(safe_task_updates, run_id=run_id)

    def _external_senses(self) -> list[str]:
        """Which configured senses are actually alive right now.

        The reactor never holds the MCP client list (it lives in cli.main), but
        every MCP tool carries its client, so liveness is observable from here.
        Guessing would let the organism schedule research it cannot perform.
        """
        senses: set[str] = set()
        for tool in self.runner.tools.values():
            client = getattr(tool, "client", None)
            alive = getattr(client, "alive", None)
            if callable(alive):
                try:
                    if alive():
                        senses.add(str(getattr(client, "server_name", "") or "mcp"))
                except Exception:
                    log.debug("MCP sense liveness check failed", exc_info=True)
                    continue
        if "webfetch" in self.runner.tools:
            senses.add("webfetch")
        return sorted(senses)

    def _external_seek_due(self, state: AgentState) -> bool:
        """Cadence plus a durable cooldown: no in-memory counter to lose on restart."""
        if state.generation % max(1, self.config.external_seek_every_generations) != 0:
            return False
        row = self.store.connection.execute(
            "SELECT created_at FROM event_log WHERE kind='external_seek_created' ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return True
        try:
            last = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return True
        return (utc_datetime_now() - last).total_seconds() >= self.config.external_seek_cooldown_seconds

    def _seek_external_evidence(self, state: AgentState, goals: list[dict[str, Any]], *, reason: str = "internal exhaustion") -> dict[str, Any] | None:
        """The harness opens the boundary on a cadence or on internal exhaustion.

        Without this valve the hypothesis space stays closed: the bounded
        fallback always produces something, so an exhaustion-only trigger never
        fired and the organism never consulted an external source on its own.
        """
        if not self.config.external_seek_enabled:
            return None
        senses = self._external_senses()
        if not senses:
            self.store.append_event("external_senses_unavailable", {
                "generation": state.generation,
                "reason": "no MCP sense is alive and webfetch is absent",
            })
            return None
        if not self._external_seek_due(state):
            return None
        goal = goals[0] if goals else None
        if goal is None:
            return None
        title = f"{EXTERNAL_SEEK_TASK_TITLE} [generation {state.generation}]"
        expected = (
            "One memory entry cites a specific external source (arXiv id, repository URL, or page URL) "
            "and states a bounded, testable improvement hypothesis derived from it; or the search is "
            "recorded as yielding nothing applicable, with the queries used."
        )
        with self.store.transaction():
            task_id = self.store.add_task(
                title,
                str(goal["goal_id"]),
                expected_new_fact=expected,
                area="research",
            )
            self.store.append_event("external_seek_created", {
                "task_id": task_id,
                "goal_id": str(goal["goal_id"]),
                "generation": state.generation,
                "senses": senses,
                "reason": reason,
            })
        return self.store.task_work(task_id)

    def _archive_proposal(self, proposal: dict[str, Any]) -> None:
        """Every accepted proposal becomes a stepping stone with a descriptor."""
        from .idea_archive import cell_key, classify
        try:
            subsystem, change_type, source = classify(
                scope=proposal.get("scope") or [],
                kind=str(proposal.get("kind", "")),
                validation=str(proposal.get("validation", "")),
                inspiration_ref=str(proposal.get("inspiration_ref", "") or ""),
            )
            self.store.archive_idea({
                "title": proposal["title"],
                "problem_description": proposal.get("problem", ""),
                "hypothesis": proposal.get("hypothesis", ""),
                "expected_new_fact": proposal.get("expected_new_fact", ""),
                "validation": proposal.get("validation", ""),
                "inspiration_ref": proposal.get("inspiration_ref", ""),
                "subsystem": subsystem,
                "change_type": change_type,
                "evidence_source": source,
                "cell_key": cell_key(subsystem, change_type, source),
                "quality": float(proposal.get("quality", 0.5) or 0.5),
                "novelty": 1.0,
                "task_id": proposal.get("task_id"),
                "proposal_id": proposal.get("proposal_id"),
            })
        except Exception as exc:
            self.store.append_event("idea_archive_rejected", {
                "reason": f"archive failed: {exc}",
                "title": str(proposal.get("title", ""))[:200],
            })

    def _archive_run_self_improvements(self, run_id: str, selected_work: dict[str, Any] | None, value_estimate: float) -> None:
        """Archive proposals that went through the tool, not only planner ideas.

        The archive was fed exclusively by `_run_autonomous_planning`, so the
        organism's real self-improvement traffic never became a stepping stone:
        promoted self-improvement work left `idea_archive` empty. Quality is the
        run's measured value, so a promoted change outranks an untested one.
        """
        task_id = None
        if isinstance(selected_work, dict):
            task = selected_work.get("task")
            if isinstance(task, dict):
                task_id = task.get("task_id")
        quality = max(0.5, min(float(value_estimate), 1.0))
        for event in self.store.recent_run_events(run_id, limit=None):
            if event.get("kind") != "tool_result":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            call = payload.get("call")
            result = payload.get("result")
            if not isinstance(call, dict) or call.get("tool_name") != "propose_self_improvement":
                continue
            if not isinstance(result, dict) or result.get("ok") is not True:
                continue
            raw_arguments = call.get("arguments")
            arguments: dict[str, Any] = raw_arguments if isinstance(raw_arguments, dict) else {}
            raw_hypothesis = arguments.get("hypothesis")
            hypothesis: dict[str, Any] = raw_hypothesis if isinstance(raw_hypothesis, dict) else {}
            problem = str(hypothesis.get("problem") or "").strip()
            self._archive_proposal({
                "title": problem[:300] or "self-improvement proposal",
                "problem": problem,
                "hypothesis": str(hypothesis.get("expected_behavior") or ""),
                "expected_new_fact": str(hypothesis.get("expected_behavior") or ""),
                "validation": str(hypothesis.get("validation") or ""),
                "inspiration_ref": str(arguments.get("inspiration_ref") or ""),
                "scope": self._proposal_scope(arguments),
                "kind": "improvement",
                "quality": quality,
                "task_id": task_id,
                "proposal_id": result.get("proposal_id"),
            })

    @staticmethod
    def _proposal_scope(arguments: dict[str, Any]) -> list[str]:
        """The paths a proposal touches, from either edit form."""
        scope: list[str] = []
        changes = arguments.get("changes")
        if isinstance(changes, list):
            scope.extend(
                change["path"]
                for change in changes
                if isinstance(change, dict) and isinstance(change.get("path"), str)
            )
        files = arguments.get("files")
        if isinstance(files, dict):
            scope.extend(str(path) for path in files)
        return scope

    def _record_idea_outcome(self, selected_work: dict[str, Any] | None, value_estimate: float) -> None:
        """Close the loop: the archived idea behind a task learns how it went."""
        task_id = None
        if isinstance(selected_work, dict):
            task = selected_work.get("task")
            if isinstance(task, dict):
                task_id = task.get("task_id")
        if not task_id:
            return
        row = self.store.connection.execute(
            "SELECT idea_id FROM idea_archive WHERE task_id=? AND status IN ('active','materialized') LIMIT 1",
            (str(task_id),),
        ).fetchone()
        if row is None:
            return
        self.store.record_idea_outcome(
            str(row["idea_id"]),
            quality=max(0.0, min(float(value_estimate), 1.0)),
            status="validated" if float(value_estimate) >= 1.0 else "active",
        )

    def _sleep_without_work(self, state: AgentState, reason: str) -> None:
        with self.store.transaction():
            state = self.store.state()
            if state.lifecycle != LifecycleState.SLEEP:
                self.store.transition(state, LifecycleState.SLEEP, reason=reason)
            state.active_run_id = None
            state.retry_count += 1
            delay = min(600.0, max(self.config.wake_interval_seconds, 2 ** min(state.retry_count, 9)))
            state.next_wake_at = (utc_datetime_now() + timedelta(seconds=delay)).isoformat().replace("+00:00", "Z")
            state.next_plan = {"next": "run autonomous planning on next wake", "reason": reason}
            self.store.set_state(state)
            self.store.append_event("planner_backoff", {"reason": reason, "retry_count": state.retry_count, "delay_seconds": delay})

    def _record_no_active_goal(self, state: AgentState, inbox_events: list[dict[str, Any]]) -> None:
        """Record an active-goal gap once per generation, never every tick.

        The gap used to park the organism in sleep before the inbox or the
        autonomous planner could run. It is now only a durable observation; the
        generation check keeps a persistent gap from spamming the event log.
        """
        previous = self.store.connection.execute(
            "SELECT payload FROM event_log WHERE kind='no_active_goal' ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if previous is not None:
            try:
                payload = json.loads(previous[0])
            except (TypeError, json.JSONDecodeError):
                payload = {}
            if payload.get("generation") == state.generation:
                return
        self.store.append_event(
            "no_active_goal",
            {
                "generation": state.generation,
                "pending_inbox": len(inbox_events),
                "note": "no active goal; the inbox and autonomous planner still have wake authority",
            },
        )

    def _record_provider_request_telemetry(self, run_id: str) -> None:
        """Re-emit the assembled per-request identity as a durable event.

        The ReAct runner already computes ``tool_count`` and ``request_chars``
        for every provider request and stores them in the transcript; copying
        the last one into the event log lets the daily metrics snapshot weigh
        the MCP tool surface against the request size without re-serializing
        any payload.
        """
        meta: dict[str, Any] = {}
        row = self.store.connection.execute(
            "SELECT payload FROM transcript WHERE run_id=? AND kind='provider_request' ORDER BY sequence DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row is not None:
            try:
                parsed = json.loads(row[0])
            except (TypeError, json.JSONDecodeError):
                parsed = {}
            if isinstance(parsed, dict):
                meta = parsed
        tool_count = meta.get("tool_count")
        if not isinstance(tool_count, int):
            tool_count = len(self.runner.tools)
        request_chars = meta.get("request_chars")
        self.store.append_event(
            "provider_request",
            {
                "tool_count": tool_count,
                "request_chars": request_chars if isinstance(request_chars, int) else None,
                "model": str(meta.get("model") or self._provider_model()),
                "step": meta.get("step"),
            },
            run_id,
        )

    def _provider_model(self) -> str:
        """Best-effort model identity without a network probe.

        A single provider exposes ``model`` directly; a fallback chain is
        represented by its first configured model so the event stays a scalar.
        """
        model = getattr(self.provider, "model", None)
        if isinstance(model, str) and model:
            return model
        chain = getattr(self.provider, "providers", None)
        if isinstance(chain, (list, tuple)):
            for item in chain:
                candidate = getattr(item, "model", None)
                if isinstance(candidate, str) and candidate:
                    return candidate
        return type(self.provider).__name__

    @staticmethod
    def _is_retryable_environment_blocker(report: str) -> bool:
        """Keep transient worktree/tool environment failures retryable."""
        try:
            payload = json.loads(report)
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict) or str(payload.get("status", "")).upper() != "BLOCKED":
            return False
        text = " ".join(str(payload.get(key, "")) for key in ("blocker", "summary", "next_hypothesis")).casefold()
        markers = (
            "dirty worktree",
            "worktree",
            "untracked file",
            "modified tracked",
            "environment",
            "executable is unavailable",
            "file or directory not found",
            "permission denied",
        )
        return any(marker in text for marker in markers)

    def _run_had_provider_failure(self, run_id: str) -> bool:
        """Report whether the provider chain, not the model, ended the run.

        Only a run that died because no provider answered is exempt from the
        task's give-up budget. Merely having retried a provider along the way is
        not enough: a run that consumed its own time or token budget after a few
        provider retries is a model-side failure, and treating it as an outage
        made a task retry forever while the give-up budget never moved.
        """
        for event in self.store.recent_run_events(run_id, limit=None):
            if event["kind"] == "run_result_committed":
                failure = str((event.get("payload") or {}).get("failure") or "")
                return "all providers failed" in failure
        return False

    def _consecutive_provider_failures(self) -> int:
        """Count trailing runs that died because no provider answered.

        A streak, not a total: one run with any other outcome resets it. This is
        the signal that stayed invisible when many runs failed in a row on an
        empty provider error with a recorded cooldown of zero.
        """
        rows = self.store.connection.execute(
            "SELECT status, failure FROM run_results ORDER BY created_at DESC, run_id DESC LIMIT ?",
            (max(1, self.config.provider_lockout_threshold),),
        ).fetchall()
        streak = 0
        for row in rows:
            if str(row["status"]) == RunStatus.NEEDS_RECOVERY.value and "all providers failed" in str(row["failure"] or ""):
                streak += 1
            else:
                break
        return streak

    def _provider_status_summary(self) -> dict[str, Any]:
        """Compact provider state for the escalation alert, never a network call."""
        probe = getattr(self.provider, "health_probe", None)
        if not callable(probe):
            return {}
        try:
            result = probe()
        except Exception as exc:
            return {"probe_error": str(exc)[:300]}
        if not isinstance(result, dict):
            return {}
        summary: dict[str, Any] = {"ok": result.get("ok"), "active_provider": result.get("active_provider")}
        providers = result.get("providers")
        if isinstance(providers, dict):
            summary["blocked"] = {
                name: detail.get("cooldown_remaining_seconds")
                for name, detail in providers.items()
                if isinstance(detail, dict) and detail.get("blocked")
            }
        return summary

    def _runtime_facts(self) -> dict[str, Any]:
        """Canonical environment facts the model kept rediscovering the hard way.

        Repeated bash calls failed with "/usr/bin/python3: No module
        named pytest" while a high-confidence memory recorded the working
        command that retrieval never returned.
        """
        root = Path(self.config.self_improvement_root or Path.cwd())
        venv_python = root / ".venv" / "bin" / "python"
        scratch = Path("/tmp/skynet-scratch")
        facts: dict[str, Any] = {
            "workspace": str(root),
            "test_command": f"{venv_python} -m pytest -q" if venv_python.exists() else f"{sys.executable} -m pytest -q",
            "python": sys.executable,
            "tool_count": len(self.runner.tools),
            "state_database": "state/skynet.sqlite3 (query it read-only with the db tool; never guess a table name)",
            "durable_code_change": "only propose_self_improvement makes a code change durable; editing the main worktree is quarantined",
            "scratch_directory": str(scratch),
        }
        try:
            scratch.mkdir(parents=True, exist_ok=True)
        except OSError:
            facts.pop("scratch_directory", None)
        # The model repeatedly queried tables that do not exist (`events` instead
        # of `event_log`, `planner_proposals` instead of `planner_attempts`), so
        # the schema is injected instead of being rediscovered by failure.
        with suppress(Exception):
            facts["tables"] = sorted(
                row[0]
                for row in self.store.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            )
        missing = [binary for binary in ("sqlite3", "jq") if shutil.which(binary) is None]
        if missing:
            facts["missing_binaries"] = missing
        return facts

    def _cosmetic_streak(self) -> dict[str, Any]:
        """Read the self-improvement registry for a run of cosmetic changes."""
        root = self.config.self_improvement_root or Path.cwd()
        try:
            manager = SelfImprovementManager(root)
            return manager.cosmetic_streak()
        except Exception as exc:
            log.debug("cosmetic streak check failed: %s", exc)
            return {"streak": 0}

    def _worktree_state(self) -> dict[str, Any]:
        """Report modified tracked files at wake, before any work is chosen.

        The dirty-worktree refusal used to arrive ten steps into a proposal; the
        organism now knows the state of its own tree before it starts.
        """
        root = Path(self.config.self_improvement_root or Path.cwd())
        if not (root / ".git").exists():
            return {}
        try:
            completed = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root, capture_output=True, text=True, check=False, timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        if completed.returncode:
            return {}
        files = [line[3:].strip().strip('"') for line in completed.stdout.splitlines() if line.strip()]
        if not files:
            return {"dirty": False}
        return {
            "dirty": True,
            "files": files[:20],
            "count": len(files),
            "note": "modified tracked files block a proposal and are quarantined; revert or quarantine them before proposing",
        }

    def _reboot_outcome(self) -> dict[str, Any]:
        """Report how the organism's own last promotion ended."""
        state_dir = Path(self.config.state_path).parent
        guard_path = state_dir / "reboot-guard.json"
        try:
            guard = json.loads(guard_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(guard, dict):
            return {}
        if guard.get("completed_at"):
            outcome = "accepted"
        elif guard.get("rolled_back"):
            outcome = "rolled_back"
        elif guard.get("failed"):
            outcome = "health_failed"
        elif guard.get("active"):
            outcome = "in_progress"
        else:
            return {}
        result: dict[str, Any] = {
            "outcome": outcome,
            "proposal_id": guard.get("proposal_id"),
            "commit": guard.get("commit"),
            "healthy_cycles": guard.get("healthy_cycles"),
        }
        if guard.get("rollback_error"):
            result["rollback_error"] = str(guard["rollback_error"])[:500]
        return result

    def _record_daily_metrics(self) -> None:
        """Persist one reproducible snapshot per UTC day into evaluations."""
        state_dir = Path(self.config.state_path).parent
        try:
            written = metrics.record_daily_snapshot(
                self.store,
                since_days=1.0,
                registry_path=state_dir / "self-improvement-proposals.json",
                state_dir=state_dir,
            )
        except Exception as exc:
            log.warning("daily metrics snapshot failed: %s", exc, exc_info=True)
            return
        if written is None:
            return
        self.store.append_event(
            "metrics_snapshot",
            {"window_days": written.get("window_days"), "marker": metrics.daily_marker()},
        )

    def _maybe_run_daily_maintenance(self) -> None:
        """Run retention/decay at most once per UTC day.

        The marker lives in a sidecar JSON file so the daily cadence does not
        depend on ``store.py`` or on the metrics snapshot being enabled. A
        missing or corrupt marker is treated as "due" so maintenance cannot be
        silently skipped.
        """
        marker_path = Path(self.config.state_path).parent / "maintenance-marker.json"
        today = metrics.daily_marker()
        try:
            stored = json.loads(marker_path.read_text(encoding="utf-8")).get("marker")
        except (OSError, ValueError, AttributeError):
            stored = None
        if stored == today:
            return
        self._run_daily_maintenance()
        try:
            marker_path.write_text(json.dumps({"marker": today}), encoding="utf-8")
        except OSError as exc:
            log.warning("daily maintenance marker write failed: %s", exc, exc_info=True)

    def _run_daily_maintenance(self) -> None:
        """Retention and GC, once per day, so the state cannot grow forever."""
        try:
            pruned = self.store.prune_event_log(self.config.event_retention_days)
            if pruned:
                self.store.append_event(
                    "history_pruned",
                    {"event_log": pruned, "retention_days": self.config.event_retention_days},
                )
        except Exception as exc:
            log.warning("event log retention failed: %s", exc, exc_info=True)
        try:
            decayed = self.store.decay_memory_confidence()
            if decayed["faded"] or decayed["dropped"]:
                self.store.append_event("memory_confidence_decayed", decayed)
        except Exception as exc:
            log.warning("memory confidence decay failed: %s", exc, exc_info=True)

    def _claims_uncaptured_changes(self, run_id: str, report: str) -> bool:
        """Detect a COMPLETED report that claims file changes nothing captured.

        Code changes are only durable through the self-improvement proposal
        workflow; a prototype edited in a scratch directory leaves the main
        worktree untouched.
        """
        try:
            payload = json.loads(report)
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict) or not payload.get("changes"):
            return False
        captured = {"propose_self_improvement", "promote_self_improvement"}
        for event in self.store.recent_run_events(run_id, limit=None):
            if event["kind"] != "tool_result":
                continue
            result = event.get("payload", {})
            tool_name = result.get("call", {}).get("tool_name")
            if tool_name in captured and result.get("result", {}).get("ok") is True:
                return False
        return True

    def _run_autonomous_planning(self, state: AgentState, goals: list[dict[str, Any]], trigger: str) -> list[str]:
        with self.store.transaction():
            current = self.store.state()
            self.store.transition(current, LifecycleState.PLAN, reason="no ready work; autonomous planning")
        tasks = [dict(row) for row in self.store.connection.execute("SELECT * FROM tasks ORDER BY updated_at DESC LIMIT 100")]
        memories = self.store.search_memories(self._planner_memory_query(state, goals), limit=20)
        previous = state.next_plan if isinstance(state.next_plan, dict) else {}
        proposals = self.autonomous_planner.generate(
            generation=state.generation,
            goals=goals,
            tasks=tasks,
            memories=memories,
            previous=previous,
            trigger=trigger,
        )
        for proposal in proposals:
            self._archive_proposal(proposal)
        return [str(item["task_id"]) for item in proposals if item.get("task_id")]

    @staticmethod
    def _inbox_notifications(inbox_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Surface pending owner messages as notifications, not as orders.

        The organism decides what to do with each one: act with its own tools,
        turn it into a task, or dismiss it with ``acknowledge_inbox``. Nothing is
        consumed here and no task is created, so the notification repeats on the
        next wake until the organism decides.
        """
        notifications: list[dict[str, Any]] = []
        for event in inbox_events:
            payload = event.get("payload")
            text = ""
            if isinstance(payload, dict):
                for key in ("text", "answer", "message"):
                    candidate = payload.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        text = candidate
                        break
            notifications.append({
                "kind": "inbox_notification",
                "event_id": str(event.get("event_id", "")),
                "message_kind": str(event.get("kind", "user_message")),
                "text": text[:4000],
                "note": (
                    "A pending owner message. Decide what to do with it: act with your own tools, "
                    "create a task, or dismiss it with acknowledge_inbox(event_id, decision). It stays "
                    "pending until you decide."
                ),
            })
        return notifications

    def _ensure_roadmap_seed(self) -> bool:
        """Re-seed the curated roadmap pool when no active work remains.

        ``handoff.seed`` is idempotent and is the only writer of ``area='roadmap'``,
        so calling it here cannot duplicate tasks. It is the fixed pool the empty
        planner falls back to, instead of sleeping with nothing to do.
        """
        from .handoff import seed

        # Only a truly empty goal table is re-seeded. A merely blocked goal is
        # revived by the scheduler; creating roadmap work here would mask it.
        if self.store.connection.execute("SELECT 1 FROM goals LIMIT 1").fetchone() is not None:
            return False
        root = Path(self.config.self_improvement_root or Path.cwd())
        try:
            summary = seed(self.store, root=root)
        except Exception:
            log.exception("roadmap re-seed failed; continuing without it")
            return False
        return bool(summary.get("tasks_created")) if isinstance(summary, dict) else False

    def _create_planner_fallback(self, goals: list[dict[str, Any]], generation: int) -> str | None:
        goal = goals[0] if goals else None
        if not goal:
            return None
        goal_id = str(goal["goal_id"])
        existing = self.store.connection.execute(
            "SELECT task_id, status FROM tasks WHERE goal_id=? AND title LIKE ? ORDER BY created_at DESC LIMIT 1",
            (goal_id, FALLBACK_TASK_TITLE + "%"),
        ).fetchone()
        if existing and existing["status"] in {"pending", "running"}:
            return str(existing["task_id"])
        # The previous fallback finished, so a fresh bounded diagnostic keeps the
        # organism alive instead of parking it in sleep. The generation suffix
        # gives it a new fingerprint and bounds it to one pending fallback.
        title = f"{FALLBACK_TASK_TITLE} [generation {generation}]"
        with self.store.transaction():
            task_id = self.store.add_task(title, goal_id=goal_id, expected_new_fact="A focused diagnostic establishes one verified fact or blocker", area="recovery")
            self.store.append_event(
                "planner_fallback_created",
                {"task_id": task_id, "goal_id": goal_id, "generation": generation, "reason": "planner produced no ready task"},
            )
        return task_id

    def _select_work(self) -> dict[str, Any] | None:
        return PortfolioPlanner(
            self.store,
            epsilon=self.config.planner_epsilon,
            hypothesis_ttl_days=self.config.hypothesis_ttl_days,
            cell_scarcity_weight=self.config.cell_scarcity_weight,
        ).select()

    def _request_self_improvement_restart(self, deferred_restart: dict[str, Any]) -> None:
        """Request a process restart after a completed promotion, without failing the cycle.

        The reboot request file written during promotion stays on disk, so a
        missed call is recoverable: the supervisor observes a pending request
        with no open window on the next cycle and retries the restart. Repeated
        failures escalate through a durable event counter and an outbox alert
        instead of being swallowed.
        """
        if self._self_improvement_restart is None:
            with self.store.transaction():
                self.store.append_event("restart_skipped", {"reason": "no restart callback configured", **deferred_restart})
            return
        try:
            self._self_improvement_restart()
        except Exception as exc:
            self.store.record_restart_failure(
                proposal_id=str(deferred_restart.get("proposal_id", "")),
                commit=str(deferred_restart.get("commit", "")),
                error=str(exc),
                limit=self.config.restart_failure_limit,
                source="reactor",
            )

    def set_self_improvement_health(self, health_check: Any) -> None:
        for name in ("propose_self_improvement", "promote_self_improvement"):
            tool = self.runner.tools.get(name)
            if isinstance(tool, (PromoteSelfImprovementTool, SelfImprovementTool)):
                tool.health_check = health_check

    def set_self_improvement_restart(self, restart: Any) -> None:
        self._self_improvement_restart = restart

    def set_stop_event(self, stop_event: Any) -> None:
        """Let a blocking owner-dialogue wait end when the process stops."""
        tool = self.runner.tools.get("ask_user")
        if isinstance(tool, AskUserTool):
            tool.set_stop_event(stop_event)

    @staticmethod
    @contextmanager
    def _defer_watchdog_signal() -> Iterator[None]:
        """Keep interrupt signals pending until the durable accounting commits.

        A watchdog or shutdown signal that lands in the middle of the post-run
        transaction would roll back memory consolidation, criteria evaluation,
        task updates, and the checkpoint. Deferring the signal keeps that work
        atomic; the signal is delivered right after the transaction commits.
        """
        if current_thread() is not main_thread():
            yield
            return
        deferred = {signal.SIGALRM, signal.SIGTERM, signal.SIGINT}
        try:
            signal.pthread_sigmask(signal.SIG_BLOCK, deferred)
        except (AttributeError, OSError, ValueError):
            yield
            return
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, deferred)

    def watchdog(self) -> bool:
        """Mark a run stale when its durable heartbeat exceeded the watchdog window."""
        if not self._run_lock.acquire(blocking=False):
            return False
        try:
            cutoff = (utc_datetime_now() - timedelta(seconds=self.config.watchdog_timeout_seconds)).isoformat().replace("+00:00", "Z")
            run_id = self.store.stale_active_run(cutoff)
            if run_id is None:
                return False
            self._interrupt_run(run_id, "watchdog_timeout")
            return True
        finally:
            self._run_lock.release()

    def _interrupt_run(self, run_id: str, reason: str) -> None:
        with self.store.transaction():
            state = self.store.state()
            if state.active_run_id != run_id:
                return
            self.store.finish_run(run_id, RunStatus.INTERRUPTED)
            self.store.commit_run_result(run_id, RunStatus.INTERRUPTED, "Run interrupted by the harness before a Finish Report.", 0, 0, reason)
            self.store.append_event("watchdog_timeout", {"run_id": run_id, "reason": reason}, run_id)
            state.active_run_id = None
            self.store.transition(state, LifecycleState.RECOVERING, run_id=run_id, reason=reason)
            state.next_plan = {**state.next_plan, "recovery": {"reason": reason, "run_id": run_id}}
            self.store.set_state(state)

    def interrupt_stale_run(self, *, reason: str = "watchdog_timeout", force: bool = False) -> RunStatus | None:
        """Interrupt the active run from the watchdog boundary.

        Unlike :meth:`watchdog`, this is safe to call while ``tick()`` is
        blocked inside a tool call (from a SIGALRM handler): it does not touch
        the run lock. Unless ``force`` is set, the run is only interrupted when
        its durable heartbeat is actually stale.
        """
        state = self.store.state()
        run_id = state.active_run_id
        if run_id is None:
            return None
        if not force:
            cutoff = (utc_datetime_now() - timedelta(seconds=self.config.watchdog_timeout_seconds)).isoformat().replace("+00:00", "Z")
            if self.store.stale_active_run(cutoff) != run_id:
                return None
        self._interrupt_run(run_id, reason)
        return RunStatus.INTERRUPTED

    def recover(self) -> None:
        """Mark an interrupted run and preserve its bounded episode for recovery."""
        with self.store.transaction():
            state = self.store.state()
            if state.active_run_id is not None:
                interrupted_run_id = state.active_run_id
                episode = self.store.recent_run_events(interrupted_run_id, limit=None)
                classification = self.store.classify_recovery(interrupted_run_id)
                recovery_status = RunStatus.NEEDS_RECOVERY if classification["status"] == "result_committed" else RunStatus.INTERRUPTED
                self.store.finish_run(interrupted_run_id, recovery_status)
                self.store.commit_run_result(
                    interrupted_run_id,
                    recovery_status,
                    "Run interrupted by a process restart before a Finish Report.",
                    0,
                    0,
                    "process_restart",
                )
                started_event = next((event for event in episode if event["kind"] == "run_started"), None)
                if started_event and isinstance(started_event["payload"].get("pending_work"), list):
                    for work in started_event["payload"]["pending_work"]:
                        task = work.get("task") if isinstance(work, dict) else None
                        if isinstance(task, dict) and isinstance(task.get("task_id"), str):
                            self.store.apply_task_updates([{"task_id": task["task_id"], "status": "pending"}], run_id=interrupted_run_id)
                self.store.append_event(
                    "run_interrupted",
                    {"reason": "process_restart", "episode_events": len(episode), "classification": classification},
                    interrupted_run_id,
                )
                state.next_plan = {
                    **state.next_plan,
                    "recovery": {
                        "reason": "process_restart",
                        "run_id": interrupted_run_id,
                        "episode_event_count": len(episode),
                        "episode_snapshot_id": self.store.snapshot_episode(interrupted_run_id)["snapshot_id"],
                        "classification": classification,
                    },
                }
            state.active_run_id = None
            self.store.transition(state, LifecycleState.RECOVER, run_id=state.active_run_id, reason="process restart recovery")

    # `MemoryStore._normalize_terms` keeps only the *last* `_MAX_QUERY_TERMS`
    # (24) normalized terms, so a long `next_plan["initial_prompt"]` could
    # silently evict the goal and task words: the searched terms were the
    # previous run's bookkeeping and the situation-defining words never reached
    # the index. Long parts are capped, and the situation-defining parts are
    # appended last so the recall window keeps them.
    _MEMORY_QUERY_PART_CHARS = 300

    @staticmethod
    def _join_query_parts(parts: list[str]) -> str:
        bounded = [str(part)[:Reactor._MEMORY_QUERY_PART_CHARS] for part in parts if part]
        return " ".join(bounded).strip()[:1000]

    @staticmethod
    def _planner_memory_query(state: AgentState, goals: list[dict[str, Any]]) -> str:
        """Derive the planner's recall query instead of searching a fixed phrase.

        The fixed string "autonomous planning next bounded work" recalled nothing
        useful about the actual situation; the goal title, the previous outcome
        and the intended next step are what the planning decision depends on.
        """
        next_plan = state.next_plan if isinstance(state.next_plan, dict) else {}
        previous = next_plan.get("previous_outcome")
        parts: list[str] = []
        if isinstance(previous, dict):
            parts.append(str(previous.get("summary", "")))
        parts.append(str(next_plan.get("initial_prompt", "")))
        parts.append(str(next_plan.get("next", "")))
        parts.append("autonomous planning next bounded work")
        if goals:
            parts.append(str(goals[0].get("title", "")))
        return Reactor._join_query_parts(parts)

    @staticmethod
    def _memory_query(selected_work: dict[str, Any] | None, inbox_events: list[dict[str, Any]], next_plan: dict[str, Any]) -> str:
        # `selected_work` is `task_work()`'s envelope, so the title lives under
        # `task`/`goal`; reading `selected_work["title"]` returned an empty string
        # for every task, which made the query empty and recall silently dead.
        parts: list[str] = []
        for event in inbox_events[:3]:
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                parts.append(str(payload.get("text", "")))
        parts.append(str(next_plan.get("initial_prompt", "")))
        parts.append(str(next_plan.get("next", "")))
        if selected_work:
            raw_task = selected_work.get("task")
            raw_goal = selected_work.get("goal")
            task: dict[str, Any] = raw_task if isinstance(raw_task, dict) else {}
            goal: dict[str, Any] = raw_goal if isinstance(raw_goal, dict) else {}
            parts.append(str(goal.get("title", "")))
            title = task.get("title") or goal.get("title") or selected_work.get("title", "")
            parts.append(str(title))
            parts.append(str(task.get("expected_new_fact", "")))
        return Reactor._join_query_parts(parts)

    def close(self) -> None:
        self.store.close()

    @staticmethod
    def _intention(selected_work: dict[str, Any] | None, next_plan: dict[str, Any]) -> dict[str, Any]:
        if selected_work:
            title = selected_work.get("task", {}).get("title") or selected_work.get("goal", {}).get("title") or "selected work"
        else:
            title = "inspect durable state and choose useful work"
        return {"title": str(title), "why_now": str(next_plan.get("initial_prompt", "continue the highest-priority work"))}

    @staticmethod
    def _success_criteria(selected_work: dict[str, Any] | None) -> list[dict[str, Any]]:
        if selected_work:
            title = selected_work.get("task", {}).get("title") or selected_work.get("goal", {}).get("title") or "selected work"
        else:
            title = "selected work"
        return [{
            "criterion": f"produce verified evidence for the selected work: {title}",
            "kind": "verified_progress",
            "required": True,
        }]
