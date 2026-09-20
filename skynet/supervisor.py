from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from .checkpoints import CheckpointError, CheckpointManager
from .lock import ProcessLock
from .models import LifecycleState
from .provider import LLMProvider, Tool
from .reactor import HEALTHY_LIFECYCLES, Reactor, ReactorConfig
from .recovery import RebootGuard
from .rollback import consume_request
from .self_improvement import SelfImprovementManager

log = logging.getLogger("skynet.supervisor")


class Supervisor:
    """Process boundary. It owns the lock, but does not start itself on import."""

    def __init__(self, provider: LLMProvider, tools: Mapping[str, Tool], config: ReactorConfig | None = None, *, root: str | Path | None = None) -> None:
        self.config = config or ReactorConfig()
        self.lock = ProcessLock(Path(self.config.state_path).with_suffix(".lock"))
        self.reactor = Reactor(provider, tools, self.config)
        self.root = Path(root or Path.cwd())
        self.reboot_guard = RebootGuard(Path(self.config.state_path).parent, max_window_age_seconds=self.config.reboot_window_max_age_seconds)
        self.self_improvement = SelfImprovementManager(self.root)
        self._converge_handoff()
        self._restart_callback: Callable[[], None] | None = None
        self._started = False

    def _converge_handoff(self) -> None:
        """Re-seed the roadmap at startup so the curated task list is the live one.

        `seed` is idempotent and is the only writer of `area='roadmap'`, so it is
        the only place that can retire a task whose entry was deleted from
        `ROADMAP_TASKS`. Seeding here converges the live state on the curated
        list without an operator step, and a failure is logged rather than fatal
        because the roadmap is scaffolding, not a precondition for running.
        """
        from .handoff import seed

        try:
            summary = seed(self.reactor.store, root=self.root)
        except Exception:
            log.exception("roadmap handoff convergence failed; continuing")
            return
        if summary["tasks_created"] or summary["tasks_retired"]:
            log.info(
                "roadmap handoff converged created=%s retired=%s",
                summary["tasks_created"],
                summary["tasks_retired"],
            )

    def start(self) -> None:
        if self._started:
            return
        self.lock.acquire()
        try:
            self.reactor.store.runtime_log.write("supervisor_start", {"state_path": str(self.config.state_path)})
            log.info("supervisor starting; recovering durable state")
            # Startup performs no destructive filesystem work; a rollback
            # request is executed by consume_request, which already refuses a
            # commit that is not an ancestor of HEAD.
            rollback = consume_request(self.root)
            if rollback is not None:
                self.reactor.store.append_event("rollback_request_executed", rollback)
                if rollback.get("applied"):
                    # The running tree is not the tree on disk; restart through the
                    # same deferred path a promotion uses.
                    self.reactor.store.raise_alert(
                        "rollback_requested_by_agent",
                        {"rollback_commit": rollback.get("rollback_commit"), "reason": rollback.get("reason")},
                        severity="critical",
                        dedup_key="rollback_requested_by_agent",
                    )
                    if self._restart_callback is not None:
                        self._restart_callback()
            self.reactor.recover()
            repaired = self.reactor.store.repair_status_consistency()
            if repaired["hypotheses"] or repaired["tasks"]:
                log.warning("repaired inconsistent task/hypothesis statuses: %s", repaired)
            backfilled = self.reactor.store.backfill_missing_run_results()
            if backfilled:
                log.warning("backfilled %d run result rows", backfilled)
            self.reboot_guard.begin(Path(self.config.state_path).parent / "reboot-request.json")
            reconciled = self.self_improvement.reconcile_awaiting_reboot()
            if reconciled:
                log.warning("reconciled previous self-improvement proposals: %s", reconciled)
                self.reactor.store.append_event("improvement_proposals_reconciled", {"resolved": reconciled})
            orphans = self.self_improvement.prune_orphan_worktrees()
            if orphans:
                log.warning("removed %d orphaned self-improvement worktrees", len(orphans))
                self.reactor.store.append_event("improvement_worktrees_pruned", {"removed": orphans})
            self._started = True
        except BaseException:
            self.lock.release()
            raise

    def observe_reboot(self) -> dict[str, object]:
        request_path = Path(self.config.state_path).parent / "reboot-request.json"
        state = self.reactor.store.state()
        if request_path.exists() and not self.reboot_guard.path.exists() and state.active_run_id is None:
            # A deferred restart may be missed. The request file is durable
            # and unconsumed, so the restart is retried here instead of waiting
            # for an unrelated future reboot.
            proposal_id, commit = self._read_request_identity(request_path)
            self.reactor.store.append_event(
                "restart_recovered",
                {"reason": "pending reboot request with no open window", "proposal_id": proposal_id},
            )
            if self._restart_callback is not None:
                try:
                    self._restart_callback()
                except Exception as exc:
                    self.reactor.store.record_restart_failure(
                        proposal_id=proposal_id,
                        commit=commit,
                        error=str(exc),
                        limit=self.config.restart_failure_limit,
                        source="supervisor_recovery",
                    )
            return {"active": False, "ok": True, "changed": True, "restart_requested": True}
        rollback_action: Callable[[str], None] | None = None
        if (self.root / ".git").exists():
            def rollback(commit: str) -> None:
                checkpoint = CheckpointManager(self.root)
                if not checkpoint.is_ancestor(commit):
                    # A target that is not an ancestor would discard every
                    # later commit. Record the refusal durably, leave the tree
                    # untouched and let the window stay open with the failure.
                    self.reactor.store.append_event(
                        "rollback_refused",
                        {"commit": commit, "reason": "rollback commit is not an ancestor of HEAD"},
                    )
                    raise CheckpointError(
                        f"refusing to roll back to {commit}: it is not an ancestor of HEAD, "
                        "so resetting would discard later commits"
                    )
                checkpoint._git(["reset", "--hard", commit])
            rollback_action = rollback
        elif self.reboot_guard.path.exists():
            def refuse_without_git(commit: str) -> None:
                # No git worktree means no usable rollback commit. The refusal
                # is recorded and nothing on disk is moved.
                self.reactor.store.append_event(
                    "rollback_refused",
                    {"commit": commit, "reason": "no git worktree to roll back"},
                )
                raise CheckpointError(f"refusing to roll back to {commit}: no git worktree")
            rollback_action = refuse_without_git
        # The reboot window judges process health only. A provider outage is a
        # transient external condition and must never roll back a promotion
        # that already started successfully.
        health = self.health_check(include_provider=False)
        result = self.reboot_guard.observe(health, rollback_action)
        request = result.get("proposal_id")
        if isinstance(request, str) and result.get("changed"):
            self.self_improvement.mark_reboot_result(request, result)
        if result.get("rolled_back"):
            # The tree on disk was reset, but this process still holds the
            # broken promoted code in memory. Reuse the deferred restart path a
            # promotion uses so the rollback actually takes effect.
            self.reactor.store.append_event(
                "rollback_restart_requested",
                {"commit": result.get("commit"), "proposal_id": request if isinstance(request, str) else ""},
            )
            if self._restart_callback is not None:
                try:
                    self._restart_callback()
                except Exception as exc:
                    self.reactor.store.record_restart_failure(
                        proposal_id=request if isinstance(request, str) else "",
                        commit=str(result.get("commit", "")),
                        error=str(exc),
                        limit=self.config.restart_failure_limit,
                        source="supervisor_reboot_rollback",
                    )
        if result.get("changed"):
            self.reactor.store.runtime_log.write("reboot_observation", {"health": health, "result": result})
            if result.get("completed") and isinstance(request, str):
                self.reactor.store.reset_planning_context(
                    reason="self_improvement_accepted",
                    proposal_id=request,
                    commit=str(result.get("commit", "")),
                )
            if result.get("completed") or result.get("quarantined"):
                # A finished or quarantined window unblocks every other
                # awaiting_reboot entry at once, not only on the next start.
                reconciled = self.self_improvement.reconcile_awaiting_reboot()
                if reconciled:
                    self.reactor.store.append_event("improvement_proposals_reconciled", {"resolved": reconciled})
        return result

    def stop(self) -> None:
        if not self._started:
            return
        log.info("supervisor stopping")
        try:
            self.reactor.store.runtime_log.write("supervisor_stop", {})
        finally:
            self.reactor.close()
            self.lock.release()
            self._started = False

    def set_restart_callback(self, callback: Callable[[], None]) -> None:
        self._restart_callback = callback

    @staticmethod
    def _read_request_identity(request_path: Path) -> tuple[str, str]:
        try:
            payload = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "", ""
        if not isinstance(payload, dict):
            return "", ""
        return str(payload.get("proposal_id", "")), str(payload.get("commit", ""))

    def restart_service(self, service: str) -> None:
        """Request a restart of this service without invoking a shell."""
        if not service or service != Path(service).name:
            raise ValueError("service must be a unit name")
        subprocess.Popen(["systemctl", "restart", service], close_fds=True)

    def health_check(self, *, include_provider: bool = True) -> dict[str, object]:
        """Run cheap, side-effect-free checks at the process boundary.

        ``include_provider=False`` yields the process-only health used by the
        reboot guard, where an unreachable provider is not a failed promotion.
        """
        checks: dict[str, bool] = {}
        details: dict[str, str] = {}
        healthy = HEALTHY_LIFECYCLES | {LifecycleState.RECOVERING}
        try:
            state = self.reactor.store.state()
            checks["state"] = state.lifecycle in healthy
            details["lifecycle"] = state.lifecycle.value
        except Exception as exc:
            checks["state"] = False
            details["state_error"] = str(exc)[:500]
        try:
            tables = {
                row["name"] for row in self.reactor.store.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            required = {"agent_state", "runs", "run_results", "event_log", "inbox", "outbox", "capability_effects", "episode_snapshots", "memory_consolidations", "recovery_reconciliations", "schema_migrations"}
            checks["schema"] = required <= tables
            details["missing_tables"] = ",".join(sorted(required - tables))
        except Exception as exc:
            checks["schema"] = False
            details["schema_error"] = str(exc)[:500]
        probe = getattr(self.reactor.provider, "health_probe", None)
        if include_provider and callable(probe):
            provider_health = cast(dict[str, object], probe())
            checks["provider_reachable"] = bool(provider_health.get("ok"))
            details["provider"] = str(provider_health)
        return {"ok": all(checks.values()), "checks": checks, "details": details}
