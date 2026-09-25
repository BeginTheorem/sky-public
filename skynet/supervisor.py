from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from .checkpoints import CheckpointError, CheckpointManager
from .lock import ProcessLock
from .memory_audit import checkpoint_memory_audit, memory_audit_continuity_report, verify_chain
from .models import LifecycleState
from .proposal_registry import audit_promotion_registry, sweep_environment_blocks
from .provider import LLMProvider, Tool
from .reactor import HEALTHY_LIFECYCLES, SAFETY_NET_FINGERPRINTS, Reactor, ReactorConfig
from .recovery import RebootGuard
from .rollback import consume_request
from .self_improvement import SelfImprovementManager

log = logging.getLogger("skynet.supervisor")


def _client_first(services: list[str]) -> list[str]:
    """Order units so the Telegram client is restarted before the reactor.

    ``SKYNET_TELEGRAM_SERVICE`` is authoritative when set: the named unit is
    moved first even if it does not contain ``telegram``. With the env var unset
    the name match is the fallback, and an ambiguous candidate set (no unit looks
    like a client, or several do) is logged rather than raised -- a restart must
    still happen, and silently returning the reactor first is the stale-client
    bug this ordering exists to prevent. Every other unit keeps its relative
    order.
    """
    configured = os.getenv("SKYNET_TELEGRAM_SERVICE", "").strip()
    if configured:
        clients = [unit for unit in services if unit == configured]
        if not clients:
            log.warning(
                "configured telegram service %s is not in the restart set %s; client-first ordering skipped",
                configured,
                services,
            )
    else:
        clients = [unit for unit in services if "telegram" in unit.lower()]
        if not clients:
            log.warning(
                "no telegram unit in the restart set %s and SKYNET_TELEGRAM_SERVICE is unset; "
                "client-first ordering skipped",
                services,
            )
        elif len(clients) > 1:
            log.warning("ambiguous telegram client set %s; keeping the given order", clients)
            return list(services)
    others = [unit for unit in services if unit not in clients]
    return clients + others


def sync_deployed_commit(root: str | Path) -> bool:
    """Point ``.deployed-commit`` at the commit the worktree is actually on.

    ``scripts/deploy.sh`` and ``scripts/rollback.sh`` are the only writers, so
    every other way HEAD moves -- an in-place self-improvement restart, a manual
    pull, a checkout -- leaves the marker naming the last deployed commit while
    the tree has moved on. The marker is observability, not a control input: it
    is excluded from the checkpoint hash (``checkpoints.is_clean``) and nothing
    reads it to decide behaviour, so repairing it at startup only makes "what is
    deployed?" answerable from the tree itself.

    Best-effort by construction: a non-git root, a git failure, or an
    unwritable marker is logged and ignored, never fatal. Returns ``True`` when
    the marker was rewritten, ``False`` when it was already current or could not
    be read/written.
    """
    root_path = Path(root)
    marker = root_path / ".deployed-commit"
    if not (root_path / ".git").exists():
        return False
    try:
        current = CheckpointManager(root_path).current_commit()
    except CheckpointError as exc:
        log.warning("deployed-commit sync skipped: %s", exc)
        return False
    if not current:
        return False
    try:
        recorded = marker.read_text(encoding="utf-8").strip()
    except OSError:
        recorded = ""
    if recorded == current:
        return False
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{marker.name}.", dir=root_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(current + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, marker)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
    except OSError as exc:
        log.warning("deployed-commit sync failed: %s", exc)
        return False
    log.info("deployed-commit synced to %s", current)
    return True


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
        `ROADMAP_TASKS`. It used to run only from the manual `skynet handoff`
        command, so a database that predates a curation change kept selecting
        dropped work: at generation 172 the live table held 18 area='roadmap'
        rows against 4 curated entries, and the planner selected a task whose
        acceptance criterion was already met. Seeding here converges the live
        state on the curated list without an operator step, and a failure is
        logged rather than fatal because the roadmap is scaffolding, not a
        precondition for running.
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
            self.reactor.store.append_event("supervisor_start", {"state_path": str(self.config.state_path)})
            log.info("supervisor starting; recovering durable state")
            # The deployed marker is written only by deploy.sh/rollback.sh, so a
            # tree that moved without them would report a stale commit. Repair
            # it before any lifecycle work; it is best-effort and never fatal.
            sync_deployed_commit(self.root)
            # The runtime backup/release subsystem is gone: deploy shares one
            # git history, and its restore could quarantine .git/, .venv/ and
            # tests/ (P0-2). Startup performs no destructive filesystem work;
            # a rollback request is executed by consume_request, which already
            # refuses a commit that is not an ancestor of HEAD.
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
            repaired = self.reactor.store.repair_status_consistency(exclude_fingerprints=SAFETY_NET_FINGERPRINTS)
            if repaired["hypotheses"] or repaired["tasks"]:
                log.warning("repaired inconsistent task/hypothesis statuses: %s", repaired)
            backfilled = self.reactor.store.backfill_missing_run_results()
            if backfilled:
                log.warning("backfilled %d run result rows", backfilled)
            self.reboot_guard.begin(Path(self.config.state_path).parent / "reboot-request.json")
            # Housekeeping below is best-effort. Each call touches an external
            # file or git worktree, so an unwritable/corrupt proposals file or a
            # broken worktree must not prevent the organism from starting at all;
            # a failure is logged and startup continues.
            try:
                reconciled = self.self_improvement.reconcile_awaiting_reboot()
                if reconciled:
                    log.warning("reconciled previous self-improvement proposals: %s", reconciled)
                    self.reactor.store.append_event("improvement_proposals_reconciled", {"resolved": reconciled})
            except Exception:
                log.exception("awaiting-reboot reconciliation failed; continuing")
            # The retry guard stops retrying an environment-blocked change once
            # its bounded ceiling is spent, but nothing wrote that outcome back:
            # the record stayed blocked_by_environment, which is terminal to
            # every other sweep, while the guard still read it as pending
            # repair. Close those records at startup so the two agree.
            try:
                closed = sweep_environment_blocks(self.self_improvement.proposals_path)
                if closed:
                    log.warning("closed environment-blocked proposals: %s", closed)
                    self.reactor.store.append_event("improvement_environment_blocks_resolved", {"resolved": closed})
            except Exception:
                log.exception("environment-block sweep failed; continuing")
            # A record can read 'accepted' while its promoted commit is no
            # longer an ancestor of HEAD: reconcile only revisits awaiting_reboot
            # and promoting rows, and an external rollback resets HEAD past
            # every later promotion. Nothing rewrites the record here -- the
            # detector only names the contradiction so the loss is observable.
            try:
                orphans = audit_promotion_registry(self.self_improvement.proposals_path, self.root)
                if orphans:
                    log.warning("promotions that are neither live nor rolled back: %s", orphans)
                    self.reactor.store.append_event("promotion_liveness_violations", {"violations": orphans})
            except Exception:
                log.exception("promotion-liveness audit failed; continuing")
            # The owner keeps his own continuity with a permanent read-only log
            # of memory-change metadata and offered to hold the same obligation
            # for this one. The store is compared against the last checkpoint:
            # a mismatch is reported as a fact -- that it changed, when, and by
            # how many rows and bytes -- together with the two provenance counts
            # that separate an in-band write from an outside one. No content is
            # read back out of the log and nothing is repaired here.
            try:
                state_dir = Path(self.config.state_path).parent
                continuity = memory_audit_continuity_report(self.reactor.store.connection, state_dir)
                if continuity.get("status") == "changed_externally":
                    log.warning("memory store changed outside the audit checkpoints: %s", continuity)
                    self.reactor.store.append_event("memory_audit_external_change", continuity)
                elif continuity.get("status") == "no_baseline":
                    checkpoint_memory_audit(self.reactor.store.connection, state_dir, reason="startup_baseline")
                # The continuity report above compares the store against the
                # checkpoint, and both live in files the checkpoint writer
                # rewrites -- so a writer who re-issues both consistently
                # reports "unchanged" over an outside memory change and erases
                # the only trace of it (measured: one forged line
                # plus a matching checkpoint turned changed_externally into
                # unchanged, while verify_chain already named the break at line
                # 1). verify_chain was reachable only from the test suite, so
                # that detection reached nothing the organism could see. It runs
                # here on every start and reports; it never repairs, and never
                # reads content back out of the log.
                chain = verify_chain(state_dir)
                if not chain.get("ok", True):
                    log.warning("memory audit chain does not verify: %s", chain)
                    self.reactor.store.append_event("memory_audit_chain_broken", chain)
            except Exception:
                log.exception("memory-audit continuity check failed; continuing")
            try:
                orphans = self.self_improvement.prune_orphan_worktrees()
                if orphans:
                    log.warning("removed %d orphaned self-improvement worktrees", len(orphans))
                    self.reactor.store.append_event("improvement_worktrees_pruned", {"removed": orphans})
            except Exception:
                log.exception("orphan worktree pruning failed; continuing")
            self._started = True
        except BaseException:
            self.lock.release()
            raise

    def observe_reboot(self) -> dict[str, object]:
        request_path = Path(self.config.state_path).parent / "reboot-request.json"
        state = self.reactor.store.state()
        if request_path.exists() and not self.reboot_guard.path.exists() and state.active_run_id is None:
            # A deferred restart was missed (for example the watchdog killed the
            # cycle after the promotion was merged). The request file is durable
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
                # No git worktree means no usable rollback commit. The runtime
                # backup fallback used to quarantine unrelated files (P0-2);
                # now the refusal is recorded and nothing on disk is moved.
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
            self.reactor.store.append_event("reboot_observation", {"health": health, "result": result})
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
            self.reactor.store.append_event("supervisor_stop", {})
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

    def restart_service(self, service: str | list[str]) -> None:
        """Request a restart of the given unit(s) without invoking a shell.

        The Telegram control bot is a separate systemd unit that imports the
        same tree, so a promotion only takes effect after *both* units restart.
        Restarting one unit left the bot serving a stale module in memory: a
        promoted POLL_TIMEOUT change was committed at 22:40 while the bot
        process was still the one started at 17:33. Accepting a list keeps the
        reboot path aligned with the CLI's own ``start``/``stop``/``reboot``,
        which already act on the pair.

        The client is ordered before the reactor inside this method, not at the
        call site: ``systemctl restart a b`` starts the stop job for ``a`` first,
        and this callback runs inside the reactor process, so naming the reactor
        first killed the client before systemd handled the rest of the
        transaction. The order was previously encoded only in ``cli.py``, so any
        other caller could reintroduce the stale-client bug.
        """
        services = [service] if isinstance(service, str) else list(service)
        if not services:
            raise ValueError("at least one unit name is required")
        for unit in services:
            if not unit or unit != Path(unit).name:
                raise ValueError("service must be a unit name")
        services = _client_first(services)
        self._record_restart_requested(services)
        subprocess.Popen(["systemctl", "restart", *services], close_fds=True)

    def _record_restart_requested(self, units: list[str]) -> None:
        """Make a requested restart observable in the durable event log.

        The failure path already wrote ``restart_failed``/``restart_escalated``,
        but the success path wrote nothing, so "did the restart actually fire?"
        had no answer in state: event_log held zero restart or promotion rows
        across generations 9-10, and two episodes could not verify that a
        promoted change had reached the Telegram bot process. Recording the
        request here - at the single choke point every restart path goes
        through - makes the answer a row instead of an inference. The write is
        failure-tolerant: observability must never prevent the restart itself.
        """
        try:
            with self.reactor.store.transaction():
                self.reactor.store.append_event(
                    "restart_requested",
                    {"units": list(units), "pid": os.getpid(), "source": "supervisor"},
                )
        except Exception:
            log.exception("could not record restart request for units=%s", units)

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
