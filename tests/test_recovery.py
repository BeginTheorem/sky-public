"""Split from the former monolithic CoreTests suite."""

from __future__ import annotations

import json
import os
import signal
import tempfile
import threading
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from helpers import FakeProvider, FixtureTool

from skynet.models import LifecycleState, RunStatus
from skynet.reactor import Reactor, ReactorConfig, WatchdogTimeout
from skynet.recovery import RebootGuard
from skynet.store import StateStore


class CoreTests(unittest.TestCase):
    def test_interrupt_stale_run_marks_active_run_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            from skynet.models import Budget, RunRecord
            from skynet.time import utc_now
            reactor_config = ReactorConfig(state_path=Path(directory) / "state.sqlite3", watchdog_timeout_seconds=5)
            reactor = Reactor(FakeProvider(), {}, reactor_config)
            store = reactor.store
            run = RunRecord("interrupted-run", 1, RunStatus.RUNNING, utc_now(), Budget())
            with store.transaction():
                store.create_run(run)
                state = store.state()
                state.active_run_id = "interrupted-run"
                store.transition(state, LifecycleState.REACT, run_id="interrupted-run", reason="test")
                store.set_state(state)
            # The in-memory ReAct counters are gone once the process is killed;
            # the durable running totals must survive into the ledger row.
            with store.transaction():
                store.touch_run("interrupted-run", "react", steps=7, usage_tokens=4242)
            self.assertIsNone(reactor.interrupt_stale_run(reason="watchdog_timeout"))
            result = reactor.interrupt_stale_run(reason="watchdog_timeout", force=True)
            self.assertEqual(result, RunStatus.INTERRUPTED)
            status = store.connection.execute("SELECT status FROM runs WHERE run_id=?", ("interrupted-run",)).fetchone()[0]
            self.assertEqual(status, "interrupted")
            ledger = store.connection.execute("SELECT steps, usage_tokens FROM run_results WHERE run_id=?", ("interrupted-run",)).fetchone()
            self.assertEqual((ledger[0], ledger[1]), (7, 4242))
            self.assertIsNone(store.state().active_run_id)
            self.assertEqual(store.state().lifecycle, LifecycleState.RECOVERING)
            store.close()
    def test_watchdog_timeout_is_base_exception(self) -> None:
        self.assertTrue(issubclass(WatchdogTimeout, BaseException))
        self.assertFalse(issubclass(WatchdogTimeout, Exception))
    def test_recovery_handles_lifecycle_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                from skynet.models import Budget, RunRecord
                from skynet.time import utc_now
                store.create_run(RunRecord("run-chaos", 1, RunStatus.RUNNING, utc_now(), Budget()))
                state = store.state()
                state.active_run_id = "run-chaos"
                store.set_state(state)
                for phase in ("initial", "common", "finish"):
                    store.append_event("react_phase", {"phase": phase}, "run-chaos")
            store.close()
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            reactor.recover()
            self.assertEqual(reactor.store.state().lifecycle.value, "recover")
            reactor.close()
    def test_recovery_keeps_task_in_normal_work_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            with store.transaction():
                from skynet.models import Budget, RunRecord
                from skynet.time import utc_now
                goal_id = store.add_goal("recovery goal")
                task_id = store.add_task("repeating recovery task", goal_id)
                store.create_run(RunRecord("run-recovery", 1, RunStatus.RUNNING, utc_now(), Budget()))
                store.connection.execute(
                    "UPDATE tasks SET status='running', attempts=3 WHERE task_id=?",
                    (task_id,),
                )
                state = store.state()
                state.active_run_id = "run-recovery"
                store.set_state(state)
                store.append_event(
                    "run_started",
                    {"pending_work": [{"kind": "task", "task": {"task_id": task_id}}]},
                    "run-recovery",
                )
            store.close()
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=path))
            reactor.recover()
            self.assertEqual(
                reactor.store.connection.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0],
                "pending",
            )
            reactor.close()
    def test_episode_snapshot_contains_sqlite_transcript_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                store.append_transcript("provider_request", {"messages": []}, "run-1")
                snapshot = store.snapshot_episode("run-1")
            self.assertEqual(snapshot["transcript"][0]["kind"], "provider_request")
            stored = store.connection.execute("SELECT payload FROM episode_snapshots WHERE run_id=?", ("run-1",)).fetchone()[0]
            self.assertIn('"transcript"', stored)
            store.close()
    def test_recovery_preserves_bounded_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            run_id = "interrupted-run"
            with store.transaction():
                from skynet.models import Budget, RunRecord
                from skynet.time import utc_now
                store.create_run(RunRecord(run_id, 1, RunStatus.RUNNING, utc_now(), Budget()))
                state = store.state()
                state.active_run_id = run_id
                store.set_state(state)
                store.append_event("run_started", {"run_id": run_id}, run_id)
                store.append_event("tool_result", {"result": {"ok": True}}, run_id)
            store.close()
            reactor = Reactor(FakeProvider(), {"fixture_tool": FixtureTool()}, ReactorConfig(state_path=path))
            reactor.recover()
            recovery = reactor.store.state().next_plan["recovery"]
            self.assertEqual(recovery["run_id"], run_id)
            self.assertEqual(recovery["episode_event_count"], 2)
            self.assertTrue(recovery["episode_snapshot_id"])
            self.assertEqual(reactor.store.connection.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()[0], "interrupted")
            reactor.close()
    def test_memory_consolidation_is_versioned_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                first = store.consolidate_versioned("episode-1", "run-1", [{"kind": "fact", "content": "durable fact"}])
                second = store.consolidate_versioned("episode-1", "run-1", [{"kind": "fact", "content": "durable fact"}])
            self.assertEqual(first["output_version"], 1)
            self.assertFalse(first["replayed"])
            self.assertTrue(second["replayed"])
            self.assertEqual(store.connection.execute("SELECT version FROM memory_meta").fetchone()[0], 1)
            store.close()
    def test_recovery_classifies_unresolved_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                store.append_event("tool_call", {"call_id": "call-1", "tool_name": "write"}, "run-1")
            classification = store.classify_recovery("run-1")
            self.assertEqual(classification, {"status": "unknown_outcome", "call_ids": ["call-1"]})
            store.reconcile_tool_call("run-1", "call-1", "not_applied", {"reason": "verified absent"})
            self.assertEqual(store.classify_recovery("run-1")["status"], "retryable")
            store.close()
    def test_reconciliation_is_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                store.append_event("tool_call", {"call_id": "call-1", "tool_name": "write"}, "run-1")
            first = store.reconcile_tool_call("run-1", "call-1", "not_applied", {"reason": "verified absent"})
            second = store.reconcile_tool_call("run-1", "call-1", "unknown", {"reason": "inspection unavailable"})
            self.assertEqual(second, first)
            self.assertEqual(store.classify_recovery("run-1")["status"], "retryable")
            store.close()
    def test_recovery_classifies_committed_result_before_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                from skynet.models import Budget, RunRecord
                from skynet.time import utc_now
                store.create_run(RunRecord("run-1", 1, RunStatus.RUNNING, utc_now(), Budget()))
                store.commit_run_result("run-1", RunStatus.COMPLETED, "durable result", 1, 2, "")
            self.assertEqual(store.classify_recovery("run-1")["status"], "result_committed")
            store.close()
    def test_episode_snapshot_is_immutable_and_unbounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                for index in range(150):
                    store.append_event("tool_result", {"call": {"call_id": str(index)}, "result": {"ok": True}}, "run-1")
                first = store.snapshot_episode("run-1")
                store.append_event("tool_result", {"call": {"call_id": "late"}, "result": {"ok": True}}, "run-1")
                second = store.snapshot_episode("run-1")
            self.assertEqual(len(first["events"]), 150)
            self.assertEqual(first["snapshot_id"], second["snapshot_id"])
            self.assertEqual(len(second["events"]), 150)
            store.close()
    def test_watchdog_ignores_unrelated_stale_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=path, watchdog_timeout_seconds=1))
            with reactor.store.transaction():
                from skynet.models import Budget, RunRecord
                from skynet.time import utc_now
                reactor.store.create_run(RunRecord("stale-orphan", 1, RunStatus.RUNNING, utc_now(), Budget()))
                reactor.store.create_run(RunRecord("fresh-active", 1, RunStatus.RUNNING, utc_now(), Budget()))
                reactor.store.connection.execute("UPDATE runs SET heartbeat_at=? WHERE run_id=?", ("2000-01-01T00:00:00+00:00", "stale-orphan"))
                state = reactor.store.state()
                state.active_run_id = "fresh-active"
                reactor.store.set_state(state)
            self.assertFalse(reactor.watchdog())
            self.assertEqual(reactor.store.connection.execute("SELECT status FROM runs WHERE run_id=?", ("fresh-active",)).fetchone()[0], "running")
            reactor.close()
    def test_watchdog_interrupts_stale_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=path, watchdog_timeout_seconds=1))
            run_id = "stale-run"
            with reactor.store.transaction():
                from skynet.models import Budget, RunRecord
                from skynet.time import utc_now
                reactor.store.create_run(RunRecord(run_id, 1, RunStatus.RUNNING, utc_now(), Budget()))
                reactor.store.connection.execute("UPDATE runs SET heartbeat_at=? WHERE run_id=?", ("2000-01-01T00:00:00+00:00", run_id))
                state = reactor.store.state()
                state.active_run_id = run_id
                reactor.store.set_state(state)
            self.assertTrue(reactor.watchdog())
            self.assertEqual(reactor.store.connection.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()[0], "interrupted")
            reactor.close()
    def test_recovery_keeps_episode_out_of_agent_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            with reactor.store.transaction():
                state = reactor.store.state()
                state.active_run_id = "run-1"
                reactor.store.set_state(state)
                for index in range(40):
                    reactor.store.append_event("tool_result", {"output": "x" * 1000, "index": index}, "run-1")
            reactor.recover()
            recovery = reactor.store.state().next_plan["recovery"]
            self.assertNotIn("episode", recovery)
            self.assertEqual(recovery["episode_event_count"], 40)
            self.assertTrue(recovery["episode_snapshot_id"])
            reactor.close()
    def test_watchdog_does_not_write_while_reactor_run_lock_is_owned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            self.assertTrue(reactor._run_lock.acquire(blocking=False))
            try:
                self.assertFalse(reactor.watchdog())
            finally:
                reactor._run_lock.release()
                reactor.close()
    def test_watchdog_reads_stale_state_from_another_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            from skynet.models import Budget, RunRecord
            from skynet.time import utc_now
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                store.create_run(RunRecord("wd-run", 1, RunStatus.RUNNING, utc_now(), Budget()))
                state = store.state()
                state.active_run_id = "wd-run"
                store.set_state(state)
            future = (datetime.now(UTC) + timedelta(seconds=60)).isoformat().replace("+00:00", "Z")
            fresh = (datetime.now(UTC) - timedelta(seconds=60)).isoformat().replace("+00:00", "Z")
            seen: dict[str, object] = {}

            def read() -> None:
                seen["stale"] = store.stale_active_run(future)
                seen["fresh"] = store.stale_active_run(fresh)

            thread = threading.Thread(target=read)
            thread.start()
            thread.join(timeout=5)
            self.assertEqual(seen.get("stale"), "wd-run")
            self.assertIsNone(seen.get("fresh"))
            store.close()

    def test_stale_run_watchdog_signals_main_thread(self) -> None:
        from skynet.models import Budget, RunRecord
        from skynet.time import utc_now
        from skynet.watchdog import StaleRunWatchdog

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=path))
            with reactor.store.transaction():
                reactor.store.create_run(RunRecord("stale-run", 1, RunStatus.RUNNING, utc_now(), Budget()))
                reactor.store.connection.execute("UPDATE runs SET heartbeat_at=? WHERE run_id=?", ("2000-01-01T00:00:00+00:00", "stale-run"))
                state = reactor.store.state()
                state.active_run_id = "stale-run"
                reactor.store.set_state(state)
            watchdog = StaleRunWatchdog(reactor, interval_seconds=1, timeout_seconds=5)
            with patch("skynet.watchdog.time.sleep"), patch("skynet.watchdog.os.kill") as kill:
                watchdog._check_once()
            kill.assert_called_once_with(os.getpid(), signal.SIGALRM)
            reactor.close()

    def test_stale_run_watchdog_ignores_absent_or_changed_runs(self) -> None:
        from skynet.watchdog import StaleRunWatchdog

        reactor = MagicMock()
        watchdog = StaleRunWatchdog(reactor, interval_seconds=1, timeout_seconds=5)
        with patch("skynet.watchdog.time.sleep"), patch("skynet.watchdog.os.kill") as kill:
            reactor.store.stale_active_run.return_value = None
            watchdog._check_once()
            reactor.store.stale_active_run.side_effect = ["run-a", None]
            watchdog._check_once()
        kill.assert_not_called()

    def test_stale_run_watchdog_loop_survives_check_failures(self) -> None:
        from skynet.watchdog import StaleRunWatchdog

        watchdog = StaleRunWatchdog(MagicMock(), interval_seconds=1, timeout_seconds=5)
        with patch.object(watchdog, "_check_once", side_effect=RuntimeError("boom")) as check, patch.object(watchdog._stop, "wait", side_effect=[False, True]):
            watchdog.run()
        self.assertEqual(check.call_count, 1)

    def test_watchdog_signal_is_blocked_during_durable_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            deferred = {signal.SIGALRM, signal.SIGTERM, signal.SIGINT}
            with patch("skynet.reactor.signal.pthread_sigmask") as mask, reactor._defer_watchdog_signal():
                mask.assert_called_once_with(signal.SIG_BLOCK, deferred)
            self.assertEqual(mask.call_args_list[-1], call(signal.SIG_UNBLOCK, deferred))
            reactor.close()

    def test_watchdog_signal_is_not_blocked_on_worker_threads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            calls: list[object] = []

            def worker() -> None:
                with patch("skynet.reactor.signal.pthread_sigmask", side_effect=lambda *args: calls.append(args)), \
                     reactor._defer_watchdog_signal():
                    pass

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(10)
            self.assertFalse(thread.is_alive())
            self.assertEqual(calls, [])
            reactor.close()

    def test_backfill_missing_run_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.connection.execute(
                "INSERT INTO runs(run_id, attempt, status, started_at, budget) VALUES (?, 1, 'interrupted', '2026-01-01T00:00:00Z', '{}')",
                ("run-x",),
            )
            store.connection.commit()
            self.assertEqual(store.backfill_missing_run_results(), 1)
            row = store.connection.execute("SELECT status, failure FROM run_results WHERE run_id=?", ("run-x",)).fetchone()
            self.assertEqual(row["status"], "interrupted")
            self.assertEqual(row["failure"], "missing_result_backfilled")
            self.assertEqual(store.backfill_missing_run_results(), 0)
            store.close()

    def test_malformed_reboot_guard_is_quarantined_not_wedging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state / "reboot-guard.json").write_text("{not json", encoding="utf-8")
            guard = RebootGuard(state)
            result = guard.observe({"ok": True})
            self.assertTrue(result["quarantined"])
            self.assertFalse(result["active"])
            self.assertFalse((state / "reboot-guard.json").exists())
            quarantined = list((state / "quarantine").glob("reboot-guard-*.json"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(quarantined[0].read_text(encoding="utf-8"), "{not json")
            # The next cycle proceeds instead of raising forever.
            self.assertFalse(guard.observe({"ok": True})["active"])

    def test_stale_reboot_window_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            guard = RebootGuard(state, max_window_age_seconds=60.0)
            started = (datetime.now(UTC) - timedelta(days=2)).isoformat().replace("+00:00", "Z")
            guard._atomic_write({
                "commit": "abcdef1",
                "rollback_commit": "abcdef2",
                "proposal_id": "proposal-x",
                "started_at": started,
                "healthy_cycles": 0,
                "failed": False,
            })
            result = guard.observe({"ok": True})
            self.assertTrue(result["quarantined"])
            self.assertEqual(result["proposal_id"], "proposal-x")
            self.assertFalse((state / "reboot-guard.json").exists())

    def test_fresh_reboot_window_still_advances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            guard = RebootGuard(state, max_window_age_seconds=60.0)
            guard._atomic_write({
                "commit": "abcdef1",
                "rollback_commit": "abcdef2",
                "proposal_id": "proposal-x",
                "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "healthy_cycles": 0,
                "failed": False,
            })
            result = guard.observe({"ok": True})
            self.assertTrue(result["active"])
            self.assertEqual(result["healthy_cycles"], 1)
            self.assertFalse(result.get("quarantined", False))

    def test_open_reboot_window_names_the_promoted_commit(self) -> None:
        """An un-finished window must not be anonymous.

        Measured on the live event log: 81 promotions carried
        restart_after_checkpoint, 55 of their windows produced no
        reboot_observation row at all, and only 2 of the 32 rows that exist
        named the promoted commit -- both on the terminal branch. The
        still-open branch returned no commit and no proposal_id, so
        ``reboot_observation`` (payload = {health, result}) could not be
        attributed to the promotion it was judging. This pins the identity
        travelling with every in-window observation.
        """
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            guard = RebootGuard(state, health_window_cycles=3)
            guard._atomic_write({
                "commit": "abcdef1",
                "rollback_commit": "abcdef2",
                "proposal_id": "proposal-x",
                "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "healthy_cycles": 0,
                "active": True,
                "failed": False,
            })
            first = guard.observe({"ok": True})
            self.assertTrue(first["active"])
            self.assertEqual(first["commit"], "abcdef1")
            self.assertEqual(first["proposal_id"], "proposal-x")
            second = guard.observe({"ok": True})
            self.assertTrue(second["active"])
            self.assertEqual(second["healthy_cycles"], 2)
            self.assertEqual(second["commit"], "abcdef1")
            terminal = guard.observe({"ok": True})
            self.assertTrue(terminal["completed"])
            self.assertEqual(terminal["commit"], "abcdef1")
            self.assertEqual(terminal["proposal_id"], "proposal-x")
            # A closed window still answers with the same identity.
            closed = guard.observe({"ok": True})
            self.assertFalse(closed["active"])
            self.assertEqual(closed["commit"], "abcdef1")
            self.assertEqual(closed["proposal_id"], "proposal-x")

    def test_open_reboot_window_reports_its_size_not_only_the_count(self) -> None:
        """healthy_cycles alone cannot be read as "1 of N" without N.

        The live rows record healthy_cycles 1 or 2 and never 3, and the window
        size (health_window_cycles, default 3) was in no durable row, so
        "the window is still open" and "the window silently shrank" were the
        same observation. This pins the size travelling with the count.
        """
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            guard = RebootGuard(state, health_window_cycles=2)
            guard._atomic_write({
                "commit": "abcdef1",
                "rollback_commit": "abcdef2",
                "proposal_id": "proposal-x",
                "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "healthy_cycles": 0,
                "active": True,
                "failed": False,
            })
            opening = guard.observe({"ok": True})
            self.assertTrue(opening["active"])
            self.assertEqual((opening["healthy_cycles"], opening["window_cycles"]), (1, 2))
            closing = guard.observe({"ok": True})
            self.assertTrue(closing["completed"])
            self.assertEqual((closing["healthy_cycles"], closing["window_cycles"]), (2, 2))

    def test_begin_marks_the_window_active_in_the_durable_bytes(self) -> None:
        """Reactor._reboot_outcome reads the guard file, not the return value.

        Its ``in_progress`` branch tested ``guard.get("active")``, but
        ``begin`` never wrote that key -- only a later ``observe`` did, and it
        writes it only when the window closes. So an open window was invisible
        to the start envelope and the branch was unreachable. Measured live:
        18 of 108 run envelopes carried a reboot_outcome observation, and all
        18 read ``accepted``; none ever read ``in_progress``.
        """
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            request = state / "reboot-request.json"
            request.write_text(
                json.dumps({"commit": "abcdef1", "rollback_commit": "abcdef2", "proposal_id": "proposal-x"}),
                encoding="utf-8",
            )
            guard = RebootGuard(state)
            self.assertIsNotNone(guard.begin(request))
            persisted = json.loads((state / "reboot-guard.json").read_text(encoding="utf-8"))
            self.assertTrue(persisted["active"])
            self.assertEqual(persisted["commit"], "abcdef1")
            self.assertEqual(persisted["proposal_id"], "proposal-x")
            self.assertNotIn("completed_at", persisted)


class RollbackFailureTests(unittest.TestCase):
    def _guard_with_failed_health(self, state: Path, rollback):
        from skynet.recovery import RebootGuard

        guard = RebootGuard(state, max_window_age_seconds=3600.0)
        guard._atomic_write({
            "commit": "abcdef1",
            "rollback_commit": "abcdef2",
            "proposal_id": "proposal-x",
            "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "healthy_cycles": 0,
            "failed": False,
        })
        return guard.observe({"ok": False}, rollback)

    def test_failed_rollback_is_recorded_and_does_not_claim_rolled_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)

            def failing(commit: str) -> None:
                del commit
                raise RuntimeError("git reset refused")

            with self.assertRaises(RuntimeError):
                self._guard_with_failed_health(state, failing)
            payload = json.loads((state / "reboot-guard.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["rollback_error"], "git reset refused")
            self.assertEqual(payload["rollback_attempts"], 1)
            # Claiming rolled_back while the broken code still runs would be a
            # false durable record; the window stays open and is bounded by age.
            self.assertFalse(payload.get("rolled_back", False))

    def test_successful_rollback_closes_the_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            seen: list[str] = []

            def ok(commit: str) -> None:
                seen.append(commit)

            result = self._guard_with_failed_health(state, ok)
            self.assertEqual(seen, ["abcdef2"])
            self.assertTrue(result["rolled_back"])
            payload = json.loads((state / "reboot-guard.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["rolled_back"])
            self.assertNotIn("rollback_error", payload)
