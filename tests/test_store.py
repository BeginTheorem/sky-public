"""Split from the former monolithic CoreTests suite."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

from skynet import metrics
from skynet.models import RunStatus
from skynet.outbox import deliver_to_http
from skynet.planner import PlannerCandidate
from skynet.store import StateStore


def _candidate(goal_id: str, task_id: str) -> PlannerCandidate:
    """A minimal planner candidate for `record_planner_decision`."""
    return PlannerCandidate(
        workstream_id=goal_id,
        title="streak candidate",
        task_id=task_id,
        score=0.5,
        criticality=0.25,
        novelty=0.5,
        repetition_penalty=0.0,
        reason="test",
    )


class CoreTests(unittest.TestCase):
    def test_reset_short_memory_preserves_durable_state_and_audits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("durable goal")
            store.add_task("durable task", goal_id)
            store.consolidate("run-1", [{"kind": "fact", "content": "durable memory", "confidence": 1.0}])
            state = store.state()
            state.next_plan = {
                "initial_prompt": "continue old line",
                "previous_outcome": {"report": "old report"},
                "next": "old next step",
                "recovery": {"reason": "loop"},
            }
            state.retry_count = 4
            state.next_wake_at = "2099-01-01T00:00:00+00:00"
            store.set_state(state)

            details = store.reset_short_memory()

            self.assertEqual(store.state().next_plan, {})
            self.assertEqual(store.state().retry_count, 0)
            self.assertIsNotNone(store.state().next_wake_at)
            self.assertIsNotNone(store.connection.execute("SELECT goal_id FROM goals WHERE goal_id=?", (goal_id,)).fetchone())
            self.assertEqual(store.search_memories("durable memory")[0]["content"], "durable memory")
            event = store.connection.execute(
                "SELECT kind, payload FROM event_log WHERE kind='short_memory_reset' ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            self.assertIsNotNone(event)
            self.assertEqual(json.loads(event["payload"])["reason"], "manual_cli_reset")
            self.assertEqual(details["generation"], store.state().generation)
            store.close()
    def test_reset_short_memory_rejects_active_run_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            state = store.state()
            state.active_run_id = "active-run"
            store.set_state(state)
            with self.assertRaisesRegex(RuntimeError, "active run"):
                store.reset_short_memory()
            store.close()
    def test_reset_short_memory_interrupts_stopped_run_with_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            from skynet.models import Budget, RunRecord
            from skynet.time import utc_now
            store.create_run(RunRecord("stopped-run", 1, RunStatus.RUNNING, utc_now(), Budget()))
            state = store.state()
            state.active_run_id = "stopped-run"
            store.set_state(state)
            details = store.reset_short_memory(allow_interrupted_run=True)
            self.assertEqual(details["interrupted_run_id"], "stopped-run")
            self.assertIsNone(store.state().active_run_id)
            run = store.connection.execute("SELECT status FROM runs WHERE run_id=?", ("stopped-run",)).fetchone()
            self.assertIsNotNone(run)
            self.assertEqual(run["status"], RunStatus.INTERRUPTED.value)
            store.close()
    def test_goal_priority_is_stored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("high priority", priority=3.0)
            value = store.connection.execute("SELECT priority FROM goals WHERE goal_id=?", (goal_id,)).fetchone()[0]
            self.assertEqual(value, 3.0)
            store.close()
    def test_outbox_expired_lease_can_be_reclaimed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                message_id = store.add_outbox("test", {"value": 1})
            self.assertEqual(store.claim_outbox(lease_seconds=300)[0]["message_id"], message_id)
            self.assertEqual(store.claim_outbox(lease_seconds=0), [])
            store.connection.execute("UPDATE outbox SET claimed_at='2000-01-01T00:00:00+00:00' WHERE message_id=?", (message_id,))
            self.assertEqual(store.claim_outbox(lease_seconds=300)[0]["message_id"], message_id)
            store.close()
    def test_run_result_commit_reconciles_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                first = store.commit_run_result("run-1", RunStatus.COMPLETED, "done", 2, 3, "")
                second = store.commit_run_result("run-1", RunStatus.FAILED, "other", 9, 9, "bad")
            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["status"], "failed")
            result = store.run_result("run-1")
            self.assertIsNotNone(result)
            self.assertEqual(cast(dict[str, object], result)["report"], "other")
            store.close()
    def test_outbox_http_delivery_sends_idempotency_key(self) -> None:
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append((self.headers["Idempotency-Key"], self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(204)
                self.end_headers()

            def log_message(self, format, *args):
                del format, args
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                store = StateStore(Path(directory) / "state.sqlite3")
                message_id = store.add_outbox("test", {"text": "hello"})
                delivered = deliver_to_http(store, f"http://127.0.0.1:{server.server_port}")
                self.assertEqual(delivered, 1)
                self.assertEqual(received[0][0], message_id)
                self.assertEqual(store.pending_outbox(), [])
                store.close()
        finally:
            server.shutdown()
            server.server_close()
    def test_numbered_passes_share_planner_fingerprints(self) -> None:
        from skynet.planner import hypothesis_fingerprint, structural_fingerprint

        first = "Run bounded self-improvement pass 14 on MCP recovery"
        second = "Run bounded self-improvement pass 23 on MCP recovery"
        self.assertEqual(
            hypothesis_fingerprint(area=first, problem=first, expected_behavior=first),
            hypothesis_fingerprint(area=second, problem=second, expected_behavior=second),
        )
        self.assertEqual(
            structural_fingerprint(area=first, target=first, behavior_kind="task"),
            structural_fingerprint(area=second, target=second, behavior_kind="task"),
        )
    def test_legacy_task_fingerprints_are_backfilled_with_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            with store.transaction():
                store.connection.execute(
                    "UPDATE tasks SET hypothesis_fingerprint=NULL, structural_fingerprint=NULL, status='completed' "
                    "WHERE task_id=?",
                    (store.add_task("Run bounded self-improvement pass 1 on planner migration"),),
                )
            store.close()

            reopened = StateStore(path)
            task = reopened.connection.execute("SELECT hypothesis_fingerprint FROM tasks").fetchone()
            hypothesis = reopened.connection.execute("SELECT status FROM hypotheses WHERE fingerprint=?", (task[0],)).fetchone()
            self.assertIsNotNone(task[0])
            self.assertEqual(hypothesis[0], "completed")
            reopened.close()
    def test_planner_reset_preserves_durable_history_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("durable portfolio", priority=4)
            task_id = store.add_task("bounded task", goal_id)
            store.append_event("evidence", {"task_id": task_id})
            first = store.reset_planning_context(reason="self_improvement_accepted", proposal_id="p1", commit="abc1234")
            second = store.reset_planning_context(reason="self_improvement_accepted", proposal_id="p1", commit="abc1234")
            self.assertFalse(first["idempotent"])
            self.assertTrue(second["idempotent"])
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0], 1)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 1)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='evidence'").fetchone()[0], 1)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='planner_reset'").fetchone()[0], 1)
            store.close()
    def test_store_deduplicates_inbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                store.add_inbox_event("same", "test", {"x": 1})
                store.add_inbox_event("same", "test", {"x": 2})
            count = store.connection.execute("SELECT COUNT(*) FROM inbox").fetchone()[0]
            self.assertEqual(count, 1)
            store.close()
    def test_read_only_store_does_not_migrate_or_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            writable = StateStore(path)
            writable.add_goal("inspect")
            writable.close()
            readonly = StateStore(path, read_only=True)
            self.assertEqual(readonly.connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0], 1)
            readonly.close()
    def test_schema_migration_journal_is_versioned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            row = store.connection.execute("SELECT version, name FROM schema_migrations ORDER BY version DESC").fetchone()
            self.assertEqual((row[0], row[1]), (12, "run-progress"))
            store.close()

    def test_schema_v4_upgrade_adds_alerts_area_and_pinned(self) -> None:
        """A v3 database must gain the v4 columns and tables without data loss."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            goal_id = store.add_goal("upgrade me", priority=1.0)
            task_id = store.add_task("legacy task", goal_id)
            store.consolidate("run-legacy", [{"kind": "fact", "content": "legacy memory", "confidence": 0.7}])
            # Simulate a genuine v3 database: no v4 columns, table, or indexes.
            store.connection.execute("UPDATE schema_migrations SET version=3 WHERE version=4")
            store.connection.execute("DROP INDEX IF EXISTS idx_tasks_area")
            store.connection.execute("DROP INDEX IF EXISTS idx_memories_pinned")
            store.connection.execute("DROP INDEX IF EXISTS idx_alerts_pending")
            store.connection.execute("ALTER TABLE tasks DROP COLUMN area")
            store.connection.execute("ALTER TABLE memories DROP COLUMN pinned")
            store.connection.execute("DROP TABLE alerts")
            store.close()

            upgraded = StateStore(path)
            columns = {row["name"] for row in upgraded.connection.execute("PRAGMA table_info(tasks)")}
            self.assertIn("area", columns)
            memory_columns = {row["name"] for row in upgraded.connection.execute("PRAGMA table_info(memories)")}
            self.assertIn("pinned", memory_columns)
            self.assertEqual(upgraded.connection.execute("SELECT COUNT(*) FROM alerts").fetchone()[0], 0)
            self.assertEqual(upgraded.connection.execute("SELECT area FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0], "general")
            self.assertEqual(upgraded.connection.execute("SELECT content FROM memories").fetchone()[0], "legacy memory")
            versions = [row[0] for row in upgraded.connection.execute("SELECT version FROM schema_migrations ORDER BY version")]
            self.assertEqual(versions[-1], 12)
            # v11 names the decay clock: the column must exist after an upgrade
            # even though an older database already had every earlier column.
            self.assertIn("decayed_at", {row["name"] for row in upgraded.connection.execute("PRAGMA table_info(memories)")})
            upgraded.close()

    def test_connection_sets_busy_timeout_and_wal_synchronous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            self.assertEqual(store.connection.execute("PRAGMA busy_timeout").fetchone()[0], 5000)
            self.assertEqual(str(store.connection.execute("PRAGMA synchronous").fetchone()[0]).lower(), "1")
            self.assertEqual(str(store.connection.execute("PRAGMA journal_mode").fetchone()[0]).lower(), "wal")
            store.close()

    def test_confidence_decay_fades_stale_memories_and_keeps_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.consolidate("run-1", [
                {"kind": "fact", "content": "stale guess", "confidence": 0.9},
                {"kind": "fact", "content": "pinned truth", "confidence": 0.5},
                {"kind": "fact", "content": "almost gone", "confidence": 0.11},
            ])
            pinned = store.connection.execute("SELECT memory_id FROM memories WHERE content='pinned truth'").fetchone()[0]
            store.set_memory_pinned(str(pinned))
            store.connection.execute("UPDATE memories SET updated_at='2020-01-01T00:00:00Z'")
            result = store.decay_memory_confidence(factor=0.5, stale_days=7.0, drop_below=0.1)
            self.assertEqual(result["faded"], 2, "pinned memories are not faded")
            self.assertEqual(result["dropped"], 1, "the weakest unpinned memory is removed")
            rows = {row["content"]: row["confidence"] for row in store.connection.execute("SELECT content, confidence FROM memories")}
            self.assertAlmostEqual(rows["stale guess"], 0.855, places=3)
            self.assertEqual(rows["pinned truth"], 0.5)
            self.assertNotIn("almost gone", rows)
            store.close()

    def test_importance_weighted_decay_slows_high_confidence_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.consolidate("run-1", [
                {"kind": "fact", "content": "well evidenced", "confidence": 0.9},
                {"kind": "fact", "content": "weak guess", "confidence": 0.3},
            ])
            store.connection.execute("UPDATE memories SET updated_at='2020-01-01T00:00:00Z'")
            store.decay_memory_confidence(factor=0.5, stale_days=7.0, drop_below=0.0)
            rows = {row["content"]: row["confidence"] for row in store.connection.execute("SELECT content, confidence FROM memories")}
            self.assertAlmostEqual(rows["well evidenced"], 0.855, places=3)
            self.assertAlmostEqual(rows["weak guess"], 0.195, places=3)
            self.assertGreater(rows["well evidenced"], rows["weak guess"], "high confidence must decay slower")
            # A second pass inside the same stale window must be a no-op.
            store.decay_memory_confidence(factor=0.5, stale_days=7.0, drop_below=0.0)
            rows_after = {row["content"]: row["confidence"] for row in store.connection.execute("SELECT content, confidence FROM memories")}
            self.assertAlmostEqual(rows_after["well evidenced"], 0.855, places=3)
            self.assertAlmostEqual(rows_after["weak guess"], 0.195, places=3)
            store.close()

    def test_memory_validity_api_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            memory_id = store.remember_memory("alpha durable fact", kind="fact", confidence=0.4, evidence="run-1")
            self.assertEqual(store.remember_memory("alpha durable fact", kind="fact", confidence=0.8), memory_id)
            row = store.connection.execute("SELECT confidence, status, valid_from FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
            self.assertAlmostEqual(row["confidence"], 0.8, places=3)
            self.assertEqual(row["status"], "active")
            self.assertTrue(row["valid_from"])
            corrected = store.correct_memory(memory_id, content="alpha corrected fact", evidence="run-2")
            old = store.connection.execute("SELECT status, superseded_by, valid_to FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
            self.assertEqual(old["status"], "superseded")
            self.assertEqual(old["superseded_by"], corrected)
            self.assertTrue(old["valid_to"])
            self.assertTrue(store.forget_memory(corrected))
            self.assertFalse(store.forget_memory(corrected))
            with self.assertRaises(ValueError):
                store.correct_memory("missing-id", content="x", evidence="y")
            store.close()

    def test_a_memory_cannot_supersede_itself(self) -> None:
        # Correcting a memory to the same content upserts the same row, so the
        # old and new ids coincide; marking it superseded would silently drop
        # the corrected fact from search.
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            memory_id = store.remember_memory("same text", kind="fact")
            corrected = store.correct_memory(memory_id, content="same text", evidence="run-2")
            self.assertEqual(corrected, memory_id)
            row = store.connection.execute("SELECT status FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
            self.assertEqual(row["status"], "active")
            self.assertTrue(store.search_memories("same text"))
            store.close()

    def test_event_log_retention_keeps_the_safety_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.append_event("planner_decision", {"routine": True})
            store.append_event("provider_lockout", {"consecutive_failures": 3})
            store.append_event("task_gave_up", {"task_id": "t"})
            # Backdate everything beyond the window.
            store.connection.execute("UPDATE event_log SET created_at='2020-01-01T00:00:00Z'")
            self.assertEqual(store.prune_event_log(0), 0, "a disabled window must be a no-op")
            removed = store.prune_event_log(30)
            self.assertEqual(removed, 1, "only the routine event is pruned")
            kinds = {row[0] for row in store.connection.execute("SELECT kind FROM event_log")}
            self.assertEqual(kinds, {"provider_lockout", "task_gave_up"})
            store.close()

    def test_event_log_retention_keeps_the_planner_and_restart_post_mortem(self) -> None:
        # "The planner crashed" and "there was genuinely no work" leave the same
        # empty portfolio, so the kinds that separate them must survive retention;
        # a restart request is the row that makes restarts observable at all.
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            protected = {
                "restart_requested",
                "planner_invalid_response",
                "planner_provider_error",
                "planner_fallback_capped",
                "judge_health_degraded",
            }
            for kind in sorted(protected):
                store.append_event(kind, {"evidence": kind})
            store.append_event("planner_decision", {"routine": True})
            store.connection.execute("UPDATE event_log SET created_at='2020-01-01T00:00:00Z'")
            removed = store.prune_event_log(30)
            self.assertEqual(removed, 1, "only the routine event is pruned")
            kinds = {row[0] for row in store.connection.execute("SELECT kind FROM event_log")}
            self.assertEqual(kinds, protected)
            store.close()

    def test_hot_query_paths_are_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            names = {row[0] for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='index'")}
            for expected in (
                "idx_event_log_kind",
                "idx_outbox_pending",
                "idx_inbox_pending",
                "idx_hypotheses_structural",
                "idx_proposals_fingerprint",
                "idx_tasks_area",
                "idx_memories_pinned",
                "idx_alerts_pending",
            ):
                self.assertIn(expected, names)
            store.close()

    def test_raise_alert_deduplicates_inside_the_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            first = store.raise_alert("provider_lockout", {"providers": ["ollama"]}, severity="critical")
            second = store.raise_alert("provider_lockout", {"providers": ["ollama", "openrouter"]}, severity="critical")
            self.assertTrue(first["new"])
            self.assertFalse(second["new"])
            self.assertEqual(second["occurrences"], 2)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM alerts").fetchone()[0], 1)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM outbox WHERE kind='alert'").fetchone()[0], 1)
            events = [row[0] for row in store.connection.execute("SELECT kind FROM event_log")]
            self.assertEqual(events.count("alert_raised"), 1)
            pending = store.pending_alerts()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["severity"], "critical")
            self.assertEqual(pending[0]["payload"]["providers"], ["ollama", "openrouter"])
            store.mark_alert_delivered(first["alert_id"], "telegram")
            self.assertEqual(store.pending_alerts(), [])
            collapsed = store.raise_alert("provider_lockout", {"providers": []}, dedup_window_seconds=3600.0)
            self.assertFalse(collapsed["new"], "a delivered alert inside the window must not re-arm")
            store.close()

    def test_raise_alert_rearms_after_the_dedup_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            first = store.raise_alert("disk_low", {"free_gb": 4})
            store.mark_alert_delivered(first["alert_id"], "telegram")
            rearmed = store.raise_alert("disk_low", {"free_gb": 2}, dedup_window_seconds=0.0)
            self.assertTrue(rearmed["new"])
            self.assertEqual(rearmed["occurrences"], 1)
            self.assertEqual(len(store.pending_alerts()), 1)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM outbox WHERE kind='alert'").fetchone()[0], 2)
            store.close()

    def test_outbox_dead_letters_after_the_attempt_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.add_outbox("agent_response", {"run_id": "r1"})
            state = ""
            for attempt in range(5):
                claimed = store.claim_outbox(limit=1, lease_seconds=0.0)
                self.assertEqual(len(claimed), 1, f"attempt {attempt} must be claimable")
                state = store.mark_outbox_failed(claimed[0]["message_id"], "telegram 429", max_attempts=5)
            self.assertEqual(state, "dead")
            self.assertEqual(store.claim_outbox(limit=5, lease_seconds=0.0), [])
            row = store.connection.execute("SELECT delivery_state, attempts, last_error FROM outbox").fetchone()
            self.assertEqual((row[0], row[1], row[2]), ("dead", 5, "telegram 429"))
            store.close()

    def test_selected_task_streak_counts_consecutive_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("streak", priority=1.0)
            task_id = store.add_task("streak task", goal_id)
            other = store.add_task("other task", goal_id)
            self.assertEqual(store.selected_task_streak(task_id), 0)
            for _ in range(3):
                store.record_planner_decision([], _candidate(goal_id, task_id))
            self.assertEqual(store.selected_task_streak(task_id), 3)
            store.record_planner_decision([], _candidate(goal_id, other))
            self.assertEqual(store.selected_task_streak(task_id), 0)
            self.assertEqual(store.selected_task_streak(other), 1)
            store.close()

    def test_pinned_memories_are_returned_without_a_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.consolidate("run-pin", [
                {"kind": "fact", "content": "the test runner is .venv/bin/python -m pytest -q", "confidence": 0.95},
                {"kind": "observation", "content": "an ordinary observation", "confidence": 0.4},
            ])
            rows = store.connection.execute("SELECT memory_id, content FROM memories ORDER BY confidence DESC").fetchall()
            self.assertTrue(store.set_memory_pinned(rows[0]["memory_id"]))
            self.assertFalse(store.set_memory_pinned("missing-id"))
            pinned = store.pinned_memories()
            self.assertEqual([item["content"] for item in pinned], ["the test runner is .venv/bin/python -m pytest -q"])
            store.close()

    def test_runtime_log_rotates_and_survives_write_errors(self) -> None:
        from skynet.runtime_log import RuntimeLog

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.jsonl"
            log = RuntimeLog(path, max_bytes=200, backups=2)
            for index in range(12):
                log.write("event", {"index": index, "payload": "x" * 60})
            self.assertTrue(path.exists())
            self.assertTrue((path.parent / "runtime.jsonl.1").exists())
            self.assertGreater(path.parent.joinpath("runtime.jsonl.1").stat().st_size, 0)
            with patch("pathlib.Path.open", side_effect=OSError("disk full")):
                log.write("event", {"index": "boom"})

    def test_outbox_failed_delivery_is_retried_with_error_recorded(self) -> None:
        from skynet.outbox import deliver_to_http, deliver_to_jsonl

        class FailingHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(500)
                self.end_headers()

            def log_message(self, format, *args):
                return

        server = HTTPServer(("127.0.0.1", 0), FailingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                store = StateStore(Path(directory) / "state.sqlite3")
                store.add_outbox("event", {"n": 1}, "m1")
                with patch("skynet.outbox.sleep"):
                    delivered = deliver_to_http(store, f"http://127.0.0.1:{server.server_port}/", retries=1, backoff_seconds=0.0)
                self.assertEqual(delivered, 0)
                row = store.connection.execute("SELECT delivery_state, attempts, last_error FROM outbox WHERE message_id='m1'").fetchone()
                self.assertEqual(row["delivery_state"], "pending")
                self.assertEqual(row["attempts"], 1)
                self.assertTrue(row["last_error"])
                stream = MagicMock()
                stream.__enter__ = MagicMock(return_value=stream)
                stream.__exit__ = MagicMock(return_value=False)
                stream.write.side_effect = OSError("disk full")
                with patch("skynet.outbox.Path.open", return_value=stream):
                    delivered = deliver_to_jsonl(store, Path(directory) / "out.jsonl")
                self.assertEqual(delivered, 0)
                store.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_prune_run_history_keeps_the_newest_runs_and_the_audit_trail(self) -> None:
        from skynet.models import Budget, RunRecord
        from skynet.time import utc_now

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.add_goal("keep me", priority=1.0)
            for index in range(3):
                run_id = f"run-{index}"
                store.create_run(RunRecord(run_id, 1, RunStatus.COMPLETED, utc_now(), Budget()))
                store.append_transcript("react_history", {"messages": [index]}, run_id)
                store.append_event("tool_call", {"call_id": run_id, "tool_name": "bash", "arguments": {}}, run_id)
                store.snapshot_episode(run_id)

            pruned = store.prune_run_history(2)
            self.assertEqual(pruned["transcript"], 1)
            self.assertEqual(pruned["episodes"], 1)
            held = [row[0] for row in store.connection.execute("SELECT DISTINCT run_id FROM transcript WHERE run_id IS NOT NULL").fetchall()]
            self.assertEqual(sorted(held), ["run-1", "run-2"])
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM episode_snapshots WHERE run_id='run-0'").fetchone()[0], 0)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM event_log WHERE run_id='run-0'").fetchone()[0], 1)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0], 1)
            store.close()

    def test_capability_effect_referenced_only_by_retained_transcript_survives(self) -> None:
        # The folded tool result keeps its address in two retained places: the
        # tool_result event and the react_history tool message, whose `content` is
        # itself a JSON string. Transcript pruning is by run count, not days, so a
        # retained transcript can outlive the 30-day effect window and the row it
        # names must not be deleted.
        from skynet.models import Budget, RunRecord
        from skynet.time import utc_now

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.create_run(RunRecord("run-1", 1, RunStatus.COMPLETED, utc_now(), Budget()))
            key = "run-1:0:call-folded"
            store.record_effect(key, "bash", "hash", {"ok": True, "blob": "x" * 5000}, "applied")
            folded = json.dumps({
                "ok": True,
                "truncated": True,
                "preview": "x" * 10,
                "effect_key": key,
                "full_result_in": "capability_effects.idempotency_key",
            })
            store.append_transcript("react_history", {"messages": [{"role": "tool", "tool_call_id": "call-folded", "content": folded}]}, "run-1")
            store.connection.execute("UPDATE capability_effects SET created_at='2020-01-01T00:00:00Z'")
            removed = store.prune_capability_effects(30)
            self.assertEqual(removed, 0, "the retained transcript still names the row")
            self.assertIsNotNone(store.effect(key), "the address in the retained transcript must resolve")
            store.close()

    def test_capability_effect_not_referenced_by_retained_history_is_pruned(self) -> None:
        # The guard must not silently disable pruning: an aged row nothing retained
        # points at is still reclaimed.
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            key = "run-1:0:call-orphan"
            store.record_effect(key, "bash", "hash", {"ok": True}, "applied")
            store.connection.execute("UPDATE capability_effects SET created_at='2020-01-01T00:00:00Z'")
            removed = store.prune_capability_effects(30)
            self.assertEqual(removed, 1)
            self.assertIsNone(store.effect(key))
            store.close()

    def test_capability_effect_pin_is_released_when_its_transcript_is_pruned(self) -> None:
        # The pin is bounded: once prune_run_history drops the run holding the
        # pointer, the next effect pass reclaims the row.
        from skynet.models import Budget, RunRecord
        from skynet.time import utc_now

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            for index in range(3):
                store.create_run(RunRecord(f"run-{index}", 1, RunStatus.COMPLETED, utc_now(), Budget()))
            key = "run-0:0:call-folded"
            store.record_effect(key, "bash", "hash", {"ok": True}, "applied")
            folded = json.dumps({"ok": True, "truncated": True, "preview": "x", "effect_key": key, "full_result_in": "capability_effects.idempotency_key"})
            store.append_transcript("react_history", {"messages": [{"role": "tool", "tool_call_id": "call-folded", "content": folded}]}, "run-0")
            store.connection.execute("UPDATE capability_effects SET created_at='2020-01-01T00:00:00Z'")
            self.assertEqual(store.prune_capability_effects(30), 0, "pinned while the run-0 transcript is retained")
            store.prune_run_history(2)
            self.assertEqual(store.prune_capability_effects(30), 1, "the pin is released with the transcript")
            store.close()

    def test_repair_status_consistency_cancels_unselectable_pending_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("g", priority=1.0)
            task_id = store.add_task("dead row", goal_id)
            fingerprint = store.connection.execute("SELECT hypothesis_fingerprint FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0]
            store.connection.execute("UPDATE hypotheses SET status='completed' WHERE fingerprint=?", (fingerprint,))
            store.connection.commit()
            self.assertEqual(store.repair_status_consistency()["tasks"], 1)
            self.assertEqual(store.connection.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0], "cancelled")
            self.assertEqual(store.repair_status_consistency(), {"hypotheses": 0, "tasks": 0})
            store.close()

    def test_repair_status_consistency_leaves_a_cancelled_ready_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("g", priority=1.0)
            task_id = store.add_task("cancelled", goal_id)
            store.connection.execute("UPDATE tasks SET status='cancelled' WHERE task_id=?", (task_id,))
            store.connection.commit()
            self.assertEqual(store.repair_status_consistency(), {"hypotheses": 0, "tasks": 0})
            fingerprint = store.connection.execute("SELECT hypothesis_fingerprint FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0]
            self.assertEqual(store.connection.execute("SELECT status FROM hypotheses WHERE fingerprint=?", (fingerprint,)).fetchone()[0], "ready")
            store.close()

    def test_repair_status_consistency_excludes_safety_net_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("g", priority=1.0)
            task_id = store.add_task("safety net", goal_id)
            fingerprint = store.connection.execute("SELECT hypothesis_fingerprint FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0]
            store.connection.execute("UPDATE hypotheses SET status='completed' WHERE fingerprint=?", (fingerprint,))
            store.connection.commit()
            self.assertEqual(store.repair_status_consistency(exclude_fingerprints=(fingerprint,))["tasks"], 0)
            self.assertEqual(store.connection.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0], "pending")
            # Without the exclusion the same row is repair-eligible, proving the
            # exclusion (not the row shape) is what protected it.
            self.assertEqual(store.repair_status_consistency()["tasks"], 1)
            self.assertEqual(store.connection.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0], "cancelled")
            store.close()

    def test_task_update_without_status_keeps_current_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("g", priority=1.0)
            task_id = store.add_task("done", goal_id)
            store.connection.execute("UPDATE tasks SET status='completed' WHERE task_id=?", (task_id,))
            store.connection.commit()
            store.apply_task_updates([{"task_id": task_id, "outcome": "finished"}], run_id="run-1")
            self.assertEqual(store.connection.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0], "completed")
            store.close()

    def test_outbox_delivery_drains_pending(self) -> None:
        from skynet.outbox import deliver_to_jsonl
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.add_outbox("run_finished", {"run_id": "r1"})
            store.add_outbox("run_finished", {"run_id": "r2"})
            target = Path(directory) / "outbox.jsonl"
            self.assertEqual(deliver_to_jsonl(store, target), 2)
            self.assertEqual(store.pending_outbox(), [])
            self.assertEqual(len(target.read_text(encoding="utf-8").strip().splitlines()), 2)
            store.close()

    def test_outcome_tokens_separate_completed_from_waste(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.commit_run_result("done-1", RunStatus.COMPLETED, "ok", 1, 100, "")
            store.commit_run_result("done-2", RunStatus.COMPLETED, "ok", 1, 300, "")
            store.commit_run_result("failed", RunStatus.FAILED, "no", 1, 5000, "boom")
            store.commit_run_result("recover", RunStatus.NEEDS_RECOVERY, "?", 1, 2000, "stuck")
            outcomes = metrics.outcome_mix(store.connection, "1970-01-01T00:00:00Z")
            self.assertEqual(outcomes["tokens_total"], 7400)
            self.assertEqual(outcomes["tokens_per_completed_run"], 200, "only completed runs pay")
            self.assertEqual(outcomes["tokens_wasted"], 7000, "failed and recovering runs are waste")
            store.close()

    def test_livelock_streak_reports_longest_run_not_selection_sum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            for task_id in ("A", "A", "B", "A", "A", "C", "C", "C"):
                store.record_planner_decision([], _candidate("goal-1", task_id))
            streaks = metrics.livelock_streaks(store.connection, "1970-01-01T00:00:00Z", threshold=3)
            self.assertEqual(streaks, [{"task_id": "C", "streak": 3}], "A was selected 4 times but never 3 in a row")
            store.close()

    def test_inflight_outbox_lease_survives_a_second_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            with patch("skynet.store._INFLIGHT_LEASES_RECOVERED", False):
                store = StateStore(path)
                message_id = store.add_outbox("test", {"value": 1})
                store.claim_outbox(lease_seconds=300.0)
                store.close()
                second = StateStore(path)
                state = second.connection.execute(
                    "SELECT delivery_state FROM outbox WHERE message_id=?", (message_id,)
                ).fetchone()[0]
                self.assertEqual(state, "delivering", "a second open must not steal a live lease")
                self.assertEqual(second.recover_inflight_outbox(), 1)
                recovered = second.connection.execute(
                    "SELECT delivery_state FROM outbox WHERE message_id=?", (message_id,)
                ).fetchone()[0]
                self.assertEqual(recovered, "pending")
                second.close()

    def test_v6_planner_resets_column_is_dropped_on_upgrade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            store.connection.execute("ALTER TABLE planner_resets ADD COLUMN previous_context TEXT NOT NULL DEFAULT ''")
            store.close()
            upgraded = StateStore(path)
            columns = {row["name"] for row in upgraded.connection.execute("PRAGMA table_info(planner_resets)")}
            self.assertNotIn("previous_context", columns)
            upgraded.reset_planning_context(reason="upgrade-check")
            upgraded.close()

    def test_unexpired_outbox_lease_is_not_reclaimed_by_startup_recovery(self) -> None:
        # `recover_inflight_outbox` is startup recovery, so it must reclaim only
        # leases older than the window: a live lease another process is holding
        # would double-deliver if reset.
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            fresh = store.add_outbox("test", {"value": "fresh"})
            stale = store.add_outbox("test", {"value": "stale"})
            store.claim_outbox(lease_seconds=1.0)
            store.connection.execute("UPDATE outbox SET claimed_at='2000-01-01T00:00:00Z' WHERE message_id=?", (stale,))
            store.connection.commit()
            self.assertEqual(store.recover_inflight_outbox(stale_after_seconds=300.0), 1)
            states = dict(store.connection.execute("SELECT message_id, delivery_state FROM outbox").fetchall())
            self.assertEqual(states[stale], "pending")
            self.assertEqual(states[fresh], "delivering", "a lease inside the window is live")
            # The explicit operator call without a window keeps the original
            # "recover every delivering row" behaviour.
            self.assertEqual(store.recover_inflight_outbox(), 1)
            store.close()

    def test_run_effect_capabilities_are_scoped_to_the_run_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.record_effect("run-1:0:call-a", "send_message_to_user", "h1", {"ok": True}, "applied")
            store.record_effect("run-1:1:call-b", "send_message_to_user", "h2", {"ok": True}, "applied")
            store.record_effect("run-2:0:call-c", "bash", "h3", {"ok": True}, "applied")
            self.assertEqual(store.run_effect_capabilities("run-1"), {"send_message_to_user": 2})
            self.assertEqual(store.run_effect_capabilities("run-2"), {"bash": 1})
            self.assertEqual(store.run_effect_capabilities("run-3"), {})
            store.close()

    def test_snapshot_episode_stores_a_transcript_pointer_not_a_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.append_transcript("provider_request", {"messages": []}, "run-1")
            snapshot = store.snapshot_episode("run-1")
            stored = json.loads(
                store.connection.execute(
                    "SELECT payload FROM episode_snapshots WHERE run_id=?", ("run-1",)
                ).fetchone()[0]
            )
            self.assertEqual(stored["transcript"], {"rebuilt_from": "transcript", "rows_at_freeze": 1})
            self.assertEqual(snapshot["transcript"][0]["kind"], "provider_request", "the read path rebuilds the projection")
            store.close()

    def test_compact_episode_snapshots_rewrites_only_byte_identical_legacy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.append_transcript("provider_request", {"messages": []}, "run-1")
            rebuilt = store._snapshot_transcript("run-1")
            events = [{"sequence": 1, "kind": "tool_result", "payload": {"ok": True}}]
            legacy = json.dumps({"events": events, "transcript": rebuilt}, ensure_ascii=False, sort_keys=True)
            store.connection.execute(
                "INSERT INTO episode_snapshots(snapshot_id, run_id, first_sequence, last_sequence, payload_hash, payload, created_at) "
                "VALUES ('legacy', 'run-1', 1, 1, 'old', ?, '2026-01-01T00:00:00Z')",
                (legacy,),
            )
            mismatched = json.dumps({"events": [], "transcript": [{"kind": "wrong"}]}, sort_keys=True)
            store.connection.execute(
                "INSERT INTO episode_snapshots(snapshot_id, run_id, first_sequence, last_sequence, payload_hash, payload, created_at) "
                "VALUES ('mismatch', 'run-2', 0, 0, 'old', ?, '2026-01-01T00:00:00Z')",
                (mismatched,),
            )
            store.connection.commit()

            result = store.compact_episode_snapshots()
            self.assertEqual(result["snapshots"], 1)
            self.assertGreater(result["bytes"], 0)
            rows = {
                row["run_id"]: json.loads(row["payload"])
                for row in store.connection.execute("SELECT run_id, payload FROM episode_snapshots").fetchall()
            }
            self.assertEqual(rows["run-1"]["transcript"], {"rebuilt_from": "transcript", "rows_at_freeze": len(rebuilt)})
            self.assertEqual(rows["run-1"]["events"], events, "events are not rebuildable and must be untouched")
            self.assertEqual(rows["run-2"]["transcript"], [{"kind": "wrong"}], "a mismatched rebuild keeps the old bytes")
            stored_hash = store.connection.execute(
                "SELECT payload_hash FROM episode_snapshots WHERE run_id='run-1'"
            ).fetchone()[0]
            self.assertNotEqual(stored_hash, "old")
            store.close()

    def test_capability_effect_referenced_only_by_episode_snapshot_survives(self) -> None:
        # The snapshot stores the run's tool_result events verbatim, so an
        # effect address lives there too; a snapshot is pruned by run count, not
        # age, so the pin must hold past the effect retention window.
        from skynet.models import Budget, RunRecord
        from skynet.time import utc_now

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.create_run(RunRecord("run-1", 1, RunStatus.COMPLETED, utc_now(), Budget()))
            key = "run-1:0:call-ev"
            store.record_effect(key, "bash", "hash", {"ok": True}, "applied")
            store.append_event(
                "tool_result",
                {"call": {"call_id": "call-ev", "tool_name": "bash"}, "result": {"ok": True, "effect_key": key}},
                "run-1",
            )
            store.snapshot_episode("run-1")
            # Drop the event-log holder so the snapshot is the only remaining
            # pointer to the effect row.
            store.connection.execute("DELETE FROM event_log WHERE kind='tool_result'")
            store.connection.execute("UPDATE capability_effects SET created_at='2020-01-01T00:00:00Z'")
            self.assertEqual(store.prune_capability_effects(30), 0, "the retained snapshot still names the row")
            self.assertIsNotNone(store.effect(key), "the address in the retained snapshot must resolve")
            store.connection.execute("DELETE FROM episode_snapshots WHERE run_id='run-1'")
            self.assertEqual(store.prune_capability_effects(30), 1, "with the snapshot gone the pin releases")
            store.close()
