"""The asynchronous owner dialogue: durable questions and outbound messages."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from skynet.dialogue import AskUserTool, ReadInboxTool, SendMessageToUserTool
from skynet.outbox import render_outbox_message
from skynet.store import StateStore


class DialogueTests(unittest.TestCase):
    def _store(self, directory: str) -> StateStore:
        return StateStore(Path(directory) / "state.sqlite3")

    def test_asking_persists_and_queues_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            result = AskUserTool(store).execute(
                {"question": "Prefer openrouter or ollama?", "options": ["openrouter", "ollama"]}, idempotency_key="k"
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["state"], "queued")
            # Durable before anything else: a question that only lives in a
            # prompt is lost when the run is interrupted.
            self.assertEqual(len(store.open_questions()), 1)
            kinds = [row[0] for row in store.connection.execute("SELECT kind FROM outbox")]
            self.assertEqual(kinds, ["user_question"])
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='question_asked'").fetchone()[0], 1
            )
            store.close()

    def test_an_answer_closes_the_question_and_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            question_id = store.ask_question("Which host?", run_id="run-1")
            self.assertEqual(store.open_questions()[0]["question"], "Which host?")
            self.assertTrue(store.answer_question(question_id, "203.0.113.7", source="telegram"))
            self.assertEqual(store.open_questions(), [])
            row = store.connection.execute(
                "SELECT answer, source FROM user_questions WHERE question_id=?", (question_id,)
            ).fetchone()
            self.assertEqual((row["answer"], row["source"]), ("203.0.113.7", "telegram"))
            # Answering twice is refused, not silently overwritten.
            self.assertFalse(store.answer_question(question_id, "something else"))
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='question_answered'").fetchone()[0], 1
            )
            store.close()

    def test_an_expired_question_stops_being_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            store.ask_question("Too late?", ttl_seconds=60.0)
            store.connection.execute("UPDATE user_questions SET expires_at='2020-01-01T00:00:00Z'")
            self.assertEqual(store.open_questions(), [])
            self.assertEqual(
                store.connection.execute("SELECT status FROM user_questions").fetchone()[0], "expired"
            )
            store.close()

    def test_the_question_cap_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            tool = AskUserTool(store, max_open=2)
            for index in range(2):
                self.assertTrue(tool.execute({"question": f"q{index}"}, idempotency_key=f"k{index}")["ok"])
            refused = tool.execute({"question": "q3"}, idempotency_key="k3")
            self.assertFalse(refused["ok"])
            self.assertIn("too many open questions", refused["error"])
            self.assertEqual(len(store.open_questions()), 2)
            store.close()

    def test_an_empty_question_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            self.assertFalse(AskUserTool(store).execute({"question": "   "}, idempotency_key="k")["ok"])
            self.assertFalse(SendMessageToUserTool(store).execute({"message": ""}, idempotency_key="k")["ok"])
            store.close()

    def test_sending_a_message_queues_it_without_a_reply(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            result = SendMessageToUserTool(store).execute(
                {"message": "Promotion accepted", "severity": "warning"}, idempotency_key="k"
            )
            self.assertTrue(result["ok"])
            self.assertEqual(
                store.connection.execute("SELECT kind FROM outbox").fetchone()[0], "agent_message"
            )
            self.assertEqual(store.open_questions(), [])
            store.close()

    def test_replayed_message_call_does_not_queue_a_second_owner_message(self) -> None:
        """A crash between the outbox write and the effect commit must not duplicate.

        `react.py` writes `capability_effects` only after the tool returns, so the
        window between `add_outbox` and `record_effect` is real. On the resumed
        run the identical call is re-issued because no cached result exists; if
        the outbox row is keyed by a fresh uuid, a second owner message is queued
        and delivered. The call's effect key must be the row identity instead.
        Measured on a fresh store: 2 pending rows unpatched, 1 with the stable key
        (arXiv:2608.01710v1, semantic replay / durable token-independent state).
        """
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            tool = SendMessageToUserTool(store)
            effect_key = "run-a:7:call_1"
            first = tool.execute({"message": "still working", "severity": "info"}, idempotency_key=effect_key)
            # Simulated crash here: no `record_effect`, so the resumed run finds
            # no cached result and re-issues the identical call.
            self.assertIsNone(store.effect(effect_key))
            second = tool.execute({"message": "still working", "severity": "info"}, idempotency_key=effect_key)
            self.assertTrue(first["ok"])
            self.assertTrue(second["ok"])
            pending = [message for message in store.pending_outbox() if message["kind"] == "agent_message"]
            self.assertEqual(len(pending), 1, "one owner message per call identity, not one per issuance")
            self.assertEqual(pending[0]["payload"].get("idempotency_key"), effect_key)
            store.close()

    def test_distinct_message_calls_still_queue_distinct_messages(self) -> None:
        """The dedup is per call identity: a genuinely new call must still be sent."""
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            tool = SendMessageToUserTool(store)
            tool.execute({"message": "first", "severity": "info"}, idempotency_key="run-a:7:call_1")
            tool.execute({"message": "second", "severity": "info"}, idempotency_key="run-a:8:call_2")
            pending = [message for message in store.pending_outbox() if message["kind"] == "agent_message"]
            self.assertEqual(len(pending), 2, "distinct calls are distinct effects")
            store.close()

    def test_replayed_question_call_does_not_ask_the_owner_twice(self) -> None:
        """A crash between the outbox write and the effect commit must not duplicate.

        `react.py` writes `capability_effects` only after the tool returns, so the
        resumed run re-issues the identical call with no cached result. With a
        fresh uuid4 as the row key the owner is asked twice; measured on a fresh
        store: 2 `user_questions` rows and 2 `user_question` outbox rows
        unpatched, 1 patched (arXiv:2608.01710v1, semantic replay).
        """
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            tool = AskUserTool(store)
            effect_key = "run-a:9:call_1"
            first = tool.execute({"question": "continue?", "options": ["yes", "no"]}, idempotency_key=effect_key)
            # Simulated crash here: no `record_effect`, so the resumed run finds
            # no cached result and re-issues the identical call.
            self.assertIsNone(store.effect(effect_key))
            second = tool.execute({"question": "continue?", "options": ["yes", "no"]}, idempotency_key=effect_key)
            self.assertTrue(first["ok"])
            self.assertTrue(second["ok"])
            self.assertEqual(first["question_id"], second["question_id"])
            self.assertEqual(len(store.open_questions(limit=10)), 1, "one question per call identity")
            queued = [row for row in store.pending_outbox() if row["kind"] == "user_question"]
            self.assertEqual(len(queued), 1)
            self.assertEqual(queued[0]["message_id"], effect_key)
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='question_asked'").fetchone()[0], 1
            )
            store.close()

    def test_a_derived_question_id_keeps_the_answer_prefix_routable(self) -> None:
        """The call identity must not become the row id: `/answer <prefix>` routes.

        Every call in one run shares the `{run_id}:` prefix, so using the effect
        key as `question_id` would make every question in a run render the same
        8-character prefix and `telegram_bot.handle_answer` would reject both as
        ambiguous. The id is derived from the key instead and stays distinct.
        """
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            # A wide cap keeps the open-question guard out of the way: the point
            # here is the id derivation, not the cap (which the replay also
            # respects: at the cap a replayed call is refused, never duplicated).
            tool = AskUserTool(store, max_open=5)
            ids = [
                tool.execute({"question": f"q{index}"}, idempotency_key=f"run-a:{index}:call_{index}")["question_id"]
                for index in range(3)
            ]
            self.assertEqual(len({value[:8] for value in ids}), 3, "distinct calls need distinct answer prefixes")
            self.assertTrue(all(value.startswith("ask-") for value in ids))
            # The derived id is stable for the same call, so a replay resolves to
            # the same question and the owner's answer still lands.
            replayed = tool.execute({"question": "q0"}, idempotency_key="run-a:0:call_0")["question_id"]
            self.assertEqual(replayed, ids[0])
            self.assertTrue(store.answer_question(ids[0], "yes"))
            self.assertEqual(len(store.open_questions(limit=10)), 2)
            store.close()

    def test_read_inbox_returns_pending_owner_messages_without_consuming_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            store.add_inbox_event("m1", "user_message", {"text": "first"})
            store.add_inbox_event("m2", "user_message", {"text": "second"})
            # A different pending kind must not be returned by the read tool and
            # must not consume the small limit before ``user_message`` rows.
            store.add_inbox_event("a1", "user_answer", {"answer": "an answer"})
            result = ReadInboxTool(store).execute({}, idempotency_key="read")
            self.assertTrue(result["ok"])
            self.assertEqual(result["count"], 2)
            self.assertEqual([item["event_id"] for item in result["messages"]], ["m1", "m2"])
            self.assertEqual([item["text"] for item in result["messages"]], ["first", "second"])
            # Read-only: nothing is consumed, and a second read returns the same.
            self.assertEqual(len(store.pending_inbox()), 3)
            again = ReadInboxTool(store).execute({}, idempotency_key="read-2")
            self.assertEqual([item["event_id"] for item in again["messages"]], ["m1", "m2"])
            store.close()

    def test_read_inbox_is_empty_when_the_queue_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            result = ReadInboxTool(store).execute({}, idempotency_key="read")
            self.assertTrue(result["ok"])
            self.assertEqual(result["count"], 0)
            self.assertEqual(result["messages"], [])
            store.close()

    def test_read_inbox_bounds_the_limit_and_the_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            for index in range(25):
                store.add_inbox_event(f"m{index}", "user_message", {"text": "x" * 5000})
            tool = ReadInboxTool(store)
            result = tool.execute({"limit": 1000}, idempotency_key="read")
            # The tool cap, not the argument, owns the bound.
            self.assertEqual(result["count"], tool.MAX_LIMIT)
            self.assertTrue(all(len(item["text"]) <= tool.MAX_TEXT_CHARS for item in result["messages"]))
            store.close()

    def test_outbox_rendering_covers_the_new_kinds(self) -> None:
        question = render_outbox_message("user_question", {"question_id": "abcdef1234", "question": "Prefer?", "options": ["a", "b"]})
        self.assertIn("[QUESTION abcdef12]", question)
        self.assertIn("/answer", question)
        message = render_outbox_message("agent_message", {"message": "done", "severity": "critical"})
        self.assertEqual(message, "[CRITICAL] done")


class AlertReconciliationTests(unittest.TestCase):
    def test_a_delivered_outbox_message_closes_its_pending_alert(self) -> None:
        # The jsonl drain used to consume the same lease without marking the alert
        # row, so alerts_pending counted a delivered alert forever.
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            alert_id = store.raise_alert("livelock_suspected", {"task_id": "t"}, severity="warning")["alert_id"]
            self.assertEqual(len(store.pending_alerts()), 1)
            message = store.claim_outbox(limit=1, lease_seconds=0.0)[0]
            store.mark_outbox_delivered(message["message_id"])
            self.assertEqual(store.reconcile_delivered_alerts(), 1)
            self.assertEqual(store.pending_alerts(), [])
            row = store.connection.execute("SELECT channel FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
            self.assertEqual(row["channel"], "reconciled")
            # Idempotent: nothing left to reconcile.
            self.assertEqual(store.reconcile_delivered_alerts(), 0)
            store.close()

    def test_a_dead_lettered_message_does_not_close_its_alert(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.raise_alert("provider_lockout", {"providers": []}, severity="critical")
            message = store.claim_outbox(limit=1, lease_seconds=0.0)[0]
            store.mark_outbox_failed(message["message_id"], "telegram down", max_attempts=1)
            self.assertEqual(store.reconcile_delivered_alerts(), 0)
            self.assertEqual(len(store.pending_alerts()), 1)
            store.close()


class BlockingWaitTests(unittest.TestCase):
    def test_wait_returns_the_answer_when_the_owner_replies(self) -> None:
        import threading
        import time

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            tool = AskUserTool(store)

            def answer_soon() -> None:
                # A separate connection: sqlite objects are bound to their thread.
                import sqlite3

                time.sleep(0.1)
                connection = sqlite3.connect(str(path), timeout=5.0)
                try:
                    connection.execute(
                        "UPDATE user_questions SET status='answered', answered_at='now', answer=?, source='telegram' "
                        "WHERE status='open'",
                        ("yes, prefer ollama",),
                    )
                    connection.commit()
                finally:
                    connection.close()

            thread = threading.Thread(target=answer_soon, daemon=True)
            thread.start()
            result = tool.execute(
                {"question": "Prefer ollama or openrouter?", "wait_seconds": 5},
                idempotency_key="wait",
            )
            thread.join(timeout=5)
            self.assertTrue(result["ok"])
            self.assertEqual(result["state"], "answered")
            self.assertEqual(result["answer"], "yes, prefer ollama")
            store.close()

    def test_wait_times_out_and_tells_the_model_to_proceed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            tool = AskUserTool(store)
            tool.POLL_INTERVAL_SECONDS = 0.01
            result = tool.execute(
                {"question": "Anyone there?", "wait_seconds": 0.05},
                idempotency_key="timeout",
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["state"], "expired_wait")
            self.assertIn("assumption", result["note"])
            # The question stays queued: the owner may still answer it later.
            self.assertEqual(len(store.open_questions()), 1)
            store.close()

    def test_no_wait_argument_keeps_the_async_behaviour(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            result = AskUserTool(store).execute({"question": "async please"}, idempotency_key="k")
            self.assertEqual(result["state"], "queued")
            store.close()

    def test_ask_user_schema_is_honest_about_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            description = AskUserTool(store).schema["function"]["description"]
            self.assertNotIn("does not block", description)
            self.assertIn("Non-blocking by default", description)
            self.assertIn("wait_seconds", description)
            store.close()

    def test_stop_event_ends_a_wait_without_an_answer(self) -> None:
        import threading

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            tool = AskUserTool(store)
            tool.POLL_INTERVAL_SECONDS = 0.01
            stop = threading.Event()
            stop.set()
            tool.set_stop_event(stop)
            result = tool.execute({"question": "Anybody?", "wait_seconds": 5}, idempotency_key="stop")
            self.assertEqual(result["state"], "expired_wait")
            self.assertEqual(len(store.open_questions()), 1)
            store.close()
