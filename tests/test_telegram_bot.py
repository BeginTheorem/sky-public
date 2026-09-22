import importlib
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

import socks

from skynet.outbox import TelegramTransport
from skynet.telegram_bot import (
    RECEIPT_LIMIT,
    RECEIPT_MAX_ATTEMPTS,
    RECEIPT_READ,
    RECEIPT_RETRY_BACKOFF_SECONDS,
    RECEIPT_SEEN,
    TELEGRAM_ALLOWED_REACTIONS,
    PrivateBot,
    TelegramAPI,
    chunk_html,
    format_runtime_record,
    redact,
    runtime_event_kind,
)

try:
    importlib.import_module("skynet.reporting")
except ModuleNotFoundError:
    # reporting.py is authored in parallel; tests only need the render hook.
    _reporting_stub = types.ModuleType("skynet.reporting")
    cast(Any, _reporting_stub).render_recent = lambda store, limit=5: []
    cast(Any, _reporting_stub).recent_run_pairs = lambda store, limit=5: []
    sys.modules["skynet.reporting"] = _reporting_stub


class FakeAPI:
    def __init__(self):
        self.sent = []
        self.calls = []

    def send(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text, kwargs))

    def call(self, method, params=None):
        self.calls.append((method, params or {}))
        return {"ok": True, "result": True}


class TelegramBotTests(unittest.TestCase):
    def test_redaction(self):
        self.assertNotIn("secret-value", redact("API_KEY=secret-value"))
        formatted = format_runtime_record({"kind": "provider_response", "payload": {"text": "answer", "reasoning_content": "private"}})
        self.assertIn("answer", formatted)
        self.assertNotIn("private", formatted)
        wrapped = {"kind": "event", "payload": {"event_kind": "finish_report", "payload": {"text": "wrapped"}}}
        self.assertEqual(runtime_event_kind(wrapped), "finish_report")
        self.assertIn("finish_report", format_runtime_record(wrapped))
        self.assertIn("wrapped", format_runtime_record(wrapped))
        invalid_time = format_runtime_record({"timestamp": "telegram-filter-test", "kind": "finish_report", "payload": {"text": "kept"}})
        self.assertIn("unknown time", invalid_time)
        self.assertIn("kept", invalid_time)
        formatted_time = format_runtime_record({"timestamp": "2026-09-15T02:40:52Z", "kind": "finish_report", "payload": {}})
        self.assertIn("02:40:52 [UTC", formatted_time)

    def test_authorization_requires_both_user_and_chat(self):
        bot = PrivateBot(cast(TelegramAPI, FakeAPI()), Path("state/skynet.sqlite3"), 7, 8, "skynet.service")
        self.assertTrue(bot.authorized({"message": {"from": {"id": 7}, "chat": {"id": 8}}}))
        self.assertFalse(bot.authorized({"message": {"from": {"id": 7}, "chat": {"id": 9}}}))
        self.assertFalse(bot.authorized({"message": {"from": {"id": 9}, "chat": {"id": 8}}}))

    def test_unauthorized_update_does_not_reply_or_execute(self):
        api = FakeAPI()
        bot = PrivateBot(cast(TelegramAPI, api), Path("state/skynet.sqlite3"), 7, 8, "skynet.service")
        with patch("skynet.telegram_bot.subprocess.run") as run:
            bot.process({"update_id": 1, "message": {"from": {"id": 9}, "chat": {"id": 8}, "text": "/stop"}})
        self.assertEqual(api.sent, [])
        run.assert_not_called()

    def test_send_command_was_removed_and_never_injects_text(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.sqlite3"
            bot = PrivateBot(cast(TelegramAPI, api), state, 7, 8, "skynet.service")
            bot.process({"update_id": 1, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/send hello"}})
            # /send is gone: it is answered as an unknown command and no inbox
            # event is created, so a legacy command cannot inject text.
            self.assertTrue(any("Неизвестная команда" in sent[1] for sent in api.sent))
            self.assertEqual(bot.store().pending_inbox(), [])
            self.assertNotIn("/send", api.sent[-1][1])

    def test_help_does_not_advertise_send(self):
        api = FakeAPI()
        bot = PrivateBot(cast(TelegramAPI, api), Path("state/skynet.sqlite3"), 7, 8, "skynet.service")
        bot.process({"update_id": 2, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/help"}})
        self.assertNotIn("/send", api.sent[-1][1])

    def test_callback_is_checked_and_confirmation_is_executed(self):
        api = FakeAPI()
        bot = PrivateBot(cast(TelegramAPI, api), Path("state/skynet.sqlite3"), 7, 8, "skynet.service")
        with patch("skynet.telegram_bot.subprocess.run") as run:
            run.side_effect = [Mock(returncode=0, stdout="ok", stderr=""), Mock(returncode=0, stdout="active", stderr="")]
            bot.process({"callback_query": {"id": "cb", "from": {"id": 7}, "message": {"chat": {"id": 8}}, "data": "confirm:restart"}})
        self.assertEqual(api.calls[0][0], "answerCallbackQuery")
        self.assertEqual(run.call_args_list[0].args[0], ["systemctl", "restart", "skynet.service"])

    def test_stop_command_executes_without_second_confirmation(self):
        api = FakeAPI()
        bot = PrivateBot(cast(TelegramAPI, api), Path("state/skynet.sqlite3"), 7, 8, "skynet.service")
        with patch("skynet.telegram_bot.subprocess.run") as run:
            run.side_effect = [Mock(returncode=0, stdout="", stderr=""), Mock(returncode=3, stdout="inactive", stderr="")]
            bot.process({"message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/stop"}})
        self.assertEqual(run.call_args_list[0].args[0], ["systemctl", "stop", "skynet.service"])

    def test_systemctl_timeout_still_replies_and_keeps_bot_alive(self):
        api = FakeAPI()
        bot = PrivateBot(cast(TelegramAPI, api), Path("state/skynet.sqlite3"), 7, 8, "skynet.service")
        with patch("skynet.telegram_bot.subprocess.run") as run:
            run.side_effect = [subprocess.TimeoutExpired(["systemctl", "stop"], 20), Mock(returncode=3, stdout="inactive", stderr="")]
            bot.systemctl(8, "stop")
        self.assertEqual(len(api.sent), 2)
        self.assertIn("timeout", api.sent[0][1])
        self.assertIn("inactive", api.sent[1][1])

    def test_api_call_filters_ipv6_when_using_socks_proxy(self):
        api = TelegramAPI("token", "socks5://127.0.0.1:9050")
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'{"ok": true, "result": true}'
        with patch("skynet.telegram_bot.request.urlopen", return_value=response), patch("skynet.telegram_bot.socks.set_default_proxy") as set_proxy, patch("skynet.telegram_bot.socks.socksocket"):
            api.call("getMe")
        self.assertEqual(set_proxy.call_args_list[0].args, (socks.SOCKS5, "127.0.0.1", 9050))
        self.assertEqual(set_proxy.call_args_list[0].kwargs, {"rdns": True})

    def test_api_call_resolves_only_ipv4_for_socks_proxy(self):
        api = TelegramAPI("token", "socks5://127.0.0.1:9050")
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'{"ok": true, "result": true}'
        with patch("skynet.telegram_bot.request.urlopen", return_value=response), patch("skynet.telegram_bot.socks.set_default_proxy"), patch("skynet.telegram_bot.socks.socksocket"):
            original = socket.getaddrinfo
            api.call("getMe")
            self.assertIs(socket.getaddrinfo, original)

    def test_api_call_preserves_hostname_for_tor_remote_dns(self):
        api = TelegramAPI("token", "socks5://127.0.0.1:9050")
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'{"ok": true, "result": true}'
        with patch("skynet.telegram_bot.request.urlopen", return_value=response), patch("skynet.telegram_bot.socks.set_default_proxy"), patch("skynet.telegram_bot.socks.socksocket"), patch("skynet.telegram_bot.socket.getaddrinfo") as getaddrinfo:
            api.call("getMe")
            getaddrinfo.assert_not_called()

    def test_logs_renders_one_message_per_recent_entry(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.sqlite3"
            bot = PrivateBot(cast(TelegramAPI, api), state, 7, 8, "skynet.service")
            with patch("skynet.reporting.render_recent", return_value=["SCHEDULER 1", "REPORT 1"]) as render:
                bot.process({"message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/logs"}})
        self.assertEqual([sent[1] for sent in api.sent], ["SCHEDULER 1", "REPORT 1"])
        self.assertEqual(render.call_args.args[1], 5)

    def test_logs_argument_requests_that_many_runs(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.sqlite3"
            bot = PrivateBot(cast(TelegramAPI, api), state, 7, 8, "skynet.service")
            with patch("skynet.reporting.render_recent", return_value=["only"]) as render:
                bot.process({"message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/logs 3"}})
        self.assertEqual(render.call_args.args[1], 3)
        self.assertEqual([sent[1] for sent in api.sent], ["only"])

    def test_logs_clamps_range_and_falls_back_when_empty(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.sqlite3"
            bot = PrivateBot(cast(TelegramAPI, api), state, 7, 8, "skynet.service")
            with patch("skynet.reporting.render_recent", return_value=["x"]) as render:
                bot.process({"message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/logs 99"}})
                self.assertEqual(render.call_args.args[1], 10)
                bot.process({"message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/logs 0"}})
                self.assertEqual(render.call_args.args[1], 1)
            with patch("skynet.reporting.render_recent", return_value=[]):
                bot.process({"message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/logs"}})
        self.assertEqual(api.sent[-1][1], "Логов пока нет.")

    def test_polling_does_not_send_logs_automatically(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = PrivateBot(cast(TelegramAPI, api), Path(directory) / "state.sqlite3", 7, 8, "skynet.service")
            with patch.object(bot, "process"), patch.object(bot, "send_logs"), \
                 patch.object(api, "call", side_effect=[{"ok": True, "result": True}, KeyboardInterrupt]), \
                 self.assertRaises(KeyboardInterrupt):
                bot.run()
        self.assertEqual(api.sent, [])

    def test_number_state_roundtrip_and_corrupt_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offset.json"
            bot = PrivateBot(cast(TelegramAPI, FakeAPI()), Path(directory) / "state.sqlite3", 1, 1, "skynet.service")
            self.assertEqual(bot.load_number(path), 0)
            bot.save_number(path, 42)
            self.assertEqual(bot.load_number(path), 42)
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(bot.load_number(path, default=7), 7)

    def test_sanitize_redacts_credentials_recursively(self):
        from skynet.telegram_bot import sanitize

        cleaned = sanitize({"reasoning_content": "hidden", "api_key": "x", "nested": {"foo_token": "y", "keep": 1}, "items": [{"secret": "z"}]})
        self.assertNotIn("reasoning_content", cleaned)
        self.assertNotIn("api_key", cleaned)
        self.assertEqual(cleaned["nested"]["foo_token"], "[redacted]")
        self.assertEqual(cleaned["nested"]["keep"], 1)
        self.assertEqual(cleaned["items"][0], {})


class TelegramOwnerChannelTests(unittest.TestCase):
    def _bot(self, api, directory):
        return PrivateBot(cast(TelegramAPI, api), Path(directory) / "state.sqlite3", 7, 8, "skynet.service")

    def test_status_renders_metrics_and_clamps_window(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            with patch("skynet.telegram_bot.metrics.snapshot", return_value={"proposals": {}, "resources": {}}) as snapshot, patch(
                "skynet.telegram_bot.metrics.format_report", return_value="REPORT"
            ):
                bot.process({"update_id": 1, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/status 7"}})
                self.assertEqual(snapshot.call_args.kwargs["since_days"], 7.0)
                bot.process({"update_id": 2, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/status 999"}})
                self.assertEqual(snapshot.call_args.kwargs["since_days"], 90.0)
                bot.process({"update_id": 3, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/status"}})
                self.assertEqual(snapshot.call_args.kwargs["since_days"], 1.0)
            self.assertEqual(api.sent[-1][1], "REPORT")

    def test_metrics_includes_proposals_and_resources_json(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            data = {"proposals": {"accepted": 1}, "resources": {"db_bytes": 10}}
            with patch("skynet.telegram_bot.metrics.snapshot", return_value=data), patch(
                "skynet.telegram_bot.metrics.format_report", return_value="BASE"
            ):
                bot.process({"update_id": 4, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/metrics"}})
            text = api.sent[-1][1]
            self.assertIn("BASE", text)
            self.assertIn('"accepted": 1', text)

    def test_alerts_lists_pending_most_recent_first(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            store = bot.store()
            store.raise_alert("first", {"n": 1})
            store.connection.execute("UPDATE alerts SET last_seen_at='2020-01-01T00:00:00Z' WHERE kind='first'")
            store.raise_alert("second", {"n": 2})
            bot.process({"update_id": 5, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/alerts"}})
            text = api.sent[-1][1]
            self.assertIn("first", text)
            self.assertIn("second", text)
            self.assertLess(text.index("second"), text.index("first"))

    def test_unauthorized_cannot_read_status_alerts_metrics(self):
        api = FakeAPI()
        bot = PrivateBot(cast(TelegramAPI, api), Path("state/skynet.sqlite3"), 7, 8, "skynet.service")
        with patch.object(bot, "send_metrics") as send_metrics, patch.object(bot, "send_alerts") as send_alerts:
            for update_id, text in enumerate(("/status", "/alerts", "/metrics"), start=10):
                bot.process({"update_id": update_id, "message": {"from": {"id": 9}, "chat": {"id": 8}, "text": text}})
        send_metrics.assert_not_called()
        send_alerts.assert_not_called()

    def test_offset_write_is_atomic_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telegram-offset.json"
            bot = self._bot(FakeAPI(), directory)
            bot.save_number(path, 5)
            with patch("skynet.telegram_bot.os.replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
                bot.save_number(path, 9)
            self.assertEqual(bot.load_number(path), 5)
            self.assertEqual(list(Path(directory).glob("*.tmp-*")), [])

    def test_duplicate_update_id_is_not_processed_twice(self):
        api = FakeAPI()
        bot = PrivateBot(cast(TelegramAPI, api), Path("state/skynet.sqlite3"), 7, 8, "skynet.service")
        update = {"update_id": 11, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/help"}}
        with patch.object(bot, "execute") as execute:
            bot.process(update)
            bot.process(update)
        self.assertEqual(execute.call_count, 1)

    def test_command_exception_advances_offset_and_notifies(self):
        with tempfile.TemporaryDirectory() as directory:
            api = FakeAPI()
            bot = self._bot(api, directory)
            offset_path = Path(directory) / "telegram-offset.json"
            bot.offset_path = offset_path
            update = {"update_id": 41, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/status"}}

            def fake_call(method, params=None):
                api.calls.append((method, params or {}))
                if method == "getMe":
                    return {"ok": True, "result": True}
                if method == "getUpdates":
                    bot.request_stop()
                    return {"ok": True, "result": [update]}
                return {"ok": True, "result": True}

            api.call = fake_call
            with patch.object(bot, "process", side_effect=RuntimeError("kaboom")):
                bot.run()
            self.assertEqual(bot.load_number(offset_path), 42)
            self.assertTrue(any("kaboom" in sent[1] for sent in api.sent))

    def test_stop_event_exits_loop_between_iterations(self):
        with tempfile.TemporaryDirectory() as directory:
            api = FakeAPI()
            bot = self._bot(api, directory)
            bot.offset_path = Path(directory) / "telegram-offset.json"
            iterations: list[int] = []

            def fake_call(method, params=None):
                api.calls.append((method, params or {}))
                if method == "getMe":
                    return {"ok": True, "result": True}
                iterations.append(1)
                return {"ok": True, "result": []}

            api.call = fake_call
            original = bot._drain_safely

            def drain_and_stop():
                original()
                if len(iterations) >= 2:
                    bot.request_stop()

            bot._drain_safely = drain_and_stop
            bot.run()
            self.assertEqual(len(iterations), 2)
            self.assertTrue(bot._stop.is_set())

    def test_request_stop_sets_event(self):
        bot = PrivateBot(cast(TelegramAPI, FakeAPI()), Path("state/skynet.sqlite3"), 7, 8, "skynet.service")
        self.assertFalse(bot._stop.is_set())
        bot.request_stop(15, None)
        self.assertTrue(bot._stop.is_set())

    def test_answer_routes_by_short_prefix_to_the_right_question(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            store = bot.store()
            first = store.ask_question("первый вопрос")
            newest = store.ask_question("второй вопрос")
            bot.handle_answer(8, f"{first[:8]} мой ответ")
            self.assertIn(first[:8], api.sent[-1][1])
            answered = store.answer_for(first)
            self.assertIsNotNone(answered)
            assert answered is not None
            self.assertEqual(answered["answer"], "мой ответ")
            self.assertIsNone(store.answer_for(newest))
            store.close()

    def test_answer_rejects_an_ambiguous_prefix_without_consuming(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            store = bot.store()
            first = store.ask_question("первый вопрос")
            second = store.ask_question("второй вопрос")
            store.connection.execute("UPDATE user_questions SET question_id='abcd0001-0000-0000-0000-000000000000' WHERE question_id=?", (first,))
            store.connection.execute("UPDATE user_questions SET question_id='abcd0002-0000-0000-0000-000000000000' WHERE question_id=?", (second,))
            bot.handle_answer(8, "abcd общий ответ")
            self.assertIn("неоднозначен", api.sent[-1][1])
            self.assertIsNone(store.answer_for("abcd0001-0000-0000-0000-000000000000"))
            self.assertIsNone(store.answer_for("abcd0002-0000-0000-0000-000000000000"))
            self.assertEqual(len(store.open_questions()), 2)
            store.close()

    def test_answer_still_accepts_the_full_question_id(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            store = bot.store()
            question = store.ask_question("единственный вопрос")
            bot.handle_answer(8, f"{question} полный ответ")
            answered = store.answer_for(question)
            self.assertIsNotNone(answered)
            assert answered is not None
            self.assertEqual(answered["answer"], "полный ответ")
            store.close()

    def test_plain_message_becomes_user_message_without_unknown_command(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            bot.process({"update_id": 20, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "привет, организм"}})
            self.assertEqual(api.sent, [])
            rows = bot.store().pending_inbox()
            self.assertEqual([row["kind"] for row in rows], ["user_message"])
            self.assertEqual(rows[0]["payload"]["text"], "привет, организм")
            self.assertEqual(rows[0]["payload"]["source"], "telegram")

    def test_reply_to_single_open_question_records_answer(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            store = bot.store()
            question = store.ask_question("единственный вопрос")
            bot.process({
                "update_id": 21,
                "message": {
                    "from": {"id": 7},
                    "chat": {"id": 8},
                    "text": "мой ответ",
                    "reply_to_message": {"message_id": 5},
                },
            })
            self.assertEqual(api.sent, [])
            answered = store.answer_for(question)
            self.assertIsNotNone(answered)
            assert answered is not None
            self.assertEqual(answered["answer"], "мой ответ")
            kinds = [row["kind"] for row in store.pending_inbox()]
            self.assertIn("user_message", kinds)
            self.assertIn("user_answer", kinds)
            store.close()

    def test_reply_with_two_open_questions_stays_general_message(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            store = bot.store()
            first = store.ask_question("первый вопрос")
            second = store.ask_question("второй вопрос")
            bot.process({
                "update_id": 22,
                "message": {
                    "from": {"id": 7},
                    "chat": {"id": 8},
                    "text": "общий ответ",
                    "reply_to_message": {"message_id": 6},
                },
            })
            self.assertEqual(api.sent, [])
            self.assertIsNone(store.answer_for(first))
            self.assertIsNone(store.answer_for(second))
            kinds = [row["kind"] for row in store.pending_inbox()]
            self.assertEqual(kinds, ["user_message"])
            store.close()

    def test_transport_chunks_text_longer_than_4096(self):
        api = FakeAPI()
        TelegramTransport(api, 8).send("x" * 5000)
        sends = [call for call in api.calls if call[0] == "sendMessage"]
        self.assertEqual(len(sends), 2)
        self.assertEqual(len(sends[0][1]["text"]), 4096)


class ReceiptEmojiTests(unittest.TestCase):
    def test_receipts_are_inside_the_api_accepted_reaction_set(self) -> None:
        # The Bot API rejects any reaction outside this set with 400
        # REACTION_INVALID, which silently turns a read upgrade into a no-op.
        self.assertIn(RECEIPT_SEEN, TELEGRAM_ALLOWED_REACTIONS)
        self.assertIn(RECEIPT_READ, TELEGRAM_ALLOWED_REACTIONS)
        self.assertNotEqual(RECEIPT_SEEN, RECEIPT_READ)


class ReceiptLedgerTests(unittest.TestCase):
    """The read-upgrade ledger is bounded; a refused reaction cannot storm."""

    def _bot_with_read_event(self, api, directory, event_id="evt-1"):
        bot = PrivateBot(cast(TelegramAPI, api), Path(directory) / "state.sqlite3", 7, 8, "skynet.service")
        bot._receipts[101] = event_id
        bot._receipt_order.append(101)
        bot.store().append_event("inbox_delivered_in_run", {"event_ids": [event_id]})
        return bot

    @staticmethod
    def _reaction_calls(api):
        return [call for call in api.calls if call[0] == "setMessageReaction"]

    def test_failed_read_reaction_is_bounded_and_expires(self):
        with tempfile.TemporaryDirectory() as directory:
            api = FakeAPI()

            def failing_call(method, params=None):
                api.calls.append((method, params or {}))
                if method == "setMessageReaction":
                    raise RuntimeError("REACTION_INVALID")
                return {"ok": True, "result": True}

            api.call = failing_call
            bot = self._bot_with_read_event(api, directory)
            clock = {"now": 1000.0}
            with patch("skynet.telegram_bot.RECEIPT_REFRESH_INTERVAL_SECONDS", 0.0), \
                 patch("skynet.telegram_bot.time.monotonic", side_effect=lambda: clock["now"]):
                for _ in range(20):
                    clock["now"] += 1000.0
                    bot._refresh_receipts()
            self.assertEqual(len(self._reaction_calls(api)), RECEIPT_MAX_ATTEMPTS)
            self.assertNotIn(101, bot._receipts)
            self.assertEqual(list(bot._receipt_order), [])

    def test_backoff_delays_the_next_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            api = FakeAPI()

            def failing_call(method, params=None):
                api.calls.append((method, params or {}))
                if method == "setMessageReaction":
                    raise RuntimeError("REACTION_INVALID")
                return {"ok": True, "result": True}

            api.call = failing_call
            bot = self._bot_with_read_event(api, directory)
            clock = {"now": 1000.0}
            with patch("skynet.telegram_bot.RECEIPT_REFRESH_INTERVAL_SECONDS", 0.0), \
                 patch("skynet.telegram_bot.time.monotonic", side_effect=lambda: clock["now"]):
                bot._refresh_receipts()
                self.assertEqual(len(self._reaction_calls(api)), 1)
                clock["now"] += RECEIPT_RETRY_BACKOFF_SECONDS / 2
                bot._refresh_receipts()
                self.assertEqual(len(self._reaction_calls(api)), 1)
                clock["now"] += RECEIPT_RETRY_BACKOFF_SECONDS
                bot._refresh_receipts()
                self.assertEqual(len(self._reaction_calls(api)), 2)

    def test_refresh_pass_is_capped_per_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            api = FakeAPI()
            bot = self._bot_with_read_event(api, directory)
            for message_id in range(102, 120):
                bot._receipts[message_id] = f"evt-{message_id}"
                bot._receipt_order.append(message_id)
                bot.store().append_event("inbox_delivered_in_run", {"event_ids": [f"evt-{message_id}"]})
            bot._refresh_receipts()
            # The reactions succeed, but one pass attempts only its batch.
            from skynet.telegram_bot import RECEIPT_REFRESH_BATCH
            self.assertLessEqual(len(self._reaction_calls(api)), RECEIPT_REFRESH_BATCH)

    def test_refresh_finds_an_entry_outside_the_recent_window(self):
        # The old query read the most recent RECEIPT_LIMIT delivery events, so a
        # burst of more than RECEIPT_LIMIT events between two passes stranded an
        # older message on "seen" forever. The query is now driven by the ledger's
        # own event ids and cannot miss an entry it still holds.
        with tempfile.TemporaryDirectory() as directory:
            api = FakeAPI()
            bot = self._bot_with_read_event(api, directory, event_id="evt-old")
            for index in range(RECEIPT_LIMIT + 5):
                bot.store().append_event("inbox_delivered_in_run", {"event_ids": [f"evt-{index}"]})
            with patch("skynet.telegram_bot.RECEIPT_REFRESH_INTERVAL_SECONDS", 0.0):
                bot._refresh_receipts()
            self.assertEqual(len(self._reaction_calls(api)), 1)
            self.assertNotIn(101, bot._receipts)

    def test_persisted_retry_deadline_survives_a_restart(self):
        # The attempt counter was persisted but the backoff deadline was not, so a
        # restart granted one immediate retry. The deadline is now a wall-clock
        # `retry_after` in the sidecar and is honoured on load.
        with tempfile.TemporaryDirectory() as directory:
            api = FakeAPI()
            bot = self._bot_with_read_event(api, directory)
            bot._receipt_read_attempts[101] = 1
            clock = {"wall": 10_000.0, "mono": 5_000.0}
            with patch("skynet.telegram_bot.time.time", side_effect=lambda: clock["wall"]), \
                 patch("skynet.telegram_bot.time.monotonic", side_effect=lambda: clock["mono"]):
                bot._receipt_read_retry_after[101] = clock["mono"] + RECEIPT_RETRY_BACKOFF_SECONDS
                bot._save_receipts()
                restarted = PrivateBot(cast(TelegramAPI, api), Path(directory) / "state.sqlite3", 7, 8, "skynet.service")
                with patch("skynet.telegram_bot.RECEIPT_REFRESH_INTERVAL_SECONDS", 0.0):
                    restarted._refresh_receipts()
                    self.assertEqual(len(self._reaction_calls(api)), 0, "the persisted deadline must block the first retry")
                    clock["mono"] += RECEIPT_RETRY_BACKOFF_SECONDS + 1
                    restarted._refresh_receipts()
            self.assertEqual(len(self._reaction_calls(api)), 1)

    def test_utf16_chunking_keeps_emoji_messages_within_limit(self):
        text = "😀" * 3000
        parts = chunk_html(text, limit=4096)
        for part in parts:
            self.assertLessEqual(len(part.encode("utf-16-le")) // 2, 4096)
        self.assertEqual("".join(parts), text)

