import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import socks

from skynet.outbox import TelegramTransport, markup_for_outbox
from skynet.telegram_bot import (
    RECEIPT_LIMIT,
    RECEIPT_MAX_ATTEMPTS,
    RECEIPT_READ,
    RECEIPT_RETRY_BACKOFF_SECONDS,
    RECEIPT_SEEN,
    TELEGRAM_ALLOWED_REACTIONS,
    ApiError,
    PrivateBot,
    TelegramAPI,
    chunk_html,
    redact,
)
from skynet.telegram_ui import help_keyboard, logs_keyboard, parse_callback, question_keyboard, status_keyboard


class FakeAPI:
    def __init__(self):
        self.sent: list[tuple[int, str, dict]] = []
        self.calls: list[tuple[str, dict]] = []

    def send(self, chat_id, text, markup=None, **kwargs):
        if markup is not None:
            kwargs["reply_markup"] = markup
        self.sent.append((chat_id, text, kwargs))

    def answer_callback(self, callback_id, text):
        self.calls.append(("answerCallbackQuery", {"callback_query_id": callback_id, "text": text}))

    def edit_message(self, chat_id, message_id, text, markup=None):
        payload = {"chat_id": chat_id, "message_id": message_id, "text": text}
        if markup is not None:
            payload["reply_markup"] = markup
        self.calls.append(("editMessageText", payload))
        return True

    def call(self, method, params=None):
        self.calls.append((method, params or {}))
        return {"ok": True, "result": True}


class TelegramBotTests(unittest.TestCase):
    def test_redaction(self):
        self.assertNotIn("secret-value", redact("API_KEY=secret-value"))

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
        self.assertEqual(run.call_args_list[0].args[0], ["systemctl", "--user", "restart", "skynet.service"])

    def test_stop_command_executes_without_second_confirmation(self):
        api = FakeAPI()
        # A temp ledger, so the park this stop writes cannot land in the real state dir.
        with tempfile.TemporaryDirectory() as directory:
            bot = PrivateBot(cast(TelegramAPI, api), Path(directory) / "state.sqlite3", 7, 8, "skynet.service")
            with patch("skynet.telegram_bot.subprocess.run") as run:
                run.side_effect = [Mock(returncode=0, stdout="", stderr=""), Mock(returncode=3, stdout="inactive", stderr="")]
                bot.process({"message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/stop"}})
        self.assertEqual(run.call_args_list[0].args[0], ["systemctl", "--user", "stop", "skynet.service"])

    def test_systemctl_timeout_still_replies_and_keeps_bot_alive(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = PrivateBot(cast(TelegramAPI, api), Path(directory) / "state.sqlite3", 7, 8, "skynet.service")
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


class TelegramOwnerChannelTests(unittest.TestCase):
    def _bot(self, api, directory):
        return PrivateBot(cast(TelegramAPI, api), Path(directory) / "state.sqlite3", 7, 8, "skynet.service")

    def test_cold_start_confirms_backlog_without_executing_it(self):
        """A command left in the backlog must not run after a cold start.

        With no offset file (fresh deploy, purge, state reset) Telegram replays
        every pending update, so a queued ``/stop`` would stop the service seconds
        after it comes up. The first poll must confirm the backlog and resume past
        it instead of executing it.
        """
        with tempfile.TemporaryDirectory() as directory:
            api = FakeAPI()
            bot = self._bot(api, directory)
            offset_path = Path(directory) / "telegram-offset.json"
            bot.offset_path = offset_path
            stale = {"update_id": 77, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/stop"}}
            polls: list[dict] = []

            def fake_call(method, params=None):
                api.calls.append((method, params or {}))
                if method == "getMe":
                    return {"ok": True, "result": True}
                if method == "getUpdates":
                    polls.append(params or {})
                    if len(polls) == 1:
                        return {"ok": True, "result": [stale]}
                    bot.request_stop()
                    return {"ok": True, "result": []}
                return {"ok": True, "result": True}

            api.call = fake_call
            with patch.object(bot, "execute") as execute:
                bot.run()

            self.assertEqual(polls[0].get("offset"), -1)  # drain the backlog, don't deliver it
            execute.assert_not_called()
            self.assertEqual(polls[1].get("offset"), 78)  # resume past the stale update
            self.assertEqual(bot.load_number(offset_path), 78)

    def test_status_renders_the_card_and_metrics_keeps_the_window(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            with patch("skynet.telegram_bot.status_card.card", return_value="КАРТОЧКА"), patch(
                "skynet.telegram_bot.metrics.snapshot", return_value={"proposals": {}, "resources": {}}
            ) as snapshot, patch("skynet.telegram_bot.metrics.format_report", return_value="REPORT"):
                bot.process({"update_id": 1, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/metrics 7"}})
                self.assertEqual(snapshot.call_args.kwargs["since_days"], 7.0)
                bot.process({"update_id": 2, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/metrics 999"}})
                self.assertEqual(snapshot.call_args.kwargs["since_days"], 90.0)
                bot.process({"update_id": 3, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/metrics"}})
                self.assertEqual(snapshot.call_args.kwargs["since_days"], 1.0)
                bot.process({"update_id": 4, "message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/status 7"}})
            self.assertEqual(api.sent[-1][1], "КАРТОЧКА", "the window argument belongs to /metrics, not the card")
            self.assertIn("REPORT", api.sent[-2][1])

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
        with patch.object(bot, "send_card") as send_card, patch.object(bot, "send_metrics") as send_metrics, patch.object(
            bot, "send_alerts"
        ) as send_alerts:
            for update_id, text in enumerate(("/status", "/alerts", "/metrics"), start=10):
                bot.process({"update_id": update_id, "message": {"from": {"id": 9}, "chat": {"id": 8}, "text": text}})
        send_card.assert_not_called()
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
                if method == "setMyCommands":
                    # Startup publishes the command menu; that is not a poll iteration.
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


class TelegramInterfaceTests(unittest.TestCase):
    """Buttons: the callback grammar, the answers they carry, and the cards that hold them."""

    def _bot(self, api, directory):
        return PrivateBot(cast(TelegramAPI, api), Path(directory) / "state.sqlite3", 7, 8, "skynet.service")

    def _callback(self, data, *, message_id=11, text="🔐 Запрос root-доступа", user=7, chat=8):
        return {
            "callback_query": {
                "id": "cb1",
                "from": {"id": user},
                "message": {"message_id": message_id, "chat": {"id": chat}, "text": text},
                "data": data,
            }
        }

    def test_callback_grammar(self):
        self.assertEqual(parse_callback("ui:logs:10"), ("ui", ["logs", "10"]))
        self.assertEqual(parse_callback("q:abc123:1"), ("q", ["abc123", "1"]))
        self.assertEqual(parse_callback("confirm:restart"), ("confirm", ["restart"]))
        self.assertEqual(parse_callback("cancel"), ("cancel", []))
        self.assertEqual(parse_callback(""), ("", []))
        self.assertEqual(parse_callback(None), ("", []))  # type: ignore[arg-type]

    def test_keyboards_are_well_formed(self):
        for keyboard in (status_keyboard(), logs_keyboard(10), help_keyboard()):
            buttons = [button for row in keyboard["inline_keyboard"] for button in row]
            self.assertTrue(buttons)
            for button in buttons:
                self.assertLessEqual(len(button["callback_data"]), 64)
                self.assertTrue(button["text"])
                self.assertTrue(button["callback_data"])
        self.assertIn("ui:status", [b["callback_data"] for b in status_keyboard()["inline_keyboard"][0]])
        self.assertIn("10 ✅", [b["text"] for row in logs_keyboard(10)["inline_keyboard"] for b in row])
        self.assertIsNone(question_keyboard("", ["да"]))
        self.assertIsNone(question_keyboard("qid", []))

    def test_question_button_answers_and_strips_the_keyboard(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            question_id = bot.store().ask_question(
                "Разрешить root?", options=["Разрешить", "Запретить"], ttl_seconds=900
            )
            # The prompt renders the short id, exactly as the question message does.
            bot.process(self._callback(f"q:{question_id[:8]}:0"))
            recorded = bot.store().answer_for(question_id)
        self.assertIsNotNone(recorded)
        assert recorded is not None  # narrows the type for the checker, as the other tests here do
        self.assertEqual(recorded["answer"], "Разрешить")
        self.assertEqual(recorded["source"], "telegram-button")
        # The edit lands first and the toast carries the outcome, so the owner sees which
        # answer was recorded rather than a generic "принято".
        self.assertEqual([name for name, _ in api.calls], ["editMessageText", "answerCallbackQuery"])
        self.assertEqual(api.calls[0][1]["text"], "🔐 Запрос root-доступа\n\n✅ ответ: Разрешить")
        self.assertNotIn("reply_markup", api.calls[0][1])
        self.assertEqual(api.calls[1][1]["text"], "Ответ: Разрешить")

    def test_a_closed_question_cannot_be_answered_twice(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            question_id = bot.store().ask_question("Разрешить root?", options=["Разрешить"], ttl_seconds=900)
            bot.store().answer_question(question_id, "Запретить", source="telegram")
            bot.process(self._callback(f"q:{question_id[:8]}:0"))
        self.assertEqual([name for name, _ in api.calls], ["answerCallbackQuery"])
        self.assertEqual(api.calls[0][1]["text"], "Вопрос уже закрыт")

    def test_unauthorized_callback_does_nothing(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            bot.process(self._callback("ui:status", user=9))
        self.assertEqual(api.calls, [])
        self.assertEqual(api.sent, [])

    def test_status_card_carries_its_buttons(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            with patch("skynet.telegram_bot.status_card.card", return_value="КАРТОЧКА"):
                bot.process({"message": {"from": {"id": 7}, "chat": {"id": 8}, "text": "/status"}})
        markup = api.sent[-1][2].get("reply_markup")
        if not isinstance(markup, dict):
            self.fail("the status card carries no keyboard")
        self.assertIn("ui:status", [b["callback_data"] for row in markup["inline_keyboard"] for b in row])

    def test_refresh_edits_the_card_in_place(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            with patch("skynet.telegram_bot.status_card.card", return_value="КАРТОЧКА"):
                bot.process(self._callback("ui:status", message_id=42, text="карточка"))
        self.assertEqual([name for name, _ in api.calls], ["answerCallbackQuery", "editMessageText"])
        self.assertEqual(api.calls[1][1]["message_id"], 42)
        self.assertEqual(api.sent, [])

    def test_logs_window_button_re_renders_that_many_runs(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            with patch.object(bot, "send_logs") as send_logs:
                bot.process(self._callback("ui:logs:10"))
        send_logs.assert_called_once_with(8, 10)

    def test_menu_is_published_for_the_owners_chat(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            bot.register_commands()
        method, params = api.calls[-1]
        self.assertEqual(method, "setMyCommands")
        self.assertEqual(params["scope"], {"type": "chat", "chat_id": 8})
        names = [item["command"] for item in params["commands"]]
        for expected in ("status", "logs", "metrics", "alerts", "providers", "answer", "stop", "help"):
            self.assertIn(expected, names)

    def test_a_failing_menu_call_is_not_fatal(self):
        api = FakeAPI()
        api.call = Mock(side_effect=RuntimeError("network down"))  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as directory:
            bot = self._bot(api, directory)
            bot.register_commands()  # must not raise

    def test_edit_message_tolerates_not_modified(self):
        api = TelegramAPI.__new__(TelegramAPI)
        api.call = Mock(side_effect=ApiError("Bad Request: message is not modified", {}))  # type: ignore[method-assign]
        self.assertFalse(api.edit_message(8, 1, "карточка", status_keyboard()))

    def test_outbox_attaches_buttons_only_to_questions(self):
        api = FakeAPI()
        transport = TelegramTransport(cast(TelegramAPI, api), 8)
        transport.send_markup("вопрос", question_keyboard("qid", ["Разрешить"]) or {})
        self.assertIn("reply_markup", api.calls[-1][1])
        self.assertIsNone(markup_for_outbox("alert", {}))
        self.assertIsNone(markup_for_outbox("agent_response", {"run_id": "x"}))
        self.assertIsNotNone(markup_for_outbox("user_question", {"question_id": "qid", "options": ["Разрешить"]}))
        self.assertIsNone(markup_for_outbox("user_question", {"question_id": "qid", "options": []}))

