"""The owner's stop is a stop: parking the carrier against the revive tick.

His `/stop` from the bot stopped the carrier on 2026-10-16 at 22:07:54 and the revive
timer started it again at 01:00:30, because from inside the box a deliberate owner stop
and an accident look identical. The brake used to be "stop the timer too", which only the
curator can apply -- never the owner from his phone. Now his stop writes a park that the
tick honours, his `/start` clears it, and the external probe stays quiet instead of
paging him about an outage he ordered.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest.mock import patch

from skynet import revive
from skynet.telegram_bot import PrivateBot

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE = REPO_ROOT / "scripts" / "external_probe.py"


def _load_probe() -> ModuleType:
    """Import the probe script by path: the gate lints ``skynet``/``tests`` only."""
    spec = importlib.util.spec_from_file_location("skynet_external_probe_under_test", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Result:
    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _Recorder:
    """Fake systemctl runner that records every argv it is given."""

    def __init__(self, state: str = "inactive") -> None:
        self.state = state
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> _Result:
        self.calls.append(list(argv))
        return _Result(stdout=f"{self.state}\n")


class _FakeAPI:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str, dict[str, Any]]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def send(self, chat_id: int, text: str, markup: Any = None, **kwargs: Any) -> None:
        if markup is not None:
            kwargs["reply_markup"] = markup
        self.sent.append((chat_id, text, kwargs))

    def answer_callback(self, callback_id: str, text: str) -> None:
        self.calls.append(("answerCallbackQuery", {"callback_query_id": callback_id, "text": text}))

    def edit_message(self, chat_id: int, message_id: int, text: str, markup: Any = None) -> bool:
        payload: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": text}
        if markup is not None:
            payload["reply_markup"] = markup
        self.calls.append(("editMessageText", payload))
        return True

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((method, params or {}))
        return {"ok": True, "result": True}


class ParkFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name) / "state" / "skynet.sqlite3"
        self.state.parent.mkdir(parents=True, exist_ok=True)
        patcher = patch.dict(os.environ, {"SKYNET_STATE": str(self.state)}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SKYNET_PARK_FILE", None)

    def test_no_file_means_not_parked(self) -> None:
        self.assertIsNone(revive.park_state())

    def test_a_park_carries_its_reason_and_lifts_with_clear(self) -> None:
        record = revive.set_park("owner /stop from the Telegram bot")
        self.assertEqual(record["parked"], True)
        state = cast(dict[str, Any], revive.park_state())
        self.assertIn("Telegram", str(state["reason"]))
        revive.clear_park()
        self.assertIsNone(revive.park_state())

    def test_a_corrupt_or_unparked_file_is_not_a_park(self) -> None:
        path = revive.park_path()
        path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(revive.park_state())
        path.write_text(json.dumps({"parked": False}), encoding="utf-8")
        self.assertIsNone(revive.park_state())

    def test_clearing_an_absent_park_is_not_an_error(self) -> None:
        revive.clear_park()
        self.assertIsNone(revive.park_state())


class ParkedReviveTests(ParkFileTests):
    def test_a_parked_carrier_is_not_revived(self) -> None:
        revive.set_park("owner stop")
        runner = _Recorder(state="inactive")
        self.assertEqual(revive.revive(runner=runner), 0)
        self.assertEqual(runner.calls, [], "the tick must not even ask systemctl while parked")

    def test_the_same_tick_starts_her_once_the_park_is_lifted(self) -> None:
        revive.set_park("owner stop")
        runner = _Recorder(state="inactive")
        revive.revive(runner=runner)
        revive.clear_park()
        self.assertEqual(revive.revive(runner=runner), 0)
        self.assertTrue(any("start" in call for call in runner.calls))


class BotParkTests(ParkFileTests):
    def _bot(self) -> tuple[PrivateBot, _FakeAPI]:
        api = _FakeAPI()
        bot = PrivateBot(cast(Any, api), self.state, 7, 8, "skynet.service")
        return bot, api

    def test_the_bots_stop_writes_the_park_before_the_unit_moves(self) -> None:
        bot, _api = self._bot()
        with patch("skynet.telegram_bot.subprocess.run", return_value=_Result(stdout="inactive")):
            bot.execute(8, "/stop", "")
        state = cast(dict[str, Any], revive.park_state())
        self.assertIsNotNone(state, "an owner /stop must leave a park behind")
        self.assertIn("Telegram", str(state["reason"]))

    def test_the_bots_start_lifts_it(self) -> None:
        bot, _api = self._bot()
        revive.set_park("owner stop")
        with patch("skynet.telegram_bot.subprocess.run", return_value=_Result(stdout="active")):
            bot.execute(8, "/start", "")
        self.assertIsNone(revive.park_state())

    def test_a_confirmed_stop_parks_too(self) -> None:
        # The confirm path is the one a button tap walks, and it must not be a
        # second, forgetful way to stop her.
        bot, _api = self._bot()
        with patch("skynet.telegram_bot.subprocess.run", return_value=_Result(stdout="inactive")):
            bot.handle_callback({"id": "cb-1", "data": "confirm:stop", "message": {"message_id": 4, "text": "stop?"}})
        self.assertIsNotNone(revive.park_state())


class ProbeParkTests(ParkFileTests):
    def test_the_probe_stays_quiet_while_she_is_parked(self) -> None:
        probe = _load_probe()
        revive.set_park("owner stop")
        env = {
            "SKYNET_TELEGRAM_BOT_TOKEN": "123:test",
            "SKYNET_TELEGRAM_ALLOWED_CHAT_ID": "42",
            "SKYNET_EXTERNAL_PROBE_LOG": str(self.state.with_name("external_probe.log")),
            "SKYNET_PROBE_ALERT_AFTER": "1",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(probe, "service_status", return_value="inactive"),
            patch.object(probe, "send_alert") as alert,
        ):
            self.assertEqual(probe.main(), 0)
        alert.assert_not_called()
        log = self.state.with_name("external_probe.log")
        self.assertIn("parked", log.read_text(encoding="utf-8"))

    def test_an_unparked_outage_still_pages_him(self) -> None:
        # The park must not have become a mute button for real outages.
        probe = _load_probe()
        env = {
            "SKYNET_TELEGRAM_BOT_TOKEN": "123:test",
            "SKYNET_TELEGRAM_ALLOWED_CHAT_ID": "42",
            "SKYNET_EXTERNAL_PROBE_LOG": str(self.state.with_name("probe2.log")),
            "SKYNET_PROBE_ALERT_AFTER": "1",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(probe, "service_status", return_value="inactive"),
            patch.object(probe, "send_alert") as alert,
        ):
            self.assertEqual(probe.main(), 0)
        self.assertTrue(alert.called)

    def test_an_active_service_never_touches_the_park(self) -> None:
        probe = _load_probe()
        # Pin the log: without it the probe falls back to the live state file and every
        # gate run appends a fake "active" line, resetting the counter that pages him.
        env = {
            "SKYNET_TELEGRAM_BOT_TOKEN": "123:test",
            "SKYNET_TELEGRAM_ALLOWED_CHAT_ID": "42",
            "SKYNET_EXTERNAL_PROBE_LOG": str(self.state.with_name("probe-active.log")),
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(probe, "service_status", return_value="active"),
            patch.object(probe, "send_alert") as alert,
        ):
            self.assertEqual(probe.main(), 0)
        alert.assert_not_called()
        log = self.state.with_name("probe-active.log")
        self.assertIn("service_active", log.read_text(encoding="utf-8"), "the pin, not the live log, took the write")


if __name__ == "__main__":
    unittest.main()
