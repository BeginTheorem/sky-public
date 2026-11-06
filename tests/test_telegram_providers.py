"""``/providers``: the owner flips one rung, and the next heartbeat makes it real.

The analogue of Hermes' ``models``: every rung of the chain, its class, the state it is in,
and a button per rung. Two things are pinned here -- that the card distinguishes the three
states that look identical from inside the chain (on, switched off from the phone, parked by
an env flag), and that a tap writes DESIRED state only, so a run already in flight keeps the
ladder it started with.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from skynet.providers import (
    active_chain_names,
    applied_chain_names,
    apply_provider_prefs,
    configured_provider_names,
    load_provider_prefs,
    money_boost_state_path,
)
from skynet.telegram_bot import PrivateBot
from skynet.telegram_ui import parse_callback, providers_keyboard

CHAIN = "openrouter_deepseek,openrouter_glm,nvidia_ultra,zai_glm47_flash"


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


class ProvidersCardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name) / "state" / "skynet.sqlite3"
        self.state.parent.mkdir(parents=True, exist_ok=True)
        env = {"SKYNET_STATE": str(self.state), "SKYNET_PROVIDER_CHAIN": CHAIN, "SKYNET_MONEY_BOOST": "true", "OPENROUTER_ENABLED": "true"}
        patcher = patch.dict(os.environ, env, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SKYNET_PARK_FILE", None)
        for key in ("SKYNET_PROVIDER_PREFS", "SKYNET_PROVIDER_APPLIED"):
            os.environ.pop(key, None)
        # Same pin as test_provider_prefs: the unit's absolute SKYNET_MONEY_BOOST_STATE
        # would otherwise override this suite's temp state dir and the owner's live switch
        # would decide the chain these cases assert on.
        os.environ.pop("SKYNET_MONEY_BOOST_STATE", None)
        apply_provider_prefs()

    def _bot(self) -> tuple[PrivateBot, _FakeAPI]:
        api = _FakeAPI()
        return PrivateBot(cast(Any, api), self.state, 7, 8, "skynet.service"), api

    def test_the_money_boost_path_resolves_inside_the_temp_state_dir(self) -> None:
        # The unit exports SKYNET_MONEY_BOOST_STATE at an absolute path outside the temp
        # root; with the pin dropped, the owner's live switch decides the very chain these
        # cases assert on. Fail on the leak here instead of eight assertions down.
        self.assertEqual(money_boost_state_path(), self.state.parent / "money-boost.json")

    def test_the_card_lists_every_rung_by_name_and_class(self) -> None:
        bot, _api = self._bot()
        text = bot.providers_text()
        for name in configured_provider_names():
            self.assertIn(name, text)
        self.assertIn("Платные:", text)
        self.assertIn("Бесплатные:", text)
        self.assertIn("Money boost: вкл", text)

    def test_the_command_sends_the_card_with_a_button_per_rung(self) -> None:
        bot, api = self._bot()
        bot.execute(8, "/providers", "")
        self.assertEqual(len(api.sent), 1)
        _chat, text, kwargs = api.sent[0]
        self.assertIn("openrouter_deepseek", text)
        keyboard = cast(dict[str, Any], kwargs["reply_markup"])["inline_keyboard"]
        data = [button["callback_data"] for row in keyboard for button in row]
        self.assertIn("pv:openrouter_deepseek", data)
        self.assertIn("pv:zai_glm47_flash", data)
        self.assertIn("ui:provider", data)

    def test_a_parked_flag_is_named_separately_from_a_rung_he_switched_off(self) -> None:
        bot, _api = self._bot()
        with patch.dict(os.environ, {"OPENROUTER_GLM_ENABLED": "false"}):
            text = bot.providers_text()
        self.assertIn("запаркован флагом", text)
        # The flag stays out of the desired-state file: it is his older switch.
        self.assertEqual(load_provider_prefs()["off"], [])

    def test_the_card_reports_the_park_state_of_the_carrier(self) -> None:
        from skynet.revive import PARK_FILE, clear_park, set_park

        bot, _api = self._bot()
        park = self.state.with_name(PARK_FILE)
        set_park("owner stop", park)
        self.addCleanup(clear_park, park)
        self.assertIn("Стоянка владельца: да", bot.providers_text())


class ProvidersToggleTests(ProvidersCardTests):
    def test_a_tap_writes_desired_state_and_says_it_waits_for_the_tick(self) -> None:
        bot, api = self._bot()
        bot.handle_callback({"id": "cb-9", "data": "pv:openrouter_deepseek", "message": {"message_id": 5, "text": "карточка"}})
        self.assertEqual(load_provider_prefs()["off"], ["openrouter_deepseek"])
        toast = [payload["text"] for method, payload in api.calls if method == "answerCallbackQuery"]
        self.assertTrue(any("heartbeat" in text for text in toast))
        # Desired only: the live chain has not moved and nothing is pinned yet --
        # the snapshot appears when the tick applies his decision.
        self.assertIn("openrouter_deepseek", active_chain_names())
        self.assertIsNone(applied_chain_names())

    def test_the_tick_is_what_moves_the_chain(self) -> None:
        bot, _api = self._bot()
        bot.handle_callback({"id": "cb-10", "data": "pv:openrouter_deepseek", "message": {"message_id": 5, "text": "карточка"}})
        names, changed = apply_provider_prefs()
        self.assertTrue(changed)
        self.assertEqual(names[0], "openrouter_glm")

    def test_a_second_tap_puts_the_rung_back(self) -> None:
        bot, _api = self._bot()
        for tap in range(2):
            bot.handle_callback({"id": f"cb-{tap}", "data": "pv:openrouter_glm", "message": {"message_id": 5, "text": "карточка"}})
        self.assertEqual(load_provider_prefs()["off"], [])

    def test_the_card_is_edited_in_place_after_a_tap(self) -> None:
        bot, api = self._bot()
        bot.handle_callback({"id": "cb-11", "data": "pv:openrouter_glm", "message": {"message_id": 77, "text": "карточка"}})
        edits = [payload for method, payload in api.calls if method == "editMessageText"]
        self.assertTrue(edits)
        self.assertEqual(edits[0]["message_id"], 77)

    def test_a_stale_button_without_a_name_is_refused(self) -> None:
        bot, api = self._bot()
        bot.handle_callback({"id": "cb-12", "data": "pv", "message": {"message_id": 5, "text": "карточка"}})
        self.assertEqual(load_provider_prefs()["off"], [])
        self.assertTrue(any("устарела" in payload["text"] for method, payload in api.calls if method == "answerCallbackQuery"))


class CallbackGrammarTests(unittest.TestCase):
    def test_the_callback_carries_a_rung_name_and_nothing_else(self) -> None:
        kind, args = parse_callback("pv:openrouter_deepseek")
        self.assertEqual(kind, "pv")
        self.assertEqual(args, ["openrouter_deepseek"])

    def test_two_rungs_sit_on_one_row(self) -> None:
        keyboard = providers_keyboard([("✅ a", "a"), ("⛔ b", "b"), ("✅ c", "c")])
        rows = cast(list[list[dict[str, str]]], keyboard["inline_keyboard"])
        self.assertEqual(len(rows[0]), 2)
        self.assertEqual(len(rows[1]), 1)
        self.assertEqual(rows[0][1]["callback_data"], "pv:b")

    def test_the_menu_advertises_the_command(self) -> None:
        api = _FakeAPI()
        bot = PrivateBot(cast(Any, api), Path("state/skynet.sqlite3"), 7, 8, "skynet.service")
        bot.register_commands()
        menus = [params for method, params in api.calls if method == "setMyCommands"]
        self.assertTrue(menus)
        names = [item["command"] for item in cast(list[dict[str, str]], menus[0]["commands"])]
        self.assertIn("providers", names)
        self.assertNotIn("provider", names)


if __name__ == "__main__":
    unittest.main()
