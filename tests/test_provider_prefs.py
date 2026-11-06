"""The owner's provider switches, and what the heartbeat makes of them.

Owner decision 2026-10-16: the bot's ``/providers`` is the analogue of Hermes' ``models``
-- he flips one rung at a time from his phone, and the REAL change lands on the next
heartbeat, never under a run already in flight. Two files carry that: the desired state the
bot writes, and the applied state the reactor copies across at the top of a tick. The
context ceiling follows the class of the rung that now sits first (paid 700k, free 100k).
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from skynet.providers import (
    FREE_CONTEXT_CEILING_CHARS,
    FREE_CONTEXT_CEILING_TOKENS,
    PAID_CONTEXT_CEILING_CHARS,
    PAID_CONTEXT_CEILING_TOKENS,
    active_chain_names,
    applied_chain_names,
    apply_provider_prefs,
    configured_provider_names,
    context_ceiling_chars,
    context_ceiling_tokens,
    load_provider_prefs,
    money_boost_state_path,
    provider_class,
    set_provider_off,
)

CHAIN = "openrouter_deepseek,openrouter_glm,nvidia_ultra,zai_glm47_flash"


class ProviderPrefsTestCase(unittest.TestCase):
    """Every case runs in a temp state dir, so no test can touch the live ledger."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.state = self.root / "state" / "skynet.sqlite3"
        self.state.parent.mkdir(parents=True, exist_ok=True)
        env = {
            "SKYNET_STATE": str(self.state),
            "SKYNET_PROVIDER_CHAIN": CHAIN,
            "SKYNET_MONEY_BOOST": "true",
            # The paid rungs are off by default (``_DEFAULT_ENABLED``): a chain that
            # carries them has to say so, exactly as the live env does.
            "OPENROUTER_ENABLED": "true",
        }
        patcher = patch.dict(os.environ, env, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SKYNET_PROVIDER_PREFS", None)
        os.environ.pop("SKYNET_PROVIDER_APPLIED", None)
        # The live unit exports this at an ABSOLUTE path outside the temp root, and it
        # outranks the ``SKYNET_STATE`` above: inherited, the suite reads the owner's live
        # switch instead of its own chain, which reddened 8 cases in the service env.
        os.environ.pop("SKYNET_MONEY_BOOST_STATE", None)
        for key in ("SKYNET_PAID_CONTEXT_TOKENS", "SKYNET_FREE_CONTEXT_TOKENS", "SKYNET_PAID_CONTEXT_CHARS", "SKYNET_FREE_CONTEXT_CHARS"):
            os.environ.pop(key, None)


class HermeticityTests(ProviderPrefsTestCase):
    """The suite decides its own money-boost; the owner's live switch never reaches it."""

    def test_the_money_boost_path_resolves_inside_the_temp_state_dir(self) -> None:
        # The unit exports SKYNET_MONEY_BOOST_STATE at an absolute path outside the temp
        # root. If that pin is dropped again, this case names the leak directly, instead
        # of leaving eight unrelated assertions to fail on the live chain.
        self.assertEqual(money_boost_state_path(), self.state.parent / "money-boost.json")


class CeilingTests(ProviderPrefsTestCase):
    def test_the_class_of_a_rung_decides_the_ceiling(self) -> None:
        self.assertEqual(provider_class("openrouter_deepseek"), "paid")
        self.assertEqual(provider_class("openrouter_glm"), "paid")
        self.assertEqual(provider_class("nvidia_ultra"), "free")
        self.assertEqual(context_ceiling_tokens("openrouter_deepseek"), PAID_CONTEXT_CEILING_TOKENS)
        self.assertEqual(context_ceiling_tokens("nvidia_ultra"), FREE_CONTEXT_CEILING_TOKENS)
        self.assertEqual(context_ceiling_chars("openrouter_glm"), PAID_CONTEXT_CEILING_CHARS)
        self.assertEqual(context_ceiling_chars("zai_glm47_flash"), FREE_CONTEXT_CEILING_CHARS)

    def test_the_owner_numbers_are_the_defaults(self) -> None:
        self.assertEqual((FREE_CONTEXT_CEILING_TOKENS, PAID_CONTEXT_CEILING_TOKENS), (100_000, 700_000))
        self.assertEqual((FREE_CONTEXT_CEILING_CHARS, PAID_CONTEXT_CEILING_CHARS), (100_000, 700_000))

    def test_an_env_override_moves_one_class_only(self) -> None:
        with patch.dict(os.environ, {"SKYNET_PAID_CONTEXT_TOKENS": "128000"}):
            self.assertEqual(context_ceiling_tokens("openrouter_deepseek"), 128_000)
            self.assertEqual(context_ceiling_tokens("nvidia_ultra"), FREE_CONTEXT_CEILING_TOKENS)

    def test_a_nonsense_override_is_ignored_rather_than_obeyed(self) -> None:
        with patch.dict(os.environ, {"SKYNET_FREE_CONTEXT_TOKENS": "zero"}):
            self.assertEqual(context_ceiling_tokens("nvidia_ultra"), FREE_CONTEXT_CEILING_TOKENS)
        with patch.dict(os.environ, {"SKYNET_FREE_CONTEXT_TOKENS": "-5"}):
            self.assertEqual(context_ceiling_tokens("nvidia_ultra"), FREE_CONTEXT_CEILING_TOKENS)


class DesiredVersusAppliedTests(ProviderPrefsTestCase):
    def test_without_any_file_the_environment_alone_decides(self) -> None:
        self.assertEqual(active_chain_names(), configured_provider_names())
        self.assertIsNone(applied_chain_names())

    def test_a_toggle_writes_desired_state_and_changes_nothing_yet(self) -> None:
        before = active_chain_names()
        set_provider_off("openrouter_deepseek", True)
        self.assertEqual(load_provider_prefs()["off"], ["openrouter_deepseek"])
        self.assertEqual(active_chain_names(), before, "desired state must not move the live chain")

    def test_the_heartbeat_is_what_applies_it(self) -> None:
        set_provider_off("openrouter_deepseek", True)
        names, changed = apply_provider_prefs()
        self.assertTrue(changed)
        self.assertEqual(names[0], "openrouter_glm")
        self.assertEqual(active_chain_names()[0], "openrouter_glm")
        self.assertNotIn("openrouter_deepseek", active_chain_names())
        # A second tick with nothing new is not a change, so the caller events once.
        _names_again, changed_again = apply_provider_prefs()
        self.assertFalse(changed_again)

    def test_switching_back_on_restores_the_rung_at_its_place(self) -> None:
        set_provider_off("openrouter_deepseek", True)
        apply_provider_prefs()
        set_provider_off("openrouter_deepseek", False)
        _names, changed = apply_provider_prefs()
        self.assertTrue(changed)
        self.assertEqual(active_chain_names()[0], "openrouter_deepseek")

    def test_an_enabled_flag_bites_at_once_while_a_toggle_waits(self) -> None:
        # The owner's older switches (env flags) are hot by design; the phone switch
        # is tick-bound. The two must not be confused with each other.
        apply_provider_prefs()
        with patch.dict(os.environ, {"OPENROUTER_GLM_ENABLED": "false"}):
            self.assertNotIn("openrouter_glm", active_chain_names())

    def test_every_rung_switched_off_yields_nothing_and_says_so(self) -> None:
        for name in configured_provider_names():
            set_provider_off(name, True)
        names, changed = apply_provider_prefs()
        self.assertEqual(names, [])
        self.assertTrue(changed)
        self.assertEqual(applied_chain_names(), [])

    def test_a_corrupt_prefs_file_is_an_empty_switch_not_a_crash(self) -> None:
        path = self.state.with_name("provider-prefs.json")
        path.write_text("{not json", encoding="utf-8")
        self.assertEqual(load_provider_prefs()["off"], [])
        self.assertEqual(active_chain_names(), configured_provider_names())

    def test_money_boost_off_still_parkes_the_paid_rungs(self) -> None:
        with patch.dict(os.environ, {"SKYNET_MONEY_BOOST": "false"}):
            apply_provider_prefs()
            live = active_chain_names()
        self.assertNotIn("openrouter_deepseek", live)
        self.assertNotIn("openrouter_glm", live)
        self.assertEqual(live[0], "nvidia_ultra")

    def test_the_applied_file_holds_the_effective_list(self) -> None:
        # The snapshot is written while an owner switch exists -- it pins HIS
        # decision; with no file there is nothing to pin.
        set_provider_off("ollama_gemma", True)
        names, _changed = apply_provider_prefs()
        written = json.loads(cast(str, self.state.with_name("provider-applied.json").read_text(encoding="utf-8")))
        self.assertEqual(written["providers"], names)

    def test_no_switch_means_no_snapshot_at_all(self) -> None:
        # A fresh deploy must behave exactly as before: no applied file, no filter.
        apply_provider_prefs()
        self.assertFalse(self.state.with_name("provider-applied.json").exists())


if __name__ == "__main__":
    unittest.main()
