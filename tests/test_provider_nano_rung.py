"""The nano rung: the cheap fast one, measured before it was wired.

Measured 2026-11-02 on NIM with the key this chain already uses, interleaved against
the rungs it would sit beside (5 rounds each, one prompt):

    nvidia/nemotron-3-nano-omni-30b-a3b-reasoning   5/5 answers, 1.0 s median, 75 tokens
    nvidia/nemotron-3-ultra-550b-a55b               5/5 answers, 1.6 s median, 76 tokens
    nvidia/nemotron-3.5-super-vl-120b-a12b (head)   4/5 answers, 1.0 s median, 82 tokens

so it is as fast as the head and the least talkative of the three. It gets a rung of
its own rather than sharing the super rung because `FallbackProvider` keys cooldowns by
provider name: a 429 on one model must not bench another behind the same endpoint.

These tests pin what the wiring promises: the rung is built from the environment chain in
the configured order, it can be parked by its own flag without touching its neighbours,
its model and budget are overridable, and the canonical name is what a legacy token
resolves to.
"""

from __future__ import annotations

import os
import unittest
from typing import Any, cast
from unittest.mock import patch

from skynet.providers import OpenAICompatibleProvider, active_chain_names, build_provider

NANO_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


def _environment(**overrides: str) -> dict[str, str]:
    base = {
        "SKYNET_PROVIDER_CHAIN": "nvidia_super,nvidia_nano,nvidia_lightning",
        "NVIDIA_API_KEY": "nvidia-secret",
    }
    base.update(overrides)
    return base


class NanoRungTests(unittest.TestCase):
    def test_the_rung_is_built_and_keeps_the_configured_order(self) -> None:
        with patch.dict(os.environ, _environment(), clear=True):
            provider = cast(Any, build_provider())
            self.assertEqual(active_chain_names(), ["nvidia_super", "nvidia_nano", "nvidia_lightning"])
        self.assertEqual([item.name for item in provider.providers], ["nvidia_super", "nvidia_nano", "nvidia_lightning"])
        nano = cast(OpenAICompatibleProvider, provider.providers[1])
        self.assertEqual(nano.model, NANO_MODEL)
        self.assertEqual(nano.max_output_tokens, 8192)

    def test_the_rung_is_parked_by_its_own_flag_without_benching_its_neighbours(self) -> None:
        with patch.dict(os.environ, _environment(NEMOTRON_NANO_ENABLED="false"), clear=True):
            provider = cast(Any, build_provider())
            self.assertEqual(active_chain_names(), ["nvidia_super", "nvidia_lightning"])
        self.assertEqual([item.name for item in provider.providers], ["nvidia_super", "nvidia_lightning"])
        # The shared NVIDIA switch still governs it, the same way it governs the others.
        with patch.dict(os.environ, _environment(NVIDIA_ENABLED="false"), clear=True):
            self.assertEqual(active_chain_names(), [])

    def test_the_legacy_token_resolves_to_the_canonical_name(self) -> None:
        with patch.dict(os.environ, _environment(SKYNET_PROVIDER_CHAIN="nemotron_nano"), clear=True):
            provider = cast(Any, build_provider())
            self.assertEqual(active_chain_names(), ["nvidia_nano"])
        self.assertEqual([item.name for item in provider.providers], ["nvidia_nano"])

    def test_model_and_budget_are_overridable_from_the_environment(self) -> None:
        with patch.dict(
            os.environ,
            _environment(NEMOTRON_NANO_MODEL="nvidia/some-other-nano", NEMOTRON_NANO_MAX_OUTPUT_TOKENS="4096"),
            clear=True,
        ):
            provider = cast(Any, build_provider())
        nano = cast(OpenAICompatibleProvider, provider.providers[1])
        self.assertEqual(nano.model, "nvidia/some-other-nano")
        self.assertEqual(nano.max_output_tokens, 4096)


if __name__ == "__main__":
    unittest.main()
