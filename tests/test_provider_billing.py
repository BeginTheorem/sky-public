"""A rung that cannot pay must not kill the ladder.

Measured live 2026-10-16: the paid rung answered HTTP 402 with
``{"error":{"message":"Недостаточно средств"}}``. The body carries no quota marker, so
``_classify_status`` fell through to ``invalid_request`` -- a category ``_is_fatal``
treats as chain-fatal -- and one empty wallet killed every call, free tail included:
ten runs in forty minutes died at zero steps while that rung took 43 strikes.

Billing is provider-local: skip the rung, cool it long, keep the tail alive.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from typing import cast

from skynet.models import ModelTurn
from skynet.providers.errors import ProviderError
from skynet.providers.fallback import BILLING_PROVIDER_COOLDOWN_SECONDS, MIN_PROVIDER_COOLDOWN_SECONDS, FallbackProvider
from skynet.providers.openai_compatible import OpenAICompatibleProvider

# The body openrouter actually sent, verbatim.
BROKE_BODY = '{"error":{"message":"Недостаточно средств","type":"invalid_request_error"}}'


class BillingClassificationTests(unittest.TestCase):
    def test_a_402_is_billing_and_not_a_broken_request(self) -> None:
        category, retryable, cooldown = OpenAICompatibleProvider._classify_status(402, BROKE_BODY)
        self.assertEqual(category, "billing")
        self.assertFalse(retryable)
        self.assertEqual(cooldown, 0.0)

    def test_a_spent_quota_still_outranks_billing(self) -> None:
        # A free allowance that resets at midnight is a different fact from an empty
        # balance, and it keeps its until-next-UTC-day cooldown.
        category, retryable, _cooldown = OpenAICompatibleProvider._classify_status(
            402, '{"error":{"code":"insufficient_quota","type":"insufficient_quota"}}'
        )
        self.assertEqual(category, "quota")
        self.assertTrue(retryable)

    def test_a_401_is_still_chain_fatal(self) -> None:
        # The fix must not have widened into the credential case: a bad key IS a
        # broken chain, and it stays fatal.
        self.assertEqual(OpenAICompatibleProvider._classify_status(401, "")[0], "auth")
        self.assertTrue(FallbackProvider._is_fatal(ProviderError("bad key", category="auth", retryable=False)))


class BillingLadderTests(unittest.TestCase):
    def test_billing_is_not_chain_fatal(self) -> None:
        self.assertFalse(FallbackProvider._is_fatal(ProviderError("broke", category="billing", retryable=False)))

    def test_a_broke_rung_is_cooled_for_half_an_hour(self) -> None:
        class Broke:
            name = "openrouter_deepseek"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("broke", category="billing", retryable=False)

        fallback = FallbackProvider([Broke()])
        with self.assertRaises(ProviderError):
            fallback.complete([], max_tokens=1)
        blocked = cast(dict[str, float], fallback._blocked_until or {})
        self.assertIn("openrouter_deepseek", blocked)
        remaining = blocked["openrouter_deepseek"] - time.time()
        self.assertGreater(remaining, MIN_PROVIDER_COOLDOWN_SECONDS)
        self.assertLessEqual(remaining, BILLING_PROVIDER_COOLDOWN_SECONDS)

    def test_the_ladder_walks_past_a_rung_that_cannot_pay(self) -> None:
        class Broke:
            name = "openrouter_deepseek"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("broke", category="billing", retryable=False)

        class Free:
            name = "nvidia_ultra"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="free tail")

        with self.assertLogs("skynet.providers.fallback", level="INFO") as captured:
            result = FallbackProvider([Broke(), Free()]).complete([], max_tokens=1)
        self.assertEqual(result.text, "free tail")
        self.assertTrue(any("fallback_failure" in line and "openrouter_deepseek" in line for line in captured.output))

    def test_a_broke_rung_does_not_stop_the_next_call_from_using_the_tail(self) -> None:
        # The cooldown is what keeps a ten-rung ladder from re-striking the empty
        # wallet on every call: by the second call the broke rung must not be
        # consulted at all.
        class Broke:
            name = "openrouter_glm"
            strikes = 0

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                Broke.strikes += 1
                raise ProviderError("broke", category="billing", retryable=False)

        class Free:
            name = "zai_glm47_flash"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="tail")

        fallback = FallbackProvider([Broke(), Free()])
        self.assertEqual(fallback.complete([], max_tokens=1).text, "tail")
        self.assertTrue(fallback.complete([], max_tokens=1).text == "tail")
        self.assertEqual(Broke.strikes, 1)

    def test_the_cooldown_state_survives_a_restart(self) -> None:
        # A dead wallet outlives the process: the cooldown has to be on disk, or a
        # restart would buy the ladder one more strike at a rung that cannot pay.
        class Broke:
            name = "openrouter_deepseek"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("broke", category="billing", retryable=False)

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "provider-fallback.json"
            first = FallbackProvider([Broke()], state_path=state)
            with self.assertRaises(ProviderError):
                first.complete([], max_tokens=1)
            self.assertTrue(state.exists())
            self.assertIn("openrouter_deepseek", state.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
