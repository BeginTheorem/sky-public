"""A rung that keeps failing must step aside, not sit at the top of every call.

Measured live 2026-10-16/17: the category cooldowns (2s network, 10s server) are
shorter than the time one hanging rung takes to fail, so election picked that same
rung first on every attempt for hours -- twelve failures an hour, all on
``nvidia_ultra``, while the three healthy rungs behind it were never reached.
Counting consecutive failures and growing the cooldown turns a repeat offender
into a rung that steps aside, without touching the owner's grouped-strike scheme
and without overriding a wait the provider itself stated.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from typing import cast

from skynet.models import ModelTurn
from skynet.providers.errors import ProviderError
from skynet.providers.fallback import (
    MAX_ESCALATED_COOLDOWN_SECONDS,
    MIN_PROVIDER_COOLDOWN_SECONDS,
    FallbackProvider,
)

NAME = "nvidia_ultra"


class _Rung:
    name = NAME

    def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
        raise ProviderError("504", category="server", retryable=True, cooldown_seconds=10.0)


def _remaining(fallback: FallbackProvider) -> float:
    blocked = cast(dict[str, float], fallback._blocked_until or {})
    return blocked[NAME] - time.time()


class CooldownEscalationTests(unittest.TestCase):
    def test_the_cooldown_grows_with_the_streak(self) -> None:
        fallback = FallbackProvider([_Rung()])
        exc = ProviderError("504", category="server", retryable=True, cooldown_seconds=10.0)
        fallback._cool_after_failure(NAME, exc)
        first = _remaining(fallback)
        fallback._cool_after_failure(NAME, exc)
        second = _remaining(fallback)
        fallback._cool_after_failure(NAME, exc)
        third = _remaining(fallback)
        # The category default (10s) is below the floor, so the first is the floor
        # and each repeat doubles it.
        self.assertGreater(first, MIN_PROVIDER_COOLDOWN_SECONDS - 1)
        self.assertGreater(second, first * 1.5)
        self.assertGreater(third, second * 1.5)

    def test_it_is_capped(self) -> None:
        fallback = FallbackProvider([_Rung()])
        exc = ProviderError("504", category="server", retryable=True, cooldown_seconds=10.0)
        for _ in range(12):
            fallback._cool_after_failure(NAME, exc)
        self.assertLessEqual(_remaining(fallback), MAX_ESCALATED_COOLDOWN_SECONDS + 1)

    def test_an_answer_resets_the_streak(self) -> None:
        fallback = FallbackProvider([_Rung()])
        exc = ProviderError("504", category="server", retryable=True, cooldown_seconds=10.0)
        fallback._cool_after_failure(NAME, exc)
        fallback._cool_after_failure(NAME, exc)
        fallback._clear_cooldown(NAME)
        fallback._cool_after_failure(NAME, exc)
        # Back to the base cooldown: one outage does not tax the rung forever.
        self.assertLess(_remaining(fallback), MIN_PROVIDER_COOLDOWN_SECONDS * 2)

    def test_a_provider_stated_wait_is_never_escalated(self) -> None:
        # The owner's rule: a wait the provider itself stated wins, so the
        # escalation must not multiply it into a longer bench than it asked for.
        fallback = FallbackProvider([_Rung()])
        exc = ProviderError("429", category="rate_limit", retryable=True, cooldown_seconds=10.0, cooldown_explicit=True)
        fallback._cool_after_failure(NAME, exc)
        first = _remaining(fallback)
        fallback._cool_after_failure(NAME, exc)
        second = _remaining(fallback)
        self.assertAlmostEqual(first, second, delta=1.0)

    def test_block_provider_honours_a_stated_wait_exactly(self) -> None:
        # Even below the 30s floor: a provider that states ten seconds is obeyed.
        fallback = FallbackProvider([_Rung()])
        exc = ProviderError("429", category="rate_limit", retryable=True, cooldown_seconds=10.0, cooldown_explicit=True)
        fallback._block_provider(NAME, exc)
        self.assertLess(_remaining(fallback), 15.0)

    def test_billing_keeps_its_long_floor(self) -> None:
        # The cap must never SHORTEN the half-hour billing cooldown.
        fallback = FallbackProvider([_Rung()])
        exc = ProviderError("broke", category="billing", retryable=False, cooldown_seconds=0.0)
        for _ in range(6):
            fallback._cool_after_failure(NAME, exc)
        self.assertGreater(_remaining(fallback), MAX_ESCALATED_COOLDOWN_SECONDS)

    def test_the_streak_survives_a_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "provider-fallback.json"
            first = FallbackProvider([_Rung()], state_path=state)
            exc = ProviderError("504", category="server", retryable=True, cooldown_seconds=10.0)
            first._cool_after_failure(NAME, exc)
            first._cool_after_failure(NAME, exc)
            stored = state.read_text(encoding="utf-8")
            self.assertIn("failure_streak", stored)
            second = FallbackProvider([_Rung()], state_path=state)
            second._cool_after_failure(NAME, exc)
            # Third consecutive failure across a restart: the escalation carried.
            self.assertGreater(_remaining(second), MIN_PROVIDER_COOLDOWN_SECONDS * 3)


if __name__ == "__main__":
    unittest.main()
