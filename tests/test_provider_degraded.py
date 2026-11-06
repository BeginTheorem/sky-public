"""A rung whose deployment is DEGRADED belongs to the rung, not to the request.

The body NVIDIA answered with, verbatim (measured live 2026-10-16/17)::

    HTTP 400 {"status":400,"title":"Bad Request","detail":"Function id
    '948fe171-ce7a-4332-8bc0-5e14e90259f9': DEGRADED function cannot be invoked"}

A 400 is a 400 to any status-code rule, so it classified as ``invalid_request`` --
which ``fallback._is_fatal`` treats as chain-fatal. The whole ladder therefore
aborted on the first rung, every planner pass died as ``planner_provider_error``,
runs sat at zero steps for hours, and the three healthy rungs standing behind the
degraded one were never tried (twelve failures an hour, all on that rung).
"""

from __future__ import annotations

import unittest

from skynet.models import ModelTurn
from skynet.providers.errors import ProviderError
from skynet.providers.fallback import FallbackProvider
from skynet.providers.openai_compatible import DEGRADED_PROVIDER_COOLDOWN_SECONDS, OpenAICompatibleProvider

DEGRADED_BODY = (
    '{"status":400,"title":"Bad Request","detail":"Function id '
    "'948fe171-ce7a-4332-8bc0-5e14e90259f9': DEGRADED function cannot be invoked\"}"
)


class DegradedClassificationTests(unittest.TestCase):
    def test_a_degraded_deployment_is_not_a_broken_request(self) -> None:
        category, retryable, cooldown = OpenAICompatibleProvider._classify_status(400, DEGRADED_BODY)
        self.assertEqual(category, "server")
        self.assertTrue(retryable)
        self.assertEqual(cooldown, DEGRADED_PROVIDER_COOLDOWN_SECONDS)

    def test_it_does_not_abort_the_ladder(self) -> None:
        self.assertFalse(FallbackProvider._is_fatal(ProviderError("degraded", category="server", retryable=True)))

    def test_a_plain_400_is_still_a_broken_request(self) -> None:
        # The fix must not widen into the malformed-payload case: a genuinely
        # wrong request still aborts the chain instead of walking it.
        category, retryable, _cooldown = OpenAICompatibleProvider._classify_status(400, '{"error":{"message":"bad arguments"}}')
        self.assertEqual(category, "invalid_request")
        self.assertFalse(retryable)
        self.assertTrue(FallbackProvider._is_fatal(ProviderError("bad", category="invalid_request", retryable=False)))

    def test_the_status_alone_decides_nothing(self) -> None:
        # 503 without the marker keeps the ordinary short server cooldown.
        self.assertEqual(OpenAICompatibleProvider._classify_status(503, "")[2], 10.0)


class DegradedLadderTests(unittest.TestCase):
    def test_the_ladder_walks_past_a_degraded_rung(self) -> None:
        class Degraded:
            name = "nvidia_ultra"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError(
                    "nvidia_ultra request failed: HTTP 400: DEGRADED function cannot be invoked",
                    category="server",
                    retryable=True,
                    cooldown_seconds=DEGRADED_PROVIDER_COOLDOWN_SECONDS,
                )

        class Healthy:
            name = "nvidia_lightning"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="answered by the rung behind it")

        with self.assertLogs("skynet.providers.fallback", level="INFO") as captured:
            result = FallbackProvider([Degraded(), Healthy()]).complete([], max_tokens=1)
        self.assertEqual(result.text, "answered by the rung behind it")
        self.assertTrue(any("fallback_failure" in line and "nvidia_ultra" in line for line in captured.output))

    def test_a_degraded_rung_is_benched_for_its_long_cooldown(self) -> None:
        class Degraded:
            name = "nvidia_ultra"
            strikes = 0

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                Degraded.strikes += 1
                raise ProviderError("degraded", category="server", retryable=True, cooldown_seconds=DEGRADED_PROVIDER_COOLDOWN_SECONDS)

        class Healthy:
            name = "zai_glm47_flash"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="tail")

        fallback = FallbackProvider([Degraded(), Healthy()])
        self.assertEqual(fallback.complete([], max_tokens=1).text, "tail")
        self.assertEqual(fallback.complete([], max_tokens=1).text, "tail")
        # The second call must not consult the degraded rung at all.
        self.assertEqual(Degraded.strikes, 1)


if __name__ == "__main__":
    unittest.main()
