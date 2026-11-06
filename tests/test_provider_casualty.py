"""A provider outage must not be charged to the task's budget.

The reactor exempts a run from the give-up budget when the CHAIN, not the model, ended
it -- but it detected that by looking for the text "all providers failed", which a
chain-fatal provider error never produces: it is re-raised straight from the rung, so the
surviving text names the model. Measured 2026-10-16 on an empty wallet (HTTP 402,
``invalid_request`` and therefore fatal): three zero-step casualties were charged to the
bounded diagnostic, ``task_gave_up`` blocked it, and the reactor minted a fresh twin every
~15 minutes -- generations 20, 23 and 26 of one task.

The widened rule stays narrow on purpose: the exemption is for a run that produced NO step
and whose own stream carries a chain-FATAL classification. A merely retryable strike is
still a model-side failure when the run then died on its own budget -- the rule the live
server paid for (``tests/test_liveness.py``).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast

from skynet.reactor import Reactor

BROKE_TEXT = 'openrouter request failed: HTTP 402: {"error":{"message":"Недостаточно средств","type":"invalid_request_error"}}'


class _StoreDouble:
    """Just the one store method ``_run_had_provider_failure`` reads."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def recent_run_events(self, run_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        del run_id, limit
        return self._events


def _verdict(events: list[dict[str, Any]]) -> bool:
    reactor = cast(Any, SimpleNamespace(store=_StoreDouble(events)))
    return cast(bool, Reactor._run_had_provider_failure(reactor, "run-1"))


def _committed(*, failure: str, steps: int) -> dict[str, Any]:
    return {"kind": "run_result_committed", "payload": {"failure": failure, "steps": steps}}


def _classified(category: str, *, retryable: bool) -> dict[str, Any]:
    return {"kind": "provider_error_classified", "payload": {"category": category, "retryable": retryable, "step": 1}}


class ProviderCasualtyTests(unittest.TestCase):
    def test_a_zero_step_fatal_run_is_an_outage_however_the_text_reads(self) -> None:
        # The live text of 2026-10-16 carried no chain marker at all.
        self.assertTrue(_verdict([_committed(failure=BROKE_TEXT, steps=0), _classified("invalid_request", retryable=False)]))

    def test_a_bad_credential_on_a_zero_step_run_is_an_outage(self) -> None:
        self.assertTrue(_verdict([_committed(failure="openrouter request failed: HTTP 403", steps=0), _classified("auth", retryable=False)]))

    def test_a_retryable_strike_on_a_zero_step_run_is_still_charged(self) -> None:
        # The live rule this method exists for: a run that retried a provider and then
        # died is a model-side failure, and the task's budget must move.
        self.assertFalse(_verdict([_committed(failure="time budget", steps=0), _classified("network", retryable=True)]))

    def test_a_run_that_worked_before_dying_is_not_an_outage(self) -> None:
        self.assertFalse(_verdict([_committed(failure="model spent its budget", steps=7), _classified("invalid_request", retryable=False)]))

    def test_a_zero_step_run_without_a_classification_stays_charged(self) -> None:
        # No classification row means nothing accused the chain: keep the historical
        # judgment rather than exempting every silent death.
        self.assertFalse(_verdict([_committed(failure="harness gave up", steps=0), {"kind": "provider_error"}]))

    def test_the_chain_marker_still_wins_on_its_own(self) -> None:
        self.assertTrue(_verdict([_committed(failure="all providers failed after 3 attempts: a; b", steps=12)]))

    def test_a_run_with_no_committed_result_is_not_an_outage(self) -> None:
        self.assertFalse(_verdict([_classified("invalid_request", retryable=False)]))


if __name__ == "__main__":
    unittest.main()
