from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ..models import ModelTurn
from ..provider import LLMProvider, Message, ToolSchema
from .errors import ProviderError

log = logging.getLogger("skynet.providers.fallback")

MIN_PROVIDER_COOLDOWN_SECONDS = 30.0

# The two terminal texts the chain can raise, as recorded in
# ``run_results.failure``. They name DIFFERENT failure classes and are disjoint
# on the live ledger (measured 2026-09-25, 174 committed runs): 5 runs died with
# the cooldown text below -- the terminal model call was refused before it
# reached any provider -- and 12 died with a ladder that had struck at least one
# provider first. A reader that only matches the shared "all providers failed"
# substring cannot separate them, which is why the exact pair is named here.
CHAIN_WIDE_COOLDOWN_MARKER = "every provider is in cooldown"
ZERO_STRIKE_MARKER = "all providers failed after 0 attempts"


def is_chain_wide_cooldown_abort(failure: str) -> bool:
    """Whether a run's terminal failure text is a zero-strike chain cooldown.

    "Every provider is in cooldown" is not a dead provider: it is the chain
    declaring WHEN it can answer again (``blocked_until``), and the abort is
    reached without a single strike having been spent. Both markers are required
    together, so the 12 recorded runs whose ladder struck a provider and then
    gave up stay outside this class -- they are provider failures, not an
    unreachable chain.

    This is a classification of RECORDED text, and it is deliberately narrow:
    it answers "did this episode die because no provider could be reached at
    all", nothing more. Measured effect on the counters the reactor moves: each
    such run charged its task +1 ``attempts`` (the run was started) and 0
    ``consecutive_model_failures`` (the reactor's own "all providers failed"
    exemption already covered it), so a reader must not treat this class as the
    cause of the task give-up budget.
    """
    text = str(failure or "")
    return ZERO_STRIKE_MARKER in text and CHAIN_WIDE_COOLDOWN_MARKER in text


@dataclass(slots=True)
class FallbackProvider:
    providers: list[LLMProvider]
    max_attempts: int = 0
    timeout_ladder: tuple[float, ...] = ()
    strike_delay_seconds: float = 0.0
    _blocked_until: dict[str, float] | None = None
    event_logger: Callable[[str, dict[str, object]], None] | None = None
    state_path: Path | None = None
    ladder_deadline_seconds: float = 600.0
    # How long a call may wait out a chain-wide cooldown before admitting the
    # chain is down. 0.0 keeps the historical behaviour (raise immediately).
    max_wait_for_cooldown_seconds: float = 0.0
    # Optional supplier of the currently active provider names. Re-read at every
    # call so enabling or disabling the paid provider takes effect without a
    # process restart; provider instances (and their cooldowns) are preserved.
    active_names: Callable[[], list[str]] | None = None
    # Last active view that was logged, used only to journal real chain changes.
    _active_snapshot: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("FallbackProvider requires at least one provider")
        if self.max_attempts <= 0:
            self.max_attempts = len(self.providers)
        # An explicit ladder IS the organism's provider-retry mechanism:
        # ReActConfig.provider_retries stays 0 because retries live here, and a
        # provider struck once inside a call stays eligible for the remaining
        # rungs even after that failure opens its cooldown. A one-rung ladder
        # therefore disables retries entirely instead of shortening them, so a
        # single transient fault that a second strike would have absorbed ends
        # the episode. Measured on the live database: 7 needs_recovery runs
        # fail with "all providers failed after 1 attempts:
        # openrouter[invalid_response]" while the deployed
        # SKYNET_PROVIDER_TIMEOUT_LADDER=900 supplies exactly one rung, and 6 of
        # those 7 wrote no memories at all. Reusing the last configured timeout
        # as a second rung keeps the operator's timeout budget while restoring
        # the one retry the ladder exists for. An empty ladder is untouched: it
        # keeps the documented legacy one-strike-per-provider behaviour.
        if len(self.timeout_ladder) == 1:
            self.timeout_ladder = (self.timeout_ladder[0], self.timeout_ladder[0])
        self._blocked_until = {}
        self._load_state()

    def _load_state(self) -> None:
        if self.state_path is None:
            return
        try:
            stored = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(stored, dict):
            return
        names = {self._name(provider) for provider in self.providers}
        blocked = self._blocked_until if self._blocked_until is not None else {}
        stored_blocked = stored.get("blocked_until")
        if isinstance(stored_blocked, dict):
            for key, value in stored_blocked.items():
                name = str(key)
                if name not in names:
                    # A provider that left the chain must not linger as garbage.
                    continue
                try:
                    blocked[name] = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
        self._blocked_until = blocked
        key_state = stored.get("key_state")
        if isinstance(key_state, dict):
            for provider in self.providers:
                restore = getattr(provider, "restore_key_state", None)
                if not callable(restore):
                    continue
                state = key_state.get(self._name(provider))
                if not isinstance(state, dict):
                    continue
                try:
                    restore(state)
                except Exception:
                    log.warning("could not restore provider key state", exc_info=True)

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        payload: dict[str, object] = {"blocked_until": self._blocked_until or {}}
        key_state: dict[str, object] = {}
        for provider in self.providers:
            state = getattr(provider, "key_state", None)
            if callable(state):
                try:
                    state = state()
                except Exception:
                    state = None
            if isinstance(state, dict) and state:
                key_state[self._name(provider)] = state
        if key_state:
            payload["key_state"] = key_state
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{self.state_path.name}.", dir=self.state_path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(json.dumps(payload, sort_keys=True))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.state_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        except OSError:
            log.warning("could not persist provider fallback state", exc_info=True)

    @staticmethod
    def _name(provider: LLMProvider) -> str:
        return str(getattr(provider, "name", type(provider).__name__))

    def _log(self, event: str, payload: dict[str, object]) -> None:
        log.info("%s %s", event, payload)
        if self.event_logger is None:
            return
        try:
            self.event_logger(event, payload)
        except Exception:
            # A diagnostic sink must never abort the provider ladder: the store
            # can be momentarily unwritable or reached from a worker thread.
            log.warning("provider event logger failed for %s", event, exc_info=True)

    def set_event_logger(self, logger: Callable[[str, dict[str, object]], None]) -> None:
        self.event_logger = logger

    def _blocked_remaining(self) -> dict[str, float]:
        now = time.time()
        remaining: dict[str, float] = {}
        for name, until in (self._blocked_until or {}).items():
            value = until - now
            if value > 0:
                remaining[name] = round(value, 1)
        return remaining

    def _clear_cooldown(self, name: str) -> None:
        """Forget a cooldown the provider has just disproved by answering.

        A retryable failure opens a cooldown so the next cycle backs off, but a
        later *successful* strike from the same provider proves it is healthy
        now. Leaving that stale cooldown in place aborted the very next step of
        the same run on the live switch: a 500 on one step put
        nemotron in a 30s cooldown, the same step's retry then succeeded, and the
        following step was skipped as still cooling down -- three such aborts
        escalated to provider_lockout while the provider was actively answering.
        """
        blocked = self._blocked_until or {}
        if name in blocked:
            blocked.pop(name, None)
            self._save_state()

    def _active_providers(self) -> list[LLMProvider]:
        """The subset of the constructed chain the supplier currently lists.

        ``self.providers`` is the full set built once at startup and is never
        mutated: a provider that leaves the active chain (money-boost off) must
        come back when it is re-enabled (money-boost on) without a process
        restart. Cooldowns live in ``self._blocked_until`` keyed by name, so
        they survive every reload. An empty or failed supplier keeps the full
        chain rather than starving the organism.
        """
        if self.active_names is None:
            return list(self.providers)
        try:
            active = {str(name) for name in self.active_names()}
        except Exception:
            log.exception("provider chain supplier failed; keeping the current chain")
            return list(self.providers)
        if not active:
            return list(self.providers)
        selected = [provider for provider in self.providers if self._name(provider) in active]
        if not selected:
            return list(self.providers)
        names = tuple(self._name(provider) for provider in selected)
        if names != self._active_snapshot:
            self._active_snapshot = names
            self._log("provider_chain_reloaded", {"providers": list(names)})
        return selected

    def complete(self, messages: Sequence[Message], *, max_tokens: int, tools: Sequence[ToolSchema] = ()) -> ModelTurn:
        failures: list[str] = []
        providers = list(self._active_providers()[:self.max_attempts])
        # An empty ladder keeps the legacy behaviour: one point strike per
        # provider using that provider's own timeout, without delays. A
        # configured ladder spends its rungs on round-robin strikes, so a slow
        # but healthy provider gets more room on later attempts.
        strikes: list[float | None] = list(self.timeout_ladder) if self.timeout_ladder else [None] * len(providers)
        position = 0
        strikes_done = 0
        first_strike_at: float | None = None
        struck_this_call: set[str] = set()
        # Time spent inside provider calls only. Sleeps between strikes and the
        # ladder deadline wait are infrastructure, not model thinking, so they
        # are excluded from the budget the reactor charges.
        model_seconds = 0.0

        def _select() -> LLMProvider | None:
            """Pick the next eligible provider, advancing the round-robin."""
            nonlocal position
            for _ in range(len(providers)):
                candidate = providers[position % len(providers)]
                position += 1
                name = self._name(candidate)
                # A provider already struck in this call stays eligible for the
                # remaining rungs; a cooldown only gates the next cycle.
                if name in struck_this_call:
                    return candidate
                blocked_until = (self._blocked_until or {}).get(name, 0.0)
                if blocked_until > time.time():
                    self._log("fallback_skipped", {"provider": name, "reason": "cooldown", "remaining_seconds": round(blocked_until - time.time(), 1)})
                    continue
                return candidate
            return None

        waited_for_cooldown = False
        for strike, rung in enumerate(strikes):
            if first_strike_at is not None and self.ladder_deadline_seconds > 0:
                elapsed = time.monotonic() - first_strike_at
                if elapsed > self.ladder_deadline_seconds:
                    self._log("fallback_ladder_deadline", {"elapsed_seconds": round(elapsed, 1), "deadline_seconds": self.ladder_deadline_seconds, "attempts": strikes_done})
                    break
            provider = _select()
            if provider is None and not waited_for_cooldown:
                # Nothing was even tried before the abort. If the chain itself
                # said when it can answer and that instant is near, one bounded
                # wait preserves the episode that would otherwise be discarded
                # with zero strikes. See _wait_out_short_cooldown.
                waited_for_cooldown = True
                self._wait_out_short_cooldown(strikes_done)
                provider = _select()
            if provider is None:
                # Every candidate is inside its cooldown. Breaking here used to
                # raise "all providers failed after 0 strikes: " with an empty
                # reason and the LATEST cooldown; name the blocked providers and
                # the earliest recovery point instead, so the reason stays
                # diagnosable and the caller backs off until the chain can
                # actually answer again. The "all providers failed" substring is
                # load-bearing for reactor failure classification, so it stays.
                blocked = self._blocked_remaining()
                cooling = "; ".join(f"{name}={seconds}s" for name, seconds in sorted(blocked.items()))
                message = (
                    f"all providers failed after {strikes_done} attempts: "
                    f"every provider is in cooldown ({cooling})"
                )  # the exact pair is_chain_wide_cooldown_abort matches; keep them in step
                self._log("fallback_all_cooling", {"attempts": strikes_done, "blocked_remaining": blocked})
                raise ProviderError(
                    message,
                    category="unavailable",
                    retryable=True,
                    cooldown_seconds=min(blocked.values(), default=MIN_PROVIDER_COOLDOWN_SECONDS),
                )
            name = self._name(provider)
            if strike and self.timeout_ladder and self.strike_delay_seconds > 0:
                time.sleep(self.strike_delay_seconds)
            if first_strike_at is None:
                first_strike_at = time.monotonic()
            strikes_done += 1
            struck_this_call.add(name)
            self._log("fallback_attempt", {"provider": name, "attempt": strike + 1, "attempts": len(strikes), "timeout_seconds": rung})
            try:
                strike_started = time.monotonic()
                result = self._strike(provider, messages, max_tokens=max_tokens, tools=tools, timeout_seconds=rung)
                model_seconds += max(0.0, time.monotonic() - strike_started)
                result.model_seconds = model_seconds
                # The planner's provider call passes through here and nowhere
                # else, so its stop reason has to be recorded on this line: a
                # reply cut off at the output ceiling leaves no other trace.
                truncated = result.finish_reason == "length"
                # ``max_tokens`` travels here as the *requested* ceiling, which
                # is not necessarily the one that was enforced: a provider may
                # refuse more than its own limit, or spend the budget on its
                # reasoning channel. Recording the requested value beside the
                # stop reason is what makes a cut-off reply diagnosable.
                # Measured: three consecutive planner generations
                # ended at exactly ``completion_tokens`` 2240 with
                # ``finish_reason='length'`` and ``text_chars`` 0 while the
                # planner had asked for 16384, and no record said how far apart
                # those two numbers were, so the operator instruction to raise
                # the knob could neither be confirmed nor refuted.
                self._log(
                    "fallback_selected",
                    {
                        "provider": name,
                        "attempt": strike + 1,
                        "completion_tokens": result.completion_tokens,
                        "finish_reason": result.finish_reason,
                        "requested_max_tokens": max_tokens,
                        "reasoning_chars": len(result.reasoning_content),
                    },
                )
                if truncated:
                    # A distinct event rather than another payload field: the
                    # planner already gets ``planner_output_truncated``, but a
                    # cut-off *episode* turn left no record a reader could find
                    # without scanning every ``provider_response``. This event
                    # carries the three numbers that decide whether the ceiling
                    # or the request is at fault.
                    self._log(
                        "provider_output_truncated",
                        {
                            "provider": name,
                            "attempt": strike + 1,
                            "completion_tokens": result.completion_tokens,
                            "requested_max_tokens": max_tokens,
                            "reasoning_chars": len(result.reasoning_content),
                            "text_chars": len(result.text),
                        },
                    )
                if strike:
                    # A retry that answers is the one outcome no other durable
                    # event records: ``fallback_failure`` is durable but says
                    # nothing about the recovery, and the successful selection
                    # below is indistinguishable from a first-strike answer.
                    # The retry-amplification paper names exactly this quantity
                    # -- success rate after retry -- as instrumentation without
                    # which a retry storm cannot be reconstructed after the
                    # fact (arXiv:2608.25403, sec. 8.1). Measured on the live
                    # database it is rare: 14 recovering calls against 2623
                    # first-strike selections, so it cannot flood the log.
                    self._log(
                        "fallback_retry_recovered",
                        {
                            "provider": name,
                            "attempt": strike + 1,
                            "attempts": len(strikes),
                            "prior_failures": list(failures),
                        },
                    )
                self._clear_cooldown(name)
                return result
            except ProviderError as exc:
                failures.append(f"{name}[{exc.category}]: {exc}")
                cooldown = max(0.0, exc.cooldown_seconds)
                if exc.retryable:
                    # A zero cooldown must never mean "retry the same dead
                    # provider on the very next cycle".
                    cooldown = max(cooldown, MIN_PROVIDER_COOLDOWN_SECONDS)
                    blocked = self._blocked_until if self._blocked_until is not None else {}
                    blocked[name] = time.time() + cooldown
                    self._save_state()
                else:
                    cooldown = 0.0
                if exc.category == "invalid_response":
                    # A malformed response is produced by ONE provider, so it is
                    # provider-local, not chain-fatal: measured live, a single
                    # openrouter "malformed streamed tool call" aborted an episode
                    # whose other providers were never struck. Cool this
                    # provider and keep the ladder going; if every provider
                    # answers malformed, the loop still ends in the aggregate
                    # "all providers failed" error below instead of retrying
                    # forever. invalid_request and configuration stay fatal
                    # because those describe a broken request, not a broken
                    # provider.
                    cooldown = max(cooldown, MIN_PROVIDER_COOLDOWN_SECONDS)
                    blocked = self._blocked_until if self._blocked_until is not None else {}
                    blocked[name] = time.time() + cooldown
                    self._save_state()
                self._log("fallback_failure", {"provider": name, "category": exc.category, "retryable": exc.retryable, "cooldown_seconds": cooldown, "attempt": strike + 1, "error": str(exc)[:500]})
                if exc.category in {"invalid_request", "configuration"} or (not exc.retryable and exc.category != "invalid_response"):
                    raise
            except Exception as exc:
                failures.append(f"{name}[unexpected]: {exc}")
                self._log("fallback_failure", {"provider": name, "category": "unexpected", "retryable": True, "attempt": strike + 1, "error": str(exc)[:500]})
        blocked_remaining = self._blocked_remaining()
        message = f"all providers failed after {strikes_done} attempts: {'; '.join(failures)}"
        if blocked_remaining:
            message += f" | blocked_until={blocked_remaining}"
        # The earliest remaining cooldown is when the chain can answer again;
        # the latest would over-sleep every provider that recovered sooner.
        raise ProviderError(message, category="unavailable", retryable=True, cooldown_seconds=min(blocked_remaining.values(), default=0.0))

    def _wait_out_short_cooldown(self, strikes_done: int) -> float:
        """Back off until the chain's own declared recovery point, if it is near.

        Every provider sitting inside its cooldown is not a failure of any one
        of them: it is a statement about WHEN the chain can answer, and this
        class already computes that instant (``blocked_until``). Raising
        immediately spends the whole episode on that statement and discards the
        work already inside it. Measured on the live database: 5 of the 21
        needs_recovery runs died with ZERO strikes attempted purely because the
        earliest recorded cooldown had 12.2-30.0s left, two of them after 22 and
        41 verified steps.

        The wait is deliberately narrow. It fires only when this call has not
        struck a single provider yet (the measured defect), only while the
        earliest horizon is no longer than ``max_wait_for_cooldown_seconds``
        -- the same magnitude as MIN_PROVIDER_COOLDOWN_SECONDS, i.e. the
        smallest backoff this class already considers meaningful -- and only
        once per call. A longer horizon stays a decision for the caller: the
        reactor owns cross-run scheduling, with its own jittered backoff.
        """
        if self.max_wait_for_cooldown_seconds <= 0 or strikes_done:
            return 0.0
        blocked = self._blocked_remaining()
        if not blocked:
            return 0.0
        horizon = min(blocked.values())
        if horizon > self.max_wait_for_cooldown_seconds:
            self._log(
                "fallback_cooldown_wait_skipped",
                {"remaining_seconds": horizon, "bound_seconds": self.max_wait_for_cooldown_seconds},
            )
            return 0.0
        self._log(
            "fallback_cooldown_wait",
            {"remaining_seconds": horizon, "bound_seconds": self.max_wait_for_cooldown_seconds},
        )
        time.sleep(horizon)
        return horizon

    def _strike(self, provider: LLMProvider, messages: Sequence[Message], *, max_tokens: int, tools: Sequence[ToolSchema], timeout_seconds: float | None) -> ModelTurn:
        if timeout_seconds is not None and getattr(provider, "accepts_timeout_override", False):
            # The override is an optional capability, not part of the protocol:
            # the flag above is the contract, so the widened call is cast.
            strike = cast(Callable[..., ModelTurn], provider.complete)
            return strike(messages, max_tokens=max_tokens, tools=tools, timeout_seconds=timeout_seconds)
        return provider.complete(messages, max_tokens=max_tokens, tools=tools)

    def health_probe(self) -> dict[str, object]:
        details: dict[str, object] = {}
        active = None
        now = time.time()
        for provider in self._active_providers():
            name = self._name(provider)
            blocked_until = (self._blocked_until or {}).get(name, 0.0)
            if blocked_until > now:
                # Local wall-clock state only: no network probe for a provider
                # we already know is blocked.
                result: dict[str, object] = {"ok": False, "blocked": True, "cooldown_remaining_seconds": round(blocked_until - now, 1)}
            else:
                probe = getattr(provider, "health_probe", None)
                try:
                    result = dict(probe()) if probe else {"ok": True}
                except Exception as exc:
                    log.warning("provider health probe failed", extra={"provider": name}, exc_info=True)
                    result = {"ok": False, "error": str(exc)[:500]}
                result["blocked"] = False
                result["cooldown_remaining_seconds"] = 0.0
                if active is None and result.get("ok"):
                    active = name
            details[name] = result
        return {"ok": active is not None, "active_provider": active, "blocked_until": self._blocked_remaining(), "providers": details}
