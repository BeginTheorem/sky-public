from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, NoReturn, cast

from ..models import ModelTurn
from ..provider import LLMProvider, Message, ToolSchema
from .errors import ProviderError

log = logging.getLogger("skynet.providers.fallback")

MIN_PROVIDER_COOLDOWN_SECONDS = 30.0

# A rung that cannot pay does not heal by being retried: the balance is the
# owner's to refill, not the chain's to wait out. It is still provider-LOCAL, so
# the ladder continues past it (see ``_is_fatal``) -- this cooldown only keeps the
# chain from re-striking the empty wallet on every call in the meantime. Half an
# hour: long enough that a ten-rung ladder does not burn its calls on a broke
# rung, short enough that a top-up is picked up in the same session.
BILLING_PROVIDER_COOLDOWN_SECONDS = 1800.0

#: A rung that fails again and again multiplies its OWN cooldown, so a dead rung
#: steps further out of the way instead of being re-elected at the top of every
#: call. Measured live 2026-10-16/17: with the 2-10s category cooldowns the ladder
#: picked the hanging NVIDIA rung first on every attempt for hours -- twelve
#: failures an hour, all on that one rung -- while the three healthy rungs behind
#: it were never reached. The streak resets the moment the rung answers, so a
#: healthy provider that hits one 500 is not punished for it.
FAILURE_STREAK_COOLDOWN_GROWTH = 2.0
MAX_ESCALATED_COOLDOWN_SECONDS = 900.0


def is_chain_fatal(category: str, *, retryable: bool) -> bool:
    """Whether a provider fault ends the WHOLE ladder, by category.

    The single authority for the judgment ``FallbackProvider._is_fatal`` applies, so a
    reader that only has the stored classification (the reactor's per-run events) can
    reach the same verdict without a second list drifting from this one. A chain-fatal
    fault is re-raised straight from the rung, which is why such a run's failure text
    names the model and never the chain.
    """
    return category in {"invalid_request", "configuration"} or (
        not retryable and category not in {"invalid_response", "prompt_too_long", "billing"}
    )

# The terminal texts the chain can raise, as recorded in ``run_results.failure``.
# They name DIFFERENT failure classes and are disjoint on the live ledger
# (measured 2026-09-25, 174 committed runs): 5 runs died with the cooldown text
# below -- the terminal model call was refused before it reached any provider --
# and 12 died with a ladder that had struck at least one provider first. A reader
# that only matches the shared "all providers failed" substring cannot separate
# them, which is why the exact pairs are named here.
CHAIN_WIDE_COOLDOWN_MARKER = "every provider is in cooldown"
ZERO_STRIKE_MARKER = "all providers failed after 0 attempts"
# The terminal text of the patient cycle (owner decision 2026-09-27): the whole
# chain failed ``dead_passes`` times in a row. The reactor keys its 30-minute
# lockout sleep on this marker, the same way it keys the zero-strike abort on
# CHAIN_WIDE_COOLDOWN_MARKER.
DEAD_PASS_MARKER = "dead passes"  # noqa: S105 -- an event-text marker, not a credential


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


def is_chain_dead_terminal(failure: str) -> bool:
    """Whether a run died because the whole chain failed every pass in a row.

    The patient cycle (the ReAct path) retries a dead chain in place instead of
    ending the episode: only after ``dead_passes`` consecutive whole-chain
    failures does it raise. That terminal text carries ``DEAD_PASS_MARKER`` and
    keeps the "all providers failed" substring the reactor's provider-failure
    classification already matches, so a reader can tell it apart and the
    reactor can jump straight to the lockout sleep.
    """
    text = str(failure or "")
    return "all providers failed" in text and DEAD_PASS_MARKER in text


@dataclass(slots=True)
class FallbackProvider:
    providers: list[LLMProvider]
    max_attempts: int = 0
    # The patient cycle (owner decision 2026-09-27), used by the ReAct path.
    # Each provider gets ``attempts_per_provider`` grouped strikes separated by
    # the escalating ``attempt_waits``; a provider that fails its last attempt is
    # locked for ``provider_block_seconds``; the whole chain is then re-passed in
    # place, and only ``dead_passes`` consecutive whole-chain failures end the
    # episode. The library defaults keep the historical single-strike behaviour
    # so callers with their own short budgets (memory loop, planner) are
    # unchanged; ``build_provider()`` wires the operator's numbers from env.
    attempts_per_provider: int = 1
    attempt_waits: tuple[float, ...] = ()
    provider_block_seconds: float = 300.0
    dead_passes: int = 1
    # Per-attempt budgets: a streaming rung is bounded by SILENCE between chunks
    # (the transport refreshes the idle timer on every received chunk), while a
    # non-streaming rung can only be bounded by the whole wait -- a provider that
    # thinks for minutes and then sends everything at once must not be cut off.
    attempt_timeout_seconds: float = 900.0
    stream_idle_seconds: float = 120.0
    # Called while the patient cycle sleeps (slices of ``wait_slice_seconds``),
    # so the reactor can keep the run heartbeat fresh: the stale-run watchdog
    # watches the durable heartbeat and would otherwise kill a run that is
    # deliberately waiting out provider cooldowns. Same shape as ``event_logger``,
    # and providers must not import the store.
    wait_callback: Callable[[float], None] | None = None
    wait_slice_seconds: float = 5.0
    _blocked_until: dict[str, float] | None = None
    #: Consecutive failures per rung, persisted beside the cooldowns: it is what
    #: grows a repeat offender's cooldown and it must survive a restart, or a
    #: hanging rung starts at the short category cooldown again every time.
    _failure_streak: dict[str, int] | None = None
    event_logger: Callable[[str, dict[str, object]], None] | None = None
    state_path: Path | None = None
    # Hard wall-clock cap for one patient call (0 = no cap). It bounds the
    # "just wait" loop so a silently hanging chain cannot hold the organism
    # forever; the owner's 3-dead-pass rule ends it much earlier in practice.
    call_deadline_seconds: float = 0.0
    # How long a zero-strike chain-wide cooldown may be waited out once per
    # non-patient call before admitting the chain is down. 0.0 keeps the
    # historical behaviour (raise immediately).
    max_wait_for_cooldown_seconds: float = 0.0
    # Optional supplier of the currently active provider names. Re-read at every
    # call so enabling or disabling the paid provider takes effect without a
    # process restart; provider instances (and their cooldowns) are preserved.
    active_names: Callable[[], list[str]] | None = None
    # Last active view that was logged, used only to journal real chain changes.
    _active_snapshot: tuple[str, ...] = ()

    # Optional callable that picks an orchestration strategy for this call.
    # Return "parallel" to strike every unblocked provider concurrently and
    # answer with the first successful rung (wall-clock = fastest answering
    # provider instead of the sum of failures). Return anything else (or None,
    # the default) to keep the historical one-strike sequential path
    # byte-for-byte. Re-read at every call so the operator can switch strategy
    # without a restart.
    strategy_selector: Callable[[], str] | None = None

    # Advertises the patient call to the ReAct runner without making every fake
    # provider in the tests accept the ``patient`` keyword.
    supports_patient_wait: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("FallbackProvider requires at least one provider")
        if self.max_attempts <= 0:
            self.max_attempts = len(self.providers)
        if self.attempts_per_provider < 1:
            self.attempts_per_provider = 1
        # Retries live here, not in the ReAct loop: ``ReActConfig.provider_retries``
        # stays 0. The patient cycle is therefore the only retry mechanism the
        # ReAct path has, and the grouped attempts above are what absorb a
        # transient fault instead of ending the episode.
        self.attempt_waits = self._normalised_waits(self.attempts_per_provider)
        self._blocked_until = {}
        self._failure_streak = {}
        self._load_state()

    def _normalised_waits(self, attempts: int) -> tuple[float, ...]:
        """Exactly ``attempts - 1`` waits: pad short ladders with the last value."""
        waits = tuple(float(wait) for wait in self.attempt_waits)
        needed = max(0, attempts - 1)
        if len(waits) >= needed:
            return waits[:needed]
        if not waits:
            return (0.0,) * needed
        return waits + (waits[-1],) * (needed - len(waits))

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
        streak = self._failure_streak if self._failure_streak is not None else {}
        stored_streak = stored.get("failure_streak")
        if isinstance(stored_streak, dict):
            for key, value in stored_streak.items():
                name = str(key)
                if name not in names:
                    continue
                try:
                    streak[name] = max(0, int(value))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
        self._failure_streak = streak
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
        if self._failure_streak:
            payload["failure_streak"] = self._failure_streak
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

    def set_wait_callback(self, callback: Callable[[float], None] | None) -> None:
        """Install the heartbeat pulse used while the patient cycle sleeps."""
        self.wait_callback = callback

    def _blocked_remaining(self) -> dict[str, float]:
        now = time.time()
        remaining: dict[str, float] = {}
        for name, until in (self._blocked_until or {}).items():
            value = until - now
            if value > 0:
                remaining[name] = round(value, 1)
        return remaining

    def _earliest_block(self) -> float:
        """The raw (unrounded) horizon the chain itself declared, or 0.0.

        ``_blocked_remaining`` rounds for the log; a sleep must use the exact
        value, or a sub-0.1s horizon rounds to 0.0 and the wait spins instead of
        sleeping.
        """
        now = time.time()
        horizons = [until - now for until in (self._blocked_until or {}).values() if until - now > 0]
        return min(horizons) if horizons else 0.0

    def _is_blocked(self, name: str) -> bool:
        return (self._blocked_until or {}).get(name, 0.0) > time.time()

    def _remaining(self, name: str) -> float:
        return round(max(0.0, (self._blocked_until or {}).get(name, 0.0) - time.time()), 1)

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
        streak = self._failure_streak if self._failure_streak is not None else {}
        if name in blocked or name in streak:
            blocked.pop(name, None)
            # The rung answered, so its failure streak is history: the next
            # failure starts from the base cooldown instead of inheriting the
            # escalation a previous outage earned.
            streak.pop(name, None)
            self._failure_streak = streak
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

    # Forcing one tool is an optional capability of the members (the closing turn asks for it).
    # The chain advertises it because it forwards the choice to whichever member supports it, and
    # silently drops it for a member that does not.
    accepts_tool_choice = True

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        tools: Sequence[ToolSchema] = (),
        patient: bool = False,
        tool_choice: dict[str, Any] | None = None,
    ) -> ModelTurn:
        """Answer through the chain.

        ``patient=True`` is the ReAct path (owner decision 2026-09-27): the chain
        is retried in place -- each provider gets ``attempts_per_provider``
        grouped strikes separated by escalating waits, a provider that fails its
        last strike is locked, and a fully dead chain is waited out and re-passed
        instead of ending the episode. The default keeps the historical
        one-strike-per-provider behaviour for callers with their own short
        budgets (memory loop, planner).
        """
        providers = list(self._active_providers()[:self.max_attempts])
        if patient:
            return self._complete_patient(providers, messages, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice)
        if self.strategy_selector is not None and self.strategy_selector() == "parallel":
            return self._complete_parallel(providers, messages, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice)
        return self._complete_sequential(providers, messages, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice)

    # -- non-patient path (memory loop, planner, any short-budget caller) ------

    def _complete_sequential(self, providers: list[LLMProvider], messages: Sequence[Message], *, max_tokens: int, tools: Sequence[ToolSchema], tool_choice: dict[str, Any] | None = None) -> ModelTurn:
        failures: list[str] = []
        model_seconds = 0.0
        candidates: list[LLMProvider] = []
        for provider in providers:
            name = self._name(provider)
            if self._is_blocked(name):
                self._log("fallback_skipped", {"provider": name, "reason": "cooldown", "remaining_seconds": self._remaining(name)})
                continue
            candidates.append(provider)
        if not candidates and self._wait_out_short_cooldown(0):
            candidates = [provider for provider in providers if not self._is_blocked(self._name(provider))]
        if not candidates:
            # Every candidate is inside its cooldown. Name the blocked providers
            # and the earliest recovery point instead of raising an empty reason,
            # so the reason stays diagnosable and the caller backs off until the
            # chain can actually answer again. The "all providers failed"
            # substring is load-bearing for reactor failure classification.
            blocked = self._blocked_remaining()
            cooling = "; ".join(f"{name}={seconds}s" for name, seconds in sorted(blocked.items()))
            message = f"all providers failed after 0 attempts: every provider is in cooldown ({cooling})"
            self._log("fallback_all_cooling", {"attempts": 0, "blocked_remaining": blocked})
            raise ProviderError(
                message,
                category="unavailable",
                retryable=True,
                cooldown_seconds=min(blocked.values(), default=MIN_PROVIDER_COOLDOWN_SECONDS),
            )
        attempts = max(1, self.attempts_per_provider)
        waits = self._normalised_waits(attempts)
        for _position, provider in enumerate(candidates):
            name = self._name(provider)
            for attempt in range(1, attempts + 1):
                if attempt > 1:
                    wait = waits[attempt - 2]
                    if wait > 0:
                        time.sleep(wait)
                self._log("fallback_attempt", {"provider": name, "attempt": attempt, "attempts": attempts})
                try:
                    strike_started = time.monotonic()
                    result = self._strike(provider, messages, max_tokens=max_tokens, tools=tools, timeout_seconds=None, tool_choice=tool_choice)
                    model_seconds += max(0.0, time.monotonic() - strike_started)
                    result.model_seconds = model_seconds
                    self._record_selection(name, attempt, attempts, result, max_tokens)
                    if attempt > 1:
                        # A retry that answers is the one outcome no other durable
                        # event records: ``fallback_failure`` is durable but says
                        # nothing about the recovery, and the successful selection
                        # below is indistinguishable from a first-strike answer.
                        self._log("fallback_retry_recovered", {"provider": name, "attempt": attempt, "attempts": attempts, "prior_failures": list(failures)})
                    self._clear_cooldown(name)
                    return result
                except ProviderError as exc:
                    failures.append(f"{name}[{exc.category}]: {exc}")
                    self._log("fallback_failure", {"provider": name, "category": exc.category, "retryable": exc.retryable, "cooldown_seconds": exc.cooldown_seconds, "attempt": attempt, "error": str(exc)[:500]})
                    if self._is_fatal(exc):
                        raise
                    if attempt < attempts:
                        continue
                    self._cool_after_failure(name, exc)
                except Exception as exc:
                    failures.append(f"{name}[unexpected]: {exc}")
                    self._log("fallback_failure", {"provider": name, "category": "unexpected", "retryable": True, "attempt": attempt, "error": str(exc)[:500]})
                    break
        blocked_remaining = self._blocked_remaining()
        message = f"all providers failed after {len(candidates)} attempts: {'; '.join(failures)}"
        if blocked_remaining:
            message += f" | blocked_until={blocked_remaining}"
        # The earliest remaining cooldown is when the chain can answer again;
        # the latest would over-sleep every provider that recovered sooner.
        raise ProviderError(message, category="unavailable", retryable=True, cooldown_seconds=min(blocked_remaining.values(), default=0.0))

    # -- non-patient parallel path (latency-optimized: strike every unblocked rung concurrently) ----

    def _complete_parallel(self, providers: list[LLMProvider], messages: Sequence[Message], *, max_tokens: int, tools: Sequence[ToolSchema], tool_choice: dict[str, Any] | None = None) -> ModelTurn:
        """Strike every unblocked provider at once; answer with the first success.

        Wall-clock is the fastest answering rung instead of the sum of every
        failing one. Failures are cooled and each successful selection is
        journaled under the same lock; the whole call returns as soon as any
        provider answers. When no provider answers, the call raises the same
        terminal error the sequential path raises so reactor failure
        classification is unchanged.
        """
        candidates = [provider for provider in providers if not self._is_blocked(self._name(provider))]
        if not candidates:
            # Delegate to the sequential path so the "all providers failed after 0 attempts"
            # terminal text stays load-bearing for reactor failure classification.
            return self._complete_sequential(providers, messages, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice)

        successes: list[ModelTurn] = []
        failures: list[str] = []
        lock = threading.Lock()
        answered = threading.Event()

        def strike(position: int, provider: LLMProvider) -> None:
            name = self._name(provider)
            timeout = self._attempt_timeout(provider)
            try:
                strike_started = time.monotonic()
                result = self._strike(provider, messages, max_tokens=max_tokens, tools=tools, timeout_seconds=timeout, tool_choice=tool_choice)
                result.model_seconds = max(0.0, time.monotonic() - strike_started)
                with lock:
                    self._record_selection(name, position + 1, len(candidates), result, max_tokens)
                    self._clear_cooldown(name)
                    if not answered.is_set():
                        successes.append(result)
                        answered.set()
            except ProviderError as exc:
                with lock:
                    failures.append(f"{name}[{exc.category}]: {exc}")
                self._cool_after_failure(name, exc)
                self._log("fallback_failure", {"provider": name, "category": exc.category, "retryable": exc.retryable, "cooldown_seconds": exc.cooldown_seconds, "attempt": position + 1, "error": str(exc)[:500]})
            except Exception as exc:
                with lock:
                    failures.append(f"{name}[unexpected]: {exc}")
                self._log("fallback_failure", {"provider": name, "category": "unexpected", "retryable": True, "attempt": position + 1, "error": str(exc)[:500]})

        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            futures = [pool.submit(strike, index, provider) for index, provider in enumerate(candidates)]
            for future in as_completed(futures):
                future.result()  # surface any error raised inside strike itself

        if successes:
            return successes[0]

        blocked_remaining = self._blocked_remaining()
        message = f"all providers failed after {len(candidates)} attempts: {'; '.join(failures)}"
        if blocked_remaining:
            message += f" | blocked_until={blocked_remaining}"
        raise ProviderError(message, category="unavailable", retryable=True, cooldown_seconds=min(blocked_remaining.values(), default=0.0))

    # -- patient path (ReAct: wait the network out, never end the episode) -----

    def _complete_patient(self, providers: list[LLMProvider], messages: Sequence[Message], *, max_tokens: int, tools: Sequence[ToolSchema], tool_choice: dict[str, Any] | None = None) -> ModelTurn:
        """Owner's retry scheme.

        One provider at a time, ``attempts_per_provider`` grouped strikes with
        ``attempt_waits`` between them; after the last failure the provider is
        locked (5 min by default, or the cooldown the provider itself stated).
        When the whole chain is locked the pass is repeated after the earliest
        lock expires -- the call keeps waiting instead of ending the run -- and
        only ``dead_passes`` consecutive dead passes raise the terminal error the
        reactor turns into a 30-minute lockout sleep.
        """
        failures: list[str] = []
        attempts = max(1, self.attempts_per_provider)
        waits = self._normalised_waits(attempts)
        deadline = None if self.call_deadline_seconds <= 0 else time.monotonic() + self.call_deadline_seconds
        model_seconds = 0.0
        passes = 0
        while True:
            passes += 1
            attempts_made = 0
            for provider in providers:
                name = self._name(provider)
                if self._is_blocked(name):
                    self._log("fallback_skipped", {"provider": name, "reason": "cooldown", "remaining_seconds": self._remaining(name)})
                    continue
                for attempt in range(attempts):
                    if attempt and waits[attempt - 1] > 0 and not self._pulse_sleep(waits[attempt - 1], deadline):
                        self._raise_chain_dead(passes, failures, "call deadline reached")
                    if deadline is not None and time.monotonic() >= deadline:
                        self._raise_chain_dead(passes, failures, "call deadline reached")
                    attempts_made += 1
                    timeout = self._attempt_timeout(provider)
                    self._log("fallback_attempt", {"provider": name, "attempt": attempt + 1, "attempts": attempts, "pass": passes, "timeout_seconds": timeout})
                    try:
                        strike_started = time.monotonic()
                        result = self._strike(provider, messages, max_tokens=max_tokens, tools=tools, timeout_seconds=timeout, tool_choice=tool_choice)
                        model_seconds += max(0.0, time.monotonic() - strike_started)
                        result.model_seconds = model_seconds
                        self._record_selection(name, attempt + 1, attempts, result, max_tokens)
                        if attempt or passes > 1:
                            self._log("fallback_retry_recovered", {"provider": name, "attempt": attempt + 1, "attempts": attempts, "pass": passes, "prior_failures": list(failures)})
                        self._clear_cooldown(name)
                        return result
                    except ProviderError as exc:
                        failures.append(f"{name}[{exc.category}]: {exc}")
                        self._log("fallback_failure", {"provider": name, "category": exc.category, "retryable": exc.retryable, "cooldown_seconds": exc.cooldown_seconds, "attempt": attempt + 1, "pass": passes, "error": str(exc)[:500]})
                        if self._is_fatal(exc):
                            raise
                        if attempt + 1 >= attempts:
                            self._block_provider(name, exc)
                    except Exception as exc:
                        failures.append(f"{name}[unexpected]: {exc}")
                        self._log("fallback_failure", {"provider": name, "category": "unexpected", "retryable": True, "attempt": attempt + 1, "pass": passes, "error": str(exc)[:500]})
                        if attempt + 1 >= attempts:
                            self._block_provider(name)
            if attempts_made == 0:
                # Nothing was even tried: the whole chain sits in a cooldown.
                # "Every provider is in cooldown" is a statement about WHEN the
                # chain can answer, not a dead chain, so wait the declared
                # horizon out and pass again rather than counting a dead pass.
                # A horizon that cannot be waited out (already elapsed, or past
                # the call deadline) ends the call instead of spinning.
                if not self._wait_out_chain(deadline):
                    self._raise_chain_dead(passes, failures, "every provider is in cooldown")
                continue
            if passes >= max(1, self.dead_passes):
                break
            self._log("fallback_dead_pass", {"pass": passes, "attempts": attempts_made, "blocked_remaining": self._blocked_remaining()})
            if not self._wait_out_chain(deadline):
                break
        return self._raise_chain_dead(passes, failures, None)

    def _record_selection(self, name: str, attempt: int, attempts: int, result: ModelTurn, max_tokens: int) -> None:
        """Journal a successful strike, including a cut-off reply.

        ``max_tokens`` travels here as the *requested* ceiling, which is not
        necessarily the one that was enforced: a provider may refuse more than
        its own limit, or spend the budget on its reasoning channel. Recording
        the requested value beside the stop reason is what makes a cut-off reply
        diagnosable. Measured: three consecutive planner generations ended at
        exactly ``completion_tokens`` 2240 with ``finish_reason='length'`` and
        ``text_chars`` 0 while the planner had asked for 16384, and no record said
        how far apart those two numbers were, so the operator instruction to
        raise the knob could neither be confirmed nor refuted.
        """
        self._log(
            "fallback_selected",
            {
                "provider": name,
                "attempt": attempt,
                "attempts": attempts,
                "completion_tokens": result.completion_tokens,
                "finish_reason": result.finish_reason,
                "requested_max_tokens": max_tokens,
                "reasoning_chars": len(result.reasoning_content),
                "cost_tokens": result.total_tokens,
                "latency_seconds": result.model_seconds,
            },
        )
        if result.finish_reason == "length":
            # A distinct event rather than another payload field: the planner
            # already gets ``planner_output_truncated``, but a cut-off *episode*
            # turn left no record a reader could find without scanning every
            # ``provider_response``. This event carries the three numbers that
            # decide whether the ceiling or the request is at fault.
            self._log(
                "provider_output_truncated",
                {
                    "provider": name,
                    "attempt": attempt,
                    "completion_tokens": result.completion_tokens,
                    "requested_max_tokens": max_tokens,
                    "reasoning_chars": len(result.reasoning_content),
                    "text_chars": len(result.text),
                },
            )

    @staticmethod
    def _is_fatal(exc: ProviderError) -> bool:
        """A broken request or a bad credential is chain-fatal, not provider-local.

        Retrying it would fail identically everywhere: the owner's ladder applies
        to network/server/timeout faults, while ``invalid_request`` and
        ``configuration`` describe a broken request and abort the chain at once.
        ``invalid_response`` stays provider-local (a malformed reply is produced
        by ONE provider), ``auth`` is non-retryable so it aborts too, and
        ``prompt_too_long`` is the same payload that may still fit a
        longer-context rung -- so it skips this provider instead of killing the
        episode. ``billing`` joins that last group: an empty balance belongs to
        one rung, and the free tail behind it does not need money to answer.
        """
        return is_chain_fatal(exc.category, retryable=exc.retryable)

    def _escalated_cooldown(self, name: str, cooldown: float) -> float:
        """Grow a rung's cooldown by its consecutive-failure streak, capped.

        The owner's retry scheme gives each rung its grouped strikes, but the
        cooldown a category default opens (2-10s) is shorter than the time one
        hanging rung takes to fail, so election kept picking that same rung at the
        top of every call. Counting the streak turns a repeat offender into a
        rung that steps aside: 30s, 60s, 120s ... up to
        ``MAX_ESCALATED_COOLDOWN_SECONDS``. A cooldown the *provider* itself
        stated is never touched (the owner's "his 10 seconds" rule), which is why
        callers pass those straight through instead of landing here.
        """
        streak = 0
        state = self._failure_streak if self._failure_streak is not None else {}
        try:
            streak = int(state.get(name, 0))
        except (TypeError, ValueError):
            streak = 0
        streak += 1
        state[name] = streak
        self._failure_streak = state
        grown = max(cooldown, 0.0) * (FAILURE_STREAK_COOLDOWN_GROWTH ** (streak - 1))
        return max(cooldown, min(grown, MAX_ESCALATED_COOLDOWN_SECONDS))

    def _cool_after_failure(self, name: str, exc: ProviderError) -> None:
        """Non-patient cooling: a retryable failure brakes for at least the floor.

        A zero cooldown must never mean "retry the same dead provider on the very
        next cycle". ``invalid_response`` is cooled too even though it is
        non-retryable, because the ladder continues past it. ``billing`` gets the
        longest floor of the family: a rung whose key cannot pay answers
        identically on every strike until the owner refills it, so cooling it for
        thirty seconds would only buy thirty seconds of noise. A failure that
        repeats without an answer in between escalates (``_escalated_cooldown``),
        so the plate a failing rung occupies keeps growing.
        """
        if exc.retryable or exc.category in {"invalid_response", "billing"}:
            floor = BILLING_PROVIDER_COOLDOWN_SECONDS if exc.category == "billing" else MIN_PROVIDER_COOLDOWN_SECONDS
            cooldown = max(max(0.0, exc.cooldown_seconds), floor)
            if not exc.cooldown_explicit:
                cooldown = self._escalated_cooldown(name, cooldown)
            blocked = self._blocked_until if self._blocked_until is not None else {}
            blocked[name] = time.time() + cooldown
            self._save_state()

    def _block_provider(self, name: str, exc: ProviderError | None = None) -> float:
        """Lock a provider after its last grouped attempt in the patient cycle.

        The default lock is ``provider_block_seconds`` (5 min). A cooldown the
        provider ITSELF stated (a ``Retry-After``, a spent quota until the next
        UTC day) wins -- even when it is shorter than the default, which is the
        owner's "his 10 seconds" rule.
        """
        if exc is not None and exc.cooldown_explicit and exc.cooldown_seconds > 0:
            cooldown = float(exc.cooldown_seconds)
        else:
            # Not a value the provider stated, so a rung that keeps failing its
            # grouped strikes is benched longer each time instead of returning to
            # the top of the next call.
            cooldown = self._escalated_cooldown(name, max(0.0, self.provider_block_seconds))
        blocked = self._blocked_until if self._blocked_until is not None else {}
        blocked[name] = time.time() + cooldown
        self._save_state()
        self._log("fallback_provider_blocked", {"provider": name, "cooldown_seconds": round(cooldown, 1), "stated_by_provider": bool(exc is not None and exc.cooldown_explicit)})
        return cooldown

    def _attempt_timeout(self, provider: LLMProvider) -> float:
        """Per-attempt budget: idle silence for a streamer, the whole wait otherwise."""
        if getattr(provider, "streams", False):
            return self.stream_idle_seconds
        return self.attempt_timeout_seconds

    def _pulse_sleep(self, seconds: float, deadline: float | None) -> bool:
        """Sleep in slices, pulsing the heartbeat; False when the deadline hits."""
        remaining = max(0.0, seconds)
        while remaining > 0:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            slice_seconds = remaining if self.wait_slice_seconds <= 0 else min(self.wait_slice_seconds, remaining)
            self._pulse(slice_seconds)
            time.sleep(slice_seconds)
            remaining -= slice_seconds
        return not (deadline is not None and time.monotonic() >= deadline)

    def _pulse(self, seconds: float) -> None:
        callback = self.wait_callback
        if callback is None:
            return
        try:
            callback(seconds)
        except Exception:
            log.warning("provider wait pulse failed", exc_info=True)

    def _wait_out_chain(self, deadline: float | None) -> bool:
        """Wait the earliest declared cooldown out, pulsing the heartbeat.

        False means there is nothing to wait for or the call deadline forbids it
        (the caller then ends the cycle with the terminal error).
        """
        horizon = self._earliest_block()
        if horizon <= 0:
            return False
        self._log("fallback_cooldown_wait", {"remaining_seconds": round(horizon, 1), "bound_seconds": None if deadline is None else round(max(0.0, deadline - time.monotonic()), 1)})
        return self._pulse_sleep(horizon, deadline)

    def _raise_chain_dead(self, passes: int, failures: list[str], note: str | None) -> NoReturn:
        """End the patient cycle with the terminal text the reactor keys on."""
        blocked = self._blocked_remaining()
        reason = "; ".join(failures) or "no provider answered"
        message = f"all providers failed after {passes} {DEAD_PASS_MARKER}: {reason}"
        if note:
            message += f" | {note}"
        if blocked:
            message += f" | blocked_until={blocked}"
        raise ProviderError(message, category="unavailable", retryable=True, cooldown_seconds=min(blocked.values(), default=0.0))

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

    def _strike(
        self,
        provider: LLMProvider,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        tools: Sequence[ToolSchema],
        timeout_seconds: float | None,
        tool_choice: dict[str, Any] | None = None,
    ) -> ModelTurn:
        # Both overrides are optional capabilities, not part of the protocol: the flags are the
        # contract, so the widened call is cast. A member that cannot force a tool simply is not
        # asked to -- the closing turn degrades to the tool being offered, as before.
        kwargs: dict[str, Any] = {"max_tokens": max_tokens, "tools": tools}
        if timeout_seconds is not None and getattr(provider, "accepts_timeout_override", False):
            kwargs["timeout_seconds"] = timeout_seconds
        if tool_choice is not None and getattr(provider, "accepts_tool_choice", False):
            kwargs["tool_choice"] = tool_choice
        return cast(Callable[..., ModelTurn], provider.complete)(messages, **kwargs)

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
