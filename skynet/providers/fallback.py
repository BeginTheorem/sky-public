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
        if self.event_logger is not None:
            self.event_logger(event, payload)

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
        for strike, rung in enumerate(strikes):
            if first_strike_at is not None and self.ladder_deadline_seconds > 0:
                elapsed = time.monotonic() - first_strike_at
                if elapsed > self.ladder_deadline_seconds:
                    self._log("fallback_ladder_deadline", {"elapsed_seconds": round(elapsed, 1), "deadline_seconds": self.ladder_deadline_seconds, "attempts": strikes_done})
                    break
            provider = None
            for _ in range(len(providers)):
                candidate = providers[position % len(providers)]
                position += 1
                name = self._name(candidate)
                # A provider already struck in this call stays eligible for the
                # remaining rungs; a cooldown only gates the next cycle.
                if name in struck_this_call:
                    provider = candidate
                    break
                blocked_until = (self._blocked_until or {}).get(name, 0.0)
                if blocked_until > time.time():
                    self._log("fallback_skipped", {"provider": name, "reason": "cooldown", "remaining_seconds": round(blocked_until - time.time(), 1)})
                    continue
                provider = candidate
                break
            if provider is None:
                # Every candidate is inside its cooldown. Name the blocked
                # providers and the earliest recovery point, so the reason
                # stays diagnosable and the caller backs off until the chain
                # can answer again. The "all providers failed" substring is
                # load-bearing for failure classification, so it stays.
                blocked = self._blocked_remaining()
                cooling = "; ".join(f"{name}={seconds}s" for name, seconds in sorted(blocked.items()))
                message = (
                    f"all providers failed after {strikes_done} attempts: "
                    f"every provider is in cooldown ({cooling})"
                )
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
                self._log("fallback_selected", {"provider": name, "attempt": strike + 1})
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
                self._log("fallback_failure", {"provider": name, "category": exc.category, "retryable": exc.retryable, "cooldown_seconds": cooldown, "attempt": strike + 1, "error": str(exc)[:500]})
                if exc.category in {"invalid_request", "invalid_response", "configuration"} or not exc.retryable:
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
