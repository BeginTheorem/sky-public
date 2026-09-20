from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..models import ModelTurn
from ..provider import Message, ToolSchema
from .errors import ProviderError
from .key_rotator import KeyRotator
from .openai_compatible import OpenAICompatibleProvider

MIN_PROVIDER_COOLDOWN_SECONDS = 30.0


@dataclass(slots=True)
class OllamaCloudProvider:
    base_url: str
    keys: list[str]
    model: str
    cooldown_seconds: float = 60.0
    max_attempts: int = 3
    timeout_seconds: float = 120.0
    max_output_tokens: int | None = 8192
    network_cooldown_seconds: float = 60.0
    permanent_backoff_seconds: float = KeyRotator.PERMANENT_BACKOFF_SECONDS
    _rotator: KeyRotator = field(init=False, repr=False)

    accepts_timeout_override = True

    def __post_init__(self) -> None:
        self._rotator = KeyRotator(self.keys, self.cooldown_seconds, permanent_backoff_seconds=self.permanent_backoff_seconds)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def key_state(self) -> dict[str, object]:
        return self._rotator.to_dict()

    def restore_key_state(self, state: object) -> None:
        self._rotator.restore(state)

    def complete(self, messages: Sequence[Message], *, max_tokens: int, tools: Sequence[ToolSchema] = (), timeout_seconds: float | None = None) -> ModelTurn:
        if not self._rotator.size:
            raise ProviderError("Ollama has no configured API keys", category="configuration", provider=self.name, model=self.model)
        errors: list[str] = []
        cooldown = 0.0
        for _ in range(max(1, self.max_attempts)):
            slot = self._rotator.next_available()
            if slot is None:
                # Every key is cooling down: report when the earliest returns so
                # the fallback chain backs off too instead of retrying at once.
                earliest = self._rotator.earliest_available_in()
                errors.append(f"all {self._rotator.total} keys cooling down (earliest in {earliest:.0f}s)")
                cooldown = max(cooldown, earliest, MIN_PROVIDER_COOLDOWN_SECONDS)
                break
            client = OpenAICompatibleProvider(
                self.name,
                self.base_url,
                slot.key,
                self.model,
                timeout_seconds=timeout_seconds or self.timeout_seconds,
                max_output_tokens=self.max_output_tokens,
                network_cooldown_seconds=self.network_cooldown_seconds,
            )
            try:
                return client.complete(messages, max_tokens=max_tokens, tools=tools)
            except ProviderError as exc:
                errors.append(str(exc)[:500])
                self._rotator.mark_failed(slot.index, permanent=exc.category == "auth", cooldown_seconds=exc.cooldown_seconds)
                if exc.category == "quota":
                    cooldown = max(cooldown, float(exc.cooldown_seconds or 0.0))
                # A bad key is not retryable at the fallback level, but the
                # rotator has already marked it permanent so other keys are
                # still worth trying.
                if not exc.retryable and exc.category != "auth":
                    raise
        raise ProviderError(
            f"Ollama attempts exhausted: {'; '.join(errors)}",
            category="unavailable",
            retryable=True,
            cooldown_seconds=max(cooldown, MIN_PROVIDER_COOLDOWN_SECONDS),
            provider=self.name,
            model=self.model,
        )

    def health_probe(self) -> dict[str, object]:
        status = self._rotator.status()
        available = sum(1 for item in status if item["available"])
        return {"ok": available > 0, "provider": self.name, "model": self.model, "available_keys": available, "total_keys": len(status), "keys": status}
