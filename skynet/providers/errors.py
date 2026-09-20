from __future__ import annotations


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        category: str = "unknown",
        retryable: bool = False,
        cooldown_seconds: float = 0.0,
        provider: str = "unknown",
        model: str = "",
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.cooldown_seconds = cooldown_seconds
        self.provider = provider
        self.model = model
        self.http_status = http_status
