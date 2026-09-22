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

def tool_arguments(value: object) -> dict[str, object]:
    """Return a tool call's arguments as a mapping, or raise invalid_response.

    A provider may answer with a syntactically valid JSON payload that is not
    the object shape a tool call needs (``null``, a list, a string, a number).
    Measured (arXiv:2608.06790v1 fault taxonomy, value fault on tool call
    fields): such a turn parses cleanly, so the chain logs it as
    ``fallback_selected`` and the corrupt arguments reach the ReAct loop, where
    ``arguments.get(...)`` raises AttributeError inside every tool. Treating it
    as a provider-local malformed response cools that provider and lets the
    ladder continue, exactly like a JSON syntax error.
    """
    if not isinstance(value, dict):
        raise ProviderError(
            f"tool call arguments are not a JSON object: {type(value).__name__}",
            category="invalid_response",
            retryable=False,
        )
    return value
