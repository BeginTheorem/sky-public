from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from .models import ModelTurn

Message = dict[str, Any]

ToolSchema = dict[str, Any]


class LLMProvider(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        tools: Sequence[ToolSchema] = (),
    ) -> ModelTurn:
        """Return one model turn.

        A per-call ``timeout_seconds`` override is deliberately *not* part of
        this contract: it is an optional capability. A provider that honours it
        sets ``accepts_timeout_override = True`` and widens its own signature;
        the fallback strike ladder checks that flag before passing the keyword.
        Requiring it here made every fake provider in the suite protocol-
        incompatible without changing behaviour.
        """
        ...


class Tool(Protocol):
    name: str

    @property
    def schema(self) -> ToolSchema:
        ...

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        ...
