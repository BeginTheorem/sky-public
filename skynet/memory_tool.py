"""On-demand memory access: the organism queries and curates its own memory.

The ReAct prompt carries only a small automatic injection of pinned and recently
relevant memories; this tool is how the organism goes deeper than that. `search`
retrieves more on demand, while `remember`, `forget`, `pin`, `unpin` and
`correct` let it curate what future cycles will see instead of waiting for the
memory loop to consolidate it. Every store call is guarded, so a half-integrated
store returns a structured error instead of crashing the run.
"""

from __future__ import annotations

import inspect
from typing import Any

from .provider import Tool
from .store import StateStore


class MemoryTool(Tool):
    """Query and curate the durable memory store from inside a run."""

    name = "memory"
    capability_kind = "write"

    ACTIONS = ("search", "remember", "forget", "pin", "unpin", "correct")
    DEFAULT_SEARCH_LIMIT = 10
    MIN_SEARCH_LIMIT = 1
    MAX_SEARCH_LIMIT = 50
    DEFAULT_CONFIDENCE = 0.6
    MAX_CONTENT_CHARS = 4000
    MAX_EVIDENCE_CHARS = 2000

    def __init__(self, store: StateStore) -> None:
        self.store = store

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Query and curate your own long-term memory. The prompt carries only a small "
                    "automatic injection, so use action='search' whenever the task needs more than "
                    "that. action='remember' stores a new memory for future cycles; action='forget' "
                    "removes one; action='pin'/'unpin' control whether it is always injected; "
                    "action='correct' supersedes a wrong memory with a corrected version and cites "
                    "the evidence that justifies the change."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": list(self.ACTIONS)},
                        "query": {"type": "string", "description": "Search text; required for action='search'."},
                        "limit": {
                            "type": "integer",
                            "minimum": self.MIN_SEARCH_LIMIT,
                            "maximum": self.MAX_SEARCH_LIMIT,
                            "description": "Maximum results for action='search' (default 10).",
                        },
                        "include_inactive": {
                            "type": "boolean",
                            "description": "Also return forgotten or superseded memories in action='search'.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Memory text for action='remember', or the corrected text for action='correct'.",
                        },
                        "kind": {"type": "string", "description": "Memory kind for action='remember' (default observation)."},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "pinned": {"type": "boolean", "description": "Store the new memory as always-injected."},
                        "memory_id": {"type": "string", "description": "Target memory for forget/pin/unpin/correct."},
                        "evidence": {"type": "string", "description": "Cited evidence; required for action='correct'."},
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        del idempotency_key
        action = str(arguments.get("action", "")).strip()
        if action == "search":
            return self._search(arguments)
        if action == "remember":
            return self._remember(arguments)
        if action == "forget":
            return self._forget(arguments)
        if action in {"pin", "unpin"}:
            return self._set_pinned(action, arguments)
        if action == "correct":
            return self._correct(arguments)
        return {"ok": False, "error": f"unknown action: {action or '<missing>'}"}

    def _search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "query must not be empty"}
        limit = self._clamp_int(
            arguments.get("limit", self.DEFAULT_SEARCH_LIMIT),
            default=self.DEFAULT_SEARCH_LIMIT,
            low=self.MIN_SEARCH_LIMIT,
            high=self.MAX_SEARCH_LIMIT,
        )
        include_inactive = self._as_bool(arguments.get("include_inactive", False))
        missing = self._missing("search_memories")
        if missing is not None:
            return missing
        try:
            if self._supports_include_inactive():
                memories = self.store.search_memories(query, limit=limit, include_inactive=include_inactive)
            else:
                # Mid-integration: the store has search but not the inactive filter yet.
                memories = self.store.search_memories(query, limit=limit)
        except Exception as exc:
            return {"ok": False, "error": f"search_memories failed: {exc}"}
        return {"ok": True, "memories": list(memories)}

    def _remember(self, arguments: dict[str, Any]) -> dict[str, Any]:
        content = str(arguments.get("content", "")).strip()
        if not content:
            return {"ok": False, "error": "content must not be empty"}
        missing = self._missing("remember_memory")
        if missing is not None:
            return missing
        kind = str(arguments.get("kind", "observation")).strip() or "observation"
        confidence = self._clamp_float(
            arguments.get("confidence", self.DEFAULT_CONFIDENCE),
            default=self.DEFAULT_CONFIDENCE,
            low=0.0,
            high=1.0,
        )
        pinned = self._as_bool(arguments.get("pinned", False))
        try:
            memory_id = self.store.remember_memory(
                content[: self.MAX_CONTENT_CHARS], kind=kind, confidence=confidence, pinned=pinned
            )
        except Exception as exc:
            return {"ok": False, "error": f"remember_memory failed: {exc}"}
        return {"ok": True, "memory_id": memory_id}

    def _forget(self, arguments: dict[str, Any]) -> dict[str, Any]:
        memory_id = str(arguments.get("memory_id", "")).strip()
        if not memory_id:
            return {"ok": False, "error": "memory_id must not be empty"}
        missing = self._missing("forget_memory")
        if missing is not None:
            return missing
        try:
            forgotten = bool(self.store.forget_memory(memory_id))
        except Exception as exc:
            return {"ok": False, "error": f"forget_memory failed: {exc}"}
        return {"ok": True, "forgotten": forgotten}

    def _set_pinned(self, action: str, arguments: dict[str, Any]) -> dict[str, Any]:
        memory_id = str(arguments.get("memory_id", "")).strip()
        if not memory_id:
            return {"ok": False, "error": "memory_id must not be empty"}
        missing = self._missing("set_memory_pinned")
        if missing is not None:
            return missing
        try:
            updated = bool(self.store.set_memory_pinned(memory_id, action == "pin"))
        except Exception as exc:
            return {"ok": False, "error": f"set_memory_pinned failed: {exc}"}
        return {"ok": True, "updated": updated}

    def _correct(self, arguments: dict[str, Any]) -> dict[str, Any]:
        memory_id = str(arguments.get("memory_id", "")).strip()
        content = str(arguments.get("content", "")).strip()
        evidence = str(arguments.get("evidence", "")).strip()
        if not memory_id:
            return {"ok": False, "error": "memory_id must not be empty"}
        if not content:
            return {"ok": False, "error": "content must not be empty"}
        if not evidence:
            return {"ok": False, "error": "evidence must not be empty"}
        missing = self._missing("correct_memory")
        if missing is not None:
            return missing
        try:
            new_id = self.store.correct_memory(
                memory_id, content=content[: self.MAX_CONTENT_CHARS], evidence=evidence[: self.MAX_EVIDENCE_CHARS]
            )
        except Exception as exc:
            return {"ok": False, "error": f"correct_memory failed: {exc}"}
        return {"ok": True, "memory_id": new_id}

    def _supports_include_inactive(self) -> bool:
        """Whether the store's search accepts the inactive filter (mid-integration safe)."""
        try:
            parameters = inspect.signature(self.store.search_memories).parameters
        except (TypeError, ValueError):
            return False
        if "include_inactive" in parameters:
            return True
        return any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())

    def _missing(self, method: str) -> dict[str, Any] | None:
        """Report a store that does not expose the method instead of crashing."""
        if not callable(getattr(self.store, method, None)):
            return {"ok": False, "error": f"store does not expose {method}"}
        return None

    @staticmethod
    def _clamp_int(value: object, *, default: int, low: int, high: int) -> int:
        try:
            number = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
        return max(low, min(number, high))

    @staticmethod
    def _clamp_float(value: object, *, default: float, low: float, high: float) -> float:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
        return max(low, min(number, high))

    @staticmethod
    def _as_bool(value: object, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().casefold() in {"true", "1", "yes", "on"}
        if value is None:
            return default
        return bool(value)
