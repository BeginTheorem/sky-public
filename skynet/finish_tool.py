"""The finish tool: the episode's closing contract, offered as a call instead of prose.

The ReAct prompt describes the Finish Report as a JSON object returned in text, and the harness
parsed that text. Measured on the live ledger 2026-10-19: of 69 closing turns in build runs, 15
came back as prose ("Задача выполнена…", "the finish tool isn't available in my tool inventory,
so here is my Finish Report delivered as text"), of which the one bounded repair turn saved 16 --
and every one of the 39 recorded calls to `finish` hit `unknown tool: finish`, because the model
reaches for the closing call the ReAct pattern teaches it and the harness had no such tool. Two
runs ended unreadable (NEEDS_RECOVERY) with their evidence carried forward.

So the report travels the channel the model is already reaching for: a tool call whose arguments
are the contract's fields. The runner intercepts this tool before dispatch -- `execute` is never
reached; it exists so that a stray call fails loudly instead of silently doing nothing.
"""

from __future__ import annotations

from typing import Any

from .model_contracts import FINISH_REPORT_SCHEMA

FINISH_TOOL_NAME = "finish"


class FinishTool:
    """The Finish Report schema, exposed as the episode's closing tool."""

    name = FINISH_TOOL_NAME
    # Closing is not an effect: it ends the episode and reports, so it must not travel as a
    # write and must never be blocked by the effect-once machinery.
    capability_kind = "read"

    def __init__(self) -> None:
        self._schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Close this episode by returning the Finish Report: the arguments ARE the report. "
                    "Call it once, when this episode's work is done, blocked, or must be recovered. "
                    "It is the only accepted way to finish -- a report written as prose is rejected."
                ),
                "parameters": FINISH_REPORT_SCHEMA,
            },
        }

    @property
    def schema(self) -> dict[str, Any]:
        return self._schema

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        del arguments, idempotency_key
        return {
            "ok": False,
            "error": "the finish tool is intercepted by the runner before dispatch",
            "failure_class": "internal",
        }
