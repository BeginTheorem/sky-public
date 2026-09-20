"""Owner dialogue: durable questions and outbound messages.

The asynchronous half of the owner channel. A question is persisted and queued
for delivery, then the run continues on its own assumption; the answer arrives
as an inbox message on a later wake and as a memory, so it survives the
interruption of the run that asked it. The default path never blocks. Passing
wait_seconds>0 opts into a bounded blocking wait; the run heartbeat is refreshed
while waiting so the watchdog stays calm.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

from .provider import Tool


class AskUserTool(Tool):
    """Ask the owner one bounded question; async by default, optionally waiting."""

    name = "ask_user"
    capability_kind = "write"

    # A bounded wait, not an unbounded one: the provider ladder plus a long block
    # used to exceed the watchdog window, so the cap keeps the run inside it and
    # the heartbeat is refreshed while waiting.
    MAX_WAIT_SECONDS = 600.0
    HEARTBEAT_INTERVAL_SECONDS = 20.0
    POLL_INTERVAL_SECONDS = 2.0

    def __init__(self, store: Any, *, default_ttl_seconds: float = 86_400.0, max_open: int = 3) -> None:
        self.store = store
        self.default_ttl_seconds = default_ttl_seconds
        self.max_open = max_open
        self._stop_event: Any = None

    def set_stop_event(self, event: Any) -> None:
        """The runner's stop event, so SIGTERM ends a wait without an answer.

        The runner must call this once when it constructs the tool; until then
        only WatchdogTimeout can interrupt a wait.
        """
        self._stop_event = event

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Ask the owner a question when a decision is genuinely theirs. The question is "
                    "stored durably and pushed to the owner channel; the answer arrives later as an "
                    "inbox message and as a memory. Non-blocking by default: wait_seconds=0 queues "
                    "the question and keeps working. Passing wait_seconds>0 blocks up to that long "
                    "for an answer; if none arrives, continue on your own explicit assumption and "
                    "record it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "minLength": 1, "maxLength": 2000},
                        "options": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 200},
                            "maxItems": 8,
                            "description": "Optional short choices the owner can pick from.",
                        },
                        "ttl_seconds": {"type": "number", "minimum": 60, "maximum": 604800},
                        "wait_seconds": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 600,
                            "description": "Wait this long for an answer before continuing (0 = do not wait).",
                        },
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        del idempotency_key
        question = str(arguments.get("question", "")).strip()
        if not question:
            return {"ok": False, "error": "question must not be empty"}
        options = [str(item)[:200] for item in arguments.get("options", []) if str(item).strip()][:8]
        open_now = self.store.open_questions(limit=self.max_open + 1)
        if len(open_now) >= self.max_open:
            return {
                "ok": False,
                "error": f"too many open questions ({len(open_now)}); wait for an answer or proceed on your own assumption",
                "open_questions": [item["question"][:200] for item in open_now],
            }
        try:
            ttl = float(arguments.get("ttl_seconds", self.default_ttl_seconds))
        except (TypeError, ValueError):
            ttl = self.default_ttl_seconds
        question_id = self.store.ask_question(question, options=options, ttl_seconds=ttl)
        try:
            wait_seconds = max(0.0, min(float(arguments.get("wait_seconds", 0) or 0), self.MAX_WAIT_SECONDS))
        except (TypeError, ValueError):
            wait_seconds = 0.0
        if wait_seconds > 0:
            answered = self._wait_for_answer(question_id, wait_seconds)
            if answered is not None:
                return {
                    "ok": True,
                    "question_id": question_id,
                    "state": "answered",
                    "answer": answered["answer"],
                    "source": answered.get("source"),
                }
            return {
                "ok": True,
                "question_id": question_id,
                "state": "expired_wait",
                "note": "no answer within the wait; proceed on an explicit assumption and record it",
            }
        return {
            "ok": True,
            "question_id": question_id,
            "state": "queued",
            "note": "the owner may answer later; do not wait for it, proceed and record the assumption you act on",
        }

    def _wait_for_answer(self, question_id: str, wait_seconds: float) -> dict[str, Any] | None:
        """Poll for an answer, refreshing the run heartbeat so the watchdog is calm."""
        deadline = time.monotonic() + wait_seconds
        last_beat = 0.0
        while time.monotonic() < deadline:
            answer = self.store.answer_for(question_id)
            if answer is not None:
                return answer
            if self._stop_event is not None and self._stop_event.is_set():
                return None
            now = time.monotonic()
            if now - last_beat >= self.HEARTBEAT_INTERVAL_SECONDS:
                run_id = self.store.state().active_run_id
                if run_id:
                    with contextlib.suppress(Exception):
                        self.store.touch_run(str(run_id), "waiting_for_user", progress=False)
                last_beat = now
            time.sleep(self.POLL_INTERVAL_SECONDS)
        return self.store.answer_for(question_id)


class AcknowledgeInboxTool(Tool):
    """Close a pending owner notification after deciding what to do with it.

    A pending owner message is surfaced as a notification, never as an order and
    never as an auto-created task: the organism decides. This tool is how that
    decision becomes durable — ``acted`` or ``ignored`` consumes the message,
    ``deferred`` leaves it pending for a later wake.
    """

    name = "acknowledge_inbox"
    capability_kind = "write"

    def __init__(self, store: Any) -> None:
        self.store = store

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Close a pending owner notification (kind inbox_notification) after you have decided "
                    "what to do with it. Use decision='acted' when you handled it, 'ignored' when you "
                    "judged it needs no action, or 'deferred' to keep it pending for a later wake. The "
                    "message stays pending until this call, so do not leave notifications open by accident."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "string", "minLength": 1, "maxLength": 100},
                        "decision": {"type": "string", "enum": ["acted", "ignored", "deferred"]},
                        "note": {"type": "string", "maxLength": 500, "description": "Optional one-line reason recorded with the decision."},
                    },
                    "required": ["event_id", "decision"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        del idempotency_key
        event_id = str(arguments.get("event_id", "")).strip()
        if not event_id:
            return {"ok": False, "error": "event_id must not be empty"}
        decision = str(arguments.get("decision", "")).strip()
        if decision not in {"acted", "ignored", "deferred"}:
            return {"ok": False, "error": "decision must be one of: acted, ignored, deferred"}
        if event_id not in {str(event["event_id"]) for event in self.store.pending_inbox()}:
            return {"ok": True, "state": "already_closed"}
        if decision == "deferred":
            return {"ok": True, "state": "deferred", "note": "the notification stays pending for a later wake"}
        self.store.consume_inbox(event_id)
        self.store.append_event(
            "inbox_acknowledged",
            {"event_id": event_id, "decision": decision, "note": str(arguments.get("note", ""))[:500]},
        )
        return {"ok": True, "state": "closed", "decision": decision}


class SendMessageToUserTool(Tool):
    """Send the owner a message without expecting a reply."""

    name = "send_message_to_user"
    capability_kind = "write"

    def __init__(self, store: Any) -> None:
        self.store = store

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Send the owner a short status message through the owner channel. Use it to report a decision, a blocker or a result; do not use it for routine narration.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "minLength": 1, "maxLength": 4000},
                        "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        message = str(arguments.get("message", "")).strip()
        if not message:
            return {"ok": False, "error": "message must not be empty"}
        severity = str(arguments.get("severity", "info"))
        if severity not in {"info", "warning", "critical"}:
            severity = "info"
        self.store.add_outbox("agent_message", {"message": message[:4000], "severity": severity, "idempotency_key": idempotency_key})
        self.store.append_event("agent_message_queued", {"severity": severity, "message": message[:300]})
        return {"ok": True, "state": "queued"}
