from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from time import sleep
from typing import Any, Protocol
from urllib import error, request

from .store import StateStore

log = logging.getLogger("skynet.outbox")

TELEGRAM_MESSAGE_LIMIT = 4096
DEFAULT_BATCH = 5
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BACKLOG_LIMIT = 20
DEFAULT_RATE_LIMIT_CAP = 60.0
RATE_LIMIT_RETRIES = 1
BACKLOG_SUPPRESS_ERROR = "suppressed: backlog limit"
JSON_RENDER_LIMIT = 8000
AGENT_RESPONSE_SUMMARY_LIMIT = 400
AGENT_RESPONSE_LOGS_HINT = "Полный отчёт доступен командой /logs"


class OutboxRateLimitError(RuntimeError):
    def __init__(self, retry_after: float) -> None:
        self.retry_after = max(1.0, float(retry_after))
        super().__init__(f"transport rate limit; retry after {self.retry_after:g}s")


class Transport(Protocol):
    def send(self, text: str) -> None: ...


def chunk_text(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    text = text or "(empty)"
    return [text[index:index + limit] for index in range(0, len(text), limit)] or ["(empty)"]


def _compact_json(value: Any, limit: int = JSON_RENDER_LIMIT) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = str(value)
    return rendered if len(rendered) <= limit else rendered[:limit] + "…"


def render_alert(payload: dict[str, Any]) -> str:
    severity = str(payload.get("severity", "warning"))
    kind = str(payload.get("kind", "alert"))
    occurrences = payload.get("occurrences", 1)
    return f"[{severity.upper()}] {kind} x{occurrences}\n{_compact_json(payload.get('payload', {}))}"


def render_agent_response(payload: dict[str, Any]) -> str:
    run_id = str(payload.get("run_id", "unknown"))
    status = str(payload.get("status", "unknown"))
    report = payload.get("report")
    summary = ""
    if isinstance(report, str):
        try:
            parsed = json.loads(report)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            candidate = parsed.get("summary")
            if isinstance(candidate, str) and candidate.strip():
                summary = candidate.strip()
        if not summary:
            summary = report
    elif isinstance(report, dict):
        candidate = report.get("summary")
        summary = candidate.strip() if isinstance(candidate, str) and candidate.strip() else _compact_json(report)
    else:
        summary = _compact_json(report)
    summary = " ".join(summary.split())
    if len(summary) > AGENT_RESPONSE_SUMMARY_LIMIT:
        summary = summary[:AGENT_RESPONSE_SUMMARY_LIMIT].rstrip() + "…"
    return f"run={run_id} status={status}\n{summary}\n{AGENT_RESPONSE_LOGS_HINT}"


def render_user_question(payload: dict[str, Any]) -> str:
    question_id = str(payload.get("question_id", ""))[:8]
    question = str(payload.get("question", "")).strip()
    options = payload.get("options")
    lines = [f"[QUESTION {question_id}] {question}"]
    if isinstance(options, list) and options:
        lines.append("варианты: " + ", ".join(str(item) for item in options[:8]))
    lines.append("ответить: /answer <текст>  или  /answer " + question_id + " <текст>")
    return "\n".join(lines)


def render_agent_message(payload: dict[str, Any]) -> str:
    severity = str(payload.get("severity", "info")).upper()
    return f"[{severity}] {str(payload.get('message', '')).strip()}"


def render_outbox_message(kind: str, payload: dict[str, Any]) -> str:
    if kind == "alert":
        return render_alert(payload)
    if kind == "agent_response":
        return render_agent_response(payload)
    if kind == "user_question":
        return render_user_question(payload)
    if kind == "agent_message":
        return render_agent_message(payload)
    return f"{kind}\n{_compact_json(payload)}"


class TelegramTransport:
    """Deliver outbox text through the bot's existing HTTP/proxy layer."""

    def __init__(self, api: Any, chat_id: int, *, limit: int = TELEGRAM_MESSAGE_LIMIT) -> None:
        self.api = api
        self.chat_id = chat_id
        self.limit = limit

    def send(self, text: str) -> None:
        from .telegram_bot import redact

        for part in chunk_text(text, self.limit):
            try:
                self.api.call("sendMessage", {"chat_id": self.chat_id, "text": redact(part)})
            except Exception as exc:
                retry_after = getattr(exc, "retry_after", None)
                if retry_after is not None:
                    raise OutboxRateLimitError(float(retry_after)) from exc
                raise


class OutboxDrainer:
    """Bounded, rate-limited delivery of the outbox to a single transport.

    The bot owns the token and the poll loop, so it also owns this projection:
    the drainer never schedules itself and never touches the lifecycle.
    """

    def __init__(
        self,
        store: StateStore,
        transport: Transport,
        *,
        batch: int = DEFAULT_BATCH,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backlog_limit: int = DEFAULT_BACKLOG_LIMIT,
        enabled: bool = True,
        sleep: Callable[[float], None] = sleep,
        rate_limit_cap: float = DEFAULT_RATE_LIMIT_CAP,
    ) -> None:
        self.store = store
        self.transport = transport
        self.batch = max(1, batch)
        self.max_attempts = max(1, max_attempts)
        self.backlog_limit = max(0, backlog_limit)
        self.enabled = enabled
        self._sleep = sleep
        self.rate_limit_cap = max(0.0, rate_limit_cap)
        self._backlog_checked = False

    def drain(self) -> int:
        # Reconcile any alert whose message already left the queue before
        # sending anything new.
        try:
            self.store.reconcile_delivered_alerts()
        except Exception:
            log.exception("alert reconciliation failed")
        if not self.enabled:
            return 0
        if not self._backlog_checked:
            self._backlog_checked = True
            self._suppress_backlog()
        delivered = 0
        for message in self.store.claim_outbox(limit=self.batch):
            if self._deliver(message):
                delivered += 1
        return delivered

    def _suppress_backlog(self) -> None:
        if self.backlog_limit <= 0:
            return
        pending_count = int(
            self.store.connection.execute(
                "SELECT COUNT(*) FROM outbox WHERE delivery_state='pending'"
            ).fetchone()[0]
        )
        surplus = pending_count - self.backlog_limit
        if surplus <= 0:
            return
        suppressed = self.store.claim_outbox(limit=surplus)
        count = 0
        for message in suppressed:
            self.store.mark_outbox_failed(message["message_id"], BACKLOG_SUPPRESS_ERROR, max_attempts=1)
            count += 1
        if not count:
            return
        log.warning("outbox backlog suppressed count=%s limit=%s", count, self.backlog_limit)
        self.store.append_event("outbox_backlog_suppressed", {"count": count, "limit": self.backlog_limit})
        self.store.raise_alert(
            "outbox_backlog",
            {"suppressed": count, "limit": self.backlog_limit},
            severity="warning",
            dedup_key="outbox-backlog",
        )

    def _deliver(self, message: dict[str, Any]) -> bool:
        message_id = str(message.get("message_id", ""))
        kind = str(message.get("kind", "unknown"))
        payload = message.get("payload")
        if not isinstance(payload, dict):
            payload = {"raw": payload}
        try:
            self._send_with_rate_limit(render_outbox_message(kind, payload))
        except Exception as exc:
            state = self.store.mark_outbox_failed(message_id, str(exc), max_attempts=self.max_attempts)
            log.warning("outbox delivery failed id=%s kind=%s state=%s error=%s", message_id, kind, state, exc)
            if state == "dead":
                self._dead_letter(message_id, kind, str(exc))
            return False
        try:
            self.store.mark_outbox_delivered(message_id)
        except Exception:
            log.exception("outbox mark-delivered failed id=%s", message_id)
            return False
        if kind == "alert":
            alert_id = payload.get("alert_id")
            if isinstance(alert_id, str) and alert_id:
                try:
                    self.store.mark_alert_delivered(alert_id, "telegram")
                except Exception:
                    log.exception("alert mark-delivered failed id=%s", alert_id)
        return True

    def _send_with_rate_limit(self, text: str) -> None:
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                self.transport.send(text)
                return
            except OutboxRateLimitError as exc:
                if attempt >= RATE_LIMIT_RETRIES:
                    raise
                delay = min(max(float(exc.retry_after), 1.0), self.rate_limit_cap)
                log.warning("outbox rate limited; sleeping %.1fs", delay)
                self._sleep(delay)

    def _dead_letter(self, message_id: str, kind: str, error: str) -> None:
        try:
            self.store.append_event("outbox_dead_letter", {"message_id": message_id, "kind": kind, "error": error})
            self.store.raise_alert(
                "outbox_dead_letter",
                {"message_id": message_id, "kind": kind, "error": error},
                severity="critical",
                dedup_key=f"outbox-dead:{kind}",
            )
        except Exception:
            log.exception("outbox dead-letter escalation failed id=%s", message_id)


def deliver_to_jsonl(store: StateStore, path: str | Path, limit: int = 50) -> int:
    """Deliver claimed messages to a local append-only integration boundary."""
    messages = store.claim_outbox(limit)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    delivered = 0
    with target.open("a", encoding="utf-8") as stream:
        for message in messages:
            try:
                stream.write(json.dumps(message, ensure_ascii=False) + "\n")
                stream.flush()
                store.mark_outbox_delivered(message["message_id"])
                delivered += 1
            except OSError as exc:
                store.mark_outbox_failed(message["message_id"], str(exc))
    return delivered


def deliver_to_http(
    store: StateStore,
    url: str,
    *,
    api_key: str = "",
    limit: int = 50,
    timeout_seconds: float = 15.0,
    retries: int = 2,
    backoff_seconds: float = 1.0,
) -> int:
    """Deliver claimed messages to an idempotent HTTP integration boundary."""
    delivered = 0
    for message in store.claim_outbox(limit):
        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Idempotency-Key": message["message_id"]}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = request.Request(url, data=body, headers=headers, method="POST")
        try:
            for attempt in range(max(0, retries) + 1):
                try:
                    with request.urlopen(req, timeout=timeout_seconds) as response:
                        if not 200 <= response.status < 300:
                            raise OSError(f"HTTP {response.status}")
                    store.mark_outbox_delivered(message["message_id"])
                    delivered += 1
                    break
                except (error.HTTPError, error.URLError, TimeoutError, OSError):
                    if attempt >= retries:
                        raise
                    sleep(min(backoff_seconds * (2**attempt), 30.0))
        except (error.HTTPError, error.URLError, TimeoutError, OSError) as exc:
            store.mark_outbox_failed(message["message_id"], str(exc))
    return delivered
