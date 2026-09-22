from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from time import monotonic, sleep
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
DEFERRED_ERROR = "deferred: rate-limit window"
JSON_RENDER_LIMIT = 8000
AGENT_RESPONSE_SUMMARY_LIMIT = 400
AGENT_RESPONSE_LOGS_HINT = "Полный отчёт доступен командой /logs"


class OutboxRateLimitError(RuntimeError):
    def __init__(self, retry_after: float) -> None:
        self.retry_after = max(1.0, float(retry_after))
        super().__init__(f"transport rate limit; retry after {self.retry_after:g}s")


class OutboxDeferredError(RuntimeError):
    """A rate limit deferred a message instead of failing it."""

    def __init__(self, retry_after: float) -> None:
        self.retry_after = max(1.0, float(retry_after))
        super().__init__(f"deferred by rate limit; retry after {self.retry_after:g}s")


class Transport(Protocol):
    def send(self, text: str) -> None: ...


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
        from .telegram_bot import chunk_html, redact

        for part in chunk_html(redact(text), self.limit):
            try:
                self.api.call(
                    "sendMessage",
                    {"chat_id": self.chat_id, "text": part, "parse_mode": "HTML"},
                )
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
        block_on_rate_limit: bool = True,
    ) -> None:
        self.store = store
        self.transport = transport
        self.batch = max(1, batch)
        self.max_attempts = max(1, max_attempts)
        self.backlog_limit = max(0, backlog_limit)
        self.enabled = enabled
        self._sleep = sleep
        self.rate_limit_cap = max(0.0, rate_limit_cap)
        # A 429 describes the channel, not the message, and drain() is called
        # synchronously from the single-threaded poll loop. Sleeping here stalls
        # the owner's next command for the whole flood window while the queue it
        # waits for is already paused; the deferral path below and the post-window
        # probe deliver the same messages without holding the caller.
        self.block_on_rate_limit = bool(block_on_rate_limit)
        self._backlog_checked = False
        self._rate_limited_until = 0.0

    def drain(self) -> int:
        # Two consumers used to race for the same lease; reconcile any alert
        # whose message already left the queue before sending anything new.
        try:
            self.store.reconcile_delivered_alerts()
        except Exception:
            log.exception("alert reconciliation failed")
        if not self.enabled:
            return 0
        if not self._backlog_checked:
            self._backlog_checked = True
            self._suppress_backlog()
        if not self._rate_limit_window_open():
            # The flood-wait window is still running. Sending anything now only
            # deepens the refusal, so the queue is left untouched and a drain
            # costs zero requests; the probe below runs once the window expires.
            return 0
        if self._rate_limited_until > 0:
            return self._probe_after_rate_limit()
        delivered = 0
        claimed = self.store.claim_outbox(limit=self.batch)
        for position, message in enumerate(claimed):
            if position and not self._rate_limit_window_open():
                # The channel met this batch with an unannounced flood wait.
                # The message that discovered it already paused the queue for
                # the whole window; attempting the rest would spend one request
                # each to learn the same refusal and leave them claimed in
                # `delivering` for a lease. Return the untouched tail instead.
                for leftover in claimed[position:]:
                    self._defer(str(leftover.get("message_id", "")))
                break
            if self._deliver(message):
                delivered += 1
        return delivered

    def _rate_limit_window_open(self) -> bool:
        """True when sending may be attempted at all."""
        return monotonic() >= self._rate_limited_until

    def _rate_limit_pause(self, retry_after: float) -> None:
        """Postpone the whole queue for one Telegram flood-wait window.

        Telegram's `retry_after` describes the channel, not the message that
        happened to hit it. Treating it as a per-message failure burned one unit
        of every queued message's attempt budget per drain cycle, so a sustained
        flood wait dead-lettered the owner's alerts although nothing was wrong
        with them. Meanwhile the window is remembered and every send is skipped;
        a single probe request per window measures when the refusal has passed,
        so a long refusal costs one request per window instead of one per message
        per poll.
        """
        delay = min(max(float(retry_after), 1.0), self.rate_limit_cap)
        self._rate_limited_until = max(self._rate_limited_until, monotonic() + delay)

    def _probe_after_rate_limit(self) -> int:
        """Try exactly one queued message once the flood-wait window has expired."""
        if self._rate_limited_until <= 0:
            return 0
        for message in self.store.claim_outbox(limit=1):
            message_id = str(message.get("message_id", ""))
            kind = str(message.get("kind", "unknown"))
            payload = message.get("payload")
            if not isinstance(payload, dict):
                payload = {"raw": payload}
            try:
                self._send_with_rate_limit(render_outbox_message(kind, payload), consume_attempt=False)
            except OutboxDeferredError:
                self._defer(message_id)
                return 0
            except Exception as exc:
                state = self.store.mark_outbox_failed(message_id, str(exc), max_attempts=self.max_attempts)
                log.warning("outbox probe failed id=%s state=%s error=%s", message_id, state, exc)
                if state == "dead":
                    self._dead_letter(message_id, kind, str(exc))
                return 0
            self._mark_delivered(message_id, kind, payload)
            # The channel accepted a send again: leave the flood-wait state so the
            # next drain returns to full batch delivery instead of probing.
            self._rate_limited_until = 0.0
            return 1
        return 0

    def _mark_delivered(self, message_id: str, kind: str, payload: dict[str, Any]) -> None:
        try:
            self.store.mark_outbox_delivered(message_id)
        except Exception:
            log.exception("outbox mark-delivered failed id=%s", message_id)
            return
        if kind == "alert":
            alert_id = payload.get("alert_id")
            if isinstance(alert_id, str) and alert_id:
                try:
                    self.store.mark_alert_delivered(alert_id, "telegram")
                except Exception:
                    log.exception("alert mark-delivered failed id=%s", alert_id)

    def _defer(self, message_id: str) -> None:
        """Return a claimed message to `pending` without spending an attempt.

        ``claim`` increments ``attempts``; a message Telegram refused for a
        channel-wide reason must not pay for that, or the dead-letter cap expires
        while the message itself is still perfectly deliverable.
        """
        try:
            with self.store.transaction():
                self.store.connection.execute(
                    "UPDATE outbox SET delivery_state='pending', claimed_at=NULL, last_error=?, "
                    "attempts=CASE WHEN attempts > 0 THEN attempts - 1 ELSE 0 END "
                    "WHERE message_id=? AND delivery_state='delivering'",
                    (DEFERRED_ERROR, message_id),
                )
        except Exception:
            log.warning("outbox defer failed id=%s", message_id, exc_info=True)

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
        except OutboxDeferredError:
            self._defer(message_id)
            return False
        except Exception as exc:
            state = self.store.mark_outbox_failed(message_id, str(exc), max_attempts=self.max_attempts)
            log.warning("outbox delivery failed id=%s kind=%s state=%s error=%s", message_id, kind, state, exc)
            if state == "dead":
                self._dead_letter(message_id, kind, str(exc))
            return False
        self._mark_delivered(message_id, kind, payload)
        return True

    def _send_with_rate_limit(self, text: str, *, consume_attempt: bool = True) -> None:
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                self.transport.send(text)
                return
            except OutboxRateLimitError as exc:
                self._rate_limit_pause(exc.retry_after)
                if not consume_attempt or not self.block_on_rate_limit:
                    # The probe measures the window with a single request; the
                    # configured non-blocking mode hands the whole penalty to the
                    # window plus that probe. Either way a blocking sleep here
                    # would stall the poll loop for the length of the flood wait.
                    raise OutboxDeferredError(exc.retry_after) from exc
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
