"""Private Telegram control channel for SkyNet.

The bot is deliberately a separate process. It exposes only fixed commands and
accepts updates from one configured user in one configured private chat.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from threading import Event, Lock
from typing import Any
from urllib import parse, request
from urllib.error import HTTPError
from uuid import uuid4

import socks

from . import metrics, status_card
from .outbox import OutboxDrainer, TelegramTransport
from .providers import (
    active_chain_names,
    applied_chain_names,
    configured_provider_names,
    context_ceiling_tokens,
    load_provider_prefs,
    money_boost_enabled,
    provider_class,
    set_provider_off,
)
from .revive import PARK_FILE, clear_park, park_state, set_park
from .store import StateStore
from .systemctl import systemctl_command
from .telegram_ui import help_keyboard, logs_keyboard, parse_callback, providers_keyboard, status_keyboard

log = logging.getLogger("skynet.telegram")
MAX_MESSAGE = 3900
# Long-poll window. This is also the worst-case delay before a queued outbox
# message can leave: the bot drains the outbox only after getUpdates returns, so
# a reply generated while a poll is in flight waits for that poll to end. The old
# 30s window was measured as ~29.9s of delivery latency for a message queued
# mid-poll; a shorter window trades a few more requests for a much smaller worst
# case. Keep it above 0: short polling burns quota and the token only allows one
# getUpdates consumer at a time.
POLL_TIMEOUT = 5
RETRY_FALLBACK_SECONDS = 30
SEEN_UPDATE_LIMIT = 512

# Honest read receipts. Telegram marks an owner message as "read" (two ticks) the
# moment the bot fetches it with getUpdates -- a transport fact, not a statement
# that the model ever saw the text. The Bot API has no "mark as unread" method,
# so the only truthful signal we can offer is a reaction the organism controls:
# eyes while the message is merely queued, a thumbs-up once a run has actually
# carried it into a ReAct context. The read emoji must come from the API's
# allowed reaction set: U+2705 (the check mark) is rejected with
# 400 REACTION_INVALID on this bot's chat, which made every read upgrade a
# silent no-op, so it is not used. ``setMessageReaction`` is deliberately absent
# from ALLOWED_UPDATES: the API never delivers reaction updates for reactions set
# by bots, so subscribing would only add noise.
ALLOWED_UPDATES = ["message", "callback_query"]
RECEIPT_SEEN = "👀"      # eyes: fetched, not yet read by the model
RECEIPT_READ = "👍"      # thumbs up: delivered into a ReAct context (U+2705 is rejected)
RECEIPT_LIMIT = 200
# The Bot API accepts only a fixed set of emoji for setMessageReaction; anything
# outside it is rejected with 400 REACTION_INVALID and the read upgrade silently
# no-ops. Kept as data so a test can pin both receipts to the accepted set.
TELEGRAM_ALLOWED_REACTIONS = frozenset({
    "👍", "👎", "❤", "🔥", "🥰", "👏", "😁", "🤔", "🤯", "😱", "🤬", "😢",
    "🎉", "🤩", "🤮", "💩", "🙏", "👌", "🕊", "🤡", "🥱", "🥴", "😍", "🐳",
    "❤‍🔥", "🌚", "🌭", "💯", "🤣", "⚡", "🍌", "🏆", "💔", "🤨", "😐", "🍓",
    "🍾", "💋", "🖕", "😈", "😴", "😭", "🤓", "👻", "👨‍💻", "👀", "🎃", "🙈",
    "😇", "😨", "🤝", "✍", "🤗", "🎄", "☃", "💅", "🤪", "🗿", "🆒", "💘",
    "🤷‍♂", "🤷", "🤷‍♀", "😡",
})
# A failed read upgrade used to be retried on every poll (POLL_TIMEOUT is 5s),
# so one permanently refused reaction cost up to RECEIPT_LIMIT blocking network
# calls per cycle, forever. Each entry is now attempted at most this many times,
# with exponential backoff between attempts, and then dropped.
RECEIPT_MAX_ATTEMPTS = 3
RECEIPT_RETRY_BACKOFF_SECONDS = 30.0
# Receipt refresh is a side task on the single poll loop; run it on a cheap
# cadence and cap the reactions one pass may attempt so a burst of messages or a
# batch of upgrades cannot stall command intake or the outbox drain.
RECEIPT_REFRESH_INTERVAL_SECONDS = 30.0
RECEIPT_REFRESH_BATCH = 5
DEFAULT_STATUS_DAYS = 1.0
MIN_STATUS_DAYS = 0.1
MAX_STATUS_DAYS = 90.0
DEFAULT_LOG_RUNS = 5
MAX_LOG_RUNS = 10
NO_LOGS_TEXT = "Логов пока нет."
SECRET_RE = re.compile(r"(?i)(token|api[_-]?key|password|secret)=\S+")
BOT_TOKEN_RE = re.compile(r"\d{8,12}:[A-Za-z0-9_-]{30,}")


def outbox_enabled() -> bool:
    return os.getenv("SKYNET_OUTBOX_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


class TelegramRateLimitError(RuntimeError):
    def __init__(self, retry_after: float) -> None:
        self.retry_after = max(1.0, retry_after)
        super().__init__(f"Telegram rate limit; retry after {self.retry_after:g}s")


class ApiError(RuntimeError):
    """A non-429 Bot API failure, with the API's own description preserved.

    ``raise RuntimeError(str(result))`` kept the body, but callers that log
    only ``str(exc)`` could not name the cause; the description is what names
    it, so it becomes the message and stays available as an attribute. An HTTP
    error status (400 and friends) carries the same description in its JSON
    body, so it is translated here too: without that the caller only sees the
    opaque ``HTTP Error 400: Bad Request`` status line while the body that was
    already parsed and discarded named the cause.
    """

    def __init__(self, description: str, payload: dict[str, Any], http_status: int | None = None) -> None:
        self.description = description or "unknown Telegram API error"
        self.payload = payload
        self.http_status = http_status
        suffix = f" (HTTP {http_status})" if http_status is not None else ""
        super().__init__(f"Telegram API error: {self.description}{suffix}")


def redact(text: str) -> str:
    return BOT_TOKEN_RE.sub("[bot-token]", SECRET_RE.sub(r"\1=[redacted]", text))


# Telegram renders a message either as plain text or, when parse_mode is set, as
# a small markup subset. MarkdownV2 needs roughly eighteen characters escaped and
# one missed character rejects the whole send, so the renderer targets HTML:
# there only &, < and > must be escaped and every tag is balanced by construction.
HTML_ESCAPE = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;"})
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
BOLD_SPAN_RE = re.compile(r"\*\*([^*\n]+)\*\*")


def to_telegram_html(text: str) -> str:
    """Render plain text as Telegram-safe HTML.

    Escaping runs before any tag is inserted, so the result cannot carry a
    malformed entity; text with no markup (``run=... status=...``) is unchanged
    apart from the three escaped characters.
    """
    escaped = text.translate(HTML_ESCAPE)
    # Code spans are tokenized before bold is applied: substituting them first
    # let a ``**`` inside backticks become ``<b>`` nested inside ``<code>``
    # (`` `a**b**c` `` -> ``<code>a<b>b</b>c</code>``), which Telegram's HTML
    # parser can reject, and one rejected entity fails the whole send. Bold is
    # therefore applied only to the segments between code spans.
    parts: list[str] = []
    position = 0
    for match in CODE_SPAN_RE.finditer(escaped):
        parts.append(BOLD_SPAN_RE.sub(r"<b>\1</b>", escaped[position:match.start()]))
        parts.append(f"<code>{match.group(1)}</code>")
        position = match.end()
    parts.append(BOLD_SPAN_RE.sub(r"<b>\1</b>", escaped[position:]))
    return "".join(parts)


def _markup_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges carrying markup, as (start, end) over ``text``."""
    spans = [(m.start(), m.end()) for m in CODE_SPAN_RE.finditer(text)]
    spans += [(m.start(), m.end()) for m in BOLD_SPAN_RE.finditer(text)]
    return spans


def _split_outside_markup(text: str, point: int) -> int:
    """Move a split point back so no markup span straddles the boundary.

    ``chunk_html`` measures rendered prefixes and then cuts at an arbitrary
    character, so a ``code``/``**bold**`` span crossing the cut was emitted as
    two halves: both delimiters survived as literal text and the formatting was
    silently lost. Measured on this renderer, a code span beginning 5 characters
    before the boundary arrived as ``...aaa`code`` and `` span here`...`` with
    zero ``<code>`` tags in either part. Backing the cut up to the start of the
    offending span keeps the part within the limit -- a shorter prefix cannot
    render longer -- and leaves the span whole in the next part. Moving the cut
    can expose a second, overlapping span, so the scan repeats until stable; a
    span that begins at the first character cannot be helped this way and the
    caller keeps its original point.
    """
    spans = _markup_spans(text)
    for _ in range(len(spans) + 1):
        moved = False
        for start, end in spans:
            if start < point < end:
                point, moved = start, True
        if not moved:
            break
    return point


def _telegram_length(text: str) -> int:
    """Length as Telegram counts it: UTF-16 code units, not Python code points.

    Telegram's 4096 limit is measured in UTF-16 code units, so an astral
    character (an emoji, U+1F600 and friends) counts as two while ``len`` counts
    it as one. A message dense in emoji therefore passed the old check and was
    still rejected as too long; measuring the rendered prefix in UTF-16 units
    makes the bisection agree with the server.
    """
    return len(text.encode("utf-16-le")) // 2


def chunk_html(text: str, limit: int = MAX_MESSAGE) -> list[str]:
    """Split text so each rendered part fits Telegram's limit.

    Escaping and markup both inflate the text: ``&`` becomes ``&amp;`` (five
    characters) and every ``<code>``/``<b>`` span adds its own tags. Chunking the
    raw text and rendering afterwards can therefore emit a part larger than the
    limit, and Telegram then rejects the whole send as "message is too long".
    Measured on this repository's own outbox: a 4007-character ``agent_message``
    rendered to one 4322-character part, over the 4096 limit, because 24 code
    spans contributed their tags after the split. Rendered length grows
    monotonically with prefix length, so the longest prefix that still fits is
    found by bisection and the part that is measured is exactly the part sent.
    """
    text = text or "(empty)"
    parts: list[str] = []
    remaining = text
    while remaining:
        low, high = 0, len(remaining)
        while low < high:
            mid = (low + high + 1) // 2
            if _telegram_length(to_telegram_html(remaining[:mid])) <= limit:
                low = mid
            else:
                high = mid - 1
        if low <= 0:
            low = 1
        safe = _split_outside_markup(remaining, low)
        if safe > 0:
            low = safe
        parts.append(to_telegram_html(remaining[:low]))
        remaining = remaining[low:]
    return parts or ["(empty)"]


class TelegramAPI:
    def __init__(self, token: str, proxy: str, timeout: float = 45.0) -> None:
        self.base = f"https://api.telegram.org/bot{token}/"
        self.proxy = proxy
        self.timeout = timeout
        self._lock = Lock()

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        body = parse.urlencode({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value) for key, value in (params or {}).items()}).encode()
        req = request.Request(self.base + method, data=body, method="POST")
        with self._lock:
            original = socket.socket
            original_getaddrinfo = socket.getaddrinfo
            try:
                if self.proxy:
                    parsed = parse.urlparse(self.proxy)
                    if parsed.hostname is None or parsed.port is None:
                        raise ValueError("Telegram proxy must include host and port")
                    # Keep hostname resolution inside the proxy so it can choose
                    # a reachable address instead of receiving a stale local IP.
                    socks.set_default_proxy(socks.SOCKS5, parsed.hostname, parsed.port, rdns=True)
                    socket.socket = socks.socksocket
                    # Preserve the hostname in the sockaddr so PySocks performs
                    # remote DNS through the proxy, while avoiding IPv6 socket attempts.
                    def ipv4_getaddrinfo(host, port, _family=0, type=0, proto=0, _flags=0):
                        return [(socket.AF_INET, type or socket.SOCK_STREAM, proto, "", (host, port))]

                    socket.getaddrinfo = ipv4_getaddrinfo
                try:
                    with request.urlopen(req, timeout=self.timeout) as response:
                        result = json.loads(response.read().decode("utf-8"))
                except HTTPError as exc:
                    result = json.loads(exc.read().decode("utf-8"))
                    if exc.code != 429 and isinstance(result, dict) and result.get("ok") is False:
                        raise ApiError(str(result.get("description", "")), result, http_status=exc.code) from exc
                    if exc.code != 429:
                        raise
            finally:
                socket.socket = original
                socket.getaddrinfo = original_getaddrinfo
                socks.set_default_proxy()
        if result.get("error_code") == 429:
            parameters = result.get("parameters") or {}
            raise TelegramRateLimitError(float(parameters.get("retry_after", RETRY_FALLBACK_SECONDS)))
        if not result.get("ok"):
            raise ApiError(str(result.get("description", "")), result)
        return result

    def send(self, chat_id: int, text: str, markup: dict[str, Any] | None = None) -> None:
        parts = chunk_html(redact(text))
        for position, part in enumerate(parts):
            payload: dict[str, Any] = {"chat_id": chat_id, "text": part, "parse_mode": "HTML"}
            if markup is not None and position == len(parts) - 1:
                payload["reply_markup"] = markup
            self.call("sendMessage", payload)

    def edit_message(self, chat_id: int, message_id: int, text: str, markup: dict[str, Any] | None = None) -> bool:
        """Rewrite one message in place; returns False when nothing had to change.

        Telegram answers ``400 message is not modified`` when the text and keyboard are
        identical, which is the refresh button pressed twice rather than a failure — and
        an edited card beats a chat that grows a new status message per tap. Long text
        falls back to the caller's own chunked send: an edit cannot be split.
        """
        parts = chunk_html(redact(text))
        if len(parts) != 1:
            return False
        payload: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": parts[0], "parse_mode": "HTML"}
        if markup is not None:
            payload["reply_markup"] = markup
        try:
            self.call("editMessageText", payload)
            return True
        except ApiError as exc:
            if "not modified" in str(exc).lower():
                return False
            raise

    def answer_callback(self, callback_id: str, text: str) -> None:
        """Toast for a button tap. A failure here must not break the action it acknowledges."""
        with contextlib.suppress(Exception):
            self.call("answerCallbackQuery", {"callback_query_id": callback_id, "text": text[:200]})


class PrivateBot:
    def __init__(
        self,
        api: TelegramAPI,
        state_path: Path,
        allowed_user: int,
        allowed_chat: int,
        service: str,
        *,
        store: StateStore | None = None,
        drainer: OutboxDrainer | None = None,
    ) -> None:
        self.api = api
        self.state_path = state_path
        self.allowed_user = allowed_user
        self.allowed_chat = allowed_chat
        self.service = service
        self.offset_path = state_path.parent / "telegram-offset.json"
        self.receipts_path = state_path.parent / "telegram-receipts.json"
        self._store = store
        self._drainer = drainer
        self._stop = Event()
        self._seen_updates: set[int] = set()
        self._seen_order: deque[int] = deque()
        self._receipts: dict[int, str] = {}
        self._receipt_order: deque[int] = deque()
        self._receipt_read_attempts: dict[int, int] = {}
        self._receipt_read_retry_after: dict[int, float] = {}
        self._last_receipt_refresh = 0.0
        self._load_receipts()

    def store(self) -> StateStore:
        if self._store is None:
            self._store = StateStore(self.state_path)
        return self._store

    def drainer(self) -> OutboxDrainer:
        if self._drainer is None:
            self._drainer = OutboxDrainer(
                self.store(),
                TelegramTransport(self.api, self.allowed_chat),
                batch=_int_env("SKYNET_OUTBOX_BATCH", 5),
                max_attempts=_int_env("SKYNET_OUTBOX_MAX_ATTEMPTS", 5),
                backlog_limit=_int_env("SKYNET_OUTBOX_BACKLOG_LIMIT", 20, minimum=0),
                enabled=outbox_enabled(),
                # This drainer runs inside the single-threaded long-poll loop, so an
                # unannounced flood wait must not block it; the rate-limit window
                # and the post-window probe already carry the penalty.
                block_on_rate_limit=_bool_env("SKYNET_OUTBOX_BLOCK_ON_RATE_LIMIT", False),
            )
        return self._drainer

    def request_stop(self, *_args: Any) -> None:
        self._stop.set()

    def install_signal_handlers(self) -> None:
        for name in ("SIGTERM", "SIGINT"):
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                signal.signal(signum, self.request_stop)
            except (ValueError, OSError):
                log.warning("could not install %s handler", name)

    def _seen(self, update_id: int) -> bool:
        if update_id in self._seen_updates:
            return True
        self._seen_updates.add(update_id)
        self._seen_order.append(update_id)
        while len(self._seen_order) > SEEN_UPDATE_LIMIT:
            self._seen_updates.discard(self._seen_order.popleft())
        return False

    def authorized(self, update: dict[str, Any]) -> bool:
        source = update.get("message") or update.get("callback_query") or update.get("my_chat_member")
        if not source:
            return False
        user = source.get("from", {})
        chat = (source.get("chat") or source.get("message", {}).get("chat") or {})
        return user.get("id") == self.allowed_user and chat.get("id") == self.allowed_chat

    def load_number(self, path: Path, default: int = 0) -> int:
        try:
            return int(json.loads(path.read_text()).get("value", default))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return default

    def save_number(self, path: Path, value: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        try:
            with temp.open("w", encoding="utf-8") as stream:
                stream.write(json.dumps({"value": value}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        except OSError:
            with contextlib.suppress(OSError):
                temp.unlink()
            raise
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass

    def command(self, text: str) -> tuple[str, str]:
        parts = text.strip().split(maxsplit=1)
        return parts[0].split("@", 1)[0].lower(), parts[1] if len(parts) == 2 else ""

    def execute(self, chat_id: int, command: str, argument: str) -> None:
        if command == "/help":
            self.api.send(
                chat_id,
                "Канал управления SkyNet.\n"
                "/status — карточка состояния (кнопки: обновить, логи, метрики)\n"
                "/logs [число] — последние прогоны\n"
                "/metrics — метрики и ресурсы\n"
                "/alerts — оповещения\n"
                "/provider — провайдеры и money-boost\n"
                "/answer [id] <текст> — ответить на открытый вопрос; у вопроса есть кнопки\n"
                "/start | /stop | /restart — сервис\n"
                "/reset_memory | /provider_on | /provider_off — обслуживание\n"
                "Обычный текст без «/» уходит организму как сообщение.",
                help_keyboard(),
            )
        elif command == "/status":
            self.send_card(chat_id)
        elif command == "/metrics":
            self.send_metrics(chat_id, argument, verbose=True)
        elif command == "/alerts":
            self.send_alerts(chat_id)
        elif command in {"/provider", "/providers"}:
            self.send_providers(chat_id)
        elif command in {"/provider_on", "/provider_off"}:
            self.confirm(chat_id, "provider_on" if command.endswith("on") else "provider_off")
        elif command in {"/stop", "/restart", "/reset_memory"}:
            self.confirm(chat_id, command[1:])
        elif command == "/logs":
            self.send_logs(chat_id, self.log_run_limit(argument))
        elif command == "/answer":
            self.handle_answer(chat_id, argument)
        elif command == "/start":
            self.systemctl(chat_id, "start")
        else:
            self.api.send(chat_id, "Неизвестная команда. Используйте /help")

    def handle_answer(self, chat_id: int, argument: str) -> None:
        """Record the owner's answer to the newest open question, or to a named one.

        The question may be named by its full id or by a unique prefix: the
        outbox renders the short 8-char form, so a prefix must route. The answer
        also becomes an inbox event, so the organism reads it on the next wake
        even if the run that asked the question is long gone.
        """
        store = self.store()
        open_questions = store.open_questions()
        if not open_questions:
            self.api.send(chat_id, "Нет открытых вопросов.")
            return
        text = argument.strip()
        target: str | None = None
        if text:
            first, _, rest = text.partition(" ")
            ids = [str(item["question_id"]) for item in open_questions]
            if first in ids:
                target, text = first, rest.strip()
            else:
                matches = [item for item in open_questions if str(item["question_id"]).startswith(first)]
                if len(matches) > 1:
                    listing = "\n".join(f"{item['question_id'][:8]}: {item['question']}" for item in matches)
                    self.api.send(chat_id, f"Префикс «{first}» неоднозначен, уточните вопрос:\n{listing}")
                    return
                if len(matches) == 1:
                    target, text = str(matches[0]["question_id"]), rest.strip()
        if target is None:
            target = str(open_questions[-1]["question_id"])
        if not text:
            listing = "\n".join(f"{item['question_id'][:8]}: {item['question']}" for item in open_questions)
            self.api.send(chat_id, f"Использование: /answer <текст>\nОткрытые вопросы:\n{listing}")
            return
        if store.answer_question(target, text, source="telegram"):
            store.add_inbox_event(str(uuid4()), "user_answer", {"question_id": target, "answer": text, "source": "telegram"})
            self.api.send(chat_id, f"Ответ записан для вопроса {target[:8]}.")
        else:
            self.api.send(chat_id, "Вопрос не найден или уже закрыт.")

    def handle_freeform(self, chat_id: int, text: str, message: dict[str, Any]) -> None:
        """Record a free-form owner message; never answer with a canned echo.

        The message always becomes an observation. A reply to exactly one open
        question is additionally recorded as its answer, so the organism reads
        it on the next wake. When several questions are open the message stays
        a general observation and the organism decides what to do with it.
        """
        store = self.store()
        body = text.strip()
        event_id = str(uuid4())
        store.add_inbox_event(event_id, "user_message", {"text": body, "source": "telegram"})
        self._track_receipt(message, event_id)
        if not message.get("reply_to_message"):
            return
        open_questions = store.open_questions()
        if len(open_questions) != 1:
            return
        target = str(open_questions[0]["question_id"])
        if store.answer_question(target, body, source="telegram"):
            store.add_inbox_event(str(uuid4()), "user_answer", {"question_id": target, "answer": body, "source": "telegram"})

    def _track_receipt(self, message: dict[str, Any], event_id: str) -> None:
        """Remember which chat message carries which inbox event, for receipts."""
        message_id = message.get("message_id")
        if not isinstance(message_id, int):
            return
        self._receipts[message_id] = event_id
        self._receipt_order.append(message_id)
        while len(self._receipt_order) > RECEIPT_LIMIT:
            self._forget_receipt(self._receipt_order.popleft())
        self._react(message_id, RECEIPT_SEEN)
        self._save_receipts()

    def _forget_receipt(self, message_id: int) -> None:
        """Drop every trace of one receipt so the ledger cannot grow stale."""
        self._receipts.pop(message_id, None)
        self._receipt_read_attempts.pop(message_id, None)
        self._receipt_read_retry_after.pop(message_id, None)
        with contextlib.suppress(ValueError):
            self._receipt_order.remove(message_id)

    def _react(self, message_id: int, emoji: str) -> bool:
        """Set one receipt reaction; a missing permission must never break intake."""
        try:
            self.api.call(
                "setMessageReaction",
                {
                    "chat_id": self.allowed_chat,
                    "message_id": message_id,
                    "reaction": json.dumps([{"type": "emoji", "emoji": emoji}]),
                },
            )
            return True
        except Exception as exc:
            log.info("read receipt reaction failed message_id=%s emoji=%s: %s", message_id, emoji, exc)
            return False

    def _refresh_receipts(self) -> None:
        """Upgrade "fetched" to "read" for messages a run actually carried.

        The organism records ``inbox_delivered_in_run`` with the event ids it put
        in front of the model, so the check mark means the text reached a ReAct
        context -- not that a reply was sent. A permanently refused reaction used
        to be retried on every poll for the life of the entry; each entry is now
        attempted at most ``RECEIPT_MAX_ATTEMPTS`` times with exponential
        backoff, then dropped. The whole pass is a bounded side task on the
        single poll loop: it runs at most once per cadence window and attempts at
        most ``RECEIPT_REFRESH_BATCH`` reactions, so a burst of upgrades cannot
        stall command intake or the outbox drain.
        """
        if not self._receipts:
            return
        now = time.monotonic()
        if now - self._last_receipt_refresh < RECEIPT_REFRESH_INTERVAL_SECONDS:
            return
        self._last_receipt_refresh = now
        # Driven by the ledger's own event ids, not by a global most-recent-N
        # window: a window can be outrun (more than RECEIPT_LIMIT deliveries
        # between two passes), which stranded an older message on "seen" forever.
        # The ledger is bounded to RECEIPT_LIMIT entries, so the wanted set is
        # bounded; the join cannot miss an entry the ledger still holds.
        ledger_ids = sorted({str(value) for value in self._receipts.values() if value})
        if not ledger_ids:
            return
        try:
            rows = self.store().connection.execute(
                "SELECT DISTINCT delivered.value AS event_id "
                "FROM event_log, json_each(event_log.payload, '$.event_ids') AS delivered, "
                "json_each(?) AS wanted "
                "WHERE event_log.kind='inbox_delivered_in_run' AND delivered.value = wanted.value",
                (json.dumps(ledger_ids),),
            ).fetchall()
        except Exception:
            log.exception("read receipt ledger read failed")
            return
        read_ids = {str(row[0]) for row in rows}
        if not read_ids:
            return
        changed = False
        attempted = 0
        for message_id in list(self._receipt_order):
            if attempted >= RECEIPT_REFRESH_BATCH:
                break
            if self._receipts.get(message_id) not in read_ids:
                continue
            if now < self._receipt_read_retry_after.get(message_id, 0.0):
                continue
            attempted += 1
            if self._react(message_id, RECEIPT_READ):
                self._forget_receipt(message_id)
                changed = True
                continue
            attempts = self._receipt_read_attempts.get(message_id, 0) + 1
            if attempts >= RECEIPT_MAX_ATTEMPTS:
                log.warning("dropping read receipt after %s failed attempts message_id=%s", attempts, message_id)
                self._forget_receipt(message_id)
                changed = True
                continue
            self._receipt_read_attempts[message_id] = attempts
            self._receipt_read_retry_after[message_id] = now + RECEIPT_RETRY_BACKOFF_SECONDS * (2 ** (attempts - 1))
            changed = True
        if changed:
            self._save_receipts()

    def _load_receipts(self) -> None:
        """Restore the seen->read ledger so a restart cannot strand a receipt.

        Receipt state lived only in memory, so a message fetched before a bot
        restart kept the "seen" reaction forever even after a run read it, while
        the same message upgraded correctly without a restart. The map is small
        and writes are rare, so it is persisted next to the offset file. The
        failed-attempt counter is persisted too so a restart cannot reset the
        bound and re-open the retry storm. The backoff deadline is stored as a
        wall-clock ``retry_after`` (a monotonic reading does not survive a
        reboot) and translated back to the local clock on load, so one immediate
        retry is not granted across a restart. Sidecars written before the field
        existed simply have no deadline, which is backward compatible.
        """
        try:
            data = json.loads(self.receipts_path.read_text())
            order = [int(item) for item in data.get("order", [])]
            receipts = {int(key): str(value) for key, value in data.get("receipts", {}).items()}
            attempts: dict[int, int] = {}
            raw_attempts = data.get("read_attempts")
            if isinstance(raw_attempts, dict):
                for key, value in raw_attempts.items():
                    try:
                        attempts[int(key)] = max(0, int(value))
                    except (TypeError, ValueError):
                        continue
            retry_after: dict[int, float] = {}
            raw_retry_after = data.get("retry_after")
            if isinstance(raw_retry_after, dict):
                for key, value in raw_retry_after.items():
                    try:
                        retry_after[int(key)] = float(value)
                    except (TypeError, ValueError):
                        continue
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return
        wall_now = time.time()
        for message_id in order:
            if message_id in receipts and message_id not in self._receipts:
                self._receipts[message_id] = receipts[message_id]
                self._receipt_order.append(message_id)
                if message_id in attempts:
                    self._receipt_read_attempts[message_id] = attempts[message_id]
                if retry_after.get(message_id, 0.0) > wall_now:
                    self._receipt_read_retry_after[message_id] = time.monotonic() + (retry_after[message_id] - wall_now)

    def _save_receipts(self) -> None:
        """Persist the receipt ledger atomically; failure must never break intake."""
        path = self.receipts_path
        wall_now = time.time()
        mono_now = time.monotonic()
        payload = json.dumps({
            "order": list(self._receipt_order),
            "receipts": {str(key): value for key, value in self._receipts.items()},
            "read_attempts": {
                str(key): value
                for key, value in self._receipt_read_attempts.items()
                if key in self._receipts
            },
            "retry_after": {
                str(key): wall_now + max(0.0, value - mono_now)
                for key, value in self._receipt_read_retry_after.items()
                if key in self._receipts
            },
        })
        temp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with temp.open("w", encoding="utf-8") as stream:
                stream.write(payload + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        except OSError:
            with contextlib.suppress(OSError):
                temp.unlink()
            log.warning("could not persist receipt ledger", exc_info=True)

    def status_days(self, argument: str) -> float:
        try:
            value = float(argument.strip())
        except (TypeError, ValueError):
            return DEFAULT_STATUS_DAYS
        return min(MAX_STATUS_DAYS, max(MIN_STATUS_DAYS, value))

    def log_run_limit(self, argument: str) -> int:
        try:
            value = int(argument.strip())
        except (TypeError, ValueError):
            return DEFAULT_LOG_RUNS
        return min(MAX_LOG_RUNS, max(1, value))

    def status_text(self, argument: str = "", *, verbose: bool = False) -> str:
        """The status card's text, so the /status command and its button cannot drift apart."""
        data = metrics.snapshot(
            self.store(),
            since_days=self.status_days(argument),
            registry_path=self.state_path.parent / "self-improvement-proposals.json",
            state_dir=self.state_path.parent,
        )
        text = metrics.format_report(data)
        if verbose:
            extra = json.dumps(
                {"proposals": data.get("proposals"), "resources": data.get("resources")},
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            text = f"{text}\n\n{extra}"
        return text

    def card_text(self) -> str:
        """The status card: what she is doing *now*, not a window of statistics.

        ``/metrics`` keeps the aggregate dump; this is the one screen the owner opens to
        learn whether she is alive, busy, stuck, waiting for him, and whether the running
        code is the committed code.
        """
        return status_card.card(
            self.store(),
            service=self.service,
            repo=self.state_path.parent.parent,
            state_dir=self.state_path.parent,
            registry_path=self.state_path.parent / "self-improvement-proposals.json",
        )

    def send_card(self, chat_id: int) -> None:
        self.api.send(chat_id, self.card_text(), status_keyboard())

    def send_metrics(self, chat_id: int, argument: str, *, verbose: bool) -> None:
        self.api.send(chat_id, self.status_text(argument, verbose=verbose), status_keyboard())

    def send_alerts(self, chat_id: int) -> None:
        alerts = sorted(
            self.store().pending_alerts(limit=20),
            key=lambda item: str(item.get("last_seen_at", "")),
            reverse=True,
        )
        if not alerts:
            self.api.send(chat_id, "Оповещений нет.")
            return
        lines: list[str] = []
        for alert in alerts:
            lines.append(f"[{alert.get('severity')}] {alert.get('kind')} x{alert.get('occurrences')} {alert.get('last_seen_at')}")
            lines.append("  " + json.dumps(alert.get("payload"), ensure_ascii=False, default=str))
        self.api.send(chat_id, "\n".join(lines))

    def systemctl(self, chat_id: int, action: str) -> None:
        # The park is written BEFORE the unit moves: the revive timer ticks every 30
        # minutes and a stop it does not know about is a stop it undoes -- the owner's
        # 22:07:54 `/stop` came back at 01:00:30 exactly that way, and his own voice
        # command has to beat a timer. The path is the bot's own ledger: one file,
        # named by whoever knows where the state lives.
        park_file = self.state_path.with_name(PARK_FILE)
        if action == "stop":
            set_park("owner /stop from the Telegram bot", park_file)
        elif action == "start":
            clear_park(park_file)
        try:
            result = subprocess.run(systemctl_command(action, self.service), capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            self.api.send(chat_id, f"{action}: timeout; проверяю фактическое состояние")
            try:
                state = subprocess.run(systemctl_command("is-active", self.service), capture_output=True, text=True, timeout=10)
                actual = state.stdout.strip() or state.stderr.strip() or "unknown"
                self.api.send(chat_id, f"{action}: {actual}")
            except subprocess.TimeoutExpired:
                self.api.send(chat_id, f"{action}: timeout; состояние неизвестно")
            return
        log.info("systemctl action=%s service=%s returncode=%s", action, self.service, result.returncode)
        if result.returncode:
            self.api.send(chat_id, f"{action}: failed\n{result.stderr.strip()}")
            return
        state = subprocess.run(systemctl_command("is-active", self.service), capture_output=True, text=True, timeout=10)
        actual = state.stdout.strip() or state.stderr.strip() or "unknown"
        self.api.send(chat_id, f"{action}: {actual}")

    def confirm(self, chat_id: int, action: str) -> None:
        if action in {"stop", "restart"}:
            self.systemctl(chat_id, action)
        elif action == "reset_memory":
            result = subprocess.run([sys.executable, "-m", "skynet", "reset-short-memory", "--state", str(self.state_path)], capture_output=True, text=True, timeout=30)
            self.api.send(chat_id, result.stdout or result.stderr)
        elif action in {"provider_on", "provider_off"}:
            result = subprocess.run([sys.executable, "-m", "skynet", "money-boost", "on" if action.endswith("on") else "off", "--service", self.service], capture_output=True, text=True, timeout=30)
            self.api.send(chat_id, result.stdout or result.stderr)

    def send_logs(self, chat_id: int, limit: int = DEFAULT_LOG_RUNS) -> None:
        """Send the owner the last few run pairs in compact form.

        Rendering lives in ``skynet.reporting`` so the bot never re-derives the
        run/report structure and never dumps the raw English runtime record.
        """
        try:
            from . import reporting

            messages = reporting.render_recent(self.store(), limit)
        except Exception:
            log.exception("could not render recent run logs")
            messages = []
        if not messages:
            self.api.send(chat_id, NO_LOGS_TEXT, logs_keyboard(limit))
            return
        for position, message in enumerate(messages):
            # The window buttons ride the last message only: a keyboard belongs to the
            # message it sits on, and one set of controls is enough.
            self.api.send(chat_id, message, logs_keyboard(limit) if position == len(messages) - 1 else None)

    def providers_text(self) -> str:
        """The rung list the owner switches, in the three states he must tell apart.

        ON, switched off from the phone (and therefore waiting for the next heartbeat)
        and parked by an ``*_ENABLED`` flag in the environment all look identical from
        inside the chain, so the card names every one. The flags are the owner's older
        switches and this command does not touch them; what it changes is only the
        desired state the reactor copies across at the next tick.
        """
        configured = configured_provider_names()
        off = {str(item) for item in load_provider_prefs()["off"]}
        live = set(active_chain_names())
        applied = applied_chain_names()
        paid = [name for name in configured if provider_class(name) == "paid"]
        free = [name for name in configured if provider_class(name) != "paid"]

        def line(name: str) -> str:
            mark = "✅" if name in live else "⛔"
            if name in off:
                state = "выключен тобой"
            elif name not in live:
                state = "запаркован флагом"
            else:
                state = "работает"
            return f"{mark} {name} — {state}"

        active = next((name for name in configured if name in live), "")
        waiting = [name for name in configured if applied is not None and (name in live) != (name in applied)]
        boost = money_boost_enabled()
        lines = ["🧠 Провайдеры", "", "Платные:", *[line(name) for name in paid], "Бесплатные:", *[line(name) for name in free], ""]
        lines.append(f"Цепочка: {len(live)} из {len(configured)} рангов")
        if active:
            lines.append(f"Потолок класса {provider_class(active)}: {context_ceiling_tokens(active)} токенов — по первому живому рангу")
        if waiting:
            lines.append("Ждут следующего heartbeat: " + ", ".join(waiting))
        lines.append(f"Money boost: {'вкл' if boost else 'выкл'}")
        lines.append(f"Стоянка владельца: {'да' if park_state(self.state_path.with_name(PARK_FILE)) else 'нет'}")
        return "\n".join(lines)

    def send_providers(self, chat_id: int, message_id: Any = None) -> None:
        """Render the provider card, in place when it was tapped from a card."""
        live = set(active_chain_names())
        entries = [(f"{'✅' if name in live else '⛔'} {name}", name) for name in configured_provider_names()]
        text = self.providers_text()
        keyboard = providers_keyboard(entries)
        if isinstance(message_id, int) and self.api.edit_message(chat_id, message_id, text, keyboard):
            return
        self.api.send(chat_id, text, keyboard)

    def toggle_provider(self, args: list[str], callback_id: str, message_id: Any = None) -> None:
        """Flip one rung's owner switch (DESIRED state only) and re-render the card.

        Nothing about the running chain changes here: the reactor copies the desired
        state into the applied chain at the next heartbeat, so the toast says what will
        happen instead of implying the ladder already moved.
        """
        if not args:
            self.api.answer_callback(callback_id, "Кнопка устарела")
            return
        name = args[0]
        off_now = name in {str(item) for item in load_provider_prefs()["off"]}
        set_provider_off(name, not off_now)
        self.api.answer_callback(callback_id, f"{name}: {'включён' if off_now else 'выключен'} — со следующего heartbeat")
        self.send_providers(self.allowed_chat, message_id)

    def process(self, update: dict[str, Any]) -> None:
        update_id = update.get("update_id")
        if isinstance(update_id, int) and self._seen(update_id):
            log.info("ignored duplicate Telegram update=%s", update_id)
            return
        if not self.authorized(update):
            log.warning("ignored unauthorized Telegram update=%s", update_id)
            return
        callback = update.get("callback_query")
        if callback:
            self.handle_callback(callback)
            return
        message = update.get("message")
        if message and isinstance(message.get("text"), str):
            text = message["text"].strip()
            if text.startswith("/"):
                command, argument = self.command(text)
                log.info("authorized Telegram command=%s chat_id=%s", command, self.allowed_chat)
                self.execute(self.allowed_chat, command, argument)
            else:
                log.info("authorized Telegram message chat_id=%s", self.allowed_chat)
                self.handle_freeform(self.allowed_chat, text, message)

    def handle_callback(self, callback: dict[str, Any]) -> None:
        """Route one button tap. Authorization already happened in ``process``.

        Each kind maps onto the same code path its text command uses — answering a
        question, running a service action, rendering a card — so a button and a typed
        command can never drift apart, and the callback grammar (``telegram_ui``) carries
        actions only, never text to execute.
        """
        data = str(callback.get("data", ""))
        callback_id = str(callback.get("id", ""))
        log.info("authorized Telegram callback=%s chat_id=%s", data, self.allowed_chat)
        kind, args = parse_callback(data)
        raw_message = callback.get("message")
        message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
        message_id = message.get("message_id")
        if kind == "cancel":
            self.api.answer_callback(callback_id, "Отменено")
            self.api.send(self.allowed_chat, "Отменено.")
        elif kind == "confirm":
            self.api.answer_callback(callback_id, "Принято")
            self.confirm(self.allowed_chat, args[0] if args else "")
        elif kind == "q":
            self.api.answer_callback(callback_id, self.answer_from_button(args, message))
        elif kind == "pv":
            self.toggle_provider(args, callback_id, message_id)
        elif kind == "ui":
            self.api.answer_callback(callback_id, "Принято")
            self.run_card_action(args, message_id)
        else:
            self.api.answer_callback(callback_id, "Кнопка устарела")

    def answer_from_button(self, args: list[str], message: dict[str, Any]) -> str:
        """Record the answer a question button stands for; returns the toast text.

        The option index is resolved against the question the store still holds open, not
        against the label printed on the button: the store's options are the authority, and
        a question that expired or was answered meanwhile is reported as closed instead of
        being answered twice. This is also how a sudo approval is granted — the question
        simply carries ``Разрешить`` / ``Запретить`` as its options.
        """
        if len(args) < 2:
            return "Кнопка устарела"
        short_id, index_text = args[0], args[1]
        open_questions = self.store().open_questions()
        matches = [item for item in open_questions if str(item["question_id"]).startswith(short_id)]
        if len(matches) != 1:
            return "Вопрос уже закрыт"
        question = matches[0]
        raw_options = question.get("options")
        options: list[Any] = raw_options if isinstance(raw_options, list) else []
        if not options:
            return "Кнопка устарела"
        try:
            answer = str(options[int(index_text)]).strip()
        except (IndexError, TypeError, ValueError):
            return "Кнопка устарела"
        if not answer:
            return "Кнопка устарела"
        question_id = str(question["question_id"])
        if not self.store().answer_question(question_id, answer, source="telegram-button"):
            return "Вопрос уже закрыт"
        self.store().add_inbox_event(
            str(uuid4()), "user_answer", {"question_id": question_id, "answer": answer, "source": "telegram-button"}
        )
        if isinstance(message.get("message_id"), int) and isinstance(message.get("text"), str):
            # Strip the keyboard: the question is answered, and a live button would lie.
            message_id = message["message_id"]
            self.api.edit_message(self.allowed_chat, message_id, f"{message['text']}\n\n✅ ответ: {answer}")
        return f"Ответ: {answer}"

    def run_card_action(self, args: list[str], message_id: Any) -> None:
        """A card button: refresh in place when it came from a card, else send the view."""
        action = args[0] if args else ""
        if action == "status":
            text = self.card_text()
            if isinstance(message_id, int) and self.api.edit_message(self.allowed_chat, message_id, text, status_keyboard()):
                return
            self.api.send(self.allowed_chat, text, status_keyboard())
        elif action == "logs":
            self.send_logs(self.allowed_chat, self.log_run_limit(args[1] if len(args) > 1 else ""))
        elif action == "metrics":
            self.send_metrics(self.allowed_chat, "", verbose=True)
        elif action == "alerts":
            self.send_alerts(self.allowed_chat)
        elif action == "provider":
            self.send_providers(self.allowed_chat, message_id)
        else:
            self.api.send(self.allowed_chat, "Неизвестная кнопка.")

    def _drain_safely(self) -> None:
        try:
            self.drainer().drain()
        except Exception:
            log.exception("outbox drain failed; continuing")

    def _notify_failure(self, exc: BaseException) -> None:
        try:
            self.api.send(self.allowed_chat, f"Команда завершилась ошибкой: {redact(str(exc))[:200]}")
        except Exception:
            log.warning("could not notify Telegram command failure")

    def _confirm_backlog(self) -> int:
        """Confirm a pending Telegram backlog on a cold start without executing it.

        A missing offset file means fresh state (redeploy, purge, state reset):
        Telegram would then replay every pending update still inside its 24h
        window, and a stale ``/stop`` left in that backlog would stop the service
        seconds after it comes up. ``getUpdates(offset=-1)`` returns only the
        newest update and confirms everything older as soon as the next call uses
        the returned offset, so a cold start resumes from *now* and never runs a
        command sent before this process existed. With nothing pending the offset
        stays 0. Called inside the poll loop so a transient failure is retried by
        the loop's own handler instead of falling back to a replay.
        """
        pending = self.api.call("getUpdates", {"offset": -1, "timeout": 0, "allowed_updates": ALLOWED_UPDATES}).get("result", [])
        offset = max((int(update["update_id"]) for update in pending), default=-1) + 1
        self.save_number(self.offset_path, offset)
        if pending:
            log.info("cold start: confirmed %d pending Telegram update(s) without executing them; resuming at offset=%d", len(pending), offset)
        return offset

    def register_commands(self) -> None:
        """Publish the command menu for this chat; a failure is cosmetic, never fatal.

        The menu is what turns a chat into an interface: the owner types one letter and
        taps. Scoped to the owner's chat so the bot cannot advertise itself anywhere else,
        and re-published on every start because the description text lives here, in code.
        """
        commands = [
            {"command": "status", "description": "карточка состояния"},
            {"command": "logs", "description": "последние прогоны"},
            {"command": "metrics", "description": "метрики и ресурсы"},
            {"command": "alerts", "description": "оповещения"},
            {"command": "providers", "description": "ранги провайдеров: включить/выключить"},
            {"command": "answer", "description": "ответить на открытый вопрос"},
            {"command": "start", "description": "запустить сервис"},
            {"command": "stop", "description": "остановить сервис"},
            {"command": "restart", "description": "перезапустить сервис"},
            {"command": "reset_memory", "description": "сбросить короткую память"},
            {"command": "help", "description": "список команд и кнопки"},
        ]
        try:
            self.api.call(
                "setMyCommands",
                {"commands": commands, "scope": {"type": "chat", "chat_id": self.allowed_chat}},
            )
        except Exception:
            log.warning("could not register the Telegram command menu", exc_info=True)

    def run(self) -> None:
        startup_delay = 5.0
        while not self._stop.is_set():
            try:
                self.api.call("getMe")
                break
            except Exception as exc:
                log.warning("Telegram API startup failed; retrying in %.1fs: %s", startup_delay, exc)
                self._stop.wait(startup_delay)
                startup_delay = min(60.0, startup_delay * 2)
        offset = self.load_number(self.offset_path, default=-1)
        backlog_confirmed = offset >= 0
        self._drain_safely()
        self.register_commands()
        while not self._stop.is_set():
            try:
                if not backlog_confirmed:
                    offset = self._confirm_backlog()
                    backlog_confirmed = True
                updates = self.api.call("getUpdates", {"offset": offset, "timeout": POLL_TIMEOUT, "allowed_updates": ALLOWED_UPDATES}).get("result", [])
                for update in updates:
                    next_offset = max(offset, int(update["update_id"]) + 1)
                    try:
                        self.process(update)
                    except Exception as exc:
                        log.exception("Telegram command failed update=%s", update.get("update_id"))
                        self._notify_failure(exc)
                    offset = next_offset
                    self.save_number(self.offset_path, offset)
                self._drain_safely()
                self._refresh_receipts()
            except TelegramRateLimitError as exc:
                log.warning("Telegram rate limit; retrying in %.1fs", exc.retry_after)
                self._stop.wait(exc.retry_after)
            except Exception as exc:
                # A Tor circuit or Bot API timeout must not destroy the control
                # process or discard the unacknowledged log offset.
                log.warning("Telegram loop temporarily unavailable; retrying: %s", exc)
                self._stop.wait(RETRY_FALLBACK_SECONDS)

def main() -> int:
    parser = argparse.ArgumentParser(description="Private SkyNet Telegram control bot")
    parser.add_argument("--state", default=os.getenv("SKYNET_STATE", "state/skynet.sqlite3"))
    parser.add_argument("--service", default=os.getenv("SKYNET_SERVICE", "skynet.service"))
    args = parser.parse_args()
    logging.basicConfig(level=os.getenv("SKYNET_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    token = os.getenv("SKYNET_TELEGRAM_BOT_TOKEN", "")
    if not token:
        log.error("SKYNET_TELEGRAM_BOT_TOKEN is not configured")
        return 2
    try:
        user = int(os.environ["SKYNET_TELEGRAM_ALLOWED_USER_ID"])
        chat = int(os.environ["SKYNET_TELEGRAM_ALLOWED_CHAT_ID"])
    except (KeyError, ValueError):
        log.exception("Telegram allowlist is not configured")
        return 2
    proxy = os.getenv("SKYNET_TELEGRAM_PROXY", "")
    bot = PrivateBot(TelegramAPI(token, proxy), Path(args.state), user, chat, args.service)
    bot.install_signal_handlers()
    bot.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
