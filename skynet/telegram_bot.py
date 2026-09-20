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
from collections import deque
from pathlib import Path
from threading import Event, Lock
from typing import Any
from urllib import parse, request
from urllib.error import HTTPError
from uuid import uuid4

import socks

from . import metrics
from .outbox import OutboxDrainer, TelegramTransport
from .providers import money_boost_state_path
from .store import StateStore
from .time import display_timestamp

log = logging.getLogger("skynet.telegram")
MAX_MESSAGE = 3900
POLL_TIMEOUT = 30
RETRY_FALLBACK_SECONDS = 30
SEEN_UPDATE_LIMIT = 512
DEFAULT_STATUS_DAYS = 1.0
MIN_STATUS_DAYS = 0.1
MAX_STATUS_DAYS = 90.0
DEFAULT_LOG_RUNS = 5
MAX_LOG_RUNS = 10
NO_LOGS_TEXT = "Логов пока нет."
SECRET_RE = re.compile(r"(?i)(token|api[_-]?key|password|secret)=\S+")
BOT_TOKEN_RE = re.compile(r"\d{8,12}:[A-Za-z0-9_-]{30,}")
SENSITIVE_KEYS = {"reasoning_content", "api_key", "token", "password", "secret"}


def outbox_enabled() -> bool:
    return os.getenv("SKYNET_OUTBOX_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default
def runtime_event_kind(record: dict[str, Any]) -> str:
    """Return the operational kind for direct and wrapped runtime events."""
    payload = record.get("payload")
    if record.get("kind") == "event" and isinstance(payload, dict):
        nested = payload.get("event_kind")
        if isinstance(nested, str):
            return nested
    return str(record.get("kind", "event"))
TELEGRAM_EVENT_KINDS = {
    "run_started", "run_finished", "run_failure", "tool_call", "tool_result",
    "tool_failure", "tool_retry", "provider_error_classified", "provider_retry",
    "finish_report", "emergency_finish",
    "memory_loop_started", "memory_loop_finished", "memory_loop_failed", "memory_consolidated",
    "proposal_created", "proposal_failed",
    "supervisor_start", "supervisor_stop", "watchdog_timeout",
}


def chunks(text: str, limit: int = MAX_MESSAGE) -> list[str]:
    text = text or "(empty)"
    return [text[i:i + limit] for i in range(0, len(text), limit)] or ["(empty)"]


class TelegramRateLimitError(RuntimeError):
    def __init__(self, retry_after: float) -> None:
        self.retry_after = max(1.0, retry_after)
        super().__init__(f"Telegram rate limit; retry after {self.retry_after:g}s")


def redact(text: str) -> str:
    return BOT_TOKEN_RE.sub("[bot-token]", SECRET_RE.sub(r"\1=[redacted]", text))


def sanitize(value: Any, key: str = "") -> Any:
    """Remove reasoning and credential-shaped fields before remote delivery."""
    if key.lower() in SENSITIVE_KEYS or key.lower().endswith("_token"):
        return "[redacted]"
    if isinstance(value, dict):
        return {name: sanitize(item, name) for name, item in value.items() if name.lower() not in SENSITIVE_KEYS}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def format_runtime_record(record: dict[str, Any]) -> str:
    timestamp = "unknown time"
    if record.get("timestamp"):
        try:
            timestamp = display_timestamp(str(record["timestamp"]))
        except (TypeError, ValueError):
            log.warning("ignoring invalid runtime timestamp=%r", record.get("timestamp"))
    kind = runtime_event_kind(record)
    run_id = record.get("run_id")
    payload = record.get("payload", {})
    if record.get("kind") == "event" and isinstance(payload, dict):
        payload = payload.get("payload", payload)
    payload = sanitize(payload)
    prefix = f"[{timestamp}] {kind}"
    if run_id:
        prefix += f" run={run_id}"
    return redact(prefix + "\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str))


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
            raise RuntimeError(str(result))
        return result

    def send(self, chat_id: int, text: str) -> None:
        for part in chunks(redact(text)):
            self.call("sendMessage", {"chat_id": chat_id, "text": part})


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
        self._store = store
        self._drainer = drainer
        self._stop = Event()
        self._seen_updates: set[int] = set()
        self._seen_order: deque[int] = deque()

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
            self.api.send(chat_id, "Команды: /status [дни], /alerts, /metrics, /answer <текст>, /start, /stop, /restart, /provider, /provider_on, /provider_off, /reset_memory, /logs [число], /send <текст>. Обычный текст без «/» уходит организму как сообщение.")
        elif command == "/status":
            self.send_metrics(chat_id, argument, verbose=False)
        elif command == "/metrics":
            self.send_metrics(chat_id, argument, verbose=True)
        elif command == "/alerts":
            self.send_alerts(chat_id)
        elif command == "/provider":
            path = money_boost_state_path()
            enabled = False
            with contextlib.suppress(OSError, ValueError, TypeError, json.JSONDecodeError):
                enabled = bool(json.loads(path.read_text()).get("enabled", False))
            self.api.send(chat_id, json.dumps({"money_boost": enabled, "state_path": str(path)}, ensure_ascii=False))
        elif command in {"/provider_on", "/provider_off"}:
            self.confirm(chat_id, "provider_on" if command.endswith("on") else "provider_off")
        elif command in {"/stop", "/restart", "/reset_memory"}:
            self.confirm(chat_id, command[1:])
        elif command == "/logs":
            self.send_logs(chat_id, self.log_run_limit(argument))
        elif command == "/answer":
            self.handle_answer(chat_id, argument)
        elif command == "/send":
            if not argument.strip():
                self.api.send(chat_id, "Использование: /send <сообщение>")
                return
            self.store().add_inbox_event(str(uuid4()), "user_message", {"text": argument.strip(), "source": "telegram"})
            self.api.send(chat_id, "Сообщение поставлено в inbox.")
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
        store.add_inbox_event(str(uuid4()), "user_message", {"text": body, "source": "telegram"})
        if not message.get("reply_to_message"):
            return
        open_questions = store.open_questions()
        if len(open_questions) != 1:
            return
        target = str(open_questions[0]["question_id"])
        if store.answer_question(target, body, source="telegram"):
            store.add_inbox_event(str(uuid4()), "user_answer", {"question_id": target, "answer": body, "source": "telegram"})

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

    def send_metrics(self, chat_id: int, argument: str, *, verbose: bool) -> None:
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
        self.api.send(chat_id, text)

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
        try:
            result = subprocess.run(["systemctl", action, self.service], capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            self.api.send(chat_id, f"{action}: timeout; проверяю фактическое состояние")
            try:
                state = subprocess.run(["systemctl", "is-active", self.service], capture_output=True, text=True, timeout=10)
                actual = state.stdout.strip() or state.stderr.strip() or "unknown"
                self.api.send(chat_id, f"{action}: {actual}")
            except subprocess.TimeoutExpired:
                self.api.send(chat_id, f"{action}: timeout; состояние неизвестно")
            return
        log.info("systemctl action=%s service=%s returncode=%s", action, self.service, result.returncode)
        if result.returncode:
            self.api.send(chat_id, f"{action}: failed\n{result.stderr.strip()}")
            return
        state = subprocess.run(["systemctl", "is-active", self.service], capture_output=True, text=True, timeout=10)
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
            self.api.send(chat_id, NO_LOGS_TEXT)
            return
        for message in messages:
            self.api.send(chat_id, message)

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
            action = str(callback.get("data", ""))
            log.info("authorized Telegram callback=%s chat_id=%s", action, self.allowed_chat)
            self.api.call("answerCallbackQuery", {"callback_query_id": callback.get("id"), "text": "Принято"})
            if action == "cancel":
                self.api.send(self.allowed_chat, "Отменено.")
            elif action.startswith("confirm:"):
                self.confirm(self.allowed_chat, action.split(":", 1)[1])
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
        offset = self.load_number(self.offset_path)
        self._drain_safely()
        while not self._stop.is_set():
            try:
                updates = self.api.call("getUpdates", {"offset": offset, "timeout": POLL_TIMEOUT, "allowed_updates": ["message", "callback_query"]}).get("result", [])
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
            except TelegramRateLimitError as exc:
                log.warning("Telegram rate limit; retrying in %.1fs", exc.retry_after)
                self._stop.wait(exc.retry_after)
            except Exception as exc:
                # A proxy circuit or Bot API timeout must not destroy the control
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
