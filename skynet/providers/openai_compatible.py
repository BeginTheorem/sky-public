from __future__ import annotations

import json
import socket
import threading
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from urllib import error, request
from urllib.parse import urlparse

try:
    import socks as _socks
    _HAS_SOCKS = True
except ImportError:
    _HAS_SOCKS = False

from ..models import ModelTurn, ToolCall
from ..provider import Message, ToolSchema
from ..time import utc_datetime_now
from .errors import ProviderError, tool_arguments

# `urllib` has no SOCKS support, so the process-global socket factory is
# patched for the duration of a proxy-scoped request. A depth counter keeps
# nested/concurrent scopes from restoring the patch while another scope is
# still active.
_PROXY_LOCK = threading.Lock()
_PROXY_DEPTH = 0
_PROXY_ORIGINAL: Any = None


def _socks_connection_factory(socks_type: int, proxy_host: str, proxy_port: int):
    def _patched(addr: tuple[str, int], timeout: float = 30, _source_addr: tuple[str, int] | None = None) -> socket.socket:
        s = _socks.socksocket()  # type: ignore[name-defined]
        s.set_proxy(socks_type, proxy_host, proxy_port)
        s.settimeout(timeout)
        s.connect(addr)
        return s  # type: ignore[return-value]
    return _patched


@dataclass(slots=True)
class OpenAICompatibleProvider:
    name: str
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 600.0
    max_output_tokens: int | None = None
    reject_reasoning_leakage: bool = False
    temperature: float | None = None
    top_p: float | None = None
    proxy_url: str | None = None
    cooldown_overrides: dict[int, float] | None = None
    network_cooldown_seconds: float = 2.0
    default_headers: dict[str, str] | None = None

    accepts_timeout_override = True

    @contextmanager  # type: ignore[type-var]
    def _proxy_scope(self):
        global _PROXY_DEPTH, _PROXY_ORIGINAL
        if not self.proxy_url:
            yield
            return
        if not _HAS_SOCKS:
            raise ProviderError(
                f"{self.name} proxy configured but pysocks is not installed",
                category="configuration",
                retryable=False,
                provider=self.name,
                model=self.model,
            )
        parsed = urlparse(self.proxy_url)
        socks_type = _socks.SOCKS5  # type: ignore[name-defined]
        if parsed.scheme == "socks4":
            socks_type = _socks.SOCKS4  # type: ignore[name-defined]
        proxy_host = parsed.hostname or "127.0.0.1"
        proxy_port = parsed.port or 9050
        with _PROXY_LOCK:
            if _PROXY_DEPTH == 0:
                _PROXY_ORIGINAL = socket.create_connection
                socket.create_connection = _socks_connection_factory(socks_type, proxy_host, proxy_port)  # type: ignore[assignment]
            _PROXY_DEPTH += 1
        try:
            yield
        finally:
            with _PROXY_LOCK:
                _PROXY_DEPTH -= 1
                if _PROXY_DEPTH <= 0:
                    _PROXY_DEPTH = 0
                    if _PROXY_ORIGINAL is not None:
                        socket.create_connection = _PROXY_ORIGINAL  # type: ignore[assignment]
                        _PROXY_ORIGINAL = None

    def health_probe(self) -> dict[str, object]:
        url = f"{self.base_url.rstrip('/')}/models"
        headers = {"Accept": "application/json", **(self.default_headers or {})}
        if self.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with self._proxy_scope(), request.urlopen(request.Request(url, headers=headers), timeout=5.0) as response:
                return {"ok": 200 <= response.status < 300, "provider": self.name, "model": self.model, "endpoint": url, "hostname": urlparse(url).hostname, "proxy": self.proxy_url}
        except error.HTTPError as exc:
            return {"ok": False, "provider": self.name, "model": self.model, "endpoint": url, "http_status": exc.code, "proxy": self.proxy_url}
        except (error.URLError, TimeoutError, OSError) as exc:
            return {"ok": False, "provider": self.name, "model": self.model, "endpoint": url, "error": str(exc)[:500], "proxy": self.proxy_url}

    def complete(self, messages: Sequence[Message], *, max_tokens: int, tools: Sequence[ToolSchema] = (), timeout_seconds: float | None = None) -> ModelTurn:
        effective_max_tokens = min(max_tokens, self.max_output_tokens) if self.max_output_tokens else max_tokens
        payload: dict[str, Any] = {"model": self.model, "messages": list(messages), "max_tokens": effective_max_tokens, "stream": False}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if tools:
            payload["tools"] = list(tools)
        headers = {"Content-Type": "application/json", "Accept": "application/json", **(self.default_headers or {})}
        if self.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(f"{self.base_url.rstrip('/')}/chat/completions", data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with self._proxy_scope(), request.urlopen(req, timeout=timeout_seconds or self.timeout_seconds) as response:
                data = json.load(response)
        except error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
            except OSError:
                detail = "response body unavailable"
            category, retryable, cooldown = self._classify_status(exc.code, detail)
            if self.cooldown_overrides and exc.code in self.cooldown_overrides:
                # An override may lengthen a short cooldown, but it must not
                # shorten a computed one (e.g. quota-until-midnight).
                cooldown = max(cooldown, self.cooldown_overrides[exc.code])
            raise ProviderError(f"{self.name} request failed: HTTP {exc.code}: {detail}", category=category, retryable=retryable, cooldown_seconds=cooldown, provider=self.name, model=self.model, http_status=exc.code) from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise ProviderError(f"{self.name} request failed: {exc}", category="network", retryable=True, cooldown_seconds=self.network_cooldown_seconds, provider=self.name, model=self.model) from exc
        return self._parse(data)

    def _parse(self, data: dict[str, Any]) -> ModelTurn:
        try:
            choice = data["choices"][0]
            message = choice["message"]
            calls = [ToolCall(tool_name=item["function"]["name"], arguments=tool_arguments(json.loads(item["function"].get("arguments", "{}"))), call_id=item.get("id", "")) for item in message.get("tool_calls", [])]
            text = message.get("content") or ""
            reasoning_content = message.get("reasoning_content") or ""
            if not isinstance(reasoning_content, str):
                reasoning_content = ""
            if self.reject_reasoning_leakage and not calls and _looks_like_reasoning_leak(text, reasoning_content):
                raise ProviderError(
                    f"{self.name} returned reasoning leakage instead of a final response",
                    category="response_quality",
                    retryable=True,
                    cooldown_seconds=2.0,
                    provider=self.name,
                    model=self.model,
                )
            usage = data.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            total = _as_int(usage.get("total_tokens", 0))
            prompt = _as_int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
            completion = _as_int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
            return ModelTurn(
                text=text, reasoning_content=reasoning_content,
                tool_calls=calls, usage_tokens=total, prompt_tokens=prompt, completion_tokens=completion,
                finish_reason=_as_finish_reason(choice.get("finish_reason")),
            )
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(f"{self.name} returned an unsupported response", category="invalid_response", provider=self.name, model=self.model) from exc

    @staticmethod
    def _classify_status(status: int, detail: str = "") -> tuple[str, bool, float]:
        if status in {402, 429} and _is_quota_exhausted(detail):
            # A spent allowance is not a minute-long rate limit: cool the key
            # (and the provider) until the next UTC day instead of hammering it.
            return "quota", True, _seconds_until_next_utc_day()
        if status in {401, 403}:
            return "auth", False, 0.0
        if status == 429:
            return "rate_limit", True, 60.0
        if status in {408, 500, 502, 503, 504, 529}:
            return "server", True, 10.0
        if status == 451:
            return "unavailable", True, 10.0
        if 500 <= status < 600:
            return "server", True, 10.0
        return "invalid_request" if 400 <= status < 500 else "http", False, 0.0


def _as_finish_reason(value: Any) -> str | None:
    """Normalise a provider stop reason; an absent or malformed one is None."""
    if isinstance(value, str) and value:
        return value
    return None


def _as_int(value: Any) -> int:
    """Parse a token count without ever turning a valid response into an error."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _looks_like_reasoning_leak(text: str, reasoning_content: str = "") -> bool:
    # A populated reasoning field is a definitive leak signal. Bare prose such
    # as "We need to fix the bug" is a legitimate answer and must not be
    # rejected, so only explicit chain-of-thought framing is matched in text.
    if reasoning_content.strip():
        return True
    lowered = text.lstrip().lower()
    return (
        " thinking" in lowered
        or "</think>" in lowered
        or lowered.startswith(("analysis:", "chain of thought", "chain-of-thought", "<thinking>"))
    )


_QUOTA_MARKERS = ("usage limit", "usage_limit", "quota", "upgrade for higher limits", "free_tier_limit", "freeusage_limit_error")


def _has_quota_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _QUOTA_MARKERS)


def _structured_error_text(detail: str) -> str | None:
    """Return ``error.code``/``error.type`` text, or None when absent."""
    try:
        parsed = json.loads(detail)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    error = parsed.get("error")
    if not isinstance(error, dict):
        return None
    fields = [error.get("code"), error.get("type")]
    text = " ".join(str(item) for item in fields if item not in (None, ""))
    return text or None


def _is_quota_exhausted(detail: str) -> bool:
    # Prefer a structured marker; fall back to the raw body only when the
    # response carries no structured code/type at all.
    structured = _structured_error_text(detail)
    if structured is not None:
        return _has_quota_marker(structured)
    return _has_quota_marker(detail)


def _seconds_until_next_utc_day() -> float:
    now = utc_datetime_now()
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1.0, (tomorrow - now).total_seconds())
