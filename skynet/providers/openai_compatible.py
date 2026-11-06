from __future__ import annotations

import json
import logging
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

log = logging.getLogger("skynet.providers.openai_compatible")

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
    # Endpoint-specific body fields merged into the payload verbatim (e.g.
    # Z.AI's ``thinking`` switch). Absent means the endpoint's own default.
    extra_body: dict[str, Any] | None = None

    accepts_timeout_override = True
    # Probed against the NVIDIA NIM ring 2026-10-19: both the named form and "required" answer
    # 200 with a tool call, so the closing turn can be forced to be a call rather than prose.
    accepts_tool_choice = True

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
        # No Tor default any more: an explicit proxy URL normally carries its
        # port, and the conventional SOCKS default (1080) is the neutral
        # fallback when it does not.
        proxy_port = parsed.port or 1080
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

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        tools: Sequence[ToolSchema] = (),
        timeout_seconds: float | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> ModelTurn:
        effective_max_tokens = min(max_tokens, self.max_output_tokens) if self.max_output_tokens else max_tokens
        if self.max_output_tokens and max_tokens > self.max_output_tokens:
            # The clamp is invisible in the payload, so a provider ceiling lower
            # than the requested budget used to be undiagnosable from outside.
            # Log the gap here; the transport still records both numbers in
            # ``fallback_selected``.
            log.warning("%s output ceiling clamps %s -> %s tokens", self.name, max_tokens, effective_max_tokens)
        payload: dict[str, Any] = {"model": self.model, "messages": list(messages), "max_tokens": effective_max_tokens, "stream": False}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if tools:
            payload["tools"] = list(tools)
        if tool_choice is not None and tools:
            # Only meaningful with tools on the request; a server that refuses the field fails the
            # call instead of quietly answering in prose, which is the point of asking for it.
            payload["tool_choice"] = tool_choice
        if self.extra_body:
            payload.update(self.extra_body)
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
            # ``cooldown_explicit`` marks a wait the PROVIDER stated, so the
            # patient cycle honours it even when it is shorter than its own
            # 5-minute block. A spent quota body or a ``Retry-After`` header is
            # the provider's word; a category default is only our estimate and an
            # operator override must not be shortened by it.
            explicit = category == "quota"
            retry_after = _retry_after_seconds(exc)
            if retry_after is not None:
                cooldown, explicit = retry_after, True
            elif self.cooldown_overrides and exc.code in self.cooldown_overrides:
                # An override may lengthen a short cooldown, but it must not
                # shorten a computed one (e.g. quota-until-midnight).
                cooldown = max(cooldown, self.cooldown_overrides[exc.code])
            raise ProviderError(f"{self.name} request failed: HTTP {exc.code}: {detail}", category=category, retryable=retryable, cooldown_seconds=cooldown, cooldown_explicit=explicit, provider=self.name, model=self.model, http_status=exc.code) from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise ProviderError(f"{self.name} request failed: {exc}", category="network", retryable=True, cooldown_seconds=self.network_cooldown_seconds, provider=self.name, model=self.model) from exc
        return self._parse(data)

    def _parse(self, data: dict[str, Any]) -> ModelTurn:
        try:
            choice = data["choices"][0]
            message = choice["message"]
            # ``or []``, not a default: Mistral-shaped replies carry an explicit
            # ``"tool_calls": null``, and iterating None turned every Mistral
            # turn into an ``invalid_response`` (measured live 2026-09-27, both
            # rungs, before the key was trusted end-to-end).
            calls = [ToolCall(tool_name=item["function"]["name"], arguments=tool_arguments(json.loads(item["function"].get("arguments", "{}"))), call_id=item.get("id", "")) for item in (message.get("tool_calls") or [])]
            text = _content_text(message.get("content"))
            reasoning_content = message.get("reasoning_content") or ""
            if not isinstance(reasoning_content, str):
                reasoning_content = ""
            finish_reason = choice.get("finish_reason")
            if finish_reason == "error":
                # A 200 whose generation Mistral aborted carries
                # ``finish_reason="error"`` with whatever partial text it had
                # (measured 2026-09-27: a degenerate repetition was cut at ~200
                # tokens). Accepting it hands the loop a truncated reply, so it is
                # a provider-local quality failure like the empty ``length`` stop.
                raise ProviderError(
                    f"{self.name} aborted the generation (finish_reason=error)",
                    category="response_quality",
                    retryable=True,
                    cooldown_seconds=2.0,
                    provider=self.name,
                    model=self.model,
                )
            if finish_reason in {"length", "model_length"} and not text.strip() and not calls:
                # A cut-off reply with nothing visible is a provider failure, not
                # an answer: returning it as a success is what let the
                # output ceiling freeze the planner for seven hours.
                # A ``length`` stop with visible text is a long but real answer
                # and stays valid.
                raise ProviderError(
                    f"{self.name} stopped at the output ceiling with no visible content",
                    category="response_quality",
                    retryable=True,
                    cooldown_seconds=2.0,
                    provider=self.name,
                    model=self.model,
                )
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
            # Cache accounting differs by vendor: OpenAI-shaped responses nest it in
            # `prompt_tokens_details.cached_tokens`, Anthropic-shaped ones report
            # `cache_read_input_tokens` at the top level. Both are copied, never
            # inferred: a provider that says nothing leaves the count at zero.
            details = usage.get("prompt_tokens_details")
            details = details if isinstance(details, dict) else {}
            cached = _as_int(details.get("cached_tokens", usage.get("cache_read_input_tokens", 0)))
            return ModelTurn(
                text=text, reasoning_content=reasoning_content,
                tool_calls=calls, usage_tokens=total, prompt_tokens=prompt, completion_tokens=completion,
                cached_tokens=cached,
                finish_reason=_as_finish_reason(finish_reason),
            )
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(f"{self.name} returned an unsupported response", category="invalid_response", provider=self.name, model=self.model) from exc

    @staticmethod
    def _classify_status(status: int, detail: str = "") -> tuple[str, bool, float]:
        if status in {402, 429} and _is_quota_exhausted(detail):
            # A spent allowance is not a minute-long rate limit: cool the key
            # (and the provider) until the next UTC day instead of hammering it.
            return "quota", True, _seconds_until_next_utc_day()
        if status == 402:
            # A 402 is the key saying it cannot PAY for this request: a spent
            # balance, not a broken payload. Measured live 2026-10-16: openrouter
            # answers 402 with ``{"error":{"message":"Недостаточно средств"}}``,
            # which carries no quota marker, so the rung fell through to
            # ``invalid_request`` -- and ``fallback._is_fatal`` aborts the WHOLE
            # chain on that category. One empty wallet thus killed the free tail
            # standing behind it: ten runs in forty minutes died at zero steps
            # (``needs_recovery``), the paid rung counting 43 strikes. A balance is
            # a property of ONE rung, so billing stays provider-local: skip the
            # rung, cool it long, keep the ladder alive.
            return "billing", False, 0.0
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
        if status == 400 and _is_prompt_too_long(detail):
            # A prompt above THIS rung's context ceiling is provider-local, not a
            # broken request: the same payload may fit a longer-context rung, so
            # the patient cycle skips this provider instead of aborting the whole
            # chain (see ``fallback._is_fatal``). Measured 2026-09-27: Mistral
            # answers 400 with type ``invalid_request_prompt_too_long`` (code
            # 3059, "Prompt 138892 > 32768 maximum context length"), Z.AI with
            # code 1261 and "Prompt exceeds max length".
            return "prompt_too_long", False, 0.0
        if status == 400 and _is_provider_degraded(detail):
            # The status says "bad request", the body says the rung's deployment
            # is DEGRADED. Rung-local and transient, so the ladder skips it after
            # a long cooldown instead of aborting -- see
            # ``DEGRADED_PROVIDER_COOLDOWN_SECONDS``.
            return "server", True, DEGRADED_PROVIDER_COOLDOWN_SECONDS
        return "invalid_request" if 400 <= status < 500 else "http", False, 0.0


def _retry_after_seconds(exc: error.HTTPError) -> float | None:
    """The provider's own wait hint, when the response carries one.

    Only the numeric (seconds) form of ``Retry-After`` is honoured: the HTTP-date
    form would need clock-skew handling for a value the transport already
    expresses in seconds on every endpoint this chain talks to.
    """
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("Retry-After")
    except Exception:
        return None
    if not value:
        return None
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def _content_text(content: Any) -> str:
    """Normalize a chat message ``content`` to plain text.

    A provider may return the content as a string, as ``None``, or as a list of
    content parts (``[{"type": "text", "text": "..."}]``). Falling back on
    ``content or ""`` only covers the falsy cases: a non-empty list is truthy and
    travelled unchanged into ``ModelTurn.text``, where every downstream
    ``text.strip()`` raised ``AttributeError: 'list' object has no attribute
    'strip'``. That is the crash of 2026-10-13 -- the memory loop died on a
    provider reply whose ``content`` was a parts list and the failed cycle began
    the restart storm that ended in systemd's start-limit. Non-text parts
    (images and other payloads) are dropped rather than stringified.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                value = part.get("text")
                if isinstance(value, str):
                    chunks.append(value)
        return "".join(chunks)
    return ""


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

_PROMPT_TOO_LONG_MARKERS = ("prompt_too_long", "exceeds max length", "maximum context length")

#: NVIDIA states a degraded deployment in the BODY of an HTTP 400
#: (``Function id '<uuid>': DEGRADED function cannot be invoked``), which no
#: status-code rule can see. Quoted from the live response, not paraphrased
#: (measured 2026-10-16/17 on the nemotron-3-ultra rung).
_PROVIDER_DEGRADED_MARKERS = ("degraded function cannot be invoked", "function is degraded")

#: How long a rung whose deployment is degraded steps aside. A rollout or a
#: recovery takes minutes, not seconds, and the ladder has other rungs to try --
#: while ``invalid_request`` (the category this body used to land in) is
#: chain-fatal and killed the whole episode on the first rung.
DEGRADED_PROVIDER_COOLDOWN_SECONDS = 300.0


def _is_provider_degraded(detail: str) -> bool:
    """Whether a 400 body says the PROVIDER's deployment is unhealthy.

    ``invalid_request`` is the right verdict for a malformed payload; it is the
    wrong one for a rung whose model deployment is being rolled out, because
    ``fallback._is_fatal`` aborts the whole ladder on that category. Measured
    live: the rung answered this body for hours while the three rungs behind it
    were answering 200, and every run died at zero steps.
    """
    return any(marker in detail.lower() for marker in _PROVIDER_DEGRADED_MARKERS)


def _is_prompt_too_long(detail: str) -> bool:
    """Whether a 400 reports a prompt above the model's context ceiling.

    Both the structured ``type``/``code`` (Mistral 3059) and the raw message
    ("Prompt 138892 > 32768 maximum context length") are checked, because a rung
    may state the limit only in the message text (Z.AI code 1261).
    """
    structured = _structured_error_text(detail)
    return any(marker in f"{structured or ''} {detail}".lower() for marker in _PROMPT_TOO_LONG_MARKERS)


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
