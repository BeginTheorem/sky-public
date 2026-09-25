from __future__ import annotations

import json
import select
import time
from collections.abc import Sequence
from typing import Any
from urllib import error, request

from ..models import ModelTurn, ToolCall
from ..provider import Message, ToolSchema
from .errors import ProviderError, tool_arguments
from .openai_compatible import OpenAICompatibleProvider, _as_finish_reason, _as_int, _looks_like_reasoning_leak


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenAI-compatible streaming client for the OpenRouter API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        name: str = "openrouter",
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        retry_delay_seconds: float = 1.0,
        chunk_timeout_seconds: float | None = None,
        stream_deadline_seconds: float | None = None,
        reject_reasoning_leakage: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            reject_reasoning_leakage=reject_reasoning_leakage,
        )
        self.max_attempts = max(1, max_attempts)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)
        # The connect timeout only covers opening the stream. A healthy but slow
        # completion is bounded by *silence* (the idle timeout): as long as
        # chunks keep arriving the stream is allowed to run. ``stream_deadline``
        # is an optional overall cap and is off by default, because a total
        # deadline cuts off long reasoning even when it is actively streaming.
        self.chunk_timeout_seconds = max(0.1, chunk_timeout_seconds or 900.0)
        self.stream_deadline_seconds = stream_deadline_seconds

    def complete(self, messages: Sequence[Message], *, max_tokens: int, tools: Sequence[ToolSchema] = (), timeout_seconds: float | None = None) -> ModelTurn:
        # ``timeout_seconds`` from the fallback ladder is an idle/hang budget, not
        # a total-stream deadline: a slow but active reasoning stream must not be
        # cut off. The overall cap lives in ``stream_deadline_seconds``.
        idle_seconds = max(0.1, timeout_seconds) if timeout_seconds is not None else self.chunk_timeout_seconds
        last_error: ProviderError | None = None
        for attempt in range(self.max_attempts):
            try:
                return self._complete_once(messages, max_tokens=max_tokens, tools=tools, idle_seconds=idle_seconds)
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt + 1 >= self.max_attempts:
                    raise
                if self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds)
        if last_error is None:
            raise AssertionError("retry loop exited without an error to raise")
        raise last_error

    def _complete_once(self, messages: Sequence[Message], *, max_tokens: int, tools: Sequence[ToolSchema] = (), idle_seconds: float | None = None) -> ModelTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = list(tools)
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        idle = max(0.1, idle_seconds) if idle_seconds is not None else self.chunk_timeout_seconds
        deadline = None if self.stream_deadline_seconds is None else time.monotonic() + self.stream_deadline_seconds
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                self._set_chunk_timeout(response, idle_seconds=idle, deadline=deadline)
                return self._parse_sse(response, idle_seconds=idle, deadline=deadline)
        except error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
            except OSError:
                detail = "response body unavailable"
            category, retryable, cooldown = self._classify_status(exc.code, detail)
            raise ProviderError(
                f"openrouter request failed: HTTP {exc.code}: {detail}",
                category=category,
                retryable=retryable,
                cooldown_seconds=cooldown,
                provider=self.name,
                model=self.model,
                http_status=exc.code,
            ) from exc
        except (error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise ProviderError(
                f"openrouter request failed: {exc}",
                category="network",
                retryable=True,
                cooldown_seconds=2.0,
                provider=self.name,
                model=self.model,
            ) from exc

    def _parse_sse(self, response: Any, *, idle_seconds: float, deadline: float | None = None) -> ModelTurn:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        usage_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason: str | None = None
        saw_done = False
        try:
            iterator = iter(response)
            while True:
                self._refresh_chunk_timeout(response, idle_seconds=idle_seconds, deadline=deadline)
                try:
                    raw_line = next(iterator)
                except StopIteration:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("openrouter SSE stream exceeded its deadline")
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ProviderError("openrouter returned malformed SSE JSON", category="invalid_response", provider=self.name, model=self.model) from exc
                if chunk.get("usage"):
                    usage = chunk["usage"]
                    if isinstance(usage, dict):
                        usage_tokens = _as_int(usage.get("total_tokens", 0))
                        prompt_tokens = _as_int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
                        completion_tokens = _as_int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
                choices = chunk.get("choices") or []
                for choice in choices:
                    # The stop reason arrives on the final content chunk, and
                    # the stream can still carry a later usage-only chunk.
                    reason = _as_finish_reason(choice.get("finish_reason"))
                    if reason is not None:
                        finish_reason = reason
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        text_parts.append(content)
                    # OpenRouter names the reasoning channel differently from the
                    # OpenAI-compatible spelling. Measured on the live wire
                    # (175 chunks of one finish request): the delta
                    # vocabulary is ``reasoning`` in 97 chunks plus
                    # ``reasoning_details``, and ``reasoning_content`` never
                    # appears. Reading only ``reasoning_content`` therefore left
                    # ``ModelTurn.reasoning_content`` empty on every openrouter turn,
                    # so a turn whose tokens went to the reasoning channel was
                    # indistinguishable from one that returned nothing. The
                    # aliases are read first-non-empty-per-delta so a provider
                    # that sends both spellings cannot have the same text
                    # counted twice.
                    for key in ("reasoning_content", "reasoning"):
                        reasoning = delta.get(key)
                        if isinstance(reasoning, str) and reasoning:
                            reasoning_parts.append(reasoning)
                            break
                    for call in delta.get("tool_calls") or []:
                        index = int(call.get("index", 0))
                        target = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        call_id = call.get("id")
                        # Some providers repeat id/name in every delta; only the
                        # first non-empty value is meaningful, arguments accumulate.
                        if isinstance(call_id, str) and call_id and not target["id"]:
                            target["id"] = call_id
                        function = call.get("function") or {}
                        name = function.get("name")
                        arguments = function.get("arguments")
                        if isinstance(name, str) and name and not target["name"]:
                            target["name"] = name
                        if isinstance(arguments, str):
                            target["arguments"] += arguments
        except (TypeError, ValueError) as exc:
            raise ProviderError(
                "openrouter returned malformed SSE data",
                category="invalid_response",
                provider=self.name,
                model=self.model,
            ) from exc
        except TimeoutError as exc:
            raise ProviderError(
                "openrouter SSE stream timed out waiting for completion",
                category="network",
                retryable=True,
                cooldown_seconds=2.0,
                provider=self.name,
                model=self.model,
            ) from exc
        if not saw_done and finish_reason is None:
            # A stream is terminated by *either* protocol terminator: the
            # ``[DONE]`` sentinel or a terminal ``finish_reason``. Requiring the
            # sentinel alone discarded a completed response whose tool calls and
            # usage had already arrived in full. Live evidence: three
            # ``fallback_failure`` records "openrouter SSE stream ended before
            # [DONE]"; the last two ended the ReAct episode at
            # step 24 and step 39 with ``needs_recovery``. A stream that closed
            # with neither terminator really was cut off mid-generation, so that
            # case stays a retryable transport failure.
            raise ProviderError(
                "openrouter SSE stream ended before [DONE]",
                category="network",
                retryable=True,
                cooldown_seconds=2.0,
                provider=self.name,
                model=self.model,
            )
        try:
            calls = [ToolCall(tool_name=item["name"], arguments=tool_arguments(json.loads(item["arguments"] or "{}")), call_id=item["id"]) for item in tool_calls.values()]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ProviderError("openrouter returned malformed streamed tool call", category="invalid_response", provider=self.name, model=self.model) from exc
        text = "".join(text_parts)
        reasoning_content = "".join(reasoning_parts)
        if finish_reason == "length" and not text.strip() and not calls:
            # The same guard as the OpenAI-compatible path: a ceiling stop with
            # no visible content is a provider failure, not an answer. Measured
            # on the live run: this provider stopped at completion_tokens 2240
            # with an empty text and the ladder counted it as a success.
            raise ProviderError(
                "openrouter stopped at the output ceiling with no visible content",
                category="response_quality",
                retryable=True,
                cooldown_seconds=2.0,
                provider=self.name,
                model=self.model,
            )
        if self.reject_reasoning_leakage and not calls and _looks_like_reasoning_leak(text, reasoning_content):
            raise ProviderError(
                "openrouter returned reasoning leakage instead of a final response",
                category="response_quality",
                retryable=True,
                cooldown_seconds=2.0,
                provider=self.name,
                model=self.model,
            )
        return ModelTurn(text=text, reasoning_content=reasoning_content, tool_calls=calls, usage_tokens=usage_tokens, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, finish_reason=finish_reason)

    def _read_timeout(self, idle_seconds: float, deadline: float | None) -> float:
        timeout = idle_seconds
        if deadline is not None:
            timeout = min(timeout, max(0.1, deadline - time.monotonic()))
        return timeout

    def _set_chunk_timeout(self, response: Any, *, idle_seconds: float, deadline: float | None = None) -> bool:
        """Make an idle SSE connection fail instead of blocking the ReAct loop.

        Returns True when a socket timeout was applied. The private
        ``fp.raw._sock`` path is preferred, but the response may wrap the
        socket differently, so every plausible target is tried.
        """
        timeout = self._read_timeout(idle_seconds, deadline)
        raw = getattr(getattr(response, "fp", None), "raw", None)
        target = getattr(raw, "_sock", None)
        if target is None or not hasattr(target, "settimeout"):
            target = raw if raw is not None and hasattr(raw, "settimeout") else None
        if target is None:
            fp = getattr(response, "fp", None)
            if fp is not None and hasattr(fp, "settimeout"):
                target = fp
        if target is None and hasattr(response, "settimeout"):
            target = response
        if target is None:
            return False
        target.settimeout(timeout)
        return True

    @staticmethod
    def _fileno(response: Any) -> int | None:
        fileno = getattr(response, "fileno", None)
        if callable(fileno):
            try:
                result = fileno()
            except (OSError, ValueError):
                return None
            return result if isinstance(result, int) else None
        raw = getattr(getattr(response, "fp", None), "raw", None)
        sock = getattr(raw, "_sock", None)
        if sock is not None:
            try:
                return sock.fileno()
            except (OSError, ValueError):
                return None
        return None

    def _wait_readable(self, response: Any, *, idle_seconds: float, deadline: float | None) -> None:
        """Bound a read when no socket timeout could be applied."""
        fd = self._fileno(response)
        if fd is None:
            return
        try:
            readable, _, _ = select.select([fd], [], [], self._read_timeout(idle_seconds, deadline))
        except (OSError, ValueError):
            return
        if not readable:
            raise TimeoutError("openrouter SSE stream idle past its idle timeout")

    def _refresh_chunk_timeout(self, response: Any, *, idle_seconds: float, deadline: float | None) -> None:
        """Refresh the socket timeout before every blocking SSE read."""
        if not self._set_chunk_timeout(response, idle_seconds=idle_seconds, deadline=deadline):
            self._wait_readable(response, idle_seconds=idle_seconds, deadline=deadline)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("openrouter SSE stream exceeded its deadline")
