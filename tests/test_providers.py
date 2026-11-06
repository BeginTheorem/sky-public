from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import patch

from skynet.models import ModelTurn
from skynet.providers import build_provider
from skynet.providers.errors import ProviderError
from skynet.providers.fallback import FallbackProvider, is_chain_dead_terminal
from skynet.providers.key_rotator import KeyRotator, KeySlot
from skynet.providers.ollama import OllamaCloudProvider
from skynet.providers.openai_compatible import OpenAICompatibleProvider
from skynet.providers.openrouter import OpenRouterProvider

#: Both provider-switch paths are the OWNER's phone controls. When
#: ``state/provider-prefs.json`` exists, ``active_chain_names()`` filters the chain
#: by what the last tick applied -- so one real tap silently changed what the
#: assertions below saw (measured 2026-10-17: a live prefs file turned seven
#: assertions here into ``[] != [the environment's chain]``, while the module was
#: only ever meant to test the environment). Pin both paths to a file that never
#: exists for the whole module; the switch mechanism has its own suite.
_OWNER_SWITCH_ABSENT = Path(tempfile.gettempdir()) / "skynet-tests-no-owner-switches.json"
_SWITCH_PATH_PATCHERS: list[Any] = []


def setUpModule() -> None:
    for target in ("skynet.providers.provider_prefs_path", "skynet.providers.provider_applied_path"):
        patcher = patch(target, return_value=_OWNER_SWITCH_ABSENT)
        patcher.start()
        _SWITCH_PATH_PATCHERS.append(patcher)


def tearDownModule() -> None:
    for patcher in _SWITCH_PATH_PATCHERS:
        patcher.stop()
    _SWITCH_PATH_PATCHERS.clear()


class _Handler(BaseHTTPRequestHandler):
    statuses: ClassVar[list[int]] = []
    bodies: ClassVar[list[str]] = []
    auths: ClassVar[list[str]] = []
    payloads: ClassVar[list[dict[str, Any]]] = []
    reasoning_response = False

    def do_POST(self) -> None:
        _Handler.auths.append(self.headers.get("Authorization", ""))
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            _Handler.payloads.append(cast(dict[str, Any], json.loads(raw.decode())) if raw else {})
        except ValueError:
            _Handler.payloads.append({})
        status = _Handler.statuses.pop(0) if _Handler.statuses else 200
        override = _Handler.bodies.pop(0) if _Handler.bodies else None
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        message = {"content": "ok", "tool_calls": []}
        if _Handler.reasoning_response:
            message = {"content": "We need to analyze this first.", "reasoning_content": "hidden reasoning"}
        body = {"choices": [{"message": message}], "usage": {"total_tokens": 3}}
        payload = override if override is not None else json.dumps(body if status == 200 else {"error": "failed"})
        self.wfile.write(payload.encode())

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"data": []}')

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class ProviderTests(unittest.TestCase):
    def test_http_529_is_retryable_overload(self) -> None:
        self.assertEqual(OpenAICompatibleProvider._classify_status(529), ("server", True, 10.0))
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        _Handler.statuses = []
        _Handler.bodies = []
        _Handler.auths = []
        _Handler.payloads = []
        _Handler.reasoning_response = False

    def test_rotator_round_robin_and_masked_status(self) -> None:
        rotator = KeyRotator(["first-secret", "second-secret"], cooldown_seconds=60)
        first = cast(KeySlot, rotator.next_available())
        second = cast(KeySlot, rotator.next_available())
        self.assertEqual(first.key, "first-secret")
        self.assertEqual(second.key, "second-secret")
        rotator.mark_failed(0, permanent=True)
        self.assertFalse(rotator.status()[0]["available"])
        self.assertNotIn("first-secret", str(rotator.status()))

    def test_ollama_rotates_after_rate_limit(self) -> None:
        _Handler.statuses = [429, 200]
        provider = OllamaCloudProvider(self.base_url, ["key-one", "key-two"], "test-model", cooldown_seconds=60, max_attempts=2)
        result = provider.complete([{"role": "user", "content": "hello"}], max_tokens=10)
        self.assertEqual(result.text, "ok")
        self.assertEqual(_Handler.auths, ["Bearer key-one", "Bearer key-two"])

    def test_fallback_moves_to_next_provider(self) -> None:
        class Failed:
            name = "failed"
            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                from skynet.providers.errors import ProviderError
                raise ProviderError("down", category="network", retryable=True)
        class Good:
            name = "good"
            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="fallback")
        self.assertEqual(FallbackProvider([Failed(), Good()]).complete([], max_tokens=1).text, "fallback")

    def test_fallback_logs_failure_and_selected_provider(self) -> None:
        from skynet.providers.errors import ProviderError

        class Failed:
            name = "nvidia_deepseek"
            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("busy", category="server", retryable=True, cooldown_seconds=30)

        class Good:
            name = "nvidia"
            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="ok")

        with self.assertLogs("skynet.providers.fallback", level="INFO") as captured:
            result = FallbackProvider([Failed(), Good()]).complete([], max_tokens=1)
        self.assertEqual(result.text, "ok")
        self.assertTrue(any("fallback_failure" in line and "nvidia_deepseek" in line for line in captured.output))
        self.assertTrue(any("fallback_selected" in line and "nvidia" in line for line in captured.output))

    def test_provider_failure_reaches_the_durable_event_log(self) -> None:
        # A provider strike and the whole chain going into cooldown used to be
        # written only to state/runtime.jsonl, so a failed run left no durable
        # post-mortem. The reactor wires the store's record_provider_event as
        # the provider hook; prove the promoted kinds land in event_log.
        from skynet.store import StateStore

        class Failed:
            name = "openrouter"
            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("down", category="network", retryable=True, cooldown_seconds=30)

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            try:
                fallback = FallbackProvider([Failed()])
                fallback.set_event_logger(store.record_provider_event)
                with self.assertRaisesRegex(Exception, "all providers failed"):
                    fallback.complete([], max_tokens=1)
                with self.assertRaisesRegex(Exception, "all providers failed"):
                    fallback.complete([], max_tokens=1)
                kinds = {
                    str(row[0])
                    for row in store.connection.execute(
                        "SELECT kind FROM event_log WHERE kind IN ('fallback_failure', 'fallback_all_cooling')"
                    )
                }
            finally:
                store.close()
        self.assertIn("fallback_failure", kinds)
        self.assertIn("fallback_all_cooling", kinds)

    def test_fallback_attempt_is_not_reported_as_a_strike(self) -> None:
        # The pre-attempt signal says an attempt began, not that one failed:
        # ~91% of successful first tries were logged as a strike, so operator
        # reports read a success as a failure.
        class Good:
            name = "openrouter"
            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="ok")

        with self.assertLogs("skynet.providers.fallback", level="INFO") as captured:
            result = FallbackProvider([Good()]).complete([], max_tokens=1)
        self.assertEqual(result.text, "ok")
        joined = "\n".join(captured.output)
        self.assertIn("fallback_attempt", joined)
        self.assertNotIn("fallback_failure", joined)
        self.assertNotIn("fallback_strike", joined)

    def test_fallback_skips_provider_during_cooldown(self) -> None:
        from skynet.providers.errors import ProviderError

        class DeepSeek:
            name = "nvidia_deepseek"
            calls = 0
            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                self.calls += 1
                raise ProviderError("rate limited", category="rate_limit", retryable=True, cooldown_seconds=60)

        provider = DeepSeek()
        fallback = FallbackProvider([provider])
        with self.assertRaisesRegex(Exception, "all providers failed"):
            fallback.complete([], max_tokens=1)
        with self.assertRaisesRegex(Exception, "all providers failed"):
            fallback.complete([], max_tokens=1)
        self.assertEqual(provider.calls, 1)

    def test_chain_wide_cooldown_is_waited_out_when_the_horizon_is_short(self) -> None:
        """The live shape: every provider cooling, earliest horizon ~0.5s.

        Measured on state/skynet.sqlite3: 5 of the 21 needs_recovery runs died
        with ZERO strikes purely because the whole chain was inside a cooldown
        whose earliest recorded recovery was 12.2-30.0s away -- two of them
        after 22 and 41 verified steps. A horizon the chain itself declared is
        not a reason to discard an episode that has not tried anything yet.
        """
        class Healthy:
            name = "healthy"
            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="ok")

        fallback = FallbackProvider([Healthy()], max_wait_for_cooldown_seconds=5.0)
        fallback._blocked_until = {"healthy": time.time() + 0.5}
        started = time.monotonic()
        with self.assertLogs("skynet.providers.fallback", level="INFO") as captured:
            result = fallback.complete([], max_tokens=1)
        self.assertEqual(result.text, "ok")
        self.assertGreaterEqual(time.monotonic() - started, 0.4)
        self.assertTrue(any("fallback_cooldown_wait" in line for line in captured.output))

    def test_a_long_chain_wide_cooldown_is_not_waited_out(self) -> None:
        # The bound exists so a chain that is down for minutes is not waited on
        # inside one call: that decision belongs to the reactor's cross-run
        # backoff, which has its own escalation and jitter.
        class Down:
            name = "down"
            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("down", category="network", retryable=True, cooldown_seconds=300)

        fallback = FallbackProvider([Down()], max_wait_for_cooldown_seconds=5.0)
        with self.assertRaisesRegex(Exception, "all providers failed"):
            fallback.complete([], max_tokens=1)
        started = time.monotonic()
        with self.assertRaisesRegex(Exception, "all providers failed"):
            fallback.complete([], max_tokens=1)
        self.assertLess(time.monotonic() - started, 1.0)

    def test_fallback_health_excludes_provider_in_cooldown(self) -> None:
        class Healthy:
            name = "healthy"
            def complete(self, messages, *, max_tokens, tools=()):
                return ModelTurn(text="healthy")
            def health_probe(self) -> dict[str, object]:
                return {"ok": True}

        fallback = FallbackProvider([Healthy()])
        assert fallback._blocked_until is not None
        fallback._blocked_until["healthy"] = time.time() + 60
        result = fallback.health_probe()
        self.assertFalse(result["ok"])
        self.assertIsNone(result["active_provider"])

    def test_fallback_health_continues_after_probe_exception(self) -> None:
        class Broken:
            name = "broken"
            def complete(self, messages, *, max_tokens, tools=()):
                return ModelTurn(text="broken")
            def health_probe(self) -> dict[str, object]:
                raise RuntimeError("probe exploded")

        class Healthy:
            name = "healthy"
            def complete(self, messages, *, max_tokens, tools=()):
                return ModelTurn(text="healthy")
            def health_probe(self) -> dict[str, object]:
                return {"ok": True}

        result = FallbackProvider([Broken(), Healthy()]).health_probe()
        self.assertTrue(result["ok"])
        self.assertEqual(result["active_provider"], "healthy")
        providers = cast(dict[str, dict[str, object]], result["providers"])
        broken = providers["broken"]
        self.assertEqual(broken["ok"], False)
        self.assertEqual(broken["error"], "probe exploded")
        self.assertEqual(broken["blocked"], False)

    def test_reasoning_leakage_is_retryable(self) -> None:
        _Handler.reasoning_response = True
        provider = OpenAICompatibleProvider(
            name="nvidia",
            base_url=self.base_url,
            api_key="key",
            model="test-model",
            reject_reasoning_leakage=True,
        )
        with self.assertRaisesRegex(ProviderError, "reasoning leakage") as raised:
            provider.complete([{"role": "user", "content": "hard task"}], max_tokens=100)
        self.assertEqual(raised.exception.category, "response_quality")
        self.assertTrue(raised.exception.retryable)




    def test_quota_429_gets_next_day_cooldown(self) -> None:
        noon = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        with patch("skynet.providers.openai_compatible.utc_datetime_now", return_value=noon):
            category, retryable, cooldown = OpenAICompatibleProvider._classify_status(
                429, '{"error":{"message":"you have reached your monthly usage limit, upgrade for higher limits"}}'
            )
        self.assertEqual(category, "quota")
        self.assertTrue(retryable)
        # The cooldown ends at the next UTC midnight.
        self.assertEqual(cooldown, 12 * 60 * 60)

    def test_plain_429_keeps_the_minute_cooldown(self) -> None:
        category, retryable, cooldown = OpenAICompatibleProvider._classify_status(429, '{"error":{"message":"rate limit exceeded"}}')
        self.assertEqual(category, "rate_limit")
        self.assertTrue(retryable)
        self.assertEqual(cooldown, 60.0)

    def test_ollama_quota_exhaustion_cools_every_key_and_reports_it(self) -> None:
        from skynet.providers.errors import ProviderError

        class QuotaClient:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("monthly usage limit", category="quota", retryable=True, cooldown_seconds=7200.0, provider="ollama")

        provider = OllamaCloudProvider(self.base_url, ["key-one", "key-two"], "test-model", max_attempts=2)
        with patch("skynet.providers.ollama.OpenAICompatibleProvider", QuotaClient), self.assertRaises(ProviderError) as raised:
            provider.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        self.assertEqual(raised.exception.cooldown_seconds, 7200.0)
        probe = provider.health_probe()
        self.assertFalse(probe["ok"])
        self.assertEqual(probe["available_keys"], 0)

    def test_fallback_moves_on_reasoning_leakage(self) -> None:
        from skynet.providers.errors import ProviderError

        class LeakyNvidia:
            name = "nvidia"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("reasoning leakage", category="response_quality", retryable=True)

        ollama = OpenAICompatibleProvider(name="ollama", base_url=self.base_url, api_key="key", model="test-model")
        self.assertEqual(FallbackProvider([LeakyNvidia(), ollama]).complete([], max_tokens=1).text, "ok")

    def test_openrouter_parses_streaming_text_and_tool_calls(self) -> None:
        class Response:
            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n',
                    b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n',
                    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"bash","arguments":"{\\"command\\":\\"pwd\\"}"}}]}}]}\n\n',
                    b'data: {"usage":{"total_tokens":7}}\n\n',
                    b'data: [DONE]\n\n',
                ])
        provider = OpenRouterProvider(base_url=self.base_url, api_key="openrouter-secret", model="openai/gpt-5.6-luna")
        result = provider._parse_sse(Response(), idle_seconds=60.0)
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.usage_tokens, 7)
        self.assertEqual(result.tool_calls[0].tool_name, "bash")
        self.assertEqual(result.tool_calls[0].arguments, {"command": "pwd"})

    def test_openrouter_stream_deadline_is_retryable(self) -> None:
        class Response:
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                time.sleep(0.02)
                yield b'data: [DONE]\n\n'

        provider = OpenRouterProvider(
            base_url=self.base_url,
            api_key="openrouter-secret",
            model="openai/gpt-5.6-luna",
            timeout_seconds=0.01,
        )
        with self.assertRaisesRegex(ProviderError, "timed out") as raised:
            provider._parse_sse(Response(), idle_seconds=60.0, deadline=time.monotonic() + 0.01)
        self.assertEqual(raised.exception.category, "network")
        self.assertTrue(raised.exception.retryable)

    def test_openrouter_accepts_null_stream_fields(self) -> None:
        class Response:
            def __iter__(self):
                return iter([
                    b'data: {"choices":null}\n\n',
                    b'data: {"choices":[{"delta":{"content":null,"tool_calls":[{"index":0,"id":null,"function":{"name":null,"arguments":null}}]}}]}\n\n',
                    b'data: {"usage":{"total_tokens":3},"choices":null}\n\n',
                    b'data: [DONE]\n\n',
                ])

        provider = OpenRouterProvider(base_url=self.base_url, api_key="openrouter-secret", model="openai/gpt-5.6-luna")
        result = provider._parse_sse(Response(), idle_seconds=60.0)
        self.assertEqual(result.text, "")
        self.assertEqual(result.usage_tokens, 3)
        self.assertEqual(result.tool_calls[0].arguments, {})

    def test_openrouter_terminal_finish_reason_ends_the_stream_without_done(self) -> None:
        # A stream is terminated by either the [DONE] sentinel or a terminal
        # finish_reason. Requiring the sentinel alone discarded a complete answer
        # whose tool calls and usage had already arrived.
        class Response:
            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"content":"answered"}}]}\n\n',
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":5}}\n\n',
                ])

        provider = OpenRouterProvider(base_url=self.base_url, api_key="openrouter-secret", model="openai/gpt-5.6-luna")
        result = provider._parse_sse(Response(), idle_seconds=60.0)
        self.assertEqual(result.text, "answered")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage_tokens, 5)

    def test_openrouter_stream_with_neither_terminator_stays_retryable(self) -> None:
        class Response:
            def __iter__(self):
                return iter([b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'])

        provider = OpenRouterProvider(base_url=self.base_url, api_key="openrouter-secret", model="openai/gpt-5.6-luna")
        with self.assertRaisesRegex(ProviderError, "before \\[DONE\\]") as raised:
            provider._parse_sse(Response(), idle_seconds=60.0)
        self.assertEqual(raised.exception.category, "network")
        self.assertTrue(raised.exception.retryable)

    def test_openrouter_length_stop_with_empty_text_is_retryable(self) -> None:
        """A ceiling stop with no visible content must strike, not "succeed".

        Measured: the planner asked for 16384 tokens, openrouter stopped at
        2240 with an empty text and ``finish_reason='length'``, and the ladder
        returned it as a success -- so no retry or fallback ever ran.
        """
        class Response:
            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"reasoning":"thinking hard"}}]}\n\n',
                    b'data: {"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"total_tokens":2240}}\n\n',
                ])

        provider = OpenRouterProvider(base_url=self.base_url, api_key="openrouter-secret", model="openai/gpt-5.6-luna")
        with self.assertRaises(ProviderError) as raised:
            provider._parse_sse(Response(), idle_seconds=60.0)
        self.assertEqual(raised.exception.category, "response_quality")
        self.assertTrue(raised.exception.retryable)

    def test_openrouter_reads_the_reasoning_alias_and_does_not_double_count(self) -> None:
        # OpenRouter spells the reasoning channel `reasoning`, not `reasoning_content`;
        # a delta that carries both spellings must still count the text once.
        class Response:
            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"reasoning":"thinking"}}]}\n\n',
                    b'data: {"choices":[{"delta":{"reasoning_content":"both","reasoning":"both-again"}}]}\n\n',
                    b'data: [DONE]\n\n',
                ])

        provider = OpenRouterProvider(base_url=self.base_url, api_key="openrouter-secret", model="openai/gpt-5.6-luna")
        result = provider._parse_sse(Response(), idle_seconds=60.0)
        self.assertEqual(result.reasoning_content, "thinkingboth")

    def test_fallback_selected_logs_finish_reason_and_completion_tokens(self) -> None:
        class Good:
            name = "openrouter"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="ok", completion_tokens=7, finish_reason="length")

        captured: list[tuple[str, dict[str, object]]] = []
        provider = FallbackProvider([Good()])
        provider.set_event_logger(lambda event, payload: captured.append((event, payload)))
        self.assertEqual(provider.complete([], max_tokens=1).text, "ok")
        selected = next(payload for event, payload in captured if event == "fallback_selected")
        self.assertEqual(selected["finish_reason"], "length")
        self.assertEqual(selected["completion_tokens"], 7)

    def test_fallback_selected_records_requested_ceiling_beside_stop_reason(self) -> None:
        """A cut-off reply must be diagnosable from the log line alone.

        Measured: three consecutive planner generations stopped at
        completion_tokens 2240 with finish_reason='length' while the planner
        requested 16384, and the recorded line could not say how far apart the
        two numbers were -- so the operator instruction to raise the knob could
        neither be confirmed nor refuted. The requested ceiling, the reasoning
        volume and a dedicated truncation event close that gap.
        """
        class CutOff:
            name = "openrouter"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="", reasoning_content="thinking hard", completion_tokens=2240, finish_reason="length")

        captured: list[tuple[str, dict[str, object]]] = []
        provider = FallbackProvider([CutOff()])
        provider.set_event_logger(lambda event, payload: captured.append((event, payload)))
        provider.complete([], max_tokens=16384)
        selected = next(payload for event, payload in captured if event == "fallback_selected")
        self.assertEqual(selected["requested_max_tokens"], 16384)
        self.assertEqual(selected["completion_tokens"], 2240)
        self.assertEqual(selected["reasoning_chars"], len("thinking hard"))
        truncated = next(payload for event, payload in captured if event == "provider_output_truncated")
        self.assertEqual(truncated["requested_max_tokens"], 16384)
        self.assertEqual(truncated["completion_tokens"], 2240)
        self.assertEqual(truncated["text_chars"], 0)

    def test_no_truncation_event_when_a_turn_stops_normally(self) -> None:
        """``stop`` and ``tool_calls`` are normal ends; only ``length`` is a cut-off."""
        class Normal:
            name = "openrouter"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="done", completion_tokens=12, finish_reason="stop")

        captured: list[tuple[str, dict[str, object]]] = []
        provider = FallbackProvider([Normal()])
        provider.set_event_logger(lambda event, payload: captured.append((event, payload)))
        provider.complete([], max_tokens=16384)
        self.assertFalse(any(event == "provider_output_truncated" for event, _ in captured))

    def test_openai_parse_propagates_finish_reason(self) -> None:
        from skynet.providers.openai_compatible import _as_finish_reason

        self.assertIsNone(_as_finish_reason(None))
        self.assertIsNone(_as_finish_reason(""))
        self.assertIsNone(_as_finish_reason(3))
        self.assertEqual(_as_finish_reason("tool_calls"), "tool_calls")
        provider = OpenAICompatibleProvider(name="nvidia", base_url=self.base_url, api_key="key", model="m")
        turn = provider._parse({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})
        self.assertEqual(turn.finish_reason, "stop")
        absent = provider._parse({"choices": [{"message": {"content": "hi"}}]})
        self.assertIsNone(absent.finish_reason)

    def test_openai_parse_joins_list_content_parts(self) -> None:
        """A parts-list ``content`` must become text, never reach ``ModelTurn.text``.

        The memory loop died on ``turn.text.strip()`` with ``AttributeError:
        'list' object has no attribute 'strip'`` when a provider returned
        ``content`` as ``[{"type": "text", "text": ...}]``; the failed cycle
        started the 2026-10-13 restart storm. ``content or ""`` never covered it
        because a non-empty list is truthy.
        """
        provider = OpenAICompatibleProvider(name="mistral", base_url=self.base_url, api_key="key", model="m")
        parts = provider._parse(
            {"choices": [{"message": {"content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]}}]}
        )
        self.assertEqual(parts.text, "hello world")
        # Bare string parts count; non-text parts (images) are dropped, not stringified.
        mixed = provider._parse(
            {"choices": [{"message": {"content": ["plain", {"type": "image_url", "image_url": {"url": "x"}}]}}]}
        )
        self.assertEqual(mixed.text, "plain")
        # ``None`` and a bare string keep their old behaviour.
        self.assertEqual(provider._parse({"choices": [{"message": {"content": None}}]}).text, "")
        self.assertEqual(provider._parse({"choices": [{"message": {"content": "hi"}}]}).text, "hi")

    def test_length_stop_with_empty_text_is_a_retryable_provider_error(self) -> None:
        """The output-ceiling guard: no visible content means no usable answer.

        Returning an empty ``length`` turn as success is what let the
        ceiling freeze the planner for seven hours.
        """
        from skynet.providers.errors import ProviderError

        provider = OpenAICompatibleProvider(name="nemotron", base_url=self.base_url, api_key="key", model="m")
        with self.assertRaises(ProviderError) as raised:
            provider._parse(
                {"choices": [{"message": {"content": "", "reasoning_content": "thinking"}, "finish_reason": "length"}]}
            )
        self.assertEqual(raised.exception.category, "response_quality")
        self.assertTrue(raised.exception.retryable)

    def test_length_stop_with_visible_text_stays_a_valid_answer(self) -> None:
        provider = OpenAICompatibleProvider(name="nemotron", base_url=self.base_url, api_key="key", model="m")
        turn = provider._parse({"choices": [{"message": {"content": "a long but real answer"}, "finish_reason": "length"}]})
        self.assertEqual(turn.text, "a long but real answer")
        self.assertEqual(turn.finish_reason, "length")

    def test_proxy_403_is_a_fatal_auth_error(self) -> None:
        """With no Tor circuit to rotate, a proxy 403 is a fatal auth error."""
        provider = OpenAICompatibleProvider(
            name="nemotron", base_url=self.base_url, api_key="key", model="m", proxy_url="socks5://127.0.0.1:10808"
        )
        _Handler.statuses = [403]
        with (
            patch.object(OpenAICompatibleProvider, "_proxy_scope", return_value=nullcontext()),
            self.assertRaises(ProviderError) as raised,
        ):
            provider.complete([{"role": "user", "content": "hello"}], max_tokens=10)
        self.assertEqual(raised.exception.category, "auth")
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(_Handler.statuses, [], "a 403 must not spend a second request")

    def test_a_successful_strike_clears_the_provider_cooldown(self) -> None:
        """A provider that just answered must not stay cooling down.

        Measured on the live switch: a 500 on one step put nemotron in
        a 30s cooldown, the same step's retry then succeeded, and the following
        step was skipped as still cooling down -- aborting the run and, after
        three such aborts, escalating to provider_lockout while the provider was
        actively answering.
        """
        class Flaky:
            name = "flaky"

            def __init__(self) -> None:
                self.calls = 0

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                self.calls += 1
                if self.calls == 1:
                    raise ProviderError("boom", category="server", retryable=True, cooldown_seconds=30.0)
                return ModelTurn(text="ok", finish_reason="stop")

        captured: list[tuple[str, dict[str, object]]] = []
        provider = FallbackProvider([Flaky()], attempts_per_provider=2, attempt_waits=(0.0,))
        provider.set_event_logger(lambda event, payload: captured.append((event, payload)))
        self.assertEqual(provider.complete([], max_tokens=1, patient=True).text, "ok")
        self.assertEqual(provider._blocked_remaining(), {}, "a success must clear the cooldown it disproved")
        # The next step of the same run must reach the provider, not skip it.
        self.assertEqual(provider.complete([], max_tokens=1, patient=True).text, "ok")
        self.assertFalse(any(event == "fallback_skipped" for event, _ in captured))

    def test_patient_cycle_escalates_waits_and_attempt_timeouts(self) -> None:
        from skynet.providers.errors import ProviderError

        class Flaky:
            name = "flaky"
            accepts_timeout_override = True

            def __init__(self) -> None:
                self.timeouts: list[float | None] = []

            def complete(self, *_args: object, timeout_seconds: float | None = None, **_kwargs: object) -> ModelTurn:
                self.timeouts.append(timeout_seconds)
                raise ProviderError("busy", category="server", retryable=True)

        provider = Flaky()
        fallback = FallbackProvider(
            [provider],
            attempts_per_provider=4,
            attempt_waits=(0.01, 0.02, 0.03),
            provider_block_seconds=300.0,
            attempt_timeout_seconds=900.0,
        )
        started = time.monotonic()
        with self.assertRaisesRegex(Exception, "all providers failed"):
            fallback.complete([], max_tokens=1, patient=True)
        elapsed = time.monotonic() - started
        # Four grouped strikes, each bounded by the non-streaming whole-wait
        # budget, separated by the configured escalating waits. The ladder is
        # normalised to exactly ``attempts - 1`` waits (see build_provider() for
        # the production 5/10/15 values).
        self.assertEqual(fallback.attempt_waits, (0.01, 0.02, 0.03))
        self.assertEqual(provider.timeouts, [900.0, 900.0, 900.0, 900.0])
        self.assertGreaterEqual(elapsed, 0.05)

    def test_patient_reports_model_time_without_the_waits(self) -> None:
        """Backoff between strikes is infrastructure, not model thinking.

        Charging it to the run's model budget made slow-but-healthy chains
        exhaust a run at a few dozen steps.
        """
        from skynet.providers.errors import ProviderError

        class Flaky:
            name = "flaky"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("busy", category="server", retryable=True)

        class Good:
            name = "good"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="ok")

        fallback = FallbackProvider([Flaky(), Good()], attempts_per_provider=2, attempt_waits=(0.2,))
        started = time.monotonic()
        turn = fallback.complete([], max_tokens=1, patient=True)
        wall = time.monotonic() - started
        self.assertEqual(turn.text, "ok")
        self.assertGreater(wall, 0.15)
        self.assertLess(turn.model_seconds, wall - 0.1)

    def test_patient_groups_attempts_per_provider_in_chain_order(self) -> None:
        from skynet.providers.errors import ProviderError

        seen: list[str] = []

        class Flaky:
            def __init__(self, name: str) -> None:
                self.name = name

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                seen.append(self.name)
                raise ProviderError("busy", category="server", retryable=True)

        fallback = FallbackProvider([Flaky("a"), Flaky("b")], attempts_per_provider=2, attempt_waits=(0.0,))
        with self.assertRaisesRegex(Exception, "all providers failed"):
            fallback.complete([], max_tokens=1, patient=True)
        # The owner's scheme is grouped, not round-robin: both strikes on "a"
        # happen before "b" is touched.
        self.assertEqual(seen, ["a", "a", "b", "b"])

    def test_default_call_strikes_each_provider_once(self) -> None:
        from skynet.providers.errors import ProviderError

        seen: list[str] = []

        class Flaky:
            def __init__(self, name: str) -> None:
                self.name = name

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                seen.append(self.name)
                raise ProviderError("busy", category="server", retryable=True)

        fallback = FallbackProvider([Flaky("a"), Flaky("b")])
        started = time.monotonic()
        with self.assertRaisesRegex(Exception, "all providers failed"):
            fallback.complete([], max_tokens=1)
        self.assertEqual(seen, ["a", "b"])
        self.assertLess(time.monotonic() - started, 1.0)

    def test_patient_locks_a_provider_after_its_last_strike(self) -> None:
        from skynet.providers.errors import ProviderError

        class Down:
            name = "down"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("down", category="server", retryable=True, cooldown_seconds=10.0)

        events: list[tuple[str, dict[str, object]]] = []
        fallback = FallbackProvider([Down()], attempts_per_provider=2, attempt_waits=(0.0,), provider_block_seconds=300.0)
        fallback.set_event_logger(lambda event, payload: events.append((event, payload)))
        with self.assertRaisesRegex(Exception, "dead passes"):
            fallback.complete([], max_tokens=1, patient=True)
        remaining = fallback._blocked_remaining()["down"]
        # The provider's own 10s cooldown does not shorten the owner's 5-minute
        # block: a category/countdown number is not the provider's word.
        self.assertGreaterEqual(remaining, 299.0)
        self.assertTrue(any(event == "fallback_provider_blocked" for event, _ in events))

    def test_patient_honours_a_cooldown_the_provider_itself_stated(self) -> None:
        from skynet.providers.errors import ProviderError

        class Down:
            name = "down"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("busy", category="rate_limit", retryable=True, cooldown_seconds=10.0, cooldown_explicit=True)

        fallback = FallbackProvider([Down()], attempts_per_provider=1, provider_block_seconds=300.0)
        with self.assertRaisesRegex(Exception, "dead passes"):
            fallback.complete([], max_tokens=1, patient=True)
        remaining = fallback._blocked_remaining()["down"]
        # "His 10 seconds": an explicit provider hint wins even when shorter.
        self.assertLessEqual(remaining, 10.5)
        self.assertGreater(remaining, 8.0)

    def test_patient_ends_after_the_configured_dead_passes(self) -> None:
        from skynet.providers.errors import ProviderError

        seen: list[str] = []

        class Down:
            name = "a"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                seen.append(self.name)
                raise ProviderError("down", category="network", retryable=True)

        fallback = FallbackProvider(
            [Down()],
            attempts_per_provider=1,
            provider_block_seconds=0.05,
            dead_passes=3,
            call_deadline_seconds=0.0,
        )
        events: list[str] = []
        fallback.set_event_logger(lambda event, _payload: events.append(event))
        with self.assertRaises(ProviderError) as raised:
            fallback.complete([], max_tokens=1, patient=True)
        self.assertTrue(is_chain_dead_terminal(str(raised.exception)))
        # Three whole-chain passes, each provider struck once per pass.
        self.assertEqual(seen, ["a", "a", "a"])
        self.assertEqual(events.count("fallback_dead_pass"), 2)

    def test_patient_repasses_a_fully_locked_chain_instead_of_ending(self) -> None:
        from skynet.providers.errors import ProviderError

        class Down:
            name = "down"
            calls = 0

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                self.calls += 1
                raise ProviderError("down", category="network", retryable=True)

        class Good:
            name = "good"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="ok")

        fallback = FallbackProvider([Down(), Good()], attempts_per_provider=1, provider_block_seconds=0.05, dead_passes=3)
        events: list[str] = []
        fallback.set_event_logger(lambda event, _payload: events.append(event))
        # Both rungs start locked: the pass has nothing to strike, so the call
        # waits the declared horizon out and re-passes instead of ending the
        # episode -- the run is not interrupted, the network is just waited on.
        fallback._blocked_until = {"down": time.time() + 0.05, "good": time.time() + 0.05}
        result = fallback.complete([], max_tokens=1, patient=True)
        self.assertEqual(result.text, "ok")
        self.assertIn("fallback_cooldown_wait", events)

    def test_patient_pulses_the_wait_callback(self) -> None:
        from skynet.providers.errors import ProviderError

        class Down:
            name = "down"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("down", category="network", retryable=True)

        pulses: list[float] = []
        fallback = FallbackProvider(
            [Down()],
            attempts_per_provider=2,
            attempt_waits=(0.05,),
            provider_block_seconds=0.05,
            dead_passes=2,
            wait_slice_seconds=0.01,
        )
        fallback.set_wait_callback(pulses.append)
        with self.assertRaisesRegex(Exception, "dead passes"):
            fallback.complete([], max_tokens=1, patient=True)
        # The heartbeat must beat while the chain sleeps, or the stale-run
        # watchdog kills a deliberately waiting run.
        self.assertGreaterEqual(len(pulses), 2)

    def test_patient_aborts_at_once_on_a_broken_request(self) -> None:
        from skynet.providers.errors import ProviderError

        class Broken:
            name = "broken"
            calls = 0

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                self.calls += 1
                raise ProviderError("bad request", category="invalid_request", retryable=False)

        broken = Broken()
        fallback = FallbackProvider([broken], attempts_per_provider=4, attempt_waits=(0.0, 0.0, 0.0))
        with self.assertRaises(ProviderError):
            fallback.complete([], max_tokens=1, patient=True)
        self.assertEqual(broken.calls, 1, "a broken request must not be retried across the ladder")

    def test_patient_uses_idle_budget_for_streamers_and_whole_wait_otherwise(self) -> None:
        from skynet.providers.errors import ProviderError

        class Streamer:
            name = "streamer"
            streams = True
            accepts_timeout_override = True
            seen: ClassVar[list[float | None]] = []

            def complete(self, *_args: object, timeout_seconds: float | None = None, **_kwargs: object) -> ModelTurn:
                self.seen.append(timeout_seconds)
                raise ProviderError("busy", category="server", retryable=True)

        class Silent:
            name = "silent"
            accepts_timeout_override = True
            seen: ClassVar[list[float | None]] = []

            def complete(self, *_args: object, timeout_seconds: float | None = None, **_kwargs: object) -> ModelTurn:
                self.seen.append(timeout_seconds)
                raise ProviderError("busy", category="server", retryable=True)

        streamer, silent = Streamer(), Silent()
        fallback = FallbackProvider(
            [streamer, silent],
            attempts_per_provider=1,
            provider_block_seconds=0.0,
            dead_passes=1,
            stream_idle_seconds=120.0,
            attempt_timeout_seconds=900.0,
        )
        with self.assertRaises(ProviderError):
            fallback.complete([], max_tokens=1, patient=True)
        self.assertEqual(streamer.seen, [120.0], "a streamer is bounded by silence between chunks")
        self.assertEqual(silent.seen, [900.0], "a silent provider gets the whole-think budget")

    def test_openrouter_slow_stream_is_not_killed_by_the_connect_timeout(self) -> None:
        class Response:
            def __enter__(self) -> Response:
                return self

            def __exit__(self, *_exc: object) -> bool:
                return False

            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
                time.sleep(0.05)
                yield b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
                time.sleep(0.05)
                yield b'data: [DONE]\n\n'

        provider = OpenRouterProvider(base_url=self.base_url, api_key="openrouter-secret", model="m", timeout_seconds=0.01)
        with patch("skynet.providers.openrouter.request.urlopen", return_value=Response()):
            result = provider.complete([], max_tokens=1)
        self.assertEqual(result.text, "ab")

    def test_openrouter_stream_deadline_still_bounds_a_runaway_stream(self) -> None:
        class Response:
            def __enter__(self) -> Response:
                return self

            def __exit__(self, *_exc: object) -> bool:
                return False

            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
                time.sleep(0.05)
                yield b'data: [DONE]\n\n'

        provider = OpenRouterProvider(
            base_url=self.base_url,
            api_key="openrouter-secret",
            model="m",
            timeout_seconds=5.0,
            stream_deadline_seconds=0.01,
        )
        with patch("skynet.providers.openrouter.request.urlopen", return_value=Response()), self.assertRaisesRegex(Exception, "timed out"):
            provider.complete([], max_tokens=1)

    def test_openrouter_timeout_override_is_an_idle_budget_not_a_deadline(self) -> None:
        """The fallback ladder's rung must bound silence, never total stream time.

        Treating it as a total deadline cut off long, actively-streaming
        reasoning responses and produced the openrouter strike storm.
        """
        provider = OpenRouterProvider(
            base_url=self.base_url,
            api_key="openrouter-secret",
            model="m",
            timeout_seconds=5.0,
            chunk_timeout_seconds=900.0,
            stream_deadline_seconds=None,
        )
        # The override narrows the silence budget; it never becomes a total cap.
        self.assertEqual(provider._read_timeout(0.1, None), 0.1)
        self.assertEqual(provider._read_timeout(900.0, None), 900.0)
        # A deadline, when an operator sets one, only shrinks the read window
        # (floored at 0.1s so a near-expired deadline still attempts a read).
        self.assertLessEqual(provider._read_timeout(900.0, time.monotonic() + 0.05), 0.1)
        # Off by default: no total-stream cap.
        self.assertIsNone(provider.stream_deadline_seconds)

    def test_factory_builds_ollama_gemma_and_nemotron_ultra_chain(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "ollama_gemma,nvidia_ultra",
            "OLLAMA_API_KEYS": "secret-one,secret-two",
            "OLLAMA_MODEL": "model-a",
            "NVIDIA_API_KEY": "nvidia-secret",
            "OPENROUTER_ENABLED": "false",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual([item.name for item in provider.providers], ["ollama_gemma", "nvidia_ultra"])
        ultra = provider.providers[1]
        self.assertEqual(ultra.model, "nvidia/nemotron-3-ultra-550b-a55b")
        self.assertEqual(ultra.temperature, 1.0)
        self.assertEqual(ultra.top_p, 0.95)
        self.assertEqual(ultra.timeout_seconds, 120.0)
        self.assertEqual(ultra.max_output_tokens, 8192)

    def test_factory_builds_nvidia_deepseek_before_nemotron(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "openrouter_deepseek,nvidia_deepseek,nvidia_ultra,ollama_gemma",
            "DEEPSEEK_API_KEY": "deepseek-secret",
            "NVIDIA_API_KEY": "nvidia-secret",
            "OLLAMA_API_KEYS": "ollama-secret",
            "OPENROUTER_ENABLED": "false",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual([item.name for item in provider.providers], ["nvidia_deepseek", "nvidia_ultra", "ollama_gemma"])
        deepseek = provider.providers[0]
        self.assertEqual(deepseek.model, "deepseek-ai/deepseek-v4.1-flash")
        self.assertEqual(deepseek.timeout_seconds, 120.0)
        self.assertEqual(deepseek.max_output_tokens, 8192)
        self.assertEqual(deepseek.cooldown_overrides[429], 300.0)
        self.assertEqual(deepseek.network_cooldown_seconds, 60.0)

    def test_factory_can_enable_openrouter_explicitly(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "ollama_gemma,nvidia_ultra,openrouter_deepseek",
            "OLLAMA_API_KEYS": "secret-one,secret-two",
            "NVIDIA_API_KEY": "nvidia-secret",
            "OPENROUTER_ENABLED": "true",
            "OPENROUTER_API_KEY": "openrouter-secret",
            "SKYNET_MONEY_BOOST": "true",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual([item.name for item in provider.providers], ["ollama_gemma", "nvidia_ultra", "openrouter_deepseek"])

    def test_money_boost_hot_reload_works_both_ways_and_keeps_the_full_set(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "openrouter_deepseek,ollama_gemma",
            "OPENROUTER_ENABLED": "true",
            "OPENROUTER_API_KEY": "openrouter-secret",
            "OLLAMA_API_KEYS": "ollama-secret",
        }
        with tempfile.TemporaryDirectory() as directory:
            boost = Path(directory) / "money-boost.json"
            with patch.dict(os.environ, environment, clear=True), patch("skynet.providers.money_boost_state_path", return_value=boost):
                provider = cast(Any, build_provider())
                from skynet.providers import active_chain_names

                def active_names() -> list[str]:
                    return [item.name for item in provider._active_providers()]

                full_set = ["openrouter_deepseek", "ollama_gemma"]
                self.assertEqual([item.name for item in provider.providers], full_set)
                # money-boost off: openrouter_deepseek is constructed but not in
                # the active view, and a cooldown on it is retained for later.
                self.assertEqual(active_chain_names(), ["ollama_gemma"])
                self.assertEqual(active_names(), ["ollama_gemma"])
                assert provider._blocked_until is not None
                provider._blocked_until["openrouter_deepseek"] = time.time() + 60
                # money-boost on: openrouter_deepseek returns to the active view
                # with no restart, and the full set was never mutated.
                boost.write_text('{"enabled": true}', encoding="utf-8")
                self.assertEqual(active_chain_names(), ["openrouter_deepseek", "ollama_gemma"])
                self.assertEqual(active_names(), ["openrouter_deepseek", "ollama_gemma"])
                self.assertEqual([item.name for item in provider.providers], full_set)
                assert provider._blocked_until is not None
                self.assertIn("openrouter_deepseek", provider._blocked_until)
                # money-boost off again: openrouter_deepseek leaves the view but
                # stays built.
                boost.write_text('{"enabled": false}', encoding="utf-8")
                self.assertEqual(active_chain_names(), ["ollama_gemma"])
                self.assertEqual(active_names(), ["ollama_gemma"])
                self.assertEqual([item.name for item in provider.providers], full_set)

    def test_active_view_filters_per_call_without_mutating_the_full_set(self) -> None:
        class Named:
            def __init__(self, name: str) -> None:
                self.name = name

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text=self.name)

        active = {"a"}
        fallback = FallbackProvider([Named("a"), Named("b")], active_names=lambda: sorted(active))
        self.assertEqual(fallback.complete([], max_tokens=1).text, "a")
        active.clear()
        active.add("b")
        self.assertEqual(fallback.complete([], max_tokens=1).text, "b")
        active.clear()
        active.add("a")
        self.assertEqual(fallback.complete([], max_tokens=1).text, "a")
        self.assertEqual([cast(Any, provider).name for provider in fallback.providers], ["a", "b"])

    def test_the_backup_paid_rung_is_parked_by_its_own_flag(self) -> None:
        """The backup rung is parkable without benching the primary it shares a key with.

        Both paid rungs live on one endpoint and one credential, but not on one
        model: a key whose catalog carries only the primary answers HTTP 403 for
        the backup, and 403 classifies as ``auth`` -- which ``fallback._is_fatal``
        treats as chain-fatal. So the backup needs a flag of its own, ahead of the
        shared one (measured 2026-10-16: openrouter lists exactly
        ``deepseek/deepseek-v4.1-flash``).
        """
        from skynet.providers import active_chain_names

        chain = {
            "SKYNET_PROVIDER_CHAIN": "openrouter_deepseek,openrouter_glm,ollama_gemma",
            "OPENROUTER_ENABLED": "true",
            "OPENROUTER_API_KEY": "openrouter-secret",
            "OLLAMA_API_KEYS": "ollama-secret",
        }
        with tempfile.TemporaryDirectory() as directory:
            boost = Path(directory) / "money-boost.json"
            boost.write_text('{"enabled": true}', encoding="utf-8")
            with patch.dict(os.environ, chain, clear=True), patch("skynet.providers.money_boost_state_path", return_value=boost):
                # Unset, the backup follows the shared flag: nothing changes for a
                # key that does carry the model.
                self.assertEqual(active_chain_names(), ["openrouter_deepseek", "openrouter_glm", "ollama_gemma"])
                self.assertEqual([item.name for item in cast(Any, build_provider()).providers], ["openrouter_deepseek", "openrouter_glm", "ollama_gemma"])
            parked = {**chain, "OPENROUTER_GLM_ENABLED": "false"}
            with patch.dict(os.environ, parked, clear=True), patch("skynet.providers.money_boost_state_path", return_value=boost):
                self.assertEqual(active_chain_names(), ["openrouter_deepseek", "ollama_gemma"], "the backup leaves the active view")
                self.assertEqual([item.name for item in cast(Any, build_provider()).providers], ["openrouter_deepseek", "ollama_gemma"], "...and is not even constructed")
            shared = {**chain, "OPENROUTER_ENABLED": "false"}
            with patch.dict(os.environ, shared, clear=True), patch("skynet.providers.money_boost_state_path", return_value=boost):
                self.assertEqual(active_chain_names(), ["ollama_gemma"], "the shared flag still parks the whole endpoint")

    def test_provider_enable_knobs_are_unified_and_back_compatible(self) -> None:
        from skynet.providers import active_chain_names

        chain = {
            "SKYNET_PROVIDER_CHAIN": "nvidia_deepseek,nvidia_super,ollama_gemma",
            "DEEPSEEK_API_KEY": "deepseek-secret",
            "NVIDIA_API_KEY": "nvidia-secret",
            "OLLAMA_API_KEYS": "ollama-secret",
        }
        # Canonical names disable the provider in both the constructor and the
        # hot-reload filter.
        with patch.dict(os.environ, {**chain, "NVIDIA_DEEPSEEK_ENABLED": "false", "NEMOTRON_SUPER_ENABLED": "false"}, clear=True):
            provider = cast(Any, build_provider())
            self.assertEqual(active_chain_names(), ["ollama_gemma"])
        self.assertEqual([item.name for item in provider.providers], ["ollama_gemma"])
        # Legacy names still work, and the canonical flag wins when both exist.
        with patch.dict(os.environ, {**chain, "DEEPSEEK_ENABLED": "false", "NVIDIA_ENABLED": "false"}, clear=True):
            provider = cast(Any, build_provider())
            self.assertEqual(active_chain_names(), ["ollama_gemma"])
        self.assertEqual([item.name for item in provider.providers], ["ollama_gemma"])
        with patch.dict(
            os.environ,
            {**chain, "DEEPSEEK_ENABLED": "false", "NVIDIA_DEEPSEEK_ENABLED": "true", "NVIDIA_ENABLED": "false", "NEMOTRON_ENABLED": "true"},
            clear=True,
        ):
            provider = cast(Any, build_provider())
            self.assertEqual(active_chain_names(), ["nvidia_deepseek", "nvidia_super", "ollama_gemma"])
        self.assertEqual([item.name for item in provider.providers], ["nvidia_deepseek", "nvidia_super", "ollama_gemma"])

    def test_factory_enables_openrouter_with_money_boost_and_bounds(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "openrouter_deepseek,ollama_gemma",
            "OPENROUTER_ENABLED": "true",
            "OPENROUTER_API_KEY": "openrouter-secret",
            "OPENROUTER_TIMEOUT_SECONDS": "30",
            "OPENROUTER_MAX_ATTEMPTS": "3",
            "OLLAMA_API_KEYS": "ollama-secret",
        }
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "money-boost.json"
            state.write_text('{"enabled": true}\n', encoding="utf-8")
            with patch.dict(os.environ, environment, clear=True), patch("skynet.providers.money_boost_state_path", return_value=state):
                provider = cast(Any, build_provider())
        self.assertEqual([item.name for item in provider.providers], ["openrouter_deepseek", "ollama_gemma"])
        openrouter = cast(OpenRouterProvider, provider.providers[0])
        self.assertEqual(openrouter.timeout_seconds, 30.0)
        self.assertEqual(openrouter.max_attempts, 3)

    def test_factory_builds_openrouter_deepseek_then_glm_backup_rung(self) -> None:
        """OpenRouter carries two models as two rungs: DeepSeek first, GLM as backup.

        The rungs must be separately named, or a DeepSeek cooldown would bench
        the GLM backup too (FallbackProvider keys cooldowns by provider name).
        """
        environment = {
            "SKYNET_PROVIDER_CHAIN": "openrouter_deepseek,openrouter_glm",
            "OPENROUTER_ENABLED": "true",
            "OPENROUTER_API_KEY": "openrouter-secret",
            "OPENROUTER_MODEL": "deepseek/deepseek-v4.1-flash",
        }
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "money-boost.json"
            state.write_text('{"enabled": true}', encoding="utf-8")
            with patch.dict(os.environ, environment, clear=True), patch("skynet.providers.money_boost_state_path", return_value=state):
                provider = cast(Any, build_provider())
                from skynet.providers import active_chain_names

                self.assertEqual(active_chain_names(), ["openrouter_deepseek", "openrouter_glm"])
        self.assertEqual([item.name for item in provider.providers], ["openrouter_deepseek", "openrouter_glm"])
        primary = cast(OpenRouterProvider, provider.providers[0])
        backup = cast(OpenRouterProvider, provider.providers[1])
        self.assertEqual(primary.name, "openrouter_deepseek")
        self.assertEqual(primary.model, "deepseek/deepseek-v4.1-flash")
        self.assertEqual(backup.name, "openrouter_glm")
        self.assertEqual(backup.model, "z-ai/glm-5.3-flash")

    def test_backup_rung_leaves_the_active_view_without_money_boost(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "openrouter_deepseek,openrouter_glm",
            "OPENROUTER_ENABLED": "true",
            "OPENROUTER_API_KEY": "openrouter-secret",
        }
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "money-boost.json"
            state.write_text('{"enabled": false}', encoding="utf-8")
            with patch.dict(os.environ, environment, clear=True), patch("skynet.providers.money_boost_state_path", return_value=state):
                provider = cast(Any, build_provider())
                from skynet.providers import active_chain_names

                self.assertEqual(active_chain_names(), [])
                self.assertEqual([item.name for item in provider.providers], ["openrouter_deepseek", "openrouter_glm"])

    def test_factory_builds_requested_provider_order(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "openrouter_deepseek,nvidia_deepseek,nvidia_super,ollama_gemma",
            "OPENROUTER_ENABLED": "true", "OPENROUTER_API_KEY": "openrouter-secret",
            "SKYNET_MONEY_BOOST": "true",
            "DEEPSEEK_API_KEY": "nvidia-secret", "NVIDIA_API_KEY": "nvidia-secret",
            "OLLAMA_API_KEYS": "ollama-secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual([item.name for item in provider.providers], ["openrouter_deepseek", "nvidia_deepseek", "nvidia_super", "ollama_gemma"])

    def test_451_is_classified_as_unavailable_retryable(self) -> None:
        category, retryable, cooldown = OpenAICompatibleProvider._classify_status(451)
        self.assertEqual(category, "unavailable")
        self.assertTrue(retryable)
        self.assertGreater(cooldown, 0)

    def test_fallback_moves_to_next_provider_on_451(self) -> None:
        from skynet.providers.errors import ProviderError

        class BlockedNvidia:
            name = "nvidia"
            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("nvidia request failed: HTTP 451:", category="unavailable", retryable=True, cooldown_seconds=10.0)

        ollama = OpenAICompatibleProvider(name="ollama", base_url=self.base_url, api_key="key", model="test-model")
        result = FallbackProvider([BlockedNvidia(), ollama]).complete([], max_tokens=1)
        self.assertEqual(result.text, "ok")

    def test_factory_passes_proxy_url_to_nvidia_nemotron(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "nvidia_ultra",
            "NVIDIA_API_KEY": "nvidia-secret",
            "NVIDIA_PROXY_URL": "socks5://127.0.0.1:10808",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider = cast(Any, build_provider())
        nemotron = provider.providers[0]
        self.assertEqual(nemotron.name, "nvidia_ultra")
        self.assertEqual(nemotron.proxy_url, "socks5://127.0.0.1:10808")

    def test_factory_builds_zai_free_rungs_after_nemotron(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "nvidia_super,zai_glm47_flash,zai_glm45_flash,ollama_gemma",
            "NVIDIA_API_KEY": "nvidia-secret",
            "ZAI_API_KEY": "zai-secret",
            "OLLAMA_API_KEYS": "ollama-secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual(
            [item.name for item in provider.providers],
            ["nvidia_super", "zai_glm47_flash", "zai_glm45_flash", "ollama_gemma"],
        )
        primary, backup = provider.providers[1], provider.providers[2]
        self.assertEqual(primary.model, "glm-4.7-flash")
        self.assertEqual(backup.model, "glm-4.5-flash")
        self.assertEqual(primary.base_url, "https://api.z.ai/api/paas/v4")
        self.assertEqual(primary.timeout_seconds, 300.0)
        self.assertEqual(primary.cooldown_overrides[429], 120.0)
        # The endpoint ceiling (131072) is above every budget we send, so the
        # rung must not clamp our output.
        self.assertIsNone(primary.max_output_tokens)
        # Thinking is off by default: the switch is sent explicitly disabled.
        self.assertEqual(primary.extra_body, {"thinking": {"type": "disabled"}})
        self.assertEqual(backup.extra_body, {"thinking": {"type": "disabled"}})

    def test_zai_aliases_normalise_to_canonical_rungs(self) -> None:
        from skynet.providers import canonical_provider_name

        self.assertEqual(canonical_provider_name("zai"), "zai_glm47_flash")
        self.assertEqual(canonical_provider_name("zai_glm45"), "zai_glm45_flash")
        self.assertEqual(canonical_provider_name("zai_glm47_flash"), "zai_glm47_flash")

    def test_zai_default_chain_places_the_free_rungs_before_ollama(self) -> None:
        from skynet.providers import DEFAULT_CHAIN

        self.assertEqual(
            DEFAULT_CHAIN,
            "openrouter_deepseek,openrouter_glm,nvidia_deepseek,nvidia_super,"
            "nvidia_ultra,nvidia_lightning,"
            "zai_glm47_flash,zai_glm45_flash,mistral_leanstral,"
            "mistral_codestral,muse_glimmer,ollama_gemma",
        )

    def test_factory_builds_nemotron_lightning_rung(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "nvidia_lightning",
            "NVIDIA_API_KEY": "nvidia-secret",
            "NEMOTRON_LIGHTNING_ENABLED": "true",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual([item.name for item in provider.providers], ["nvidia_lightning"])
        lightning = provider.providers[0]
        self.assertEqual(lightning.model, "nvidia/nemotron-3.5-lightning-30b-a3b")
        self.assertEqual(lightning.temperature, 1.0)
        self.assertEqual(lightning.top_p, 0.95)
        self.assertEqual(lightning.timeout_seconds, 120.0)
        self.assertEqual(lightning.max_output_tokens, 8192)

    def test_factory_builds_muse_glimmer_rung_at_the_tail(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "mistral_codestral,muse_glimmer,ollama_gemma",
            "NVIDIA_API_KEY": "nvidia-secret",
            "MISTRAL_API_KEY": "mistral-secret",
            "OLLAMA_API_KEYS": "ollama-secret",
            "MUSE_GLIMMER_ENABLED": "true",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual(
            [item.name for item in provider.providers],
            ["mistral_codestral", "muse_glimmer", "ollama_gemma"],
        )
        glimmer = provider.providers[1]
        self.assertEqual(glimmer.model, "meta/muse-glimmer-30b")
        self.assertEqual(glimmer.temperature, 1.0)
        self.assertEqual(glimmer.top_p, 0.95)
        self.assertEqual(glimmer.timeout_seconds, 120.0)
        self.assertEqual(glimmer.max_output_tokens, 8192)

    def test_new_nim_aliases_normalise_to_canonical_rungs(self) -> None:
        from skynet.providers import canonical_provider_name

        self.assertEqual(canonical_provider_name("nemotron_lightning"), "nvidia_lightning")
        self.assertEqual(canonical_provider_name("nemotron_3_5_lightning"), "nvidia_lightning")
        self.assertEqual(canonical_provider_name("muse"), "muse_glimmer")
        self.assertEqual(canonical_provider_name("glimmer"), "muse_glimmer")

    def test_mistral_rungs_wire_from_env_with_the_120s_429_brake(self) -> None:
        env = {
            "SKYNET_PROVIDER_CHAIN": "mistral_leanstral,mistral_codestral",
            "MISTRAL_API_KEY": "mistral-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual([item.name for item in provider.providers], ["mistral_leanstral", "mistral_codestral"])
        leanstral, codestral = provider.providers
        self.assertEqual(leanstral.model, "labs-leanstral-1-5")
        self.assertEqual(codestral.model, "codestral-latest")
        self.assertEqual(leanstral.base_url, "https://api.mistral.ai/v1")
        for rung in (leanstral, codestral):
            # leanstral rejects a sampling temperature (measured 2026-09-27), the
            # free tier served ~14k output tokens with no ceiling error, and the
            # rungs use no proxy.
            self.assertIsNone(rung.temperature)
            self.assertIsNone(rung.top_p)
            self.assertIsNone(rung.max_output_tokens)
            self.assertIsNone(rung.proxy_url)
            self.assertEqual(rung.cooldown_overrides[429], 120.0)

    def test_mistral_aliases_normalise_to_canonical_rungs(self) -> None:
        from skynet.providers import canonical_provider_name

        self.assertEqual(canonical_provider_name("leanstral"), "mistral_leanstral")
        self.assertEqual(canonical_provider_name("codestral"), "mistral_codestral")
        self.assertEqual(canonical_provider_name("mistral_leanstral"), "mistral_leanstral")

    def test_mistral_rungs_skip_without_key_or_when_disabled(self) -> None:
        chain = {
            "SKYNET_PROVIDER_CHAIN": "mistral_leanstral,mistral_codestral,ollama_gemma",
            "OLLAMA_API_KEYS": "ollama-secret",
        }
        with patch.dict(os.environ, chain, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual([item.name for item in provider.providers], ["ollama_gemma"])
        with patch.dict(os.environ, {**chain, "MISTRAL_API_KEY": "secret", "MISTRAL_ENABLED": "false"}, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual([item.name for item in provider.providers], ["ollama_gemma"])

    def test_mistral_rungs_are_not_money_boost_gated(self) -> None:
        from skynet.providers import PAID_PROVIDER_NAMES, active_chain_names

        self.assertNotIn("mistral_leanstral", PAID_PROVIDER_NAMES)
        self.assertNotIn("mistral_codestral", PAID_PROVIDER_NAMES)
        environment = {
            "SKYNET_PROVIDER_CHAIN": "mistral_leanstral,ollama_gemma",
            "MISTRAL_API_KEY": "secret",
            "OLLAMA_API_KEYS": "ollama-secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(active_chain_names(), ["mistral_leanstral", "ollama_gemma"])

    def test_a_prompt_over_the_context_ceiling_is_provider_local(self) -> None:
        mistral = '{"object":"error","message":"Prompt 138892 > 32768 maximum context length","type":"invalid_request_prompt_too_long","code":"3059"}'
        self.assertEqual(OpenAICompatibleProvider._classify_status(400, mistral), ("prompt_too_long", False, 0.0))
        # Z.AI names the ceiling only in the message text.
        zai = '{"error":{"message":"Prompt exceeds max length","code":"1261","type":"invalid_request_error"}}'
        self.assertEqual(OpenAICompatibleProvider._classify_status(400, zai), ("prompt_too_long", False, 0.0))
        # An unrelated 400 is still a broken request.
        self.assertEqual(
            OpenAICompatibleProvider._classify_status(400, '{"error":{"type":"invalid_request_error","message":"bad schema"}}'),
            ("invalid_request", False, 0.0),
        )

    def test_prompt_too_long_does_not_abort_the_chain(self) -> None:
        class Known:
            name = "known"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="ok")

        fallback = FallbackProvider([Known()])
        self.assertFalse(fallback._is_fatal(ProviderError("too long", category="prompt_too_long", retryable=False)))
        self.assertFalse(fallback._is_fatal(ProviderError("malformed", category="invalid_response", retryable=False)))
        self.assertTrue(fallback._is_fatal(ProviderError("broken", category="invalid_request", retryable=False)))
        self.assertTrue(fallback._is_fatal(ProviderError("bad key", category="auth", retryable=False)))

    def test_aborted_generation_is_a_retryable_quality_failure(self) -> None:
        _Handler.bodies = [
            json.dumps({"choices": [{"finish_reason": "error", "message": {"content": "partial abab"}}], "usage": {"total_tokens": 5}})
        ]
        provider = OpenAICompatibleProvider(name="mistral_codestral", base_url=self.base_url, api_key="key", model="codestral-latest")
        with self.assertRaises(ProviderError) as raised:
            provider.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        self.assertEqual(raised.exception.category, "response_quality")
        self.assertTrue(raised.exception.retryable)

    def test_model_length_without_content_is_retryable(self) -> None:
        _Handler.bodies = [json.dumps({"choices": [{"finish_reason": "model_length", "message": {"content": ""}}], "usage": {"total_tokens": 5}})]
        provider = OpenAICompatibleProvider(name="plain", base_url=self.base_url, api_key="key", model="m")
        with self.assertRaises(ProviderError) as raised:
            provider.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        self.assertEqual(raised.exception.category, "response_quality")

    def test_an_explicit_null_tool_calls_is_not_a_malformed_response(self) -> None:
        # Mistral-shaped replies carry ``"tool_calls": null``; iterating that
        # None used to raise and turn every Mistral turn into invalid_response.
        _Handler.bodies = [
            json.dumps({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok", "tool_calls": None}}], "usage": {"total_tokens": 3}})
        ]
        provider = OpenAICompatibleProvider(name="mistral_codestral", base_url=self.base_url, api_key="key", model="codestral-latest")
        turn = provider.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        self.assertEqual(turn.text, "ok")
        self.assertEqual(turn.tool_calls, [])
        self.assertEqual(turn.finish_reason, "stop")

    def test_zai_rungs_skip_without_key_or_when_disabled(self) -> None:
        chain = {
            "SKYNET_PROVIDER_CHAIN": "zai_glm47_flash,zai_glm45_flash,ollama_gemma",
            "OLLAMA_API_KEYS": "ollama-secret",
        }
        with patch.dict(os.environ, chain, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual([item.name for item in provider.providers], ["ollama_gemma"])
        with patch.dict(os.environ, {**chain, "ZAI_API_KEY": "zai-secret", "ZAI_ENABLED": "false"}, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual([item.name for item in provider.providers], ["ollama_gemma"])

    def test_zai_rungs_are_not_money_boost_gated(self) -> None:
        from skynet.providers import PAID_PROVIDER_NAMES, active_chain_names

        self.assertNotIn("zai_glm47_flash", PAID_PROVIDER_NAMES)
        self.assertNotIn("zai_glm45_flash", PAID_PROVIDER_NAMES)
        environment = {
            "SKYNET_PROVIDER_CHAIN": "zai_glm47_flash,ollama_gemma",
            "ZAI_API_KEY": "zai-secret",
            "OLLAMA_API_KEYS": "ollama-secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(active_chain_names(), ["zai_glm47_flash", "ollama_gemma"])

    def test_zai_thinking_can_be_reenabled_by_env(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "zai_glm47_flash",
            "ZAI_API_KEY": "zai-secret",
            "ZAI_THINKING": "true",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider = cast(Any, build_provider())
        # No ``thinking`` field at all: the endpoint default applies again.
        self.assertEqual(provider.providers[0].extra_body, {})

    def test_extra_body_is_merged_into_the_request_payload(self) -> None:
        provider = OpenAICompatibleProvider(
            name="zai", base_url=self.base_url, api_key="key", model="glm-4.7-flash",
            extra_body={"thinking": {"type": "disabled"}},
        )
        provider.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        payload = _Handler.payloads[-1]
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["model"], "glm-4.7-flash")

    def test_tool_choice_is_carried_only_when_asked_for(self) -> None:
        """A normal step must let the model choose; the closing turn forces one tool.

        Probed live against the NVIDIA NIM ring 2026-10-19: both the named form and "required"
        answer 200 with a tool call (`finish_reason: tool_calls`, no text), which is why the field
        can be sent at all -- and why a server that refuses it fails the call instead of quietly
        answering in prose.
        """
        tools = [{"type": "function", "function": {"name": "finish", "parameters": {"type": "object", "additionalProperties": False}}}]
        provider = OpenAICompatibleProvider(name="nim", base_url=self.base_url, api_key="key", model="m")
        provider.complete([{"role": "user", "content": "hi"}], max_tokens=5, tools=tools)
        self.assertNotIn("tool_choice", _Handler.payloads[-1], "a normal step must let the model choose its own tool")
        provider.complete(
            [{"role": "user", "content": "close"}],
            max_tokens=5,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "finish"}},
        )
        self.assertEqual(_Handler.payloads[-1]["tool_choice"], {"type": "function", "function": {"name": "finish"}})
        self.assertTrue(provider.accepts_tool_choice, "the capability is a flag, like the timeout override")

    def test_tool_choice_without_tools_is_not_sent(self) -> None:
        provider = OpenAICompatibleProvider(name="nim", base_url=self.base_url, api_key="key", model="m")
        provider.complete(
            [{"role": "user", "content": "hi"}],
            max_tokens=5,
            tool_choice={"type": "function", "function": {"name": "finish"}},
        )
        self.assertNotIn("tool_choice", _Handler.payloads[-1], "forcing a tool that was not offered is not a servable request")

    def test_no_extra_body_leaves_the_payload_untouched(self) -> None:
        provider = OpenAICompatibleProvider(name="plain", base_url=self.base_url, api_key="key", model="m")
        provider.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        self.assertNotIn("thinking", _Handler.payloads[-1])

    def test_proxy_scope_is_reentrant_and_restores_socket(self) -> None:
        import socket

        provider = OpenAICompatibleProvider(name="proxied", base_url="http://example.invalid", api_key="", model="m", proxy_url="socks5://127.0.0.1:10808")
        original = socket.create_connection
        with provider._proxy_scope():
            patched = socket.create_connection
            self.assertIsNot(patched, original)
            with provider._proxy_scope():
                self.assertIs(socket.create_connection, patched)
            self.assertIs(socket.create_connection, patched)
        self.assertIs(socket.create_connection, original)

    def test_all_keys_permanent_gives_positive_backoff_and_message(self) -> None:
        rotator = KeyRotator(["one", "two"], permanent_backoff_seconds=1800.0)
        rotator.mark_failed(0, permanent=True)
        rotator.mark_failed(1, permanent=True)
        self.assertGreater(rotator.earliest_available_in(), 0.0)
        self.assertEqual(rotator.total, 2)

        provider = OllamaCloudProvider(self.base_url, ["one", "two"], "test-model")
        provider._rotator.mark_failed(0, permanent=True)
        provider._rotator.mark_failed(1, permanent=True)
        with self.assertRaises(ProviderError) as raised:
            provider.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        tail = str(raised.exception).split("exhausted:", 1)[1].strip()
        self.assertTrue(tail)
        self.assertGreater(raised.exception.cooldown_seconds, 0.0)

    def test_zero_cooldown_retryable_persists_min_block(self) -> None:
        class Down:
            name = "down"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("down", category="network", retryable=True, cooldown_seconds=0.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provider-fallback.json"
            fallback = FallbackProvider([Down()], state_path=path)
            with self.assertRaises(ProviderError):
                fallback.complete([], max_tokens=1)
            stored = json.loads(path.read_text(encoding="utf-8"))
            remaining = stored["blocked_until"]["down"] - time.time()
            self.assertGreaterEqual(remaining, 29.0)

    def test_aggregate_failure_message_is_actionable(self) -> None:
        class Alpha:
            name = "alpha"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("alpha down", category="network", retryable=True, cooldown_seconds=0.0)

        class Beta:
            name = "beta"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("beta busy", category="server", retryable=True, cooldown_seconds=5.0)

        with self.assertRaises(ProviderError) as raised:
            FallbackProvider([Alpha(), Beta()]).complete([], max_tokens=1)
        message = str(raised.exception)
        for fragment in ("alpha", "network", "beta", "server", "blocked_until"):
            self.assertIn(fragment, message)
        self.assertGreater(raised.exception.cooldown_seconds, 0.0)

    def test_cooldown_state_round_trips_between_instances(self) -> None:
        class Down:
            name = "down"
            calls = 0

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                self.calls += 1
                raise ProviderError("down", category="network", retryable=True, cooldown_seconds=0.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provider-fallback.json"
            first = Down()
            fallback = FallbackProvider([first], state_path=path)
            with self.assertRaises(ProviderError):
                fallback.complete([], max_tokens=1)
            self.assertEqual(first.calls, 1)
            second = Down()
            reloaded = FallbackProvider([second], state_path=path)
            with self.assertRaises(ProviderError):
                reloaded.complete([], max_tokens=1)
            self.assertEqual(second.calls, 0)

    def test_key_rotator_state_round_trips_through_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provider-fallback.json"
            provider = OllamaCloudProvider(self.base_url, ["one", "two"], "test-model")
            provider._rotator.mark_failed(0, permanent=True)
            provider._rotator.mark_failed(1, cooldown_seconds=45.0)
            FallbackProvider([provider], state_path=path)._save_state()
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("key_state", stored)
            self.assertIn("ollama_gemma", stored["key_state"])

            reloaded = OllamaCloudProvider(self.base_url, ["one", "two"], "test-model")
            FallbackProvider([reloaded], state_path=path)
            status = reloaded._rotator.status()
            self.assertTrue(status[0]["permanent"])
            self.assertFalse(status[1]["permanent"])
            self.assertGreater(cast(float, status[1]["cooldown_remaining"]), 0.0)

    def test_load_state_discards_unknown_provider(self) -> None:
        class Known:
            name = "known"

            def complete(self, messages, *, max_tokens, tools=()):
                return ModelTurn(text="known")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provider-fallback.json"
            path.write_text(
                json.dumps({"blocked_until": {"opencode_free": time.time() + 999, "known": time.time() + 50}}),
                encoding="utf-8",
            )
            fallback = FallbackProvider([Known()], state_path=path)
            assert fallback._blocked_until is not None
            self.assertNotIn("opencode_free", fallback._blocked_until)
            self.assertIn("known", fallback._blocked_until)

    def test_call_deadline_ends_the_patient_cycle(self) -> None:
        class Slow:
            name = "slow"
            calls = 0

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                self.calls += 1
                time.sleep(0.05)
                raise ProviderError("busy", category="server", retryable=True, cooldown_seconds=0.0)

        slow = Slow()
        fallback = FallbackProvider(
            [slow],
            attempts_per_provider=3,
            attempt_waits=(0.05, 0.05),
            provider_block_seconds=0.05,
            dead_passes=3,
            call_deadline_seconds=0.01,
        )
        with self.assertRaises(ProviderError):
            fallback.complete([], max_tokens=1, patient=True)
        self.assertEqual(slow.calls, 1)

    def test_patient_cycle_skips_a_provider_whose_prompt_is_too_long(self) -> None:
        class TooLong:
            name = "too-long"
            calls = 0

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                self.calls += 1
                raise ProviderError("prompt too long", category="prompt_too_long", retryable=False)

        class Works:
            name = "works"
            calls = 0

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                self.calls += 1
                return ModelTurn(text="done")

        too_long, works = TooLong(), Works()
        fallback = FallbackProvider(
            [too_long, works],
            attempts_per_provider=2,
            attempt_waits=(0.0,),
            provider_block_seconds=0.01,
        )
        turn = fallback.complete([], max_tokens=1, patient=True)
        self.assertEqual(turn.text, "done")
        self.assertEqual(works.calls, 1)
        self.assertGreaterEqual(too_long.calls, 1)

    def test_factory_wires_the_patient_cycle_from_env(self) -> None:
        env = {
            "SKYNET_PROVIDER_CHAIN": "nvidia_ultra",
            "NVIDIA_API_KEY": "test-key",
            "NEMOTRON_ULTRA_ENABLED": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            provider = cast(Any, build_provider())
        # The owner's numbers (2026-09-27): 4 grouped strikes per provider with
        # 5/10/15s waits, a 5-minute block, 3 dead passes before giving up, a
        # 2-minute silence budget for streamers and 15 minutes for a silent one.
        self.assertEqual(provider.attempts_per_provider, 4)
        self.assertEqual(provider.attempt_waits, (5.0, 10.0, 15.0))
        self.assertEqual(provider.provider_block_seconds, 300.0)
        self.assertEqual(provider.dead_passes, 3)
        self.assertEqual(provider.stream_idle_seconds, 120.0)
        self.assertEqual(provider.attempt_timeout_seconds, 900.0)

    def test_deepseek_quota_marker_keeps_until_midnight_over_override(self) -> None:
        noon = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        _Handler.statuses = [429]
        _Handler.bodies = ['{"error":{"message":"you have reached your usage limit"}}']
        provider = OpenAICompatibleProvider(
            name="nvidia_deepseek",
            base_url=self.base_url,
            api_key="key",
            model="test-model",
            cooldown_overrides={429: 300.0},
        )
        with patch("skynet.providers.openai_compatible.utc_datetime_now", return_value=noon), self.assertRaises(ProviderError) as raised:
            provider.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        self.assertEqual(raised.exception.category, "quota")
        self.assertEqual(raised.exception.cooldown_seconds, 12 * 60 * 60)

    def test_deepseek_plain_429_uses_the_override(self) -> None:
        _Handler.statuses = [429]
        _Handler.bodies = ['{"error":{"message":"rate limit exceeded"}}']
        provider = OpenAICompatibleProvider(
            name="nvidia_deepseek",
            base_url=self.base_url,
            api_key="key",
            model="test-model",
            cooldown_overrides={429: 300.0},
        )
        with self.assertRaises(ProviderError) as raised:
            provider.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        self.assertEqual(raised.exception.category, "rate_limit")
        self.assertEqual(raised.exception.cooldown_seconds, 300.0)

    def test_quota_at_2359_does_not_extend_past_midnight(self) -> None:
        just_before = datetime(2026, 1, 1, 23, 59, 0, tzinfo=UTC)
        with patch("skynet.providers.openai_compatible.utc_datetime_now", return_value=just_before):
            category, _, cooldown = OpenAICompatibleProvider._classify_status(429, '{"error":{"message":"usage limit"}}')
        self.assertEqual(category, "quota")
        self.assertGreater(cooldown, 0.0)
        self.assertLess(cooldown, 120.0)

    def test_auth_and_unknown_5xx_classification(self) -> None:
        category, retryable, _ = OpenAICompatibleProvider._classify_status(401)
        self.assertEqual(category, "auth")
        self.assertFalse(retryable)
        category, retryable, _ = OpenAICompatibleProvider._classify_status(521)
        self.assertEqual(category, "server")
        self.assertTrue(retryable)

    def test_response_without_usage_parses(self) -> None:
        _Handler.statuses = [200]
        _Handler.bodies = [json.dumps({"choices": [{"message": {"content": "hi", "tool_calls": []}}]})]
        provider = OpenAICompatibleProvider(name="ollama", base_url=self.base_url, api_key="key", model="test-model")
        result = provider.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        self.assertEqual(result.text, "hi")
        self.assertEqual(result.usage_tokens, 0)

        _Handler.statuses = [200]
        _Handler.bodies = [json.dumps({"choices": [{"message": {"content": "hi"}}], "usage": None})]
        result = provider.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        self.assertEqual(result.text, "hi")

    def test_legitimate_answer_is_not_a_reasoning_leak(self) -> None:
        provider = OpenAICompatibleProvider(name="nvidia", base_url=self.base_url, api_key="key", model="m", reject_reasoning_leakage=True)
        result = provider._parse({"choices": [{"message": {"content": "We need to fix the bug"}}]})
        self.assertEqual(result.text, "We need to fix the bug")
        with self.assertRaises(ProviderError):
            provider._parse({"choices": [{"message": {"content": "answer", "reasoning_content": "hidden"}}]})
        with self.assertRaises(ProviderError):
            provider._parse({"choices": [{"message": {"content": "analysis: hidden chain of thought"}}]})
        leak_tag = chr(60) + "thinking" + chr(62) + "hmm"
        with self.assertRaises(ProviderError):
            provider._parse({"choices": [{"message": {"content": leak_tag}}]})

    def test_openrouter_stream_without_done_is_retryable_and_falls_through(self) -> None:
        class Response:
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

        provider = OpenRouterProvider(base_url=self.base_url, api_key="openrouter-secret", model="m")
        with self.assertRaises(ProviderError) as raised:
            provider._parse_sse(Response(), idle_seconds=60.0)
        self.assertEqual(raised.exception.category, "network")
        self.assertTrue(raised.exception.retryable)

        class NoDoneOpenRouter:
            name = "openrouter"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                raise ProviderError("openrouter SSE stream ended before [DONE]", category="network", retryable=True, cooldown_seconds=2.0)

        ollama = OpenAICompatibleProvider(name="ollama", base_url=self.base_url, api_key="key", model="test-model")
        result = FallbackProvider([NoDoneOpenRouter(), ollama]).complete([], max_tokens=1)
        self.assertEqual(result.text, "ok")

    def test_openrouter_repeated_tool_deltas_are_deduplicated(self) -> None:
        class Response:
            def __iter__(self):
                chunks = [
                    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "bash", "arguments": "{\"cmd\":"}}]}}]},
                    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "bash", "arguments": "\"pwd\"}"}}]}}]},
                ]
                for chunk in chunks:
                    yield ("data: " + json.dumps(chunk) + "\n\n").encode()
                yield b"data: [DONE]\n\n"

        provider = OpenRouterProvider(base_url=self.base_url, api_key="openrouter-secret", model="m")
        result = provider._parse_sse(Response(), idle_seconds=60.0)
        self.assertEqual(result.tool_calls[0].tool_name, "bash")
        self.assertEqual(result.tool_calls[0].call_id, "call_1")
        self.assertEqual(result.tool_calls[0].arguments, {"cmd": "pwd"})

    def test_health_probe_all_blocked_reports_cooldowns(self) -> None:
        class Net:
            name = "net"

            def complete(self, messages, *, max_tokens, tools=()):
                return ModelTurn(text="net")

            def health_probe(self) -> dict[str, object]:
                raise AssertionError("health_probe must not touch the network while blocked")

        fallback = FallbackProvider([Net()])
        assert fallback._blocked_until is not None
        fallback._blocked_until["net"] = time.time() + 120
        result = fallback.health_probe()
        self.assertFalse(result["ok"])
        providers = cast(dict[str, dict[str, object]], result["providers"])
        self.assertTrue(providers["net"]["blocked"])
        self.assertGreater(cast(float, providers["net"]["cooldown_remaining_seconds"]), 0.0)
