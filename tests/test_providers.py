from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import patch

from skynet.models import ModelTurn
from skynet.providers import build_provider
from skynet.providers.errors import ProviderError
from skynet.providers.fallback import FallbackProvider
from skynet.providers.key_rotator import KeyRotator, KeySlot
from skynet.providers.ollama import OllamaCloudProvider
from skynet.providers.openai_compatible import OpenAICompatibleProvider
from skynet.providers.openrouter import OpenRouterProvider


class _Handler(BaseHTTPRequestHandler):
    statuses: ClassVar[list[int]] = []
    bodies: ClassVar[list[str]] = []
    auths: ClassVar[list[str]] = []
    reasoning_response = False

    def do_POST(self) -> None:
        _Handler.auths.append(self.headers.get("Authorization", ""))
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

    def test_fallback_attempt_is_not_reported_as_a_strike(self) -> None:
        # The pre-attempt signal says an attempt began, not that one failed.
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

    def test_fallback_ladder_escalates_timeouts_and_pauses_between_strikes(self) -> None:
        from skynet.providers.errors import ProviderError

        class Flaky:
            name = "flaky"
            accepts_timeout_override = True

            def __init__(self) -> None:
                self.rungs: list[float | None] = []

            def complete(self, *_args: object, timeout_seconds: float | None = None, **_kwargs: object) -> ModelTurn:
                self.rungs.append(timeout_seconds)
                raise ProviderError("busy", category="server", retryable=True)

        provider = Flaky()
        fallback = FallbackProvider([provider], timeout_ladder=(0.5, 1.0, 1.5), strike_delay_seconds=0.05)
        started = time.monotonic()
        with self.assertRaisesRegex(Exception, "all providers failed"):
            fallback.complete([], max_tokens=1)
        self.assertEqual(provider.rungs, [0.5, 1.0, 1.5])
        self.assertGreaterEqual(time.monotonic() - started, 0.1)

    def test_fallback_reports_model_time_without_strike_delay(self) -> None:
        """Backoff between strikes is infrastructure, not model thinking."""
        from skynet.providers.errors import ProviderError

        class Flaky:
            name = "flaky"
            accepts_timeout_override = True

            def complete(self, *_args: object, timeout_seconds: float | None = None, **_kwargs: object) -> ModelTurn:
                raise ProviderError("busy", category="server", retryable=True)

        class Good:
            name = "good"

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                return ModelTurn(text="ok")

        fallback = FallbackProvider([Flaky(), Good()], timeout_ladder=(1.0, 1.0), strike_delay_seconds=0.2)
        started = time.monotonic()
        turn = fallback.complete([], max_tokens=1)
        wall = time.monotonic() - started
        self.assertEqual(turn.text, "ok")
        self.assertGreater(wall, 0.15)
        self.assertLess(turn.model_seconds, wall - 0.1)

    def test_fallback_ladder_round_robins_across_providers(self) -> None:
        from skynet.providers.errors import ProviderError

        seen: list[str] = []

        class Flaky:
            accepts_timeout_override = True

            def __init__(self, name: str) -> None:
                self.name = name

            def complete(self, *_args: object, timeout_seconds: float | None = None, **_kwargs: object) -> ModelTurn:
                seen.append(self.name)
                raise ProviderError("busy", category="server", retryable=True)

        fallback = FallbackProvider([Flaky("a"), Flaky("b")], timeout_ladder=(1.0, 1.0, 1.0))
        with self.assertRaisesRegex(Exception, "all providers failed"):
            fallback.complete([], max_tokens=1)
        self.assertEqual(seen, ["a", "b", "a"])

    def test_fallback_without_a_ladder_strikes_each_provider_once(self) -> None:
        from skynet.providers.errors import ProviderError

        seen: list[str] = []

        class Flaky:
            def __init__(self, name: str) -> None:
                self.name = name

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                seen.append(self.name)
                raise ProviderError("busy", category="server", retryable=True)

        fallback = FallbackProvider([Flaky("a"), Flaky("b")], strike_delay_seconds=5.0)
        started = time.monotonic()
        with self.assertRaisesRegex(Exception, "all providers failed"):
            fallback.complete([], max_tokens=1)
        self.assertEqual(seen, ["a", "b"])
        self.assertLess(time.monotonic() - started, 1.0)

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
        """The fallback ladder's rung must bound silence, never total stream time."""
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

    def test_factory_builds_ollama_and_nvidia_chain(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "ollama,nvidia",
            "OLLAMA_API_KEYS": "secret-one,secret-two",
            "OLLAMA_MODEL": "model-a",
            "NVIDIA_API_KEY": "nvidia-secret",
            "OPENROUTER_ENABLED": "false",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual([getattr(item, "name", "ollama") for item in provider.providers], ["ollama", "nvidia"])
        nvidia = provider.providers[1]
        self.assertEqual(nvidia.temperature, 1.0)
        self.assertEqual(nvidia.top_p, 0.95)
        self.assertEqual(nvidia.timeout_seconds, 120.0)
        self.assertEqual(nvidia.max_output_tokens, 8192)
        self.assertTrue(nvidia.reject_reasoning_leakage)

    def test_factory_builds_nvidia_deepseek_before_nvidia(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "openrouter,nvidia_deepseek,nvidia,ollama",
            "DEEPSEEK_API_KEY": "deepseek-secret",
            "NVIDIA_API_KEY": "nvidia-secret",
            "OLLAMA_API_KEYS": "ollama-secret",
            "OPENROUTER_ENABLED": "false",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual([item.name for item in provider.providers], ["nvidia_deepseek", "nvidia", "ollama"])
        deepseek = provider.providers[0]
        self.assertEqual(deepseek.model, "deepseek-ai/deepseek-v4-flash-0731")
        self.assertEqual(deepseek.timeout_seconds, 120.0)
        self.assertEqual(deepseek.max_output_tokens, 8192)
        self.assertEqual(deepseek.cooldown_overrides[429], 300.0)
        self.assertEqual(deepseek.network_cooldown_seconds, 60.0)

    def test_factory_can_enable_openrouter_explicitly(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "ollama,nvidia,openrouter",
            "OLLAMA_API_KEYS": "secret-one,secret-two",
            "NVIDIA_API_KEY": "nvidia-secret",
            "OPENROUTER_ENABLED": "true",
            "OPENROUTER_API_KEY": "openrouter-secret",
            "SKYNET_MONEY_BOOST": "true",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual([getattr(item, "name", "ollama") for item in provider.providers], ["ollama", "nvidia", "openrouter"])

    def test_money_boost_hot_reload_works_both_ways_and_keeps_the_full_set(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "openrouter,ollama",
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

                full_set = ["openrouter", "ollama"]
                self.assertEqual([item.name for item in provider.providers], full_set)
                # money-boost off: openrouter is constructed but not in the active
                # view, and a cooldown on it is retained for later.
                self.assertEqual(active_chain_names(), ["ollama"])
                self.assertEqual(active_names(), ["ollama"])
                assert provider._blocked_until is not None
                provider._blocked_until["openrouter"] = time.time() + 60
                # money-boost on: openrouter returns to the active view with no
                # restart, and the full set was never mutated.
                boost.write_text('{"enabled": true}', encoding="utf-8")
                self.assertEqual(active_chain_names(), ["openrouter", "ollama"])
                self.assertEqual(active_names(), ["openrouter", "ollama"])
                self.assertEqual([item.name for item in provider.providers], full_set)
                assert provider._blocked_until is not None
                self.assertIn("openrouter", provider._blocked_until)
                # money-boost off again: openrouter leaves the view but stays built.
                boost.write_text('{"enabled": false}', encoding="utf-8")
                self.assertEqual(active_chain_names(), ["ollama"])
                self.assertEqual(active_names(), ["ollama"])
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

    def test_provider_enable_knobs_are_unified_and_back_compatible(self) -> None:
        from skynet.providers import active_chain_names

        chain = {
            "SKYNET_PROVIDER_CHAIN": "nvidia_deepseek,nemotron,ollama",
            "DEEPSEEK_API_KEY": "deepseek-secret",
            "NVIDIA_API_KEY": "nvidia-secret",
            "OLLAMA_API_KEYS": "ollama-secret",
        }
        # Canonical names disable the provider in both the constructor and the
        # hot-reload filter.
        with patch.dict(os.environ, {**chain, "NVIDIA_DEEPSEEK_ENABLED": "false", "NEMOTRON_ENABLED": "false"}, clear=True):
            provider = cast(Any, build_provider())
            self.assertEqual(active_chain_names(), ["ollama"])
        self.assertEqual([item.name for item in provider.providers], ["ollama"])
        # Legacy names still work, and the canonical flag wins when both exist.
        with patch.dict(os.environ, {**chain, "DEEPSEEK_ENABLED": "false", "NVIDIA_ENABLED": "false"}, clear=True):
            provider = cast(Any, build_provider())
            self.assertEqual(active_chain_names(), ["ollama"])
        self.assertEqual([item.name for item in provider.providers], ["ollama"])
        with patch.dict(
            os.environ,
            {**chain, "DEEPSEEK_ENABLED": "false", "NVIDIA_DEEPSEEK_ENABLED": "true", "NVIDIA_ENABLED": "false", "NEMOTRON_ENABLED": "true"},
            clear=True,
        ):
            provider = cast(Any, build_provider())
            self.assertEqual(active_chain_names(), ["nvidia_deepseek", "nemotron", "ollama"])
        self.assertEqual([item.name for item in provider.providers], ["nvidia_deepseek", "nemotron", "ollama"])

    def test_factory_enables_openrouter_with_money_boost_and_bounds(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "openrouter,ollama",
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
        self.assertEqual([item.name for item in provider.providers], ["openrouter", "ollama"])
        openrouter = cast(OpenRouterProvider, provider.providers[0])
        self.assertEqual(openrouter.timeout_seconds, 30.0)
        self.assertEqual(openrouter.max_attempts, 3)

    def test_factory_builds_requested_provider_order(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "openrouter,nvidia_deepseek,nemotron,ollama",
            "OPENROUTER_ENABLED": "true", "OPENROUTER_API_KEY": "openrouter-secret",
            "SKYNET_MONEY_BOOST": "true",
            "DEEPSEEK_API_KEY": "nvidia-secret", "NVIDIA_API_KEY": "nvidia-secret",
            "OLLAMA_API_KEYS": "ollama-secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider = cast(Any, build_provider())
        self.assertEqual([item.name for item in provider.providers], ["openrouter", "nvidia_deepseek", "nemotron", "ollama"])

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

    def test_factory_passes_proxy_url_to_nvidia(self) -> None:
        environment = {
            "SKYNET_PROVIDER_CHAIN": "nvidia",
            "NVIDIA_API_KEY": "nvidia-secret",
            "NVIDIA_PROXY_URL": "socks5://127.0.0.1:1080",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider = cast(Any, build_provider())
        nvidia = provider.providers[0]
        self.assertEqual(nvidia.proxy_url, "socks5://127.0.0.1:1080")

    def test_proxy_scope_is_reentrant_and_restores_socket(self) -> None:
        import socket

        provider = OpenAICompatibleProvider(name="socks", base_url="http://example.invalid", api_key="", model="m", proxy_url="socks5://127.0.0.1:1080")
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
            self.assertIn("ollama", stored["key_state"])

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

    def test_ladder_deadline_stops_further_strikes(self) -> None:
        class Slow:
            name = "slow"
            calls = 0

            def complete(self, *_args: object, **_kwargs: object) -> ModelTurn:
                self.calls += 1
                time.sleep(0.05)
                raise ProviderError("busy", category="server", retryable=True, cooldown_seconds=0.0)

        slow = Slow()
        fallback = FallbackProvider([slow], timeout_ladder=(1.0, 1.0, 1.0), ladder_deadline_seconds=0.01)
        with self.assertRaises(ProviderError):
            fallback.complete([], max_tokens=1)
        self.assertEqual(slow.calls, 1)

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
