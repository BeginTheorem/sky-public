from __future__ import annotations

import json
import os
from pathlib import Path

from .errors import ProviderError
from .fallback import FallbackProvider
from .ollama import OllamaCloudProvider
from .openai_compatible import OpenAICompatibleProvider
from .openrouter import OpenRouterProvider


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_provider() -> FallbackProvider:
    """Build the configured provider chain from environment variables.

    Every provider in the configured chain is constructed even when it is
    currently inactive, so the money-boost flag can be toggled at runtime: the
    fallback provider filters by `active_chain_names()` on each call instead of
    the process having to restart.
    """
    chain = _split(os.getenv("SKYNET_PROVIDER_CHAIN", DEFAULT_CHAIN)) or _split(DEFAULT_CHAIN)
    providers = []
    for name in chain:
        if name in {"ollama", "ollama_cloud"}:
            keys = _split(os.getenv("OLLAMA_API_KEYS", ""))
            legacy_key = os.getenv("SKYNET_LLM_API_KEY", "")
            if not keys and legacy_key:
                keys = [legacy_key]
            if not keys or not _provider_enabled("ollama", True):
                continue
            providers.append(OllamaCloudProvider(
                base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1"),
                keys=keys,
                model=os.getenv("OLLAMA_MODEL", os.getenv("SKYNET_LLM_MODEL", "gemma4:31b")),
                cooldown_seconds=float(os.getenv("OLLAMA_KEY_COOLDOWN_SECONDS", "60")),
                max_attempts=int(os.getenv("OLLAMA_MAX_ATTEMPTS", "3")),
                timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
                max_output_tokens=int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "8192")),
                network_cooldown_seconds=float(os.getenv("OLLAMA_NETWORK_COOLDOWN_SECONDS", "60")),
            ))
        elif name == "nvidia_deepseek":
            key = os.getenv("DEEPSEEK_API_KEY", os.getenv("NVIDIA_API_KEY", ""))
            if _provider_enabled("nvidia_deepseek", True) and key:
                providers.append(OpenAICompatibleProvider(
                    name="nvidia_deepseek",
                    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                    api_key=key,
                    model=os.getenv("DEEPSEEK_MODEL", "deepseek-ai/deepseek-v4-flash-0731"),
                    timeout_seconds=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120")),
                    max_output_tokens=int(os.getenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "8192")),
                    temperature=1.0,
                    top_p=0.95,
                    proxy_url=os.getenv("DEEPSEEK_PROXY_URL", os.getenv("NVIDIA_PROXY_URL", "")) or None,
                    cooldown_overrides={
                        429: float(os.getenv("DEEPSEEK_RATE_LIMIT_COOLDOWN_SECONDS", "300")),
                        500: float(os.getenv("DEEPSEEK_SERVER_COOLDOWN_SECONDS", "60")),
                        502: float(os.getenv("DEEPSEEK_SERVER_COOLDOWN_SECONDS", "60")),
                        503: float(os.getenv("DEEPSEEK_SERVER_COOLDOWN_SECONDS", "60")),
                        504: float(os.getenv("DEEPSEEK_SERVER_COOLDOWN_SECONDS", "60")),
                    },
                    network_cooldown_seconds=float(os.getenv("DEEPSEEK_NETWORK_COOLDOWN_SECONDS", "60")),
                ))
        elif name in {"nvidia", "nim"}:
            key = os.getenv("NVIDIA_API_KEY", "")
            if _provider_enabled(name, True) and key:
                proxy = os.getenv("NVIDIA_PROXY_URL", "")
                providers.append(OpenAICompatibleProvider(
                    name="nvidia",
                    base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                    api_key=key,
                    model=os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
                    timeout_seconds=float(os.getenv("NVIDIA_TIMEOUT_SECONDS", "120")),
                    max_output_tokens=int(os.getenv("NVIDIA_MAX_OUTPUT_TOKENS", "8192")),
                    temperature=1.0,
                    top_p=0.95,
                    reject_reasoning_leakage=True,
                    proxy_url=proxy or None,
                ))
        elif name in {"nemotron", "nvidia_nemotron"}:
            key = os.getenv("NVIDIA_API_KEY", "")
            if _provider_enabled("nemotron", True) and key:
                providers.append(OpenAICompatibleProvider(
                    name="nemotron", base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                    api_key=key, model=os.getenv("NEMOTRON_MODEL", os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")),
                    timeout_seconds=float(os.getenv("NEMOTRON_TIMEOUT_SECONDS", os.getenv("NVIDIA_TIMEOUT_SECONDS", "120"))),
                    max_output_tokens=int(os.getenv("NEMOTRON_MAX_OUTPUT_TOKENS", os.getenv("NVIDIA_MAX_OUTPUT_TOKENS", "8192"))),
                    temperature=1.0, top_p=0.95, reject_reasoning_leakage=True,
                    proxy_url=os.getenv("NEMOTRON_PROXY_URL", os.getenv("NVIDIA_PROXY_URL", "")) or None,
                ))
        elif name in {"openrouter"}:
            key = os.getenv("OPENROUTER_API_KEY", "")
            if _provider_enabled("openrouter", False) and key:
                providers.append(OpenRouterProvider(
                    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                    api_key=key,
                    model=os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash"),
                    timeout_seconds=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30")),
                    max_attempts=int(os.getenv("OPENROUTER_MAX_ATTEMPTS", "1")),
                    retry_delay_seconds=float(os.getenv("OPENROUTER_RETRY_DELAY_SECONDS", "1")),
                    # The idle timeout is the hang budget: it only fires on
                    # silence, so a long active reasoning stream is not cut off.
                    # The overall deadline is off unless an operator sets it.
                    chunk_timeout_seconds=float(os.getenv("OPENROUTER_CHUNK_TIMEOUT_SECONDS", "900")),
                    stream_deadline_seconds=_optional_seconds(os.getenv("OPENROUTER_STREAM_DEADLINE_SECONDS", "0")),
                ))
        elif name in {"openai", "generic"}:
            key = os.getenv("OPENAI_API_KEY", "")
            if key:
                proxy = os.getenv("OPENAI_PROXY_URL", "")
                providers.append(OpenAICompatibleProvider(
                    name="openai",
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    api_key=key,
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    proxy_url=proxy or None,
                ))
    if not providers:
        raise ProviderError("no provider is configured", category="configuration", retryable=False)
    state_path = os.getenv("SKYNET_PROVIDER_STATE", "")
    if not state_path and os.getenv("SKYNET_STATE"):
        state_path = str(Path(os.environ["SKYNET_STATE"]).with_name("provider-fallback.json"))
    return FallbackProvider(
        providers,
        max_attempts=int(os.getenv("SKYNET_PROVIDER_MAX_ATTEMPTS", str(len(providers)))),
        timeout_ladder=_timeout_ladder(os.getenv("SKYNET_PROVIDER_TIMEOUT_LADDER", "60,120,180,240,360")),
        strike_delay_seconds=float(os.getenv("SKYNET_PROVIDER_STRIKE_DELAY", "30")),
        state_path=Path(state_path) if state_path else None,
        ladder_deadline_seconds=float(os.getenv("SKYNET_PROVIDER_LADDER_DEADLINE", "600")),
        active_names=active_chain_names,
    )


def _timeout_ladder(value: str) -> tuple[float, ...]:
    ladder: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            seconds = float(item)
        except ValueError:
            continue
        if seconds > 0:
            ladder.append(seconds)
    return tuple(ladder)


def _optional_seconds(value: str) -> float | None:
    text = value.strip()
    if not text or text in {"0", "none", "off"}:
        return None
    try:
        seconds = float(text)
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def _enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


# Canonical enable flag per provider, shared by build_provider() and
# active_chain_names(). Legacy aliases are accepted so an existing
# config/skynet.env keeps working; the canonical (first) name wins. Before this
# map the constructor read DEEPSEEK_ENABLED / NVIDIA_ENABLED while the
# hot-reload filter read NVIDIA_DEEPSEEK_ENABLED / NEMOTRON_ENABLED, so toggling
# the documented knob at runtime did nothing.
_ENABLE_FLAGS: dict[str, tuple[str, ...]] = {
    "ollama": ("OLLAMA_ENABLED",),
    "ollama_cloud": ("OLLAMA_ENABLED",),
    "nvidia_deepseek": ("NVIDIA_DEEPSEEK_ENABLED", "DEEPSEEK_ENABLED"),
    "nvidia": ("NVIDIA_ENABLED",),
    "nim": ("NVIDIA_ENABLED",),
    "nemotron": ("NEMOTRON_ENABLED", "NVIDIA_ENABLED"),
    "nvidia_nemotron": ("NEMOTRON_ENABLED", "NVIDIA_ENABLED"),
    "openrouter": ("OPENROUTER_ENABLED",),
    "openai": ("OPENAI_ENABLED",),
    "generic": ("OPENAI_ENABLED",),
}


def _provider_enabled(name: str, default: bool) -> bool:
    """Whether the named provider is enabled, canonical flag first."""
    for flag in _ENABLE_FLAGS.get(name, (f"{name.upper()}_ENABLED",)):
        value = os.getenv(flag)
        if value is not None:
            return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


# Provider's default chain. Every implementation stays in place, not deleted:
# enabling or disabling one is an environment decision (`*_ENABLED=false`,
# `SKYNET_PROVIDER_CHAIN=...`), so the chain changes without a code change.
DEFAULT_CHAIN = "openrouter,nvidia_deepseek,nemotron,ollama"


def active_chain_names() -> list[str]:
    """The provider names the configuration currently wants, read fresh.

    Used by the fallback provider to hot-reload the paid provider without a
    service restart: the chain is a property of the environment plus the
    money-boost flag, not of the process start.
    """
    chain = _split(os.getenv("SKYNET_PROVIDER_CHAIN", DEFAULT_CHAIN)) or _split(DEFAULT_CHAIN)
    if not _money_boost_enabled():
        chain = [name for name in chain if name not in {"openrouter"}]
    return [name for name in chain if _provider_enabled(name, True)]


def money_boost_state_path() -> Path:
    configured = os.getenv("SKYNET_MONEY_BOOST_STATE", "")
    if configured:
        return Path(configured)
    state = os.getenv("SKYNET_STATE", "state/skynet.sqlite3")
    return Path(state).with_name("money-boost.json")


def _money_boost_enabled() -> bool:
    path = money_boost_state_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return bool(value.get("enabled", False))
    except (OSError, TypeError, ValueError):
        return _enabled("SKYNET_MONEY_BOOST", False)


__all__ = ["FallbackProvider", "OllamaCloudProvider", "OpenAICompatibleProvider", "OpenRouterProvider", "ProviderError", "active_chain_names", "build_provider", "money_boost_state_path"]
