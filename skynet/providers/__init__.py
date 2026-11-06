from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .errors import ProviderError
from .fallback import FallbackProvider
from .ollama import OllamaCloudProvider
from .openai_compatible import OpenAICompatibleProvider
from .openrouter import OpenRouterProvider


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


# Canonical provider names. Each rung is one (endpoint, model) pair, so the
# name states which model sits behind it and a cooldown on one model never
# benches another that shares the endpoint (FallbackProvider keys cooldowns by
# provider name). Legacy aliases stay accepted so an older config/skynet.env
# keeps working; the canonical name is what the chain is normalised to.
PROVIDER_ALIASES: dict[str, str] = {
    "openrouter_deepseek": "openrouter_deepseek",
    "openrouter": "openrouter_deepseek",
    "open_router": "openrouter_deepseek",
    "openrouter_glm": "openrouter_glm",
    "open_router_glm": "openrouter_glm",
    "nvidia_deepseek": "nvidia_deepseek",
    # The owner's rotation names the four NIM rungs by model: deepseek, super,
    # ultra, lightning (2026-10-17, when NVIDIA moved the lineup to
    # nemotron-3.5-super-vl / nemotron-3.5-lightning). The long
    # ``nvidia_nemotron_*`` names stay accepted so an existing config, test or
    # parked snapshot keeps resolving; the short name is what the chain carries.
    "nvidia_super": "nvidia_super",
    "nvidia_nemotron_super": "nvidia_super",
    "nemotron": "nvidia_super",
    "nvidia_nemotron": "nvidia_super",
    "nemotron_super": "nvidia_super",
    "nvidia_ultra": "nvidia_ultra",
    "nvidia_nemotron_ultra": "nvidia_ultra",
    "nemotron_ultra": "nvidia_ultra",
    "nvidia_lightning": "nvidia_lightning",
    "nvidia_nemotron_lightning": "nvidia_lightning",
    "nemotron_lightning": "nvidia_lightning",
    "nemotron_3_5_lightning": "nvidia_lightning",
    # The nano rung is the cheap fast one: 30B A3B, reasoning, measured 5/5 in
    # 1.0s median with 75 output tokens on the probe prompt (2026-11-02), against
    # 1.6s/76 for ultra and 1.0s/82 for the super-vl head it was measured beside.
    # It earns a rung of its own for the same reason as the others: a cooldown is
    # keyed by provider name, so sharing a rung with super would bench both.
    "nvidia_nano": "nvidia_nano",
    "nvidia_nemotron_nano": "nvidia_nano",
    "nemotron_nano": "nvidia_nano",
    "nemotron_3_nano": "nvidia_nano",
    "ollama_gemma": "ollama_gemma",
    "ollama": "ollama_gemma",
    "ollama_cloud": "ollama_gemma",
    "zai_glm47_flash": "zai_glm47_flash",
    "zai": "zai_glm47_flash",
    "zai_glm45_flash": "zai_glm45_flash",
    "zai_glm45": "zai_glm45_flash",
    "mistral_leanstral": "mistral_leanstral",
    "leanstral": "mistral_leanstral",
    "mistral_codestral": "mistral_codestral",
    "codestral": "mistral_codestral",
    "muse_glimmer": "muse_glimmer",
    "muse": "muse_glimmer",
    "glimmer": "muse_glimmer",
    "muse_glimmer_30b": "muse_glimmer",
}


def canonical_provider_name(name: str) -> str:
    """Normalise a chain token to its canonical provider name (or pass through)."""
    return PROVIDER_ALIASES.get(name, name)


def build_provider() -> FallbackProvider:
    """Build the configured provider chain from environment variables.

    Every provider in the configured chain is constructed even when it is
    currently inactive, so the money-boost flag can be toggled at runtime: the
    fallback provider filters by `active_chain_names()` on each call instead of
    the process having to restart.
    """
    chain = _split(os.getenv("SKYNET_PROVIDER_CHAIN", DEFAULT_CHAIN)) or _split(DEFAULT_CHAIN)
    providers = []
    for raw_name in chain:
        name = canonical_provider_name(raw_name)
        if name == "ollama_gemma":
            keys = _split(os.getenv("OLLAMA_API_KEYS", ""))
            legacy_key = os.getenv("SKYNET_LLM_API_KEY", "")
            if not keys and legacy_key:
                keys = [legacy_key]
            if not keys or not _provider_enabled("ollama_gemma", True):
                continue
            providers.append(OllamaCloudProvider(
                name="ollama_gemma",
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
                    # The rung is named for the model it is configured to carry.
                    # NIM lists deepseek-ai/deepseek-v4.1-flash (the renamed id)
                    # but serves no bytes on either the streaming or the
                    # non-streaming path (measured 2026-09-26: idle past 60s,
                    # connection dropped), so this rung is normally the one
                    # disabled in a live env.
                    model=os.getenv("DEEPSEEK_MODEL", "deepseek-ai/deepseek-v4.1-flash"),
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
        elif name == "nvidia_ultra":
            key = os.getenv("NVIDIA_API_KEY", "")
            if _provider_enabled("nvidia_ultra", True) and key:
                providers.append(OpenAICompatibleProvider(
                    name="nvidia_ultra",
                    base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                    api_key=key,
                    model=os.getenv("NEMOTRON_ULTRA_MODEL", os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")),
                    timeout_seconds=float(os.getenv("NEMOTRON_ULTRA_TIMEOUT_SECONDS", os.getenv("NVIDIA_TIMEOUT_SECONDS", "120"))),
                    max_output_tokens=int(os.getenv("NEMOTRON_ULTRA_MAX_OUTPUT_TOKENS", os.getenv("NVIDIA_MAX_OUTPUT_TOKENS", "8192"))),
                    temperature=1.0,
                    top_p=0.95,
                    proxy_url=os.getenv("NEMOTRON_ULTRA_PROXY_URL", os.getenv("NVIDIA_PROXY_URL", "")) or None,
                ))
        elif name == "nvidia_super":
            key = os.getenv("NVIDIA_API_KEY", "")
            if _provider_enabled("nvidia_super", True) and key:
                providers.append(OpenAICompatibleProvider(
                    name="nvidia_super",
                    base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                    api_key=key,
                    # NIM renamed the rung's model on 2026-10-17: the free catalog
                    # now lists ``nemotron-3.5-super-vl-120b-a12b`` (vision-language,
                    # 120B A12B) and no longer carries the 3-super id this rung was
                    # built against. Confirmed against ``GET /v1/models`` plus three
                    # live completions before the switch.
                    model=os.getenv("NEMOTRON_SUPER_MODEL", os.getenv("NEMOTRON_MODEL", os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b"))),
                    timeout_seconds=float(os.getenv("NEMOTRON_SUPER_TIMEOUT_SECONDS", os.getenv("NEMOTRON_TIMEOUT_SECONDS", os.getenv("NVIDIA_TIMEOUT_SECONDS", "120")))),
                    max_output_tokens=int(os.getenv("NEMOTRON_SUPER_MAX_OUTPUT_TOKENS", os.getenv("NEMOTRON_MAX_OUTPUT_TOKENS", os.getenv("NVIDIA_MAX_OUTPUT_TOKENS", "8192")))),
                    # No leak guard: nemotron returns reasoning in ``content`` on
                    # the no-tools planner path, and the guard turned a parseable
                    # reply into ``response_quality`` provider errors on the live
                    # switch. The ReAct Finish-Report parser already refuses prose,
                    # so the provider-level guard only hurts.
                    temperature=1.0,
                    top_p=0.95,
                    proxy_url=os.getenv("NEMOTRON_SUPER_PROXY_URL", os.getenv("NEMOTRON_PROXY_URL", os.getenv("NVIDIA_PROXY_URL", ""))) or None,
                ))
        elif name == "nvidia_lightning":
            key = os.getenv("NVIDIA_API_KEY", "")
            if _provider_enabled("nvidia_lightning", True) and key:
                providers.append(OpenAICompatibleProvider(
                    name="nvidia_lightning",
                    base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                    api_key=key,
                    # 30B A3B Mamba-2/MoE hybrid: measured 0.7-2.9s on plain text
                    # and 1M-token context, so it is the fast NIM rung. It is a
                    # reasoning model (``reasoning_content``), and on a tools turn
                    # it can spend the whole output budget thinking before it
                    # emits the call (measured once in three: 1024 tokens,
                    # finish_reason=length, no call). That is the retryable
                    # empty-length case the response-quality guard already
                    # handles; the tricky part is only the budget sizing.
                    model=os.getenv("NEMOTRON_LIGHTNING_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"),
                    timeout_seconds=float(os.getenv("NEMOTRON_LIGHTNING_TIMEOUT_SECONDS", os.getenv("NVIDIA_TIMEOUT_SECONDS", "120"))),
                    max_output_tokens=int(os.getenv("NEMOTRON_LIGHTNING_MAX_OUTPUT_TOKENS", os.getenv("NVIDIA_MAX_OUTPUT_TOKENS", "8192"))),
                    temperature=1.0,
                    top_p=0.95,
                    proxy_url=os.getenv("NEMOTRON_LIGHTNING_PROXY_URL", os.getenv("NVIDIA_PROXY_URL", "")) or None,
                ))
        elif name == "nvidia_nano":
            key = os.getenv("NVIDIA_API_KEY", "")
            if _provider_enabled("nvidia_nano", True) and key:
                providers.append(OpenAICompatibleProvider(
                    name="nvidia_nano",
                    base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                    api_key=key,
                    # 30B A3B omni reasoning model, and the cheapest fast rung NIM
                    # offers: measured 5/5 answers, 1.0s median, 75 output tokens
                    # (2026-11-02, interleaved against the rungs this chain already
                    # carries). Its budget is what matters -- like every nemotron
                    # here it spends tokens thinking first, and a tight ceiling
                    # turns a good answer into the response-quality failure class.
                    model=os.getenv("NEMOTRON_NANO_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),
                    timeout_seconds=float(os.getenv("NEMOTRON_NANO_TIMEOUT_SECONDS", os.getenv("NVIDIA_TIMEOUT_SECONDS", "120"))),
                    max_output_tokens=int(os.getenv("NEMOTRON_NANO_MAX_OUTPUT_TOKENS", os.getenv("NVIDIA_MAX_OUTPUT_TOKENS", "8192"))),
                    temperature=1.0,
                    top_p=0.95,
                    proxy_url=os.getenv("NEMOTRON_NANO_PROXY_URL", os.getenv("NVIDIA_PROXY_URL", "")) or None,
                ))
        elif name == "zai_glm47_flash":
            key = os.getenv("ZAI_API_KEY", "")
            if _provider_enabled("zai_glm47_flash", True) and key:
                providers.append(OpenAICompatibleProvider(
                    name="zai_glm47_flash",
                    base_url=os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4"),
                    api_key=key,
                    model=os.getenv("ZAI_MODEL", "glm-4.7-flash"),
                    # Free-tier latency measured up to ~167s on a ~190k-token
                    # prompt, so the rung's own budget is generous. When a
                    # timeout ladder is configured, its rung still governs the
                    # ladder strikes.
                    timeout_seconds=float(os.getenv("ZAI_TIMEOUT_SECONDS", "300")),
                    proxy_url=os.getenv("ZAI_PROXY_URL", "") or None,
                    # 429 is Z.AI code 1305 ("service may be temporarily
                    # overloaded"): a measured burst was blocked for ~50-70s, so
                    # the rung brakes longer than the 60s default.
                    cooldown_overrides={429: float(os.getenv("ZAI_RATE_LIMIT_COOLDOWN_SECONDS", "120"))},
                    extra_body=_zai_extra_body(),
                ))
        elif name == "zai_glm45_flash":
            key = os.getenv("ZAI_API_KEY", "")
            if _provider_enabled("zai_glm45_flash", True) and key:
                providers.append(OpenAICompatibleProvider(
                    name="zai_glm45_flash",
                    base_url=os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4"),
                    api_key=key,
                    model=os.getenv("ZAI_GLM45_MODEL", "glm-4.5-flash"),
                    timeout_seconds=float(os.getenv("ZAI_TIMEOUT_SECONDS", "300")),
                    proxy_url=os.getenv("ZAI_PROXY_URL", "") or None,
                    cooldown_overrides={429: float(os.getenv("ZAI_RATE_LIMIT_COOLDOWN_SECONDS", "120"))},
                    extra_body=_zai_extra_body(),
                ))
        elif name == "mistral_leanstral":
            key = os.getenv("MISTRAL_API_KEY", "")
            if _provider_enabled("mistral_leanstral", True) and key:
                providers.append(OpenAICompatibleProvider(
                    name="mistral_leanstral",
                    base_url=os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
                    api_key=key,
                    model=os.getenv("MISTRAL_LEANSTRAL_MODEL", "labs-leanstral-1-5"),
                    timeout_seconds=float(os.getenv("MISTRAL_TIMEOUT_SECONDS", "300")),
                    # No temperature/top_p and no output clamp. The model rejects a
                    # sampling temperature ("top_p must be 1 when using green mode",
                    # measured 2026-09-27), and the free tier served 13893
                    # completion tokens with no ceiling error, so the reactor's own
                    # SKYNET_OUTPUT_TOKENS budget flows through unclamped.
                    cooldown_overrides={429: float(os.getenv("MISTRAL_RATE_LIMIT_COOLDOWN_SECONDS", "120"))},
                ))
        elif name == "mistral_codestral":
            key = os.getenv("MISTRAL_API_KEY", "")
            if _provider_enabled("mistral_codestral", True) and key:
                providers.append(OpenAICompatibleProvider(
                    name="mistral_codestral",
                    base_url=os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
                    api_key=key,
                    # ``mistral-code-latest`` is the same model as
                    # ``codestral-latest`` (both ids resolve to codestral-2508),
                    # so one rung carries it instead of a duplicate whose shared
                    # per-model quota would 429 immediately.
                    model=os.getenv("MISTRAL_CODESTRAL_MODEL", "codestral-latest"),
                    timeout_seconds=float(os.getenv("MISTRAL_TIMEOUT_SECONDS", "300")),
                    cooldown_overrides={429: float(os.getenv("MISTRAL_RATE_LIMIT_COOLDOWN_SECONDS", "120"))},
                ))
        elif name == "muse_glimmer":
            key = os.getenv("NVIDIA_API_KEY", "")
            if _provider_enabled("muse_glimmer", True) and key:
                providers.append(OpenAICompatibleProvider(
                    name="muse_glimmer",
                    base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                    api_key=key,
                    # 29.6B dense multimodal model from Meta. We do not send
                    # images; it is here for its text reasoning and tool calling
                    # (measured: clean ``finish_reason=tool_calls`` in 5.7s, and
                    # a separate ``reasoning_content`` field). It is the slowest
                    # of the NIM rungs, so it sits at the tail of the chain.
                    model=os.getenv("MUSE_GLIMMER_MODEL", "meta/muse-glimmer-30b"),
                    timeout_seconds=float(os.getenv("MUSE_GLIMMER_TIMEOUT_SECONDS", os.getenv("NVIDIA_TIMEOUT_SECONDS", "120"))),
                    max_output_tokens=int(os.getenv("MUSE_GLIMMER_MAX_OUTPUT_TOKENS", os.getenv("NVIDIA_MAX_OUTPUT_TOKENS", "8192"))),
                    temperature=1.0,
                    top_p=0.95,
                    proxy_url=os.getenv("MUSE_GLIMMER_PROXY_URL", os.getenv("NVIDIA_PROXY_URL", "")) or None,
                ))
        elif name == "openrouter_deepseek":
            key = os.getenv("OPENROUTER_API_KEY", "")
            if _provider_enabled("openrouter_deepseek", False) and key:
                providers.append(OpenRouterProvider(
                    name="openrouter_deepseek",
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
        elif name == "openrouter_glm":
            # A second rung on the same openrouter endpoint: openrouter_deepseek
            # is the primary model, this one is the backup. A distinct ``name``
            # keeps their cooldowns independent, so a bad DeepSeek window does
            # not bench the backup.
            key = os.getenv("OPENROUTER_API_KEY", "")
            if _provider_enabled("openrouter_glm", False) and key:
                providers.append(OpenRouterProvider(
                    name="openrouter_glm",
                    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                    api_key=key,
                    model=os.getenv("OPENROUTER_GLM_MODEL", "z-ai/glm-5.3-flash"),
                    timeout_seconds=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30")),
                    max_attempts=int(os.getenv("OPENROUTER_MAX_ATTEMPTS", "1")),
                    retry_delay_seconds=float(os.getenv("OPENROUTER_RETRY_DELAY_SECONDS", "1")),
                    chunk_timeout_seconds=float(os.getenv("OPENROUTER_CHUNK_TIMEOUT_SECONDS", "900")),
                    stream_deadline_seconds=_optional_seconds(os.getenv("OPENROUTER_STREAM_DEADLINE_SECONDS", "0")),
                ))
    if not providers:
        raise ProviderError("no provider is configured", category="configuration", retryable=False)
    state_path = os.getenv("SKYNET_PROVIDER_STATE", "")
    if not state_path and os.getenv("SKYNET_STATE"):
        state_path = str(Path(os.environ["SKYNET_STATE"]).with_name("provider-fallback.json"))
    return FallbackProvider(
        providers,
        max_attempts=int(os.getenv("SKYNET_PROVIDER_MAX_ATTEMPTS", str(len(providers)))),
        # The patient cycle (owner decision 2026-09-27): per-provider grouped
        # strikes with escalating waits, a 5-minute lock after the last failure,
        # and an in-place re-pass of a dead chain. Only the ReAct path opts in
        # (``complete(..., patient=True)``); the memory loop and the planner keep
        # the single-strike behaviour their own short budgets fit.
        attempts_per_provider=int(os.getenv("SKYNET_PROVIDER_ATTEMPTS_PER_PROVIDER", "4")),
        attempt_waits=_seconds_tuple(os.getenv("SKYNET_PROVIDER_ATTEMPT_WAITS", "5,10,15")),
        provider_block_seconds=float(os.getenv("SKYNET_PROVIDER_BLOCK_SECONDS", "300")),
        dead_passes=int(os.getenv("SKYNET_PROVIDER_DEAD_PASSES", "3")),
        # A non-streaming rung can only be bounded by the whole wait (it thinks
        # silently and then sends everything at once); a streaming rung is
        # bounded by silence between chunks.
        attempt_timeout_seconds=float(os.getenv("SKYNET_PROVIDER_ATTEMPT_TIMEOUT", "900")),
        stream_idle_seconds=float(os.getenv("SKYNET_PROVIDER_STREAM_IDLE", "120")),
        # Hard cap for one patient call: the 3-dead-pass rule ends it far
        # earlier in practice, but a silently hanging chain must not hold the
        # organism forever.
        call_deadline_seconds=float(os.getenv("SKYNET_PROVIDER_CALL_DEADLINE", "14400")),
        state_path=Path(state_path) if state_path else None,
        # A zero-strike chain-wide cooldown whose earliest horizon is this short
        # is waited out once per NON-patient call instead of discarding the
        # episode with zero strikes (measured: 5 of 21 needs_recovery runs,
        # 12.2-30.0s horizons). "0" restores the raise-immediately behaviour.
        max_wait_for_cooldown_seconds=float(os.getenv("SKYNET_PROVIDER_COOLDOWN_WAIT", "30")),
        active_names=active_chain_names,
    )


def _seconds_tuple(value: str) -> tuple[float, ...]:
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
# config/skynet.env keeps working; the canonical (first) name wins.
_ENABLE_FLAGS: dict[str, tuple[str, ...]] = {
    "ollama_gemma": ("OLLAMA_GEMMA_ENABLED", "OLLAMA_ENABLED"),
    "nvidia_deepseek": ("NVIDIA_DEEPSEEK_ENABLED", "DEEPSEEK_ENABLED"),
    "nvidia_ultra": ("NEMOTRON_ULTRA_ENABLED", "NVIDIA_ENABLED"),
    "nvidia_super": ("NEMOTRON_SUPER_ENABLED", "NEMOTRON_ENABLED", "NVIDIA_ENABLED"),
    "nvidia_lightning": ("NEMOTRON_LIGHTNING_ENABLED", "NVIDIA_ENABLED"),
    "nvidia_nano": ("NEMOTRON_NANO_ENABLED", "NVIDIA_ENABLED"),
    "openrouter_deepseek": ("OPENROUTER_ENABLED",),
    # The backup rung gets its own canonical flag ahead of the shared one, so a
    # model the key does not carry can be parked without benching the primary on
    # the same endpoint. Measured 2026-10-16: openrouter's catalog holds exactly
    # one model (`deepseek/deepseek-v4.1-flash`), and a request for
    # `z-ai/glm-5.3-flash` answers HTTP 403, which classifies as `auth` and
    # `fallback._is_fatal` treats as chain-fatal -- an active mis-keyed rung would
    # kill every call instead of costing one attempt.
    "openrouter_glm": ("OPENROUTER_GLM_ENABLED", "OPENROUTER_ENABLED"),
    "zai_glm47_flash": ("ZAI_ENABLED",),
    "zai_glm45_flash": ("ZAI_ENABLED",),
    "mistral_leanstral": ("MISTRAL_ENABLED",),
    "mistral_codestral": ("MISTRAL_ENABLED",),
    "muse_glimmer": ("MUSE_GLIMMER_ENABLED", "NVIDIA_ENABLED"),
}

# Providers that are off unless OPENROUTER_ENABLED turns them on.
_DEFAULT_ENABLED: dict[str, bool] = {"openrouter_deepseek": False, "openrouter_glm": False}


def _provider_enabled(name: str, default: bool) -> bool:
    """Whether the named provider is enabled, canonical flag first."""
    for flag in _ENABLE_FLAGS.get(name, (f"{name.upper()}_ENABLED",)):
        value = os.getenv(flag)
        if value is not None:
            return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _zai_extra_body() -> dict[str, object]:
    """Body fields for the Z.AI rungs.

    GLM thinking is on by default and is billed to the same output budget: on a
    no-tools turn (Finish report, planner) a spent budget leaves ``content``
    empty with ``finish_reason=length``, which the shared response-quality guard
    refuses and retries (measured on the live endpoint). Disabled by default;
    ``ZAI_THINKING=true`` restores the endpoint default by sending no field.
    """
    if _enabled("ZAI_THINKING", False):
        return {}
    return {"thinking": {"type": "disabled"}}


# The full provider archive in chain order. The other implementations are kept
# in place, not deleted -- disabling is an environment decision
# (`*_ENABLED=false`, a different `SKYNET_PROVIDER_CHAIN`) so they can be
# re-enabled without a code change.
DEFAULT_CHAIN = "openrouter_deepseek,openrouter_glm,nvidia_deepseek,nvidia_super,nvidia_ultra,nvidia_lightning,zai_glm47_flash,zai_glm45_flash,mistral_leanstral,mistral_codestral,muse_glimmer,ollama_gemma"

# Providers that are only active while money-boost is on: the paid openrouter
# rungs. The backup-model rung is gated exactly like the primary one.
PAID_PROVIDER_NAMES = frozenset({"openrouter_deepseek", "openrouter_glm"})

# The owner's own switches, one per rung, plus the context ceiling that follows
# from a rung's class. Two files on purpose (owner decision 2026-10-16): the bot
# writes the DESIRED state, the reactor copies it into the APPLIED state at the
# next tick, because the owner's rule is that a switch lands only at the next
# heartbeat -- a tap must never reshape the ladder under a run already in flight. Both
# files absent means the plain environment chain, so a fresh deploy needs no bootstrap.
PROVIDER_PREFS_FILE = "provider-prefs.json"
PROVIDER_APPLIED_FILE = "provider-applied.json"
# Free rungs are cheap and short-windowed, paid rungs are the long-context ones,
# so the window is a property of the class and not of the caller. Tokens are what
# the ReAct window counts, characters what the planner's payload counts: each knob
# gets the same two numbers in its own unit (700k chars is the planner's paid
# default this replaces).
FREE_CONTEXT_CEILING_TOKENS = 100_000
PAID_CONTEXT_CEILING_TOKENS = 700_000
FREE_CONTEXT_CEILING_CHARS = 100_000
PAID_CONTEXT_CEILING_CHARS = 700_000


def _provider_state_path(filename: str) -> Path:
    state = os.getenv("SKYNET_STATE", "state/skynet.sqlite3")
    return Path(state).with_name(filename)


def provider_prefs_path() -> Path:
    """Where the owner's DESIRED switches live -- the bot's write target."""
    configured = os.getenv("SKYNET_PROVIDER_PREFS", "")
    return Path(configured) if configured else _provider_state_path(PROVIDER_PREFS_FILE)


def provider_applied_path() -> Path:
    """Where the tick parks what it actually applied -- the chain's source."""
    configured = os.getenv("SKYNET_PROVIDER_APPLIED", "")
    return Path(configured) if configured else _provider_state_path(PROVIDER_APPLIED_FILE)


def _read_provider_file(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def provider_class(name: str) -> str:
    """``paid`` or ``free``: the class is what decides the context ceiling."""
    return "paid" if canonical_provider_name(name) in PAID_PROVIDER_NAMES else "free"


def context_ceiling_tokens(name: str) -> int:
    """Per-request ReAct window for this rung's class, in tokens.

    Env-tunable on purpose (``SKYNET_PAID_CONTEXT_TOKENS`` /
    ``SKYNET_FREE_CONTEXT_TOKENS``): the two numbers the owner named are a
    starting point, and a rung whose model serves a smaller window is corrected
    with a restart rather than with a patch.
    """
    paid = provider_class(name) == "paid"
    return _positive_int(
        "SKYNET_PAID_CONTEXT_TOKENS" if paid else "SKYNET_FREE_CONTEXT_TOKENS",
        PAID_CONTEXT_CEILING_TOKENS if paid else FREE_CONTEXT_CEILING_TOKENS,
    )


def context_ceiling_chars(name: str) -> int:
    """Planner payload cap for this rung's class, in characters (env-tunable)."""
    paid = provider_class(name) == "paid"
    return _positive_int(
        "SKYNET_PAID_CONTEXT_CHARS" if paid else "SKYNET_FREE_CONTEXT_CHARS",
        PAID_CONTEXT_CEILING_CHARS if paid else FREE_CONTEXT_CEILING_CHARS,
    )


def _positive_int(flag: str, default: int) -> int:
    """An env override when it parses as a positive integer, else the default."""
    try:
        value = int(str(os.getenv(flag, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def load_provider_prefs() -> dict[str, list[str]]:
    """The owner's desired switches as ``{"off": [names]}``, read tolerantly.

    A missing or unreadable file is not an error: it means "no owner switch", so
    the chain is exactly what the environment says.
    """
    raw = _read_provider_file(provider_prefs_path()).get("off")
    names = [canonical_provider_name(str(item)) for item in raw] if isinstance(raw, list) else []
    return {"off": sorted(set(names))}


def set_provider_off(name: str, off: bool) -> dict[str, object]:
    """Flip one rung's owner switch; return the DESIRED state it wrote.

    Deliberately desired-only: nothing about the running chain changes here. The
    reactor applies it at the next tick, which is the whole point of the two files.
    """
    canonical = canonical_provider_name(name)
    disabled = {str(item) for item in load_provider_prefs()["off"]}
    disabled.add(canonical) if off else disabled.discard(canonical)
    written: dict[str, object] = {"off": sorted(disabled), "updated_at": time.time()}
    path = provider_prefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(written, ensure_ascii=False, indent=2), encoding="utf-8")
    return written


def configured_provider_names() -> list[str]:
    """Every rung the environment names, in chain order and unfiltered.

    The bot's list has to show what the environment DECLARES: a rung parked by its
    ``*_ENABLED`` flag, or switched off from the phone, is an owner decision to
    display, not a rung to hide. ``active_chain_names`` answers the narrower
    question of what the chain is answering with.
    """
    return _configured_chain()


def _configured_chain() -> list[str]:
    chain = _split(os.getenv("SKYNET_PROVIDER_CHAIN", DEFAULT_CHAIN)) or _split(DEFAULT_CHAIN)
    return [canonical_provider_name(item) for item in chain]


def _effective_chain_names() -> list[str]:
    """Env order minus the owner's switches, the money boost and the off flags."""
    names = _configured_chain()
    disabled = {str(item) for item in load_provider_prefs()["off"]}
    if disabled:
        names = [name for name in names if name not in disabled]
    if not _money_boost_enabled():
        names = [name for name in names if name not in PAID_PROVIDER_NAMES]
    return [name for name in names if _provider_enabled(name, _DEFAULT_ENABLED.get(name, True))]


def apply_provider_prefs() -> tuple[list[str], bool]:
    """Copy the desired switches into the applied state; ``(names, changed)``.

    Called by the reactor at the top of a tick -- the heartbeat boundary the owner
    asked for. ``changed`` is True only when the effective list actually moved, so
    the caller can event it once instead of on every tick.

    With **no desired-state file** nothing is written and nothing is pinned: the
    applied snapshot exists to hold a decision the owner made, and until he makes
    one the chain is exactly the environment's, read live as it always was. That
    also keeps a fresh deploy byte-identical to the old behaviour instead of
    freezing a snapshot nobody asked for.
    """
    effective = _effective_chain_names()
    path = provider_applied_path()
    if not provider_prefs_path().exists():
        return effective, False
    previous = _read_provider_file(path).get("providers")
    was = [str(item) for item in previous] if isinstance(previous, list) else None
    if was != effective:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"providers": effective, "updated_at": time.time()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return effective, was != effective


def applied_chain_names() -> list[str] | None:
    """The chain the last tick applied, or None when no tick has run yet."""
    value = _read_provider_file(provider_applied_path()).get("providers")
    return [str(item) for item in value] if isinstance(value, list) else None


def active_chain_names() -> list[str]:
    """The provider names the configuration currently wants, read fresh.

    Used by the fallback provider to hot-reload the chain without a service
    restart: the chain is a property of the environment, the money-boost flag and
    the owner's switches -- not of the process start. Tokens are normalised to
    their canonical provider names so a legacy alias never fails to match the
    built provider.

    The owner's switches enter through the APPLIED file rather than the desired
    one: a tap in the bot changes nothing until the next tick copies it across,
    while an ``*_ENABLED`` flag still bites at once (a rung whose flag went false
    must stop answering immediately). With no applied file yet the chain is
    exactly the environment's, so a fresh deploy behaves as before.
    """
    names = _configured_chain()
    applied = applied_chain_names()
    if applied is not None:
        allowed = set(applied)
        names = [name for name in names if name in allowed]
    if not _money_boost_enabled():
        names = [name for name in names if name not in PAID_PROVIDER_NAMES]
    return [name for name in names if _provider_enabled(name, _DEFAULT_ENABLED.get(name, True))]


def money_boost_enabled() -> bool:
    """Whether the paid rungs are on: the state file first, then the env var.

    Public because two readers must agree: the chain itself and the card the owner
    reads when he decides which rung to switch. A card that consulted only the file
    would call a paid chain "money boost: off" on a fresh deploy.
    """
    return _money_boost_enabled()


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


__all__ = [
    "FREE_CONTEXT_CEILING_CHARS",
    "FREE_CONTEXT_CEILING_TOKENS",
    "PAID_CONTEXT_CEILING_CHARS",
    "PAID_CONTEXT_CEILING_TOKENS",
    "PAID_PROVIDER_NAMES",
    "PROVIDER_ALIASES",
    "PROVIDER_APPLIED_FILE",
    "PROVIDER_PREFS_FILE",
    "FallbackProvider",
    "OllamaCloudProvider",
    "OpenAICompatibleProvider",
    "OpenRouterProvider",
    "ProviderError",
    "active_chain_names",
    "applied_chain_names",
    "apply_provider_prefs",
    "build_provider",
    "canonical_provider_name",
    "configured_provider_names",
    "context_ceiling_chars",
    "context_ceiling_tokens",
    "load_provider_prefs",
    "money_boost_enabled",
    "money_boost_state_path",
    "provider_applied_path",
    "provider_class",
    "provider_prefs_path",
    "set_provider_off",
]
