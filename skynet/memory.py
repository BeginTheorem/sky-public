from __future__ import annotations

import hashlib
import json
import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model_contracts import MEMORY_RESPONSE_SCHEMA, parse_json_object, validate_shape
from .models import AgentRunResult, Budget, ModelTurn
from .provider import LLMProvider, Message
from .runtime_log import verbose_enabled, verbose_write

MEMORY_LOOP_INSTRUCTION = (
    "Read the completed episode and extract durable strategic observations. You may suggest evidence and planner hints, but "
    "the deterministic harness owns portfolio ranking and task selection. "
    "Return JSON only with memory_candidates, next_plan, initial_prompt, goal_updates, task_updates, and evaluation. "
    "When new evidence contradicts an existing memory, emit a candidate with supersedes_memory_id set to the contradicted "
    "memory's id and put the contradicting evidence in evidence; never supersede on a guess. "
    "Goal/task updates must reference existing IDs and include evidence in outcome when completing work. "
    "Do not include reasoning. Do not resurrect completed, blocked, exhausted, or reset planning context."
)


@dataclass(slots=True)
class MemoryResult:
    """Bounded output of the independent post-Run memory pass."""

    memory_candidates: list[dict[str, Any]] = field(default_factory=list)
    next_plan: dict[str, Any] = field(default_factory=dict)
    initial_prompt: str = ""
    goal_updates: list[dict[str, Any]] = field(default_factory=list)
    task_updates: list[dict[str, Any]] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=dict)


def _runtime_log_path(runtime_log: Callable[..., None] | None) -> Path | None:
    """Best-effort recovery of the store's runtime-log path from the callback.

    The reactor passes a lambda wrapping ``store.runtime_log.write``; the path
    is reachable through its closure. A bound ``RuntimeLog.write`` exposes
    ``__self__.path`` directly.
    """
    owner = getattr(runtime_log, "__self__", None)
    path = getattr(owner, "path", None)
    if path is not None:
        return Path(path)
    for cell in getattr(runtime_log, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        candidate = getattr(getattr(value, "store", None), "runtime_log", None)
        candidate_path = getattr(candidate, "path", None)
        if candidate_path is not None:
            return Path(candidate_path)
    return None


class MemoryLoop:
    """Compress one finished episode without exposing model reasoning traces."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        budget: Budget | None = None,
        timeout_seconds: float = 180.0,
        episode_char_budget: int = 150_000,
    ) -> None:
        self.provider = provider
        self.budget = budget or Budget(steps=1, tokens=50_000, seconds=180.0, output_tokens=8_192)
        self.timeout_seconds = timeout_seconds
        # What actually bounds the episode prompt. The token field above is only
        # logged: claiming a token budget that is never enforced was a lie in the
        # documentation.
        self.episode_char_budget = max(12_000, int(episode_char_budget))

    def consolidate(
        self,
        episode: Sequence[dict[str, Any]],
        finish: AgentRunResult,
        system_prompt: str,
        runtime_log: Callable[..., None] | None = None,
        chat_history: Sequence[Message] | None = None,
        short_memory: Sequence[dict[str, Any]] | None = None,
        verbose_base_path: str | Path | None = None,
    ) -> MemoryResult:
        episode_payload = self._bounded_episode(episode)
        prompt = {
            "short_memory": self._bounded_context(short_memory or [], 24_000, "short_memory"),
            # The ReAct transcript starts with its own system prompt, and the
            # Memory Loop adds a system prompt of its own: keeping both wasted
            # tokens on every consolidation.
            "chat_history": self._bounded_context(
                [message for message in (chat_history or []) if str(message.get("role", "")) != "system"],
                48_000,
                "chat_history",
            ),
            "episode": episode_payload,
            "finish_report": finish.report,
            "technical_result": {"status": finish.status.value, "steps": finish.steps, "usage_tokens": finish.usage_tokens, "failure": finish.failure},
            "instruction": MEMORY_LOOP_INSTRUCTION,
        }
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        if runtime_log is not None:
            encoded = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
            runtime_log("provider_request", {"phase": "memory", "message_count": len(messages), "request_chars": len(encoded), "request_sha256": hashlib.sha256(encoded.encode()).hexdigest(), "episode_char_budget": self.episode_char_budget, "logged_token_budget_is_not_enforced": self.budget.tokens, "max_output_tokens": self.budget.output_tokens}, run_id=None)
        verbose_path = verbose_base_path if verbose_base_path is not None else _runtime_log_path(runtime_log)
        if verbose_path is not None and verbose_enabled(verbose_path):
            verbose_write("provider_request", {"phase": "memory", "messages": messages}, base_path=verbose_path, run_id=None)
        result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

        def complete() -> None:
            try:
                result_queue.put(("ok", self.provider.complete(messages, max_tokens=self.budget.output_tokens, tools=())))
            except BaseException as exc:
                result_queue.put(("error", exc))

        worker = threading.Thread(target=complete, name="skynet-memory-provider", daemon=True)
        worker.start()
        worker.join(self.timeout_seconds)
        if worker.is_alive():
            exc = TimeoutError(f"memory loop provider timeout after {self.timeout_seconds:.1f}s")
            if runtime_log is not None:
                runtime_log("memory_degraded", {"error": str(exc)}, run_id=None)
            return MemoryResult(evaluation={"status": "degraded", "error": str(exc)})
        kind, value = result_queue.get()
        if kind == "error":
            exc = value if isinstance(value, Exception) else RuntimeError(str(value))
            if runtime_log is not None:
                runtime_log("memory_degraded", {"error": str(exc)[:500]}, run_id=None)
            return MemoryResult(evaluation={"status": "degraded", "error": str(exc)[:500]})
        turn = value
        if not isinstance(turn, ModelTurn):
            exc = TypeError("memory provider returned invalid turn")
            if runtime_log is not None:
                runtime_log("memory_degraded", {"error": str(exc)}, run_id=None)
            return MemoryResult(evaluation={"status": "degraded", "error": str(exc)})

        if runtime_log is not None:
            runtime_log(
                "provider_response",
                {"phase": "memory", "text_preview": turn.text[:1000], "text_chars": len(turn.text), "tool_call_count": 0, "usage_tokens": turn.total_tokens, "prompt_tokens": turn.prompt_tokens, "completion_tokens": turn.completion_tokens},
                run_id=None,
            )
        if verbose_path is not None and verbose_enabled(verbose_path):
            verbose_write("provider_response", {"phase": "memory", "text": turn.text, "usage_tokens": turn.total_tokens}, base_path=verbose_path, run_id=None)
        try:
            return self._parse(turn)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if runtime_log is not None:
                runtime_log("memory_degraded", {"error": str(exc)[:500]}, run_id=None)
            return MemoryResult(evaluation={"status": "degraded", "error": str(exc)[:500]})

    def _bounded_episode(self, episode: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep the memory prompt bounded while retaining the durable full episode."""
        limit_chars = self.episode_char_budget
        selected: list[dict[str, Any]] = []
        size = 0
        for event in reversed(episode):
            encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            if selected and size + len(encoded) > limit_chars:
                break
            selected.append(event)
            size += len(encoded)
        selected.reverse()
        if len(selected) < len(episode):
            return [{"kind": "memory_context_truncated", "omitted_events": len(episode) - len(selected)}, *selected]
        return selected
    @staticmethod
    def _bounded_context(items: Sequence[dict[str, Any]], limit_chars: int, kind: str) -> list[dict[str, Any]]:
        """Keep auxiliary memory inputs bounded, retaining the newest entries."""
        selected: list[dict[str, Any]] = []
        size = 0
        for item in reversed(items):
            encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            if selected and size + len(encoded) > limit_chars:
                break
            selected.append(item)
            size += len(encoded)
        selected.reverse()
        if len(selected) < len(items):
            return [{"kind": f"{kind}_truncated", "omitted_items": len(items) - len(selected)}, *selected]
        return selected

    @staticmethod
    def _parse(turn: ModelTurn) -> MemoryResult:
        text = turn.text.strip()
        if not text:
            raise ValueError("memory loop returned an empty response")
        data = parse_json_object(text)
        validate_shape(data, MEMORY_RESPONSE_SCHEMA)
        candidates = data.get("memory_candidates", [])
        plan = data.get("next_plan", {})
        initial_prompt = data.get("initial_prompt", "")
        goal_updates = data.get("goal_updates", [])
        task_updates = data.get("task_updates", [])
        evaluation = data.get("evaluation", {})
        return MemoryResult(
            memory_candidates=candidates if isinstance(candidates, list) else [],
            next_plan=plan if isinstance(plan, dict) else {},
            initial_prompt=str(initial_prompt) if initial_prompt else "",
            goal_updates=goal_updates if isinstance(goal_updates, list) else [],
            task_updates=task_updates if isinstance(task_updates, list) else [],
            evaluation=evaluation if isinstance(evaluation, dict) else {},
        )
