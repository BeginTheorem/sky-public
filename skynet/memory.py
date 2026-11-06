from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model_contracts import MAX_MODEL_REASKS, MEMORY_RESPONSE_SCHEMA, corrective_reask_instruction, drop_unknown_fields, parse_json_object, validate_shape
from .models import AgentRunResult, Budget, ModelTurn
from .provider import LLMProvider, Message
from .runtime_log import verbose_enabled, verbose_write

MEMORY_LOOP_INSTRUCTION = (
    "Read the completed episode and extract durable strategic observations. You may suggest evidence and planner hints, but "
    "the deterministic harness owns portfolio ranking and task selection. "
    "Return JSON only with memory_candidates, next_plan, initial_prompt, goal_updates, task_updates, and evaluation. "
    "When new evidence contradicts an existing memory, emit a candidate with supersedes_memory_id set to the contradicted "
    "memory's id and put the contradicting evidence in evidence; never supersede on a guess. "
    "A synthesis that generalizes several memories into a higher-level claim must name the memories it is derived from, one "
    "per line in evidence as '<memory_id>: <what it supports>', and be emitted only when those rows are actually in the "
    "memory_context or pinned_memories given to you; a claim that cites no memory and no episode event is a guess, so keep "
    "it out of memory_candidates. "
    "Goal/task updates must reference existing IDs and include evidence in outcome when completing work. "
    "Do not include reasoning. Do not resurrect completed, blocked, exhausted, or reset planning context."
)

# Marks text cut to fit a budget rather than omitted entirely, so a reader can
# tell a shortened item apart from one that was never sent, and can see how much
# was dropped without a second field to account for.
def _truncation_mark(omitted: int) -> str:
    return f" ...[truncated by the memory-loop budget: {omitted} chars omitted]"




def memory_digest_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """``SKYNET_MEMORY_DIGEST=0`` restores the raw re-send; digesting is the default.

    The raw path sends the episode twice (as an event slice and as the transcript) and pays for
    bulk the consolidator never reads: measured 74 calls at 100% fresh billing, 23.8% of the
    epoch's fresh input.
    """
    source = os.environ if environ is None else environ
    return str(source.get("SKYNET_MEMORY_DIGEST", "1")).strip().casefold() not in {"0", "false", "no", "off"}


def _first_line(value: object, limit: int) -> str:
    text = str(value or "").strip().splitlines()
    return (text[0] if text else "")[:limit]


def _encode(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _digest_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """One event, rendered as its facts.

    The loop consolidates an episode into durable memory, so what it needs from an event is what
    happened, to what, and whether it worked. Bulk is not a fact: a raw tool result averages
    4,100 chars, and the consolidator's question about it ("did it work, on what, and what did it
    say when it did not") is answerable from the identifiers and the first line.

    Events arrive as ``{"kind": ..., "payload": {...}}`` -- the payload holds the substance --
    but a flattened dict is tolerated so a caller that already unwrapped one still digests.
    """
    kind = str(event.get("kind") or "event")
    raw_payload = event.get("payload")
    payload: Mapping[str, Any] = raw_payload if isinstance(raw_payload, Mapping) else event
    if kind == "tool_call":
        raw_arguments = payload.get("arguments")
        arguments: Mapping[str, Any] = raw_arguments if isinstance(raw_arguments, Mapping) else {}
        compact = {
            key: (value if isinstance(value, (int, float, bool)) else str(value)[:200])
            for key, value in list(arguments.items())[:6]
        }
        return {
            "kind": "tool_call",
            "tool": str(payload.get("tool_name") or payload.get("tool") or event.get("tool_name") or ""),
            "arguments": compact,
        }
    if kind == "tool_result":
        raw_call = payload.get("call")
        call: Mapping[str, Any] = raw_call if isinstance(raw_call, Mapping) else {}
        raw_result = payload.get("result")
        result: Mapping[str, Any] = raw_result if isinstance(raw_result, Mapping) else payload
        ok = bool(result.get("ok")) if "ok" in result else bool(result.get("returncode") == 0)
        row: dict[str, Any] = {
            "kind": "tool_result",
            "tool": str(call.get("tool_name") or payload.get("tool_name") or event.get("tool_name") or ""),
            "ok": ok,
        }
        if ok:
            row["head"] = _first_line(result.get("stdout") or result.get("output") or result.get("result"), 400)
        else:
            # An error line is a fact the next cycle may need verbatim.
            row["error"] = _first_line(result.get("error") or result.get("stderr") or result.get("message"), 400)
        return row
    if kind == "run_started":
        raw_intention = payload.get("intention")
        intention: Mapping[str, Any] = raw_intention if isinstance(raw_intention, Mapping) else {}
        observations = payload.get("observations")
        return {
            "kind": "run_started",
            "wake_cause": str(payload.get("wake_cause") or ""),
            "intention": {key: str(value)[:200] for key, value in list(intention.items())[:4]},
            "obs_note": f"{len(observations) if isinstance(observations, list) else 0} observations dropped (bulk)",
        }
    if kind == "finish_report":
        return {"kind": "finish_report", "report": str(payload.get("report") or payload)[:2000]}
    if kind in MemoryLoop.FACT_KINDS:
        rendered = json.dumps(payload, ensure_ascii=False, default=str)
        if len(rendered) <= 600:
            return {"kind": kind, "payload": json.loads(rendered)}
        return {"kind": kind, "payload_head": _first_line(rendered, 500)}
    if kind in MemoryLoop.TELEMETRY_KINDS:
        return {"kind": kind, "note": "telemetry dropped"}
    rendered = json.dumps(payload, ensure_ascii=False, default=str)
    return {"kind": kind, "summary": _first_line(rendered, 300)}


def digest_episode(episode: Sequence[dict[str, Any]], limit_chars: int) -> list[dict[str, Any]]:
    """A bounded, fact-only view of one episode: newest events kept, omissions marked."""
    rows: list[dict[str, Any]] = []
    size = 0
    omitted = 0
    for event in reversed(episode):
        row = _digest_event(event)
        encoded = _encode(row)
        if size + encoded > limit_chars and rows:
            omitted += 1
            continue
        rows.append(row)
        size += encoded
    rows.reverse()
    if omitted:
        return [{"kind": "digest_truncated", "omitted_events": omitted}, *rows]
    return rows


def digest_history(chat_history: Sequence[Message], limit_chars: int) -> list[dict[str, Any]]:
    """What the model *did*, from the transcript: actions and tool outcomes, no reasoning.

    The episode digest already carries the tool rows; this keeps the assistant's own action
    sequence (which tool it chose with which arguments) in the order it chose them, and drops
    the reasoning prose the loop's contract says it must not consolidate anyway.
    """
    rows: list[dict[str, Any]] = []
    for message in chat_history:
        role = str(message.get("role") or "")
        if role == "system":
            continue
        if role == "assistant":
            calls = message.get("tool_calls") or []
            for call in calls:
                function = call.get("function") if isinstance(call, Mapping) else None
                function = function if isinstance(function, Mapping) else {}
                rows.append({
                    "kind": "action",
                    "tool": str(function.get("name") or ""),
                    "arguments": _first_line(function.get("arguments"), 200),
                })
        elif role == "tool":
            rows.append({"kind": "result", "tool_call_id": str(message.get("tool_call_id") or ""), "head": _first_line(message.get("content"), 240)})
    size = sum(_encode(row) for row in rows)
    if size <= limit_chars:
        return rows
    # Keep the tail: the end of an episode is what the consolidation is about.
    kept: list[dict[str, Any]] = []
    running = 0
    for row in reversed(rows):
        encoded = _encode(row)
        if running + encoded > limit_chars and kept:
            break
        kept.append(row)
        running += encoded
    kept.reverse()
    return [{"kind": "history_digest_truncated", "omitted_rows": len(rows) - len(kept)}, *kept]

def _truncated_item(item: dict[str, Any], limit_chars: int, kind: str) -> dict[str, Any]:
    """Return ``item`` reduced to fit ``limit_chars``, keeping valid JSON.

    A single item larger than the whole budget must not be admitted whole: the
    callers' running-total guard compares only *subsequent* items against the
    limit, so one oversized row used to set the prompt size and the configured
    budget bounded nothing -- measured on the live ledger (generation 184,
    read-only copy), 27 of the 29 degraded Memory-Loop episodes sent a single
    transcript row over 48,000 chars (max 460,118) and the built request
    measured 2.9x its configured input budget. The longest text-bearing field is
    prefix-truncated and marked; every other field is kept, because kind,
    position and timestamp are what make the row interpretable.
    """
    encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= limit_chars:
        return item
    for field_name in ("content", "payload", "report", "text"):
        value = item.get(field_name)
        if not isinstance(value, str) or not value:
            continue
        mark = _truncation_mark(len(value))
        overhead = len(encoded) - len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        keep = limit_chars - overhead - len(mark)
        while keep > 0:
            trimmed = dict(item)
            trimmed[field_name] = value[:keep] + mark
            result = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
            if len(result) <= limit_chars:
                return trimmed
            # Escaping can expand the kept prefix, so shave the excess.
            keep -= len(result) - limit_chars
    # No text field to shorten (or the wrapper alone exceeds the budget): the
    # only honest reduction left is to send the item's shape, not its content.
    return {"kind": f"{kind}_item_truncated", "omitted_chars": len(encoded)}



@dataclass(slots=True)
class MemoryResult:
    """Bounded output of the independent post-Run memory pass."""

    memory_candidates: list[dict[str, Any]] = field(default_factory=list)
    next_plan: dict[str, Any] = field(default_factory=dict)
    initial_prompt: str = ""
    goal_updates: list[dict[str, Any]] = field(default_factory=list)
    task_updates: list[dict[str, Any]] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=dict)
    # What the pass cost, reported by the loop itself: the run row the reactor opens
    # for a consolidation has no other way to learn it (`run_results.usage_tokens`
    # and `.steps` are written from here). `model_calls` counts provider attempts,
    # including the re-asks a malformed reply costs.
    usage_tokens: int = 0
    model_calls: int = 0
    # The largest assembled request this pass sent, in characters. The loop's real
    # bound is characters, not the token field it logs: the reactor states a soft
    # token budget on the pass's run row, and this is what lets the row be checked
    # against what actually went out.
    peak_request_chars: int = 0
    # A reply that carried undeclared keys (:func:`drop_unknown_fields`) or a
    # section that failed its own contract is salvaged rather than discarded, so
    # the harness records what it had to tolerate for post-mortem.
    dropped_fields: list[str] = field(default_factory=list)
    rejected_sections: list[str] = field(default_factory=list)


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

    # The caps that actually bound one consolidation request. The token figure in the
    # loop's ``budget`` is logged rather than enforced; these are the real bound, and
    # :meth:`request_ceiling_tokens` is how the reactor states an honest (soft) budget
    # on the pass's run row instead of quoting a number nothing enforces.
    SHORT_MEMORY_CHARS = 24_000
    CHAT_HISTORY_CHARS = 48_000
    DEFAULT_EPISODE_CHAR_BUDGET = 150_000

    # Digested payload budgets. The raw figures above bound a *re-send* of the episode: raw
    # tool_result rows average 4,100 chars and a run_started envelope ~17,000, so the payload
    # was 74,000 tokens of which the loop only ever needed the outcomes, the identifiers and
    # the error lines. Measured on the live ledger: 74 calls, 4.69M prompt tokens, 100% fresh
    # (a one-shot prompt cannot be cached), 23.8% of the epoch's whole fresh input at 4% of its
    # volume. The digest keeps the facts and drops the bulk.
    DEFAULT_DIGEST_EPISODE_CHARS = 24_000
    DEFAULT_DIGEST_HISTORY_CHARS = 8_000
    #: Event kinds whose payload is dropped to a size note: they say nothing a consolidator
    #: can use, and there are thousands of them.
    TELEMETRY_KINDS = ("provider_request", "provider_response", "scheduler_decision", "heartbeat")
    #: Kinds whose payload is a fact and kept, bounded, in full-ish form.
    FACT_KINDS = (
        "memory_event", "memory_consolidated", "checkpoint", "effect_reapplied", "plan_recorded",
        "planner_decision", "proposal_validated", "proposal_changes_applied", "task_progress",
        "context_compacted", "context_limit", "policy_denied", "idea_learnability_rejected",
    )

    @classmethod
    def request_ceiling_tokens(cls, *, episode_char_budget: int | None = None, output_tokens: int = 8_192) -> int:
        """The honest token ceiling of one request: the caps, converted, plus output room.

        ``//3`` is the same conservative estimator ``ReActRunner._request_tokens`` uses.
        The caller adds whatever headroom it wants; the point of this number is that it
        is measured from what the loop enforces, not from an aspiration.
        """
        episode = cls.DEFAULT_EPISODE_CHAR_BUDGET if episode_char_budget is None else max(12_000, int(episode_char_budget))
        return (cls.SHORT_MEMORY_CHARS + cls.CHAT_HISTORY_CHARS + episode) // 3 + int(output_tokens)

    @classmethod
    def digested_ceiling_tokens(
        cls,
        *,
        digest_episode_chars: int = DEFAULT_DIGEST_EPISODE_CHARS,
        digest_history_chars: int = DEFAULT_DIGEST_HISTORY_CHARS,
        output_tokens: int = 8_192,
    ) -> int:
        """The honest ceiling of a digested request: measured from the bound it enforces."""
        return (
            cls.SHORT_MEMORY_CHARS + max(1_000, int(digest_history_chars)) + max(2_000, int(digest_episode_chars))
        ) // 3 + int(output_tokens)

    def __init__(
        self,
        provider: LLMProvider,
        *,
        budget: Budget | None = None,
        timeout_seconds: float = 180.0,
        episode_char_budget: int = DEFAULT_EPISODE_CHAR_BUDGET,
        digest: bool = True,
        digest_episode_chars: int = DEFAULT_DIGEST_EPISODE_CHARS,
        digest_history_chars: int = DEFAULT_DIGEST_HISTORY_CHARS,
    ) -> None:
        self.provider = provider
        self.budget = budget or Budget(steps=1, tokens=50_000, seconds=180.0, output_tokens=8_192)
        self.timeout_seconds = timeout_seconds
        # Mirrors the class caps so a reader sees the bound next to the loop it bounds.
        self.short_memory_chars = self.SHORT_MEMORY_CHARS
        self.chat_history_chars = self.CHAT_HISTORY_CHARS
        # What actually bounds the episode prompt. The token field above is only
        # logged: claiming a token budget that is never enforced was a lie in the
        # documentation.
        self.episode_char_budget = max(12_000, int(episode_char_budget))
        # Digesting is the default because the raw path re-sends what the event log already
        # records, twice (episode and chat_history are the same episode in two renderings).
        self.digest = digest
        self.digest_episode_chars = max(2_000, int(digest_episode_chars))
        self.digest_history_chars = max(1_000, int(digest_history_chars))

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
        if self.digest:
            episode_payload = digest_episode(episode, self.digest_episode_chars)
            history_payload = digest_history(chat_history or [], self.digest_history_chars)
        else:
            episode_payload = self._bounded_episode(episode)
            # The ReAct transcript starts with its own system prompt, and the
            # Memory Loop adds a system prompt of its own: keeping both wasted
            # tokens on every consolidation.
            history_payload = self._bounded_context(
                [message for message in (chat_history or []) if str(message.get("role", "")) != "system"],
                self.chat_history_chars,
                "chat_history",
            )
        prompt = {
            "short_memory": self._bounded_context(short_memory or [], self.short_memory_chars, "short_memory"),
            "chat_history": history_payload,
            "episode": episode_payload,
            "finish_report": finish.report,
            "technical_result": {"status": finish.status.value, "steps": finish.steps, "usage_tokens": finish.usage_tokens, "failure": finish.failure},
            "instruction": MEMORY_LOOP_INSTRUCTION,
        }
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        verbose_path = verbose_base_path if verbose_base_path is not None else _runtime_log_path(runtime_log)
        # One shared deadline for the reply and every re-ask: three re-asks must
        # not multiply the loop's timeout by four.
        deadline = time.monotonic() + self.timeout_seconds
        attempt = 0
        usage_tokens = 0
        peak_request_chars = 0
        while True:
            attempt += 1
            peak_request_chars = max(peak_request_chars, self._log_request(messages, runtime_log, verbose_path, attempt))
            kind, value = self._call_provider(messages, deadline)
            if kind == "timeout":
                exc = TimeoutError(f"memory loop provider timeout after {self.timeout_seconds:.1f}s")
                self._log_degraded(runtime_log, str(exc))
                return MemoryResult(evaluation={"status": "degraded", "error": str(exc)}, usage_tokens=usage_tokens, model_calls=attempt, peak_request_chars=peak_request_chars)
            if kind == "error":
                exc = value if isinstance(value, Exception) else RuntimeError(str(value))
                self._log_degraded(runtime_log, str(exc)[:500])
                return MemoryResult(evaluation={"status": "degraded", "error": str(exc)[:500]}, usage_tokens=usage_tokens, model_calls=attempt, peak_request_chars=peak_request_chars)
            turn = value
            if not isinstance(turn, ModelTurn):
                exc = TypeError("memory provider returned invalid turn")
                self._log_degraded(runtime_log, str(exc))
                return MemoryResult(evaluation={"status": "degraded", "error": str(exc)}, usage_tokens=usage_tokens, model_calls=attempt, peak_request_chars=peak_request_chars)
            usage_tokens += int(getattr(turn, "total_tokens", 0) or 0)
            self._log_response(turn, runtime_log, verbose_path, attempt)
            try:
                parsed = self._parse(turn)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                # A tool call where the memory loop has no tools is the dominant
                # live failure: 14 of the last 30 memory_degraded events were
                # "structured model response contains no acceptable JSON value".
                # One shared deadline bounds the re-asks, exactly as it bounds
                # the first reply.
                if attempt > MAX_MODEL_REASKS or time.monotonic() >= deadline:
                    self._log_degraded(runtime_log, str(exc)[:500])
                    return MemoryResult(evaluation={"status": "degraded", "error": str(exc)[:500]}, usage_tokens=usage_tokens, model_calls=attempt, peak_request_chars=peak_request_chars)
                if runtime_log is not None:
                    runtime_log(
                        "memory_reask",
                        {
                            "attempt": attempt,
                            "error": str(exc)[:300],
                            "attempted_tools": [str(call.tool_name) for call in turn.tool_calls],
                        },
                        run_id=None,
                    )
                messages = [
                    *messages,
                    {"role": "assistant", "content": turn.text},
                    {
                        "role": "user",
                        "content": corrective_reask_instruction(
                            phase="memory loop", tool_calls=turn.tool_calls, parser_error=str(exc)[:300]
                        ),
                    },
                ]
                continue
            parsed.usage_tokens = usage_tokens
            parsed.model_calls = attempt
            parsed.peak_request_chars = peak_request_chars
            return parsed

    def _bounded_episode(self, episode: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep the memory prompt bounded while retaining the durable full episode."""
        limit_chars = self.episode_char_budget
        selected: list[dict[str, Any]] = []
        size = 0
        for event in reversed(episode):
            encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            if not selected and len(encoded) > limit_chars:
                # Same admission rule as _bounded_context: the newest item is
                # the one this function exists to keep, so an item too large for
                # the whole budget is shortened rather than let through whole.
                trimmed = _truncated_item(event, max(1, limit_chars // 2), "episode")
                selected.append(trimmed)
                size += len(json.dumps(trimmed, ensure_ascii=False, separators=(",", ":")))
                continue
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
            if not selected and len(encoded) > limit_chars:
                # The guard below reads `selected and ...`, so the FIRST item --
                # always the newest, the one this function exists to keep -- was
                # admitted whole however large it was: one oversized transcript
                # row set the prompt size and the budget it was meant to enforce
                # bounded nothing. Measured on the live ledger (generation 184,
                # read-only copy): in 27 of the 29 episodes whose Memory Loop
                # degraded, a single transcript row exceeds 48,000 chars (max
                # 460,118); `_bounded_context` returned 2 items totalling
                # 558,781 chars against a 48,000-char limit (11.6x), the built
                # request measured 2.9x the 50,000-token `memory_input_tokens`
                # budget, and all 29 had to be recovered or were lost. An item
                # that cannot fit is now shortened and marked instead of
                # admitted whole; older items fill only what is left.
                trimmed = _truncated_item(item, max(1, limit_chars // 2), kind)
                selected.append(trimmed)
                size += len(json.dumps(trimmed, ensure_ascii=False, separators=(",", ":")))
                continue
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
        # A single undeclared key used to fail the whole consolidation
        # (`initial_prompt_note` beside `initial_prompt`). Drop the
        # extras and validate each section on its own so one malformed section
        # cannot discard the candidates and plan that were perfectly good.
        dropped = drop_unknown_fields(data, MEMORY_RESPONSE_SCHEMA)
        properties = MEMORY_RESPONSE_SCHEMA["properties"]
        rejected: list[str] = []

        def section(key: str) -> Any:
            if key not in data:
                return None
            try:
                validate_shape(data[key], properties[key], path=f"response.{key}")
            except ValueError:
                rejected.append(f"response.{key}")
                return None
            return data[key]

        candidates = section("memory_candidates")
        plan = section("next_plan")
        initial_prompt = section("initial_prompt")
        goal_updates = section("goal_updates")
        task_updates = section("task_updates")
        evaluation = section("evaluation")
        if all(value is None for value in (candidates, plan, initial_prompt, goal_updates, task_updates, evaluation)):
            raise ValueError("memory response carried no usable section")
        return MemoryResult(
            memory_candidates=candidates if isinstance(candidates, list) else [],
            next_plan=plan if isinstance(plan, dict) else {},
            initial_prompt=str(initial_prompt) if isinstance(initial_prompt, str) and initial_prompt else "",
            goal_updates=goal_updates if isinstance(goal_updates, list) else [],
            task_updates=task_updates if isinstance(task_updates, list) else [],
            evaluation=evaluation if isinstance(evaluation, dict) else {},
            dropped_fields=dropped,
            rejected_sections=rejected,
        )

    def _call_provider(self, messages: list[Message], deadline: float) -> tuple[str, object]:
        """One provider call bounded by the shared memory-loop deadline.

        The deadline is shared across the initial reply and every re-ask, so
        three re-asks cannot extend the loop's timeout. Returns ``("ok", turn)``,
        ``("error", exc)`` or ``("timeout", None)``; an over-running worker is a
        daemon and is left to finish on its own, exactly as before.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ("timeout", None)
        result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

        def complete() -> None:
            try:
                result_queue.put(("ok", self.provider.complete(messages, max_tokens=self.budget.output_tokens, tools=())))
            except BaseException as exc:
                result_queue.put(("error", exc))

        worker = threading.Thread(target=complete, name="skynet-memory-provider", daemon=True)
        worker.start()
        worker.join(remaining)
        if worker.is_alive():
            return ("timeout", None)
        return result_queue.get()

    def _log_request(self, messages: list[Message], runtime_log: Callable[..., None] | None, verbose_path: str | Path | None, attempt: int) -> int:
        """Log the request and return its assembled size in characters."""
        encoded = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        if runtime_log is not None:
            runtime_log("provider_request", {"phase": "memory", "attempt": attempt, "message_count": len(messages), "request_chars": len(encoded), "request_sha256": hashlib.sha256(encoded.encode()).hexdigest(), "episode_char_budget": self.episode_char_budget, "logged_token_budget_is_not_enforced": self.budget.tokens, "max_output_tokens": self.budget.output_tokens}, run_id=None)
        if verbose_path is not None and verbose_enabled(verbose_path):
            verbose_write("provider_request", {"phase": "memory", "attempt": attempt, "messages": messages}, base_path=verbose_path, run_id=None)
        return len(encoded)

    def _log_response(self, turn: ModelTurn, runtime_log: Callable[..., None] | None, verbose_path: str | Path | None, attempt: int) -> None:
        if runtime_log is not None:
            runtime_log(
                "provider_response",
                {"phase": "memory", "attempt": attempt, "text_preview": turn.text[:1000], "text_chars": len(turn.text), "tool_call_count": len(turn.tool_calls), "usage_tokens": turn.total_tokens, "prompt_tokens": turn.prompt_tokens, "completion_tokens": turn.completion_tokens, "cached_tokens": turn.cached_tokens, "finish_reason": turn.finish_reason},
                run_id=None,
            )
        if verbose_path is not None and verbose_enabled(verbose_path):
            verbose_write("provider_response", {"phase": "memory", "attempt": attempt, "text": turn.text, "usage_tokens": turn.total_tokens}, base_path=verbose_path, run_id=None)

    @staticmethod
    def _log_degraded(runtime_log: Callable[..., None] | None, error: str) -> None:
        if runtime_log is not None:
            runtime_log("memory_degraded", {"error": error}, run_id=None)
