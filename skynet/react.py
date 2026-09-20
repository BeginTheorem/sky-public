from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import os
import signal
import subprocess
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model_contracts import FINISH_REPORT_SCHEMA, parse_json_object, validate_shape
from .models import AgentRunResult, ModelTurn, RunStatus, StartEnvelope
from .provider import LLMProvider, Message, Tool
from .providers.errors import ProviderError
from .runtime_log import verbose_enabled, verbose_write
from .store import StateStore

log = logging.getLogger("skynet.react")
MAX_RESPONSE_LOG_PREVIEW = 2000
WORKTREE_STATUS_TIMEOUT_SECONDS = 5.0
WORKTREE_WARNING_MAX_FILES = 20


# Child process groups registered by tools during the current tool call. A tool
# (for example BashTool) calls ``register_process_group(os.getpgid(child.pid))``
# right after spawning a session leader. If the call is interrupted by a
# BaseException (WatchdogTimeout from SIGALRM, a SIGTERM-driven interrupt, or
# KeyboardInterrupt), the runner kills and reaps whatever is still registered.
# A context variable keeps concurrent calls in different threads isolated.
_current_process_groups: contextvars.ContextVar[set[int] | None] = contextvars.ContextVar(
    "skynet_tool_process_groups", default=None
)


def register_process_group(pgid: int) -> None:
    """Register a live child process group for cleanup if the call is interrupted.

    Safe to call from any tool and a no-op outside a tracked tool call, so a
    tool can always register without knowing whether the runner is watching.
    """
    groups = _current_process_groups.get()
    if groups is None:
        return
    try:
        groups.add(int(pgid))
    except (TypeError, ValueError):
        return


@contextlib.contextmanager
def tracked_process_group() -> Iterator[set[int]]:
    """Expose the current call's process-group set to tools for one call."""
    groups: set[int] = set()
    token = _current_process_groups.set(groups)
    try:
        yield groups
    finally:
        _current_process_groups.reset(token)


def _terminate_process_groups(groups: set[int]) -> None:
    """Kill and best-effort reap registered process groups. Never raises."""
    for pgid in tuple(groups):
        # Already gone, not ours, or not permitted: nothing left to clean.
        with contextlib.suppress(OSError):
            os.killpg(pgid, signal.SIGKILL)
        deadline = time.monotonic() + 1.0
        while True:
            try:
                reaped, _ = os.waitpid(pgid, os.WNOHANG)
            except (ChildProcessError, OSError):
                break
            if reaped:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)


@dataclass(slots=True)
class ReActConfig:
    max_steps: int = 100
    max_tokens: int = 200_000  # maximum input context for one provider request
    output_tokens: int = 8_192
    timeout_seconds: float = 1800.0
    context_warning_ratio: float = 0.5
    context_warning_ratio_high: float = 0.75
    # Model time, not wall time: the budget enforced at the top of run() and
    # before each tool call is provider latency. Without an in-band note the
    # model only learns the budget is gone once the episode is discarded, so
    # the budget is disclosed in band (budget-conditioned control).
    time_warning_ratio: float = 0.7
    finish_threshold_ratio: float = 0.875  # protective forced Finish, not normal completion
    context_finish_reserve: int = 25_000
    # Provider retries live in FallbackProvider's strike ladder; the episode
    # level must not duplicate them.
    provider_retries: int = 0
    provider_retry_backoff_seconds: float = 1.0
    read_tool_retries: int = 1
    tool_retry_backoff_seconds: float = 0.5
    max_compactions: int = 3
    compaction_keep_messages: int = 4
    max_repeated_responses: int = 3
    tool_result_max_chars: int = 8_000


class ReActRunner:
    def __init__(
        self,
        provider: LLMProvider,
        store: StateStore,
        tools: Mapping[str, Tool],
        config: ReActConfig | None = None,
        *,
        worktree_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.provider = provider
        self.store = store
        self.tools = tools
        self.config = config or ReActConfig()
        # Optional main-worktree root used to warn the model when bash leaves
        # tracked files modified. Unset keeps the runner byte-for-byte as before.
        self.worktree_root = Path(worktree_root) if worktree_root else None
        self._worktree_status_step: int | None = None
        self._worktree_status: list[str] = []

    def run(self, start: StartEnvelope, system_prompt: str) -> AgentRunResult:
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "START ENVELOPE\n" + json.dumps(start.as_dict(), ensure_ascii=False) + "\nEND START ENVELOPE"},
        ]
        self._record_history(start.run_id, messages)
        model_seconds = 0.0
        used_tokens = 0
        context_warning_sent = False
        context_warning_high_sent = False
        time_warning_sent = False
        compactions = 0
        phase = "initial"
        response_fingerprints: dict[str, int] = {}
        deferred_control: dict[str, Any] = {}
        failed_proposals: set[str] = set()
        self._worktree_status_step = None
        self._worktree_status = []
        steps_limit = max(1, min(int(start.budget.steps), self.config.max_steps))
        time_limit = max(0.001, min(float(start.budget.seconds), self.config.timeout_seconds))
        request_limit = max(1, min(int(start.budget.tokens), self.config.max_tokens))
        response_token_limit = max(1, min(int(start.budget.output_tokens), self.config.output_tokens))
        for tool in self.tools.values():
            reset = getattr(tool, "reset_run_scope", None)
            if callable(reset):
                reset()
        tool_schemas = [tool.schema for tool in self.tools.values()]
        self.store.append_event("react_phase", {"phase": phase}, start.run_id)
        self.store.touch_run(start.run_id, phase)

        for step in range(steps_limit):
            if model_seconds >= time_limit:
                status, report, finish_usage = self._finish(messages, start.run_id, reason="time budget")
                if status == RunStatus.NEEDS_RECOVERY:
                    return AgentRunResult(status, report or "ReAct time budget exhausted", step, used_tokens + finish_usage, "time budget")
                return AgentRunResult(status, report, step, used_tokens + finish_usage, "time budget")
            request_tokens = self._request_tokens(messages, tool_schemas)
            reserve = min(self.config.context_finish_reserve, max(0, request_limit // 8))
            if request_tokens >= request_limit * self.config.finish_threshold_ratio:
                status, report, finish_usage = self._finish(messages, start.run_id, reason="protective context threshold")
                return AgentRunResult(status, report, step, used_tokens + finish_usage, "protective context threshold")
            if request_tokens + reserve >= request_limit:
                if len(messages) > 3 and compactions < self.config.max_compactions:
                    messages = self._compact_messages(messages)
                    compactions += 1
                    self.store.append_event(
                        "context_compacted",
                        {"estimated_tokens_before": request_tokens, "messages_after": len(messages), "attempt": compactions},
                        start.run_id,
                    )
                    continue
                self.store.append_transcript(
                    "context_limit",
                    {"estimated_tokens": request_tokens, "limit": request_limit},
                    start.run_id,
                )
                return self._context_budget_result(messages, start.run_id, step, used_tokens)
            if not context_warning_sent and request_tokens >= request_limit * self.config.context_warning_ratio:
                self.store.append_transcript(
                    "context_warning",
                    {"estimated_tokens": request_tokens, "limit": request_limit},
                    start.run_id,
                )
                messages.append({"role": "system", "content": "Context warning: keep the next action focused and avoid returning large raw files."})
                context_warning_sent = True
            if not context_warning_high_sent and request_tokens >= request_limit * self.config.context_warning_ratio_high:
                self.store.append_transcript(
                    "context_warning",
                    {"estimated_tokens": request_tokens, "limit": request_limit, "remaining": request_limit - request_tokens},
                    start.run_id,
                )
                messages.append({"role": "system", "content": "Critical context warning: finish the current task or compact the history now."})
                context_warning_high_sent = True
            # The time budget must be disclosed once, in band, so the model can
            # close the episode with a real Finish Report instead of hitting a
            # budget-exhausted error before tool execution.
            if not time_warning_sent and model_seconds >= time_limit * self.config.time_warning_ratio:
                self.store.append_transcript(
                    "time_warning",
                    {
                        "model_seconds": round(model_seconds, 3),
                        "limit": time_limit,
                        "remaining_seconds": round(max(0.0, time_limit - model_seconds), 3),
                    },
                    start.run_id,
                )
                messages.append({
                    "role": "system",
                    "content": (
                        f"Time budget: {model_seconds:.0f}s of {time_limit:.0f}s of model time used "
                        f"({100.0 * model_seconds / max(time_limit, 0.001):.0f}%); "
                        "stop exploring and return the Finish Report now."
                    ),
                })
                time_warning_sent = True
            request_tokens = self._request_tokens(messages, tool_schemas)
            if request_tokens >= request_limit * self.config.finish_threshold_ratio:
                status, report, finish_usage = self._finish(messages, start.run_id, reason="protective context threshold")
                return AgentRunResult(status, report, step, used_tokens + finish_usage, "protective context threshold")
            if request_tokens + reserve >= request_limit:
                compacted = self._compact_messages(messages)
                if compacted == messages or compactions >= self.config.max_compactions:
                    return self._context_budget_result(messages, start.run_id, step, used_tokens)
                messages = compacted
                compactions += 1
                self.store.append_event(
                    "context_compacted",
                    {"estimated_tokens_before": request_tokens, "messages_after": len(messages), "attempt": compactions},
                    start.run_id,
                )
                continue
            log.info("run=%s step=%d requesting model turn", start.run_id, step + 1)
            self.store.touch_run(start.run_id, phase, progress=True)
            try:
                turn, turn_seconds = self._complete_with_retry(
                    messages,
                    max_tokens=response_token_limit,
                    tools=tool_schemas,
                    run_id=start.run_id,
                    step=step,
                )
            except Exception as exc:
                self.store.append_event(
                    "provider_failure",
                    {"step": step + 1, "error": str(exc)[:1000]},
                    start.run_id,
                )
                self._record_history(start.run_id, messages)
                return AgentRunResult(RunStatus.NEEDS_RECOVERY, f"Provider unavailable: {str(exc)[:500]}", step, used_tokens, str(exc))
            model_seconds += turn_seconds
            used_tokens += turn.total_tokens
            # The fingerprint covers the tool calls too: many providers send no
            # preamble text, so a text-only fingerprint would treat unrelated
            # calls (or none at all) as the same response.
            call_signature = json.dumps(
                [[str(call.tool_name), json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, default=str)] for call in turn.tool_calls],
                ensure_ascii=False,
            )
            response_fingerprint = hashlib.sha256(
                (turn.text.strip() + "|" + call_signature).encode("utf-8")
            ).hexdigest()
            response_fingerprints[response_fingerprint] = response_fingerprints.get(response_fingerprint, 0) + 1
            # Checked before any other handling: a response without tool calls
            # used to terminate the run immediately, so repeated identical
            # responses could never accumulate and this guard was unreachable.
            # The comparison now covers tool-calling responses too, because
            # repeating the same request forever is exactly the loop to stop.
            if response_fingerprints[response_fingerprint] > self.config.max_repeated_responses:
                self.store.append_event("doom_loop", {"phase": phase, "kind": "model_response"}, start.run_id)
                status, report, finish_usage = self._finish(messages, start.run_id)
                if status == RunStatus.NEEDS_RECOVERY:
                    return AgentRunResult(status, report or "Repeated model response detected", step + 1, used_tokens + finish_usage, "doom loop")
                return AgentRunResult(status, report, step + 1, used_tokens + finish_usage, "doom loop")
            log.info(
                "run=%s step=%d model response tool_calls=%d usage_tokens=%d",
                start.run_id,
                step + 1,
                len(turn.tool_calls),
                turn.usage_tokens,
            )
            if not turn.tool_calls:
                log.info(
                    "run=%s step=%d model final response=%s",
                    start.run_id,
                    step + 1,
                    turn.text[:MAX_RESPONSE_LOG_PREVIEW],
                )
                messages.append({"role": "assistant", "content": turn.text})
                self._record_history(start.run_id, messages)
                self.store.append_event("finish_report", {"step": step + 1, "text": turn.text, "usage_tokens": turn.total_tokens, "prompt_tokens": turn.prompt_tokens, "completion_tokens": turn.completion_tokens}, start.run_id)
                report = turn.text.strip()
                status, report = self._parse_finish_report(report, start.run_id)
                return AgentRunResult(status, report, step + 1, used_tokens)

            if phase == "initial":
                phase = "common"
                self.store.append_event("react_phase", {"phase": phase}, start.run_id)
                self.store.touch_run(start.run_id, phase)

            assistant_message: Message = {
                "role": "assistant",
                "content": turn.text,
                "tool_calls": [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": json.dumps(call.arguments, separators=(",", ":")),
                        },
                    }
                    for call in turn.tool_calls
                ],
            }
            messages.append(assistant_message)

            for call in turn.tool_calls:
                # Record every attempted invocation, including unknown tools, so
                # recovery can reconcile the same observable event that produced
                # the result.
                self.store.append_event("tool_call", {"call_id": call.call_id, "tool_name": call.tool_name, "arguments": call.arguments}, start.run_id)
                if model_seconds >= time_limit:
                    self.store.append_event(
                        "budget_exhausted",
                        {"budget": "time", "step": step + 1, "call_id": call.call_id, "tool_name": call.tool_name, "model_seconds": round(model_seconds, 3)},
                        start.run_id,
                    )
                    self._record_history(start.run_id, messages)
                    return AgentRunResult(
                        RunStatus.NEEDS_RECOVERY,
                        "ReAct time budget exhausted before tool execution",
                        step + 1,
                        used_tokens,
                        "time budget",
                    )
                tool = self.tools.get(call.tool_name)
                if tool is None:
                    result = {"ok": False, "error": f"unknown tool: {call.tool_name}"}
                else:
                    effect_key = f"{start.run_id}:{step}:{call.call_id}"
                    arguments_hash = hashlib.sha256(json.dumps(call.arguments, sort_keys=True).encode()).hexdigest()
                    result = self.store.effect(effect_key)
                    if result is None:
                        result = self._execute_tool_with_retry(
                            tool,
                            call.arguments,
                            effect_key,
                            start.run_id,
                            step,
                        )
                        self.store.record_effect(effect_key, call.tool_name, arguments_hash, result, "applied" if result.get("ok") else "failed")
                        if call.tool_name == "propose_self_improvement" and not result.get("ok"):
                            failure_key = hashlib.sha256(
                                json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
                            ).hexdigest()
                            if failure_key in failed_proposals:
                                result = {
                                    "ok": False,
                                    "error": "identical self-improvement failure already occurred in this run; continue with a different bounded task",
                                    "failure_class": "repeat_guard",
                                    "do_not_retry_unchanged": True,
                                }
                            failed_proposals.add(failure_key)
                result = self._apply_worktree_warning(result, call.tool_name, start.run_id, step)
                log.info("run=%s tool=%s ok=%s", start.run_id, call.tool_name, result.get("ok"))
                self.store.append_event(
                    "tool_result",
                    {
                        "call": {
                            "call_id": call.call_id,
                            "tool_name": call.tool_name,
                            "arguments": call.arguments,
                        },
                        "result": self._bounded_result(result),
                    },
                    start.run_id,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": self._bounded_tool_content(result),
                })
                # A hard denial is checked first: a result carries only one of
                # these today, but if a tool ever returns several flags the hard
                # decision must not be shadowed by the soft one.
                if result.get("policy_denied"):
                    # The sibling branches below emit an event, so the metrics
                    # allowlist (skynet/metrics.py) and the retention set
                    # (skynet/store.py PROTECTED_EVENT_KINDS) both count
                    # 'policy_denied' - yet nothing ever wrote it, so a hard
                    # denial was invisible to every reporting surface and lived
                    # only inside the tool_result payload. A permission decision
                    # must be recorded as its own event.
                    self.store.append_event(
                        "policy_denied",
                        {"tool": call.tool_name, "protected_path": result.get("protected_path")},
                        start.run_id,
                    )
                elif result.get("policy_warning"):
                    self.store.append_event(
                        "policy_soft_denied",
                        {"tool": call.tool_name, "protected_path": result.get("protected_path")},
                        start.run_id,
                    )
                elif result.get("policy_override"):
                    self.store.append_event(
                        "policy_override_confirmed",
                        {"tool": call.tool_name, "arguments": call.arguments},
                        start.run_id,
                    )
                control_action = result.get("control_action")
                if isinstance(control_action, dict) and control_action.get("type") == "restart_after_checkpoint":
                    deferred_control = dict(control_action)
                    self.store.append_event("deferred_control", deferred_control, start.run_id)
                    status, report, finish_usage = self._finish(messages, start.run_id, reason="deferred restart after checkpoint")
                    if status in {RunStatus.NEEDS_RECOVERY, RunStatus.BLOCKED}:
                        # The promotion already succeeded and is tool-verified; a
                        # harness stop or a cautious model verdict must not
                        # record the successful promotion as a failed task.
                        status = RunStatus.COMPLETED
                        report = "Self-improvement promoted; restart deferred until memory and checkpoint complete."
                    return AgentRunResult(
                        status,
                        report,
                        step + 1,
                        used_tokens + finish_usage,
                        "deferred restart",
                        deferred_control,
                    )
        self._record_history(start.run_id, messages)
        self.store.append_event("budget_exhausted", {"budget": "steps", "steps": steps_limit}, start.run_id)
        return AgentRunResult(RunStatus.NEEDS_RECOVERY, "ReAct step budget exhausted", steps_limit, used_tokens, "step budget")

    def _context_budget_result(self, messages: list[Message], run_id: str, step: int, used_tokens: int) -> AgentRunResult:
        status, report, finish_usage = self._emergency_finish(messages, run_id)
        return AgentRunResult(status, report, step, used_tokens + finish_usage, "context budget")

    def _finish(self, messages: list[Message], run_id: str, *, reason: str = "normal") -> tuple[RunStatus, str, int]:
        """Force a Finish turn and return the parsed status, report, and usage."""
        self._record_history(run_id, messages)
        self.store.append_event("react_phase", {"phase": "finish", "reason": reason}, run_id)
        finish_messages = self._finish_messages(messages)
        finish_messages.append({"role": "system", "content": "Finish phase. Do not call tools. Return only one JSON object matching the Finish Report contract in the ReAct system prompt."})
        reserve = min(self.config.context_finish_reserve, max(0, self.config.max_tokens // 8))
        if self._request_tokens(finish_messages, []) + reserve >= self.config.max_tokens:
            return self._local_finish(run_id, reason, "Finish context exceeded the per-request limit")
        try:
            turn, _turn_seconds = self._complete_with_retry(finish_messages, max_tokens=self.config.output_tokens, tools=[], run_id=run_id, step=-1)
        except Exception as exc:
            self.store.append_event("finish_failure", {"error": str(exc)[:1000]}, run_id)
            return self._local_finish(run_id, reason, f"Provider Finish failed: {str(exc)[:300]}")
        report = turn.text.strip()
        self._record_history(run_id, [*messages, {"role": "assistant", "content": report}])
        if not report:
            return RunStatus.NEEDS_RECOVERY, f"Finish Report missing ({reason})", turn.total_tokens
        status, report = self._parse_finish_report(report, run_id)
        if status == RunStatus.NEEDS_RECOVERY:
            return status, report, turn.total_tokens
        self.store.append_event("finish_report", {"text": report, "usage_tokens": turn.total_tokens, "prompt_tokens": turn.prompt_tokens, "completion_tokens": turn.completion_tokens, "forced": reason != "normal", "reason": reason}, run_id)
        return status, report, turn.total_tokens

    def _parse_finish_report(self, report: str, run_id: str) -> tuple[RunStatus, str]:
        try:
            data = parse_json_object(report)
            # Accept the old minimal status object while providers roll forward;
            # new responses still use the closed Finish contract.
            if data.get("status") in {"completed", "COMPLETED", "blocked", "BLOCKED"} and not {"evidence", "actions", "changes", "tests", "blocker", "next_hypothesis"} <= set(data):
                status = str(data["status"]).upper()
                summary = str(data.get("summary", "")).strip()
                data = {"status": status, "summary": summary or "legacy report", "evidence": [summary or "legacy report"], "actions": [], "changes": [], "tests": [], "blocker": summary if status == "BLOCKED" else "", "next_hypothesis": ""}
            validate_shape(data, FINISH_REPORT_SCHEMA)
            if data["status"] == "COMPLETED" and not data["evidence"]:
                raise ValueError("COMPLETED Finish Report requires evidence")
            if data["status"] == "BLOCKED" and not data["blocker"]:
                raise ValueError("BLOCKED Finish Report requires blocker")
            normalized = json.dumps(data, ensure_ascii=False)
            return RunStatus(str(data["status"]).lower()), normalized
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            # A prose reply is not a Finish Report. Accepting any non-JSON text
            # as COMPLETED marked tasks done with no durable change, so a broken
            # report must stay recoverable instead of closing the task.
            self.store.append_event(
                "finish_invalid",
                {"error": str(exc)[:500], "report_prefix": report[:200]},
                run_id,
            )
            return RunStatus.NEEDS_RECOVERY, f"Finish Report invalid: {exc}"

    def _local_finish(self, run_id: str, reason: str, detail: str) -> tuple[RunStatus, str, int]:
        report = f"Finish Report (harness): cycle stopped safely. Reason: {reason}. {detail}. Full evidence remains in the durable event log."
        self.store.append_event("finish_report", {"text": report, "usage_tokens": 0, "forced": True, "local": True, "reason": reason}, run_id)
        # A harness-generated stop is not verified work; keep the run
        # recoverable so the task is retried instead of silently completed.
        # The deferred-restart path upgrades this to COMPLETED only when a
        # promotion actually succeeded.
        return RunStatus.NEEDS_RECOVERY, report, 0

    def _emergency_finish(self, messages: list[Message], run_id: str) -> tuple[RunStatus, str, int]:
        self.store.append_event("emergency_finish", {"reason": "context budget"}, run_id)
        return self._finish(messages, run_id)

    def _complete_with_retry(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        tools: list[dict[str, Any]],
        run_id: str,
        step: int,
    ) -> tuple[ModelTurn, float]:
        attempt = 0
        while True:
            try:
                request_chars, request_sha256 = self._request_fingerprint(messages)
                request_meta = {
                    "step": step + 1,
                    "attempt": attempt + 1,
                    "message_count": len(messages),
                    "tool_count": len(tools),
                    "request_chars": request_chars,
                    "request_sha256": request_sha256,
                    "model": self._provider_model(),
                    "max_output_tokens": max_tokens,
                }
                self.store.runtime_log.write("provider_request", request_meta, run_id=run_id)
                if verbose_enabled(self.store.runtime_log.path):
                    verbose_write(
                        "provider_request",
                        {"step": step + 1, "attempt": attempt + 1, "messages": messages, "tools": tools, "model": self._provider_model(), "max_output_tokens": max_tokens},
                        base_path=self.store.runtime_log.path,
                        run_id=run_id,
                    )
                # The transcript keeps the request identity, not the payload: storing
                # the whole message list on every attempt grows the database
                # quadratically with the number of steps.
                self.store.append_transcript("provider_request", request_meta, run_id)
                call_started = time.monotonic()
                turn = self.provider.complete(messages, max_tokens=max_tokens, tools=tools)
                # A fallback chain reports model-only time; wall clock would also
                # charge its backoff sleeps to the run's model budget.
                reported_seconds = float(getattr(turn, "model_seconds", 0.0) or 0.0)
                turn_seconds = reported_seconds if reported_seconds > 0.0 else max(0.0, time.monotonic() - call_started)
                self.store.runtime_log.write(
                    "provider_response",
                    {
                        "step": step + 1,
                        "attempt": attempt + 1,
                        "text_preview": turn.text[:1000],
                        "text_chars": len(turn.text),
                        "tool_call_count": len(turn.tool_calls),
                        "tool_names": [call.tool_name for call in turn.tool_calls],
                        "usage_tokens": turn.total_tokens,
                        "prompt_tokens": turn.prompt_tokens,
                        "completion_tokens": turn.completion_tokens,
                    },
                    run_id=run_id,
                )
                if verbose_enabled(self.store.runtime_log.path):
                    verbose_write(
                        "provider_response",
                        {"step": step + 1, "attempt": attempt + 1, "text": turn.text, "tool_calls": [
                            {"call_id": call.call_id, "tool_name": call.tool_name, "arguments": call.arguments}
                            for call in turn.tool_calls
                        ], "usage_tokens": turn.total_tokens},
                        base_path=self.store.runtime_log.path,
                        run_id=run_id,
                    )
                self.store.append_transcript(
                    "provider_response",
                    {"step": step, "attempt": attempt + 1, "text": turn.text, "tool_calls": [
                        {"call_id": call.call_id, "tool_name": call.tool_name, "arguments": call.arguments}
                        for call in turn.tool_calls
                    ], "usage_tokens": turn.total_tokens, "prompt_tokens": turn.prompt_tokens, "completion_tokens": turn.completion_tokens},
                    run_id,
                )
                return turn, turn_seconds
            except Exception as exc:
                self.store.runtime_log.write(
                    "provider_error",
                    {"step": step + 1, "attempt": attempt + 1, "error": str(exc)[:1000]},
                    run_id=run_id,
                )
                self.store.append_transcript("provider_error", {"step": step, "attempt": attempt + 1, "error": str(exc)[:1000]}, run_id)
                retryable = not isinstance(exc, ProviderError) or exc.retryable
                if isinstance(exc, ProviderError):
                    self.store.append_event("provider_error_classified", {"step": step + 1, "category": exc.category, "retryable": exc.retryable, "cooldown_seconds": exc.cooldown_seconds}, run_id)
                if not retryable or attempt >= self.config.provider_retries:
                    raise
                self.store.append_event(
                    "provider_retry",
                    {"step": step + 1, "attempt": attempt + 1, "error": str(exc)[:1000]},
                    run_id,
                )
                self._backoff(self.config.provider_retry_backoff_seconds, attempt)
                attempt += 1

    def _execute_tool_with_retry(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        effect_key: str,
        run_id: str,
        step: int,
    ) -> dict[str, Any]:
        retries = self.config.read_tool_retries if getattr(tool, "capability_kind", "read") == "read" else 0
        attempt = 0
        with tracked_process_group() as groups:
            interrupted = False
            try:
                while True:
                    try:
                        return tool.execute(arguments, idempotency_key=effect_key)
                    except Exception as exc:
                        if attempt >= retries:
                            self.store.append_event(
                                "tool_failure",
                                {"step": step + 1, "tool": tool.name, "error": str(exc)[:1000]},
                                run_id,
                            )
                            return {"ok": False, "error": f"tool execution failed: {str(exc)[:500]}"}
                        self.store.append_event(
                            "tool_retry",
                            {"step": step + 1, "tool": tool.name, "attempt": attempt + 1, "error": str(exc)[:1000]},
                            run_id,
                        )
                        self._backoff(self.config.tool_retry_backoff_seconds, attempt)
                        attempt += 1
            except BaseException:
                # A WatchdogTimeout (SIGALRM) or a SIGTERM-driven interrupt must
                # not leave the tool's child process group running. Kill it here,
                # then re-raise untouched so the original exception is never
                # masked. Handled Exceptions and successful returns kill nothing.
                interrupted = True
                raise
            finally:
                if interrupted:
                    _terminate_process_groups(groups)

    def _worktree_dirty_files(self, step: int) -> list[str]:
        """Return modified tracked files, cached once per step."""
        if self.worktree_root is None:
            return []
        if self._worktree_status_step == step:
            return self._worktree_status
        self._worktree_status_step = step
        self._worktree_status = self._read_worktree_status()
        return self._worktree_status

    def _read_worktree_status(self) -> list[str]:
        """Run a bounded ``git status``; any failure is silent and means clean."""
        try:
            completed = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=str(self.worktree_root),
                capture_output=True,
                text=True,
                timeout=WORKTREE_STATUS_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if completed.returncode != 0:
            return []
        files: list[str] = []
        for line in completed.stdout.splitlines():
            # Porcelain format is "XY <path>"; skip the two status columns.
            path = line[3:].strip() if len(line) > 3 else ""
            if path:
                files.append(path)
        return files

    def _apply_worktree_warning(self, result: dict[str, Any], tool_name: str, run_id: str, step: int) -> dict[str, Any]:
        """Warn the model in-band when bash leaves tracked files modified."""
        if tool_name != "bash" or self.worktree_root is None:
            return result
        files = self._worktree_dirty_files(step)
        if not files:
            return result
        listed = files[:WORKTREE_WARNING_MAX_FILES]
        self.store.append_event(
            "worktree_dirty_after_bash",
            {"step": step + 1, "files": listed, "count": len(files)},
            run_id,
        )
        warning = (
            "bash left modified tracked files in the main worktree ("
            + ", ".join(listed)
            + "). Code changes are durable only through propose_self_improvement; "
            "revert unintended edits before proposing."
        )
        updated = dict(result)
        updated["worktree_warning"] = warning
        return updated

    @staticmethod
    def _backoff(base: float, attempt: int) -> None:
        delay = max(0.0, min(float(base) * (2**attempt), 30.0))
        if delay:
            time.sleep(delay)

    @staticmethod
    def _tail_start(messages: list[Message], keep: int) -> int:
        """Return a tail start that never splits a tool call from its result."""
        start = min(len(messages), max(2, len(messages) - max(0, keep)))
        while start > 2 and start < len(messages) and messages[start].get("role") == "tool":
            start -= 1
        return start

    def _finish_messages(self, messages: list[Message]) -> list[Message]:
        keep = max(2, self.config.compaction_keep_messages)
        return messages[:2] + messages[self._tail_start(messages, keep):]

    @staticmethod
    def _request_tokens(messages: list[Message], tools: list[dict[str, Any]]) -> int:
        """Conservative provider-independent estimate including tool schemas."""
        serialized = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False, separators=(",", ":"))
        return max(1, (len(serialized) + 2) // 3)

    @staticmethod
    def _request_fingerprint(messages: list[Message]) -> tuple[int, str]:
        """Serialise the request once for both its size and its hash."""
        serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        return len(serialized), hashlib.sha256(serialized.encode()).hexdigest()

    def _provider_model(self) -> str:
        """Best-effort model identity for request telemetry.

        A single provider exposes ``model``; the fallback wrapper does not, so
        the first chain member stands in for the chain. The class name is the
        last resort and keeps the telemetry key present for every provider.
        """
        model = getattr(self.provider, "model", None)
        if isinstance(model, str) and model:
            return model
        providers = getattr(self.provider, "providers", None)
        if isinstance(providers, (list, tuple)) and providers:
            nested = getattr(providers[0], "model", None)
            if isinstance(nested, str) and nested:
                return nested
        name = getattr(self.provider, "name", None)
        return str(name) if name else type(self.provider).__name__

    def _record_history(self, run_id: str, messages: list[Message]) -> None:
        sanitized = [
            {key: value for key, value in message.items() if key != "reasoning_content"}
            for message in messages
        ]
        self.store.set_transcript("react_history", {"messages": sanitized}, run_id)

    def _compact_messages(self, messages: list[Message]) -> list[Message]:
        """Keep the durable Start and a bounded tail; never invent a model summary."""
        keep = max(2, self.config.compaction_keep_messages)
        if len(messages) <= keep:
            return messages
        start = self._tail_start(messages, keep - 3)
        compacted = [*messages[:2], {"role": "system", "content": "[Context compacted by harness. Earlier tool exchanges were persisted in the event log.]"}, *messages[start:]]
        if self._request_tokens(compacted, []) >= self._request_tokens(messages, []):
            return messages
        return compacted

    def _bounded_result(self, result: dict[str, Any]) -> dict[str, Any]:
        content = json.dumps(result, ensure_ascii=False)
        if len(content) <= self.config.tool_result_max_chars:
            return result
        return {"ok": bool(result.get("ok")), "truncated": True, "preview": content[: self.config.tool_result_max_chars]}

    def _bounded_tool_content(self, result: dict[str, Any]) -> str:
        return json.dumps(self._bounded_result(result), ensure_ascii=False)
