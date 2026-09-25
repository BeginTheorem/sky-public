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
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from .model_contracts import FINISH_REPORT_SCHEMA, clamp_to_schema, parse_json_object, validate_shape
from .models import AgentRunResult, ModelTurn, RunStatus, StartEnvelope
from .provider import LLMProvider, Message, Tool
from .providers.errors import ProviderError
from .runtime_log import verbose_enabled, verbose_write
from .store import StateStore

log = logging.getLogger("skynet.react")
MAX_RESPONSE_LOG_PREVIEW = 2000
WORKTREE_STATUS_TIMEOUT_SECONDS = 5.0
WORKTREE_WARNING_MAX_FILES = 20
RUN_PROGRESS_MESSAGE_LIMIT = 300
RUN_PROGRESS_TOOL_LIMIT = 60

# Called by the runner when model time crosses a progress boundary. The payload
# is deliberately a plain dict of already-derived facts (no store or outbox
# knowledge), so the runner stays the ReAct owner and the Reactor keeps being the
# only writer of the owner outbox.
RunProgressCallback = Callable[[dict[str, Any]], None]


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
    # model only learns the budget is gone once the episode is discarded
    # (arXiv:2604.01664: budget-conditioned control beats the budget-free form).
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
    # An owner message that arrives mid-episode is invisible to a loop that reads
    # the inbox only when the envelope is built. Measured on this repository: a
    # message posted while an episode ran never entered that run's observations
    # and was found only by querying the inbox table directly. The literature
    # says the cooperative form is the weak one (arXiv:2606.06460v4, Exp. 4: a
    # mid-task notice riding tool output was never acknowledged, 0/20; agents'
    # own stop rate was 28/120 = 23% and model-dependent, while a harness-level
    # interceptor stopped 120/120 with no false trips). Delivery is therefore the
    # harness's job, not a tool the model must remember to call. Injected as a
    # system message at a step boundary; one event is delivered at most once.
    inbox_delivery_max_per_step: int = 3
    # A two-hour run leaves the owner channel silent until Finish. Emit a short
    # factual progress note at most once per this interval of MODEL time (0
    # disables it). Model time, not wall time, so provider backoff is never
    # billed as progress. The count is additionally capped at
    # ``run budget // interval`` and stops once the Finish phase begins.
    run_progress_seconds: float = 3600.0


class ReActRunner:
    def __init__(
        self,
        provider: LLMProvider,
        store: StateStore,
        tools: Mapping[str, Tool],
        config: ReActConfig | None = None,
        *,
        worktree_root: str | os.PathLike[str] | None = None,
        on_progress: RunProgressCallback | None = None,
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
        # Owner progress emission is opt-in: without a callback the runner is
        # byte-for-byte as before, and the Reactor is the only outbox writer.
        self._on_progress = on_progress
        self._progress_next_at = 0.0
        self._progress_sent = 0
        self._finish_phase = False

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
        last_tool = ""
        response_fingerprints: dict[str, int] = {}
        deferred_control: dict[str, Any] = {}
        failed_proposals: set[str] = set()
        self._worktree_status_step = None
        self._worktree_status = []
        self._finish_phase = False
        self._progress_sent = 0
        progress_interval = float(self.config.run_progress_seconds)
        self._progress_next_at = progress_interval if progress_interval > 0 else 0.0
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

        # Only the notifications the START ENVELOPE actually carried count as
        # already delivered; anything else in the inbox is new input. Deriving the
        # cursor from the envelope (rather than re-reading the inbox) also closes
        # the race where a message arrives between envelope build and this line.
        delivered_inbox: set[str] = set()
        for observation in start.observations:
            if isinstance(observation, dict) and observation.get("kind") == "inbox_notification":
                delivered_inbox.add(str(observation.get("event_id", "")))
        self._record_envelope_delivery(delivered_inbox, start.run_id)
        for step in range(steps_limit):
            self._deliver_new_inbox(messages, delivered_inbox, start.run_id)
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
                context_warning_high_sent = True            # The time budget was enforced but never disclosed: measured live,
            # 13 runs died at ~1800-2100s of model time with "time budget
            # exhausted before tool execution". Tell the model once, in band, so
            # it can close the episode with a real Finish Report.
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
            # Emitted after every finish-triggering guard above, so a run that is
            # about to close never sends a "still running" note, and never after
            # the Finish phase has begun.
            self._emit_run_progress(start.run_id, step, model_seconds, last_tool, time_limit)
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
                # A provider that dies mid-episode used to return this prose
                # string as the whole report, erasing the tool calls the episode
                # had already executed. That is the same erasure the unreadable-
                # finish path was repaired for, at a second site: measured
                # on the live ledger, 7 of the 11 needs_recovery runs
                # died here with 26-85 successful tool calls each (a352322c: 74
                # over 45 steps / 3.2M tokens; d01cecd8: 85 over 38) and a report
                # naming none of them -- seven times the volume of the path just
                # repaired. The status stays NEEDS_RECOVERY and the "Provider
                # unavailable: ..." prefix is kept verbatim, because reactor.py
                # persists this report and downstream readers match the prefix;
                # only the loss of the execution record is removed.
                return AgentRunResult(
                    RunStatus.NEEDS_RECOVERY,
                    self._provider_failure_report(f"Provider unavailable: {str(exc)[:500]}", start.run_id, messages),
                    step,
                    used_tokens,
                    str(exc),
                )
            model_seconds += turn_seconds
            used_tokens += turn.total_tokens
            # Persist the running totals each turn: the in-memory counters are
            # lost on a SIGTERM/watchdog kill, which used to leave the run's
            # ledger row at steps=0, tokens=0.
            self.store.touch_run(start.run_id, phase, steps=step + 1, usage_tokens=used_tokens)
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
                self._finish_phase = True
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
                parsed_report = self._read_finish_report(report, start.run_id)
                if parsed_report is not None:
                    status, report = parsed_report
                    return AgentRunResult(status, report, step + 1, used_tokens)
                # The reply is unreadable, but this episode already executed tools.
                # Ask once more before discarding that work: a zero-length or prose
                # closing turn must not be the only chance to report it.
                repaired = self._repair_finish_report(report, start.run_id, messages, step=step)
                if repaired is not None:
                    status, report, repair_usage = repaired
                    self.store.append_event("finish_report", {"step": step + 1, "text": report, "usage_tokens": repair_usage, "repaired": True}, start.run_id)
                    return AgentRunResult(status, report, step + 1, used_tokens + repair_usage)
                status, report = self._parse_finish_report(report, start.run_id)
                if status == RunStatus.NEEDS_RECOVERY:
                    status, report = self._carry_evidence_forward(report, start.run_id, messages)
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
                    # The queued call is refused, but the calls the episode
                    # already executed must not vanish from its report: measured
                    # in the sweep test, this exit discarded a successful tool
                    # result before the carrier was wired in.
                    status, report = self._carry_evidence_forward(
                        "ReAct time budget exhausted before tool execution",
                        start.run_id,
                        messages,
                        situation="time_budget_before_tool",
                    )
                    return AgentRunResult(status, report, step + 1, used_tokens, "time budget")
                tool = self.tools.get(call.tool_name)
                if tool is None:
                    result = {"ok": False, "error": f"unknown tool: {call.tool_name}"}
                    result_address = ""
                else:
                    effect_key = f"{start.run_id}:{step}:{call.call_id}"
                    result_address = effect_key
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
                        self.store.record_effect(effect_key, call.tool_name, arguments_hash, result, self.store.effect_status(result))
                        previous_runs = self.store.effect_reapplied_runs(call.tool_name, arguments_hash, exclude_run_id=start.run_id)
                        # `previous_runs` is run-scoped by construction, so a call
                        # re-issued inside this run with a regenerated call_id was
                        # invisible: 15 of the live ledger's 58 repeated
                        # identities repeat only inside one run,
                        # 15/15 with differing results, and no event named any of
                        # them. The run-agnostic count is what makes that cell of
                        # the effect-exactly-once violation (arXiv:2608.03836v3)
                        # visible without serving a recorded result.
                        prior_application_count = self.store.effect_identity_prior_count(
                            call.tool_name, arguments_hash, exclude_effect_key=effect_key
                        )
                        # `previous_runs` grows with every later re-application, so
                        # `if previous_runs:` re-emitted this event for each repeat
                        # and reported one fact 2, 3, ... times (measured: three
                        # identical executions produced two events for one
                        # identity). The announcement is once per identity: a
                        # non-empty list alone is not enough (an identity that had
                        # already repeated twice before this signal existed would
                        # never announce), so it is paired with the durable
                        # `effect_reapplied` pointer, which is a protected event
                        # kind and therefore survives retention. This event is a
                        # pointer, not a running total: its `previous_run_count` is
                        # the number of other runs at announcement time, 1 for the
                        # common transition and more for a late-announced identity,
                        # so a reader that needs the current number of other runs
                        # re-derives it from `effect_reapplied_runs` with the
                        # payload's `arguments_hash` (measured on the live ledger:
                        # 12 identities repeat across runs, and the
                        # one-time event understates the worst of them, a `db` query
                        # applied by 10 runs, as 1).
                        if (previous_runs or prior_application_count) and not self.store.effect_reapplied_announced(call.tool_name, arguments_hash):
                            self.store.append_event(
                                "effect_reapplied",
                                {
                                    "step": step + 1,
                                    "tool": call.tool_name,
                                    "capability_kind": getattr(tool, "capability_kind", "read"),
                                    "arguments_hash": arguments_hash,
                                    "previous_run_ids": previous_runs[:3],
                                    "previous_run_count": len(previous_runs),
                                    # Zero here means the repeat is inside this
                                    # run, not across runs: the two are different
                                    # recovery cells, so the post-mortem keeps
                                    # them separable.
                                    "prior_application_count": prior_application_count,
                                    "same_run_repeat": not previous_runs and prior_application_count > 0,
                                },
                                start.run_id,
                            )
                        # A protected-path warning is not a failure: the identical
                        # resubmission is the intended next step, so it must not
                        # be blocked by the repeat guard.
                        if call.tool_name == "propose_self_improvement" and not result.get("ok") and not result.get("warned"):
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
                last_tool = call.tool_name
                log.info("run=%s tool=%s ok=%s", start.run_id, call.tool_name, result.get("ok"))
                self.store.append_event(
                    "tool_result",
                    {
                        "call": {
                            "call_id": call.call_id,
                            "tool_name": call.tool_name,
                            "arguments": call.arguments,
                        },
                        "result": self._bounded_result(result, result_address),
                    },
                    start.run_id,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": self._bounded_tool_content(result, result_address),
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
        # Measured on the unmodified tree: an episode that burned its step budget
        # after a successful tool call (the live probe used two) returned this
        # bare prose and named none of them. The statuses and the failure label
        # are unchanged; only the executed-evidence record is no longer dropped.
        status, report = self._carry_evidence_forward(
            "ReAct step budget exhausted", start.run_id, messages, situation="step_budget"
        )
        return AgentRunResult(status, report, steps_limit, used_tokens, "step budget")

    def _context_budget_result(self, messages: list[Message], run_id: str, step: int, used_tokens: int) -> AgentRunResult:
        status, report, finish_usage = self._emergency_finish(messages, run_id)
        return AgentRunResult(status, report, step, used_tokens + finish_usage, "context budget")

    def _finish(self, messages: list[Message], run_id: str, *, reason: str = "normal") -> tuple[RunStatus, str, int]:
        """Force a Finish turn and return the parsed status, report, and usage."""
        self._finish_phase = True
        self._record_history(run_id, messages)
        self.store.append_event("react_phase", {"phase": "finish", "reason": reason}, run_id)
        finish_messages = self._finish_messages(messages)
        finish_messages.append({"role": "system", "content": "Finish phase. Do not call tools. Return only one JSON object matching the Finish Report contract in the ReAct system prompt."})
        reserve = min(self.config.context_finish_reserve, max(0, self.config.max_tokens // 8))
        if self._request_tokens(finish_messages, []) + reserve >= self.config.max_tokens:
            return self._local_finish(messages, run_id, reason, "Finish context exceeded the per-request limit")
        try:
            turn, _turn_seconds = self._complete_with_retry(finish_messages, max_tokens=self.config.output_tokens, tools=[], run_id=run_id, step=-1)
        except Exception as exc:
            self.store.append_event("finish_failure", {"error": str(exc)[:1000]}, run_id)
            return self._local_finish(messages, run_id, reason, f"Provider Finish failed: {str(exc)[:300]}")
        report = turn.text.strip()
        self._record_history(run_id, [*messages, {"role": "assistant", "content": report}])
        parsed_report = self._read_finish_report(report, run_id)
        if parsed_report is None:
            # The context was just reserved for a Finish request, so the one
            # bounded repair turn is affordable by construction.
            repaired = self._repair_finish_report(report, run_id, finish_messages, step=-1, force=True)
            if repaired is not None:
                status, report, repair_usage = repaired
                self.store.append_event("finish_report", {"text": report, "usage_tokens": repair_usage, "forced": reason != "normal", "reason": reason, "repaired": True}, run_id)
                return status, report, turn.total_tokens + repair_usage
            if not report:
                status, carried = self._carry_evidence_forward(f"Finish Report missing ({reason})", run_id, messages)
                return status, carried, turn.total_tokens
            status, report = self._parse_finish_report(report, run_id)
            if status == RunStatus.NEEDS_RECOVERY:
                status, report = self._carry_evidence_forward(report, run_id, messages)
            return status, report, turn.total_tokens
        status, report = parsed_report
        self.store.append_event("finish_report", {"text": report, "usage_tokens": turn.total_tokens, "prompt_tokens": turn.prompt_tokens, "completion_tokens": turn.completion_tokens, "forced": reason != "normal", "reason": reason}, run_id)
        return status, report, turn.total_tokens

    def _parse_finish_report(self, report: str, run_id: str, *, record_failure: bool = True) -> tuple[RunStatus, str]:
        try:
            data = parse_json_object(report)
            # Accept the old minimal status object while providers roll forward;
            # new responses still use the closed Finish contract.
            if data.get("status") in {"completed", "COMPLETED", "blocked", "BLOCKED"} and not {"evidence", "actions", "changes", "tests", "blocker", "next_hypothesis"} <= set(data):
                status = str(data["status"]).upper()
                summary = str(data.get("summary", "")).strip()
                data = {"status": status, "summary": summary or "legacy report", "evidence": [summary or "legacy report"], "actions": [], "changes": [], "tests": [], "blocker": summary if status == "BLOCKED" else "", "next_hypothesis": ""}
            # An over-length summary or citation, or more citations than the
            # contract allows, is a wording problem, not a broken report: clamp
            # it and keep the episode's real status. Only structural violations
            # still downgrade the run to recovery.
            clamped = clamp_to_schema(data, FINISH_REPORT_SCHEMA)
            if clamped:
                self.store.append_event(
                    "finish_clamped",
                    {"paths": clamped, "report_chars": len(report)},
                    run_id,
                )
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
            if record_failure:
                self.store.append_event(
                    "finish_invalid",
                    {"error": str(exc)[:500], "report_prefix": report[:200]},
                    run_id,
                )
            return RunStatus.NEEDS_RECOVERY, f"Finish Report invalid: {exc}"

    def _read_finish_report(self, report: str, run_id: str) -> tuple[RunStatus, str] | None:
        """Parse a closing reply without recording a failure.

        The caller must tell "unreadable, ask once more" from "unreadable, give
        up", so the failure event is written by the caller, not the parser.
        """
        status, parsed = self._parse_finish_report(report, run_id, record_failure=False)
        if status == RunStatus.NEEDS_RECOVERY:
            return None
        return status, parsed

    @staticmethod
    def _episode_has_durable_work(messages: list[Message]) -> bool:
        """True when the episode executed at least one tool call.

        Only such an episode has work that discarding its report would destroy:
        a run that executed nothing loses nothing by being retried normally.
        """
        return any(message.get("role") == "tool" for message in messages)

    @staticmethod
    def _succeeded_tool_names(messages: list[Message]) -> list[str]:
        """The tools the episode actually ran successfully, in order.

        Three conditions, each of which a weaker reading of "a tool ran" would
        drop. Only a ``tool`` message proves execution, because an assistant
        message carrying ``tool_calls`` is the request. A call the harness
        refused -- an unknown tool name, a policy denial -- still appends a
        ``tool`` message, so the result must also report ``ok`` as not False.
        The name is taken from the request that preceded the result, not from
        the payload, so a result cannot rename the tool that produced it.
        """
        names: list[str] = []
        requested: list[str] = []
        for message in messages:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    function = call.get("function") if isinstance(call, dict) else None
                    if isinstance(function, dict):
                        requested.append(str(function.get("name", "")))
            elif message.get("role") == "tool":
                name = requested.pop(0) if requested else ""
                try:
                    payload = json.loads(str(message.get("content", "")))
                except (TypeError, ValueError):
                    continue
                if not isinstance(payload, dict) or payload.get("ok") is False:
                    continue
                if name:
                    names.append(name)
        return names

    # Every non-COMPLETED AgentRunResult that can be reached after a successful
    # tool call must route its report through this carrier (or through ``_finish``
    # and ``_provider_failure_report``, which call it). The census of the sites is
    # asserted by ``test_every_agent_run_result_exit_carries_executed_evidence``;
    # this table is what makes a new situation a two-line addition instead of a
    # fourth copy of the same reasoning.
    EVIDENCE_CARRY_SITUATIONS: ClassVar[dict[str, tuple[str, str]]] = {
        "finish_unreadable": (
            "The closing turn was not a readable Finish Report and the repair turn failed",
            "the model returned no readable Finish Report after two attempts",
        ),
        "provider_unavailable": (
            "The model turn failed with the provider unavailable",
            "the provider was unavailable and no readable report could be produced",
        ),
        "time_budget_before_tool": (
            "The time budget was exhausted before the queued tool call could run",
            "the model-time budget ended the episode before a tool call could be executed",
        ),
        "step_budget": (
            "The step budget was exhausted before the episode could close",
            "the step budget ended the episode before it returned a readable report",
        ),
        "harness_stop": (
            "The harness stopped the cycle safely",
            "the harness stopped the cycle before the model returned a readable report",
        ),
    }

    def _carry_evidence_forward(
        self, report: str, run_id: str, messages: list[Message], *, situation: str = "finish_unreadable"
    ) -> tuple[RunStatus, str]:
        """Close an unreadable episode without discarding the work it did.

        The parser refuses prose and empty replies on purpose -- accepting any
        text as COMPLETED closed tasks with no durable change -- but the refusal
        used to replace the whole report with the parser's error string, so an
        episode that had executed dozens of tool calls ended with a report that
        named none of them. Measured on the live ledger: three runs
        (9b944e12, 8869a882, b01c5fdb) finished needs_recovery with
        ``failure=''`` and a report that was the bare string "Finish Report
        invalid: structured model response contains no acceptable JSON value",
        after 96, 28 and 49 successful tool results respectively. Only a
        successful call counts: a refused or unknown-tool call changed nothing,
        so its episode is the no-work control and keeps the bare rejection.

        The status stays NEEDS_RECOVERY -- a report nobody could read is still
        not a verified completion -- and only the evidence is carried over, so
        this can lose no verification, only stop erasing an execution record.
        An episode with no executed tool call is returned unchanged.
        """
        executed = self._succeeded_tool_names(messages)
        if not executed:
            return RunStatus.NEEDS_RECOVERY, report
        evidence = [f"executed tool: {name}" for name in executed]
        why, blocker = self.EVIDENCE_CARRY_SITUATIONS.get(
            situation, self.EVIDENCE_CARRY_SITUATIONS["finish_unreadable"]
        )
        carried = json.dumps({
            "status": "NEEDS_RECOVERY",
            "summary": (
                f"{why}; "
                f"{len(executed)} tool call(s) did execute, so their evidence is carried forward "
                f"instead of being discarded. Reported error: {report[:300]}"
            ),
            "evidence": evidence or ["no evidence recorded"],
            "actions": [f"tool call: {name}" for name in executed],
            "changes": [],
            "tests": [],
            "blocker": blocker,
            "next_hypothesis": "retry the finish turn with the episode's tool results; the work itself already ran",
        }, ensure_ascii=False)
        self.store.append_event(
            "finish_evidence_carried",
            {
                "tool_calls": len(executed),
                "tools": executed[:20],
                "reason": situation,
                "parser_error": report[:300],
            },
            run_id,
        )
        return RunStatus.NEEDS_RECOVERY, carried

    def _provider_failure_report(self, report: str, run_id: str, messages: list[Message]) -> str:
        """Keep the ``Provider unavailable: ...`` text and stop erasing the work.

        The prefix is preserved verbatim because ``reactor.py`` stores this
        return value as the run row's report and downstream readers match the
        prefix. What changes is that a report which carries executed evidence
        becomes a readable JSON object naming the tools the episode had already
        run, so a recovered episode holds its own record of what it did. An
        episode with no successful tool call keeps the bare prose string, exactly
        like the unreadable-finish control, and records no carried event.
        """
        if not self._succeeded_tool_names(messages):
            return report
        status, carried = self._carry_evidence_forward(report, run_id, messages, situation="provider_unavailable")
        if status != RunStatus.NEEDS_RECOVERY:
            return report
        try:
            data = json.loads(carried)
        except ValueError:
            return report
        data["summary"] = f"{report}. {data['summary']}"
        data["blocker"] = report
        data["next_hypothesis"] = (
            "retry the same episode once the provider is reachable; the tool work already ran and is recorded in evidence"
        )
        return json.dumps(data, ensure_ascii=False)

    def _repair_finish_report(
        self,
        report: str,
        run_id: str,
        messages: list[Message],
        *,
        step: int = -1,
        force: bool = False,
    ) -> tuple[RunStatus, str, int] | None:
        """One bounded repair turn when the closing reply is not a report.

        Live evidence: 4 of the 7 recorded ``finish_invalid`` events are replies
        the parser could not read - one zero-length turn and three prose turns -
        and each discarded an episode whose work was already durable (event_log
        6464 sits after that run's own promotion was committed). Detection
        without repair is the expensive half of the pair: closing detection into
        a re-run recovered 45% of flagged failures against a 16% resampling
        control, for about one extra model call per run (arXiv:2608.02464).

        Returns None when the episode has no work at risk, when the turn is
        unaffordable, when the provider fails, or when the retry is unreadable
        too. The caller then reports the original failure, so this can only
        recover a run, never close one without a valid report.
        """
        if not self._episode_has_durable_work(messages):
            self.store.append_event("finish_repair_skipped", {"reason": "no tool calls in the episode"}, run_id)
            return None
        if not force:
            reserve = min(self.config.context_finish_reserve, max(0, self.config.max_tokens // 8))
            if self._request_tokens(messages, []) + reserve >= self.config.max_tokens:
                self.store.append_event("finish_repair_skipped", {"reason": "context budget"}, run_id)
                return None
        unreadable = "the reply was empty" if not report.strip() else "the reply was not one JSON object"
        repair_messages = [
            *messages,
            {
                "role": "system",
                "content": (
                    f"Your previous closing reply was not a readable Finish Report ({unreadable}). "
                    "Do not call tools. Return only one JSON object matching the Finish Report "
                    "contract in the system prompt."
                ),
            },
        ]
        try:
            turn, _turn_seconds = self._complete_with_retry(
                repair_messages, max_tokens=self.config.output_tokens, tools=[], run_id=run_id, step=step
            )
        except Exception as exc:
            self.store.append_event("finish_repair_failed", {"error": str(exc)[:500]}, run_id)
            return None
        retry_report = turn.text.strip()
        parsed = self._read_finish_report(retry_report, run_id)
        if parsed is None:
            self.store.append_event(
                "finish_repair_failed",
                {"error": "retry reply was not a Finish Report", "report_chars": len(retry_report)},
                run_id,
            )
            return None
        status, parsed_report = parsed
        self.store.append_event(
            "finish_repair",
            {"usage_tokens": turn.total_tokens, "report_chars": len(retry_report)},
            run_id,
        )
        return status, parsed_report, turn.total_tokens

    def _local_finish(self, messages: list[Message], run_id: str, reason: str, detail: str) -> tuple[RunStatus, str, int]:
        report = f"Finish Report (harness): cycle stopped safely. Reason: {reason}. {detail}. Full evidence remains in the durable event log."
        self.store.append_event("finish_report", {"text": report, "usage_tokens": 0, "forced": True, "local": True, "reason": reason}, run_id)
        # A harness stop is not the episode's fault, but it is still a stop after
        # work: measured in the sweep test, this exit discarded six successful
        # tool results and returned the harness sentence alone. An episode with
        # no executed tool call keeps that sentence verbatim.
        _status, carried = self._carry_evidence_forward(report, run_id, messages, situation="harness_stop")
        report = carried
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
                        "finish_reason": turn.finish_reason,
                        # An empty text with a non-zero completion count is
                        # otherwise unreadable: reasoning_chars>0 means the
                        # provider spent the tokens on the reasoning channel,
                        # 0 means it really returned nothing.
                        "reasoning_chars": len(turn.reasoning_content),
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
                # A provider call that raised is a cycle-level deviation, not a
                # high-frequency attempt trace: keep it durable so a failed run
                # can be post-mortem'd after retention. append_event mirrors it
                # into the runtime projection under kind "event".
                self.store.append_event(
                    "provider_error",
                    {"step": step + 1, "attempt": attempt + 1, "error": str(exc)[:1000]},
                    run_id,
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

    def _record_envelope_delivery(self, delivered: set[str], run_id: str) -> None:
        """Record the inbox events the START ENVELOPE itself carried.

        The set is seeded from the envelope's notifications, so a message that
        reached the model through the envelope was already delivered -- yet only
        ``_deliver_new_inbox`` ever wrote ``inbox_delivered_in_run``. A message
        carried by the envelope therefore left no ledger entry while the run
        consumed it (``acknowledge_inbox`` sets ``inbox.consumed_at``), and the
        read-receipt ledger kept waiting for a delivery event that could never
        arrive: the owner's own chat still showed the "queued" reaction for a
        message the organism had already read and answered. Recording the
        envelope's ids closes that gap without delivering anything twice, because
        the same set is the ``_deliver_new_inbox`` cursor.
        """
        event_ids = sorted(item for item in delivered if item)
        if not event_ids:
            return
        self.store.append_event("inbox_delivered_in_run", {"count": len(event_ids), "event_ids": event_ids}, run_id)

    def _deliver_new_inbox(self, messages: list[Message], delivered: set[str], run_id: str) -> list[str]:
        """Inject owner messages that arrived after this episode started.

        The inbox is read once per wake, when the START ENVELOPE is built, so
        live input sent while the episode runs would otherwise stay invisible
        until the next wake. Reading it again at a step boundary and appending
        anything not already carried by the envelope is harness-level delivery:
        it does not depend on the model choosing to call a read tool. Bounded by
        ``inbox_delivery_max_per_step`` and failure-tolerant, because a delivery
        that cannot be read must never kill a running episode.
        """
        try:
            # Only owner messages are injected; the inbox also carries other
            # pending kinds (answers, notifications) that are not conversation.
            pending = [event for event in self.store.pending_inbox() if str(event.get("kind")) == "user_message"]
        except Exception:
            log.exception("inbox delivery read failed; continuing without it")
            return []
        fresh: list[dict[str, Any]] = []
        for event in pending:
            event_id = str(event.get("event_id", ""))
            if not event_id or event_id in delivered:
                continue
            delivered.add(event_id)
            fresh.append(event)
            if len(fresh) >= self.config.inbox_delivery_max_per_step:
                break
        if not fresh:
            return []
        lines: list[str] = []
        for event in fresh:
            payload = event.get("payload")
            text = ""
            if isinstance(payload, dict):
                for key in ("text", "answer", "message"):
                    candidate = payload.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        text = candidate.strip()
                        break
            lines.append(f"- [{event.get('kind', 'user_message')!s}] {text[:1500]}")
        # One inbox read per step boundary: the count is taken from the same
        # snapshot, so the read is reused rather than issued twice. `delivered`
        # now includes `fresh`, so the remainder is the not-yet-injected backlog.
        unread = len(fresh) + sum(1 for event in pending if str(event.get("event_id", "")) not in delivered)
        messages.append({
            "role": "system",
            "content": (
                f"You have {unread} unread owner message(s). "
                "New owner message(s) arrived while this episode is running. This is live input, "
                "not an order: read it, decide whether it changes the current work, and continue. "
                "No immediate reply is required.\n" + "\n".join(lines)
            ),
        })
        event_ids = [str(event.get("event_id", "")) for event in fresh]
        self.store.append_event("inbox_delivered_in_run", {"count": len(event_ids), "event_ids": event_ids}, run_id)
        return event_ids

    def _emit_run_progress(
        self,
        run_id: str,
        step: int,
        model_seconds: float,
        last_tool: str,
        time_limit: float,
    ) -> None:
        """Emit at most one short owner note per interval of model time.

        Bounded on four axes: a positive interval, at most one note per boundary
        crossing, a hard cap of ``time_limit // interval`` per episode, and a
        stop once the Finish phase begins. The note carries only facts the runner
        already owns -- completed step count, elapsed model minutes, the last
        tool name -- never an outcome claim. It is emitted before the model turn
        and after every finish-triggering guard, so it cannot fire from the
        Finish phase or race the finish response.
        """
        interval = float(self.config.run_progress_seconds)
        if interval <= 0 or self._on_progress is None or self._finish_phase:
            return
        cap = int(time_limit // interval)
        if cap <= 0 or self._progress_sent >= cap:
            return
        if model_seconds < self._progress_next_at:
            return
        # Advance past the current model time so a single long turn that spans
        # several boundaries still produces one note, never a burst.
        self._progress_next_at = (int(model_seconds // interval) + 1) * interval
        self._progress_sent += 1
        tool = str(last_tool or "").strip()[:RUN_PROGRESS_TOOL_LIMIT]
        minutes = int(model_seconds // 60)
        message = (
            f"Прогон продолжается: шаг {step}, прошло ~{minutes} мин модельного времени, "
            f"последний инструмент: {tool or '—'}."
        )[:RUN_PROGRESS_MESSAGE_LIMIT]
        payload = {
            "run_id": run_id,
            "step": step,
            "model_seconds": round(model_seconds, 1),
            "last_tool": tool,
            "message": message,
        }
        try:
            self._on_progress(payload)
        except Exception:
            # Owner progress is observability, not part of the run's accounting:
            # a broken callback must never kill a running episode.
            log.exception("run progress callback failed; continuing")

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

    def _bounded_result(self, result: dict[str, Any], effect_key: str = "") -> dict[str, Any]:
        """Bound a tool result for the context window, keeping it addressable.

        A result larger than `tool_result_max_chars` is replaced by a preview,
        which is what the model and every event-log reader see. The full result
        is not lost -- `record_effect` persists it verbatim in
        `capability_effects` under the run-scoped `effect_key` -- but until now
        nothing in the bounded form said where it went, so the removal was
        silent and the preview was the only surviving copy in the log.

        Measured on the live ledger (generation 40): of 2347
        `tool_result` events, 142 were truncated, holding 528,000 bytes of
        preview; all 142 full results were still present in `capability_effects`
        and would have been reachable from an emitted key.

        The key embeds the `step`, and neither the `tool_call` nor the
        `tool_result` event carries it -- yet the step is not needed to recover a
        payload, because `call_id` is unique within a run and `record_effect`
        stores exactly one row per `run_id:step:call_id`. Re-measured
        (generation 42) on the 33 folded events emitted before this
        key existed: all 33 resolve by `(run_id, call_id)` alone, 0 ambiguous,
        0 missing. The emitted key therefore buys a direct row name instead of a
        two-column join; what it is actually required for is keeping the
        eviction non-silent (arXiv:2608.21690, Scroll, Sec. 2.4: eviction is
        only safe under the invariant that everything removed stays
        addressable, with a pointer kept in place of the removed payload).

        A key is emitted only when one actually exists: unknown tools never call
        `record_effect`, so they pass an empty key and keep the old shape.
        """
        content = json.dumps(result, ensure_ascii=False)
        if len(content) <= self.config.tool_result_max_chars:
            return result
        bounded: dict[str, Any] = {"ok": bool(result.get("ok")), "truncated": True, "preview": content[: self.config.tool_result_max_chars]}
        if effect_key:
            bounded["effect_key"] = effect_key
            bounded["full_result_in"] = "capability_effects.idempotency_key"
        return bounded

    def _bounded_tool_content(self, result: dict[str, Any], effect_key: str = "") -> str:
        """The copy the model reads: bounded like the log copy, and addressed.

        This is the second call site of the same bounding helper. Until now it
        hard-coded the empty key, so a folded result reached the model as a bare
        preview while only the `event_log` copy carried `effect_key` (measured
        at generation 41: 34 folded `tool_result` events with 1 addressed, 33
        folded tool messages in `react_history` with 0). The model therefore saw
        a silent eviction it could neither detect nor undo, even though a `db`
        SELECT against `capability_effects.idempotency_key` can return the full
        payload. Passing the same run-scoped key keeps the two copies pointing
        at one row; the default keeps the one-argument call site unchanged.
        """
        return json.dumps(self._bounded_result(result, effect_key), ensure_ascii=False)
