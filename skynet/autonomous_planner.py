"""Bounded LLM proposal generation; execution remains deterministic and local."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .idea_archive import CELLS_TOTAL, CHANGE_TYPES, EVIDENCE_SOURCES, SUBSYSTEMS, learnability_defect
from .model_contracts import MAX_MODEL_REASKS, PLANNER_RESPONSE_SCHEMA, corrective_reask_instruction, validate_shape
from .models import ModelTurn
from .plan_tools import plan_tool_schemas
from .planner import (
    hypothesis_fingerprint,
    normalize_hypothesis_text,
    structural_fingerprint,
    title_prefix,
    title_similarity,
)
from .planner_contract import PLANNER_INSTRUCTION, PLANNER_RETRY_INSTRUCTION, PLANNER_STALL_INSTRUCTION, parse_planner_reply, planner_system_prompt
from .provider import LLMProvider, Message

# How much reconnaissance one planning pass may do. Rounds bound the model calls,
# calls-per-round bound one burst, and the result cap keeps a `git log` from filling the
# planner's window. All three are budgets, not limits the model is told about.
MAX_TOOL_ROUNDS = 4
MAX_TOOL_CALLS_PER_ROUND = 3
TOOL_RESULT_MAX_CHARS = 8_000


def plan_pass_token_ceiling(max_input_chars: int, output_tokens: int) -> int:
    """The token allowance ONE planning pass books -- and the ceiling its own
    mid-pass guard enforces. One formula, one number: the run row and the guard
    read it from here, so a heartbeat that re-sizes the planner's window moves
    both together and they cannot drift apart.

    It is the pass's worst case, not a prediction: the input ceiling converted
    with the ReAct runner's own ``//3`` estimator plus output room, with 10%
    headroom on top -- the same shape as the memory pass's figure.

    Why the guard is this number and not a smaller per-step share (measured
    2026-10-21): the old frozen constant was ``floor(274_688 / 8) * 0.95`` --
    the booking divided by the eight calls a pass may make. Every call re-sends
    its whole prompt, so a real per-call cost is roughly constant (~33-38k on
    the live ledger) while that share was 32.6k: the guard fired at step 2 with
    66-77k spent of a 238k allowance, 29 of 32 passes that day and 103 of 111
    the day before, each retried minutes later and killed again. A guard that
    trips with three quarters of the allowance unspent costs work instead of
    protecting it; sized at the booking, only a pass that outspends what it
    booked stops.
    """
    return int((max_input_chars // 3 + output_tokens) * 1.1)



class AutonomousPlanner:
    # A proposal whose title similarity to an already-RESOLVED proposal crosses
    # this ratio is the same question asked again. Neither number is a guess. On
    # the 80-proposal live ledger 0.55 is the highest threshold that still fires,
    # and it fires exactly once: the true positive "measure the BM25-vs-dense
    # top-k crossover N", asked 2026-09-24 and asked again 2026-09-25 minutes
    # after the first one answered it. 0.60 fires zero times, and whole-population
    # 0.55 adds one false positive -- the recurring "Pay the owner channel"
    # recovery habit at 0.64, whose re-issue is legitimate -- which is why
    # recovery is excluded by kind rather than the threshold raised.
    TITLE_PARAPHRASE_RATIO = 0.55
    TITLE_PARAPHRASE_KINDS = frozenset({"research", "validation"})

    def __init__(self, provider: LLMProvider, store: Any, *, output_tokens: int = 16_384, max_input_chars: int = 700_000, timeout_seconds: float = 120.0, max_active_goals: int = 8, hypothesis_ttl_days: float = 30.0, paraphrase_history_limit: int = 200) -> None:
        self.provider = provider
        self.store = store
        self.output_tokens = output_tokens
        self.max_input_chars = max_input_chars
        # The pass's token allowance: derived here from the window it was built
        # with, and moved by ``resize_window`` when the heartbeat re-sizes the
        # planner. The reactor books the plan run row off this same number.
        self.max_pass_tokens = plan_pass_token_ceiling(self.max_input_chars, self.output_tokens)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_active_goals = max(1, int(max_active_goals))
        self.hypothesis_ttl_days = max(0.0, float(hypothesis_ttl_days))
        # How many resolved proposals the paraphrase gate scans. Deliberately
        # larger than the bounded prompt's own view, because the model can only
        # avoid asking what it was shown, and its prompt is capped.
        self.paraphrase_history_limit = max(1, int(paraphrase_history_limit))
        # The outcome of the most recent generate() call. The reactor reads it
        # to distinguish "the planner crashed" from "the planner found nothing",
        # which the empty list return value cannot express on its own.
        self.last_status: str = ""

    def resize_window(self, max_input_chars: int) -> None:
        """Re-size the context window AND the pass's token allowance with it.

        The reactor calls this on every heartbeat that moves the active rung. A
        window that moves has to move the booking with it, or the pass is
        measured against the class of the previous heartbeat -- and a caller that
        assigns ``max_input_chars`` directly would move the window while leaving
        the allowance behind, which is exactly the drift this method removes.
        """
        self.max_input_chars = int(max_input_chars)
        self.max_pass_tokens = plan_pass_token_ceiling(self.max_input_chars, self.output_tokens)

    def _complete_bounded(self, messages: Sequence[Message], deadline: float, tools: Sequence[Any] = ()) -> ModelTurn:
        """Bound the planner provider call by the shared planning deadline.

        A stuck request must not freeze PLAN, and the deadline is shared across
        the reply, its re-asks and its reconnaissance rounds, so three re-asks and four
        tool rounds cannot multiply the wait. ``tools`` are the request-side schemas of
        the pass's read-only set; empty means the pass scouts with nothing and a tool
        call in its reply stays a parse failure.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"planner provider timeout after {self.timeout_seconds:.1f}s")
        result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

        def complete() -> None:
            try:
                result_queue.put(("ok", self.provider.complete(messages, max_tokens=self.output_tokens, tools=tools)))
            except BaseException as exc:
                result_queue.put(("error", exc))

        worker = threading.Thread(target=complete, name="skynet-planner-provider", daemon=True)
        worker.start()
        worker.join(remaining)
        if worker.is_alive():
            raise TimeoutError(f"planner provider timeout after {self.timeout_seconds:.1f}s")
        kind, value = result_queue.get()
        if kind == "error":
            raise value if isinstance(value, BaseException) else RuntimeError(str(value))
        if not isinstance(value, ModelTurn):
            raise TypeError("planner provider returned invalid turn")
        return value

    def generate(
        self,
        *,
        generation: int,
        goals: list[dict[str, Any]],
        tasks: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        previous: dict[str, Any],
        trigger: str,
        exploration: Mapping[str, Any] | None = None,
        tools: Mapping[str, Any] | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        run_id: str = "",
    ) -> list[dict[str, Any]]:
        """One bounded planning pass, with reconnaissance when tools are supplied.

        With ``tools`` the pass scouts before it proposes: a reply that asks for a
        read-only tool is executed, its result is fed back, and the pass continues while
        its round budget and the shared deadline allow. Without ``tools`` the shipped
        contract stands unchanged -- a tool call in the reply is a parse failure.
        """
        attempt = self.store.start_planner_attempt(generation, trigger, "explore-alternatives")
        self.last_status = ""
        tool_schemas: Sequence[Any] = plan_tool_schemas(tools) if tools else ()
        payload = self._bounded_payload(goals, tasks, memories, previous, exploration=exploration)
        messages = [{"role": "system", "content": planner_system_prompt()}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        deadline = time.monotonic() + self.timeout_seconds
        try:
            turn: ModelTurn = self._complete_bounded(messages, deadline, tools=tool_schemas)
        except Exception as exc:
            self._fail_attempt(attempt, trigger, "provider_error", str(exc))
            return []
        cumulative_usage = turn.total_tokens
        retried = False
        first_truncated = self._truncated(turn)
        last_error: Exception | None = None
        parsed: Any = None
        reask = 0
        tool_rounds = 0
        while True:
            # Reconnaissance: the pass reads before it proposes. Bounded by rounds and by
            # the shared deadline, executed on this thread (the store connection is not
            # thread-shared), and a tool fault comes back as a result, never as a raise.
            if tools and turn.tool_calls and tool_rounds < max_tool_rounds and time.monotonic() < deadline:
                tool_rounds += 1
                messages = [*messages, self._assistant_message(turn), *self._tool_messages(turn, tools, attempt, trigger, tool_rounds, run_id)]
                try:
                    turn = self._complete_bounded(messages, deadline, tools=tool_schemas)
                except Exception as provider_exc:
                    self._fail_attempt(attempt, trigger, "provider_error", f"tool round provider error: {provider_exc}")
                    return []
                cumulative_usage += turn.total_tokens
                step = 1 + tool_rounds
                if cumulative_usage > self.max_pass_tokens:
                    self._fail_attempt(attempt, trigger, "plan_budget_mid_pass_exceeded", f"mid-pass abort at step {step}: {cumulative_usage} > {self.max_pass_tokens} (the pass's booked token allowance)")
                    return []
                continue
            # One bounded re-ask per parse failure, up to MAX_MODEL_REASKS: the
            # model is told exactly what was wrong -- including the tool calls it
            # emitted in a phase that has none -- and asked for one object. It
            # never loops, so a model that keeps emitting the same shape cannot
            # burn the run budget. The reply's own stop reason rides along, so a
            # first attempt cut off at the output ceiling stays diagnosable even
            # when a later re-ask repairs it.
            try:
                parsed = self._parse(turn.text)
                break
            except Exception as exc:
                last_error = exc
                # Classify failure for discriminative retry
                failure_kind = self._failure_type(turn, str(exc))
                if reask >= MAX_MODEL_REASKS:
                    break
                reask += 1
                retried = True
                self.store.append_event("planner_retry", {
                    "attempt_id": attempt,
                    "trigger": trigger,
                    "reason": str(exc)[:500],
                    "finish_reason": turn.finish_reason,
                    "text_chars": len(turn.text),
                    "reask": reask,
                    "attempted_tools": [str(call.tool_name) for call in turn.tool_calls],
                    "failure_kind": failure_kind,
                })
                # Provide failure-specific retry hint
                if failure_kind == "truncation":
                    retry_hint = "Output was truncated at the token ceiling. Continue from the cutoff point and emit only the JSON object."
                elif failure_kind == "prose":
                    retry_hint = "Prose detected. Emit only the JSON object matching the schema — no explanatory text, no formatting, no extra fields."
                else:
                    retry_hint = ""
                # The forcing re-ask below carries no tool schemas, so its instruction must not
                # invite a call the request cannot execute.
                retry_instruction = PLANNER_RETRY_INSTRUCTION + "\n\n" + corrective_reask_instruction(
                    phase="planner", tool_calls=turn.tool_calls, parser_error=str(exc)[:1500], tools_available=False
                )
                if retry_hint:
                    retry_instruction += "\n\n" + retry_hint
                correction = [*messages, {"role": "assistant", "content": turn.text}, {"role": "user", "content": retry_instruction}]
                step = 1 + tool_rounds + reask
                if cumulative_usage > self.max_pass_tokens:
                    self._fail_attempt(attempt, trigger, "plan_budget_mid_pass_exceeded", f"mid-pass abort before re-ask at step {step}: {cumulative_usage} > {self.max_pass_tokens} (the pass's booked token allowance)")
                    return []
                try:
                    turn = self._complete_bounded(correction, deadline)
                    cumulative_usage += turn.total_tokens
                except Exception as provider_exc:
                    self._fail_attempt(attempt, trigger, "provider_error", f"retry provider error: {provider_exc}")
                    return []
        if parsed is None:
            # A completion stopped by the output ceiling is a distinct event from
            # a model that answered garbage: the first is a budget knob
            # (SKYNET_PLANNER_OUTPUT_TOKENS), the second is a contract break.
            # Either the first or the last attempt hitting the ceiling names the
            # budget problem.
            if first_truncated or self._truncated(turn):
                self._fail_attempt(
                    attempt,
                    trigger,
                    "output_truncated",
                    f"reply stopped at the output ceiling (first={first_truncated}, last={turn.finish_reason}); retried {MAX_MODEL_REASKS} times; still no decodable content",
                )
            else:
                self._fail_attempt(attempt, trigger, "invalid_response", f"retried {MAX_MODEL_REASKS} times; still invalid: {last_error}")
            return []
        data = [item for item in parsed.get("proposals", []) if isinstance(item, dict)]
        self._apply_goal_proposals(parsed.get("goal_proposals"), goals, attempt)
        accepted = 0
        rejected = 0
        deduplicated = 0
        result: list[dict[str, Any]] = []
        for raw in data[:3]:
            proposal, rejection_reason = self._validate(raw, goals)
            if proposal is None:
                rejected += 1
                self.store.record_planner_proposal(self._safe_rejected(raw), attempt, "rejected", reason=rejection_reason)
                continue
            if self._fingerprints_occupied(proposal["hypothesis_fingerprint"], proposal["structural_fingerprint"]):
                deduplicated += 1
                self.store.record_planner_proposal(proposal, attempt, "deduplicated", reason="duplicate fingerprint")
                continue
            paraphrase = self._paraphrase_of_resolved_work(proposal)
            if paraphrase is not None:
                deduplicated += 1
                self._record_paraphrase(proposal, attempt, paraphrase)
                continue
            # Learnability is checked only for genuinely new work: a duplicate is
            # reported as a duplicate, not as unlearnable, so the rejection reason
            # stays diagnostic instead of masking what actually happened.
            defect = learnability_defect(raw) if isinstance(raw, dict) else None
            if defect is not None:
                rejected += 1
                self.store.append_event("idea_learnability_rejected", {
                    "reason": defect,
                    "title": str(raw.get("title", ""))[:200] if isinstance(raw, dict) else "",
                })
                self.store.record_planner_proposal(proposal, attempt, "rejected", reason=defect)
                continue
            with self.store.transaction():
                task_id = self.store.add_task(proposal["title"], proposal["goal_id"], expected_new_fact=proposal["expected_new_fact"], hypothesis_fingerprint=proposal["hypothesis_fingerprint"], structural_fingerprint=proposal["structural_fingerprint"], area=str(proposal["kind"]))
                self.store.record_planner_proposal(proposal, attempt, "accepted", task_id=task_id)
            proposal["task_id"] = task_id
            result.append(proposal)
            accepted += 1
        if accepted:
            status = "completed"
        elif not data:
            status = "no_proposals"
        elif deduplicated == len(data):
            status = "all_deduplicated"
        elif rejected + deduplicated == len(data):
            status = "all_rejected" if rejected else "all_deduplicated"
        else:
            status = "no_work"
        # The retry is visible in the attempt row's failure field, which stays
        # the one place a reader of planner_attempts sees it without joining
        # event_log.
        self.store.finish_planner_attempt(attempt, status, proposal_count=len(data), failure="repaired_after_retry" if retried else "")
        self.last_status = status
        if not accepted:
            self.store.append_event("planner_no_work", {"attempt_id": attempt, "trigger": trigger, "reason": status, "rejected": rejected, "deduplicated": deduplicated})
        return result

    @staticmethod
    def _truncated(turn: ModelTurn) -> bool:
        """Whether the provider itself says the completion hit the output ceiling.

        ``finish_reason == "length"`` is the provider's own stop reason, not an
        inference from token counts: a reply that ends there was cut off by
        ``max_tokens``, so its undecodable text names a budget problem rather
        than a malformed model answer.
        """
        return turn.finish_reason == "length"
    @staticmethod
    def _failure_type(turn: ModelTurn, parser_error: str) -> str:
        """Classify the failure kind for discriminative retry.

        - 'truncation': finish_reason == 'length' (output ceiling hit)
        - 'prose': finish_reason == 'stop' with parse/JSON error (model emitted prose instead of JSON)
        - 'other': any other failure mode

        This classification enables targeted retry feedback and measurement.
        """
        if turn.finish_reason == "length":
            return "truncation"
        if turn.finish_reason == "stop" and parser_error:
            # Prose or malformed JSON when the model chose to stop
            return "prose"
        return "other"

    def _fail_attempt(self, attempt: str, trigger: str, status: str, failure: str) -> None:
        """Close a failed attempt and make the failure kind a distinct event.

        ``planner_generation_finished`` already carries the status, but a
        dedicated event keeps "the model replied with garbage" separable from
        "the provider was unreachable" and from "the reply was cut off at the
        output ceiling" without parsing the status vocabulary.
        """
        self.store.finish_planner_attempt(attempt, status, failure=failure[:1000])
        self.last_status = status
        event = {
            "invalid_response": "planner_invalid_response",
            "output_truncated": "planner_output_truncated",
        }.get(status, "planner_provider_error")
        self.store.append_event(event, {"attempt_id": attempt, "trigger": trigger, "failure": failure[:500]})

    def _fingerprints_occupied(self, hypothesis_fingerprint_value: str, structural_fingerprint_value: str) -> bool:
        """A fingerprint is occupied only by terminal hypotheses or genuinely live work.

        Cancelled or ready hypotheses left behind by a cancelled or completed task
        must not block a fresh proposal; a pending or running task still does.
        A terminal hypothesis stops occupying the fingerprint after the TTL, so a
        solved-then-regressed area can be revisited instead of being suppressed
        forever by a monotonically consumed hypothesis space.
        """
        from datetime import timedelta

        from .time import utc_datetime_now

        cutoff = (utc_datetime_now() - timedelta(days=self.hypothesis_ttl_days)).isoformat().replace("+00:00", "Z")
        if self.store.connection.execute(
            "SELECT 1 FROM hypotheses WHERE (fingerprint=? OR structural_fingerprint=?) "
            "AND status IN ('completed','rejected','exhausted') AND updated_at >= ? LIMIT 1",
            (hypothesis_fingerprint_value, structural_fingerprint_value, cutoff),
        ).fetchone():
            return True
        if self.store.connection.execute(
            "SELECT 1 FROM tasks WHERE (hypothesis_fingerprint=? OR structural_fingerprint=?) AND status IN ('pending','running') LIMIT 1",
            (hypothesis_fingerprint_value, structural_fingerprint_value),
        ).fetchone():
            return True
        return self.store.connection.execute(
            "SELECT 1 FROM planner_proposals p WHERE (p.hypothesis_fingerprint=? OR p.structural_fingerprint=?) "
            "AND p.status='accepted' AND EXISTS (SELECT 1 FROM tasks t WHERE t.task_id=p.created_task_id AND t.status IN ('pending','running')) LIMIT 1",
            (hypothesis_fingerprint_value, structural_fingerprint_value),
        ).fetchone() is not None

    def _paraphrase_of_resolved_work(self, proposal: dict[str, Any]) -> dict[str, Any] | None:
        """A proposal that re-asks a question whose task already reached a terminal state.

        The fingerprint gate is exact, and by design the TTL frees a resolved
        area so a solved-then-regressed problem can be revisited. A paraphrase
        defeats both, because it hashes differently and nothing else notices that
        its question was already answered.

        Scope is deliberate on both edges. Only ``TITLE_PARAPHRASE_KINDS`` are
        compared, because among ``recovery`` proposals a near-identical title is
        a legitimate recurrence ("drain the inbox again"); and only a prior
        proposal that received a task already in a terminal state counts, so work
        that is still live is neither duplicated nor blocked.
        """
        kind = str(proposal.get("kind", ""))
        if kind not in self.TITLE_PARAPHRASE_KINDS:
            return None
        if not normalize_hypothesis_text(proposal.get("title", "")).strip():
            return None
        rows = self.store.connection.execute(
            "SELECT p.title, p.kind, p.created_at, t.status AS task_status FROM planner_proposals p "
            "JOIN tasks t ON t.task_id = p.created_task_id "
            "WHERE p.created_task_id IS NOT NULL AND p.kind IN ('research','validation') "
            "AND t.status IN ('completed','cancelled','failed') ORDER BY p.created_at DESC LIMIT ?",
            (self.paraphrase_history_limit,),
        ).fetchall()
        for row in rows:
            ratio = title_similarity(proposal.get("title", ""), row["title"])
            if ratio >= self.TITLE_PARAPHRASE_RATIO:
                return {"ratio": round(ratio, 3), "title": title_prefix(row["title"]), "kind": row["kind"], "created_at": row["created_at"], "task_status": row["task_status"]}
        return None

    def _record_paraphrase(self, proposal: dict[str, Any], attempt: str, match: dict[str, Any]) -> None:
        """Visible under its own name: a paraphrase is not byte-identical work."""
        self.store.record_planner_proposal(
            proposal, attempt, "deduplicated",
            reason=f"paraphrase of resolved work (similarity {match['ratio']}): {match['title']}",
        )
        self.store.append_event("planner_paraphrase_deduplicated", {"attempt_id": attempt, "title": title_prefix(proposal.get("title", "")), "kind": str(proposal.get("kind", "")), "match": match})

    # How many of the reactor's own rows the prompt carries. Both callers hand
    # this function their collection best-first -- `_run_autonomous_planning`
    # reads tasks with ORDER BY updated_at DESC and search_memories ranks by
    # score -- so the window is taken from the head and the trimmer below drops
    # from the tail. Keeping the far end instead showed the model the *oldest*
    # rows the reactor held: at the generation-220 planner call the 40 tasks in
    # the prompt ended at 2026-09-24T11:06Z and held none of that day's 36 tasks,
    # while the completed task that already answered the question sat at rank 6
    # of 100 newest-first, so a re-issue was indistinguishable from new work.
    PAYLOAD_TASK_WINDOW = 40
    PAYLOAD_MEMORY_WINDOW = 20

    MAX_TOOL_ROUNDS = 4
    MAX_TOOL_CALLS_PER_ROUND = 3
    TOOL_RESULT_MAX_CHARS = 8_000

    @staticmethod
    def _assistant_message(turn: ModelTurn) -> Message:
        """The assistant turn as the provider expects it back, calls included."""
        return {
            "role": "assistant",
            "content": turn.text,
            "tool_calls": [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.tool_name, "arguments": json.dumps(call.arguments, separators=(",", ":"))},
                }
                for call in turn.tool_calls
            ],
        }

    def _tool_messages(
        self,
        turn: ModelTurn,
        tools: Mapping[str, Any],
        attempt: str,
        trigger: str,
        round_index: int,
        run_id: str,
    ) -> list[Message]:
        """Run the calls this turn asked for and return one result message per call.

        Bounded twice: ``MAX_TOOL_CALLS_PER_ROUND`` calls per turn, and each result is
        capped so one `git log` cannot fill the planner's window. A tool fault -- an
        unknown name, a refusal from the read-only shell, an exception inside the tool --
        comes back as an ``ok: false`` payload the model can read and react to, and is
        recorded as a durable ``planner_tool_call`` event so the pass's reconnaissance is
        auditable after the fact.
        """
        messages: list[Message] = []
        for call in list(turn.tool_calls)[: self.MAX_TOOL_CALLS_PER_ROUND]:
            tool = tools.get(call.tool_name)
            started = time.monotonic()
            if tool is None:
                payload: dict[str, Any] = {"ok": False, "error": f"unknown tool {call.tool_name}"}
            else:
                try:
                    payload = dict(tool.execute(dict(call.arguments), idempotency_key=f"{attempt}:r{round_index}:{call.call_id}"))
                except Exception as exc:  # a tool fault must not kill the pass
                    payload = {"ok": False, "error": f"tool raised: {exc}"[:500]}
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if len(encoded) > self.TOOL_RESULT_MAX_CHARS:
                payload = {"ok": bool(payload.get("ok")), "truncated": True, "preview": encoded[: self.TOOL_RESULT_MAX_CHARS]}
                encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            self.store.append_event(
                "planner_tool_call",
                {
                    "attempt_id": attempt,
                    "trigger": trigger,
                    "round": round_index,
                    "tool_name": call.tool_name,
                    "ok": bool(payload.get("ok")),
                    "result_chars": len(encoded),
                    "seconds": round(time.monotonic() - started, 3),
                },
                run_id or None,
            )
            messages.append({"role": "tool", "tool_call_id": call.call_id, "content": encoded})
        return messages

    def _bounded_payload(self, goals: list[dict[str, Any]], tasks: list[dict[str, Any]], memories: list[dict[str, Any]], previous: dict[str, Any], exploration: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Trim complete JSON records, never cut a serialized document mid-value.

        Every collection arrives best-first, so both the window and the
        over-budget trim work from the head: the newest task and the
        highest-scoring memory are the rows that must survive the cap.
        """
        payload: dict[str, Any] = {
            "goals": list(goals),
            "tasks": list(tasks[: self.PAYLOAD_TASK_WINDOW]),
            "memories": list(memories[: self.PAYLOAD_MEMORY_WINDOW]),
            "previous_outcome": previous,
            "cell_coverage": self._cell_coverage(),
            "instruction": PLANNER_INSTRUCTION,
        }
        if exploration:
            # Under a stall the empty-cell preference is strengthened with the
            # explicit untried list and its own instruction. This is a directive
            # to the one planner, never a second selection contour.
            payload["exploration"] = {
                **dict(exploration),
                "untried_cells": self._empty_cells(24),
                "instruction": PLANNER_STALL_INSTRUCTION,
            }
        while len(json.dumps(payload, ensure_ascii=False)) > self.max_input_chars and (payload["memories"] or payload["tasks"] or payload["goals"]):
            collection = payload["memories"] or payload["tasks"] or payload["goals"]
            collection.pop()
        if len(json.dumps(payload, ensure_ascii=False)) > self.max_input_chars:
            payload["previous_outcome"] = {"truncated": True}
        if len(json.dumps(payload, ensure_ascii=False)) > self.max_input_chars:
            payload["goals"] = [{"truncated": True, "count": len(goals)}]
        if len(json.dumps(payload, ensure_ascii=False)) > self.max_input_chars:
            # Last resort: the instruction itself is longer than the budget. It is
            # a fixed contract, so trimming it is only reachable with an
            # artificially tiny max_input_chars; the bound still holds.
            budget = max(0, self.max_input_chars - 100)
            payload = {"instruction": PLANNER_INSTRUCTION[:budget], "truncated": True}
        return payload

    def _empty_cells(self, limit: int) -> list[str]:
        """Descriptor cells with no active occupant, in a stable order."""
        filled: set[str] = set()
        if hasattr(self.store, "idea_cells"):
            try:
                filled = set(self.store.idea_cells())
            except Exception:
                filled = set()
        return sorted(
            f"{subsystem}|{change_type}|{source}"
            for subsystem in SUBSYSTEMS
            for change_type in CHANGE_TYPES
            for source in EVIDENCE_SOURCES
            if f"{subsystem}|{change_type}|{source}" not in filled
        )[:limit]

    def _cell_coverage(self) -> dict[str, Any]:
        """Cheap structural hint: which behavioural cells are still empty."""
        cells: dict[str, int] = {}
        if hasattr(self.store, "idea_cells"):
            try:
                cells = self.store.idea_cells()
            except Exception:
                cells = {}
        return {
            "axes": {
                "subsystem": list(SUBSYSTEMS),
                "change_type": list(CHANGE_TYPES),
                "evidence_source": list(EVIDENCE_SOURCES),
            },
            "cells_total": CELLS_TOTAL,
            "cells_filled": len(cells),
            "empty_examples": self._empty_cells(12),
        }

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        """Decode and validate a planner reply.

        Decoding moved to the unprotected ``planner_contract`` module (the
        instrument is repairable by the organism); validation stays here (the
        decision is not). ``parse_planner_reply`` only reshapes text into a JSON
        value and normalizes a top-level list; this protected call is what
        refuses anything that is not a schema-valid object.
        """
        data = parse_planner_reply(text)
        validate_shape(data, PLANNER_RESPONSE_SCHEMA)
        return data

    def _apply_goal_proposals(self, raw: Any, goals: list[dict[str, Any]], attempt: str) -> list[str]:
        """Create at most one bounded goal per planning attempt.

        The model never owns the lifecycle: it proposes, and the harness caps the
        number of active goals, de-duplicates by fingerprint and records the
        decision. Without this the goal set is frozen at genesis forever.
        """
        if not isinstance(raw, list) or not raw:
            return []
        active = [goal for goal in goals if goal.get("status") == "active"]
        if len(active) >= self.max_active_goals:
            self.store.append_event("goal_proposal_rejected", {"reason": "active goal cap reached", "active": len(active), "cap": self.max_active_goals})
            return []
        # De-duplicate on the normalized title: an existing goal's original
        # expected_behavior is not recoverable (the genesis goal stores no
        # constraints), so a fingerprint over it would never match.
        known = {normalize_hypothesis_text(goal.get("title", "")) for goal in active}
        created: list[str] = []
        for candidate in raw[:1]:
            validated = self._validate_goal(candidate)
            if validated is None:
                self.store.append_event("goal_proposal_rejected", {"reason": "invalid goal proposal", "attempt": attempt})
                continue
            if normalize_hypothesis_text(validated["title"]) in known:
                self.store.append_event("goal_proposal_rejected", {"reason": "duplicate goal fingerprint", "title": validated["title"]})
                continue
            with self.store.transaction():
                goal_id = self.store.add_goal(
                    validated["title"],
                    priority=float(validated.get("priority", 0.5)),
                    constraints={
                        "problem": validated["problem"],
                        "expected_behavior": validated["expected_behavior"],
                        "validation": validated["validation"],
                        "source": "autonomous_planner",
                    },
                )
                self.store.append_event(
                    "goal_proposal_accepted",
                    {"goal_id": goal_id, "title": validated["title"], "fingerprint": validated["fingerprint"], "attempt": attempt},
                )
            created.append(goal_id)
        return created

    @staticmethod
    def _validate_goal(raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        required = ("title", "problem", "expected_behavior", "validation")
        for key in required:
            value = raw.get(key)
            if not isinstance(value, str) or not value.strip():
                return None
        title = str(raw["title"]).strip()
        if len(title) > 300:
            return None
        try:
            priority = max(0.0, min(float(raw.get("priority", 0.5)), 4.0))
        except (TypeError, ValueError):
            priority = 0.5
        return {
            "title": title,
            "problem": str(raw["problem"]).strip(),
            "expected_behavior": str(raw["expected_behavior"]).strip(),
            "validation": str(raw["validation"]).strip(),
            "priority": priority,
            "fingerprint": hypothesis_fingerprint(area="goal", problem=title, expected_behavior=str(raw["expected_behavior"]).strip()),
        }

    @staticmethod
    def _validate(raw: dict[str, Any], goals: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
        """Validate a proposal and return (candidate, rejection_reason).

        Returns (candidate, None) on success, (None, reason) on failure.
        Specific rejection reasons identify which validation check failed,
        enabling diagnostic coverage tracking (DCR@3-style).
        """
        required = ("goal_id", "title", "problem", "hypothesis", "expected_new_fact", "validation", "scope", "kind")
        # Ensure all required keys are present and non-empty (except for collections)
        for key in required:
            val = raw.get(key)
            if val is None:
                return None, f"missing_{key}"
            if not isinstance(val, (list, dict)) and not val:
                return None, f"empty_{key}"
        if not isinstance(raw.get("scope"), list):
            return None, "scope_not_list"
        if str(raw["kind"]) not in {"engineering", "research", "validation", "recovery", "observation", "self_improvement"}:
            return None, f"invalid_kind:{raw.get('kind')}"
        if not any(str(goal.get("goal_id")) == str(raw["goal_id"]) and goal.get("status") == "active" for goal in goals):
            return None, "goal_inactive_or_missing"

        candidate = {key: raw[key] for key in required}
        if len(str(candidate["title"])) > 300:
            return None, "title_too_long"
        if len(candidate["scope"]) > 12:
            return None, "scope_too_large"

        candidate["goal_id"] = str(candidate["goal_id"])
        candidate["title"] = str(candidate["title"]).strip()
        candidate["inspiration_ref"] = str(raw.get("inspiration_ref", "") or "").strip()[:400]
        candidate["parent_idea_id"] = str(raw.get("parent_idea_id", "") or "").strip()[:64] or None
        candidate["hypothesis_fingerprint"] = hypothesis_fingerprint(area=candidate["kind"], problem=candidate["problem"], expected_behavior=candidate["expected_new_fact"], files=candidate["scope"])
        candidate["structural_fingerprint"] = structural_fingerprint(area=candidate["kind"], target=candidate["problem"], behavior_kind=candidate["kind"])
        return candidate, None

    @staticmethod
    def _safe_rejected(raw: dict[str, Any]) -> dict[str, Any]:
        import hashlib
        raw_text = json.dumps(raw, sort_keys=True, default=str)
        rejected_fingerprint = hashlib.sha256(raw_text.encode()).hexdigest()
        return {"goal_id": str(raw.get("goal_id", "")), "title": str(raw.get("title", "")), "problem": str(raw.get("problem", "")), "hypothesis": str(raw.get("hypothesis", "")), "expected_new_fact": str(raw.get("expected_new_fact", "")), "validation": str(raw.get("validation", "")), "scope": raw.get("scope", []) if isinstance(raw.get("scope", []), list) else [], "kind": str(raw.get("kind", "")), "hypothesis_fingerprint": "rejected-" + rejected_fingerprint, "structural_fingerprint": "rejected-" + rejected_fingerprint}
