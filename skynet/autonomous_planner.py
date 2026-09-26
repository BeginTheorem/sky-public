"""Bounded LLM proposal generation; execution remains deterministic and local."""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Sequence
from typing import Any

from .idea_archive import CELLS_TOTAL, CHANGE_TYPES, EVIDENCE_SOURCES, SUBSYSTEMS, learnability_defect
from .model_contracts import PLANNER_RESPONSE_SCHEMA, validate_shape
from .models import ModelTurn
from .planner import (
    hypothesis_fingerprint,
    normalize_hypothesis_text,
    structural_fingerprint,
    title_prefix,
    title_similarity,
)
from .planner_contract import PLANNER_INSTRUCTION, PLANNER_RETRY_INSTRUCTION, parse_planner_reply, planner_system_prompt
from .provider import LLMProvider, Message


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

    def _complete_bounded(self, messages: Sequence[Message]) -> ModelTurn:
        """Bound the planner provider call; a stuck request must not freeze PLAN."""
        result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

        def complete() -> None:
            try:
                result_queue.put(("ok", self.provider.complete(messages, max_tokens=self.output_tokens, tools=())))
            except BaseException as exc:
                result_queue.put(("error", exc))

        worker = threading.Thread(target=complete, name="skynet-planner-provider", daemon=True)
        worker.start()
        worker.join(self.timeout_seconds)
        if worker.is_alive():
            raise TimeoutError(f"planner provider timeout after {self.timeout_seconds:.1f}s")
        kind, value = result_queue.get()
        if kind == "error":
            raise value if isinstance(value, BaseException) else RuntimeError(str(value))
        if not isinstance(value, ModelTurn):
            raise TypeError("planner provider returned invalid turn")
        return value

    def generate(self, *, generation: int, goals: list[dict[str, Any]], tasks: list[dict[str, Any]], memories: list[dict[str, Any]], previous: dict[str, Any], trigger: str) -> list[dict[str, Any]]:
        attempt = self.store.start_planner_attempt(generation, trigger, "explore-alternatives")
        self.last_status = ""
        payload = self._bounded_payload(goals, tasks, memories, previous)
        messages = [{"role": "system", "content": planner_system_prompt()}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        try:
            turn: ModelTurn = self._complete_bounded(messages)
        except Exception as exc:
            self._fail_attempt(attempt, trigger, "provider_error", str(exc))
            return []
        retried = False
        try:
            parsed = self._parse(turn.text)
        except Exception as first_exc:
            # One bounded repair attempt: the model is told exactly what was
            # wrong and asked for one object. It never loops, so a model that
            # keeps emitting the same shape cannot burn the run budget. The
            # turn's own stop reason rides along, so a first attempt cut off at
            # the output ceiling stays diagnosable even when the retry repairs it.
            retried = True
            # Remember whether the FIRST reply was cut off: if the retry also
            # fails, the truncation must stay diagnosable even when the retry
            # ended for an unrelated reason (a non-truncated garbage reply).
            first_truncated = self._truncated(turn)
            self.store.append_event("planner_retry", {
                "attempt_id": attempt,
                "trigger": trigger,
                "reason": str(first_exc)[:500],
                "finish_reason": turn.finish_reason,
                "text_chars": len(turn.text),
            })
            correction = [*messages, {"role": "assistant", "content": turn.text}, {"role": "user", "content": PLANNER_RETRY_INSTRUCTION}]
            try:
                turn = self._complete_bounded(correction)
            except Exception as exc:
                self._fail_attempt(attempt, trigger, "provider_error", f"retry provider error: {exc}")
                return []
            try:
                parsed = self._parse(turn.text)
            except Exception as retry_exc:
                # A completion stopped by the output ceiling is a distinct event
                # from a model that answered garbage: the first is a budget knob
                # (SKYNET_PLANNER_OUTPUT_TOKENS), the second is a contract break.
                # Either attempt hitting the ceiling names the budget problem.
                if first_truncated or self._truncated(turn):
                    self._fail_attempt(
                        attempt,
                        trigger,
                        "output_truncated",
                        f"reply stopped at the output ceiling (first={first_truncated}, retry=finish_reason={turn.finish_reason}); retried once; still no decodable content",
                    )
                else:
                    self._fail_attempt(attempt, trigger, "invalid_response", f"retried once; still invalid: {retry_exc}")
                return []
        data = [item for item in parsed.get("proposals", []) if isinstance(item, dict)]
        self._apply_goal_proposals(parsed.get("goal_proposals"), goals, attempt)
        accepted = 0
        rejected = 0
        deduplicated = 0
        result: list[dict[str, Any]] = []
        for raw in data[:3]:
            proposal = self._validate(raw, goals)
            if proposal is None:
                rejected += 1
                self.store.record_planner_proposal(self._safe_rejected(raw), attempt, "rejected", reason="invalid proposal")
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

    def _bounded_payload(self, goals: list[dict[str, Any]], tasks: list[dict[str, Any]], memories: list[dict[str, Any]], previous: dict[str, Any]) -> dict[str, Any]:
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

    def _cell_coverage(self) -> dict[str, Any]:
        """Cheap structural hint: which behavioural cells are still empty."""
        cells: dict[str, int] = {}
        if hasattr(self.store, "idea_cells"):
            try:
                cells = self.store.idea_cells()
            except Exception:
                cells = {}
        filled = set(cells)
        return {
            "axes": {
                "subsystem": list(SUBSYSTEMS),
                "change_type": list(CHANGE_TYPES),
                "evidence_source": list(EVIDENCE_SOURCES),
            },
            "cells_total": CELLS_TOTAL,
            "cells_filled": len(filled),
            "empty_examples": sorted(
                f"{subsystem}|{change_type}|{source}"
                for subsystem in SUBSYSTEMS
                for change_type in CHANGE_TYPES
                for source in EVIDENCE_SOURCES
                if f"{subsystem}|{change_type}|{source}" not in filled
            )[:12],
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
    def _validate(raw: dict[str, Any], goals: list[dict[str, Any]]) -> dict[str, Any] | None:
        required = ("goal_id", "title", "problem", "hypothesis", "expected_new_fact", "validation", "scope", "kind")
        # Ensure all required keys are present and non-empty (except for collections)
        for key in required:
            val = raw.get(key)
            if val is None:
                return None
            if not isinstance(val, (list, dict)) and not val:
                return None
        if not isinstance(raw.get("scope"), list):
            return None
        if str(raw["kind"]) not in {"engineering", "research", "validation", "recovery", "observation", "self_improvement"}:
            return None
        if not any(str(goal.get("goal_id")) == str(raw["goal_id"]) and goal.get("status") == "active" for goal in goals):
            return None
        candidate = {key: raw[key] for key in required}
        if len(str(candidate["title"])) > 300 or len(candidate["scope"]) > 12:
            return None
        candidate["goal_id"] = str(candidate["goal_id"])
        candidate["title"] = str(candidate["title"]).strip()
        candidate["inspiration_ref"] = str(raw.get("inspiration_ref", "") or "").strip()[:400]
        candidate["parent_idea_id"] = str(raw.get("parent_idea_id", "") or "").strip()[:64] or None
        candidate["hypothesis_fingerprint"] = hypothesis_fingerprint(area=candidate["kind"], problem=candidate["problem"], expected_behavior=candidate["expected_new_fact"], files=candidate["scope"])
        candidate["structural_fingerprint"] = structural_fingerprint(area=candidate["kind"], target=candidate["problem"], behavior_kind=candidate["kind"])
        return candidate

    @staticmethod
    def _safe_rejected(raw: dict[str, Any]) -> dict[str, Any]:
        import hashlib
        raw_text = json.dumps(raw, sort_keys=True, default=str)
        rejected_fingerprint = hashlib.sha256(raw_text.encode()).hexdigest()
        return {"goal_id": str(raw.get("goal_id", "")), "title": str(raw.get("title", "")), "problem": str(raw.get("problem", "")), "hypothesis": str(raw.get("hypothesis", "")), "expected_new_fact": str(raw.get("expected_new_fact", "")), "validation": str(raw.get("validation", "")), "scope": raw.get("scope", []) if isinstance(raw.get("scope", []), list) else [], "kind": str(raw.get("kind", "")), "hypothesis_fingerprint": "rejected-" + rejected_fingerprint, "structural_fingerprint": "rejected-" + rejected_fingerprint}
