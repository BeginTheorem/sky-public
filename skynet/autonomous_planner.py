"""Bounded LLM proposal generation; execution remains deterministic and local."""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Sequence
from typing import Any

from .idea_archive import CELLS_TOTAL, CHANGE_TYPES, EVIDENCE_SOURCES, SUBSYSTEMS, learnability_defect
from .model_contracts import PLANNER_RESPONSE_SCHEMA, json_contract, parse_json_object, validate_shape
from .models import ModelTurn
from .planner import hypothesis_fingerprint, normalize_hypothesis_text, structural_fingerprint
from .provider import LLMProvider, Message

PLANNER_INSTRUCTION = """Return exactly one JSON object matching the planner response contract. Generate at most 3 bounded proposals for active goals.
Each proposal must contain goal_id, title, problem, hypothesis, expected_new_fact, validation, scope (list), kind.
The proposal must be useful without user input, have observable validation, and be smaller than a broad project.
Do not repeat completed, blocked, exhausted, pending, or rejected work. Do not generate numbered pass/iteration/cycle variants.
Do not execute tools or describe tool calls. Allowed kind values: engineering, research, validation, recovery, observation, self_improvement.
For kind=research, set inspiration_ref to the external source (arXiv id, repository URL, or page URL) and make expected_new_fact the distilled, testable claim taken from it; a research proposal without a source will be rejected.
Prefer proposals that occupy an empty descriptor cell: the payload lists cell_coverage, and an idea in an unexplored subsystem/change-type/evidence-source combination outranks a third variation of an already-worked one.
A proposal may name parent_idea_id to develop an archived idea instead of starting from nothing."""


class AutonomousPlanner:
    def __init__(self, provider: LLMProvider, store: Any, *, output_tokens: int = 4096, max_input_chars: int = 700_000, timeout_seconds: float = 120.0, max_active_goals: int = 8, hypothesis_ttl_days: float = 30.0) -> None:
        self.provider = provider
        self.store = store
        self.output_tokens = output_tokens
        self.max_input_chars = max_input_chars
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_active_goals = max(1, int(max_active_goals))
        self.hypothesis_ttl_days = max(0.0, float(hypothesis_ttl_days))

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
        payload = self._bounded_payload(goals, tasks, memories, previous)
        messages = [{"role": "system", "content": "You are SkyNet's bounded autonomous planner. The harness owns execution. Return exactly one JSON object matching this schema: " + json_contract(PLANNER_RESPONSE_SCHEMA)}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        try:
            turn: ModelTurn = self._complete_bounded(messages)
        except Exception as exc:
            self.store.finish_planner_attempt(attempt, "provider_error", failure=str(exc)[:1000])
            return []
        try:
            parsed = self._parse(turn.text)
        except Exception as exc:
            self.store.finish_planner_attempt(attempt, "invalid_response", failure=str(exc)[:1000])
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
        self.store.finish_planner_attempt(attempt, status, proposal_count=len(data))
        if not accepted:
            self.store.append_event("planner_no_work", {"attempt_id": attempt, "trigger": trigger, "reason": status, "rejected": rejected, "deduplicated": deduplicated})
        return result

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

    def _bounded_payload(self, goals: list[dict[str, Any]], tasks: list[dict[str, Any]], memories: list[dict[str, Any]], previous: dict[str, Any]) -> dict[str, Any]:
        """Trim complete JSON records, never cut a serialized document mid-value."""
        payload: dict[str, Any] = {
            "goals": list(goals),
            "tasks": list(tasks[-40:]),
            "memories": list(memories[-20:]),
            "previous_outcome": previous,
            "cell_coverage": self._cell_coverage(),
            "instruction": PLANNER_INSTRUCTION,
        }
        while len(json.dumps(payload, ensure_ascii=False)) > self.max_input_chars and (payload["memories"] or payload["tasks"] or payload["goals"]):
            collection = payload["memories"] or payload["tasks"] or payload["goals"]
            collection.pop(0)
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
        value = text.strip()
        if "{" in value:
            value = value[value.find("{"): value.rfind("}") + 1]
        data = parse_json_object(value)
        validate_shape(data, PLANNER_RESPONSE_SCHEMA)
        return data if isinstance(data, dict) else {}

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
