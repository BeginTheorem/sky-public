"""Deterministic portfolio planner for bounded autonomous work."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9_]+")
_REPETITION_MARKER_RE = re.compile(r"\b(?:pass|iteration|cycle)\s+\d+\b")


def normalize_text(value: Any) -> str:
    return " ".join(_WORD_RE.findall(str(value).casefold()))


def normalize_hypothesis_text(value: Any) -> str:
    """Ignore numbered retry markers when comparing durable work."""
    return normalize_text(_REPETITION_MARKER_RE.sub("", str(value)))


def hypothesis_fingerprint(*, area: str, problem: str, expected_behavior: str, files: Iterable[str] = (), selector: Iterable[str] = ()) -> str:
    payload = "\n".join([
        normalize_hypothesis_text(area), normalize_hypothesis_text(problem), normalize_hypothesis_text(expected_behavior),
        " ".join(sorted(normalize_text(item) for item in files)),
        " ".join(sorted(normalize_text(item) for item in selector)),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()


def structural_fingerprint(*, area: str, target: str, behavior_kind: str) -> str:
    payload = "\n".join((normalize_hypothesis_text(area), normalize_hypothesis_text(target), normalize_text(behavior_kind)))
    return hashlib.sha256(payload.encode()).hexdigest()


# A task fingerprint is in one of three states. The old novelty/repetition pair
# was computed from the same EXISTS predicate, so it carried one bit in two
# columns and scored every real candidate identically.
STATE_NOVELTY = {"new": 1.0, "live": 0.45, "terminal": 0.0}
STATE_REPETITION = {"new": 0.0, "live": 0.25, "terminal": 1.0}


# A task's area is the only behavioural signal an existing task carries, so it
# is mapped onto the archive's subsystem axis. Unknown areas fall back to
# "general"; the mapping never invents a descriptor the model chose.
_SUBSYSTEM_BY_AREA = {
    "reactor": "reactor", "recovery": "reactor", "bootstrap": "reactor", "watchdog": "reactor",
    "memory": "memory",
    "scheduler": "scheduler", "planning": "scheduler", "roadmap": "scheduler",
    "tools": "tools",
    "providers": "providers",
}


@dataclass(frozen=True, slots=True)
class PlannerCandidate:
    workstream_id: str
    title: str
    task_id: str | None
    score: float
    criticality: float
    novelty: float
    repetition_penalty: float
    reason: str
    expected_value: float = 0.0
    cell_key: str = ""
    cell_scarcity: float = 0.0


class PortfolioPlanner:
    """Ranks a bounded portfolio and never invents numbered replacement work."""

    def __init__(self, store: Any, *, max_workstreams: int = 4, epsilon: float = 0.0, hypothesis_ttl_days: float = 30.0, cell_scarcity_weight: float = 0.20) -> None:
        self.store = store
        self.max_workstreams = max_workstreams
        self.epsilon = max(0.0, min(float(epsilon), 1.0))
        self.hypothesis_ttl_days = max(0.0, float(hypothesis_ttl_days))
        self.cell_scarcity_weight = max(0.0, min(float(cell_scarcity_weight), 1.0))
        # A decision can be computed now and recorded later, once the caller
        # knows what the autonomous planner did. That is what lets a cycle
        # write one decision whose reason reflects the whole selection step.
        self._pending_decision: tuple[list[PlannerCandidate], PlannerCandidate | None, str, float | None] | None = None

    def _cell_for(self, row: dict[str, Any]) -> str:
        from .idea_archive import cell_key
        area = str(row.get("area") or "general").lower()
        return cell_key(_SUBSYSTEM_BY_AREA.get(area, "general"), "workflow", "own-repo")

    def _cell_scarcity(self) -> dict[str, float]:
        """1/(1+occupancy) per filled cell: an empty direction outranks a crowded one.

        A greedy scalar ranking converges on the same few areas; the scarcity
        term is the quality-diversity nudge that keeps the repertoire spread
        (arXiv:2506.13131), and it is deliberately smaller than criticality so it
        can never dominate a genuinely more important task.
        """
        try:
            cells = self.store.idea_cells()
        except Exception:
            cells = {}
        return {str(key): 1.0 / (1.0 + int(value)) for key, value in cells.items()}

    def rank(self) -> list[PlannerCandidate]:
        rows = self.store.planner_candidates(limit=1000, terminal_ttl_days=self.hypothesis_ttl_days)
        scarcity = self._cell_scarcity()
        best_by_workstream: dict[str, PlannerCandidate] = {}
        seen_fingerprints: set[str] = set()
        seen_structural_fingerprints: set[str] = set()
        for row in rows:
            criticality = max(0.0, min(float(row.get("priority", 0.0)) / 4.0, 1.0))
            state = str(row.get("fingerprint_state") or "new")
            novelty = STATE_NOVELTY.get(state, 0.45)
            repetition = STATE_REPETITION.get(state, 0.0)
            if repetition >= 1.0 or novelty <= 0.0:
                continue
            fingerprint = str(row.get("hypothesis_fingerprint") or "")
            structural = str(row.get("structural_fingerprint") or "")
            if fingerprint and fingerprint in seen_fingerprints:
                continue
            if structural and structural in seen_structural_fingerprints:
                continue
            attempts = max(0, int(row.get("attempts", 0) or 0))
            # A repeatedly retried task loses to fresh work; without this a
            # retryable-blocked task returns to pending and is re-selected
            # forever while alternatives exist.
            attempt_penalty = 0.1 * min(attempts, 5)
            expected_value = max(0.0, min(float(row.get("area_success_rate") or 0.0), 1.0))
            cell = self._cell_for(row)
            cell_scarcity = scarcity.get(cell, 1.0)
            score = (
                0.40 * criticality
                + 0.30 * novelty
                + 0.15 * expected_value
                + self.cell_scarcity_weight * cell_scarcity
                - 0.35 * repetition
                - attempt_penalty
            )
            candidate = PlannerCandidate(
                workstream_id=str(row.get("area") or row.get("goal_id") or "general"),
                title=str(row["title"]),
                task_id=str(row["task_id"]) if row.get("task_id") else None,
                score=score, criticality=criticality, novelty=novelty,
                repetition_penalty=repetition, reason="highest novelty-adjusted criticality",
                expected_value=expected_value, cell_key=cell, cell_scarcity=cell_scarcity,
            )
            previous = best_by_workstream.get(candidate.workstream_id)
            if previous is None or (candidate.score, candidate.novelty) > (previous.score, previous.novelty):
                best_by_workstream[candidate.workstream_id] = candidate
            if fingerprint:
                seen_fingerprints.add(fingerprint)
            if structural:
                seen_structural_fingerprints.add(structural)
        candidates = list(best_by_workstream.values())
        candidates.sort(key=lambda item: (-item.score, -item.criticality, -item.novelty, item.workstream_id))
        return candidates[: self.max_workstreams]

    def select(self, *, record: bool = True) -> dict[str, Any] | None:
        ranked = self.rank()
        selected = ranked[0] if ranked else None
        draw: float | None = None
        explore = False
        if self.epsilon > 0.0 and len(ranked) > 1 and hasattr(self.store, "next_random"):
            # Without exploration the ranking can never be evaluated: every
            # cycle picks the same top candidate, so a bad score function looks
            # exactly like a good one. The draw is seeded, durable and recorded.
            draw_value = self.store.next_random()
            draw = float(draw_value) if draw_value is not None else 0.0
            if draw < self.epsilon:
                index = 1 + int(draw * 10_000) % (len(ranked) - 1)
                selected = ranked[index]
                explore = True
        self._pending_decision = (
            ranked,
            selected,
            "epsilon_greedy_explore" if explore else (selected.reason if selected else "no novel work"),
            draw,
        )
        if record:
            self.record_decision()
        if selected is None or selected.task_id is None:
            return self._select_from_archive(record=record)
        return self.store.task_work(selected.task_id)

    def record_decision(self, *, reason: str | None = None) -> None:
        """Persist the decision computed by the last ``select`` call.

        ``select(record=False)`` defers the write so the caller can pass the
        reason the whole cycle actually produced; calling this twice for one
        decision is a programming error and the pending slot is cleared to make
        the second call a no-op instead of a duplicate row.
        """
        if self._pending_decision is None:
            return
        ranked, selected, default_reason, draw = self._pending_decision
        self._pending_decision = None
        self.store.record_planner_decision(ranked, selected, reason=reason or default_reason, draw=draw)

    def _select_from_archive(self, *, record: bool = True) -> dict[str, Any] | None:
        """An exhausted portfolio is not an exhausted search.

        Before the organism concludes "no novel work", sample an archived
        stepping stone and materialize it. DGM keeps every valid ancestor
        precisely because an archived idea can pay off long after it was opened,
        so the archive is the substrate that keeps the boundary open when the
        task pool is empty.
        """
        if not hasattr(self.store, "sample_archive_parents"):
            return None
        goal = self.store.connection.execute(
            "SELECT goal_id FROM goals WHERE status='active' ORDER BY priority DESC LIMIT 1"
        ).fetchone()
        if goal is None:
            return None
        parents = self.store.sample_archive_parents(k=1)
        if not parents:
            return None
        task_id = self.store.materialize_idea(str(parents[0]["idea_id"]), str(goal["goal_id"]))
        if record:
            self.store.record_planner_decision([], None, reason="archive_materialized")
        else:
            self._pending_decision = ([], None, "archive_materialized", None)
        return self.store.task_work(task_id)
