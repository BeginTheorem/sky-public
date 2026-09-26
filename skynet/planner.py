"""Deterministic portfolio planner for bounded autonomous work."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9_]+")
_REPETITION_MARKER_RE = re.compile(r"\b(?:pass|iteration|cycle)\s+\d+\b")
_SENTENCE_END = ".!?"


def normalize_text(value: Any) -> str:
    return " ".join(_WORD_RE.findall(str(value).casefold()))


def normalize_hypothesis_text(value: Any) -> str:
    """Ignore numbered retry markers when comparing durable work."""
    return normalize_text(_REPETITION_MARKER_RE.sub("", str(value)))


def proposal_title_tokens(value: Any) -> frozenset[str]:
    """Token set of a proposal title under the fingerprint normalization."""
    return frozenset(normalize_hypothesis_text(value).split())


def title_similarity(left: Any, right: Any) -> float:
    """Jaccard similarity of two titles under the fingerprint normalization.

    The fingerprint collapses only text that normalizes to the same string, so a
    paraphrased re-issue of an already-answered question hashes differently and
    passes the exact dedup gate. A title-token Jaccard survives that paraphrase;
    the threshold that makes it decision-grade is measured in
    ``AutonomousPlanner._paraphrase_of_resolved_work``.
    """
    left_tokens = proposal_title_tokens(left)
    right_tokens = proposal_title_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def title_prefix(value: Any, limit: int = 120) -> str:
    """Cut a title at the last sentence end before ``limit``, else at ``limit``."""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    cut = text.rfind(_SENTENCE_END, 0, limit)
    if cut > 0:
        return text[: cut + 1]
    space = text.rfind(" ", 0, limit)
    return text[:space] if space > 0 else text[:limit]


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


# The single valence channel. A charge is stored as (value, draws): the value is
# a bounded 0..1 reading of how the recent cycles went, and the draw count is how
# much that reading is trusted. With no draws the channel is inert, so an empty
# ledger reproduces the historical selection byte for byte.
VALENCE_PRIOR = 0.5
VALENCE_WINDOW_DRAWS = 20.0
VALENCE_TILT_WEIGHT = 0.5


def valence_decidable_band(base: float, *, weight: float = VALENCE_TILT_WEIGHT) -> tuple[float, float]:
    """The interval of thresholds the channel can reach from a base threshold.

    This is the channel's whole reach, and it is small: the tilt factor is
    ``1 - weight * (2 * effective - 1)`` with ``effective`` in [0, 1] and the
    weight capped at 1, so a threshold can never leave
    ``[base * (1 - weight), base * (1 + weight)]``. Every draw outside that
    interval is evidence about nothing -- the channel cannot change that
    decision at any charge -- so the band is what a falsifier's sample size has
    to be computed from, and it is why a band-restricted replay is the only
    honest test of this channel. At the shipped defaults (``weight`` 0.5,
    ``planner_epsilon`` 0.1) the reach is [0.05, 0.15]: a tenth of the draw
    range, and only the ``[threshold, base)`` half of it can matter while the
    reading sits above the prior, because the tilt only lowers the threshold in
    that direction. The band is asymptotic rather than reached: a charge known
    through a finite number of draws covers only the fraction
    ``confidence = draws / (draws + VALENCE_WINDOW_DRAWS)`` of it, so the live
    charge (``draws == VALENCE_WINDOW_DRAWS``) sits at confidence 0.5, reaches
    [0.075, 0.125] at the shipped base, and leaves only [0.075, 0.1) on the
    deciding side -- 0.025 of the draw range, which is why the falsifier needs
    a reach this function makes computable.
    """
    bounded_base = max(0.0, min(float(base), 1.0))
    bounded_weight = max(0.0, min(float(weight), 1.0))
    return (max(0.0, bounded_base * (1.0 - bounded_weight)),
            min(1.0, bounded_base * (1.0 + bounded_weight)))


def valence_band_occupancy(
    rows: Iterable[dict[str, Any]],
    *,
    base: float = 0.0,
    weight: float = VALENCE_TILT_WEIGHT,
    value: float = VALENCE_PRIOR,
    draws: int = 0,
) -> dict[str, Any]:
    """Count the recorded draws a charged valence channel could actually decide.

    The falsifier that judges this channel compares a charged run against an
    uncharged one, so its sample is not "the decisions the planner made" but
    "the decisions whose draw fell inside the tilt's reach", and the reach is
    fixed by ``valence_decidable_band``: a threshold can never leave
    ``[base * (1 - weight), base * (1 + weight)]``. Above the prior the tilt only
    lowers the threshold, so the deciding interval is ``[threshold, base)``
    rather than the whole band: a draw at or above the base cannot be moved by a
    down-tilt at any amplitude.

    A second term decides whether an in-band decision can testify at all, and it
    is counted here rather than assumed by the caller: ``select`` takes an
    exploring rank only when ``len(ranked) > 1``, and the rank is
    ``1 + int(draw * 10_000) % (len(ranked) - 1)``. On a one-candidate decision
    that expression is 0 for every draw, so changing the threshold changes
    nothing and the row is in the band but cannot testify. Counting such rows as
    evidence is how a falsifier gets its sample size from nowhere.

    ``rows`` are the journal's exploration blocks (mappings carrying ``draw``
    and ``candidates``). The returned mapping reports ``total``, ``deciding``,
    ``decisive``, the ``reach`` and ``threshold`` the counts were taken against,
    ``reachable`` (whether the down-tilt actually moved the threshold), and the
    two shares. The sizing target is ``planner_epsilon`` itself rather than a
    hand-picked number: arXiv:1706.01905 section 9.1 calibrates exploration
    noise to ``delta := -log(1 - eps + eps / |A|)``, the KL divergence of an
    eps-greedy policy, so a calibrated perturbation moves about ``eps`` of the
    decisions (section 9: the two noise sources must have similar distances).
    """
    reach = valence_decidable_band(base, weight=weight)
    threshold = valence_tilt(base, value=value, draws=draws, weight=weight)
    total = deciding = decisive = 0
    for row in rows:
        draw = row.get("draw")
        if draw is None:
            continue
        total += 1
        if not (threshold <= float(draw) < base):
            continue
        deciding += 1
        if len(row.get("candidates") or []) >= 2:
            decisive += 1
    return {
        "total": total,
        "deciding": deciding,
        "decisive": decisive,
        "reach": reach,
        "threshold": threshold,
        "reachable": threshold <= base,
        "share": (deciding / total) if total else 0.0,
        "decisive_share": (decisive / total) if total else 0.0,
    }


def valence_tilt(base: float, *, value: float, draws: int,
                 weight: float = VALENCE_TILT_WEIGHT) -> float:
    """The exploration threshold this valence channel implies.

    The channel acts on the ONE actuator that already exists: the epsilon-greedy
    exploration threshold in `PortfolioPlanner.select`. A drawn threshold is
    deliberately bounded and identity-preserving in three ways, because a
    selection influence nobody can replay is not auditable:

    * a zero-draw channel returns `base` exactly, so a fresh or disabled channel
      cannot change one byte of the decision;
    * a neutral reading (``value == VALENCE_PRIOR``, or any reading known only
      through zero information) returns `base` exactly;
    * the returned threshold stays in [0, 1] and is never more than
      ``weight * base`` away from `base`, so valence can bias the rate of
      exploration but cannot replace the policy with itself. ``weight`` is the
      tilt amplitude and it is capped at 1 exactly as the tilt caps it: it is
      the one number that decides how much of the draw range the channel can act
      on at all (``valence_decidable_band``), the shipped default is 0.5, and a
      configuration cannot escape its own band by asking for more.

    The direction is the affect hypothesis under test, not a proven policy: a
    high reading LOWERS the threshold (exploit the ranking that just worked) and
    a low reading RAISES it (explore, because insisting on a ranking whose window
    just went badly is the trap). ``draws`` is the confidence; the offset shrinks
    to zero as draws -> 0, so a channel with almost no history cannot swing the
    decision, and the live charge (draws == window) sits at half confidence.
    """
    bounded_base = max(0.0, min(float(base), 1.0))
    bounded_value = max(0.0, min(float(value), 1.0))
    bounded_weight = max(0.0, min(float(weight), 1.0))
    confidence = max(0, int(draws)) / (max(0, int(draws)) + VALENCE_WINDOW_DRAWS)
    effective = VALENCE_PRIOR + (bounded_value - VALENCE_PRIOR) * confidence
    factor = 1.0 - bounded_weight * (2.0 * effective - 1.0)
    return max(0.0, min(1.0, bounded_base * factor))


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

    def __init__(
        self,
        store: Any,
        *,
        max_workstreams: int = 4,
        epsilon: float = 0.0,
        hypothesis_ttl_days: float = 30.0,
        cell_scarcity_weight: float = 0.20,
        valence: tuple[float, int] | None = None,
        tilt_weight: float = VALENCE_TILT_WEIGHT,
    ) -> None:
        self.store = store
        self.max_workstreams = max_workstreams
        self.epsilon = max(0.0, min(float(epsilon), 1.0))
        self.hypothesis_ttl_days = max(0.0, float(hypothesis_ttl_days))
        self.cell_scarcity_weight = max(0.0, min(float(cell_scarcity_weight), 1.0))
        # The charge is passed in and never read here: the caller owns the window
        # (one charge per selection), so the tilt stays a pure function of an
        # argument and a replay can reproduce it from the journal alone.
        value, draws = valence if valence is not None else (VALENCE_PRIOR, 0)
        self.valence = (max(0.0, min(float(value), 1.0)), max(0, int(draws)))
        # The amplitude is configuration, so it is passed in rather than read
        # from the module: the threshold stays a pure function of this planner's
        # own arguments, and a replay can reproduce it from the journal plus the
        # config the row was written under. A module read here would have made
        # the band a property of the import order, not of the decision.
        self.tilt_weight = max(0.0, min(float(tilt_weight), 1.0))
        self.threshold = valence_tilt(self.epsilon, value=self.valence[0],
                                      draws=self.valence[1], weight=self.tilt_weight)
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
            if draw < self.threshold:
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
        # Both arms of the counterfactual reach the journal. ``threshold`` is the
        # tilted one the draw was actually decided against; without the untilted
        # base a replay cannot ask "would this draw have flared with the channel
        # absent?", which is the falsifier the channel is judged by. ``epsilon`` is
        # exactly the base the tilt was derived from, so no new quantity is added.
        self.store.record_planner_decision(
            ranked, selected, reason=reason or default_reason, draw=draw,
            threshold=self.threshold, base_threshold=self.epsilon,
        )

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
        # Goals may share a priority -- the live ledger holds two active goals
        # at 4.0 -- so `priority DESC` alone let the row's physical position
        # choose which goal a materialized archive idea was attached to: the
        # measured query returned eaa65414 under one insertion order and
        # dcc0e38a under the reverse. `created_at` keeps the intended tie-break
        # (the older goal wins) and `goal_id` makes it a function of the corpus.
        goal = self.store.connection.execute(
            "SELECT goal_id FROM goals WHERE status='active' ORDER BY priority DESC, created_at, goal_id LIMIT 1"
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
