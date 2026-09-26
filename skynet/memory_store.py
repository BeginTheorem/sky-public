from __future__ import annotations

import contextlib
import math
import re
import sqlite3
from typing import Any

MEMORY_KINDS = (
    "fact",
    "lesson",
    "constraint",
    "outcome",
    "hypothesis",
    "measurement",
    "risk",
    "procedure",
    "observation",
)

KIND_ALIASES: dict[str, str] = {
    "durable_fact": "fact",
    "technical_fact": "fact",
    "code_fact": "fact",
    "runtime_fact": "fact",
    "engineering_fact": "fact",
    # Observed on the live database after the first migration; the map must cover
    # every kind the model actually emits or `UNIQUE(kind, content)` fragments.
    "engineering": "fact",
    "blocker": "risk",
    "blocked": "risk",
    "blockage": "risk",
    "question": "observation",
    "answer": "fact",
    "user_dialogue": "fact",
    "dialogue": "fact",
    "decision": "outcome",
    "result": "outcome",
    "error": "risk",
    "failure": "risk",
    "technical_observation": "observation",
    "strategic_observation": "observation",
    "bug_observation": "observation",
    "lesson_learned": "lesson",
    "operational_lesson": "lesson",
    "operational_guideline": "procedure",
    "operational_knowledge": "procedure",
    "operational_note": "observation",
    "best_practice": "procedure",
    "design_constraint": "constraint",
    "system_constraint": "constraint",
    "technical_constraint": "constraint",
    "tool_constraint": "constraint",
    "test_constraint": "constraint",
    "measurement_limitation": "measurement",
    "system_invariant": "constraint",
    "invariant": "constraint",
    "system_logic": "fact",
    "harness_behavior": "observation",
    "episode_outcome": "outcome",
    "run_outcome": "outcome",
    "bug_report": "observation",
    "bug_fix": "outcome",
    "root_cause": "fact",
    "validated_hypothesis": "hypothesis",
    "strategic_decision": "outcome",
    "strategy": "procedure",
    "follow_up": "observation",
    "bottleneck": "risk",
    "artifact": "observation",
    "history": "observation",
    "pattern": "observation",
    "tooling": "procedure",
    "insight": "observation",
}


def normalize_memory_kind(kind: str) -> str:
    """Map an arbitrary model-supplied kind onto the canonical vocabulary."""
    if kind in MEMORY_KINDS:
        return kind
    return KIND_ALIASES.get(kind, "observation")


_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "if", "in",
    "into", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or",
    "our", "she", "so", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "to", "too", "up", "us", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "you", "your",
})

_MAX_QUERY_TERMS = 24
_RRF_K = 60
_RRF_WEIGHTS = (("bm25", 1.0), ("recency", 0.5), ("confidence", 0.35))
# The fused order may lift a memory above its lexical position by at most this
# many slots. `recency`/`confidence` are non-relevance axes: measured on 82 real
# labelled planner queries they add no candidate BM25 did not already return and
# an UNBOUNDED lift costs 3.4x recall (held-out hit@5 0.102 -> 0.343 at 1),
# while the intentional fixture lift still fires. A memory with no lexical rank
# is placed after every memory that has one.
#
# Generation 227 re-measured that claim against the FUSION FUNCTION itself, on
# 184 frozen envelopes (139 labelled) replayed from a read-only ledger copy:
# replicated fusion plus THIS placement reproduces MemoryStore.search on 184/184
# envelopes on both query arms, 0 of 417 top-3 slots come from outside the bm25
# pool, and the page SET is unchanged under 20 score-preserving tie shuffles
# (0/139) -- so on this reader a fused order can only REORDER the lexical pool.
# Replacing the rank fusion by a relative-score convex combination (per-axis
# min-max, same weights) moved h@5 by +.0072, 95% interval [-.0216,+.0432], on
# the planner arm and by exactly 0 on the injected arm, while its h@1 and MRR
# intervals lay wholly BELOW zero on the planner arm: the alternative function
# is not adopted, and no fusion-function change should be proposed for this
# reader without a mechanism that changes pool MEMBERSHIP rather than order.
_RRF_MAX_LIFT = 1


class MemoryStore:
    """Persistence facade for memories and their search projection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.fts_available = self._ensure_fts()

    def _ensure_fts(self) -> bool:
        exists = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchone()
        if not exists:
            try:
                self.connection.execute(
                    "CREATE VIRTUAL TABLE memories_fts USING fts5(memory_id UNINDEXED, kind, content)"
                )
            except sqlite3.OperationalError:
                return False
        self.fts_available = True
        if self._fts_counts_differ():
            with contextlib.suppress(sqlite3.OperationalError):
                self.rebuild_index()
        return True

    def _fts_counts_differ(self) -> bool:
        memories = self.connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        indexed = self.connection.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
        return memories != indexed

    def rebuild_index(self) -> None:
        if not self.fts_available:
            return
        self.connection.execute("DELETE FROM memories_fts")
        self.connection.execute(
            "INSERT INTO memories_fts(memory_id, kind, content) SELECT memory_id, kind, content FROM memories"
        )

    def index_memory(self, memory_id: str, kind: str, content: str) -> None:
        if not self.fts_available:
            return
        self.connection.execute("DELETE FROM memories_fts WHERE memory_id=?", (memory_id,))
        self.connection.execute(
            "INSERT INTO memories_fts(memory_id, kind, content) VALUES (?, ?, ?)",
            (memory_id, kind, content),
        )

    def _delete_memory(self, memory_id: str) -> None:
        self.connection.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))
        if self.fts_available:
            self.connection.execute("DELETE FROM memories_fts WHERE memory_id=?", (memory_id,))

    def drop_index(self, memory_ids: list[str]) -> int:
        """Remove memories from the FTS index (used when they are deleted)."""
        if not self.fts_available or not memory_ids:
            return 0
        placeholders = ",".join("?" for _ in memory_ids)
        try:
            return self.connection.execute(
                f"DELETE FROM memories_fts WHERE memory_id IN ({placeholders})", memory_ids
            ).rowcount
        except sqlite3.Error:
            return 0

    def normalize_legacy_kinds(self) -> int:
        """Rewrite legacy kinds to canonical ones, merging duplicates. Idempotent."""
        placeholders = ",".join("?" for _ in KIND_ALIASES)
        rows = self.connection.execute(
            f"SELECT memory_id, kind, content, confidence FROM memories WHERE kind IN ({placeholders})",
            tuple(KIND_ALIASES),
        ).fetchall()
        changed = 0
        for row in rows:
            canonical = KIND_ALIASES[row["kind"]]
            existing = self.connection.execute(
                "SELECT memory_id, confidence FROM memories WHERE kind=? AND content=?",
                (canonical, row["content"]),
            ).fetchone()
            if existing is not None and existing["confidence"] >= row["confidence"]:
                self._delete_memory(row["memory_id"])
            else:
                if existing is not None:
                    self._delete_memory(existing["memory_id"])
                self.connection.execute(
                    "UPDATE memories SET kind=? WHERE memory_id=?", (canonical, row["memory_id"])
                )
                self.index_memory(row["memory_id"], canonical, row["content"])
            changed += 1
        return changed

    @staticmethod
    def _terms(value: str) -> list[str]:
        return re.findall(r"[\w-]+", value.casefold())

    @classmethod
    def _normalize_terms(cls, value: str) -> list[str]:
        """The query terms the search uses: the LAST `_MAX_QUERY_TERMS`.

        Callers append the situation-defining parts last on purpose, so a term
        that repeats must be anchored at its last occurrence: `dict.fromkeys`
        kept the first, so a goal word that had already appeared in an earlier
        part stayed anchored there and was pushed out of the window by the later
        parts' own tokens. Measured on the 55 live `run_started` envelopes that
        carry a previous report, all six goal-title terms stay inside the window
        on 55/55 envelopes, against 45/55 before, while a query of 200 distinct
        words is unchanged (terms 176..199).
        """
        tokens = [token for token in cls._terms(value) if len(token) >= 2 and token not in _STOPWORDS]
        deduped = list(dict.fromkeys(reversed(tokens)))[::-1]
        return deduped[-_MAX_QUERY_TERMS:]

    @staticmethod
    def _document_terms(value: str) -> list[str]:
        return [token for token in MemoryStore._terms(value) if len(token) >= 2 and token not in _STOPWORDS]

    def search(
        self,
        query: str,
        limit: int = 20,
        *,
        include_inactive: bool = False,
        allowed: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Rank memories for ``query``; ``allowed`` restricts every axis to a cohort.

        ``allowed`` is the counterpart of the scorecard's ``--as-of``: every fusion axis
        is filtered to the given memory ids, so a page can be read against the corpus a
        published figure was measured on instead of today's ledger. An EMPTY set is a
        real cohort (no memory existed) and must not degrade to "no restriction", which
        is why the test is ``is not None`` and not truthiness.

        A supplied cohort is the WHOLE membership predicate: it replaces today's
        ``status='active'`` clause instead of intersecting it. The caller built the set
        from the validity window of an instant, which already decides membership there, so
        intersecting it with today's status re-introduces exactly the later events the
        reconstruction removed -- a memory alive at the instant and superseded afterwards
        vanished from the replay, and the past figure became a function of the present.
        Measured on the live ledger at 2026-09-22T23:40:45Z: three cohort members were
        superseded later, two of them sat in S3's replayed order, and hub top-1 read 2/5
        through the re-filter where the same instant without it reads 3/5.
        """
        terms = self._normalize_terms(query)
        if not terms:
            return []
        limit = max(1, min(int(limit), 50))
        if self.fts_available:
            return self._fts_search(terms, limit, include_inactive=include_inactive, allowed=allowed)
        return self._fallback_search(terms, limit, include_inactive=include_inactive, allowed=allowed)

    def _cohort_clause(self, alias: str, allowed: set[str] | None) -> tuple[str, tuple[Any, ...]]:
        """The ``AND <alias>.memory_id IN (...)`` fragment restricting an axis to a cohort."""
        if allowed is None:
            return "", ()
        return (
            f" AND {alias}.memory_id IN ({','.join('?' for _ in sorted(allowed))})",
            tuple(sorted(allowed)),
        )

    @staticmethod
    def _place_bounded(
        scores: dict[str, float],
        lexical_order: list[str],
        limit: int,
        *,
        max_lift: int = _RRF_MAX_LIFT,
    ) -> list[tuple[str, float]]:
        """The shipped bounded placement: fused scores -> ``(id, score)`` page slots.

        Extracted verbatim from ``_fts_search`` so a replay harness can reproduce a
        published page instead of re-implementing the three passes. The shipped page is
        NOT the fused order -- measured on the 165 frozen envelopes of the generation-206
        ledger, the shipped page equals the pooled fused order on 0/165 envelopes on
        either query arm -- so a rig that only re-ranks cannot check itself against
        ``MemoryStore.search`` at all, and every scratch harness that priced an
        alternative reader (generations 206, 227) re-implemented these passes by hand.

        ``lexical_order`` is the bm25 axis in rank order: only a memory inside it has a
        bounded lift budget of ``max_lift`` slots above its own lexical rank. Pass 2 fills
        only what pass 1 left, so a memory no BM25 match returned can never displace a
        lexically retrieved one; pass 3 rescues a lexical memory whose bounded slot fell
        past the page, so the page stays as full as the unbounded fusion kept it. Equal
        scores are broken by the stable external memory id, not by the pool's row order.
        """
        lexical_rank = {memory_id: rank for rank, memory_id in enumerate(lexical_order, start=1)}
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        slots: list[str | None] = [None] * limit
        for memory_id, _score in ranked:
            rank = lexical_rank.get(memory_id)
            if rank is None:
                continue
            for index in range(max(0, rank - 1 - max_lift), limit):
                if slots[index] is None:
                    slots[index] = memory_id
                    break
        for memory_id, _score in ranked:
            if lexical_rank.get(memory_id) is not None:
                continue
            for index in range(limit):
                if slots[index] is None:
                    slots[index] = memory_id
                    break
        placed = {memory_id for memory_id in slots if memory_id is not None}
        for memory_id, _score in ranked:
            if memory_id in placed:
                continue
            for index in range(limit):
                if slots[index] is None:
                    slots[index] = memory_id
                    break
        return [(memory_id, scores[memory_id]) for memory_id in slots if memory_id is not None]

    def _fts_search(
        self,
        terms: list[str],
        limit: int,
        *,
        include_inactive: bool = False,
        allowed: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        candidate_limit = limit * 5
        cohort_only = allowed is not None
        active_join = "" if (include_inactive or cohort_only) else " AND m.status='active'"
        # Anchored with ``WHERE 1=1`` so the cohort fragment (which starts with AND) is
        # always a valid continuation, with or without the status clause: composing
        # "FROM memories" + "" + " AND ..." was a syntax error, so the one call that
        # asked for a cohort without the status filter could not run at all.
        active_where = "" if (include_inactive or cohort_only) else " AND status='active'"
        bm25_cohort, bm25_params = self._cohort_clause("m", allowed)
        axis_cohort, axis_params = self._cohort_clause("memories", allowed)
        bm25_rows = self.connection.execute(
            """SELECT m.memory_id FROM memories_fts
               JOIN memories m ON m.memory_id = memories_fts.memory_id
               WHERE memories_fts MATCH ?""" + active_join + bm25_cohort + """
               ORDER BY bm25(memories_fts), m.memory_id
               LIMIT ?""",
            (match, *bm25_params, candidate_limit),
        ).fetchall()
        recency_rows = self.connection.execute(
            "SELECT memory_id FROM memories WHERE 1=1" + active_where + axis_cohort + " ORDER BY updated_at DESC, memory_id LIMIT ?",
            (*axis_params, candidate_limit),
        ).fetchall()
        confidence_rows = self.connection.execute(
            "SELECT memory_id FROM memories WHERE 1=1" + active_where + axis_cohort + " ORDER BY confidence DESC, updated_at DESC, memory_id LIMIT ?",
            (*axis_params, candidate_limit),
        ).fetchall()
        scores: dict[str, float] = {}
        for rows, (_axis, weight) in zip((bm25_rows, recency_rows, confidence_rows), _RRF_WEIGHTS, strict=False):
            for rank, row in enumerate(rows, start=1):
                scores[row["memory_id"]] = scores.get(row["memory_id"], 0.0) + weight / (_RRF_K + rank)
        if not scores:
            return []
        ordered = self._place_bounded(scores, [row["memory_id"] for row in bm25_rows], limit)
        if not ordered:
            return []
        ids = [memory_id for memory_id, _ in ordered]
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(
            f"SELECT memory_id, kind, content, confidence, source_run, updated_at FROM memories WHERE memory_id IN ({placeholders})"
            + ("" if (include_inactive or cohort_only) else " AND status='active'"),
            ids,
        ).fetchall()
        by_id = {row["memory_id"]: dict(row) for row in rows}
        results: list[dict[str, Any]] = []
        for memory_id, score in ordered:
            item = by_id.get(memory_id)
            if item is None:
                continue
            item["score"] = score
            results.append(item)
        return results

    def _fallback_search(
        self,
        terms: list[str],
        limit: int,
        *,
        include_inactive: bool = False,
        allowed: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Replay the corpus by hand, ordered by a key that cannot tie.

        This branch runs when the durable FTS projection is missing or unreadable,
        so it is the reader a re-measurement falls back to; a page that depends on
        the physical row order cannot be re-measured against the figure it
        published. Measured 2026-09-26 (738 active memories, 182 frozen planner
        envelopes): the hand-built score lands on two decimals or coarser, so the
        key this branch used to carry, ``(score, confidence)``, tied on 171/182
        queries -- and two ledgers holding the same logical corpus in different
        insertion orders returned DIFFERENT 20-slot pages, because ``sorted``
        leaves equal keys in the pool's row order and ``ORDER BY updated_at DESC``
        alone leaves that order to the query plan. This is the failure mode Lin and
        Yang describe for Lucene (arXiv:1807.05798v2): a tie broken by an indexer-
        assigned id is not repeatable across index instances, and the remedy is to
        break it by the stable external document id. Two edits, one per line: the
        pool is ordered ``updated_at DESC, memory_id`` so the membership of the
        500-row window stops depending on the scan, and the final key ends in
        ``memory_id``. The key negates instead of using ``reverse=True``, because
        reverse inverts the identifier too and would order equal-score memories by
        descending id -- arbitrary in exactly the way this change exists to remove.
        Wherever the old key was already unique the order is unchanged (measured: 0
        of 182 live pages move), so the edit decides only the pages that were
        arbitrary before it.
        """
        active_where = "" if (include_inactive or allowed is not None) else " AND status='active'"
        cohort_where, cohort_params = self._cohort_clause("memories", allowed)
        rows = self.connection.execute(
            "SELECT memory_id, kind, content, confidence, source_run, updated_at FROM memories WHERE 1=1"
            + active_where
            + cohort_where
            + " ORDER BY updated_at DESC, memory_id LIMIT 500",
            cohort_params,
        ).fetchall()
        documents = [self._document_terms(row["content"]) for row in rows]
        document_frequency = {term: sum(term in document for document in documents) for term in set(terms)}
        average_length = sum(len(document) for document in documents) / max(len(documents), 1)
        scored: list[dict[str, Any]] = []
        for row, document in zip(rows, documents, strict=False):
            score = 0.0
            for term in terms:
                frequency = document.count(term)
                if not frequency:
                    continue
                df = document_frequency[term]
                idf = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.2 * (1.0 - 0.75 + 0.75 * len(document) / max(average_length, 1.0))
                score += idf * (frequency * 2.2 / denominator)
            if score:
                item = dict(row)
                item["score"] = score
                scored.append(item)
        return sorted(scored, key=lambda item: (-item["score"], -item["confidence"], item["memory_id"]))[:limit]
