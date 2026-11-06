from __future__ import annotations

import calendar
import contextlib
import math
import re
import sqlite3
import time
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
_RRF_WEIGHTS = (("bm25", 1.0), ("recency", 0.5), ("confidence", 0.35), ("attention", 0.25))
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


def _symbolize_content(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of row with content symbolized if trust_source='owner_untrusted'.

    DualView (arXiv:2607.03821v2) defense: AgentView sees symbols, HumanView keeps originals.
    """
    if row.get("trust_source") == "owner_untrusted":
        out = dict(row)
        out["content"] = "[SYMBOL: owner_untrusted]"
        return out
    return row


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
        self.connection.commit()

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
        lifecycle_stage: str | None = None,
        scope: str | None = None,
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
        intersecting it with today's status         re-introduces exactly the later events the
        reconstruction removed -- a memory alive at the instant and superseded afterwards
        vanished from the replay, and the past figure became a function of the present.
        Measured on the live ledger at 2026-09-22T23:40:45Z: three cohort members were
        superseded later, two of them sat in S3's replayed order, and hub top-1 read 2/5
        through the re-filter where the same instant without it reads 3/5.

        Automatic lifecycle promotion (Oracle Agent Memory, arXiv:2607.13157v1):
        - raw → extracted after 3 positive retrievals with scope widening (thread→agent→global)
        - extracted → consolidated after 5 distinct session references with scope widening
        - Cross-session retrieval boost for multi-hop queries
        """
        terms = self._normalize_terms(query)
        if not terms:
            return []
        limit = max(1, min(int(limit), 50))
        if self.fts_available:
            fts_results = self._fts_search(terms, limit, include_inactive=include_inactive, allowed=allowed, lifecycle_stage=lifecycle_stage, scope=scope)
        else:
            fts_results = self._fallback_search(terms, limit, include_inactive=include_inactive, allowed=allowed, lifecycle_stage=lifecycle_stage, scope=scope)

        # Deduplicate and re-rank (simple approach: keep first occurrence)
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for item in fts_results:
            mid = item.get("memory_id", "")
            if mid not in seen:
                seen.add(mid)
                deduped.append(item)

        # ========== Automatic Memory Lifecycle Promotion ==========
        # After search results are returned, promote memories based on retrieval evidence
        # Raw → Extracted after N=3 positive retrievals
        # Extracted → Consolidated after M=5 distinct session references
        # Scope widening: thread → agent → global
        try:
            with self.connection:
                # Get all memory IDs that were returned in this search
                returned_ids = [item.get("memory_id") for item in deduped if item.get("memory_id")]
                if returned_ids:
                    placeholders = ",".join("?" for _ in returned_ids)
                    # Increment retrieval_count for each returned memory
                    self.connection.execute(
                        f"UPDATE memories SET retrieval_count = retrieval_count + 1 WHERE memory_id IN ({placeholders})",
                        returned_ids,
                    )
                    # For each memory, check if we should promote based on positive outcomes
                    # Positive outcome: memory was clicked/followed in subsequent turns
                    # We track this via positive_outcome_count
                    # Promotion to extracted after 3 positive retrievals
                    memories = self.connection.execute(
                        f"SELECT memory_id, lifecycle_stage, scope, retrieval_count, positive_outcome_count, last_session_id FROM memories WHERE memory_id IN ({placeholders})",
                        returned_ids,
                    ).fetchall()
                    by_id = {row["memory_id"]: dict(row) for row in memories}
                    for mid in returned_ids:
                        item = by_id.get(mid)
                        if item is None:
                            continue
                        lc_stage = item.get("lifecycle_stage", "raw")
                        retr_count = item.get("retrieval_count", 0)
                        pos_outcome = item.get("positive_outcome_count", 0)
                        # Raw → Extracted after 3 positive retrievals
                        if lc_stage == "raw" and retr_count >= 3 and pos_outcome >= 3:
                            new_scope = item.get("scope", "thread")
                            # Scope widening: thread → agent → global
                            scope_order = {"thread": 0, "agent": 1, "global": 2}
                            current_level = scope_order.get(new_scope, 0)
                            if current_level < 2:  # Can still widen
                                new_scope = list(scope_order.keys())[min(current_level + 1, 2)]
                            self.connection.execute(
                                "UPDATE memories SET lifecycle_stage=?, scope=? WHERE memory_id=?",
                                ("extracted", new_scope, mid)
                            )
                            # Reset counter after promotion
                            self.connection.execute(
                                "UPDATE memories SET positive_outcome_count=0 WHERE memory_id=?",
                                (mid,)
                            )
                        # Extracted → Consolidated after 5 distinct session references
                        elif lc_stage == "extracted" and item.get("last_session_id") and pos_outcome >= 5:
                            new_scope = item.get("scope", "thread")
                            scope_order = {"thread": 0, "agent": 1, "global": 2}
                            current_level = scope_order.get(new_scope, 0)
                            if current_level < 2:
                                new_scope = list(scope_order.keys())[min(current_level + 1, 2)]
                            self.connection.execute(
                                "UPDATE memories SET lifecycle_stage=?, scope=? WHERE memory_id=?",
                                ("consolidated", new_scope, mid)
                            )
                            # Reset counter after promotion
                            self.connection.execute(
                                "UPDATE memories SET positive_outcome_count=0 WHERE memory_id=?",
                                (mid,)
                            )
                    # Also boost cross-session memories in future searches
                    # Mark memories with cross-session evidence
                    for mid in returned_ids:
                        item = by_id.get(mid)
                        if item is None:
                            continue
                        session_id = item.get("last_session_id")
                        if session_id:
                            # This memory has cross-session evidence
                            self.connection.execute(
                                "UPDATE memories SET last_session_id=? WHERE memory_id=?",
                                (session_id, mid)
                            )
        except Exception as e:
            # Promotion failures should not break the search; continue silently
            # Log at debug level to avoid noise
            import logging
            logging.getLogger(__name__).debug("Lifecycle promotion skipped: %s", e)

        # Return up to limit
        limit = max(1, min(limit, 100))
        return deduped[:limit]

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
        lifecycle_stage: str | None = None,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        candidate_limit = limit * 5
        cohort_only = allowed is not None
        # Separate clauses for JOIN context (with 'm.' prefix) and WHERE context (without prefix)
        lifecycle_clause_join = " AND m.lifecycle_stage=?" if lifecycle_stage is not None else ""
        scope_clause_join = " AND m.scope=?" if scope is not None else ""
        lifecycle_clause_where = " AND lifecycle_stage=?" if lifecycle_stage is not None else ""
        scope_clause_where = " AND scope=?" if scope is not None else ""
        active_join = "" if (include_inactive or cohort_only) else " AND m.status='active'"
        active_join += lifecycle_clause_join + scope_clause_join
        # Anchored with ``WHERE 1=1`` so the cohort fragment (which starts with AND) is
        # always a valid continuation, with or without the status clause: composing
        # "FROM memories" + "" + " AND ..." was a syntax error, so the one call that
        # asked for a cohort without the status filter could not run at all.
        active_where = "" if (include_inactive or cohort_only) else " AND status='active'"
        active_where += lifecycle_clause_where + scope_clause_where
        bm25_cohort, bm25_params = self._cohort_clause("m", allowed)
        axis_cohort, axis_params = self._cohort_clause("memories", allowed)
        lifecycle_params = (lifecycle_stage,) if lifecycle_stage is not None else ()
        scope_params = (scope,) if scope is not None else ()
        bm25_params_full = (match, *bm25_params, *lifecycle_params, *scope_params, candidate_limit)
        bm25_rows = self.connection.execute(
            """SELECT m.memory_id FROM memories_fts
               JOIN memories m ON m.memory_id = memories_fts.memory_id
               WHERE memories_fts MATCH ?""" + active_join + bm25_cohort + """
               ORDER BY bm25(memories_fts), m.memory_id
               LIMIT ?""",
            bm25_params_full,
        ).fetchall()
        lifecycle_params = (lifecycle_stage,) if lifecycle_stage is not None else ()
        scope_params = (scope,) if scope is not None else ()
        recency_params = (*axis_params, *lifecycle_params, *scope_params, candidate_limit)
        recency_rows = self.connection.execute(
            "SELECT memory_id FROM memories WHERE 1=1" + active_where + axis_cohort + " ORDER BY updated_at DESC, memory_id LIMIT ?",
            recency_params,
        ).fetchall()
        confidence_rows = self.connection.execute(
            "SELECT memory_id FROM memories WHERE 1=1" + active_where + axis_cohort + " ORDER BY confidence DESC, updated_at DESC, memory_id LIMIT ?",
            recency_params,
        ).fetchall()
        attention_rows = self._attention_pool(candidate_limit, include_inactive=include_inactive, allowed=allowed, lifecycle_stage=lifecycle_stage, scope=scope)
        scores: dict[str, float] = {}
        # Handle both 3-axis and 4-axis fusion modes for backward compatibility
        if len(_RRF_WEIGHTS) == 3:
            # Legacy 3-axis mode (without attention)
            for rows, (_axis, weight) in zip((bm25_rows, recency_rows, confidence_rows), _RRF_WEIGHTS, strict=False):
                for rank, row in enumerate(rows, start=1):
                    memory_id = row[0]
                    scores[memory_id] = scores.get(memory_id, 0.0) + weight / (_RRF_K + rank)
        else:
            # New 4-axis mode (with attention)
            for rows, (_axis, weight) in zip((bm25_rows, recency_rows, confidence_rows, attention_rows), _RRF_WEIGHTS, strict=False):
                for rank, row in enumerate(rows, start=1):
                    memory_id = row[0]
                    scores[memory_id] = scores.get(memory_id, 0.0) + weight / (_RRF_K + rank)
        if not scores:
            return []
        ordered = self._place_bounded(scores, [row["memory_id"] for row in bm25_rows], limit)
        if not ordered:
            return []
        ids = [memory_id for memory_id, _ in ordered]
        placeholders = ",".join("?" for _ in ids)
        lifecycle_clause = " AND lifecycle_stage=?" if lifecycle_stage is not None else ""
        scope_clause = " AND scope=?" if scope is not None else ""
        extra_where = ("" if (include_inactive or cohort_only) else " AND status='active'") + lifecycle_clause + scope_clause
        extra_params = []
        if lifecycle_stage is not None:
            extra_params.append(lifecycle_stage)
        if scope is not None:
            extra_params.append(scope)
        # Include lifecycle_stage and scope in SELECT when those filters are applied
        select_cols = "memory_id, kind, content, confidence, source_run, updated_at, trust_source"
        if lifecycle_stage is not None:
            select_cols += ", lifecycle_stage"
        if scope is not None:
            select_cols += ", scope"
        rows = self.connection.execute(
            f"SELECT {select_cols} FROM memories WHERE memory_id IN ({placeholders})"
            + extra_where,
            (*ids, *extra_params),
        ).fetchall()
        by_id = {row["memory_id"]: dict(row) for row in rows}
        results: list[dict[str, Any]] = []
        for memory_id, score in ordered:
            item = by_id.get(memory_id)
            if item is None:
                continue
            item["score"] = score
            results.append(_symbolize_content(item))
        return results

    def _fallback_search(
        self,
        terms: list[str],
        limit: int,
        *,
        include_inactive: bool = False,
        allowed: set[str] | None = None,
        lifecycle_stage: str | None = None,
        scope: str | None = None,
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
        lifecycle_clause = " AND lifecycle_stage=?" if lifecycle_stage is not None else ""
        scope_clause = " AND scope=?" if scope is not None else ""
        active_where = "" if (include_inactive or allowed is not None) else " AND status='active'"
        active_where += lifecycle_clause + scope_clause
        cohort_where, cohort_params = self._cohort_clause("memories", allowed)
        lifecycle_params = (lifecycle_stage,) if lifecycle_stage is not None else ()
        scope_params = (scope,) if scope is not None else ()
        rows = self.connection.execute(
            "SELECT memory_id, kind, content, confidence, source_run, updated_at, trust_source FROM memories WHERE 1=1"
            + active_where
            + cohort_where
            + " ORDER BY updated_at DESC, memory_id LIMIT 500",
            (*cohort_params, *lifecycle_params, *scope_params),
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
                scored.append(_symbolize_content(item))
        return sorted(scored, key=lambda item: (-item["score"], -item["confidence"], item["memory_id"]))[:limit]

    @staticmethod
    def _compute_attention_score(retrieval_count: int, positive_outcome_count: int, complexity_tier: str) -> float:
        """Reinforcement score for a memory, from its accumulated usage evidence.

        Deliberately NOT a query-relevance score. There is no embedding provider,
        and a lexical "similarity" here only duplicates the BM25 axis -- measured,
        it lifted an exact stale match above a fresh confident partial one, the
        opposite of what the recency/confidence axes are for. The signal BM25
        cannot carry is usage: how often search returned the memory
        (``retrieval_count``, bumped on every hit) plus positive outcomes, scaled
        by the complexity tier. ``min(1.0, ...)`` saturates at ten uses so one
        popular memory cannot run away with the axis.
        """
        usage = retrieval_count + positive_outcome_count
        if usage <= 0:
            return 0.0
        tier_multipliers = {"Simple": 1.0, "Hybrid": 1.2, "Complex": 1.4}
        return min(1.0, usage / 10.0) * tier_multipliers.get(complexity_tier, 1.0)

    def _attention_pool(
        self,
        limit: int,
        *,
        include_inactive: bool = False,
        allowed: set[str] | None = None,
        lifecycle_stage: str | None = None,
        scope: str | None = None,
    ) -> list[tuple[str, float]]:
        """Rank memories by reinforcement evidence (see ``_compute_attention_score``)."""
        candidates: list[tuple[str, float]] = []

        query = "SELECT memory_id, retrieval_count, positive_outcome_count, complexity_tier FROM memories WHERE 1=1"
        params = []
        if not include_inactive and allowed is None:
            query += " AND status='active'"
        if allowed is not None:
            placeholders = ",".join("?" for _ in allowed)
            query += f" AND memory_id IN ({placeholders})"
            params.extend(sorted(allowed))
        if lifecycle_stage is not None:
            query += " AND lifecycle_stage=?"
            params.append(lifecycle_stage)
        if scope is not None:
            query += " AND scope=?"
            params.append(scope)
        query += " ORDER BY updated_at DESC LIMIT 500"

        try:
            rows = self.connection.execute(query, params).fetchall()
        except sqlite3.Error:
            return []

        for row in rows:
            memory_id, retrieval_count, positive_outcome_count, complexity_tier = row
            score = self._compute_attention_score(
                retrieval_count or 0, positive_outcome_count or 0, complexity_tier or "Simple"
            )
            if score > 0:
                candidates.append((memory_id, score))

        candidates.sort(key=lambda x: (-x[1], x[0]))
        return candidates






class SpacedReplayScheduler:
    """Reinforce consolidated memories on an expanding schedule.

    Spaced repetition with growing intervals, driven from the reactor's
    per-cycle housekeeping. Based on:
    - Oracle Agent Memory (arXiv:2607.13157v1): memory lifecycle with
      consolidation and retrieval stages
    - TiMem (arXiv:2601.02845v2): stratified scheduling with temporal hierarchy
    - Learn the Time to Learn (arXiv:2209.08660v2): replay scheduling matters
      more than what to replay

    Intervals follow exponential backoff: 1, 3, 7, 30 and 90 days. Replay does
    not re-embed anything -- there is no embedding provider -- it advances
    ``replay_level`` and increments ``retrieval_count``, the usage evidence the
    attention axis ranks on.
    """

    # Spaced repetition intervals in seconds
    DEFAULT_INTERVALS = (
        86400,       # 1 day
        259200,      # 3 days
        604800,      # 7 days
        2592000,     # 30 days
        7776000,     # 90 days
    )

    def __init__(
        self,
        connection: sqlite3.Connection,
        intervals: tuple[int, ...] | None = None,
    ) -> None:
        self.connection = connection
        self.intervals = intervals or self.DEFAULT_INTERVALS

    @staticmethod
    def _epoch_utc(value: Any) -> float | None:
        """Parse a stored UTC timestamp (``YYYY-MM-DDTHH:MM:SS``) to epoch seconds.

        The stored timestamps are UTC, so ``time.mktime`` would read them as
        local time and shift the schedule by the host offset (three hours on the
        MSK server); ``calendar.timegm`` is the correct inverse of the
        ``time.gmtime`` the writers use.
        """
        if not isinstance(value, str):
            return None
        try:
            return calendar.timegm(time.strptime(value[:19], "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, TypeError):
            return None

    def get_memories_due_for_replay(self, *, now: float | None = None) -> list[dict[str, Any]]:
        """Consolidated memories whose next replay interval has elapsed."""
        now = time.time() if now is None else now
        rows = self.connection.execute(
            """SELECT memory_id, content, kind, confidence, scope, lifecycle_stage,
                      retrieval_count, positive_outcome_count, last_session_id,
                      trust_source, updated_at, replay_level, last_replay_at
               FROM memories
               WHERE lifecycle_stage = 'consolidated'
                 AND status = 'active'
                 AND replay_level IS NOT NULL
                 AND replay_level >= 0
            """
        ).fetchall()

        due: list[dict[str, Any]] = []
        for row in rows:
            replay_level = row["replay_level"]
            if replay_level >= len(self.intervals):
                continue  # the schedule is exhausted
            interval = self.intervals[replay_level]
            anchor = self._epoch_utc(row["last_replay_at"])
            if anchor is None:
                anchor = self._epoch_utc(row["updated_at"])
            if anchor is None:
                continue
            if (now - anchor) >= interval:
                due.append(_symbolize_content(dict(row)))
        return due

    def replay_memory(self, memory_id: str) -> bool:
        """Advance one memory's replay schedule and its usage evidence.

        There is no embedding to reinforce, so a "replay" is the bookkeeping the
        schedule needs: advance ``replay_level`` and bump ``retrieval_count``,
        the usage evidence the attention axis ranks on.
        """
        row = self.connection.execute(
            "SELECT replay_level FROM memories WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return False
        replay_level = row["replay_level"] if row["replay_level"] is not None else 0
        now_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        self.connection.execute(
            """UPDATE memories
               SET replay_level = ?, last_replay_at = ?, retrieval_count = retrieval_count + 1
               WHERE memory_id = ?""",
            (replay_level + 1, now_str, memory_id),
        )
        return True

    def run_replay_cycle(self, *, max_per_cycle: int = 50) -> dict[str, Any]:
        """Replay the due memories, bounded so one cycle cannot do unbounded work."""
        due = self.get_memories_due_for_replay()
        replayed = 0
        for mem in due[:max_per_cycle]:
            if self.replay_memory(mem["memory_id"]):
                replayed += 1
        if replayed:
            self.connection.commit()
        return {"replayed": replayed, "candidates": len(due), "intervals": self.intervals}

    def initialize_replay_tracking(self) -> int:
        """Seed ``replay_level`` for consolidated memories that lack it. Idempotent."""
        rows = self.connection.execute(
            """SELECT memory_id FROM memories
               WHERE lifecycle_stage = 'consolidated'
                 AND (replay_level IS NULL OR last_replay_at IS NULL)"""
        ).fetchall()
        now_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        for row in rows:
            self.connection.execute(
                "UPDATE memories SET replay_level = 0, last_replay_at = ? WHERE memory_id = ?",
                (now_str, row["memory_id"]),
            )
        if rows:
            self.connection.commit()
        return len(rows)

