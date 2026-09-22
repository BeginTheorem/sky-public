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

    def search(self, query: str, limit: int = 20, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        terms = self._normalize_terms(query)
        if not terms:
            return []
        limit = max(1, min(int(limit), 50))
        if self.fts_available:
            return self._fts_search(terms, limit, include_inactive=include_inactive)
        return self._fallback_search(terms, limit, include_inactive=include_inactive)

    def _fts_search(self, terms: list[str], limit: int, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        candidate_limit = limit * 5
        active_join = "" if include_inactive else " AND m.status='active'"
        active_where = "" if include_inactive else " WHERE status='active'"
        bm25_rows = self.connection.execute(
            """SELECT m.memory_id FROM memories_fts
               JOIN memories m ON m.memory_id = memories_fts.memory_id
               WHERE memories_fts MATCH ?""" + active_join + """
               ORDER BY bm25(memories_fts), m.memory_id
               LIMIT ?""",
            (match, candidate_limit),
        ).fetchall()
        recency_rows = self.connection.execute(
            "SELECT memory_id FROM memories" + active_where + " ORDER BY updated_at DESC, memory_id LIMIT ?",
            (candidate_limit,),
        ).fetchall()
        confidence_rows = self.connection.execute(
            "SELECT memory_id FROM memories" + active_where + " ORDER BY confidence DESC, updated_at DESC, memory_id LIMIT ?",
            (candidate_limit,),
        ).fetchall()
        scores: dict[str, float] = {}
        for rows, (_axis, weight) in zip((bm25_rows, recency_rows, confidence_rows), _RRF_WEIGHTS, strict=False):
            for rank, row in enumerate(rows, start=1):
                scores[row["memory_id"]] = scores.get(row["memory_id"], 0.0) + weight / (_RRF_K + rank)
        if not scores:
            return []
        ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
        ids = [memory_id for memory_id, _ in ordered]
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(
            f"SELECT memory_id, kind, content, confidence, source_run, updated_at FROM memories WHERE memory_id IN ({placeholders})"
            + ("" if include_inactive else " AND status='active'"),
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

    def _fallback_search(self, terms: list[str], limit: int, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        active_where = "" if include_inactive else " WHERE status='active'"
        rows = self.connection.execute(
            "SELECT memory_id, kind, content, confidence, source_run, updated_at FROM memories"
            + active_where
            + " ORDER BY updated_at DESC LIMIT 500"
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
        return sorted(scored, key=lambda item: (item["score"], item["confidence"]), reverse=True)[:limit]
