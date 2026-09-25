"""Split from the former monolithic CoreTests suite."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from skynet.memory import MemoryLoop
from skynet.memory_store import _RRF_MAX_LIFT, KIND_ALIASES, MEMORY_KINDS, MemoryStore, normalize_memory_kind
from skynet.models import AgentRunResult, ModelTurn, RunStatus
from skynet.store import StateStore


def _seed_memories(store: StateStore, rows: list[tuple[str, str, str, float, str]]) -> None:
    for memory_id, kind, content, confidence, updated_at in rows:
        store.connection.execute(
            "INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at) VALUES(?,?,?,?,NULL,?)",
            (memory_id, kind, content, confidence, updated_at),
        )
        assert store.memory_store is not None
        store.memory_store.index_memory(memory_id, kind, content)
    store.connection.commit()


class CoreTests(unittest.TestCase):
    def test_memory_loop_provider_timeout_degrades_without_blocking(self) -> None:
        class HangingProvider:
            def complete(self, messages, *, max_tokens, tools=()):
                time.sleep(0.05)
                return ModelTurn(text="{}")

        started = time.monotonic()
        result = MemoryLoop(HangingProvider(), timeout_seconds=0.001).consolidate(
            [], AgentRunResult(RunStatus.COMPLETED, "done"), "system"
        )
        self.assertLess(time.monotonic() - started, 0.04)
        self.assertEqual(result.evaluation["status"], "degraded")
    def test_memory_search_uses_bm25(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                store.consolidate("run", [
                    {"kind": "fact", "content": "Python reactor durable state", "confidence": 0.8},
                    {"kind": "fact", "content": "Python Python Python historical noise", "confidence": 0.2},
                    {"kind": "fact", "content": "Gateway timeout recovery", "confidence": 0.9},
                ])
            results = store.search_memories("Python reactor", limit=5)
            self.assertEqual(results[0]["content"], "Python reactor durable state")
            store.close()
    def test_memory_search_uses_durable_fts_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                store.consolidate("run-1", [{"kind": "fact", "content": "SQLite durable search index", "confidence": 0.8}])
            self.assertTrue(store.memory_store is not None)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0], 1)
            results = store.search_memories("durable search", limit=5)
            self.assertEqual(results[0]["content"], "SQLite durable search index")
            store.close()
    def test_consolidation_logs_one_bounded_memory_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                store.consolidate_versioned("ep-1", "run-1", [{"kind": "fact", "content": "seed"}])
                seed_id = store.connection.execute("SELECT memory_id FROM memories WHERE content='seed'").fetchone()[0]
                store.consolidate_versioned("ep-2", "run-1", [
                    {"kind": "fact", "content": "fresh"},
                    {"kind": "fact", "content": "replacement", "supersedes_memory_id": seed_id, "evidence": "contradicted"},
                    {"kind": "fact", "content": "ignored", "supersedes_memory_id": "does-not-exist"},
                ])
            events = [event for event in store.recent_run_events("run-1", limit=None) if event["kind"] == "memory_event"]
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["payload"]["added"], 1)
            payload = events[-1]["payload"]
            self.assertEqual(payload["run_id"], "run-1")
            self.assertEqual(payload["added"], 3)
            self.assertEqual(payload["superseded"], 1)
            self.assertEqual(len(payload["memory_ids"]), 3)
            self.assertEqual(store.connection.execute("SELECT status FROM memories WHERE memory_id=?", (seed_id,)).fetchone()[0], "superseded")
            store.close()

    def test_memory_event_ids_are_capped_at_twenty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                store.consolidate_versioned("ep-1", "run-1", [{"kind": "fact", "content": f"item-{index}"} for index in range(25)])
            payload = [event for event in store.recent_run_events("run-1", limit=None) if event["kind"] == "memory_event"][-1]["payload"]
            self.assertEqual(payload["added"], 25)
            self.assertEqual(len(payload["memory_ids"]), 20)
            store.close()

    def test_episode_for_memory_excludes_reactor_internal_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with store.transaction():
                store.append_event("run_started", {"start": True}, "run")
                store.append_event("react_phase", {"phase": "common"}, "run")
                store.append_event("tool_call", {"name": "clock"}, "run")
                store.append_event("memory_loop_finished", {"count": 1}, "run")
                store.append_event("finish_report", {"text": "done"}, "run")
            self.assertEqual([event["kind"] for event in store.episode_for_memory("run")], ["run_started", "tool_call", "finish_report"])
            store.close()
    def test_memory_loop_bounds_auxiliary_context(self) -> None:
        from skynet.memory import MemoryLoop

        class CapturingProvider:
            def __init__(self) -> None:
                self.payload = None

            def complete(self, messages, *, max_tokens, tools=()):
                self.payload = json.loads(messages[1]["content"])
                return ModelTurn(text='{"memory_candidates": [], "next_plan": {}}')

        provider = CapturingProvider()
        MemoryLoop(provider).consolidate(
            [],
            AgentRunResult(RunStatus.COMPLETED, "report"),
            "system",
            chat_history=[{"role": "user", "content": "x" * 60_000}, {"role": "assistant", "content": "recent"}],
            short_memory=[{"kind": "fact", "content": "y" * 30_000}, {"kind": "fact", "content": "latest"}],
        )
        self.assertEqual(cast(dict, provider.payload)["chat_history"][0]["kind"], "chat_history_truncated")
        self.assertEqual(cast(dict, provider.payload)["chat_history"][-1]["content"], "recent")
        self.assertEqual(cast(dict, provider.payload)["short_memory"][0]["kind"], "short_memory_truncated")
        self.assertEqual(cast(dict, provider.payload)["short_memory"][-1]["content"], "latest")
    def test_memory_loop_shortens_an_item_larger_than_its_own_budget(self) -> None:
        """One oversized newest item must not set the prompt size.

        The running-total guard compares only *later* items against the limit,
        so the first (newest) item was admitted whole however large it was. On
        the live ledger that made 27 of the 29 degraded Memory-Loop episodes
        send 460k-char transcript rows against a 48,000-char limit. The bounded
        rule keeps the newest item, shortened and marked, inside its budget.
        """
        from skynet.memory import MemoryLoop

        class CapturingProvider:
            def __init__(self) -> None:
                self.payload = None

            def complete(self, messages, *, max_tokens, tools=()):
                self.payload = json.loads(messages[1]["content"])
                return ModelTurn(text='{"memory_candidates": [], "next_plan": {}}')

        provider = CapturingProvider()
        huge = "z" * 400_000
        MemoryLoop(provider, episode_char_budget=20_000).consolidate(
            [{"kind": "tool_result", "payload": huge}],
            AgentRunResult(RunStatus.COMPLETED, "report"),
            "system",
            chat_history=[{"role": "user", "content": huge}],
        )
        payload = cast(dict, provider.payload)
        episode_chars = len(json.dumps(payload["episode"], ensure_ascii=False))
        chat_chars = len(json.dumps(payload["chat_history"], ensure_ascii=False))
        self.assertLessEqual(episode_chars, 20_000, f"episode sent {episode_chars} chars against a 20,000 budget")
        self.assertLessEqual(chat_chars, 48_000, f"chat_history sent {chat_chars} chars against a 48,000 budget")
        self.assertIn("truncated by the memory-loop budget", json.dumps(payload["chat_history"]))
    def test_memory_loop_degrades_on_non_json_provider_response(self) -> None:
        from skynet.memory import MemoryLoop

        class InvalidMemoryProvider:
            def complete(self, messages, *, max_tokens, tools=()):
                return ModelTurn(text="not json")

        result = MemoryLoop(InvalidMemoryProvider()).consolidate([], AgentRunResult(RunStatus.COMPLETED, "ok"), "system")
        self.assertEqual(result.evaluation["status"], "degraded")
    def test_memory_loop_drops_an_undeclared_field_instead_of_degrading(self) -> None:
        from skynet.memory import MemoryLoop

        class StaticProvider:
            def __init__(self, text: str) -> None:
                self.text = text

            def complete(self, messages, *, max_tokens, tools=()):
                return ModelTurn(text=self.text)

        payload = {
            "memory_candidates": [{"kind": "fact", "content": "kept", "confidence": 0.5}],
            "next_plan": {"next": "keep going"},
            "initial_prompt": "prompt",
            "initial_prompt_note": "the model invented this key on 2026-09-22",
            "goal_updates": [],
            "task_updates": [],
            "evaluation": {},
        }
        result = MemoryLoop(StaticProvider(json.dumps(payload))).consolidate(
            [], AgentRunResult(RunStatus.COMPLETED, "ok"), "system"
        )
        self.assertNotEqual(result.evaluation.get("status"), "degraded")
        self.assertEqual(len(result.memory_candidates), 1)
        self.assertEqual(result.initial_prompt, "prompt")
        self.assertIn("response.initial_prompt_note", result.dropped_fields)
    def test_memory_loop_keeps_valid_sections_when_one_is_malformed(self) -> None:
        from skynet.memory import MemoryLoop

        class StaticProvider:
            def __init__(self, text: str) -> None:
                self.text = text

            def complete(self, messages, *, max_tokens, tools=()):
                return ModelTurn(text=self.text)

        payload = {
            "memory_candidates": [{"kind": "fact", "content": "kept", "confidence": 0.5}],
            "next_plan": "not-an-object",
            "initial_prompt": "prompt",
            "goal_updates": [],
            "task_updates": [],
            "evaluation": {},
        }
        result = MemoryLoop(StaticProvider(json.dumps(payload))).consolidate(
            [], AgentRunResult(RunStatus.COMPLETED, "ok"), "system"
        )
        self.assertNotEqual(result.evaluation.get("status"), "degraded")
        self.assertEqual(len(result.memory_candidates), 1)
        self.assertEqual(result.next_plan, {})
        self.assertIn("response.next_plan", result.rejected_sections)
    def test_memory_loop_still_degrades_when_no_section_is_usable(self) -> None:
        from skynet.memory import MemoryLoop

        class StaticProvider:
            def __init__(self, text: str) -> None:
                self.text = text

            def complete(self, messages, *, max_tokens, tools=()):
                return ModelTurn(text=self.text)

        payload = {"memory_candidates": "wrong type", "next_plan": "wrong type", "initial_prompt": 7}
        result = MemoryLoop(StaticProvider(json.dumps(payload))).consolidate(
            [], AgentRunResult(RunStatus.COMPLETED, "ok"), "system"
        )
        self.assertEqual(result.evaluation.get("status"), "degraded")
    def test_memory_query_keeps_goal_title_terms_inside_the_tail_window(self) -> None:
        from skynet.reactor import Reactor

        goal_title = "Reactor checkpoint retention pointer audit"
        selected_work = {
            "task": {
                "title": "Confirm the acceptance criterion " * 20,
                "expected_new_fact": "generic acceptance boilerplate wording " * 20,
            },
            "goal": {"title": goal_title},
        }
        query = Reactor._memory_query(selected_work, [], {})
        terms = MemoryStore._normalize_terms(query)
        for token in ("reactor", "checkpoint", "retention", "pointer", "audit"):
            self.assertIn(token, terms)
    def test_memory_loop_receives_clean_history_and_short_memory(self) -> None:
        from skynet.memory import MemoryLoop

        class CapturingProvider:
            def __init__(self) -> None:
                self.messages = []

            def complete(self, messages, *, max_tokens, tools=()):
                self.messages = messages
                return ModelTurn(text='{"memory_candidates": [], "next_plan": {}}')

        provider = CapturingProvider()
        MemoryLoop(provider).consolidate(
            [],
            AgentRunResult(RunStatus.COMPLETED, "report"),
            "system",
            chat_history=[{"role": "assistant", "content": "answer"}],
            short_memory=[{"kind": "fact", "content": "known"}],
        )
        payload = json.loads(provider.messages[1]["content"])
        self.assertEqual(payload["chat_history"][0]["content"], "answer")
        self.assertEqual(payload["short_memory"][0]["content"], "known")
    def test_memory_loop_writes_full_dump_when_verbose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "state.sqlite3"

            class CapturingProvider:
                def complete(self, messages, *, max_tokens, tools=()):
                    return ModelTurn(text='{"memory_candidates": [], "next_plan": {}}')

            with patch.dict(os.environ, {"SKYNET_VERBOSE_PROVIDER": "1"}):
                MemoryLoop(CapturingProvider()).consolidate(
                    [], AgentRunResult(RunStatus.COMPLETED, "report"), "system", verbose_base_path=base
                )
            lines = [json.loads(line) for line in (Path(directory) / "verbose.jsonl").read_text(encoding="utf-8").splitlines()]
            kinds = [line["kind"] for line in lines]
            self.assertIn("provider_request", kinds)
            self.assertIn("provider_response", kinds)
            request = next(line for line in lines if line["kind"] == "provider_request")
            self.assertEqual(request["payload"]["messages"][0]["role"], "system")
            response = next(line for line in lines if line["kind"] == "provider_response")
            self.assertIn("memory_candidates", response["payload"]["text"])

    def test_react_history_for_memory_removes_reasoning_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.append_transcript(
                "react_history",
                {"messages": [
                    {"role": "system", "content": "system"},
                    {"role": "assistant", "content": "tool", "reasoning_content": "private"},
                    {"role": "tool", "tool_call_id": "call-1", "content": "result"},
                ]},
                "run-1",
            )
            history = store.react_history_for_memory("run-1")
            self.assertEqual(history[1], {"role": "assistant", "content": "tool"})
            self.assertEqual(history[2]["content"], "result")
            store.close()

    def test_memory_fallback_search_scores_when_fts_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.connection.execute("INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at) VALUES('m1','fact','durable python knowledge',1.0,NULL,?)", ("2026-01-01T00:00:00Z",))
            store.connection.execute("INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at) VALUES('m2','fact','unrelated note',0.5,NULL,?)", ("2026-01-01T00:00:00Z",))
            store.connection.commit()
            memory_store = store.memory_store
            assert memory_store is not None
            memory_store.fts_available = False
            results = store.search_memories("python", limit=5)
            self.assertEqual([item["memory_id"] for item in results], ["m1"])
            self.assertEqual(store.search_memories("", limit=5), [])
            store.close()


class MemoryRetrievalTests(unittest.TestCase):
    def test_memory_search_long_query_returns_relevant_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed_memories(store, [
                ("m-rare", "fact", "zymurgy calibration procedure", 0.95, "2026-06-01T00:00:00Z"),
                ("m-py1", "fact", "python reactor runtime", 0.5, "2026-05-01T00:00:00Z"),
                ("m-py2", "fact", "python gateway timeout", 0.5, "2026-05-02T00:00:00Z"),
                ("m-py3", "fact", "python durable index", 0.5, "2026-05-03T00:00:00Z"),
                ("m-py4", "fact", "python search memory", 0.5, "2026-05-04T00:00:00Z"),
                ("m-py5", "fact", "python lesson constraint", 0.5, "2026-05-05T00:00:00Z"),
                ("m-py6", "fact", "python observation outcome", 0.5, "2026-05-06T00:00:00Z"),
                ("m-x1", "fact", "alpha bravo", 0.1, "2020-01-01T00:00:00Z"),
                ("m-x2", "fact", "charlie delta", 0.1, "2020-01-02T00:00:00Z"),
                ("m-x3", "fact", "echo foxtrot", 0.1, "2020-01-03T00:00:00Z"),
                ("m-x4", "fact", "golf hotel", 0.1, "2020-01-04T00:00:00Z"),
                ("m-x5", "fact", "india juliet", 0.1, "2020-01-05T00:00:00Z"),
            ])
            query = "python zymurgy " + " ".join(f"filler{i}" for i in range(1, 21))
            results = store.search_memories(query, limit=20)
            self.assertGreater(len(results), 0)
            ids = [item["memory_id"] for item in results]
            self.assertIn("m-rare", ids)
            self.assertIn("m-py1", ids)
            self.assertLess(ids.index("m-rare"), ids.index("m-x1"))
            store.close()

    def test_memory_search_stopword_only_and_empty_queries_return_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed_memories(store, [("m1", "fact", "durable python knowledge", 1.0, "2026-01-01T00:00:00Z")])
            self.assertEqual(store.search_memories("the and of to", limit=5), [])
            self.assertEqual(store.search_memories("", limit=5), [])
            self.assertEqual(store.search_memories("a", limit=5), [])
            store.close()

    def test_memory_search_term_cap_keeps_tail(self) -> None:
        query = " ".join(f"word{i}" for i in range(200))
        terms = MemoryStore._normalize_terms(query)
        self.assertEqual(len(terms), 24)
        self.assertEqual(terms[0], "word176")
        self.assertEqual(terms[-1], "word199")
        self.assertNotIn("word0", terms)
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed_memories(store, [("m-tail", "fact", "word199 marker", 0.5, "2026-01-01T00:00:00Z")])
            results = store.search_memories(query, limit=5)
            self.assertIn("m-tail", [item["memory_id"] for item in results])
            store.close()

    def test_normalize_terms_anchors_a_repeat_at_its_last_occurrence(self) -> None:
        # Callers append the situation-defining parts last, so a goal term that
        # already appeared earlier must be anchored at its last occurrence; the
        # old first-occurrence dedup pushed it out of the 24-term window.
        query = "goal " + " ".join(f"f{index}" for index in range(30)) + " goal"
        terms = MemoryStore._normalize_terms(query)
        self.assertEqual(len(terms), 24)
        self.assertEqual(terms[-1], "goal")

    def test_rrf_lifts_recent_confident_partial_match_above_stale_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed_memories(store, [
                ("m-stale", "fact", "alpha beta gamma delta", 0.10, "2020-01-01T00:00:00Z"),
                ("m-fresh", "fact", "alpha", 0.95, "2026-01-01T00:00:00Z"),
                ("m-recent-a", "fact", "unrelated recent item one", 0.90, "2026-02-01T00:00:00Z"),
                ("m-recent-b", "fact", "unrelated recent item two", 0.99, "2026-03-01T00:00:00Z"),
            ])
            ids = [item["memory_id"] for item in store.search_memories("alpha beta gamma delta", limit=10)]
            self.assertLess(ids.index("m-fresh"), ids.index("m-stale"))
            store.close()

    def test_rrf_bounds_a_non_relevance_lift_to_one_slot(self) -> None:
        # The intentional lift above must not become unbounded. `recency` and
        # `confidence` are non-relevance axes, so a memory may rise at most
        # `_RRF_MAX_LIFT` slots above its own BM25 rank. Measured on 82 real
        # labelled planner queries, the old unbounded lift scored held-out
        # hit@5 0.102 / MRR 0.082 against 0.343 / 0.214 with the bound.
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            rows = [("m-weak", "fact", "alpha note", 0.99, "2026-12-01T00:00:00Z")]
            rows.extend(
                (
                    f"m-lex-{index:02d}", "fact",
                    f"alpha beta gamma delta epsilon zeta item{index}",
                    0.05, f"2018-01-{index + 1:02d}T00:00:00Z",
                )
                for index in range(12)
            )
            _seed_memories(store, rows)
            query = "alpha beta gamma delta epsilon zeta"
            match = " OR ".join(f'"{term}"' for term in query.split())
            lexical_ranks = [row[0] for row in store.connection.execute(
                "SELECT m.memory_id FROM memories_fts"
                " JOIN memories m ON m.memory_id = memories_fts.memory_id"
                " WHERE memories_fts MATCH ? AND m.status='active'"
                " ORDER BY bm25(memories_fts), m.memory_id LIMIT 100", (match,))]
            ids = [item["memory_id"] for item in store.search_memories(query, limit=20)]
            # m-weak is newest and most confident but matches only one term;
            # BM25 puts it last of the thirteen.
            self.assertEqual(lexical_ranks[-1], "m-weak")
            lexical_rank = lexical_ranks.index("m-weak")  # 0-based
            self.assertEqual(lexical_rank + 1, 13)
            # The invariant: no memory may sit more than `_RRF_MAX_LIFT` places
            # above where BM25 put it. Unbounded, m-weak lands 6th of 20; here it
            # cannot pass slot 13 - 1 - _RRF_MAX_LIFT.
            self.assertIn("m-weak", ids)
            self.assertGreaterEqual(ids.index("m-weak"), lexical_rank - _RRF_MAX_LIFT)
            store.close()

    def test_rrf_more_matching_terms_wins_on_equal_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed_memories(store, [
                ("m-strong", "fact", "alpha beta gamma", 0.5, "2025-01-01T00:00:00Z"),
                ("m-weak", "fact", "alpha", 0.5, "2025-01-01T00:00:00Z"),
            ])
            ids = [item["memory_id"] for item in store.search_memories("alpha beta gamma", limit=10)]
            self.assertLess(ids.index("m-strong"), ids.index("m-weak"))
            store.close()

    def test_ensure_fts_skips_rebuild_when_counts_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            store.consolidate("run", [{"kind": "fact", "content": "alpha beta", "confidence": 0.5}])
            store.close()
            with patch.object(MemoryStore, "rebuild_index") as rebuild:
                reopened = StateStore(path)
                rebuild.assert_not_called()
            reopened.close()

    def test_ensure_fts_rebuilds_when_counts_differ(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            store.consolidate("run", [{"kind": "fact", "content": "alpha beta", "confidence": 0.5}])
            store.connection.execute("DELETE FROM memories_fts")
            store.connection.commit()
            store.close()
            with patch.object(MemoryStore, "rebuild_index") as rebuild:
                reopened = StateStore(path)
                rebuild.assert_called_once()
            reopened.close()

    def test_read_only_connection_falls_back_to_manual_bm25(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            store.connection.execute(
                "INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at) VALUES('m1','fact','durable python knowledge',1.0,NULL,?)",
                ("2026-01-01T00:00:00Z",),
            )
            store.connection.execute("DROP TABLE memories_fts")
            store.connection.commit()
            store.close()
            connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            memory_store = MemoryStore(connection)
            self.assertFalse(memory_store.fts_available)
            results = memory_store.search("python", limit=5)
            self.assertEqual([item["memory_id"] for item in results], ["m1"])
            self.assertIn("score", results[0])
            connection.close()

    def test_normalize_memory_kind_maps_legacy_and_unknown(self) -> None:
        for legacy in KIND_ALIASES:
            self.assertIn(normalize_memory_kind(legacy), MEMORY_KINDS)
        for canonical in MEMORY_KINDS:
            self.assertEqual(normalize_memory_kind(canonical), canonical)
        self.assertEqual(normalize_memory_kind("episodic"), "observation")
        self.assertEqual(normalize_memory_kind("totally_unknown"), "observation")

    def test_normalize_legacy_kinds_converts_merges_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed_memories(store, [
                ("m-legacy-1", "technical_fact", "shared fact", 0.4, "2026-01-01T00:00:00Z"),
                ("m-legacy-2", "code_fact", "unique legacy", 0.7, "2026-01-02T00:00:00Z"),
                ("m-canonical", "fact", "shared fact", 0.9, "2026-01-03T00:00:00Z"),
                ("m-keep", "fact", "untouched", 0.5, "2026-01-04T00:00:00Z"),
            ])
            assert store.memory_store is not None
            changed = store.memory_store.normalize_legacy_kinds()
            self.assertEqual(changed, 2)
            kinds = {row["memory_id"]: row["kind"] for row in store.connection.execute("SELECT memory_id, kind FROM memories")}
            self.assertEqual(kinds["m-legacy-2"], "fact")
            self.assertNotIn("m-legacy-1", kinds)
            self.assertEqual(kinds["m-canonical"], "fact")
            self.assertEqual(kinds["m-keep"], "fact")
            self.assertEqual(store.connection.execute("SELECT confidence FROM memories WHERE memory_id='m-canonical'").fetchone()[0], 0.9)
            shared = [item for item in store.search_memories("shared fact", limit=10) if item["content"] == "shared fact"]
            self.assertEqual(len(shared), 1)
            self.assertEqual(shared[0]["memory_id"], "m-canonical")
            self.assertEqual(store.memory_store.normalize_legacy_kinds(), 0)
            store.close()

    def test_memory_search_results_carry_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed_memories(store, [
                ("m1", "fact", "alpha durable state", 0.8, "2026-01-01T00:00:00Z"),
                ("m2", "fact", "unrelated note", 0.2, "2026-01-02T00:00:00Z"),
            ])
            results = store.search_memories("alpha", limit=5)
            self.assertTrue(results)
            for item in results:
                self.assertIn("score", item)
            store.close()


class MemoryPromptTests(unittest.TestCase):
    def test_the_memory_prompt_drops_the_react_system_message(self) -> None:
        # The ReAct transcript starts with its own system prompt and the Memory
        # Loop adds one; keeping both wasted tokens on every consolidation.
        from skynet.memory import MemoryLoop
        from skynet.models import AgentRunResult, ModelTurn, RunStatus

        class CapturingProvider:
            def __init__(self) -> None:
                self.payload = None

            def complete(self, messages, *, max_tokens, tools=()):
                self.payload = json.loads(messages[1]["content"])
                return ModelTurn(text='{"memory_candidates": [], "next_plan": {}}')

        provider = CapturingProvider()
        MemoryLoop(provider).consolidate(
            [],
            AgentRunResult(RunStatus.COMPLETED, "report"),
            "system",
            chat_history=[
                {"role": "system", "content": "you are the react agent"},
                {"role": "user", "content": "hello"},
            ],
        )
        roles = [message["role"] for message in cast(dict, provider.payload)["chat_history"]]
        self.assertNotIn("system", roles)
        self.assertEqual(roles, ["user"])

    def test_the_memory_prompt_carries_identity_once(self) -> None:
        # The ReAct system prompt embeds SOUL.md and the Memory system prompt
        # repeats that identity. Dropping the ReAct system turn must leave the
        # identity text exactly once in the assembled Memory prompt.
        from skynet.memory import MemoryLoop
        from skynet.models import AgentRunResult, ModelTurn, RunStatus

        identity = "SOUL identity marker: I am SkyNet."

        class CapturingProvider:
            def __init__(self) -> None:
                self.messages = []

            def complete(self, messages, *, max_tokens, tools=()):
                self.messages = messages
                return ModelTurn(text='{"memory_candidates": [], "next_plan": {}}')

        provider = CapturingProvider()
        MemoryLoop(provider).consolidate(
            [],
            AgentRunResult(RunStatus.COMPLETED, "report"),
            f"Memory system prompt. {identity}",
            chat_history=[
                {"role": "system", "content": f"ReAct system prompt. {identity}"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "working"},
            ],
        )
        assembled = json.dumps(provider.messages, ensure_ascii=False)
        self.assertEqual(assembled.count(identity), 1)
        payload = json.loads(provider.messages[1]["content"])
        self.assertEqual([message["role"] for message in payload["chat_history"]], ["user", "assistant"])
