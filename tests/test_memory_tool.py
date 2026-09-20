"""The on-demand memory tool: search, remember, forget, pin/unpin, correct."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from skynet.memory_tool import MemoryTool
from skynet.store import StateStore


def _has(store: StateStore, method: str) -> bool:
    return callable(getattr(store, method, None))


class MemoryToolTests(unittest.TestCase):
    def _store(self, directory: str) -> StateStore:
        return StateStore(Path(directory) / "state.sqlite3")

    def _seed(self, store: StateStore, content: str, kind: str = "fact", confidence: float = 0.8) -> str:
        store.consolidate("seed-run", [{"kind": kind, "content": content, "confidence": confidence}])
        row = store.connection.execute("SELECT memory_id FROM memories WHERE content=?", (content,)).fetchone()
        self.assertIsNotNone(row)
        return str(row["memory_id"])

    def test_schema_is_openai_style_and_names_every_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            tool = MemoryTool(store)
            self.assertEqual(tool.name, "memory")
            self.assertEqual(tool.capability_kind, "write")
            schema = tool.schema
            self.assertEqual(schema["type"], "function")
            function = schema["function"]
            self.assertEqual(function["name"], "memory")
            self.assertEqual(
                function["parameters"]["properties"]["action"]["enum"],
                ["search", "remember", "forget", "pin", "unpin", "correct"],
            )
            self.assertEqual(function["parameters"]["required"], ["action"])
            self.assertIn("search", function["description"])
            self.assertIn("correct", function["description"])
            self.assertIn("supersedes", function["description"])
            store.close()

    def test_remember_then_search_finds_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            if not _has(store, "remember_memory"):
                self.skipTest("store.remember_memory not present yet")
            tool = MemoryTool(store)
            remembered = tool.execute(
                {"action": "remember", "content": "the reactor owns lifecycle", "kind": "fact", "confidence": 0.9},
                idempotency_key="r1",
            )
            self.assertTrue(remembered["ok"], remembered)
            self.assertTrue(remembered["memory_id"])
            found = tool.execute({"action": "search", "query": "reactor lifecycle"}, idempotency_key="s1")
            self.assertTrue(found["ok"], found)
            self.assertIn("the reactor owns lifecycle", [item["content"] for item in found["memories"]])
            store.close()

    def test_search_limit_and_empty_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            tool = MemoryTool(store)
            for index in range(5):
                self._seed(store, f"shared token observation {index}")
            result = tool.execute({"action": "search", "query": "shared token", "limit": 2}, idempotency_key="s")
            self.assertTrue(result["ok"], result)
            self.assertLessEqual(len(result["memories"]), 2)
            # An out-of-range limit is clamped, not rejected.
            clamped = tool.execute({"action": "search", "query": "shared token", "limit": 999}, idempotency_key="s2")
            self.assertTrue(clamped["ok"], clamped)
            self.assertLessEqual(len(clamped["memories"]), 50)
            self.assertFalse(tool.execute({"action": "search", "query": "   "}, idempotency_key="s3")["ok"])
            self.assertFalse(tool.execute({"action": "search"}, idempotency_key="s4")["ok"])
            store.close()

    def test_search_excludes_inactive_memories_when_the_store_supports_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            columns = {row["name"] for row in store.connection.execute("PRAGMA table_info(memories)")}
            if "inactive" in columns:
                column, inactive_value = "inactive", 1
            elif "active" in columns:
                column, inactive_value = "active", 0
            elif "status" in columns:
                column, inactive_value = "status", "superseded"
            else:
                self.skipTest("store has no active/inactive memory column yet")
            tool = MemoryTool(store)
            memory_id = self._seed(store, "obsolete fact alpha")
            store.connection.execute(
                f"UPDATE memories SET {column}=? WHERE memory_id=?", (inactive_value, memory_id)
            )
            store.connection.commit()
            default = tool.execute({"action": "search", "query": "obsolete fact alpha"}, idempotency_key="s1")
            self.assertNotIn("obsolete fact alpha", [item["content"] for item in default["memories"]])
            included = tool.execute(
                {"action": "search", "query": "obsolete fact alpha", "include_inactive": True},
                idempotency_key="s2",
            )
            self.assertIn("obsolete fact alpha", [item["content"] for item in included["memories"]])
            store.close()

    def test_forget_removes_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            if not _has(store, "forget_memory"):
                self.skipTest("store.forget_memory not present yet")
            tool = MemoryTool(store)
            memory_id = self._seed(store, "a memory to forget")
            forgotten = tool.execute({"action": "forget", "memory_id": memory_id}, idempotency_key="f1")
            self.assertTrue(forgotten["ok"], forgotten)
            self.assertTrue(forgotten["forgotten"])
            row = store.connection.execute("SELECT 1 FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
            self.assertIsNone(row)
            again = tool.execute({"action": "forget", "memory_id": memory_id}, idempotency_key="f2")
            self.assertTrue(again["ok"])
            self.assertFalse(again["forgotten"])
            store.close()

    def test_pin_and_unpin_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            tool = MemoryTool(store)
            memory_id = self._seed(store, "always inject this")
            pinned = tool.execute({"action": "pin", "memory_id": memory_id}, idempotency_key="p1")
            self.assertTrue(pinned["ok"], pinned)
            self.assertTrue(pinned["updated"])
            self.assertIn("always inject this", [item["content"] for item in store.pinned_memories()])
            unpinned = tool.execute({"action": "unpin", "memory_id": memory_id}, idempotency_key="p2")
            self.assertTrue(unpinned["ok"], unpinned)
            self.assertTrue(unpinned["updated"])
            self.assertNotIn("always inject this", [item["content"] for item in store.pinned_memories()])
            missing = tool.execute({"action": "pin", "memory_id": "does-not-exist"}, idempotency_key="p3")
            self.assertTrue(missing["ok"])
            self.assertFalse(missing["updated"])
            store.close()

    def test_correct_supersedes_with_a_new_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            if not _has(store, "correct_memory"):
                self.skipTest("store.correct_memory not present yet")
            tool = MemoryTool(store)
            old_id = self._seed(store, "the service listens on port 5000")
            corrected = tool.execute(
                {
                    "action": "correct",
                    "memory_id": old_id,
                    "content": "the service listens on port 8080",
                    "evidence": "config/skynet.env:12",
                },
                idempotency_key="c1",
            )
            self.assertTrue(corrected["ok"], corrected)
            new_id = corrected["memory_id"]
            self.assertNotEqual(new_id, old_id)
            found = tool.execute({"action": "search", "query": "service port 8080"}, idempotency_key="c2")
            self.assertIn("the service listens on port 8080", [item["content"] for item in found["memories"]])
            store.close()

    def test_invalid_arguments_return_ok_false_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            tool = MemoryTool(store)
            cases = [
                {},
                {"action": "bogus"},
                {"action": "search"},
                {"action": "search", "query": "   "},
                {"action": "remember"},
                {"action": "remember", "content": "   "},
                {"action": "forget"},
                {"action": "pin"},
                {"action": "unpin"},
                {"action": "correct", "memory_id": "x"},
                {"action": "correct", "memory_id": "x", "content": "c"},
                {"action": "correct", "content": "c", "evidence": "e"},
                # A syntactically valid correction of an unknown id must be an
                # error result, not a raised ValueError from the store.
                {"action": "correct", "memory_id": "missing", "content": "c", "evidence": "e"},
            ]
            for arguments in cases:
                result = tool.execute(arguments, idempotency_key="bad")
                self.assertFalse(result["ok"], arguments)
                self.assertIn("error", result)
            store.close()

    def test_missing_store_method_returns_a_structured_error(self) -> None:
        class Bare:
            pass

        tool = MemoryTool(Bare())  # type: ignore[arg-type]
        result = tool.execute({"action": "search", "query": "anything"}, idempotency_key="k")
        self.assertFalse(result["ok"])
        self.assertIn("store does not expose", result["error"])
