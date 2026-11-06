"""The planning pass's read-only senses: what it may run, and what it may never run."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from skynet.memory_tool import MemoryTool
from skynet.plan_tools import PlanBashTool, plan_tool_schemas, plan_toolset, refusal_reason
from skynet.store import StateStore


class ReadOnlyBashTest(unittest.TestCase):
    def test_inspection_is_allowed(self) -> None:
        for command in (
            "git log --oneline -5",
            "git log | head -3",
            "git show HEAD:README.md",
            "ls -la",
            "cat pyproject.toml",
            "find . -name '*.py' | wc -l",
            "ps aux",
            "df -h",
            "jq '.project' pyproject.toml",
            "git status --porcelain",
        ):
            self.assertIsNone(refusal_reason(command), command)

    def test_writes_and_root_are_refused(self) -> None:
        cases = {
            "rm -rf state": "refused token",
            "echo hi > out.txt": "refused token",
            "ls | tee listing.txt": "refused token",
            "git commit -m 'x'": "read-only git",
            "git add -A": "read-only git",
            "git clean -fdx": "read-only git",
            "sudo ls /root": "refused token",
            "chmod 777 file": "refused token",
            "cat $(pwd)/file": "command substitution",
            "cat `whoami`": "command substitution",
            "ls && rm x": "refused token",
            "python3 -c 'import os'": "allowlist",
            "curl http://example.com": "allowlist",
            "env": "allowlist",
        }
        for command, fragment in cases.items():
            reason = refusal_reason(command)
            self.assertIsNotNone(reason, f"{command!r} must be refused")
            assert reason is not None
            self.assertIn(fragment, reason, command)

    def test_the_tool_runs_an_allowed_command_in_its_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "marker.txt").write_text("hello", encoding="utf-8")
            result = PlanBashTool(root).execute({"command": "ls"}, idempotency_key="k")
            self.assertTrue(result["ok"])
            payload = cast(dict[str, Any], result["result"])
            self.assertEqual(payload["exit_code"], 0)
            self.assertIn("marker.txt", payload["output"])

    def test_the_tool_refuses_with_the_reason_instead_of_running(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = PlanBashTool(Path(directory)).execute({"command": "rm -rf ."}, idempotency_key="k")
            self.assertFalse(result["ok"])
            self.assertIn("refused", str(result["error"]))

    def test_the_schema_names_the_read_only_surface(self) -> None:
        function = PlanBashTool(Path("/tmp")).schema["function"]
        self.assertEqual(function["name"], "bash_readonly")
        self.assertIn("read-only", function["description"])


class _MCPToolDouble:
    """A sensor tool as the plan toolset sees one: a read-only tool with a server name."""

    capability_kind = "read"

    def __init__(self, name: str, server: str) -> None:
        self.name = name
        self.client = _ClientDouble(server)
        self.schema: dict[str, Any] = {"type": "function", "function": {"name": name, "description": f"{server} {name}", "parameters": {}}}

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        return {"ok": True}


class _ClientDouble:
    """What the toolset reads off an MCP client: the server's name."""

    def __init__(self, server_name: str) -> None:
        self.server_name = server_name


class _OwnToolDouble:
    """An own tool as the plan toolset sees one."""

    def __init__(self, name: str, *, kind: str = "read") -> None:
        self.name = name
        self.capability_kind = kind
        self.schema: dict[str, Any] = {"type": "function", "function": {"name": name, "description": name, "parameters": {}}}

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        return {"ok": True}


class PlanToolsetTest(unittest.TestCase):
    @staticmethod
    def _own(name: str, *, kind: str = "read") -> _OwnToolDouble:
        return _OwnToolDouble(name, kind=kind)

    def test_the_toolset_is_read_only_and_named_as_the_model_sees_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            available: dict[str, Any] = {name: self._own(name) for name in ("webfetch", "read", "grep", "structure", "bash", "write")}
            available["search"] = _MCPToolDouble("search", "ddg")
            available["fetch_content"] = _MCPToolDouble("fetch_content", "ddg")
            available["get_abstract"] = _MCPToolDouble("get_abstract", "arxiv")
            described = plan_toolset(available, memory_tool=MemoryTool(store, actions=("search",)), root=directory)
            self.assertEqual(sorted(described), ["bash_readonly", "fetch_content", "grep", "memory", "read", "search", "structure", "webfetch"])
            self.assertNotIn("bash", described, "the writing shell is not the plan pass's shell")
            self.assertNotIn("write", described)
            self.assertNotIn("get_abstract", described, "only the named sensor serves the plan pass")
            for tool in described.values():
                self.assertEqual(str(getattr(tool, "capability_kind", "read")), "read", f"{getattr(tool, 'name', '?')} must be read-only")
            self.assertEqual(described["memory"].schema["function"]["parameters"]["properties"]["action"]["enum"], ["search"])
            store.close()

    def test_a_write_capable_tool_in_the_map_is_a_construction_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            available: dict[str, Any] = {name: self._own(name) for name in ("webfetch", "read", "grep", "structure")}
            available["read"] = self._own("read", kind="write")
            with self.assertRaises(ValueError):
                plan_toolset(available, memory_tool=MemoryTool(store, actions=("search",)), root=directory)
            store.close()

    def test_the_schemas_carry_the_read_only_shell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            available: dict[str, Any] = {name: self._own(name) for name in ("read",)}
            described = plan_toolset(available, memory_tool=MemoryTool(store, actions=("search",)), root=directory)
            names = [schema["function"]["name"] for schema in plan_tool_schemas(described)]
            self.assertIn("bash_readonly", names)
            self.assertIn("memory", names)
            store.close()


class MemoryToolRestrictionTest(unittest.TestCase):
    def test_a_search_only_instance_refuses_curation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            tool = MemoryTool(store, actions=("search",))
            self.assertEqual(tool.capability_kind, "read")
            result = tool.execute({"action": "remember", "content": "x"}, idempotency_key="k")
            self.assertFalse(result["ok"])
            self.assertIn("not available on this instance", str(result["error"]))
            store.close()

    def test_the_full_instance_still_declares_its_write_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            self.assertEqual(MemoryTool(store).capability_kind, "write")
            store.close()

    def test_an_unknown_action_is_refused_at_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with self.assertRaises(ValueError):
                MemoryTool(store, actions=("search", "summon"))
            store.close()


if __name__ == "__main__":
    unittest.main()
