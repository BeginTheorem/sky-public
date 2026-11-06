"""The skills directory: an index in the prompt, the bodies on demand."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from helpers import FakeProvider

from skynet.reactor import SKILLS_BLOCK_MARKER, Reactor, ReactorConfig
from skynet.skills import (
    MAX_BODY_CHARS,
    MAX_DESCRIPTION_CHARS,
    MAX_INDEX_CHARS,
    SkillError,
    SkillTool,
    discover,
    index_block,
    read_skill,
    skills_root,
)

FRONT_MATTER_SKILL = """---
name: sample-procedure
description: How to do the thing without breaking the other thing.
---

## Steps

1. First step.
2. Second step.
"""


def write_skill(root: Path, filename: str, text: str) -> Path:
    directory = skills_root(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(text, encoding="utf-8")
    return path


class SkillDiscoveryTest(unittest.TestCase):
    def test_front_matter_names_the_skill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_skill(root, "whatever.md", FRONT_MATTER_SKILL)
            found = discover(root)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].name, "sample-procedure")
            self.assertEqual(found[0].description, "How to do the thing without breaking the other thing.")

    def test_a_plain_file_still_counts(self) -> None:
        # A procedure written in a hurry is still a procedure: the stem names it and
        # the first line of prose describes it.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_skill(root, "restart-a-unit.md", "# Restart a unit\n\nThe safe way.\n")
            found = discover(root)
            self.assertEqual(found[0].name, "restart-a-unit")
            self.assertEqual(found[0].description, "Restart a unit")

    def test_documentation_is_not_a_skill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_skill(root, "README.md", "# Skills\n\nHow this directory works.\n")
            write_skill(root, "_draft.md", "Half-written idea.\n")
            write_skill(root, ".hidden.md", "Not a skill either.\n")
            self.assertEqual(discover(root), ())

    def test_a_file_with_nothing_to_say_is_not_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_skill(root, "blank.md", "---\nname: blank\n---\n\n")
            write_skill(root, "Bad Name.md", "Whatever.\n")
            self.assertEqual(discover(root), ())

    def test_one_name_is_published_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_skill(root, "a.md", "---\nname: same\ndescription: first.\n---\nbody\n")
            write_skill(root, "b.md", "---\nname: same\ndescription: second.\n---\nbody\n")
            found = discover(root)
            self.assertEqual([skill.name for skill in found], ["same"])
            self.assertEqual(found[0].description, "first.")

    def test_a_missing_directory_is_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(discover(Path(directory)), ())


class SkillIndexTest(unittest.TestCase):
    def test_no_skills_means_no_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(index_block(Path(directory)), "")

    def test_the_index_names_every_skill_with_one_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_skill(root, "one.md", FRONT_MATTER_SKILL)
            write_skill(root, "two.md", "Second procedure.\nMore text that is body, not index.\n")
            text = index_block(root)
            self.assertIn("sample-procedure: How to do the thing", text)
            self.assertIn("two: Second procedure.", text)
            self.assertIn("skill(action='read'", text)
            self.assertNotIn("First step.", text, "the index must not carry bodies")

    def test_a_long_description_is_clipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_skill(root, "long.md", f"---\nname: long\ndescription: {'x' * 500}\n---\nbody\n")
            line = next(line for line in index_block(root).splitlines() if line.startswith("- long:"))
            self.assertLessEqual(len(line) - len("- long: "), MAX_DESCRIPTION_CHARS)
            self.assertTrue(line.endswith("…"))

    def test_the_whole_index_is_clipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(30):
                write_skill(root, f"skill-{index:02d}.md", f"---\nname: skill-{index:02d}\ndescription: {'y' * 300}\n---\nbody\n")
            text = index_block(root)
            self.assertIn("index truncated", text)
            self.assertLessEqual(len(text), MAX_INDEX_CHARS + 60)


class SkillReadTest(unittest.TestCase):
    def test_read_returns_the_body_without_front_matter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_skill(root, "one.md", FRONT_MATTER_SKILL)
            body = read_skill(root, "sample-procedure")
            self.assertIn("## Steps", body)
            self.assertNotIn("description:", body)

    def test_read_refuses_a_name_that_is_a_path(self) -> None:
        # The lookup runs against the discovered set, so a name can never become a path.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_skill(root, "one.md", FRONT_MATTER_SKILL)
            for name in ("../outside", "/etc/passwd", "sub/one", "..", ""):
                with self.assertRaises(SkillError):
                    read_skill(root, name)

    def test_an_unknown_name_lists_what_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_skill(root, "one.md", FRONT_MATTER_SKILL)
            with self.assertRaises(SkillError) as caught:
                read_skill(root, "no-such-skill")
            self.assertIn("sample-procedure", str(caught.exception))

    def test_a_long_body_is_truncated_with_a_notice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_skill(root, "big.md", "---\nname: big\ndescription: big.\n---\n" + "z" * (MAX_BODY_CHARS * 2))
            body = read_skill(root, "big")
            self.assertIn(f"[truncated at {MAX_BODY_CHARS} chars]", body)
            self.assertLessEqual(len(body), MAX_BODY_CHARS + 40)


class SkillToolTest(unittest.TestCase):
    def tool(self, root: Path) -> SkillTool:
        return SkillTool(root)

    def test_list_names_every_skill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_skill(root, "one.md", FRONT_MATTER_SKILL)
            write_skill(root, "two.md", "Second procedure.\n")
            result = self.tool(root).execute({"action": "list"}, idempotency_key="k")
            self.assertTrue(result["ok"])
            payload = cast(dict[str, Any], result["result"])
            self.assertEqual(payload["count"], 2)
            self.assertEqual([item["name"] for item in payload["skills"]], ["sample-procedure", "two"])

    def test_read_returns_the_body(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_skill(root, "one.md", FRONT_MATTER_SKILL)
            result = self.tool(root).execute({"action": "read", "name": "sample-procedure"}, idempotency_key="k")
            self.assertTrue(result["ok"])
            payload = cast(dict[str, Any], result["result"])
            self.assertIn("Second step.", payload["body"])

    def test_read_needs_a_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.tool(Path(directory)).execute({"action": "read"}, idempotency_key="k")
            self.assertFalse(result["ok"])
            self.assertIn("name is required", cast(str, result["error"]))

    def test_an_unknown_action_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.tool(Path(directory)).execute({"action": "write"}, idempotency_key="k")
            self.assertFalse(result["ok"])
            self.assertIn("action must be one of", cast(str, result["error"]))

    def test_an_unknown_skill_is_an_error_not_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.tool(Path(directory)).execute({"action": "read", "name": "ghost"}, idempotency_key="k")
            self.assertFalse(result["ok"])
            self.assertIn("unknown skill", cast(str, result["error"]))

    def test_an_empty_directory_lists_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.tool(Path(directory)).execute({"action": "list"}, idempotency_key="k")
            self.assertTrue(result["ok"])
            payload = cast(dict[str, Any], result["result"])
            self.assertEqual(payload["count"], 0)

    def test_the_tool_carries_a_schema(self) -> None:
        function = SkillTool(Path("/tmp")).schema["function"]
        self.assertEqual(function["name"], "skill")
        self.assertEqual(function["parameters"]["properties"]["action"]["enum"], ["list", "read"])


class SkillsInThePromptTest(unittest.TestCase):
    def test_a_skill_lands_in_the_prompt_as_an_index_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_skill(root, "one.md", FRONT_MATTER_SKILL)
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root))
            prompt = reactor.config.system_prompt
            self.assertIn(SKILLS_BLOCK_MARKER, prompt)
            self.assertIn("sample-procedure", prompt)
            self.assertNotIn("Second step.", prompt, "bodies are read on demand, never injected")
            self.assertIn("skill", reactor.runner.tools)
            reactor.close()

    def test_no_skills_no_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root))
            self.assertNotIn(SKILLS_BLOCK_MARKER, reactor.config.system_prompt)
            reactor.close()


if __name__ == "__main__":
    unittest.main()
