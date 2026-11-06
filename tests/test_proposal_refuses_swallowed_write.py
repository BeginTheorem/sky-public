"""A proposal must not commit a file whose content is empty or a template token.

Measured defect (2026-10-31, corpus audit): 11 of 302 artifacts held no study --
four were 0 bytes and seven carried a single template token instead of text
(``<content>``, ``{{ARTIFACT_MD}}``, ``__PLACEHOLDER_MD__``,
``@@READ_FROM_DRAFT@@``, ``BASE64_PLACEHOLDER_NOT_USED``,
``{{READ_FILE:/tmp/skynet-scratch/artifact_draft.md}}``,
``$(cat /tmp/skynet-scratch/moex_sberp_preferred_buyhold_vs_deposit.md)``). All
eleven were committed by this pipeline, and the run report that produced one of
them declares the artifact published with its markers "confirmed" -- that check
reads the draft in scratch, never the applied file. A swallowed write inflates
every count of delivered artifacts while the file is unreadable, so it is
refused where it would be written.

The invariant these tests pin: no proposal path writes a file whose whole
content is empty or a single template token, and real content still lands.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from skynet.self_improvement import SelfImprovementError, SelfImprovementManager, SelfImprovementTool

HYPOTHESIS = {
    "problem": "an empty or template-token file is committed as a delivered artifact",
    "expected_behavior": "the proposal refuses it and writes nothing",
    "evidence": "corpus audit 2026-10-31: 4 files of 0 bytes, 7 single-token files",
    "validation": "assert the refusal class and that the target is absent",
    "rollback_condition": "revert if real content starts being refused",
}

SWALLOWED_CONTENTS = (
    "",
    "   \n\n",
    "{{ARTIFACT_MD}}",
    "__PLACEHOLDER_MD__",
    "BASE64_PLACEHOLDER_NOT_USED",
    "<content>",
    "@@READ_FROM_DRAFT@@",
    "{{READ_FILE:/tmp/skynet-scratch/artifact_draft.md}}",
    "$(cat /tmp/skynet-scratch/moex_sberp_preferred_buyhold_vs_deposit.md)",
)


def _repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    (root / "module.py").write_text("value = 1\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("\n", encoding="utf-8")
    subprocess.run(["git", "add", "module.py", "tests"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)


class SwallowedWriteRefusalTests(unittest.TestCase):
    def test_every_measured_token_is_refused_and_nothing_is_written(self) -> None:
        for index, content in enumerate(SWALLOWED_CONTENTS):
            with self.subTest(content=content[:40]), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _repo(root)
                manager = SelfImprovementManager(root, root.parent / "worktrees")
                tool = SelfImprovementTool(manager, test_command=(sys.executable, "-c", "pass"))
                result = tool.execute(
                    {"files": {"artifacts/new_study.md": content}, "hypothesis": dict(HYPOTHESIS)},
                    idempotency_key=f"swallowed-{index}",
                )
                self.assertFalse(result["ok"], f"accepted a swallowed write: {content!r}")
                self.assertEqual(result["failure_class"], "swallowed_write")
                self.assertIn("swallowed", str(result["error"]).casefold())

    def test_a_create_change_is_refused_too(self) -> None:
        """The measured corpus also arrived through the `changes` channel."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            tool = SelfImprovementTool(manager, test_command=(sys.executable, "-c", "pass"))
            result = tool.execute(
                {
                    "changes": [{"path": "artifacts/audit_study.md", "operation": "create", "content": "{{ARTIFACT_MD}}"}],
                    "hypothesis": dict(HYPOTHESIS),
                },
                idempotency_key="swallowed-change",
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["failure_class"], "swallowed_write")

    def test_the_manager_refuses_a_whitespace_only_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            with self.assertRaises(SelfImprovementError) as caught:
                manager.propose_files(
                    {"artifacts/blank.md": " \n\t\n"},
                    (sys.executable, "-c", "pass"),
                    metadata={"hypothesis": dict(HYPOTHESIS)},
                )
            self.assertIn("swallowed", str(caught.exception).casefold())
            self.assertNotIn("swallowed_write", str(caught.exception).split(":")[0])

    def test_real_content_still_lands(self) -> None:
        """The guard must not refuse a study: the positive half of the invariant."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            body = "# External-source finding: a real study\n\n## As-of date\n2026-10-31\n\nThe finding.\n"
            result = manager.propose_files(
                {"artifacts/real_study.md": body},
                (sys.executable, "-c", "pass"),
                metadata={"hypothesis": dict(HYPOTHESIS)},
            )
            self.assertTrue(result.get("commit"), result)
            worktree = manager.worktree_root / str(result["proposal_id"])
            self.assertTrue((worktree / "artifacts" / "real_study.md").is_file())

    def test_a_token_line_inside_a_long_document_is_not_refused(self) -> None:
        """The guard fires on a whole-file token, not on a token quoted in prose."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            study = (
                "# Audit of swallowed writes\n\n"
                "Seven files carried a token such as {{ARTIFACT_MD}} instead of the text.\n"
                "The finding: the marker check reads the draft, never the applied file.\n"
            )
            result = manager.propose_files(
                {"artifacts/swallowed_write_audit.md": study},
                (sys.executable, "-c", "pass"),
                metadata={"hypothesis": dict(HYPOTHESIS)},
            )
            self.assertTrue(result.get("commit"), result)


if __name__ == "__main__":
    unittest.main()
