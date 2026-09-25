"""A proposal that supplies new files AND exact-anchor changes must apply both.

Measured defect (generation 186): proposal
073938d987a64ff0ba9b7a7da8a460db was submitted through
``propose_self_improvement`` with ``files={"tests/test_executed_evidence_invariant.py": ...}``
plus nine anchor changes to ``skynet/react.py``. The tool result's
``applied.files`` listed only ``skynet/react.py``, the registry recorded
``files_changed: 1``, and the promoted commit 9088322 touches only
``skynet/react.py`` -- the new test file the run's own report claims was
silently dropped. Root cause: ``propose_files`` branches
``if patch / elif changes / else`` and calls ``apply_files`` only in the
``else`` branch, so a request carrying both is only half applied and the gate
still reports success.

The invariant these tests pin: every path (new content) named in a proposal
must exist in the committed worktree, whatever combination of ``files``,
``changes`` and ``patch`` the proposal carried.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast

from skynet.self_improvement import SelfImprovementManager, SelfImprovementTool

HYPOTHESIS = {
    "problem": "a proposal with files+changes drops the new files",
    "expected_behavior": "both the new file and the edited file are committed",
    "evidence": "generation 186 proposal 073938d987a64ff0ba9b7a7da8a460db",
    "validation": "assert both paths exist after propose_files returns",
    "rollback_condition": "revert if an existing-file path is applied without an anchor",
}


def _repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    (root / "module.py").write_text("value = 1\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("\n", encoding="utf-8")
    (root / "tests" / "test_smoke.py").write_text(
        "import unittest\n\nclass Smoke(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "module.py", "tests"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)


class ProposalAppliesEveryNamedPathTests(unittest.TestCase):
    def test_new_file_is_applied_alongside_anchor_changes(self) -> None:
        """The measured defect: files + changes must apply both, not just changes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            result = manager.propose_files(
                {"skynet/new_module.py": "value = 1\n"},
                (sys.executable, "-c", "pass"),
                changes=[{"path": "module.py", "operation": "replace", "old": "value = 1", "new": "value = 2"}],
                metadata={"hypothesis": HYPOTHESIS},
            )
            proposal = manager.worktree_root / str(result["proposal_id"])
            self.assertTrue(
                (proposal / "skynet" / "new_module.py").exists(),
                "the new file named in `files` was not applied alongside `changes`",
            )
            self.assertEqual((proposal / "module.py").read_text(encoding="utf-8"), "value = 2\n")
            committed = subprocess.run(
                ["git", "show", "--name-only", "--format=", str(result["commit"])],
                cwd=proposal,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
            self.assertIn("skynet/new_module.py", cast(list[str], committed))
            self.assertIn("module.py", cast(list[str], committed))

    def test_new_file_is_applied_alongside_patch(self) -> None:
        """The same drop happens on the `patch` branch, which is tried first."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            patch = (
                "diff --git a/module.py b/module.py\n"
                "--- a/module.py\n"
                "+++ b/module.py\n"
                "@@ -1 +1 @@\n"
                "-value = 1\n"
                "+value = 3\n"
            )
            result = manager.propose_files(
                {"tests/test_from_patch.py": "def test_p():\n    assert True\n"},
                (sys.executable, "-c", "pass"),
                patch=patch,
                metadata={"hypothesis": HYPOTHESIS},
            )
            # `applied_paths` must be the union of both channels, not only the
            # patch's: with the new file missing from it the gate-protected
            # check never sees the test file, so the drop is silent here too.
            self.assertEqual(result.get("status"), "warned_protected")
            self.assertIn("tests/test_from_patch.py", cast(list[str], result.get("protected_paths", [])))

    def test_the_live_case_new_test_file_beside_react_changes_is_not_swallowed(self) -> None:
        """The exact shape that was silently dropped on generation 186.

        Proposal 073938d987a64ff0ba9b7a7da8a460db carried
        ``files={"tests/test_executed_evidence_invariant.py": ...}`` together
        with anchor changes to ``skynet/react.py``. Because only the ``changes``
        branch ran, the new file was never applied -- and because
        ``applied_paths`` therefore named only ``skynet/react.py``, the
        gate-protected check never saw the test file either, so the drop was
        silent in both directions. The patched code must apply the file and then
        warn, exactly as a files-only proposal naming that path would.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            result = manager.propose_files(
                {"tests/test_executed_evidence_invariant.py": "def test_ok():\n    assert True\n"},
                (sys.executable, "-c", "pass"),
                changes=[{"path": "module.py", "operation": "replace", "old": "value = 1", "new": "value = 2"}],
                metadata={"hypothesis": HYPOTHESIS},
            )
            self.assertEqual(result.get("status"), "warned_protected")
            self.assertIn("tests/test_executed_evidence_invariant.py", cast(list[str], result.get("protected_paths", [])))

    def test_a_files_entry_for_an_existing_path_is_refused_loudly(self) -> None:
        """The `files` channel stays new-files-only, even beside `changes`.

        Driven through the tool, which is the surface a proposal actually
        arrives on: with `changes` present the old guard was skipped, so this
        request would have overwritten the tracked file with no anchor.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            tool = SelfImprovementTool(manager, test_command=(sys.executable, "-c", "pass"))
            result = tool.execute(
                {
                    "files": {"module.py": "clobbered = True\n"},
                    "changes": [{"path": "module.py", "operation": "replace", "old": "value = 1", "new": "value = 2"}],
                    "hypothesis": dict(HYPOTHESIS),
                },
                idempotency_key="files-existing-beside-changes",
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["failure_class"], "invalid_payload")
            self.assertIn("module.py", str(result["error"]))
            self.assertEqual((root / "module.py").read_text(encoding="utf-8"), "value = 1\n")


if __name__ == "__main__":
    unittest.main()
