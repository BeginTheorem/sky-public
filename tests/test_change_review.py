"""The deterministic change review rejects degeneracy and warns otherwise.

Measured motivation: the dead helper ``_parse_finish_report_with_error`` and the
dead flag ``use_portfolio_planner`` were both promoted by the gate and reverted
by hand, and comment-only edits have claimed mechanisms. The review is the
deterministic core of RRSI's critic (arXiv:2609.24972): no model, no benchmark.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from helpers import commit_all, git_repo

from skynet.change_review import FileChange, review_change
from skynet.self_improvement import SelfImprovementError, SelfImprovementManager, classify_failure


class ReviewUnitTests(unittest.TestCase):
    def _review(self, change: FileChange, *, files: dict[str, str] | None = None) -> list:
        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory)
            for name, content in (files or {}).items():
                path = worktree / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            return review_change([change], worktree=worktree)

    def test_comment_only_python_edit_is_rejected(self) -> None:
        findings = self._review(FileChange("mod.py", "# note\n", "# changed\n"))
        self.assertEqual([f.code for f in findings if f.severity == "reject"], ["degenerate_change"])

    def test_whitespace_only_edit_is_rejected(self) -> None:
        findings = self._review(FileChange("mod.py", "value = 1\n", "value  =  1\n"))
        self.assertEqual([f.code for f in findings if f.severity == "reject"], ["degenerate_change"])

    def test_deleting_only_a_comment_is_rejected(self) -> None:
        findings = self._review(FileChange("mod.py", "# only a comment\n", None))
        self.assertEqual([f.code for f in findings if f.severity == "reject"], ["degenerate_change"])

    def test_empty_new_file_is_not_a_degenerate_change(self) -> None:
        self.assertEqual(review_change([FileChange("pkg/__init__.py", None, "\n")], worktree=Path("/nonexistent")), [])

    def test_a_modification_is_not_degenerate(self) -> None:
        findings = self._review(FileChange("mod.py", "value = 1\n", "value = 2\n"))
        self.assertEqual(findings, [])

    def test_byte_identical_content_is_not_reviewed(self) -> None:
        # A mode-only change (chmod) has no textual difference; the gate owns it.
        findings = self._review(FileChange("mod.py", "value = 1\n", "value = 1\n"))
        self.assertEqual(findings, [])

    def test_whitespace_inside_a_string_is_not_a_whitespace_only_edit(self) -> None:
        findings = self._review(FileChange("mod.py", 'MSG = "hello world"\n', 'MSG = "helloworld"\n'))
        self.assertEqual(findings, [])

    def test_a_private_symbol_referenced_in_a_non_python_file_is_not_flagged(self) -> None:
        text = "def _helper():\n    return 1\n"
        findings = self._review(
            FileChange("pkg.py", None, text),
            files={"pkg.py": text, "config/notes.toml": 'name = "_helper"\n'},
        )
        self.assertEqual(findings, [])

    def test_a_markdown_heading_only_edit_is_not_degenerate(self) -> None:
        findings = self._review(FileChange("README.md", "text\n", "# New heading\n"))
        self.assertEqual(findings, [])

    def test_new_private_symbol_without_a_reference_is_rejected(self) -> None:
        text = "def _dead():\n    return 1\n"
        findings = self._review(FileChange("pkg.py", None, text), files={"pkg.py": text})
        rejected = [f for f in findings if f.severity == "reject"]
        self.assertEqual([f.code for f in rejected], ["unused_symbol"])
        self.assertIn("_dead", rejected[0].detail)

    def test_new_unused_constant_is_rejected(self) -> None:
        text = "MAX_ROWS = 5\n"
        findings = self._review(FileChange("pkg.py", None, text), files={"pkg.py": text})
        self.assertEqual([f.code for f in findings if f.severity == "reject"], ["unused_symbol"])

    def test_a_referenced_private_symbol_is_not_flagged(self) -> None:
        text = "def _live():\n    return 1\n\n_live()\n"
        findings = self._review(FileChange("pkg.py", None, text), files={"pkg.py": text})
        self.assertEqual(findings, [])

    def test_a_decorated_private_symbol_is_not_flagged(self) -> None:
        text = "@decorator\ndef _handled():\n    return 1\n"
        findings = self._review(FileChange("pkg.py", None, text), files={"pkg.py": text})
        self.assertEqual(findings, [])

    def test_a_new_public_symbol_is_only_warned(self) -> None:
        text = "def public_addition():\n    return 1\n"
        findings = self._review(FileChange("pkg.py", None, text), files={"pkg.py": text})
        self.assertEqual([(f.code, f.severity) for f in findings], [("unused_symbol", "warn")])

    def test_new_private_method_without_a_reference_is_rejected(self) -> None:
        # The dead MemoryStore TMT helpers were added as private methods and
        # slipped past the module-level rule; a class body is now scanned too.
        text = "class Widget:\n    def _dead(self):\n        return 1\n"
        findings = self._review(FileChange("pkg.py", None, text), files={"pkg.py": text})
        rejected = [f for f in findings if f.severity == "reject"]
        self.assertEqual([f.code for f in rejected], ["unused_symbol"])
        self.assertIn("_dead", rejected[0].detail)

    def test_a_referenced_private_method_is_not_flagged(self) -> None:
        text = "class C:\n    def _live(self):\n        return 1\n\n    def run(self):\n        return self._live()\n\n\nC()\n"
        findings = self._review(FileChange("pkg.py", None, text), files={"pkg.py": text})
        self.assertEqual(findings, [])

    def test_a_decorated_private_method_is_not_flagged(self) -> None:
        text = "class C:\n    @property\n    def _x(self):\n        return 1\n\n\nC()\n"
        findings = self._review(FileChange("pkg.py", None, text), files={"pkg.py": text})
        self.assertEqual(findings, [])

    def test_a_new_public_method_is_only_warned(self) -> None:
        text = "class Widget:\n    def helper(self):\n        return 1\n"
        findings = self._review(FileChange("pkg.py", None, text), files={"pkg.py": text})
        self.assertEqual([f for f in findings if f.severity == "reject"], [])
        self.assertNotIn("helper", "".join(f.detail for f in findings))

    def test_an_existing_private_method_is_not_flagged(self) -> None:
        text = "class C:\n    def _dead(self):\n        return 1\n"
        findings = self._review(FileChange("pkg.py", text, text), files={"pkg.py": text})
        self.assertEqual(findings, [])


def _repo(root: Path) -> None:
    git_repo(root)
    (root / "module.py").write_text("value = 1\n# note\n", encoding="utf-8")
    commit_all(root, "base")


class GateIntegrationTests(unittest.TestCase):
    def test_comment_only_change_is_rejected_by_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            manager = SelfImprovementManager(root, root / "worktrees")
            proposal = manager.propose()
            try:
                manager.apply_changes(proposal, [{"path": "module.py", "operation": "replace", "old": "# note", "new": "# changed"}])
                with self.assertRaises(SelfImprovementError) as caught:
                    manager.validate_and_commit(proposal, (sys.executable, "-c", "pass"), applied_paths=["module.py"], fingerprint="f")
                self.assertEqual(caught.exception.reason_code, "degenerate_change")
                self.assertEqual(classify_failure(caught.exception), "degenerate_change")
            finally:
                manager.discard(proposal)

    def test_new_unused_private_helper_is_rejected_by_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            manager = SelfImprovementManager(root, root / "worktrees")
            proposal = manager.propose()
            try:
                manager.apply_files(proposal, {"skynet/dead_helper.py": "def _dead():\n    return 1\n"})
                with self.assertRaises(SelfImprovementError) as caught:
                    manager.validate_and_commit(proposal, (sys.executable, "-c", "pass"), applied_paths=["skynet/dead_helper.py"], fingerprint="f")
                self.assertEqual(caught.exception.reason_code, "degenerate_change")
            finally:
                manager.discard(proposal)

    def test_a_real_change_passes_the_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            manager = SelfImprovementManager(root, root / "worktrees", gate_suite=(sys.executable, "-c", "pass"))
            proposal = manager.propose()
            try:
                manager.apply_changes(proposal, [{"path": "module.py", "operation": "replace", "old": "value = 1", "new": "value = 2"}])
                commit = manager.validate_and_commit(proposal, (sys.executable, "-c", "pass"), applied_paths=["module.py"], fingerprint="f")
                self.assertNotEqual(commit, proposal.base_commit)
            finally:
                manager.discard(proposal)


if __name__ == "__main__":
    unittest.main()
