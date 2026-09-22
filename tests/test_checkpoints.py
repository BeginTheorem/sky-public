"""Split from the former monolithic CoreTests suite."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from skynet.checkpoints import CheckpointError, CheckpointManager


class CoreTests(unittest.TestCase):
    def test_checkpoint_boundary_rejects_untracked_source_and_tracked_edits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            tracked = root / "tracked.txt"
            tracked.write_text("base", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            (root / "tests").mkdir()
            (root / "tests" / "test_recovery.py").write_text("# untracked", encoding="utf-8")
            manager = CheckpointManager(root)
            self.assertFalse(manager.is_clean())
            self.assertFalse(manager.is_clean(allow_untracked=True))
            (root / "tests" / "test_recovery.py").unlink()
            self.assertTrue(manager.is_clean(allow_untracked=True))
            tracked.write_text("modified", encoding="utf-8")
            self.assertFalse(manager.is_clean(allow_untracked=True))
    def test_checkpoint_manager_requires_clean_worktree_for_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            import subprocess
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "tracked.txt").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            manager = CheckpointManager(root)
            checkpoint = manager.create({"ok": True})
            self.assertEqual(checkpoint.commit, manager.current_commit())
            (root / "tracked.txt").write_text("changed\n", encoding="utf-8")
            self.assertEqual(manager.rollback(), checkpoint.commit)
            self.assertEqual((root / "tracked.txt").read_text(encoding="utf-8"), "one\n")
    def test_checkpoint_boundary_rejects_dirty_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            import subprocess
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "tracked.txt").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            (root / "tracked.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaises(CheckpointError):
                CheckpointManager(root).create({"ok": True})
    def test_checkpoint_ignores_runtime_artifacts(self) -> None:
        cases = (
            {".deployed-commit": "abc123\n", ".deployment-refresh": "\n"},
            {"inspirations/agentos-mcp/.pytest-cache/marker": "generated\n"},
        )
        for artifacts in cases:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                subprocess.run(["git", "init", "-q"], cwd=root, check=True)
                subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
                subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
                (root / "tracked.txt").write_text("one\n", encoding="utf-8")
                subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
                subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
                for relative, content in artifacts.items():
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                self.assertTrue(CheckpointManager(root).is_clean(), artifacts)


class AncestryTests(unittest.TestCase):
    def test_is_ancestor_accepts_an_ancestor_and_rejects_a_stranger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "f.txt").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "f.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "one"], cwd=root, check=True)
            first = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            (root / "f.txt").write_text("two\n", encoding="utf-8")
            subprocess.run(["git", "add", "f.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "two"], cwd=root, check=True)
            second = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            # An unrelated commit that is not reachable from HEAD.
            subprocess.run(["git", "checkout", "-q", "--detach", first], cwd=root, check=True)
            (root / "g.txt").write_text("side\n", encoding="utf-8")
            subprocess.run(["git", "add", "g.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "side"], cwd=root, check=True)
            side = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            subprocess.run(["git", "checkout", "-q", second], cwd=root, check=True)

            checkpoint = CheckpointManager(root)
            self.assertTrue(checkpoint.is_ancestor(first))
            self.assertTrue(checkpoint.is_ancestor(second))
            # A stale rollback_commit from an older promotion must be refused:
            # resetting to it would discard every later commit.
            self.assertFalse(checkpoint.is_ancestor(side))
            self.assertFalse(checkpoint.is_ancestor("0" * 40))
