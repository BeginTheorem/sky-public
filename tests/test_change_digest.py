"""The change digest must identify the patch, not only the patched path.

Measured collision fixed here: proposals b1a54578 (four type guards)
and 0d3b1925 (a ``json_tree`` holder subquery) both edit only
``skynet/store.py`` and were both reported as change_digest ``6248ede0...``
because the digest was ``sha256`` over the sorted path names. The registry and
the runtime log carry that value as the change identity, so a path-only digest
cannot tell two promotions of one file apart.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

from skynet.self_improvement import change_content_digest


def _repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=root, check=True)
    (root / "mod.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "mod.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)


class ChangeDigestTest(unittest.TestCase):
    def test_two_patches_to_one_path_do_not_share_a_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            (root / "mod.py").write_text("value = 2\n", encoding="utf-8")
            first = change_content_digest(root, ["mod.py"])
            (root / "mod.py").write_text("value = 3\n", encoding="utf-8")
            second = change_content_digest(root, ["mod.py"])
            self.assertNotEqual(first, second)

    def test_digest_is_content_derived_and_order_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            (root / "mod.py").write_text("value = 2\n", encoding="utf-8")
            (root / "other.py").write_text("x = 1\n", encoding="utf-8")
            left = change_content_digest(root, ["mod.py", "other.py"])
            right = change_content_digest(root, ["other.py", "mod.py"])
            self.assertEqual(left, right)
            # Replaying the same change yields the same identity, across roots.
            with tempfile.TemporaryDirectory() as second_directory:
                other = Path(second_directory)
                _repo(other)
                (other / "mod.py").write_text("value = 2\n", encoding="utf-8")
                (other / "other.py").write_text("x = 1\n", encoding="utf-8")
                self.assertEqual(left, change_content_digest(other, ["mod.py", "other.py"]))

    def test_deletion_does_not_collapse_to_the_empty_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            self.assertNotEqual(change_content_digest(root, ["gone.py"]), change_content_digest(root, []))


if __name__ == "__main__":
    unittest.main()
