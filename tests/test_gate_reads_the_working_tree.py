"""A check must read the tree it is judging, never the committed tree behind it.

A check whose subject is this repository's own ``HEAD`` -- ``git archive HEAD``, ``git show
HEAD``, ``git ls-tree HEAD`` -- asks about the tree *as it was before the change under test*.
That shape is how a defect is added invisibly and then can never be removed: the check sees the
old tree and passes, the change commits, and from then on the check fails on a file that the very
change which would fix it is not allowed to touch. Measured twice in two days: the public-release
scanner refused ``artifacts/index.md:76`` and ``artifacts/fallback_parallel_strategy_wiring.md:128``
only after they had landed, and no proposal could repair either -- the second blocked every durable
change for six hours, leaving a root request as the organism's only channel.

So the rule here is mechanical: nothing in this repository may materialise the content of a bare
``HEAD``. Reading files is how a check about the working tree works; "is the committed tree clean?"
is a release question and belongs to the release step, wherever that step lives.

Naming a commit is not reading one, and a throwaway repository is not this one, so those stay
legal: ``rev-parse HEAD`` (the deploy marker and the rollback assertions), ``git show <hash>`` of
a commit a test just created, ``git reset --hard HEAD~1`` in a temporary fixture, and a policy
fixture that merely *quotes* a command string.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()
SCANNED_SUFFIXES = (".py", ".sh")

# Built by concatenation so this guard does not flag itself.
_GIT = "g" + "it"
_HEAD = "HE" + "AD"
_CONTENT_COMMANDS = ("archive", "show", "ls-tree", "cat-file")

# A git call that materialises committed content ...
_GIT_CONTENT = re.compile(rf"\b{_GIT}\b[^\n]*\b(?:{'|'.join(_CONTENT_COMMANDS)})\b")
# ... whose target is a bare HEAD: not HEAD:path, not HEAD~1, not HEAD^, not a hash.
_BARE_HEAD = re.compile(rf"""(?:["']{_HEAD}["']|(?<![\w:]){_HEAD}(?![\w:._~^/@-]))""")

# Reviewed exceptions: repository-relative path -> why it must read a commit. Empty on purpose:
# an entry is a hole somebody decided on, so it must never appear quietly.
ALLOWED: dict[str, str] = {}


def _tracked_files() -> list[Path]:
    """The tracked files, or an empty list when git is unavailable."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [REPO_ROOT / name for name in result.stdout.split("\0") if name]


class GateReadsTheWorkingTreeTests(unittest.TestCase):
    def test_no_tracked_code_materialises_the_committed_tree(self) -> None:
        offenders: list[str] = []
        for path in _tracked_files():
            if path.suffix not in SCANNED_SUFFIXES or path.resolve() == SELF:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in ALLOWED:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if _GIT_CONTENT.search(line) and _BARE_HEAD.search(line):
                    offenders.append(f"{rel}:{lineno}: {line.strip()[:120]}")
        self.assertEqual(
            offenders,
            [],
            "a check reads the committed tree instead of the working tree:\n  " + "\n  ".join(offenders),
        )

    def test_the_guard_still_recognises_the_shape_it_forbids(self) -> None:
        shapes = [
            f'subprocess.run(["{_GIT}", "archive", "--format=tar", "{_HEAD}"])',
            f'subprocess.run(["{_GIT}", "-C", str(REPO_ROOT), "archive", "{_HEAD}"])',
            f'run("{_GIT} show {_HEAD}")',
            f'subprocess.check_output(["{_GIT}", "ls-tree", "{_HEAD}"])',
        ]
        for shape in shapes:
            self.assertTrue(
                _GIT_CONTENT.search(shape) and _BARE_HEAD.search(shape),
                f"the guard must still catch: {shape}",
            )

    def test_legal_shapes_stay_legal(self) -> None:
        legal = [
            f'subprocess.check_output(["{_GIT}", "rev-parse", "{_HEAD}"])',  # naming a commit
            f'subprocess.run(["{_GIT}", "reset", "--hard", "{_HEAD}~1"], cwd=root)',  # a temp fixture
            f'subprocess.run(["{_GIT}", "show", "--format=", commit], cwd=proposal)',  # a hash
            f'"{_GIT} show {_HEAD}:README.md",',  # a policy fixture quoting a command
            "from skynet.idea_archive import cell_key",  # an idea archive is not a git archive
        ]
        for shape in legal:
            self.assertFalse(
                _GIT_CONTENT.search(shape) and _BARE_HEAD.search(shape),
                f"the guard must not flag: {shape}",
            )


if __name__ == "__main__":
    unittest.main()
