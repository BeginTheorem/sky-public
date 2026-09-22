"""The published tree must not carry the operator's personal paths.

Infrastructure values live in gitignored ``config/*.env`` files; a *tracked*
file that hardcodes the operator's home directory makes a public copy fail on
any other host, and the failure is silent until deploy. This guard keeps the
de-hardcoding from regressing: every tracked file is scanned, so a new script or
unit template cannot reintroduce the path unnoticed.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Built by concatenation so this guard does not flag itself.
_OPERATOR = "lord" + "-" + "lucifer"
FORBIDDEN = (_OPERATOR, "/home/" + _OPERATOR)

# Tracked text files worth scanning; binaries (wheels, sqlite, caches) are skipped.
TEXT_SUFFIXES = {
    ".py", ".sh", ".in", ".conf", ".toml", ".md", ".example", ".ini", ".cfg",
    ".txt", ".yml", ".yaml", ".json", ".service", ".env",
}


def _tracked_files() -> list[Path]:
    """Return the tracked files, or an empty list when git is unavailable."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [REPO_ROOT / name for name in result.stdout.split("\0") if name]


class NoHardcodedPersonalPathsTests(unittest.TestCase):
    def test_tracked_files_do_not_carry_the_operator_path(self) -> None:
        offenders: list[str] = []
        for path in _tracked_files():
            if path.suffix not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if any(token in text for token in FORBIDDEN):
                offenders.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(
            offenders,
            [],
            f"tracked files still hardcode the operator path: {offenders}",
        )

    def test_deploy_units_are_templates(self) -> None:
        """Unit files must ship as @ROOT@ templates, not rendered copies."""
        for name in ("skynet.service.in", "skynet-telegram.service.in", "skynet-rollback.service.in"):
            text = (REPO_ROOT / "deploy" / name).read_text(encoding="utf-8")
            self.assertIn("@ROOT@", text, f"{name} lost its @ROOT@ placeholder")
            self.assertNotIn("/home/", text, f"{name} carries an absolute home path")


if __name__ == "__main__":
    unittest.main()
