"""Hard structural invariants that are about correctness, not tidiness.

These are the layering rules AGENTS.md states as invariants. A regression here
is a design break, not a style nit, so this is a real gate. The tree is parsed
with ``ast`` rather than imported, so the check needs no third-party dependency
and cannot be fooled by import side effects.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

SKYNET = Path(__file__).resolve().parent.parent / "skynet"


def _internal_imports(path: Path) -> set[str]:
    """Top-level ``skynet`` submodules imported by ``path`` (relative or absolute)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                found.add(node.module.split(".")[0])
            elif node.module and node.module.startswith("skynet."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("skynet."):
                    found.add(alias.name.split(".")[1])
    return found


def _module(name: str) -> Path:
    return SKYNET / f"{name}.py"


class DependencyContractTests(unittest.TestCase):
    def test_providers_stay_leaves(self) -> None:
        """A provider must not reach into the store, the lifecycle, or the app."""
        forbidden = {"store", "reactor", "react", "cli", "memory", "self_improvement", "supervisor"}
        offenders: dict[str, list[str]] = {}
        for path in sorted((SKYNET / "providers").glob("*.py")):
            hit = _internal_imports(path) & forbidden
            if hit:
                offenders[path.name] = sorted(hit)
        self.assertEqual(offenders, {}, f"providers must not import store/reactor/...: {offenders}")

    def test_memory_does_not_import_the_store(self) -> None:
        """memory is store-free by design -- which is why it cannot emit durable events."""
        self.assertNotIn("store", _internal_imports(_module("memory")))

    def test_store_does_not_import_the_reactor(self) -> None:
        """The reactor owns the lifecycle; the store must not depend on it."""
        self.assertNotIn("reactor", _internal_imports(_module("store")))

    def test_judge_health_is_a_pure_reader(self) -> None:
        """judge_health reads planner_attempts through arguments, not imports."""
        self.assertEqual(_internal_imports(_module("judge_health")), set())


if __name__ == "__main__":
    unittest.main()
