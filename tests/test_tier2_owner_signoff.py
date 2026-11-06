"""Tier-2: a change to the owner's own levers is not promoted on the organism's word.

Measured 2026-10-31 on the live registry: every promotion this pipeline ever made was decided
inside the organism -- propose, test, merge, request its own restart -- and nothing in that
path required an act from outside the process, including for the scripts that deploy, roll
back and judge it. Project Chrysalis draws the opposite line and calls it dual-custody human
approval; their critic's reading is the one that matters here: the gate around the agent, not
the agent's speed, is what makes self-evolution governable.

The invariant these tests pin: a proposal touching a path that carries the owner's levers (the
gate scripts, deploy and rollback, the unit files, the reboot request, the checkpoint and sudo
machinery) is written and tested like any other, but it stays parked until the owner answers a
durable question, silence is a refusal, and only ``Разрешить`` resumes it.

Deliberately NOT in that list: the gate-protected meta-loop paths (``skynet/self_improvement.py``,
``planner.py``, ``metrics.py``, ``idea_archive.py``, ``autonomous_planner.py``). The owner decided
those are warned, not held, so the organism can repair its own judge after an identical
resubmission; one test pins that decision so no later change reverses it by accident.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from skynet.self_improvement import (
    SelfImprovementError,
    SelfImprovementManager,
    SelfImprovementTool,
    owner_signoff_paths,
)

HYPOTHESIS = {
    "problem": "a change to the deploy script promotes itself",
    "expected_behavior": "it parks until the owner answers",
    "evidence": "registry audit 2026-10-31: no promotion ever required an outside act",
    "validation": "assert the parked status, the durable question and an unmoved HEAD",
    "rollback_condition": "revert if a real improvement starts being refused",
}

AUTHORITY_CHANGE = {"scripts/deploy.sh": "#!/bin/bash\nset -euo pipefail\necho deploy\n"}
ORDINARY_CHANGE = {"skynet/new_module.py": "value = 2\n"}


class FakeStore:
    """The slice of StateStore this boundary touches, with everything visible to assertions."""

    def __init__(self, answer: str | None = None) -> None:
        self.answer = answer
        self.questions: list[dict[str, Any]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.alerts: list[str] = []

    def append_event(self, kind: str, payload: dict[str, Any]) -> None:
        self.events.append((kind, payload))

    def raise_alert(self, kind: str, payload: dict[str, Any], **kwargs: Any) -> None:
        self.alerts.append(kind)

    def ask_question(self, question: str, **kwargs: Any) -> str:
        question_id = f"ask-{len(self.questions) + 1}"
        self.questions.append({"question_id": question_id, "question": question, **kwargs})
        return question_id

    def answer_for(self, question_id: str) -> dict[str, Any] | None:
        if self.answer is None:
            return None
        return {"question_id": question_id, "answer": self.answer, "source": "owner"}


def _repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    (root / "module.py").write_text("value = 1\n", encoding="utf-8")
    # The gate runs a suite stage of its own, and an empty suite exits non-zero: the repo
    # needs one passing test for any proposal to reach the promotion boundary at all.
    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("\n", encoding="utf-8")
    (root / "tests" / "test_smoke.py").write_text(
        "import unittest\n\nclass Smoke(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "module.py", "tests"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)


def _head(root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


class OwnerSignoffPathTests(unittest.TestCase):
    def test_the_list_holds_the_owner_levers_and_not_the_open_meta_loop(self) -> None:
        hits = owner_signoff_paths(
            [
                "scripts/test.sh",
                "./scripts/deploy.sh",
                "deploy/skynet.service.in",
                "state/reboot-request.json",
                "skynet/sudo_gate.py",
                "skynet/self_improvement.py",
                "skynet/planner.py",
                "skynet/store.py",
                "tests/test_something.py",
            ]
        )
        self.assertEqual(
            hits,
            sorted(
                [
                    "scripts/test.sh",
                    "scripts/deploy.sh",
                    "deploy/skynet.service.in",
                    "state/reboot-request.json",
                    "skynet/sudo_gate.py",
                ]
            ),
        )


class OwnerSignoffPromotionTests(unittest.TestCase):
    def _manager(self, root: Path, store: FakeStore | None = None) -> SelfImprovementManager:
        manager = SelfImprovementManager(root, root.parent / "worktrees")
        manager.set_store(store if store is not None else FakeStore())
        return manager

    def test_a_tier_two_change_parks_at_the_owner_and_never_merges_itself(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            before = _head(root)
            store = FakeStore()
            manager = self._manager(root, store)
            result = manager.propose_files(AUTHORITY_CHANGE, (sys.executable, "-c", "pass"), metadata={"hypothesis": dict(HYPOTHESIS)})
            record = manager._read_proposals()[str(result["proposal_id"])]
            with self.assertRaises(SelfImprovementError) as caught:
                manager.promote_pending(str(result["proposal_id"]), {"ok": True})
            self.assertEqual(getattr(caught.exception, "reason_code", ""), "owner_signoff_required")
            self.assertIn("scripts/deploy.sh", str(caught.exception))
            # Nothing moved: no merge, no reboot request.
            self.assertEqual(_head(root), before)
            self.assertFalse((root / "state" / "reboot-request.json").exists())
            parked = manager._read_proposals()[str(result["proposal_id"])]
            self.assertEqual(parked["status"], "awaiting_owner")
            self.assertEqual(parked["owner_signoff_paths"], ["scripts/deploy.sh"])
            # The owner was actually asked, once, in Telegram terms he can answer.
            self.assertEqual(len(store.questions), 1)
            self.assertIn("scripts/deploy.sh", store.questions[0]["question"])
            self.assertIn("Разрешить", store.questions[0]["question"])
            # The owner must read why: the hypothesis sits on the record itself, and an
            # earlier version read a nested key and asked him with an empty reason.
            self.assertIn(HYPOTHESIS["problem"], store.questions[0]["question"])
            self.assertIn(HYPOTHESIS["expected_behavior"], store.questions[0]["question"])
            self.assertEqual(parked["owner_question_id"], store.questions[0]["question_id"])
            self.assertTrue(any(kind == "proposal_awaiting_owner" for kind, _ in store.events))
            self.assertEqual(record["status"], "validated")

    def test_the_tool_reports_the_boundary_as_a_failure_the_run_can_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            before = _head(root)
            manager = self._manager(root)
            tool = SelfImprovementTool(manager, health_check=lambda: {"ok": True}, test_command=(sys.executable, "-c", "pass"))
            result = tool.execute(
                {"files": dict(AUTHORITY_CHANGE), "hypothesis": dict(HYPOTHESIS)},
                idempotency_key="tier-two",
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["failure_class"], "owner_signoff_required")
            self.assertEqual(_head(root), before)

    def test_silence_leaves_it_parked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            before = _head(root)
            store = FakeStore()
            manager = self._manager(root, store)
            result = manager.propose_files(AUTHORITY_CHANGE, (sys.executable, "-c", "pass"), metadata={"hypothesis": dict(HYPOTHESIS)})
            proposal_id = str(result["proposal_id"])
            with self.assertRaises(SelfImprovementError):
                manager.promote_pending(proposal_id, {"ok": True})
            self.assertEqual(manager.reconcile_awaiting_owner(), [])
            self.assertEqual(manager._read_proposals()[proposal_id]["status"], "awaiting_owner")
            self.assertEqual(_head(root), before)
            self.assertEqual(len(store.questions), 1)

    def test_the_owner_approval_resumes_the_parked_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            store = FakeStore()
            manager = self._manager(root, store)
            result = manager.propose_files(AUTHORITY_CHANGE, (sys.executable, "-c", "pass"), metadata={"hypothesis": dict(HYPOTHESIS)})
            proposal_id = str(result["proposal_id"])
            with self.assertRaises(SelfImprovementError):
                manager.promote_pending(proposal_id, {"ok": True})
            store.answer = "Разрешить"
            resolved = manager.reconcile_awaiting_owner()
            self.assertEqual([entry["status"] for entry in resolved], ["awaiting_reboot"])
            self.assertEqual(_head(root), str(result["commit"]))
            request = root / "state" / "reboot-request.json"
            self.assertTrue(request.exists())
            self.assertTrue(any(kind == "proposal_owner_approved" for kind, _ in store.events))

    def test_the_owner_refusal_rejects_the_parked_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            before = _head(root)
            store = FakeStore()
            manager = self._manager(root, store)
            result = manager.propose_files(AUTHORITY_CHANGE, (sys.executable, "-c", "pass"), metadata={"hypothesis": dict(HYPOTHESIS)})
            proposal_id = str(result["proposal_id"])
            with self.assertRaises(SelfImprovementError):
                manager.promote_pending(proposal_id, {"ok": True})
            store.answer = "Запретить"
            resolved = manager.reconcile_awaiting_owner()
            self.assertEqual([entry["status"] for entry in resolved], ["rejected"])
            self.assertEqual(manager._read_proposals()[proposal_id]["status"], "rejected")
            self.assertEqual(manager._read_proposals()[proposal_id]["failure_class"], "owner_refused")
            self.assertEqual(_head(root), before)
            self.assertFalse(Path(str(result["worktree"])).exists())

    def test_an_ordinary_change_still_promotes_without_asking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            store = FakeStore()
            manager = self._manager(root, store)
            result = manager.propose_files(ORDINARY_CHANGE, (sys.executable, "-c", "pass"), metadata={"hypothesis": dict(HYPOTHESIS)})
            promoted = manager.promote_pending(str(result["proposal_id"]), {"ok": True})
            self.assertEqual(promoted, str(result["commit"]))
            self.assertEqual(_head(root), str(result["commit"]))
            self.assertEqual(store.questions, [])


if __name__ == "__main__":
    unittest.main()
