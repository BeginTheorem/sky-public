"""Split from the former monolithic CoreTests suite."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from helpers import commit_all, git_repo

from skynet.outbox import render_alert
from skynet.recovery import RebootGuard, RecoveryError
from skynet.self_improvement import (
    DEFAULT_TEST_COMMAND,
    ImprovementProposal,
    SelfImprovementError,
    SelfImprovementManager,
    SelfImprovementTool,
    classify_failure,
    gate_suite_environment,
    patch_structure_error,
)
from skynet.store import StateStore
from skynet.time import utc_now


def _manager(root: Path) -> SelfImprovementManager:
    """A manager whose worktrees live inside the test root.

    The default for a repository under /tmp is the shared
    /tmp/.skynet-improvements, which a root-run organism leaves root-owned; that
    made these tests fail on the server while passing locally.
    """
    return SelfImprovementManager(root, root / "worktrees")


class CoreTests(unittest.TestCase):
    def test_self_improvement_rejects_untracked_source_from_prior_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            tracked = root / "tracked.py"
            tracked.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            (root / "skynet").mkdir()
            (root / "skynet" / "prior_cycle.py").write_text("value = 2\n", encoding="utf-8")
            manager = SelfImprovementManager(root, root / "proposals")
            with self.assertRaises(SelfImprovementError):
                manager.propose()
    def test_self_improvement_requires_complete_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tool = SelfImprovementTool(SelfImprovementManager(Path(directory)))
            result = tool.execute({"files": {"new.txt": "x"}}, idempotency_key="missing-hypothesis")
            self.assertFalse(result["ok"])
            self.assertEqual(result["failure_class"], "invalid_payload")
    def test_gate_suite_environment_drops_only_service_provider_flags(self) -> None:
        environment = {
            "OLLAMA_ENABLED": "false",
            "NVIDIA_ENABLED": "false",
            "DEEPSEEK_ENABLED": "false",
            "SKYNET_MONEY_BOOST": "true",
            "PATH": "/usr/bin",
        }
        with patch.dict(os.environ, environment, clear=True):
            gate_env = gate_suite_environment()
        for name in ("OLLAMA_ENABLED", "NVIDIA_ENABLED", "DEEPSEEK_ENABLED", "SKYNET_MONEY_BOOST"):
            self.assertNotIn(name, gate_env)
        self.assertEqual(gate_env.get("PATH"), "/usr/bin")
    def test_self_improvement_schema_describes_hypothesis_and_patch_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = SelfImprovementManager(Path(directory))
            schema = cast(dict, SelfImprovementTool(manager).schema)["function"]["parameters"]
            hypothesis = schema["properties"]["hypothesis"]
            self.assertEqual(set(hypothesis["required"]), {"problem", "expected_behavior", "evidence", "validation", "rollback_condition"})
            changes = schema["properties"]["changes"]["items"]
            self.assertTrue(changes["allOf"])
    def test_reboot_guard_quarantines_malformed_state(self) -> None:
        bad_payloads = (
            {"commit": "abc", "rollback_commit": "def", "healthy_cycles": "1"},
            {"commit": "not-a-commit", "rollback_commit": "abcdef1", "healthy_cycles": 0},
        )
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            guard = RebootGuard(state)
            for payload in bad_payloads:
                guard.path.write_text(json.dumps(payload), encoding="utf-8")
                result = guard.observe({"ok": True})
                self.assertTrue(result["quarantined"])
                self.assertFalse(guard.path.exists())
                self.assertGreaterEqual(len(list((state / "quarantine").glob("reboot-guard-*.json"))), 1)
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            guard = RebootGuard(state)
            for value in ("[]", '"bad"', "null"):
                guard.path.write_text(value, encoding="utf-8")
                result = guard.observe({"ok": True})
                self.assertTrue(result["quarantined"])
                self.assertFalse(guard.path.exists())

    def test_self_improvement_tool_requires_files_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = SelfImprovementManager(directory)
            result = SelfImprovementTool(manager).execute({}, idempotency_key="x")
            self.assertFalse(result["ok"])
    def test_self_improvement_isolated_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            import subprocess
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "README.md").write_text("one\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "__init__.py").write_text("\n", encoding="utf-8")
            (root / "tests" / "test_smoke.py").write_text("import unittest\n\nclass Smoke(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md", "tests"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            proposal = manager.propose()
            manager.apply_files(proposal, {"notes.txt": "proposal\n"})
            # A valid module: this test is about worktree isolation, and the
            # harness gate now (correctly) refuses a tree with a broken import.
            manager.apply_files(proposal, {"skynet/module.py": "value = 1\n"})
            commit = manager.validate_and_commit(proposal, ("python", "-c", "pass"))
            manager.promote(proposal, commit, {"ok": True, "verified": True})
            request = json.loads((root / "state" / "reboot-request.json").read_text(encoding="utf-8"))
            self.assertEqual(request["commit"], commit)
            self.assertEqual(request["rollback_commit"], proposal.base_commit)
            self.assertEqual(request["health"]["commit"], commit)
            self.assertTrue((root / "notes.txt").exists())
            manager.discard(proposal)
    def test_self_improvement_applies_exact_patch_and_rejects_ambiguous_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            source = root / "module.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "module.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            proposal = manager.propose()
            manager.apply_changes(proposal, [{"path": "module.py", "operation": "replace", "old": "value = 1", "new": "value = 2"}])
            self.assertEqual((proposal.worktree / "module.py").read_text(encoding="utf-8"), "value = 2\n")
            (proposal.worktree / "module.py").write_text("value = 2\nvalue = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(SelfImprovementError, "match exactly once"):
                manager.apply_changes(proposal, [{"path": "module.py", "operation": "replace", "old": "value", "new": "x"}])
            manager.discard(proposal)
    def test_self_improvement_replace_keeps_line_boundaries(self) -> None:
        # The proposal_failed rows carried store.py with the import
        # line concatenated with itself ("utc_nowfrom .time import ..."). The
        # applier is an exact single-match string replace, so a replacement must
        # never splice the anchor into its neighbours: pin that boundary here.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            source = root / "module.py"
            source.write_text("from .time import utc_now\n\nvalue = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "module.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            proposal = manager.propose()
            manager.apply_changes(
                proposal,
                [{"path": "module.py", "operation": "replace", "old": "from .time import utc_now", "new": "from .time import utc_datetime_now, utc_now"}],
            )
            text = (proposal.worktree / "module.py").read_text(encoding="utf-8")
            self.assertEqual(text, "from .time import utc_datetime_now, utc_now\n\nvalue = 1\n")
            self.assertNotIn("utc_nowfrom", text)
            compile(text, "module.py", "exec")
            manager.discard(proposal)
    def test_self_improvement_distinguishes_missing_anchor_from_ambiguous_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            source = root / "module.py"
            source.write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "module.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            proposal = manager.propose()
            with self.assertRaisesRegex(SelfImprovementError, r"patch anchor not found \(0 matches\)"):
                manager.apply_changes(proposal, [{"path": "module.py", "operation": "replace", "old": "absent line", "new": "x"}])
            (proposal.worktree / "module.py").write_text("value = 1\nvalue = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(SelfImprovementError, r"must match exactly once \(2 matches given\)"):
                manager.apply_changes(proposal, [{"path": "module.py", "operation": "replace", "old": "value = 1", "new": "x"}])
            manager.discard(proposal)
    def test_self_improvement_repairs_transport_escaped_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            source = root / "module.py"
            source.write_text('value = is_ok("x")\n', encoding="utf-8")
            subprocess.run(["git", "add", "module.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            proposal = manager.propose()
            escaped = 'value = is_ok(\\"x\\")\\n'
            applied = manager.apply_changes(proposal, [{"path": "module.py", "operation": "replace", "old": escaped, "new": "value = 2\n"}])
            self.assertEqual((proposal.worktree / "module.py").read_text(encoding="utf-8"), "value = 2\n")
            self.assertEqual(applied.get("anchor_repairs"), 1)
            with self.assertRaisesRegex(SelfImprovementError, r"patch anchor not found \(0 matches\)"):
                manager.apply_changes(proposal, [{"path": "module.py", "operation": "replace", "old": "absent line", "new": "x"}])
            manager.discard(proposal)
    def test_self_improvement_does_not_nest_worktree_collections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection = root / ".skynet-improvements"
            worktree = collection / "abc123"
            worktree.mkdir(parents=True)
            manager = SelfImprovementManager(worktree)
            self.assertEqual(manager.worktree_root, collection)
            collection_manager = SelfImprovementManager(collection)
            self.assertEqual(collection_manager.worktree_root, collection)

    def test_self_improvement_prunes_only_orphaned_worktrees(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "module.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            manager = SelfImprovementManager(root, root / "worktrees")
            proposal = manager.propose()
            orphan = manager.worktree_root / "orphan"
            orphan.mkdir(parents=True)
            (orphan / "junk.txt").write_text("stale", encoding="utf-8")
            removed = manager.prune_orphan_worktrees()
            self.assertEqual(removed, [str(orphan)])
            self.assertFalse(orphan.exists())
            self.assertTrue(proposal.worktree.exists())
            manager.discard(proposal)

    def test_self_improvement_rejects_identical_failed_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "README.md").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            changes = [{"path": "README.md", "operation": "replace", "old": "one", "new": "two"}]
            with self.assertRaises(SelfImprovementError):
                manager.propose_files({}, (sys.executable, "-c", "raise SystemExit(1)"), changes=changes, metadata={"hypothesis": {"problem": "test"}})
            with self.assertRaisesRegex(SelfImprovementError, "identical proposal"):
                manager.propose_files({}, (sys.executable, "-c", "pass"), changes=changes, metadata={"hypothesis": {"problem": "test"}})
    def test_self_improvement_tool_requires_patch_for_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("one\n", encoding="utf-8")
            manager = _manager(root)
            tool = SelfImprovementTool(manager)
            result = tool.execute({"files": {"README.md": "two\n"}}, idempotency_key="legacy-existing")
            self.assertFalse(result["ok"])
            self.assertEqual(result["failure_class"], "invalid_payload")
            self.assertTrue(result["do_not_retry_unchanged"])
    def test_self_improvement_tool_splits_shell_string_and_passes_argv_through(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("one\n", encoding="utf-8")
            manager = _manager(root)
            tool = SelfImprovementTool(manager, test_command=(sys.executable, "-c", "pass"))
            with patch.object(manager, "propose_files", return_value={"ok": True, "proposal_id": "p"}) as propose:
                tool.execute({"changes": [{"path": "README.md", "operation": "replace", "old": "one", "new": "two"}], "test_command": "python -c 'pass'", "hypothesis": {"problem": "test", "expected_behavior": "two", "evidence": "fixture", "validation": "pass", "rollback_condition": "fail"}}, idempotency_key="command-normalize")
            self.assertEqual(propose.call_args.args[1], ("python", "-c", "pass"))
    def test_self_improvement_manager_normalizes_test_command(self) -> None:
        self.assertEqual(SelfImprovementManager._normalize_test_command(("python", "-c", "pass"))[0], sys.executable)
        self.assertEqual(SelfImprovementManager._normalize_test_command((".venv/bin/pytest", "-q"))[:3], (sys.executable, "-m", "pytest"))
        self.assertEqual(SelfImprovementManager._normalize_test_command(("python3", "-V"))[0], sys.executable)
    def test_self_improvement_uses_current_python_interpreter_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = SelfImprovementManager(Path(directory))
            tool = SelfImprovementTool(manager)
            self.assertEqual(tool.test_command[0], sys.executable)
    def test_reboot_guard_requires_health_window_and_explicit_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "reboot-request.json"
            request.write_text(json.dumps({"commit": "abc1234", "rollback_commit": "def1234", "proposal_id": "p1"}), encoding="utf-8")
            guard = RebootGuard(root, health_window_cycles=2)
            self.assertIsNotNone(guard.begin(request))
            self.assertEqual(guard.observe({"ok": True})["healthy_cycles"], 1)
            self.assertTrue(guard.observe({"ok": True})["completed"])

            request.write_text(json.dumps({"commit": "def4567", "rollback_commit": "abc4567"}), encoding="utf-8")
            guard.begin(request)
            with self.assertRaises(RecoveryError):
                guard.observe({"ok": False, "error": "broken"})
            rolled_back: list[str] = []
            request.write_text(json.dumps({"commit": "abc7890", "rollback_commit": "def7890"}), encoding="utf-8")
            guard.begin(request)
            result = guard.observe({"ok": False}, rolled_back.append)
            self.assertTrue(result["rolled_back"])
            self.assertEqual(rolled_back, ["def7890"])
            self.assertFalse(guard.observe({"ok": False})["active"])
    def test_reboot_guard_begin_quarantines_invalid_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "reboot-request.json"
            for field in ("commit", "rollback_commit"):
                payload = {"commit": "abc1234", "rollback_commit": "def1234"}
                payload[field] = "not-a-commit"
                request.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(RecoveryError):
                    RebootGuard(root).begin(request)
                self.assertFalse(request.exists())
                self.assertTrue(list(root.glob("reboot-request.json.corrupt-*")))
                for quarantined in root.glob("reboot-request.json.corrupt-*"):
                    quarantined.unlink()
                self.assertFalse((root / "reboot-guard.json").exists())
    def test_reboot_guard_persists_failure_before_rollback_callback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "reboot-request.json"
            request.write_text(json.dumps({"commit": "abc1234", "rollback_commit": "def1234"}), encoding="utf-8")
            guard = RebootGuard(root)
            guard.begin(request)
            with self.assertRaisesRegex(RuntimeError, "rollback failed"):
                guard.observe({"ok": False, "error": "broken"}, lambda _commit: (_ for _ in ()).throw(RuntimeError("rollback failed")))
            persisted = json.loads((root / "reboot-guard.json").read_text(encoding="utf-8"))
            self.assertTrue(persisted["failed"])
            self.assertFalse(persisted.get("rolled_back", False))
    def test_identical_reproposal_resumes_promotion_of_a_validated_change(self) -> None:
        # A change that passed the full gate but whose promotion failed (health
        # gate, or a restart before promotion) must not be trapped by the
        # duplicate guard: re-proposing it identically resumes promotion with the
        # stored commit instead of raising "already attempted".
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "README.md").write_text("one\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "__init__.py").write_text("\n", encoding="utf-8")
            (root / "tests" / "test_smoke.py").write_text("import unittest\n\nclass Smoke(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md", "tests"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            hypothesis = {"problem": "missing note", "expected_behavior": "note is present", "evidence": "test fixture", "validation": "run smoke test", "rollback_condition": "smoke test fails"}
            first = manager.propose_files({}, ("python", "-c", "pass"), changes=[{"path": "notes.txt", "operation": "create", "content": "proposal\n"}], metadata={"hypothesis": hypothesis})
            self.assertTrue(first["ok"], first)
            self.assertTrue(first["promotion_required"])
            proposal_id = str(first["proposal_id"])
            again = manager.propose_files({}, ("python", "-c", "pass"), changes=[{"path": "notes.txt", "operation": "create", "content": "proposal\n"}], metadata={"hypothesis": hypothesis})
            self.assertTrue(again["ok"], again)
            self.assertTrue(again.get("resumed_validation"), again)
            self.assertTrue(again["promotion_required"], again)
            self.assertEqual(str(again["proposal_id"]), proposal_id)
            record = manager._read_proposals()[proposal_id]
            self.assertEqual(record["status"], "validated")
            manager.discard(ImprovementProposal(proposal_id, Path(str(record["worktree"])), str(record["base_commit"])))

    def test_self_improvement_proposal_automatically_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "README.md").write_text("one\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "__init__.py").write_text("\n", encoding="utf-8")
            (root / "tests" / "test_smoke.py").write_text("import unittest\n\nclass Smoke(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md", "tests"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            tool = SelfImprovementTool(manager, lambda: {"ok": True}, ("python", "-c", "pass"))
            result = tool.execute({"files": {"notes.txt": "automatic\n"}, "hypothesis": {"problem": "missing note", "expected_behavior": "note is present", "evidence": "test fixture", "validation": "run smoke test", "rollback_condition": "smoke test fails"}}, idempotency_key="auto-1")
            result_map = cast(dict[str, object], result)
            self.assertTrue(result_map["ok"], result_map)
            self.assertFalse(result_map["promotion_required"])
            self.assertTrue(cast(dict[str, object], result_map["promotion"])["reboot_requested"])
            record = manager._read_proposals()[str(result_map["proposal_id"])]
            manager.discard(ImprovementProposal(str(record["proposal_id"]), Path(str(record["worktree"])), str(record["base_commit"])))
    def test_self_improvement_marks_acceptance_and_rollback_durably(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = _manager(root)
            manager._write_proposals({"p1": {"proposal_id": "p1", "status": "awaiting_reboot"}, "p2": {"proposal_id": "p2", "status": "awaiting_reboot"}})
            manager.mark_reboot_result("p1", {"completed": True, "proposal_id": "p1"})
            manager.mark_reboot_result("p2", {"rolled_back": True, "proposal_id": "p2"})
            proposals = json.loads((root / "state" / "self-improvement-proposals.json").read_text())
            self.assertEqual(proposals["p1"]["status"], "accepted")
            self.assertEqual(proposals["p2"]["status"], "rolled_back")
    def test_self_improvement_test_timeout_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "README.md").write_text("one\n", encoding="utf-8")
            base = commit_all(root, "base")
            (root / "README.md").write_text("changed\n", encoding="utf-8")
            manager = _manager(root)
            proposal = ImprovementProposal("p", root, base)
            real_run = subprocess.run

            def timeout_unless_git(argv, *args, **kwargs):
                # The worktree must stay readable so the protected-path guard
                # runs; only the gate command itself times out.
                if isinstance(argv, (list, tuple)) and argv and str(argv[0]).endswith("git"):
                    return real_run(argv, *args, **kwargs)
                raise subprocess.TimeoutExpired("gate", 600)

            with patch("skynet.self_improvement.subprocess.run", side_effect=timeout_unless_git), \
                 self.assertRaisesRegex(SelfImprovementError, "timed out"):
                manager.validate_and_commit(proposal, (sys.executable, "-c", "pass"))
    def test_self_improvement_reconciles_awaiting_reboot_without_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            subprocess.run(["git", "add", "a.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            state = root / "state"
            state.mkdir()
            (state / "self-improvement-proposals.json").write_text(json.dumps({
                "p1": {"status": "awaiting_reboot", "promoted_commit": commit, "worktree": ""},
            }), encoding="utf-8")
            manager = _manager(root)
            resolved = manager.reconcile_awaiting_reboot()
            self.assertEqual(resolved, [{"proposal_id": "p1", "status": "accepted"}])
            self.assertEqual(manager._read_proposals()["p1"]["status"], "accepted")

    def test_self_improvement_reconciles_after_a_completed_reboot_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            commit = commit_all(root, "base")
            state = root / "state"
            state.mkdir(exist_ok=True)
            (state / "self-improvement-proposals.json").write_text(json.dumps({
                "p1": {"status": "awaiting_reboot", "promoted_commit": commit, "worktree": ""},
            }), encoding="utf-8")
            (state / "reboot-guard.json").write_text(json.dumps({
                "commit": commit, "rollback_commit": commit, "proposal_id": "p1",
                "healthy_cycles": 3, "failed": False, "active": False,
                "completed_at": "2026-09-17T17:21:56.496362Z",
            }), encoding="utf-8")
            manager = _manager(root)
            resolved = manager.reconcile_awaiting_reboot()
            self.assertEqual(resolved, [{"proposal_id": "p1", "status": "accepted"}])
            self.assertEqual(manager._read_proposals()["p1"]["status"], "accepted")
    def test_self_improvement_reconciles_a_rolled_back_reboot_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            base = commit_all(root, "base")
            (root / "a.txt").write_text("b\n", encoding="utf-8")
            gone = commit_all(root, "rolled-back")
            subprocess.run(["git", "reset", "--hard", base], cwd=root, check=True)
            state = root / "state"
            state.mkdir(exist_ok=True)
            (state / "self-improvement-proposals.json").write_text(json.dumps({
                "p1": {"status": "awaiting_reboot", "promoted_commit": gone, "worktree": ""},
            }), encoding="utf-8")
            (state / "reboot-guard.json").write_text(json.dumps({
                "commit": gone, "rollback_commit": base, "proposal_id": "p1",
                "failed": True, "rolled_back": True, "active": False,
            }), encoding="utf-8")
            manager = _manager(root)
            resolved = manager.reconcile_awaiting_reboot()
            self.assertEqual(resolved, [{"proposal_id": "p1", "status": "rolled_back"}])
            self.assertEqual(manager._read_proposals()["p1"]["status"], "rolled_back")
    def test_self_improvement_reconciliation_waits_for_an_open_reboot_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            commit = commit_all(root, "base")
            state = root / "state"
            state.mkdir(exist_ok=True)
            (state / "self-improvement-proposals.json").write_text(json.dumps({
                "p1": {"status": "awaiting_reboot", "promoted_commit": commit, "worktree": ""},
            }), encoding="utf-8")
            manager = _manager(root)
            for guard in (
                {"commit": commit, "rollback_commit": commit, "healthy_cycles": 0, "failed": False},
                {"commit": commit, "rollback_commit": commit, "healthy_cycles": 1, "failed": False, "active": True},
                {"commit": commit, "rollback_commit": commit, "healthy_cycles": 1, "failed": False, "active": False},
                {"commit": commit, "rollback_commit": commit, "failed": True, "active": False},
                "{not json",
            ):
                (state / "reboot-guard.json").write_text(json.dumps(guard) if isinstance(guard, dict) else guard, encoding="utf-8")
                self.assertEqual(manager.reconcile_awaiting_reboot(), [])
                self.assertEqual(manager._read_proposals()["p1"]["status"], "awaiting_reboot")
            (state / "reboot-guard.json").unlink()
            self.assertEqual(manager.reconcile_awaiting_reboot(), [{"proposal_id": "p1", "status": "accepted"}])
    def test_self_improvement_reconciliation_keeps_undecidable_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            commit = commit_all(root, "base")
            state = root / "state"
            state.mkdir(exist_ok=True)
            manager = _manager(root)
            for record in (
                {"status": "awaiting_reboot", "worktree": ""},
                {"status": "awaiting_reboot", "promoted_commit": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "worktree": ""},
            ):
                (state / "self-improvement-proposals.json").write_text(json.dumps({"p1": record}), encoding="utf-8")
                self.assertEqual(manager.reconcile_awaiting_reboot(), [])
                self.assertEqual(manager._read_proposals()["p1"]["status"], "awaiting_reboot")
            self.assertEqual(commit, subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())

    def test_classify_failure_maps_failures_to_durable_classes(self) -> None:
        cases = {
            "invalid proposal path: ../etc/passwd escapes worktree": "path_violation",
            "hypothesis payload must be an object": "invalid_payload",
            "identical proposal was already attempted": "invalid_payload",
            "proposal test command must be an argv list, not a shell string": "invalid_payload",
            "proposal test command contains control characters": "invalid_payload",
            "proposal test path is unavailable: tests/test_self_improvement.py": "invalid_payload",
            "patch does not match exactly once": "patch_mismatch",
            "patch requires a non-empty old or anchor value": "invalid_payload",
            "patch anchor not found (0 matches): skynet/autonomous_planner.py": "anchor_not_found",
            "patch anchor must match exactly once (3 matches given): skynet/autonomous_planner.py": "anchor_ambiguous",
            "IndentationError: unexpected indent": "syntax_error",
            "ModuleNotFoundError: No module named 'x'": "import_error",
            "systemctl restart failed: permission denied": "administrative_test_failure",
            "proposal tests timed out after 600s": "timeout",
            "proposal validation executable is unavailable: /nope": "environment_failure",
            "promotion requires a passing health gate": "health_check_failure",
            "proposal tests failed (pytest -q): 3 failed": "regression_failure",
            "something entirely new": "unknown",
        }
        for error, expected in cases.items():
            self.assertEqual(classify_failure(error), expected, error)

    def test_empty_anchor_is_a_payload_error_not_a_stale_anchor(self) -> None:
        """A missing/empty old or anchor is malformed payload, not a stale anchor.

        Registry replay: 10 records carry failure_class='patch_mismatch'; 9 are
        genuine zero-match anchor failures and 1 (fa1aee89ca9e4a168620e892fc0f4b7c)
        is 'patch requires a non-empty old or anchor value'. Text matching alone
        routed that record into the stale-anchor bucket and inflated the
        anchor-failure signal it is meant to measure.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("alpha\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root / "worktrees")
            metadata = {"hypothesis": {"problem": "p", "expected_behavior": "e", "evidence": "v", "validation": "v", "rollback_condition": "r"}}
            for change in (
                {"path": "module.py", "operation": "replace", "old": "", "new": "x"},
                {"path": "module.py", "operation": "replace", "new": "x"},
            ):
                with self.assertRaisesRegex(SelfImprovementError, "non-empty old or anchor"):
                    manager.propose_files({}, changes=[dict(change)], test_command=("python", "-c", "pass"), metadata=metadata)
                record = list(manager._read_proposals().values())[-1]
                self.assertEqual(record["failure_class"], "invalid_payload", record)
                self.assertEqual(record["status"], "rejected", record)
                self.assertEqual(record.get("reason_code", ""), "")

    def test_anchor_failure_reason_codes_are_distinct_and_persisted(self) -> None:
        """Zero-match and multi-match anchors must not collapse into one code.

        A stale anchor and a count-dropped ambiguous anchor share the
        human-readable prefix "patch anchor must match exactly once", so the
        persisted rejection record must carry an explicit reason_code and
        match_count instead of a single generic patch_mismatch label. The
        transport-escape repair path must never be recorded as a rejection.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text('alpha\nMARKER\nquoted "literal"\nbeta\nMARKER\ngamma\n', encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root / "worktrees")
            metadata = {"hypothesis": {"problem": "p", "expected_behavior": "e", "evidence": "v", "validation": "v", "rollback_condition": "r"}}
            cases = {
                "zero_match": ("MARKER_MISSING", "anchor_not_found", r"0 matches", 0),
                "multi_match": ("MARKER", "anchor_ambiguous", r"2 matches given", 2),
            }
            codes: list[str] = []
            for (anchor, reason_code, phrase, match_count) in cases.values():
                with self.assertRaisesRegex(SelfImprovementError, phrase):
                    manager.propose_files({}, changes=[{"path": "module.py", "operation": "replace", "old": anchor, "new": "x"}], test_command=("python", "-c", "pass"), metadata=metadata)
                registry = json.loads((root / "state" / "self-improvement-proposals.json").read_text(encoding="utf-8"))
                persisted = [item for item in registry.values() if item.get("reason_code") == reason_code]
                self.assertEqual(len(persisted), 1, registry)
                self.assertEqual(persisted[0]["failure_class"], reason_code)
                self.assertEqual(persisted[0]["match_count"], match_count)
                codes.append(str(persisted[0]["reason_code"]))
            dropped = SelfImprovementError("patch anchor must match exactly once: skynet/autonomous_planner.py", reason_code="anchor_ambiguous", match_count=3)
            self.assertEqual(classify_failure(dropped), "anchor_ambiguous")
            self.assertEqual(classify_failure(str(dropped)), "patch_mismatch")
            codes.append(classify_failure(str(dropped)))
            self.assertEqual(len(set(codes)), 3, codes)
            proposal = manager.propose()
            applied = manager.apply_changes(proposal, [{"path": "module.py", "operation": "replace", "old": 'quoted \\"literal\\"', "new": 'quoted "literal" done'}])
            self.assertEqual(applied.get("anchor_repairs"), 1, applied)
            self.assertIn('quoted "literal" done', (proposal.worktree / "module.py").read_text(encoding="utf-8"))
            manager.discard(proposal)

    def test_gate_defaults_to_pytest_and_detects_a_failing_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "tests").mkdir()
            (root / "tests" / "test_sample.py").write_text("def test_ok():\n    assert True\n\ndef test_broken():\n    assert False\n", encoding="utf-8")
            commit_all(root, "failing-tests")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            self.assertIn("pytest", " ".join(SelfImprovementTool(manager).test_command))
            with self.assertRaisesRegex(SelfImprovementError, "tests failed"):
                manager.propose_files({}, changes=[{"path": "note.txt", "operation": "create", "content": "x"}])
            record = next(iter(manager._read_proposals().values()))
            self.assertEqual(record["status"], "rejected")
            self.assertEqual(record["failure_class"], "regression_failure")

    def test_self_improvement_environment_failure_is_blocked_not_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "README.md").write_text("one\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            with self.assertRaisesRegex(SelfImprovementError, "executable is unavailable"):
                manager.propose_files({}, ("/nonexistent/skynet-runner",), changes=[{"path": "README.md", "operation": "replace", "old": "one", "new": "two"}])
            record = next(iter(manager._read_proposals().values()))
            self.assertEqual(record["status"], "blocked_by_environment")
            self.assertEqual(record["failure_class"], "environment_failure")

    def test_promotion_and_reboot_require_a_passing_health_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = _manager(root)
            proposal = ImprovementProposal("p", root / "worktree", "base")
            with self.assertRaisesRegex(SelfImprovementError, "passing health gate"):
                manager.promote(proposal, "commit", {"ok": False})
            with self.assertRaisesRegex(SelfImprovementError, "passing health gate"):
                manager.request_reboot(proposal, "commit", {"ok": False})
            self.assertFalse((root / "state" / "reboot-request.json").exists())

    def test_promote_pending_restores_validated_status_when_health_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = _manager(root)
            manager._write_proposals({"p1": {"proposal_id": "p1", "status": "validated", "worktree": str(root / "wt"), "base_commit": "base", "commit": "c"}})
            with self.assertRaisesRegex(SelfImprovementError, "passing health gate"):
                manager.promote_pending("p1", {"ok": False})
            self.assertEqual(manager._read_proposals()["p1"]["status"], "validated")

    def test_proposal_without_changes_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "README.md").write_text("one\n", encoding="utf-8")
            base = commit_all(root, "base")
            manager = _manager(root)
            with self.assertRaisesRegex(SelfImprovementError, "no changes"):
                manager.validate_and_commit(ImprovementProposal("p", root, base), (sys.executable, "-c", "pass"))

    def test_open_window_withholds_only_its_own_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            commit = commit_all(root, "base")
            state = root / "state"
            state.mkdir(exist_ok=True)
            (state / "self-improvement-proposals.json").write_text(json.dumps({
                "p1": {"status": "awaiting_reboot", "promoted_commit": commit, "worktree": ""},
                "p2": {"status": "awaiting_reboot", "promoted_commit": commit, "worktree": ""},
            }), encoding="utf-8")
            (state / "reboot-guard.json").write_text(json.dumps({
                "commit": commit, "rollback_commit": commit, "proposal_id": "p1",
                "healthy_cycles": 1, "failed": False,
            }), encoding="utf-8")
            manager = _manager(root)
            self.assertEqual(manager.reconcile_awaiting_reboot(), [{"proposal_id": "p2", "status": "accepted"}])
            proposals = manager._read_proposals()
            self.assertEqual(proposals["p1"]["status"], "awaiting_reboot")
            self.assertEqual(proposals["p2"]["status"], "accepted")

    def test_gate_runs_inside_the_proposal_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "marker.txt").write_text("main\n", encoding="utf-8")
            commit = commit_all(root, "base")
            # An isolated worktree root: the default for a repository under /tmp
            # is the shared /tmp/.skynet-improvements, which is root-owned on the
            # server after a root-run organism and made this test fail there.
            manager = SelfImprovementManager(root, root / "worktrees")
            proposal = manager.propose()
            try:
                manager.apply_files(proposal, {"marker.txt": "worktree\n"})
                command = (sys.executable, "-c", "import sys; sys.exit(0 if open('marker.txt').read().strip() == 'worktree' else 1)")
                promoted = manager.validate_and_commit(proposal, command)
                self.assertNotEqual(promoted, commit)
                self.assertEqual((root / "marker.txt").read_text(encoding="utf-8"), "main\n")
            finally:
                manager.discard(proposal)

    def test_promoting_record_recreates_lost_reboot_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "tracked.txt").write_text("base\n", encoding="utf-8")
            commit = commit_all(root, "base")
            state = root / "state"
            state.mkdir()
            (state / "self-improvement-proposals.json").write_text(json.dumps({
                "p1": {"status": "promoting", "commit": commit, "base_commit": commit, "worktree": ""},
            }), encoding="utf-8")
            manager = _manager(root)
            resolved = manager.reconcile_awaiting_reboot()
            self.assertEqual(resolved[0]["status"], "awaiting_reboot")
            request = json.loads((state / "reboot-request.json").read_text(encoding="utf-8"))
            self.assertEqual(request["proposal_id"], "p1")
            self.assertEqual(manager._read_proposals()["p1"]["status"], "awaiting_reboot")

    def _proposal_metadata(self) -> dict[str, object]:
        return {"hypothesis": {"problem": "p", "expected_behavior": "e", "evidence": "v", "validation": "v", "rollback_condition": "r"}}

    def test_absolute_path_inside_repo_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            proposal = manager.propose()
            try:
                manager.apply_changes(proposal, [{"path": str(root / "module.py"), "operation": "replace", "old": "value = 1", "new": "value = 2"}])
                self.assertEqual((proposal.worktree / "module.py").read_text(encoding="utf-8"), "value = 2\n")
                applied = manager.apply_files(proposal, {str(root / "skynet" / "new_mod.py"): "x = 1\n"})
                self.assertEqual(applied, ["skynet/new_mod.py"])
                self.assertTrue((proposal.worktree / "skynet" / "new_mod.py").is_file())
            finally:
                manager.discard(proposal)

    def test_absolute_path_outside_repo_is_rejected_with_expected_form(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            proposal = manager.propose()
            try:
                with self.assertRaisesRegex(SelfImprovementError, "expected a path inside"):
                    manager.apply_files(proposal, {"/etc/passwd": "x"})
                with self.assertRaisesRegex(SelfImprovementError, "expected a path inside"):
                    manager.apply_changes(proposal, [{"path": "../evil.py", "operation": "create", "content": "x"}])
            finally:
                manager.discard(proposal)

    def test_backslash_path_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            proposal = manager.propose()
            try:
                applied = manager.apply_files(proposal, {"skynet\\nested\\new_mod.py": "x = 1\n"})
                self.assertEqual(applied, ["skynet/nested/new_mod.py"])
                self.assertTrue((proposal.worktree / "skynet" / "nested" / "new_mod.py").is_file())
            finally:
                manager.discard(proposal)

    def test_shell_string_test_command_runs_in_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            proposal = manager.propose()
            try:
                manager.apply_files(proposal, {"note.txt": "x\n"})
                command = 'python -c "import pathlib; pathlib.Path(\'gate.txt\').write_text(\'ran\')"'
                commit = manager.validate_and_commit(proposal, command)
                self.assertTrue(commit)
                self.assertEqual((proposal.worktree / "gate.txt").read_text(encoding="utf-8"), "ran")
            finally:
                manager.discard(proposal)

    def test_unusable_test_command_falls_back_to_default_and_records_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "tests").mkdir()
            (root / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            result = manager.propose_files({}, "", changes=[{"path": "note.txt", "operation": "create", "content": "x"}], metadata=self._proposal_metadata())
            self.assertTrue(result["test_command_fallback"])
            record = manager._read_proposals()[str(result["proposal_id"])]
            self.assertTrue(record["test_command_fallback"])
            self.assertEqual(cast(list, record["test_command"]), list(DEFAULT_TEST_COMMAND))
            manager.discard(ImprovementProposal(str(record["proposal_id"]), Path(str(record["worktree"])), str(record["base_commit"])))

    def test_control_characters_in_test_command_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = SelfImprovementManager(Path(directory))
            with self.assertRaisesRegex(SelfImprovementError, "control characters"):
                manager._coerce_test_command(("python", "-c", "bad\nvalue"))

    def test_mangled_argv_fragment_reports_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            proposal = manager.propose()
            try:
                manager.apply_files(proposal, {"note.txt": "x\n"})
                with self.assertRaises(SelfImprovementError) as caught:
                    manager.validate_and_commit(proposal, ('python3","-m","pytest","tests/x.py',))
                message = str(caught.exception)
                self.assertIn('python3","-m","pytest', message)
                self.assertIn("would have been used instead", message)
            finally:
                manager.discard(proposal)

    def test_missing_clean_test_path_reports_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            proposal = manager.propose()
            try:
                manager.apply_files(proposal, {"note.txt": "x\n"})
                with self.assertRaises(SelfImprovementError) as caught:
                    manager.validate_and_commit(proposal, (sys.executable, "-m", "pytest", "tests/missing_test.py"))
                message = str(caught.exception)
                self.assertIn("proposal test path is unavailable: tests/missing_test.py", message)
                self.assertIn("would have been used instead", message)
            finally:
                manager.discard(proposal)

    def test_venv_python_spellings_normalize_to_current_interpreter(self) -> None:
        for spelling in ("./.venv/bin/python3", ".venv/bin/python3", "./.venv/bin/python", "/opt/project/.venv/bin/python3"):
            self.assertEqual(SelfImprovementManager._normalize_test_command((spelling, "-V"))[0], sys.executable)

    def test_dirty_tracked_file_is_quarantined_and_proposal_proceeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / ".gitignore").write_text("state/\n", encoding="utf-8")
            (root / "module.py").write_text("original\n", encoding="utf-8")
            commit_all(root, "base")
            (root / "module.py").write_text("prototype\n", encoding="utf-8")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            proposal = manager.propose()
            try:
                self.assertEqual(manager.last_quarantined, ["module.py"])
                self.assertEqual((root / "module.py").read_text(encoding="utf-8"), "original\n")
                quarantined = list((root / "state" / "quarantine").glob("worktree-*/module.py"))
                self.assertEqual(len(quarantined), 1)
                self.assertEqual(quarantined[0].read_text(encoding="utf-8"), "prototype\n")
                events = [json.loads(line)["kind"] for line in (root / "state" / "runtime.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertIn("worktree_quarantined", events)
            finally:
                manager.discard(proposal)

    def test_dirty_worktree_refuse_keeps_old_behaviour(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("original\n", encoding="utf-8")
            commit_all(root, "base")
            (root / "module.py").write_text("prototype\n", encoding="utf-8")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            with self.assertRaisesRegex(SelfImprovementError, "no modified tracked files"):
                manager.propose(on_dirty="refuse")
            self.assertEqual((root / "module.py").read_text(encoding="utf-8"), "prototype\n")
            self.assertFalse((root / "state" / "quarantine").exists())

    def test_modified_state_file_is_refused_not_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "state").mkdir()
            (root / "state" / "keep.txt").write_text("a\n", encoding="utf-8")
            commit_all(root, "base")
            (root / "state" / "keep.txt").write_text("b\n", encoding="utf-8")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            with self.assertRaisesRegex(SelfImprovementError, "refusing to quarantine organism state"):
                manager.propose()
            self.assertEqual((root / "state" / "keep.txt").read_text(encoding="utf-8"), "b\n")

    def test_untracked_files_are_not_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("original\n", encoding="utf-8")
            commit_all(root, "base")
            (root / "loose.txt").write_text("scratch\n", encoding="utf-8")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            self.assertEqual(manager._quarantine_dirty_worktree(), [])
            self.assertTrue((root / "loose.txt").is_file())

    def test_repeat_failure_carries_previous_class_and_reset_clears_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = SelfImprovementManager(Path(directory))
            tool = SelfImprovementTool(manager)
            arguments = {"files": {"new.txt": "x"}, "hypothesis": self._proposal_metadata()["hypothesis"]}
            with patch.object(manager, "propose_files", side_effect=SelfImprovementError("boom")):
                first = tool.execute(arguments, idempotency_key="a")
                second = tool.execute(arguments, idempotency_key="b")
            self.assertNotIn("previous_failure", first)
            self.assertEqual(second["failure_key"], first["failure_key"])
            previous = cast(dict, second["previous_failure"])
            self.assertEqual(previous["failure_class"], "unknown")
            self.assertEqual(previous["error"], "boom")
            tool.reset_run_scope()
            with patch.object(manager, "propose_files", side_effect=SelfImprovementError("boom")):
                third = tool.execute(arguments, idempotency_key="c")
            self.assertNotIn("previous_failure", third)

    def test_identical_proposal_error_names_previous_proposal_and_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "README.md").write_text("one\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            changes = [{"path": "README.md", "operation": "replace", "old": "one", "new": "two"}]
            metadata = self._proposal_metadata()
            with self.assertRaises(SelfImprovementError):
                manager.propose_files({}, (sys.executable, "-c", "raise SystemExit(1)"), changes=changes, metadata=metadata)
            previous_id = next(iter(manager._read_proposals()))
            tool = SelfImprovementTool(manager, test_command=(sys.executable, "-c", "raise SystemExit(1)"))
            result = tool.execute({"changes": changes, "hypothesis": metadata["hypothesis"]}, idempotency_key="dup")
            self.assertFalse(result["ok"])
            self.assertIn(previous_id, str(result["error"]))
            self.assertIn("failure_class=regression_failure", str(result["error"]))
            self.assertEqual(cast(dict, result["previous_proposal"])["proposal_id"], previous_id)
            self.assertEqual(cast(dict, result["previous_proposal"])["failure_class"], "regression_failure")

    def test_gate_selfcheck_proves_which_tree_is_tested(self) -> None:
        # A runner that leaves cwd out of sys.path silently falls through to the
        # editable-install finder and validates the MAIN tree while returning
        # rc=0. The selfcheck stage makes that class of quiet false accepts
        # impossible: it asserts that `skynet` resolves inside the worktree.
        import sys as _sys

        from skynet.self_improvement import GATE_SELFCHECK_SNIPPET

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worktree = root / "wt"
            (worktree / "skynet").mkdir(parents=True)
            (worktree / "skynet" / "__init__.py").write_text("", encoding="utf-8")
            manager = _manager(root)
            # Positive: the worktree carries the package, so the stage passes.
            manager._run_gate_stage(
                "selfcheck",
                [_sys.executable, "-c", GATE_SELFCHECK_SNIPPET],
                worktree,
                env={**os.environ, "SKYNET_PROPOSAL_WORKTREE": str(worktree)},
            )
            # Negative: a worktree that does not contain the package is refused,
            # instead of silently testing whatever the finder resolves to.
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaises(SelfImprovementError) as caught:
                manager._run_gate_stage(
                    "selfcheck",
                    [_sys.executable, "-c", GATE_SELFCHECK_SNIPPET],
                    empty,
                    env={**os.environ, "SKYNET_PROPOSAL_WORKTREE": str(empty)},
                )
            self.assertIn("gate stage selfcheck failed", str(caught.exception))

    def test_gitignored_proposal_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / ".gitignore").write_text("*.pem\n", encoding="utf-8")
            (root / "README.md").write_text("one\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            with self.assertRaisesRegex(SelfImprovementError, "ignored proposal path"):
                manager.propose_files({}, (sys.executable, "-c", "pass"), changes=[{"path": "skynet/service_cert.pem", "operation": "create", "content": "secret"}], metadata=self._proposal_metadata())
            record = next(iter(manager._read_proposals().values()))
            self.assertEqual(record["failure_class"], "ignored_path")
            self.assertEqual(record["status"], "rejected")

    def test_a_weak_test_command_cannot_replace_the_harness_suite(self) -> None:
        # The registry showed proposals declaring `grep -q`, a `python3 -c` mock
        # or an import as their "test". The harness suite must still run: a
        # model-chosen command can only add a bounded stage.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "README.md").write_text("one\n", encoding="utf-8")
            (root / "skynet").mkdir()
            (root / "skynet" / "__init__.py").write_text("", encoding="utf-8")
            commit_all(root, "base")

            ran: list[list[str]] = []
            manager = SelfImprovementManager(root, root.parent / "worktrees", gate_suite=(sys.executable, "-c", "print('harness-suite-ran')"))
            original = manager._run_gate_stage

            def spy(stage, argv, worktree, **kwargs):
                ran.append([stage, *argv])
                return original(stage, argv, worktree, **kwargs)

            manager._run_gate_stage = spy  # type: ignore[method-assign]
            # A trivially weak "test" that cannot validate code is supplied as
            # the model's command; it passes, so only the harness suite stands
            # between this proposal and a promotion.
            manager.propose_files(
                {},
                ["grep", "-q", "one", "README.md"],
                changes=[{"path": "skynet/mod.py", "operation": "create", "content": "value = 1\n"}],
                metadata=self._proposal_metadata(),
            )
            stages = [entry[0] for entry in ran]
            self.assertIn("selfcheck", stages)
            self.assertIn("compileall", stages)
            self.assertIn("suite", stages, "the harness suite must run even when the model supplies its own command")

    def test_import_smoke_catches_a_broken_import(self) -> None:
        # pytest passes while an import no test touches is broken; the promoted
        # code then fails to start. The import-smoke stage closes that gap.
        import sys as _sys

        from skynet.self_improvement import GATE_IMPORT_SMOKE_SNIPPET

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "skynet"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "ok.py").write_text("value = 1\n", encoding="utf-8")
            manager = _manager(root)
            manager._run_gate_stage(
                "import-smoke", [_sys.executable, "-c", GATE_IMPORT_SMOKE_SNIPPET], root
            )
            (package / "broken.py").write_text("import definitely_not_a_module\n", encoding="utf-8")
            with self.assertRaises(SelfImprovementError) as caught:
                manager._run_gate_stage(
                    "import-smoke", [_sys.executable, "-c", GATE_IMPORT_SMOKE_SNIPPET], root
                )
            self.assertIn("gate stage import-smoke failed", str(caught.exception))
            self.assertIn("definitely_not_a_module", str(caught.exception))

    def test_diff_size_and_cosmetic_streak(self) -> None:
        # A self-improving system that only polishes rust looks productive while
        # changing nothing; the size of each accepted diff is the signal.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "a.txt").write_text("one\n", encoding="utf-8")
            base = commit_all(root, "base")
            manager = _manager(root)
            (root / "a.txt").write_text("two\n", encoding="utf-8")
            small = commit_all(root, "tiny change")
            size = manager._diff_size(small)
            self.assertEqual(size["files_changed"], 1)
            self.assertEqual(size["lines_changed"], 2)
            self.assertGreater(size["lines_added"], 0)
            # A missing commit is tolerated rather than raising.
            self.assertEqual(manager._diff_size("0" * 40), {})
            # The streak counts consecutive tiny diffs, newest first.
            manager._write_proposals({
                "p1": {"proposal_id": "p1", "status": "validated", "validated_at": "2026-01-01T00:00:00Z", "lines_changed": 1},
                "p2": {"proposal_id": "p2", "status": "validated", "validated_at": "2026-01-02T00:00:00Z", "lines_changed": 2},
                "p3": {"proposal_id": "p3", "status": "validated", "validated_at": "2026-01-03T00:00:00Z", "lines_changed": 1},
            })
            self.assertEqual(manager.cosmetic_streak()["streak"], 3)
            # A substantial change breaks the streak.
            manager._write_proposals({
                "p4": {"proposal_id": "p4", "status": "validated", "validated_at": "2026-01-04T00:00:00Z", "lines_changed": 40},
                **manager._read_proposals(),
            })
            self.assertEqual(manager.cosmetic_streak()["streak"], 0)
            self.assertTrue(base)

    def test_promoting_reconcile_cleans_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            base = commit_all(root, "base")
            (root / "a.txt").write_text("b\n", encoding="utf-8")
            dangling = commit_all(root, "dangling")
            subprocess.run(["git", "reset", "--hard", base], cwd=root, check=True)
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            proposal = manager.propose()
            manager._write_proposals({
                proposal.proposal_id: {"proposal_id": proposal.proposal_id, "status": "promoting", "commit": dangling, "base_commit": proposal.base_commit, "worktree": str(proposal.worktree)},
            })
            resolved = manager.reconcile_awaiting_reboot()
            self.assertEqual(resolved, [{"proposal_id": proposal.proposal_id, "status": "validated", "reason": "promotion interrupted before the merge"}])
            self.assertFalse(proposal.worktree.exists())

    def test_stale_validated_proposals_are_swept_and_fresh_ones_kept(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            stale = manager.propose()
            fresh = manager.propose()
            manager._write_proposals({
                stale.proposal_id: {"proposal_id": stale.proposal_id, "status": "validated", "worktree": str(stale.worktree), "validated_at": "2020-01-01T00:00:00Z"},
                fresh.proposal_id: {"proposal_id": fresh.proposal_id, "status": "validated", "worktree": str(fresh.worktree), "validated_at": utc_now()},
            })
            swept = manager.sweep_stale_proposals()
            self.assertEqual([item["proposal_id"] for item in swept], [stale.proposal_id])
            records = manager._read_proposals()
            self.assertEqual(records[stale.proposal_id]["status"], "rejected")
            self.assertEqual(records[stale.proposal_id]["failure_class"], "stale_proposal")
            self.assertFalse(stale.worktree.exists())
            self.assertEqual(records[fresh.proposal_id]["status"], "validated")
            self.assertTrue(fresh.worktree.exists())
            manager.discard(fresh)

    def test_promote_pending_rebases_a_stale_proposal_and_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / ".gitignore").write_text("state/\n", encoding="utf-8")
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            result = manager.propose_files({}, (sys.executable, "-c", "pass"), changes=[{"path": "added.txt", "operation": "create", "content": "new\n"}], metadata=self._proposal_metadata())
            proposal_id = str(result["proposal_id"])
            (root / "b.txt").write_text("b\n", encoding="utf-8")
            commit_all(root, "advance")
            promoted = manager.promote_pending(proposal_id, {"ok": True})
            self.assertTrue(promoted)
            self.assertEqual(manager._read_proposals()[proposal_id]["status"], "awaiting_reboot")
            self.assertTrue((root / "added.txt").is_file())
            manager._cleanup_worktree(str(manager._read_proposals()[proposal_id]["worktree"]))

    def test_promote_pending_marks_conflicting_stale_base_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / ".gitignore").write_text("state/\n", encoding="utf-8")
            (root / "a.txt").write_text("one\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            result = manager.propose_files({}, (sys.executable, "-c", "pass"), changes=[{"path": "a.txt", "operation": "replace", "old": "one", "new": "proposal"}], metadata=self._proposal_metadata())
            proposal_id = str(result["proposal_id"])
            (root / "a.txt").write_text("main-change\n", encoding="utf-8")
            commit_all(root, "conflict")
            with self.assertRaisesRegex(SelfImprovementError, "main worktree moved"):
                manager.promote_pending(proposal_id, {"ok": True})
            record = manager._read_proposals()[proposal_id]
            self.assertEqual(record["status"], "rejected")
            self.assertEqual(record["failure_class"], "stale_base_commit")
            self.assertFalse(Path(str(record["worktree"])).exists())

    def test_worktree_quota_blocks_new_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            first = manager.propose()
            second = manager.propose()
            manager._write_proposals({
                first.proposal_id: {"proposal_id": first.proposal_id, "status": "validated", "worktree": str(first.worktree)},
                second.proposal_id: {"proposal_id": second.proposal_id, "status": "validated", "worktree": str(second.worktree)},
            })
            with patch.dict(os.environ, {"SKYNET_MAX_WORKTREES": "2"}), self.assertRaises(SelfImprovementError) as caught:
                manager.propose()
            self.assertEqual(caught.exception.reason_code, "worktree_quota")
            manager.discard(first)
            manager.discard(second)

    def test_corrupt_registry_is_quarantined_and_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir()
            manager = _manager(root)
            manager._write_proposals({"old": {"proposal_id": "old", "change_fingerprint": "fingerprint-old", "status": "rejected"}})
            registry = state / "self-improvement-proposals.json"
            registry.write_text(registry.read_text(encoding="utf-8") + "{not json", encoding="utf-8")
            self.assertEqual(manager._read_proposals(), {})
            quarantined = list((state / "quarantine").glob("self-improvement-proposals-*.json"))
            self.assertEqual(len(quarantined), 1)
            self.assertIn("fingerprint-old", quarantined[0].read_text(encoding="utf-8"))
            manager._write_proposals({"fresh": {"proposal_id": "fresh", "status": "rejected"}})
            self.assertNotIn("fingerprint-old", registry.read_text(encoding="utf-8"))
            self.assertEqual(list(manager._read_proposals()), ["fresh"])

    def test_registry_write_fsyncs_the_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir()
            manager = _manager(root)
            opened: dict[int, str] = {}
            fsynced: list[int] = []
            real_open = os.open
            real_fsync = os.fsync

            def tracking_open(path, flags, *args, **kwargs):
                fd = real_open(path, flags, *args, **kwargs)
                opened[fd] = str(path)
                return fd

            def tracking_fsync(fd):
                fsynced.append(fd)
                return real_fsync(fd)

            with patch("skynet.self_improvement.os.open", side_effect=tracking_open), patch("skynet.self_improvement.os.fsync", side_effect=tracking_fsync):
                manager._write_proposals({"p": {"status": "rejected"}})
            self.assertTrue(any(opened.get(fd) == str(state) for fd in fsynced), (opened, fsynced))

    def test_classify_failure_learns_hygiene_classes(self) -> None:
        cases = {
            "ignored proposal path cannot be committed: skynet/service_cert.pem": "ignored_path",
            "stale_proposal expired after 24h without promotion": "stale_proposal",
            "main worktree moved since proposal was created": "stale_base_commit",
            "worktree quota reached (8 active of 8)": "worktree_quota",
            "proposal modifies gate-protected path(s): conftest.py; the test suite cannot be changed": "protected_path",
            "patch could not be applied with git apply: error: patch failed: module.py:1": "patch_apply_failed",
        }
        for error, expected in cases.items():
            self.assertEqual(classify_failure(error), expected, error)
        self.assertEqual(classify_failure(SelfImprovementError("x", reason_code="worktree_quota")), "worktree_quota")

    def test_proposal_cannot_edit_the_gate_inputs(self) -> None:
        # The gate runs the suite from the worktree the proposal edits, so a
        # proposal that rewrites tests/, conftest.py, the pytest config or the
        # meta-loop is warned on first submission: it is NOT applied to the main
        # worktree and the model must re-submit the identical change to proceed.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_sample.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            metadata = self._proposal_metadata()
            for path in ("conftest.py", "tests/test_new.py", "pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg", "skynet/self_improvement.py"):
                result = manager.propose_files(
                    {},
                    (sys.executable, "-c", "pass"),
                    changes=[{"path": path, "operation": "create", "content": "x\n"}],
                    metadata=metadata,
                )
                self.assertFalse(result["ok"], path)
                self.assertTrue(result["warned"], path)
                self.assertTrue(result["awaiting_resubmission"], path)
                self.assertEqual(result["status"], "warned_protected", path)
                protected = result["protected_paths"]
                if not isinstance(protected, list):
                    self.fail(f"protected_paths is not a list: {protected!r}")
                self.assertIn(path, protected, path)
            # Every warning keeps its worktree and carries no failure verdict.
            for record in manager._read_proposals().values():
                self.assertEqual(record["status"], "warned_protected")
                self.assertEqual(record["failure_class"], "")
            # A warning still denies: nothing was applied to the main worktree.
            self.assertFalse((root / "conftest.py").exists())
            self.assertFalse((root / "tests" / "test_new.py").exists())
            self.assertEqual((root / "module.py").read_text(encoding="utf-8"), "value = 1\n")

    def test_protected_warning_alert_carries_no_rollback_condition(self) -> None:
        # The owner reported "a pile of warnings pointing at an emergency
        # rollback of the work tree". The alert payload embedded the whole
        # hypothesis contract, whose mandatory ``rollback_condition`` field is
        # the only source of the word "rollback" in a warning that rolls nothing
        # back. Measured on the live store: 9 such alerts, 2745-4471
        # rendered characters, up to 9 "rollback" occurrences.
        with tempfile.TemporaryDirectory() as directory:
            # The store lives outside the repo: a sqlite file inside it would be
            # an untracked path and the proposal boundary refuses a dirty tree.
            root = Path(directory) / "repo"
            root.mkdir()
            manager, metadata = self._protected_repo(str(root))
            store = StateStore(Path(directory) / "state.sqlite3")
            manager.set_store(store)
            change = {"path": "tests/test_new.py", "operation": "create", "content": "def test_new():\n    assert True\n"}
            warned = manager.propose_files({}, (sys.executable, "-c", "pass"), changes=[change], metadata=metadata)
            self.assertTrue(warned["warned"], warned)
            row = store.connection.execute(
                "SELECT payload FROM alerts WHERE kind='gate_protected_warned'"
            ).fetchone()
            self.assertIsNotNone(row)
            payload = json.loads(row["payload"])
            rendered = render_alert(
                {"kind": "gate_protected_warned", "severity": "warning", "occurrences": 1, "payload": payload}
            )
            self.assertNotIn("rollback", rendered.lower(), rendered)
            self.assertNotIn("hypothesis", payload)
            # The negative control: the durable event keeps the contract, so the
            # information is not lost, only the misleading alert is.
            event = store.connection.execute(
                "SELECT payload FROM event_log WHERE kind='gate_protected_warned'"
            ).fetchone()
            self.assertIsNotNone(event)
            self.assertIn("rollback_condition", event["payload"])
            self.assertIn("tests/test_new.py", json.dumps(payload))
            store.close()

    def _protected_repo(self, directory: str) -> tuple[SelfImprovementManager, dict[str, object]]:
        root = Path(directory)
        git_repo(root)
        (root / "module.py").write_text("value = 1\n", encoding="utf-8")
        (root / "tests").mkdir()
        (root / "tests" / "test_sample.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        commit_all(root, "base")
        return SelfImprovementManager(root, root.parent / "worktrees"), self._proposal_metadata()

    def test_warned_proposal_is_validated_on_identical_resubmission(self) -> None:
        # The release half of the warning. The first submission only warns; the
        # identical fingerprint is then acknowledged, the full gate runs and the
        # proposal validates. Without this the warning would be a dead end again.
        with tempfile.TemporaryDirectory() as directory:
            manager, metadata = self._protected_repo(directory)
            change = {"path": "tests/test_new.py", "operation": "create", "content": "def test_new():\n    assert True\n"}
            warned = manager.propose_files({}, (sys.executable, "-c", "pass"), changes=[change], metadata=metadata)
            self.assertTrue(warned["warned"], warned)
            self.assertFalse(warned["ok"], warned)
            fingerprint = str(warned["change_fingerprint"])

            released = manager.propose_files({}, (sys.executable, "-c", "pass"), changes=[change], metadata=metadata)
            self.assertFalse(released.get("warned", False), released)
            self.assertTrue(released.get("ok"), released)
            self.assertTrue(released.get("promotion_required"), released)
            record = manager._read_proposals()[str(released["proposal_id"])]
            self.assertEqual(record["status"], "validated")
            self.assertEqual(record["change_fingerprint"], fingerprint)

    def test_different_fingerprint_gets_a_fresh_warning(self) -> None:
        # A warning is bound to the exact diff: a different change must never
        # ride on another proposal's acknowledgement.
        with tempfile.TemporaryDirectory() as directory:
            manager, metadata = self._protected_repo(directory)
            first = {"path": "tests/test_a.py", "operation": "create", "content": "def test_a():\n    assert True\n"}
            second = {"path": "tests/test_b.py", "operation": "create", "content": "def test_b():\n    assert True\n"}
            warned_a = manager.propose_files({}, (sys.executable, "-c", "pass"), changes=[first], metadata=metadata)
            self.assertTrue(warned_a["warned"], warned_a)
            warned_b = manager.propose_files({}, (sys.executable, "-c", "pass"), changes=[second], metadata=metadata)
            self.assertTrue(warned_b["warned"], warned_b)
            self.assertFalse(warned_b["ok"], warned_b)
            self.assertNotEqual(warned_a["change_fingerprint"], warned_b["change_fingerprint"])
            fingerprints = {str(record.get("change_fingerprint")) for record in manager._read_proposals().values()}
            self.assertIn(str(warned_a["change_fingerprint"]), fingerprints)
            self.assertIn(str(warned_b["change_fingerprint"]), fingerprints)

    def test_warned_worktrees_are_reclaimed_before_the_quota_wedges(self) -> None:
        # With the gate's own inputs warn-only, a run can warn on many distinct
        # protected diffs. Each warning holds a worktree; if those counted
        # against the quota forever, a handful of them would wedge
        # self-improvement for the life of the process. The quota must reclaim
        # old warnings instead of refusing new work, and a reclaimed warning
        # must be re-issuable rather than refused as already attempted.
        with tempfile.TemporaryDirectory() as directory:
            manager, metadata = self._protected_repo(directory)
            with patch.dict(os.environ, {"SKYNET_MAX_WORKTREES": "2"}):
                for index in range(5):
                    change = {"path": f"tests/test_{index}.py", "operation": "create", "content": f"def test_{index}():\n    assert True\n"}
                    result = manager.propose_files({}, (sys.executable, "-c", "pass"), changes=[change], metadata=metadata)
                    self.assertTrue(result.get("warned"), result)
                active = [record for record in manager._read_proposals().values() if record.get("worktree")]
                self.assertLessEqual(len(active), 2)
                repeat = manager.propose_files({}, (sys.executable, "-c", "pass"), changes=[{"path": "tests/test_0.py", "operation": "create", "content": "def test_0():\n    assert True\n"}], metadata=metadata)
                self.assertTrue(repeat.get("warned"), repeat)

    def test_an_unreadable_worktree_is_refused_not_assumed_safe(self) -> None:
        # A git failure used to yield an empty change set, which silently
        # disabled the protected-path guard. An unknown change set is now a
        # rejection instead of a free pass.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("one\n", encoding="utf-8")
            manager = SelfImprovementManager(root, root / "worktrees")
            proposal = ImprovementProposal("p", root, "base")
            with self.assertRaisesRegex(SelfImprovementError, "could not determine"):
                manager.validate_and_commit(proposal, (sys.executable, "-c", "pass"))

    def test_patch_proposal_is_applied_gated_and_promoted(self) -> None:
        # A proposal delivered as a unified diff is applied with `git apply`,
        # then runs the same gate and promotion path as an anchor proposal.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            (root / ".gitignore").write_text("state/\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            tool = SelfImprovementTool(manager, lambda: {"ok": True}, (sys.executable, "-c", "pass"))
            patch = (
                "diff --git a/module.py b/module.py\n"
                "--- a/module.py\n"
                "+++ b/module.py\n"
                "@@ -1 +1 @@\n"
                "-value = 1\n"
                "+value = 2\n"
            )
            result = tool.execute({"patch": patch, "hypothesis": self._proposal_metadata()["hypothesis"]}, idempotency_key="patch-1")
            self.assertTrue(result["ok"], result)
            self.assertEqual((root / "module.py").read_text(encoding="utf-8"), "value = 2\n")
            record = manager._read_proposals()[str(result["proposal_id"])]
            self.assertEqual(record["status"], "awaiting_reboot")
            self.assertIn("module.py", cast(dict, result["applied"])["files"])
            manager._cleanup_worktree(str(record["worktree"]))

    def test_reconcile_marks_rolled_back_only_when_head_is_behind_the_commit(self) -> None:
        """The discriminating predicate: not-an-ancestor is not a rollback.

        A rollback moves the tree *backwards* below the promoted commit, so the
        commit contains HEAD. A commit on an unrelated line of history contains
        HEAD as little as it is contained by it, and calling that a rollback
        would assert a fact git never established.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            commit_all(root, "base")
            (root / "a.txt").write_text("b\n", encoding="utf-8")
            promoted = commit_all(root, "promoted")
            subprocess.run(["git", "branch", "keep", promoted], cwd=root, check=True)
            subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=root, check=True)
            # an unrelated root, sharing no ancestry with HEAD either way
            subprocess.run(["git", "checkout", "-q", "--orphan", "orph"], cwd=root, check=True)
            (root / "b.txt").write_text("x\n", encoding="utf-8")
            subprocess.run(["git", "add", "b.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "orphan"], cwd=root, check=True)
            orphan = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            subprocess.run(["git", "checkout", "-q", "master"], cwd=root, check=True)
            state = root / "state"
            state.mkdir()
            (state / "self-improvement-proposals.json").write_text(json.dumps({
                "rolled": {"status": "awaiting_reboot", "promoted_commit": promoted, "worktree": ""},
                "unrelated": {"status": "awaiting_reboot", "promoted_commit": orphan, "worktree": ""},
            }), encoding="utf-8")
            manager = _manager(root)
            resolved = manager.reconcile_awaiting_reboot()
            by_id = {entry["proposal_id"]: entry["status"] for entry in resolved}
            self.assertEqual(by_id.get("rolled"), "rolled_back")
            # The negative control must NOT be written down as a rollback; it is
            # left awaiting review instead of being marked rolled_back.
            self.assertNotIn("unrelated", by_id)
            record = manager._read_proposals()["unrelated"]
            self.assertEqual(record["status"], "awaiting_reboot")
            self.assertNotIn("reboot_result", record)

    def test_reconcile_does_not_re_promote_an_undecidable_interrupted_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            commit_all(root, "base")
            subprocess.run(["git", "checkout", "-q", "--orphan", "orph"], cwd=root, check=True)
            (root / "b.txt").write_text("x\n", encoding="utf-8")
            subprocess.run(["git", "add", "b.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "orphan"], cwd=root, check=True)
            orphan = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            subprocess.run(["git", "checkout", "-q", "master"], cwd=root, check=True)
            state = root / "state"
            state.mkdir()
            (state / "self-improvement-proposals.json").write_text(json.dumps({
                "p1": {"status": "promoting", "promoted_commit": orphan, "base_commit": orphan, "worktree": ""},
            }), encoding="utf-8")
            manager = _manager(root)
            resolved = manager.reconcile_awaiting_reboot()
            self.assertEqual(resolved, [])
            self.assertEqual(manager._read_proposals()["p1"]["status"], "promoting")

    def test_damaged_patch_payload_is_retryable_not_a_verdict(self) -> None:
        """A payload the transport truncated must not burn the change fingerprint.

        Measured on the runtime log: every patch payload ever submitted (11 of
        11, 2026-09-25) arrives structurally malformed, and proposal 701d3510 --
        the only submission carrying a byte-correct, locally validated file --
        was recorded as a verdict about its change. A payload that cannot be a
        well-formed diff is an environment condition, so the same intended change
        stays submittable.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            truncated = (
                "diff --git a/module.py b/module.py\n"
                "--- a/module.py\n"
                "+++ b/module.py\n"
                "@@ -1,3 +1,3 @@\n"
                " value = 1\n"
                "+value = 2"
            )
            with self.assertRaises(SelfImprovementError):
                manager.propose_files({}, (sys.executable, "-c", "pass"), patch=truncated, metadata=self._proposal_metadata())
            record = next(iter(manager._read_proposals().values()))
            self.assertEqual(record["failure_class"], "patch_payload_corrupt")
            self.assertEqual(record["status"], "blocked_by_environment")

    def test_structural_check_separates_damage_from_a_real_non_match(self) -> None:
        """The shape check must flag transport damage and pass a valid diff on."""
        good = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
        self.assertIsNone(patch_structure_error(good))
        self.assertIsNone(patch_structure_error(good.rstrip("\n") + "\n"))
        # Multi-file payloads and the "no newline at end of file" marker are
        # well-formed and must reach git apply unchanged.
        multi = good + "diff --git a/y.py b/y.py\nnew file mode 100644\n--- /dev/null\n+++ b/y.py\n@@ -0,0 +1,2 @@\n+one\n+two\n"
        self.assertIsNone(patch_structure_error(multi))
        marker = 'diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n\\ No newline at end of file\n+b\n'
        self.assertIsNone(patch_structure_error(marker))
        # Damage: unterminated, and a hunk body that ends early.
        self.assertIsNotNone(patch_structure_error(good.rstrip("\n")))
        self.assertIsNotNone(patch_structure_error(good + "@@ -9,4 +9,4 @@\n-one\n+two\n"))
        # An empty payload and prose without a file header are rejected too, so
        # the check cannot be satisfied by "not terminated" alone.
        self.assertEqual(patch_structure_error(""), "patch is empty")
        self.assertEqual(patch_structure_error("   \n\n"), "patch is empty")
        self.assertIn("no file header", str(patch_structure_error("just prose, no diff at all\n")))

    def test_failed_patch_is_rejected_with_patch_apply_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            bad_patch = (
                "diff --git a/module.py b/module.py\n"
                "--- a/module.py\n"
                "+++ b/module.py\n"
                "@@ -1 +1 @@\n"
                "-absent line\n"
                "+value = 2\n"
            )
            with self.assertRaises(SelfImprovementError):
                manager.propose_files({}, (sys.executable, "-c", "pass"), patch=bad_patch, metadata=self._proposal_metadata())
            record = next(iter(manager._read_proposals().values()))
            self.assertEqual(record["failure_class"], "patch_apply_failed")
            self.assertEqual(record["status"], "rejected")

