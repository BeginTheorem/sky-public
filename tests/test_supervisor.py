"""Split from the former monolithic CoreTests suite."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from datetime import UTC
from pathlib import Path
from typing import cast
from unittest.mock import patch

from helpers import FakeProvider, FixtureTool

from skynet.checkpoints import CheckpointError
from skynet.reactor import Reactor, ReactorConfig
from skynet.supervisor import Supervisor


def _iso(seconds: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(seconds, tz=UTC).isoformat()

class CoreTests(unittest.TestCase):
    def test_supervisor_health_check_reports_schema_and_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            reactor = Reactor(FakeProvider(), {"fixture_tool": FixtureTool()}, ReactorConfig(state_path=path))
            supervisor = Supervisor(FakeProvider(), {"fixture_tool": FixtureTool()}, ReactorConfig(state_path=Path(directory) / "supervisor.sqlite3"), root=Path(directory))
            supervisor.start()
            health = supervisor.health_check()
            self.assertTrue(health["ok"])
            self.assertTrue(cast(dict, health["checks"])["schema"])
            supervisor.stop()
            reactor.close()
    def test_supervisor_owns_reboot_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.sqlite3"
            request = Path(directory) / "reboot-request.json"
            request.write_text(json.dumps({"commit": "abcdef1", "rollback_commit": "abcdef2"}), encoding="utf-8")
            supervisor = Supervisor(FakeProvider(), {}, ReactorConfig(state_path=state_path), root=directory)
            supervisor.start()
            self.assertEqual(supervisor.observe_reboot()["active"], True)
            supervisor.stop()
    def test_external_startup_rollback_restores_previous_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "version.txt").write_text("old\n", encoding="utf-8")
            subprocess.run(["git", "add", "version.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "old"], cwd=root, check=True)
            old_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            (root / "version.txt").write_text("broken\n", encoding="utf-8")
            subprocess.run(["git", "add", "version.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "broken"], cwd=root, check=True)
            state = root / "state"
            state.mkdir()
            (state / "reboot-guard.json").write_text(json.dumps({"rollback_commit": old_commit}), encoding="utf-8")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            calls = root / "systemctl.calls"
            (fake_bin / "systemctl").write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {calls}\n", encoding="utf-8")
            (fake_bin / "systemctl").chmod(0o755)
            env = {**os.environ, "SKYNET_ROOT": str(root), "SKYNET_SERVICE": "test.service", "PATH": f"{fake_bin}:{os.environ['PATH']}"}
            subprocess.run(["bash", str(Path(__file__).parents[1] / "scripts" / "skynet-startup-rollback.sh")], env=env, check=True)
            self.assertEqual((root / "version.txt").read_text(encoding="utf-8"), "old\n")
            self.assertFalse((state / "reboot-guard.json").exists())
            self.assertIn("restart test.service", calls.read_text(encoding="utf-8"))

    def test_shell_scripts_do_not_default_to_a_literal_home(self) -> None:
        # A default written as "\$HOME/..." is the literal string "$HOME/..."
        # instead of a path, and a script using it creates a bogus "$HOME"
        # directory in the operator's home.
        scripts = Path(__file__).parents[1] / "scripts"
        offenders = [
            script.name
            for script in sorted(scripts.glob("*.sh"))
            if ":-\\$HOME" in script.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])

    def test_deploy_and_rollback_use_the_shared_history(self) -> None:
        # Deploy pushes the local branch and never wipes the server repository:
        # the server's own self-improvement commits must survive a deploy.
        scripts = Path(__file__).parents[1] / "scripts"
        deploy = (scripts / "deploy.sh").read_text(encoding="utf-8")
        rollback = (scripts / "rollback.sh").read_text(encoding="utf-8")
        self.assertNotIn("server-repo.git", deploy)
        self.assertNotIn("server-repo.git", rollback)
        self.assertNotIn('rm -rf "$target/.git"', deploy)
        self.assertIn("receive.denyCurrentBranch updateInstead", deploy)
        self.assertIn("git push --quiet", deploy)
        self.assertIn("git merge-base --is-ancestor", deploy)
        self.assertIn("reset --hard", rollback)
        self.assertNotIn("fetch", rollback)

    def test_startup_rollback_request_survives_a_malformed_guard(self) -> None:
        # The request is the authoritative reboot intent: a malformed guard used
        # to suppress a valid request and leave the broken version running.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "version.txt").write_text("old\n", encoding="utf-8")
            subprocess.run(["git", "add", "version.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "old"], cwd=root, check=True)
            old_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            (root / "version.txt").write_text("broken\n", encoding="utf-8")
            subprocess.run(["git", "add", "version.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "broken"], cwd=root, check=True)
            state = root / "state"
            state.mkdir()
            (state / "reboot-request.json").write_text(json.dumps({"rollback_commit": old_commit}), encoding="utf-8")
            (state / "reboot-guard.json").write_text("{not json", encoding="utf-8")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            (fake_bin / "systemctl").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            (fake_bin / "systemctl").chmod(0o755)
            env = {**os.environ, "SKYNET_ROOT": str(root), "SKYNET_SERVICE": "test.service", "PATH": f"{fake_bin}:{os.environ['PATH']}"}
            subprocess.run(["bash", str(Path(__file__).parents[1] / "scripts" / "skynet-startup-rollback.sh")], env=env, check=True)
            self.assertEqual((root / "version.txt").read_text(encoding="utf-8"), "old\n")
            self.assertFalse((state / "reboot-request.json").exists())

    def test_startup_rollback_refuses_a_commit_that_is_not_an_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "version.txt").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "version.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "one"], cwd=root, check=True)
            first = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            subprocess.run(["git", "checkout", "-q", "--detach", first], cwd=root, check=True)
            (root / "side.txt").write_text("side\n", encoding="utf-8")
            subprocess.run(["git", "add", "side.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "side"], cwd=root, check=True)
            side = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            # Back to the original commit: `side` is now NOT an ancestor of HEAD.
            subprocess.run(["git", "checkout", "-q", "--detach", first], cwd=root, check=True)
            state = root / "state"
            state.mkdir()
            (state / "reboot-request.json").write_text(json.dumps({"rollback_commit": side}), encoding="utf-8")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            (fake_bin / "systemctl").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            (fake_bin / "systemctl").chmod(0o755)
            env = {**os.environ, "SKYNET_ROOT": str(root), "SKYNET_SERVICE": "test.service", "PATH": f"{fake_bin}:{os.environ['PATH']}"}
            completed = subprocess.run(
                ["bash", str(Path(__file__).parents[1] / "scripts" / "skynet-startup-rollback.sh")],
                env=env, capture_output=True, text=True, check=False,
            )
            self.assertIn("is not an ancestor", completed.stderr)
            # HEAD is untouched: a stale commit must not discard later work.
            self.assertEqual(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), first)

    def test_external_startup_rollback_skips_finished_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "version.txt").write_text("accepted\n", encoding="utf-8")
            subprocess.run(["git", "add", "version.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "accepted"], cwd=root, check=True)
            (root / "version.txt").write_text("unrelated-crash\n", encoding="utf-8")
            subprocess.run(["git", "add", "version.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "unrelated"], cwd=root, check=True)
            state = root / "state"
            state.mkdir()
            (state / "reboot-guard.json").write_text(json.dumps({
                "rollback_commit": "abcdef1", "active": False, "completed_at": _iso(0.0),
            }), encoding="utf-8")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            calls = root / "systemctl.calls"
            (fake_bin / "systemctl").write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {calls}\n", encoding="utf-8")
            (fake_bin / "systemctl").chmod(0o755)
            env = {**os.environ, "SKYNET_ROOT": str(root), "SKYNET_SYSTEMCTL": str(fake_bin / "systemctl")}
            subprocess.run(["bash", str(Path(__file__).parents[1] / "scripts" / "skynet-startup-rollback.sh")], env=env, check=True)
            self.assertEqual((root / "version.txt").read_text(encoding="utf-8"), "unrelated-crash\n")
            self.assertFalse(calls.exists())

    def test_startup_rollback_refusal_never_touches_the_working_tree(self) -> None:
        # A refused rollback must leave the working tree byte-for-byte alone
        # and must not restart the service; no file may be moved into
        # state/quarantine/, including .git/, .venv/ and tests/.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "version.txt").write_text("broken\n", encoding="utf-8")
            (root / ".venv").mkdir()
            (root / ".venv" / "marker").write_text("venv\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_x.py").write_text("x = 1\n", encoding="utf-8")
            state = root / "state"
            state.mkdir()
            (state / "reboot-request.json").write_text(
                json.dumps({"commit": "abcdef1", "rollback_commit": "not-a-real-commit"}),
                encoding="utf-8",
            )
            before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
            env = {**os.environ, "SKYNET_ROOT": str(root), "SKYNET_SERVICE": "test.service"}
            completed = subprocess.run(
                ["bash", str(Path(__file__).parents[1] / "scripts" / "skynet-startup-rollback.sh")],
                env=env, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
            self.assertEqual(before, after)
            self.assertEqual((root / ".venv" / "marker").read_text(encoding="utf-8"), "venv\n")
            self.assertEqual((root / "tests" / "test_x.py").read_text(encoding="utf-8"), "x = 1\n")
            self.assertFalse((state / "quarantine").exists())

    def test_supervisor_startup_does_not_touch_the_working_tree(self) -> None:
        # No startup subsystem may quarantine .git/, .venv/ and tests/.
        # Startup must leave them alone and must not create a backup manifest.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / ".venv").mkdir()
            (root / ".venv" / "marker").write_text("venv\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_x.py").write_text("x = 1\n", encoding="utf-8")
            (root / "skynet").mkdir()
            (root / "config").mkdir()
            (root / "deploy").mkdir()
            state = root / "state"
            state.mkdir()
            (root / ".deployment-refresh").write_text("\n", encoding="utf-8")
            (state / "reboot-request.json").write_text(
                json.dumps({"commit": "abcdef1", "rollback_commit": "abcdef2"}),
                encoding="utf-8",
            )
            supervisor = Supervisor(FakeProvider(), {}, ReactorConfig(state_path=state / "state.sqlite3"), root=root)
            supervisor.start()
            try:
                self.assertTrue((root / ".git").is_dir())
                self.assertTrue((root / ".venv" / "marker").is_file())
                self.assertTrue((root / "tests" / "test_x.py").is_file())
                self.assertFalse((state / "runtime-backup.json").exists())
                self.assertFalse((state / "runtime-backups").exists())
                self.assertFalse((state / "quarantine").exists())
            finally:
                supervisor.stop()

    def test_supervisor_restart_service_rejects_unit_escape_and_uses_argv(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch("skynet.supervisor.subprocess.Popen") as popen:
            supervisor = Supervisor(FakeProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"), root=directory)
            with self.assertRaises(ValueError):
                supervisor.restart_service("../evil")
            supervisor.restart_service("skynet.service")
            popen.assert_called_once_with(["systemctl", "restart", "skynet.service"], close_fds=True)
            supervisor.reactor.close()

    def test_provider_outage_does_not_fail_the_reboot_window(self) -> None:
        class DownProvider:
            def health_probe(self):
                return {"ok": False, "active_provider": "", "providers": {}}

            def complete(self, messages, *, max_tokens, tools=()):
                raise AssertionError("provider must not be called during the reboot window")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir()
            (state / "reboot-request.json").write_text(json.dumps({"commit": "abcdef1", "rollback_commit": "abcdef2"}), encoding="utf-8")
            supervisor = Supervisor(DownProvider(), {}, ReactorConfig(state_path=state / "state.sqlite3"), root=root)
            supervisor.start()
            try:
                self.assertFalse(supervisor.health_check()["ok"])
                self.assertTrue(supervisor.health_check(include_provider=False)["ok"])
                result = supervisor.observe_reboot()
                self.assertTrue(result["active"])
                self.assertFalse(result.get("rolled_back"))
                guard = json.loads((state / "reboot-guard.json").read_text(encoding="utf-8"))
                self.assertEqual(guard["healthy_cycles"], 1)
                self.assertFalse(guard["failed"])
            finally:
                supervisor.stop()

    def test_pending_reboot_request_triggers_restart_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir()
            supervisor = Supervisor(FakeProvider(), {}, ReactorConfig(state_path=state / "state.sqlite3"), root=root)
            supervisor.start()
            try:
                calls: list[int] = []
                supervisor.set_restart_callback(lambda: calls.append(1))
                (state / "reboot-request.json").write_text(
                    json.dumps({"commit": "abcdef1", "rollback_commit": "abcdef2", "proposal_id": "proposal-x"}),
                    encoding="utf-8",
                )
                result = supervisor.observe_reboot()
                self.assertTrue(result["restart_requested"])
                self.assertEqual(calls, [1])
                self.assertEqual(
                    supervisor.reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='restart_recovered'").fetchone()[0],
                    1,
                )
            finally:
                supervisor.stop()

    def test_failing_recovered_restart_escalates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir()
            supervisor = Supervisor(FakeProvider(), {}, ReactorConfig(state_path=state / "state.sqlite3", restart_failure_limit=2), root=root)
            supervisor.start()
            try:
                def failing() -> None:
                    raise RuntimeError("systemctl unavailable")

                supervisor.set_restart_callback(failing)
                request = {"commit": "abcdef1", "rollback_commit": "abcdef2", "proposal_id": "proposal-x"}
                (state / "reboot-request.json").write_text(json.dumps(request), encoding="utf-8")
                supervisor.observe_reboot()
                supervisor.observe_reboot()
                self.assertEqual(supervisor.reactor.store.count_restart_failures("proposal-x"), 2)
                self.assertEqual(
                    supervisor.reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='restart_escalated'").fetchone()[0],
                    1,
                )
            finally:
                supervisor.stop()

    def test_failed_health_window_after_rollback_triggers_restart(self) -> None:
        # A successful git rollback changes the tree on disk, but the
        # process keeps running the broken promoted code in memory. The same
        # deferred restart callback a promotion uses must fire.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "version.txt").write_text("good\n", encoding="utf-8")
            subprocess.run(["git", "add", "version.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "good"], cwd=root, check=True)
            good = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            (root / "version.txt").write_text("broken\n", encoding="utf-8")
            subprocess.run(["git", "add", "version.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "broken"], cwd=root, check=True)
            state = root / "state"
            state.mkdir()
            (state / "reboot-request.json").write_text(
                json.dumps({"commit": "abcdef1", "rollback_commit": good, "proposal_id": "proposal-x"}),
                encoding="utf-8",
            )
            supervisor = Supervisor(FakeProvider(), {}, ReactorConfig(state_path=state / "state.sqlite3"), root=root)
            supervisor.start()
            try:
                calls: list[int] = []
                supervisor.set_restart_callback(lambda: calls.append(1))
                with patch.object(supervisor, "health_check", return_value={"ok": False, "checks": {}, "details": {}}):
                    result = supervisor.observe_reboot()
                self.assertTrue(result.get("rolled_back"))
                self.assertEqual((root / "version.txt").read_text(encoding="utf-8"), "good\n")
                self.assertEqual(calls, [1])
            finally:
                supervisor.stop()

    def test_rollback_to_a_non_ancestor_is_refused_without_touching_the_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "version.txt").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "version.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "one"], cwd=root, check=True)
            first = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            subprocess.run(["git", "checkout", "-q", "--detach", first], cwd=root, check=True)
            (root / "side.txt").write_text("side\n", encoding="utf-8")
            subprocess.run(["git", "add", "side.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "side"], cwd=root, check=True)
            side = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            subprocess.run(["git", "checkout", "-q", "--detach", first], cwd=root, check=True)
            state = root / "state"
            state.mkdir()
            (state / "reboot-request.json").write_text(
                json.dumps({"commit": first, "rollback_commit": side, "proposal_id": "proposal-x"}),
                encoding="utf-8",
            )
            supervisor = Supervisor(FakeProvider(), {}, ReactorConfig(state_path=state / "state.sqlite3"), root=root)
            supervisor.start()
            try:
                with patch.object(supervisor, "health_check", return_value={"ok": False, "checks": {}, "details": {}}), \
                     self.assertRaises(CheckpointError):
                    supervisor.observe_reboot()
                self.assertEqual(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), first)
                self.assertEqual(
                    supervisor.reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='rollback_refused'").fetchone()[0],
                    1,
                )
            finally:
                supervisor.stop()
