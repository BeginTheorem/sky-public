"""The organism's own rollback request."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from skynet.rollback import RollbackRequestTool, consume_request, request_path


def _repo(root: Path) -> tuple[str, str]:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=root, check=True)
    (root / "f.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "one"], cwd=root, check=True)
    first = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    (root / "f.txt").write_text("two\n", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "two"], cwd=root, check=True)
    second = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return first, second


class RollbackRequestTests(unittest.TestCase):
    def test_request_defaults_to_head_parent_and_is_durable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, _ = _repo(root)
            tool = RollbackRequestTool(root)
            result = tool.execute({"reason": "the promotion broke the provider chain"}, idempotency_key="k")
            self.assertTrue(result["ok"])
            self.assertEqual(result["rollback_commit"], first)
            self.assertTrue(request_path(root).exists())
            payload = json.loads(request_path(root).read_text(encoding="utf-8"))
            self.assertTrue(payload["dangerous"])
            # One outstanding request at a time.
            again = tool.execute({"reason": "a second reason that is long enough"}, idempotency_key="k2")
            self.assertFalse(again["ok"])
            self.assertIn("already pending", again["error"])

    def test_a_short_reason_and_a_non_ancestor_commit_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _repo(root)
            tool = RollbackRequestTool(root)
            self.assertFalse(tool.execute({"reason": "short"}, idempotency_key="k")["ok"])
            refused = tool.execute({"reason": "a good long reason", "commit": "0" * 40}, idempotency_key="k")
            self.assertFalse(refused["ok"])
            self.assertIn("not an ancestor", refused["error"])
            self.assertFalse(request_path(root).exists())

    def test_consume_applies_the_rollback_and_removes_the_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, _ = _repo(root)
            RollbackRequestTool(root).execute({"reason": "roll back to the known good commit"}, idempotency_key="k")
            result = consume_request(root)
            assert result is not None
            self.assertTrue(result["applied"])
            self.assertEqual(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), first)
            self.assertFalse(request_path(root).exists())
            # Idempotent: nothing left to consume.
            self.assertIsNone(consume_request(root))

    def test_consume_refuses_a_target_that_is_no_longer_an_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = _repo(root)
            RollbackRequestTool(root).execute({"reason": "roll back to the known good commit"}, idempotency_key="k")
            # History moves on: the recorded target is still an ancestor here, so
            # use an unrelated commit to prove the guard.
            subprocess.run(["git", "checkout", "-q", "--detach", first], cwd=root, check=True)
            (root / "side.txt").write_text("side\n", encoding="utf-8")
            subprocess.run(["git", "add", "side.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "side"], cwd=root, check=True)
            side = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            subprocess.run(["git", "checkout", "-q", "--detach", second], cwd=root, check=True)
            payload = json.loads(request_path(root).read_text(encoding="utf-8"))
            payload["rollback_commit"] = side
            request_path(root).write_text(json.dumps(payload), encoding="utf-8")
            result = consume_request(root)
            assert result is not None
            self.assertFalse(result["applied"])
            self.assertIn("no longer an ancestor", result["error"])
            self.assertEqual(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), second)
