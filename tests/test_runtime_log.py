"""Runtime log kind filtering and the verbose provider dump."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skynet.runtime_log import RuntimeLog, verbose_enabled, verbose_write


class RuntimeLogTests(unittest.TestCase):
    def test_kind_filter_keeps_only_the_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.jsonl"
            log = RuntimeLog(path, kinds="fallback_failure,cycle_error")
            log.write("tool_result", {"noisy": True})
            log.write("fallback_failure", {"provider": "openrouter"})
            log.write("cycle_error", {"error": "boom"})
            kinds = [json.loads(line)["kind"] for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(kinds, ["fallback_failure", "cycle_error"])

    def test_no_filter_writes_everything(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.jsonl"
            log = RuntimeLog(path)
            log.write("tool_result", {})
            log.write("anything", {})
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)


class VerboseWriteTests(unittest.TestCase):
    def test_env_switch_enables_the_dump(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "state.sqlite3"
            with patch.dict(os.environ, {"SKYNET_VERBOSE_PROVIDER": "1"}):
                self.assertTrue(verbose_enabled(base))
                verbose_write("provider_request", {"messages": [{"role": "user", "content": "hi"}]}, base_path=base)
            lines = (Path(directory) / "verbose.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["kind"], "provider_request")

    def test_state_file_enables_the_dump(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "state.sqlite3"
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(verbose_enabled(base))
                (Path(directory) / "verbose.json").write_text('{"enabled": true}', encoding="utf-8")
                self.assertTrue(verbose_enabled(base))

    def test_verbose_write_redacts_obvious_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "state.sqlite3"
            with patch.dict(os.environ, {"SKYNET_VERBOSE_PROVIDER": "true"}):
                verbose_write(
                    "provider_request",
                    {
                        "api_key": "sk-live-abcdef123456",
                        "token": "super-secret-token",
                        "header": "Bearer abc.def.ghi",
                    },
                    base_path=base,
                )
            text = (Path(directory) / "verbose.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("sk-live-abcdef123456", text)
            self.assertNotIn("super-secret-token", text)
            self.assertNotIn("abc.def.ghi", text)
            self.assertIn("[REDACTED]", text)

    def test_verbose_write_is_a_noop_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "state.sqlite3"
            with patch.dict(os.environ, {}, clear=True):
                verbose_write("provider_response", {"text": "secret"}, base_path=base)
            self.assertFalse((Path(directory) / "verbose.jsonl").exists())

    def test_verbose_write_compacts_past_the_byte_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "state.sqlite3"
            path = Path(directory) / "verbose.jsonl"
            with patch.dict(os.environ, {"SKYNET_VERBOSE_PROVIDER": "1"}):
                for index in range(20):
                    verbose_write("provider_response", {"text": f"marker-{index}-" + "x" * 40}, base_path=base, max_bytes=300)
            text = path.read_text(encoding="utf-8")
            self.assertLessEqual(path.stat().st_size, 300)
            self.assertIn("marker-19", text)
            self.assertNotIn("marker-0-", text)

    def test_verbose_write_never_raises_on_unwritable_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory) / "blocker"
            blocker.write_text("not a directory", encoding="utf-8")
            base = blocker / "state.sqlite3"
            with patch.dict(os.environ, {"SKYNET_VERBOSE_PROVIDER": "1"}):
                verbose_write("provider_request", {"messages": []}, base_path=base)

    def test_verbose_write_sets_restrictive_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "state.sqlite3"
            with patch.dict(os.environ, {"SKYNET_VERBOSE_PROVIDER": "1"}):
                verbose_write("provider_request", {"messages": []}, base_path=base)
            mode = (Path(directory) / "verbose.jsonl").stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)
