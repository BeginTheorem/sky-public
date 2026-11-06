"""The external host probe: the log is its state, and no secret lands on disk."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "external_probe.py"


def _probe() -> Any:
    """Load the script (it lives outside the package) without executing a probe."""
    spec = importlib.util.spec_from_file_location("external_probe", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(Any, module)


class ConsecutiveFailuresTests(unittest.TestCase):
    def setUp(self) -> None:
        self.probe = _probe()

    def test_an_empty_log_counts_nothing(self) -> None:
        self.assertEqual(self.probe.consecutive_failures([]), 0)

    def test_a_healthy_tail_counts_nothing(self) -> None:
        lines = [
            "2026-10-16T00:00:00 | service_status=inactive | host=h | message=alert_sent",
            "2026-10-16T00:15:00 | service_status=active | host=h | message=service_active",
        ]
        self.assertEqual(self.probe.consecutive_failures(lines), 0)

    def test_failed_checks_at_the_tail_are_counted(self) -> None:
        lines = [
            "2026-10-16T00:00:00 | service_status=active | host=h | message=service_active",
            "2026-10-16T00:15:00 | service_status=unknown | host=h | message=alert_pending",
            "2026-10-16T00:30:00 | service_status=inactive | host=h | message=alert_sent",
        ]
        self.assertEqual(self.probe.consecutive_failures(lines), 2)

    def test_lines_carrying_no_status_are_ignored(self) -> None:
        lines = ["garbage", "2026-10-16T00:15:00 | host=h | message=alert_pending"]
        self.assertEqual(self.probe.consecutive_failures(lines), 0)


class ThresholdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.probe = _probe()

    def test_the_default_is_patient(self) -> None:
        os.environ.pop("SKYNET_PROBE_ALERT_AFTER", None)
        self.assertEqual(self.probe.alert_after(), self.probe.ALERT_AFTER)
        self.assertGreaterEqual(self.probe.ALERT_AFTER, 2, "one failed check is not an outage")

    def test_an_override_wins(self) -> None:
        with patch.dict(os.environ, {"SKYNET_PROBE_ALERT_AFTER": "5"}):
            self.assertEqual(self.probe.alert_after(), 5)

    def test_garbage_falls_back_to_the_default(self) -> None:
        with patch.dict(os.environ, {"SKYNET_PROBE_ALERT_AFTER": "soon"}):
            self.assertEqual(self.probe.alert_after(), self.probe.ALERT_AFTER)

    def test_zero_cannot_mean_alert_on_every_tick(self) -> None:
        with patch.dict(os.environ, {"SKYNET_PROBE_ALERT_AFTER": "0"}):
            self.assertEqual(self.probe.alert_after(), 1)


class LogPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.probe = _probe()

    def test_an_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / "probe.log")
            with patch.dict(os.environ, {"SKYNET_EXTERNAL_PROBE_LOG": target}):
                self.assertEqual(self.probe.log_path(), Path(target))

    def test_the_default_sits_in_the_tree_that_owns_the_script(self) -> None:
        os.environ.pop("SKYNET_EXTERNAL_PROBE_LOG", None)
        self.assertEqual(self.probe.log_path(), ROOT / "state" / "external_probe.log")

    def test_an_appended_failure_is_what_the_counter_reads_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.log"
            self.probe.append_log(path, "inactive", "host", "alert_pending")
            self.probe.append_log(path, "active", "host", "service_active")
            self.assertEqual(self.probe.consecutive_failures(path.read_text(encoding="utf-8").splitlines()), 0)
            self.probe.append_log(path, "inactive", "host", "alert_sent")
            self.assertEqual(self.probe.consecutive_failures(path.read_text(encoding="utf-8").splitlines()), 1)


class NoSecretOnDiskTests(unittest.TestCase):
    def test_the_token_never_travels_through_a_file(self) -> None:
        source = PROBE.read_text(encoding="utf-8")
        for forbidden in ("/tmp/skynet-alert", "config_file", "message_file"):
            self.assertNotIn(forbidden, source, "the token must reach curl on stdin, never a shared path")

    def test_the_script_names_no_real_home_directory(self) -> None:
        self.assertNotIn("/home/", PROBE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
