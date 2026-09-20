"""Bounded system probe: schema, degradation and log rotation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skynet import system_probe
from skynet.system_probe import SystemProbe, SystemProbeLog, default_services

REQUIRED_KEYS = {
    "loadavg",
    "cpu_percent",
    "cpu_per_core",
    "cpu_count",
    "memory",
    "disk",
    "network",
    "processes",
    "services",
    "home_listing",
    "tmp_listing",
    "agent_commands",
    "sudo_commands",
    "bash_history",
    "probe_error",
}


class _Result:
    def __init__(self, stdout: str = "") -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


class DefaultServicesTests(unittest.TestCase):
    def test_default_services_is_frozen(self) -> None:
        self.assertEqual(
            default_services(),
            ["skynet.service", "skynet-telegram.service", "ssh", "systemd-journald", "cron"],
        )

    def test_default_services_returns_a_fresh_list(self) -> None:
        first = default_services()
        first.append("mutated")
        self.assertNotIn("mutated", default_services())


class SystemProbeSampleTests(unittest.TestCase):
    def test_sample_returns_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probe = SystemProbe(Path(directory))
            with patch.object(system_probe.subprocess, "run", return_value=_Result("")):
                sample = probe.sample(agent_commands=["echo hello"])
            self.assertTrue(REQUIRED_KEYS.issubset(sample.keys()))
            self.assertEqual(sample["agent_commands"], ["echo hello"])

    def test_sample_survives_missing_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probe = SystemProbe(Path(directory))
            with patch.object(system_probe.subprocess, "run", side_effect=FileNotFoundError):
                sample = probe.sample(agent_commands=["x"])
            self.assertTrue(REQUIRED_KEYS.issubset(sample.keys()))
            self.assertEqual(sample["sudo_commands"], [])
            self.assertEqual(sample["home_listing"], [])
            self.assertEqual(sample["services"][0]["active"], None)
            self.assertEqual(sample["services"][0]["failed"], None)

    def test_sample_truncates_agent_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probe = SystemProbe(Path(directory))
            commands = [f"cmd-{index}" for index in range(60)]
            with patch.object(system_probe.subprocess, "run", return_value=_Result("")):
                sample = probe.sample(agent_commands=commands)
            self.assertEqual(len(sample["agent_commands"]), 50)
            self.assertEqual(sample["agent_commands"][0], "cmd-10")


class SystemProbeLogTests(unittest.TestCase):
    def test_log_keeps_newest_max_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            log = SystemProbeLog(path, max_records=5)
            for index in range(20):
                log.write({"i": index})
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 5)
            self.assertEqual(json.loads(lines[0])["i"], 15)
            self.assertEqual(json.loads(lines[-1])["i"], 19)

    def test_log_byte_cap_compacts_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            log = SystemProbeLog(path, max_records=1000, max_bytes=200)
            for index in range(50):
                log.write({"i": index, "pad": "x" * 20})
            self.assertLessEqual(path.stat().st_size, 200)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertGreater(len(lines), 0)
            self.assertLess(len(lines), 50)

    def test_log_on_unwritable_path_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory) / "blocker"
            blocker.write_text("not a directory", encoding="utf-8")
            log = SystemProbeLog(blocker / "probe.jsonl")
            log.write({"i": 1})
            self.assertFalse((blocker / "probe.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
