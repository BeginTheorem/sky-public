"""Tests for the heartbeat crash-forensics system probe."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, cast
from unittest.mock import patch

from skynet import heartbeat
from skynet.reactor import Reactor
from skynet.store import StateStore


class FakeProbe:
    instances: ClassVar[list[FakeProbe]] = []

    def __init__(self, workspace, *, services=None, timeout=10.0) -> None:
        self.workspace = workspace
        self.services = services
        self.timeout = timeout
        self.agent_commands: list[str] | None = None
        FakeProbe.instances.append(self)

    def sample(self, *, agent_commands=None) -> dict[str, object]:
        self.agent_commands = agent_commands
        return {"cpu_percent": 1.5, "memory_percent": 2.5, "process_count": 3}


class RaisingProbe:
    def __init__(self, workspace, *, services=None, timeout=10.0) -> None:
        raise RuntimeError("probe exploded")


class FakeProbeLog:
    writes: ClassVar[list[tuple[Path, dict[str, object]]]] = []

    def __init__(self, path, *, max_records=250, max_bytes=10_000_000) -> None:
        self.path = Path(path)

    def write(self, payload) -> None:
        FakeProbeLog.writes.append((self.path, payload))
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\n")


class FakeReactor:
    def __init__(self, state_path: Path) -> None:
        self.config = SimpleNamespace(state_path=state_path)
        self.store = StateStore(state_path)
        self.ticks = 0

    def watchdog(self) -> None:
        pass

    def tick(self, cause: str):
        self.ticks += 1

    def close(self) -> None:
        self.store.close()


class HeartbeatProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        heartbeat._wake_count = 0
        FakeProbe.instances = []
        FakeProbeLog.writes = []

    def _wake(self, reactor: FakeReactor, times: int) -> None:
        for _ in range(times):
            heartbeat.wake(cast(Reactor, reactor), "timer")

    def _probe_path(self, root: Path) -> Path:
        return root / "system-probe.jsonl"

    def test_first_wake_probes_then_every_nth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = FakeReactor(root / "state.sqlite3")
            try:
                with patch.object(heartbeat, "SystemProbe", FakeProbe), patch.object(
                    heartbeat, "SystemProbeLog", FakeProbeLog
                ), patch.dict("os.environ", {"SKYNET_SYSTEM_PROBE_EVERY": "3"}):
                    self._wake(reactor, 1)
                    self.assertEqual(len(FakeProbe.instances), 1)
                    self._wake(reactor, 2)
                    self.assertEqual(len(FakeProbe.instances), 1)
                    self._wake(reactor, 1)
                    self.assertEqual(len(FakeProbe.instances), 2)
            finally:
                reactor.close()
            probe_path = self._probe_path(root)
            self.assertTrue(probe_path.exists())
            records = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(records), 2)

    def test_probe_records_durable_event_with_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = FakeReactor(root / "state.sqlite3")
            try:
                with patch.object(heartbeat, "SystemProbe", FakeProbe), patch.object(
                    heartbeat, "SystemProbeLog", FakeProbeLog
                ), patch.dict("os.environ", {"SKYNET_SYSTEM_PROBE_EVERY": "1"}):
                    self._wake(reactor, 1)
                rows = reactor.store.connection.execute(
                    "SELECT payload FROM event_log WHERE kind='system_probe'"
                ).fetchall()
                self.assertEqual(len(rows), 1)
                payload = json.loads(rows[0]["payload"])
                self.assertEqual(payload["path"], str(self._probe_path(root)))
                self.assertEqual(payload["cpu_percent"], 1.5)
                self.assertEqual(payload["memory_percent"], 2.5)
                self.assertEqual(payload["process_count"], 3)
            finally:
                reactor.close()

    def test_agent_commands_are_sourced_from_bash_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = FakeReactor(root / "state.sqlite3")
            try:
                reactor.store.append_event(
                    "tool_call", {"call_id": "c1", "tool_name": "bash", "arguments": {"command": "ls -la"}}
                )
                reactor.store.append_event(
                    "tool_call", {"call_id": "c2", "tool_name": "read", "arguments": {"path": "x"}}
                )
                with patch.object(heartbeat, "SystemProbe", FakeProbe), patch.object(
                    heartbeat, "SystemProbeLog", FakeProbeLog
                ), patch.dict("os.environ", {"SKYNET_SYSTEM_PROBE_EVERY": "1"}):
                    self._wake(reactor, 1)
                self.assertEqual(FakeProbe.instances[0].agent_commands, ["ls -la"])
            finally:
                reactor.close()

    def test_probe_failure_does_not_escape_wake(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = FakeReactor(root / "state.sqlite3")
            try:
                with patch.object(heartbeat, "SystemProbe", RaisingProbe), patch.object(
                    heartbeat, "SystemProbeLog", FakeProbeLog
                ), patch.dict("os.environ", {"SKYNET_SYSTEM_PROBE_EVERY": "1"}):
                    result = heartbeat.wake(cast(Reactor, reactor), "timer")
                self.assertIsNone(result)
                self.assertEqual(reactor.ticks, 1)
                rows = reactor.store.connection.execute(
                    "SELECT kind FROM event_log ORDER BY sequence"
                ).fetchall()
                kinds = [row["kind"] for row in rows]
                self.assertIn("heartbeat", kinds)
                self.assertIn("heartbeat_result", kinds)
            finally:
                reactor.close()

    def test_env_cadence_defaults_to_20(self) -> None:
        with patch.dict("os.environ", {"SKYNET_SYSTEM_PROBE_EVERY": "0"}):
            self.assertEqual(heartbeat._probe_every(), 1)


if __name__ == "__main__":
    unittest.main()
