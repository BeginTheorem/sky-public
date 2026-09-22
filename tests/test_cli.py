"""Split from the former monolithic CoreTests suite."""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

from skynet.cli import _clean_start_command as clean_start
from skynet.cli import _window_days, build_parser, main
from skynet.lock import ProcessLock
from skynet.store import StateStore


class CoreTests(unittest.TestCase):
    def test_cli_exposes_checkpoint_and_rollback(self) -> None:
        parser = build_parser()
        checkpoint = parser.parse_args(["--root", "/tmp/project", "checkpoint"])
        rollback = parser.parse_args(["--root", "/tmp/project", "rollback"])
        self.assertEqual(checkpoint.command, "checkpoint")
        self.assertEqual(rollback.command, "rollback")
        reconcile = parser.parse_args(["reconcile", "run:call"])
        self.assertEqual(reconcile.command, "reconcile")

    def _clean_state(self, directory):
        state = Path(directory) / "state" / "skynet.sqlite3"
        state.parent.mkdir(parents=True, exist_ok=True)
        return state

    def test_cli_clean_start_archives_state_and_restarts_services(self) -> None:
        import tarfile

        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            (state.parent / "runtime.jsonl").write_text("{}\n", encoding="utf-8")
            (state.parent / "money-boost.json").write_text('{"enabled": true}\n', encoding="utf-8")
            (state.parent / "skynet.lock").write_text("", encoding="utf-8")
            store = StateStore(state)
            store.close()
            actions: list[tuple[str, list[str]]] = []

            def fake(action, services):
                actions.append((action, list(services)))
                return subprocess.CompletedProcess(["systemctl", action], 0, "", "")

            with patch("skynet.cli._systemctl_action", side_effect=fake):
                code = clean_start(str(state), "skynet.service", "skynet-telegram.service")
            self.assertEqual(code, 0)
            self.assertEqual([action for action, _ in actions], ["stop", "enable", "start"])
            self.assertFalse((state.parent / "runtime.jsonl").exists())
            self.assertFalse((state.parent / "money-boost.json").exists())
            self.assertFalse((state.parent / "skynet.lock").exists())
            self.assertTrue(state.exists())
            archives = list((state.parent / "archive").glob("full-reset-before-*.tar.gz"))
            self.assertEqual(len(archives), 1)
            with tarfile.open(archives[0]) as archive:
                names = archive.getnames()
            self.assertTrue(any(name.endswith("runtime.jsonl") for name in names))
            reopened = StateStore(state)
            self.assertIsNotNone(reopened.state().lifecycle)
            reopened.close()

    def test_cli_clean_start_aborts_when_stop_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            with patch("skynet.cli._systemctl_action", return_value=subprocess.CompletedProcess(["systemctl"], 3, "", "stop failed")):
                self.assertEqual(clean_start(str(state), "skynet.service", "skynet-telegram.service"), 3)
            self.assertFalse((state.parent / "archive").exists())

    def test_cli_reset_dispatcher_cancels_work_and_keeps_durable_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            store = StateStore(state)
            goal_id = store.add_goal("keep me", priority=1.0)
            store.add_task("pending work", goal_id=goal_id)
            store.connection.execute("INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at) VALUES('m1','fact','durable memory',1.0,NULL,?)", ("2026-01-01T00:00:00Z",))
            store.connection.commit()
            store.close()
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "reset-dispatcher"]):
                self.assertEqual(main(), 0)
            reopened = StateStore(state)
            self.assertEqual(reopened.connection.execute("SELECT status FROM tasks").fetchone()[0], "cancelled")
            self.assertEqual(reopened.connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0], 1)
            self.assertEqual(reopened.connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 1)
            reopened.close()

    def test_cli_start_stop_propagate_systemctl_failure(self) -> None:
        with patch("skynet.cli._systemctl_action", return_value=subprocess.CompletedProcess(["systemctl"], 0, "", "")) as action:
            with patch.object(sys, "argv", ["skynet", "--service", "skynet.service", "start"]):
                self.assertEqual(main(), 0)
            action.assert_called_once_with("start", ["skynet.service", "skynet-telegram.service"])
        with patch("skynet.cli._systemctl_action", return_value=subprocess.CompletedProcess(["systemctl"], 5, "", "boom")), \
             patch.object(sys, "argv", ["skynet", "--service", "skynet.service", "stop"]):
            self.assertEqual(main(), 5)

    def _fake_supervisor(self, events, interruptions, on_interrupt=None, captured=None):
        class FakeStatus:
            value = "interrupted"

        class FakeStore:
            def append_event(self, event, *args, **kwargs):
                events.append(event)

            def state(self):
                class State:
                    active_run_id = None
                    next_wake_at = None
                    lifecycle = type("Lifecycle", (), {"value": "sleep"})()

                return State()

        class FakeReactor:
            def __init__(self):
                self.store = FakeStore()

            def set_self_improvement_restart(self, callback):
                del callback

            def set_self_improvement_health(self, callback):
                del callback

            def set_stop_event(self, stop_event):
                del stop_event

            def interrupt_stale_run(self, *, reason, force):
                interruptions.append(reason)
                if on_interrupt is not None:
                    on_interrupt()
                return FakeStatus()

        class FakeSupervisor:
            def __init__(self, *args, **kwargs):
                if captured is not None and len(args) >= 3:
                    captured["config"] = args[2]
                del kwargs
                self.reactor = FakeReactor()

            def start(self):
                if captured is not None:
                    captured["sigterm_at_start"] = signal.getsignal(signal.SIGTERM)

            def stop(self):
                pass

            def set_restart_callback(self, callback):
                del callback

            def restart_service(self, service):
                del service

            def observe_reboot(self):
                pass

            def health_check(self):
                return {"ok": True, "checks": {"provider_reachable": True}, "details": {}}

        return FakeSupervisor

    def test_cli_loop_survives_a_failing_cycle(self) -> None:
        events: list[str] = []
        stop = threading.Event()
        supervisor = self._fake_supervisor(events, [])

        def release_stop():
            # Stop only after the loop journaled the failed cycle, so the
            # assertion observes the failure path instead of racing a fixed
            # sleep against it.
            deadline = time.monotonic() + 5.0
            while "cycle_error" not in events and time.monotonic() < deadline:
                time.sleep(0.01)
            stop.set()

        threading.Thread(target=release_stop, daemon=True).start()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", ["skynet", "--state", str(Path(directory) / "state.sqlite3"), "--interval", "0.05", "run"]
        ), patch("skynet.cli.build_provider", return_value=object()), patch(
            "skynet.cli.default_tools", return_value={}
        ), patch("skynet.cli.Supervisor", supervisor), patch(
            "skynet.cli.heartbeat_wake", side_effect=RuntimeError("boom")
        ), patch("skynet.cli.StaleRunWatchdog", MagicMock()), patch(
            "skynet.cli._stop_event", return_value=stop
        ):
            code = main()
        self.assertEqual(code, 0)
        self.assertEqual(events.count("cycle_error"), 1)

    def test_stop_event_seam_is_not_shared_with_unrelated_threads(self) -> None:
        # Regression guard: MCPStdioClient starts a stderr drain thread during
        # the same startup window as the loop. Replacing the loop's stop event
        # must not hand that event to Thread.__init__, or the worker thread's
        # own start would set the loop's stop flag.
        stop = threading.Event()
        with patch("skynet.cli._stop_event", return_value=stop):
            worker = threading.Thread(target=lambda: None, daemon=True)
            worker.start()
            worker.join(timeout=2)
        self.assertFalse(stop.is_set())

    def test_cli_sigterm_interrupts_the_running_cycle(self) -> None:
        events: list[str] = []
        interruptions: list[str] = []
        supervisor = self._fake_supervisor(events, interruptions)

        def wake_then_hang(*args, **kwargs):
            del args, kwargs
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
            return

        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.object(
                sys, "argv", ["skynet", "--state", str(Path(directory) / "state.sqlite3"), "--interval", "0.05", "run"]
            ), patch("skynet.cli.build_provider", return_value=object()), patch(
                "skynet.cli.default_tools", return_value={}
            ), patch("skynet.cli.Supervisor", supervisor), patch(
                "skynet.cli.heartbeat_wake", side_effect=wake_then_hang
            ), patch("skynet.cli.StaleRunWatchdog", MagicMock()):
                code = main()
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
        self.assertEqual(code, 0)
        # A SIGTERM is a clean owner stop, not a stale-run watchdog timeout.
        self.assertIn("owner_stop", interruptions)
        self.assertLess(time.monotonic() - started, 3.0)

    def test_mcp_servers_are_discovered_from_the_environment(self) -> None:
        # Adding a sensor used to require editing cli.py; any
        # SKYNET_MCP_<NAME>_COMMAND now registers one, and _DISABLED removes one.
        from skynet.cli import _mcp_config

        with patch.dict(os.environ, {
            "SKYNET_MCP_DDG_COMMAND": "ddg-server",
            "SKYNET_MCP_GITHUB_COMMAND": "gh --read-only",
            "SKYNET_MCP_FOO_COMMAND": "foo --bar",
            "SKYNET_MCP_BAR_COMMAND": "bar",
            "SKYNET_MCP_BAR_DISABLED": "1",
            "SKYNET_MCP_TIMEOUT": "30",
            "SKYNET_MCP_TOOL_TIMEOUT": "180",
        }, clear=False):
            for name in ("DDG", "GITHUB", "FOO", "BAR"):
                os.environ[f"SKYNET_MCP_{name}_COMMAND"] = os.environ[f"SKYNET_MCP_{name}_COMMAND"]
            config = _mcp_config()
        self.assertEqual(config["foo"], ["foo", "--bar"])
        self.assertEqual(config["github"], ["gh", "--read-only"])
        self.assertNotIn("bar", config, "a disabled server is skipped")
        # The timeout keys must not be mistaken for server names.
        self.assertNotIn("timeout", config)
        self.assertNotIn("tool", config)

    def test_log_alias_filters_by_kind_and_message_lists_inbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            store = StateStore(state)
            store.append_event("cycle_error", {"error": "boom"})
            store.append_event("tool_result", {"noisy": True})
            store.add_inbox_event("i1", "user_message", {"text": "hello"})
            store.close()
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "log", "10", "--kinds", "cycle_error"]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(), 0)
            rows = [json.loads(line) for line in output.getvalue().splitlines() if line.strip().startswith("{")]
            self.assertEqual([row["kind"] for row in rows], ["cycle_error"])
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "message"]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(), 0)
            self.assertIn("hello", output.getvalue())

    def test_jsonl_outbox_drain_is_opt_in(self) -> None:
        # Two consumers racing for one outbox lease meant an alert could land in
        # outbox.jsonl and never be marked delivered, so the owner never saw it.
        from skynet.cli import _jsonl_drain_enabled

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SKYNET_OUTBOX_JSONL", None)
            self.assertFalse(_jsonl_drain_enabled())
        for value in ("1", "true", "yes", "on", "TRUE"):
            with patch.dict(os.environ, {"SKYNET_OUTBOX_JSONL": value}):
                self.assertTrue(_jsonl_drain_enabled(), value)
        for value in ("0", "false", "no", "off"):
            with patch.dict(os.environ, {"SKYNET_OUTBOX_JSONL": value}):
                self.assertFalse(_jsonl_drain_enabled(), value)

    def test_cli_memory_search_returns_a_seeded_hit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            store = StateStore(state)
            with store.transaction():
                store.consolidate("run-1", [{"kind": "fact", "content": "SQLite durable search index", "confidence": 0.9}])
            store.close()
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "memory", "durable search"]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main()
            self.assertEqual(code, 0)
            results = json.loads(output.getvalue())
            self.assertTrue(any("durable search" in item["content"] for item in results), results)

    def test_cli_metrics_renders_a_report_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            store = StateStore(state)
            goal_id = store.add_goal("metrics goal", priority=1.0)
            store.add_task("metrics task", goal_id)
            store.raise_alert("provider_lockout", {"providers": ["ollama"]}, severity="critical")
            store.close()
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "metrics"]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main()
            self.assertEqual(code, 0)
            report = output.getvalue()
            self.assertIn("SkyNet metrics", report)
            self.assertIn("alerts_pending=1", report)
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "metrics"]), patch.dict(
                os.environ, {"SKYNET_METRICS_JSON": "1"}
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main()
            self.assertEqual(code, 0)
            data = json.loads(output.getvalue())
            self.assertIn("outcomes", data)
            self.assertIn("delivery", data)
            self.assertEqual(data["delivery"]["alerts_pending"], 1)

    def test_cli_alerts_lists_pending_and_recent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            store = StateStore(state)
            store.raise_alert("disk_low", {"free_gb": 3}, severity="warning")
            store.close()
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "alerts"]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main()
            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual([item["kind"] for item in payload["pending"]], ["disk_low"])
            self.assertEqual(len(payload["recent"]), 1)

    def test_metrics_window_argument_is_clamped(self) -> None:
        self.assertEqual(_window_days(None), 1.0)
        self.assertEqual(_window_days("7"), 7.0)
        self.assertEqual(_window_days("7d"), 7.0)
        self.assertEqual(_window_days("999"), 90.0)
        self.assertEqual(_window_days("0"), 0.1)
        self.assertEqual(_window_days("garbage"), 1.0)

    def test_cli_mcp_collision_is_renamed_and_recorded(self) -> None:
        captured: dict[str, object] = {}
        supervisor = self._fake_supervisor([], [], captured=captured)
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.sqlite3"
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "--once", "run"]), patch(
                "skynet.cli.build_provider", return_value=object()
            ), patch("skynet.cli.default_tools", return_value={"bash": MagicMock()}), patch(
                "skynet.cli.start_mcp_servers", return_value=({"bash": MagicMock()}, MagicMock(skipped_servers=["arxiv"]))
            ), patch.dict(os.environ, {"SKYNET_MCP_DDG_COMMAND": "ddg"}), patch(
                "skynet.cli.Supervisor", supervisor
            ), patch("skynet.cli.heartbeat_wake", return_value=None), patch("skynet.cli.StaleRunWatchdog", MagicMock()):
                code = main()
            self.assertEqual(code, 0)
            store = StateStore(state, read_only=True)
            try:
                kinds = [row[0] for row in store.connection.execute("SELECT kind FROM event_log")]
                self.assertIn("tool_name_collision", kinds)
                self.assertIn("mcp_server_unavailable", kinds)
                self.assertEqual(
                    store.connection.execute("SELECT COUNT(*) FROM alerts WHERE kind='mcp_server_unavailable'").fetchone()[0],
                    1,
                )
            finally:
                store.close()

    def test_cli_interval_reaches_reactor_config(self) -> None:
        captured: dict[str, object] = {}
        supervisor = self._fake_supervisor([], [], captured=captured)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", ["skynet", "--state", str(Path(directory) / "state.sqlite3"), "--interval", "300", "--once", "run"]
        ), patch("skynet.cli.build_provider", return_value=object()), patch(
            "skynet.cli.default_tools", return_value={}
        ), patch("skynet.cli.Supervisor", supervisor), patch(
            "skynet.cli.heartbeat_wake", return_value=None
        ), patch("skynet.cli.StaleRunWatchdog", MagicMock()):
            code = main()
        self.assertEqual(code, 0)
        config = captured["config"]
        self.assertEqual(cast(Any, config).wake_interval_seconds, 300.0)

    def test_cli_planner_output_tokens_reaches_reactor_config(self) -> None:
        captured: dict[str, object] = {}
        supervisor = self._fake_supervisor([], [], captured=captured)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", ["skynet", "--state", str(Path(directory) / "state.sqlite3"), "--once", "run"]
        ), patch("skynet.cli.build_provider", return_value=object()), patch(
            "skynet.cli.default_tools", return_value={}
        ), patch("skynet.cli.Supervisor", supervisor), patch(
            "skynet.cli.heartbeat_wake", return_value=None
        ), patch("skynet.cli.StaleRunWatchdog", MagicMock()), patch.dict(
            os.environ, {"SKYNET_PLANNER_OUTPUT_TOKENS": "12345"}
        ):
            code = main()
        self.assertEqual(code, 0)
        config = captured["config"]
        self.assertEqual(cast(Any, config).planner_output_tokens, 12_345)

    def test_cli_memory_loop_timeout_reaches_reactor_config(self) -> None:
        captured: dict[str, object] = {}
        supervisor = self._fake_supervisor([], [], captured=captured)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", ["skynet", "--state", str(Path(directory) / "state.sqlite3"), "--once", "run"]
        ), patch("skynet.cli.build_provider", return_value=object()), patch(
            "skynet.cli.default_tools", return_value={}
        ), patch("skynet.cli.Supervisor", supervisor), patch(
            "skynet.cli.heartbeat_wake", return_value=None
        ), patch("skynet.cli.StaleRunWatchdog", MagicMock()), patch.dict(
            os.environ, {"SKYNET_MEMORY_LOOP_TIMEOUT": "450"}
        ):
            code = main()
        self.assertEqual(code, 0)
        config = captured["config"]
        self.assertEqual(cast(Any, config).memory_loop_timeout_seconds, 450.0)

    def test_cli_installs_signal_handlers_before_supervisor_start(self) -> None:
        captured: dict[str, object] = {}
        supervisor = self._fake_supervisor([], [], captured=captured)
        try:
            with tempfile.TemporaryDirectory() as directory, patch.object(
                sys, "argv", ["skynet", "--state", str(Path(directory) / "state.sqlite3"), "--once", "run"]
            ), patch("skynet.cli.build_provider", return_value=object()), patch(
                "skynet.cli.default_tools", return_value={}
            ), patch("skynet.cli.Supervisor", supervisor), patch(
                "skynet.cli.heartbeat_wake", return_value=None
            ), patch("skynet.cli.StaleRunWatchdog", MagicMock()):
                code = main()
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
        self.assertEqual(code, 0)
        self.assertIsNot(captured["sigterm_at_start"], signal.SIG_DFL)
        self.assertIsNot(captured["sigterm_at_start"], signal.SIG_IGN)

    def test_cli_second_alarm_while_interrupting_does_not_kill_the_loop(self) -> None:
        events: list[str] = []
        interruptions: list[str] = []
        stop = threading.Event()
        nested = {"sent": False}

        def on_interrupt() -> None:
            if not nested["sent"]:
                nested["sent"] = True
                os.kill(os.getpid(), signal.SIGALRM)
                stop.set()

        supervisor = self._fake_supervisor(events, interruptions, on_interrupt=on_interrupt)

        def wake_then_alarm(*args, **kwargs):
            del args, kwargs
            os.kill(os.getpid(), signal.SIGALRM)
            time.sleep(5)
            return

        try:
            with self.assertLogs("skynet.cli", level="WARNING") as captured_logs, tempfile.TemporaryDirectory() as directory, patch.object(
                sys, "argv", ["skynet", "--state", str(Path(directory) / "state.sqlite3"), "--interval", "0.05", "run"]
            ), patch("skynet.cli.build_provider", return_value=object()), patch(
                "skynet.cli.default_tools", return_value={}
            ), patch("skynet.cli.Supervisor", supervisor), patch(
                "skynet.cli.heartbeat_wake", side_effect=wake_then_alarm
            ), patch("skynet.cli.StaleRunWatchdog", MagicMock()), patch(
                "skynet.cli._stop_event", return_value=stop
            ):
                code = main()
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
        self.assertEqual(code, 0)
        self.assertTrue(nested["sent"])
        self.assertIn("watchdog_timeout", interruptions)
        self.assertIn("watchdog alarm suppressed", "\n".join(captured_logs.output))

    def test_cli_forget_uses_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            store = StateStore(state)
            with store.transaction():
                store.consolidate("run-1", [
                    {"kind": "fact", "content": "keep this durable memory", "confidence": 0.9},
                    {"kind": "fact", "content": "delete this stale memory", "confidence": 0.9},
                ])
            store.close()
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "forget", "stale"]):
                self.assertEqual(main(), 0)
            reopened = StateStore(state)
            contents = [row[0] for row in reopened.connection.execute("SELECT content FROM memories ORDER BY content")]
            self.assertEqual(contents, ["keep this durable memory"])
            reopened.close()

    def test_cli_forget_without_target_errors_instead_of_wiping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            store = StateStore(state)
            with store.transaction():
                store.consolidate("run-1", [{"kind": "fact", "content": "must survive", "confidence": 0.9}])
            store.close()
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "forget"]):
                self.assertNotEqual(main(), 0)
            reopened = StateStore(state)
            self.assertEqual(reopened.connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 1)
            reopened.close()

    def test_cli_reset_reports_the_lock_holder_with_a_stop_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            store = StateStore(state)
            store.add_task("pending work")
            store.close()
            holder = ProcessLock(Path(state).with_suffix(".lock"))
            holder.acquire()
            try:
                with patch.object(sys, "argv", ["skynet", "--state", str(state), "reset-dispatcher"]):
                    errors = io.StringIO()
                    with contextlib.redirect_stderr(errors):
                        code = main()
            finally:
                holder.release()
            self.assertEqual(code, 1)
            self.assertIn("skynet.service", errors.getvalue())
            self.assertIn("systemctl stop", errors.getvalue())

    def test_cli_reset_force_stops_resets_and_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            store = StateStore(state)
            store.add_task("pending work")
            store.close()
            calls = Path(directory) / "systemctl.calls"
            fake = Path(directory) / "systemctl"
            fake.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {calls}\n", encoding="utf-8")
            fake.chmod(0o755)
            attempts = {"count": 0}
            real_acquire = ProcessLock.acquire

            def first_attempt_fails(self) -> None:
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("Another SkyNet process owns the lock")
                return real_acquire(self)

            with patch.dict(os.environ, {"SKYNET_SYSTEMCTL": str(fake)}), patch.object(
                ProcessLock, "acquire", first_attempt_fails
            ), patch.object(
                sys, "argv", ["skynet", "--state", str(state), "--service", "skynet.service", "--force", "reset-dispatcher"]
            ):
                self.assertEqual(main(), 0)
            self.assertEqual(calls.read_text(encoding="utf-8").splitlines(), ["stop skynet.service", "start skynet.service"])
            reopened = StateStore(state)
            self.assertEqual(reopened.connection.execute("SELECT status FROM tasks").fetchone()[0], "cancelled")
            reopened.close()

    def test_cli_start_and_stop_include_the_telegram_unit(self) -> None:
        with patch("skynet.cli._systemctl_action", return_value=subprocess.CompletedProcess(["systemctl"], 0, "", "")) as action:
            with patch.object(sys, "argv", ["skynet", "--service", "skynet.service", "--telegram-service", "skynet-telegram.service", "start"]):
                self.assertEqual(main(), 0)
            action.assert_called_once_with("start", ["skynet.service", "skynet-telegram.service"])
        with patch("skynet.cli._systemctl_action", return_value=subprocess.CompletedProcess(["systemctl"], 0, "", "")) as action:
            with patch.object(sys, "argv", ["skynet", "--service", "skynet.service", "--telegram-service", "skynet-telegram.service", "stop"]):
                self.assertEqual(main(), 0)
            action.assert_called_once_with("stop", ["skynet.service", "skynet-telegram.service"])

    def test_cli_money_boost_writes_under_the_state_path_and_stays_hot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            store = StateStore(state)
            store.close()
            with patch.dict(os.environ, {"SKYNET_MONEY_BOOST_STATE": ""}), patch(
                "skynet.cli._systemctl_action", return_value=subprocess.CompletedProcess(["systemctl"], 0, "", "")
            ) as action, patch.object(sys, "argv", ["skynet", "--state", str(state), "money-boost", "on"]), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(), 0)
            self.assertTrue((state.parent / "money-boost.json").exists())
            self.assertEqual(json.loads((state.parent / "money-boost.json").read_text(encoding="utf-8")), {"enabled": True})
            # The running process hot-reloads the chain, so no restart is issued.
            action.assert_not_called()
            self.assertIn("hot-reload", out.getvalue())

    def test_cli_timezone_flag_reaches_the_log_configuration(self) -> None:
        with patch("skynet.cli.logging.basicConfig") as basic, patch.object(sys, "argv", ["skynet", "--timezone", "Europe/Berlin", "time"]):
            self.assertEqual(main(), 0)
        self.assertIn("[Europe/Berlin]", basic.call_args.kwargs["format"])

    def test_cli_log_formatter_uses_the_configured_timezone(self) -> None:
        import logging as logging_module

        from skynet.cli import UTCLogFormatter

        formatter = UTCLogFormatter("%(message)s", "%Y-%m-%dT%H:%M:%S", "Asia/Tokyo")
        record = logging_module.LogRecord("t", logging_module.INFO, "p", 1, "m", (), None)
        record.created = 0  # 1970-01-01T00:00:00Z is 09:00 in Tokyo.
        self.assertEqual(formatter.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "1970-01-01T09:00:00")

    def _stub_reporting(self, renderer):
        # reporting.py is created in parallel; injecting a stub keeps this test
        # independent of its import while still exercising the render_recent call.
        import types

        import skynet

        module = types.ModuleType("skynet.reporting")
        cast(Any, module).render_recent = renderer
        return patch.dict(sys.modules, {"skynet.reporting": module}), patch.object(
            skynet, "reporting", module, create=True
        )

    def test_logs_mandatory_tier_renders_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            StateStore(state).close()
            renderer = MagicMock(return_value=["SCHEDULER run-1", "REPORT ok"])
            modules_patch, attr_patch = self._stub_reporting(renderer)
            with modules_patch, attr_patch, patch.object(
                sys, "argv", ["skynet", "--state", str(state), "logs", "--tier", "mandatory"]
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main()
            self.assertEqual(code, 0)
            self.assertIn("SCHEDULER run-1", output.getvalue())
            self.assertIn("REPORT ok", output.getvalue())
            renderer.assert_called_once()
            self.assertEqual(renderer.call_args.args[1], 5)

    def test_logs_mandatory_tier_clamps_limit_to_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            StateStore(state).close()
            renderer = MagicMock(return_value=[])
            modules_patch, attr_patch = self._stub_reporting(renderer)
            with modules_patch, attr_patch, patch.object(
                sys, "argv", ["skynet", "--state", str(state), "logs", "--tier", "mandatory", "--limit", "500"]
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            self.assertEqual(renderer.call_args.args[1], 100)

    def test_logs_system_tier_prints_probe_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            StateStore(state).close()
            probe = state.parent / "system-probe.jsonl"
            probe.write_text(
                json.dumps({"cpu": 1.5, "ts": "t1"}) + "\n" + json.dumps({"cpu": 2.5, "ts": "t2"}) + "\n",
                encoding="utf-8",
            )
            with patch.object(
                sys, "argv", ["skynet", "--state", str(state), "logs", "--tier", "system", "--limit", "1"]
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(), 0)
            text = output.getvalue()
            self.assertIn('"cpu": 2.5', text)
            self.assertNotIn('"cpu": 1.5', text)

    def test_logs_verbose_tier_handles_a_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            StateStore(state).close()
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "logs", "--tier", "verbose"]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(), 0)
            self.assertIn("verbose logging is off", output.getvalue())
            self.assertIn("verbose.jsonl", output.getvalue())

    def test_cli_verbose_toggles_state_file_and_reports_the_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self._clean_state(directory)
            StateStore(state).close()
            verbose_path = state.parent / "verbose.json"
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "verbose", "status"]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(), 0)
            status = json.loads(output.getvalue())
            self.assertFalse(status["verbose"])
            self.assertEqual(status["state_path"], str(verbose_path))
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "verbose", "on"]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(), 0)
            self.assertEqual(json.loads(verbose_path.read_text(encoding="utf-8")), {"enabled": True})
            self.assertIn(str(verbose_path), output.getvalue())
            self.assertIn("hot-reload", output.getvalue())
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "verbose", "status"]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(), 0)
            self.assertTrue(json.loads(output.getvalue())["verbose"])
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "verbose", "off"]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            self.assertEqual(json.loads(verbose_path.read_text(encoding="utf-8")), {"enabled": False})
            with patch.object(sys, "argv", ["skynet", "--state", str(state), "verbose", "bogus"]):
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    self.assertEqual(main(), 2)
            self.assertIn("on, off, or status", errors.getvalue())
