"""Tests for the MCP stdio client and server startup."""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

from skynet.mcp import MCPClientList, MCPError, MCPStdioClient, discover_mcp_tools, start_mcp_servers


class CoreTests(unittest.TestCase):
    def test_mcp_stdio_client_discovers_and_calls_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "mcp_fixture.py"
            helper.write_text("""import json, sys\ndef read(): return json.loads(sys.stdin.buffer.readline())\ndef send(value): sys.stdout.buffer.write((json.dumps(value)+'\\n').encode()); sys.stdout.buffer.flush()\nwhile True:\n msg=read()\n if msg.get('method')=='initialize': send({'jsonrpc':'2.0','id':msg['id'],'result':{}})\n elif msg.get('method')=='tools/list': send({'jsonrpc':'2.0','id':msg['id'],'result':{'tools':[{'name':'fixture_echo','description':'echo','inputSchema':{'type':'object','properties':{'text':{'type':'string'}},'additionalProperties':False}}]}})\n elif msg.get('method')=='tools/call': send({'jsonrpc':'2.0','id':msg['id'],'result':{'content':[{'type':'text','text':msg['params']['arguments']['text']}]}})\n""", encoding="utf-8")
            client = MCPStdioClient(["python", str(helper)])
            try:
                client.start()
                tools = discover_mcp_tools(client)
                self.assertEqual(tools["fixture_echo"].execute({"text": "ok"}, idempotency_key="x")["ok"], True)
            finally:
                client.close()
    def test_mcp_rejects_non_object_response(self) -> None:
        client = MCPStdioClient(["fixture"], restart_attempts=0)
        client.process = MagicMock(stdin=MagicMock(), stdout=MagicMock())
        client._write = MagicMock()
        client._read = MagicMock(return_value=[])
        with self.assertRaisesRegex(MCPError, "invalid MCP response object"):
            client.request("tools/list", {})
    def test_mcp_remote_error_does_not_restart_process(self) -> None:
        client = MCPStdioClient(["fixture"], restart_attempts=1)
        client.process = MagicMock(stdin=MagicMock(), stdout=MagicMock())
        client._write = MagicMock()
        client._read = MagicMock(return_value={"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "unknown"}})
        with patch.object(client, "close") as close, patch.object(client, "start") as start, self.assertRaises(MCPError):
            client.request("tools/call", {})
        close.assert_not_called()
        start.assert_not_called()
    def test_mcp_start_closes_client_when_discovery_fails(self) -> None:
        import skynet.mcp as mcp_module

        class FakeClient:
            instances: ClassVar[list[FakeClient]] = []

            def __init__(self, command, *, timeout_seconds):
                self.closed = False
                self.__class__.instances.append(self)

            def start(self):
                return None

            def close(self):
                self.closed = True

        with patch.object(mcp_module, "MCPStdioClient", FakeClient), patch.object(
            mcp_module, "discover_mcp_tools", side_effect=MCPError("discovery failed")
        ), self.assertRaises(MCPError):
            start_mcp_servers({"fixture": ["fixture"]})

        self.assertEqual(len(FakeClient.instances), 1)
        self.assertTrue(FakeClient.instances[0].closed)
    def test_mcp_rejects_oversized_frame(self) -> None:
        client = MCPStdioClient(["python", "-c", ""], max_frame_bytes=4)
        client._stdout_buffer.extend(b"12345")
        selector = MagicMock()
        with self.assertRaises(MCPError):
            client._read_until(selector, 0, b"\n", time.monotonic() + 1, 4)
    def test_mcp_process_failure_can_be_restarted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "mcp_crash_fixture.py"
            helper.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
            client = MCPStdioClient(["python", str(helper)], timeout_seconds=1, restart_attempts=2)
            spawns: list[int] = []
            original_start_process = client._start_process

            def counting_start_process() -> None:
                spawns.append(1)
                original_start_process()

            client._start_process = counting_start_process
            try:
                with self.assertRaises(MCPError):
                    client.start()
                self.assertIsNone(client.process)
                # A failed handshake must not restart: one spawn, no recursion,
                # and the attempt stays within the configured budget.
                self.assertEqual(client.restart_count, 0)
                self.assertLessEqual(len(spawns), client.restart_attempts + 1)
            finally:
                client.close()

    def test_mcp_start_does_not_recurse_on_immediately_dying_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "mcp_crash_fixture.py"
            helper.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
            client = MCPStdioClient(["python", str(helper)], timeout_seconds=1, restart_attempts=3)
            spawns: list[int] = []
            original_start_process = client._start_process

            def counting_start_process() -> None:
                spawns.append(1)
                original_start_process()

            client._start_process = counting_start_process
            try:
                started = time.monotonic()
                with self.assertRaises(MCPError):
                    client.start()
                elapsed = time.monotonic() - started
                # Exactly one process spawn proves start() never re-entered
                # itself.
                self.assertEqual(len(spawns), 1)
                self.assertLessEqual(client.restart_count, client.restart_attempts)
                self.assertIsNone(client.process)
                self.assertLess(elapsed, 5)
            finally:
                client.close()

    def test_mcp_restarts_are_bounded_by_restart_count(self) -> None:
        client = MCPStdioClient(["fixture"], restart_attempts=2)
        client.process = MagicMock(stdin=MagicMock(), stdout=MagicMock())
        client.process.poll.return_value = 3
        client._write = MagicMock()
        client._read = MagicMock()
        with patch.object(client, "close") as close, patch.object(client, "start") as start, self.assertRaises(MCPError):
            client.request("tools/list", {})
        self.assertEqual(client.restart_count, 2)
        self.assertEqual(close.call_count, 2)
        self.assertEqual(start.call_count, 2)
    def test_mcp_content_length_read_keeps_pipelined_bytes(self) -> None:
        client = MCPStdioClient(["python", "-c", ""], timeout_seconds=1)
        body = b'{"jsonrpc":"2.0","id":1,"result":{}}'
        client._stdout_buffer.extend(body + b"EXTRA")
        chunk = client._read_exact(MagicMock(), 0, len(body), time.monotonic() + 1)
        self.assertEqual(chunk, body)
        self.assertEqual(bytes(client._stdout_buffer), b"EXTRA")

    def test_mcp_reads_extra_headers_before_the_body(self) -> None:
        import json as json_module
        import os as os_module

        client = MCPStdioClient(["python", "-c", ""], timeout_seconds=1)
        body = json_module.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}).encode()
        frame = b"Content-Length: " + str(len(body)).encode() + b"\nX-Trace: abc\n\n" + body
        read_fd, write_fd = os_module.pipe()
        try:
            os_module.write(write_fd, frame)
            os_module.close(write_fd)
            client.process = MagicMock()
            client.process.stdout = MagicMock()
            client.process.stdout.fileno.return_value = read_fd
            self.assertEqual(client._read(), {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
        finally:
            os_module.close(read_fd)

    def test_mcp_replays_idempotent_method_but_never_tools_call(self) -> None:
        client = MCPStdioClient(["fixture"], restart_attempts=2)
        client.process = MagicMock(stdin=MagicMock(), stdout=MagicMock())
        client._write = MagicMock()
        client._read = MagicMock(side_effect=[MCPError("broken pipe"), {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}])
        with patch.object(client, "close") as close, patch.object(client, "start") as start:
            self.assertEqual(client.list_tools(), [])
        close.assert_called_once()
        start.assert_called_once()

        client._read = MagicMock(side_effect=MCPError("broken pipe"))
        with patch.object(client, "close") as close, patch.object(client, "start") as start, self.assertRaises(MCPError):
            client.call_tool("fixture_echo", {}, idempotency_key="k")
        close.assert_not_called()
        start.assert_not_called()

    def test_mcp_framing_is_independent_of_chunk_boundaries(self) -> None:
        import json as json_module
        import os as os_module
        import random

        rng = random.Random(1337)
        body = json_module.dumps({"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}).encode()
        frame = b"Content-Length: " + str(len(body)).encode() + b"\nX-Trace: t\n\n" + body
        for split in sorted({rng.randrange(0, len(frame) + 1) for _ in range(8)}):
            client = MCPStdioClient(["python", "-c", ""], timeout_seconds=1)
            read_fd, write_fd = os_module.pipe()
            try:
                client._stdout_buffer.extend(frame[:split])
                os_module.write(write_fd, frame[split:])
                os_module.close(write_fd)
                client.process = MagicMock()
                client.process.stdout = MagicMock()
                client.process.stdout.fileno.return_value = read_fd
                self.assertEqual(client._read(), {"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}, f"split={split}")
            finally:
                os_module.close(read_fd)

    def test_mcp_client_does_not_leak_descriptors(self) -> None:
        from helpers import count_open_fds, wait_for_fd_count

        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "mcp_fixture.py"
            helper.write_text("import json, sys\ndef read(): return json.loads(sys.stdin.buffer.readline())\ndef send(value): sys.stdout.buffer.write((json.dumps(value)+'\\n').encode()); sys.stdout.buffer.flush()\nwhile True:\n msg=read()\n if msg.get('method')=='initialize': send({'jsonrpc':'2.0','id':msg['id'],'result':{}})\n", encoding="utf-8")
            baseline = count_open_fds()
            client = MCPStdioClient(["python", str(helper)])
            client.start()
            self.assertGreater(count_open_fds(), baseline)
            client.close()
            self.assertLessEqual(wait_for_fd_count(baseline), baseline)

    def test_mcp_startup_skips_unavailable_sensor_when_degrading(self) -> None:
        import skynet.mcp as mcp_module

        closed: list[str] = []

        class FakeClient:
            def __init__(self, command, *, timeout_seconds: float = 30.0) -> None:
                self.command = list(command)

            def start(self) -> None:
                if self.command[0] == "broken":
                    raise OSError("missing binary")

            def close(self) -> None:
                closed.append(self.command[0])

        class FakeTool:
            name = "sensor_probe"

        with patch.object(mcp_module, "MCPStdioClient", FakeClient), patch.object(
            mcp_module, "discover_mcp_tools", return_value={"probe": FakeTool()}
        ):
            tools, clients = start_mcp_servers({"broken": ["broken"], "healthy": ["healthy"]}, skip_failures=True)

        self.assertIn("probe", tools)
        self.assertEqual(len(clients), 1)
        self.assertIn("broken", closed)
        self.assertEqual(clients.skipped_servers, ["broken"])
        self.assertEqual(tools.skipped_servers, ["broken"])

    def test_mcp_close_kills_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "mcp_group_fixture.py"
            pid_file = Path(directory) / "child.pid"
            helper.write_text(
                "import json, subprocess, sys\n"
                "child = subprocess.Popen(['sleep', '60'])\n"
                "open(sys.argv[1], 'w').write(str(child.pid))\n"
                "def read():\n"
                "    line = sys.stdin.buffer.readline()\n"
                "    return json.loads(line) if line else None\n"
                "def send(value):\n"
                "    sys.stdout.buffer.write((json.dumps(value) + '\\n').encode())\n"
                "    sys.stdout.buffer.flush()\n"
                "while True:\n"
                "    message = read()\n"
                "    if message is None:\n"
                "        break\n"
                "    if message.get('method') == 'initialize':\n"
                "        send({'jsonrpc': '2.0', 'id': message['id'], 'result': {}})\n",
                encoding="utf-8",
            )
            client = MCPStdioClient(["python", str(helper), str(pid_file)], timeout_seconds=5)
            try:
                client.start()
                child_pid = int(pid_file.read_text(encoding="utf-8"))
                client.close()
                deadline = time.monotonic() + 5
                gone = False
                while time.monotonic() < deadline:
                    try:
                        os.kill(child_pid, 0)
                    except ProcessLookupError:
                        gone = True
                        break
                    time.sleep(0.05)
                self.assertTrue(gone, "grandchild survived close()")
            finally:
                client.close()

    def test_mcp_health_reports_dead_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / "mcp_alive_fixture.py"
            helper.write_text(
                "import json, sys\n"
                "def send(value):\n"
                "    sys.stdout.buffer.write((json.dumps(value) + '\\n').encode())\n"
                "    sys.stdout.buffer.flush()\n"
                "while True:\n"
                "    line = sys.stdin.buffer.readline()\n"
                "    if not line:\n"
                "        break\n"
                "    message = json.loads(line)\n"
                "    if message.get('method') == 'initialize':\n"
                "        send({'jsonrpc': '2.0', 'id': message['id'], 'result': {}})\n",
                encoding="utf-8",
            )
            client = MCPStdioClient(["python", str(helper)], timeout_seconds=5)
            try:
                client.start()
                client.server_name = "dying"
                client.tool_names = ["probe"]
                self.assertTrue(client.alive())
                process = client.process
                assert process is not None
                process.kill()
                process.wait(timeout=5)
                self.assertFalse(client.alive())
                health = client.health()
                self.assertEqual(health["dying"]["alive"], False)
                self.assertEqual(health["dying"]["tools"], 1)
                with self.assertRaisesRegex(MCPError, "dying"):
                    client.call_tool("probe", {}, idempotency_key="k")
            finally:
                client.close()

    def test_mcp_tools_call_uses_longer_timeout(self) -> None:
        client = MCPStdioClient(["python", "-c", ""], timeout_seconds=30.0, tool_timeout_seconds=180.0)
        client.process = MagicMock(stdin=MagicMock(), stdout=MagicMock())
        client.process.poll.return_value = None
        client._write = MagicMock()
        client._read = MagicMock(return_value={"jsonrpc": "2.0", "id": 1, "result": {}})
        client.request("tools/call", {})
        client._read.assert_called_with(180.0)
        client._next_id = 1
        client._read.reset_mock()
        client._read.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}
        client.request("tools/list", {})
        client._read.assert_called_with(30.0)

    def test_mcp_tool_timeout_defaults_from_environment(self) -> None:
        with patch.dict(os.environ, {"SKYNET_MCP_TOOL_TIMEOUT": "250"}):
            client = MCPStdioClient(["python", "-c", ""])
        self.assertEqual(client.tool_timeout_seconds, 250.0)

    def test_mcp_client_list_health_and_restart(self) -> None:
        client = MagicMock()
        client.server_name = "alpha"
        client.health.return_value = {"alpha": {"alive": True, "tools": 2, "restarted": 1}}
        clients = MCPClientList([client])
        self.assertEqual(clients.health(), {"alpha": {"alive": True, "tools": 2, "restarted": 1}})
        clients.restart("alpha")
        client.restart.assert_called_once_with("alpha")
        with self.assertRaises(MCPError):
            clients.restart("missing")
