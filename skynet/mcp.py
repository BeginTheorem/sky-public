from __future__ import annotations

import contextlib
import json
import logging
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .provider import ToolSchema

log = logging.getLogger("skynet.mcp")


class MCPError(RuntimeError):
    pass


class MCPRemoteError(MCPError):
    """A JSON-RPC application error returned by an otherwise-live server."""



def _env_timeout(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1.0, float(raw))
    except ValueError:
        return default


class MCPStdioClient:
    """Bounded MCP JSON-RPC client using the standard stdio NDJSON framing."""

    def __init__(self, command: Sequence[str], *, timeout_seconds: float = 30.0, tool_timeout_seconds: float | None = None, max_frame_bytes: int = 16 * 1024 * 1024, restart_attempts: int = 1) -> None:
        self.command = list(command)
        self.timeout_seconds = timeout_seconds
        self.tool_timeout_seconds = tool_timeout_seconds if tool_timeout_seconds is not None else _env_timeout("SKYNET_MCP_TOOL_TIMEOUT", 180.0)
        self.max_frame_bytes = max_frame_bytes
        self.restart_attempts = restart_attempts
        self.server_name = ""
        self.tool_names: list[str] = []
        self.restart_count = 0
        self.process: subprocess.Popen[bytes] | None = None
        self._next_id = 1
        self._io_lock = threading.RLock()
        self._stdout_buffer = bytearray()
        self._stderr_buffer = bytearray()
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> None:
        if self.process is not None:
            return
        self._start_process()
        try:
            # The handshake must never restart: a restart calls start() again
            # from inside start(), so an immediately-dying server would recurse
            # without bound. A failed handshake simply closes and propagates.
            result = self.request(
                "initialize",
                {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "skynet", "version": "0.1"}},
                allow_restart=False,
            )
            if not isinstance(result, dict):
                raise MCPError("MCP initialize returned an invalid result")
            self.notify("notifications/initialized", {})
        except Exception:
            self.close()
            raise

    def _start_process(self) -> None:
        command = list(self.command)
        if command and command[0] in {"python", "python3"}:
            command[0] = sys.executable
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self._stdout_buffer.clear()
        self._stderr_buffer.clear()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, args=(self.process,), daemon=True)
        self._stderr_thread.start()

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        self.process = None
        self._terminate_process_group(process)
        thread = self._stderr_thread
        self._stderr_thread = None
        if thread is not None:
            thread.join(timeout=1)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
        """Kill the whole session group, not just the leader.

        The child is started with ``start_new_session=True`` so its
        grandchildren (Playwright browsers, npx node processes) share its
        process group and would otherwise survive.
        """
        try:
            pgid: int | None = os.getpgid(process.pid)
        except (ProcessLookupError, OSError, TypeError, ValueError):
            pgid = None

        def signal_group(sig: int) -> None:
            if pgid is not None:
                try:
                    if pgid != os.getpgid(0):
                        os.killpg(pgid, sig)
                        return
                except (ProcessLookupError, OSError):
                    pass
            with contextlib.suppress(ProcessLookupError, OSError):
                process.send_signal(sig)

        signal_group(signal.SIGTERM)
        try:
            process.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            pass
        signal_group(signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=2)

    def _drain_stderr(self, process: subprocess.Popen[bytes]) -> None:
        stream = process.stderr
        if stream is None:
            return
        while True:
            try:
                chunk = stream.read(4096)
            except (ValueError, OSError):
                return
            if not chunk:
                return
            self._stderr_buffer.extend(chunk)
            if len(self._stderr_buffer) > 16 * 1024:
                del self._stderr_buffer[:-16 * 1024]

    def _diagnostic(self) -> str:
        return bytes(self._stderr_buffer).decode("utf-8", errors="replace").strip()[-2000:]

    def _label(self) -> str:
        return self.server_name or "MCP server"

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def health(self) -> dict[str, dict[str, Any]]:
        name = self.server_name or "mcp"
        return {name: {"alive": self.alive(), "tools": len(self.tool_names), "restarted": self.restart_count}}

    def restart(self, server_name: str = "") -> None:
        if server_name and self.server_name and server_name != self.server_name:
            raise MCPError(f"server name mismatch: {server_name} != {self.server_name}")
        with self._io_lock:
            self._restart_locked()

    def _restart_locked(self) -> None:
        # Count the attempt before spawning so a restart that itself fails
        # still consumes the budget and cannot loop.
        self.restart_count += 1
        self.close()
        self.start()

    # Methods that are safe to resend after a transport restart. `tools/call`
    # is deliberately absent: a timeout may follow a side effect already
    # applied by the server, so it must never be replayed automatically.
    IDEMPOTENT_METHODS = frozenset({"initialize", "tools/list", "notifications/initialized"})

    def _restart_allowed(self, method: str, allow_restart: bool) -> bool:
        """Whether a transport failure may be recovered by restarting.

        ``tools/call`` is never replayed; ``allow_restart=False`` disables the
        restart for the initialize handshake; ``restart_count`` caps the total
        number of restarts so a crashing server cannot restart forever.
        """
        return allow_restart and method in self.IDEMPOTENT_METHODS and self.restart_count < self.restart_attempts

    def request(self, method: str, params: dict[str, Any], *, allow_restart: bool = True) -> Any:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise MCPError("MCP client is not started")
        request_id = self._next_id
        self._next_id += 1
        with self._io_lock:
            while True:
                exit_code = self.process.poll() if self.process is not None else None
                if isinstance(exit_code, int):
                    if self._restart_allowed(method, allow_restart):
                        self._restart_locked()
                        continue
                    raise MCPError(f"{self._label()} is not running (exit code {exit_code})")
                try:
                    self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
                    while True:
                        response = self._read(self._timeout_for(method))
                        if not isinstance(response, dict):
                            raise MCPError("invalid MCP response object")
                        if response.get("id") != request_id:
                            continue
                        if "error" in response:
                            raise MCPRemoteError(str(response["error"]))
                        return response.get("result")
                except MCPRemoteError:
                    # The server answered; retrying could duplicate a side effect.
                    raise
                except (MCPError, OSError, ValueError) as exc:
                    if not self._restart_allowed(method, allow_restart):
                        detail = self._diagnostic()
                        message = f"{self._label()} transport failure during {method}: {exc}" if method == "tools/call" else str(exc)
                        if detail and detail not in message:
                            message = f"{message}; MCP stderr: {detail}"
                        raise MCPError(message) from exc
                    self._restart_locked()

    def _timeout_for(self, method: str) -> float:
        return self.tool_timeout_seconds if method == "tools/call" else self.timeout_seconds

    def notify(self, method: str, params: dict[str, Any]) -> None:
        with self._io_lock:
            self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def list_tools(self) -> list[dict[str, Any]]:
        result = self.request("tools/list", {})
        tools = result.get("tools", []) if isinstance(result, dict) else []
        return [tool for tool in tools if isinstance(tool, dict)]

    def call_tool(self, name: str, arguments: dict[str, Any], idempotency_key: str = "") -> dict[str, Any]:
        payload = {"name": name, "arguments": arguments}
        if idempotency_key:
            payload["_meta"] = {"idempotencyKey": idempotency_key}
        result = self.request("tools/call", payload)
        return result if isinstance(result, dict) else {"content": result}

    def _write(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise MCPError("MCP client is not started")
        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        # MCP stdio uses newline-delimited JSON. Accepting Content-Length on read
        # keeps compatibility with the historical fixture transport.
        self.process.stdin.write(body + b"\n")
        self.process.stdin.flush()

    def _read(self, timeout_seconds: float | None = None) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise MCPError("MCP client is not started")
        selector = selectors.DefaultSelector()
        fd = self.process.stdout.fileno()
        os.set_blocking(fd, False)
        selector.register(fd, selectors.EVENT_READ)
        deadline = time.monotonic() + (self.timeout_seconds if timeout_seconds is None else timeout_seconds)
        try:
            line_end = self._read_until(selector, fd, b"\n", deadline, self.max_frame_bytes)
            first_line = bytes(self._stdout_buffer[:line_end]).strip()
            del self._stdout_buffer[:line_end + 1]
            if first_line.lower().startswith(b"content-length:"):
                headers: dict[str, str] = {"content-length": first_line.split(b":", 1)[1].strip().decode("ascii")}
                while True:
                    end = self._read_until(selector, fd, b"\n", deadline, self.max_frame_bytes)
                    header = bytes(self._stdout_buffer[:end]).strip()
                    del self._stdout_buffer[:end + 1]
                    if not header:
                        break
                    key, separator, value = header.decode("ascii", errors="replace").partition(":")
                    if not separator:
                        raise MCPError("invalid MCP response header")
                    headers[key.lower().strip()] = value.strip()
                length = int(headers["content-length"])
                if length < 0 or length > self.max_frame_bytes:
                    raise MCPError("MCP response exceeds frame limit")
                body = self._read_exact(selector, fd, length, deadline)
                return json.loads(body.decode("utf-8"))
            return json.loads(first_line.decode("utf-8"))
        except (KeyError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MCPError("invalid MCP response framing") from exc
        finally:
            selector.close()

    def _read_exact(self, selector: selectors.BaseSelector, fd: int, needed: int, deadline: float) -> bytes:
        """Read exactly `needed` bytes, leaving any pipelined frame in the buffer."""
        while len(self._stdout_buffer) < needed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError("MCP response timed out")
            if not selector.select(remaining):
                raise MCPError("MCP response timed out")
            chunk = os.read(fd, min(64 * 1024, max(1, needed - len(self._stdout_buffer))))
            if not chunk:
                raise MCPError("MCP process closed stdout")
            self._stdout_buffer.extend(chunk)
        body = bytes(self._stdout_buffer[:needed])
        del self._stdout_buffer[:needed]
        return body

    def _read_until(self, selector: selectors.BaseSelector, fd: int, marker: bytes, deadline: float, limit: int) -> int:
        while True:
            position = self._stdout_buffer.find(marker)
            if position >= 0:
                if position > limit:
                    raise MCPError("MCP response exceeds frame limit")
                return position
            if len(self._stdout_buffer) > limit:
                raise MCPError("MCP response exceeds frame limit")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError("MCP response timed out")
            if not selector.select(remaining):
                raise MCPError("MCP response timed out")
            chunk = os.read(fd, 64 * 1024)
            if not chunk:
                raise MCPError("MCP process closed stdout")
            self._stdout_buffer.extend(chunk)


class MCPToolMap(dict):
    """Dict of exposed MCP tools that also reports skipped servers."""

    def __init__(self, *args: Any, skipped_servers: Iterable[str] = (), **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skipped_servers: list[str] = list(skipped_servers)


class MCPClientList(list):
    """List of MCP clients with per-server restart and health helpers."""

    def __init__(self, items: Iterable[MCPStdioClient] = (), *, skipped_servers: Iterable[str] = ()) -> None:
        super().__init__(items)
        self.skipped_servers: list[str] = list(skipped_servers)

    def restart(self, server_name: str) -> None:
        for client in self:
            if client.server_name == server_name:
                client.restart(server_name)
                return
        raise MCPError(f"unknown MCP server: {server_name}")

    def health(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for client in self:
            result.update(client.health())
        return result


def start_mcp_servers(
    config: Mapping[str, Sequence[str]],
    *,
    timeout_seconds: float = 30.0,
    skip_failures: bool = False,
) -> tuple[MCPToolMap, MCPClientList]:
    """Start configured stdio servers and expose their discovered tools.

    A sensor failure must not kill the organism: with ``skip_failures`` a
    server that cannot start is logged and skipped, while the remaining
    servers still come up. Without it the first failure aborts startup.
    Skipped server names are recorded on both returned containers.
    """
    tools: dict[str, MCPTool] = {}
    clients: list[MCPStdioClient] = []
    skipped: list[str] = []
    try:
        for server_name, command in config.items():
            client = MCPStdioClient(command, timeout_seconds=timeout_seconds)
            client.server_name = server_name
            try:
                client.start()
                discovered = discover_mcp_tools(client)
            except Exception as exc:
                client.close()
                if not skip_failures:
                    raise
                log.warning("MCP server %s unavailable: %s", server_name, exc)
                skipped.append(server_name)
                continue
            client.tool_names = list(discovered)
            clients.append(client)
            for tool_name, tool in discovered.items():
                exposed_name = tool_name if tool_name not in tools else f"{server_name}_{tool_name}"
                tool.name = exposed_name
                tools[exposed_name] = tool
        return MCPToolMap(tools, skipped_servers=skipped), MCPClientList(clients, skipped_servers=skipped)
    except Exception:
        for client in clients:
            client.close()
        raise


@dataclass(slots=True)
class MCPTool:
    client: MCPStdioClient
    remote_name: str
    name: str
    description: str
    input_schema: dict[str, Any]
    capability_kind: str = "read"

    @property
    def schema(self) -> ToolSchema:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.input_schema}}

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        try:
            return {"ok": True, "result": self.client.call_tool(self.remote_name, dict(arguments), idempotency_key)}
        except MCPError as exc:
            return {"ok": False, "error": str(exc)}


def discover_mcp_tools(client: MCPStdioClient) -> dict[str, MCPTool]:
    return {
        str(item["name"]): MCPTool(
            client=client,
            remote_name=str(item["name"]),
            name=str(item["name"]),
            description=str(item.get("description", "MCP tool")),
            input_schema=item.get("inputSchema", {"type": "object", "additionalProperties": False}),
        )
        for item in client.list_tools()
        if item.get("name")
    }
