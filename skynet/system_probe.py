"""Bounded OS/system sampler for crash forensics.

When the organism dies there is usually no witness: the traceback is gone, the
process table is gone, and the only durable evidence is what some external
observer managed to write down before the lights went out. This module is that
observer. It samples load, memory, disk, network, processes, systemd units and
shell history, and every external command is run with a timeout so a wedged
``systemctl`` can never hang the caller. Failures are captured as data (``None``
or an empty list), never raised.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any

import psutil

DEFAULT_SERVICES = [
    "skynet.service",
    "skynet-telegram.service",
    "ssh",
    "systemd-journald",
    "cron",
]

_LISTING_LINES = 30
_HISTORY_LINES = 50
_COMMAND_LINES = 50
_TOP_PROCESSES = 15
_MAX_AGENT_COMMAND_CHARS = 300
_MAX_CMD_LINE_CHARS = 200


def default_services() -> list[str]:
    """The units whose liveness explains most of the organism's behaviour."""
    return list(DEFAULT_SERVICES)


class SystemProbe:
    """Cheap, best-effort sampler; a missing value is data, not an exception."""

    def __init__(
        self,
        workspace: Path,
        *,
        services: Sequence[str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.workspace = Path(workspace)
        self.services = list(services) if services is not None else default_services()
        self.timeout = max(0.1, float(timeout))
        # psutil reports 0.0 until it has a baseline; prime it once at
        # construction so the first real sample is a delta, not a zero.
        try:
            psutil.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None, percpu=True)
        except (psutil.Error, OSError):
            pass

    def _run(self, args: Sequence[str]) -> str | None:
        """Run a bounded external command; ``None`` means it did not answer."""
        try:
            completed = subprocess.run(
                list(args),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            # FileNotFoundError (no systemctl/sudo), TimeoutExpired, AccessDenied:
            # all of them are normal on a stripped-down host and all are data.
            return None
        return completed.stdout or ""

    def _listing(self, path: Path) -> list[str]:
        output = self._run(["ls", "-la", str(path)])
        if output is None:
            return []
        return output.splitlines()[:_LISTING_LINES]

    def _workspace_mount(self) -> Path:
        """Longest partition mountpoint that is a prefix of the workspace."""
        try:
            target = self.workspace.resolve()
        except OSError:
            target = self.workspace
        best: Path | None = None
        try:
            partitions = psutil.disk_partitions(all=False)
        except (psutil.Error, OSError):
            return target
        for partition in partitions:
            mount = Path(partition.mountpoint)
            try:
                target.relative_to(mount)
            except ValueError:
                continue
            if best is None or len(str(mount)) > len(str(best)):
                best = mount
        return best if best is not None else target

    def _disk(self) -> list[dict[str, Any]]:
        mounts: list[Path] = [self._workspace_mount()]
        with contextlib.suppress(psutil.Error, OSError):
            mounts.extend(Path(partition.mountpoint) for partition in psutil.disk_partitions(all=False))
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for mount in mounts:
            key = str(mount)
            if key in seen:
                continue
            seen.add(key)
            try:
                usage = psutil.disk_usage(key)
            except (psutil.Error, OSError):
                continue
            rows.append(
                {
                    "mountpoint": key,
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent,
                }
            )
        return rows

    def _network(self) -> dict[str, Any]:
        network: dict[str, Any] = {
            "bytes_sent": None,
            "bytes_recv": None,
            "packets_sent": None,
            "packets_recv": None,
            "established": None,
        }
        try:
            counters = psutil.net_io_counters()
            network.update(
                {
                    "bytes_sent": counters.bytes_sent,
                    "bytes_recv": counters.bytes_recv,
                    "packets_sent": counters.packets_sent,
                    "packets_recv": counters.packets_recv,
                }
            )
        except (psutil.Error, OSError):
            pass
        try:
            established = 0
            for connection in psutil.net_connections(kind="inet"):
                if connection.status == psutil.CONN_ESTABLISHED:
                    established += 1
            network["established"] = established
        except (psutil.Error, OSError):
            pass
        return network

    def _processes(self) -> dict[str, Any]:
        count = 0
        rows: list[dict[str, Any]] = []
        try:
            iterator = psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_percent", "memory_info", "cmdline"]
            )
            for process in iterator:
                count += 1
                info = process.info
                if not info:
                    continue
                memory_info = info.get("memory_info")
                rss = int(getattr(memory_info, "rss", 0) or 0)
                command = info.get("cmdline") or []
                cmdline = " ".join(str(part) for part in command)[:_MAX_CMD_LINE_CHARS]
                rows.append(
                    {
                        "pid": info.get("pid"),
                        "name": info.get("name"),
                        "cpu_percent": info.get("cpu_percent"),
                        "memory_percent": info.get("memory_percent"),
                        "rss": rss,
                        "cmdline": cmdline,
                    }
                )
        except (psutil.Error, OSError):
            pass
        rows.sort(key=lambda row: row["rss"], reverse=True)
        return {"count": count, "top": rows[:_TOP_PROCESSES]}

    def _services(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in self.services:
            active_output = self._run(["systemctl", "is-active", name])
            failed_output = self._run(["systemctl", "is-failed", name])
            active = active_output.strip() == "active" if active_output is not None else None
            failed = failed_output.strip() == "failed" if failed_output is not None else None
            rows.append({"name": name, "active": active, "failed": failed})
        return rows

    def _sudo_commands(self) -> list[str]:
        output = self._run(
            ["sudo", "-n", "journalctl", "-q", "_COMM=sudo", "-n", "50", "--no-pager"]
        )
        if not output:
            return []
        return output.splitlines()[:_COMMAND_LINES]

    def _bash_history(self) -> list[str]:
        try:
            text = (Path.home() / ".bash_history").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return [line[:_MAX_AGENT_COMMAND_CHARS] for line in text.splitlines()[-_HISTORY_LINES:]]

    def sample(self, *, agent_commands: Sequence[str] | None = None) -> dict[str, Any]:
        """One bounded snapshot; every section degrades to ``None``/``[]``."""
        errors: list[str] = []

        loadavg: list[float] | None = None
        try:
            loadavg = list(os.getloadavg())
        except (OSError, AttributeError) as exc:
            errors.append(f"loadavg: {type(exc).__name__}")

        cpu_percent: float | None = None
        cpu_per_core: list[float] | None = None
        cpu_count: int | None = None
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_per_core = list(psutil.cpu_percent(interval=None, percpu=True))
            cpu_count = psutil.cpu_count()
        except (psutil.Error, OSError) as exc:
            errors.append(f"cpu: {type(exc).__name__}")

        memory: dict[str, Any] = {
            "total": None,
            "available": None,
            "used": None,
            "percent": None,
            "swap": {"total": None, "used": None, "percent": None},
        }
        try:
            virtual = psutil.virtual_memory()
            swap = psutil.swap_memory()
            memory.update(
                {
                    "total": virtual.total,
                    "available": virtual.available,
                    "used": virtual.used,
                    "percent": virtual.percent,
                    "swap": {"total": swap.total, "used": swap.used, "percent": swap.percent},
                }
            )
        except (psutil.Error, OSError) as exc:
            errors.append(f"memory: {type(exc).__name__}")

        try:
            disk = self._disk()
        except (psutil.Error, OSError) as exc:
            errors.append(f"disk: {type(exc).__name__}")
            disk = []

        try:
            network = self._network()
        except (psutil.Error, OSError) as exc:
            errors.append(f"network: {type(exc).__name__}")
            network = {
                "bytes_sent": None,
                "bytes_recv": None,
                "packets_sent": None,
                "packets_recv": None,
                "established": None,
            }

        try:
            processes = self._processes()
        except (psutil.Error, OSError) as exc:
            errors.append(f"processes: {type(exc).__name__}")
            processes = {"count": None, "top": []}

        commands = [str(command)[:_MAX_AGENT_COMMAND_CHARS] for command in (agent_commands or [])]

        return {
            "loadavg": loadavg,
            "cpu_percent": cpu_percent,
            "cpu_per_core": cpu_per_core,
            "cpu_count": cpu_count,
            "memory": memory,
            "disk": disk,
            "network": network,
            "processes": processes,
            "services": self._services(),
            "home_listing": self._listing(Path.home()),
            "tmp_listing": self._listing(Path("/tmp")),
            "agent_commands": commands[-_COMMAND_LINES:],
            "sudo_commands": self._sudo_commands(),
            "bash_history": self._bash_history(),
            "probe_error": "; ".join(errors) or None,
        }


class SystemProbeLog:
    """JSONL sink for probe snapshots with a count ring and a byte cap.

    Crash forensics must survive the crash, so this writer never raises on
    ``OSError``: an unwritable log is worse than no log, but not worth killing
    the sampler for. The count ring keeps the newest ``max_records`` lines; the
    byte cap rewrites the file compactly when it grows past ``max_bytes``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_records: int = 250,
        max_bytes: int = 10_000_000,
    ) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self.max_records = max(1, int(max_records))
        self.max_bytes = max(0, int(max_bytes))

    def _read_lines(self) -> list[str]:
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return [line + "\n" for line in text.splitlines() if line.strip()]

    def _fit(self, lines: list[str]) -> list[str]:
        """Newest suffix whose UTF-8 size fits ``max_bytes`` (at least one line)."""
        if not self.max_bytes:
            return lines
        kept: deque[str] = deque()
        total = 0
        for line in reversed(lines):
            size = len(line.encode("utf-8"))
            if kept and total + size > self.max_bytes:
                break
            kept.appendleft(line)
            total += size
        return list(kept)

    def write(self, payload: dict[str, Any]) -> None:
        """Append one JSON line, then enforce the ring and the byte cap."""
        line = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")) + "\n"
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                lines = self._read_lines()
                lines.append(line)
                rewrite = False
                if len(lines) > self.max_records:
                    lines = lines[-self.max_records :]
                    rewrite = True
                if self.max_bytes and sum(len(item.encode("utf-8")) for item in lines) > self.max_bytes:
                    lines = self._fit(lines)
                    rewrite = True
                if rewrite:
                    self.path.write_text("".join(lines), encoding="utf-8")
                else:
                    with self.path.open("a", encoding="utf-8") as stream:
                        stream.write(line)
                        stream.flush()
            except OSError:
                # A crash log that cannot be written must not crash the sampler.
                return
