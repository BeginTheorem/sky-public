from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .models import RunStatus
from .reactor import Reactor
from .system_probe import SystemProbe, SystemProbeLog, default_services

logger = logging.getLogger("skynet.heartbeat")

# Module-level wake counter. The first wake anchors the probe cadence: the
# organism's very first moments are the ones most likely to take the host
# down, so wake #1 always dumps, then every Nth wake after it.
_wake_count = 0


def _probe_every() -> int:
    """Wakes between dumps; a malformed env value falls back to 20."""
    try:
        return max(1, int(os.getenv("SKYNET_SYSTEM_PROBE_EVERY", "20")))
    except ValueError:
        return 20


def _probe_services() -> list[str]:
    """Services to watch: explicit SKYNET_WATCH_SERVICES, else the defaults."""
    configured = os.getenv("SKYNET_WATCH_SERVICES", "")
    services = [name.strip() for name in configured.split(",") if name.strip()]
    return services or default_services()


def _recent_agent_commands(reactor: Reactor, limit: int = 20) -> list[str]:
    """Recover the organism's latest bash commands from the durable event log.

    ``react.py`` records each invocation as a ``tool_call`` event whose payload
    is ``{"call_id", "tool_name", "arguments"}``; bash's arguments carry the
    command under ``"command"``. The shape is read defensively because a
    foreign or older event must degrade to "no commands", never crash a wake.
    """
    try:
        rows = reactor.store.connection.execute(
            "SELECT payload FROM event_log WHERE kind='tool_call' ORDER BY sequence DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception:
        return []
    commands: list[str] = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("tool_name") != "bash":
            continue
        arguments = payload.get("arguments")
        command = arguments.get("command") if isinstance(arguments, dict) else None
        if isinstance(command, str) and command:
            commands.append(command)
    return commands


def _take_system_probe(reactor: Reactor) -> None:
    """Sample the host and append one JSONL record for crash forensics."""
    state_path = Path(reactor.config.state_path)
    probe_path = state_path.parent / "system-probe.jsonl"
    probe = SystemProbe(state_path.parent, services=_probe_services())
    sample = probe.sample(agent_commands=_recent_agent_commands(reactor))
    SystemProbeLog(probe_path).write(sample)
    # The real sampler nests memory/processes; accept a flat fake too.
    memory = sample.get("memory")
    processes = sample.get("processes")
    reactor.store.append_event(
        "system_probe",
        {
            "path": str(probe_path),
            "cpu_percent": sample.get("cpu_percent"),
            "memory_percent": memory.get("percent") if isinstance(memory, dict) else sample.get("memory_percent"),
            "process_count": processes.get("count") if isinstance(processes, dict) else sample.get("process_count"),
        },
    )


def _maybe_probe(reactor: Reactor) -> None:
    """Probe on the first wake and every Nth wake thereafter.

    A probe is forensic, not load-bearing: any failure is logged and swallowed
    so it can never stop ``reactor.tick`` from running.
    """
    global _wake_count
    _wake_count += 1
    if (_wake_count - 1) % _probe_every() != 0:
        return
    try:
        _take_system_probe(reactor)
    except Exception:
        logger.warning("system probe failed; continuing wake", exc_info=True)


def wake(reactor: Reactor, cause: str = "timer") -> RunStatus | None:
    """Attempt one wake; an active run is reported as busy, never duplicated."""
    _maybe_probe(reactor)
    reactor.store.append_event("heartbeat", {"cause": cause})
    reactor.watchdog()
    status = reactor.tick(cause)
    if status is not None:
        outcome = status.value
    elif reactor.store.state().active_run_id is not None:
        outcome = "busy"
    else:
        outcome = reactor.store.state().lifecycle.value
    reactor.store.append_event("heartbeat_result", {"cause": cause, "status": outcome})
    return status
