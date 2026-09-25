from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import signal
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4
from zoneinfo import ZoneInfo

from .models import Budget
from .time import DISPLAY_TIMEZONE, display_timestamp, utc_now


class UTCLogFormatter(logging.Formatter):
    def __init__(self, fmt: str | None = None, datefmt: str | None = None, timezone_name: str | None = None) -> None:
        super().__init__(fmt, datefmt)
        self.timezone_name = timezone_name or DISPLAY_TIMEZONE

    def formatTime(self, record, datefmt=None):
        value = datetime.fromtimestamp(record.created, ZoneInfo(self.timezone_name))
        return value.strftime(datefmt) if datefmt else value.isoformat(timespec="seconds")


def _log_format(timezone_name: str) -> str:
    """Build the log line format with the configured timezone label."""
    return f"%(asctime)s [{timezone_name}] [%(levelname)s] %(name)s: %(message)s"
from .checkpoints import CheckpointError, CheckpointManager
from .heartbeat import wake as heartbeat_wake
from .lock import ProcessLock
from .mcp import MCPStdioClient, start_mcp_servers
from .provider import Tool
from .providers import build_provider
from .reactor import ReactorConfig, WatchdogTimeout
from .recovery import RecoveryError
from .store import StateStore
from .supervisor import Supervisor
from .tools import default_tools
from .watchdog import StaleRunWatchdog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one bounded SkyNet lifecycle tick.")
    parser.add_argument("--once", action="store_true", help="run one tick and exit")
    parser.add_argument("--interval", type=float, default=60.0, help="seconds between ticks")
    parser.add_argument("--state", default=os.getenv("SKYNET_STATE", "state/skynet.sqlite3"))
    parser.add_argument("command", nargs="?", choices=("run", "start", "stop", "reboot", "clean-start", "checkpoint", "rollback", "reconcile", "send", "status", "health", "goals", "inbox", "outbox", "deliver", "memory", "logs", "time", "reset-short-memory", "reset-dispatcher", "forget", "money-boost", "verbose", "metrics", "alerts", "handoff", "clean", "log", "message"), default="run")
    parser.add_argument("value", nargs="?")
    parser.add_argument("--kinds", default="", help="comma-separated event kinds to include in `log`")
    parser.add_argument("--tier", choices=("system", "mandatory", "advanced", "verbose"), default="advanced", help="which log tier `logs` reads")
    parser.add_argument("--limit", type=int, default=None, help="maximum entries to print for the selected `logs` tier")
    parser.add_argument("--service", default=os.getenv("SKYNET_SERVICE", "skynet.service"))
    parser.add_argument("--telegram-service", default=os.getenv("SKYNET_TELEGRAM_SERVICE", "skynet-telegram.service"))
    parser.add_argument("--force", action="store_true", help="stop the service, perform the reset, then restart it")
    parser.add_argument("--root", default=os.getenv("SKYNET_ROOT", os.getcwd()))
    parser.add_argument("--timezone", default=os.getenv("SKYNET_DISPLAY_TIMEZONE"), help="IANA timezone for displayed timestamps, e.g. UTC")
    return parser


def _systemctl_action(action: str, services: list[str]) -> subprocess.CompletedProcess[str]:
    # SKYNET_SYSTEMCTL is the same test/ops override the startup rollback script
    # uses; when set it is invoked directly (no sudo) so a fake binary works.
    executable = os.getenv("SKYNET_SYSTEMCTL") or "systemctl"
    command = [executable, action, *services]
    if os.geteuid() != 0 and not os.getenv("SKYNET_SYSTEMCTL"):
        command = ["sudo", "-n", *command]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _clean_start_command(state_path: str, service: str, telegram_service: str) -> int:
    state = Path(state_path)
    state_dir = state.parent
    services = [service, telegram_service]
    stopped = _systemctl_action("stop", services)
    if stopped.returncode:
        print(stopped.stderr.strip() or "failed to stop services", file=sys.stderr)
        return stopped.returncode
    state_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = state_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive_path = archive_dir / f"full-reset-before-{stamp}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for child in state_dir.iterdir():
            if child.name != "archive":
                archive.add(child, arcname=str(Path(state_dir.name) / child.name))
    for name in (
        state.name, f"{state.name}-wal", f"{state.name}-shm", "runtime.jsonl",
        "provider-fallback.json", "checkpoint.json", "reboot-guard.json",
        "runtime-backup.json", "self-improvement-proposals.json", "money-boost.json",
        "skynet.lock",
    ):
        target = state_dir / name
        if target.is_dir():
            import shutil
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
    store = StateStore(state)
    store.close()
    enabled = _systemctl_action("enable", services)
    if enabled.returncode:
        print(enabled.stderr.strip() or "failed to enable services", file=sys.stderr)
        return enabled.returncode
    started = _systemctl_action("start", services)
    if started.returncode:
        print(started.stderr.strip() or "failed to start services", file=sys.stderr)
        return started.returncode
    print(json.dumps({"started": services, "state": str(state), "archive": str(archive_path)}, ensure_ascii=False))
    return 0


def _runtime_env_value(root: str, name: str) -> str:
    value = os.getenv(name, "")
    if value:
        return value
    env_path = Path(root) / "config" / "skynet.env"
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def _money_boost_command(value: str | None, root: str, state_path: str) -> int:
    configured_path = os.getenv("SKYNET_MONEY_BOOST_STATE", "")
    path = Path(configured_path) if configured_path else Path(state_path).with_name("money-boost.json")
    if value not in {"on", "off", "status"}:
        print("money-boost requires on, off, or status", file=sys.stderr)
        return 2
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError, TypeError):
        current = {}
    enabled = bool(current.get("enabled", False))
    if value == "status":
        configured = bool(_runtime_env_value(root, "OPENROUTER_API_KEY"))
        print(json.dumps({"money_boost": enabled, "openrouter_configured": configured, "state_path": str(path)}, ensure_ascii=False))
        return 0
    target = value == "on"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"enabled": target}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    # The running FallbackProvider re-reads this file through
    # active_chain_names() on every call, so no service restart is needed:
    # toggling money-boost is genuinely hot in both directions.
    print(json.dumps({"money_boost": target, "applied": "hot-reload", "state_path": str(path)}, ensure_ascii=False))
    return 0


def _verbose_command(value: str | None, state_path: str) -> int:
    """Toggle or inspect the provider-dump state file read by the service."""
    path = Path(state_path).with_name("verbose.json")
    if value not in {"on", "off", "status"}:
        print("verbose requires on, off, or status", file=sys.stderr)
        return 2
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError, TypeError):
        current = {}
    enabled = bool(current.get("enabled", False))
    if value == "status":
        print(json.dumps({"verbose": enabled, "state_path": str(path)}, ensure_ascii=False))
        return 0
    target = value == "on"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"enabled": target}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    # The running service reads this file (SKYNET_VERBOSE_PROVIDER remains the
    # runtime override), so the toggle is hot and no service restart is issued.
    print(json.dumps({
        "verbose": target,
        "applied": "hot-reload",
        "state_path": str(path),
        "note": "the running service reads this file; no restart needed",
    }, ensure_ascii=False))
    return 0


def _logs_limit(args: argparse.Namespace, default: int, low: int, high: int) -> int:
    """Resolve the effective `logs` limit, honouring the legacy `log <N>` form."""
    raw = getattr(args, "limit", None)
    if raw is None:
        positional = getattr(args, "value", None)
        if positional is not None and str(positional).strip().isdigit():
            raw = int(str(positional).strip())
    if raw is None:
        raw = default
    return max(low, min(int(raw), high))


def _logs_command(store: StateStore, args: argparse.Namespace) -> int:
    """Single entry point for the log tiers; each tier owns its own source."""
    tier = getattr(args, "tier", "advanced") or "advanced"
    state_dir = Path(args.state).parent
    if tier == "mandatory":
        limit = _logs_limit(args, 5, 1, 100)
        from . import reporting
        rendered = reporting.render_recent(store, limit)
        print(rendered if isinstance(rendered, str) else "\n".join(str(line) for line in rendered))
        return 0
    if tier == "system":
        limit = _logs_limit(args, 20, 1, 250)
        probe_path = state_dir / "system-probe.jsonl"
        if not probe_path.exists():
            print(f"no system probe records at {probe_path}")
            return 0
        for line in probe_path.read_text(encoding="utf-8").splitlines()[-limit:]:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(line)
                continue
            print(json.dumps(record, ensure_ascii=False, indent=2, default=str))
        return 0
    if tier == "verbose":
        limit = _logs_limit(args, 50, 1, 10_000)
        verbose_path = state_dir / "verbose.jsonl"
        if not verbose_path.exists():
            enabled = False
            state_file = state_dir / "verbose.json"
            try:
                if state_file.exists():
                    enabled = bool(json.loads(state_file.read_text(encoding="utf-8")).get("enabled", False))
            except (OSError, json.JSONDecodeError, TypeError):
                enabled = False
            if enabled:
                print(f"verbose logging is on but {verbose_path} has no output yet")
            else:
                print(f"verbose logging is off ({verbose_path} absent); enable it with `skynet verbose on`")
            return 0
        lines = verbose_path.read_text(encoding="utf-8").splitlines()[-limit:]
        if not lines:
            print(f"verbose logging is on but {verbose_path} is empty")
            return 0
        print("\n".join(lines))
        return 0
    # advanced: the compact technical view that `logs` has always produced.
    limit = _logs_limit(args, 500, 1, 10_000)
    kinds = {item.strip() for item in (getattr(args, "kinds", "") or "").split(",") if item.strip()}
    if kinds:
        placeholders = ",".join("?" for _ in kinds)
        rows = store.connection.execute(
            f"SELECT kind, payload, run_id, created_at FROM event_log WHERE kind IN ({placeholders}) "
            "ORDER BY sequence DESC LIMIT ?",
            (*sorted(kinds), limit),
        ).fetchall()
        for row in rows:
            print(json.dumps(dict(row), ensure_ascii=False, default=str))
        return 0
    runtime_path = state_dir / "runtime.jsonl"
    if runtime_path.exists():
        lines = runtime_path.read_text(encoding="utf-8").splitlines()[-limit:]
        print("\n".join(lines))
    else:
        print(json.dumps([dict(row) for row in store.connection.execute("SELECT * FROM event_log ORDER BY sequence DESC LIMIT ?", (limit,))], default=str))
    return 0


def _stop_event() -> threading.Event:
    """Create the event that stops the supervisor loop.

    The loop obtains its stop event through this factory instead of calling
    ``threading.Event`` inline, so a test can replace only the loop's own
    event. Patching the stdlib ``threading.Event`` globally also intercepts
    ``Event()`` calls made by unrelated threads: ``MCPStdioClient.start()``
    creates its stderr drain thread at the same point in startup, and
    ``Thread.__init__`` stores the mocked event as ``Thread._started``, which
    ``_bootstrap_inner`` sets when that thread starts. The MCP thread would
    then set the CLI loop's stop flag and the loop would exit before running
    its first cycle.
    """
    return threading.Event()


def _mcp_config() -> dict[str, list[str]]:
    """Build the MCP server map from SKYNET_MCP_<NAME>_COMMAND.

    The four sensors used to be hardcoded keys, so adding one required editing
    this file. Any SKYNET_MCP_<NAME>_COMMAND now registers a server, and
    SKYNET_MCP_<NAME>_DISABLED=1 removes one without touching the command.
    """
    prefix, suffix = "SKYNET_MCP_", "_COMMAND"
    config: dict[str, list[str]] = {}
    for key in sorted(os.environ):
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        name = key[len(prefix): -len(suffix)].lower()
        if not name:
            continue
        if os.getenv(f"{prefix}{name.upper()}_DISABLED", "").strip().casefold() in {"1", "true", "yes", "on"}:
            continue
        command = shlex.split(os.environ.get(key, "") or "")
        if command:
            config[name] = command
    return config


def _jsonl_drain_enabled() -> bool:
    """Whether the loop drains the outbox to a local jsonl file.

    Off by default: the Telegram drainer owns the queue, and two consumers
    racing for the same lease meant an alert could be written to the jsonl file
    and never marked delivered, so the owner never saw it and `alerts_pending`
    lied. Operators can still opt in with SKYNET_OUTBOX_JSONL=1, or use the
    explicit `skynet deliver` command.
    """
    return os.getenv("SKYNET_OUTBOX_JSONL", "false").strip().casefold() in {"1", "true", "yes", "on"}


def _drain_outbox(store: StateStore, target: str) -> None:
    """Deliver queued notifications to the local append-only boundary each cycle."""
    if not _jsonl_drain_enabled():
        return
    try:
        if not store.pending_outbox(limit=1):
            return
        from .outbox import deliver_to_jsonl
        delivered = deliver_to_jsonl(store, target)
        if delivered:
            store.append_event("outbox_delivered", {"count": delivered, "target": target})
    except Exception:
        logging.getLogger("skynet.cli").exception("outbox delivery failed; continuing")


def _forget_memories(store: StateStore, target: str) -> dict[str, object]:
    """Delete memories matching a memory id or a content substring."""
    rows = store.connection.execute(
        "SELECT memory_id, content FROM memories WHERE memory_id=? OR content LIKE ?",
        (target, f"%{target}%"),
    ).fetchall()
    memory_ids = [row["memory_id"] for row in rows]
    fts_available = bool(store.memory_store is not None and store.memory_store.fts_available)
    for memory_id in memory_ids:
        store.connection.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))
        if fts_available:
            store.connection.execute("DELETE FROM memories_fts WHERE memory_id=?", (memory_id,))
    store.connection.commit()
    return {"forgotten": len(memory_ids), "target": target, "memory_ids": memory_ids}


def _window_days(value: str | None) -> float:
    """Parse a metrics window argument like `7` or `7d`; clamp to [0.1, 90]."""
    if not value:
        return 1.0
    text = str(value).strip().casefold().rstrip("d")
    try:
        days = float(text)
    except ValueError:
        return 1.0
    return max(0.1, min(days, 90.0))


def _reset_command(args: argparse.Namespace) -> int:
    """Run a manual reset, honouring the service lock and ``--force``."""
    process_lock = ProcessLock(Path(args.state).with_suffix(".lock"))
    service_stopped = False
    try:
        process_lock.acquire()
    except RuntimeError as exc:
        if not args.force:
            print(
                f"{exc}. Stop the service first: sudo systemctl stop {args.service} "
                f"(or re-run with --force to stop, reset, and restart it).",
                file=sys.stderr,
            )
            return 1
        stopped = _systemctl_action("stop", [args.service])
        if stopped.returncode:
            print(stopped.stderr.strip() or f"failed to stop {args.service}", file=sys.stderr)
            return stopped.returncode
        service_stopped = True
        try:
            process_lock.acquire()
        except RuntimeError as retry_exc:
            print(str(retry_exc), file=sys.stderr)
            _systemctl_action("start", [args.service])
            return 1
    store: StateStore | None = None
    try:
        if args.command == "forget" and not args.value:
            print("forget requires a memory id or content pattern", file=sys.stderr)
            return 2
        store = StateStore(args.state)
        if args.command == "reset-dispatcher":
            details = store.reset_dispatcher()
        elif args.command == "forget":
            details = _forget_memories(store, str(args.value))
        else:
            details = store.reset_short_memory(allow_interrupted_run=True)
        print(json.dumps(details, ensure_ascii=False, default=str))
        return 0
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()
        process_lock.release()
        if service_stopped:
            restarted = _systemctl_action("start", [args.service])
            if restarted.returncode:
                print(restarted.stderr.strip() or f"failed to restart {args.service}", file=sys.stderr)


def main() -> int:
    args = build_parser().parse_args()
    timezone_name = args.timezone or DISPLAY_TIMEZONE
    log_format = _log_format(timezone_name)
    logging.basicConfig(
        level=os.getenv("SKYNET_LOG_LEVEL", "INFO"),
        format=log_format,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(UTCLogFormatter(log_format, "%Y-%m-%dT%H:%M:%S", timezone_name))
    if args.command == "money-boost":
        return _money_boost_command(args.value, args.root, args.state)
    if args.command == "verbose":
        return _verbose_command(args.value, args.state)
    if args.command == "clean":
        args.command = "clean-start"
    if args.command == "log":
        args.command = "logs"
    if args.command == "clean-start":
        return _clean_start_command(args.state, args.service, args.telegram_service)
    if args.command in {"start", "stop", "reboot"}:
        action = "restart" if args.command == "reboot" else args.command
        services = [args.service, args.telegram_service]
        completed = _systemctl_action(action, services)
        if completed.returncode:
            print(completed.stderr.strip() or f"systemctl {action} failed", file=sys.stderr)
            return completed.returncode
        print(f"{action} requested for {', '.join(services)}")
        return 0
    if args.command in {"checkpoint", "rollback"}:
        manager = CheckpointManager(args.root)
        try:
            if args.command == "checkpoint":
                state_store = StateStore(args.state, read_only=True)
                try:
                    state = state_store.state()
                    tables = {row["name"] for row in state_store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    required = {"agent_state", "runs", "run_results", "event_log", "checkpoints", "inbox", "outbox", "capability_effects", "episode_snapshots", "memory_consolidations", "recovery_reconciliations", "schema_migrations"}
                    health = {"ok": required <= tables, "checks": {"schema": required <= tables, "state": state.lifecycle.value}}
                finally:
                    state_store.close()
                if not health["ok"]:
                    print(json.dumps(health, default=str), file=sys.stderr)
                    return 1
                print(json.dumps(manager.create(health).as_dict(), default=str))
                return 0
            print(json.dumps({"rolled_back_to": manager.rollback()}, default=str))
            return 0
        except CheckpointError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if args.command in {"reset-short-memory", "reset-dispatcher", "forget"}:
        return _reset_command(args)
    if args.command == "time":
        now = utc_now()
        print(json.dumps({"utc": now, "display": display_timestamp(now, args.timezone)}, ensure_ascii=False))
        return 0
    if args.command != "run":
        try:
            store = StateStore(args.state, read_only=args.command in {"status", "health", "goals", "inbox", "outbox", "memory", "logs", "metrics", "alerts", "message"})
        except (OSError, sqlite3.OperationalError) as exc:
            print(f"cannot open state database; run this diagnostic through the service account: {exc}", file=sys.stderr)
            return 1
        try:
            if args.command == "send":
                if not args.value:
                    raise SystemExit("send requires a message")
                store.add_inbox_event(str(uuid4()), "user_message", {"text": args.value})
                print("queued")
            elif args.command == "reconcile":
                if not args.value or ":" not in args.value:
                    raise SystemExit("reconcile requires RUN_ID:CALL_ID")
                run_id, call_id = args.value.split(":", 1)
                payload = json.loads(os.getenv("SKYNET_RECONCILIATION_RESULT", "{}"))
                status = os.getenv("SKYNET_RECONCILIATION_STATUS", "unknown")
                if status not in {"confirmed", "not_applied", "unknown"}:
                    raise SystemExit("SKYNET_RECONCILIATION_STATUS must be confirmed, not_applied, or unknown")
                with store.transaction():
                    print(json.dumps(store.reconcile_tool_call(run_id, call_id, status, payload)))
            elif args.command == "status":
                print(json.dumps(store.state().__dict__ if hasattr(store.state(), "__dict__") else {"lifecycle": store.state().lifecycle.value, "generation": store.state().generation, "active_run_id": store.state().active_run_id, "next_plan": store.state().next_plan}, default=str))
            elif args.command == "health":
                state = store.state()
                tables = {row["name"] for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                required = {"agent_state", "runs", "run_results", "event_log", "checkpoints", "inbox", "outbox", "capability_effects", "episode_snapshots", "memory_consolidations", "recovery_reconciliations", "schema_migrations"}
                result = {"ok": required <= tables, "checks": {"schema": required <= tables, "state": state.lifecycle.value}}
                print(json.dumps(result, default=str))
                return 0 if result["ok"] else 1
            elif args.command == "goals":
                print(json.dumps([dict(row) for row in store.connection.execute("SELECT * FROM goals ORDER BY priority DESC")], default=str))
            elif args.command == "inbox":
                print(json.dumps(store.pending_inbox(), default=str))
            elif args.command == "outbox":
                print(json.dumps(store.pending_outbox(), default=str))
            elif args.command == "deliver":
                from .outbox import deliver_to_http, deliver_to_jsonl
                endpoint = os.getenv("SKYNET_OUTBOX_ENDPOINT", "")
                if endpoint:
                    print(json.dumps({"delivered": deliver_to_http(store, endpoint, api_key=os.getenv("SKYNET_OUTBOX_API_KEY", ""))}))
                else:
                    target = args.value or os.getenv("SKYNET_OUTBOX", "state/outbox.jsonl")
                    print(json.dumps({"delivered": deliver_to_jsonl(store, target)}))
            elif args.command == "memory":
                # The read-only StateStore deliberately has no memory_store, so
                # build the search facade directly on the read-only connection.
                from .memory_store import MemoryStore
                try:
                    results = MemoryStore(store.connection).search(args.value or "")
                except Exception as exc:
                    print(f"memory search unavailable: {exc}", file=sys.stderr)
                    return 1
                print(json.dumps(results, ensure_ascii=False, default=str))
            elif args.command == "metrics":
                from . import metrics
                data = metrics.snapshot(
                    store,
                    since_days=_window_days(args.value),
                    registry_path=Path(args.state).parent / "self-improvement-proposals.json",
                    state_dir=Path(args.state).parent,
                )
                if os.getenv("SKYNET_METRICS_JSON") or (args.value or "").endswith("json"):
                    print(json.dumps(data, ensure_ascii=False, default=str))
                else:
                    print(metrics.format_report(data))
            elif args.command == "handoff":
                from .handoff import seed
                writable = StateStore(args.state)
                try:
                    print(json.dumps(seed(writable, root=args.root), ensure_ascii=False, default=str))
                finally:
                    writable.close()
            elif args.command == "alerts":
                pending = store.pending_alerts(limit=50)
                recent = store.alerts.recent(limit=50)
                print(json.dumps({"pending": pending, "recent": recent}, ensure_ascii=False, default=str))
            elif args.command == "message":
                # The operator's view of the agent-to-owner channel.
                pending = store.pending_inbox(limit=50)
                if not pending:
                    print("no pending messages")
                for item in pending:
                    print(json.dumps(item, ensure_ascii=False, default=str))
            elif args.command == "logs":
                return _logs_command(store, args)
        finally:
            store.close()
        return 0
    try:
        provider = build_provider()
    except RuntimeError:
        logging.getLogger("skynet.cli").exception("provider configuration failed")
        return 1
    config = ReactorConfig(state_path=args.state, self_improvement_root=args.root, wake_interval_seconds=args.interval, budget=Budget(
        steps=int(os.getenv("SKYNET_MAX_STEPS", "100")),
        tokens=int(os.getenv("SKYNET_REACT_INPUT_TOKENS", "200000")),
        seconds=float(os.getenv("SKYNET_MAX_SECONDS", "3600")),
        output_tokens=int(os.getenv("SKYNET_OUTPUT_TOKENS", "8192")),
    ), memory_input_tokens=int(os.getenv("SKYNET_MEMORY_INPUT_TOKENS", "50000")),
        memory_loop_timeout_seconds=float(os.getenv("SKYNET_MEMORY_LOOP_TIMEOUT", "300")),
        planner_output_tokens=int(os.getenv("SKYNET_PLANNER_OUTPUT_TOKENS", "16384")),
        run_progress_seconds=float(os.getenv("SKYNET_RUN_PROGRESS_SECONDS", "3600")),
        transcript_retention_runs=int(os.getenv("SKYNET_TRANSCRIPT_RETENTION_RUNS", "200")),
        task_giveup_failures=int(os.getenv("SKYNET_TASK_GIVEUP_FAILURES", "3")),
        restart_failure_limit=int(os.getenv("SKYNET_RESTART_FAILURE_LIMIT", "3")),
        reboot_window_max_age_seconds=float(os.getenv("SKYNET_REBOOT_WINDOW_MAX_AGE", "86400")),
        watchdog_timeout_seconds=float(os.getenv("SKYNET_WATCHDOG_SECONDS", "3900")),
        provider_lockout_threshold=int(os.getenv("SKYNET_PROVIDER_LOCKOUT_THRESHOLD", "3")),
        provider_lockout_seconds=float(os.getenv("SKYNET_PROVIDER_LOCKOUT_SECONDS", "1800")),
        livelock_streak_limit=int(os.getenv("SKYNET_LIVELOCK_STREAK_LIMIT", "3")),
        pinned_memory_limit=int(os.getenv("SKYNET_PINNED_MEMORY_LIMIT", "32")),
        memory_inject_limit=int(os.getenv("SKYNET_MEMORY_INJECT_LIMIT", "3")),
        metrics_snapshot_enabled=os.getenv("SKYNET_METRICS_SNAPSHOT", "true").strip().casefold() not in {"0", "false", "no", "off"},
        planner_epsilon=float(os.getenv("SKYNET_PLANNER_EPSILON", "0.1")),
        hypothesis_ttl_days=float(os.getenv("SKYNET_HYPOTHESIS_TTL_DAYS", "30")),
        cell_scarcity_weight=float(os.getenv("SKYNET_CELL_SCARCITY_WEIGHT", "0.20")),
        external_seek_enabled=os.getenv("SKYNET_EXTERNAL_SEEK", "true").strip().casefold() not in {"0", "false", "no", "off"},
        external_seek_every_generations=int(os.getenv("SKYNET_EXTERNAL_SEEK_EVERY", "4")),
        external_seek_cooldown_seconds=float(os.getenv("SKYNET_EXTERNAL_SEEK_COOLDOWN", "3600")),
        fallback_repeat_generations=int(os.getenv("SKYNET_FALLBACK_REPEAT_GENERATIONS", "4")),
        fallback_repeat_seconds=float(os.getenv("SKYNET_FALLBACK_REPEAT_SECONDS", "900")),
        stall_alert_seconds=float(os.getenv("SKYNET_STALL_ALERT_SECONDS", "3600")),
        external_seek_min_seconds=float(os.getenv("SKYNET_EXTERNAL_SEEK_MIN_SECONDS", "3600")),
        criterion_epoch_generations=int(os.getenv("SKYNET_CRITERION_EPOCH_GENERATIONS", "100")),
        archive_parent_k=int(os.getenv("SKYNET_ARCHIVE_PARENT_K", "2")),
        archive_parent_lambda=float(os.getenv("SKYNET_ARCHIVE_PARENT_LAMBDA", "10")),
        archive_parent_alpha0=float(os.getenv("SKYNET_ARCHIVE_PARENT_ALPHA0", "0.5")),
        max_active_goals=int(os.getenv("SKYNET_MAX_ACTIVE_GOALS", "8")),
        event_retention_days=int(os.getenv("SKYNET_EVENT_RETENTION_DAYS", "30")),
        effect_retention_days=int(os.getenv("SKYNET_EFFECT_RETENTION_DAYS", "30")))
    tools = cast(dict[str, Tool], default_tools())
    mcp_clients: list[MCPStdioClient] = []
    mcp_config = _mcp_config()
    if mcp_config:
        discovered, mcp_clients = start_mcp_servers(
            mcp_config,
            timeout_seconds=float(os.getenv("SKYNET_MCP_TIMEOUT", "30")),
            skip_failures=True,
        )
        discovered = cast(dict[str, Tool], discovered)
        # A sensor must never silently replace a builtin: a shadowed `bash` or
        # `propose_self_improvement` would change what the organism is without
        # any trace. Prefix the intruder and record it.
        collisions = sorted(set(discovered) & set(tools))
        for name in collisions:
            discovered[f"mcp_{name}"] = discovered.pop(name)
        if collisions:
            logging.getLogger("skynet.cli").warning("MCP tools renamed to avoid shadowing builtins: %s", collisions)
        tools.update(discovered)
        skipped = list(getattr(mcp_clients, "skipped_servers", []) or [])
        if collisions or skipped:
            try:
                startup_store = StateStore(args.state, read_only=False)
                try:
                    if collisions:
                        startup_store.append_event("tool_name_collision", {"renamed": collisions})
                    if skipped:
                        startup_store.append_event("mcp_server_unavailable", {"servers": skipped})
                        startup_store.raise_alert(
                            "mcp_server_unavailable",
                            {"servers": skipped, "message": "configured MCP sensor(s) did not start; the organism is partially blind"},
                            severity="warning",
                            dedup_key="mcp_server_unavailable",
                        )
                finally:
                    startup_store.close()
            except (OSError, sqlite3.OperationalError) as exc:
                logging.getLogger("skynet.cli").warning("could not record MCP startup state: %s", exc)
    supervisor = Supervisor(provider, tools, config, root=args.root)
    # A promotion must restart every unit that imports this tree: the reactor
    # and the Telegram control bot. Restarting only the reactor left the bot
    # running stale code indefinitely.
    #
    # The reactor goes LAST, and that ordering is load-bearing. This callback
    # runs inside the reactor's own process, so `systemctl restart` starts a
    # stop job for whichever unit is named first: naming the reactor first
    # killed the client before systemd handled the remaining jobs of the same
    # transaction, and the bot was never restarted at all - it stayed the
    # process started at 01:50:10 while the tree moved on to f0fb0486 at
    # 02:09:39. Verified on throwaway transient units: reactor-first restarted
    # the reactor three times and left the other unit untouched in three of
    # three trials; other-unit-first restarted both units in three of three.
    restart_units = [args.telegram_service, args.service]
    supervisor.reactor.set_self_improvement_restart(lambda: supervisor.restart_service(restart_units))
    supervisor.set_restart_callback(lambda: supervisor.restart_service(restart_units))
    supervisor.reactor.set_self_improvement_health(supervisor.health_check)

    # Install signal handling before startup recovery: a systemd stop during
    # reconcile/checkpoint/runtime backup must not run with default handlers.
    wake_cause = "startup"
    stop_event = _stop_event()
    supervisor.reactor.set_stop_event(stop_event)
    in_cycle = False
    interrupt_raised = False
    watchdog_handling = False

    def request_stop(signum, frame):
        nonlocal interrupt_raised
        del signum, frame
        stop_event.set()
        if in_cycle and not interrupt_raised and not watchdog_handling:
            # Interrupt the running cycle durably instead of waiting for the
            # whole ReAct budget; the reactor turns this into a recoverable
            # run interruption.
            interrupt_raised = True
            raise WatchdogTimeout()
        logging.getLogger("skynet.cli").info("shutdown requested; finishing current bounded cycle")

    def watchdog_alarm(signum, frame):
        nonlocal watchdog_handling
        del signum, frame
        if watchdog_handling:
            # A second alarm while the first interrupt is still being handled
            # must not escape the loop and kill the process.
            logging.getLogger("skynet.cli").warning("watchdog alarm suppressed; already handling a stale run")
            return
        watchdog_handling = True
        raise WatchdogTimeout()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGALRM, watchdog_alarm)

    try:
        supervisor.start()
    except RecoveryError:
        logging.getLogger("skynet.cli").exception("invalid reboot request")
        supervisor.stop()
        return 1
    try:
        stale_watchdog = StaleRunWatchdog(
            supervisor.reactor,
            interval_seconds=min(30.0, max(5.0, args.interval)),
            timeout_seconds=config.watchdog_timeout_seconds,
        )
        stale_watchdog.start()
        try:
            while not stop_event.is_set():
                try:
                    logging.getLogger("skynet.cli").info("wake cause=%s", wake_cause)
                    health = supervisor.health_check()
                    if not args.once and not stop_event.is_set():
                        supervisor.observe_reboot()
                    checks = cast(dict[str, object], health["checks"])
                    if not health["ok"] and not checks.get("provider_reachable", True):
                        logging.getLogger("skynet.cli").error("provider health failed; skipping ReAct cycle: %s", health)
                        stop_event.wait(min(args.interval, 60.0))
                        wake_cause = "provider-retry"
                        continue
                    interrupt_raised = False
                    in_cycle = True
                    try:
                        status = heartbeat_wake(supervisor.reactor, wake_cause)
                    except WatchdogTimeout:
                        logging.getLogger("skynet.cli").warning("stale run watchdog interrupted the cycle")
                        try:
                            # Block SIGALRM while the interrupt is handled so a
                            # second alarm pends and is then suppressed by the
                            # handler instead of unwinding the whole loop.
                            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
                            try:
                                # A set stop event means SIGTERM/SIGINT, not the
                                # SIGALRM stale-run watchdog: record the clean
                                # owner stop and keep it out of the deviation mix.
                                reason = "owner_stop" if stop_event.is_set() else "watchdog_timeout"
                                status = supervisor.reactor.interrupt_stale_run(reason=reason, force=True)
                            finally:
                                signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM})
                        finally:
                            watchdog_handling = False
                    finally:
                        in_cycle = False
                    if status is not None:
                        cycle_status = status.value
                    else:
                        current_state = supervisor.reactor.store.state()
                        cycle_status = "busy" if current_state.active_run_id else current_state.lifecycle.value
                    logging.getLogger("skynet.cli").info("cycle status=%s", cycle_status)
                    _drain_outbox(supervisor.reactor.store, os.getenv("SKYNET_OUTBOX") or str(Path(config.state_path).parent / "outbox.jsonl"))
                    health = supervisor.health_check()
                    if args.once:
                        return 0
                    if stop_event.is_set():
                        return 0
                    wake_cause = "timer"
                    delay = args.interval
                    next_wake = supervisor.reactor.store.state().next_wake_at
                    if next_wake:
                        delay = max(0.0, (datetime.fromisoformat(next_wake) - datetime.now(UTC)).total_seconds())
                    stop_event.wait(max(1.0, delay))
                except WatchdogTimeout:
                    in_cycle = False
                    watchdog_handling = False
                    logging.getLogger("skynet.cli").warning("watchdog signal outside a cycle; continuing")
                except Exception:
                    in_cycle = False
                    logging.getLogger("skynet.cli").exception("cycle failed; continuing after backoff")
                    try:
                        supervisor.reactor.store.append_event("cycle_error", {"error": "see journal"})
                    except Exception:
                        logging.getLogger("skynet.cli").exception("could not journal cycle_error")
                    stop_event.wait(min(max(args.interval, 5.0), 60.0))
                    wake_cause = "cycle-error"
        finally:
            stale_watchdog.stop()
            stale_watchdog.join(timeout=1)
    finally:
        supervisor.stop()
        for mcp_client in mcp_clients:
            mcp_client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
