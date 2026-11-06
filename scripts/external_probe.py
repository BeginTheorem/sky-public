#!/usr/bin/env python3
"""External host probe: is the carrier alive, and does the owner hear about it if not.

It runs from its own systemd timer, outside the organism's process tree, because an alert
raised inside a process cannot survive that process. It keeps no state file of its own:
the log it appends to *is* the state, so consecutive failures are read back from the tail
and one outage produces one alert instead of one per tick.

The unit sets SKYNET_TELEGRAM_BOT_TOKEN and SKYNET_TELEGRAM_ALLOWED_CHAT_ID; optional are
SKYNET_TELEGRAM_PROXY, SKYNET_EXTERNAL_PROBE_LOG and SKYNET_PROBE_ALERT_AFTER. No secret
is written to disk: the token reaches curl through its config on stdin.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

SERVICE = "skynet.service"
ALERT_AFTER = 2


def parked_reason() -> str:
    """The owner's park, when he has her parked: a stopped carrier is not an outage.

    The control bot's ``/stop`` writes this file and ``/start`` removes it; the revive
    tick honours the same record. Without this check a deliberate park would page him
    about an outage he ordered himself, which is exactly the false alarm that makes an
    alert channel worthless.
    """
    override = os.environ.get("SKYNET_PARK_FILE")
    state = os.environ.get("SKYNET_STATE", "state/skynet.sqlite3")
    path = Path(override) if override else Path(state).with_name("owner-park.json")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return ""
    if not isinstance(record, dict) or not record.get("parked"):
        return ""
    return str(record.get("reason") or "parked by the owner")


def log_path() -> Path:
    """Probe history: next to the tree that owns this script, unless an override says otherwise.

    Derived, never written out — the release scanner rejects a shipped file that names a
    real home directory.
    """
    override = os.environ.get("SKYNET_EXTERNAL_PROBE_LOG")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "state" / "external_probe.log"


def alert_after() -> int:
    """How many failed checks in a row justify waking the owner."""
    try:
        return max(1, int(os.environ.get("SKYNET_PROBE_ALERT_AFTER", str(ALERT_AFTER))))
    except ValueError:
        return ALERT_AFTER


def service_status(service: str = SERVICE) -> str:
    """``active`` / ``inactive`` / whatever systemd prints; ``unknown`` when it will not answer."""
    runtime = os.environ.get("XDG_RUNTIME_DIR", "/run/user/" + str(os.getuid()))
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", service],
            capture_output=True,
            text=True,
            env={**os.environ, "XDG_RUNTIME_DIR": runtime},
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() or "inactive"


def consecutive_failures(lines: Iterable[str]) -> int:
    """Failed checks at the tail of the log: the log is the only state this probe keeps."""
    count = 0
    for line in reversed(list(lines)):
        if "| service_status=" not in line:
            continue
        status = line.split("| service_status=", 1)[1].split("|", 1)[0].strip()
        if status == "active":
            break
        count += 1
    return count


def append_log(path: Path, status: str, host: str, outcome: str) -> None:
    entry = f"{datetime.now().isoformat()} | service_status={status} | host={host} | message={outcome}\n"
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
    except OSError as error:
        print(f"WARNING: Failed to write log: {error}", file=sys.stderr)


def send_alert(message: str, token: str, chat_id: str) -> None:
    """One Telegram message. The token travels on stdin, so nothing lands in a shared directory."""
    config = f'url = "https://api.telegram.org/bot{token}/sendMessage"\ndata-urlencode = "chat_id={chat_id}"\n'
    command = ["curl", "-sS", "--max-time", "10", "-o", "/dev/null", "-K", "-"]
    proxy = os.environ.get("SKYNET_TELEGRAM_PROXY")
    if proxy:
        command.extend(["--proxy", proxy])
    command.extend(["--data-urlencode", f"text={message}"])
    subprocess.run(command, input=config, capture_output=True, text=True, check=False)


def main() -> int:
    token = os.environ.get("SKYNET_TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("SKYNET_TELEGRAM_ALLOWED_CHAT_ID", "") or os.environ.get("SKYNET_TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("ERROR: SKYNET_TELEGRAM_BOT_TOKEN and SKYNET_TELEGRAM_ALLOWED_CHAT_ID must be set", file=sys.stderr)
        return 1

    path = log_path()
    host = socket.gethostname() or "unknown"
    status = service_status()
    park = parked_reason()
    if park:
        # The owner parked her: the log records why the probe went quiet, and no alert
        # is sent for a state he asked for.
        append_log(path, status, host, f"parked: {park}")
        print(f"SkyNet is parked by the owner ({park}); not an outage")
        return 0
    failures = consecutive_failures(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0

    if status == "active":
        append_log(path, status, host, "service_active")
        print("SkyNet service is active")
        return 0

    failures += 1
    threshold = alert_after()
    if failures == threshold:
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        message = (
            f"SkyNet host probe: {SERVICE} is unreachable.\n"
            f"checks in a row: {failures}\n"
            f"host: {host}\n"
            f"time: {stamp}"
        )
        send_alert(message, token, chat_id)
        append_log(path, status, host, "alert_sent")
        print("Alert sent (or attempted) for unreachable SkyNet service")
        return 0

    append_log(path, status, host, "alert_pending")
    print(f"SkyNet service is {status}: {failures} check(s) in a row, alert at {threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
