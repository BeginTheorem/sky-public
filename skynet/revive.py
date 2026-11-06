"""Bring the carrier unit back after it died.

The organism is allowed to stop her own unit: a bash step that runs
``systemctl --user stop skynet.service`` kills its own cgroup, so the trailing
``systemctl --user start`` on the same command line never executes, and
``Restart=always`` deliberately does not cover an explicit stop. That is exactly
how the carrier died on 2026-10-15 and stayed down for eight hours.

Owner decision: do **not** forbid the stop - a guard would fight the model
instead of the failure. Resurrect her instead. ``skynet-revive.timer`` calls this
module every 30 minutes; when the carrier is not running, it is started again, so
a death costs at most one tick and never a night.

The brake for a deliberate maintenance stop is the timer itself
(``systemctl --user stop skynet-revive.timer``, or ``disable`` it): while the
timer is stopped nothing is revived.

That brake is not available to the owner from his phone, and on 2026-10-16 it
cost him a night: his `/stop` at 22:07:54 was undone by this same tick at
01:00:30, because a stop by the owner and a death by accident were
indistinguishable here. So the owner's stop now writes a **park** (``set_park``)
and ``revive`` refuses to start a parked carrier; his `/start` clears it
(``clear_park``). The timer keeps ticking while parked -- a cheap no-op that also
means the park cannot be silently forgotten by a timer that was never re-armed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .systemctl import systemctl_command

CARRIER_UNIT = "skynet.service"

# The owner's park, and why it exists. His `/stop` from Telegram stopped the
# carrier on 2026-10-16 at 22:07:54; this timer started it again at 01:00:30,
# because from here a deliberate owner stop and a death look identical. The brake
# for a maintenance stop used to be "stop the timer too" -- a rule only the
# curator can apply, never the owner from his phone. So an owner stop now writes a
# park file and the tick honours it: the carrier stays down until he says
# `/start`, whatever the timer wants. The park carries a reason and a timestamp
# rather than a bare flag, so a later reader can tell who parked it and when.
PARK_FILE = "owner-park.json"


def park_path() -> Path:
    """``state/owner-park.json`` beside the ledger, or its env override."""
    configured = os.getenv("SKYNET_PARK_FILE", "")
    if configured:
        return Path(configured)
    state = os.getenv("SKYNET_STATE", "state/skynet.sqlite3")
    return Path(state).with_name(PARK_FILE)


def park_state(path: Path | None = None) -> dict[str, Any] | None:
    """The park record while the owner has her parked, else None.

    ``path`` lets a caller that already knows the ledger pin the file explicitly (the
    control bot does, so its own state path decides where the park lives); the tick
    resolves it from the environment, and in production both name the same file.
    """
    try:
        value = json.loads((path or park_path()).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(value, dict) or not value.get("parked"):
        return None
    return value


def set_park(reason: str = "owner stop", path: Path | None = None) -> dict[str, Any]:
    """Park the carrier: the revive tick will not start it until this is cleared."""
    record: dict[str, Any] = {"parked": True, "reason": reason, "at": time.time()}
    target = path or park_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def clear_park(path: Path | None = None) -> None:
    """Lift the park (the owner's `/start`); a missing file is already clear."""
    try:
        (path or park_path()).unlink()
    except FileNotFoundError:
        return

# States in which the tick does nothing. ``activating``/``reloading``: the unit is
# already on its way up, so a competing start job must not be queued.
# ``deactivating``: it is on its way *down* (an explicit stop can take up to
# TimeoutStopSec), and the next tick is the right moment to bring it back rather
# than racing the stop.
NO_ACTION_STATES = ("active", "activating", "reloading", "deactivating")

Runner = Callable[..., Any]


def _run(argv: Sequence[str]) -> Any:
    return subprocess.run(list(argv), capture_output=True, text=True, check=False)


def active_state(*, unit: str = CARRIER_UNIT, runner: Runner | None = None) -> str:
    """The unit's ``ActiveState``, lowercased, or ``unknown`` when unreadable."""
    runner = runner or _run
    result = runner(systemctl_command("show", "-p", "ActiveState", "--value", unit))
    state = str(getattr(result, "stdout", "") or "").strip().lower()
    return state or "unknown"


def needs_revival(state: str) -> bool:
    """True unless the carrier is already running or on its way up or down.

    ``unknown`` (systemctl unreadable) counts as *needs revival* on purpose: the
    start that follows is idempotent - a blind tick against a live carrier is a
    no-op, while a skipped tick costs another 30 minutes of downtime.
    """
    return str(state or "").strip().lower() not in NO_ACTION_STATES


def revive(*, unit: str = CARRIER_UNIT, runner: Runner | None = None) -> int:
    """Start the carrier when it is down. Return the start's exit status.

    A parked carrier is not down: it is down **on purpose**, by the owner's own
    word, and the tick's whole job is to respect that until he lifts it.
    """
    runner = runner or _run
    park = park_state()
    if park is not None:
        print(f"revive: parked by the owner ({park.get('reason', 'no reason given')}); leaving {unit} down")
        return 0
    state = active_state(unit=unit, runner=runner)
    if not needs_revival(state):
        print(f"revive: {unit} is {state}; nothing to do")
        return 0
    print(f"revive: {unit} is {state}; starting it")
    # A crash loop can leave the unit in a start-limit lockout, which makes the
    # start fail for up to ``StartLimitIntervalSec``. The tick period is longer
    # than that interval, so this is belt and braces rather than the mechanism -
    # but a resurrected unit must never be blocked by a stale lock.
    runner(systemctl_command("reset-failed", unit))
    result = runner(systemctl_command("start", unit))
    status = int(getattr(result, "returncode", 1) or 0)
    if status != 0:
        stderr = str(getattr(result, "stderr", "") or "").strip()
        print(f"revive: start of {unit} failed (rc={status}){': ' + stderr if stderr else ''}", file=sys.stderr)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``skynet-revive.service`` oneshot."""
    del argv
    # A user unit started by the user manager normally inherits the runtime dir,
    # but that value is what makes ``systemctl --user`` reachable at all, so it is
    # defaulted here as well instead of being templated into the unit file.
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    unit = os.getenv("SKYNET_REVIVE_UNIT", CARRIER_UNIT)
    return revive(unit=unit)


if __name__ == "__main__":  # pragma: no cover - exercised through the unit
    raise SystemExit(main())
