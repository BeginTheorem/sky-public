"""Git's optional locks, and the debris a killed index refresh leaves behind.

A read-only ``git status`` refreshes the index as an *optional* side effect, and that
refresh takes ``.git/index.lock``. Kill the process at the wrong moment — a subprocess
timeout, a stopped unit, a SIGKILL from the step budget — and a 0-byte ``index.lock``
stays behind that refuses every later git *write*: no commit, no merge, no publication.

Measured on 2026-10-28: the five-second ``git status`` in ``Reactor._worktree_state``
was killed mid-refresh and left such a lock at 22:18:56. It blocked durable publication
for five hours (17 blocked cycles against 12 completed), and it could not be cleared
from inside the organism at all: ``.git/`` is a hard-denied path for bash, and the sudo
gate the refusal points to cannot work on this box — the unit sets
``NoNewPrivileges=yes``, so sudo refuses every elevation at the kernel level. The owner
approved the removal three times and the gate reported ``sudo_ran`` with returncode 1
each time, which the organism then explained to him as a container misconfiguration.

Two rules keep that class of failure out:

* every read-only git call runs with ``GIT_OPTIONAL_LOCKS=0``, so it never takes the
  index lock at all;
* a *stale* lock is swept by the organism's own code — it owns its debris, and needing
  root (or an escape from the read-only workspace policy) to clear it is a design bug,
  not a safeguard.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

# Read-only git gets a generous budget: the point of the timeout is to bound a hang, not
# to race a large worktree. Five seconds is what killed the refresh that started this.
READ_ONLY_GIT_TIMEOUT_SECONDS = 30.0

# A live writer's lock is also 0 bytes, so age is half the test: nothing legitimate holds
# one for this long, and the process check below closes the other half.
STALE_LOCK_AGE_SECONDS = 300.0
_LOCK_NAME = "index.lock"

# Set on every read-only call. git documents it as "complete the operation without
# performing optional sub-operations that require taking a lock", and the index refresh
# behind status/diff is exactly such a sub-operation.
READ_ONLY_GIT_ENV: dict[str, str] = {"GIT_OPTIONAL_LOCKS": "0"}


def git_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for a git call that must not take optional locks."""
    env = dict(os.environ if base is None else base)
    env.update(READ_ONLY_GIT_ENV)
    return env


def install_git_read_only_environment() -> None:
    """Make "no optional locks" the process-wide default, and say so out loud.

    Per-call environments cover the call sites this module knows about; this covers the
    ones it does not, including every git invocation a future edit adds. It is deliberately
    a process setting rather than an import-time side effect: a hidden global that changes
    how subprocesses behave is exactly the kind of magic that costs an afternoon later.
    Write operations are unaffected — required locks are still taken.
    """
    os.environ.update(READ_ONLY_GIT_ENV)


def read_only_git(
    args: Sequence[str],
    cwd: str | Path,
    *,
    timeout: float = READ_ONLY_GIT_TIMEOUT_SECONDS,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run git read-only: optional locks off, output captured, failures returned not raised.

    ``GIT_OPTIONAL_LOCKS=0`` is the whole point of this wrapper. A killed ``status`` used to
    leave a lock behind that outlived the call that made it, so the fix belongs at the call
    site, not in a cleanup that runs after the damage.
    """
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=git_env(env),
    )


def git_dir(root: str | Path) -> Path | None:
    """The repository's git directory, following the ``gitdir:`` file of a linked worktree."""
    dot_git = Path(root) / ".git"
    if dot_git.is_dir():
        return dot_git
    if not dot_git.is_file():
        return None
    try:
        text = dot_git.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    candidate = Path(text.split(":", 1)[1].strip())
    return candidate if candidate.is_absolute() else (dot_git.parent / candidate)


def git_processes_running() -> bool:
    """Whether any git process is alive on this box — the holder half of the staleness test.

    ``/proc`` is read directly rather than shelling out to ps/pgrep: the sweep must work
    from inside the harness, where the sandbox denies nothing but never guarantees a
    process listing. A machine without /proc (or a process that exits mid-scan) reads as
    "no git running", which is the safe direction only because age is tested as well.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        return False
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        first = raw.split(b"\0", 1)[0]
        if first.rsplit(b"/", 1)[-1] == b"git":
            return True
    return False


def clear_stale_index_lock(
    root: str | Path,
    *,
    max_age_seconds: float = STALE_LOCK_AGE_SECONDS,
    now: float | None = None,
    git_running: Callable[[], bool] | None = None,
) -> Path | None:
    """Remove a 0-byte ``.git/index.lock`` that no live git process can own.

    Returns the removed path, or ``None`` when there was nothing to do. A lock is stale only
    when all three hold: it is empty (git writes the payload elsewhere and only then commits
    the file), it is older than ``max_age_seconds``, and no git process is alive to own it.
    Anything else is left alone — this must never interrupt a writer it cannot see.
    """
    directory = git_dir(root)
    if directory is None:
        return None
    lock = directory / _LOCK_NAME
    try:
        info = lock.stat()
    except OSError:
        return None
    if info.st_size != 0:
        return None
    if (time.time() if now is None else now) - info.st_mtime < max_age_seconds:
        return None
    if (git_running or git_processes_running)():
        return None
    try:
        lock.unlink()
    except OSError:
        return None
    return lock
