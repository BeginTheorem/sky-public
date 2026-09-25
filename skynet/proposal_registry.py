"""Resolve self-improvement proposals the environment blocked for good.

The retry rule lives in ``skynet/self_improvement.py``: an environment failure
does not burn the change fingerprint, so the identical patch may be attempted
again once the toolchain is repaired, up to a bounded ceiling. The ceiling
exists so a persistent environment fault cannot loop, but nothing wrote the
outcome back: a record past its ceiling stayed ``blocked_by_environment``
forever. That status is terminal (it is in ``TERMINAL_PROPOSAL_STATUSES``, so
its worktree is reclaimed), and the retry guard reads it as still retryable, so
the entry was simultaneously retired and reported as pending repair -- a
contradiction no sweep resolved, because every other sweep looks for
``validated`` or ``awaiting_reboot``.

This module closes that hole. Once the retry budget is spent the environment
fault is no longer a pending condition, and the registry's own integrity rule
decides the record: a ``failure_class`` outside ``BLOCKED_FAILURE_CLASSES``
must be recorded as ``rejected``. The decision is therefore not a new policy,
only the missing write of a decision the guard already made.

The guard's unit is the change fingerprint, not the record. It walks every
record and a single spent one makes the whole fingerprint permanently refused
-- yet each attempt is a separate record, so a fingerprint retried to the
ceiling leaves several records behind and only the last one carries the spent
counter. The same is true of a fingerprint with any terminal sibling: the guard
refuses it because the change was already rejected, validated, awaiting reboot
or accepted. Deciding per record therefore left the earlier siblings
``blocked_by_environment`` forever with a guard that already refused their
fingerprint: the same terminal-status dead end, reached through the sibling
records of one change. The fingerprint is refused as a unit, so the group is
closed together.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .time import utc_now

BLOCKED_STATUS = "blocked_by_environment"
ENVIRONMENT_ATTEMPTS_KEY = "environment_attempts"
# Deliberately outside BLOCKED_FAILURE_CLASSES: the fault is no longer pending,
# so check_registry_integrity requires the record to read as rejected.
EXHAUSTED_FAILURE_CLASS = "environment_retries_exhausted"
# The retry guard treats a missing counter as one spent attempt; the sweep must
# agree, or a legacy record would be read as retryable by one and spent by the
# other.
ASSUMED_ATTEMPTS_WHEN_ABSENT = 1


def _retry_ceiling() -> int:
    """Read the retry ceiling from its owner without a module-level import.

    ``skynet.self_improvement`` imports this module, so importing the constant at
    module scope is a cycle. The ceiling must still have exactly one source of
    truth: a sweep that disagreed with the retry guard would either close a
    record the guard still retries or leave a spent one blocked forever.
    """
    from .self_improvement import MAX_ENVIRONMENT_ATTEMPTS

    return MAX_ENVIRONMENT_ATTEMPTS


def _spent_attempts(record: dict[str, Any]) -> int:
    value = record.get(ENVIRONMENT_ATTEMPTS_KEY)
    if isinstance(value, bool) or not isinstance(value, int):
        return ASSUMED_ATTEMPTS_WHEN_ABSENT
    return max(value, 0)


def _fingerprint(record: dict[str, Any]) -> str:
    """The change identity the retry guard refuses as a unit, or "" if absent."""
    value = record.get("change_fingerprint")
    return value if isinstance(value, str) and value else ""


# The retry guard in ``self_improvement.py`` refuses a change fingerprint once
# ANY sibling record is terminal, not only when the environment budget is spent.
# A blocked record whose fingerprint has a terminal sibling can therefore never
# be retried, and this sweep must close it too. This mirrors the guard's own set
# (``self_improvement.py:1331``); it cannot be imported because that module
# imports this one.
TERMINAL_SIBLING_STATUSES = frozenset({"rejected", "validated", "awaiting_reboot", "accepted"})

# Statuses whose record claims the promoted commit is the code now running.
# Only these are checked for liveness; a rejected or environment-blocked record
# never claimed a commit, so a missing one is not a violation for it.
LIVE_CLAIMING_STATUSES = frozenset({"accepted", "awaiting_reboot", "promoting"})


def audit_promotion_liveness(
    proposals: dict[str, Any],
    in_history: Callable[[str], bool | None],
) -> list[dict[str, Any]]:
    """Report promotions that claim to be live while their commit is not.

    ``reconcile_awaiting_reboot`` only ever visits ``awaiting_reboot`` and
    ``promoting`` records, so once a record reads ``accepted`` nothing revisits
    it. An external rollback (``request_rollback``, or the startup script) resets
    HEAD backwards past every later promotion, and the records whose commits
    then left HEAD keep claiming ``accepted`` with no recorded rollback reason:
    a promotion that is neither live nor explicitly rolled back.

    Pure and read-only on purpose. A naive repair -- mark every non-ancestor
    record rolled back -- has a false-positive path of its own when unrelated
    history is dropped, so this function only names the contradiction and leaves
    the decision to a caller that can judge it. ``in_history`` returns ``None``
    when ancestry cannot be decided (unreadable registry entry, absent commit,
    failed git call); an undecidable row is never reported as a violation.
    """
    violations: list[dict[str, Any]] = []
    for proposal_id, record in proposals.items():
        if not isinstance(record, dict):
            continue
        status = str(record.get("status", ""))
        if status not in LIVE_CLAIMING_STATUSES:
            continue
        commit = str(record.get("promoted_commit") or record.get("commit") or "")
        if not commit:
            # Nothing was ever claimed, so there is no contradiction to report.
            continue
        ancestor = in_history(commit)
        if ancestor is False:
            violations.append(
                {
                    "proposal_id": proposal_id,
                    "status": status,
                    "commit": commit,
                    "reason": "record claims the promotion is live but the commit is not an ancestor of HEAD",
                }
            )
    return violations


def audit_promotion_registry(proposals_path: Path | str, root: Path | str) -> list[dict[str, Any]]:
    """Read the registry and report promotions that are neither live nor rolled back.

    Best-effort: an unreadable registry or a missing git worktree yields no
    violations rather than an exception, because this runs in startup
    housekeeping beside the reconcile sweep and must never block a start.
    """
    path = Path(proposals_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []

    def in_history(commit: str) -> bool | None:
        try:
            check = subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                cwd=str(root), capture_output=True, text=True, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if check.returncode == 0:
            return True
        if check.returncode == 1:
            return False
        # 128 and friends mean git could not answer at all (unknown commit,
        # no repository); that is undecidable, not a violation.
        return None

    return audit_promotion_liveness(data, in_history)


def _spent_fingerprints(proposals: dict[str, Any], ceiling: int) -> set[str]:
    """Fingerprints the retry guard refuses as a unit.

    A fingerprint is spent when any record carries it and is either terminal
    (the guard refuses it outright) or environment-blocked with its own attempt
    budget exhausted. Every other blocked record of that fingerprint is
    unretryable too, so the sweep closes the whole group.
    """
    spent: set[str] = set()
    for record in proposals.values():
        if not isinstance(record, dict):
            continue
        fingerprint = _fingerprint(record)
        if not fingerprint:
            continue
        status = record.get("status")
        terminal = status in TERMINAL_SIBLING_STATUSES
        spent_budget = status == BLOCKED_STATUS and _spent_attempts(record) >= ceiling
        if terminal or spent_budget:
            spent.add(fingerprint)
    return spent


def resolve_exhausted_environment_blocks(
    proposals: dict[str, Any],
    *,
    max_attempts: int | None = None,
    timestamp: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reject environment-blocked records whose retry budget is spent.

    Pure: the caller owns persistence. A record still inside its budget is left
    untouched because it is genuinely retryable, and rewriting it would destroy
    the retryability the guard depends on.
    """
    ceiling = _retry_ceiling() if max_attempts is None else max_attempts
    stamp = timestamp or utc_now()
    spent_groups = _spent_fingerprints(proposals, ceiling)
    resolved: list[dict[str, Any]] = []
    for proposal_id, record in proposals.items():
        if not isinstance(record, dict) or record.get("status") != BLOCKED_STATUS:
            continue
        attempts = _spent_attempts(record)
        fingerprint = _fingerprint(record)
        spent_here = attempts >= ceiling
        if not spent_here and (not fingerprint or fingerprint not in spent_groups):
            continue
        previous_class = str(record.get("failure_class") or "unknown")
        reason = (
            "environment retry budget exhausted"
            if spent_here
            else "sibling record of the same change fingerprint is already terminal"
        )
        record["status"] = "rejected"
        record["failure_class"] = EXHAUSTED_FAILURE_CLASS
        record["failure"] = (
            f"environment-blocked after {attempts} of {ceiling} attempts ({previous_class}); "
            f"{reason}, so the guard refuses the identical patch and the record is closed"
        )
        record["resolution"] = {
            "reason": reason,
            "previous_failure_class": previous_class,
            "environment_attempts": attempts,
            "resolved_at": stamp,
        }
        resolved.append(
            {
                "proposal_id": proposal_id,
                "status": "rejected",
                "failure_class": EXHAUSTED_FAILURE_CLASS,
                "previous_failure_class": previous_class,
                "environment_attempts": attempts,
                "spent_by_own_attempts": spent_here,
            }
        )
    return proposals, resolved


def sweep_environment_blocks(
    proposals_path: Path | str,
    *,
    max_attempts: int | None = None,
    timestamp: str | None = None,
) -> list[dict[str, Any]]:
    """Read the registry, close its spent environment blocks, write it back."""
    path = Path(proposals_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # A corrupt registry is quarantined by its owner, not overwritten here.
        return []
    if not isinstance(data, dict):
        return []
    # Preserve every top-level record, including any non-dict value an external
    # writer left behind. Filtering them out here discarded them on the rewrite;
    # the sweep only ever edits dict records and must not silently drop data it
    # does not understand.
    proposals = dict(data)
    proposals, resolved = resolve_exhausted_environment_blocks(
        proposals, max_attempts=max_attempts, timestamp=timestamp
    )
    if not resolved:
        return []
    _atomic_write(path, proposals)
    return resolved


def _atomic_write(path: Path, proposals: dict[str, Any]) -> None:
    """Replace the registry in one step so a crash cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(proposals, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
