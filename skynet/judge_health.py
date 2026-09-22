"""Read-only health of the protected planner contract, surfaced to the owner.

The judge's decisions are closed to self-improvement, and that is correct. But
it produced a failure mode nobody anticipated: on 2026-09-20 the planner failed
15 of 18 attempts with ``structured model response must be a JSON object``
because its decoder destroyed a legitimate top-level JSON array, and the empty
portfolio it returned was indistinguishable from "nothing to do". The organism
silently re-created two generic fallback tasks in 69% of runs for a full day.
This module watches the durable attempt log and escalates that degradation
through the existing alert/outbox channel instead of letting it stay silent.

It is deliberately **not** gate-protected: the organism may maintain its own
instrumentation of the judge. It reads ``planner_attempts`` and calls
``store.raise_alert`` only; it never validates a proposal, never selects work
and never owns lifecycle. The Reactor, the sole lifecycle owner, calls it once
per cycle.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("skynet.judge_health")

# Statuses that mean the planner reached a decision (including a legitimate
# "no work"): the instrument decoded the reply and the judge decided.
DECIDED_PLANNER_STATUSES = frozenset({
    "completed",
    "no_proposals",
    "all_deduplicated",
    "all_rejected",
    "no_work",
})
# Statuses that mean the planner did not reach a decision: the reply could not
# be decoded, or the provider could not answer. Either way ``generate()``
# returns ``[]``, which is exactly the masked condition the 2026-09-20 incident
# produced. The payload names the class so a provider outage is not misreported
# as a contract break.
FAILED_PLANNER_STATUSES = frozenset({"invalid_response", "provider_error", "output_truncated"})

# 12 attempts is a small, bounded recent window: enough to see a systemic break
# without remembering a repaired contract forever.
DEFAULT_WINDOW = 12
# Five samples is the floor: one bad reply is noise, five consecutive failures
# is a broken instrument.
DEFAULT_MIN_SAMPLES = 5
# A majority-broken planner is degraded; the incident was 83% broken.
DEFAULT_THRESHOLD = 0.5
# Six hours, matching the alert dedup default: a degraded judge is re-surfaced
# at most a few times a day, never once per cycle.
DEFAULT_COOLDOWN_SECONDS = 21_600.0

ALERT_KIND = "planner_judge_degraded"
ALERT_DEDUP_KEY = "planner-judge-degraded"
EVENT_KIND = "judge_health_degraded"
# The durable post-restart boundary that RebootGuard.begin writes at startup.
GUARD_FILENAME = "reboot-guard.json"


def restart_boundary(state_dir: Path | str | None) -> str | None:
    """Timestamp of the process's own restart boundary, or ``None``.

    The watchdog exists to notice a planner that is broken *now*. A window that
    reaches back before the last restart derives its verdict from attempts made
    by code that is no longer running: on 2026-09-20 the promoted contract fix
    could not clear the alert because the nine failures that triggered it
    (``event_log`` seq 6477, generations 14..43) were still the newest rows in
    ``planner_attempts`` afterwards, so every cycle re-derived rate 0.182.

    ``RebootGuard.begin`` writes ``started_at`` at startup, which makes the
    guard file under the state directory the one durable restart boundary the
    organism already owns. This reads it and returns its ``started_at``.

    Any failure to read or parse it returns ``None``, which means "no boundary,
    use the whole window". That is deliberately fail-open: an unreadable guard
    must never silence a genuinely degraded planner, and it must never raise
    into the reactor cycle. The boundary only ever *excludes* older rows; it can
    never invent samples, so it cannot make a healthy planner look degraded.
    """
    if state_dir is None:
        return None
    try:
        guard = json.loads((Path(state_dir) / GUARD_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(guard, dict):
        return None
    started_at = guard.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        return None
    return started_at


def planner_success_rate(store: Any, *, window: int = DEFAULT_WINDOW, since: str | None = None) -> dict[str, Any]:
    """Decided-vs-failed planner attempts over the last ``window`` finished rows.

    Only finished attempts count: a row still ``started`` is in flight and says
    nothing about the contract. The status vocabulary is read from the durable
    ``planner_attempts`` table, so no new column or migration is needed.

    ``since`` excludes attempts that *started* before a restart boundary, so the
    verdict describes the running code rather than a corridor that has already
    been repaired. The filter is applied in SQL on ``started_at`` and the
    ``LIMIT`` still bounds the scan, so a long run of pre-boundary rows cannot
    push the window open. An empty post-boundary window reports ``samples`` 0,
    which ``check_judge_health`` treats as "not enough evidence" and never as
    degradation. ``since`` is compared as the stored ISO-8601 string, which is
    valid because every writer stores a UTC ``...Z`` timestamp of one fixed
    shape with microsecond precision: same length, same field widths, so
    lexicographic order is chronological order.
    """
    bounded_window = max(1, int(window))
    if since:
        rows = store.connection.execute(
            "SELECT status FROM planner_attempts WHERE finished_at IS NOT NULL "
            "AND started_at IS NOT NULL AND started_at >= ? "
            "ORDER BY finished_at DESC LIMIT ?",
            (since, bounded_window),
        ).fetchall()
    else:
        rows = store.connection.execute(
            "SELECT status FROM planner_attempts WHERE finished_at IS NOT NULL "
            "ORDER BY finished_at DESC LIMIT ?",
            (bounded_window,),
        ).fetchall()
    statuses = [str(row["status"]) for row in rows]
    decided = sum(1 for status in statuses if status in DECIDED_PLANNER_STATUSES)
    invalid_response = statuses.count("invalid_response")
    provider_error = statuses.count("provider_error")
    output_truncated = statuses.count("output_truncated")
    failed = sum(1 for status in statuses if status in FAILED_PLANNER_STATUSES)
    samples = decided + failed
    return {
        "window": bounded_window,
        "samples": samples,
        "decided": decided,
        "failed": failed,
        "invalid_response": invalid_response,
        "provider_error": provider_error,
        "output_truncated": output_truncated,
        "success_rate": (decided / samples) if samples else 1.0,
    }


def check_judge_health(
    store: Any,
    *,
    window: int = DEFAULT_WINDOW,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    threshold: float = DEFAULT_THRESHOLD,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    since: str | None = None,
) -> dict[str, Any]:
    """Escalate a degraded planner to the owner; never raise.

    A call below ``min_samples`` or at/above ``threshold`` is a no-op, so a
    single bad reply cannot fire it. A degraded call goes through
    ``store.raise_alert``, whose durable dedup window is the rate limit: a
    second call inside ``cooldown_seconds`` only bumps the occurrence counter
    and does not re-notify. The whole body is guarded so a fault in the watchdog
    cannot crash the reactor cycle.

    ``since`` is an optional restart boundary forwarded to
    ``planner_success_rate``: attempts that started before it belong to code
    that is no longer running and cannot testify about the contract now.
    """
    try:
        observation = planner_success_rate(store, window=window, since=since)
        degraded = observation["samples"] >= max(1, int(min_samples)) and observation["success_rate"] < float(threshold)
        alerted = False
        occurrences = 0
        if degraded:
            detail = {
                "success_rate": round(float(observation["success_rate"]), 3),
                "samples": observation["samples"],
                "decided": observation["decided"],
                "failed": observation["failed"],
                "invalid_response": observation["invalid_response"],
                "provider_error": observation["provider_error"],
                "output_truncated": observation["output_truncated"],
                "window": observation["window"],
            }
            result = store.raise_alert(
                ALERT_KIND,
                {
                    **detail,
                    "note": (
                        "The planner is not reaching a decision. Its decisions stay closed, "
                        "but the instrument (skynet/planner_contract.py) is editable; a "
                        "provider_error-only window is the provider chain, not the contract."
                    ),
                },
                severity="warning",
                dedup_key=ALERT_DEDUP_KEY,
                dedup_window_seconds=max(0.0, float(cooldown_seconds)),
            )
            alerted = bool(result.get("new"))
            occurrences = int(result.get("occurrences", 0))
            if alerted:
                store.append_event(EVENT_KIND, detail)
        return {**observation, "degraded": degraded, "alerted": alerted, "occurrences": occurrences}
    except Exception as exc:
        log.warning("judge health check failed: %s", exc, exc_info=True)
        return {"degraded": False, "alerted": False, "error": str(exc)}
