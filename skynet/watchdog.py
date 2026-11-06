from __future__ import annotations

import logging
import os
import signal
import threading
import time
from datetime import timedelta

from .reactor import Reactor
from .time import utc_datetime_now

log = logging.getLogger("skynet.watchdog")


class StaleRunWatchdog(threading.Thread):
    """Interrupt the main thread when the active run made no durable progress.

    The main loop blocks inside ``tick()`` while a tool call runs. If that
    tool call hangs (pipe deadlock, stuck provider, browser), the ordinary
    between-tick watchdog can never fire. This daemon thread watches the
    durable ``heartbeat_at`` and, once the run is stale, raises SIGALRM in the
    main thread so the Reactor can interrupt the run durably.
    """

    def __init__(
        self,
        reactor: Reactor,
        *,
        interval_seconds: float = 30.0,
        timeout_seconds: float = 1200.0,
    ) -> None:
        super().__init__(daemon=True, name="skynet-stale-watchdog")
        self.reactor = reactor
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self._stop = threading.Event()
        self._consecutive_stale = 0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._check_once()
            except Exception:
                log.exception("stale-run watchdog check failed")

    def _check_once(self) -> None:
        cutoff = (utc_datetime_now() - timedelta(seconds=self.timeout_seconds)).isoformat().replace("+00:00", "Z")
        active_run_id = self.reactor.store.stale_active_run(cutoff)
        if active_run_id is None:
            # Heartbeat advanced or no active run: reset the stale counter.
            self._consecutive_stale = 0
            return
        # Re-read right before signalling so a just-finished tick (active run
        # already cleared) does not get interrupted while sleeping.
        time.sleep(0.05)
        if self.reactor.store.stale_active_run(cutoff) != active_run_id:
            # The run changed or cleared between reads: reset the counter.
            self._consecutive_stale = 0
            return
        # Same run confirmed stale on this check.
        self._consecutive_stale += 1
        if self._consecutive_stale < 2:
            # First strike: record the stale observation, do not interrupt.
            log.warning(
                "stale run %s detected (strike %d); awaiting confirmation",
                active_run_id,
                self._consecutive_stale,
            )
            return
        # Second consecutive stale check: interrupt.
        log.warning("stale run %s confirmed; interrupting main thread", active_run_id)
        self._consecutive_stale = 0
        try:
            os.kill(os.getpid(), signal.SIGALRM)
        except (OSError, ValueError):
            log.warning("failed to raise SIGALRM for stale run %s", active_run_id, exc_info=True)
