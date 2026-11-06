from __future__ import annotations

import os
import signal
from unittest.mock import MagicMock, patch

from skynet.watchdog import StaleRunWatchdog


def test_single_stale_check_does_not_interrupt() -> None:
    reactor = MagicMock()
    watchdog = StaleRunWatchdog(reactor, interval_seconds=1, timeout_seconds=5)
    with patch("skynet.watchdog.time.sleep"), patch("skynet.watchdog.os.kill") as kill:
        reactor.store.stale_active_run.return_value = "run-a"
        watchdog._check_once()
    kill.assert_not_called()
    assert watchdog._consecutive_stale == 1


def test_two_consecutive_stale_checks_interrupt() -> None:
    reactor = MagicMock()
    watchdog = StaleRunWatchdog(reactor, interval_seconds=1, timeout_seconds=5)
    with patch("skynet.watchdog.time.sleep"), patch("skynet.watchdog.os.kill") as kill:
        reactor.store.stale_active_run.return_value = "run-a"
        watchdog._check_once()
        kill.assert_not_called()
        watchdog._check_once()
    kill.assert_called_once_with(os.getpid(), signal.SIGALRM)
    assert watchdog._consecutive_stale == 0


def test_heartbeat_advance_resets_stale_count() -> None:
    reactor = MagicMock()
    watchdog = StaleRunWatchdog(reactor, interval_seconds=1, timeout_seconds=5)
    with patch("skynet.watchdog.time.sleep"), patch("skynet.watchdog.os.kill") as kill:
        reactor.store.stale_active_run.return_value = "run-a"
        watchdog._check_once()
        assert watchdog._consecutive_stale == 1
        kill.assert_not_called()
        # Heartbeat advanced: stale_active_run returns None -> counter resets.
        reactor.store.stale_active_run.return_value = None
        watchdog._check_once()
        assert watchdog._consecutive_stale == 0
        kill.assert_not_called()
        # A single stale after reset does not interrupt.
        reactor.store.stale_active_run.return_value = "run-a"
        watchdog._check_once()
        assert watchdog._consecutive_stale == 1
        kill.assert_not_called()
