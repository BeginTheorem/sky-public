"""The judge-health watchdog: fires on real degradation, stays quiet otherwise.

It reads the durable ``planner_attempts`` log, escalates through the existing
alert/outbox channel, is rate-limited by the alert dedup window, and must never
raise into the reactor cycle. See ``skynet/judge_health.py``.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from skynet.judge_health import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_THRESHOLD,
    EVENT_KIND,
    check_judge_health,
    planner_success_rate,
    restart_boundary,
)
from skynet.reactor import Reactor, ReactorConfig
from skynet.store import StateStore


class _BrokenStore:
    """A store whose durable read fails; the watchdog must swallow that."""

    @property
    def connection(self) -> Any:
        raise RuntimeError("store unavailable")


class _DummyProvider:
    def complete(self, messages: Any, *, max_tokens: int, tools: Any = ()) -> Any:
        raise AssertionError("the judge-health test must not call the provider")


def _seed(store: StateStore, statuses: list[str]) -> None:
    for generation, status in enumerate(statuses):
        attempt = store.start_planner_attempt(generation, "test")
        store.finish_planner_attempt(attempt, status)


def _alert_count(store: StateStore) -> int:
    return int(store.connection.execute("SELECT COUNT(*) FROM outbox WHERE kind='alert'").fetchone()[0])


def _event_count(store: StateStore) -> int:
    return int(store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind=?", (EVENT_KIND,)).fetchone()[0])


class JudgeHealthTriggerTests(unittest.TestCase):
    def test_below_minimum_sample_does_not_fire(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed(store, ["invalid_response"] * (DEFAULT_MIN_SAMPLES - 1))
            result = check_judge_health(store)
            self.assertFalse(result["degraded"])
            self.assertFalse(result["alerted"])
            self.assertEqual(_alert_count(store), 0)
            store.close()

    def test_degraded_below_threshold_fires(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed(store, ["invalid_response"] * 6)
            result = check_judge_health(store)
            self.assertTrue(result["degraded"])
            self.assertTrue(result["alerted"])
            self.assertEqual(result["success_rate"], 0.0)
            self.assertEqual(result["invalid_response"], 6)
            self.assertEqual(_alert_count(store), 1)
            self.assertEqual(_event_count(store), 1)
            store.close()

    def test_healthy_above_threshold_does_not_fire(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed(store, ["completed"] * 6)
            result = check_judge_health(store)
            self.assertFalse(result["degraded"])
            self.assertEqual(result["success_rate"], 1.0)
            self.assertEqual(_alert_count(store), 0)
            store.close()

    def test_boundary_exactly_at_threshold_is_not_degraded(self) -> None:
        # 3 failed + 3 decided = 0.5, which is not below the 0.5 threshold.
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed(store, ["invalid_response"] * 3 + ["completed"] * 3)
            result = check_judge_health(store)
            self.assertEqual(result["success_rate"], DEFAULT_THRESHOLD)
            self.assertFalse(result["degraded"])
            store.close()

    def test_provider_errors_count_as_failure_and_are_named(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed(store, ["provider_error"] * 6)
            result = check_judge_health(store)
            self.assertTrue(result["degraded"])
            self.assertEqual(result["provider_error"], 6)
            self.assertEqual(result["invalid_response"], 0)
            payload = store.pending_alerts()[0]["payload"]
            self.assertEqual(payload["provider_error"], 6)
            store.close()

    def test_truncated_output_counts_as_failure_and_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed(store, ["output_truncated"] * 6)
            result = check_judge_health(store)
            self.assertTrue(result["degraded"])
            self.assertEqual(result["output_truncated"], 6)
            self.assertEqual(result["invalid_response"], 0)
            store.close()

    def test_legitimate_no_work_is_not_a_failure(self) -> None:
        # A judge that decides "nothing to do" is working; it must not be alerted.
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed(store, ["no_proposals", "no_work", "all_rejected", "all_deduplicated", "completed"])
            result = check_judge_health(store)
            self.assertFalse(result["degraded"])
            self.assertEqual(result["samples"], 5)
            self.assertEqual(result["failed"], 0)
            store.close()


class JudgeHealthRateLimitTests(unittest.TestCase):
    def test_second_call_inside_cooldown_does_not_renotify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed(store, ["invalid_response"] * 6)
            first = check_judge_health(store, cooldown_seconds=3600.0)
            second = check_judge_health(store, cooldown_seconds=3600.0)
            self.assertTrue(first["alerted"])
            self.assertFalse(second["alerted"])
            self.assertEqual(second["occurrences"], 2)
            self.assertEqual(_alert_count(store), 1)
            self.assertEqual(_event_count(store), 1)
            store.close()

    def test_alert_rearms_after_the_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed(store, ["invalid_response"] * 6)
            check_judge_health(store, cooldown_seconds=0.0)
            # Re-arming needs a delivered alert: an undelivered one only counts up.
            pending = store.pending_alerts()
            store.mark_alert_delivered(pending[0]["alert_id"], "telegram")
            rearmed = check_judge_health(store, cooldown_seconds=0.0)
            self.assertTrue(rearmed["alerted"])
            self.assertEqual(_event_count(store), 2)
            store.close()


class JudgeHealthResilienceTests(unittest.TestCase):
    def test_broken_store_never_raises(self) -> None:
        result = check_judge_health(_BrokenStore())
        self.assertFalse(result["degraded"])
        self.assertFalse(result["alerted"])
        self.assertIn("error", result)

    def test_reactor_wiring_surfaces_the_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            _seed(store, ["invalid_response"] * 6)
            store.close()

            reactor = Reactor(_DummyProvider(), {}, ReactorConfig(state_path=path, self_improvement_root=Path(directory)))
            try:
                reactor._check_judge_health()
                self.assertEqual(_alert_count(reactor.store), 1)
            finally:
                reactor.close()

    def test_malformed_env_override_never_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            reactor = Reactor(_DummyProvider(), {}, ReactorConfig(state_path=path, self_improvement_root=Path(directory)))
            try:
                with patch.dict(os.environ, {"SKYNET_JUDGE_HEALTH_THRESHOLD": "not-a-number"}):
                    reactor._check_judge_health()
            finally:
                reactor.close()


class JudgeHealthRestartBoundaryTests(unittest.TestCase):
    def test_restart_boundary_reads_the_guard_started_at(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "reboot-guard.json").write_text(
                json.dumps({"started_at": "2026-09-21T00:00:00.000000Z", "release_id": "x"}),
                encoding="utf-8",
            )
            self.assertEqual(restart_boundary(directory), "2026-09-21T00:00:00.000000Z")

    def test_restart_boundary_is_fail_open(self) -> None:
        self.assertIsNone(restart_boundary(None))
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(restart_boundary(directory), "missing guard")
            (Path(directory) / "reboot-guard.json").write_text("not json", encoding="utf-8")
            self.assertIsNone(restart_boundary(directory), "malformed guard")
            (Path(directory) / "reboot-guard.json").write_text("[1, 2]", encoding="utf-8")
            self.assertIsNone(restart_boundary(directory), "non-dict guard")
            (Path(directory) / "reboot-guard.json").write_text(json.dumps({"started_at": None}), encoding="utf-8")
            self.assertIsNone(restart_boundary(directory), "no usable started_at")

    def test_since_excludes_pre_restart_attempts(self) -> None:
        # The watchdog judges the running code, not a repaired corridor: a
        # window reaching before the restart re-derived the old failure rate.
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed(store, ["invalid_response"] * 6)
            store.connection.execute("UPDATE planner_attempts SET started_at='2000-01-01T00:00:00.000000Z'")
            _seed(store, ["completed"] * 6)
            full = planner_success_rate(store, window=12)
            self.assertEqual(full["success_rate"], 0.5)
            bounded = planner_success_rate(store, window=12, since="2020-01-01T00:00:00.000000Z")
            self.assertEqual(bounded["samples"], 6)
            self.assertEqual(bounded["success_rate"], 1.0)
            store.close()

    def test_empty_post_boundary_window_is_not_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed(store, ["invalid_response"] * 6)
            result = check_judge_health(store, since="2999-01-01T00:00:00.000000Z")
            self.assertEqual(result["samples"], 0)
            self.assertFalse(result["degraded"])
            self.assertFalse(result["alerted"])
            self.assertEqual(_alert_count(store), 0)
            store.close()


if __name__ == "__main__":
    unittest.main()
