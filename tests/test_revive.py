"""The revive tick: start the carrier unit exactly when it is down.

Two halves are pinned here. The decision logic (which states mean "she is down")
is pure and is exercised with a fake systemctl runner; the wiring (the timer that
calls it every 30 minutes, the oneshot that runs it, and ``deploy.sh`` shipping
both) is read out of the tracked templates, because a unit that is never
installed revives nobody.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from skynet import revive


#: The park file is durable production state: ``/stop`` in the owner's bot writes
#: ``state/owner-park.json`` and the tick resolves it from ``SKYNET_STATE`` when a
#: test does not pin it. His real park then made four assertions here fail
#: (measured 2026-10-17, after a maintenance stop wrote the file), while these
#: tests are about the tick's decision -- not about the owner's file. Pin it for
#: the whole module; the park mechanism has its own assertions below.
def setUpModule() -> None:
    os.environ["SKYNET_PARK_FILE"] = str(Path(tempfile.gettempdir()) / "skynet-tests-no-park.json")


def tearDownModule() -> None:
    os.environ.pop("SKYNET_PARK_FILE", None)


REPO_ROOT = Path(__file__).resolve().parent.parent
TIMER = REPO_ROOT / "deploy" / "skynet-revive.timer.in"
SERVICE = REPO_ROOT / "deploy" / "skynet-revive.service.in"
DEPLOY = REPO_ROOT / "scripts" / "deploy.sh"


class _Result:
    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _Recorder:
    """Fake runner: records every argv, answers ``show`` and ``start``."""

    def __init__(self, state: str, start_rc: int = 0) -> None:
        self.state = state
        self.start_rc = start_rc
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> _Result:
        recorded = list(argv)
        self.calls.append(recorded)
        if "show" in recorded:
            return _Result(stdout=F"{self.state}\n")
        if "start" in recorded:
            return _Result(returncode=self.start_rc, stderr="boom" if self.start_rc else "")
        return _Result()

    def verbs(self) -> list[str]:
        """The systemctl verb of every recorded call (``--user`` is a flag)."""
        verbs: list[str] = []
        for recorded in self.calls:
            for argument in recorded[1:]:
                if not argument.startswith("-"):
                    verbs.append(argument)
                    break
        return verbs


class NeedsRevivalTests(unittest.TestCase):
    def test_live_and_transitional_states_are_skipped(self) -> None:
        for state in ("active", "activating", "reloading", "deactivating", "ACTIVE", " active "):
            with self.subTest(state=state):
                self.assertFalse(revive.needs_revival(state))

    def test_down_states_need_revival(self) -> None:
        for state in ("inactive", "failed", "unknown", "", "dead"):
            with self.subTest(state=state):
                self.assertTrue(revive.needs_revival(state))

    def test_unreadable_state_still_revives(self) -> None:
        """A broken probe must not keep her down; a blind start is a no-op."""
        runner = _Recorder(state="")
        revive.revive(runner=runner)
        self.assertIn("start", runner.verbs())


class ReviveTests(unittest.TestCase):
    def test_inactive_carrier_is_started(self) -> None:
        runner = _Recorder(state="inactive")
        self.assertEqual(revive.revive(runner=runner), 0)
        self.assertEqual(runner.verbs(), ["show", "reset-failed", "start"])

    def test_active_carrier_is_left_alone(self) -> None:
        runner = _Recorder(state="active")
        self.assertEqual(revive.revive(runner=runner), 0)
        self.assertEqual(runner.verbs(), ["show"])

    def test_failed_start_is_reported_upward(self) -> None:
        runner = _Recorder(state="failed", start_rc=1)
        self.assertEqual(revive.revive(runner=runner), 1)

    def test_active_state_reads_systemctl_output(self) -> None:
        runner = _Recorder(state="failed")
        self.assertEqual(revive.active_state(runner=runner), "failed")


class UnitWiringTests(unittest.TestCase):
    """The templates and the deploy script must agree on one mechanism."""

    def test_timer_triggers_the_revive_oneshot_every_30_minutes(self) -> None:
        text = TIMER.read_text(encoding="utf-8")
        self.assertIn("Unit=skynet-revive.service", text)
        self.assertIn("OnUnitActiveSec=30min", text)
        self.assertIn("OnBootSec=30min", text)
        self.assertIn("WantedBy=timers.target", text)

    def test_oneshot_runs_the_revive_module_from_the_rendered_root(self) -> None:
        text = SERVICE.read_text(encoding="utf-8")
        self.assertIn("ExecStart=@ROOT@/.venv/bin/python -m skynet.revive", text)
        self.assertIn("@ROOT@", text)
        self.assertNotIn("/home/", text)

    def test_timer_is_tracked_without_an_absolute_home(self) -> None:
        self.assertNotIn("/home/", TIMER.read_text(encoding="utf-8"))

    def test_deploy_ships_and_enables_the_timer(self) -> None:
        text = DEPLOY.read_text(encoding="utf-8")
        self.assertIn("skynet-revive.service skynet-revive.timer", text)
        self.assertIn("enable skynet.service skynet-telegram.service skynet-revive.timer", text)

    def test_carrier_unit_the_revive_oneshot_starts_is_skynet(self) -> None:
        self.assertEqual(revive.CARRIER_UNIT, "skynet.service")


if __name__ == "__main__":
    unittest.main()
