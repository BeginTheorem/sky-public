"""The StartEnvelope must name a scratch directory a run can actually write.

Measured 2026-09-26 (generation 242): /tmp on this host is a 3.9 GB tmpfs and 13
stale per-generation ledger copies (241-289 MB each, 2.9 GB) filled it to 100%.
The episode's own `cp state/skynet.sqlite3` then failed with ENOSPC, one line
after the prompt had pointed at that directory as the place to work, while the
envelope had said nothing about the filesystem it was naming. These tests pin
the facts that make that condition visible before a run stages a file.

Both threshold cases are driven by patching the floor, never by the host's real
free space: `tempfile.TemporaryDirectory()` lives on the same tmpfs as the
scratch directory, so an assertion that a tempdir is "healthy" would pass or
fail with how full the operator's /tmp happens to be (this host was at 316 MB
free, below the 512 MB floor, when this file was written) rather than with the
code under test.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skynet.models import ModelTurn
from skynet.reactor import Reactor, ReactorConfig


class _ProseProvider:
    """Answers in prose, so the episode runs without a valid Finish Report."""

    def complete(self, messages, *, max_tokens, tools=()):
        return ModelTurn(text="I inspected the state but did not produce a report.", usage_tokens=1)


class ScratchHeadroomFactsTests(unittest.TestCase):
    def _reactor(self, directory: str) -> Reactor:
        root = Path(directory)
        return Reactor(
            _ProseProvider(),
            {},
            ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root),
        )

    def _runtime_facts(self, directory: str, scratch: Path) -> dict:
        reactor = self._reactor(directory)
        with patch("skynet.reactor._scratch_root", return_value=scratch):
            reactor.tick("test")
        row = reactor.store.connection.execute(
            "SELECT payload FROM event_log WHERE kind='run_started' ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        reactor.close()
        if row is None:
            self.fail("the run recorded no run_started event")
        observations = {item["kind"]: item for item in json.loads(str(row[0]))["observations"]}
        return observations["runtime_facts"]

    def test_scratch_facts_follow_the_resolved_root_and_report_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "skynet-scratch"
            scratch.mkdir()
            # Floor zero: free bytes are above it on any real filesystem, so the
            # flag must be absent whatever this host's /tmp currently holds.
            with patch("skynet.reactor.SCRATCH_LOW_HEADROOM_BYTES", 0):
                facts = self._runtime_facts(directory, scratch)
            # The fact follows the root that was resolved (the same one the
            # policy boundary authorises), not a hardcoded /tmp path that bash
            # may not even be allowed to touch.
            self.assertEqual(facts["scratch_directory"], str(scratch))
            self.assertGreater(facts["scratch_budget_bytes"], 0)
            self.assertGreater(facts["scratch_headroom_bytes"], 0)
            self.assertLessEqual(facts["scratch_headroom_bytes"], facts["scratch_budget_bytes"])
            # A healthy filesystem keeps exactly the keys the pinned liveness
            # test already asserts: the flag appears only in the low case.
            self.assertNotIn("scratch_low_headroom", facts)

    def test_low_headroom_is_flagged_when_free_space_is_at_the_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "skynet-scratch"
            scratch.mkdir()
            with patch("skynet.reactor.SCRATCH_LOW_HEADROOM_BYTES", 1 << 62):
                facts = self._runtime_facts(directory, scratch)
            self.assertTrue(facts["scratch_low_headroom"])
            self.assertGreaterEqual(facts["scratch_headroom_bytes"], 0)

    def test_an_unusable_scratch_is_not_advertised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory) / "blocker"
            blocker.write_text("a file, not a directory", encoding="utf-8")
            facts = self._runtime_facts(directory, blocker / "skynet-scratch")
            self.assertNotIn("scratch_directory", facts)
            self.assertNotIn("scratch_headroom_bytes", facts)
            self.assertNotIn("scratch_low_headroom", facts)


if __name__ == "__main__":
    unittest.main()
