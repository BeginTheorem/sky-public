"""The append-only memory audit: metadata only, chained, never rewritten."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from helpers import FakeProvider, FixtureTool

from skynet.memory_audit import (
    ANCHOR_NAME,
    AUDITED_COLUMNS,
    CHECKPOINT_NAME,
    GENESIS,
    LOG_NAME,
    _digest_text,
    checkpoint_memory_audit,
    memory_audit_continuity_report,
    memory_digest,
    verify_chain,
)
from skynet.models import RunStatus
from skynet.reactor import Reactor, ReactorConfig
from skynet.store import StateStore
from skynet.supervisor import Supervisor


class MemoryAuditTests(unittest.TestCase):
    def _store(self, directory: str) -> StateStore:
        return StateStore(Path(directory) / "state.sqlite3")

    def _seed(self, store: StateStore, content: str, *, source_run: str = "seed-run") -> str:
        store.consolidate(source_run, [{"kind": "fact", "content": content, "confidence": 0.8}])
        row = store.connection.execute("SELECT memory_id FROM memories WHERE content=?", (content,)).fetchone()
        self.assertIsNotNone(row)
        return str(row["memory_id"])

    def test_checkpoint_records_metadata_and_never_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            private_note = "a sentence that must not reach the log"
            self._seed(store, private_note)
            result = checkpoint_memory_audit(store.connection, directory, reason="episode_end", run_id="run-1")
            self.assertEqual(result["status"], "recorded")
            text = (Path(directory) / LOG_NAME).read_text(encoding="utf-8")
            self.assertNotIn(private_note, text)
            record = json.loads(text.strip().splitlines()[0])
            self.assertEqual(record["rows"], 1)
            self.assertEqual(record["reason"], "episode_end")
            self.assertEqual(record["previous"], GENESIS)
            self.assertIn("bytes", record)
            self.assertGreater(record["bytes"], 0)
            store.close()

    def test_chain_links_every_line_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            self._seed(store, "first")
            first = checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            self._seed(store, "second")
            second = checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            self.assertEqual((first["sequence"], second["sequence"]), (1, 2))
            self.assertTrue(verify_chain(directory)["ok"])
            lines = (Path(directory) / LOG_NAME).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertNotEqual(json.loads(lines[0])["previous"], json.loads(lines[1])["previous"])
            store.close()

    def test_rewritten_last_line_is_caught_by_the_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            self._seed(store, "first")
            checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            self._seed(store, "second")
            checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            path = Path(directory) / LOG_NAME
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            tampered = json.loads(lines[-1])
            tampered["rows"] = 99
            lines[-1] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = verify_chain(directory)
            self.assertFalse(report["ok"])
            # Nothing follows the edited line, so only the checkpoint's hash sees it.
            self.assertEqual(report["broken_at"], 2)
            store.close()

    def test_mid_log_rewrite_breaks_the_chain_at_the_next_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            self._seed(store, "first")
            checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            self._seed(store, "second")
            checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            path = Path(directory) / LOG_NAME
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            tampered = json.loads(lines[0])
            tampered["rows"] = 99
            lines[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = verify_chain(directory)
            self.assertFalse(report["ok"])
            self.assertEqual(report["broken_at"], 2)
            # The append path keeps working and records the break in the log.
            appended = checkpoint_memory_audit(store.connection, directory, reason="startup")
            self.assertEqual(appended["status"], "recorded")
            self.assertEqual(appended["chain_broken_at"], 2)
            store.close()

    def test_removing_the_newest_line_is_reported_not_silently_accepted(self) -> None:
        """A chain walk alone cannot see a committed line being deleted.

        Dropping the tail leaves a shorter chain that agrees with itself, and a
        chain walk only compares a line with its predecessor, so it verified the
        truncated log as clean. Worse, the next append re-anchored ``previous``
        on the surviving prefix and erased the evidence permanently. The
        checkpoint's sequence is the commitment that catches it (arXiv
        cs/0302010); this test pins the detection.
        """
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            for index in range(3):
                self._seed(store, f"seed {index}")
                checkpoint_memory_audit(store.connection, directory, reason=f"episode_{index}")
            path = Path(directory) / LOG_NAME
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            report = verify_chain(directory)
            self.assertFalse(report["ok"])
            self.assertEqual(report["broken_at"], 3)
            # The next append must carry the break forward instead of re-anchoring
            # the chain and hiding the removal.
            appended = checkpoint_memory_audit(store.connection, directory, reason="next_episode")
            self.assertEqual(appended["status"], "recorded")
            self.assertEqual(appended["chain_broken_at"], 3)
            store.close()

    def test_deleting_the_whole_log_is_reported_when_a_checkpoint_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            self._seed(store, "seed")
            checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            (Path(directory) / LOG_NAME).unlink()
            report = verify_chain(directory)
            self.assertFalse(report["ok"])
            self.assertEqual(report["broken_at"], 1)
            store.close()

    def test_a_log_without_a_checkpoint_is_still_clean(self) -> None:
        """The new check must not fire on a fresh state directory."""
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(verify_chain(directory)["ok"])

    def test_continuity_report_is_silent_when_nothing_changed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            self._seed(store, "stable")
            checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            report = memory_audit_continuity_report(store.connection, directory)
            self.assertEqual(report["status"], "unchanged")
            store.close()

    def test_continuity_report_sees_an_out_of_band_insert_and_names_the_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            # A consolidate caller that passes a synthetic id (``handoff-N`` does
            # exactly this) lands in ``pseudo_run_rows``: attributed, but to a run
            # that never existed. The two counts exist to tell that apart from a
            # row with no attribution at all.
            self._seed(store, "in band", source_run="handoff-0")
            checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            # Exactly what an outside writer does: a direct INSERT with no run.
            store.connection.execute(
                "INSERT INTO memories(memory_id, kind, content, confidence, updated_at, pinned, status, valid_from) "
                "VALUES ('deadbeefdeadbeefdeadbeefdeadbeef', 'fact', 'injected from outside', 0.9, "
                "'2026-09-22T00:00:00Z', 0, 'active', '2026-09-22T00:00:00Z')"
            )
            store.connection.commit()
            report = memory_audit_continuity_report(store.connection, directory)
            self.assertEqual(report["status"], "changed_externally")
            self.assertEqual(report["rows_delta"], 1)
            self.assertEqual(report["unattributed_rows"], 1)
            self.assertEqual(report["pseudo_run_rows"], 1)
            self.assertNotEqual(report["digest"], report["checkpoint_digest"])
            self.assertNotIn("injected from outside", json.dumps(report))
            store.close()

    def test_durable_log_line_carries_the_count_that_names_an_outside_write(self) -> None:
        # The module promises that the two provenance counts let a reader of the
        # log separate an in-band write from an outside one. They were computed
        # only in the live report, so the durable line showed *that* the store
        # changed but never *what kind* of change it was: the attribution died
        # with the process that printed it. The counts now travel in the record,
        # so the same comparison works against a line read later, or against a
        # line written by a previous generation.
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            # In band: attributed, but to a run that never existed (``handoff-0``
            # does exactly this), so it is attributed yet not unattributed.
            self._seed(store, "in band", source_run="handoff-0")
            checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            baseline = json.loads((Path(directory) / LOG_NAME).read_text(encoding="utf-8").strip().splitlines()[-1])
            self.assertIn("unattributed_rows", baseline)
            self.assertEqual(baseline["unattributed_rows"], 0)
            # Outside: a direct INSERT with no run, exactly what the audit exists
            # to make visible.
            store.connection.execute(
                "INSERT INTO memories(memory_id, kind, content, confidence, updated_at, pinned, status, valid_from) "
                "VALUES ('feedfacefeedfacefeedfacefeedface', 'fact', 'written behind the harness', 0.9, "
                "'2026-09-22T00:00:00Z', 0, 'active', '2026-09-22T00:00:00Z')"
            )
            store.connection.commit()
            report = memory_audit_continuity_report(store.connection, directory)
            self.assertEqual(report["status"], "changed_externally")
            # The durable line alone names the outside write: the count moved by
            # exactly one while the in-band row left it at zero.
            self.assertEqual(report["unattributed_rows"], baseline["unattributed_rows"] + 1)
            # The checkpoint is a pointer to that line, not a live reading, so it
            # must still carry the count as of the checkpoint: the outside write
            # happened after it and cannot have been folded into it.
            checkpoint = json.loads((Path(directory) / CHECKPOINT_NAME).read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["unattributed_rows"], baseline["unattributed_rows"])
            self.assertEqual((checkpoint["sequence"], checkpoint["rows"]), (baseline["sequence"], baseline["rows"]))
            # One predicate, two readers: the count in the digest and the count in
            # the report cannot drift apart.
            self.assertEqual(report["unattributed_rows"], memory_digest(store.connection)["unattributed_rows"])
            # The count is not content (negative control): the chained digest is
            # computed over the audited column values only, so an independent
            # re-implementation of that chain -- which never sees a count --
            # reproduces it exactly. A count folded into the digest would break
            # this, and the live 349-row store would not still hash to the digest
            # its pre-change line 4 recorded.
            chain = hashlib.sha256()
            for row in store.connection.execute(
                f"SELECT {', '.join(AUDITED_COLUMNS)} FROM memories ORDER BY memory_id"
            ).fetchall():
                chain.update("\x1f".join("" if value is None else str(value) for value in row).encode("utf-8", "replace"))
                chain.update(b"\x1e")
            self.assertEqual(chain.hexdigest(), memory_digest(store.connection)["digest"])
            self.assertNotIn("written behind the harness", json.dumps(report))
            store.close()

    def test_in_place_edit_changes_the_digest_at_constant_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            memory_id = self._seed(store, "one word matters")
            checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            store.connection.execute("UPDATE memories SET content='one word matters!' WHERE memory_id=?", (memory_id,))
            store.connection.commit()
            report = memory_audit_continuity_report(store.connection, directory)
            self.assertEqual(report["status"], "changed_externally")
            self.assertEqual(report["rows_delta"], 0)
            self.assertEqual(report["bytes_delta"], 1)
            store.close()

    def test_report_without_a_checkpoint_says_so_instead_of_guessing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            self._seed(store, "no baseline yet")
            report = memory_audit_continuity_report(store.connection, directory)
            self.assertEqual(report["status"], "no_baseline")
            self.assertFalse((Path(directory) / CHECKPOINT_NAME).exists())
            store.close()

    def test_unwritable_directory_is_reported_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            self._seed(store, "still fine")
            result = checkpoint_memory_audit(store.connection, Path(directory) / "state.sqlite3", reason="episode_end")
            self.assertEqual(result["status"], "failed")
            store.close()

    def test_digest_is_order_independent_of_insertion_and_depends_on_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            self._seed(store, "alpha")
            self._seed(store, "beta")
            first = memory_digest(store.connection)
            store.connection.execute("UPDATE memories SET confidence=0.123 WHERE content='beta'")
            store.connection.commit()
            second = memory_digest(store.connection)
            self.assertEqual(first["rows"], second["rows"])
            self.assertNotEqual(first["digest"], second["digest"])
            store.close()


    def test_anchors_are_appended_once_per_checkpoint_and_named_by_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            self._seed(store, "first")
            checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            self._seed(store, "second")
            checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            log_lines = (Path(directory) / LOG_NAME).read_text(encoding="utf-8").strip().splitlines()
            anchors = [
                json.loads(line)
                for line in (Path(directory) / ANCHOR_NAME).read_text(encoding="utf-8").strip().splitlines()
            ]
            # One anchor per checkpoint, each committing to the line at its own
            # sequence -- the position, not only the tail.
            self.assertEqual([anchor["sequence"] for anchor in anchors], [1, 2])
            for anchor in anchors:
                self.assertEqual(
                    anchor["line_hash"],
                    hashlib.sha256(log_lines[anchor["sequence"] - 1].encode("utf-8")).hexdigest(),
                )
            self.assertTrue(verify_chain(directory)["ok"])
            store.close()

    def test_coordinated_rewrite_of_log_and_checkpoint_is_caught_by_the_anchor(self) -> None:
        """The hole this closes: log and checkpoint were one writer's two files.

        A rewrite of *both*, internally consistent, verified clean because the
        only evidence lived in the two files the same writer rewrites (measured
        on the pre-anchor code: a forged 3-line chain plus a matching
        checkpoint returned ``ok=True`` where the real log had 5 lines). RFC 9162
        section 2.1.4.2 names this split view and requires a commitment the log
        operator cannot re-issue; the anchor file is appended once per checkpoint
        and never rewritten, so it is that commitment here.
        """
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            for index in range(5):
                self._seed(store, f"seed {index}")
                checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            # Exactly what a writer holding state/ does: forge a shorter,
            # self-consistent chain and a checkpoint that agrees with it.
            previous = GENESIS
            forged: list[str] = []
            for index in range(1, 4):
                record = {
                    "sequence": index,
                    "at": f"2026-01-0{index}T00:00:00Z",
                    "reason": "episode_end",
                    "rows": 100 + index,
                    "bytes": 1000 + index,
                    "digest": "a" * 64,
                    "previous": previous,
                }
                line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                forged.append(line)
                previous = hashlib.sha256(line.encode("utf-8")).hexdigest()
            (Path(directory) / LOG_NAME).write_text("\n".join(forged) + "\n", encoding="utf-8")
            last = json.loads(forged[-1])
            (Path(directory) / CHECKPOINT_NAME).write_text(
                json.dumps(
                    {**last, "line_hash": hashlib.sha256(forged[-1].encode("utf-8")).hexdigest()},
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            report = verify_chain(directory)
            self.assertFalse(report["ok"])
            self.assertEqual(report["broken_at"], 4)
            self.assertIn("anchor", report["reason"])
            store.close()

    def test_consistent_interior_edit_of_both_files_is_caught_by_the_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            for index in range(3):
                self._seed(store, f"seed {index}")
                checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            path = Path(directory) / LOG_NAME
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            edited = json.loads(lines[1])
            edited["rows"] = 999
            lines[1] = json.dumps(edited, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            # Re-chain everything after the edit so the chain walk is satisfied.
            previous = hashlib.sha256(lines[0].encode("utf-8")).hexdigest()
            for index in range(1, len(lines)):
                record = json.loads(lines[index])
                record["previous"] = previous
                lines[index] = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                previous = hashlib.sha256(lines[index].encode("utf-8")).hexdigest()
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            last = json.loads(lines[-1])
            (Path(directory) / CHECKPOINT_NAME).write_text(
                json.dumps(
                    {**last, "line_hash": hashlib.sha256(lines[-1].encode("utf-8")).hexdigest()},
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            report = verify_chain(directory)
            self.assertFalse(report["ok"])
            self.assertEqual(report["broken_at"], 2)
            self.assertIn("anchor", report["reason"])
            store.close()

    def test_checkpoint_rolled_back_behind_an_anchor_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            for index in range(3):
                self._seed(store, f"seed {index}")
                checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            lines = (Path(directory) / LOG_NAME).read_text(encoding="utf-8").strip().splitlines()
            first = json.loads(lines[0])
            (Path(directory) / CHECKPOINT_NAME).write_text(
                json.dumps({**first, "line_hash": hashlib.sha256(lines[0].encode("utf-8")).hexdigest()}, sort_keys=True),
                encoding="utf-8",
            )
            report = verify_chain(directory)
            self.assertFalse(report["ok"])
            self.assertEqual(report["broken_at"], 3)
            self.assertIn("rolled back", report["reason"])
            store.close()

    def test_a_log_written_before_anchors_existed_still_verifies(self) -> None:
        """The honest limit, pinned rather than hidden.

        An absent anchor file (a log written before this change, or a writer that
        deleted it) leaves the chain walk and the checkpoint as the only checks,
        which is the state the anchor exists to improve on. The mechanism raises
        the cost of a coordinated rewrite; it is not cryptographic protection
        against a writer with full access to state/, and this test keeps that
        limit visible instead of implying otherwise.
        """
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            self._seed(store, "seed")
            checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            (Path(directory) / ANCHOR_NAME).unlink()
            self.assertTrue(verify_chain(directory)["ok"])
            # The next checkpoint re-anchors, so the window is bounded by one
            # checkpoint rather than left open forever.
            checkpoint_memory_audit(store.connection, directory, reason="episode_end")
            self.assertTrue((Path(directory) / ANCHOR_NAME).exists())
            self.assertTrue(verify_chain(directory)["ok"])
            store.close()

class MemoryAuditWiringTests(unittest.TestCase):
    """The audit is worthless if the cycle never writes it."""

    def test_a_completed_cycle_appends_an_episode_end_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(
                FakeProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=root / "state.sqlite3", self_improvement_root=root),
            )
            try:
                self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
                lines = (root / LOG_NAME).read_text(encoding="utf-8").strip().splitlines()
                self.assertEqual(len(lines), 1)
                record = json.loads(lines[0])
                self.assertEqual(record["reason"], "episode_end")
                # The cycle clears active_run_id before the checkpoint, so the
                # recorded run must be the one the episode actually ran as.
                self.assertIsInstance(record["run_id"], str)
                self.assertTrue(record["run_id"])
                self.assertEqual(
                    reactor.store.connection.execute("SELECT COUNT(*) FROM runs WHERE run_id=?", (record["run_id"],)).fetchone()[0],
                    1,
                )
                self.assertTrue(record["rows"] >= 1)
                self.assertTrue(verify_chain(root)["ok"])
                self.assertEqual(
                    reactor.store.connection.execute(
                        "SELECT COUNT(*) FROM event_log WHERE kind='memory_audit_anomaly'"
                    ).fetchone()[0],
                    0,
                )
            finally:
                reactor.close()

    def test_startup_reports_a_rewritten_log_the_continuity_check_calls_unchanged(self) -> None:
        """The log and the checkpoint are re-issued by the same writer.

        A rewrite of both used to be indistinguishable from no change at all:
        the continuity report compared the store against a checkpoint re-issued
        over the outside write, so it printed ``unchanged`` while the only trace
        of that write was gone. ``verify_chain`` already named the break, but
        nothing outside this module's tests called it, so the detection never
        reached the organism. Startup now calls it and records the fact.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root / "state"
            state_dir.mkdir()
            reactor = Reactor(
                FakeProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=state_dir / "state.sqlite3", self_improvement_root=root),
            )
            try:
                reactor.tick("test")
                store = reactor.store
                store.connection.execute(
                    "INSERT INTO memories (memory_id, kind, content, confidence, source_run, updated_at) "
                    "VALUES ('outside', 'fact', 'written outside the checkpoint', 0.9, NULL, '2026-09-22T00:00:00Z')"
                )
                store.connection.commit()
                self.assertEqual(
                    memory_audit_continuity_report(store.connection, state_dir)["status"], "changed_externally"
                )
                current = memory_digest(store.connection)
                record = {
                    "sequence": 1,
                    "at": "2026-09-22T00:00:00.000000Z",
                    "reason": "episode_end",
                    "run_id": None,
                    "generation": None,
                    **current,
                    "previous": GENESIS,
                }
                line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                (state_dir / LOG_NAME).write_text(line + "\n", encoding="utf-8")
                (state_dir / CHECKPOINT_NAME).write_text(
                    json.dumps({**record, "line_hash": _digest_text(line)}, sort_keys=True), encoding="utf-8"
                )
                # The continuity report on its own is satisfied by the forgery ...
                self.assertEqual(
                    memory_audit_continuity_report(store.connection, state_dir)["status"], "unchanged"
                )
            finally:
                reactor.close()
            supervisor = Supervisor(
                FakeProvider(), {}, ReactorConfig(state_path=state_dir / "state.sqlite3"), root=root
            )
            supervisor.start()
            try:
                kinds = {
                    str(row[0])
                    for row in supervisor.reactor.store.connection.execute(
                        "SELECT kind FROM event_log WHERE kind LIKE 'memory_audit%'"
                    )
                }
            finally:
                supervisor.stop()
            # ... and the anchor witness, now actually called, catches what it cannot.
            self.assertIn("memory_audit_chain_broken", kinds)

    def test_startup_does_not_report_a_break_for_an_intact_chain(self) -> None:
        """The new check must be silent on untouched state, or it is just noise."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root / "state"
            state_dir.mkdir()
            reactor = Reactor(
                FakeProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=state_dir / "state.sqlite3", self_improvement_root=root),
            )
            try:
                reactor.tick("test")
            finally:
                reactor.close()
            self.assertTrue(verify_chain(state_dir)["ok"])
            supervisor = Supervisor(
                FakeProvider(), {}, ReactorConfig(state_path=state_dir / "state.sqlite3"), root=root
            )
            supervisor.start()
            try:
                kinds = {
                    str(row[0])
                    for row in supervisor.reactor.store.connection.execute(
                        "SELECT kind FROM event_log WHERE kind LIKE 'memory_audit%'"
                    )
                }
            finally:
                supervisor.stop()
            self.assertNotIn("memory_audit_chain_broken", kinds)

    def test_startup_reports_a_change_and_baselines_an_empty_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root
            store = StateStore(root / "state.sqlite3")
            try:
                # No checkpoint yet: the startup check must baseline, not guess.
                self.assertEqual(memory_audit_continuity_report(store.connection, state_dir)["status"], "no_baseline")
                checkpoint_memory_audit(store.connection, state_dir, reason="startup_baseline")
                self.assertEqual(memory_audit_continuity_report(store.connection, state_dir)["status"], "unchanged")
                store.connection.execute(
                    "INSERT INTO memories(memory_id, kind, content, confidence, updated_at, pinned, status, valid_from) "
                    "VALUES ('f' * 32, 'fact', 'written behind the harness', 0.9, '2026-09-22T00:00:00Z', 0, 'active', "
                    "'2026-09-22T00:00:00Z')"
                )
                store.connection.commit()
                report = memory_audit_continuity_report(store.connection, state_dir)
                self.assertEqual(report["status"], "changed_externally")
                self.assertEqual(report["rows_delta"], 1)
                self.assertEqual(report["log_lines_appended"], 0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
