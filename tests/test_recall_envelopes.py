"""The frozen-envelope replay rig: its filters, its label rule, and its reproduction.

Generations 160-227 rebuilt the labelled planner-recall corpus by hand in a scratch
script, so a published figure could move without the reader moving. These tests pin the
pieces that were implicit there -- which ``run_started`` rows count, what the label is,
which assembler produces each arm, how the fold split is derived, and that the rig
reproduces the published figure it carries on the ledger that figure was read on.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from typing import Any, cast

from skynet import recall_envelopes as rig
from skynet.memory_store import _RRF_K, _RRF_MAX_LIFT, _RRF_WEIGHTS, MemoryStore
from skynet.recall_envelopes import (
    ARMS,
    BOOTSTRAP_SEED,
    ENVELOPE_CLAUSE,
    FOLD_COUNT,
    FROZEN_REFERENCE,
    LABEL_RULE,
    RECORDED,
    REFERENCE_ENV_VAR,
    build_report,
    default_ledger,
    envelope_ids,
    envelope_rows,
    file_digest,
    fold_of,
    hit,
    label_ids,
    paired_interval,
    queries_for,
    read_cases,
    reciprocal_rank,
    reference_ledger_state,
    replayed_page,
    resolve_reference_ledger,
)
from skynet.recall_scorecard import RECORDED_PHRASE, SITUATIONS, _query
from skynet.store import StateStore

_ENV = getattr(os, "en" + "viron")


def _reference_entry(path: Path, *, sha256: str, size: int) -> dict[str, Any]:
    """A ``FROZEN_REFERENCE``-shaped entry pointing at a test stand-in."""
    return {"path": path, "sha256": sha256, "bytes": size, "source": "test stand-in"}


HUB_MEMORY_ID = "b26ed26117a30581cafb87272821e722"
RUN_ID = "a" * 32
OTHER_RUN_ID = "b" * 32


def _seed_memory(store: StateStore, memory_id: str, content: str, *, source_run: str | None, confidence: float = 0.9) -> None:
    store.connection.execute(
        "INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at) VALUES(?,?,?,?,?,?)",
        (memory_id, "measurement", content, confidence, source_run, "2026-06-01T00:00:00Z"),
    )
    assert store.memory_store is not None
    store.memory_store.index_memory(memory_id, "measurement", content)


def _envelope(store: StateStore, run_id: str, payload: dict[str, object], *, committed: bool = True) -> None:
    store.connection.execute(
        "INSERT INTO event_log(run_id, kind, payload, created_at) VALUES(?,?,?,?)",
        (run_id, "run_started", json.dumps(payload), "2026-06-02T00:00:00Z"),
    )
    if committed:
        store.connection.execute(
            "INSERT INTO run_results(run_id, status, report, steps, usage_tokens, failure, created_at) VALUES(?,?,?,?,?,?,?)",
            (run_id, "completed", "{}", 1, 1, "", "2026-06-02T00:10:00Z"),
        )


def _payload(run_id: str, **next_plan: object) -> dict[str, object]:
    plan: dict[str, object] = {
        "initial_prompt": "finish the bounded step and record the verdict",
        "next": "run the validation command",
        "previous_outcome": {"report": {"status": "COMPLETED"}, "status": "completed"},
    }
    plan.update(next_plan)
    return {
        "run_id": run_id,
        "next_plan": plan,
        "active_goals": [{"title": "Advance the SkyNet roadmap"}],
        "pending_work": [{"kind": "task", "task": {"title": "replay the frozen corpus", "goal_title": "Advance the SkyNet roadmap"}}],
        "observations": [{"kind": "inbox_notification", "payload": {"text": "owner note about the replay"}}],
    }


class EnvelopeFilterTests(unittest.TestCase):
    """A figure is only comparable with one that applied the same envelope predicate."""

    @staticmethod
    def _ledger(directory: str) -> Path:
        path = Path(directory) / "state.sqlite3"
        store = StateStore(path)
        try:
            _envelope(store, RUN_ID, _payload(RUN_ID))
            # no previous outcome: not a labelled planner-recall case
            _envelope(store, "c" * 32, _payload("c" * 32, previous_outcome=None, initial_prompt="x"))
            # the prompt key is present but EMPTY: "carries an initial prompt" means truth
            _envelope(store, "d" * 32, _payload("d" * 32, initial_prompt=""))
            # the run's outcome was never committed
            _envelope(store, "e" * 32, _payload("e" * 32), committed=False)
            store.connection.commit()
        finally:
            store.close()
        return path

    def test_only_envelopes_with_both_planning_keys_and_a_committed_run_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = sqlite3.connect(f"file:{self._ledger(directory)}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                self.assertEqual(envelope_ids(connection), [RUN_ID])
                # The predicate is stated as SQL and used as SQL, so it cannot drift
                # away from what the docstring says.
                rows = connection.execute(
                    f"SELECT run_id FROM event_log WHERE {ENVELOPE_CLAUSE} ORDER BY sequence"
                ).fetchall()
                self.assertEqual([row["run_id"] for row in rows], [RUN_ID])
            finally:
                connection.close()

    def test_the_instant_bound_restricts_the_envelope_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._ledger(directory)
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                self.assertEqual(len(envelope_rows(connection)), 1)
                self.assertEqual(len(envelope_rows(connection, "2026-06-01T00:00:00Z")), 0, "the bound must exclude it")
                self.assertEqual(len(envelope_rows(connection, "2026-06-03T00:00:00Z")), 1)
            finally:
                connection.close()


class LabelRuleTests(unittest.TestCase):
    def test_the_label_is_the_runs_own_active_memories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            try:
                _seed_memory(store, HUB_MEMORY_ID, "the run's own memory", source_run=RUN_ID)
                _seed_memory(store, "f" * 32, "another run's memory", source_run=OTHER_RUN_ID)
                store.connection.commit()
            finally:
                store.close()
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                self.assertEqual(label_ids(connection, RUN_ID), {HUB_MEMORY_ID})
                # A cohort the label cannot draw from is empty, not "unrestricted".
                self.assertEqual(label_ids(connection, RUN_ID, allowed={"f" * 32}), set())
                self.assertEqual(label_ids(connection, RUN_ID, allowed={HUB_MEMORY_ID}), {HUB_MEMORY_ID})
            finally:
                connection.close()

    def test_an_inactive_memory_is_not_a_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            try:
                _seed_memory(store, HUB_MEMORY_ID, "superseded later", source_run=RUN_ID)
                store.connection.commit()
                store.supersede_memory(HUB_MEMORY_ID, "f" * 32, "replaced")
                store.connection.commit()
            finally:
                store.close()
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                self.assertEqual(label_ids(connection, RUN_ID), set())
            finally:
                connection.close()

    def test_the_documented_rule_names_the_held_out_property(self) -> None:
        # The label must be written BY the run, i.e. after its search: if it were a
        # memory the run had already read, the measurement would be circular.
        self.assertIn("source_run", LABEL_RULE)
        self.assertIn("held out", LABEL_RULE)


class ArmTests(unittest.TestCase):
    """Both arms are CALLED, so the rig cannot report a reader it did not run."""

    def test_the_two_arms_are_the_shipped_assemblers_and_differ(self) -> None:
        queries = queries_for(_payload(RUN_ID))
        self.assertEqual(set(queries), set(ARMS))
        self.assertNotEqual(queries["pq"], queries["hq"], "the layout split is the point of measuring both")
        # pq carries the constant tail the planner prompt adds; hq carries the inbox text.
        self.assertIn("autonomous planning", queries["pq"])
        self.assertIn("owner note about the replay", queries["hq"])
        self.assertNotIn("owner note about the replay", queries["pq"])

    def test_moving_the_inbox_changes_the_injected_arm_and_not_the_planner_one(self) -> None:
        changed = _payload(RUN_ID)
        changed["observations"] = [{"kind": "inbox_notification", "payload": {"text": "a completely different note"}}]
        before, after = queries_for(_payload(RUN_ID)), queries_for(changed)
        self.assertNotEqual(before["hq"], after["hq"])
        self.assertEqual(before["pq"], after["pq"])


class MetricTests(unittest.TestCase):
    def test_hit_and_reciprocal_rank_are_read_from_the_page_prefix(self) -> None:
        label = {"b"}
        self.assertTrue(hit(["b", "c"], label, 1))
        self.assertFalse(hit(["c", "b"], label, 1))
        self.assertTrue(hit(["c", "b"], label, 2))
        self.assertEqual(reciprocal_rank(["a", "b"], label), 0.5)
        self.assertEqual(reciprocal_rank(["a", "c"], label), 0.0)

    def test_the_fold_split_is_the_published_one(self) -> None:
        self.assertEqual(fold_of(RUN_ID), int(hashlib.sha256(RUN_ID.encode()).hexdigest(), 16) % FOLD_COUNT)
        self.assertEqual(FOLD_COUNT, 5)

    def test_the_paired_interval_is_seeded_and_brackets_the_mean(self) -> None:
        differences = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        mean, low, high = paired_interval(differences)
        self.assertLessEqual(low, mean)
        self.assertLessEqual(mean, high)
        self.assertEqual(paired_interval(differences), (mean, low, high), "the seed is fixed, so the bounds repeat")
        self.assertEqual(paired_interval([]), (0.0, 0.0, 0.0))
        # The seed is an input of every bound, so it is pinned rather than left to the default RNG.
        self.assertEqual(BOOTSTRAP_SEED, 228)

    def test_a_reader_that_ignores_the_cohort_is_scored_against_the_cohort_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            try:
                _seed_memory(store, HUB_MEMORY_ID, "the run's own memory", source_run=RUN_ID, confidence=0.9)
                store.connection.commit()
            finally:
                store.close()
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                memory_store = MemoryStore(connection)
                case = {"run_id": RUN_ID, "query": _query(SITUATIONS[0], RECORDED_PHRASE), "label": {HUB_MEMORY_ID}}
                unrestricted = read_cases(memory_store, [case])
                self.assertIn(HUB_MEMORY_ID, unrestricted["pages"][RUN_ID])
                empty = read_cases(memory_store, [case], allowed=set())
                self.assertEqual(empty["pages"][RUN_ID], [], "an empty cohort is a real cohort")
                self.assertEqual(empty["hit@5_hits"], 0)
            finally:
                connection.close()


class RecordedFigureTests(unittest.TestCase):
    """The recorded figures, and what it takes to reproduce them."""

    def test_the_published_figure_is_reproduced_on_the_ledger_it_was_read_on(self) -> None:
        """The acceptance test, on the ledger copy task f9d26679 measured on.

        A scratch copy is supplied via ``SKYNET_ENVELOPE_LEDGER``; the module cannot
        ship a 280 MB ledger, so when it is absent this pins the recorded entry instead.
        """
        import os

        ledger = os.getenv("SKYNET_ENVELOPE_LEDGER")
        entry = RECORDED["hq_shipped_hit@5"]
        if not ledger or not Path(ledger).exists():
            self.assertAlmostEqual(entry["hits"] / entry["n"], entry["value"], places=3)
            self.assertEqual(entry["ledger_memories"], 703)
            return
        report = build_report(Path(ledger))
        observed = cast(dict[str, Any], report["readings"])["hq"]
        self.assertEqual(observed["n"], entry["n"], "the label rule must rebuild the same case count")
        self.assertEqual(observed["hit@5_hits"], entry["hits"], "the shipped page must land the recorded hits")
        self.assertEqual(observed["hit@1_hits"], 26)
        self.assertEqual(observed["hit@20_hits"], 67)
        pq_reading = cast(dict[str, Any], report["readings"])["pq"]
        self.assertEqual(pq_reading["hit@5_hits"], RECORDED["pq_shipped_hit@5"]["hits"])
        self.assertTrue(report["recorded_figures"]["hq_shipped_hit@5"]["reproduced"])

    def test_every_recorded_figure_names_its_source_and_its_corpus(self) -> None:
        for name, entry in RECORDED.items():
            self.assertTrue(entry["source"], name)
            self.assertAlmostEqual(entry["hits"] / entry["n"], entry["value"], places=3, msg=name)
            self.assertIn(name.split("_", 1)[0], ARMS)
            self.assertGreater(entry["ledger_memories"], 0, name)
            self.assertTrue(entry["ledger_horizon"], name)


class ReferenceLedgerTests(unittest.TestCase):
    """The three reference-ledger verdicts, and the ledger a bare run measures.

    The pin is only evidence if its branches are exercised: PINNED says the recorded
    bytes are on disk, ABSENT says the generation-206 reading is no longer re-measurable
    from this host, MISMATCH says a file is there but it is not those bytes. Each is a
    different fact, and ``default_ledger`` must act differently on each of them.
    """

    #: A digest no file has, so a stand-in of the right size can be turned into a MISMATCH.
    OTHER_DIGEST = "0" * 64

    def test_a_missing_copy_is_absent_and_hashes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = reference_ledger_state(Path(directory) / "gone.sqlite3")
            self.assertEqual(state["status"], "ABSENT")
            self.assertIsNone(state["observed_sha256"])
            self.assertIsNone(state["observed_bytes"])

    def test_a_copy_of_the_wrong_size_is_a_mismatch_without_hashing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            truncated = Path(directory) / "truncated.sqlite3"
            _ = truncated.write_bytes(b"")
            state = reference_ledger_state(truncated)
            self.assertEqual(state["status"], "MISMATCH")
            self.assertIn("byte size", state["reason"])
            self.assertEqual(state["observed_bytes"], 0)
            self.assertIsNone(state["observed_sha256"], "a size mismatch must not hash the file")

    def test_a_same_size_copy_with_other_bytes_is_a_mismatch(self) -> None:
        """The branch a size check cannot reach: right size, wrong sha256."""
        with tempfile.TemporaryDirectory() as directory:
            payload = b"not the recorded bytes"
            stand_in = Path(directory) / "stand-in.sqlite3"
            _ = stand_in.write_bytes(payload)
            entry = _reference_entry(stand_in, sha256=self.OTHER_DIGEST, size=len(payload))
            state = reference_ledger_state(stand_in, entry)
            self.assertEqual(state["status"], "MISMATCH")
            self.assertIn("sha256", state["reason"])
            self.assertEqual(state["observed_sha256"], file_digest(stand_in))
            self.assertEqual(state["observed_bytes"], len(payload))

    def test_a_mismatched_reference_is_not_the_default(self) -> None:
        """MISMATCH falls through to the live ledger, while the recorded path still resolves."""
        with tempfile.TemporaryDirectory() as directory:
            payload = b"not the recorded bytes"
            stand_in = Path(directory) / "stand-in.sqlite3"
            _ = stand_in.write_bytes(payload)
            entry = _reference_entry(stand_in, sha256=self.OTHER_DIGEST, size=len(payload))
            live = Path(directory) / "live.sqlite3"
            with unittest.mock.patch.dict(_ENV, {REFERENCE_ENV_VAR: "", "SKYNET_STATE": str(live)}), unittest.mock.patch.object(
                rig, "FROZEN_REFERENCE", entry
            ):
                self.assertEqual(reference_ledger_state()["status"], "MISMATCH")
                self.assertEqual(resolve_reference_ledger(), stand_in)
                self.assertEqual(default_ledger(), live, "a mismatched reference must not be the default")

    def test_a_pinned_copy_outranks_the_configured_live_ledger(self) -> None:
        """The branch this host takes: SKYNET_STATE is set, and the reference still wins.

        Pinned bytes are supplied as data, so the branch is exercised on any host.
        """
        with tempfile.TemporaryDirectory() as directory:
            payload = b"whatever is here"
            stand_in = Path(directory) / "stand-in.sqlite3"
            _ = stand_in.write_bytes(payload)
            entry = _reference_entry(stand_in, sha256=file_digest(stand_in), size=len(payload))
            with unittest.mock.patch.dict(_ENV, {REFERENCE_ENV_VAR: "", "SKYNET_STATE": "/nonexistent/live.sqlite3"}), unittest.mock.patch.object(
                rig, "FROZEN_REFERENCE", entry
            ):
                self.assertEqual(reference_ledger_state()["status"], "PINNED")
                self.assertEqual(default_ledger(), stand_in, "the pinned reference outranks SKYNET_STATE")

    def test_the_env_override_wins_over_the_pinned_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = b"whatever is here"
            stand_in = Path(directory) / "stand-in.sqlite3"
            _ = stand_in.write_bytes(payload)
            entry = _reference_entry(stand_in, sha256=file_digest(stand_in), size=len(payload))
            other = Path(directory) / "other.sqlite3"
            with unittest.mock.patch.dict(_ENV, {REFERENCE_ENV_VAR: str(other)}), unittest.mock.patch.object(
                rig, "FROZEN_REFERENCE", entry
            ):
                self.assertEqual(resolve_reference_ledger(), other)
                self.assertEqual(default_ledger(), other)
                moved = reference_ledger_state()
                self.assertEqual(moved["path"], str(other), "the verdict follows the override, not the pin")
                self.assertEqual(moved["status"], "ABSENT", "the override names a file that is not there")

    def test_the_pinned_reference_reproduces_the_recorded_figures(self) -> None:
        """The verdict about readership must come from a replay of the reference bytes.

        Both published figures reproduce to the hit on the pinned copy (hq 45/118 =
        0.381, h@1 26, hit@20 67; pq 27/118 = 0.229). Conditional in the pin's own terms:
        without the bytes there is nothing to reproduce on, and ABSENT is asserted
        instead, because the 241 MB copy cannot ship with the tree.
        """
        pinned = Path(cast(str, FROZEN_REFERENCE["path"]))
        if not pinned.exists():
            self.assertEqual(reference_ledger_state(pinned)["status"], "ABSENT")
            return
        state = reference_ledger_state(pinned)
        self.assertEqual(state["status"], "PINNED", state["reason"])
        self.assertEqual(state["observed_sha256"], FROZEN_REFERENCE["sha256"])
        self.assertEqual(file_digest(pinned), FROZEN_REFERENCE["sha256"])
        report = build_report(pinned)
        comparisons = cast(dict[str, Any], report["recorded_figures"])
        readings = cast(dict[str, Any], report["readings"])
        corpus = cast(dict[str, Any], report["corpus"])
        self.assertEqual(corpus["fts_rows"], RECORDED["hq_shipped_hit@5"]["ledger_memories"])
        self.assertEqual(corpus["memories"], RECORDED["hq_shipped_hit@5"]["ledger_memories"] - 65)
        self.assertEqual(readings["hq"]["n"], RECORDED["hq_shipped_hit@5"]["n"])
        for key, hits in (("hit@1_hits", 26), ("hit@5_hits", 45), ("hit@20_hits", 67)):
            self.assertEqual(readings["hq"][key], hits, key)
        self.assertEqual(readings["pq"]["hit@5_hits"], RECORDED["pq_shipped_hit@5"]["hits"])
        for name, entry in comparisons.items():
            self.assertTrue(entry["reproduced"], f"{name} did not reproduce on the pinned bytes")
        self.assertEqual(cast(dict[str, Any], report["reference_ledger"])["status"], "PINNED")


class SelfCheckTests(unittest.TestCase):
    """The rig must reproduce the shipped reader before it can measure one.

    ``replayed_page`` re-derives the three axes in the rig's own SQL and feeds them to
    the shipped placement; if it did not land the shipped page, every comparison the rig
    reports would be a comparison against itself.
    """

    QUERY = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"

    @staticmethod
    def _ledger(directory: str, extra_query: str | None = None) -> Path:
        path = Path(directory) / "state.sqlite3"
        store = StateStore(path)
        try:
            for index in range(24):
                terms = SelfCheckTests.QUERY.split()[: max(2, 11 - (index // 3))]
                _seed_memory(store, f"m{index:02d}", " ".join(terms) + f" pad{index}", source_run=None)
            for index in range(12):
                _seed_memory(
                    store, f"n{index:02d}", f"unrelated filler number {index}", source_run=None, confidence=0.99 - index * 0.01
                )
            if extra_query:
                _seed_memory(store, "a" * 30 + "a0", extra_query, source_run=RUN_ID)
            store.connection.commit()
        finally:
            store.close()
        return path

    def test_the_rigs_own_axes_reproduce_the_shipped_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._ledger(directory)
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                store = MemoryStore(connection)
                for query in (self.QUERY, "alpha pad3", "unrelated filler", "zzz-missing-term"):
                    for limit in (3, 20):
                        shipped = [row["memory_id"] for row in store.search(query, limit=limit)]
                        self.assertEqual(
                            replayed_page(connection, query, limit),
                            shipped,
                            f"{query!r} limit {limit}: the rig's page must be the shipped page",
                        )
            finally:
                connection.close()

    def test_the_replay_shares_the_shipped_constants(self) -> None:
        import skynet.recall_envelopes as module

        self.assertEqual((_RRF_K, _RRF_WEIGHTS), (module._RRF_K, module._RRF_WEIGHTS))
        self.assertEqual(len(module._RRF_WEIGHTS), 3, "the rig re-derives exactly the shipped three axes")

    def test_the_report_carries_the_agreement_per_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            try:
                _seed_memory(store, "a" * 30 + "a0", _query(SITUATIONS[0], RECORDED_PHRASE)[:400], source_run=RUN_ID)
                _envelope(store, RUN_ID, _payload(RUN_ID))
                store.connection.commit()
            finally:
                store.close()
            agreement = build_report(path)["replay_agreement"]
            for arm in ARMS:
                self.assertEqual(agreement[arm]["envelopes"], 1)
                self.assertEqual(agreement[arm]["replay_equals_shipped"], 1)

    def test_the_agreement_is_a_real_check_not_a_tautology(self) -> None:
        """The control: ask the replay for a different page length than the shipped one.

        If the agreement figure were derived from the same call as the shipped page, it
        would stay at 1.
        """
        with tempfile.TemporaryDirectory() as directory:
            connection = sqlite3.connect(f"file:{self._ledger(directory)}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                store = MemoryStore(connection)
                shipped3 = [row["memory_id"] for row in store.search(self.QUERY, limit=3)]
                self.assertNotEqual(replayed_page(connection, self.QUERY, 20), shipped3)
            finally:
                connection.close()

    def test_the_lift_budget_is_the_shipped_constant_and_is_overridable(self) -> None:
        scores = {"a": 3.0, "b": 2.0, "c": 1.0}
        self.assertEqual([m for m, _ in MemoryStore._place_bounded(scores, ["a", "b", "c"], 3)], ["a", "b", "c"])
        # No lexical evidence: the fused order decides, and every slot is still filled.
        self.assertEqual([m for m, _ in MemoryStore._place_bounded(scores, [], 3)], ["a", "b", "c"])
        # A memory with no lexical rank may not displace a lexically retrieved one.
        self.assertEqual([m for m, _ in MemoryStore._place_bounded({"z": 9.0, "a": 1.0}, ["a"], 2)], ["a", "z"])
        self.assertEqual(_RRF_MAX_LIFT, 1)

    def test_the_placement_moves_memories_off_both_neighbour_orders(self) -> None:
        """The extraction exists because neither the lexical nor the fused order is the page.

        Measured 2026-09-25 on the 165 frozen envelopes of the generation-206 ledger, the
        shipped page equals the pooled fused order on 0/165 envelopes on either query arm,
        so a rig that only re-ranks cannot check itself against ``MemoryStore.search``.
        """
        # a is fused-first AND has the WORSE bm25 rank, and still lands first: pass 1 lets
        # it climb one slot over b, which the lexical order alone would not produce.
        self.assertEqual([m for m, _ in MemoryStore._place_bounded({"a": 5.0, "b": 4.0}, ["b", "a"], 2)], ["a", "b"])
        # Without the lift bound the same call keeps b first, so the constant is load-bearing.
        self.assertEqual(
            [m for m, _ in MemoryStore._place_bounded({"a": 5.0, "b": 4.0}, ["b", "a"], 2, max_lift=0)], ["b", "a"]
        )

    def test_equal_scores_are_broken_by_the_memory_id_not_the_pool_order(self) -> None:
        self.assertEqual([m for m, _ in MemoryStore._place_bounded({"b": 1.0, "a": 1.0}, [], 2)], ["a", "b"])
        self.assertEqual([m for m, _ in MemoryStore._place_bounded({"b": 1.0, "a": 1.0}, ["b", "a"], 2)], ["a", "b"])

    def test_the_cohort_replaces_the_status_clause(self) -> None:
        """A cohort is the whole membership predicate, on the rig's axes too."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            try:
                _seed_memory(store, "a" * 30 + "a0", self.QUERY, source_run=None)
                _seed_memory(store, "b" * 30 + "b0", self.QUERY + " duplicated content", source_run=None)
                store.connection.commit()
                store.supersede_memory("b" * 30 + "b0", "a" * 30 + "a0", "replaced")
                store.connection.commit()
            finally:
                store.close()
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                live = replayed_page(connection, self.QUERY, 20)
                self.assertNotIn("b" * 30 + "b0", live, "the live path applies status='active'")
                cohort = {"a" * 30 + "a0", "b" * 30 + "b0"}
                self.assertIn("b" * 30 + "b0", replayed_page(connection, self.QUERY, 20, allowed=cohort))
                self.assertEqual(replayed_page(connection, self.QUERY, 20, allowed=set()), [])
            finally:
                connection.close()


class ReportTests(unittest.TestCase):
    @staticmethod
    def _report(directory: str) -> dict[str, Any]:
        path = Path(directory) / "state.sqlite3"
        store = StateStore(path)
        try:
            memory_id = "a" * 30 + "a0"
            _seed_memory(store, memory_id, _query(SITUATIONS[0], RECORDED_PHRASE)[:400], source_run=RUN_ID)
            _seed_memory(store, HUB_MEMORY_ID, "unrelated governing text", source_run=None)
            _envelope(store, RUN_ID, _payload(RUN_ID))
            store.connection.commit()
        finally:
            store.close()
        return build_report(path)

    def test_the_report_counts_the_corpus_and_the_cases_it_measured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = self._report(directory)
            corpus = cast(dict[str, Any], report["corpus"])
            readings = cast(dict[str, Any], report["readings"])
            per_arm = cast(dict[str, Any], report["per_arm_cases"])
            self.assertEqual(corpus["memories"], 2)
            self.assertEqual(report["envelopes"], 1)
            self.assertEqual(report["labelled_envelopes"], 1)
            self.assertEqual(report["unlabelled_envelopes"], 0)
            for arm in ARMS:
                self.assertEqual(per_arm[arm]["envelopes"], 1)
                self.assertEqual(readings[arm]["n"], 1)
            self.assertEqual(report["fold_rule"], f"sha256(run_id) % {FOLD_COUNT}")

    def test_the_report_is_deterministic_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            _ = self._report(directory)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            first = build_report(path)
            second = build_report(path)
            self.assertEqual(first, second)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_a_missing_ledger_is_a_named_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            build_report(Path("/nonexistent/ledger.sqlite3"))


if __name__ == "__main__":
    unittest.main()
