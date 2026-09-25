"""The hub-ablation report decides local contamination from general dilution.

A cohort replay can show that a published ranking figure moved because later
memories exist. It cannot say whether the cause is the later memories that quote
the fixture vocabulary or every later memory. These tests pin the ablation that
answers it: the partitions it reports, the memory it names for a contested slot,
and that it only reads the ledger.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from skynet.memory_store import MemoryStore
from skynet.recall_ablation import REFERENCE_AS_OF, hub_ablation
from skynet.recall_scorecard import HUB_MEMORY_ID, RECORDED_PHRASE, SITUATIONS, _query
from skynet.store import StateStore

AS_OF = "2026-06-01T00:00:00.000000Z"
BEFORE = "2026-05-01T00:00:00.000000Z"
LOW_OVERLAP = "l" * 32  # shares one term with S1's query and none with the others
QUOTER = "q" * 32  # quotes four of S1's query terms verbatim
SLOT_HOLDER = "c" * 32  # old, but wins the S2-S4 top-1 slots on the metadata axes


def _terms(situation_key: str) -> list[str]:
    situation = next(item for item in SITUATIONS if item.key == situation_key)
    return MemoryStore._normalize_terms(_query(situation, RECORDED_PHRASE))


def _seed(store: StateStore, memory_id: str, kind: str, content: str, confidence: float, updated_at: str) -> None:
    store.connection.execute(
        "INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at) VALUES(?,?,?,?,NULL,?)",
        (memory_id, kind, content, confidence, updated_at),
    )
    assert store.memory_store is not None
    store.memory_store.index_memory(memory_id, kind, content)


def _ledger(directory: str) -> Path:
    """A small ledger in which the hub is displaced by a memory sharing one term."""
    path = Path(directory) / "state.sqlite3"
    store = StateStore(path)
    query_terms = _terms("S1")
    _seed(store, HUB_MEMORY_ID, "measurement", " ".join(query_terms[:6]), 0.9, BEFORE)
    _seed(store, SLOT_HOLDER, "outcome", query_terms[0] + " released", 0.9, "2026-05-02T00:00:00Z")
    _seed(store, QUOTER, "measurement", " ".join(query_terms[:4]), 0.5, "2026-07-01T00:00:00Z")
    _seed(store, LOW_OVERLAP, "observation", query_terms[0] + " unrelated", 0.99, "2026-08-01T00:00:00Z")
    store.connection.commit()
    store.close()
    return path


class HubAblationTests(unittest.TestCase):
    def test_the_partitions_count_only_the_quoting_memories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = hub_ablation(_ledger(directory), AS_OF, min_terms=3)
            self.assertEqual(report["active_memories"], 4)
            self.assertEqual(report["cohort_memories"], 2)
            self.assertEqual(report["post_reference_memories"], 2)
            # The four-term quoter is in the min_terms partition; the memory that
            # shares a single term is not, but both share at least one term.
            self.assertEqual(report["quoters_min_terms"], 1)
            self.assertEqual(report["quoters_any_term"], 2)

    def test_removing_the_quoters_leaves_the_displacement_standing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = hub_ablation(_ledger(directory), AS_OF, min_terms=3)
            readings = report["readings"]
            # The low-overlap memory, not the quoter, holds the top-1 slot...
            self.assertEqual(readings["live"]["per_fixture"]["S2"]["top1"], LOW_OVERLAP)
            # ...so dropping every memory that quotes >= 3 fixture terms does not
            # restore the hub. The cause is general dilution, not contamination.
            self.assertEqual(readings["live_minus_quoters_ge_3"]["hub_top1"], readings["live"]["hub_top1"])
            self.assertEqual(readings["live_minus_quoters_ge_3"]["per_fixture"]["S2"]["top1"], LOW_OVERLAP)
            self.assertEqual(readings["live_minus_quoters_ge_3"]["per_fixture"]["S2"]["hub_rank"], 3)

    def test_removing_every_term_sharing_memory_restores_the_cohort_reading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = hub_ablation(_ledger(directory), AS_OF, min_terms=3)
            readings = report["readings"]
            # Both later memories share at least one term with S1, so this
            # removal set is exactly the cohort and must read identically.
            self.assertEqual(readings["live_minus_all_quoters"], readings["cohort"])
            self.assertLess(readings["live"]["per_fixture"]["S2"]["hub_rank"], 20)

    def test_readings_name_the_memory_holding_each_top1_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = hub_ablation(_ledger(directory), AS_OF, min_terms=3)
            for name, reading in report["readings"].items():
                self.assertEqual(set(reading["per_fixture"]), {situation.key for situation in SITUATIONS}, name)
                for entry in reading["per_fixture"].values():
                    self.assertGreaterEqual(entry["hub_rank"] or 99, 1)
                    self.assertIsInstance(entry["top1_shared_terms"], list)

    def test_a_contested_slot_names_the_removal_that_raises_the_figure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = hub_ablation(_ledger(directory), AS_OF, min_terms=3)
            cohort = report["readings"]["cohort"]
            self.assertEqual(report["contested_cohort_top1"], [SLOT_HOLDER])
            self.assertEqual(len(report["single_removals"]), 1)
            entry = report["single_removals"][0]
            self.assertEqual(entry["memory_id"], SLOT_HOLDER)
            # The slot holder wins S2, S3 and S4, so removing it hands exactly
            # those three slots back to the hub.
            self.assertEqual(entry["hub_top1"], cohort["hub_top1"] + 3)
            self.assertTrue(entry["raises_hub_top1"])
            # This fixture is not the published corpus, so its figure is not the
            # published 4/5; the two fields are independent and both are pinned.
            self.assertFalse(entry["equals_reference_hub_top1"])
            self.assertEqual(entry["equals_reference_hub_top1"], entry["hub_top1"] == 4)

    def test_no_contested_slot_is_reported_without_removals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            hub = " ".join(_terms("S1")[:6])
            store.connection.execute(
                "INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at) VALUES(?,?,?,?,NULL,?)",
                (HUB_MEMORY_ID, "measurement", hub, 0.9, BEFORE),
            )
            assert store.memory_store is not None
            store.memory_store.index_memory(HUB_MEMORY_ID, "measurement", hub)
            store.connection.commit()
            store.close()
            report = hub_ablation(path, AS_OF, min_terms=3)
            self.assertEqual(report["contested_cohort_top1"], [])
            self.assertEqual(report["single_removals"], [])

    def test_the_report_is_deterministic_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _ledger(directory)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            first = hub_ablation(path, AS_OF, min_terms=3)
            second = hub_ablation(path, AS_OF, min_terms=3)
            self.assertEqual(first, second)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_the_default_instant_is_the_published_cohort_instant(self) -> None:
        self.assertEqual(REFERENCE_AS_OF, "2026-09-22T23:40:45.000000Z")


if __name__ == "__main__":
    unittest.main()
