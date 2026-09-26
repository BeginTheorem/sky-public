"""The scorecard's REPRODUCED/DRIFTED verdict is a difference interval, not equality.

A published figure that moved by one scenario is not evidence of drift: the five
fixture situations cannot separate a real ranking movement from fixture-to-fixture
variation. These tests pin the rule's three verdicts, the difference interval that
decides them, its stated inputs, the marginal-overlap test it replaces, and the
measured effect on the figures frozen in ``skynet.recall_scorecard``, and the reader
split -- the planner layout and the injected-run layout are different queries, so a
verdict is only meaningful with the reader it was measured on stated beside it.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from skynet.reactor import Reactor
from skynet.recall_scorecard import (
    BUDGET_REASON_BEYOND_BOUND,
    BUDGET_REASON_INSIDE_BAND,
    BUDGET_REASON_NO_DENOMINATOR,
    HUB_MEMORY_ID,
    INJECTED_LAYOUT,
    PLANNER_LAYOUT,
    RECORDED_PHRASE,
    REFERENCE_GOVERNING_TOP3,
    REFERENCE_GOVERNING_TOP20,
    REFERENCE_GOVERNING_VERDICTS,
    REFERENCE_HUB_TOP1,
    REFERENCE_HUB_TOP2,
    REQUIRED_N_SEARCH_LIMIT,
    SHIPPED_PHRASE,
    SITUATIONS,
    VERDICT_MIN_SCENARIOS,
    VERDICT_ROPE,
    _injected_query,
    _pooled_fused_order,
    _query,
    as_of_cohort,
    budget_text,
    build_report,
    classify_verdict,
    difference_interval,
    direction_detail,
    figure_observations,
    harness_supply,
    layout_comparison,
    required_n,
    required_n_detail,
    verdict_detail,
    wilson_interval,
)

# The cohort fixtures below: one memory inside the sample instant, one after it.
COHORT_INSTANT = "2026-06-01T00:00:00.000000Z"
COHORT_MEMORY_WRITTEN_AT = "2026-05-01T00:00:00Z"
POST_COHORT_WRITTEN_AT = "2026-07-01T00:00:00Z"
POST_COHORT_DISPLACER_ID = "f" * 32
# The cohort/status fixtures: a rival for the hub's top-1 slot and a member that was
# alive at the sample instant but superseded afterwards.
SUPERSEDED_MEMBER_ID = "b" * 30 + "b0"
RIVAL_MEMBER_ID = "a" * 30 + "a0"

LIVE_HUB_TOP1 = 0  # measured 2026-09-25 on a read-only copy of the live ledger
LIVE_HUB_TOP2 = 2
LIVE_GOVERNING_TOP3 = 5
LIVE_GOVERNING_TOP20 = 13
LIVE_GOVERNING_VERDICTS = 13
LIVE_ACTIVE_MEMORIES = 550  # the corpus the live figures above were read on

# The budget table, measured 2026-09-25 on a read-only copy of the live ledger.
LIVE_REQUIRED_N = {"hub_top1": 5, "hub_top2": 6, "governing_top3": None, "governing_top20": 35}


class WilsonIntervalTests(unittest.TestCase):
    def test_interval_brackets_the_rate_and_stays_in_the_unit_interval(self) -> None:
        for hits, total in ((0, 5), (2, 5), (5, 5), (3, 8), (13, 13), (8, 24)):
            low, high = wilson_interval(hits, total)
            self.assertLessEqual(0.0, low)
            self.assertLessEqual(low, hits / total)
            self.assertLessEqual(hits / total, high)
            self.assertLessEqual(high, 1.0)

    def test_zero_denominator_is_the_whole_unit_interval(self) -> None:
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_interval_narrows_as_the_denominator_grows(self) -> None:
        wide = wilson_interval(8, 10)
        narrow = wilson_interval(80, 100)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_the_stated_level_pins_the_interval_width(self) -> None:
        # 4 scenarios, 0 hits: the 95% Wilson upper bound is 0.490. A different
        # level would move this number, so the rule's stated coverage is pinned.
        upper = wilson_interval(0, VERDICT_MIN_SCENARIOS)[1]
        self.assertLess(abs(upper - 0.4899), 0.001)

    def test_worst_case_pair_is_separated_only_from_three_scenarios(self) -> None:
        # 0/n vs n/n is the widest possible split, so it fixes the floor of the
        # difference interval: below VERDICT_MIN_SCENARIOS it does not clear the
        # ROPE band, and at VERDICT_MIN_SCENARIOS it does, with a scenario of margin.
        for total in range(1, VERDICT_MIN_SCENARIOS):
            self.assertNotEqual(
                classify_verdict((0, total), (total, total)), "DRIFTED", f"n={total} already separates 0/n from n/n"
            )
        self.assertEqual(classify_verdict((0, VERDICT_MIN_SCENARIOS), (VERDICT_MIN_SCENARIOS, VERDICT_MIN_SCENARIOS)), "DRIFTED")


class DifferenceIntervalTests(unittest.TestCase):
    """The interval the verdict is read from, and why marginal overlap is not it."""

    def test_reproduces_the_published_worked_example(self) -> None:
        # The method's reference implementation publishes (0.1705, 0.809) for the
        # interval of 9/10 - 3/10; a wrong combination formula would not match it.
        low, high = difference_interval((3, 10), (9, 10))
        self.assertLess(abs(low - 0.1705), 0.0015)
        self.assertLess(abs(high - 0.8090), 0.0015)

    def test_brackets_the_observed_difference(self) -> None:
        for recorded, observed in (((4, 5), (0, 5)), ((3, 8), (5, 13)), ((80, 100), (20, 100)), ((8, 24), (8, 24))):
            low, high = difference_interval(recorded, observed)
            difference = observed[0] / observed[1] - recorded[0] / recorded[1]
            self.assertLessEqual(low, difference)
            self.assertLessEqual(difference, high)

    def test_interval_narrows_as_both_denominators_grow(self) -> None:
        wide = difference_interval((80, 100), (20, 100))
        narrow = difference_interval((800, 1000), (200, 1000))
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_overlap_of_the_marginal_intervals_is_not_the_difference_interval(self) -> None:
        # The two marginal 95% intervals for 4/5 and 0/5 still touch, so the old
        # overlap test could not separate them; the difference interval does.
        _, high_recorded = wilson_interval(4, 5)
        low_observed, _ = wilson_interval(0, 5)
        self.assertLess(low_observed, high_recorded)  # they overlap
        _, high = difference_interval((4, 5), (0, 5))
        self.assertLess(high, -VERDICT_ROPE)  # and the difference still clears the band

    def test_a_separated_difference_the_overlap_test_missed(self) -> None:
        # 0/4 vs 4/5 is a 0.80 gap whose marginal intervals overlap, so the
        # overlap test called it UNDERDETERMINED; the difference interval separates it.
        low_recorded, _ = wilson_interval(0, 4)
        _, high_observed = wilson_interval(4, 5)
        self.assertLess(low_recorded, high_observed)
        self.assertEqual(classify_verdict((0, 4), (4, 5)), "DRIFTED")

    def test_zero_denominator_is_the_whole_interval(self) -> None:
        self.assertEqual(difference_interval((4, 0), (4, 5)), (-1.0, 1.0))
        self.assertEqual(difference_interval((4, 5), (0, 0)), (-1.0, 1.0))


class VerdictRuleTests(unittest.TestCase):
    def test_exact_integer_match_reproduces(self) -> None:
        self.assertEqual(classify_verdict((4, 5), (4, 5)), "REPRODUCED")
        self.assertEqual(classify_verdict((8, 24), (8, 24)), "REPRODUCED")
        self.assertEqual(classify_verdict((0, 5), (0, 5)), "REPRODUCED")

    def test_an_equal_rate_over_different_denominators_reproduces(self) -> None:
        # 2/4 and 3/6 are the same rate, so this is not drift.
        self.assertEqual(classify_verdict((2, 4), (3, 6)), "REPRODUCED")

    def test_overlapping_intervals_are_underdetermined_not_drifted(self) -> None:
        # One scenario moved, the rate gap is below the ROPE band: no call.
        self.assertEqual(classify_verdict((4, 5), (3, 5)), "UNDERDETERMINED")
        self.assertEqual(classify_verdict((3, 8), (4, 8)), "UNDERDETERMINED")

    def test_a_large_separated_gap_drifts(self) -> None:
        self.assertEqual(classify_verdict((80, 100), (20, 100)), "DRIFTED")

    def test_a_difference_inside_the_rope_band_does_not_drift(self) -> None:
        # 80 scenarios: 8/80 vs 0/80 gives a difference interval reaching into the
        # ROPE band, which is not a practical difference, so no drift call.
        self.assertEqual(classify_verdict((8, 80), (0, 80)), "UNDERDETERMINED")
        # fifteen scenarios more push the whole interval past the band.
        self.assertEqual(classify_verdict((15, 80), (0, 80)), "DRIFTED")
        # a large corpus with a small difference stays no-call.
        self.assertEqual(classify_verdict((100, 1000), (90, 1000)), "UNDERDETERMINED")

    def test_below_the_floor_no_verdict_beyond_reproduction_is_possible(self) -> None:
        for total in range(1, VERDICT_MIN_SCENARIOS):
            for recorded in range(total + 1):
                for observed in range(total + 1):
                    verdict = classify_verdict((recorded, total), (observed, total))
                    self.assertIn(verdict, {"REPRODUCED", "UNDERDETERMINED"}, f"{recorded}/{total} vs {observed}/{total}")

    def test_missing_context_is_underdetermined(self) -> None:
        self.assertEqual(classify_verdict((4, 0), (4, 5)), "UNDERDETERMINED")
        self.assertEqual(classify_verdict((4, 5), (0, 0)), "UNDERDETERMINED")

    def test_detail_carries_every_input_the_verdict_used(self) -> None:
        detail = verdict_detail((4, 5), (0, 5))
        self.assertEqual(detail["verdict"], "DRIFTED")
        self.assertEqual(detail["recorded"], {"hits": 4, "total": 5})
        self.assertEqual(detail["observed"], {"hits": 0, "total": 5})
        self.assertAlmostEqual(detail["recorded_rate"], 0.8)
        self.assertAlmostEqual(detail["observed_rate"], 0.0)
        self.assertEqual(detail["recorded_interval"], wilson_interval(4, 5))
        self.assertEqual(detail["observed_interval"], wilson_interval(0, 5))
        self.assertEqual(detail["difference_interval"], difference_interval((4, 5), (0, 5)))
        self.assertEqual(detail["rope"], VERDICT_ROPE)
        self.assertEqual(detail["min_scenarios"], VERDICT_MIN_SCENARIOS)

    def test_the_hub_top1_gap_is_drifted_by_the_difference_interval(self) -> None:
        # The gap of the one published figure the difference interval separates.
        _, high = difference_interval((REFERENCE_HUB_TOP1, len(SITUATIONS)), (LIVE_HUB_TOP1, len(SITUATIONS)))
        self.assertLess(high, -VERDICT_ROPE)
        self.assertEqual(classify_verdict((REFERENCE_HUB_TOP1, len(SITUATIONS)), (LIVE_HUB_TOP1, len(SITUATIONS))), "DRIFTED")


class DirectionRuleTests(unittest.TestCase):
    def test_equal_readings_reproduce_and_a_fall_reverses(self) -> None:
        self.assertEqual(direction_detail(5, 5, 5)["verdict"], "REPRODUCED")
        self.assertEqual(direction_detail(5, 4, 5)["verdict"], "REVERSED")
        self.assertEqual(direction_detail(5, 3, 0)["verdict"], "UNDERDETERMINED")


class PublishedFiguresTests(unittest.TestCase):
    """The measured verdict change for the figures held in memory."""

    def test_live_corpus_is_larger_than_the_reference_cohort(self) -> None:
        self.assertGreater(LIVE_ACTIVE_MEMORIES, 380)

    def test_hub_top1_is_drifted_under_the_difference_interval(self) -> None:
        # The old exact-integer rule also called this DRIFTED, but for the wrong
        # reason; the marginal-overlap rule hid it as UNDERDETERMINED because the
        # two 95% intervals still touch by 0.058 even though the difference is clear.
        detail = verdict_detail((REFERENCE_HUB_TOP1, len(SITUATIONS)), (LIVE_HUB_TOP1, len(SITUATIONS)))
        self.assertEqual(detail["verdict"], "DRIFTED")
        self.assertNotEqual(detail["recorded"]["hits"], detail["observed"]["hits"])

    def test_hub_top2_and_both_governing_figures_are_underdetermined(self) -> None:
        situations = len(SITUATIONS)
        self.assertEqual(verdict_detail((REFERENCE_HUB_TOP2, situations), (LIVE_HUB_TOP2, situations))["verdict"], "UNDERDETERMINED")
        self.assertEqual(
            verdict_detail((REFERENCE_GOVERNING_TOP3, REFERENCE_GOVERNING_VERDICTS), (LIVE_GOVERNING_TOP3, LIVE_GOVERNING_VERDICTS))["verdict"],
            "UNDERDETERMINED",
        )
        self.assertEqual(
            verdict_detail((REFERENCE_GOVERNING_TOP20, REFERENCE_GOVERNING_VERDICTS), (LIVE_GOVERNING_TOP20, LIVE_GOVERNING_VERDICTS))["verdict"],
            "UNDERDETERMINED",
        )

    def test_the_two_constant_tail_figures_stay_reproduced(self) -> None:
        # 8/24 and 10/24 are exact matches, so the new rule does not disturb them.
        self.assertEqual(classify_verdict((8, 24), (8, 24)), "REPRODUCED")
        self.assertEqual(classify_verdict((10, 24), (10, 24)), "REPRODUCED")

    def test_the_floor_covers_every_reference_denominator(self) -> None:
        self.assertGreaterEqual(len(SITUATIONS), VERDICT_MIN_SCENARIOS)
        self.assertLessEqual(VERDICT_ROPE, 0.25)


class ReaderLayoutTests(unittest.TestCase):
    """A ranking figure is measured through one of two readers; name which one.

    ``_query`` mirrors ``Reactor._planner_memory_query``; the query whose hits enter an
    agent run's prompt is ``Reactor._memory_query``, and the scorecard calls that
    function rather than re-deriving its parts. These tests pin that the two layouts are
    genuinely different queries, that each assembly names its function, and that the
    comparison is scored from the same per-fixture readings the report carries.
    """

    @staticmethod
    def _pair(
        governing: dict[str, int | None],
        *,
        planner_top1: bool,
        planner_top2: bool,
        injected_top1: bool,
        injected_top2: bool,
    ) -> dict[str, Any]:
        """One synthetic fixture pair: what each reader returned for its memories."""

        def reading(top1: bool, top2: bool) -> dict[str, Any]:
            return {
                "goal_intact": True,
                "fused_top1": HUB_MEMORY_ID if top1 else "x" * 32,
                "fused_top3": [HUB_MEMORY_ID if top2 else "y" * 32],
                "pooled_top3": dict(governing),
                "pooled_top20": dict(governing),
            }

        return {
            "label": "synthetic",
            "goal": "synthetic goal",
            "governing": [{"memory_id": mid, "label": mid} for mid in governing],
            "recorded": reading(planner_top1, planner_top2),
            "injected": reading(injected_top1, injected_top2),
        }

    def test_the_two_layouts_are_different_queries_for_every_fixture(self) -> None:
        for situation in SITUATIONS:
            planner = _query(situation, SHIPPED_PHRASE)
            injected = _injected_query(situation)
            self.assertNotEqual(planner, injected, situation.key)
            # The constant phrase is the planner layout's own tail; the injected query is
            # derived from the envelope and never carries a fixed phrase.
            self.assertNotIn(SHIPPED_PHRASE, injected, situation.key)

    def test_the_injected_query_is_the_shipped_reader_not_a_copy_of_it(self) -> None:
        # The scorecard must not re-derive the injected part list: the function it
        # replays is Reactor._memory_query itself, called with the fixture's plan and a
        # task envelope whose title is the fixture label.
        situation = SITUATIONS[0]
        expected = Reactor._memory_query(
            {"task": {"title": situation.label, "goal_title": situation.goal}},
            [],
            {"initial_prompt": situation.initial_prompt, "next": situation.next_step},
        )
        self.assertEqual(_injected_query(situation), expected)

    def test_the_assemblies_name_the_function_that_built_each_query(self) -> None:
        self.assertIn("Reactor._planner_memory_query", PLANNER_LAYOUT)
        self.assertIn("Reactor._join_query_parts", PLANNER_LAYOUT)
        self.assertIn("Reactor._memory_query", INJECTED_LAYOUT)
        self.assertNotEqual(PLANNER_LAYOUT, INJECTED_LAYOUT)

    def test_every_figure_names_its_reader_and_carries_its_margin(self) -> None:
        pair = self._pair({"a" * 32: 3, "b" * 32: None}, planner_top1=False, planner_top2=True, injected_top1=True, injected_top2=True)
        layout = layout_comparison({"S1": pair})
        self.assertEqual(layout["readers"]["planner"], PLANNER_LAYOUT)
        self.assertEqual(layout["readers"]["injected"], INJECTED_LAYOUT)
        self.assertEqual(set(layout["figures"]), {"hub_top1", "hub_top2", "governing_top3", "governing_top20"})
        for name, entry in layout["figures"].items():
            for reader in ("planner", "injected"):
                detail = entry["readers"][reader]
                self.assertEqual(detail["reader"], layout["readers"][reader], name)
                self.assertEqual(detail["layout"], reader, name)
                observed_rate = detail["observed"]["hits"] / detail["observed"]["total"]
                recorded_rate = detail["recorded"]["hits"] / detail["recorded"]["total"]
                self.assertLess(abs(detail["gap"] - (observed_rate - recorded_rate)), 1e-12, name)

    def test_the_same_recorded_figure_can_read_differently_under_the_two_readers(self) -> None:
        # 4/5 recorded; the planner reader sees the hub at neither top-1 nor top-2 and the
        # injected reader sees it at both, which changes the observed rate and so the
        # verdict -- the point of scoring the fixtures twice.
        pair = self._pair({"a" * 32: 1}, planner_top1=False, planner_top2=False, injected_top1=True, injected_top2=True)
        layout = layout_comparison({"S1": pair})
        hub1 = layout["figures"]["hub_top1"]["readers"]
        self.assertEqual(hub1["planner"]["observed"]["hits"], 0)
        self.assertEqual(hub1["injected"]["observed"]["hits"], 1)
        # 0/5 against 4/5 clears the band; 1/5 against 4/5 does not. Same recorded
        # figure, same fixtures, opposite verdicts -- and now both are labelled.
        self.assertEqual(hub1["planner"]["verdict"], "DRIFTED")
        self.assertEqual(hub1["injected"]["verdict"], "UNDERDETERMINED")
        self.assertIn("hub_top1", layout["verdict_changes"])
        self.assertEqual(layout["verdict_change_count"], len(layout["verdict_changes"]))

    def test_a_moved_governing_memory_is_named_per_reader(self) -> None:
        listed = "l" * 32
        absent = "a" * 32
        pair = self._pair({listed: 3, absent: None}, planner_top1=False, planner_top2=True, injected_top1=False, injected_top2=True)
        # The injected reading places the first memory in the top-3 and drops the other.
        pair["injected"]["pooled_top3"] = {listed: None, absent: 4}
        layout = layout_comparison({"S1": pair})
        moved = layout["per_pair"]["S1"]["membership_moved"]
        self.assertEqual({entry["memory_id"] for entry in moved}, {listed, absent})
        self.assertEqual(layout["governing_rank_change_count"], 2)
        self.assertEqual(layout["governing_verdicts"], 2)
        self.assertEqual(layout["per_pair"]["S1"]["planner"]["governing"][listed], {"top3": 3, "top20": 3})

    def test_figure_observations_read_a_reading_by_name(self) -> None:
        pairs = {"S1": self._pair({"a" * 32: 1}, planner_top1=True, planner_top2=True, injected_top1=False, injected_top2=False)}
        planner = figure_observations(pairs, "recorded")
        injected = figure_observations(pairs, "injected")
        # The hub figures are over the frozen fixture count, not the synthetic dict's;
        # only the one pair given here can contribute a hit.
        self.assertEqual(planner["hub_top1"], (1, len(SITUATIONS)))
        self.assertEqual(injected["hub_top1"], (0, len(SITUATIONS)))
        self.assertEqual(planner["governing_top3"], (1, 1))
        self.assertEqual(injected["governing_top20"], (1, 1))

    def test_the_report_carries_both_readings_of_every_fixture(self) -> None:
        # build_report must measure the injected layout through the ledger as well; a
        # missing pair key would silently score the planner reading twice.
        import inspect

        source = inspect.getsource(build_report)
        self.assertIn('"injected": _read(', source)
        self.assertIn("layout_comparison(pairs)", source)


class ReaderCohortTests(unittest.TestCase):
    """Every reader of the report must be measured on the same corpus.

    ``--as-of`` restricts the pooled replay to the cohort a published figure was read
    on. A reader that silently ignores the restriction is scored against today's much
    larger corpus, so the corpus difference is reported as a difference between readers
    -- the exact confusion the layout block exists to remove. These tests build a ledger
    whose post-cohort memory displaces the hub and pin that each reader stays inside the
    cohort.
    """

    @staticmethod
    def _ledger(directory: str) -> Path:
        """A ledger whose post-cohort memory quotes the hub's own query terms."""
        from skynet.memory_store import MemoryStore
        from skynet.store import StateStore

        path = Path(directory) / "state.sqlite3"
        store = StateStore(path)
        terms = MemoryStore._normalize_terms(_query(SITUATIONS[0], RECORDED_PHRASE))
        hub_text = " ".join(terms[:8])
        store.connection.execute(
            "INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at) VALUES(?,?,?,?,NULL,?)",
            (HUB_MEMORY_ID, "measurement", hub_text, 0.9, COHORT_MEMORY_WRITTEN_AT),
        )
        assert store.memory_store is not None
        store.memory_store.index_memory(HUB_MEMORY_ID, "measurement", hub_text)
        # Same vocabulary, higher confidence, written after the cohort instant: only a
        # reader that ignores ``--as-of`` can see it win the top-1 slot.
        later_text = hub_text + " extra"
        store.connection.execute(
            "INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at) VALUES(?,?,?,?,NULL,?)",
            (POST_COHORT_DISPLACER_ID, "outcome", later_text, 0.99, POST_COHORT_WRITTEN_AT),
        )
        store.memory_store.index_memory(POST_COHORT_DISPLACER_ID, "outcome", later_text)
        store.connection.commit()
        store.close()
        return path

    def test_every_reading_in_the_report_stays_inside_the_as_of_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = build_report(self._ledger(directory), COHORT_INSTANT)
            self.assertEqual(report["cohort_memories"], 1)
            self.assertEqual(report["active_memories"], 2)
            self.assertEqual(report["cohort_excluded_active"], 1)
            for key, pair in report["pairs"].items():
                for reading in ("recorded", "shipped", "phrase_free", "injected"):
                    self.assertNotEqual(
                        pair[reading]["fused_top1"], POST_COHORT_DISPLACER_ID, f"{key}/{reading} escaped the cohort"
                    )
                    self.assertEqual(pair[reading]["fused_top1"], HUB_MEMORY_ID, f"{key}/{reading} lost the hub")

    def test_the_membership_comparison_is_taken_on_one_corpus(self) -> None:
        # The layout block compares two readers; if one of them ignores the cohort, the
        # "moved" entries it reports are corpus movement, not reader movement.
        with tempfile.TemporaryDirectory() as directory:
            report = build_report(self._ledger(directory), COHORT_INSTANT)
            layout = report["layout"]
            self.assertEqual(layout["readers"]["planner"], PLANNER_LAYOUT)
            self.assertEqual(layout["readers"]["injected"], INJECTED_LAYOUT)
            for pair in layout["per_pair"].values():
                self.assertTrue(pair["planner"]["hub_top1"])
                self.assertTrue(pair["injected"]["hub_top1"])
                self.assertEqual(pair["membership_moved"], [])


    @staticmethod
    def _cohort_ledger(directory: str, post_cohort: int) -> tuple[Path, set[str]]:
        """A ledger whose cohort holds the hub and S1's governing memories.

        ``post_cohort`` adds same-vocabulary memories written after the sample instant
        at a higher confidence, so a page read on the live ledger puts them above every
        cohort memory; the returned set is the cohort those memories must not enter.
        """
        from skynet.memory_store import MemoryStore
        from skynet.store import StateStore

        path = Path(directory) / "state.sqlite3"
        store = StateStore(path)
        terms = MemoryStore._normalize_terms(_query(SITUATIONS[0], RECORDED_PHRASE))
        hub_text = " ".join(terms[:8])
        rows = [(HUB_MEMORY_ID, "measurement", hub_text, 0.90)]
        rows += [(mid, "lesson", f"{hub_text} {label}", 0.70) for mid, label in SITUATIONS[0].governing]
        for memory_id, kind, content, confidence in rows:
            store.connection.execute(
                "INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at) VALUES(?,?,?,?,NULL,?)",
                (memory_id, kind, content, confidence, COHORT_MEMORY_WRITTEN_AT),
            )
            assert store.memory_store is not None
            store.memory_store.index_memory(memory_id, kind, content)
        for index in range(post_cohort):
            memory_id = f"{index + 1:032x}"
            content = f"{hub_text} later{index}"
            store.connection.execute(
                "INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at) VALUES(?,?,?,?,NULL,?)",
                (memory_id, "outcome", content, 0.99, POST_COHORT_WRITTEN_AT),
            )
            assert store.memory_store is not None
            store.memory_store.index_memory(memory_id, "outcome", content)
        store.connection.commit()
        store.close()
        return path, {HUB_MEMORY_ID, *(mid for mid, _label in SITUATIONS[0].governing)}

    def test_the_shipped_page_is_read_on_the_cohort_too(self) -> None:
        """``shipped_top3``/``shipped_top20`` come from ``MemoryStore.search``.

        That reader took no cohort argument while every pooled rank took one, so under
        ``--as-of`` the printed shipped page of runs 1-4 was still read on today's
        ledger: a post-cohort flood took the page over and the replay reported the
        larger corpus as a property of the reading. Measured on the live ledger, 17 of
        20 shipped-top-3 readings and 12 of 20 shipped-top-20 readings moved once the
        page took the cohort.
        """
        for post_cohort in (0, 12, 40):
            with tempfile.TemporaryDirectory() as directory:
                path, cohort_ids = self._cohort_ledger(directory, post_cohort)
                report = build_report(path, COHORT_INSTANT)
                self.assertEqual(report["cohort_memories"], len(cohort_ids))
                for key, pair in report["pairs"].items():
                    for reading in ("recorded", "shipped", "phrase_free", "injected"):
                        data = pair[reading]
                        self.assertTrue(
                            set(data["shipped_top3"]) <= cohort_ids, f"{key}/{reading} shipped_top3 escaped the cohort"
                        )
                        for memory_id, rank in data["shipped_top20"].items():
                            if rank is not None:
                                self.assertTrue(memory_id in cohort_ids, f"{key}/{reading} shipped_top20 escaped")
                        for entry in pair["governing"]:
                            memory_id = entry["memory_id"]
                            if memory_id in cohort_ids:
                                self.assertIsNotNone(
                                    data["shipped_top20"][memory_id], f"{key}/{reading} dropped a governing memory"
                                )

    def test_the_live_and_cohort_pages_differ_so_the_pin_is_not_vacuous(self) -> None:
        # Without this, ``test_the_shipped_page_is_read_on_the_cohort_too`` would also pass
        # on a build that ignored the cohort argument, if the flood never reached the page.
        with tempfile.TemporaryDirectory() as directory:
            path, _cohort_ids = self._cohort_ledger(directory, 40)
            pair = build_report(path, COHORT_INSTANT)["pairs"]["S1"]
            live = build_report(path, None)["pairs"]["S1"]
            governing = [entry["memory_id"] for entry in pair["governing"]]
            for reading in ("recorded", "shipped", "phrase_free", "injected"):
                live_ranks = {mid: live[reading]["shipped_top20"][mid] for mid in governing}
                cohort_ranks = {mid: pair[reading]["shipped_top20"][mid] for mid in governing}
                self.assertTrue(
                    all(rank is not None for rank in cohort_ranks.values()), f"{reading} lost a governing memory on the cohort"
                )
                self.assertTrue(
                    any(rank is None for rank in live_ranks.values()),
                    f"{reading}: the flood does not reach the shipped page, so the pin is vacuous",
                )
            self.assertNotEqual(live["recorded"]["shipped_top3"], pair["recorded"]["shipped_top3"])

class CohortIsTheWholeMembershipPredicateTests(unittest.TestCase):
    """A supplied cohort REPLACES today's status clause; it does not intersect it.

    ``as_of_cohort`` decides membership from the validity window of the replay instant,
    so intersecting its result with today's ``status='active'`` re-introduces exactly
    the later events the reconstruction removed: a memory alive at the instant and
    superseded afterwards silently vanishes from the replay, and the replayed figure
    becomes a function of the present. Measured on the live ledger at
    2026-09-22T23:40:45Z, three cohort members had been superseded later, two of them
    sat in S3's replayed order, and the hub top-1 figure read 2/5 through the
    intersection where the same instant without it reads 3/5 -- the figure the module's
    own docstring claims for that cohort.
    """

    INSIDE_INSTANT = "2026-05-01T00:00:00.000000Z"
    SAMPLE_INSTANT = "2026-06-01T00:00:00.000000Z"
    SUPERSEDED_LATER_AT = "2026-07-01T00:00:00.000000Z"
    # The fixture vocabulary: S1's recorded query terms, so the hub can be outranked.
    TERMS_USED = 16

    @staticmethod
    def _ledger(directory: str, *, include_superseded: bool = True) -> Path:
        """A cohort of three: the hub, a same-vocabulary rival, and a member removed later.

        The rival outranks the hub on confidence, so the hub's top-1 slot depends on
        which of the three survive. ``SUPERSEDED_MEMBER`` is written inside the cohort
        window and superseded one month AFTER the sample instant: a correct replay of the
        instant keeps it, and a replay that also applies today's status loses it.
        """
        from skynet.memory_store import MemoryStore
        from skynet.store import StateStore

        situation = SITUATIONS[0]
        terms = MemoryStore._normalize_terms(_query(situation, RECORDED_PHRASE))
        shared = " ".join(terms[: CohortIsTheWholeMembershipPredicateTests.TERMS_USED])
        path = Path(directory) / "state.sqlite3"
        store = StateStore(path)
        rows = (
            (HUB_MEMORY_ID, "measurement", shared, 0.90),
            (RIVAL_MEMBER_ID, "lesson", f"{shared} rival {RIVAL_MEMBER_ID[-4:]}", 0.99),
        )
        if include_superseded:
            rows += ((SUPERSEDED_MEMBER_ID, "lesson", f"{shared} sup {SUPERSEDED_MEMBER_ID[-4:]}", 0.99),)
        for memory_id, kind, content, confidence in rows:
            store.connection.execute(
                "INSERT INTO memories(memory_id,kind,content,confidence,source_run,updated_at) VALUES(?,?,?,?,NULL,?)",
                (memory_id, kind, content, confidence, CohortIsTheWholeMembershipPredicateTests.INSIDE_INSTANT),
            )
            assert store.memory_store is not None
            store.memory_store.index_memory(memory_id, kind, content)
        store.connection.commit()
        if include_superseded:
            # Alive at the sample instant, inactive now: only a replay that stays inside
            # the instant can still see it.
            store.supersede_memory(SUPERSEDED_MEMBER_ID, HUB_MEMORY_ID, "superseded after the sample instant")
            store.connection.execute(
                "UPDATE memories SET valid_to=? WHERE memory_id=?",
                (CohortIsTheWholeMembershipPredicateTests.SUPERSEDED_LATER_AT, SUPERSEDED_MEMBER_ID),
            )
        store.connection.commit()
        store.close()
        return path

    def _cohort_order(self, path: Path) -> tuple[list[str], set[str]]:
        import sqlite3

        from skynet.memory_store import MemoryStore

        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            criteria = MemoryStore._normalize_terms(_query(SITUATIONS[0], RECORDED_PHRASE))
            cohort = as_of_cohort(connection, self.SAMPLE_INSTANT)
            return _pooled_fused_order(connection, criteria, 20, allowed=cohort), cohort
        finally:
            connection.close()

    def test_a_member_superseded_after_the_instant_still_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._ledger(directory)
            order, cohort = self._cohort_order(path)
            self.assertEqual(len(cohort), 3)
            self.assertIn(SUPERSEDED_MEMBER_ID, cohort, "the member was alive at the instant")
            self.assertIn(
                SUPERSEDED_MEMBER_ID,
                order,
                "the replay re-applied today's status and dropped a memory that was alive at the instant",
            )

    def test_the_figure_is_a_function_of_the_instant_only(self) -> None:
        """The same instant, replayed twice, must not move when a LATER event is undone.

        The control ledger is the same fixture with the post-instant supersession never
        having happened. Both files describe the same corpus at ``SAMPLE_INSTANT``, so
        every figure read at that instant must agree.
        """
        import shutil
        import sqlite3

        with tempfile.TemporaryDirectory() as directory:
            path = self._ledger(directory)
            control = Path(directory) / "control.sqlite3"
            shutil.copyfile(path, control)
            connection = sqlite3.connect(str(control))
            connection.execute(
                "UPDATE memories SET status='active', valid_to=NULL WHERE memory_id=?", (SUPERSEDED_MEMBER_ID,)
            )
            connection.commit()
            connection.close()
            observed = build_report(path, self.SAMPLE_INSTANT)["verdicts"]["hub_top1"]["observed"]
            counterfactual = build_report(control, self.SAMPLE_INSTANT)["verdicts"]["hub_top1"]["observed"]
            self.assertEqual(observed, counterfactual, "a post-instant supersession rewrote a past figure")

    def test_the_pin_is_not_vacuous(self) -> None:
        # Without the removed-later member the hub's top-1 slot changes, so the two pins
        # above would also pass on a build that ignored the member for an unrelated reason.
        with tempfile.TemporaryDirectory() as with_directory, tempfile.TemporaryDirectory() as without_directory:
            head_with = self._cohort_order(self._ledger(with_directory))[0][:1]
            head_without = self._cohort_order(self._ledger(without_directory, include_superseded=False))[0][:1]
            self.assertNotEqual(head_with, head_without, "the fixture does not depend on the removed-later member")

    def test_a_cohort_with_include_inactive_is_a_runnable_query(self) -> None:
        """``include_inactive=True`` with a cohort was a SQL syntax error on both paths.

        The status clause was composed as ``FROM memories`` + ``" WHERE status='active'"``
        or ``""``, and the cohort fragment always starts with ``AND``, so asking for
        inactive rows inside a cohort produced ``FROM memories AND memory_id IN (...)``.
        Both the FTS and the fallback path are pinned, and the inactive member must be
        reachable exactly when it is asked for.
        """
        import sqlite3

        from skynet.memory_store import MemoryStore

        with tempfile.TemporaryDirectory() as directory:
            path = self._ledger(directory)
            connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                store = MemoryStore(connection)
                cohort = {HUB_MEMORY_ID, RIVAL_MEMBER_ID, SUPERSEDED_MEMBER_ID}
                criteria = _query(SITUATIONS[0], RECORDED_PHRASE)
                for include_inactive in (True, False):
                    for allowed in (None, cohort, set()):
                        rows = store.search(criteria, limit=5, include_inactive=include_inactive, allowed=allowed)
                        self.assertIsInstance(rows, list)
                # A supplied cohort is the membership predicate, so the member that was
                # alive at the instant is present with or without ``include_inactive``.
                for include_inactive in (True, False):
                    rows = store.search(criteria, limit=5, include_inactive=include_inactive, allowed=cohort)
                    self.assertIn(SUPERSEDED_MEMBER_ID, [row["memory_id"] for row in rows])
                # Without a cohort the shipped reader's status clause is unchanged, so an
                # inactive memory is reachable only when it is explicitly asked for.
                self.assertNotIn(
                    SUPERSEDED_MEMBER_ID, [row["memory_id"] for row in store.search(criteria, limit=5)]
                )
                self.assertIn(
                    SUPERSEDED_MEMBER_ID,
                    [row["memory_id"] for row in store.search(criteria, limit=5, include_inactive=True)],
                )
            finally:
                connection.close()


class RequiredNBudgetTests(unittest.TestCase):
    """The budget an UNDERDETERMINED verdict needs, and whether it can be supplied.

    The projection is the topic-set-size question: how many comparable situations
    does this comparison need? (Sakai, "Topic set size design", Information Retrieval
    Journal 19:256-283, 2016, DOI 10.1007/s10791-015-9273-z, which derives n from a
    power requirement plus pilot variance, and reports that different measures need
    substantially different n.) These tests pin the projection, the no-budget case,
    the reachability flag, and the measured live figures.
    """

    def test_the_projection_agrees_with_the_verdict_rule_it_is_derived_from(self) -> None:
        for recorded, observed in (((4, 5), (0, 5)), ((5, 5), (2, 5)), ((8, 8), (4, 12))):
            required = required_n(recorded, observed)
            self.assertIsNotNone(required)
            assert required is not None  # narrowed for the reader, asserted above
            self.assertGreaterEqual(required, VERDICT_MIN_SCENARIOS)
            self.assertEqual(classify_verdict(self._at_n(recorded, required), self._at_n(observed, required)), "DRIFTED")

    def test_the_projection_starts_at_the_verdict_floor(self) -> None:
        # The two rules cannot disagree about the smallest usable n: below the floor
        # classify_verdict refuses a drift call whatever the gap, so the search starts there.
        self.assertGreaterEqual(required_n((1, 1), (0, 20)) or 0, VERDICT_MIN_SCENARIOS)

    def test_a_gap_inside_the_band_has_no_budget_at_any_n(self) -> None:
        # governing top-3's live gap (+0.0096) and 8/24 vs 9/24 are closer together than
        # the band, so no fixture count can decide them: None is a finding about the
        # effect size, not a failed search.
        self.assertIsNone(required_n((3, 8), (5, 13)))
        self.assertIsNone(required_n((8, 24), (9, 24)))
        self.assertIsNone(required_n((0, 5), (0, 5)))

    def test_the_search_bound_is_reported_and_never_silently_exceeded(self) -> None:
        detail = required_n_detail((0, 1), (1, 400))
        self.assertIsNone(detail["required_n"])
        self.assertEqual(detail["search_limit"], REQUIRED_N_SEARCH_LIMIT)

    def test_the_measured_live_budgets_are_pinned(self) -> None:
        situations = len(SITUATIONS)
        table = {
            "hub_top1": ((REFERENCE_HUB_TOP1, situations), (LIVE_HUB_TOP1, situations)),
            "hub_top2": ((REFERENCE_HUB_TOP2, situations), (LIVE_HUB_TOP2, situations)),
            "governing_top3": ((REFERENCE_GOVERNING_TOP3, REFERENCE_GOVERNING_VERDICTS), (LIVE_GOVERNING_TOP3, LIVE_GOVERNING_VERDICTS)),
            "governing_top20": ((REFERENCE_GOVERNING_TOP20, REFERENCE_GOVERNING_VERDICTS), (LIVE_GOVERNING_TOP20, LIVE_GOVERNING_VERDICTS)),
        }
        self.assertEqual({name: required_n(*pair) for name, pair in table.items()}, LIVE_REQUIRED_N)

    def test_reachability_is_decided_against_the_measured_supply(self) -> None:
        supply = {"labelled_cases": 179}
        self.assertTrue(required_n_detail((5, 5), (2, 5), supply)["reachable"])
        self.assertTrue(required_n_detail((6, 8), (13, 13), supply)["reachable"])
        self.assertFalse(required_n_detail((5, 5), (2, 5), {"labelled_cases": 5})["reachable"])
        self.assertIsNone(required_n_detail((3, 8), (5, 13), supply)["reachable"])

    def test_supply_counts_distinct_envelopes_carrying_the_label_context(self) -> None:
        import sqlite3

        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE event_log (kind TEXT, payload TEXT)")
        connection.executemany(
            "INSERT INTO event_log (kind, payload) VALUES (?, ?)",
            [
                ("run_started", '{"run_id": "a", "next_plan": {"previous_outcome": {}}}'),
                ("run_started", '{"run_id": "a", "next_plan": {"previous_outcome": {}}}'),
                ("run_started", '{"run_id": "b", "next_plan": {"previous_outcome": {}}}'),
                ("run_started", '{"run_id": "c"}'),
                ("run_finished", '{"run_id": "d", "next_plan": {"previous_outcome": {}}}'),
            ],
        )
        try:
            self.assertEqual(harness_supply(connection)["labelled_cases"], 2)
        finally:
            connection.close()

    def test_the_supply_is_bounded_by_the_replay_instant(self) -> None:
        """The reachability flag is an input of the replay, so the supply is replayed too.

        ``build_report(path, as_of)`` restricts every ranking figure to the cohort that
        existed at ``as_of``; without the same bound on ``harness_supply`` the printed
        supply and the ``reachable`` flag were read on today's event log, so a replay
        claimed the harness could pay for an n the cohort could not. Measured on the
        live ledger over nine historical instants, that disagreement moves 6 of the 32
        (figure, instant) cells that carry a budget.
        """
        import sqlite3

        from skynet.store import StateStore

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            try:
                for run_id, created in (
                    ("a" * 8, "2026-05-01T00:00:00Z"),
                    ("b" * 8, "2026-05-20T00:00:00Z"),
                    ("c" * 8, "2026-07-01T00:00:00Z"),  # after COHORT_INSTANT
                ):
                    store.connection.execute(
                        "INSERT INTO event_log(run_id, kind, payload, created_at) VALUES(?,?,?,?)",
                        (run_id, "run_started", json.dumps({"run_id": run_id, "previous_outcome": {}}), created),
                    )
                store.connection.commit()
            finally:
                store.close()
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                self.assertEqual(harness_supply(connection)["labelled_cases"], 3)
                bounded = harness_supply(connection, COHORT_INSTANT)
                self.assertEqual(bounded["labelled_cases"], 2)
                self.assertNotEqual(bounded["description"], harness_supply(connection)["description"])
            finally:
                connection.close()
            report = build_report(path, COHORT_INSTANT)
            self.assertEqual(report["harness_supply"]["labelled_cases"], 2)
            # The live replay keeps the whole ledger, so the bound changes nothing there.
            self.assertEqual(build_report(path, None)["harness_supply"]["labelled_cases"], 3)

    def test_a_reachability_verdict_is_decided_on_the_bounded_supply(self) -> None:
        # The flag and the count must agree: an n between the cohort supply and today's
        # is reachable on today's log and not on the replayed one.
        bounded = {"labelled_cases": 4}
        unbounded = {"labelled_cases": 198}
        recorded, observed = (REFERENCE_HUB_TOP1, len(SITUATIONS)), (0, len(SITUATIONS))
        required = required_n(recorded, observed)
        assert required is not None
        self.assertGreater(required, 4)
        self.assertLessEqual(required, 198)
        self.assertTrue(required_n_detail(recorded, observed, unbounded)["reachable"])
        self.assertFalse(required_n_detail(recorded, observed, bounded)["reachable"])

    def test_a_gap_wider_than_the_band_is_never_reported_as_undecidable(self) -> None:
        # The regression: required_n returns None both when the gap cannot be decided
        # at ANY n and when the deciding n merely sits past the search bound. Reporting
        # both as "the two rates are closer together than the band ... can never drift"
        # asserts the opposite of the measurement for a gap that exceeds the band.
        for recorded, observed in (((0, 4), (2, 19)), ((1, 4), (4, 11))):
            gap = abs(observed[0] / observed[1] - recorded[0] / recorded[1])
            self.assertGreater(gap, VERDICT_ROPE)
            detail = required_n_detail(recorded, observed)
            self.assertIsNone(detail["required_n"])
            self.assertEqual(detail["reason"], BUDGET_REASON_BEYOND_BOUND)
            self.assertAlmostEqual(detail["projected_gap"], gap)
            text = budget_text(detail)
            self.assertIn("search bound", text)
            self.assertNotIn("closer together than", text)
            self.assertNotIn("never drift", text)

    def test_a_widened_search_bound_confirms_the_projection(self) -> None:
        # The reason is a claim about the projection, so it is checkable: raising the
        # bound returns a finite n, and the verdict rule agrees at that n.
        recorded, observed = (0, 4), (2, 19)
        beyond = required_n(recorded, observed, limit=REQUIRED_N_SEARCH_LIMIT * 4)
        self.assertIsNotNone(beyond)
        assert beyond is not None
        self.assertGreater(beyond, REQUIRED_N_SEARCH_LIMIT)
        self.assertEqual(classify_verdict(self._at_n(recorded, beyond), self._at_n(observed, beyond)), "DRIFTED")

    def test_a_gap_inside_the_band_still_says_it_can_never_drift(self) -> None:
        detail = required_n_detail((3, 8), (5, 13))
        self.assertIsNone(detail["required_n"])
        self.assertEqual(detail["reason"], BUDGET_REASON_INSIDE_BAND)
        self.assertLessEqual(detail["projected_gap"], VERDICT_ROPE)
        self.assertIn("never drift", budget_text(detail))

    def test_a_missing_denominator_states_that_reason(self) -> None:
        detail = required_n_detail((0, 0), (1, 5))
        self.assertIsNone(detail["required_n"])
        self.assertEqual(detail["reason"], BUDGET_REASON_NO_DENOMINATOR)  # not a claim about the gap
        self.assertIsNone(detail["projected_gap"])
        self.assertIn("no denominator", budget_text(detail))

    @staticmethod
    def _at_n(rate_pair: tuple[int, int], total: int) -> tuple[int, int]:
        hits, denominator = rate_pair
        return (round(hits / denominator * total), total)


if __name__ == "__main__":
    unittest.main()
