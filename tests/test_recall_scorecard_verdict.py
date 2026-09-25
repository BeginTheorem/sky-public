"""The scorecard's REPRODUCED/DRIFTED verdict is a difference interval, not equality.

A published figure that moved by one scenario is not evidence of drift: the five
fixture situations cannot separate a real ranking movement from fixture-to-fixture
variation. These tests pin the rule's three verdicts, the difference interval that
decides them, its stated inputs, the marginal-overlap test it replaces, and the
measured effect on the figures frozen in ``skynet.recall_scorecard``.
"""

from __future__ import annotations

import unittest

from skynet.recall_scorecard import (
    REFERENCE_GOVERNING_TOP3,
    REFERENCE_GOVERNING_TOP20,
    REFERENCE_GOVERNING_VERDICTS,
    REFERENCE_HUB_TOP1,
    REFERENCE_HUB_TOP2,
    REQUIRED_N_SEARCH_LIMIT,
    SITUATIONS,
    VERDICT_MIN_SCENARIOS,
    VERDICT_ROPE,
    classify_verdict,
    difference_interval,
    direction_detail,
    harness_supply,
    required_n,
    required_n_detail,
    verdict_detail,
    wilson_interval,
)

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

    @staticmethod
    def _at_n(rate_pair: tuple[int, int], total: int) -> tuple[int, int]:
        hits, denominator = rate_pair
        return (round(hits / denominator * total), total)


if __name__ == "__main__":
    unittest.main()
