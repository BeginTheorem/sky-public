"""Tests for the heuristic provider-selection index.

The scorer is pure, unwired, and publishes no measured results, so these tests
pin the arithmetic and the documented properties (a cooling provider zeroes the
index; the factors stay in range; selection is deterministic) rather than any
performance claim.
"""

from __future__ import annotations

import math
import unittest

from skynet.scheduler import (
    AREA_CAPABILITIES,
    FACTOR_FLOOR,
    FACTOR_WEIGHTS,
    PROVIDER_CAPABILITIES,
    PairScore,
    ProviderIndexScorer,
    ProviderSpec,
    TaskSpec,
)


class FactorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = ProviderIndexScorer()

    def test_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(FACTOR_WEIGHTS.values()), 1.0)

    def test_unseen_provider_sits_at_the_prior(self) -> None:
        provider = ProviderSpec(provider_id="p", name="unknown")
        self.assertAlmostEqual(self.scorer.completion_factor(provider), 0.5)

    def test_completion_rises_with_success_rate(self) -> None:
        weak = ProviderSpec(provider_id="w", name="unknown", success_count=1, failure_count=9)
        strong = ProviderSpec(provider_id="s", name="unknown", success_count=9, failure_count=1)
        self.assertLess(self.scorer.completion_factor(weak), self.scorer.completion_factor(strong))
        self.assertGreater(self.scorer.completion_factor(strong), 0.5)
        self.assertLess(self.scorer.completion_factor(weak), 0.5)

    def test_availability_is_zero_while_cooling(self) -> None:
        cooling = ProviderSpec(provider_id="c", name="unknown", cooldown_remaining=12.0, success_count=5, failure_count=0)
        self.assertEqual(self.scorer.availability_factor(cooling), 0.0)
        assert self.scorer.availability_factor(cooling) != self.scorer.completion_factor(cooling)

    def test_urgency_rises_with_priority_and_attempts(self) -> None:
        low = self.scorer.urgency_factor(TaskSpec(task_id="a", priority=0.0, attempts=0))
        priority = self.scorer.urgency_factor(TaskSpec(task_id="b", priority=4.0, attempts=0))
        retries = self.scorer.urgency_factor(TaskSpec(task_id="c", priority=0.0, attempts=5))
        self.assertLess(low, priority)
        self.assertLess(low, retries)
        self.assertAlmostEqual(priority, 0.6)
        self.assertAlmostEqual(retries, 0.4)

    def test_urgency_and_fit_stay_inside_their_band(self) -> None:
        top = self.scorer.urgency_factor(TaskSpec(task_id="t", priority=99.0, attempts=99))
        self.assertLessEqual(top, 1.0)
        self.assertGreaterEqual(top, FACTOR_FLOOR)
        floor = self.scorer.urgency_factor(TaskSpec(task_id="t", priority=0.0, attempts=0))
        self.assertGreaterEqual(floor, FACTOR_FLOOR)

    def test_fit_uses_jaccard_overlap(self) -> None:
        engineering = TaskSpec(task_id="t", area="engineering")
        overlapping = ProviderSpec(provider_id="p", name="provider", capabilities=frozenset({"code", "refactor"}))
        identical = ProviderSpec(provider_id="p", name="provider", capabilities=AREA_CAPABILITIES["engineering"])
        disjoint = ProviderSpec(provider_id="p", name="provider", capabilities=frozenset({"embed"}))
        # engineering = {code, refactor, test, debug, api}; overlap = 2, union = 5.
        self.assertAlmostEqual(self.scorer.fit_factor(engineering, overlapping), 2 / 5)
        self.assertAlmostEqual(self.scorer.fit_factor(engineering, identical), 1.0)
        self.assertAlmostEqual(self.scorer.fit_factor(engineering, disjoint), FACTOR_FLOOR)

    def test_an_unknown_area_and_provider_are_a_weak_match(self) -> None:
        # Neither vocabulary is ever empty: an unknown area resolves to the
        # "general" area vocabulary and an unlisted provider to {"general"}, so
        # the pair lands on the floor rather than a special-case 1.0.
        task = TaskSpec(task_id="t", area="unknown-area")
        provider = ProviderSpec(provider_id="p", name="unknown-provider", capabilities=frozenset())
        self.assertAlmostEqual(self.scorer.fit_factor(task, provider), FACTOR_FLOOR)

    def test_area_and_provider_vocabularies_are_mapped(self) -> None:
        self.assertEqual(self.scorer.area_capabilities("ENGINEERING"), AREA_CAPABILITIES["engineering"])
        self.assertEqual(self.scorer.area_capabilities("mystery"), AREA_CAPABILITIES["general"])
        self.assertIn("reasoning", self.scorer.provider_capabilities("nvidia_ultra"))
        self.assertIn("fast", self.scorer.provider_capabilities("zai_glm47_flash"))
        self.assertIn("local", self.scorer.provider_capabilities("ollama_gemma"))
        self.assertEqual(
            {fragment for fragment, _ in PROVIDER_CAPABILITIES},
            {"nemotron", "nvidia", "openrouter", "zai", "glm", "ollama", "deepseek"},
        )


class ScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = ProviderIndexScorer()

    def test_index_is_the_weighted_geometric_mean(self) -> None:
        task = TaskSpec(task_id="t", area="engineering", priority=4.0, attempts=5)
        provider = ProviderSpec(provider_id="p", name="nvidia_ultra", success_count=3, failure_count=1)
        scored = self.scorer.score(task, provider)
        expected = math.exp(
            FACTOR_WEIGHTS["completion"] * math.log(scored.components["completion"])
            + FACTOR_WEIGHTS["availability"] * math.log(scored.components["availability"])
            + FACTOR_WEIGHTS["urgency"] * math.log(scored.components["urgency"])
            + FACTOR_WEIGHTS["fit"] * math.log(scored.components["fit"])
        )
        self.assertAlmostEqual(scored.index, expected)
        self.assertGreater(scored.index, 0.0)
        self.assertLessEqual(scored.index, 1.0)

    def test_a_cooling_provider_scores_exactly_zero(self) -> None:
        task = TaskSpec(task_id="t", area="engineering", priority=4.0)
        provider = ProviderSpec(provider_id="p", name="nvidia_ultra", cooldown_remaining=300.0, success_count=100, failure_count=0)
        scored = self.scorer.score(task, provider)
        self.assertEqual(scored.index, 0.0)
        self.assertEqual(scored.components["availability"], 0.0)

    def test_only_availability_may_zero_the_index(self) -> None:
        # A genuinely weak factor is floored, not zeroed: otherwise a provider
        # that merely looks like a poor fit would silently become unusable.
        task = TaskSpec(task_id="t", area="memory", priority=0.0, attempts=0)
        provider = ProviderSpec(provider_id="p", name="zai_glm47_flash", success_count=0, failure_count=20)
        scored = self.scorer.score(task, provider)
        self.assertGreater(scored.index, 0.0)
        self.assertGreaterEqual(scored.components["fit"], FACTOR_FLOOR)


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = ProviderIndexScorer()

    def test_select_returns_the_best_pair(self) -> None:
        task = TaskSpec(task_id="t", area="engineering", priority=4.0)
        healthy = ProviderSpec(provider_id="healthy", name="nvidia_ultra", success_count=8, failure_count=1)
        cooling = ProviderSpec(provider_id="cooling", name="zai_glm47_flash", cooldown_remaining=90.0, success_count=8, failure_count=1)
        chosen = self.scorer.select([task], [cooling, healthy])
        assert chosen is not None
        self.assertEqual(chosen.provider_id, "healthy")

    def test_select_none_when_every_provider_is_cooling(self) -> None:
        task = TaskSpec(task_id="t", area="engineering")
        cooling = ProviderSpec(provider_id="c", name="zai_glm47_flash", cooldown_remaining=60.0)
        self.assertIsNone(self.scorer.select([task], [cooling]))

    def test_select_none_without_inputs(self) -> None:
        self.assertIsNone(self.scorer.select([], []))
        self.assertIsNone(self.scorer.select([], [ProviderSpec(provider_id="p", name="zai_glm47_flash")]))
        self.assertIsNone(self.scorer.select([TaskSpec(task_id="t")], []))

    def test_ties_are_resolved_by_id_regardless_of_order(self) -> None:
        providers = [ProviderSpec(provider_id="b", name="zai_glm47_flash"), ProviderSpec(provider_id="a", name="zai_glm47_flash")]
        tasks = [TaskSpec(task_id="z", area="engineering", priority=4.0), TaskSpec(task_id="a", area="engineering", priority=4.0)]
        first = self.scorer.select(tasks, providers)
        second = self.scorer.select(list(reversed(tasks)), list(reversed(providers)))
        assert first is not None and second is not None
        self.assertEqual((first.task_id, first.provider_id), ("a", "a"))
        self.assertEqual((second.task_id, second.provider_id), ("a", "a"))

    def test_a_more_urgent_task_wins_a_tie_on_provider(self) -> None:
        provider = ProviderSpec(provider_id="p", name="nvidia_ultra", success_count=5, failure_count=5)
        calm = TaskSpec(task_id="calm", area="engineering", priority=0.0, attempts=0)
        urgent = TaskSpec(task_id="urgent", area="engineering", priority=4.0, attempts=4)
        chosen = self.scorer.select([calm, urgent], [provider])
        assert chosen is not None
        self.assertEqual(chosen.task_id, "urgent")

    def test_pair_score_is_json_friendly(self) -> None:
        provider = ProviderSpec(provider_id="p", name="zai_glm47_flash")
        scored: PairScore = self.scorer.score(TaskSpec(task_id="t"), provider)
        self.assertEqual(sorted(scored.components), sorted(FACTOR_WEIGHTS))


if __name__ == "__main__":
    unittest.main()
