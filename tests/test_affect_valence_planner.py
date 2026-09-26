"""The valence channel must reach the actuator it was built for, and be inert when empty.

The channel is the single persisted ``(value, draws)`` reading of how the recent
cycles went. These tests pin the two halves that make it auditable: the tilt is a
pure function of its arguments, and the planner it is passed to records the exact
threshold it applied, so a replay can reconstruct the decision from the journal.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar

from helpers import FakeProvider

from skynet.planner import (
    VALENCE_PRIOR,
    VALENCE_TILT_WEIGHT,
    VALENCE_WINDOW_DRAWS,
    PortfolioPlanner,
    valence_band_occupancy,
    valence_decidable_band,
    valence_tilt,
)
from skynet.reactor import Reactor, ReactorConfig
from skynet.store import StateStore


def _add_tasks(store: StateStore) -> None:
    goal_id = store.add_goal("portfolio", priority=4.0)
    for index in range(4):
        store.add_task(
            f"task {index}",
            goal_id,
            expected_new_fact="fact",
            hypothesis_fingerprint=f"fp-{index}",
            structural_fingerprint=f"sf-{index}",
            area=f"area-{index}",
        )


def _store(directory: str) -> StateStore:
    store = StateStore(Path(directory) / "state.sqlite3")
    _add_tasks(store)
    return store


def _finish(store: StateStore, statuses: list[str]) -> None:
    """Give the store a terminal-run window. The LAST status is the newest run."""
    for index, status in enumerate(statuses):
        store.connection.execute(
            "INSERT INTO runs(run_id, attempt, status, started_at, finished_at, budget) VALUES (?, 1, ?, ?, ?, '{}')",
            (f"run-{index}", status, f"2026-09-25T00:00:{index:02d}Z", f"2026-09-25T00:01:{index:02d}Z"),
        )


class ValenceTiltTests(unittest.TestCase):
    def test_an_empty_channel_is_identity(self) -> None:
        self.assertEqual(valence_tilt(0.1, value=VALENCE_PRIOR, draws=0), 0.1)
        # One observation cannot swing a decision either.
        self.assertLess(abs(valence_tilt(0.1, value=1.0, draws=1) - 0.1), 0.01)

    def test_a_good_window_lowers_the_threshold_and_a_bad_one_raises_it(self) -> None:
        good = valence_tilt(0.1, value=1.0, draws=int(VALENCE_WINDOW_DRAWS))
        bad = valence_tilt(0.1, value=0.0, draws=int(VALENCE_WINDOW_DRAWS))
        self.assertLess(good, 0.1)
        self.assertGreater(bad, 0.1)
        # Bounded: never more than VALENCE_TILT_WEIGHT * base away, and never outside [0, 1].
        self.assertGreaterEqual(good, 0.1 * (1.0 - VALENCE_TILT_WEIGHT))
        self.assertLessEqual(bad, 0.1 * (1.0 + VALENCE_TILT_WEIGHT))

    def test_a_neutral_reading_is_identity_at_any_draw_count(self) -> None:
        for draws in (0, 1, 5, 1000):
            self.assertEqual(valence_tilt(0.1, value=VALENCE_PRIOR, draws=draws), 0.1)


class ChargedChannelReachesTheActuatorTests(unittest.TestCase):
    def test_charge_derives_value_and_draws_from_the_same_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            _finish(store, ["failed"] * 5 + ["completed"] * 15)
            charge = store.charge_affect_valence(window=20)
            self.assertEqual(charge["draws"], 20)
            self.assertAlmostEqual(charge["valence"], 15 / 20)
            self.assertEqual(store.affect_valence(), (0.75, 20))
            # Idempotent: the charge is a function of the window, not a total.
            self.assertEqual(store.charge_affect_valence(window=20), charge)
            store.close()

    def test_an_empty_window_leaves_the_channel_inert(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            self.assertEqual(store.affect_valence(), (VALENCE_PRIOR, 0))
            charge = store.charge_affect_valence(window=20)
            self.assertEqual(charge["draws"], 0)
            self.assertEqual(charge["valence"], VALENCE_PRIOR)
            store.close()

    def test_reactor_selection_journals_the_tilted_threshold_and_the_charge(self) -> None:
        """The whole path, from the ledger's run window to the journal row."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=root / "state.sqlite3", planner_epsilon=0.1))
            _add_tasks(reactor.store)
            _finish(reactor.store, ["failed"] * 5 + ["completed"] * 15)
            planner, work = reactor._plan_select()
            self.assertIsNotNone(work)
            expected = valence_tilt(0.1, value=0.75, draws=20)
            self.assertNotEqual(expected, 0.1)
            self.assertAlmostEqual(planner.threshold, expected)
            planner.record_decision()
            payload = json.loads(
                reactor.store.connection.execute(
                    "SELECT payload FROM event_log WHERE kind='planner_decision' ORDER BY sequence DESC LIMIT 1"
                ).fetchone()["payload"]
            )
            self.assertAlmostEqual(payload["exploration"]["threshold"], expected, places=6)
            self.assertEqual(payload["exploration"]["valence"], {"value": 0.75, "draws": 20})
            reactor.close()

    def test_an_inert_channel_applies_the_untilted_threshold_exactly(self) -> None:
        """An empty ledger must reproduce the historical threshold byte for byte."""
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            planner = PortfolioPlanner(store, epsilon=1.0)
            self.assertEqual(planner.threshold, 1.0)
            self.assertEqual(planner.threshold, planner.epsilon)
            planner.select()
            payload = json.loads(
                store.connection.execute("SELECT payload FROM event_log WHERE kind='planner_decision' ORDER BY sequence DESC LIMIT 1").fetchone()["payload"]
            )
            self.assertEqual(payload["exploration"]["threshold"], 1.0)
            # No charge was ever written, so no valence is journalled.
            self.assertNotIn("valence", payload["exploration"])
            self.assertEqual(store.affect_valence(), (VALENCE_PRIOR, 0))
            store.close()

    def test_the_reactor_charges_before_selecting_so_the_tilt_is_persisted(self) -> None:
        """A cycle with an empty ledger still writes the inert charge, threshold unchanged."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=root / "state.sqlite3"))
            _add_tasks(reactor.store)
            planner, work = reactor._plan_select()
            self.assertIsNotNone(work)
            self.assertEqual(reactor.store.affect_valence()[1], 0)
            self.assertEqual(planner.threshold, reactor.config.planner_epsilon)
            reactor.close()

    def test_the_reactor_charges_the_channel_from_its_configured_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=root / "state.sqlite3", affect_valence_window=20))
            _finish(reactor.store, ["failed"] * 5 + ["completed"] * 15)
            _, work = reactor._plan_select()
            self.assertIsNone(work)  # the fixture has no task, the charge is what matters
            self.assertEqual(reactor.store.affect_valence(), (0.75, 20))
            reactor.close()

    def test_the_window_is_both_the_lookback_and_the_draw_count(self) -> None:
        """The newest window is the one that is read: the same ledger, two windows, two readings."""
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3", affect_valence_window=4))
            _finish(reactor.store, ["failed"] * 16 + ["failed", "completed", "completed", "completed"])
            short = reactor.store.charge_affect_valence(window=reactor.config.affect_valence_window)
            self.assertEqual(short["draws"], 4)
            self.assertAlmostEqual(short["valence"], 0.75)
            long = reactor.store.charge_affect_valence(window=20)
            self.assertEqual(long["draws"], 20)
            self.assertAlmostEqual(long["valence"], 3 / 20)
            reactor.close()


class TheDrawIsComparedAgainstTheTiltedThresholdTests(unittest.TestCase):
    """The one behavioural claim: the charge moves WHICH candidate a draw selects.

    The draw is seeded and durable, so it is pinned here to the single value that
    separates the two arms: with base epsilon 0.1 a draw of 0.11 exploits, and a
    low reading (which raises the threshold to 0.125) explores instead.
    """

    def test_a_low_reading_makes_a_draw_explore_that_the_base_threshold_would_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            top = PortfolioPlanner(store).rank()[0]
            store.next_random = lambda: 0.11  # type: ignore[method-assign]
            inert = PortfolioPlanner(store, epsilon=0.1, valence=(VALENCE_PRIOR, 0))
            self.assertEqual(inert.threshold, 0.1)
            self.assertEqual(inert.select()["task"]["task_id"], top.task_id)
            low = PortfolioPlanner(store, epsilon=0.1, valence=(0.0, 20))
            self.assertGreater(low.threshold, 0.11)
            self.assertNotEqual(low.select()["task"]["task_id"], top.task_id)
            store.close()

    def test_a_high_reading_keeps_the_top_candidate_where_a_low_one_explores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            top = PortfolioPlanner(store).rank()[0]
            store.next_random = lambda: 0.11  # type: ignore[method-assign]
            high = PortfolioPlanner(store, epsilon=0.1, valence=(1.0, 20))
            self.assertLess(high.threshold, 0.11)
            self.assertEqual(high.select()["task"]["task_id"], top.task_id)
            store.close()


class TheJournalCarriesBothArmsOfTheCounterfactualTests(unittest.TestCase):
    """The channel is judged by a replay, so the record must make the replay decidable.

    Measured on the live journal at commit 2507832: the three decision rows that
    carry a charge reproduce their recorded ``chosen_rank`` exactly, but only the
    *tilted* threshold was ever written, so "would this draw have flared with the
    channel absent?" -- the pre-registered falsifier -- could not be computed from
    live data at all, and the tilt band ([0.075, 0.125] around epsilon 0.1) caught
    just 2 of the 65 recorded draws. These tests pin the second key that closes the
    gap, on the record alone.
    """

    @staticmethod
    def _ranks(payload: dict) -> tuple[int, int]:
        """(rank the draw produced, rank the same draw produces with no channel)."""
        ex = payload["exploration"]
        count = len(payload["candidates"])

        def rank(threshold: float) -> int:
            if threshold > 0.0 and count > 1 and ex["draw"] < threshold:
                return 1 + int(ex["draw"] * 10_000) % (count - 1)
            return 0

        return rank(ex["threshold"]), rank(ex["base_threshold"])

    @staticmethod
    def _journal(reactor: Reactor) -> dict:
        payload = reactor.store.connection.execute(
            "SELECT payload FROM event_log WHERE kind='planner_decision' ORDER BY sequence DESC LIMIT 1"
        ).fetchone()["payload"]
        return json.loads(payload)

    def test_the_journal_records_the_untilted_arm_beside_the_tilted_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3", planner_epsilon=0.1))
            _add_tasks(reactor.store)
            _finish(reactor.store, ["failed"] * 5 + ["completed"] * 15)
            planner, work = reactor._plan_select()
            self.assertIsNotNone(work)
            planner.record_decision()
            payload = self._journal(reactor)
            self.assertEqual(payload["exploration"]["base_threshold"], 0.1)
            self.assertAlmostEqual(payload["exploration"]["threshold"], valence_tilt(0.1, value=0.75, draws=20), places=6)
            # The replay of the record reproduces the decision that was really taken.
            self.assertEqual(self._ranks(payload)[0], payload["exploration"]["chosen_rank"])
            reactor.close()

    def test_the_counterfactual_is_decidable_from_the_record_alone(self) -> None:
        """A draw inside the tilt band must show the channel as decisive, on the record only."""
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3", planner_epsilon=0.1))
            _add_tasks(reactor.store)
            # A good window lowers the threshold to 0.08; a draw of 0.09 is above it
            # (exploit) but below the untilted 0.1 (explore). Exactly the case whose
            # verdict the falsifier turns on.
            _finish(reactor.store, ["failed"] * 5 + ["completed"] * 15)
            reactor.store.next_random = lambda: 0.09  # type: ignore[method-assign]
            planner, work = reactor._plan_select()
            self.assertIsNotNone(work)
            planner.record_decision()
            payload = self._journal(reactor)
            with_channel, without_channel = self._ranks(payload)
            self.assertEqual(with_channel, 0)          # the draw did not flare
            self.assertEqual(without_channel, 1)       # with no channel it would have
            self.assertNotEqual(with_channel, without_channel, "the record must make the flip visible")
            self.assertEqual(payload["exploration"]["chosen_rank"], with_channel)
            reactor.close()

    def test_an_empty_channel_writes_both_arms_identical(self) -> None:
        """draws == 0 is the identity, and the record must show it as one."""
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            planner = PortfolioPlanner(store, epsilon=1.0)
            planner.select()
            payload = json.loads(
                store.connection.execute("SELECT payload FROM event_log WHERE kind='planner_decision' ORDER BY sequence DESC LIMIT 1").fetchone()["payload"]
            )
            self.assertEqual(payload["exploration"]["base_threshold"], 1.0)
            self.assertEqual(payload["exploration"]["threshold"], payload["exploration"]["base_threshold"])
            store.close()

    def test_a_row_without_the_base_arm_keeps_its_legacy_shape(self) -> None:
        """The pre-wiring decision rows are not retro-editable, so the writer must not invent the key."""
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            candidates = PortfolioPlanner(store).rank()
            store.record_planner_decision(candidates, candidates[0], draw=0.5, threshold=0.1)
            payload = json.loads(
                store.connection.execute("SELECT payload FROM event_log WHERE kind='planner_decision' ORDER BY sequence DESC LIMIT 1").fetchone()["payload"]
            )
            self.assertEqual(set(payload["exploration"]), {"seed", "draw", "chosen_rank", "threshold"})
            store.close()

class ValenceWindowPinTests(unittest.TestCase):
    def test_the_window_default_is_pinned(self) -> None:
        config = ReactorConfig(state_path=Path("/tmp/unused.sqlite3"))
        self.assertEqual(config.affect_valence_window, 20)
        self.assertEqual(config.planner_epsilon, 0.1)


class ValenceAmplitudeTests(unittest.TestCase):
    """The amplitude is explicit, and it is what bounds the falsifier's sample.

    The channel's reach is not a matter of opinion: the tilt factor is
    ``1 - weight * (2 * effective - 1)``, so the threshold can only land in
    ``[base * (1 - weight), base * (1 + weight)]``. A verdict about the channel
    is a statement about the draws inside that interval, and the interval is
    what the amplitude chooses. These tests pin the reach, the default, and the
    live base of the shipped deployment, so a later replay can be scored against
    a band that is a property of the code rather than of someone's reading.
    """

    def test_decidable_band_is_the_asymptotic_reach_of_the_tilt(self) -> None:
        # The band is the endpoint of the tilt as the confidence goes to one:
        # it is approached, not reached, at a finite draw count.
        base, weight = 0.1, VALENCE_TILT_WEIGHT
        low, high = valence_decidable_band(base, weight=weight)
        self.assertAlmostEqual(low, 0.05, places=9)
        self.assertAlmostEqual(high, 0.15, places=9)
        self.assertGreater(valence_tilt(base, value=1.0, draws=10 ** 7), low)
        self.assertLess(valence_tilt(base, value=0.0, draws=10 ** 7), high)
        self.assertLess(abs(valence_tilt(base, value=1.0, draws=10 ** 7) - low), 1e-4)
        self.assertLess(abs(valence_tilt(base, value=0.0, draws=10 ** 7) - high), 1e-4)
        # Monotone: more draws, more of the nominal reach.
        down = [valence_tilt(base, value=1.0, draws=d) for d in (1, 20, 100, 10 ** 7)]
        self.assertEqual(down, sorted(down, reverse=True))
        # Never wider than the base itself: the channel cannot own the policy.
        self.assertLessEqual(high - low, base + 1e-12)

    def test_the_live_charge_uses_only_half_of_that_reach(self) -> None:
        # A charge's confidence is draws / (draws + VALENCE_WINDOW_DRAWS), so the
        # live charge (one window of draws, i.e. draws == VALENCE_WINDOW_DRAWS)
        # sits at confidence 0.5 and can reach only half the nominal band:
        # [0.075, 0.125] at the shipped base, not [0.05, 0.15]. The deciding
        # interval while the reading sits above the prior is therefore
        # [threshold, 0.1) = [0.075, 0.1) at a maximally high reading -- a
        # quarter of the nominal reach and 0.025 of the draw range.
        base = 0.1
        window = int(VALENCE_WINDOW_DRAWS)
        low, high = valence_decidable_band(base)
        self.assertAlmostEqual(valence_tilt(base, value=1.0, draws=window), 0.075, places=9)
        self.assertAlmostEqual(valence_tilt(base, value=0.0, draws=window), 0.125, places=9)
        self.assertAlmostEqual(valence_tilt(base, value=1.0, draws=window) - low, (base - low) / 2.0, places=9)
        self.assertAlmostEqual(high - valence_tilt(base, value=0.0, draws=window), (high - base) / 2.0, places=9)
        self.assertAlmostEqual(base - valence_tilt(base, value=1.0, draws=window), 0.025, places=9)

    def test_decidable_band_at_the_shipped_defaults_is_a_tenth_of_the_draw_range(self) -> None:
        # The live deployment: epsilon 0.1 (ReactorConfig.planner_epsilon) and
        # amplitude 0.5 (ReactorConfig.affect_valence_tilt_weight). This is the
        # number a sample size has to be computed from.
        low, high = valence_decidable_band(0.1)
        self.assertAlmostEqual(low, 0.05, places=9)
        self.assertAlmostEqual(high, 0.15, places=9)
        self.assertAlmostEqual(high - low, 0.1, places=9)

    def _assert_band(self, weight: float, low: float, high: float) -> None:
        got_low, got_high = valence_decidable_band(0.1, weight=weight)
        self.assertAlmostEqual(got_low, low, places=9)
        self.assertAlmostEqual(got_high, high, places=9)

    def test_amplitude_widens_the_band_linearly_and_is_capped(self) -> None:
        self._assert_band(1.0, 0.0, 0.2)
        self._assert_band(0.25, 0.075, 0.125)
        self._assert_band(0.0, 0.1, 0.1)
        # Above 1 the policy would be replaced by the channel, so the knob is
        # clamped exactly as the tilt clamps it: no config can escape the band.
        self.assertEqual(valence_decidable_band(0.1, weight=4.0), valence_decidable_band(0.1, weight=1.0))

    def test_default_amplitude_is_unchanged_and_reaches_the_planner(self) -> None:
        self.assertEqual(valence_tilt(0.1, value=0.9, draws=20), valence_tilt(0.1, value=0.9, draws=20, weight=VALENCE_TILT_WEIGHT))
        planner = PortfolioPlanner(StubStore(), epsilon=1.0, valence=(0.75, 20))
        self.assertEqual(planner.tilt_weight, VALENCE_TILT_WEIGHT)
        self.assertEqual(planner.threshold, valence_tilt(1.0, value=0.75, draws=20))
        wide = PortfolioPlanner(StubStore(), epsilon=1.0, valence=(0.75, 20), tilt_weight=1.0)
        self.assertLess(wide.threshold, planner.threshold)

    def test_reactor_passes_its_configured_amplitude_to_the_planner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(
                FakeProvider(), {},
                ReactorConfig(state_path=Path(directory) / "state.sqlite3",
                              planner_epsilon=0.1, affect_valence_tilt_weight=1.0),
            )
            planner, _ = reactor._plan_select()
            self.assertEqual(planner.tilt_weight, 1.0)
            self.assertEqual(planner.threshold, valence_tilt(0.1, value=VALENCE_PRIOR, draws=0))


class StubStore:
    """The minimum a planner needs; every method the tilt path touches is inert."""

    def planner_candidates(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def idea_cells(self) -> dict[str, int]:
        return {}


class ValenceAmplitudeConfigTests(unittest.TestCase):
    def test_the_shipped_amplitude_is_the_pinned_constant(self) -> None:
        # The field default, the module constant and the deployment's own base
        # are three places that must agree, or the measured band is wrong.
        config = ReactorConfig(state_path=Path("/tmp/unused.sqlite3"))
        self.assertEqual(config.affect_valence_tilt_weight, VALENCE_TILT_WEIGHT)
        self.assertEqual(VALENCE_TILT_WEIGHT, 0.5)
        low, high = valence_decidable_band(config.planner_epsilon,
                                          weight=config.affect_valence_tilt_weight)
        self.assertAlmostEqual(low, 0.05, places=9)
        self.assertAlmostEqual(high, 0.15, places=9)
        # The reach is a tenth of the draw range, and only its lower half can
        # matter while the reading sits above the prior: that is the whole
        # sample the falsifier has to work with at the shipped defaults.
        self.assertLessEqual(high - low, 0.1 + 1e-9)


class ValenceBandOccupancyTests(unittest.TestCase):
    """The falsifier's sample size must be computable, not estimated by hand.

    The channel is only judged on draws inside its reach, and only a decision
    with two or more candidates can be switched by it. These tests pin both
    filters against the shipped defaults and the live charge, and pin the
    calibrated rate the occupancy has to reach before a verdict is affordable.
    """

    BASE: ClassVar[float] = 0.1
    LIVE: ClassVar[dict[str, Any]] = {"value": 0.9, "draws": 20}

    def _rows(self, draws_with_sizes: list[tuple[float, int]]) -> list[dict[str, Any]]:
        return [{"draw": d, "candidates": [{} for _ in range(n)]} for d, n in draws_with_sizes]

    def test_the_shipped_reach_excludes_every_draw_at_or_above_the_base(self) -> None:
        # A down-tilt cannot raise a threshold, so the deciding interval is
        # [threshold, base): the band above the base is unreachable at any charge.
        rows = self._rows([(0.085, 3), (0.09, 3), (0.1, 3), (0.12, 3), (0.14, 3)])
        occ = valence_band_occupancy(rows, base=self.BASE, weight=VALENCE_TILT_WEIGHT, **self.LIVE)
        self.assertAlmostEqual(occ["threshold"], 0.08, places=9)
        self.assertAlmostEqual(occ["reach"][1], 0.15, places=9)
        self.assertEqual(occ["total"], 5)
        # 0.085 and 0.09 are below the base and above the threshold; a draw at
        # the base itself cannot be moved downward into an exploring rank.
        self.assertEqual(occ["deciding"], 2)
        self.assertEqual(occ["decisive"], 2)

    def test_a_one_candidate_decision_is_in_band_but_cannot_testify(self) -> None:
        # The rank is 1 + int(draw * 10_000) % (len(ranked) - 1) and is taken
        # only when len(ranked) > 1, so on a one-candidate decision the channel
        # changes the threshold and nothing else.
        rows = self._rows([(0.09, 1), (0.09, 2), (0.09, 3)])
        occ = valence_band_occupancy(rows, base=self.BASE, weight=VALENCE_TILT_WEIGHT, **self.LIVE)
        self.assertEqual(occ["deciding"], 3)
        self.assertEqual(occ["decisive"], 2)

    def test_an_empty_history_has_no_share_and_does_not_divide_by_zero(self) -> None:
        occ = valence_band_occupancy([], base=self.BASE, weight=VALENCE_TILT_WEIGHT, **self.LIVE)
        self.assertEqual(occ["total"], 0)
        self.assertEqual(occ["share"], 0.0)
        self.assertEqual(occ["decisive_share"], 0.0)

    def test_an_empty_reading_is_inert_and_leaves_nothing_in_the_band(self) -> None:
        # draws == 0 leaves the threshold at the base, the channel's own neutral
        # element: [base, base) is empty, so an uncharged run has no in-band draw
        # by construction rather than by a small number.
        rows = self._rows([(0.05, 3), (0.09, 3), (0.3, 4)])
        occ = valence_band_occupancy(rows, base=self.BASE, weight=VALENCE_TILT_WEIGHT,
                                     value=VALENCE_PRIOR, draws=0)
        self.assertAlmostEqual(occ["threshold"], self.BASE, places=12)
        self.assertEqual(occ["deciding"], 0)
        self.assertEqual(occ["decisive"], 0)

    def test_the_measured_occupancy_is_the_size_the_calibration_asks_for(self) -> None:
        # Live journal, read at generation 237: 66 exploration rows, reach
        # [0.08, 0.1) at base 0.1 with the live charge, exactly 1 draw inside it
        # and that draw sits on a two-candidate decision, so the channel can
        # switch exactly 1 of the 66 recorded decisions -- about 1980 decisions
        # for a 30-flip verdict. The fixture is synthetic and the same shape, so
        # the assertion does not depend on a database.
        rows = self._rows([(0.092297, 2)] + [(0.2 + i / 200.0, 2) for i in range(65)])
        occ = valence_band_occupancy(rows, base=self.BASE, weight=VALENCE_TILT_WEIGHT, **self.LIVE)
        self.assertEqual(occ["total"], 66)
        self.assertEqual(occ["deciding"], 1)
        self.assertEqual(occ["decisive"], 1)
        self.assertAlmostEqual(occ["share"], 1 / 66, places=9)
        self.assertAlmostEqual(occ["decisive_share"], 1 / 66, places=9)
        # The calibration target is planner_epsilon itself: arXiv:1706.01905
        # section 9.1 sets delta := -log(1 - eps + eps/|A|), the KL divergence of
        # an eps-greedy policy, so a calibrated perturbation moves about eps of
        # the decisions. The shipped base leaves the occupancy an order of
        # magnitude below that, which is the sizing defect this helper makes
        # visible; a wider base reaches it (0.197 at base 0.5).
        epsilon = 0.1
        self.assertLess(occ["decisive_share"], epsilon / 5)
        wider = valence_band_occupancy(rows, base=0.5, weight=VALENCE_TILT_WEIGHT, **self.LIVE)
        self.assertGreater(wider["decisive_share"], epsilon)
        # The literal |draw - epsilon| interquantile width is a property of a
        # uniform draw rather than of the decisions the channel acts on (it is
        # ~0.35, wider than any reach this base can have), so the sizing rule
        # cannot be stated in those terms.
        self.assertGreater(0.353, valence_decidable_band(self.BASE, weight=1.0)[1] - self.BASE)

    def test_the_amplitude_clamp_widens_the_band_but_never_scales_it_unbounded(self) -> None:
        rows = self._rows([(0.05, 3), (0.09, 3)])
        wide = valence_band_occupancy([], base=self.BASE, weight=10.0, **self.LIVE)
        # The amplitude clamps at 1, so the reach is [0.0, 0.2] -- twice the base
        # -- and not the 10x a caller might expect to buy with a bigger number.
        self.assertEqual(wide["reach"], (0.0, 0.2))
        narrow = valence_band_occupancy(rows, base=self.BASE, weight=0.25, **self.LIVE)
        self.assertAlmostEqual(narrow["threshold"], 0.09, places=9)  # 0.1 * (1 - 0.25*0.4)
        self.assertEqual(narrow["deciding"], 0)  # [0.09, 0.1) excludes the draw at 0.09


if __name__ == "__main__":
    unittest.main()
