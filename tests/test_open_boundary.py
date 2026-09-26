"""Tests for the open-boundary subsystem: archive, learnability, diversity, valves."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

from helpers import FakeProvider, FixtureTool, commit_all, git_repo

from skynet import metrics
from skynet.idea_archive import effective_modes, learnability_defect
from skynet.models import Budget, RunRecord, RunStatus
from skynet.planner import PortfolioPlanner
from skynet.reactor import _EXTERNAL_SEEK_FINGERPRINTS, SAFETY_NET_FINGERPRINTS, Reactor, ReactorConfig, _live_external_seek_tasks
from skynet.self_improvement import SelfImprovementManager
from skynet.store import StateStore
from skynet.time import utc_now


def _idea(cell: str, *, quality: float, title: str = "an idea", **extra: object) -> dict[str, object]:
    """A minimal, well-formed archive row for a behavioural descriptor cell."""
    payload: dict[str, object] = {
        "cell_key": cell,
        "subsystem": "general",
        "change_type": "workflow",
        "evidence_source": "own-repo",
        "title": title,
        "quality": quality,
    }
    payload.update(extra)
    return payload


class IdeaArchiveTests(unittest.TestCase):
    def test_schema_has_archive_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            columns = {row["name"] for row in store.connection.execute("PRAGMA table_info(idea_archive)")}
            for expected in ("cell_key", "lineage_depth", "children", "evidence_source", "inspiration_ref"):
                self.assertIn(expected, columns)
            store.close()

    def test_cell_keeps_only_the_best_occupant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            cell = "memory|workflow|own-repo"
            first = store.archive_idea(_idea(cell, quality=0.4, title="first"))
            self.assertIsNotNone(first)
            inferior = store.archive_idea(_idea(cell, quality=0.3, title="second"))
            self.assertIsNone(inferior)
            incumbent = store.best_in_cell(cell)
            self.assertEqual(cast(dict, incumbent)["idea_id"], first)
            better = store.archive_idea(_idea(cell, quality=0.9, title="third"))
            self.assertIsNotNone(better)
            self.assertNotEqual(better, first)
            superseded = store.connection.execute("SELECT status FROM idea_archive WHERE idea_id=?", (first,)).fetchone()[0]
            self.assertEqual(superseded, "superseded")
            self.assertEqual(store.idea_cells()[cell], 1)
            store.close()

    def test_novelty_can_take_a_cell_from_an_equal_quality_incumbent(self) -> None:
        """Exclusive epsilon-dominance: a strictly more novel idea wins a tie.

        arXiv:1708.09251 sec. 3.1.2: a rule that needs the newcomer to be better
        on every objective "prevents most new individuals from being added".
        """
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            cell = "memory|memory-structure|paper"
            stale = store.archive_idea(_idea(cell, quality=1.0, novelty=0.5, title="stale incumbent"))
            self.assertIsNotNone(stale)
            fresher = store.archive_idea(_idea(cell, quality=1.0, novelty=0.9, title="more novel"))
            self.assertIsNotNone(fresher)
            self.assertEqual(cast(dict, store.best_in_cell(cell))["idea_id"], fresher)
            self.assertEqual(store.idea_cells()[cell], 1)
            superseded = store.connection.execute(
                "SELECT status FROM idea_archive WHERE idea_id=?", (stale,)
            ).fetchone()[0]
            self.assertEqual(superseded, "superseded")
            store.close()

    def test_quality_loss_beyond_epsilon_is_still_refused(self) -> None:
        """The relaxation is bounded: novelty must not buy a large quality drop."""
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            cell = "reactor|control-logic|own-repo"
            self.assertIsNotNone(store.archive_idea(_idea(cell, quality=1.0, novelty=0.9, title="incumbent")))
            refused = store.archive_idea(_idea(cell, quality=0.5, novelty=1.0, title="much worse quality"))
            self.assertIsNone(refused)
            self.assertEqual(store.idea_cells()[cell], 1)
            rejection = store.connection.execute(
                "SELECT payload FROM event_log WHERE kind='idea_archive_rejected' ORDER BY sequence DESC LIMIT 1"
            ).fetchone()[0]
            self.assertIn("candidate_novelty", rejection)
            self.assertIn("incumbent_novelty", rejection)
            store.close()

    def test_an_identical_idea_does_not_churn_the_cell(self) -> None:
        """No trade-off exists on an exact tie, so the cell must not rotate."""
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            cell = "tools|workflow|own-repo"
            first = store.archive_idea(_idea(cell, quality=0.9, novelty=0.9, title="original"))
            self.assertIsNotNone(first)
            self.assertIsNone(store.archive_idea(_idea(cell, quality=0.9, novelty=0.9, title="duplicate")))
            self.assertEqual(cast(dict, store.best_in_cell(cell))["idea_id"], first)
            store.close()

    def test_a_degenerate_incumbent_never_locks_a_cell(self) -> None:
        """An incumbent that scores zero on both axes must not hold the cell."""
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            cell = "providers|workflow|own-repo"
            blank = store.archive_idea(_idea(cell, quality=0.0, novelty=0.0, title="blank"))
            self.assertIsNotNone(blank)
            real = store.archive_idea(_idea(cell, quality=0.5, novelty=1.0, title="real idea"))
            self.assertIsNotNone(real)
            self.assertEqual(cast(dict, store.best_in_cell(cell))["idea_id"], real)
            store.close()

    def test_materialize_idea_creates_a_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            goal_id = store.add_goal("open boundary", priority=1.0)
            idea_id = store.archive_idea(_idea("scheduler|control-logic|own-repo", quality=0.5, title="materialize me"))
            self.assertIsNotNone(idea_id)
            task_id = store.materialize_idea(str(idea_id), goal_id)
            self.assertIsNotNone(store.task_work(task_id))
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='idea_materialized'").fetchone()[0],
                1,
            )
            self.assertEqual(
                store.connection.execute("SELECT status FROM idea_archive WHERE idea_id=?", (idea_id,)).fetchone()[0],
                "materialized",
            )
            store.close()

    def test_parent_sampling_is_journalled_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            cells = ("memory|workflow|own-repo", "scheduler|workflow|own-repo", "tools|workflow|own-repo")
            for cell, quality in zip(cells, (0.2, 0.5, 0.8), strict=False):
                self.assertIsNotNone(store.archive_idea(_idea(cell, quality=quality, title=cell)))
            store.next_random()
            before = int(store.connection.execute("SELECT draws FROM rng_state WHERE id=1").fetchone()[0])
            original = store.next_random

            def fixed_draw() -> float:
                original()
                return 0.5

            store.next_random = fixed_draw  # type: ignore[method-assign]
            parents = store.sample_archive_parents(k=2)
            after = int(store.connection.execute("SELECT draws FROM rng_state WHERE id=1").fetchone()[0])
            self.assertEqual(len(parents), 2)
            self.assertEqual(len({str(item["idea_id"]) for item in parents}), 2)
            self.assertEqual(after - before, 2)
            for item in parents:
                children = store.connection.execute(
                    "SELECT children FROM idea_archive WHERE idea_id=?", (str(item["idea_id"]),)
                ).fetchone()[0]
                self.assertEqual(children, 1)
            store.close()

    def test_saturated_idea_is_never_a_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            saturated = store.archive_idea(_idea("memory|workflow|own-repo", quality=1.0, title="saturated"))
            parent = store.archive_idea(_idea("tools|workflow|own-repo", quality=0.5, title="parent"))
            self.assertIsNotNone(saturated)
            self.assertIsNotNone(parent)
            draws = iter([0.1, 0.5, 0.9, 0.25, 0.75])
            store.next_random = lambda: next(draws)  # type: ignore[method-assign]
            for _ in range(5):
                picked = store.sample_archive_parents(k=1)
                self.assertEqual(len(picked), 1)
                self.assertEqual(str(picked[0]["idea_id"]), parent)
                self.assertNotEqual(str(picked[0]["idea_id"]), saturated)
            store.close()


def _learnable(**overrides: object) -> dict[str, object]:
    proposal: dict[str, object] = {
        "title": "Adopt retrieval augmentation for the planner",
        "expected_new_fact": "A retrieval layer reduces repeated selection of the same task",
        "validation": "Measure the drop in repeated task selection across one week",
        "kind": "engineering",
        "inspiration_ref": "",
    }
    proposal.update(overrides)
    return proposal


class ToolProposalArchiveTests(unittest.TestCase):
    def test_tool_proposal_becomes_an_archived_stepping_stone(self) -> None:
        """Real self-improvement traffic feeds the archive, not only planner ideas.

        Nine promoted commits left `idea_archive` empty because archiving ran
        exclusively inside the autonomous planner.
        """
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            reactor.store.create_run(RunRecord("run-1", 1, RunStatus.RUNNING, utc_now(), Budget()))
            reactor.store.append_event(
                "tool_result",
                {
                    "call": {
                        "call_id": "c1",
                        "tool_name": "propose_self_improvement",
                        "arguments": {
                            "hypothesis": {
                                "problem": "the db tool does not name its columns",
                                "expected_behavior": "the model stops guessing column names",
                                "validation": "a query against a real column succeeds",
                            },
                            "changes": [{"path": "skynet/tools.py"}],
                        },
                    },
                    "result": {"ok": True, "proposal_id": "prop-1"},
                },
                "run-1",
            )
            reactor._archive_run_self_improvements("run-1", None, 1.0)
            ideas = reactor.store.archived_ideas(status="active")
            self.assertEqual(len(ideas), 1)
            self.assertEqual(ideas[0]["proposal_id"], "prop-1")
            self.assertEqual(ideas[0]["evidence_source"], "own-repo")
            self.assertEqual(float(ideas[0]["quality"]), 1.0)
            reactor.close()


class LearnabilityTests(unittest.TestCase):
    def test_research_without_citation_is_rejected(self) -> None:
        defect = learnability_defect(_learnable(kind="research"))
        self.assertEqual(defect, "research proposal cites no external source")
        cited = _learnable(kind="research", inspiration_ref="arXiv:2505.22954")
        self.assertIsNone(learnability_defect(cited))

    def test_expected_new_fact_restating_the_title_is_rejected(self) -> None:
        proposal = _learnable(title="Improve planner diversity", expected_new_fact="Improve planner diversity")
        defect = learnability_defect(proposal)
        self.assertEqual(defect, "expected_new_fact restates the title; there is nothing to learn")

    def test_short_validation_is_rejected(self) -> None:
        # A short validation that is not one of the canned no-measurement
        # phrases falls through to the length rule.
        self.assertEqual(
            learnability_defect(_learnable(validation="check it")),
            "validation is not an observable signal",
        )

    def test_validation_naming_no_measurement_is_rejected(self) -> None:
        # "tests pass" names no measurement. It is only two words, so the length
        # rule fires before the explicit no-measurement rule; either way the
        # proposal is rejected.
        self.assertIsNotNone(learnability_defect(_learnable(validation="tests pass")))


class DiversityMetricsTests(unittest.TestCase):
    def test_snapshot_includes_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            diversity = cast(dict, metrics.snapshot(store)["diversity"])
            self.assertEqual(diversity["cells_total"], 144)
            self.assertEqual(diversity["cells_filled"], 0)
            self.assertEqual(diversity["coverage_ratio"], 0.0)
            self.assertEqual(diversity["effective_modes"], 0.0)
            self.assertIsNone(diversity["external_evidence_ratio"])
            store.close()

    def test_coverage_and_external_ratio_grow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.archive_idea(_idea("memory|workflow|own-repo", quality=0.5, evidence_source="own-repo"))
            store.archive_idea(_idea("tools|new-tool|paper", quality=0.5, evidence_source="paper"))
            diversity = metrics.diversity_health(store.connection, "1970-01-01T00:00:00Z")
            self.assertEqual(diversity["cells_filled"], 2)
            self.assertEqual(diversity["external_evidence_ratio"], 0.5)
            store.close()

    def test_effective_modes_single_cell_is_one(self) -> None:
        self.assertEqual(effective_modes([3]), 1.0)


class ArchiveSelectionTests(unittest.TestCase):
    def test_select_materializes_from_archive_when_portfolio_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.add_goal("open boundary", priority=1.0)
            idea_id = store.archive_idea(_idea("scheduler|control-logic|own-repo", quality=0.5, title="archived stepping stone"))
            self.assertIsNotNone(idea_id)
            selected = PortfolioPlanner(store, epsilon=0.0).select()
            self.assertIsNotNone(selected)
            task_id = cast(dict, selected)["task"]["task_id"]
            materialized = store.connection.execute(
                "SELECT task_id, status FROM idea_archive WHERE idea_id=?", (idea_id,)
            ).fetchone()
            self.assertEqual(materialized["status"], "materialized")
            self.assertEqual(materialized["task_id"], task_id)
            store.close()


class ExternalSeekTests(unittest.TestCase):
    def test_valve_creates_a_research_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(
                FakeProvider(),
                {"webfetch": Mock()},
                ReactorConfig(
                    state_path=Path(directory) / "state.sqlite3",
                    external_seek_every_generations=1,
                    external_seek_cooldown_seconds=0.0,
                ),
            )
            reactor.store.add_goal("g", priority=1.0)
            goals = reactor.store.active_work()[0]
            state = reactor.store.state()
            selected = reactor._seek_external_evidence(state, goals)
            self.assertIsNotNone(selected)
            self.assertEqual(cast(dict, selected)["task"]["area"], "research")
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='external_seek_created'").fetchone()[0],
                1,
            )
            reactor.close()

    def test_valve_is_silent_without_senses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(
                FakeProvider(),
                {},
                ReactorConfig(
                    state_path=Path(directory) / "state.sqlite3",
                    external_seek_every_generations=1,
                    external_seek_cooldown_seconds=0.0,
                ),
            )
            reactor.store.add_goal("g", priority=1.0)
            goals = reactor.store.active_work()[0]
            state = reactor.store.state()
            self.assertIsNone(reactor._seek_external_evidence(state, goals))
            self.assertEqual(
                reactor.store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='external_senses_unavailable'").fetchone()[0],
                1,
            )
            reactor.close()

    def test_valve_respects_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(
                FakeProvider(),
                {"webfetch": Mock()},
                ReactorConfig(
                    state_path=Path(directory) / "state.sqlite3",
                    external_seek_every_generations=1,
                    external_seek_cooldown_seconds=3600.0,
                ),
            )
            reactor.store.add_goal("g", priority=1.0)
            goals = reactor.store.active_work()[0]
            state = reactor.store.state()
            self.assertIsNotNone(reactor._seek_external_evidence(state, goals))
            self.assertIsNone(reactor._seek_external_evidence(state, goals))
            reactor.close()

    def test_valve_opens_on_cadence_even_with_ready_work(self) -> None:
        """The valve is cadence-driven: a busy portfolio must not seal it.

        An exhaustion-only trigger was unreachable because the bounded fallback
        always produced something, so external_seek never fired.
        """
        class WebfetchStub:
            """A real tool shape: `tick` serialises the schema, a Mock cannot be."""

            name = "webfetch"
            schema = {"type": "function", "function": {"name": name, "description": "stub", "parameters": {"type": "object", "additionalProperties": False}}}  # noqa: RUF012 - Tool protocol reads schema as an instance property

            def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
                return {"ok": True}

        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(
                FakeProvider(),
                {"webfetch": WebfetchStub()},
                ReactorConfig(
                    state_path=Path(directory) / "state.sqlite3",
                    external_seek_every_generations=1,
                    external_seek_cooldown_seconds=0.0,
                ),
            )
            goal_id = reactor.store.add_goal("g", priority=1.0)
            reactor.store.add_task("ready work", goal_id)
            state = reactor.store.state()
            state.generation = 1
            reactor.store.set_state(state)
            self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
            self.assertEqual(
                reactor.store.connection.execute(
                    "SELECT COUNT(*) FROM event_log WHERE kind='external_seek_created'"
                ).fetchone()[0],
                1,
            )
            reactor.close()


    def test_valve_does_not_stack_a_second_instance_of_itself(self) -> None:
        """An outward look already waiting must not be created again.

        Measured on the live ledger before this check existed (2026-09-25,
        read-only): the valve fired 56 times in six days and at the moment of
        every one of those firings a prior external-seek task was still live
        under the same stable template fingerprints, so asking for one
        outstanding instance would have suppressed 56 of 56 creations and
        starved nothing. Seven were simultaneously pending, the oldest four days
        old, against a mean 0.9 h from creation to a terminal state.
        """
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(
                FakeProvider(),
                {"webfetch": Mock()},
                ReactorConfig(
                    state_path=Path(directory) / "state.sqlite3",
                    external_seek_every_generations=1,
                    external_seek_cooldown_seconds=0.0,
                ),
            )
            reactor.store.add_goal("g", priority=1.0)
            goals = reactor.store.active_work()[0]
            state = reactor.store.state()
            first = reactor._seek_external_evidence(state, goals)
            self.assertIsNotNone(first)
            first_id = cast(dict, first)["task"]["task_id"]
            self.assertIsNone(reactor._seek_external_evidence(state, goals))
            self.assertEqual(
                reactor.store.connection.execute(
                    "SELECT COUNT(*) FROM event_log WHERE kind='external_seek_created'"
                ).fetchone()[0],
                1,
            )
            suppressed = reactor.store.connection.execute(
                "SELECT payload FROM event_log WHERE kind='external_seek_suppressed' "
                "ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            self.assertIsNotNone(suppressed)
            self.assertEqual(json.loads(suppressed["payload"])["live_task_id"], first_id)
            reactor.close()

    def test_valve_reopens_once_the_outstanding_look_is_settled(self) -> None:
        """The check gates duplication, not the valve: a settled task frees it.

        Every terminal status frees the valve, because none of them is
        selectable again; a suppressed duplicate must not seal the boundary.
        """
        for settled in ("completed", "cancelled", "failed", "blocked"):
            with self.subTest(settled=settled), tempfile.TemporaryDirectory() as directory:
                reactor = Reactor(
                    FakeProvider(),
                    {"webfetch": Mock()},
                    ReactorConfig(
                        state_path=Path(directory) / "state.sqlite3",
                        external_seek_every_generations=1,
                        external_seek_cooldown_seconds=0.0,
                    ),
                )
                reactor.store.add_goal("g", priority=1.0)
                goals = reactor.store.active_work()[0]
                state = reactor.store.state()
                first = reactor._seek_external_evidence(state, goals)
                self.assertIsNotNone(first)
                task_id = cast(dict, first)["task"]["task_id"]
                reactor.store.connection.execute(
                    "UPDATE tasks SET status=?, updated_at=? WHERE task_id=?",
                    (settled, utc_now(), task_id),
                )
                self.assertIsNotNone(reactor._seek_external_evidence(state, goals))
                reactor.close()

    def test_a_stale_stack_is_repaired_down_to_the_newest_live_instance(self) -> None:
        """The exemption keeps one reachable instance, not every stacked copy.

        Measured on a replica of the live ledger at generation 202 (read-only):
        6 pending copies of the external-seek template shared one terminal
        hypothesis, planner_candidates scored all 6 'terminal' and
        PortfolioPlanner.rank() returned none of them, while the valve's own
        duplicate check counted them. The repair must reduce the family to the
        newest live instance instead of exempting the whole family forever.

        Note what this does NOT claim: the valve stays shut while that survivor
        is pending with a terminal hypothesis, because nothing selects such a row
        except the firing that created it. That residual sealing is recorded for
        the next cycle rather than asserted as fixed here.
        """
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(
                FakeProvider(),
                {"webfetch": Mock()},
                ReactorConfig(
                    state_path=Path(directory) / "state.sqlite3",
                    external_seek_every_generations=1,
                    external_seek_cooldown_seconds=0.0,
                ),
            )
            reactor.store.add_goal("g", priority=1.0)
            goals = reactor.store.active_work()[0]
            state = reactor.store.state()
            first = reactor._seek_external_evidence(state, goals)
            first_id = cast(dict, first)["task"]["task_id"]
            # Two older copies of the same template survive as pending rows and
            # the hypothesis behind them is terminal, exactly as on the ledger.
            stale_ids = []
            for offset, when in enumerate(("2026-09-21T11:58:57Z", "2026-09-23T02:19:53Z")):
                stale = reactor.store.add_task(
                    f"stale copy {offset}",
                    goals[0]["goal_id"],
                    expected_new_fact="stale",
                    hypothesis_fingerprint=_EXTERNAL_SEEK_FINGERPRINTS[0],
                    structural_fingerprint=_EXTERNAL_SEEK_FINGERPRINTS[1],
                    area="research",
                )
                reactor.store.connection.execute("UPDATE tasks SET created_at=? WHERE task_id=?", (when, stale))
                stale_ids.append(stale)
            reactor.store.connection.execute(
                "UPDATE tasks SET created_at='2026-09-25T11:48:11Z' WHERE task_id=?", (first_id,)
            )
            reactor.store.connection.execute(
                "UPDATE hypotheses SET status='completed' WHERE fingerprint=?", (_EXTERNAL_SEEK_FINGERPRINTS[0],)
            )
            reactor.store.connection.commit()
            self.assertEqual(
                len(_live_external_seek_tasks(reactor.store.connection)), 3, "the stack is live before the repair"
            )
            repaired = reactor.store.repair_status_consistency(exclude_fingerprints=SAFETY_NET_FINGERPRINTS)
            self.assertEqual(repaired["tasks"], 2, "both stale copies are retired")
            statuses = dict(reactor.store.connection.execute(
                "SELECT task_id, status FROM tasks WHERE hypothesis_fingerprint=?", (_EXTERNAL_SEEK_FINGERPRINTS[0],)
            ).fetchall())
            self.assertEqual(statuses[first_id], "pending", "the newest instance keeps the safety net")
            for stale_id in stale_ids:
                self.assertEqual(statuses[stale_id], "cancelled")
            surviving = _live_external_seek_tasks(reactor.store.connection)
            self.assertEqual([row["task_id"] for row in surviving], [first_id], "one instance of the family stays live")
            # Pinned residual behaviour: the survivor is pending with a terminal
            # hypothesis, so it still holds the valve shut. Retiring it is a
            # separate change with its own evidence.
            self.assertIsNone(reactor._seek_external_evidence(state, goals))
            reactor.close()


class ProviderHealthTests(unittest.TestCase):
    def test_degradation_is_flagged_in_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            for _ in range(6):
                store.append_event("provider_error_classified", {"category": "timeout", "retryable": True, "cooldown_seconds": 30.0})
            data = metrics.snapshot(store, since_days=1.0)
            self.assertTrue(data["providers"]["degraded"])
            self.assertEqual(data["providers"]["degraded_categories"], {"timeout": 6})
            self.assertIn("DEGRADED", metrics.format_report(data))
            store.close()

    def test_a_healthy_chain_is_not_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.append_event("provider_error_classified", {"category": "invalid_request", "retryable": False, "cooldown_seconds": 0.0})
            data = metrics.snapshot(store, since_days=1.0)
            self.assertFalse(data["providers"]["degraded"])
            store.close()


class CriterionEpochTests(unittest.TestCase):
    def test_reactor_records_the_epoch_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(
                FakeProvider(),
                {"fixture_tool": FixtureTool()},
                ReactorConfig(state_path=Path(directory) / "state.sqlite3", criterion_epoch_generations=1),
            )
            self.assertEqual(reactor.tick("test"), RunStatus.COMPLETED)
            rows = reactor.store.connection.execute("SELECT payload FROM event_log WHERE kind='criterion_epoch_started'").fetchall()
            self.assertEqual(len(rows), 1)
            reactor.close()

    def test_metrics_exposes_the_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            state = store.state()
            state.generation = 250
            store.set_state(state)
            with patch.dict("os.environ", {"SKYNET_CRITERION_EPOCH_GENERATIONS": "100"}):
                data = metrics.snapshot(store, since_days=1.0)
            self.assertEqual(data["agent"]["criterion_epoch"], 2)
            store.close()


class MetaLoopProtectionTests(unittest.TestCase):
    def test_meta_loop_paths_are_warned_not_hard_protected(self) -> None:
        # Owner decision: the meta-contour is open. A first submission
        # of a meta-loop change is warned (never held or rejected), and only the
        # identical resubmission proceeds through the full gate.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_repo(root)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            commit_all(root, "base")
            manager = SelfImprovementManager(root, root.parent / "worktrees")
            metadata = {
                "hypothesis": {
                    "problem": "p",
                    "expected_behavior": "e",
                    "evidence": "ev",
                    "validation": "v",
                    "rollback_condition": "r",
                }
            }
            result = manager.propose_files(
                {},
                (sys.executable, "-c", "pass"),
                changes=[{"path": "skynet/metrics.py", "operation": "create", "content": "x = 1\n"}],
                metadata=metadata,
            )
            self.assertFalse(result["ok"], result)
            self.assertTrue(result["warned"], result)
            self.assertEqual(result["status"], "warned_protected")
            self.assertIn("skynet/metrics.py", cast(list, result["protected_paths"]))


if __name__ == "__main__":
    unittest.main()
