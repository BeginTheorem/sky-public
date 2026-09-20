"""Tests for the open-boundary subsystem: archive, learnability, diversity, valves."""
from __future__ import annotations

import tempfile
import unittest
from datetime import UTC
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

from helpers import FakeProvider, FixtureTool

from skynet import metrics
from skynet.idea_archive import effective_modes, learnability_defect
from skynet.models import Budget, RunRecord, RunStatus
from skynet.planner import PortfolioPlanner
from skynet.reactor import Reactor, ReactorConfig
from skynet.self_improvement import SelfImprovementManager
from skynet.store import StateStore
from skynet.time import utc_now


def _iso(seconds: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(seconds, tz=UTC).isoformat()

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

        Archiving that runs only inside the autonomous planner leaves
        `idea_archive` empty for promoted changes.
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
            diversity = metrics.diversity_health(store.connection, _iso(0.0))
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

        An exhaustion-only trigger is unreachable whenever the bounded fallback
        always produces something.
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
    def test_meta_loop_paths_are_gate_protected(self) -> None:
        for path in (
            "skynet/planner.py",
            "skynet/idea_archive.py",
            "skynet/autonomous_planner.py",
            "skynet/metrics.py",
        ):
            self.assertTrue(SelfImprovementManager._is_gate_protected(path), path)

if __name__ == "__main__":
    unittest.main()
