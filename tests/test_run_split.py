"""The consolidation pass keeps a run of its own: kind, verdict, cost, and the readers
that must not confuse it with an episode.

The pass used to be bookkept inside the episode's run, so its cost was charged to work
that had already finished and a degraded pass left no row to count. It has its own row
now, which means every reader that counts *episodes* has to say so -- these tests are
what keeps that true.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from helpers import FakeProvider, FixtureTool

from skynet import metrics, reporting, status_card
from skynet.memory import MemoryLoop
from skynet.model_contracts import MAX_MODEL_REASKS
from skynet.models import Budget, RunRecord, RunStatus
from skynet.reactor import Reactor, ReactorConfig
from skynet.store import StateStore


def open_run(store: StateStore, run_id: str, kind: str, status: RunStatus) -> None:
    """One finished durable run of the given kind, with a matching result row."""
    store.create_run(
        RunRecord(run_id=run_id, attempt=1, status=RunStatus.RUNNING, started_at="2026-10-16T10:00:00Z", budget=Budget(steps=1, tokens=1000, seconds=60, output_tokens=100), kind=kind)
    )
    store.finish_run(run_id, status)


def result_of(store: StateStore, run_id: str) -> dict:
    row = store.run_result(run_id)
    assert row is not None, f"no result row for {run_id}"
    return row


class MemoryPassIsItsOwnRunTest(unittest.TestCase):
    def test_a_cycle_writes_an_episode_and_a_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {"fixture_tool": FixtureTool()}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            self.assertEqual(reactor.tick("split-test"), RunStatus.COMPLETED)
            rows = {row["kind"]: row for row in reactor.store.connection.execute("SELECT run_id, kind, status, steps, usage_tokens, budget FROM runs")}
            self.assertEqual(sorted(rows), ["build", "memory"])
            build, memory = rows["build"], rows["memory"]
            self.assertEqual(build["status"], "completed")
            self.assertEqual(memory["status"], "completed")
            self.assertEqual(memory["steps"], 1, "the pass counts its own model calls")
            self.assertEqual(memory["usage_tokens"], 1, "and carries its own cost")
            # The row must state the ceiling the loop actually enforces, and the payload is
            # digested by default (see skynet/memory.py), so that is the digested ceiling.
            derived = int(MemoryLoop.digested_ceiling_tokens(output_tokens=reactor.config.budget.output_tokens) * 1.1)
            self.assertEqual(json.loads(memory["budget"])["tokens"], derived, "the row states the ceiling the loop enforces, plus headroom")
            started = json.loads(reactor.store.connection.execute("SELECT payload FROM event_log WHERE kind='memory_run_started'").fetchone()["payload"])
            self.assertEqual(started["source_run"], build["run_id"], "the pass names the episode it consolidated")
            self.assertEqual(result_of(reactor.store, build["run_id"])["failure"], "", "the episode's own verdict is untouched")
            reactor.close()

    def test_a_pass_that_cannot_run_is_named_without_failing_the_episode(self) -> None:
        # The loop reports a dead provider as a degraded pass rather than raising, so
        # the realistic outcome is `needs_recovery` -- a pass whose evidence is kept
        # for a later reader and which is never retried on its own.
        class DeadProviderDuringMemory(FakeProvider):
            def complete(self, messages, *, max_tokens, tools=()):
                if self.calls >= 2:
                    raise RuntimeError("provider went away during consolidation")
                return super().complete(messages, max_tokens=max_tokens, tools=tools)

        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(DeadProviderDuringMemory(), {"fixture_tool": FixtureTool()}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            self.assertEqual(reactor.tick("degraded-pass"), RunStatus.COMPLETED, "a pass that cannot run does not fail the episode")
            rows = {row["kind"]: row for row in reactor.store.connection.execute("SELECT run_id, kind, status FROM runs")}
            self.assertEqual(rows["build"]["status"], "completed")
            self.assertEqual(rows["memory"]["status"], "needs_recovery")
            self.assertEqual(result_of(reactor.store, rows["memory"]["run_id"])["failure"], "", "the fault ledger stays about providers")
            self.assertIn("DEGRADED memory pass", result_of(reactor.store, rows["memory"]["run_id"])["report"])
            reactor.close()

    def test_a_raising_pass_is_recorded_failed(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(FakeProvider(), {"fixture_tool": FixtureTool()}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            with patch.object(reactor.memory_loop, "consolidate", side_effect=ValueError("episode could not be serialised")):
                self.assertEqual(reactor.tick("raising-pass"), RunStatus.COMPLETED)
            rows = {row["kind"]: row for row in reactor.store.connection.execute("SELECT run_id, kind, status FROM runs")}
            self.assertEqual(rows["memory"]["status"], "failed")
            self.assertIn("Memory pass failed: episode could not be serialised", result_of(reactor.store, rows["memory"]["run_id"])["report"])
            self.assertEqual(result_of(reactor.store, rows["memory"]["run_id"])["failure"], "")
            reactor.close()


class ReadersKeepThePassApartTest(unittest.TestCase):
    def store_with_runs(self, directory: str) -> StateStore:
        store = StateStore(Path(directory) / "state.sqlite3")
        open_run(store, "ep-1", "build", RunStatus.COMPLETED)
        store.commit_run_result("ep-1", RunStatus.COMPLETED, "ok", 3, 100, "")
        open_run(store, "ep-2", "build", RunStatus.NEEDS_RECOVERY)
        store.commit_run_result("ep-2", RunStatus.NEEDS_RECOVERY, "no provider", 2, 900, "all providers failed")
        open_run(store, "pass-1", "memory", RunStatus.NEEDS_RECOVERY)
        store.commit_run_result("pass-1", RunStatus.NEEDS_RECOVERY, "DEGRADED memory pass", 1, 40, "")
        return store

    def test_outcome_mix_counts_episodes_only_and_names_the_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.store_with_runs(directory)
            outcomes = metrics.outcome_mix(store.connection, "1970-01-01T00:00:00Z")
            self.assertEqual(outcomes["total_runs"], 2, "a pass is not another episode")
            self.assertEqual(outcomes["by_status"], {"completed": 1, "needs_recovery": 1})
            self.assertEqual(outcomes["tokens_total"], 1000)
            self.assertEqual(outcomes["memory"]["runs"], 1)
            self.assertEqual(outcomes["memory"]["tokens_total"], 40)
            self.assertEqual(outcomes["memory"]["by_status"], {"needs_recovery": 1}, "a degraded pass is visible without inflating the episode rate")
            store.close()

    def test_a_pass_does_not_break_the_provider_failure_streak(self) -> None:
        # The streak is read newest-first: a pass row in front of a failing episode
        # used to reset it to zero and silence the lockout escalation.
        with tempfile.TemporaryDirectory() as directory:
            store = self.store_with_runs(directory)
            store.commit_run_result("ep-3", RunStatus.NEEDS_RECOVERY, "no provider", 1, 10, "all providers failed")
            store.close()
            reactor = Reactor(FakeProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            self.assertEqual(reactor._consecutive_provider_failures(), 2)
            reactor.close()

    def test_the_valence_window_measures_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.store_with_runs(directory)
            window = store.valence_window(window=10)
            self.assertEqual(window["stats"]["labelled_cases"], 2, "the pass is not a draw on the progress channel")
            self.assertEqual(window["stats"]["completed"], 1)
            store.close()

    def test_the_owner_run_stream_lists_episodes_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.store_with_runs(directory)
            run_ids = sorted(row["run_id"] for row in reporting._finished_runs(store, 10))
            self.assertEqual(run_ids, ["ep-1", "ep-2"], "the pass is reported by /status and /metrics, not as an episode")
            store.close()

    def test_the_card_names_the_pass_on_its_own_line(self) -> None:
        self.assertIn("прогонов ещё не было", status_card._memory_pass_line([]))
        line = status_card._memory_pass_line([{"status": "needs_recovery", "usage_tokens": 40, "finished_at": "2026-10-16T10:00:00Z"}])
        self.assertIn("Консолидация:", line)
        self.assertIn("с деградацией", line)
        self.assertIn("токенов", line)

    def test_the_card_reads_the_kind_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.store_with_runs(directory)
            kinds = {row["kind"] for row in status_card._runs(store, 5)}
            self.assertEqual(kinds, {"build", "memory"})
            store.close()


class PlanPassIsItsOwnRunTest(unittest.TestCase):
    def reactor(self, directory: str) -> Reactor:
        return Reactor(FakeProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))

    def test_the_pass_row_carries_the_planners_own_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self.reactor(directory)
            run_id = reactor._open_plan_run(generation=7, trigger="test")
            row = reactor.store.connection.execute("SELECT kind, status, last_phase, budget FROM runs WHERE run_id=?", (run_id,)).fetchone()
            self.assertEqual(row["kind"], "plan")
            self.assertEqual(row["status"], "running")
            self.assertEqual(row["last_phase"], "plan")
            budget = json.loads(row["budget"])
            self.assertEqual(budget["seconds"], reactor.autonomous_planner.timeout_seconds, "the row states the deadline the planner was given")
            # The token figure is the enforced input cap plus output room, with the same
            # 10% headroom the memory pass gets: a soft budget, because crossing our own
            # accounting is an event to record, not work to kill.
            planner = reactor.autonomous_planner
            self.assertEqual(budget["tokens"], int((planner.max_input_chars // 3 + planner.output_tokens) * 1.1))
            # Steps name the worst case the pass may actually spend: one reply, its bounded
            # parse re-asks, and one reconnaissance round each.
            self.assertEqual(budget["steps"], 1 + MAX_MODEL_REASKS + planner.MAX_TOOL_ROUNDS)
            started = json.loads(reactor.store.connection.execute("SELECT payload FROM event_log WHERE kind='plan_run_started'").fetchone()["payload"])
            self.assertEqual(started, {"generation": 7, "trigger": "test"})
            reactor.close()

    def test_the_verdict_follows_the_planner_status(self) -> None:
        # `needs_recovery` for a pass that reached no decision, `completed` for one
        # that decided (even when the decision was "nothing selectable"), `failed`
        # only when the pass raised.
        with tempfile.TemporaryDirectory() as directory:
            reactor = self.reactor(directory)
            reactor._planner_provider.calls = 2
            reactor._planner_provider.usage_tokens = 1500
            cases = {"completed": "completed", "no_work": "completed", "invalid_response": "needs_recovery", "provider_error": "needs_recovery", "raised": "failed"}
            for planner_status, expected in cases.items():
                run_id = reactor._open_plan_run(generation=1, trigger="test")
                reactor._close_plan_run(run_id, planner_status=planner_status, proposals=0, report=f"Planning pass {planner_status}")
                row = reactor.store.connection.execute("SELECT status, steps, usage_tokens FROM runs WHERE run_id=?", (run_id,)).fetchone()
                self.assertEqual(row["status"], expected, f"planner_status={planner_status}")
                self.assertEqual((row["steps"], row["usage_tokens"]), (2, 1500), "the pass reports what the recorder measured")
                result = result_of(reactor.store, run_id)
                self.assertEqual(result["failure"], "", "the fault ledger stays about providers")
                self.assertIn(planner_status, result["report"])
                finished = json.loads(reactor.store.connection.execute("SELECT payload FROM event_log WHERE kind='plan_run_finished' AND run_id=?", (run_id,)).fetchone()["payload"])
                self.assertEqual(finished["planner_status"], planner_status)
                self.assertEqual(finished["usage_tokens"], 1500)
            reactor.close()

    def test_the_wrapper_opens_and_closes_one_pass(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            reactor = self.reactor(directory)
            reactor.autonomous_planner.last_status = "no_work"
            with patch.object(reactor, "_autonomous_planning_pass", return_value=([], "no novel work")):
                task_ids, reason = reactor._run_autonomous_planning(reactor.store.state(), [], "test")
            self.assertEqual((task_ids, reason), ([], "no novel work"))
            self.assertIsNotNone(reactor._last_plan_run_id, "the episode can name the pass it consumed")
            run_id = reactor._last_plan_run_id or ""
            result = result_of(reactor.store, run_id)
            self.assertEqual(result["status"], "completed")
            self.assertIn("nothing selectable", result["report"])
            plan_runs = reactor.store.connection.execute("SELECT COUNT(*) FROM runs WHERE kind='plan'").fetchone()[0]
            self.assertEqual(plan_runs, 1)
            reactor.close()

    def test_a_raising_pass_is_failed_and_still_raises(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            reactor = self.reactor(directory)
            with patch.object(reactor, "_autonomous_planning_pass", side_effect=RuntimeError("planner exploded")), self.assertRaises(RuntimeError):
                reactor._run_autonomous_planning(reactor.store.state(), [], "test")
            row = reactor.store.connection.execute("SELECT run_id, status FROM runs WHERE kind='plan'").fetchone()
            self.assertEqual(row["status"], "failed")
            result = result_of(reactor.store, row["run_id"])
            self.assertIn("unexpected error", result["report"])
            self.assertEqual(result["failure"], "")
            reactor.close()

    def test_the_readers_keep_the_plan_pass_apart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reactor = self.reactor(directory)
            run_id = reactor._open_plan_run(generation=3, trigger="test")
            reactor._planner_provider.calls = 1
            reactor._planner_provider.usage_tokens = 700
            reactor._close_plan_run(run_id, planner_status="completed", proposals=2, report="Planning pass completed")
            outcomes = metrics.outcome_mix(reactor.store.connection, "1970-01-01T00:00:00Z")
            self.assertEqual(outcomes["total_runs"], 0, "a planning pass is not an episode")
            self.assertEqual(outcomes["plan"]["runs"], 1)
            self.assertEqual(outcomes["plan"]["by_status"], {"completed": 1})
            self.assertEqual(outcomes["plan"]["tokens_total"], 700)
            self.assertEqual(reporting._finished_runs(reactor.store, 10), [], "the owner's run stream is episodes")
            kept = reactor.store.connection.execute("SELECT COUNT(*) FROM runs WHERE kind='build'").fetchone()[0]
            self.assertEqual(kept, 0)
            self.assertIn("План:", status_card._plan_pass_line([{"status": "needs_recovery", "usage_tokens": 700, "finished_at": "2026-10-16T10:00:00Z"}]))
            reactor.close()


if __name__ == "__main__":
    unittest.main()
