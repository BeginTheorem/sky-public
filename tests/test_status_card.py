"""The status card: alive / busy / stuck / needs him / code honest — and honest about gaps."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
import unittest
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from skynet.models import Budget, RunRecord, RunStatus
from skynet.status_card import card, expected_sensors, host_facts, sensor_names
from skynet.store import StateStore

SENSOR_COMMANDS = (
    "/opt/agent/bin/arxiv-mcp-server",
    "/opt/agent/bin/duckduckgo-mcp-server",
    "/opt/agent/bin/github-mcp-server stdio --read-only",
    "/opt/agent/bin/playwright-mcp --output-dir /tmp/skynet-playwright",
)
ALL_SENSORS = (0, 1, 2, 3)
# What the card compares against when a test wants the historical four: (process marker, label).
SENSORS = (("arxiv", "arxiv"), ("duckduckgo", "ddg"), ("github", "github"), ("playwright", "playwright"))
SENSOR_ENV = {
    "SKYNET_MCP_ARXIV_COMMAND": "/opt/agent/bin/arxiv-mcp-server",
    "SKYNET_MCP_DDG_COMMAND": "/opt/agent/bin/duckduckgo-mcp-server",
    "SKYNET_MCP_GITHUB_COMMAND": "/opt/agent/bin/github-mcp-server stdio --read-only",
    "SKYNET_MCP_PLAYWRIGHT_COMMAND": "/opt/agent/bin/playwright-mcp",
}


def _probe(responses: dict[str, str]) -> Callable[[list[str]], str]:
    """A stand-in for systemctl/git: the first fragment found in the argv wins."""

    def probe(argv: list[str]) -> str:
        joined = " ".join(argv)
        for needle, value in responses.items():
            if needle in joined:
                return value
        return ""

    return probe


def _proc_tree(root: Path, *, uptime: float, sensors: tuple[int, ...]) -> None:
    """A fake /proc with only what the card reads: uptime, one PID and its children."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "uptime").write_text(f"{uptime} 0.00\n", encoding="utf-8")
    task = root / "4242" / "task" / "4242"
    task.mkdir(parents=True)
    children = [7100 + index for index in range(len(sensors))]
    (task / "children").write_text(" ".join(str(pid) for pid in children), encoding="utf-8")
    for position, pid in enumerate(children):
        (root / str(pid)).mkdir()
        (root / str(pid) / "cmdline").write_bytes(SENSOR_COMMANDS[sensors[position]].encode())


def _iso(moment: float) -> str:
    return datetime.fromtimestamp(moment, tz=UTC).isoformat().replace("+00:00", "Z")


def _seed_running_run(store: StateStore, *, now: float, steps: int = 7, tokens: int = 118_000) -> str:
    run_id = "8fb3a129-1719-4879-809b-018e7728bcda"
    store.create_run(RunRecord(run_id, 1, RunStatus.RUNNING, _iso(now - 240), Budget()))
    store.touch_run(run_id, "waiting_for_provider", steps=steps, usage_tokens=tokens)
    store.append_event("run_started", {"observations": []}, run_id)
    store.append_event(
        "decision_record",
        {"selected_work": {"kind": "task", "task": {"task_id": "task-1", "title": "Fix the reporting pipeline"}}},
        run_id,
    )
    store.append_event("tool_call", {"call_id": "call-1", "tool_name": "bash", "arguments": {}}, run_id)
    return run_id


def _unit_responses(*, active: bool = True, pid: int = 4242, next_elapse: str = "3h 8min 40s") -> dict[str, str]:
    state = "active\nSubState=running\n" if active else "inactive\nSubState=dead\n"
    return {
        "skynet.service": f"ActiveState={state}MainPID={pid}\nActiveEnterTimestampMonotonic=6400000000\n",
        "NextElapseUSecMonotonic": f"NextElapseUSecMonotonic={next_elapse}\n",
        "rev-parse": "abc1234deadbeef\n",
    }


def _card(
    store: StateStore,
    directory: Path,
    *,
    responses: dict[str, str],
    sensors: tuple[int, ...] = ALL_SENSORS,
    expected: tuple[tuple[str, str], ...] = SENSORS,
    use_env: bool = False,
) -> str:
    proc = directory / "proc"
    _proc_tree(proc, uptime=10_000.0, sensors=sensors)
    repo = directory / "repo"
    repo.mkdir(exist_ok=True)
    (repo / ".deployed-commit").write_text("abc1234deadbeef\n", encoding="utf-8")
    return card(
        store,
        service="skynet.service",
        repo=repo,
        state_dir=directory,
        registry_path=directory / "proposals.json",
        probe=_probe(responses),
        proc_root=proc,
        sensors=None if use_env else expected,
    )


class HostFactsTest(unittest.TestCase):
    def test_reads_unit_age_sensors_deploy_and_timer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / "proc"
            _proc_tree(proc, uptime=10_000.0, sensors=(0, 1))
            facts = host_facts(
                "skynet.service",
                root,
                probe=_probe(_unit_responses()),
                proc_root=proc,
                uptime=10_000.0,
                sensors=SENSORS,
            )
            self.assertTrue(facts.active)
            self.assertEqual(facts.unit, "active/running")
            self.assertAlmostEqual(facts.uptime_seconds, 3_600.0, delta=1.0)
            self.assertEqual(facts.pid, 4242)
            self.assertEqual(facts.sensors, ["arxiv", "ddg"])
            self.assertEqual(facts.deployed, "", "no marker in this repo means no claim")
            self.assertEqual(facts.head, "abc1234")
            self.assertTrue(facts.revive_armed)
            assert facts.revive_seconds is not None
            self.assertAlmostEqual(facts.revive_seconds, 1320.0, delta=1.0)

    def test_a_disarmed_timer_is_not_a_countdown(self) -> None:
        facts = host_facts(
            "skynet.service",
            Path("."),
            probe=_probe(_unit_responses(next_elapse="infinity")),
            uptime=10_000.0,
        )
        self.assertIs(facts.revive_armed, False)
        self.assertIsNone(facts.revive_seconds)

    def test_an_unreadable_timer_is_unknown_not_disarmed(self) -> None:
        facts = host_facts(
            "skynet.service",
            Path("."),
            probe=_probe(_unit_responses(next_elapse="")),
            uptime=10_000.0,
        )
        self.assertIsNone(facts.revive_armed, "an empty property must not read as 'not armed'")
        self.assertIsNone(facts.revive_seconds)

    def test_no_proc_entry_means_no_sensors(self) -> None:
        facts = host_facts(
            "skynet.service",
            Path("."),
            probe=_probe(_unit_responses(active=False, pid=0)),
            proc_root=Path("/nonexistent-proc"),
            uptime=0.0,
        )
        self.assertFalse(facts.active)
        self.assertEqual(facts.sensors, [])


class CardTest(unittest.TestCase):
    def test_live_run_names_the_task_the_step_and_the_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed_running_run(store, now=time.time())
            text = _card(store, Path(directory), responses=_unit_responses())
            self.assertIn("🧠 SkyNet · активна", text)
            self.assertIn("прогон #8fb3a129 идёт", text)
            self.assertIn("Задача: Fix the reporting pipeline", text)
            self.assertIn("Шаг 7 · фаза: ждёт провайдера · токенов 118k", text)
            self.assertIn("Прогресс: последний шаг", text)
            self.assertIn("Сенсоры 4/4: arxiv, ddg, github, playwright", text)
            self.assertIn("Деплой: HEAD abc1234 = работает · очередь канала 0", text)
            self.assertIn("Тик воскрешения: через 22м", text)
            self.assertNotIn("Ждёт тебя", text, "nothing is waiting, so the card must not invent it")
            store.close()

    def test_a_missing_sensor_is_named_not_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed_running_run(store, now=time.time())
            text = _card(store, Path(directory), responses=_unit_responses(), sensors=(0, 1))
            self.assertIn("Сенсоры 2/4: arxiv, ddg", text)
            self.assertIn("нет github, playwright", text)
            store.close()

    def test_a_stalled_run_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            now = time.time()
            run_id = _seed_running_run(store, now=now)
            store.connection.execute("UPDATE event_log SET created_at = ? WHERE run_id = ?", (_iso(now - 2400), run_id))
            text = _card(store, Path(directory), responses=_unit_responses())
            self.assertIn("⚠️ нет движения 40м", text)
            self.assertIn("похоже, встала", text)
            store.close()

    def test_waiting_line_appears_only_when_something_waits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed_running_run(store, now=time.time())
            store.raise_alert("no_progress_stall", {"last_run": "run-1"})
            text = _card(store, Path(directory), responses=_unit_responses())
            self.assertIn("Ждёт тебя: 1 оповещение", text)
            store.close()

    def test_a_down_unit_reads_as_down_and_says_why_sensors_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            text = _card(store, Path(directory), responses=_unit_responses(active=False, pid=0), sensors=())
            self.assertIn("стоит (inactive/dead)", text)
            self.assertIn("прогонов нет", text)
            self.assertIn("Сенсоры: недоступны — процесс не запущен", text)
            store.close()

    def test_deploy_drift_is_called_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed_running_run(store, now=time.time())
            responses = _unit_responses()
            responses["rev-parse"] = "9999999deadbeef\n"
            text = _card(store, Path(directory), responses=responses)
            self.assertIn("⚠️ код разошёлся: работает abc1234, HEAD 9999999", text)
            store.close()

    def test_an_empty_ledger_still_renders_every_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            text = _card(store, Path(directory), responses=_unit_responses(), sensors=())
            self.assertIn("Провайдеры: цепочка", text)
            self.assertIn("Память 0", text)
            self.assertNotIn("Прошлые:", text, "an empty ledger has no history to show")
            store.close()

    def test_a_paragraph_title_is_clipped_to_one_short_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            run_id = _seed_running_run(store, now=time.time())
            long_title = "Fix " + "a very long sentence " * 20
            store.connection.execute(
                "UPDATE event_log SET payload = ? WHERE run_id = ? AND kind = 'decision_record'",
                (
                    json.dumps({"selected_work": {"kind": "task", "task": {"task_id": "t", "title": long_title}}}),
                    run_id,
                ),
            )
            text = _card(store, Path(directory), responses=_unit_responses())
            subject = next(line for line in text.splitlines() if line.startswith("Задача: "))
            self.assertLessEqual(len(subject), len("Задача: ") + 160)
            self.assertTrue(subject.endswith("…"), subject)
            store.close()


@contextlib.contextmanager
def _mcp_env(values: dict[str, str]) -> Iterator[None]:
    """Exactly these SKYNET_MCP_* keys for the duration — the rest of the environment stays."""
    managed = [key for key in os.environ if key.startswith("SKYNET_MCP_")]
    with patch.dict(os.environ, {}, clear=False):
        for key in managed:
            os.environ.pop(key, None)
        os.environ.update(values)
        yield


class ParkedSensorTest(unittest.TestCase):
    """A parked sensor is not a missing sense: the card follows the config, not a memory of it."""

    def test_the_expected_list_comes_from_the_config(self) -> None:
        with _mcp_env(SENSOR_ENV):
            self.assertEqual(
                expected_sensors(),
                (
                    ("arxiv-mcp-server", "arxiv"),
                    ("duckduckgo-mcp-server", "ddg"),
                    ("github-mcp-server", "github"),
                    ("playwright-mcp", "playwright"),
                ),
                "the marker is the executable's basename, because that is what /proc shows",
            )

    def test_a_parked_sensor_leaves_the_list(self) -> None:
        with _mcp_env({**SENSOR_ENV, "SKYNET_MCP_PLAYWRIGHT_DISABLED": "1"}):
            expected = expected_sensors()
        self.assertEqual([label for _, label in expected], ["arxiv", "ddg", "github"])

    def test_the_reactor_and_the_card_read_one_config(self) -> None:
        from skynet.cli import _mcp_config

        with _mcp_env({**SENSOR_ENV, "SKYNET_MCP_PLAYWRIGHT_DISABLED": "1"}):
            self.assertEqual(sorted(_mcp_config()), ["arxiv", "ddg", "github"])
            self.assertEqual([label for _, label in expected_sensors()], sorted(_mcp_config()))

    def test_a_card_with_a_parked_sensor_counts_three_of_three(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with _mcp_env({**SENSOR_ENV, "SKYNET_MCP_PLAYWRIGHT_DISABLED": "1"}):
                text = _card(store, Path(directory), responses=_unit_responses(), sensors=(0, 1, 2), use_env=True)
            self.assertIn("Сенсоры 3/3: arxiv, ddg, github", text)
            store.close()

    def test_a_card_without_a_readable_config_makes_no_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            with _mcp_env({}):
                text = _card(store, Path(directory), responses=_unit_responses(), sensors=(0,), use_env=True)
            self.assertIn("Сенсоры:", text)
            self.assertIn("конфиг не читается", text, "with no config the card says it cannot compare")
            store.close()


class PassesOnTheCardTest(unittest.TestCase):
    """A cycle writes three runs; the card keeps the episode and names each pass."""

    def test_each_pass_gets_its_own_line_and_never_the_episode_s(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            now = time.time()
            _seed_running_run(store, now=now)
            for kind, status, tokens in (("plan", RunStatus.COMPLETED, 900), ("memory", RunStatus.NEEDS_RECOVERY, 1500)):
                run_id = f"{kind}-pass-run"
                store.create_run(RunRecord(run_id, 1, RunStatus.RUNNING, _iso(now - 120), Budget(), kind=kind))
                store.touch_run(run_id, kind, steps=1, usage_tokens=tokens)
                store.commit_run_result(run_id, status, f"{kind} pass", 1, tokens, "")
                store.finish_run(run_id, status)
            text = _card(store, Path(directory), responses=_unit_responses())
            self.assertIn("План: ", text)
            self.assertIn("успешно", text)
            self.assertIn("Консолидация: ", text)
            self.assertIn("с деградацией", text, "a degraded pass says so in words")
            self.assertIn("идёт", text, "the running episode is still the current run, not a pass")
            store.close()


# The marker set `expected_sensors()` derives from the live config: the basename of each
# SKYNET_MCP_*_COMMAND executable. `arxiv` and `ddg` are /bin/sh shims that exec an interpreter,
# so their children's command lines no longer carry this spelling — which is how a card reported
# "1/3" while both servers were answering.
SHIM_SENSORS = (
    ("arxiv-mcp-server", "arxiv"),
    ("duckduckgo-mcp-server", "ddg"),
    ("github-mcp-server", "github"),
    ("playwright-mcp", "playwright"),
)
SHIM_CMDLINE = "{python}\0-c\0import sys; from {module} import main; sys.exit(main())\0"
# A neutral interpreter path: the marker match below is about the *spelling* of the module
# name, and a real host path here would leak the deployment into the public export scan.
SHIM_PYTHON = "/opt/agent/share/skynet-mcp-venv/bin/python"


class ShimmedSensorTest(unittest.TestCase):
    """A console script that execs an interpreter loses its own name in /proc."""

    def _fake_proc(self, directory: Path, cmdlines: list[str]) -> Path:
        proc = directory / "proc"
        task = proc / "4242" / "task" / "4242"
        task.mkdir(parents=True)
        children = [7100 + index for index in range(len(cmdlines))]
        (task / "children").write_text(" ".join(str(pid) for pid in children), encoding="utf-8")
        for position, command in enumerate(cmdlines):
            child = proc / str(children[position])
            child.mkdir()
            (child / "cmdline").write_bytes(command.encode())
        return proc

    def _shim(self, module: str) -> str:
        return SHIM_CMDLINE.format(python=SHIM_PYTHON, module=module)

    def test_a_shimmed_sense_is_seen_by_its_module_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc = self._fake_proc(
                Path(directory),
                [
                    self._shim("arxiv_mcp_server"),
                    self._shim("duckduckgo_mcp_server.server"),
                    "/opt/agent/bin/github-mcp-server\0stdio\0--read-only\0",
                ],
            )
            found = sensor_names(4242, proc_root=proc, sensors=SHIM_SENSORS)
        self.assertEqual(found, ["arxiv", "ddg", "github"], "a shimmed sense is running, not blind")

    def test_a_sense_that_is_really_absent_is_not_invented(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc = self._fake_proc(Path(directory), [self._shim("duckduckgo_mcp_server.server")])
            found = sensor_names(4242, proc_root=proc, sensors=SHIM_SENSORS)
        self.assertEqual(found, ["ddg"], "the spelling fix must not invent a sense that is gone")


if __name__ == "__main__":
    unittest.main()
