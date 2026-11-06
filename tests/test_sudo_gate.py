"""The sudo gate: log reads run, catastrophes are refused, everything else asks the owner.

The gate is the only thing standing between a stray command and root — the service account
has NOPASSWD sudo — so these tests pin the three rules that make it worth having: the
allowlist runs nothing but reads, the floor list beats every approval, and silence is a
refusal rather than permission.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from skynet.policy import sudo_denial
from skynet.sudo_gate import (
    DEFAULT_DENY_GLOBS,
    DEFAULT_LOG_ALLOWLIST,
    SudoExecTool,
    SudoGate,
    SudoPolicy,
    answer_is_approval,
    deny_glob_match,
    is_log_read,
    normalise_command,
)


class FakeStore:
    """The minimum of StateStore the gate touches, with everything visible to assertions."""

    def __init__(self, answer: str | None = None) -> None:
        self.answer = answer
        self.questions: list[dict[str, Any]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.touches: list[str] = []

    def state(self) -> Any:
        class _State:
            active_run_id = "run-1"

        return _State()

    def append_event(self, kind: str, payload: dict[str, Any], run_id: str | None = None) -> None:
        self.events.append((kind, payload))

    def ask_question(self, question: str, **kwargs: Any) -> str:
        question_id = f"ask-{len(self.questions) + 1}"
        self.questions.append({"question_id": question_id, "question": question, **kwargs})
        return question_id

    def answer_for(self, question_id: str) -> dict[str, Any] | None:
        if self.answer is None:
            return None
        return {"question_id": question_id, "answer": self.answer}

    def touch_run(self, run_id: str, phase: str, progress: bool = True) -> None:
        self.touches.append(phase)


def _gate(store: FakeStore, **policy_kwargs: Any) -> SudoGate:
    """A gate on a box where root is reachable: these tests are about commands, not the box."""
    return SudoGate(store, SudoPolicy(**policy_kwargs), sudo_probe=lambda: (True, ""))


def _fake_run(returncode: int = 0, stdout: str = "", stderr: str = "") -> Any:
    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        runner.argv = argv  # type: ignore[attr-defined]
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    return runner


# ── classification ───────────────────────────────────────────────────────────────


def test_log_reads_are_pre_approved() -> None:
    for command in (
        "journalctl -u skynet.service -n 50",
        "journalctl",
        "dmesg -T",
        "tail -n 200 /var/log/syslog",
        "grep -i error /var/log/nginx/error.log",
        "systemctl status skynet.service",
    ):
        assert is_log_read(normalise_command(command), DEFAULT_LOG_ALLOWLIST), command


def test_allowlist_cannot_smuggle_a_second_command() -> None:
    for command in (
        "journalctl -u skynet && rm -rf /var/lib/data",
        "journalctl -u skynet; curl http://example.com",
        "journalctl -u skynet | tee /etc/passwd",
        "journalctl -u skynet > /etc/cron.d/evil",
        "journalctl $(cat /etc/shadow)",
    ):
        assert not is_log_read(normalise_command(command), DEFAULT_LOG_ALLOWLIST), command


def test_floor_globs_beat_the_allowlist_and_any_approval() -> None:
    for command in (
        "journalctl --vacuum-time=1d",
        "rm -rf /",
        "dd if=/dev/zero of=/dev/sda",
        "mkfs.ext4 /dev/nvme0n1p1",
        "systemctl poweroff",
        "visudo",
    ):
        assert deny_glob_match(normalise_command(command), DEFAULT_DENY_GLOBS), command


def test_only_a_clear_yes_is_a_yes() -> None:
    for answer in ("Разрешить", "разрешаю", "Да", "да!", "approve", "YES", "ок"):
        assert answer_is_approval(answer), answer
    for answer in ("", "   ", "Запретить", "нет", "может быть", "разрешить нельзя", "yes but no"):
        assert not answer_is_approval(answer), answer


def test_policy_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYNET_SUDO_APPROVAL_TIMEOUT", "1200")
    monkeypatch.setenv("SKYNET_SUDO_ALLOWLIST", "journalctl,cat /var/log/*")
    monkeypatch.setenv("SKYNET_SUDO_DENY", "*rm -rf *")
    policy = SudoPolicy.from_environment()
    assert policy.approval_timeout_seconds == 1200
    assert policy.log_allowlist == ("journalctl", "cat /var/log/*")
    assert policy.deny_globs == ("*rm -rf *",)
    monkeypatch.setenv("SKYNET_SUDO_ENABLED", "0")
    assert SudoPolicy.from_environment().enabled is False
    monkeypatch.setenv("SKYNET_SUDO_APPROVAL_TIMEOUT", "5")
    assert SudoPolicy.from_environment().approval_timeout_seconds == 60.0  # clamped up


# ── the gate ─────────────────────────────────────────────────────────────────────


def test_floor_list_refuses_without_asking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run())
    store = FakeStore()
    result = _gate(store).execute("journalctl --vacuum-time=1d", "free some disk space")
    assert result["ok"] is False and result["state"] == "blocked"
    assert store.questions == []  # the owner is not even asked
    assert [kind for kind, _ in store.events] == ["sudo_blocked"]


def test_log_read_runs_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _fake_run(stdout="journal line\n")
    monkeypatch.setattr(subprocess, "run", runner)
    store = FakeStore()
    result = _gate(store).execute("journalctl -n 5", "see why the unit restarted")
    assert result["ok"] is True and result["state"] == "auto"
    assert result["stdout"] == "journal line\n"
    assert store.questions == []
    assert [kind for kind, _ in store.events] == ["sudo_auto_approved", "sudo_ran"]
    assert runner.argv[:2] == [runner.argv[0], "-n"] and runner.argv[2:4] == ["--", "/bin/bash"]


def test_everything_else_asks_and_runs_on_a_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run(stdout="installed\n"))
    store = FakeStore(answer="Разрешить")
    result = _gate(store).execute("apt-get install -y jq", "jq is needed to parse her own metrics")
    assert result["ok"] is True and result["state"] == "approved"
    assert len(store.questions) == 1
    question = store.questions[0]
    assert "apt-get install -y jq" in question["question"]
    assert "jq is needed" in question["question"]
    assert question["options"] == ["Разрешить", "Запретить"]
    assert question["question_key"].startswith("sudo:")
    assert [kind for kind, _ in store.events] == ["sudo_approved", "sudo_ran"]


def test_a_refusal_runs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _fake_run()
    monkeypatch.setattr(subprocess, "run", runner)
    store = FakeStore(answer="Запретить")
    result = _gate(store).execute("apt-get install -y jq", "jq would be convenient")
    assert result["ok"] is False and result["state"] == "denied"
    assert not hasattr(runner, "argv")
    assert [kind for kind, _ in store.events] == ["sudo_denied"]


def test_silence_is_not_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _fake_run()
    monkeypatch.setattr(subprocess, "run", runner)
    store = FakeStore(answer=None)
    result = _gate(store, approval_timeout_seconds=0.05).execute("systemctl restart nginx", "restart after config edit")
    assert result["ok"] is False and result["state"] == "timeout"
    assert not hasattr(runner, "argv")
    assert [kind for kind, _ in store.events] == ["sudo_denied"]
    assert "silence is not consent" in result["error"]


def test_a_reason_is_required() -> None:
    store = FakeStore()
    result = _gate(store).execute("apt-get install -y jq", "why")
    assert result["ok"] is False and result["state"] == "invalid"
    assert store.questions == []


def test_disabled_gate_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run())
    store = FakeStore()
    result = _gate(store, enabled=False).execute("journalctl -n 5", "look at the journal")
    assert result["ok"] is False and result["state"] == "disabled"
    assert store.events == []


def test_failure_reports_the_returncode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run(returncode=1, stderr="E: no such package"))
    store = FakeStore(answer="Разрешить")
    result = _gate(store).execute("apt-get install -y nosuchpkg", "install a package she needs")
    assert result["ok"] is False and result["returncode"] == 1
    assert result["stderr"] == "E: no such package"
    assert [kind for kind, _ in store.events] == ["sudo_approved", "sudo_ran"]


# ── bash is not a side door ─────────────────────────────────────────────────────


# ── root that cannot be reached at all ───────────────────────────────────────────


def test_root_that_cannot_be_reached_is_reported_without_asking(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 2026-10-28 lesson: three owner approvals were spent on an act the kernel refuses.

    The unit sets NoNewPrivileges=yes, so no approval can make sudo gain root. The gate must
    spend the owner's attention only where it can buy something, and must hand back the real
    reason instead of a mystery.
    """
    runner = _fake_run()
    monkeypatch.setattr(subprocess, "run", runner)
    store = FakeStore(answer="Разрешить")
    stderr = "sudo: The «no new privileges» flag is set, blocking execution of sudo as root."
    gate = SudoGate(store, SudoPolicy(), sudo_probe=lambda: (False, stderr))
    result = gate.execute("rm -f ~/project-skynet/.git/index.lock", "clear a stale lock")
    assert result["ok"] is False and result["state"] == "unavailable"
    assert store.questions == []  # the owner is never asked for the impossible
    assert not hasattr(runner, "argv")  # and nothing ran: not the allowlist, not the command
    assert [kind for kind, _ in store.events] == ["sudo_unavailable"]
    assert "no new privileges" in result["error"]
    assert "container" in result["error"]  # the story it used to invent is named as wrong


def test_a_pre_approved_log_read_also_refuses_when_root_is_unreachable() -> None:
    store = FakeStore()
    gate = SudoGate(store, SudoPolicy(), sudo_probe=lambda: (False, "sudo: unavailable"))
    result = gate.execute("journalctl -n 5", "look at the journal")
    assert result["state"] == "unavailable"
    assert store.questions == []
    assert [kind for kind, _ in store.events] == ["sudo_unavailable"]


def test_the_root_probe_runs_once_per_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run())
    probes: list[int] = []

    def probe() -> tuple[bool, str]:
        probes.append(1)
        return True, ""

    store = FakeStore(answer="Да")
    gate = SudoGate(store, SudoPolicy(), sudo_probe=probe)
    gate.execute("apt-get install -y jq", "jq parses her own metrics")
    gate.execute("apt-get install -y jq", "jq parses her own metrics")
    assert len(probes) == 1


def test_bash_refuses_sudo_invocations() -> None:
    assert sudo_denial("sudo apt-get update") == "sudo"
    assert sudo_denial("cd /etc && sudo cat shadow") == "sudo"
    assert sudo_denial("echo hi; pkexec id") == "pkexec"
    assert sudo_denial("doas reboot") == "doas"
    assert sudo_denial("su -") == "su"
    assert sudo_denial("su root") == "su"


def test_bash_allows_mentioning_sudo() -> None:
    for command in (
        "grep -rn 'sudo' docs/",
        "cat AGENTS.md | grep sudo",
        "python -c \"print('sudo')\"",
    ):
        assert sudo_denial(command) is None, command


def test_tool_schema_needs_a_command_and_a_reason() -> None:
    tool = SudoExecTool(FakeStore())
    params = tool.schema["function"]["parameters"]
    assert params["required"] == ["command", "reason"]
    assert tool.name == "sudo_exec"
    assert SudoExecTool(FakeStore()).execute({"command": "id", "reason": "x"}, idempotency_key="k")["state"] == "invalid"
