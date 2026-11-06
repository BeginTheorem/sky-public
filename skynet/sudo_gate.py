"""Root on the owner's terms: the sudo gate.

Owner decision (2026-10-15): the ``service-account`` account holds full NOPASSWD sudo, and the
*harness* — not the shell — is the gate. Nothing privileged runs without the owner's own
word, and his word is asked for in Telegram, one command at a time.

A gate that asks for the impossible is worse than a closed one. The unit files set
``NoNewPrivileges=yes``, and the kernel applies that flag *before* sudo can gain privileges,
so every elevation from this process fails no matter who approves it. On 2026-10-28 the gate
spent three owner approvals on exactly such an act — ``rm -f .git/index.lock`` — reported
``sudo_ran`` with returncode 1 each time, and the organism explained the failure to the owner
as a container misconfiguration. ``sudo_available`` probes once, up front, and the gate
refuses without asking whenever root cannot be reached at all.

The shape is copied from Hermes Agent's approval mechanic, in numbers: read-only log
commands run without asking (an allowlist), unmistakably destructive commands are refused
outright and never reach the owner (block globs that beat every approval), and everything
else becomes a durable question to the owner (``store.ask_question``), blocks this run for
a bounded window while refreshing its heartbeat, and **fails closed** — silence is a
refusal, not permission. An approval is one operation: never remembered, never widened,
never reused, and the answer is written down as an event so the act is auditable.

``BashTool`` refuses a bare ``sudo`` for the same reason: a second path to root would make
this gate decorative.
"""

from __future__ import annotations

import contextlib
import fnmatch
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from .provider import Tool

log = logging.getLogger("skynet.sudo_gate")

MAX_OUTPUT = 64 * 1024
DEFAULT_TIMEOUT_SECONDS = 900.0
MAX_TIMEOUT_SECONDS = 3600.0
DEFAULT_COMMAND_TIMEOUT = 300.0
MAX_COMMAND_TIMEOUT = 3600.0
POLL_INTERVAL_SECONDS = 2.0
HEARTBEAT_INTERVAL_SECONDS = 20.0

# Reading the logs is the one thing the owner pre-approved: it changes nothing, and a
# privileged observation must not cost a wait. Every pattern is matched against the whole
# command and still has to survive the metacharacter guard below, so an allowlisted prefix
# cannot smuggle a second command behind it.
DEFAULT_LOG_ALLOWLIST: tuple[str, ...] = (
    "journalctl",
    "journalctl *",
    "dmesg",
    "dmesg -T",
    "dmesg --ctime",
    "tail * /var/log/*",
    "head * /var/log/*",
    "cat /var/log/*",
    "zcat /var/log/*",
    "zgrep * /var/log/*",
    "grep * /var/log/*",
    "ls * /var/log*",
    "systemctl status *",
    "systemctl list-units*",
    "systemctl list-timers*",
    "systemctl is-active *",
    "systemctl is-enabled *",
    "systemctl show *",
)

# Unmistakable catastrophes and self-erasure. Checked before the allowlist and before any
# approval, so an owner "yes" cannot authorise them either — the same floor Hermes keeps.
DEFAULT_DENY_GLOBS: tuple[str, ...] = (
    "*--vacuum*",
    "*--rotate*",
    "*journalctl*--flush*",
    "*rm -rf /*",
    "*rm -rf /*",
    "*mkfs*",
    "*dd *of=/dev/*",
    "*shutdown*",
    "*poweroff*",
    "*:(){*",
    "*chmod -R 777 /*",
    "*chown -R */*/* */*",
    "*> /dev/sd*",
    "*> /dev/nvme*",
    "*passwd root*",
    "*useradd*",
    "*usermod*",
    "*visudo*",
    "*sudoers.d*",
)

# Shell metacharacters turn an allowlisted prefix into an arbitrary command, so a command
# carrying any of them is never auto-approved: it goes to the owner instead.
_METACHARACTERS = re.compile(r"[;&|<>`$\n]")

_APPROVED_ANSWERS = frozenset(
    {
        "разрешить",
        "разрешаю",
        "разрешаю.",
        "да",
        "да.",
        "можно",
        "ок",
        "окей",
        "approve",
        "approved",
        "yes",
        "y",
    }
)


@dataclass(frozen=True)
class SudoPolicy:
    """The owner's calibration for this box, read from the environment."""

    enabled: bool = True
    approval_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    log_allowlist: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LOG_ALLOWLIST)
    deny_globs: tuple[str, ...] = field(default_factory=lambda: DEFAULT_DENY_GLOBS)

    @classmethod
    def from_environment(cls) -> SudoPolicy:
        return cls(
            enabled=os.getenv("SKYNET_SUDO_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"},
            approval_timeout_seconds=_bounded_float(
                os.getenv("SKYNET_SUDO_APPROVAL_TIMEOUT"), DEFAULT_TIMEOUT_SECONDS, 60.0, MAX_TIMEOUT_SECONDS
            ),
            log_allowlist=_patterns(os.getenv("SKYNET_SUDO_ALLOWLIST"), DEFAULT_LOG_ALLOWLIST),
            deny_globs=_patterns(os.getenv("SKYNET_SUDO_DENY"), DEFAULT_DENY_GLOBS),
        )


def _bounded_float(raw: object, default: float, low: float, high: float) -> float:
    try:
        value = float(str(raw).strip()) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def _patterns(raw: object, default: tuple[str, ...]) -> tuple[str, ...]:
    """Comma-separated patterns; an explicitly empty value means 'none'."""
    if raw is None:
        return default
    items = tuple(item.strip() for item in str(raw).split(",") if item.strip())
    return items or default


def normalise_command(command: str) -> str:
    """One line, collapsed whitespace — what the owner is shown and what runs."""
    return " ".join(str(command).split())


def deny_glob_match(command: str, globs: tuple[str, ...]) -> str | None:
    """The glob that forbids this command, or ``None``. Beats the allowlist and any approval."""
    lowered = command.casefold()
    for pattern in globs:
        if fnmatch.fnmatchcase(lowered, pattern.casefold()):
            return pattern
    return None


def is_log_read(command: str, allowlist: tuple[str, ...]) -> bool:
    """True when the command is a pre-approved log read, metacharacters included in the check."""
    if _METACHARACTERS.search(command):
        return False
    return any(fnmatch.fnmatchcase(command, pattern) for pattern in allowlist)


def answer_is_approval(answer: str) -> bool:
    """Only a clear yes is a yes: every other answer, including a blank one, is a refusal."""
    return str(answer).strip().casefold().rstrip("!.") in _APPROVED_ANSWERS


def sudo_available(timeout_seconds: float = 5.0) -> tuple[bool, str]:
    """Whether ``sudo`` can really gain root here, and the reason when it cannot.

    An owner approval is a scarce resource: spending one on an act the kernel will refuse
    anyway wastes his attention and teaches the organism the wrong lesson. The probe is the
    cheap end of that trade — one ``sudo -n true`` answers it, and the stderr of a refusal
    carries the real reason (under ``NoNewPrivileges`` sudo says so by name).
    """
    sudo = shutil.which("sudo") or "/usr/bin/sudo"
    try:
        done = subprocess.run(
            [sudo, "-n", "true"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"sudo could not be executed: {exc}"
    if done.returncode == 0:
        return True, ""
    return False, " ".join((done.stderr or done.stdout or "sudo refused without a message").split())


class SudoGate:
    """Ask the owner, run only on his yes, and write down what happened."""

    def __init__(self, store: Any, policy: SudoPolicy | None = None, sudo_probe: Any = None) -> None:
        self.store = store
        self.policy = policy or SudoPolicy.from_environment()
        self._stop_event: Any = None
        # Probed at most once per process: whether root is reachable is an environment fact
        # set by the unit, and a unit change restarts the process that would cache it.
        self._sudo_probe = sudo_probe or sudo_available
        self._sudo_state: tuple[bool, str] | None = None

    def root_available(self) -> tuple[bool, str]:
        """Whether root can be reached at all — probed once, never asked about."""
        state = self._sudo_state
        if state is None:
            available, why = self._sudo_probe()
            state = (bool(available), str(why))
            self._sudo_state = state
        return state

    def set_stop_event(self, event: Any) -> None:
        """The runner's stop event: SIGTERM must not leave a run blocked on a question."""
        self._stop_event = event

    def execute(self, command: str, reason: str, *, idempotency_key: str = "", timeout_seconds: float | None = None) -> dict[str, Any]:
        """Classify, ask if needed, run on approval, and report what the kernel gave back."""
        if not self.policy.enabled:
            return {"ok": False, "state": "disabled", "error": "root access is switched off (SKYNET_SUDO_ENABLED=0)"}
        command = normalise_command(command)
        if not command:
            return {"ok": False, "state": "invalid", "error": "command must be a non-empty string"}
        reason = " ".join(str(reason).split())[:400]
        if len(reason) < 8:
            return {
                "ok": False,
                "state": "invalid",
                "error": "reason must say what this is for (the owner approves the intent, not only the string)",
            }

        run_id = self._active_run_id()
        forbidden = deny_glob_match(command, self.policy.deny_globs)
        if forbidden is not None:
            self._event("sudo_blocked", {"command": command[:500], "pattern": forbidden, "reason": reason}, run_id)
            return {
                "ok": False,
                "state": "blocked",
                "blocked_by": forbidden,
                "error": (
                    f"refused by a floor rule ('{forbidden}'): this command is destructive or touches the "
                    "mechanism that keeps the organism alive, so no approval can authorise it."
                ),
            }

        available, why = self.root_available()
        if not available:
            self._event("sudo_unavailable", {"command": command[:500], "why": why[:300]}, run_id)
            return {
                "ok": False,
                "state": "unavailable",
                "error": (
                    f"root cannot be reached from this process, so the owner was not asked and neither the allowlist "
                    f"nor an approval could help: {why}. This is an environment fact about the unit "
                    f"(NoNewPrivileges=yes is applied by the kernel before sudo can gain privileges), not a judgement "
                    f"on the command. Do not retry, do not call it a missing permission, and do not report a container "
                    f"or a kernel block: work without root, or ask the owner to change the unit."
                ),
            }

        if is_log_read(command, self.policy.log_allowlist):
            self._event("sudo_auto_approved", {"command": command[:500], "reason": reason}, run_id)
            return self._run(command, run_id, state="auto", question_id=None, timeout_seconds=timeout_seconds)

        question_id = self._ask(command, reason, idempotency_key)
        answer = self._wait_for_answer(question_id)
        if answer is None:
            self._event(
                "sudo_denied",
                {"command": command[:500], "question_id": question_id, "why": "no answer inside the window"},
                run_id,
            )
            return {
                "ok": False,
                "state": "timeout",
                "question_id": question_id,
                "error": (
                    f"the owner did not answer within {self.policy.approval_timeout_seconds:.0f}s; silence is not "
                    "consent, so nothing ran. Do not retry in a loop — a later wake may ask again."
                ),
            }

        text = str(answer.get("answer", ""))
        if not answer_is_approval(text):
            self._event("sudo_denied", {"command": command[:500], "question_id": question_id, "answer": text[:200]}, run_id)
            return {
                "ok": False,
                "state": "denied",
                "question_id": question_id,
                "answer": text[:200],
                "error": "the owner refused this command; do not run it and do not ask again without new evidence.",
            }

        self._event("sudo_approved", {"command": command[:500], "question_id": question_id}, run_id)
        return self._run(command, run_id, state="approved", question_id=question_id, timeout_seconds=timeout_seconds)

    # ── internals ────────────────────────────────────────────────────────────────

    def _active_run_id(self) -> str | None:
        try:
            return self.store.state().active_run_id
        except Exception:  # pragma: no cover - a store without an active run is not fatal
            return None

    def _event(self, kind: str, payload: dict[str, Any], run_id: str | None) -> None:
        try:
            self.store.append_event(kind, payload, run_id)
        except Exception:  # pragma: no cover - auditing must never break the act itself
            log.warning("sudo gate could not record event %s", kind, exc_info=True)

    def _ask(self, command: str, reason: str, idempotency_key: str = "") -> str:
        """Queue the approval as a durable owner question; returns its id."""
        question = (
            f"🔐 Запрос root-доступа\n\n"
            f"Команда:\n{command[:1500]}\n\n"
            f"Зачем: {reason}\n\n"
            f"Окно ответа: {self.policy.approval_timeout_seconds:.0f} с. Молчание = отказ.\n"
            f"Ответ: /answer Разрешить  или  /answer Запретить"
        )
        return str(
            self.store.ask_question(
                question,
                options=["Разрешить", "Запретить"],
                ttl_seconds=self.policy.approval_timeout_seconds,
                question_key="sudo:" + (idempotency_key or command[:120]),
            )
        )

    def _wait_for_answer(self, question_id: str) -> dict[str, Any] | None:
        """Poll for the owner's answer, refreshing the run heartbeat while blocked."""
        deadline = time.monotonic() + self.policy.approval_timeout_seconds
        last_beat = 0.0
        while time.monotonic() < deadline:
            answer = self.store.answer_for(question_id)
            if answer is not None:
                return answer
            if self._stop_event is not None and self._stop_event.is_set():
                return None
            now = time.monotonic()
            if now - last_beat >= HEARTBEAT_INTERVAL_SECONDS:
                run_id = self._active_run_id()
                if run_id:
                    with contextlib.suppress(Exception):
                        self.store.touch_run(str(run_id), "waiting_for_sudo", progress=False)
                last_beat = now
            time.sleep(POLL_INTERVAL_SECONDS)
        return self.store.answer_for(question_id)

    def _run(
        self, command: str, run_id: str | None, *, state: str, question_id: str | None, timeout_seconds: float | None
    ) -> dict[str, Any]:
        sudo = shutil.which("sudo") or "/usr/bin/sudo"
        argv = [sudo, "-n", "--", "/bin/bash", "-lc", command]
        timeout = _bounded_float(timeout_seconds, DEFAULT_COMMAND_TIMEOUT, 1.0, MAX_COMMAND_TIMEOUT)
        cwd = os.getenv("SKYNET_ROOT") or os.getcwd()
        started = time.monotonic()
        try:
            done = subprocess.run(
                argv,
                capture_output=True,
                timeout=timeout,
                cwd=cwd,
                env=os.environ.copy(),
                text=True,
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            self._event(
                "sudo_failed",
                {"command": command[:500], "state": state, "why": "timeout", "timeout_seconds": timeout},
                run_id,
            )
            return {
                "ok": False,
                "state": state,
                "question_id": question_id,
                "timeout": True,
                "error": f"the approved command did not finish within {timeout:.0f}s and was killed",
                "stdout": _clip(getattr(exc, "stdout", "")),
                "stderr": _clip(getattr(exc, "stderr", "")),
            }
        except OSError as exc:
            self._event("sudo_failed", {"command": command[:500], "state": state, "why": str(exc)[:200]}, run_id)
            return {"ok": False, "state": state, "question_id": question_id, "error": f"could not execute sudo: {exc}"}

        stdout, out_truncated = _clip(done.stdout)
        stderr, err_truncated = _clip(done.stderr)
        self._event(
            "sudo_ran",
            {
                "command": command[:500],
                "state": state,
                "returncode": done.returncode,
                "duration_seconds": round(time.monotonic() - started, 2),
            },
            run_id,
        )
        return {
            "ok": done.returncode == 0,
            "state": state,
            "question_id": question_id,
            "returncode": done.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": bool(out_truncated or err_truncated),
        }


def _clip(value: object, limit: int = MAX_OUTPUT) -> tuple[str, bool]:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")
    if len(text) <= limit:
        return text, False
    return text[:limit], True


class SudoExecTool(Tool):
    """``sudo_exec`` — the only path to root, and it goes through the owner."""

    name = "sudo_exec"
    capability_kind = "write"

    def __init__(self, store: Any, gate: SudoGate | None = None) -> None:
        self.store = store
        self.gate = gate or SudoGate(store)

    def set_stop_event(self, event: Any) -> None:
        self.gate.set_stop_event(event)

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Run one command with root. This account has sudo, but the harness asks the owner in "
                    "Telegram before anything privileged runs, so expect a wait of up to the approval window "
                    "and expect refusals. Reading the logs (journalctl, dmesg, /var/log) is pre-approved and "
                    "runs immediately; unmistakably destructive commands (disk formatting, wiping /, "
                    "log vacuuming, editing sudoers) are refused outright and cannot be approved by anyone. "
                    "When the unit's NoNewPrivileges flag makes root unreachable, this tool says so at once "
                    "('state': 'unavailable') instead of asking the owner for something that cannot run. "
                    "'reason' is what the owner sees and approves — say what the command is for, concretely. "
                    "An approval covers exactly this one command: it is never remembered or reused. Never call "
                    "this to work around a policy denial from bash."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "One bounded command to run as root (no leading 'sudo').",
                        },
                        "reason": {
                            "type": "string",
                            "minLength": 8,
                            "maxLength": 400,
                            "description": "What this is for and why root is needed; shown to the owner verbatim.",
                        },
                        "timeout_seconds": {
                            "type": "number",
                            "minimum": 1,
                            "maximum": MAX_COMMAND_TIMEOUT,
                            "description": "Command timeout once the approval is in hand (default 300).",
                        },
                    },
                    "required": ["command", "reason"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        return self.gate.execute(
            str(arguments.get("command", "")),
            str(arguments.get("reason", "")),
            idempotency_key=str(idempotency_key or ""),
            timeout_seconds=arguments.get("timeout_seconds"),  # type: ignore[arg-type]
        )
