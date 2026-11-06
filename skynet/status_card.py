"""The owner's status card — what Sky is doing right now, not how the window went.

``/metrics`` answers "how did the last day go" with a machine dump; this card answers the
questions the owner actually opens the bot with: is she alive, what is she doing *now*, is
she stuck, what needs his hand — and whether the code that runs is the code that is
committed. Three cheap sources, no model call: the ledger (``runs``, ``event_log``,
questions, alerts, delivery counters), systemd (``systemctl --user show``) and the process
tree under the reactor's own PID. Every host probe goes through one injected callable, so
the card is testable without a live host.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import metrics
from .mcp import configured_servers
from .providers import money_boost_state_path
from .reporting import human_span, human_tokens, plural, run_subject, stamp, status_word
from .systemctl import systemctl_command
from .time import parse_timestamp

DEFAULT_SERVICE = "skynet.service"
TIMER_UNIT = "skynet-revive.timer"
STALE_RUN_SECONDS = 15 * 60.0
FAILURE_WINDOW_SECONDS = 3600.0
PROBE_TIMEOUT = 10.0
SUBJECT_LIMIT = 160

# How a configured sensor is labelled on the card. The *list* is not hardcoded: a sensor the
# config parked is not a missing sense, and a stopped unit must not be blamed for one that
# was never asked to start. The card names what is missing instead of printing a bare count,
# because a sense that did not come up is blind for a whole run.
SENSOR_LABELS = {
    "ddg": "ddg",
    "duckduckgo": "ddg",
    "arxiv": "arxiv",
    "github": "github",
    "playwright": "playwright",
}


def expected_sensors(environ: Mapping[str, str] | None = None) -> tuple[tuple[str, str], ...]:
    """``(process marker, label)`` for every sensor this environment asks her to start.

    The marker is the executable's own basename — that is what shows up in a child's command
    line under ``/proc``, and it is the only part of a configured command that can be trusted
    to appear there (``SKYNET_MCP_DDG_COMMAND`` starts ``duckduckgo-mcp-server``, not ``ddg``).
    A shim that ``exec``s an interpreter keeps only the module spelling of the name, which is why
    the comparison in :func:`sensor_names` accepts both — see :func:`_marker_spellings`.
    """
    sensors: list[tuple[str, str]] = []
    for name, command in configured_servers(environ).items():
        executable = Path(command[0]).name or name
        sensors.append((executable, SENSOR_LABELS.get(name, name)))
    return tuple(sensors)

STATUS_ICONS = {
    "completed": "✅",
    "interrupted": "⛔",
    "failed": "❌",
    "blocked": "⛔",
    "needs_recovery": "🛠",
    "running": "🔄",
}

PHASE_WORDS = {
    "common": "работа",
    "waiting_for_provider": "ждёт провайдера",
    "starting": "старт",
    "planning": "планирование",
    "acting": "инструменты",
    "finishing": "отчёт",
}

# systemd prints monotonic deadlines as a span ("10h 19min 54.061052s"), so a small
# tolerant parser is cheaper and safer than assuming one property name or format.
_SPAN_FACTORS = {
    "year": 31_536_000.0,
    "month": 2_592_000.0,
    "week": 604_800.0,
    "day": 86_400.0,
    "h": 3_600.0,
    "hour": 3_600.0,
    "min": 60.0,
    "minute": 60.0,
    "s": 1.0,
    "sec": 1.0,
    "ms": 0.001,
    "msec": 0.001,
    "us": 0.000001,
}
_SPAN_TOKEN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(years?|months?|weeks?|days?|hours?|minutes?|secs?|msecs?|us|h|min|s|ms)",
    re.IGNORECASE,
)

Probe = Callable[[list[str]], str]


def _subprocess_probe(argv: list[str]) -> str:
    """The default host probe: run it, keep stdout, never raise on a failed command."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=PROBE_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout


def _number(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


@dataclass
class HostFacts:
    """What only the host can say: the unit, its age, its children, deploy drift, the timer."""

    unit: str = "unknown"
    uptime_seconds: float = 0.0
    pid: int = 0
    sensors: list[str] = field(default_factory=list)
    expected_sensors: tuple[str, ...] = ()
    deployed: str = ""
    head: str = ""
    revive_seconds: float | None = None
    revive_armed: bool | None = None

    @property
    def active(self) -> bool:
        return self.unit.startswith("active")


def _show(probe: Probe, unit: str, *properties: str) -> dict[str, str]:
    """``systemctl --user show`` reduced to the requested keys."""
    argv = systemctl_command("show", unit, *[part for name in properties for part in ("-p", name)])
    facts: dict[str, str] = {}
    for line in probe(argv).splitlines():
        key, separator, value = line.partition("=")
        if separator:
            facts[key.strip()] = value.strip()
    return facts


def _read_uptime(proc_root: Path) -> float:
    try:
        return float((proc_root / "uptime").read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def _marker_spellings(executable: str) -> tuple[str, ...]:
    """Every spelling of a configured executable that can survive into ``/proc``.

    A console script that ``exec``s an interpreter loses its own name: the shim in
    ``~/.local/bin`` is replaced by ``<venv>/bin/python -c "from arxiv_mcp_server import main; ..."``,
    and only the hyphenless module name is left in the child's command line. Searching for the
    basename alone therefore reads a sense that is plainly running as blind, so both spellings
    count; a real binary (``github-mcp-server``) still matches the first one.
    """
    underscorized = executable.replace("-", "_")
    return (executable,) if underscorized == executable else (executable, underscorized)


def sensor_names(pid: int, proc_root: Path = Path("/proc"), sensors: tuple[tuple[str, str], ...] | None = None) -> list[str]:
    """Which MCP senses are alive under ``pid``, named by their command line."""
    if pid <= 0:
        return []
    expected = expected_sensors() if sensors is None else sensors
    wanted = tuple((_marker_spellings(marker), name) for marker, name in expected)
    try:
        children = (proc_root / str(pid) / "task" / str(pid) / "children").read_text(encoding="utf-8").split()
    except OSError:
        return []
    found: list[str] = []
    for child in children:
        try:
            command = (proc_root / child / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        for spellings, name in wanted:
            if name not in found and any(spelling in command for spelling in spellings):
                found.append(name)
    return found


def _deploy_state(repo: Path, probe: Probe) -> tuple[str, str]:
    """``(deployed, head)`` as short revisions; empty strings when either is unreadable."""
    deployed = ""
    with contextlib.suppress(OSError):
        deployed = (repo / ".deployed-commit").read_text(encoding="utf-8").strip()[:7]
    head = ""
    for line in probe(["git", "-C", str(repo), "rev-parse", "HEAD"]).splitlines():
        candidate = line.strip()
        if candidate:
            head = candidate[:7]
            break
    return deployed, head


def _span_seconds(text: str) -> float | None:
    """systemd's ``10h 19min 54.061052s`` → seconds; ``None`` when nothing is recognised."""
    total = 0.0
    found = False
    for value, unit in _SPAN_TOKEN.findall(str(text)):
        factor = _SPAN_FACTORS.get(unit.lower())
        if factor is None:
            continue
        total += float(value) * factor
        found = True
    return total if found else None


def _revive_seconds(probe: Probe, uptime: float) -> tuple[float | None, bool | None]:
    """``(seconds until the next revive tick, is it armed)``; ``None`` means "cannot tell".

    A monotonic-scheduled timer reports no wall-clock elapse: ``NextElapseUSecRealtime``
    comes back empty and ``NextElapseUSecMonotonic`` carries the deadline as µs since boot,
    which systemd prints as a span (``10h 19min 54.061052s``). The countdown is therefore
    that deadline minus the current uptime — one clock, no conversion. A disarmed timer
    prints ``infinity``; an unreadable one is reported as unknown so the card never claims
    "not armed" about a timer it simply failed to read.
    """
    shown = _show(probe, TIMER_UNIT, "NextElapseUSecMonotonic")
    raw = str(shown.get("NextElapseUSecMonotonic", "")).strip()
    if not raw:
        return None, None
    if raw.lower() in {"infinity", "n/a", "0"}:
        return None, False
    target = _span_seconds(raw)
    if target is None or uptime <= 0:
        return None, None
    return max(0.0, target - uptime), True


def host_facts(
    service: str = DEFAULT_SERVICE,
    repo: Path = Path("."),
    *,
    probe: Probe | None = None,
    proc_root: Path = Path("/proc"),
    uptime: float | None = None,
    sensors: tuple[tuple[str, str], ...] | None = None,
) -> HostFacts:
    probe = probe or _subprocess_probe
    if uptime is None:
        uptime = _read_uptime(proc_root)
    expected = expected_sensors() if sensors is None else sensors
    state = _show(probe, service, "ActiveState", "SubState", "MainPID", "ActiveEnterTimestampMonotonic")
    facts = HostFacts(
        unit="/".join(part for part in (state.get("ActiveState", "unknown"), state.get("SubState", "")) if part),
        pid=_number(state.get("MainPID")),
    )
    entered = _number(state.get("ActiveEnterTimestampMonotonic")) / 1_000_000
    if facts.active and 0 < entered <= uptime:
        facts.uptime_seconds = uptime - entered
    facts.sensors = sensor_names(facts.pid, proc_root, expected)
    facts.expected_sensors = tuple(name for _, name in expected)
    facts.deployed, facts.head = _deploy_state(repo, probe)
    facts.revive_seconds, facts.revive_armed = _revive_seconds(probe, uptime)
    return facts


def _runs(store: Any, limit: int = 3) -> list[dict[str, Any]]:
    rows = store.connection.execute(
        "SELECT run_id, status, kind, last_phase, steps, usage_tokens, started_at, finished_at "
        "FROM runs ORDER BY rowid DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def _pass_line(label: str, passes: list[dict[str, Any]], *, missing: str) -> str:
    """The newest pass of one kind, named apart from the episodes it serves.

    A pass keeps its own run row now, so a degraded one is visible where the owner
    looks: ``needs_recovery`` in the row means the pass could not finish its work.
    The wording is deliberately gender-neutral ("успешно", "провал"), because one
    helper serves both the consolidation pass and the planning pass.
    """
    if not passes:
        return f"{label}: {missing}"
    run = passes[0]
    status = str(run.get("status"))
    icon = STATUS_ICONS.get(status, "•")
    when = stamp(run.get("finished_at") or run.get("started_at"))
    tokens = human_tokens(run.get("usage_tokens"))
    word = {"completed": "успешно", "needs_recovery": "с деградацией", "failed": "провал", "running": "идёт"}.get(status, status_word(status))
    return f"{label}: {icon} {word} {when} · токенов {tokens}"


def _memory_pass_line(passes: list[dict[str, Any]]) -> str:
    """The newest consolidation pass."""
    return _pass_line("Консолидация", passes, missing="прогонов ещё не было")


def _plan_pass_line(passes: list[dict[str, Any]]) -> str:
    """The newest planning pass."""
    return _pass_line("План", passes, missing="прогонов ещё не было")


def _age_seconds(value: Any, now: float) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, now - parse_timestamp(str(value)).timestamp())
    except (TypeError, ValueError):
        return None


def _progress_age(store: Any, run_id: str, now: float) -> float | None:
    """How long since anything happened inside the run — the "is she stuck" signal."""
    row = store.connection.execute("SELECT MAX(created_at) AS last FROM event_log WHERE run_id = ?", (run_id,)).fetchone()
    return _age_seconds(row["last"] if row is not None else None, now)


def _events(store: Any, kinds: tuple[str, ...], since: str) -> list[dict[str, Any]]:
    marks = ", ".join("?" for _ in kinds)
    rows = store.connection.execute(
        f"SELECT kind, payload, created_at FROM event_log WHERE kind IN ({marks}) AND created_at >= ? "
        "ORDER BY sequence DESC LIMIT 200",
        (*kinds, since),
    ).fetchall()
    parsed: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        try:
            record["payload"] = json.loads(record["payload"])
        except (TypeError, ValueError):
            record["payload"] = {}
        parsed.append(record)
    return parsed


def _chain_size(store: Any) -> int:
    row = store.connection.execute(
        "SELECT payload FROM event_log WHERE kind = 'provider_chain_reloaded' ORDER BY sequence DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return 0
    try:
        providers = json.loads(row["payload"]).get("providers")
    except (TypeError, ValueError):
        return 0
    return len(providers) if isinstance(providers, list) else 0


def _provider_line(store: Any, now: float) -> str:
    """Chain size, an hour of chain failures and whether the boost is on. No live claim."""
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now - FAILURE_WINDOW_SECONDS)) + "Z"
    failures = _events(store, ("fallback_failure", "provider_unreachable"), since)
    chain = _chain_size(store)
    parts = [f"Провайдеры: цепочка {chain or '—'}"]
    if failures:
        strikes: dict[str, int] = {}
        for record in failures:
            name = str(record["payload"].get("provider") or "?")
            strikes[name] = strikes.get(name, 0) + 1
        top, count = max(strikes.items(), key=lambda item: item[1])
        parts.append(f"отказов за час {len(failures)}")
        parts.append(f"топ {top} ×{count}")
    else:
        parts.append("отказов за час нет")
    parts.append("money-boost " + ("вкл" if _money_boost_enabled() else "выкл"))
    return " · ".join(parts)


def _money_boost_enabled() -> bool:
    try:
        return bool(json.loads(money_boost_state_path().read_text(encoding="utf-8")).get("enabled", False))
    except (OSError, ValueError, TypeError):
        return False


def _header(facts: HostFacts, runs: list[dict[str, Any]], now: float) -> str:
    if facts.active:
        state = "активна"
        if facts.uptime_seconds:
            state += f" {human_span(facts.uptime_seconds)}"
    else:
        state = f"стоит ({facts.unit})"
    current = next((run for run in runs if str(run.get("status")) == "running"), None)
    if current is not None:
        spent = human_span(_age_seconds(current.get("started_at"), now) or 0.0)
        where = f"прогон #{str(current.get('run_id'))[:8]} идёт {spent}"
    elif runs:
        last = runs[0]
        when = stamp(last.get("finished_at") or last.get("started_at"))
        where = f"простаивает · последний прогон {status_word(str(last.get('status')))} {when}"
    else:
        where = "прогонов нет"
    return f"🧠 SkyNet · {state} · {where}"


def _clip(text: Any, limit: int = SUBJECT_LIMIT) -> str:
    """One flat line, short enough for a phone: a task title can be a whole paragraph."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _run_detail(store: Any, current: dict[str, Any], now: float) -> list[str]:
    lines: list[str] = []
    subject = run_subject(store, str(current.get("run_id")))
    if subject:
        lines.append(f"Задача: {_clip(subject)}")
    phase = PHASE_WORDS.get(str(current.get("last_phase")), str(current.get("last_phase") or "—"))
    step = _number(current.get("steps"))
    tokens = human_tokens(current.get("usage_tokens"))
    lines.append(f"Шаг {step} · фаза: {phase} · токенов {tokens}")
    age = _progress_age(store, str(current.get("run_id")), now)
    if age is None:
        lines.append("Прогресс: событий ещё нет")
    elif age >= STALE_RUN_SECONDS:
        lines.append(f"⚠️ нет движения {human_span(age)} — похоже, встала")
    else:
        lines.append(f"Прогресс: последний шаг {human_span(age)} назад")
    return lines


def _history_line(runs: list[dict[str, Any]], limit: int = 2) -> str:
    history: list[str] = []
    for run in runs[1 : limit + 1]:
        status = str(run.get("status"))
        icon = STATUS_ICONS.get(status, "•")
        steps = _number(run.get("steps"))
        when = stamp(run.get("finished_at") or run.get("started_at"))
        history.append(f"{icon} {steps} {plural(steps, 'шаг', 'шага', 'шагов')} ({when})")
    return "Прошлые: " + " · ".join(history) if history else ""


def _sensor_line(facts: HostFacts) -> str:
    if not facts.active or not facts.pid:
        return "Сенсоры: недоступны — процесс не запущен"
    found = list(facts.sensors)
    names = ", ".join(found) if found else "нет"
    if not facts.expected_sensors:
        return f"Сенсоры: {names} (конфиг не читается — сравнить не с чем)"
    missing = [name for name in facts.expected_sensors if name not in found]
    line = f"Сенсоры {len(found)}/{len(facts.expected_sensors)}: {names}"
    if missing:
        line += " · нет " + ", ".join(missing)
    return line


def _countdown(seconds: float) -> str:
    """A countdown in the unit a human uses: minutes, then hours, then a bare minute."""
    if seconds < 60:
        return "меньше минуты"
    if seconds < 3600:
        return f"{int(seconds // 60)}м"
    return f"{int(seconds // 3600)}ч{int((seconds % 3600) // 60):02d}м"


def _deploy_line(facts: HostFacts, delivery: dict[str, Any]) -> str:
    if not facts.head:
        state = "состояние деплоя неизвестно"
    elif facts.deployed == facts.head:
        state = f"HEAD {facts.head} = работает"
    else:
        state = f"⚠️ код разошёлся: работает {facts.deployed or '—'}, HEAD {facts.head}"
    if delivery:
        pending = _number(delivery.get("outbox_pending"))
        dead = _number(delivery.get("outbox_dead"))
        state += f" · очередь канала {pending}"
        if dead:
            state += f" (+мёртвых {dead})"
    return "Деплой: " + state


def card(
    store: Any,
    *,
    service: str = DEFAULT_SERVICE,
    repo: Path = Path("."),
    state_dir: Path | None = None,
    registry_path: Path | None = None,
    probe: Probe | None = None,
    proc_root: Path = Path("/proc"),
    now: float | None = None,
    uptime: float | None = None,
    sensors: tuple[tuple[str, str], ...] | None = None,
) -> str:
    """The card the owner reads: alive, doing what, stuck?, needs him?, code honest?"""
    now = time.time() if now is None else now
    facts = host_facts(service, repo, probe=probe, proc_root=proc_root, uptime=uptime, sensors=sensors)
    runs = _runs(store, 8)
    # A pass is a run, but it is not work on a task: the header, the current-run block
    # and the history stay about episodes, and each pass kind gets its own line so a
    # planning or consolidation problem cannot hide inside an episode's verdict.
    episodes = [run for run in runs if str(run.get("kind") or "build") == "build"]
    memory_passes = [run for run in runs if str(run.get("kind") or "build") == "memory"]
    plan_passes = [run for run in runs if str(run.get("kind") or "build") == "plan"]
    lines = [_header(facts, episodes, now)]

    current = next((run for run in episodes if str(run.get("status")) == "running"), None)
    if current is not None:
        lines.extend(_run_detail(store, current, now))
    history = _history_line(episodes)
    if history:
        lines.append(history)
    lines.append(_sensor_line(facts))

    try:
        data = metrics.snapshot(store, since_days=1.0, registry_path=registry_path, state_dir=state_dir)
    except Exception:  # the card must survive a ledger the metrics pass chokes on
        data = {}
    lines.append(_provider_line(store, now))
    memory = data.get("memory", {}) if isinstance(data, dict) else {}
    diversity = data.get("diversity", {}) if isinstance(data, dict) else {}
    planning = data.get("planning", {}) if isinstance(data, dict) else {}
    lines.append(
        f"Память {_number(memory.get('memories'))} · "
        f"идеи {_number(diversity.get('cells_filled'))}/{_number(diversity.get('cells_total'))} клеток · "
        f"задач {_number(planning.get('tasks'))}"
    )
    lines.append(_memory_pass_line(memory_passes))
    lines.append(_plan_pass_line(plan_passes))

    waiting: list[str] = []
    questions = len(store.open_questions(limit=10))
    if questions:
        waiting.append(f"{questions} {plural(questions, 'вопрос', 'вопроса', 'вопросов')}")
    alerts = len(store.pending_alerts(limit=20))
    if alerts:
        waiting.append(f"{alerts} {plural(alerts, 'оповещение', 'оповещения', 'оповещений')}")
    if waiting:
        lines.append("Ждёт тебя: " + " · ".join(waiting))

    lines.append(_deploy_line(facts, data.get("delivery", {}) if isinstance(data, dict) else {}))
    if facts.revive_armed is None:
        lines.append("Тик воскрешения: неизвестно — systemd не ответил")
    elif not facts.revive_armed:
        lines.append("Тик воскрешения: таймер не взведён")
    else:
        lines.append(f"Тик воскрешения: через {_countdown(facts.revive_seconds or 0.0)}")
    return "\n".join(lines)
