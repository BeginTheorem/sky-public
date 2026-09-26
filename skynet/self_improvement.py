from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .checkpoints import CheckpointError, CheckpointManager
from .time import parse_timestamp, utc_datetime_now, utc_now

log = logging.getLogger("skynet.self_improvement")


class SelfImprovementError(RuntimeError):
    """Failure carrying an optional durable reason code.

    The reason code keeps a stale anchor apart from an ambiguous one: once
    tool-call transport drops the match count, both share the
    "patch anchor must match exactly once" prefix and text matching alone
    cannot separate them.
    """

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "",
        match_count: int | None = None,
        previous_proposal: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.match_count = match_count
        self.previous_proposal = dict(previous_proposal) if previous_proposal else None


class ProtectedPathWarning(SelfImprovementError):
    """A proposal touches the gate's own inputs and is warned on first sight.

    The deny is only a warning: the change is recorded as ``warned_protected``
    (its worktree is kept) and the model must re-submit the identical proposal.
    The identical fingerprint is then acknowledged and proceeds through the full
    gate. It is never a hard rejection, and a different diff gets a fresh
    warning.
    """

    def __init__(self, offenders: Sequence[str], fingerprint: str, message: str) -> None:
        super().__init__(message, reason_code="protected_path")
        self.offenders = list(offenders)
        self.fingerprint = fingerprint


# The gate runs the whole suite now, and on a slow host with real git fixtures
# that can take minutes; the knob exists so an operator can raise it without
# editing code (the old 600s timeouts came from the model choosing one heavy test
# file as its entire gate).
TEST_COMMAND_TIMEOUT_SECONDS = float(os.getenv("SKYNET_GATE_TIMEOUT", "600"))
STATIC_STAGE_TIMEOUT_SECONDS = 300.0
GATE_STAGE_TIMEOUT_SECONDS = 120.0

# The gate must prove WHICH tree it tested. With `cwd=worktree`, `import skynet`
# resolves to the worktree, but a runner that leaves cwd out of sys.path (a
# console `pytest`, `tox`, `uv run pytest`, `--import-mode=importlib`) silently
# falls through to the editable-install finder and validates the MAIN tree while
# still returning rc=0 - a class of quiet false accepts that no log reveals.
# Import every module of the package from the worktree. A proposal can break an
# import that no test touches; pytest would still pass while the organism fails
# to start after the promotion.
GATE_IMPORT_SMOKE_SNIPPET = """
import importlib, pkgutil, skynet
mods = sorted(m.name for m in pkgutil.walk_packages(skynet.__path__, "skynet."))
errors = []
for name in mods:
    try:
        importlib.import_module(name)
    except Exception as exc:
        errors.append(f"{name}: {type(exc).__name__}: {exc}")
if errors:
    print("\\n".join(errors))
    raise SystemExit(1)
print("import-smoke-ok", len(mods), "modules")
"""

GATE_SELFCHECK_SNIPPET = (
    "import os, sys, skynet;"
    "wt = os.path.realpath(os.environ['SKYNET_PROPOSAL_WORKTREE']);"
    "src = getattr(skynet, '__file__', None) or next(iter(getattr(skynet, '__path__', [])), wt);"
    "real = os.path.realpath(src);"
    "assert real.startswith(wt + os.sep) or real.startswith(wt), "
    "'gate would test the main tree, not the proposal worktree: ' + real;"
    "print('gate-selfcheck-ok', real)"
)
DEFAULT_TEST_COMMAND: tuple[str, ...] = (sys.executable, "-m", "pytest", "-q")
# A payload damaged in transport is not a verdict about the change, so it
# belongs in the retryable family beside the other environment conditions:
# the same intended change must be submittable again once it arrives intact.
PATCH_PAYLOAD_CORRUPT = "patch_payload_corrupt"
BLOCKED_FAILURE_CLASSES = frozenset({PATCH_PAYLOAD_CORRUPT, "administrative_test_failure", "environment_failure", "timeout", "worktree_quota"})
# An environment failure does not burn the fingerprint, so the identical change
# is retryable once the environment is repaired. Without a ceiling that turns
# into a per-cycle loop (worktree + a full gate run each time), so the same
# fingerprint is refused after this many environment-blocked attempts.
MAX_ENVIRONMENT_ATTEMPTS = 3
ANCHOR_REASON_CODES = frozenset({"anchor_not_found", "anchor_ambiguous"})
TERMINAL_PROPOSAL_STATUSES = frozenset({"accepted", "rolled_back", "rejected", "blocked_by_environment"})
# A protected-path proposal is only warned, not rejected: its worktree must
# survive so the identical resubmission can run the full gate without being
# re-applied. Deliberately NOT terminal.
WARNED_PROTECTED = "warned_protected"
DEFAULT_PROPOSAL_TTL_HOURS = 24.0
# A warning only has to survive until the identical resubmission, which arrives
# within a run or two. A much shorter TTL than a validated proposal keeps a
# warned worktree from lingering for a day purely because the model moved on.
DEFAULT_WARNED_TTL_HOURS = 6.0
DEFAULT_MAX_WORKTREES = 8
# The orphan sweep is destructive and its root is derived, so a pattern the
# organism owns by name is exempt: a proposal id is ``uuid4().hex``, while a
# durable artifact directory is named and its absence is discoverable.
ORPHAN_SWEEP_EXEMPT_ENV = "SKYNET_KEEP_WORKTREES"
# A proposal that rewrites the suite, its configuration, or this module could
# neuter the gate and validate itself. Those paths are warned on first
# submission and only an identical resubmission reaches a stage.
GATE_PROTECTED_ROOT_FILES = frozenset({"conftest.py", "pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg"})
# The meta-loop decides what the organism works on and how it is judged. Owner
# decision opened it: these paths are warned, not held, so the
# organism can repair its own judge after an identical resubmission.
GATE_PROTECTED_PATHS = frozenset({
    "skynet/self_improvement.py",
    "skynet/idea_archive.py",
    "skynet/planner.py",
    "skynet/autonomous_planner.py",
    "skynet/metrics.py",
    # The public-release sanitizer decides what leaves the repository and what
    # the leak scanner forbids; a proposal must not weaken it silently.
    "release/rules.toml",
    "release/publish.py",
})
VENV_PYTHON_NAMES = frozenset({"python", "python3", ".venv/bin/python", "./.venv/bin/python", ".venv/bin/python3", "./.venv/bin/python3"})
# The service exports provider-enable flags (OpenRouter-only deploy) and the money
# boost; the suite asserts the factory builds each provider from
# SKYNET_PROVIDER_CHAIN, so inheriting them turns a code verdict into an
# environment verdict and rejects every proposal on an untouched tree. The
# suite stage runs with these removed, so a proposal is judged by code alone.
GATE_SUITE_ENV_EXCLUSIONS = frozenset({
    "OLLAMA_ENABLED",
    "NVIDIA_ENABLED",
    "DEEPSEEK_ENABLED",
    "SKYNET_MONEY_BOOST",
})


def change_content_digest(root: str | Path, paths: Sequence[str]) -> str:
    """Digest of *what* changed, not only *which* files changed.

    The earlier digest was ``sha256`` over the sorted path names alone, so two
    genuinely different patches to the same file were reported under the same
    identity: the four type guards submitted (proposal
    b1a54578) and the ``json_tree`` holder subquery submitted
    (proposal 0d3b1925) are both ``skynet/store.py`` and therefore both
    ``6248ede0...``, even though the second patch is not the first one. The
    registry and the runtime log carry this value as the change identity, so a
    path-only digest cannot distinguish two promotions of the same file.
    Hashing each path with its post-change bytes keeps the value derived from
    what was actually written while staying deterministic across git worktree
    roots. A path that is absent after the change (a deletion) still
    contributes its name, so deletes and creates do not collapse into the
    empty digest. The durable dedup key is ``change_fingerprint`` (see
    ``propose_self_improvement``), which is already content-derived; this value
    is the human-readable identity recorded beside it.
    """
    base = Path(root)
    hasher = hashlib.sha256()
    for relative in sorted(set(paths)):
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        try:
            payload = (base / relative).read_bytes()
        except OSError:
            payload = b""
        hasher.update(payload)
        hasher.update(b"\0")
    return hasher.hexdigest()


def gate_suite_environment() -> dict[str, str]:
    """Environment for the gate suite stage, free of service provider flags."""
    return {key: value for key, value in os.environ.items() if key not in GATE_SUITE_ENV_EXCLUSIONS}


def check_registry_integrity(path: str | Path) -> list[tuple[str, str]]:
    """Report registry entries whose status contradicts their failure_class.

    A timeout or environment failure is an environment condition, so it must be
    recorded as ``blocked_by_environment``; every other classified failure is a
    verdict about the proposal and must be ``rejected``. Entries without a
    failure_class (accepted, promoting, awaiting_reboot, ...) are not checked.

    A persisted anchor reason code is checked as well: it is written beside the
    failure class at failure time, so the two must agree and the code must carry
    its match count, otherwise a drifted record is indistinguishable from a
    consistent one to every later reader of the registry.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [("registry", f"unreadable registry: {exc}")]
    if not isinstance(data, dict):
        return [("registry", "registry is not a JSON object")]
    violations: list[tuple[str, str]] = []
    for proposal_id, record in data.items():
        if not isinstance(record, dict):
            violations.append((str(proposal_id), "entry is not a JSON object"))
            continue
        failure_class = str(record.get("failure_class", ""))
        if not failure_class:
            continue
        expected = "blocked_by_environment" if failure_class in BLOCKED_FAILURE_CLASSES else "rejected"
        status = str(record.get("status", ""))
        if status != expected:
            violations.append((str(proposal_id), f"failure_class={failure_class} recorded as {status}, expected {expected}"))
        reason_code = record.get("reason_code", "")
        if isinstance(reason_code, str) and reason_code in ANCHOR_REASON_CODES:
            if failure_class != reason_code:
                violations.append((str(proposal_id), f"reason_code={reason_code} contradicts failure_class={failure_class}"))
            match_count = record.get("match_count")
            if not isinstance(match_count, int) or isinstance(match_count, bool) or match_count < 0:
                violations.append((str(proposal_id), f"reason_code={reason_code} without a non-negative integer match_count: {match_count!r}"))
        elif failure_class in ANCHOR_REASON_CODES:
            violations.append((str(proposal_id), f"failure_class={failure_class} without a persisted reason_code"))
    return violations


_PATCH_FILE_START = re.compile(r"^(diff --git |--- )")
_PATCH_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def patch_structure_error(patch: str) -> str | None:
    """Return why *patch* is not a well-formed unified diff, else ``None``.

    Only the payload's shape is judged; ``git apply`` stays the authority on
    whether the hunks match the tree. A diff damaged in tool-call transport
    keeps its headers, so git answers "corrupt patch at line N" -- a statement
    about the payload, not about the change. Measured on the runtime log
    (11 patch payloads, 2026-09-25): all 11 arrive malformed, and proposal
    701d3510, the only submission carrying a byte-correct locally validated
    file, was recorded as a verdict about its change after dying here.
    Separating the two lets the damaged case be retried instead of burning
    the change fingerprint.
    """
    if not patch.strip():
        return "patch is empty"
    if not patch.endswith("\n"):
        return "patch is not terminated: the payload was truncated in tool-call transport"
    lines = patch.splitlines()
    if not any(_PATCH_FILE_START.match(line) for line in lines):
        return "patch carries no file header ('diff --git' or '--- ')"
    starts = [index for index, line in enumerate(lines) if _PATCH_HUNK_HEADER.match(line)]
    if not starts:
        return "patch carries no hunk header"
    for position, start in enumerate(starts):
        header = _PATCH_HUNK_HEADER.match(lines[start])
        if header is None:  # pragma: no cover - starts was built from this match
            continue
        declared_old = int(header.group(2) or 1)
        declared_new = int(header.group(4) or 1)
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        old = new = 0
        for line in lines[start + 1 : end]:
            if line.startswith("\\"):
                # "\ No newline at end of file" annotates the previous body line.
                continue
            if line.startswith(("diff --git ", "@@")):
                break
            if line.startswith("+"):
                new += 1
            elif line.startswith("-"):
                old += 1
            elif line.startswith(" ") or line == "":
                old += 1
                new += 1
            else:
                return f"hunk at line {start + 1} carries an unparseable body line: {line[:40]!r}"
        if (old, new) != (declared_old, declared_new):
            return (
                f"hunk at line {start + 1} declares -{declared_old} +{declared_new} but carries "
                f"-{old} +{new}; the payload was truncated or rewritten in tool-call transport"
            )
    return None


def classify_failure(error: BaseException | str) -> str:
    """Classify proposal failures so environment problems are not treated as bad ideas.

    An error that already carries a reason_code is reported with that code: a
    stale anchor and a count-dropped ambiguous anchor share the
    "patch anchor must match exactly once" prefix, so text matching alone
    cannot tell them apart.
    """
    reason_code = getattr(error, "reason_code", "")
    if isinstance(reason_code, str) and reason_code:
        return reason_code
    text = str(error).casefold()
    if "invalid proposal path" in text or "escapes worktree" in text:
        return "path_violation"
    if "gate-protected" in text or "protected path" in text:
        return "protected_path"
    if "git apply" in text or "patch_apply_failed" in text:
        return "patch_apply_failed"
    if "must be an object" in text or "required" in text or "payload" in text:
        return "invalid_payload"
    if "identical proposal" in text or "unchanged" in text:
        return "invalid_payload"
    if "test command" in text or "test path" in text:
        return "invalid_payload"
    if "non-empty old or anchor" in text:
        return "invalid_payload"
    if "patch" in text and ("match" in text or "anchor" in text):
        if "not found" in text or "(0 matches)" in text:
            return "anchor_not_found"
        if "matches given" in text:
            return "anchor_ambiguous"
        return "patch_mismatch"
    if "indentationerror" in text or "syntaxerror" in text:
        return "syntax_error"
    if "importerror" in text or "modulenotfounderror" in text:
        return "import_error"
    if "systemctl" in text or "permission denied" in text or "administrative" in text:
        return "administrative_test_failure"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "no such file or directory" in text or "executable" in text:
        return "environment_failure"
    if "health gate" in text or "health check" in text:
        return "health_check_failure"
    if "tests failed" in text:
        if "file or directory not found" in text:
            return "environment_failure"
        return "regression_failure"
    if "ignored proposal path" in text or "cannot be committed" in text:
        return "ignored_path"
    if "stale_proposal" in text or "expired after" in text:
        return "stale_proposal"
    if "stale_base_commit" in text or "could not be rebased" in text or "main worktree moved" in text:
        return "stale_base_commit"
    if "worktree quota" in text or "worktree_quota" in text:
        return "worktree_quota"
    return "unknown"


def _transport_unescape(value: str) -> str:
    """Decode literal escape artifacts that survive tool-call transport.

    Measured on the historical runtime log: 22 anchor submissions matched
    zero times against the file live at attempt time, and 13 of them matched
    exactly once after decoding the two-character sequences ``\\"`` and
    ``\\n`` - including all 8 skynet/autonomous_planner.py rows. The remaining
    9 are genuinely stale and must keep raising the zero-match error.
    """
    if "\\" not in value:
        return value
    return value.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")


@dataclass(frozen=True, slots=True)
class ImprovementProposal:
    proposal_id: str
    worktree: Path
    base_commit: str


class SelfImprovementManager:
    """Isolated, explicitly invoked source-change boundary."""

    def __init__(self, root: str | Path, worktree_root: str | Path | None = None, *, gate_suite: Sequence[str] | None = None, store: Any | None = None) -> None:
        self.root = Path(root).resolve()
        if worktree_root is not None:
            self.worktree_root = Path(worktree_root).resolve()
        else:
            # Never nest a second .skynet-improvements directory when the
            # manager itself runs from inside the worktree collection.
            existing = next((parent for parent in (self.root, *self.root.parents) if parent.name == ".skynet-improvements"), None)
            self.worktree_root = existing if existing is not None else (self.root.parent / ".skynet-improvements").resolve()
        self.request_path = self.root / "state" / "reboot-request.json"
        self.proposals_path = self.root / "state" / "self-improvement-proposals.json"
        # The harness owns the gate: the model's command may add a bounded check
        # but can never replace the suite (P0-14).
        self.gate_suite: tuple[str, ...] = tuple(gate_suite) if gate_suite else DEFAULT_TEST_COMMAND
        self.last_quarantined: list[str] = []
        self.runtime_log = None
        runtime_path = self.root / "state" / "runtime.jsonl"
        try:
            from .runtime_log import RuntimeLog
            self.runtime_log = RuntimeLog(runtime_path)
        except ImportError:
            pass
        # The durable store is injected by the composition root (``cli.py``): the
        # manager must not open a second StateStore itself, because construction
        # recovers inflight leases and would race the live reactor. When it is
        # absent the warning still lands in the advanced runtime log, but only a
        # store reaches the owner outbox.
        self.store = store

    def set_store(self, store: Any | None) -> None:
        """Inject the durable store used for held-proposal events and alerts."""
        self.store = store

    def _emit(self, kind: str, payload: dict[str, object]) -> None:
        """Record a durable event through the store, or the runtime log."""
        if self.store is not None:
            self.store.append_event(kind, payload)
        elif self.runtime_log is not None:
            self.runtime_log.write(kind, payload)

    def _alert_owner(self, kind: str, payload: dict[str, object]) -> None:
        """Escalate through the existing alert/outbox machinery, if wired."""
        if self.store is None:
            return
        fingerprint = str(payload.get("change_fingerprint", ""))
        self.store.raise_alert(
            kind,
            payload,
            severity="warning",
            dedup_key=f"{kind}:{fingerprint}",
        )

    def _git(self, args: Sequence[str], cwd: Path | None = None) -> str:
        result = subprocess.run(["git", *args], cwd=cwd or self.root, capture_output=True, text=True, check=False)
        if result.returncode:
            raise SelfImprovementError(result.stderr.strip() or f"git {' '.join(args)} failed")
        return result.stdout.strip()

    @staticmethod
    def _normalize_test_command(test_command: Sequence[str]) -> tuple[str, ...]:
        command = tuple(str(item) for item in test_command)
        if not command:
            raise SelfImprovementError("proposal test command must not be empty")
        first = command[0]
        if first in VENV_PYTHON_NAMES:
            return (sys.executable, *command[1:])
        candidate = Path(first)
        if candidate.is_absolute() and candidate.name in {"python", "python3"}:
            return (sys.executable, *command[1:])
        if candidate.name == "pytest":
            return (sys.executable, "-m", "pytest", *command[1:])
        return command

    def _coerce_test_command(self, test_command: str | Sequence[str] | None) -> tuple[tuple[str, ...], bool]:
        """Turn the model's command spelling into an argv list, or fall back.

        The same command must normalize to the same tokens however it is
        spelled: a shell string, one argv list, a list of command lines, or a
        list of stages. Only a ONE-element shell-shaped list used to be
        split, so a two- or three-stage command kept its whole first line as
        argv[0] and every such submission died later as "proposal validation
        executable is unavailable" -- 17 rows in the live ledger between
        2026-09-20 and 2026-09-25, each one a complete change set discarded
        over spelling. The first declared stage is the bounded check; further
        stages are recorded instead of silently dropped, and a nested argv
        stage used to be replaced by the default suite the same way. An
        unusable command still falls back to the default because the intent
        ("run the tests") is unambiguous.
        """
        if test_command is None:
            return DEFAULT_TEST_COMMAND, True
        if isinstance(test_command, str):
            if self._contains_control(test_command):
                raise SelfImprovementError("proposal test command contains control characters")
            stages: list[tuple[str, ...]] = [tuple(self._split_shell_command(test_command))]
        else:
            items = list(test_command)
            if any(self._contains_control(str(item)) for item in items):
                raise SelfImprovementError("proposal test command contains control characters")
            # The first element decides the shape, and only the first: a flat
            # argv list whose TAIL carries a quoted pytest expression
            # (`-k "planner or memory"`, ledger row 20916) must keep that
            # element as ONE token, while a command-line list always leads with
            # the separator-carrying command.
            if items and (isinstance(items[0], (list, tuple)) or self._is_command_line_head(str(items[0]))):
                stages = [self._stage_test_command(item) for item in items]
            else:
                # A flat argv list: its elements are tokens, never re-split, so
                # a quoted pytest expression such as `-k "planner or memory"`
                # stays ONE token exactly as before.
                stages = [tuple(str(item) for item in items)]
        if any(self._contains_control(part) for stage in stages for part in stage):
            raise SelfImprovementError("proposal test command contains control characters")
        stages = [stage for stage in stages if stage and stage[0].strip()]
        if not stages:
            return DEFAULT_TEST_COMMAND, True
        if len(stages) > 1 and self.runtime_log is not None:
            # Recorded, not dropped: the harness suite runs regardless, so an
            # extra stage is a declaration the proposal made about itself and
            # a silent substitution would hide it.
            self.runtime_log.write("test_command_stages_skipped", {"stages": [list(stage) for stage in stages[1:]]})
        return self._normalize_test_command(stages[0]), False

    @classmethod
    def _is_command_line_head(cls, value: str) -> bool:
        """True when the FIRST element of a list is a whole command, not argv[0].

        A command line carries shell separators; a genuine argv[0] does not. A
        checkout whose path contains a space is the one false positive, so an
        element that already names an existing file or a PATH executable is
        treated as argv[0] even when it contains a space.
        """
        if not cls._is_command_line(value):
            return False
        candidate = value.strip()
        if Path(candidate).is_file() or Path(candidate).is_dir():
            return False
        return shutil.which(candidate) is None

    @staticmethod
    def _is_command_line(value: str) -> bool:
        """True when a single string carries shell separators, i.e. is a command."""
        return any(marker in value for marker in (" ", "&&", ";", "|"))

    @classmethod
    def _stage_test_command(cls, stage: object) -> tuple[str, ...]:
        """Tokens of one declared stage: an argv list, or one command line.

        A list is already tokenized and is passed through. A string carrying a
        shell separator is split, so a quoted argument (``-c "import x; y"``)
        survives as one token; a bare word is a one-token argv.
        """
        if isinstance(stage, (list, tuple)):
            return tuple(str(item) for item in stage)
        text = str(stage)
        if cls._is_command_line(text):
            return tuple(cls._split_shell_command(text))
        return (text,)

    @staticmethod
    def _contains_control(value: str) -> bool:
        return any(char in value for char in ("\x00", "\n", "\r"))

    @staticmethod
    def _split_shell_command(value: str) -> list[str]:
        try:
            return shlex.split(value)
        except ValueError as exc:
            raise SelfImprovementError(f"proposal test command could not be parsed: {value}") from exc

    @staticmethod
    def _validate_test_command_shape(command: Sequence[str]) -> None:
        """Reject only genuinely unusable commands; shell strings are coerced."""
        if any("\x00" in item or "\n" in item or "\r" in item for item in command):
            raise SelfImprovementError("proposal test command contains control characters")

    def _normalize_relative_path(self, raw: str) -> Path:
        """Map a model-supplied path onto a repo-relative path inside the worktree.

        Models emit absolute paths into the repository and Windows separators;
        both are unambiguous and are normalized instead of rejected. Anything
        outside the repository or containing ``..`` is refused with the expected
        form so the next attempt can be corrected in one step.
        """
        candidate = raw.strip().replace("\\", "/")
        if not candidate:
            raise SelfImprovementError(
                f"invalid proposal path: {raw!r} — expected a path inside {self.root}, e.g. skynet/store.py"
            )
        path = Path(candidate)
        if path.is_absolute():
            resolved = path.resolve()
            if resolved != self.root and self.root not in resolved.parents:
                raise SelfImprovementError(
                    f"invalid proposal path: {raw} — expected a path inside {self.root}, e.g. skynet/store.py"
                )
            path = Path(resolved.relative_to(self.root).as_posix())
        if ".." in path.parts:
            raise SelfImprovementError(
                f"invalid proposal path: {raw} — expected a path inside {self.root}, e.g. skynet/store.py"
            )
        return path

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _read_proposals(self) -> dict[str, dict[str, object]]:
        try:
            raw = self.proposals_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            self._quarantine_registry(f"unreadable registry: {exc}")
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._quarantine_registry(f"invalid registry json: {exc}")
            return {}
        return value if isinstance(value, dict) else {}

    def _quarantine_registry(self, reason: str) -> None:
        """Preserve a corrupt registry instead of letting the next write erase it."""
        stamp = utc_now().replace("-", "").replace(":", "").replace(".", "")
        target = self.root / "state" / "quarantine" / f"self-improvement-proposals-{stamp}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(self.proposals_path), str(target))
        except OSError:
            log.warning("could not quarantine corrupt self-improvement registry", exc_info=True)
            return
        log.warning("quarantined corrupt self-improvement registry to %s (%s)", target, reason)
        # Registry corruption is a cycle-level deviation: route it through the
        # durable store when one is injected, not only the runtime projection.
        self._emit("registry_corrupted", {"quarantined": str(target), "reason": reason})

    def _write_proposals(self, proposals: dict[str, dict[str, object]]) -> None:
        self.proposals_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.proposals_path.name}.", dir=self.proposals_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(proposals, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.proposals_path)
            self._fsync_directory(self.proposals_path.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def mark_reboot_result(self, proposal_id: str, result: dict[str, object]) -> None:
        proposals = self._read_proposals()
        proposal = proposals.get(proposal_id)
        if proposal is None:
            return
        if result.get("completed"):
            proposal["status"] = "accepted"
        elif result.get("rolled_back"):
            proposal["status"] = "rolled_back"
        else:
            return
        proposal["reboot_result"] = result
        self._write_proposals(proposals)
        self._cleanup_worktree(str(proposal.get("worktree", "")))

    def _head_is_ancestor_of(self, commit: str) -> bool:
        """Is HEAD still an ancestor of ``commit``, i.e. was it left behind?

        A commit that is not an ancestor of HEAD is only *evidence of a
        rollback* when HEAD is still an ancestor of it: the tree moved
        backwards below that commit. The startup rollback script enforces the
        same one-way rule before it resets anything. Conservative on purpose:
        every answer other than a clean "HEAD is an ancestor" reads as ``False``,
        so an undecidable git call withholds the verdict instead of inventing one.
        """
        try:
            check = subprocess.run(
                ["git", "merge-base", "--is-ancestor", "HEAD", commit],
                cwd=self.root, capture_output=True, text=True, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return check.returncode == 0

    def reconcile_awaiting_reboot(self) -> list[dict[str, object]]:
        """Resolve proposals left awaiting_reboot when no reboot guard exists.

        An external rollback deletes both the reboot request and the guard, so
        the registry would otherwise never learn the outcome. When no reboot
        window is open, the promoted commit's presence in history decides the
        result. A proposal whose outcome cannot be decided yet (no promoted
        commit, or an ancestry check that fails for a reason other than the
        commit being absent) stays ``awaiting_reboot`` instead of being written
        down as rolled back.
        """
        proposals = self._read_proposals()
        violations = check_registry_integrity(self.proposals_path)
        if violations:
            log.warning("self-improvement registry integrity violations: %s", violations[:5])
        if not proposals:
            return []
        window_open, guarded_proposal = self._open_guard_proposal()
        if window_open and guarded_proposal is None:
            # An unreadable guard names no proposal, so nothing can be judged
            # safely and the whole sweep is withheld.
            return []
        resolved: list[dict[str, object]] = []
        changed = False
        for proposal_id, record in proposals.items():
            status = str(record.get("status", ""))
            if status not in {"awaiting_reboot", "promoting"}:
                continue
            if window_open and proposal_id == guarded_proposal:
                # Withhold only the proposal still inside its open window; every
                # other awaiting entry can be decided from git history now.
                continue
            commit = str(record.get("promoted_commit") or record.get("commit") or "")
            if not commit:
                log.warning("cannot reconcile %s: no promoted commit recorded", proposal_id)
                continue
            check = subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                cwd=self.root, capture_output=True, text=True, check=False,
            )
            if check.returncode not in (0, 1):
                log.warning(
                    "cannot reconcile %s: ancestry check on %s failed (%s)",
                    proposal_id, commit, (check.stderr or "").strip()[:200] or check.returncode,
                )
                continue
            in_history = check.returncode == 0
            # Not-an-ancestor is necessary but not sufficient for a rollback:
            # only when HEAD is still an ancestor of the commit did the tree
            # move *backwards* below it. A commit on an unrelated line of
            # history (rewritten elsewhere) satisfies neither direction, and
            # writing it down as rolled_back would record a decision the
            # evidence never made.
            lost_to_rollback = in_history is False and self._head_is_ancestor_of(commit)
            if status == "promoting":
                # A crash between the fast-forward merge and the reboot request
                # leaves the promotion merged but unrecorded. Ancestry decides:
                # merged means the request must be recreated; not merged means
                # the proposal is safe to promote again.
                if in_history:
                    record["status"] = "awaiting_reboot"
                    record["promoted_commit"] = commit
                    self._write_recovery_request(proposal_id, commit, str(record.get("base_commit", "")))
                    resolved.append({"proposal_id": proposal_id, "status": "awaiting_reboot", "request_recreated": True})
                elif lost_to_rollback:
                    record["status"] = "validated"
                    self._cleanup_worktree(str(record.get("worktree", "")))
                    resolved.append({"proposal_id": proposal_id, "status": "validated", "reason": "promotion interrupted before the merge"})
                else:
                    # Undecidable: the commit is neither reachable from HEAD nor
                    # contains it, so "not merged" cannot be told from "merged
                    # and then rolled back". Re-promoting on that guess could
                    # duplicate work; the row is left for a decidable review.
                    log.warning(
                        "cannot decide interrupted promotion %s: %s is neither an ancestor nor a descendant of HEAD",
                        proposal_id, commit,
                    )
                    continue
                changed = True
                continue
            if in_history:
                record["status"] = "accepted"
                record["reboot_result"] = {"completed": True, "rolled_back": False, "reconciled": True}
            elif lost_to_rollback:
                record["status"] = "rolled_back"
                record["reboot_result"] = {"completed": False, "rolled_back": True, "reconciled": True}
            else:
                # A missing commit plus an unrelated line of history is not proof
                # of a rollback; recording one would make the registry assert a
                # fact git never established.
                log.warning(
                    "cannot reconcile %s: %s is neither an ancestor nor a descendant of HEAD; withholding the rollback verdict",
                    proposal_id, commit,
                )
                continue
            self._cleanup_worktree(str(record.get("worktree", "")))
            resolved.append({"proposal_id": proposal_id, "status": record["status"]})
            changed = True
        if changed:
            self._write_proposals(proposals)
        return resolved

    def _write_recovery_request(self, proposal_id: str, commit: str, rollback_commit: str) -> None:
        """Recreate a reboot request lost between the merge and the request write."""
        if not rollback_commit:
            log.warning("cannot recreate reboot request for %s: no base commit recorded", proposal_id)
            return
        self.request_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"proposal_id": proposal_id, "commit": commit, "rollback_commit": rollback_commit, "health": {"recovered": True}}
        fd, temporary = tempfile.mkstemp(prefix=".reboot-request.", dir=self.request_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.request_path)
            self._fsync_directory(self.request_path.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _open_guard_proposal(self) -> tuple[bool, str | None]:
        """Report whether a reboot window is open and which proposal it covers.

        A finished window leaves its guard file on disk: ``RebootGuard.observe``
        writes ``completed_at`` on a healthy end or ``rolled_back`` after a
        rollback, and never unlinks the file. Only the proposal named by a still
        open guard is withheld from reconciliation. Unreadable or malformed
        guard state is reported as open with no identifiable proposal, so
        nothing is resolved from a guard nobody can inspect.
        """
        path = self.root / "state" / "reboot-guard.json"
        try:
            guard = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False, None
        except (OSError, json.JSONDecodeError, ValueError):
            return True, None
        if not isinstance(guard, dict):
            return True, None
        if guard.get("completed_at") or guard.get("rolled_back"):
            return False, None
        proposal_id = guard.get("proposal_id")
        return True, str(proposal_id) if proposal_id else None

    def _cleanup_worktree(self, worktree: str) -> None:
        path = Path(worktree)
        if not worktree or not path.exists():
            return
        with contextlib.suppress(SelfImprovementError):
            self._git(["worktree", "remove", "--force", str(path)])

    def prune_orphan_worktrees(self) -> list[str]:
        """Delete stale directories under the worktree root that git no longer tracks.

        Failed or interrupted proposals can leave orphaned checkouts behind
        (each one carries a full copy of the tree), so startup sweeps them.
        The same startup pass expires validated proposals that were never
        promoted, since their worktrees are otherwise never reclaimed.

        ``root`` is derived, not chosen: ``__init__`` walks up from the
        organism's own root to the nearest ``.skynet-improvements`` directory
        and otherwise takes ``root.parent / ".skynet-improvements"``. When the
        organism runs from inside that collection (every gate worktree does, and
        the collection then holds ``skynet/`` itself), the derived root is the
        collection, so a durable artifact parked beside it is inside the swept
        tree. A plain directory the organism created has no ``.git`` entry while
        a real checkout does, and ``Path.is_dir`` follows symbolic links, so the
        two rules below keep that boundary honest and both err towards
        preserving bytes:

        * a child is skipped when it is a symlink, or when it has no ``.git``
          file or directory: a directory the organism itself created is not a
          stale checkout, and deleting one silently destroys evidence;
        * a child whose name matches a pattern named in
          ``SKYNET_KEEP_WORKTREES`` (comma-separated ``fnmatch`` patterns, e.g.
          ``skynet-scratch*``) is never deleted, so a named durable directory
          can be pinned without editing and redeploying this module.

        A skipped child is logged at INFO with the rule that spared it.
        """
        self.sweep_stale_proposals()
        root = self.worktree_root
        if root == self.root or not root.is_dir():
            return []
        try:
            listed = self._git(["worktree", "list", "--porcelain"])
        except SelfImprovementError:
            log.warning("cannot list worktrees; skipping the orphan sweep", exc_info=True)
            return []
        known = {line.split(" ", 1)[1].strip() for line in listed.splitlines() if line.startswith("worktree ")}
        if not known:
            return []
        exempt = tuple(
            pattern.strip() for pattern in os.getenv(ORPHAN_SWEEP_EXEMPT_ENV, "").split(",") if pattern.strip()
        )
        removed: list[str] = []
        for child in sorted(root.iterdir()):
            if child == self.root:
                continue
            if child.is_symlink():
                log.info("orphan sweep spared %s: symbolic link", child)
                continue
            if not child.is_dir():
                continue
            if str(child) in known:
                continue
            if not (child / ".git").exists():
                log.info("orphan sweep spared %s: no .git entry, so not a checkout", child)
                continue
            if any(fnmatch.fnmatch(child.name, pattern) for pattern in exempt):
                log.info("orphan sweep spared %s: matched %s", child, ORPHAN_SWEEP_EXEMPT_ENV)
                continue
            shutil.rmtree(child, ignore_errors=True)
            if not child.exists():
                removed.append(str(child))
        if removed:
            try:
                self._git(["worktree", "prune"])
            except SelfImprovementError:
                log.warning("worktree prune failed after removing orphans", exc_info=True)
        return removed

    def sweep_stale_proposals(self) -> list[dict[str, object]]:
        """Reject validated or warned proposals that outlived the TTL."""
        try:
            ttl_hours = float(os.getenv("SKYNET_PROPOSAL_TTL_HOURS", str(DEFAULT_PROPOSAL_TTL_HOURS)))
        except ValueError:
            ttl_hours = DEFAULT_PROPOSAL_TTL_HOURS
        try:
            warned_ttl_hours = float(os.getenv("SKYNET_WARNED_TTL_HOURS", str(DEFAULT_WARNED_TTL_HOURS)))
        except ValueError:
            warned_ttl_hours = DEFAULT_WARNED_TTL_HOURS
        proposals = self._read_proposals()
        now = utc_datetime_now()
        swept: list[dict[str, object]] = []
        changed = False
        for proposal_id, record in proposals.items():
            status = record.get("status")
            if status == "validated":
                stamp = record.get("validated_at") or record.get("created_at")
                effective_ttl = ttl_hours
                expired = f"validated proposal expired after {ttl_hours:g}h without promotion"
            elif status == WARNED_PROTECTED:
                # An unacknowledged warning holds a worktree only until the
                # identical resubmission; a shorter TTL reclaims it sooner when
                # the model moved on, so warnings cannot accumulate into the
                # worktree quota.
                stamp = record.get("warned_at") or record.get("created_at")
                effective_ttl = warned_ttl_hours
                expired = f"warned proposal expired after {warned_ttl_hours:g}h without resubmission"
            else:
                continue
            if not isinstance(stamp, str):
                continue
            try:
                created = parse_timestamp(stamp)
            except ValueError:
                continue
            if (now - created).total_seconds() < effective_ttl * 3600:
                continue
            record["status"] = "rejected"
            record["failure_class"] = "stale_proposal"
            record["failure"] = expired
            self._cleanup_worktree(str(record.get("worktree", "")))
            swept.append({"proposal_id": proposal_id, "status": "rejected"})
            changed = True
        if changed:
            self._write_proposals(proposals)
            if self.runtime_log is not None:
                self.runtime_log.write("stale_proposals_swept", {"proposals": swept, "ttl_hours": ttl_hours})
        return swept

    def _modified_tracked_paths(self) -> list[str]:
        # ``_git`` strips leading whitespace, which erases the porcelain status
        # column of the first line, so the raw subprocess output is parsed here.
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise SelfImprovementError(result.stderr.strip() or "git status failed")
        paths: list[str] = []
        for line in result.stdout.splitlines():
            relative = line[3:].strip()
            if not relative:
                continue
            if " -> " in relative:
                relative = relative.split(" -> ", 1)[1]
            paths.append(relative.strip('"'))
        return paths

    def _worktree_changed_paths(self, proposal: ImprovementProposal) -> list[str] | None:
        """Every path the proposal touched, including untracked new files.

        Returns ``None`` when the worktree cannot be read. An empty list means
        the change set is genuinely empty; the two must not be conflated, or a
        git failure would silently disable the protected-path guard.
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=proposal.worktree,
                capture_output=True,
                text=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode:
            return None
        paths: list[str] = []
        for line in result.stdout.splitlines():
            relative = line[3:].strip()
            if not relative:
                continue
            if " -> " in relative:
                relative = relative.split(" -> ", 1)[1]
            paths.append(relative.strip('"'))
        return paths

    @staticmethod
    def _is_gate_protected(path: str) -> bool:
        normalized = path.replace("\\", "/").removeprefix("./")
        if normalized == "tests" or normalized.startswith("tests/"):
            return True
        if normalized in GATE_PROTECTED_ROOT_FILES:
            return True
        return normalized in GATE_PROTECTED_PATHS

    def _gate_protected_offenders(self, proposal: ImprovementProposal, applied_paths: Sequence[str] | None) -> list[str]:
        """Return the gate-protected paths a proposal touches, if any.

        The suite, its configuration and the meta-loop decide whether any change
        is safe, so a proposal that can rewrite them could neuter the gate and
        validate itself. The verdict is a warning now: the first submission is
        warned and only an identical resubmission is validated. The check runs on
        the applied paths when the caller has them and falls back to the worktree
        diff otherwise.
        """
        # The production caller always passes the applied paths; the worktree
        # diff is a defensive fallback. An explicitly empty list means "known to
        # be empty", and an unreadable worktree is refused rather than assumed
        # harmless.
        if applied_paths is not None:
            paths = list(applied_paths)
        else:
            resolved = self._worktree_changed_paths(proposal)
            if resolved is None:
                raise SelfImprovementError(
                    "could not determine the proposal's changed paths; refusing to validate",
                    reason_code="unknown_paths",
                )
            paths = resolved
        return sorted({path for path in paths if self._is_gate_protected(path)})

    def _quarantine_dirty_worktree(self) -> list[str]:
        """Preserve modified tracked files, then restore a clean main worktree.

        A model that edits sources directly through ``bash`` leaves the main
        tree dirty and would otherwise be refused at the proposal boundary. The
        prototype is moved into ``state/quarantine`` rather than discarded, and
        only tracked files are touched: untracked files and the organism's own
        ``state/`` tree are never moved.
        """
        entries = self._modified_tracked_paths()
        if not entries:
            return []
        for relative in entries:
            if relative == "state" or relative.startswith("state/"):
                raise SelfImprovementError(
                    f"refusing to quarantine organism state: {relative}; commit or revert it manually before proposing"
                )
        stamp = utc_now().replace("-", "").replace(":", "").replace(".", "")
        quarantine_dir = self.root / "state" / "quarantine" / f"worktree-{stamp}"
        moved: list[tuple[str, Path, Path]] = []
        try:
            for relative in entries:
                source = self.root / relative
                if not source.exists():
                    continue
                destination = quarantine_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                moved.append((relative, source, destination))
            if moved:
                self._git(["checkout", "--", *[relative for relative, _, _ in moved]])
        except Exception as exc:
            for _relative, source, destination in moved:
                if destination.exists() and not source.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(destination), str(source))
            raise SelfImprovementError(
                f"could not quarantine dirty worktree ({exc}); commit or revert the modified tracked files manually"
            ) from exc
        quarantined = [relative for relative, _, _ in moved]
        if quarantined and self.runtime_log is not None:
            self.runtime_log.write("worktree_quarantined", {"files": quarantined, "quarantine": str(quarantine_dir)})
        return quarantined

    def _enforce_worktree_quota(self) -> None:
        """Keep the registered worktree collection below a hard cap.

        A warned proposal holds a worktree only so the identical resubmission
        can run the gate without being re-applied. Those records are reclaimed
        (worktree removed, record dropped) before the quota can ever refuse new
        work: since the gate's own inputs are warn-only, a run that edits tests
        or the meta-loop must not be able to fill the quota with un-resubmitted
        warnings and wedge self-improvement. A reclaimed warning is simply
        re-issued if the same change is proposed again.
        """
        try:
            cap = int(os.getenv("SKYNET_MAX_WORKTREES", str(DEFAULT_MAX_WORKTREES)))
        except ValueError:
            cap = DEFAULT_MAX_WORKTREES
        if cap <= 0:
            return
        proposals = self._read_proposals()

        def active(record: Mapping[str, object]) -> bool:
            return bool(record.get("worktree")) and record.get("status") not in TERMINAL_PROPOSAL_STATUSES

        if sum(1 for record in proposals.values() if active(record)) < cap:
            return
        for record in proposals.values():
            if record.get("status") in TERMINAL_PROPOSAL_STATUSES and record.get("worktree"):
                self._cleanup_worktree(str(record.get("worktree", "")))
        warned = sorted(
            ((proposal_id, record) for proposal_id, record in proposals.items() if active(record) and record.get("status") == WARNED_PROTECTED),
            key=lambda item: str(item[1].get("warned_at") or item[1].get("created_at") or ""),
        )
        reclaimed: list[str] = []
        for proposal_id, record in warned:
            if sum(1 for item in proposals.values() if active(item)) < cap:
                break
            self._cleanup_worktree(str(record.get("worktree", "")))
            proposals.pop(proposal_id, None)
            reclaimed.append(proposal_id)
        if reclaimed:
            self._write_proposals(proposals)
            if self.runtime_log is not None:
                self.runtime_log.write("warned_worktrees_reclaimed", {"proposals": reclaimed, "cap": cap})
        active_records = [record for record in proposals.values() if active(record)]
        if len(active_records) >= cap:
            raise SelfImprovementError(
                f"worktree quota reached ({len(active_records)} active of {cap}); promote or discard a proposal first",
                reason_code="worktree_quota",
            )

    def propose(self, on_dirty: str = "quarantine") -> ImprovementProposal:
        self.last_quarantined = []
        if not CheckpointManager(self.root).is_clean(allow_untracked=True):
            if on_dirty != "quarantine":
                raise SelfImprovementError("self-improvement requires no modified tracked files in the main worktree")
            self.last_quarantined = self._quarantine_dirty_worktree()
            if not CheckpointManager(self.root).is_clean(allow_untracked=True):
                raise SelfImprovementError("self-improvement requires no modified tracked files in the main worktree")
        # Expire stale proposals on every proposal, not only at process start:
        # a warned worktree that the model never resubmitted must not wait for a
        # restart before its slot is reclaimed.
        self.sweep_stale_proposals()
        self._enforce_worktree_quota()
        proposal_id = uuid4().hex
        worktree = self.worktree_root / proposal_id
        worktree.parent.mkdir(parents=True, exist_ok=True)
        self._git(["worktree", "add", "--detach", str(worktree), "HEAD"])
        proposal = ImprovementProposal(proposal_id, worktree, self._git(["rev-parse", "HEAD"]))
        if self.runtime_log is not None:
            self.runtime_log.write("proposal_created", {"proposal_id": proposal_id, "worktree": str(worktree), "base_commit": proposal.base_commit})
        return proposal

    def apply_files(self, proposal: ImprovementProposal, files: Mapping[str, str]) -> list[str]:
        normalized: list[str] = []
        for relative, content in files.items():
            path = self._normalize_relative_path(relative)
            target = (proposal.worktree / path).resolve()
            if proposal.worktree not in target.parents:
                raise SelfImprovementError(f"proposal escapes worktree: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            normalized.append(path.as_posix())
        if self.runtime_log is not None:
            self.runtime_log.write(
                "proposal_files_applied",
                {
                    "proposal_id": proposal.proposal_id,
                    "files": sorted(normalized),
                    "change_digest": change_content_digest(proposal.worktree, normalized),
                },
            )
        return normalized

    def apply_changes(self, proposal: ImprovementProposal, changes: Sequence[Mapping[str, object]]) -> dict[str, object]:
        """Apply small exact-match edits inside an isolated proposal worktree.

        An anchor that arrives with literal escape artifacts from tool-call
        transport is decoded once, and only when the decoded form matches
        exactly once, so stale anchors still fail loudly.
        """
        changed: list[str] = []
        repairs = 0
        for change in changes:
            relative = change.get("path")
            operation = change.get("operation", "replace")
            if not isinstance(relative, str) or not isinstance(operation, str):
                raise SelfImprovementError("invalid patch payload")
            path = self._normalize_relative_path(relative)
            target = (proposal.worktree / path).resolve()
            if proposal.worktree not in target.parents:
                raise SelfImprovementError(f"proposal escapes worktree: {relative}")
            if operation == "create":
                if target.exists():
                    raise SelfImprovementError(f"patch target already exists: {relative}")
                content = change.get("content")
                if not isinstance(content, str):
                    raise SelfImprovementError("create patch requires content")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            elif operation in {"replace", "insert_before", "insert_after", "delete"}:
                if not target.is_file():
                    raise SelfImprovementError(f"patch target does not exist: {relative}")
                old = change.get("old", change.get("anchor"))
                if not isinstance(old, str) or not old:
                    raise SelfImprovementError("patch requires a non-empty old or anchor value")
                current = target.read_text(encoding="utf-8")
                matches = current.count(old)
                if matches == 0:
                    decoded = _transport_unescape(old)
                    decoded_matches = current.count(decoded)
                    if decoded != old and decoded_matches == 1:
                        old = decoded
                        matches = decoded_matches
                        repairs += 1
                if matches == 0:
                    raise SelfImprovementError(f"patch anchor not found (0 matches): {relative}", reason_code="anchor_not_found", match_count=0)
                if matches != 1:
                    raise SelfImprovementError(f"patch anchor must match exactly once ({matches} matches given): {relative}", reason_code="anchor_ambiguous", match_count=matches)
                content = change.get("content", change.get("new", ""))
                if not isinstance(content, str):
                    raise SelfImprovementError("patch content must be text")

                # An insert whose join point carries no newline on either side
                # fuses the anchor's last line with the inserted text's first
                # line, e.g. "...unconstrained.- Author of ...". That is a
                # fragmented-hunk defect of the edit format rather than a
                # verdict on the idea: edit-format studies that vary the hunk
                # measure fragmented hunks as the failure mode
                # (arXiv:2604.27296v1 sec. 3.3), and wrapped prose here makes
                # the no-newline case common (106 of 160 historical insert
                # edits in state/runtime.jsonl carry this shape). Refuse it as
                # its own reason code before writing, so the fix is a newline
                # and not an identical resubmission.
                if operation in {"insert_before", "insert_after"}:
                    joins_forward = operation == "insert_after" and not old.endswith("\n") and not content.startswith("\n")
                    joins_backward = operation == "insert_before" and not content.endswith("\n") and not old.startswith("\n")
                    if joins_forward or joins_backward:
                        raise SelfImprovementError(
                            f"{operation} would join the anchor and the inserted text onto one line: "
                            f"neither side carries a newline, so add a leading or trailing one (path: {relative})",
                            reason_code="anchor_boundary_join",
                        )
                if operation == "replace":
                    replacement = content
                elif operation == "insert_before":
                    replacement = content + old
                elif operation == "insert_after":
                    replacement = old + content
                else:
                    replacement = ""
                target.write_text(current.replace(old, replacement, 1), encoding="utf-8")
            else:
                raise SelfImprovementError(f"unsupported patch operation: {operation}")
            changed.append(path.as_posix())
        if not changed:
            raise SelfImprovementError("proposal has no changes")
        digest = change_content_digest(proposal.worktree, changed)
        if self.runtime_log is not None:
            self.runtime_log.write("proposal_changes_applied", {"proposal_id": proposal.proposal_id, "files": sorted(set(changed)), "change_digest": digest, "anchor_repairs": repairs})
        result: dict[str, object] = {"files": sorted(set(changed)), "change_digest": digest}
        if repairs:
            result["anchor_repairs"] = repairs
        return result

    def apply_patch(self, proposal: ImprovementProposal, patch: str) -> list[str]:
        """Apply a unified diff with ``git apply`` inside the worktree.

        The template path exists because exact multi-line anchors are brittle in
        tool-call transport; a diff states the intended change once and either
        applies cleanly or fails as ``patch_apply_failed``. The gate, registry
        and promotion path are identical to the anchor path.
        """
        if not patch.strip():
            raise SelfImprovementError("patch must not be empty", reason_code="patch_apply_failed")
        if self._contains_control(patch.replace("\n", "").replace("\r", "")):
            raise SelfImprovementError("patch contains control characters", reason_code="patch_apply_failed")
        structure_error = patch_structure_error(patch)
        if structure_error is not None:
            raise SelfImprovementError(
                f"patch payload is structurally damaged: {structure_error}",
                reason_code=PATCH_PAYLOAD_CORRUPT,
            )
        try:
            completed = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "--index"],
                cwd=proposal.worktree,
                input=patch,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise SelfImprovementError(f"git apply could not run: {exc}", reason_code="patch_apply_failed") from exc
        if completed.returncode:
            output = (completed.stderr or completed.stdout).strip()[:500]
            raise SelfImprovementError(
                f"patch could not be applied with git apply: {output}",
                reason_code="patch_apply_failed",
            )
        changed = self._worktree_changed_paths(proposal)
        if changed is None:
            raise SelfImprovementError(
                "patch applied but the changed paths could not be read; refusing the proposal",
                reason_code="patch_apply_failed",
            )
        digest = change_content_digest(proposal.worktree, changed)
        if self.runtime_log is not None:
            self.runtime_log.write("proposal_patch_applied", {"proposal_id": proposal.proposal_id, "files": sorted(set(changed)), "change_digest": digest})
        return sorted(set(changed))

    def _run_ruff_stage(self, worktree: Path, package: Path) -> None:
        """The project's ruff config, run from the proposal worktree.

        Ruff discovers ``pyproject.toml`` from the worktree, so the ruleset is
        exactly what ``scripts/test.sh`` enforces locally. The tree is clean, so
        any finding is a regression: unlike the retired mypy stage there is no
        baseline. An absent binary is recorded and skipped, like pyright.
        """
        executable = Path(sys.executable).parent / "ruff"
        if not executable.exists():
            if self.runtime_log is not None:
                self.runtime_log.write("gate_stage_skipped", {"stage": "ruff", "reason": "ruff is not installed"})
            return
        if not (worktree / "pyproject.toml").exists():
            # A minimal fixture worktree has no project config; judging it by
            # ruff's broad defaults would fail on fixture spacing, not on the
            # proposal. The real worktree always carries pyproject.toml (it is a
            # gate-protected file).
            if self.runtime_log is not None:
                self.runtime_log.write("gate_stage_skipped", {"stage": "ruff", "reason": "no pyproject.toml in the worktree"})
            return
        try:
            completed = subprocess.run(
                [str(executable), "check", str(worktree)],
                cwd=worktree, capture_output=True, text=True, check=False,
                timeout=STATIC_STAGE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SelfImprovementError(f"gate stage ruff could not run: {exc}") from exc
        if completed.returncode:
            output = (completed.stdout or completed.stderr)[-1500:]
            raise SelfImprovementError(f"gate stage ruff failed: {output}")

    def _run_pyright_stage(self, worktree: Path, package: Path) -> None:
        """Fail on any pyright error: the editor's language server is the bar.

        pyright is a dev dependency, so an absent binary is recorded and skipped
        like pyflakes. When present the tree is expected to be clean, so unlike
        mypy there is no baseline: any diagnostic is a regression. ``--pythonpath``
        pins import resolution to this venv even though a proposal worktree has
        no ``.venv`` of its own.
        """
        executable = Path(sys.executable).parent / "pyright"
        if not executable.exists():
            if self.runtime_log is not None:
                self.runtime_log.write("gate_stage_skipped", {"stage": "pyright", "reason": "pyright is not installed"})
            return
        try:
            completed = subprocess.run(
                [str(executable), "--pythonpath", sys.executable, str(package)],
                cwd=worktree, capture_output=True, text=True, check=False,
                timeout=STATIC_STAGE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SelfImprovementError(f"gate stage pyright could not run: {exc}") from exc
        if completed.returncode:
            output = (completed.stdout or completed.stderr)[-1500:]
            raise SelfImprovementError(f"gate stage pyright failed: {output}")

    def _run_gate_stage(
        self,
        stage: str,
        argv: Sequence[str],
        worktree: Path,
        *,
        timeout: float = GATE_STAGE_TIMEOUT_SECONDS,
        env: dict[str, str] | None = None,
    ) -> None:
        """Run one harness-owned gate stage; a failure is a verdict, not a crash."""
        try:
            completed = subprocess.run(
                list(argv), cwd=worktree, capture_output=True, text=True,
                check=False, timeout=timeout, env=env,
            )
        except FileNotFoundError as exc:
            raise SelfImprovementError(
                f"gate stage {stage} executable is unavailable: {argv[0]!r}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SelfImprovementError(f"gate stage {stage} timed out after {timeout:g}s") from exc
        if completed.returncode:
            output = completed.stderr[-1000:] or completed.stdout[-1000:]
            raise SelfImprovementError(f"gate stage {stage} failed: {output}")

    def validate_and_commit(
        self,
        proposal: ImprovementProposal,
        test_command: str | Sequence[str] = DEFAULT_TEST_COMMAND,
        *,
        applied_paths: Sequence[str] | None = None,
        fingerprint: str = "",
    ) -> str:
        test_command, fallback = self._coerce_test_command(test_command)
        if fallback and self.runtime_log is not None:
            self.runtime_log.write("test_command_fallback", {"proposal_id": proposal.proposal_id, "default_command": list(DEFAULT_TEST_COMMAND)})
        self._validate_test_command_shape(test_command)
        self._validate_test_paths(proposal, test_command)
        # A gitignored file would pass in the worktree and then vanish from the
        # commit, promoting broken code; decide it before the expensive stages.
        if applied_paths:
            self._reject_ignored_paths(proposal, applied_paths)
        # A proposal that touches the gate's own inputs is only warned: the
        # first submission of a fingerprint is recorded as warned_protected and
        # the caller turns the warning into a tool result. The identical
        # resubmission finds its record here, acknowledges it, and falls through
        # to the normal verification stages.
        offenders = self._gate_protected_offenders(proposal, applied_paths)
        if offenders:
            if self._warned_record_for_fingerprint(self._read_proposals(), fingerprint) is None:
                raise ProtectedPathWarning(
                    offenders,
                    fingerprint,
                    f"proposal modifies gate-protected path(s): {', '.join(offenders)}; "
                    "warning issued, an identical resubmission proceeds through the full gate",
                )
            self._emit(
                "gate_protected_warning_acknowledged",
                {
                    "proposal_id": proposal.proposal_id,
                    "paths": offenders,
                    "change_fingerprint": fingerprint,
                    "reason": "identical resubmission acknowledged; proceeding through the normal verification stages",
                },
            )
        # Both stages only make sense when the worktree actually carries the
        # package under test; a minimal fixture without it has nothing to prove.
        package = proposal.worktree / "skynet"
        if package.is_dir():
            self._run_gate_stage(
                "selfcheck",
                [sys.executable, "-c", GATE_SELFCHECK_SNIPPET],
                proposal.worktree,
                env={**os.environ, "SKYNET_PROPOSAL_WORKTREE": str(proposal.worktree)},
            )
            self._run_gate_stage(
                "compileall",
                [sys.executable, "-m", "compileall", "-q", str(package)],
                proposal.worktree,
            )
            self._run_gate_stage(
                "import-smoke",
                [sys.executable, "-c", GATE_IMPORT_SMOKE_SNIPPET],
                proposal.worktree,
            )
            self._run_ruff_stage(proposal.worktree, package)
            self._run_pyright_stage(proposal.worktree, package)
            # The registry showed six proposals declaring `grep -q`, a `python3
            # -c` mock or an import as their "test". A model-chosen command can
            # therefore never be the gate: the harness suite always runs, and the
            # model's command is only an additional bounded stage.
            if tuple(test_command) != self.gate_suite:
                self._run_gate_stage(
                    "suite",
                    list(self.gate_suite),
                    proposal.worktree,
                    timeout=TEST_COMMAND_TIMEOUT_SECONDS,
                    env=gate_suite_environment(),
                )
        try:
            test = subprocess.run(list(test_command), cwd=proposal.worktree, capture_output=True, text=True, check=False, timeout=TEST_COMMAND_TIMEOUT_SECONDS, env=gate_suite_environment())
        except FileNotFoundError as exc:
            raise SelfImprovementError(
                f"proposal validation executable is unavailable: {test_command[0]!r}; "
                f"the default command '{' '.join(DEFAULT_TEST_COMMAND)}' would have been used instead"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SelfImprovementError(f"proposal tests timed out after {TEST_COMMAND_TIMEOUT_SECONDS:g}s") from exc
        if test.returncode:
            output = test.stderr[-1000:] or test.stdout[-1000:]
            raise SelfImprovementError(f"proposal tests failed ({' '.join(test_command)}): {output}")
        if not self._git(["status", "--porcelain", "--untracked-files=all"], proposal.worktree):
            raise SelfImprovementError("proposal has no changes")
        self._git(["add", "--all"], proposal.worktree)
        self._git(["-c", "user.name=SkyNet Self-Improvement", "-c", "user.email=skynet@localhost", "commit", "-m", f"Self-improvement proposal {proposal.proposal_id}"], proposal.worktree)
        commit = self._git(["rev-parse", "HEAD"], proposal.worktree)
        if self.runtime_log is not None:
            self.runtime_log.write("proposal_validated", {"proposal_id": proposal.proposal_id, "commit": commit, "test_command": list(test_command)})
        return commit

    @staticmethod
    def _validate_test_paths(proposal: ImprovementProposal, test_command: Sequence[str]) -> None:
        """Check only clean relative path arguments, not mangled fragments.

        A transport-mangled argv element such as
        ``python3","-m","pytest","tests/x.py`` ends in ``.py`` but is not a
        path; validating it against the worktree produced a misleading
        "test path is unavailable" verdict. Only clean relative paths are
        checked, and a real miss names the offending item and the fallback.
        """
        for item in test_command:
            if not item.endswith((".py", ".pyc")):
                continue
            if item.startswith("-") or any(marker in item for marker in ('"', "'", ",", " ")):
                continue
            if Path(item).is_absolute():
                continue
            if not (proposal.worktree / item).is_file():
                raise SelfImprovementError(
                    f"proposal test path is unavailable: {item} (not found in the proposal worktree); "
                    f"the default command '{' '.join(DEFAULT_TEST_COMMAND)}' would have been used instead"
                )

    @staticmethod
    def _reject_ignored_paths(proposal: ImprovementProposal, applied_paths: Sequence[str]) -> None:
        """Refuse a proposal whose files a .gitignore would silently drop.

        ``git add --all`` skips ignored paths, so a passing gate could commit a
        tree that is missing the very file it imported. The commit must fail
        loudly instead.
        """
        result = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            cwd=proposal.worktree,
            input="\n".join(applied_paths) + "\n",
            capture_output=True,
            text=True,
            check=False,
        )
        ignored = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
        if ignored:
            raise SelfImprovementError(
                f"ignored proposal path cannot be committed: {', '.join(ignored)}; "
                "an ignored file cannot be committed, choose a tracked path or update .gitignore",
                reason_code="ignored_path",
            )

    def _record_validated(self, proposal: ImprovementProposal, commit: str, test_command: Sequence[str], *, fallback: bool = False) -> None:
        proposals = self._read_proposals()
        record: dict[str, object] = {
            "proposal_id": proposal.proposal_id,
            "worktree": str(proposal.worktree),
            "base_commit": proposal.base_commit,
            "commit": commit,
            "status": "validated",
            "test_command": list(test_command),
            "validated_at": utc_now(),
            # Size of the change, so a run of cosmetic edits can be told from
            # real work instead of being measured by hope.
            **self._diff_size(commit),
        }
        if fallback:
            record["test_command_fallback"] = True
        proposals[proposal.proposal_id] = record
        self._write_proposals(proposals)

    def _diff_size(self, commit: str) -> dict[str, int]:
        """Files and lines touched by a commit, bounded and failure-tolerant."""
        try:
            output = self._git(["show", "--numstat", "--format=", commit])
        except Exception:
            return {}
        files = 0
        added = 0
        removed = 0
        for line in output.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            files += 1
            for value, target in ((parts[0], "added"), (parts[1], "removed")):
                if value.isdigit():
                    if target == "added":
                        added += int(value)
                    else:
                        removed += int(value)
        if not files:
            return {}
        return {"files_changed": files, "lines_added": added, "lines_removed": removed, "lines_changed": added + removed}

    def cosmetic_streak(self, *, limit: int = 5, threshold_lines: int = 5) -> dict[str, Any]:
        """How many of the most recent validated proposals were cosmetic.

        A self-improving system that only polishes rust looks productive while
        changing nothing; a streak of tiny diffs is the signal.
        """
        proposals = self._read_proposals()
        records = [record for record in proposals.values() if isinstance(record, dict)]
        records.sort(key=lambda record: str(record.get("validated_at") or ""), reverse=True)
        streak = 0
        for record in records[:limit]:
            lines_changed = record.get("lines_changed")
            if not isinstance(lines_changed, int) or lines_changed >= threshold_lines:
                break
            streak += 1
        return {"streak": streak, "considered": min(limit, len(records)), "threshold_lines": threshold_lines}

    def promote_pending(self, proposal_id: str, health: dict[str, object]) -> str:
        record = self._read_proposals().get(proposal_id)
        if record is None or record.get("status") != "validated":
            raise SelfImprovementError(f"validated proposal not found: {proposal_id}")
        proposal = ImprovementProposal(
            proposal_id,
            Path(str(record["worktree"])),
            str(record["base_commit"]),
        )
        commit = str(record["commit"])
        try:
            proposals = self._read_proposals()
            proposals[proposal_id]["status"] = "promoting"
            self._write_proposals(proposals)
            try:
                promoted = self.promote(proposal, commit, health)
            except SelfImprovementError as exc:
                if "main worktree moved" not in str(exc):
                    raise
                rebased = self._rebase_onto_main(proposal)
                if rebased is None:
                    self._reject_stale_base(proposal_id)
                    raise
                new_base, new_commit = rebased
                proposals = self._read_proposals()
                proposals[proposal_id].update({"base_commit": new_base, "commit": new_commit})
                self._write_proposals(proposals)
                proposal = ImprovementProposal(proposal_id, proposal.worktree, new_base)
                promoted = self.promote(proposal, new_commit, health)
            proposals = self._read_proposals()
            proposals[proposal_id].update({"status": "awaiting_reboot", "promoted_commit": promoted})
            self._write_proposals(proposals)
            return promoted
        except Exception:
            proposals = self._read_proposals()
            if proposal_id in proposals and proposals[proposal_id].get("status") == "promoting":
                proposals[proposal_id]["status"] = "validated"
                self._write_proposals(proposals)
            raise

    def _rebase_onto_main(self, proposal: ImprovementProposal) -> tuple[str, str] | None:
        """Replay the proposal commit on the current HEAD, or abort on conflict."""
        new_base = self._git(["rev-parse", "HEAD"])
        result = subprocess.run(["git", "rebase", new_base], cwd=proposal.worktree, capture_output=True, text=True, check=False)
        if result.returncode:
            subprocess.run(["git", "rebase", "--abort"], cwd=proposal.worktree, capture_output=True, text=True, check=False)
            return None
        new_commit = self._git(["rev-parse", "HEAD"], cwd=proposal.worktree)
        return new_base, new_commit

    def _reject_stale_base(self, proposal_id: str) -> None:
        proposals = self._read_proposals()
        record = proposals.get(proposal_id)
        if record is None:
            return
        record["status"] = "rejected"
        record["failure_class"] = "stale_base_commit"
        record["failure"] = "main worktree moved and the proposal could not be rebased onto it"
        self._cleanup_worktree(str(record.get("worktree", "")))
        self._write_proposals(proposals)
        if self.runtime_log is not None:
            self.runtime_log.write("proposal_rejected", {"proposal_id": proposal_id, "failure_class": "stale_base_commit"})

    def request_reboot(self, proposal: ImprovementProposal, commit: str, health: dict[str, object]) -> None:
        if not health.get("ok"):
            raise SelfImprovementError("reboot request requires a passing health gate")
        self.request_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".reboot-request.", dir=self.request_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps({"proposal_id": proposal.proposal_id, "commit": commit, "rollback_commit": proposal.base_commit, "health": health}, indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.request_path)
            self._fsync_directory(self.request_path.parent)
            if self.runtime_log is not None:
                self.runtime_log.write("reboot_requested", {"proposal_id": proposal.proposal_id, "commit": commit, "rollback_commit": proposal.base_commit, "health": health})
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def promote(self, proposal: ImprovementProposal, commit: str, health: dict[str, object]) -> str:
        """Fast-forward the main worktree and record the previous commit for rollback."""
        if not health.get("ok"):
            raise SelfImprovementError("promotion requires a passing health gate")
        if self._git(["rev-parse", "HEAD"]) != proposal.base_commit:
            raise SelfImprovementError("main worktree moved since proposal was created")
        if not CheckpointManager(self.root).is_clean(allow_untracked=True):
            raise SelfImprovementError(
                "promotion requires no modified tracked files in the main worktree; "
                "commit or revert them, or let propose() quarantine them with on_dirty=quarantine"
            )
        checkpoint = CheckpointManager(self.root).create(health, allow_untracked=True)
        try:
            self._git(["merge", "--ff-only", commit])
        except Exception:
            self._git(["reset", "--hard", checkpoint.commit])
            raise
        self.request_reboot(proposal, commit, {**health, "commit": commit})
        return commit

    def discard(self, proposal: ImprovementProposal) -> None:
        self._git(["worktree", "remove", "--force", str(proposal.worktree)])

    @staticmethod
    def _fingerprint_of(record: Mapping[str, object]) -> str:
        value = record.get("change_fingerprint")
        return value if isinstance(value, str) else ""

    def _warned_record_for_fingerprint(self, proposals: Mapping[str, dict[str, object]], fingerprint: str) -> dict[str, object] | None:
        """Find the warned record for an exact fingerprint.

        ``blocked_by_protection`` is the retired status of the same mechanism;
        treating it as already-warned lets an upgrade finish a proposal that was
        held under the old owner-approval flow.
        """
        for record in proposals.values():
            if not isinstance(record, dict) or self._fingerprint_of(record) != fingerprint:
                continue
            if record.get("status") in {WARNED_PROTECTED, "blocked_by_protection"}:
                return record
        return None

    def _warned_result(self, fingerprint: str, offenders: Sequence[str]) -> dict[str, object]:
        paths = sorted(str(path) for path in offenders)
        return {
            "ok": False,
            "warned": True,
            "status": WARNED_PROTECTED,
            "change_fingerprint": fingerprint,
            "protected_paths": paths,
            "awaiting_resubmission": True,
            "error": "WARNING: this proposal touches gate-protected path(s): "
            f"{', '.join(paths)}. It was NOT applied.\n"
            "Stop and reconsider whether this edit is genuinely necessary. These paths -- the test suite, "
            "its configuration, and the self-improvement/planner/metrics modules -- decide whether your own "
            "changes are safe; editing them can weaken the very checks that verify you.\n"
            "If you are certain, resubmit the IDENTICAL change through propose_self_improvement (the same "
            f"files/changes/patch and the same hypothesis; change_fingerprint {fingerprint}). The harness will "
            "then run the full gate (ruff, pyright, pytest) from the proposal worktree. Any change to the diff "
            "produces a new fingerprint and a fresh warning.",
        }

    def _warn_protected_proposal(
        self,
        proposal: ImprovementProposal,
        fingerprint: str,
        offenders: Sequence[str],
        applied_paths: Sequence[str] | None,
        metadata: Mapping[str, object],
    ) -> dict[str, object]:
        """Warn once about a protected-path proposal instead of rejecting it.

        The worktree is kept so an identical resubmission can run the full gate
        without being re-applied; the warning is recorded as a durable,
        retention-protected event and the owner is reached through the existing
        alert/outbox channel.
        """
        record: dict[str, object] = {
            "proposal_id": proposal.proposal_id,
            "worktree": str(proposal.worktree),
            "base_commit": proposal.base_commit,
            "status": WARNED_PROTECTED,
            "change_fingerprint": fingerprint,
            "protected_paths": sorted(str(path) for path in offenders),
            "applied_paths": [str(path) for path in applied_paths] if applied_paths is not None else [],
            "warned_at": utc_now(),
            # No failure_class: a warning is not a verdict, and
            # check_registry_integrity would otherwise demand "rejected".
            "failure_class": "",
            **metadata,
        }
        proposals = self._read_proposals()
        proposals[proposal.proposal_id] = record
        self._write_proposals(proposals)
        payload: dict[str, object] = {
            "proposal_id": proposal.proposal_id,
            "paths": record["protected_paths"],
            "change_fingerprint": fingerprint,
            "reason": "proposal touches gate-protected path(s); warning issued, identical resubmission proceeds",
            "hypothesis": metadata.get("hypothesis", {}),
        }
        self._emit("gate_protected_warned", payload)
        # The durable event above keeps the whole hypothesis; the owner alert
        # must not. The hypothesis contract carries a mandatory
        # ``rollback_condition`` field, so sending it verbatim put the word
        # "rollback" in front of the owner on every protected-path warning and
        # made a warning look like an emergency rollback of the tree (measured:
        # 9 such alerts, 2745-4471 rendered characters each, up to 9
        # occurrences of "rollback"). The alert states the condition and what it
        # did -- and did not -- do; the contract stays in the event and in
        # state/self-improvement-proposals.json, both reachable on demand.
        self._alert_owner(
            "gate_protected_warned",
            {
                "proposal_id": proposal.proposal_id,
                "paths": record["protected_paths"],
                "change_fingerprint": fingerprint,
                "reason": payload["reason"],
                "effect": "no change applied to the main worktree; an identical resubmission runs the full gate",
            },
        )
        if self.runtime_log is not None:
            self.runtime_log.write("proposal_warned", {"proposal_id": proposal.proposal_id, "paths": record["protected_paths"], "change_fingerprint": fingerprint})
        return self._warned_result(fingerprint, offenders)

    def _resume_warned(self, record: Mapping[str, object], fingerprint: str, test_command: str | Sequence[str], metadata: Mapping[str, object]) -> dict[str, object]:
        """Run the full gate for a warned proposal on identical resubmission.

        The identical change is proposed again, the fingerprint matches the
        warned record, and the harness acknowledges the warning and proceeds
        through the normal verification stages (ruff, pyright, the suite). The
        stored worktree and applied paths are reused so the diff is not
        re-applied. A green gate validates the proposal; a failure is
        classified as usual.
        """
        proposal_id = str(record.get("proposal_id") or "")
        worktree = Path(str(record.get("worktree") or ""))
        base_commit = str(record.get("base_commit") or "")
        proposal = ImprovementProposal(proposal_id, worktree, base_commit)
        stored_paths = record.get("applied_paths")
        paths = [str(item) for item in stored_paths] if isinstance(stored_paths, list) else None
        raw_protected = record.get("protected_paths")
        protected = [str(item) for item in raw_protected] if isinstance(raw_protected, list) else []
        try:
            test_command, fallback = self._coerce_test_command(test_command)
            commit = self.validate_and_commit(proposal, test_command, applied_paths=paths, fingerprint=fingerprint)
            self._record_validated(proposal, commit, test_command, fallback=fallback)
            current = self._read_proposals().get(proposal_id, {})
            current.update(metadata)
            current["change_fingerprint"] = fingerprint
            current["failure_class"] = ""
            current["parent_proposal_id"] = metadata.get("parent_proposal_id", "")
            self._write_proposals(self._read_proposals() | {proposal_id: current})
            result: dict[str, object] = {
                "ok": True,
                "proposal_id": proposal_id,
                "worktree": str(worktree),
                "base_commit": base_commit,
                "commit": commit,
                "promotion_required": True,
                "warned_protected_paths": protected,
            }
            if fallback:
                result["test_command_fallback"] = True
            return result
        except Exception as exc:
            failure_class = classify_failure(exc)
            proposals = self._read_proposals()
            current = proposals.get(proposal_id)
            if isinstance(current, dict):
                current["status"] = "blocked_by_environment" if failure_class in BLOCKED_FAILURE_CLASSES else "rejected"
                current["failure_class"] = failure_class
                current["failure"] = str(exc)[:1000]
                reason_code = getattr(exc, "reason_code", "")
                if isinstance(reason_code, str) and reason_code:
                    current["reason_code"] = reason_code
                    current["match_count"] = getattr(exc, "match_count", None)
                self._write_proposals(proposals)
            if self.runtime_log is not None:
                self.runtime_log.write("proposal_failed", {"proposal_id": proposal_id, "failure_class": failure_class, "error": str(exc)[:1000]})
            self._cleanup_worktree(str(worktree))
            return {
                "ok": False,
                "status": "blocked_by_environment" if failure_class in BLOCKED_FAILURE_CLASSES else "rejected",
                "proposal_id": proposal_id,
                "failure_class": failure_class,
                "error": str(exc)[:1000],
            }

    def propose_files(
        self,
        files: Mapping[str, str],
        test_command: str | Sequence[str] = DEFAULT_TEST_COMMAND,
        *,
        changes: Sequence[Mapping[str, object]] = (),
        patch: str = "",
        metadata: Mapping[str, object] | None = None,
        on_dirty: str = "quarantine",
    ) -> dict[str, object]:
        """Run one bounded proposal from an agent.

        The returned result carries ``promotion_required`` when the change is
        validated but not yet promoted; ``SelfImprovementTool`` promotes it
        automatically when a health check is available. An identical repeat of a
        validated proposal resumes its promotion instead of being refused, so a
        change that passed the gate but failed the health gate is never trapped.
        """
        self.last_quarantined = []
        metadata = dict(metadata or {})
        fingerprint = hashlib.sha256(json.dumps({"files": dict(files), "changes": list(changes), "patch": patch, "hypothesis": metadata.get("hypothesis", {})}, sort_keys=True, default=str).encode()).hexdigest()
        previous = self._read_proposals()
        for previous_id, record in previous.items():
            if record.get("change_fingerprint") != fingerprint:
                continue
            # An environment failure is not a verdict about the patch: the same
            # change must be retryable once the environment is repaired, so it
            # does not burn the fingerprint. A change that already passed the
            # gate but whose promotion failed (typically the health gate) must
            # also be finishable: an identical repeat resumes promotion with the
            # stored commit instead of being refused, so the duplicate guard
            # cannot trap verified work. Only real rejections and changes that
            # already promoted are permanently refused; an environment-blocked
            # change is refused once its bounded attempts are spent.
            status = record.get("status")
            stored_attempts = record.get("environment_attempts")
            attempts = stored_attempts if isinstance(stored_attempts, int) else 1
            exhausted = status == "blocked_by_environment" and attempts >= MAX_ENVIRONMENT_ATTEMPTS
            if status == "validated":
                return {
                    "ok": True,
                    "proposal_id": previous_id,
                    "worktree": str(record.get("worktree", "")),
                    "base_commit": str(record.get("base_commit", "")),
                    "commit": str(record.get("commit", "")),
                    "promotion_required": True,
                    "resumed_validation": True,
                }
            if status in {"rejected", "awaiting_reboot", "accepted"} or exhausted:
                previous_class = str(record.get("failure_class") or "unknown")
                raise SelfImprovementError(
                    f"identical proposal was already attempted as {previous_id} "
                    f"(failure_class={previous_class}); change the hypothesis or patch",
                    previous_proposal={"proposal_id": previous_id, "failure_class": previous_class},
                )
        # A warned protected-path proposal is resumed here, on the next
        # identical proposal: the fingerprint is the durable key, so only the
        # exact change that was warned has its warning acknowledged.
        warned = self._warned_record_for_fingerprint(previous, fingerprint)
        if warned is not None:
            return self._resume_warned(warned, fingerprint, test_command, metadata)
        proposal = self.propose(on_dirty=on_dirty)
        quarantined = list(self.last_quarantined)
        applied_paths: list[str] | None = None
        try:
            test_command, fallback = self._coerce_test_command(test_command)
            if fallback and self.runtime_log is not None:
                self.runtime_log.write("test_command_fallback", {"proposal_id": proposal.proposal_id, "default_command": list(DEFAULT_TEST_COMMAND)})
            if patch:
                applied_paths = list(self.apply_patch(proposal, patch))
                applied: dict[str, object] | None = {"files": applied_paths, "change_digest": change_content_digest(proposal.worktree, applied_paths)}
            elif changes:
                applied = self.apply_changes(proposal, changes)
                files_value = applied.get("files")
                applied_paths = [str(item) for item in files_value] if isinstance(files_value, list) else []
            else:
                applied = None
                applied_paths = []
            # `files` is a separate channel and must apply on EVERY path, after
            # the edits so a patch is still applied to a pristine tree. The
            # branches above used to be `if patch / elif changes / else` with
            # `apply_files` only in the `else`, so a proposal carrying BOTH new
            # files and anchor changes applied only the changes. Measured on
            # generation 186: proposal 073938d987a64ff0ba9b7a7da8a460db
            # submitted `files={"tests/test_executed_evidence_invariant.py":
            # ...}` with nine anchor changes; the tool result named only
            # skynet/react.py, the registry recorded files_changed: 1, and the
            # promoted commit 9088322 contains no test file -- the gate passed
            # while half the proposal was silently discarded, and because
            # `applied_paths` omitted the file the gate-protected check never
            # saw it either (arXiv:1907.04055v1: most failures are not timely
            # detected and silently propagate across components). The union
            # below is what the ignored-path and gate-protected checks read.
            if files:
                applied_paths = list(dict.fromkeys([*applied_paths, *self.apply_files(proposal, files)]))
            commit = self.validate_and_commit(proposal, test_command, applied_paths=applied_paths, fingerprint=fingerprint)
            self._record_validated(proposal, commit, test_command, fallback=fallback)
            record = self._read_proposals()[proposal.proposal_id]
            record.update(metadata)
            record["change_fingerprint"] = fingerprint
            record["failure_class"] = ""
            record["parent_proposal_id"] = metadata.get("parent_proposal_id", "")
            self._write_proposals(self._read_proposals() | {proposal.proposal_id: record})
            result: dict[str, object] = {"ok": True, "proposal_id": proposal.proposal_id, "worktree": str(proposal.worktree), "base_commit": proposal.base_commit, "commit": commit, "promotion_required": True}
            if (changes or patch) and applied is not None:
                result["applied"] = applied
            if fallback:
                result["test_command_fallback"] = True
            if quarantined:
                result["quarantined_files"] = quarantined
            return result
        except ProtectedPathWarning as warning:
            return self._warn_protected_proposal(proposal, fingerprint, warning.offenders, applied_paths, metadata)
        except Exception as exc:
            failure_class = classify_failure(exc)
            proposals = self._read_proposals()
            record: dict[str, object] = {
                "proposal_id": proposal.proposal_id,
                "base_commit": proposal.base_commit,
                "status": "blocked_by_environment" if failure_class in BLOCKED_FAILURE_CLASSES else "rejected",
                "failure_class": failure_class,
                "failure": str(exc)[:1000],
                "change_fingerprint": fingerprint,
                **metadata,
            }
            if failure_class in BLOCKED_FAILURE_CLASSES:
                prior_attempts = [
                    item.get("environment_attempts")
                    for item in proposals.values()
                    if item.get("change_fingerprint") == fingerprint
                ]
                record["environment_attempts"] = max((value for value in prior_attempts if isinstance(value, int)), default=0) + 1
            reason_code = getattr(exc, "reason_code", "")
            if isinstance(reason_code, str) and reason_code:
                record["reason_code"] = reason_code
                record["match_count"] = getattr(exc, "match_count", None)
            proposals[proposal.proposal_id] = record
            self._write_proposals(proposals)
            if self.runtime_log is not None:
                self.runtime_log.write("proposal_failed", {"proposal_id": proposal.proposal_id, "failure_class": failure_class, "error": str(exc)[:1000]})
            self.discard(proposal)
            raise


class SelfImprovementTool:
    name = "propose_self_improvement"
    capability_kind = "write"

    def __init__(self, manager: SelfImprovementManager, health_check: Callable[[], dict[str, object]] | None = None, test_command: Sequence[str] | None = None) -> None:
        self.manager = manager
        self.health_check = health_check
        self.test_command = tuple(test_command or DEFAULT_TEST_COMMAND)
        self._failure_cache: dict[str, dict[str, str]] = {}

    def reset_run_scope(self) -> None:
        """Forget repeat-guard state at the start of every bounded run."""
        self._failure_cache.clear()

    def _failure_result(self, error: str, failure_class: str, **extra: object) -> dict[str, object]:
        """Attach a stable key and, on a repeat, the previous failure's reason."""
        failure_key = hashlib.sha256(f"{failure_class}\x00{error}".encode()).hexdigest()
        result: dict[str, object] = {"ok": False, "error": error, "failure_class": failure_class, "failure_key": failure_key, **extra}
        previous = self._failure_cache.get(failure_key)
        if previous is not None:
            result["previous_failure"] = dict(previous)
        self._failure_cache[failure_key] = {"failure_class": failure_class, "error": error}
        if len(self._failure_cache) > 64:
            self._failure_cache.pop(next(iter(self._failure_cache)))
        return result

    @property
    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Create, test, and automatically promote a bounded self-improvement proposal. The harness applies a passing change, records rollback state, and restarts the service; do not use this without a concrete evidence-backed hypothesis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "files": {"type": "object", "description": "Only for genuinely new files; do not replace existing files with complete contents.", "additionalProperties": {"type": "string"}},
                        "changes": {"type": "array", "description": "Preferred format for existing files. Each exact-anchor edit must match exactly once.", "items": {"type": "object", "properties": {"path": {"type": "string"}, "operation": {"type": "string", "enum": ["create", "replace", "insert_before", "insert_after", "delete"]}, "old": {"type": "string", "minLength": 1}, "anchor": {"type": "string", "minLength": 1}, "new": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "operation"], "additionalProperties": False, "allOf": [{"if": {"properties": {"operation": {"const": "create"}}}, "then": {"required": ["content"]}}, {"if": {"properties": {"operation": {"enum": ["replace", "insert_before", "insert_after", "delete"]}}}, "then": {"anyOf": [{"required": ["old"]}, {"required": ["anchor"]}]}}]}},
                        "patch": {"type": "string", "description": "Optional unified diff applied with `git apply` inside the worktree. Use instead of exact anchors when the change is easier to state as a diff; it must apply cleanly."},
                        "hypothesis": {"type": "object", "description": "Evidence-backed proposal contract.", "properties": {"problem": {"type": "string", "minLength": 1}, "expected_behavior": {"type": "string", "minLength": 1}, "evidence": {"type": "string", "minLength": 1}, "validation": {"type": "string", "minLength": 1}, "rollback_condition": {"type": "string", "minLength": 1}}, "required": ["problem", "expected_behavior", "evidence", "validation", "rollback_condition"], "additionalProperties": False},
                        "test_command": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                        "on_dirty": {"type": "string", "enum": ["quarantine", "refuse"], "description": "How to handle modified tracked files in the main worktree: quarantine preserves them under state/quarantine and continues; refuse aborts."},
                    },
                    "required": ["hypothesis"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        files = arguments.get("files", {})
        changes = arguments.get("changes", [])
        patch = arguments.get("patch", "")
        if not isinstance(files, dict) or not all(isinstance(path, str) and isinstance(content, str) for path, content in files.items()):
            return self._failure_result("files must be an object mapping relative paths to text", "invalid_payload")
        if not isinstance(changes, list) or not changes or not all(isinstance(change, dict) for change in changes):
            changes = []
        if not isinstance(patch, str):
            patch = ""
        if not files and not changes and not patch.strip():
            error_msg = "proposal requires files, changes, or patch"
            if self.manager.runtime_log is not None:
                self.manager.runtime_log.write("proposal_rejected", {"error": error_msg, "failure_class": "invalid_payload"})
            return self._failure_result(error_msg, "invalid_payload")
        hypothesis = arguments.get("hypothesis")
        required_hypothesis = ("problem", "expected_behavior", "evidence", "validation", "rollback_condition")
        if not isinstance(hypothesis, dict) or any(not isinstance(hypothesis.get(key), str) or not hypothesis[key].strip() for key in required_hypothesis):
            error_msg = "proposal requires a complete evidence-backed hypothesis"
            if self.manager.runtime_log is not None:
                self.manager.runtime_log.write("proposal_rejected", {"error": error_msg, "failure_class": "invalid_payload", "hypothesis": hypothesis})
            return self._failure_result(error_msg, "invalid_payload", do_not_retry_unchanged=True)
        # The `files` channel is new-content-only and that stays true on every
        # path: with `changes` or `patch` present this guard used to be skipped,
        # so a `files` entry naming an existing path would have overwritten it
        # without an anchor instead of being refused. Measured in the regression
        # test, the combined form silently clobbered the file.
        existing = [path for path in files if (self.manager.root / path).exists()]
        if existing:
            error_msg = f"use exact-anchor changes for existing files: {', '.join(existing)}"
            if self.manager.runtime_log is not None:
                self.manager.runtime_log.write("proposal_rejected", {"error": error_msg, "failure_class": "invalid_payload"})
            return self._failure_result(error_msg, "invalid_payload", do_not_retry_unchanged=True)
        try:
            requested_test_command = arguments.get("test_command")
            if isinstance(requested_test_command, str):
                try:
                    requested_test_command = shlex.split(requested_test_command)
                except ValueError:
                    requested_test_command = None
            # A nested stage list (the argv form the runbook recommends) used
            # to fail this string-only test and be replaced by the default
            # suite, so the stages the proposal declared were silently dropped.
            test_command = (
                tuple(requested_test_command)
                if isinstance(requested_test_command, list)
                and all(isinstance(item, (str, list, tuple)) for item in requested_test_command)
                else self.test_command
            )
            on_dirty_arg = arguments.get("on_dirty", "quarantine")
            on_dirty = on_dirty_arg if on_dirty_arg in ("quarantine", "refuse") else "quarantine"
            result = self.manager.propose_files(files, test_command, changes=changes, patch=patch, metadata={"hypothesis": arguments.get("hypothesis", {})}, on_dirty=on_dirty)
            if not result.get("ok") or self.health_check is None:
                return result
            health = self.health_check()
            if not health.get("ok"):
                return {**result, "promotion": {"ok": False, "error": "health gate failed", "health": health}}
            promoted = self.manager.promote_pending(str(result["proposal_id"]), health)
            return {
                **result,
                "promotion_required": False,
                "promotion": {"ok": True, "commit": promoted, "reboot_requested": True},
                "control_action": {
                    "type": "restart_after_checkpoint",
                    "proposal_id": str(result["proposal_id"]),
                    "commit": promoted,
                    "reason": "self-improvement promoted; persist memory and checkpoint before restart",
                },
            }
        except (OSError, SelfImprovementError, CheckpointError) as exc:
            failure_class = classify_failure(exc)
            failure = self._failure_result(str(exc)[:1000], failure_class, do_not_retry_unchanged=True)
            reason_code = getattr(exc, "reason_code", "")
            if isinstance(reason_code, str) and reason_code:
                failure["reason_code"] = reason_code
                failure["match_count"] = getattr(exc, "match_count", None)
            previous_proposal = getattr(exc, "previous_proposal", None)
            if isinstance(previous_proposal, Mapping):
                failure["previous_proposal"] = dict(previous_proposal)
            if self.manager.last_quarantined:
                failure["quarantined_files"] = list(self.manager.last_quarantined)
            return failure
