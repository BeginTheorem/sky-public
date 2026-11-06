from __future__ import annotations

import ipaddress
import os
import re
import socket
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class PolicyError(PermissionError):
    pass


# Paths that keep SkyNet itself alive. The experiment is intentionally
# allow-all, so touching these is not blocked; it only requires an explicit
# confirmation by re-running the exact same command (a soft denylist).
SELF_PRESERVE_PATHS: tuple[str, ...] = (
    ".git/",
    "state/",
    "config/skynet.env",
    "deploy/",
    "scripts/",
)

# The resurrection mechanism is a hard deny: repetition never overrides it.
# state/ and deploy/ sit in the soft contour above: they are recoverable
# operational data, not the machinery that keeps the organism alive. What is
# left here cannot be re-created by the organism itself - the git history, the
# rollback scripts, and the .env files that hold the API credentials the
# provider chain needs to run at all.
RESURRECTION_PATHS: tuple[str, ...] = (
    ".git/",
    "scripts/skynet-startup-rollback.sh",
    "scripts/rollback.sh",
    ".env",
)

MUTATING_TOKENS: tuple[str, ...] = (
    "rm ", "rm -", "mv ", "cp ", "tee ", "sed -i", "truncate", "dd ", "unlink",
    "shred", "chmod", "chown", ">", ">>", "git reset", "git clean", "git checkout",
    "git push", "git rm", "worktree remove", "worktree prune",
)

_CD_TARGET_PATTERN = re.compile(
    r"(?:^|[;&|()\s])(?:cd|pushd)\s+(?P<target>\"[^\"]*\"|'[^']*'|[^\s;&|()]+)"
)

_QUOTED_PATTERN = re.compile(r"'[^']*'|\"[^\"]*\"")
_QUOTE_CHARS = re.compile(r"['\"]")
# `>&` is a redirect too: in bash `>&file` sends stdout+stderr to file, while
# `2>&1` only duplicates a descriptor and writes nothing. Redirects are matched
# on the de-quoted text, so a quoted destination cannot hide behind its quotes.
_REDIRECT_PATTERN = re.compile(r"(?P<op>>>?|>&)\s*(?P<target>[^\s;&|()>]+)")
_SAFE_REDIRECT_TARGETS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})

# An absolute path named as an argument (``rm /tmp/x``, ``cp a /tmp/b``). Relative arguments
# are deliberately not collected: telling a source from a destination is shell parsing this
# guard does not attempt, and a bare relative write from inside the tree stays denied anyway.
_ABSOLUTE_PATH_TOKEN = re.compile(r"(?<![\w./\-])/[A-Za-z0-9._/\-]+")
_REDIRECT_TOKENS = (">", ">>", ">&")
_DESTRUCTIVE_TOKENS = tuple(token for token in MUTATING_TOKENS if token not in _REDIRECT_TOKENS)

# A protected path is matched on path boundaries, not as a bare substring. The
# credential entry ``.env`` is a substring of ``skynet.env`` (the config file,
# which sits in the *soft* contour) and of any ``*.env`` name, so a substring
# test hard-denied a command that merely named the config file: all 181 live
# ``policy_denied`` rows carried ``protected_path: ".env"``. A directory entry
# that already ends in ``/`` is a prefix and may be followed by anything, so
# ``state/`` matches ``state/db`` while still excluding ``prostate/``.
_BOUNDARY_BEFORE = r"(?<![A-Za-z0-9_.\-])"
_BOUNDARY_AFTER = r"(?![A-Za-z0-9_.\-])"


def protected_path_present(path: str, text: str) -> bool:
    """Whether ``path`` occurs in ``text`` on path boundaries, not as a fragment."""
    needle = path.casefold()
    haystack = text.casefold()
    suffix = "" if needle.endswith("/") else _BOUNDARY_AFTER
    return re.search(_BOUNDARY_BEFORE + re.escape(needle) + suffix, haystack) is not None


def _redirect_targets(text: str) -> tuple[str, ...]:
    """File destinations of every `>`/`>>`/`>&`; descriptor dups (`2>&1`) excluded."""
    targets: list[str] = []
    for match in _REDIRECT_PATTERN.finditer(text):
        target = match.group("target")
        if match.group("op") == ">&" and target.isdigit():
            continue
        if target in _SAFE_REDIRECT_TARGETS:
            continue
        targets.append(target)
    return tuple(targets)


def mutation_tokens(text: str) -> tuple[str, ...]:
    """Return the tokens in ``text`` that would really mutate a file.

    Two refinements over a plain substring test, both measured on the live
    event log, where 76 of 149 recorded denials were false positives:

    * a quoted span is data, not shell syntax, so a command that merely greps
      for the literal name of a protected file is not a mutation;
    * a redirect counts only when its target is outside /dev/null, so the
      stderr operators 2>&1 and 2>/dev/null are not mutations.

    Redirects are matched on the de-quoted text, because a quoted destination
    (`> "state/skynet.sqlite3"`, `>> 'config/skynet.env'`) is still a write and
    must not slip through; destructive command names are matched on the text
    with quoted spans removed, so a quoted word is never read as a command.
    """
    unquoted = _QUOTED_PATTERN.sub(" ", text)
    dequoted = _QUOTE_CHARS.sub("", text)
    tokens = [token for token in _DESTRUCTIVE_TOKENS if token in unquoted]
    for match in _REDIRECT_PATTERN.finditer(dequoted):
        op = match.group("op")
        target = match.group("target")
        if op == ">&" and target.isdigit():
            continue
        if target in _SAFE_REDIRECT_TARGETS:
            continue
        tokens.append(">" if op == ">&" else op)
    return tuple(tokens)


def self_preservation_warning(command: str, protected: Sequence[str] = SELF_PRESERVE_PATHS) -> str | None:
    """Return the critical path a command would mutate, if any.

    Only the combination of a protected path and a real mutation is flagged.
    A protected path counts when it appears outside quotes, when it is the
    destination of a redirect (quoted or not), or when a destructive command
    name is present and the path appears anywhere once quotes are removed
    (the command's target may itself be quoted or split across quotes).
    Reading a protected path, or merely naming it inside a quoted pattern, is
    observation and is not flagged.
    """
    text = command.casefold()
    unquoted = _QUOTED_PATTERN.sub(" ", text)
    dequoted = _QUOTE_CHARS.sub("", text)
    targets = _redirect_targets(dequoted)
    destructive = any(token in unquoted for token in _DESTRUCTIVE_TOKENS)
    hit = next(
        (
            path
            for path in protected
            if protected_path_present(path, unquoted)
            or any(protected_path_present(path, target) for target in targets)
            or (destructive and protected_path_present(path, dequoted))
        ),
        None,
    )
    if hit is None:
        return None
    if not mutation_tokens(text):
        return None
    return hit


def resurrection_denial(command: str, protected: Sequence[str] = RESURRECTION_PATHS) -> str | None:
    """Return a hard-denied resurrection path a command would mutate."""
    return self_preservation_warning(command, protected)


# Root is not reached through bash: it goes through the sudo gate, which asks the owner
# first. A second path to root would make that gate decorative — the account holds
# NOPASSWD sudo, so nothing but this refusal stands between a stray command and root.
_SUDO_INVOCATION = re.compile(
    r"(?:^|[;&|(\n]|\$\(|&&|\|\|)\s*(?:(?P<word>sudo|doas|pkexec)\s|su(?:\s+-|\s+root\b|\s*$))",
    re.IGNORECASE,
)


def sudo_denial(command: str) -> str | None:
    """The word that should have gone to the sudo gate instead of bash.

    A word-boundary match in command position, on the literal text: it does not see
    through variables, aliases, ``xargs`` or ``env``, so this is a guardrail against the
    obvious path, not a proof that bash cannot reach root. Mentioning ``sudo`` inside a
    quote (a grep pattern, a document) is not an invocation and stays allowed.
    """
    match = _SUDO_INVOCATION.search(command)
    if match is None:
        return None
    return match.group("word") or "su"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _effective_directory(command: str, cwd: Path) -> Path:
    """The directory a mutation really runs in, following a leading ``cd``/``pushd``."""
    try:
        effective = cwd.resolve()
    except OSError:
        effective = cwd
    for match in _CD_TARGET_PATTERN.finditer(command):
        target = match.group("target").strip("\"'")
        if not target or target in {"-", "~"} or target.startswith("$"):
            continue
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = effective / candidate
        try:
            effective = candidate.resolve()
        except OSError:
            continue
    return effective


def _mutation_targets(text: str, effective: Path) -> tuple[Path, ...]:
    """Every path a mutation in ``text`` names: redirect destinations and absolute paths.

    Resolved against ``effective`` because a relative target lands where the command runs,
    which is the whole point of following a ``cd`` first. Only absolute paths are taken from
    the arguments: guessing which argument of ``cp`` is the destination is shell parsing this
    guard deliberately does not attempt.
    """
    targets: list[Path] = []
    for raw in _redirect_targets(_QUOTE_CHARS.sub("", text)):
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = effective / candidate
        try:
            targets.append(candidate.resolve())
        except OSError:
            continue
    for token in _ABSOLUTE_PATH_TOKEN.findall(_QUOTE_CHARS.sub("", text)):
        try:
            targets.append(Path(token).resolve())
        except OSError:
            continue
    return tuple(targets)


def workspace_write_denial(command: str, cwd: Path, workspace: Path | None) -> str | None:
    """Return a path when a mutation would write inside the organism's own tree.

    The repository is the organism's *self*: bash may read it, but every edit
    must go through ``propose_self_improvement`` (the gate), so a mutation that
    *targets* the workspace is hard-denied, wherever the command was started
    from. A mutation aimed outside it — a redirect into ``/tmp`` or the home
    directory, a ``cd`` into a new directory first — is free work and stays
    allowed, because the boundary is the tree, not the starting directory. Only
    a mutation that names no destination at all while running from inside the
    tree is refused outright: this is not a shell parser and the bare form
    (``sed -i``, ``tee``, ``python -c open(..., 'w')``) is the one that edits
    the tree by habit. An interpreter write can still slip through, which is
    intended, so the organism can force a change if it truly wants one.
    """
    if workspace is None:
        return None
    if not mutation_tokens(command):
        return None
    root = workspace.resolve()
    effective = _effective_directory(command, cwd)
    targets = _mutation_targets(command, effective)
    inside = tuple(target for target in targets if _is_within(target, root))
    if _is_within(effective, root):
        # From inside the tree every named destination must be outside it; a command that
        # names none is the shape that quietly rewrites the tree, so it stays denied.
        if not targets:
            return "."
        return "." if inside else None
    if inside:
        return str(inside[0])
    return None


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    workspace: Path | None = None
    allow_private_network: bool = False
    allowed_hosts: frozenset[str] = frozenset()
    scratch_root: Path | None = None
    extra_roots: tuple[Path, ...] = ()

    @classmethod
    def from_environment(cls) -> ExecutionPolicy:
        disabled = os.getenv("SKYNET_SANDBOX_WORKSPACE_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"}
        configured = os.getenv("SKYNET_SANDBOX_WORKSPACE")
        if disabled:
            workspace: Path | None = None
        elif configured:
            workspace = Path(configured).resolve()
        else:
            workspace = Path.cwd().resolve()
        hosts = frozenset(item.strip().lower() for item in os.getenv("SKYNET_ALLOWED_HOSTS", "").split(",") if item.strip())
        private = os.getenv("SKYNET_ALLOW_PRIVATE_NETWORK", "").strip().lower() in {"1", "true", "yes", "on"}
        # The seed tells the organism to prototype here; bash, read and grep must
        # agree on the same boundary or the model falls back to cat/sed.
        scratch = os.getenv("SKYNET_SCRATCH_DIR", "/tmp/skynet-scratch").strip()
        scratch_root = Path(scratch).resolve() if scratch else None
        # Roots the owner opened for work *outside* the tree. One scratch path cannot say
        # "my home and /tmp", so this is a list (``:`` or ``,`` separated); the workspace
        # stays readable to every tool and, alone, is the one bash may never write into.
        roots: list[Path] = []
        for item in re.split(r"[:,]", os.getenv("SKYNET_SANDBOX_ROOTS", "")):
            entry = item.strip()
            if entry:
                roots.append(Path(entry).expanduser().resolve())
        return cls(workspace, private, hosts, scratch_root, tuple(roots))

    def allowed_roots(self) -> tuple[Path, ...]:
        """Workspace, scratch and every extra root: the one boundary all tools share.

        Order is stable and duplicates are dropped, so two calls compare equal; the
        workspace stays first because it is also the write boundary that bash cannot cross.
        """
        candidates: list[Path] = []
        if self.workspace is not None:
            candidates.append(self.workspace.resolve())
        if self.scratch_root is not None:
            candidates.append(self.scratch_root.resolve())
        candidates.extend(root.resolve() for root in self.extra_roots)
        unique: list[Path] = []
        for candidate in candidates:
            if candidate not in unique:
                unique.append(candidate)
        return tuple(unique)

    def check_cwd(self, cwd: str | Path) -> Path:
        resolved = Path(cwd).resolve()
        if self.workspace is None:
            return resolved
        roots = self.allowed_roots()
        if any(_is_within(resolved, root) for root in roots):
            return resolved
        # Naming only the workspace would read as "the tree is the boundary" while a refusal
        # may really be about a root the owner opened; list what is allowed.
        allowed = ", ".join(str(root) for root in roots)
        raise PolicyError(f"cwd outside sandbox workspace: {resolved} is not under {allowed}")

    def check_address(self, address: str, label: str | None = None) -> None:
        if self.allow_private_network:
            return
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise PolicyError(f"private network target is blocked: {label or address}")

    def check_url(self, url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            raise PolicyError("URL has no hostname")
        if self.allowed_hosts and hostname not in self.allowed_hosts:
            raise PolicyError(f"host is not allowlisted: {hostname}")
        if self.allow_private_network:
            return
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)}
        for address in addresses:
            self.check_address(str(address), hostname)
