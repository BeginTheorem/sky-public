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

_CD_PATTERN = re.compile(
    r"(?:^|[;&|()\s])(?:cd|pushd)\s+(?P<target>\"[^\"]*\"|'[^']*'|[^\s;&|()]+)"
)


_QUOTED_PATTERN = re.compile(r"'[^']*'|\"[^\"]*\"")
_QUOTE_CHARS = re.compile(r"['\"]")
# `>&` is a redirect too: in bash `>&file` sends stdout+stderr to file, while
# `2>&1` only duplicates a descriptor and writes nothing. Redirects are matched
# on the de-quoted text, so a quoted destination cannot hide behind its quotes.
_REDIRECT_PATTERN = re.compile(r"(?P<op>>>?|>&)\s*(?P<target>[^\s;&|()>]+)")
_SAFE_REDIRECT_TARGETS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})
_REDIRECT_TOKENS = (">", ">>", ">&")
_DESTRUCTIVE_TOKENS = tuple(token for token in MUTATING_TOKENS if token not in _REDIRECT_TOKENS)


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
            if path.casefold() in unquoted
            or any(path.casefold() in target for target in targets)
            or (destructive and path.casefold() in dequoted)
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


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def workspace_escape(command: str, cwd: Path, workspace: Path | None, extra_roots: Sequence[Path] = ()) -> str | None:
    """Return a ``cd``/``pushd`` target that resolves outside the allowed roots.

    Deliberately a cheap regex over plain ``cd <path>`` tokens, not a shell
    parser: quoted, variable and flag forms are best-effort. ``extra_roots`` is
    how the scratch directory stays consistent with read/grep instead of being
    warned about in bash and refused elsewhere.
    """
    if workspace is None:
        return None
    allowed_roots = [workspace.resolve(), *(root.resolve() for root in extra_roots)]
    for match in _CD_PATTERN.finditer(command):
        target = match.group("target").strip("\"'")
        if not target or target in {"-", "~"} or target.startswith("$"):
            continue
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not any(_is_within(resolved, root) for root in allowed_roots):
            return target
    return None


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    workspace: Path | None = None
    allow_private_network: bool = False
    allowed_hosts: frozenset[str] = frozenset()
    scratch_root: Path | None = None

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
        return cls(workspace, private, hosts, scratch_root)

    def allowed_roots(self) -> tuple[Path, ...]:
        """Workspace plus the scratch root: the one boundary all tools share."""
        roots: list[Path] = []
        if self.workspace is not None:
            roots.append(self.workspace.resolve())
        if self.scratch_root is not None:
            roots.append(self.scratch_root.resolve())
        return tuple(roots)

    def check_cwd(self, cwd: str | Path) -> Path:
        resolved = Path(cwd).resolve()
        if self.workspace is None:
            return resolved
        roots = self.allowed_roots()
        if any(_is_within(resolved, root) for root in roots):
            return resolved
        raise PolicyError(f"cwd outside sandbox workspace: {resolved} is not under {self.workspace.resolve()}")

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
