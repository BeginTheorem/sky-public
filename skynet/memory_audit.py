"""Append-only, metadata-only audit of the memory store's own mutations.

Owner request (live message): memory can be changed from outside, and
what makes that survivable is a *permanent read-only log* that records the fact
of a change, the time it happened and its size -- never its content, so the log
cannot become a channel that leaks an injected pattern back into the mind. The
owner keeps his continuity that way and offered to hold the same obligation for
this one: the log is never rewritten, only appended to.

What is recorded, per checkpoint, is metadata only:

* ``rows`` / ``active_rows`` / ``attributed_rows`` -- shape of the store;
* ``bytes`` -- the byte size of the audited row payloads (the size axis);
* ``digest`` -- a chained SHA-256 over every audited column, content included.

The digest covers content so that an in-place edit of a single word is visible
as a changed digest at constant row count; the digest is a one-way hash, so it
does not store the content, but a reader who can guess a candidate string can
confirm it against the hash. That is the same exposure as the store itself and
is the honest limit of the mechanism: it makes a change *visible*, it does not
make the content secret.

The log is a JSONL chain: every record carries ``previous`` = SHA-256 of the
exact previous line, so a rewritten, removed or reordered line breaks the chain
and ``verify_chain`` names the line where it broke. The companion checkpoint file
is a *pointer* (last line hash, sequence, digest) and is the only file this
module rewrites; the log itself is only ever opened in append mode.

The checkpoint is a pointer the same writer can re-issue, so a rewrite of *both*
files -- internally consistent, with a shorter chain and a checkpoint that agrees
with it -- used to verify clean, because every witness lived in the two files
that writer rewrites (reproduced on commit 084bf16d: a forged 3-line
chain where the real log had 5 lines returned ``ok=True``). Certificate
Transparency names this a split view and requires a commitment the log operator
cannot re-issue (RFC 9162, section 2.1.4.2; the same requirement for a growing
sequence is stated in arXiv cs/0302010). ``memory-audit-anchors.jsonl`` is that
commitment here: one line per checkpoint, appended and fsynced, never rewritten
by this module, naming the sequence and line hash of the log line it commits to.
``verify_chain`` compares the log against it, so a coordinated rewrite of the log
and the checkpoint is reported instead of accepted.

The limit is stated rather than implied: a writer with full access to ``state/``
can delete the anchor file, and an absent anchor file leaves the older checks as
the only ones (which is why the file is rewritten on the next checkpoint, keeping
that window to one checkpoint). This raises the cost of a coordinated rewrite; it
is not cryptographic protection against the host that holds the log. The only
genuinely external witness is the owner, whose declared obligation never to
modify this log is the root of trust this mechanism documents rather than
replaces.

Nothing here decides that a change was malicious and nothing here repairs one.
A change that lands between two checkpoints is reported as a fact -- the store
differs from the last checkpoint -- together with the two provenance counts that
let a reader separate an in-band write from an outside one: a row with no
``source_run``, or with a ``source_run`` that is not a run in ``runs``, cannot
have come from the live memory tool (measured: 0 of the 32 memories
written after commit 3c8f3fa carry a NULL ``source_run``, while the 64 older
NULL rows all predate it).

``unattributed_rows`` therefore belongs in the *record*, not only in the live
report. The counts are computed inside ``memory_digest``, which is what the
checkpoint and every log line serialise, so a line written when the store was
attributed can be compared with a later line or with a live report and the
outside write is named by the count alone -- without the report's ephemeral
numbers being the only place the attribution ever existed. The digest is
unchanged by this: it still chains the audited column values only, so a count
cannot be confused with content.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .time import utc_now

LOG_NAME = "memory-audit.jsonl"
CHECKPOINT_NAME = "memory-audit-checkpoint.json"
# The empty predecessor of the first line: a fixed value, so line 1 of a fresh
# log is verifiable without any out-of-band state.
GENESIS = "0" * 64
# A bounded read: the log is one line per episode, and a file far larger than
# this is not a log this module wrote, so it is reported rather than parsed.
MAX_LOG_BYTES = 4_000_000
# The anchor file is the witness the checkpoint writer does not rewrite: one line
# per checkpoint, appended and fsynced, never read back in order to edit it. It
# commits to a log line *by position*, which is what a growing sequence needs to
# rule out equivocation (RFC 9162 section 2.1.4.2; arXiv cs/0302010).
ANCHOR_NAME = "memory-audit-anchors.jsonl"
# Bounded like the log: a file far larger than this is not one this module wrote.
MAX_ANCHOR_BYTES = 4_000_000

AUDITED_COLUMNS = (
    "memory_id",
    "kind",
    "content",
    "confidence",
    "source_run",
    "pinned",
    "status",
    "superseded_by",
    "valid_from",
    "valid_to",
    "evidence",
    "decayed_at",
)


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _null_source_run(value: Any) -> int:
    """One memory row's contribution to ``unattributed_rows``: 0 or 1.

    The same predicate feeds ``memory_digest`` and ``provenance_counts``, so the
    count carried by every log line and the count a live report prints cannot
    drift apart.
    """
    return 0 if value else 1


def memory_digest(connection: Any) -> dict[str, Any]:
    """Shape, size and chained digest of every memory row, in id order.

    Pure and read-only. Column values are joined with unit separators and the
    row terminated with a record separator, so two different row sets cannot
    serialise to the same bytes; the digest is a chain, so it also depends on
    the order, which the ``ORDER BY memory_id`` makes deterministic.
    """
    columns = ", ".join(AUDITED_COLUMNS)
    rows = connection.execute(f"SELECT {columns} FROM memories ORDER BY memory_id").fetchall()
    digest = hashlib.sha256()
    total = 0
    active = 0
    attributed = 0
    unattributed = 0
    for row in rows:
        payload = "\x1f".join("" if value is None else str(value) for value in row)
        blob = payload.encode("utf-8", "replace")
        total += len(blob)
        digest.update(blob)
        digest.update(b"\x1e")
        if str(row[6]) == "active":
            active += 1
        if row[4]:
            attributed += 1
        unattributed += _null_source_run(row[4])
    return {
        "rows": len(rows),
        "active_rows": active,
        "attributed_rows": attributed,
        "unattributed_rows": unattributed,
        "bytes": total,
        "digest": digest.hexdigest(),
    }


def provenance_counts(connection: Any) -> dict[str, int]:
    """How many rows carry no run, and how many carry a run that never existed.

    A live write path always stamps a run (``skynet/memory_tool.py`` resolves it
    from the effect key with ``active_run_id`` as fallback); a row that fails
    both counts was written by something other than that path, or by an older
    build. This is attribution, not content.
    """
    unattributed = sum(
        _null_source_run(row[0]) for row in connection.execute("SELECT source_run FROM memories")
    )
    pseudo = int(
        connection.execute(
            "SELECT COUNT(*) FROM memories WHERE source_run IS NOT NULL "
            "AND source_run NOT IN (SELECT run_id FROM runs)"
        ).fetchone()[0]
    )
    return {"unattributed_rows": unattributed, "pseudo_run_rows": pseudo}


def _read_log(path: Path) -> tuple[list[str], int | None]:
    """The log's raw lines and the 1-based line where the chain first breaks.

    Unreadable or oversized logs yield ``([], None)``: this reader never raises
    and never guesses, because it runs inside startup housekeeping.
    """
    try:
        if not path.exists() or path.stat().st_size > MAX_LOG_BYTES:
            return [], None
        text = path.read_text(encoding="utf-8")
    except OSError:
        return [], None
    lines = [line for line in text.splitlines() if line.strip()]
    previous = GENESIS
    for index, raw in enumerate(lines, start=1):
        try:
            record = json.loads(raw)
        except ValueError:
            return lines, index
        if not isinstance(record, dict) or record.get("previous") != previous:
            return lines, index
        if record.get("sequence") != index:
            return lines, index
        previous = _digest_text(raw)
    return lines, None


def _missing_committed_lines(state_dir: Path | str, lines: list[str]) -> int | None:
    """The first sequence the checkpoint committed to that the log no longer has.

    A chain walk proves only that the lines *present* agree with each other. It
    cannot prove that a line the checkpoint already committed to is still there:
    dropping the newest line leaves a shorter chain that verifies cleanly, and
    the next append then re-anchors ``previous`` on the surviving prefix, which
    hides the removal permanently. Detecting that equivocation needs a
    commitment over the sequence and not only over its last element (arXiv
    cs/0302010); the checkpoint's ``sequence`` is that commitment here. A log
    shorter than its own checkpoint is therefore a break, reported at the first
    missing sequence. ``None`` means no checkpoint or a log at least as long as
    the one the checkpoint names.
    """
    checkpoint = _read_checkpoint(state_dir)
    sequence = checkpoint.get("sequence") if checkpoint else None
    if isinstance(sequence, int) and len(lines) < sequence:
        return len(lines) + 1
    return None


def verify_chain(state_dir: Path | str) -> dict[str, Any]:
    """Whether the log is intact, and where it is not.

    Three independent checks. The chain walk catches a rewritten, removed or
    reordered line at the first line that no longer agrees with its predecessor.
    That walk is blind to a rewrite of the *last* line -- nothing follows it --
    so the checkpoint's recorded ``line_hash`` is compared with the last line as
    well; a log whose tail was edited after the checkpoint is therefore also
    reported, at the sequence the checkpoint names. It is equally blind to the
    *removal* of committed lines, which leaves a shorter chain that still agrees
    with itself and which the next append would silently re-anchor; a log shorter
    than the ``sequence`` its own checkpoint names is reported as a break at the
    first missing line.
    """
    path = Path(state_dir) / LOG_NAME
    lines, broken = _read_log(path)
    report: dict[str, Any] = {"ok": broken is None, "lines": len(lines), "path": str(path)}
    if broken is not None:
        report["broken_at"] = broken
        report["reason"] = "line hash, sequence or predecessor does not match the chain"
        return report
    checkpoint = _read_checkpoint(state_dir)
    if checkpoint is not None:
        sequence = checkpoint.get("sequence")
        recorded_hash = checkpoint.get("line_hash")
        # A chain walk only proves that the lines present agree with each other.
        # It cannot prove that a line the checkpoint already committed to is
        # still there: removing the newest line leaves a shorter, internally
        # consistent chain, which is exactly the equivocation a commitment over
        # the sequence must rule out (arXiv cs/0302010). So a log shorter than
        # its own checkpoint is reported as a break, not as a clean log.
        missing = _missing_committed_lines(state_dir, lines)
        if missing is not None:
            report["ok"] = False
            report["broken_at"] = missing
            report["reason"] = "the log is shorter than its checkpoint: committed lines were removed"
        elif sequence == len(lines) and isinstance(recorded_hash, str) and recorded_hash != _digest_text(lines[-1]):
            report["ok"] = False
            report["broken_at"] = sequence
            report["reason"] = "the last line no longer hashes to the checkpoint's line_hash"
    if report["ok"]:
        # Both checks above live entirely inside the files the checkpoint writer
        # rewrites, so a writer who replaces *both* consistently proves nothing
        # about itself: a forged shorter chain plus a matching checkpoint used to
        # verify clean (reproduced on this code). The anchor file is a
        # third, append-only file the same writer never edits, so its line for a
        # sequence is a commitment that cannot be re-issued (RFC 9162 section
        # 2.1.4.2, split-view protection; arXiv cs/0302010).
        anchored = _anchor_hashes(state_dir)
        if anchored:
            highest = max(anchored)
            if highest > len(lines):
                report["ok"] = False
                report["broken_at"] = len(lines) + 1
                report["reason"] = "an anchor commits to a sequence the log no longer has"
            else:
                for index in sorted(anchored):
                    if anchored[index] != _digest_text(lines[index - 1]):
                        report["ok"] = False
                        report["broken_at"] = index
                        report["reason"] = "a log line no longer hashes to its anchor"
                        break
            if report["ok"] and checkpoint is not None:
                recorded_sequence = checkpoint.get("sequence")
                if isinstance(recorded_sequence, int) and recorded_sequence < highest:
                    report["ok"] = False
                    report["broken_at"] = highest
                    report["reason"] = "the checkpoint is older than an anchor: it was rolled back"
    return report


def _anchor_hashes(state_dir: Path | str) -> dict[int, str]:
    """Every sequence the anchor file names, mapped to the hash it recorded.

    Bounded and total: an unreadable or oversized anchor file yields ``{}``, and
    an unparsable line is skipped rather than raised, because this reader runs
    inside startup housekeeping. A missing file is not reported as a break -- it
    is a log written before anchors existed, and that limit is stated in the
    module docstring rather than hidden here.
    """
    path = Path(state_dir) / ANCHOR_NAME
    try:
        if not path.exists() or path.stat().st_size > MAX_ANCHOR_BYTES:
            return {}
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    anchors: dict[int, str] = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        sequence = record.get("sequence")
        line_hash = record.get("line_hash")
        if isinstance(sequence, int) and isinstance(line_hash, str):
            anchors[sequence] = line_hash
    return anchors


def _append_anchor(state_dir: Path | str, sequence: int, line_hash: str) -> None:
    """Append one anchor line, then fsync; the file is never rewritten.

    The anchor is written *after* the log line it names, so a process killed in
    between leaves a log line with no anchor -- not a break, because anchors are
    a lower bound on what was committed and never an upper one. The reverse, an
    anchor whose line is gone or changed, is a break.
    """
    path = Path(state_dir) / ANCHOR_NAME
    record = {"sequence": sequence, "line_hash": line_hash, "at": utc_now()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_checkpoint(state_dir: Path | str) -> dict[str, Any] | None:
    path = Path(state_dir) / CHECKPOINT_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_checkpoint(state_dir: Path | str, record: dict[str, Any]) -> None:
    """Replace the pointer atomically; the log itself is never rewritten."""
    directory = Path(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CHECKPOINT_NAME
    temporary = path.with_name(f"{path.name}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def checkpoint_memory_audit(
    connection: Any,
    state_dir: Path | str,
    *,
    reason: str,
    run_id: str | None = None,
    generation: int | None = None,
) -> dict[str, Any]:
    """Append one metadata record and move the pointer to it.

    Best-effort by contract: it runs at the end of an episode and at startup, so
    an unreadable directory or a closed store must yield a status, not an
    exception. A broken chain is never repaired here -- the new record carries
    ``chain_broken_at`` so the break stays visible in the log forever.
    """
    directory = Path(state_dir)
    log_path = directory / LOG_NAME
    try:
        current = memory_digest(connection)
        lines, broken = _read_log(log_path)
        # A truncation the checkpoint already committed past is as much a break
        # as a rewritten line, and it must be carried into this record: appending
        # over the missing tail otherwise re-anchors the chain on the surviving
        # prefix and erases the evidence of the removal.
        missing = _missing_committed_lines(directory, lines)
        if broken is None:
            broken = missing
        previous = _digest_text(lines[-1]) if lines else GENESIS
        record: dict[str, Any] = {
            "sequence": len(lines) + 1,
            "at": utc_now(),
            "reason": reason,
            "run_id": run_id,
            "generation": generation,
            **current,
            "previous": previous,
        }
        if broken is not None:
            record["chain_broken_at"] = broken
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        directory.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        line_hash = _digest_text(line)
        _append_anchor(directory, record["sequence"], line_hash)
        _write_checkpoint(directory, {**record, "line_hash": line_hash})
        return {
            "status": "recorded",
            "sequence": record["sequence"],
            "rows": current["rows"],
            "bytes": current["bytes"],
            "digest": current["digest"],
            "chain_broken_at": broken,
        }
    except Exception as exc:  # the audit must never break the cycle it observes
        return {"status": "failed", "error": str(exc)[:200]}


def _appended_since(state_dir: Path | str, sequence: Any) -> tuple[int, int]:
    """Lines appended after the checkpoint's sequence, and their total bytes."""
    path = Path(state_dir) / LOG_NAME
    lines, _broken = _read_log(path)
    if not isinstance(sequence, int):
        return 0, 0
    appended = lines[sequence:] if sequence <= len(lines) else []
    return len(appended), sum(len(line) + 1 for line in appended)


def memory_audit_continuity_report(connection: Any, state_dir: Path | str) -> dict[str, Any]:
    """Compare the store with the last checkpoint and report the difference.

    ``status`` is ``unchanged`` (the store still hashes to the recorded digest),
    ``changed_externally`` (it does not, so something wrote memories outside the
    checkpoints this process takes -- a run interrupted before its own
    checkpoint looks the same, which is why the provenance counts and the
    appended-line count are reported alongside), or ``no_baseline`` on a store
    that has never been checkpointed. A reader gets the three facts the owner
    asked for -- that it changed, when, and by how much -- and no content.
    """
    current = memory_digest(connection)
    checkpoint = _read_checkpoint(state_dir)
    if checkpoint is None:
        return {"status": "no_baseline", **current}
    recorded_rows = checkpoint.get("rows")
    recorded_digest = checkpoint.get("digest")
    if recorded_digest == current["digest"] and recorded_rows == current["rows"]:
        return {"status": "unchanged", "at": checkpoint.get("at"), **current}
    lines, bytes_appended = _appended_since(state_dir, checkpoint.get("sequence"))
    return {
        "status": "changed_externally",
        "checkpoint_at": checkpoint.get("at"),
        "checkpoint_rows": recorded_rows,
        "checkpoint_digest": recorded_digest,
        "rows_delta": current["rows"] - int(recorded_rows or 0),
        "bytes_delta": current["bytes"] - int(checkpoint.get("bytes") or 0),
        "log_lines_appended": lines,
        "log_bytes_appended": bytes_appended,
        **provenance_counts(connection),
        **current,
    }
