"""Frozen-envelope replay: the labelled planner-recall corpus as tracked code.

Generations 160-227 measured planner recall by rebuilding a corpus by hand in a scratch
script: the envelope filter, the label rule, the two query arms, the fold split and the
bootstrap seed were re-decided in each run, so a published figure could drift for reasons
that were not the reader. This module is the same replay with those choices written down.

FROZEN HERE

* The envelope set (``ENVELOPE_CLAUSE``): ``run_started`` rows carrying an OBJECT
  ``next_plan.previous_outcome`` and a NON-EMPTY ``next_plan.initial_prompt``, whose
  ``run_id`` has a ``run_results`` row. "Carries" is spelled as truth, not key presence:
  ``initial_prompt: ""`` writes the key with no planning context in it. On the
  2026-09-25 copies both spellings agree (165/165 and 185/185 rows, 0 disagreements).
* The label: ACTIVE memories whose ``source_run`` equals the envelope's own ``run_id`` --
  held out, because a run's memories are written after its search.
* The two arms, CALLED not re-derived: ``pq`` = ``Reactor._planner_memory_query`` and
  ``hq`` = ``Reactor._memory_query`` (the query whose hits reach a run's prompt).
* The fold split ``sha256(run_id) % 5``, the bootstrap seed, and the ``RECORDED`` figures
  with the memory that published each one, its n and its ledger horizon.

WHAT IT CANNOT DO

``--as-of`` restricts the replay to the memories a ledger could have held
(``recall_scorecard.as_of_cohort``), but BM25 is an INDEX-LEVEL statistic: the FTS index
still holds rows written after the instant, so their term frequencies enter every
ranking. Measured 2026-09-26 on a fresh copy of the live ledger (826 memories) bounded to
the horizon of the generation-206 copy (memories <= 2026-09-25T13:01:09.667589Z), the bm25
top-100 pool differs from the frozen copy's on 166/166 envelopes (top-20 on 164/166) and
the hq page reads h@1 .212 h@5 .381 h@20 .559 MRR .298 against .220/.381/.568/.302. So an
``--as-of`` number is a cohort-restricted approximation; only an unmodified copy of the
ledger the figure was read on reproduces it exactly. The module prints that distinction.

WHERE TO GET IT: ``envelope_rows``/``envelope_ids``, ``label_ids``, ``queries_for``,
``replayed_page``/``axis_pools`` (the rig's own axes + the shipped placement),
``read_cases`` (the SHIPPED page), ``fold_of``/``paired_interval``, ``RECORDED``, and
``build_report``, which also reports ``replay_agreement``: how many envelopes the rig's own
page reproduces ``MemoryStore.search`` on, per arm.

    .venv/bin/python -m skynet.recall_envelopes [--db PATH] [--as-of INSTANT] [--json]

THE REFERENCE COPY. The ledger the generation-206 figures were read on is kept outside
the repository, and ``FROZEN_REFERENCE["path"]`` is derived from this file's own location
(``Path(__file__).resolve().parents[2]``): in the live tree that is the copy measured in
generation 206, while a run from inside a proposal worktree derives
``<derived-worktree-root>/skynet-scratch/g206-frozen.sqlite3``, which is ``ABSENT`` and
falls back to the live ledger -- a symlink ancestor of this file has the same effect.
The startup sweep (``SelfImprovementManager.prune_orphan_worktrees``) reclaims stale
checkouts under the derived worktree root but spares a child without a ``.git`` entry, so
a directory such as ``skynet-scratch/`` parked beside that root survives it; verify with
``reference_ledger_state`` rather than assuming where the pin resolves.
``resolve_reference_ledger`` finds it through
``SKYNET_ENVELOPE_LEDGER`` first and the pinned path second, and every run prints its
state: ``PINNED`` (the recorded bytes are on disk), ``ABSENT`` (the copy is gone, so the
generation-206 reading is no longer re-measurable) or ``MISMATCH`` (a file is there but
it is not those bytes). A figure reproduced against a MISMATCH copy is not a reproduction.

With no ``--db``, a bare run measures the reference itself while its bytes verify
(``default_ledger``), so a printed reference verdict and the corpus beside it cannot
disagree; only an absent or mismatched reference falls through to ``SKYNET_STATE``.

Exit status is 0 whenever it can measure and 2 when the ledger is missing or has no
usable FTS index. The ledger is opened ``mode=ro``: this module writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any

from .memory_store import _RRF_K, _RRF_WEIGHTS, MemoryStore
from .models import AgentState
from .reactor import Reactor
from .recall_scorecard import DEFAULT_LEDGER, as_of_cohort, parse_as_of

# The envelope predicate, as code. A figure that does not say which of these filters
# it applied is not comparable with one that does.
ENVELOPE_CLAUSE = (
    "kind='run_started' AND json_type(payload,'$.next_plan.previous_outcome') = 'object' "
    "AND COALESCE(json_extract(payload,'$.next_plan.initial_prompt'), '') <> '' "
    "AND EXISTS (SELECT 1 FROM run_results rr WHERE rr.run_id = json_extract(payload,'$.run_id'))"
)
ENVELOPE_ORDER = "ORDER BY sequence DESC LIMIT 400"
LABEL_RULE = "active memories whose source_run equals the envelope's own run_id (held out: written after the search)"
FOLD_COUNT = 5
# The paired-bootstrap seed. It is printed with every interval so a later reader can
# re-derive the same bounds instead of recomputing "a" confidence interval.
BOOTSTRAP_SEED = 228
BOOTSTRAP_RESAMPLES = 2000

# The recorded figures, each with the ledger horizon and the n it was published at.
# ``ledger_memories`` is the number of memories in the copy it was read on, so a run
# that does not hold that copy knows its reproduction is against a moved corpus.
RECORDED: dict[str, dict[str, Any]] = {
    "hq_shipped_hit@5": {
        "value": 0.381,
        "hits": 45,
        "n": 118,
        "source": "memory 988d7e64c1abfc06a6021e655c9035e6 (generation 206, task f9d26679)",
        "ledger_memories": 703,
        "ledger_horizon": "2026-09-25T13:01:09.667589Z",
    },
    "pq_shipped_hit@5": {
        "value": 0.229,
        "hits": 27,
        "n": 118,
        "source": "memory 988d7e64c1abfc06a6021e655c9035e6 (generation 206, task f9d26679)",
        "ledger_memories": 703,
        "ledger_horizon": "2026-09-25T13:01:09.667589Z",
    },
}
ARMS = ("pq", "hq")

# The pinned reference copy. Its path is derived from this module's own location rather
# than spelled as an operator path (tests/test_no_hardcoded_paths scans tracked files),
# and it points at a sibling scratch directory OUTSIDE the repository: the tree may not
# carry a 241 MB binary, and the proposal-worktree root is swept at startup.
REFERENCE_ENV_VAR = "SKYNET_ENVELOPE_LEDGER"
FROZEN_REFERENCE: dict[str, Any] = {
    "path": Path(__file__).resolve().parents[2] / "skynet-scratch" / "g206-frozen.sqlite3",
    "sha256": "6bfa54ae9a7c908f0cbe864c802fc770bbbf38d1f4c2238fde0af4a173479d18",
    "bytes": 252342272,
    "source": (
        "a byte-identical copy of /tmp/skynet-scratch/g206/ledger.sqlite3, "
        "the copy the generation-206 figure was read on"
    ),
}


def file_digest(path: Path, *, chunk: int = 1 << 20) -> str:
    """The sha256 of a file, streamed so a 241 MB ledger is never read into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_reference_ledger() -> Path:
    """The reference ledger: ``SKYNET_ENVELOPE_LEDGER`` when set, else the pinned copy.

    The env var is first so a run can point the rig at another frozen copy without
    editing this module; the pinned path is the durable default.
    """
    override = os.getenv(REFERENCE_ENV_VAR)
    return Path(override) if override else Path(FROZEN_REFERENCE["path"])


def default_ledger() -> Path:
    """The ledger a bare run should measure: the reference while its bytes verify.

    The recorded figures are only re-measurable on the copy they were read on, so when
    the pinned bytes are on disk the reference IS the default measurement. Letting
    ``SKYNET_STATE`` win first (this host always sets it) made a bare run print
    ``PINNED`` while the replay header named the live ledger and both recorded figures
    read NOT reproduced -- a reference verdict reported over a corpus that never looked
    at the reference. Only with the reference absent or mismatched is the live ledger
    used, and the printed reference line always says which of the three states applied.
    """
    override = os.getenv(REFERENCE_ENV_VAR)
    if override:
        return Path(override)
    reference = Path(FROZEN_REFERENCE["path"])
    if reference_ledger_state(reference)["status"] == "PINNED":
        return reference
    configured = os.getenv("SKYNET_STATE")
    return Path(configured) if configured else Path(DEFAULT_LEDGER)


def reference_ledger_state(path: Path | None = None, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    """``PINNED``, ``ABSENT`` or ``MISMATCH`` for the reference copy: the acceptance gate's input.

    The three states are different facts, not one boolean. ABSENT means the file is gone
    and the generation-206 reading is no longer re-measurable from this host; MISMATCH
    means a file is there but it is not the recorded bytes, so a figure "reproduced" on it
    is not a reproduction. The byte size is checked before the digest: an oversized or
    undersized copy is already a mismatch and needs no 241 MB hash.
    """
    pinned = FROZEN_REFERENCE if expected is None else expected
    resolved = Path(path) if path is not None else resolve_reference_ledger()
    state: dict[str, Any] = {
        "path": str(resolved),
        "expected_sha256": pinned["sha256"],
        "expected_bytes": pinned["bytes"],
        "source": pinned.get("source", ""),
    }
    if not resolved.exists():
        state.update(
            {
                "status": "ABSENT",
                "observed_sha256": None,
                "observed_bytes": None,
                "reason": "the pinned copy is not present",
            }
        )
        return state
    size = resolved.stat().st_size
    if size != pinned["bytes"]:
        state.update(
            {
                "status": "MISMATCH",
                "observed_sha256": None,
                "observed_bytes": size,
                "reason": "byte size differs from the recorded copy",
            }
        )
        return state
    digest = file_digest(resolved)
    if digest != pinned["sha256"]:
        state.update(
            {
                "status": "MISMATCH",
                "observed_sha256": digest,
                "observed_bytes": size,
                "reason": "sha256 differs from the recorded copy",
            }
        )
        return state
    state.update(
        {
            "status": "PINNED",
            "observed_sha256": digest,
            "observed_bytes": size,
            "reason": "the recorded bytes are on disk",
        }
    )
    return state


def envelope_rows(connection: sqlite3.Connection, instant: str | None = None) -> list[sqlite3.Row]:
    """The frozen envelopes, newest first, optionally bounded by their write time."""
    query = f"SELECT sequence, run_id, payload, created_at FROM event_log WHERE {ENVELOPE_CLAUSE}"
    params: tuple[Any, ...] = ()
    if instant is not None:
        query += " AND created_at <= ?"
        params = (instant,)
    return connection.execute(f"{query} {ENVELOPE_ORDER}", params).fetchall()


def envelope_ids(connection: sqlite3.Connection, instant: str | None = None) -> list[str]:
    """The distinct run ids of the frozen envelope set, without their payloads."""
    return [row["run_id"] for row in envelope_rows(connection, instant) if row["run_id"]]


def label_ids(connection: sqlite3.Connection, run_id: str, allowed: set[str] | None = None) -> set[str]:
    """The held-out label of one envelope: its own run's active memories.

    ``allowed`` is the same cohort the page is read on. A label must be drawn from
    the corpus the figure is measured on, or a cohort-bounded replay would score its
    pages against memories the cohort did not hold.
    """
    status = "" if allowed is not None else " AND status='active'"
    rows = connection.execute(
        "SELECT memory_id FROM memories WHERE source_run=?" + status,
        (run_id,),
    ).fetchall()
    ids = {row["memory_id"] for row in rows}
    return ids if allowed is None else ids & allowed


def _selected_work(payload: dict[str, Any]) -> dict[str, Any] | None:
    pending = payload.get("pending_work") or []
    return pending[0] if pending and isinstance(pending[0], dict) else None


def _inbox_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in (payload.get("observations") or []) if isinstance(item, dict) and item.get("kind") == "inbox_notification"]


def queries_for(payload: dict[str, Any]) -> dict[str, str]:
    """Both arm queries for one frozen envelope, built by the shipped assemblers."""
    next_plan = payload.get("next_plan") or {}
    state = AgentState(
        next_plan={
            "previous_outcome": next_plan.get("previous_outcome") or {},
            "initial_prompt": next_plan.get("initial_prompt", ""),
            "next": next_plan.get("next", ""),
        }
    )
    return {
        "pq": Reactor._planner_memory_query(state, payload.get("active_goals") or []),
        "hq": Reactor._memory_query(_selected_work(payload), _inbox_events(payload), next_plan),
    }


def fold_of(run_id: str) -> int:
    """The fold a case falls in: ``sha256(run_id) % FOLD_COUNT``, as generations 174-206 split it."""
    return int(hashlib.sha256(run_id.encode()).hexdigest(), 16) % FOLD_COUNT


def axis_pools(
    connection: sqlite3.Connection,
    query: str,
    limit: int,
    *,
    allowed: set[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """The three fusion axes in rank order, re-derived by the rig.

    The SQL is written out rather than called through ``MemoryStore.search``: a
    cross-check that calls the thing it checks proves nothing. ``_RRF_K`` and
    ``_RRF_WEIGHTS`` are imported, so the check fails if either side's constants move.
    ``allowed`` REPLACES the ``status='active'`` clause, like the shipped reader.
    """
    terms = MemoryStore._normalize_terms(query)
    match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
    candidate = limit * 5
    cohort = ""
    axis_cohort = ""
    params: tuple[Any, ...] = ()
    if allowed is not None:
        cohort = f" AND m.memory_id IN ({','.join('?' for _ in sorted(allowed))})"
        axis_cohort = f" AND memories.memory_id IN ({','.join('?' for _ in sorted(allowed))})"
        params = tuple(sorted(allowed))
    active = "" if allowed is not None else " AND m.status='active'"
    axis_active = "" if allowed is not None else " AND status='active'"
    bm25 = [
        row["memory_id"]
        for row in connection.execute(
            "SELECT m.memory_id FROM memories_fts JOIN memories m ON m.memory_id = memories_fts.memory_id "
            f"WHERE memories_fts MATCH ?{active}{cohort} ORDER BY bm25(memories_fts), m.memory_id LIMIT ?",
            (match, *params, candidate),
        )
    ]
    recency = [
        row["memory_id"]
        for row in connection.execute(
            f"SELECT memories.memory_id FROM memories WHERE 1=1{axis_active}{axis_cohort} "
            "ORDER BY updated_at DESC, memory_id LIMIT ?",
            (*params, candidate),
        )
    ]
    confidence = [
        row["memory_id"]
        for row in connection.execute(
            f"SELECT memories.memory_id FROM memories WHERE 1=1{axis_active}{axis_cohort} "
            "ORDER BY confidence DESC, updated_at DESC, memory_id LIMIT ?",
            (*params, candidate),
        )
    ]
    return bm25, recency, confidence


def replayed_page(
    connection: sqlite3.Connection,
    query: str,
    limit: int = 20,
    *,
    allowed: set[str] | None = None,
) -> list[str]:
    """The page the rig's own axes + the shipped placement produce for ``query``."""
    bm25, recency, confidence = axis_pools(connection, query, limit, allowed=allowed)
    scores: dict[str, float] = {}
    for pool, (_axis, weight) in zip((bm25, recency, confidence), _RRF_WEIGHTS, strict=False):
        for rank, memory_id in enumerate(pool, start=1):
            scores[memory_id] = scores.get(memory_id, 0.0) + weight / (_RRF_K + rank)
    if not scores:
        return []
    return [memory_id for memory_id, _ in MemoryStore._place_bounded(scores, bm25, limit)]


def hit(rank: list[str], label: set[str], k: int) -> bool:
    return bool(label & set(rank[:k]))


def reciprocal_rank(rank: list[str], label: set[str]) -> float:
    for position, memory_id in enumerate(rank, start=1):
        if memory_id in label:
            return 1.0 / position
    return 0.0


def paired_interval(differences: list[float], seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float, float]:
    """Paired bootstrap over the per-envelope differences: ``(mean, low95, high95)``.

    Both readers saw the same envelope, so the sampling unit is the envelope.
    """
    if not differences:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    mean = sum(differences) / len(differences)
    draws = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(resamples)
    )
    return (mean, draws[int(0.025 * resamples)], draws[int(0.975 * resamples)])


def read_cases(
    store: MemoryStore,
    cases: list[dict[str, Any]],
    *,
    limit: int = 20,
    allowed: set[str] | None = None,
) -> dict[str, Any]:
    """Measure the SHIPPED page for one arm; ``allowed`` also bounds the label."""
    hits = {1: 0, 5: 0, limit: 0}
    reciprocal: list[float] = []
    folds: dict[int, list[int]] = {}
    pages: dict[str, list[str]] = {}
    for case in cases:
        label = case["label"] if allowed is None else (case["label"] & allowed)
        page = [row["memory_id"] for row in store.search(case["query"], limit=limit, allowed=allowed)]
        pages[case["run_id"]] = page
        for k in hits:
            hits[k] += hit(page, label, k)
        reciprocal.append(reciprocal_rank(page, label))
        fold = fold_of(case["run_id"])
        bucket = folds.setdefault(fold, [0, 0])
        bucket[1] += 1
        bucket[0] += hit(page, label, 5)
    n = len(cases)
    return {
        "n": n,
        "limit": limit,
        "hit@1": hits[1] / n if n else 0.0,
        "hit@5": hits[5] / n if n else 0.0,
        f"hit@{limit}": hits[limit] / n if n else 0.0,
        "hit@1_hits": hits[1],
        "hit@5_hits": hits[5],
        f"hit@{limit}_hits": hits[limit],
        "mrr": sum(reciprocal) / n if n else 0.0,
        "folds": {str(k): {"hit@5": v[0], "n": v[1]} for k, v in sorted(folds.items())},
        "pages": pages,
    }


def build_report(db_path: Path, instant: str | None = None) -> dict[str, Any]:
    """Rebuild the frozen corpus and measure the shipped reader on every arm."""
    if not db_path.exists():
        raise FileNotFoundError(f"ledger not found: {db_path}")
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        store = MemoryStore(connection)
        if not store.fts_available:
            raise RuntimeError(f"ledger has no usable FTS index: {db_path}")
        rows = envelope_rows(connection, instant)
        allowed = as_of_cohort(connection, instant) if instant else None
        active = int(connection.execute("SELECT COUNT(*) FROM memories WHERE status='active'").fetchone()[0])
        cases: list[dict[str, Any]] = []
        unlabelled = 0
        for row in rows:
            payload = json.loads(row["payload"])
            label = label_ids(connection, row["run_id"], allowed)
            if not label:
                unlabelled += 1
                continue
            for arm, query in queries_for(payload).items():
                if not MemoryStore._normalize_terms(query):
                    continue
                cases.append({"run_id": row["run_id"], "arm": arm, "query": query, "label": label})
        per_arm = {
            arm: {
                "n": sum(1 for case in cases if case["arm"] == arm),
                "envelopes": len({case["run_id"] for case in cases if case["arm"] == arm}),
            }
            for arm in ARMS
        }
        readings = {
            arm: read_cases(store, [case for case in cases if case["arm"] == arm], allowed=allowed)
            for arm in ARMS
        }
        # The rig checks itself: the page built from its own axis SQL + the shipped
        # placement must be the page MemoryStore.search returns, or a difference
        # between two readers would be unreadable from a difference in the rig.
        agreement: dict[str, dict[str, int]] = {}
        for arm in ARMS:
            pages = readings[arm]["pages"]
            cases_for_arm = [case for case in cases if case["arm"] == arm]
            same = sum(
                1
                for case in cases_for_arm
                if replayed_page(connection, case["query"], 20, allowed=allowed) == pages[case["run_id"]]
            )
            agreement[arm] = {"envelopes": len(cases_for_arm), "replay_equals_shipped": same}
        readings = {
            arm: {key: value for key, value in reading.items() if key != "pages"}
            for arm, reading in readings.items()
        }
        # A copy of the ledger the figure was read on can reproduce it to the hit; a
        # grown ledger can only be compared.
        comparisons: dict[str, dict[str, Any]] = {}
        for name, recorded in RECORDED.items():
            arm = name.split("_", 1)[0]
            reading = readings.get(arm) or {}
            observed_hits = reading.get("hit@5_hits")
            observed_n = reading.get("n")
            entry = {
                "arm": arm,
                "recorded": recorded["value"],
                "recorded_hits": recorded["hits"],
                "recorded_n": recorded["n"],
                "recorded_source": recorded["source"],
                "recorded_ledger_memories": recorded["ledger_memories"],
                "recorded_ledger_horizon": recorded["ledger_horizon"],
                "observed": reading.get("hit@5"),
                "observed_hits": observed_hits,
                "observed_n": observed_n,
                "reproduced": observed_hits == recorded["hits"] and observed_n == recorded["n"],
            }
            comparisons[name] = entry
        return {
            "ledger": str(db_path),
            "as_of": instant,
            "reference_ledger": reference_ledger_state(),
            "corpus": {
                "memories": active,
                "cohort_memories": len(allowed) if allowed is not None else None,
                "fts_rows": int(connection.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]),
            },
            "envelope_rule": ENVELOPE_CLAUSE,
            "label_rule": LABEL_RULE,
            "fold_rule": f"sha256(run_id) % {FOLD_COUNT}",
            "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES},
            "envelopes": len(rows),
            "labelled_envelopes": len({case["run_id"] for case in cases}),
            "unlabelled_envelopes": unlabelled,
            "per_arm_cases": per_arm,
            "readings": readings,
            "replay_agreement": agreement,
            "recorded_figures": comparisons,
        }
    finally:
        connection.close()


def _print_report(report: dict[str, Any]) -> None:
    """One line per arm plus the recorded-vs-observed verdict for each figure."""
    corpus = report["corpus"]
    scope = f"{corpus['memories']} active memories, fts rows {corpus['fts_rows']}"
    if report["as_of"]:
        scope += f"; replay restricted to the {corpus['cohort_memories']}-memory cohort at {report['as_of']}"
    print(f"frozen-envelope replay: {report['ledger']} ({scope})")
    print(
        f"  envelope rule {report['envelopes']} envelopes, {report['labelled_envelopes']} labelled"
        f" ({report['unlabelled_envelopes']} carry no active memory of their own run); label rule: {report['label_rule']}"
    )
    print(f"  fold rule {report['fold_rule']}; paired bootstrap seed {report['bootstrap']['seed']}")
    for arm in ARMS:
        r = report["readings"][arm]
        a = report["replay_agreement"][arm]
        folds = " ".join(f"f{k}:{v['hit@5']}/{v['n']}" for k, v in r["folds"].items())
        top = f"hit@{r['limit']}"
        print(
            f"  {arm} n={r['n']} h@1 {r['hit@1']:.3f} ({r['hit@1_hits']}) h@5 {r['hit@5']:.3f} ({r['hit@5_hits']}) "
            f"{top} {r[top]:.3f} MRR {r['mrr']:.3f} folds {folds}"
        )
        print(
            f"      self-check: the rig's own axes + the shipped MemoryStore._place_bounded reproduce "
            f"MemoryStore.search on {a['replay_equals_shipped']}/{a['envelopes']} envelopes"
        )
    for name, entry in report["recorded_figures"].items():
        verdict = "REPRODUCED to the hit" if entry["reproduced"] else "NOT reproduced on this corpus"
        print(
            f"  {name}: recorded {entry['recorded_hits']}/{entry['recorded_n']} = {entry['recorded']:.3f} "
            f"[{entry['recorded_source']}; ledger {entry['recorded_ledger_memories']} memories at "
            f"{entry['recorded_ledger_horizon']}]; observed {entry['observed_hits']}/{entry['observed_n']} = "
            f"{entry['observed']:.3f} -> {verdict}"
        )
    reference = report["reference_ledger"]
    observed = reference["observed_sha256"] or (
        f"size {reference['observed_bytes']}" if reference["observed_bytes"] is not None else "nothing"
    )
    print(
        f"  reference ledger: {reference['status']} at {reference['path']} "
        f"(expected sha256 {reference['expected_sha256']}, {reference['expected_bytes']} bytes; "
        f"observed {observed}; {reference['reason']})"
    )
    print(
        "  caveat: BM25 is an INDEX-LEVEL statistic, so restricting a GROWN ledger to a past instant does not "
        "reproduce a frozen copy's ranking; only an unmodified copy of the ledger the figure was read on does."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the frozen labelled-envelope corpus and measure the shipped reader.")
    parser.add_argument(
        "--db",
        default=None,
        help=(
            f"ledger path (opened read-only); {REFERENCE_ENV_VAR} first, then the pinned reference while its bytes "
            "verify, then SKYNET_STATE, then the live ledger"
        ),
    )
    parser.add_argument("--as-of", type=parse_as_of, default=None, help="restrict the replay to an ISO-8601 instant")
    parser.add_argument("--json", action="store_true", help="print the raw report as JSON instead of the table")
    args = parser.parse_args(argv)
    # Resolved after --help, so printing the usage never hashes a 241 MB ledger.
    ledger = Path(args.db) if args.db else default_ledger()
    try:
        report = build_report(ledger, args.as_of)
    except (FileNotFoundError, RuntimeError, sqlite3.Error) as exc:
        print(f"envelope replay could not read the ledger: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=1, ensure_ascii=False))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
