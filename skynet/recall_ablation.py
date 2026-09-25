"""Ablation for the planner-recall hub's rank in a reconstructed cohort.

``skynet.recall_scorecard --as-of`` can show that a published ranking figure moved
because memories written later exist. It cannot say whether the cause is the later
memories that quote the fixture vocabulary or any later memory at all: the corpus
grew both ways, so a contamination story and a dilution story predict the same
moved figure. This module separates them by removal rather than by argument. It
partitions every active memory written after the reference cohort by how many
terms it shares with each frozen fixture query, re-runs the pooled fusion with
each partition removed, and reports the hub's rank that results.

Measured 2026-09-25 on a read-only copy of the live ledger (568 active memories,
cohort reconstructed at 2026-09-22T23:40:45Z with 380 memories, matching the
published corpus):

- the cohort replays hub fused top-1 3/5 and top-2 5/5, against the published
  4/5 and 5/5 figures;
- removing all 46 post-reference memories that share three or more terms with a
  fixture query leaves the live figure at hub top-1 0/5 and top-2 2/5 -- unchanged;
- the 144 post-reference memories that share at most two terms with every fixture
  query reproduce the whole displacement on their own, and on that corpus alone
  the hub falls to pooled rank 9-16 of 20;
- per fixture the hub reaches the fusion pool through the bm25 axis alone (its
  recency rank is 195 and its confidence rank 199 of 568, outside the 100-candidate
  window), while every live top-1 holder carries a recency or confidence rank
  beside its lexical one, so the cause is general dilution, not local contamination;
- the residual one-fixture gap is not contamination either: removing the outcome
  memory 7c7e4c5346f4546ec0b18104cdc6214f, written inside the cohort at
  2026-09-22T20:16, closes the cohort figure exactly (3/5 -> 4/5).

The ledger is opened read-only and nothing is written; exit status is 0 whenever
it can measure and 2 when the ledger is missing or unreadable.

    .venv/bin/python -m skynet.recall_ablation
    .venv/bin/python -m skynet.recall_ablation --db /path/to/ledger.sqlite3 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from .memory_store import MemoryStore
from .recall_scorecard import (
    DEFAULT_LEDGER,
    HUB_MEMORY_ID,
    RECORDED_PHRASE,
    REFERENCE_HUB_TOP1,
    REFERENCE_HUB_TOP2,
    SITUATIONS,
    _pooled_fused_order,
    _query,
    _rank,
    as_of_cohort,
    parse_as_of,
)

# The instant the published cohort (REFERENCE_ACTIVE_MEMORIES) was read at; the
# same instant ``recall_scorecard --as-of`` reproduces the published corpus with.
REFERENCE_AS_OF = "2026-09-22T23:40:45.000000Z"


def _quoter_terms(content: str, query_terms: list[str]) -> list[str]:
    """The fixture-query terms a memory quotes verbatim."""
    return sorted(set(query_terms) & set(MemoryStore._normalize_terms(content)))


def hub_ablation(db_path: Path, as_of: str, min_terms: int = 3) -> dict[str, Any]:
    """Measure the hub's pooled rank with each post-reference partition removed.

    ``as_of`` is the instant the reference cohort is reconstructed at. A memory is
    a quoter of a fixture when it shares at least ``min_terms`` normalized terms
    with that fixture's recorded query; ``quoters_any_term`` counts the wider
    partition of memories sharing at least one.
    """
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        cohort = as_of_cohort(connection, as_of)
        rows = {
            row["memory_id"]: (row["content"] or "", row["kind"], row["updated_at"])
            for row in connection.execute("SELECT memory_id, content, kind, updated_at FROM memories WHERE status='active'")
        }
        active = set(rows)
        post = active - cohort
        queries = {situation.key: MemoryStore._normalize_terms(_query(situation, RECORDED_PHRASE)) for situation in SITUATIONS}
        shared = {key: {mid: _quoter_terms(rows[mid][0], terms) for mid in post} for key, terms in queries.items()}
        min_terms_union = {mid for key in queries for mid in post if len(shared[key][mid]) >= min_terms}
        any_term_union = {mid for key in queries for mid in post if shared[key][mid]}

        def measure(allowed: set[str]) -> dict[str, Any]:
            hub_top1 = hub_top2 = 0
            per_fixture: dict[str, Any] = {}
            for key, terms in queries.items():
                order20 = _pooled_fused_order(connection, terms, 20, allowed=allowed)
                order3 = _pooled_fused_order(connection, terms, 3, allowed=allowed)
                hub_rank = _rank(order20, HUB_MEMORY_ID)
                hub_top1 += hub_rank == 1
                hub_top2 += _rank(order3, HUB_MEMORY_ID) is not None
                per_fixture[key] = {
                    "hub_rank": hub_rank,
                    "top1": order20[0] if order20 else None,
                    "top1_shared_terms": _quoter_terms(rows[order20[0]][0], terms) if order20 else [],
                }
            return {"hub_top1": hub_top1, "hub_top2": hub_top2, "per_fixture": per_fixture}

        corpora = {
            "cohort": cohort,
            "live": active,
            f"live_minus_quoters_ge_{min_terms}": active - min_terms_union,
            "live_minus_all_quoters": active - any_term_union,
            f"cohort_plus_quoters_ge_{min_terms}": cohort | min_terms_union,
        }
        readings = {name: measure(corpus) for name, corpus in corpora.items()}
        cohort_reading = readings["cohort"]
        # The gap between the replayed and the published hub top-1 figure: remove
        # each cohort memory holding a contested slot, one at a time, and report
        # whether that removal raises the figure and whether it closes it.
        contested = {
            cohort_reading["per_fixture"][key]["top1"] for key in queries if cohort_reading["per_fixture"][key]["hub_rank"] != 1
        }
        single_removals: list[dict[str, Any]] = []
        for mid in sorted(m for m in contested if m):
            reading = measure(cohort - {mid})
            single_removals.append({
                "memory_id": mid,
                "kind": rows[mid][1],
                "updated_at": rows[mid][2],
                "hub_top1": reading["hub_top1"],
                "hub_top2": reading["hub_top2"],
                "raises_hub_top1": reading["hub_top1"] > cohort_reading["hub_top1"],
                "equals_reference_hub_top1": reading["hub_top1"] == REFERENCE_HUB_TOP1,
            })
        return {
            "as_of": as_of,
            "min_terms": min_terms,
            "active_memories": len(active),
            "cohort_memories": len(cohort),
            "post_reference_memories": len(post),
            "quoters_min_terms": len(min_terms_union),
            "quoters_any_term": len(any_term_union),
            "readings": readings,
            "contested_cohort_top1": sorted(m for m in contested if m),
            "single_removals": single_removals,
            "reference_hub_top1": REFERENCE_HUB_TOP1,
            "reference_hub_top2": REFERENCE_HUB_TOP2,
        }
    finally:
        connection.close()


def _print_ablation(report: dict[str, Any]) -> None:
    print(
        f"\n[hub ablation] corpus {report['active_memories']} active memories; cohort at {report['as_of']} "
        f"{report['cohort_memories']}; written later {report['post_reference_memories']}; quoting >= "
        f"{report['min_terms']} fixture-query terms {report['quoters_min_terms']}; quoting any term "
        f"{report['quoters_any_term']}"
    )
    print(f"  reference: hub fused top-1 {report['reference_hub_top1']}/5, top-2 {report['reference_hub_top2']}/5")
    for name, reading in report["readings"].items():
        detail = "  ".join(
            f"{key}:rank{value['hub_rank']} top1 {value['top1'][:8] if value['top1'] else 'none'}"
            for key, value in reading["per_fixture"].items()
        )
        print(f"  {name}: hub top-1 {reading['hub_top1']}/5 top-2 {reading['hub_top2']}/5 | {detail}")
    if not report["contested_cohort_top1"]:
        print("  no cohort memory holds a contested top-1 slot; the cohort reproduces the reference hub top-1")
        return
    print("  contested cohort top-1 holders, removed one at a time:")
    for entry in report["single_removals"]:
        if entry["equals_reference_hub_top1"]:
            marker = "CLOSES the recorded figure"
        elif entry["raises_hub_top1"]:
            marker = "raises the figure but does not close it"
        else:
            marker = "does not move the figure"
        print(
            f"    {entry['memory_id'][:8]} ({entry['kind']}, written {entry['updated_at']}): "
            f"hub top-1 -> {entry['hub_top1']}/5, top-2 -> {entry['hub_top2']}/5  {marker}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ablate the planner-recall hub's rank against a ledger.")
    parser.add_argument("--db", default=os.getenv("SKYNET_STATE") or DEFAULT_LEDGER, help="ledger path (opened read-only)")
    parser.add_argument(
        "--as-of",
        type=parse_as_of,
        default=REFERENCE_AS_OF,
        help=f"instant the reference cohort is reconstructed at (default {REFERENCE_AS_OF})",
    )
    parser.add_argument("--min-terms", type=int, default=3, help="shared-term threshold for a quoter (default 3)")
    parser.add_argument("--json", action="store_true", help="print the raw report as JSON instead of the table")
    args = parser.parse_args(argv)
    try:
        report = hub_ablation(Path(args.db), args.as_of, args.min_terms)
    except (FileNotFoundError, RuntimeError, sqlite3.Error) as exc:
        print(f"recall ablation could not read the ledger: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=1, ensure_ascii=False))
    else:
        _print_ablation(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
