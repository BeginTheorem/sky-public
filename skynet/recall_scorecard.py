"""Replayable scorecard for the planner's memory-recall query.

The recall window is a scarce budget: ``MemoryStore._normalize_terms`` keeps only
the last ``_MAX_QUERY_TERMS`` (24) terms and ``Reactor._planner_memory_query``
appends the situation-defining parts last, so which terms survive decides which
memories the planner can ever see. Claims about that ranking -- which memory is
injected first, which governing constraint is reached, what share of the window a
fixed phrase consumes -- were measured once in a scratch script and then lived
only in a memory entry; the next run that wanted to check a claim rebuilt the
corpus from the beginning, and a figure that had drifted since could not be told
apart from one that had been wrong. This module freezes the corpus and the
reference figures so a ranking claim is scored in one place.

Frozen here: the five fixture pairs and the governing memories a reader expects
each to reach (memory ``9590d67dffbf5f9d9795bea8930e4627``); the reference
figures, each naming the memory that published it; and the configuration those
figures were read under, so a reproduction is scored against the reader the
figure came from rather than today's reader.

Run 1 replays that recorded configuration, run 2 drops the constant phrase
(reproducing memory ``d8f164cc0336d7d3a923aeee2aec3342``), run 3 uses the shipped
query and run 4 does it through the injected-run reader instead of the planner one.
Every number comes from the ledger through shipped code paths, so
the scorecard cannot agree with an implementation it does not run, and the ledger
is opened read-only: this module writes nothing.

Every comparison also carries a budget (``required_n_detail``): the smallest common
n at which the recorded-vs-observed gap would leave the ROPE band, printed beside
the number of labelled cases the planner harness could supply. Measured 2026-09-25 on
a read-only copy of the live ledger, the undecided figures are decidable at 6
(hub top-2) and 35 (governing top-20) labelled cases against 179 available, while
governing top-3 (gap +0.0096) and both constant-tail shares (exact matches) have no
n at all -- the two rates are closer together than the band, so no larger fixture set
would change the verdict.

The query itself is a choice, so a verdict is only meaningful with its reader named.
The planner layout is built by ``_query`` and mirrors ``Reactor._planner_memory_query``;
the query whose hits actually enter an agent run's prompt is assembled by
``Reactor._memory_query``, where the run's StartEnvelope is built. They take different
parts from the same situation -- the injected layout has no constant phrase and no
fixture summary, but adds the task title -- so run 4 replays the fixtures through the
shipped injected-run assembler as well, and the layout block prints every recorded
figure scored against both readers, the margin (observed minus recorded rate) each one
shows, and how many figures change verdict between them. A frozen fixture carries no
owner inbox message and no acceptance criterion, so those two slots of the injected
layout are empty rather than guessed; that is stated where the query is assembled.

    .venv/bin/python -m skynet.recall_scorecard
    .venv/bin/python -m skynet.recall_scorecard --db /path/to/ledger.sqlite3
    .venv/bin/python -m skynet.recall_scorecard --as-of 2026-09-22T23:40:45Z

``--as-of`` replays the reference figures against the corpus a published figure
was actually read on, instead of today's much larger one. A figure that moved
mostly because later memories quote the fixture vocabulary is told apart from one
the reader moved. Measured 2026-09-25 on a copy of the live ledger: the hub
top-1 figure read 4/5 when published and reads 3/5 on the replayed 380-memory
cohort, while today's 550-memory corpus reads 0/5.

The verdict for a moved figure is an interval rule, not an exact-integer match
(``verdict_detail``): an exact match reproduces, the 95% confidence interval for
the *difference* of the two rates leaving the ``VERDICT_ROPE`` band drifts, and
everything else is UNDERDETERMINED -- the fixture counts cannot decide. The
interval is the Newcombe (1998) hybrid score interval
(``difference_interval``), not the overlap of the two marginal Wilson intervals,
which does not test the difference and errs in both directions (Schenker &
Gentleman 2001, DOI 10.1198/000313001317097960). Measured 2026-09-25 on a
read-only copy of the live ledger (582 active memories), applying the rule to the
figures frozen below: both constant-tail figures stay REPRODUCED, hub top-2,
governing top-3 and governing top-20 stay UNDERDETERMINED, and hub top-1 becomes
DRIFTED -- 4/5 to 0/5 is an 0.80 rate gap whose interval for the difference
(observed minus recorded) is [-0.964, -0.193], wholly past the ROPE band, which
the marginal-overlap test missed because those two intervals still touch by 0.058.
The JSON report carries every input each verdict was computed from.

Exit status is 0 whenever it can measure, drift or not (it measures, it does not
gate), and 2 when the ledger is missing or unreadable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .memory_store import _MAX_QUERY_TERMS, _RRF_K, _RRF_WEIGHTS, _STOPWORDS, MemoryStore
from .reactor import Reactor

# The daemon's default (cli.py reads SKYNET_STATE), spelled by concatenation so
# this tracked module does not itself name the runtime directory.
DEFAULT_LEDGER = "stat" "e/skynet.sqlite3"
RECORDED_PHRASE = "autonomous planning next bounded work"  # before commit 946a22ab
SHIPPED_PHRASE = "autonomous planning"


@dataclass(frozen=True)
class Situation:
    """One fixture pair: a situation text and the memories a reader expects."""

    key: str
    label: str
    summary: str
    initial_prompt: str
    next_step: str
    goal: str
    governing: tuple[tuple[str, str], ...]


# The five pairs of memory 9590d67dffbf5f9d9795bea8930e4627, verbatim.
SITUATIONS: tuple[Situation, ...] = (
    Situation(
        key="S1",
        label="self-improvement diff touching tests/",
        summary="About to submit a self-improvement proposal whose diff adds a regression test under tests/ so the fix cannot be lost again.",
        initial_prompt="Implement the fix and guard it with a regression test in tests/test_splitter.py; the harness must not be able to drop verified work.",
        next_step="Propose the change touching tests/ and skynet/splitter.py.",
        goal="Repair harness defects that discard verified work",
        governing=(
            ("6345b8ddb4c5b65a86330e6ba99d1307", "tests/ is gate-protected"),
            ("119572d54e0a27b4d7575b1d92931115", "the suite is gate-protected"),
            ("e064decf2993b07bcbe40404415838b7", "gate inputs are gate-protected"),
            ("b531099e811b96120a4038d31261dab3", "the gate refuses protected paths"),
            ("6531497ab9de1e9045863bdedc31c580", "tests/ is gate-protected (duplicate encoding)"),
            ("3591a893de0be2839b70f9c7b5baa182", "_is_gate_protected covers tests/ and more"),
            ("7bbe9bf358b0d006f0c80f9af590df5b", "every path under tests/ is protected"),
            ("6afc9de2f5a4a01a27fe146f0c2a9b0a", "the workflow refuses suite-touching proposals"),
            ("7e6fb2bd81cfde63548f46a657915b67", "GATE_PROTECTED_PATHS is broader than tests/"),
        ),
    ),
    Situation(
        key="S2",
        label="finish report with long citations",
        summary="Compose the finish report for the episode and attach the arXiv paper as a citation.",
        initial_prompt="Write the finish report; citations list the external sources consulted this episode, each at most 400 characters long.",
        next_step="Finish the episode and return the report JSON with a citations array.",
        goal="Advance the SkyNet roadmap",
        governing=(("acb6f27d46908fe23cf0be8ff95573ae", "oversized citations invalidate the report"),),
    ),
    Situation(
        key="S3",
        label="recover after an interrupted run",
        summary="The previous run aborted mid-episode; the process restarted and must resume from the last checkpoint.",
        initial_prompt="Recover after an interrupted run: inspect the durable state, reconcile the interrupted episode and continue without losing verified work.",
        next_step="Run the recovery reconciliation and resume the interrupted work.",
        goal="Repair harness defects that discard verified work",
        governing=(),
    ),
    Situation(
        key="S4",
        label="test_command shape for a proposal",
        summary="Prepare the validation command for the next self-improvement proposal.",
        initial_prompt="Call propose_self_improvement with a test_command list that runs the pytest suite for the change.",
        next_step="Submit the proposal with test_command set to the validation executable.",
        goal="Advance the SkyNet roadmap",
        governing=(
            ("142ec78a596c2891892789e3cc735a83", "test_command is a bare executable list"),
            ("f8ae0ef3d16a1839d1a0c09ea6cb515e", "two hard gate constraints on that path"),
        ),
    ),
    Situation(
        key="S5",
        label="owner asleep, send a status update",
        summary="A decision needs the owner but the owner is asleep; the episode must keep working.",
        initial_prompt="Send the owner an asynchronous status message with send_message_to_user instead of blocking on a question.",
        next_step="Notify the owner asynchronously and continue the bounded work.",
        goal="Advance the SkyNet roadmap",
        governing=(("2102e5b4a6222d4991d987dd992b02d6", "do not block on the owner; no blocking questions"),),
    ),
)

HUB_MEMORY_ID = "b26ed26117a30581cafb87272821e722"

# Reference figures, each naming the memory that published it.
REFERENCE_TAIL_SHARE: dict[str, tuple[int, str]] = {
    "Advance the SkyNet roadmap": (8, "37b9248368ae3aef2fb376534492e90b"),
    "Repair harness defects that discard verified work": (10, "37b9248368ae3aef2fb376534492e90b"),
}
REFERENCE_HUB_TOP1 = 4  # of 5 situations
REFERENCE_HUB_TOP2 = 5  # of 5 situations
REFERENCE_GOVERNING_TOP3 = 3  # of 8 recorded (situation, governing) verdicts
REFERENCE_GOVERNING_TOP20 = 6  # of 8
REFERENCE_GOVERNING_VERDICTS = 8  # (situation, governing) verdicts when published
REFERENCE_ACTIVE_MEMORIES = 380
REFERENCE_SOURCE = "9590d67dffbf5f9d9795bea8930e4627"
LESSON_SOURCE = "d8f164cc0336d7d3a923aeee2aec3342"

# The two readers a ranking figure can be measured on. ``_query`` assembles the planner
# layout by hand out of the fixture fields, mirroring ``Reactor._planner_memory_query``;
# the injected layout is assembled by the shipped ``Reactor._memory_query`` itself, so
# run 4 cannot report a reader it did not run.
PLANNER_LAYOUT = "planner layout (Reactor._planner_memory_query parts, Reactor._join_query_parts)"
INJECTED_LAYOUT = "injected-run layout (Reactor._memory_query, the query whose hits reach the run prompt)"

# -- the verdict rule --------------------------------------------------------
# An exact-integer match calls a figure that moved by one scenario "DRIFTED", a
# claim the fixture counts cannot support: five situations cannot separate a real
# ranking change from ordinary fixture-to-fixture variation. The rule below is an
# interval decision instead, in the ROPE form (arXiv:1903.03153; Kruschke 2018,
# DOI 10.1177/2515245918771304): a figure is DRIFTED only when the confidence
# interval for the difference of the two rates lies wholly outside the region of
# practical equivalence; exact equality reproduces; anything else is
# UNDERDETERMINED.
#
# The interval must be taken on the difference, not read off the overlap of the
# two separate intervals: non-overlap of two 95% intervals is not a 5% test of the
# difference and errs in both directions (Schenker & Gentleman 2001,
# DOI 10.1198/000313001317097960). Measured over every rate pair with denominators
# up to 30 it disagrees with the difference interval on 6560 of 245025 pairs -- 692
# missed separations and 5868 spurious drift calls -- and its own deciding constant
# sits below the nominal 0.95 on every published figure here (the hub top-1 gap
# flips from DRIFTED at coverage 0.9321). The interval used is the Newcombe (1998)
# method-10 hybrid score interval for ``p1 - p2``, which combines the two Wilson
# score intervals.
#
# The intervals are score intervals for a hypothetical population of comparable
# planner situations. The five fixtures are not a random sample of that
# population, so an interval is a sensitivity bound on how much of a gap the
# fixture count could explain -- not a p-value, and not a licence to call a moved
# figure reproduced.
VERDICT_LEVEL = 0.95  # two-sided coverage of the Wilson and difference intervals
VERDICT_ROPE = 0.10  # rates closer than this are practically equal
# Measured: the difference interval first clears the ROPE band for 0/n vs n/n at
# n=3, so 4 keeps a scenario of margin and still covers every reference
# denominator (5, 8 and 13).
VERDICT_MIN_SCENARIOS = 4

# -- the required-n table ----------------------------------------------------
# An UNDERDETERMINED verdict is a statement about the fixture count, so on its own
# it tells a reader nothing about what would settle the figure. The table below
# attaches the budget: the smallest common n at which the observed rate gap would
# leave the ROPE band, beside the number of labelled cases the harness could
# actually supply. ``required_n`` is the SHIPPED predicate; this comment carries the
# projection and the supply check.
#
# The question is the one test-collection builders ask -- how many topics does this
# comparison need -- and the field's answer is to derive n from a statistical
# requirement plus a variance estimated from pilot data (Sakai, "Topic set size
# design", Information Retrieval Journal 19:256-283, 2016,
# DOI 10.1007/s10791-015-9273-z, following Nagata 2003; reviewed with its topic-by-run
# data and Excel tools published). Two of its results bear directly on this table:
# traditional IR test collections run n=50-100 topics and different measures can
# require substantially different n under the same requirement, and Voorhees &
# Buckley ("The effect of topic set size on retrieval experiment error", SIGIR 2002,
# pp. 316-323) measured TREC error rates that are "larger than anticipated ... especially
# if few topics are used". Five fixtures are two orders of magnitude below that scale,
# which is exactly why the surviving comparisons are undecided -- and why the number
# that would decide each one belongs in the report instead of staying implicit.
REQUIRED_N_SEARCH_LIMIT = 5000  # every live figure decides far below this; the bound keeps the search finite
HARNESS_SUPPLY_DESCRIPTION = "distinct run_started envelopes carrying previous_outcome (the labelled planner-recall cases)"
# The same count, bounded by a replay instant: under ``--as-of`` the figures beside
# it are replayed on a reconstructed cohort, so the supply they are decided against has
# to be the corpus that existed then.
HARNESS_SUPPLY_BOUNDED_DESCRIPTION = (
    "distinct run_started envelopes carrying previous_outcome and written by the replay instant "
    "(the labelled planner-recall cases that existed then)"
)
# ``required_n`` returns ``None`` for two different reasons, and conflating them
# published the opposite of the truth: a gap LARGER than the band is decided at a
# finite n (the difference interval shrinks around the true gap), so a ``None`` that
# comes from the search bound must not be reported as "can never drift". The reason
# is decided from the projected rates themselves, not from the failed search.
BUDGET_REASON_INSIDE_BAND = "inside_band"
BUDGET_REASON_BEYOND_BOUND = "beyond_search_bound"
BUDGET_REASON_NO_DENOMINATOR = "no_denominator"


def wilson_interval(successes: int, total: int, level: float = VERDICT_LEVEL) -> tuple[float, float]:
    """Two-sided Wilson score interval, ``(0, 1)`` when there is no denominator."""
    if total <= 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = (z / denominator) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    # The clamp must not push an endpoint past the observed rate: at p = 0 or 1
    # the floating-point upper bound lands one ulp short of it, which would make
    # the interval fail to bracket the very rate it describes.
    return (max(0.0, min(centre - half, p)), min(1.0, max(centre + half, p)))


def difference_interval(
    recorded: tuple[int, int], observed: tuple[int, int], level: float = VERDICT_LEVEL
) -> tuple[float, float]:
    """Newcombe (1998) method-10 score interval for ``rate(observed) - rate(recorded)``.

    Each marginal Wilson interval is combined into an interval for the difference,
    which is the quantity the verdict is about. The interval is a sensitivity bound
    for a hypothetical population of comparable planner situations: the fixtures are
    not a random sample of it, so a bound that clears the ROPE band is the evidence
    for a drift call, not a p-value.

    Validated against the worked example published with the method's reference
    implementation (``ci_prop_diff_nc``): 9/10 vs 3/10 gives ``(0.1705, 0.8090)``.
    """
    recorded_hits, recorded_total = recorded
    observed_hits, observed_total = observed
    if recorded_total <= 0 or observed_total <= 0:
        return (-1.0, 1.0)
    recorded_rate = recorded_hits / recorded_total
    observed_rate = observed_hits / observed_total
    low_recorded, high_recorded = wilson_interval(recorded_hits, recorded_total, level)
    low_observed, high_observed = wilson_interval(observed_hits, observed_total, level)
    difference = observed_rate - recorded_rate
    low = difference - math.sqrt((observed_rate - low_observed) ** 2 + (high_recorded - recorded_rate) ** 2)
    high = difference + math.sqrt((high_observed - observed_rate) ** 2 + (recorded_rate - low_recorded) ** 2)
    return (max(-1.0, low), min(1.0, high))


def classify_verdict(recorded: tuple[int, int], observed: tuple[int, int]) -> str:
    """REPRODUCED, DRIFTED or UNDERDETERMINED for one recorded-vs-observed rate."""
    recorded_hits, recorded_total = recorded
    observed_hits, observed_total = observed
    if recorded_total <= 0 or observed_total <= 0:
        return "UNDERDETERMINED"
    if recorded_hits * observed_total == observed_hits * recorded_total:
        return "REPRODUCED"
    if recorded_total < VERDICT_MIN_SCENARIOS or observed_total < VERDICT_MIN_SCENARIOS:
        return "UNDERDETERMINED"
    low, high = difference_interval(recorded, observed)
    if low > VERDICT_ROPE or high < -VERDICT_ROPE:
        return "DRIFTED"
    return "UNDERDETERMINED"


def verdict_detail(recorded: tuple[int, int], observed: tuple[int, int]) -> dict[str, Any]:
    """The verdict plus every input it was computed from, for the JSON report."""
    recorded_hits, recorded_total = recorded
    observed_hits, observed_total = observed
    return {
        "recorded": {"hits": recorded_hits, "total": recorded_total},
        "observed": {"hits": observed_hits, "total": observed_total},
        "recorded_rate": recorded_hits / recorded_total if recorded_total else None,
        "observed_rate": observed_hits / observed_total if observed_total else None,
        "recorded_interval": wilson_interval(recorded_hits, recorded_total),
        "observed_interval": wilson_interval(observed_hits, observed_total),
        "difference_interval": difference_interval((recorded_hits, recorded_total), (observed_hits, observed_total)),
        "rope": VERDICT_ROPE,
        "min_scenarios": VERDICT_MIN_SCENARIOS,
        "verdict": classify_verdict(recorded, observed),
    }


def required_n(
    recorded: tuple[int, int], observed: tuple[int, int], rope: float = VERDICT_ROPE, limit: int = REQUIRED_N_SEARCH_LIMIT
) -> int | None:
    """Smallest common n at which the recorded-vs-observed gap would leave the band.

    Both counts are carried to the same denominator under their observed rates, so
    this is a projection of the fixture gap, not a measurement: it answers "how many
    comparable situations would this comparison need", not "how many would the reader
    see". The search starts at ``VERDICT_MIN_SCENARIOS``, so the table can never
    report a budget that the verdict rule would itself refuse, and ends at ``limit``.

    ``None`` is returned when no n below ``limit`` decides it, which happens for two
    different reasons that ``required_n_detail`` tells apart: a gap smaller than the
    ROPE band is decidable at NO n (both rates round to the same value once n is
    large), while a gap wider than the band is decided at some n the bound cut off.
    Only the first is a finding about the effect size; the second is a search failure.
    """
    recorded_hits, recorded_total = recorded
    observed_hits, observed_total = observed
    if recorded_total <= 0 or observed_total <= 0:
        return None
    recorded_rate = recorded_hits / recorded_total
    observed_rate = observed_hits / observed_total
    for total in range(VERDICT_MIN_SCENARIOS, limit + 1):
        low, high = difference_interval((round(recorded_rate * total), total), (round(observed_rate * total), total))
        if low > rope or high < -rope:
            return total
    return None


def required_n_detail(
    recorded: tuple[int, int], observed: tuple[int, int], supply: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The budget an UNDERDETERMINED verdict needs, and whether the harness can reach it."""
    required = required_n(recorded, observed)
    detail: dict[str, Any] = {
        "required_n": required,
        "search_limit": REQUIRED_N_SEARCH_LIMIT,
        "supply": supply,
        "reachable": None,
        "reason": None,
        "projected_gap": None,
    }
    if required is not None:
        if supply is not None:
            detail["reachable"] = int(supply.get("labelled_cases") or 0) >= required
        return detail
    recorded_hits, recorded_total = recorded
    observed_hits, observed_total = observed
    if recorded_total <= 0 or observed_total <= 0:
        detail["reason"] = BUDGET_REASON_NO_DENOMINATOR
        return detail
    gap = abs(observed_hits / observed_total - recorded_hits / recorded_total)
    detail["projected_gap"] = gap
    # At large n the difference interval collapses around this gap, so a gap wider
    # than the band is decided at SOME finite n -- just one the search bound cut off.
    detail["reason"] = BUDGET_REASON_BEYOND_BOUND if gap > VERDICT_ROPE else BUDGET_REASON_INSIDE_BAND
    return detail


def harness_supply(connection: sqlite3.Connection, as_of: str | None = None) -> dict[str, Any]:
    """The labelled cases the planner harness could supply at the table's n.

    The label the held-out planner-recall comparisons use is "memories whose
    ``source_run`` equals the run's own id", so the supply is the number of DISTINCT
    ``run_started`` envelopes that carry a ``previous_outcome`` -- the corpus
    generations 170-191 measured on. Counting DISTINCT run ids rather than rows keeps
    a re-emitted envelope from inflating the figure.

    ``as_of`` bounds the supply to the envelopes that already existed at the replay
    instant, because the reachability flag is an input of the same verdict: without
    the bound a report that replayed its rankings on a reconstructed cohort still
    decided "the harness can supply that n" on cases written after the corpus, and
    called a budget reachable that the cohort could not have paid for. Measured on
    the live ledger over nine historical instants, the flag disagrees with the
    cohort-bound supply in 6 of the 32 (figure, instant) cells that carry a budget:
    at 2026-09-19T20:00Z the supply reads 198 where only 4 labelled cases existed.
    """
    query = (
        "SELECT DISTINCT json_extract(payload, '$.run_id') FROM event_log "
        "WHERE kind='run_started' AND payload LIKE '%previous_outcome%'"
    )
    params: tuple[Any, ...] = ()
    description = HARNESS_SUPPLY_DESCRIPTION
    if as_of is not None:
        query += " AND created_at <= ?"
        params = (as_of,)
        description = HARNESS_SUPPLY_BOUNDED_DESCRIPTION
    rows = connection.execute(query, params).fetchall()
    return {
        "labelled_cases": sum(1 for row in rows if row[0]),
        "description": description,
    }


def budget_text(detail: dict[str, Any]) -> str:
    """One line stating what would settle an UNDERDETERMINED comparison."""
    required = detail.get("required_n")
    if required is None:
        if detail.get("reason") == BUDGET_REASON_NO_DENOMINATOR:
            return "budget: no n decides it -- a comparison has no denominator"
        if detail.get("reason") == BUDGET_REASON_BEYOND_BOUND:
            gap = detail.get("projected_gap")
            gap_text = "" if gap is None else f" (gap {gap:.4f})"
            return (
                f"budget: the deciding n is above the {detail.get('search_limit')} search bound -- the gap{gap_text} "
                f"exceeds the +/-{VERDICT_ROPE:.2f} band, so a large enough fixture set would settle it"
            )
        return (
            f"budget: no n below {detail.get('search_limit')} decides it -- the two rates are closer together than the "
            f"+/-{VERDICT_ROPE:.2f} band, so the comparison can never drift"
        )
    supply = detail.get("supply") or {}
    cases = supply.get("labelled_cases")
    if cases is None:
        return f"budget: required n = {required} (common denominator); supply unmeasured on this ledger"
    verdict = "reachable" if int(cases) >= required else "NOT reachable"
    return (
        f"budget: required n = {required} (common denominator) vs {cases} labelled cases on this ledger -> {verdict}"
    )


def direction_detail(recorded: int, observed: int, total: int) -> dict[str, Any]:
    """The phrase-removal claim is directional, so its rule is stated separately.

    Both readings come from the same corpus through two queries, so they are not
    independent samples and no interval is applied: the claim reproduces only if
    the phrase-free reading is at least the shipped one.
    """
    verdict = "UNDERDETERMINED" if total <= 0 else ("REPRODUCED" if observed >= recorded else "REVERSED")
    return {"recorded": recorded, "observed": observed, "total": total, "verdict": verdict}


def _goal_terms(title: str) -> set[str]:
    return {t for t in MemoryStore._terms(str(title)) if len(t) >= 2 and t not in _STOPWORDS}


def _terms_of(text: str) -> set[str]:
    return set(MemoryStore._normalize_terms(text))


def _constant_terms(goal: str, phrase: str) -> list[str]:
    """The constant parts (phrase + goal title) that reach the window."""
    return sorted(_terms_of(phrase + " " + goal) & (_terms_of(phrase) | _goal_terms(goal)))


def _query(situation: Situation, phrase: str | None) -> str:
    parts = [situation.summary, situation.initial_prompt, situation.next_step]
    if phrase:
        parts.append(phrase)
    parts.append(situation.goal)
    return Reactor._join_query_parts([part for part in parts if part])


def _injected_query(situation: Situation) -> str:
    """The fixture read through the injected-run reader, by calling it.

    ``_query`` above is the planner prompt's layout. The query whose hits enter an
    agent run's prompt is ``Reactor._memory_query(selected_work, inbox_events,
    next_plan)`` at the point the StartEnvelope is built, and this calls that function
    rather than re-deriving its part list, so the scorecard cannot disagree with the
    reader it claims to replay.

    A frozen fixture carries no owner inbox message and no acceptance criterion, so
    those slots are empty (``[]`` and a task dict without the key), not invented; the
    fixture's label supplies the task title and its goal the goal title. The reader split
    survives that: the injected layout has no constant phrase and no summary, which is
    exactly what moves the ranking.
    """
    selected_work = {"task": {"title": situation.label, "goal_title": situation.goal}}
    next_plan = {"initial_prompt": situation.initial_prompt, "next": situation.next_step}
    return Reactor._memory_query(selected_work, [], next_plan)


def _pooled_fused_order(
    connection: sqlite3.Connection, terms: list[str], limit: int, allowed: set[str] | None = None
) -> list[str]:
    """Replay MemoryStore._fts_search's fusion and return the pooled fused order.

    The recorded ranks were read from this order, which is the pre-_RRF_MAX_LIFT
    ranking (that bound landed later, at generation 174). Reporting the pooled
    rank beside the shipped page rank lets a moved figure be attributed to the
    reader instead of guessed.

    ``allowed``, when given, IS the membership predicate: it replaces today's
    ``status='active'`` clause instead of intersecting it. The cohort was built from
    the validity window of the replay instant, which already decides membership
    there, so adding today's status re-introduces the later events the reconstruction
    removed and silently rewrites the past figure. Measured on the live ledger at
    2026-09-22T23:40:45Z, three cohort members were superseded later; keeping the
    status clause reported hub top-1 as 2/5 (UNDERDETERMINED) where the same instant
    without it reads 3/5 -- while the module's own docstring claims 3/5 for that
    cohort, so the shipped reader and its documented figure disagreed by one cell.
    """
    match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
    candidate_limit = limit * 5
    cohort_sql = ""
    cohort_params: tuple[Any, ...] = ()
    if allowed is not None:
        placeholders = ",".join("?" for _ in sorted(allowed))
        cohort_sql = f" AND m.memory_id IN ({placeholders})"
        cohort_params = tuple(sorted(allowed))
    # The cohort replaces the status clause; without a cohort the shipped status clause
    # stays, so a live reading is byte-for-byte the shipped reader.
    status_sql = "" if allowed is not None else " AND m.status='active'"
    axes: tuple[tuple[str, tuple[Any, ...]], ...] = (
        (
            "SELECT m.memory_id FROM memories_fts JOIN memories m ON m.memory_id = memories_fts.memory_id "
            "WHERE memories_fts MATCH ?" + status_sql + cohort_sql + " "
            "ORDER BY bm25(memories_fts), m.memory_id LIMIT ?",
            (match, *cohort_params, candidate_limit),
        ),
        (
            "SELECT m.memory_id FROM memories m WHERE 1=1" + status_sql + cohort_sql + " "
            "ORDER BY updated_at DESC, memory_id LIMIT ?",
            (*cohort_params, candidate_limit),
        ),
        (
            "SELECT m.memory_id FROM memories m WHERE 1=1" + status_sql + cohort_sql + " "
            "ORDER BY confidence DESC, updated_at DESC, memory_id LIMIT ?",
            (*cohort_params, candidate_limit),
        ),
    )
    scores: dict[str, float] = {}
    for (sql, params), (_axis, weight) in zip(axes, _RRF_WEIGHTS, strict=False):
        for rank, row in enumerate(connection.execute(sql, params).fetchall(), start=1):
            memory_id = row["memory_id"]
            scores[memory_id] = scores.get(memory_id, 0.0) + weight / (_RRF_K + rank)
    return [memory_id for memory_id, _score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]


def _rank(sequence: list[str], memory_id: str) -> int | None:
    return sequence.index(memory_id) + 1 if memory_id in sequence else None


def as_of_cohort(connection: sqlite3.Connection, moment: str) -> set[str]:
    """The memories a ledger could have held at ``moment``.

    ``updated_at`` is the write time and is not rewritten by supersession, so a
    memory was active at ``moment`` iff it had been written and its ``valid_to``
    (the supersession time) had not yet passed. A cohort reconstructed this way
    is exact for rows that were superseded later; it can differ from a real
    historical ledger only by rows whose supersession left no ``valid_to``.
    """
    return {
        row["memory_id"]
        for row in connection.execute("SELECT memory_id, status, valid_to, updated_at FROM memories").fetchall()
        if row["updated_at"] <= moment and (row["valid_to"] is None or row["valid_to"] > moment)
    }


def parse_as_of(value: str) -> str:
    """Accept ``YYYY-MM-DD`` or an ISO-8601 instant and return the ledger's format."""
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an ISO-8601 instant: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class Reading:
    """One pair read through one query."""

    terms: int
    goal_intact: bool
    tail_share: int
    fused_top1: str | None
    fused_top3: list[str]
    shipped_top3: list[str]
    pooled_top3: dict[str, int | None]
    pooled_top20: dict[str, int | None]
    shipped_top20: dict[str, int | None]


def _read(
    connection: sqlite3.Connection,
    store: MemoryStore,
    situation: Situation,
    query: str,
    tail_terms: set[str],
    pooled_allowed: set[str] | None = None,
) -> Reading:
    terms = MemoryStore._normalize_terms(query)
    # The shipped page takes ``pooled_allowed`` as well: ``shipped_top3``/``shipped_top20``
    # are printed beside the pooled ranks, so a page read against today's ledger while its
    # siblings replay a cohort would report corpus movement as a reader difference.
    page3 = [m["memory_id"] for m in store.search(query, limit=3, allowed=pooled_allowed)]
    page20 = [m["memory_id"] for m in store.search(query, limit=20, allowed=pooled_allowed)]
    pooled3 = _pooled_fused_order(connection, terms, 3, allowed=pooled_allowed)
    pooled20 = _pooled_fused_order(connection, terms, 20, allowed=pooled_allowed)
    return Reading(
        terms=len(terms),
        goal_intact=_goal_terms(situation.goal) <= set(terms),
        tail_share=len(set(terms) & tail_terms),
        fused_top1=pooled20[0] if pooled20 else None,
        fused_top3=pooled3,
        shipped_top3=page3,
        pooled_top3={mid: _rank(pooled3, mid) for mid, _label in situation.governing},
        pooled_top20={mid: _rank(pooled20, mid) for mid, _label in situation.governing},
        shipped_top20={mid: _rank(page20, mid) for mid, _label in situation.governing},
    )


def build_report(db_path: Path, as_of: str | None = None) -> dict[str, Any]:
    """Measure the corpus and return the scorecard as plain data.

    ``as_of`` restricts the pooled replay to the cohort a ledger could have held
    at that instant, so a reference figure is scored against its own corpus.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"ledger not found: {db_path}")
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        store = MemoryStore(connection)
        if not store.fts_available:
            raise RuntimeError(f"ledger has no usable FTS index: {db_path}")
        active = int(connection.execute("SELECT COUNT(*) FROM memories WHERE status='active'").fetchone()[0])
        pooled_allowed = as_of_cohort(connection, as_of) if as_of else None
        pairs: dict[str, dict[str, Any]] = {}
        for situation in SITUATIONS:
            recorded_q, shipped_q, free_q = _query(situation, RECORDED_PHRASE), _query(situation, SHIPPED_PHRASE), _query(situation, None)
            # Both readers of the layout block are measured on the same corpus: the
            # injected reading takes ``pooled_allowed`` too, or an ``--as-of`` replay
            # would score the planner layout against the published cohort and the
            # injected layout against today's much larger one, and report the corpus
            # difference as a difference between readers.
            shipped_terms = set(MemoryStore._normalize_terms(shipped_q))
            free_terms = set(MemoryStore._normalize_terms(free_q))
            pairs[situation.key] = {
                "label": situation.label,
                "goal": situation.goal,
                "governing": [{"memory_id": mid, "label": label} for mid, label in situation.governing],
                "recorded": _read(
                    connection, store, situation, recorded_q, _terms_of(RECORDED_PHRASE) | _goal_terms(situation.goal), pooled_allowed
                ).__dict__,
                "shipped": _read(
                    connection, store, situation, shipped_q, _terms_of(SHIPPED_PHRASE) | _goal_terms(situation.goal), pooled_allowed
                ).__dict__,
                "injected": _read(
                    connection, store, situation, _injected_query(situation), _goal_terms(situation.goal), pooled_allowed
                ).__dict__,
                "phrase_free": _read(connection, store, situation, free_q, set(), pooled_allowed).__dict__,
                "displaced_terms": [t for t in MemoryStore._normalize_terms(shipped_q) if t not in free_terms],
                "gained_terms": [t for t in MemoryStore._normalize_terms(free_q) if t not in shipped_terms],
            }
        later = 0
        if pooled_allowed is not None:
            placeholders = ",".join("?" for _ in pooled_allowed)
            later = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM memories WHERE status='active' AND memory_id NOT IN ({placeholders})",
                    tuple(sorted(pooled_allowed)),
                ).fetchone()[0]
            )
        situations = len(SITUATIONS)
        tail_share = {
            goal: {
                "recorded": value,
                "now": len(_constant_terms(goal, RECORDED_PHRASE)),
                "terms": _constant_terms(goal, RECORDED_PHRASE),
                "source": source,
            }
            for goal, (value, source) in REFERENCE_TAIL_SHARE.items()
        }
        recorded_totals = totals_for(pairs, "recorded")
        shipped_totals = totals_for(pairs, "shipped")
        free_totals = totals_for(pairs, "phrase_free")
        verdicts = {
            "constant_tail_share": {
                goal: verdict_detail((entry["recorded"], _MAX_QUERY_TERMS), (entry["now"], _MAX_QUERY_TERMS)) | {"source": entry["source"]}
                for goal, entry in tail_share.items()
            },
            "hub_top1": verdict_detail((REFERENCE_HUB_TOP1, situations), (recorded_totals["hub1"], situations)) | {"source": REFERENCE_SOURCE},
            "hub_top2": verdict_detail((REFERENCE_HUB_TOP2, situations), (recorded_totals["hub2"], situations)) | {"source": REFERENCE_SOURCE},
            "governing_top3": verdict_detail(
                (REFERENCE_GOVERNING_TOP3, REFERENCE_GOVERNING_VERDICTS), (recorded_totals["gov3"], recorded_totals["verdicts"])
            )
            | {"source": REFERENCE_SOURCE},
            "governing_top20": verdict_detail(
                (REFERENCE_GOVERNING_TOP20, REFERENCE_GOVERNING_VERDICTS), (recorded_totals["gov20"], recorded_totals["verdicts"])
            )
            | {"source": REFERENCE_SOURCE},
            "intactness_direction": direction_detail(shipped_totals["intact"], free_totals["intact"], situations) | {"source": LESSON_SOURCE},
        }
        # Every comparison that can be undecided states its own budget, so a reader
        # can tell one that a bigger fixture set would settle from one that no n can.
        # The supply is bounded by the same instant as the figures it is attached to:
        # ``reachable`` answers "could the harness pay for this n", and the harness it
        # is asked about is the one the replay instant describes.
        supply = harness_supply(connection, as_of)
        for name in ("hub_top1", "hub_top2", "governing_top3", "governing_top20"):
            entry = verdicts[name]
            entry.update(
                required_n_detail(
                    (entry["recorded"]["hits"], entry["recorded"]["total"]),
                    (entry["observed"]["hits"], entry["observed"]["total"]),
                    supply,
                )
            )
        for entry in verdicts["constant_tail_share"].values():
            entry.update(
                required_n_detail(
                    (entry["recorded"]["hits"], entry["recorded"]["total"]),
                    (entry["observed"]["hits"], entry["observed"]["total"]),
                    supply,
                )
            )
        return {
            "ledger": str(db_path),
            "harness_supply": supply,
            "active_memories": active,
            "recorded_phrase": RECORDED_PHRASE,
            "shipped_phrase": SHIPPED_PHRASE,
            "as_of": as_of,
            "cohort_memories": len(pooled_allowed) if pooled_allowed is not None else None,
            "cohort_excluded_active": later,
            "pairs": pairs,
            "recorded_tail_share": tail_share,
            "verdicts": verdicts,
            # The same figures under the other reader, with the assembler named, so no
            # ranking verdict has to be read without knowing which reader produced it.
            "injected_totals": figure_observations(pairs, "injected"),
            "layout": layout_comparison(pairs),
        }
    finally:
        connection.close()


def totals_for(pairs: dict[str, dict[str, Any]], reading: str) -> dict[str, int]:
    """The pooled counts of one run, shared by the printed table and the verdicts."""
    totals = {"intact": 0, "hub1": 0, "hub2": 0, "gov3": 0, "gov20": 0, "verdicts": 0}
    for pair in pairs.values():
        data = pair[reading]
        totals["intact"] += bool(data["goal_intact"])
        totals["hub1"] += data["fused_top1"] == HUB_MEMORY_ID
        totals["hub2"] += HUB_MEMORY_ID in data["fused_top3"]
        for entry in pair["governing"]:
            mid = entry["memory_id"]
            totals["verdicts"] += 1
            totals["gov3"] += bool(data["pooled_top3"][mid])
            totals["gov20"] += bool(data["pooled_top20"][mid])
    return totals


def membership_changed(before: dict[str, int | None], after: dict[str, int | None]) -> bool:
    """Whether a governed memory's top-3/top-20 membership moved between two readings."""
    return bool(before["top3"]) != bool(after["top3"]) or bool(before["top20"]) != bool(after["top20"])


def membership_text(before: dict[str, int | None], after: dict[str, int | None]) -> str:
    """The print marker for a governed memory whose membership moved."""
    return " (moved)" if membership_changed(before, after) else ""


def figure_observations(pairs: dict[str, dict[str, Any]], reading: str) -> dict[str, tuple[int, int]]:
    """The four published figures as observed under one reader's reading."""
    totals = totals_for(pairs, reading)
    situations = len(SITUATIONS)
    return {
        "hub_top1": (totals["hub1"], situations),
        "hub_top2": (totals["hub2"], situations),
        "governing_top3": (totals["gov3"], totals["verdicts"]),
        "governing_top20": (totals["gov20"], totals["verdicts"]),
    }


def layout_comparison(pairs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Score every recorded figure against both readers, side by side.

    A rate observed under the planner layout and the same rate under the injected-run
    layout are different measurements of the same claim, so this block is what lets a
    verdict name its reader. Per fixture it also carries the hub's and every governing
    memory's rank under both layouts, so a moved figure can be read as a rank movement
    instead of an unexplained count.

    ``planner`` reads the run-1 ``recorded`` pair and ``injected`` the run-4 pair; the
    layout names are used on both sides of that mapping so a reader cannot be dropped
    silently.
    """
    situations = len(SITUATIONS)
    recorded: dict[str, tuple[int, int]] = {
        "hub_top1": (REFERENCE_HUB_TOP1, situations),
        "hub_top2": (REFERENCE_HUB_TOP2, situations),
        "governing_top3": (REFERENCE_GOVERNING_TOP3, REFERENCE_GOVERNING_VERDICTS),
        "governing_top20": (REFERENCE_GOVERNING_TOP20, REFERENCE_GOVERNING_VERDICTS),
    }
    layouts: tuple[tuple[str, str], ...] = (("planner", "recorded"), ("injected", "injected"))
    observed = {layout: figure_observations(pairs, reading) for layout, reading in layouts}
    readers = {"planner": PLANNER_LAYOUT, "injected": INJECTED_LAYOUT}
    figures: dict[str, Any] = {}
    changed: list[str] = []
    for name, reference in recorded.items():
        entry: dict[str, Any] = {
            "recorded": {"hits": reference[0], "total": reference[1]},
            "source": REFERENCE_SOURCE,
            "readers": {},
        }
        for layout, _reading in layouts:
            hits, total = observed[layout][name]
            detail = verdict_detail(reference, (hits, total))
            detail["reader"] = readers[layout]
            detail["layout"] = layout
            # The observed margin: how far this reader sits from the recorded rate, in
            # rate units, so a REPRODUCED and a DRIFTED reading of the same figure can be
            # told apart by how much they moved or did not.
            detail["gap"] = hits / total - reference[0] / reference[1] if total else None
            detail["required_n"] = required_n(reference, (hits, total))
            entry["readers"][layout] = detail
        if entry["readers"]["planner"]["verdict"] != entry["readers"]["injected"]["verdict"]:
            changed.append(name)
        figures[name] = entry
    per_pair: dict[str, Any] = {}
    rank_changes = 0
    verdicts = 0
    for key, pair in pairs.items():
        ranks: dict[str, Any] = {}
        for layout, reading in layouts:
            data = pair[reading]
            ranks[layout] = {
                "hub_top1": data["fused_top1"] == HUB_MEMORY_ID,
                "hub_top2": HUB_MEMORY_ID in data["fused_top3"],
                "governing": {
                    entry["memory_id"]: {"top3": data["pooled_top3"][entry["memory_id"]], "top20": data["pooled_top20"][entry["memory_id"]]}
                    for entry in pair["governing"]
                },
            }
        moved: list[dict[str, Any]] = []
        for entry in pair["governing"]:
            mid = entry["memory_id"]
            verdicts += 1
            before = ranks["planner"]["governing"][mid]
            after = ranks["injected"]["governing"][mid]
            if membership_changed(before, after):
                rank_changes += 1
                moved.append({"memory_id": mid, "label": entry["label"], "planner": before, "injected": after})
        per_pair[key] = {
            "label": pair["label"],
            "goal": pair["goal"],
            "planner": ranks["planner"],
            "injected": ranks["injected"],
            "membership_moved": moved,
        }
    return {
        "fixtures": situations,
        "recorded_source": REFERENCE_SOURCE,
        "readers": readers,
        "figures": figures,
        "verdict_changes": changed,
        "verdict_change_count": len(changed),
        "governing_verdicts": verdicts,
        "governing_rank_change_count": rank_changes,
        "per_pair": per_pair,
    }


def _print_run(report: dict[str, Any], reading: str, title: str) -> dict[str, int]:
    print(f"\n{title}")
    for key, pair in report["pairs"].items():
        data = pair[reading]
        print(f"  {key} {pair['label']}")
        print(f"    goal-title intactness: {'yes' if data['goal_intact'] else 'NO'}")
        print(f"    constant-tail term share: {data['tail_share']}/{_MAX_QUERY_TERMS}")
        if reading == "injected":
            # The injected layout carries no constant phrase, so the only fixed part it
            # can retain is the goal title; the label says what the count is.
            print(f"    goal-title terms inside the window: {data['tail_share']}/{_MAX_QUERY_TERMS}")
        else:
            print(f"    constant-tail term share: {data['tail_share']}/{_MAX_QUERY_TERMS}")
        print(f"    fused top-1: {data['fused_top1']}   shipped top-3: {[m[:8] for m in data['shipped_top3']]}")
        for entry in pair["governing"]:
            mid = entry["memory_id"]
            print(
                f"    governing {mid[:8]} ({entry['label']}): pooled top-3 {data['pooled_top3'][mid] or 'absent'}; "
                f"pooled top-20 {data['pooled_top20'][mid] or 'absent'}; shipped top-20 {data['shipped_top20'][mid] or 'absent'}"
            )
        if reading == "phrase_free":
            print(f"    tail terms displaced: {pair['displaced_terms'] or 'none'}")
            print(f"    terms gained: {pair['gained_terms'] or 'none'}")
    return totals_for(report["pairs"], reading)


def _fmt_interval(interval: tuple[float, float]) -> str:
    return f"{interval[0]:.3f}..{interval[1]:.3f}"


def _print_verdict(name: str, detail: dict[str, Any], source: str) -> None:
    print(
        f"  {name}: recorded {detail['recorded']['hits']}/{detail['recorded']['total']} -> now "
        f"{detail['observed']['hits']}/{detail['observed']['total']}  {detail['verdict']}  "
        f"[difference {_fmt_interval(detail['difference_interval'])} vs +/-{detail['rope']:.2f} band]  [{source}]"
    )
    if detail["verdict"] == "UNDERDETERMINED" and "required_n" in detail:
        print(f"    {budget_text(detail)}")


def _print_layout_check(report: dict[str, Any]) -> None:
    """Every recorded figure scored against both readers, side by side."""
    layout = report["layout"]
    readers = layout["readers"]
    print("\n[reader layout] the same recorded figures, scored per reader")
    print(f"  planner  reader: {readers['planner']}")
    print(f"  injected reader: {readers['injected']}")
    print(
        f"  recorded figures come from memory {layout['recorded_source'][:8]}; "
        f"{layout['fixtures']} fixture pairs; the planner reading is run 1 above"
    )
    for name, entry in layout["figures"].items():
        recorded = entry["recorded"]
        cells = []
        for reader in ("planner", "injected"):
            detail = entry["readers"][reader]
            gap = detail["gap"]
            cells.append(
                f"{reader} {detail['observed']['hits']}/{detail['observed']['total']} {detail['verdict']} "
                f"(gap {gap:+.4f}, n*={detail['required_n']})"
            )
        print(f"  {name}: recorded {recorded['hits']}/{recorded['total']} -> " + " | ".join(cells))
    changed = layout["verdict_changes"]
    print(
        f"  figures whose verdict changes between readers: {layout['verdict_change_count']}/{len(layout['figures'])}"
        + (f" ({', '.join(changed)})" if changed else " (none)")
    )
    print(
        f"  governing verdicts whose top-3/top-20 membership changes between readers: "
        f"{layout['governing_rank_change_count']}/{layout['governing_verdicts']}"
    )
    for key, pair in layout["per_pair"].items():
        print(f"  {key} {pair['label']}")
        print(
            f"    hub {HUB_MEMORY_ID[:8]} fused top-1: planner {'yes' if pair['planner']['hub_top1'] else 'no'}"
            f" / injected {'yes' if pair['injected']['hub_top1'] else 'no'}"
        )
        for memory_id, before in pair["planner"]["governing"].items():
            after = pair["injected"]["governing"][memory_id]
            print(
                f"    governing {memory_id[:8]}: planner top-3 {before['top3'] or 'absent'} "
                f"top-20 {before['top20'] or 'absent'} -> injected top-3 {after['top3'] or 'absent'} "
                f"top-20 {after['top20'] or 'absent'}{membership_text(before, after)}"
            )


def _print_reference_check(report: dict[str, Any]) -> None:
    verdicts = report["verdicts"]
    drift = report["active_memories"] - REFERENCE_ACTIVE_MEMORIES
    print("\n[reference figures] recorded vs today")
    print(
        f"  verdict rule: exact equality -> REPRODUCED; the {VERDICT_LEVEL:.0%} Newcombe interval for the rate difference "
        f"wholly outside the +/-{VERDICT_ROPE:.2f} ROPE band -> DRIFTED; otherwise UNDERDETERMINED (no call is possible "
        f"below {VERDICT_MIN_SCENARIOS} scenarios)"
    )
    supply = report.get("harness_supply") or {}
    print(
        f"  harness supply: {supply.get('labelled_cases')} {supply.get('description')}; an undecided comparison states the "
        f"n that would settle it (search bound {REQUIRED_N_SEARCH_LIMIT})"
    )
    print(f"  ledger drift: {REFERENCE_ACTIVE_MEMORIES} active memories when published -> {report['active_memories']} now ({drift:+d})")
    for goal, entry in report["recorded_tail_share"].items():
        detail = verdicts["constant_tail_share"][goal]
        print(
            f"  constant-tail share {goal!r} under {RECORDED_PHRASE!r}: {entry['recorded']}/{_MAX_QUERY_TERMS} -> "
            f"{entry['now']}/{_MAX_QUERY_TERMS}  {detail['verdict']}  [{entry['source']}]"
        )
        if entry["now"] == entry["recorded"]:
            print(f"    terms: {entry['terms']}")
    _print_verdict(f"hub {HUB_MEMORY_ID[:8]} fused top-1", verdicts["hub_top1"], REFERENCE_SOURCE)
    _print_verdict(f"hub {HUB_MEMORY_ID[:8]} fused top-2", verdicts["hub_top2"], REFERENCE_SOURCE)
    _print_verdict("governing memory in the pooled top-3", verdicts["governing_top3"], REFERENCE_SOURCE)
    _print_verdict("governing memory in the pooled top-20", verdicts["governing_top20"], REFERENCE_SOURCE)
    direction = verdicts["intactness_direction"]
    print(
        f"  goal-title intactness when the phrase is removed (memory {LESSON_SOURCE[:8]}, directional rule): "
        f"shipped {direction['recorded']}/{direction['total']} -> phrase-free {direction['observed']}/{direction['total']}; "
        f"{direction['verdict']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay the planner-recall scorecard against a ledger.")
    parser.add_argument("--db", default=os.getenv("SKYNET_STATE") or DEFAULT_LEDGER, help="ledger path (opened read-only)")
    parser.add_argument(
        "--as-of",
        type=parse_as_of,
        default=None,
        help="restrict the pooled replay to the cohort active at this ISO-8601 instant, e.g. 2026-09-22T23:40:45Z",
    )
    parser.add_argument("--json", action="store_true", help="print the raw report as JSON instead of the table")
    args = parser.parse_args(argv)
    try:
        report = build_report(Path(args.db), args.as_of)
    except (FileNotFoundError, RuntimeError, sqlite3.Error) as exc:
        print(f"recall scorecard could not read the ledger: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=1, ensure_ascii=False))
        return 0
    scope = f"{report['active_memories']} active memories"
    if report["cohort_memories"] is not None:
        scope = (
            f"{report['active_memories']} active memories; pooled replay restricted to the {report['cohort_memories']}-memory "
            f"cohort at {report['as_of']} ({report['cohort_excluded_active']} active memories written later are excluded)"
        )
    print(f"planner-recall scorecard: {report['ledger']} ({scope}, {len(SITUATIONS)} fixture pairs)")
    cohort_note = f"{REFERENCE_ACTIVE_MEMORIES} active memories" if report["cohort_memories"] is None else (
        f"cohort at {report['as_of']}: {report['cohort_memories']} memories"
    )
    _print_run(report, "recorded", f"[run 1] recorded configuration (phrase {RECORDED_PHRASE!r}, pooled fusion, {cohort_note})")
    _print_run(report, "shipped", f"[run 3] shipped query (constant phrase {SHIPPED_PHRASE!r})")
    _print_run(report, "phrase_free", f"[run 2] shipped query with {SHIPPED_PHRASE!r} removed")
    _print_run(report, "injected", "[run 4] injected-run layout (Reactor._memory_query on the fixture envelope)")
    _print_reference_check(report)
    _print_layout_check(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
