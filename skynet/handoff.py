"""Seed self-directed work: a scaffolding goal, its tasks and canonical memories.

The organism can plan, propose and promote, but nothing told it what the project
needs. Without a seed its only goal is the genesis bootstrap task, so
"autonomous goal setting" stayed declarative. This module gives the harness a
reproducible handoff: one scaffolding goal, a bounded task per genuinely open
item, and pinned memories that survive every retrieval.

The task list is curated here, one entry per item that is still open, and each
task carries its own acceptance criterion. Some tasks require external
evidence, so an acceptance may name a citation rather than a code change. The
seed is scaffolding, not an order: the organism may outrank it with a goal of
its own. The handoff is idempotent, so re-running it only adds what is new.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .planner import hypothesis_fingerprint, structural_fingerprint
from .time import utc_now

ROADMAP_GOAL_TITLE = "Advance the SkyNet roadmap"
ROADMAP_AREA = "roadmap"
# Scaffolding, not an order: the seed exists so the organism is not idle, but it
# must be able to outrank it with a goal of its own. The autonomous planner's
# self-set proposals score around 0.5, so a seed near 1.0 wins only a tie and
# can never structurally dominate the organism's own choices.
ROADMAP_GOAL_PRIORITY = 1.0

# One entry per genuinely open item: (title, expected_new_fact). Each task is
# self-describing; the fact is its acceptance criterion, so a task cannot be
# closed on a report alone. Closed and dropped work is deliberately absent.
ROADMAP_TASKS: tuple[tuple[str, str], ...] = (
    (
        "Resolve the proposals blocked by the environment",
        "No proposal is left in blocked_by_environment: each is retried once its toolchain is present or rejected with a recorded reason. Acceptance is the registry showing zero blocked entries with a reason for every resolution.",
    ),
    (
        "Probe the host from outside the organism",
        "An external timer probes the host over SSH and reports an unreachable machine directly to the owner channel, because in-process alerts cannot survive a dead host. Acceptance is a test of the external report path.",
    ),
    (
        "Mine the literature for one mechanism this harness lacks",
        "One arXiv source is read beyond its abstract and distilled into a memory that names the mechanism, the citation, and one bounded hypothesis applicable to this repository; or the search is recorded as unproductive with the queries used. Acceptance is the memory entry carrying a resolvable arXiv id.",
    ),
    (
        "Compare one subsystem against an open-source peer",
        "One subsystem (planner, memory, recovery, or provider fallback) is compared against a real implementation found via GitHub code search, and the comparison yields either one cited difference worth changing or one cited confirmation that the current design matches practice. Acceptance is a memory entry naming the repository, the file, and the verdict.",
    ),
)

CANONICAL_MEMORIES: tuple[str, ...] = (
    "The only durable way to change code is propose_self_improvement; editing the main worktree is quarantined and never becomes durable.",
    "Run the test suite with .venv/bin/python -m pytest -q from the repository root; the system interpreter has no pytest.",
    "The seed tasks are scaffolding, not an order: the organism may outrank them with a goal of its own when it has better evidence.",
    "Write prototypes to /tmp/skynet-scratch, never into the tracked tree: a dirty worktree blocks proposals and is quarantined.",
    "The organism's own state database is state/skynet.sqlite3; query it read-only with the db tool instead of guessing table names.",
    "Improvement hypotheses have two legitimate sources: this repository's code and runtime state, and the outside world (papers, other projects, documented failure modes). The second requires a citation.",
    "When internal work yields no new evidence, consult an external source before finishing BLOCKED: a cited, distilled finding is progress, an uncited restatement is not.",
)


def _roadmap_fingerprints(title: str, expected_new_fact: str) -> tuple[str, str]:
    return (
        hypothesis_fingerprint(area=ROADMAP_AREA, problem=title, expected_behavior=expected_new_fact),
        structural_fingerprint(area=ROADMAP_AREA, target=title, behavior_kind="roadmap"),
    )


def seed(store: Any, *, root: str | Path | None = None) -> dict[str, Any]:
    """Create the roadmap goal, its tasks and the pinned canonical memories.

    Idempotent: a task whose fingerprint already exists is skipped, and the goal
    is reused by title. Returns a summary suitable for an operator or an event.
    """
    root_path = Path(root) if root is not None else Path.cwd()
    state_path = getattr(store, "state_path", None) or getattr(store, "path", None)
    state_dir = Path(state_path).parent if state_path is not None else root_path / "state"

    existing_goal = store.connection.execute(
        "SELECT goal_id FROM goals WHERE title=? AND status='active' LIMIT 1", (ROADMAP_GOAL_TITLE,)
    ).fetchone()
    constraints = {
        "source": "skynet handoff",
        "rule": "one genuinely open item per task; the expected_new_fact is the acceptance criterion",
    }
    if existing_goal is not None:
        goal_id = str(existing_goal["goal_id"])
        goal_created = False
        # Converge on re-run: without this a corrected priority never reached an
        # existing goal and the planner kept preferring an incidental task.
        store.connection.execute(
            "UPDATE goals SET priority=?, constraints=?, updated_at=? WHERE goal_id=?",
            (ROADMAP_GOAL_PRIORITY, json.dumps(constraints, ensure_ascii=False), utc_now(), goal_id),
        )
    else:
        # Seeded just above an incidental task so the organism is not idle, but
        # below the point where scaffolding would dominate its own priorities.
        goal_id = store.add_goal(ROADMAP_GOAL_TITLE, priority=ROADMAP_GOAL_PRIORITY, constraints=constraints)
        goal_created = True

    created: list[str] = []
    skipped: list[str] = []
    for title, expected_new_fact in ROADMAP_TASKS:
        hypothesis, structural = _roadmap_fingerprints(title, expected_new_fact)
        occupied = store.connection.execute(
            "SELECT 1 FROM tasks WHERE hypothesis_fingerprint=? OR structural_fingerprint=? LIMIT 1",
            (hypothesis, structural),
        ).fetchone()
        if occupied is not None:
            skipped.append(title)
            continue
        task_id = store.add_task(
            title,
            goal_id,
            expected_new_fact=expected_new_fact,
            hypothesis_fingerprint=hypothesis,
            structural_fingerprint=structural,
            area=ROADMAP_AREA,
        )
        created.append(task_id)

    # Idempotency alone was not enough. A task seeded by an earlier revision of
    # ROADMAP_TASKS kept its `pending` status after its entry was deleted, so
    # the planner went on selecting work whose acceptance criterion was already
    # met: the doom-loop task stayed pending after its tests landed. Only this
    # module writes area=ROADMAP_AREA, so the scope is exactly the seeded set,
    # and only `pending` is retired because a `running` task belongs to an
    # episode already in flight.
    current_titles = {title for title, _ in ROADMAP_TASKS}
    retired = 0
    if current_titles:
        placeholders = ", ".join("?" for _ in current_titles)
        retired = store.connection.execute(
            f"UPDATE tasks SET status='cancelled', updated_at=? "
            f"WHERE area=? AND status='pending' AND title NOT IN ({placeholders})",
            (utc_now(), ROADMAP_AREA, *sorted(current_titles)),
        ).rowcount

    pinned_before = {item["content"] for item in store.pinned_memories(limit=64)}
    new_memories = [content for content in CANONICAL_MEMORIES if content not in pinned_before]
    if new_memories:
        store.consolidate(
            f"handoff-{store.state().generation}",
            [{"kind": "procedure", "content": content, "confidence": 0.95} for content in new_memories],
        )
        for content in new_memories:
            row = store.connection.execute("SELECT memory_id FROM memories WHERE content=?", (content,)).fetchone()
            if row is not None:
                store.set_memory_pinned(str(row["memory_id"]))

    store.append_event(
        "roadmap_seeded",
        {
            "goal_id": goal_id,
            "goal_created": goal_created,
            "tasks_created": len(created),
            "tasks_skipped": len(skipped),
            "tasks_retired": retired,
            "memories_pinned": len(new_memories),
        },
    )
    return {
        "goal_id": goal_id,
        "goal_created": goal_created,
        "tasks_created": len(created),
        "tasks_skipped": len(skipped),
        "tasks_retired": retired,
        "memories_pinned": len(new_memories),
        "state_dir": str(state_dir),
    }
