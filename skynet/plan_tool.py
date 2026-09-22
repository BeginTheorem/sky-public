"""The shared plan artifact: one explicit plan recorded inside the ReAct run.

The owner asked whether the single ReAct loop should split into "Plan" and
"Build" modes. Measurement of 42 real runs answered no for now: an external
planner already exists, the in-run planning slice is ~18% of input tokens, the
window is used to 31% with no compaction ever firing, and a second model loop
would double the structured-output failure surface. The approved alternative is
instrumentation first: let the model record a brief plan before its first
mutation, carry that plan into the next cycle's ``StartEnvelope``, and *measure*
whether it correlates with better outcomes. Nothing here gates a run.

The artifact is an ``event_log`` event (no new table, no schema bump). The
ordering question -- "did a plan precede the first mutation?" -- is answered
purely from durable ``sequence`` numbers, so ``react.py`` stays untouched.
"""

from __future__ import annotations

from typing import Any

from .provider import Tool
from .store import StateStore

#: Durable event kind written by :class:`RecordPlanTool`.
PLAN_EVENT_KIND = "plan_recorded"

#: Durable event kind the reactor writes at run finish (measurement substrate).
PLAN_OBSERVATION_EVENT_KIND = "plan_observation"

# Tools whose successful use changes durable state. The list is deliberately the
# *durable* mutation surface, not every ``capability_kind == "write"`` tool:
#   * ``propose_self_improvement`` (``skynet/self_improvement.py``) is the only
#     durable code path: it creates, gates, and promotes a proposal
#     (AGENTS.md: "durable_code_change: only propose_self_improvement makes a
#     code change durable").
#   * ``request_rollback`` (``skynet/rollback.py:47``) mutates deployed code.
#   * ``memory`` with a write action (``skynet/memory_tool.py:24``,
#     ``ACTIONS`` at ``skynet/memory_tool.py:26``) changes durable memory.
# ``bash`` (``skynet/tools.py:83``) is *excluded* on purpose: a command string
# can be a read or a write, so classifying it would mark almost every run as
# mutated and destroy the signal. Prototype file writes belong in the scratch
# directory and are not durable (AGENTS.md).
MUTATION_TOOL_NAMES = frozenset({"propose_self_improvement", "request_rollback"})
MEMORY_WRITE_ACTIONS = frozenset({"remember", "forget", "pin", "unpin", "correct"})


class RecordPlanTool(Tool):
    """Record a brief plan for the current run, durably and bounded.

    The model calls this before its first durable mutation. The argument is a
    bounded string so the tool cannot be used to dump unbounded text into the
    event log.
    """

    name = "record_plan"
    capability_kind = "write"

    # 4000 characters mirrors ``MemoryTool.MAX_CONTENT_CHARS`` (a plan is about
    # as large as a memory entry: ~1000 tokens, far below the 16k output budget
    # and the 500k ReAct window). Truncation is a deterministic prefix and is
    # flagged, never a rejection that would waste the model's turn.
    MAX_PLAN_CHARS = 4000
    MAX_STEP_CHARS = 500
    MAX_STEPS = 8

    def __init__(self, store: StateStore) -> None:
        self.store = store

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Record a brief plan for this run before making your first change to durable "
                    "state (a self-improvement proposal, a rollback, or a memory write). The plan is "
                    "carried into the next cycle's StartEnvelope as next_plan.recorded_plan so the "
                    "next run sees what this one intended. A plan is NOT required for read-only "
                    "investigation and never blocks or fails the run."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "string",
                            "description": f"The plan, one bounded string (truncated at {self.MAX_PLAN_CHARS} characters).",
                        },
                        "steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                f"Optional short list of intended steps (at most {self.MAX_STEPS}, "
                                f"each truncated at {self.MAX_STEP_CHARS} characters)."
                            ),
                        },
                    },
                    "required": ["plan"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        plan = str(arguments.get("plan", "")).strip()
        if not plan:
            return {"ok": False, "error": "plan must not be empty"}
        bounded_plan = plan[: self.MAX_PLAN_CHARS]
        truncated = len(plan) > self.MAX_PLAN_CHARS
        steps = self._bounded_steps(arguments.get("steps"))
        run_id, step_index = self._run_context(idempotency_key)
        if run_id is None:
            return {"ok": False, "error": "no active run to record a plan for"}
        payload: dict[str, Any] = {
            "run_id": run_id,
            "plan": bounded_plan,
            "steps": steps,
            "truncated": truncated,
        }
        if step_index is not None:
            payload["step_index"] = step_index
        try:
            sequence = self.store.append_event(PLAN_EVENT_KIND, payload, run_id)
        except Exception as exc:
            return {"ok": False, "error": f"append_event failed: {exc}"}
        return {
            "ok": True,
            "recorded": True,
            "sequence": sequence,
            "plan_chars": len(bounded_plan),
            "steps": len(steps),
            "truncated": truncated,
        }

    def _bounded_steps(self, raw: object) -> list[str]:
        if not isinstance(raw, list):
            return []
        steps: list[str] = []
        for item in raw[: self.MAX_STEPS]:
            text = str(item).strip()
            if text:
                steps.append(text[: self.MAX_STEP_CHARS])
        return steps

    def _run_context(self, idempotency_key: str) -> tuple[str | None, int | None]:
        """Resolve the run id and ReAct step from durable identity.

        ``react.py`` builds the effect key as ``{run_id}:{step}:{call_id}`` and
        passes it as ``idempotency_key`` (``skynet/react.py:374,675``). The
        prefix is the authoritative run id; ``active_run_id`` is the fallback so
        a tool invoked outside that exact shape still attributes the event.
        """
        run_id: str | None = None
        step_index: int | None = None
        parts = idempotency_key.split(":")
        if len(parts) == 3:
            run_id = parts[0] or None
            try:
                step_index = int(parts[1])
            except ValueError:
                step_index = None
        if run_id is None:
            try:
                run_id = self.store.state().active_run_id
            except Exception:
                run_id = None
        return run_id, step_index


def is_mutation_call(payload: object) -> bool:
    """Whether a ``tool_call`` event payload names a durable mutation.

    The event kind is always ``tool_call`` (``skynet/react.py:354``); the
    mutation is identified by tool name plus, for ``memory``, the write action.
    """
    if not isinstance(payload, dict):
        return False
    tool_name = payload.get("tool_name")
    if tool_name in MUTATION_TOOL_NAMES:
        return True
    if tool_name == "memory":
        arguments = payload.get("arguments")
        action = str(arguments.get("action", "")).strip().casefold() if isinstance(arguments, dict) else ""
        return action in MEMORY_WRITE_ACTIONS
    return False


def plan_preceded_first_mutation(store: StateStore, run_id: str) -> bool | None:
    """Whether a plan was recorded before this run's first durable mutation.

    Answers from durable ``event_log`` sequence order alone, so the ReAct loop
    needs no instrumentation. Returns ``None`` for a run with no mutation
    (a research-only run) so it is not counted as a miss, ``False`` when a
    mutation happened with no plan before it, and ``True`` otherwise.
    """
    plan_sequence: int | None = None
    mutation_sequence: int | None = None
    for event in store.recent_run_events(run_id, limit=None):
        sequence = event.get("sequence")
        if plan_sequence is None and event.get("kind") == PLAN_EVENT_KIND:
            plan_sequence = sequence
        elif mutation_sequence is None and event.get("kind") == "tool_call" and is_mutation_call(event.get("payload")):
            mutation_sequence = sequence
        if plan_sequence is not None and mutation_sequence is not None:
            break
    if mutation_sequence is None:
        return None
    if plan_sequence is None:
        return False
    return plan_sequence < mutation_sequence


def latest_recorded_plan(store: StateStore, run_id: str) -> dict[str, Any] | None:
    """The last plan payload recorded for a run, or ``None``.

    Used by the reactor to carry the plan into ``state.next_plan`` under a
    distinct key, so the next ``StartEnvelope`` sees it without clobbering the
    MemoryLoop's own ``next_plan`` writes.
    """
    latest: dict[str, Any] | None = None
    for event in store.recent_run_events(run_id, limit=None):
        if event.get("kind") != PLAN_EVENT_KIND:
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            latest = payload
    return latest
