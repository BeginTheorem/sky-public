from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class LifecycleState(StrEnum):
    BOOT = "boot"
    RECOVER = "recover"
    START = "start"
    REACT = "react"
    CONSOLIDATE = "consolidate"
    CHECKPOINT = "checkpoint"
    FAILED = "failed"
    RECOVERING = "recovering"
    PLAN = "plan"
    SLEEP = "sleep"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    NEEDS_RECOVERY = "needs_recovery"


@dataclass(slots=True)
class Budget:
    """Limits for one ReAct episode.

    ``tokens`` caps a single provider request, ``output_tokens`` caps a single
    model response, and ``seconds`` bounds the cumulative time the model spends
    producing successful responses. Tool execution, provider retries, backoff
    and network failures do not count against ``seconds``.
    """

    steps: int = 100
    tokens: int = 200_000
    seconds: float = 1800.0
    output_tokens: int = 8_192


@dataclass(slots=True)
class StartEnvelope:
    wake_cause: str
    observations: list[dict[str, Any]] = field(default_factory=list)
    active_goals: list[dict[str, Any]] = field(default_factory=list)
    pending_work: list[dict[str, Any]] = field(default_factory=list)
    previous_outcome: dict[str, Any] = field(default_factory=dict)
    next_plan: dict[str, Any] = field(default_factory=dict)
    intention: dict[str, Any] = field(default_factory=dict)
    success_criteria: list[dict[str, Any]] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    run_id: str = field(default_factory=lambda: str(uuid4()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "wake_cause": self.wake_cause,
            "observations": self.observations,
            "active_goals": self.active_goals,
            "pending_work": self.pending_work,
            "previous_outcome": self.previous_outcome,
            "next_plan": self.next_plan,
            "intention": self.intention,
            "success_criteria": self.success_criteria,
            "budget": {
                "steps": self.budget.steps,
                 "tokens": self.budget.tokens,
                 "output_tokens": self.budget.output_tokens,
                "seconds": self.budget.seconds,
            },
        }


@dataclass(slots=True)
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(slots=True)
class ModelTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_content: str = ""
    # Time actually spent waiting on the model, excluding provider backoff and
    # fallback delays. A provider that leaves it at zero lets the caller fall
    # back to wall-clock time. Charging retry/backoff to the model budget made
    # slow-but-healthy chains exhaust a run at a few dozen steps.
    model_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.usage_tokens or self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class AgentRunResult:
    """Technical result of one ReAct episode; the report is model-authored text."""

    status: RunStatus
    report: str = ""
    steps: int = 0
    usage_tokens: int = 0
    failure: str = ""
    control_action: dict[str, Any] = field(default_factory=dict)



@dataclass(slots=True)
class RunRecord:
    run_id: str
    attempt: int
    status: RunStatus
    started_at: str
    budget: Budget
    finished_at: str | None = None


@dataclass(slots=True)
class AgentState:
    lifecycle: LifecycleState = LifecycleState.BOOT
    generation: int = 0
    next_wake_at: str | None = None
    active_run_id: str | None = None
    next_plan: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
