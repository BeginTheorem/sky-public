"""Provider-selection index: a heuristic, unwired scorer for (task, provider) pairs.

Why this exists
---------------
The provider chain fails over, but nothing chooses *which* provider to try for a
given kind of work: the chain order is the policy. This module scores the
alternatives so a future decision can be made on numbers instead of position.

What it is NOT
--------------
- It is NOT the closed-form Whittle index of Ayesta et al. 2018
  (arXiv:1810.07782). That index is derived from the steady-state distribution of
  an LPS-d queue with capacities ``q``; reproducing it needs the queue's arrival
  process, which this harness does not model. What is implemented here is a
  *multiplicative heuristic inspired by* that work: a weighted geometric mean of
  four normalised factors, chosen because a zero factor zeroes the product.
- It is NOT a scheduler in the lifecycle sense. The Reactor remains the only
  owner of wake/sleep/scheduling; this module neither wakes anything nor owns
  state. The historical filename is kept for provenance, but nothing imports it:
  wiring it into work selection would create a second planner contour, which
  AGENTS.md forbids, and is therefore a separate owner decision.
- It publishes no measured results. An earlier draft carried a placeholder
  "shadow comparison" with fixed completion rates and confidence intervals; that
  was fabricated output and was removed rather than committed.

Factor mapping (each factor normalised into (0, 1])
---------------------------------------------------
- ``completion`` ~ server capacity (q): Laplace-smoothed success rate, so an
  unseen provider sits at the 0.5 prior instead of 0 or 1.
- ``availability`` ~ server availability: the completion factor, or exactly 0
  while the provider is inside a cooldown. This is the only factor allowed to
  zero the index, because it is the only one that is a hard fact.
- ``urgency`` ~ arrival-rate pressure (p): task priority and retry count.
- ``fit`` ~ scheduling-discipline fit (d): overlap between the task area's
  capability vocabulary and the provider's, as Jaccard similarity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Weights of the geometric mean. They sum to 1.0 (asserted by the tests) and are
# a design choice, not a fitted quantity: there is no measured objective to fit
# them against, and pretending otherwise would repeat the fabricated-numbers
# mistake removed from the earlier draft.
FACTOR_WEIGHTS = {
    "completion": 0.35,
    "availability": 0.25,
    "urgency": 0.20,
    "fit": 0.20,
}

# Non-critical factors are floored so a weak-but-real factor cannot silently zero
# the index: only an unavailable provider (cooldown) may do that.
FACTOR_FLOOR = 0.01
MAX_PRIORITY = 4.0
ATTEMPTS_FOR_FULL_URGENCY = 5.0

# Task areas to capability vocabularies. The vocabulary is deliberately coarse:
# it is a rough fit signal, not a claim about what a provider can do.
AREA_CAPABILITIES: dict[str, frozenset[str]] = {
    "engineering": frozenset({"code", "refactor", "test", "debug", "api"}),
    "research": frozenset({"read", "analyze", "synthesize", "cite", "search"}),
    "validation": frozenset({"test", "verify", "benchmark", "assert", "measure"}),
    "memory": frozenset({"store", "retrieve", "consolidate", "prune", "embed"}),
    "tools": frozenset({"mcp", "api", "cli", "shell", "script"}),
    "providers": frozenset({"provider", "api", "fallback", "health", "cooldown"}),
    "scheduler": frozenset({"plan", "rank", "select", "schedule", "optimize"}),
    "general": frozenset({"read", "write", "analyze", "search"}),
}

# Provider name fragment to capability vocabulary. Order matters: the first
# fragment found in the (lowercased) name wins.
PROVIDER_CAPABILITIES: tuple[tuple[str, frozenset[str]], ...] = (
    # ``deepseek`` ahead of ``nvidia``: the NIM deepseek rung is the one NVIDIA
    # rung that is not a nemotron, so the fragment order gives it the math set
    # while super/ultra/lightning fall to ``nvidia``. Both fragments match the
    # long ``nvidia_nemotron_*`` aliases too, so a parked snapshot or a test that
    # still names them keeps its capabilities.
    ("deepseek", frozenset({"reasoning", "code", "math", "structured_output"})),
    ("nemotron", frozenset({"reasoning", "code", "long_context", "structured_output"})),
    ("nvidia", frozenset({"reasoning", "code", "long_context", "structured_output"})),
    ("openrouter", frozenset({"reasoning", "code", "long_context", "structured_output"})),
    ("zai", frozenset({"fast", "code", "structured_output"})),
    ("glm", frozenset({"fast", "code", "structured_output"})),
    ("ollama", frozenset({"local", "code", "structured_output"})),
    ("deepseek", frozenset({"reasoning", "code", "math", "structured_output"})),
)


@dataclass(slots=True)
class TaskSpec:
    """The task side of a scored pair.

    ``hypothesis_fingerprint``/``structural_fingerprint`` are carried through for
    provenance and tie-breaking; neither is scored.
    """

    task_id: str
    area: str = "general"
    priority: float = 0.0
    attempts: int = 0
    hypothesis_fingerprint: str | None = None
    structural_fingerprint: str | None = None


@dataclass(slots=True)
class ProviderSpec:
    """The provider side of a scored pair.

    ``cooldown_remaining`` is read from the chain's own clock; a positive value
    is a hard fact (the provider declared it cannot answer) and zeroes the index.
    """

    provider_id: str
    name: str
    cooldown_remaining: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    capabilities: frozenset[str] = frozenset()


@dataclass(slots=True)
class PairScore:
    """A scored (task, provider) pair with its factors for diagnosis."""

    task_id: str
    provider_id: str
    index: float
    components: dict[str, float]


class ProviderIndexScorer:
    """Scores (task, provider) pairs and selects the highest index.

    Pure: it holds only the capability lookup, takes plain dataclasses, and never
    reads the store, the chain, or the clock. Cooldown state is an input.
    """

    def __init__(self) -> None:
        self._area_cache: dict[str, frozenset[str]] = {}
        self._provider_cache: dict[str, frozenset[str]] = {}

    def area_capabilities(self, area: str) -> frozenset[str]:
        key = area.strip().lower()
        cached = self._area_cache.get(key)
        if cached is None:
            cached = AREA_CAPABILITIES.get(key, AREA_CAPABILITIES["general"])
            self._area_cache[key] = cached
        return cached

    def provider_capabilities(self, name: str) -> frozenset[str]:
        key = name.strip().lower()
        cached = self._provider_cache.get(key)
        if cached is None:
            cached = frozenset({"general"})
            for fragment, capabilities in PROVIDER_CAPABILITIES:
                if fragment in key:
                    cached = capabilities
                    break
            self._provider_cache[key] = cached
        return cached

    @staticmethod
    def _clamp(value: float, low: float = FACTOR_FLOOR, high: float = 1.0) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _unit(value: float) -> float:
        """Clamp an intermediate normalisation to [0, 1] without the factor floor.

        The floor belongs to a finished factor: applying it to a component first
        would stop a zero priority from being zero and bias the blend.
        """
        return max(0.0, min(1.0, value))

    def completion_factor(self, provider: ProviderSpec) -> float:
        """Laplace-smoothed success rate: the 0.5 prior for an unseen provider."""
        attempts = max(0, provider.success_count) + max(0, provider.failure_count)
        return (max(0, provider.success_count) + 1.0) / (attempts + 2.0)

    def availability_factor(self, provider: ProviderSpec) -> float:
        """0.0 while cooling (a hard fact), else the completion factor."""
        if provider.cooldown_remaining > 0:
            return 0.0
        return self.completion_factor(provider)

    def urgency_factor(self, task: TaskSpec) -> float:
        """Priority pressure (0..MAX_PRIORITY) blended with retry pressure."""
        priority = self._unit(task.priority / MAX_PRIORITY)
        retry = self._unit(max(0, task.attempts) / ATTEMPTS_FOR_FULL_URGENCY)
        return self._clamp(0.6 * priority + 0.4 * retry)

    def fit_factor(self, task: TaskSpec, provider: ProviderSpec) -> float:
        """Jaccard overlap between the task area's vocabulary and the provider's.

        Both vocabularies always resolve to something -- an unknown area falls
        back to the "general" area vocabulary, and an unlisted provider name to
        {"general"} -- so an unknown pair lands on the floor rather than a
        special-case 1.0.
        """
        task_caps = self.area_capabilities(task.area)
        provider_caps = provider.capabilities or self.provider_capabilities(provider.name)
        union = task_caps | provider_caps
        return self._clamp(len(task_caps & provider_caps) / len(union))

    def score(self, task: TaskSpec, provider: ProviderSpec) -> PairScore:
        """Score one pair. A cooling provider scores exactly 0.0."""
        components = {
            "completion": self.completion_factor(provider),
            "availability": self.availability_factor(provider),
            "urgency": self.urgency_factor(task),
            "fit": self.fit_factor(task, provider),
        }
        providers_scores = {name: components[name] for name in FACTOR_WEIGHTS}
        if components["availability"] <= 0.0:
            return PairScore(task.task_id, provider.provider_id, 0.0, components)
        log_index = sum(weight * math.log(max(FACTOR_FLOOR, providers_scores[name])) for name, weight in FACTOR_WEIGHTS.items())
        return PairScore(task.task_id, provider.provider_id, math.exp(log_index), components)

    def select(self, tasks: Sequence[TaskSpec], providers: Sequence[ProviderSpec]) -> PairScore | None:
        """The highest-index pair, or None when nothing scores above zero.

        Inputs are walked in id order and only a strictly higher index replaces
        the incumbent, so ties resolve to the smallest task id and then provider
        id regardless of input order: an unstable tie-break would make the same
        portfolio produce different work on identical inputs.
        """
        best: PairScore | None = None
        for task in sorted(tasks, key=lambda item: item.task_id):
            for provider in sorted(providers, key=lambda item: item.provider_id):
                scored = self.score(task, provider)
                if scored.index <= 0.0:
                    continue
                if best is None or scored.index > best.index:
                    best = scored
        return best


__all__ = [
    "AREA_CAPABILITIES",
    "FACTOR_WEIGHTS",
    "PROVIDER_CAPABILITIES",
    "PairScore",
    "ProviderIndexScorer",
    "ProviderSpec",
    "TaskSpec",
]
