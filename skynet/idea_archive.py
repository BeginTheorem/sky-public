"""Behavioural descriptors and diversity math for the idea archive.

The archive is the organism's entropy sink. Selection over a finite, greedily
ranked task pool converges: either it cycles or it thermalizes into "no novel
work -> sleep". Quality-diversity keeps a structured repertoire instead, and the
descriptors below are the axes of that repertoire.

Dimensions follow MAP-Elites practice (2-4 low-dimensional behavioural axes,
tens to hundreds of cells; Mouret & Clune 2015 via arXiv:2506.13131) and the
scaffold taxonomy of arXiv:2607.13104 (prompts / memory / tools / control logic).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

SUBSYSTEMS = ("reactor", "memory", "scheduler", "tools", "providers", "general")
CHANGE_TYPES = ("new-tool", "prompt-edit", "workflow", "control-logic", "memory-structure", "eval-improvement")
EVIDENCE_SOURCES = ("own-repo", "internal-benchmark", "paper", "web")
CELLS_TOTAL = len(SUBSYSTEMS) * len(CHANGE_TYPES) * len(EVIDENCE_SOURCES)

_SUBSYSTEM_RULES = (
    (("reactor.py", "heartbeat.py", "supervisor.py", "watchdog.py", "recovery.py", "lock.py"), "reactor"),
    (("memory", "store.py"), "memory"),
    (("planner.py", "autonomous_planner.py", "handoff.py"), "scheduler"),
    (("tools.py", "mcp.py", "policy.py"), "tools"),
    (("provider", "providers/"), "providers"),
)
_PROMPT_FILES = ("model_contracts.py", "SOUL.md", "react.py")
_EVAL_FILES = ("metrics.py", "evaluation", "reporting.py")

_ARXIV_RE = re.compile(r"arxiv\.org|\b\d{4}\.\d{4,5}\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://", re.IGNORECASE)


def cell_key(subsystem: str, change_type: str, evidence_source: str) -> str:
    return f"{subsystem}|{change_type}|{evidence_source}"


def classify(*, scope: Sequence[str], kind: str, validation: str, inspiration_ref: str) -> tuple[str, str, str]:
    """Derive the behavioural descriptor deterministically from the proposal.

    Deterministic on purpose: a descriptor the model could choose freely would
    be gamed to fill cells without filling them. Scope paths decide the
    subsystem, the touched artifact decides the change type, and the citation
    decides the evidence source.
    """
    del kind
    paths = " ".join(str(item).lower() for item in scope)
    subsystem = "general"
    for needles, name in _SUBSYSTEM_RULES:
        if any(needle in paths for needle in needles):
            subsystem = name
            break
    if any(name.lower() in paths for name in _PROMPT_FILES):
        change_type = "prompt-edit"
    elif "tools.py" in paths or "mcp.py" in paths:
        change_type = "new-tool"
    elif "memory" in paths:
        change_type = "memory-structure"
    elif any(name.lower() in paths for name in _EVAL_FILES):
        change_type = "eval-improvement"
    elif subsystem in {"reactor", "scheduler", "providers"}:
        change_type = "control-logic"
    else:
        change_type = "workflow"
    ref = str(inspiration_ref or "").strip()
    if _ARXIV_RE.search(ref):
        evidence_source = "paper"
    elif _URL_RE.search(ref):
        evidence_source = "web"
    elif re.search(r"benchmark|metric|snapshot|measurement", str(validation or ""), re.IGNORECASE):
        evidence_source = "internal-benchmark"
    else:
        evidence_source = "own-repo"
    return subsystem, change_type, evidence_source


def learnability_defect(proposal: Mapping[str, object]) -> str | None:
    """Reject noise before it costs a cycle. Return None when the idea is learnable.

    Open-endedness is novelty AND learnability (arXiv:2406.04268); pure novelty
    is a stochastic trap (noisy TV). Cheap deterministic gates only - an
    expensive judge call is not justified before the invariants pass
    (arXiv:2607.13104).
    """
    title = str(proposal.get("title", "")).strip()
    fact = str(proposal.get("expected_new_fact", "")).strip()
    validation = str(proposal.get("validation", "")).strip()
    kind = str(proposal.get("kind", "")).strip()
    ref = str(proposal.get("inspiration_ref", "") or "").strip()
    if not fact or _norm(fact) == _norm(title):
        return "expected_new_fact restates the title; there is nothing to learn"
    if re.fullmatch(r"(?i)\s*(tests? pass|all green|ok|done)\s*", validation):
        return "validation names no measurement"
    if len(validation.split()) < 4:
        return "validation is not an observable signal"
    if kind == "research" and not ref:
        return "research proposal cites no external source"
    return None


def parent_weight(quality: float, children: int, *, lam: float = 10.0, alpha0: float = 0.5) -> float:
    """DGM: sigmoid(lam*(quality-alpha0)) * 1/(1+n_children)."""
    return (1.0 / (1.0 + math.exp(-lam * (float(quality) - float(alpha0))))) * (1.0 / (1.0 + max(0, int(children))))


def effective_modes(counts: Iterable[int]) -> float:
    """exp(Shannon H) over cell occupancies: the effective number of modes.

    A structural Vendi-Score proxy (arXiv:2604.18005): robust to imbalance and,
    because a cell stores one occupant, impossible to inflate with duplicates.
    """
    values = [int(c) for c in counts if int(c) > 0]
    total = sum(values)
    if not total:
        return 0.0
    entropy = -sum((v / total) * math.log(v / total) for v in values)
    return math.exp(entropy)


def lexical_uniqueness(texts: Sequence[str], *, n: int = 3) -> float:
    """IDF-weighted n-gram novelty of texts against each other.

    Sanity check from arXiv:2604.18005: rising semantic diversity without rising
    lexical uniqueness is verbose paraphrase, not new ideas.
    """
    if len(texts) < 2:
        return 1.0
    grams = [Counter(_ngrams(t, n)) for t in texts]
    docs = len(grams)
    df: Counter[str] = Counter()
    for g in grams:
        df.update(g.keys())
    score = 0.0
    weight = 0.0
    for g in grams:
        for gram in g:
            idf = math.log((1 + docs) / (1 + df[gram])) + 1.0
            score += idf * (1.0 if df[gram] == 1 else 0.0)
            weight += idf
    return round(score / weight, 6) if weight else 1.0


def structural_disorder(texts: Sequence[str]) -> float:
    """1 - phi: mean token-set cosine distance to the centroid (echo-chamber detector)."""
    sets = [set(_norm(t).split()) for t in texts if t and t.strip()]
    if len(sets) < 2:
        return 1.0
    centroid: Counter[str] = Counter()
    for s in sets:
        centroid.update(s)
    total = 0.0
    for s in sets:
        denom = math.sqrt(len(s)) * math.sqrt(sum(v * v for v in centroid.values()))
        total += (sum(centroid[t] for t in s) / denom) if denom else 0.0
    phi = total / len(sets)
    return round(max(0.0, 1.0 - phi), 6)


def _norm(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", str(value).casefold()))


def _ngrams(value: object, n: int) -> Iterable[str]:
    tokens = _norm(value).split()
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return (" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
