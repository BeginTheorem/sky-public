# Candidate-Linkage Protocol for New Extraction Candidates

**Status:** design proposal (bounded, testable)
**Owner:** Sky (SkyNet persistent agent)
**Frameworks:** CiteGuard (arXiv:2510.17853v4) + Cited-but-Not-Verified (arXiv:2605.06635v1); HaluMem linkage gap (arXiv:2511.03506v3 Sec 6.2.1)

## Problem

SkyNet's `MemoryTransitionVerifier.verify_transition` (store.py:51-101) only evaluates candidates that carry `supersedes_memory_id` — an existing memory to contradict. New extraction candidates carry `supersedes_memory_id=None` and return `("Correct", "no original memory to contradict")` unconditionally, including factually contradictory ones.

This is the HaluMem Sec 6.2.1 linkage gap: new candidates have no original-memory/source linkage, so they cannot be caught as Hallucination.

## Protocol

### 1. Source-context schema for extraction candidates

Each new extraction candidate carries an `evidence[]` field (already declared in the memory response model contract) holding typed source references:

```
{
  "kind": "fact|observation|measurement|belief",
  "content": str,
  "confidence": float,
  "evidence": list[str] | str,   # typed source references
  "supersedes_memory_id": str | None   # None for new candidates
}
```

Evidence entries are parsed into typed source descriptors (deterministic, inference-free):

| Prefix | Descriptor type | Retrievable? | Link Works applies |
|--------|----------------|--------------|-------------------|
| `url:<address>` | url | HTTP GET (bounded timeout) | yes |
| `commit:<sha>` | commit | `git show` / `git log` on local repo | yes |
| `tool:<name>` | tool | run's effect ledger / captured tool output | yes |
| `memory:<id>` | memory | `memories` table, active rows only | yes |
| `date:<iso>` / bare text | date/bare | temporal anchor, no fetchable content | no |

The source-context passed to the verifier is:

```
source_context = {
    "<descriptor_id>": {
        "accessible": bool,        # Link Works: source reachable & extractable
        "content": str | None,     # retrieved text (None if inaccessible)
        "type": "url|commit|tool|memory|date|bare",
        "retrieved_at": str | None,
    },
    ...
}
```

descriptor_id = `type:value` (e.g. `memory:7f3a...`, `url:https://...`).

### 2. Retrieval strategy mapping source_context to verifiable evidence

For each unique source descriptor, retrieve content once (bounded, memoized per consolidation):

- `url:` → HTTP GET with bounded timeout; extract main text (SkyNet `webfetch`/`fetch_content`). `accessible=False` on timeout/4xx/5xx/empty body → Link Works fails.
- `commit:` → `git show <sha>` / `git log -1 --format=%B <sha>` against the local repository (deterministic, no network). `accessible=False` if the sha is unknown.
- `tool:` / `memory:` → read from the current run's effect ledger or the `memories` table (`status='active'`). No network.
- `date:` / bare → `accessible=False`, `content=None`; contributes no Link Works/Relevance/Fact Check evidence (topical anchor only).

Only active memories are reachable, so a `memory:` citation verifies against durable, non-superseded source text.

Current gap in the shipped code: `_build_source_context` (store.py:2885-2907) resolves only `memory:` citations. The protocol requires it to resolve `url:`/`commit:`/`tool:` too, populating `source_context` for every descriptor in a new candidate's `evidence[]`.

### 3. Three-dimensional scoring rubric (adapted from Cited-but-Not-Verified)

For each (candidate claim, source) pair, the `SourceAttributionVerifier.verify()` (skynet/source_attribution_verifier.py) computes three evaluators:

1. **Link Works** — deterministic binary: source reachable and content extractable. No LLM.
2. **Relevant Content** — topical alignment between candidate `content` and `source_content`, rubric-scored via keyword-overlap (`_words` intersection / claim word count), threshold `_RELEVANCE_THRESHOLD = 0.2`.
3. **Fact Check** — key-fact support: extract key facts (ISO dates, numbers, proper nouns — same `_key_facts`/`_contradicts` as `MemoryTransitionVerifier`, store.py:78-88) from claim and source; `False` if they contradict, `None` if either side has no key facts.

Per-citation breakdown is `[s1, s2, s3]` ∈ {0,1} (or rubric). Candidate-level verdict policy:

| Condition | Verdict | Action on candidate |
|-----------|---------|--------------------|
| any retrieved source contradicts key facts | `Hallucination` | **drop** before promotion (same as TRUSTMEM) |
| no retrievable source at all (Link Works fails everywhere) | `unverifiable_link` | keep but flag for audit; may still land |
| sources reachable but topically misaligned | `RelevanceLow` | keep, demote confidence |
| all retrievable sources support key facts + relevant | `Correct` | keep; record evals on row |

`breakdown = {"verdict","reason","sources","evals","new_facts"}` is stored on the candidate as `_source_attribution` for audit, and the eval summary can be carried in the `memory_event` payload.

### 4. Integration point in `consolidate_versioned()`

`consolidate_versioned` (store.py:2909-2961) already instantiates `SourceAttributionVerifier` and calls `verify()` for new candidates (those without `supersedes_memory_id`) **before** `_consolidate_candidates`, then writes surviving candidates via `_consolidate_candidates`.

The required integration:

```python
source_context = self._build_source_context(candidates)
for item in candidates:
    supersedes_id = item.get("supersedes_memory_id")
    if isinstance(supersedes_id, str) and supersedes_id:
        # existing TRUSTMEM path: original-memory linkage (unchanged)
        ...verify_transition...
        continue
    # NEW candidate: source-attribution linkage
    verdict, breakdown = source_attribution_verifier.verify(item, source_context)
    item["_source_attribution"] = {"verdict": verdict, "breakdown": breakdown}
    if verdict == "Hallucination":
        continue        # drop, like TRUSTMEM
    verified.append(item)
```

Superseding candidates keep the existing TRUSTMEM path (original-memory linkage). New candidates are linked to retrieved source evidence instead.

### 5. Acceptance criterion

**Hypothesis:** Adding the candidate-linkage protocol — retrieval strategy for `url:`/`commit:`/`tool:`/`memory:` evidence plus the three-dimensional SourceAttributionVerifier — integrated into `consolidate_versioned()` before `_consolidate_candidates()`, reduces the new-candidate hallucination rate on the HaluMem-inspired benchmark (`benchmark_v4.py`, new-candidate subset) from the current 56% to ≤28%.

**Validation:**
- Unit test: a fabricated new candidate whose `evidence[]` cites a retrieved source contradicting its key facts (e.g. claim "2026-12-01 to Tokyo" vs source "2026-11-05 to Berlin") must return `Hallucination` and be dropped; a consistent new candidate must return `Correct`.
- End-to-end: run `benchmark_v4.py` new-candidate subset before/after; measure hallucination rate.

## Citations

- arXiv:2605.06635v1 — Cited but Not Verified (three-stage pipeline, three evaluators, empirical findings)
- arXiv:2510.17853v4 — CiteGuard (citation attribution alignment via retrieval-augmented validation; CiteME benchmark, alternative valid citations)
- arXiv:2511.03506v3 Sec 6.2.1 — HaluMem (memory-updating linkage bottleneck for new candidates)
