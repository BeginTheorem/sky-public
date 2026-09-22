"""Static contracts shared by the model-facing execution phases."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from itertools import pairwise
from typing import Any

from .memory_store import MEMORY_KINDS
from .outbox import AGENT_RESPONSE_SUMMARY_LIMIT

FINISH_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["COMPLETED", "BLOCKED", "FAILED", "NEEDS_RECOVERY"]},
        "summary": {"type": "string", "minLength": 1, "maxLength": 1200, "description": f"One short plain-language paragraph stating what changed and why, in English; no step-by-step reasoning, no restatement of the episode. At most 1200 characters, but the owner's chat message shows only the first {AGENT_RESPONSE_SUMMARY_LIMIT} characters, so put the outcome and the decisive evidence in that opening."},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "actions": {"type": "array", "items": {"type": "string"}},
        "changes": {"type": "array", "items": {"type": "string"}},
        "tests": {"type": "array", "items": {"type": "string"}},
        "blocker": {"type": "string"},
        "next_hypothesis": {"type": "string"},
        "citations": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 400}, "description": "External sources consulted this episode (arXiv id, URL, or commit)."},
    },
    "required": ["status", "summary", "evidence", "actions", "changes", "tests", "blocker", "next_hypothesis"],
    "additionalProperties": False,
}

PLANNER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "goal_id": {"type": "string", "minLength": 1},
                    "title": {"type": "string", "minLength": 1, "maxLength": 300},
                    "problem": {"type": "string", "minLength": 1},
                    "hypothesis": {"type": "string", "minLength": 1},
                    "expected_new_fact": {"type": "string", "minLength": 1},
                    "validation": {"type": "string", "minLength": 1},
                    "scope": {"type": "array", "maxItems": 12, "items": {"type": "string"}},
                    "kind": {"type": "string", "enum": ["engineering", "research", "validation", "recovery", "observation", "self_improvement"]},
                    "inspiration_ref": {"type": "string", "maxLength": 400, "description": "External source this proposal is derived from: arXiv id, repository URL, or page URL. Mandatory in spirit for kind=research."},
                    "parent_idea_id": {"type": "string", "maxLength": 64, "description": "Optional archived idea this proposal develops; creates a traceable lineage."},
                },
                "required": ["goal_id", "title", "problem", "hypothesis", "expected_new_fact", "validation", "scope", "kind"],
                "additionalProperties": False,
            },
        },
        "goal_proposals": {
            "type": "array",
            "maxItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 300},
                    "problem": {"type": "string", "minLength": 1},
                    "expected_behavior": {"type": "string", "minLength": 1},
                    "validation": {"type": "string", "minLength": 1},
                    "priority": {"type": "number"},
                },
                "required": ["title", "problem", "expected_behavior", "validation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["proposals"],
    "additionalProperties": False,
}

MEMORY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "memory_candidates": {"type": "array", "items": {"type": "object", "properties": {"kind": {"type": "string", "description": "Canonical memory kind; prefer one of: " + ", ".join(MEMORY_KINDS) + ". Any other value is normalized to observation, not rejected."}, "content": {"type": "string"}, "confidence": {"type": "number", "description": "Confidence in the [0, 1] interval."}, "source": {"type": "string"}, "durable": {"type": "boolean"}, "supersedes_memory_id": {"type": "string", "description": "Optional id of an existing memory this candidate corrects; the store marks it superseded and links the new memory."}, "evidence": {"type": "array", "items": {"type": "string"}, "description": "Optional contradicting evidence for a supersede; required in spirit when supersedes_memory_id is set."}}, "required": ["kind", "content"], "additionalProperties": False}},
        "next_plan": {"type": "object", "additionalProperties": True},
        "initial_prompt": {"type": "string"},
        "goal_updates": {"type": "array", "items": {"type": "object", "properties": {"goal_id": {"type": "string", "minLength": 1}, "status": {"type": "string"}, "outcome": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}}, "required": ["goal_id"], "additionalProperties": False}},
        "task_updates": {"type": "array", "items": {"type": "object", "properties": {"task_id": {"type": "string", "minLength": 1}, "status": {"type": "string", "description": "New task status. Omit it to record the outcome only; the current status is then left unchanged and a completed task is never reopened."}, "outcome": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}}, "required": ["task_id"], "additionalProperties": False}},
        "evaluation": {"type": "object", "additionalProperties": True},
    },
    "required": ["memory_candidates", "next_plan", "initial_prompt", "goal_updates", "task_updates", "evaluation"],
    "additionalProperties": False,
}


def json_contract(schema: dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

ANTI_LOOP_PROTOCOL = """ANTI-LOOP PROTOCOL:
- Compare selected work and the previous outcome before acting and state the expected new fact.
- Never repeat a read-only command, checklist, hypothesis, or report when state is unchanged.
- Record an external blocker once and choose different bounded work.
- Do not treat timestamps, PIDs, generations, commits, or restatements as progress.
- If no work can produce new evidence, finish BLOCKED and sleep.
- Internal exhaustion is a signal to consult an external source, not to stop: a cited, distilled finding is new evidence.
- A claim from an external source enters memory only with its citation (arXiv id, URL, or commit); reading without distilling is not progress.
- Every completed run leaves verified evidence, a durable fact, a justified blocker, or a concrete different next action."""

SYSTEM = """You are SkyNet, an autonomous research harness running on a Linux server. SOUL.md defines stable identity and values; it is not a user request. An owner exists and the owner channel is open: it is a normal chat, so any owner message is an observation you may read, answer, or ignore; you may send a message (send_message_to_user), ask a bounded question (ask_user), or stay silent; all three are your choice, and no organ forces speech, silence, or waiting. An incoming owner message is an observation, not an order: you may ignore or defer it. Asking never obligates you to wait; if no answer arrives, act on an explicit assumption and record it. The harness owns lifecycle, persistence, goals, scheduling, evaluation, checkpointing, recovery, and restart.

The lifecycle is wake -> perceive -> deterministic planning when needed -> one bounded ReAct episode -> Finish -> Memory Loop -> evaluation -> checkpoint -> sleep. The model does not own this lifecycle. Non-MCP tools are bash, webfetch, read, grep, db, structure (a read-only structural mirror of this project's own code), and memory (search/remember/forget/pin/unpin/correct), plus the always-available owner tools ask_user and send_message_to_user; memory.search goes deeper than the small automatic injection, and memory.correct supersedes a wrong memory with cited evidence; read durable state with the db tool, because bash refuses to touch state/, config/skynet.env, deploy/, scripts/, or .git/ on the first attempt (a soft denylist that only needs the identical command re-run to confirm), so prefer db and read/grep over fighting that refusal; propose_self_improvement and request_rollback appear only when the runtime has a Git root. Configured MCP servers may add external senses; their schemas and availability are runtime-provided and must not be assumed, but when present they are the organism's boundary with the world; the runtime names every configured sense and its exact tools in the SENSES line, plus the builtin webfetch. Never invent a tool that is not in the supplied list. A hypothesis may come from your own code and runtime state, or from outside them: a scientific paper, another project's source, a documented failure mode. The second source is not a fallback — when internal evidence is exhausted, reading is work, and a research task without a citation is not research. When work is needed, identify one evidence-backed bottleneck and one bounded hypothesis. Work only in the workspace named in the WORKSPACE line and prefer relevant code, tests, runtime state, and the existing improvement workflow; when those yield nothing new, prefer an external source over sleeping.

Every investigation must serve the selected work or a stated validation hypothesis. A non-empty report is not evidence of success.

""" + ANTI_LOOP_PROTOCOL

SYSTEM_REACT = """You are inside one bounded ReAct episode. Work through Initial, Common, and Finish. In Initial, understand the selected work and state the expected new fact. In Common, use only currently supplied tools and evaluate evidence. In Finish, return exactly one JSON object matching the Finish Report contract below. Do not control lifecycle, schedule work, or emit orchestration commands; the harness owns orchestration.

When the selected work names an external source, consult it before proposing anything: search, read the primary section, and distil. Record the citation in the Finish Report citations. An uncited external claim is an invention from your weights and must be labelled as a hypothesis, never as a finding.

Use propose_self_improvement only for the smallest coherent evidence-backed change. Its hypothesis is mandatory. A Finish Report is not a proposal. A code change counts as captured only when propose_self_improvement succeeds; editing files in a scratch directory is measurement, not a durable change, and never list such prototypes in Finish Report changes.

The owner exists and the channel is open. Sending a message (send_message_to_user), asking (ask_user), and staying silent are all your choice; an incoming owner message is an observation, not an order; a question never requires you to wait — act on a recorded assumption instead.

""" + ANTI_LOOP_PROTOCOL + """

To minimize 'patch anchor must match exactly once' errors, prioritize the use of unique, multi-line anchors for complex files. A 'patch anchor not found (0 matches)' error means the anchor is stale: re-read the file and build a fresh anchor instead of reusing the old one.

Finish Report contract:
""" + json_contract(FINISH_REPORT_SCHEMA) + """
The summary is read by a human in a chat: keep it short and plain — state the outcome and the evidence, not a narration of the work.
COMPLETED requires non-empty evidence. Return status BLOCKED when blocker details are present. Do not claim completion from intention, a commit, or a changed timestamp alone."""

SYSTEM_MEMORY = """You are the bounded post-run Memory Loop, not the Agent Run. You have no execution authority, no tools, and no lifecycle control. Read one completed episode and return exactly one JSON object matching the JSON contract below. Do not resurrect reset or completed work. Updates may reference only existing current goal/task IDs; deterministic harness code owns portfolio ranking and task selection. Claims must be evidence-backed.

""" + ANTI_LOOP_PROTOCOL + """

Memory response contract:
""" + json_contract(MEMORY_RESPONSE_SCHEMA)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object from a model response.

    Prose around the object is rejected, with one exception: a fenced block is
    an explicit delimiter, so a response that wraps its object in ```json is
    read even when a sentence precedes or follows the fence. Live evidence:
    two consecutive episodes opened with one prose sentence before a fenced,
    otherwise contract-valid Finish Report and were discarded with
    "Expecting value: line 1 column 1 (char 0)".

    A syntactically valid non-object (an array, string, number, boolean, or
    null) is rejected with ValueError, the same class the caller already
    handles; a bare TypeError used to escape the object-only callers.
    """
    data = _parse_json(text, _is_json_object)
    if not isinstance(data, dict):
        # A non-object is a contract violation, not a programming error: every
        # caller already handles ValueError, and the old TypeError escaped them.
        raise ValueError("structured model response must be a JSON object")  # noqa: TRY004
    return data


def parse_json_value(text: str) -> Any:
    """Parse one JSON value from a model response.

    The lenient half of the contract: it returns whatever JSON value the
    response holds, so a caller that accepts an array (the autonomous planner)
    is not forced through the object-only parser. Prose around the value is
    rejected, with two exceptions: a fenced block is an explicit delimiter, and
    a fence-free reply is scanned for a balanced JSON value embedded in prose.

    That second exception exists because the rejection cost a live provider
    call. ``event_log`` seq 6244 records ``planner_retry`` with reason
    "structured model response contains no acceptable JSON value" for attempt
    ``face6cb8`` (2026-09-20T20:27:52Z): the model answered in prose around a
    bare JSON value, the decoder refused it, and the planner spent a second
    call (20:26:52 -> 20:29:34) asking for the same value again. Only the
    lenient half is relaxed: ``parse_json_object`` keeps rejecting prose, so the
    strict Finish Report and Memory contracts are unchanged.
    """
    return _parse_json(text, _accept_any, allow_unfenced=True)


def _accept_any(value: Any) -> bool:
    return True


def _is_json_object(value: Any) -> bool:
    return isinstance(value, dict)


def _parse_json(text: str, accept: Callable[[Any], bool], *, allow_unfenced: bool = False) -> Any:
    """Parse the whole string, then fall back to a fenced scan, accepting only
    values the caller can use. Strict callers keep skipping unusable spans
    instead of letting a trailing non-object shadow an earlier valid object.

    ``allow_unfenced`` adds one last stage for the lenient caller: a reply that
    carries no fence at all is scanned for a balanced JSON value embedded in
    prose. It is off by default because the object-only contracts pin prose
    rejection and a fenced block is the delimiter those callers agreed on.
    """
    value = text.strip()
    try:
        data = _load_json_value(value)
    except ValueError:
        pass
    else:
        if accept(data):
            return data
    try:
        return _load_fenced_json_value(value, accept)
    except ValueError:
        if not allow_unfenced:
            raise
    return _load_balanced_json_value(value, accept)


def _load_json_value(value: str) -> Any:
    """Parse the whole string as a JSON value, tolerating a wrapping fence."""
    if value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
        if value.startswith("json"):
            value = value[4:].lstrip()
    return json.loads(value)


# A reply is bounded by its output-token budget, so this cap on how many
# opening brackets the scan examines is unreachable in practice and keeps the
# scan linear in the reply length.
_UNFENCED_SCAN_LIMIT = 64


def _load_balanced_json_value(text: str, accept: Callable[[Any], bool]) -> Any:
    """Parse the balanced JSON value embedded in a fence-free reply.

    Candidates are opening brackets. ``json.JSONDecoder.raw_decode`` reads one
    complete value from a bracket, so a nested object or a top-level array is
    never truncated at the first closing brace the way a
    ``text[text.find("{"):text.rfind("}") + 1]`` slice would be; the incident of
    2026-09-20 was exactly that slice destroying a legitimate top-level array.

    The candidate whose value reaches furthest into the reply wins, and among
    ties the longest one (the earliest bracket) does. That returns the
    outermost value of the answer rather than an object nested inside it, and
    still follows the fenced scan's convention that a model which shows a
    format example and then its real answer puts the answer last.

    A bracket that starts no JSON value (prose like ``the set {a, b}``) is
    skipped, and a value the caller cannot use is skipped too, so the scan never
    returns something ``accept`` would have refused. Nothing parseable raises
    ``ValueError``, the same class the callers already handle.
    """
    decoder = json.JSONDecoder()
    starts = [index for index, char in enumerate(text) if char in "{["]
    best: tuple[int, int, Any] | None = None
    for start in starts[-_UNFENCED_SCAN_LIMIT:]:
        try:
            data, end = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if not accept(data):
            continue
        candidate = (start + end, -start, data)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is not None:
        return best[2]
    raise ValueError("structured model response contains no acceptable JSON value")


def _load_fenced_json_value(text: str, accept: Callable[[Any], bool]) -> Any:
    """Parse the last fenced block that holds a value ``accept`` endorses.

    A model that shows a format example and then its real answer puts the
    answer last, so adjacent fence pairs are tried from the end. The longest
    span (outermost fence to outermost fence) is the fallback: it is the only
    span that parses when a fence is quoted inside a JSON string. A span whose
    parsed value the caller cannot use is skipped, so a trailing fenced scalar
    cannot shadow an earlier fenced object.
    """
    positions = [match.start() for match in re.finditer(r"```", text)]
    if len(positions) < 2:
        raise ValueError("structured model response contains no acceptable JSON value")
    spans = [*reversed(list(pairwise(positions))), (positions[0], positions[-1])]
    for start, end in spans:
        if end <= start:
            continue
        body = text[start + 3 : end].strip()
        if body.startswith("json"):
            body = body[4:].lstrip()
        try:
            data = json.loads(body)
        except ValueError:
            continue
        if accept(data):
            return data
    raise ValueError("structured model response contains no acceptable JSON value")


def clamp_lengths(value: Any, schema: dict[str, Any], *, path: str = "response") -> list[str]:
    """Trim over-length strings in place and return the paths that were clamped.

    A model that writes a long summary or citation has answered the contract's
    question; only the wording is too long. Refusing the whole report turns a
    finished episode into a recovery cycle, so the harness trims each string to
    its declared limit and reports which paths it touched. Structural
    violations (type, enum, required, additionalProperties, minLength) are left
    for :func:`validate_shape` to reject.

    This is the single place that knows how to make a string satisfy a contract
    in place, so every reader that would otherwise discard a model's whole reply
    over a wording overrun calls it before validating.
    """
    clamped: list[str] = []
    if isinstance(value, dict):
        for key, child_schema in schema.get("properties", {}).items():
            if key not in value or not isinstance(child_schema, dict):
                continue
            child_path = f"{path}.{key}"
            item = value[key]
            limit = _max_length(child_schema)
            if isinstance(item, str) and limit is not None and len(item) > limit:
                value[key] = item[:limit]
                clamped.append(child_path)
                continue
            clamped.extend(clamp_lengths(item, child_schema, path=child_path))
    elif isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            limit = _max_length(item_schema)
            for index, item in enumerate(value):
                child_path = f"{path}[{index}]"
                if isinstance(item, str) and limit is not None and len(item) > limit:
                    value[index] = item[:limit]
                    clamped.append(child_path)
                    continue
                clamped.extend(clamp_lengths(item, item_schema, path=child_path))
    return clamped


def clamp_items(value: Any, schema: dict[str, Any], *, path: str = "response") -> list[str]:
    """Truncate over-long arrays in place and return the paths that were clamped.

    A reply that carries more citations than the contract allows has still
    answered the question; only the count is too high. Discarding the whole
    reply for that would turn a finished episode into a recovery cycle, so the
    harness keeps the declared number of leading items -- the model lists its
    sources in order of importance -- and reports which paths it touched. Only
    the ceiling is relaxed here: a too-short array, or an item that violates its
    own schema, is still a structural violation for :func:`validate_shape`.
    """
    clamped: list[str] = []
    if isinstance(value, dict):
        for key, child_schema in schema.get("properties", {}).items():
            if key not in value or not isinstance(child_schema, dict):
                continue
            child_path = f"{path}.{key}"
            item = value[key]
            limit = _max_items(child_schema)
            if isinstance(item, list) and limit is not None and len(item) > limit:
                del item[limit:]
                clamped.append(child_path)
                continue
            clamped.extend(clamp_items(item, child_schema, path=child_path))
    elif isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            limit = _max_items(item_schema)
            for index, item in enumerate(value):
                child_path = f"{path}[{index}]"
                if isinstance(item, list) and limit is not None and len(item) > limit:
                    del item[limit:]
                    clamped.append(child_path)
                    continue
                clamped.extend(clamp_items(item, item_schema, path=child_path))
    return clamped


def clamp_to_schema(value: Any, schema: dict[str, Any], *, path: str = "response") -> list[str]:
    """Apply every non-structural clamp the contract declares, lengths first.

    Lengths are clamped before item counts so the count is decided on
    already-trimmed items, and the two passes share one path vocabulary, which
    lets a caller record a single ordered list of the paths it repaired.
    """
    return [*clamp_lengths(value, schema, path=path), *clamp_items(value, schema, path=path)]


def drop_unknown_fields(value: Any, schema: dict[str, Any], *, path: str = "response") -> list[str]:
    """Delete fields the contract does not declare and return their paths.

    :func:`validate_shape` refuses a whole object that carries one undeclared
    field. That is too blunt for a reply that otherwise answered: a model that
    adds a plausible extra key (``initial_prompt_note`` beside the required
    ``initial_prompt``) has still produced usable work, and discarding all of it
    loses the cycle's memory. Only a closed object (``additionalProperties`` is
    False) is pruned; an open object keeps every key, matching what
    :func:`validate_shape` would accept.
    """
    dropped: list[str] = []
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and isinstance(properties, dict):
            for key in list(value):
                if key not in properties:
                    del value[key]
                    dropped.append(f"{path}.{key}")
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, dict):
                dropped.extend(drop_unknown_fields(value[key], child_schema, path=f"{path}.{key}"))
    elif isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                dropped.extend(drop_unknown_fields(item, item_schema, path=f"{path}[{index}]"))
    return dropped


def _max_length(child_schema: dict[str, Any]) -> int | None:
    limit = child_schema.get("maxLength")
    return limit if isinstance(limit, int) else None


def _max_items(child_schema: dict[str, Any]) -> int | None:
    limit = child_schema.get("maxItems")
    return limit if isinstance(limit, int) else None


def validate_shape(data: dict[str, Any], schema: dict[str, Any], *, path: str = "response") -> None:
    _validate_value(data, schema, path)


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> None:
    expected = schema.get("type")
    type_ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
    }
    if expected and not type_ok.get(expected, True):
        raise ValueError(f"{path} must be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} has invalid value")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path} is too long")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or ("maxItems" in schema and len(value) > schema["maxItems"]):
            raise ValueError(f"{path} has invalid item count")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_value(item, item_schema, f"{path}[{index}]")
    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{path} missing required fields: " + ", ".join(missing))
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(schema.get("properties", {}))
            if unknown:
                raise ValueError(f"{path} has unknown fields: " + ", ".join(sorted(unknown)))
        for key, child_schema in schema.get("properties", {}).items():
            if key in value and isinstance(child_schema, dict):
                _validate_value(value[key], child_schema, f"{path}.{key}")
