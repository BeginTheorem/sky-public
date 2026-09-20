"""Static contracts shared by the model-facing execution phases."""

from __future__ import annotations

import json
import re
from itertools import pairwise
from typing import Any

from .memory_store import MEMORY_KINDS

FINISH_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["COMPLETED", "BLOCKED", "FAILED", "NEEDS_RECOVERY"]},
        "summary": {"type": "string", "minLength": 1, "maxLength": 1200, "description": "One short plain-language paragraph stating what changed and why, in English; no step-by-step reasoning, no restatement of the episode. At most 1200 characters."},
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
        "task_updates": {"type": "array", "items": {"type": "object", "properties": {"task_id": {"type": "string", "minLength": 1}, "status": {"type": "string"}, "outcome": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}}, "required": ["task_id"], "additionalProperties": False}},
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

The lifecycle is wake -> perceive -> deterministic planning when needed -> one bounded ReAct episode -> Finish -> Memory Loop -> evaluation -> checkpoint -> sleep. The model does not own this lifecycle. Non-MCP tools are bash, webfetch, read, grep, db, and memory (search/remember/forget/pin/unpin/correct), plus the always-available owner tools ask_user and send_message_to_user; memory.search goes deeper than the small automatic injection, and memory.correct supersedes a wrong memory with cited evidence; propose_self_improvement, promote_self_improvement, and request_rollback appear only when the runtime has a Git root. Configured MCP servers may add external senses; their schemas and availability are runtime-provided and must not be assumed, but when present they are the organism's boundary with the world: arxiv (search_papers, get_abstract, download_paper, read_paper, semantic_search, citation_graph, get_paper_latex_section, watch_topic, check_alerts), github read-only (search_code, search_repositories, search_commits, search_issues, get_file_contents, get_repository_tree, list_commits, issue_read, pull_request_read), ddg (search, fetch_content), playwright (browser_*), plus the builtin webfetch. Never invent a tool that is not in the supplied list. A hypothesis may come from your own code and runtime state, or from outside them: a scientific paper, another project's source, a documented failure mode. The second source is not a fallback — when internal evidence is exhausted, reading is work, and a research task without a citation is not research. When work is needed, identify one evidence-backed bottleneck and one bounded hypothesis. Work only inside the repository root and prefer relevant code, tests, runtime state, and the existing improvement workflow; when those yield nothing new, prefer an external source over sleeping.

Every investigation must serve the selected work or a stated validation hypothesis. A non-empty report is not evidence of success.

""" + ANTI_LOOP_PROTOCOL

SYSTEM_REACT = """You are inside one bounded ReAct episode. Work through Initial, Common, and Finish. In Initial, understand the selected work and state the expected new fact. In Common, use only currently supplied tools and evaluate evidence. In Finish, return exactly one JSON object matching the Finish Report contract below. Do not control lifecycle, schedule work, or emit orchestration commands; the harness owns orchestration.

When the selected work names an external source, consult it before proposing anything: search, read the primary section, and distil. Record the citation in the Finish Report citations. An uncited external claim is an invention from your weights and must be labelled as a hypothesis, never as a finding.

Use propose_self_improvement only for the smallest coherent evidence-backed change. Its hypothesis is mandatory. A Finish Report is not a proposal. A code change counts as captured only when propose_self_improvement (or promote_self_improvement) succeeds; editing files in a scratch directory is measurement, not a durable change, and never list such prototypes in Finish Report changes.

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
    read even when a sentence precedes or follows the fence. Without this
    exception, a response that prepends prose to a fenced, otherwise
    contract-valid object is discarded with a JSON decoding error.
    """
    value = text.strip()
    try:
        return _load_json_object(value)
    except ValueError:
        pass
    return _load_fenced_json_object(value)


def _load_json_object(value: str) -> dict[str, Any]:
    """Parse the whole string as a JSON object, tolerating a wrapping fence."""
    if value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
        if value.startswith("json"):
            value = value[4:].lstrip()
    data = json.loads(value)
    if not isinstance(data, dict):
        raise TypeError("structured model response must be a JSON object")
    return data


def _load_fenced_json_object(text: str) -> dict[str, Any]:
    """Parse the last fenced block that holds a JSON object.

    A model that shows a format example and then its real answer puts the
    answer last, so adjacent fence pairs are tried from the end. The longest
    span (outermost fence to outermost fence) is the fallback: it is the only
    span that parses when a fence is quoted inside a JSON string.
    """
    positions = [match.start() for match in re.finditer(r"```", text)]
    if len(positions) < 2:
        raise ValueError("structured model response must be a JSON object")
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
        if isinstance(data, dict):
            return data
    raise ValueError("structured model response must be a JSON object")


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
