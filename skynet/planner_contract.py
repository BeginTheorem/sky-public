"""The model-facing half of the autonomous planner: prompt text and reply decoding.

This module is deliberately **not** in the
``self_improvement.GATE_PROTECTED_PATHS`` warn-set. It is the *instrument* the planner
uses to talk to the model — the system prompt, the payload instruction, the
repair instruction and the text-to-value decoder. The 2026-09-20 incident is
the evidence for why an instrument must stay repairable by the organism itself:
the decoder destroyed a legitimate top-level JSON array and 15 of 18 planner
attempts failed for a day, while the empty portfolio it returned was
indistinguishable from "nothing to do".

It carries no decision logic. What counts as a valid proposal, what becomes
work, and every dedup/fingerprint/learnability/goal check stay in the
warn-set ``skynet.autonomous_planner`` module. ``parse_planner_reply`` only reshapes
text into a JSON value; it never marks anything valid.
"""

from __future__ import annotations

from typing import Any

from .model_contracts import PLANNER_RESPONSE_SCHEMA, json_contract, parse_json_value

PLANNER_INSTRUCTION = """Return exactly one JSON object matching the planner response contract. Generate at most 3 bounded proposals for active goals.
Each proposal must contain goal_id, title, problem, hypothesis, expected_new_fact, validation, scope (list), kind.
The proposal must be useful without user input, have observable validation, and be smaller than a broad project.
Do not repeat completed, blocked, exhausted, pending, or rejected work. Do not generate numbered pass/iteration/cycle variants.
Do not execute tools or describe tool calls. Allowed kind values: engineering, research, validation, recovery, observation, self_improvement.
For kind=research, set inspiration_ref to the external source (arXiv id, repository URL, or page URL) and make expected_new_fact the distilled, testable claim taken from it; a research proposal without a source will be rejected.
Prefer proposals that occupy an empty descriptor cell: the payload lists cell_coverage, and an idea in an unexplored subsystem/change-type/evidence-source combination outranks a third variation of an already-worked one.
A proposal may name parent_idea_id to develop an archived idea instead of starting from nothing."""

PLANNER_SYSTEM_PROMPT = "You are SkyNet's bounded autonomous planner. The harness owns execution. Return exactly one JSON object matching this schema: "

PLANNER_RETRY_INSTRUCTION = """Your previous reply was not a single JSON object matching the schema; return exactly one JSON object.
Do not wrap it in prose, do not return a list, and do not add fields outside the contract. Allowed kind values: engineering, research, validation, recovery, observation, self_improvement."""


def planner_system_prompt() -> str:
    """Render the planner system prompt with the response schema inline.

    The schema rendering lives here, next to the prompt it belongs to, so the
    instrument has exactly one owner and the protected planner only consumes it.
    """
    return PLANNER_SYSTEM_PROMPT + json_contract(PLANNER_RESPONSE_SCHEMA)


def parse_planner_reply(text: str) -> Any:
    """Decode a raw model reply into a JSON value, without judging it.

    The model legitimately returns a bare array of proposals, so a top-level
    list is normalized into ``{"proposals": [...]}``. Anything else is returned
    unchanged: a scalar, a malformed object or a schema-violating object all
    survive this function and are refused later by the protected validator. That
    separation is the point — a repairable decoder cannot mark work valid. A
    refusal still carries a bounded preview of the reply, so the text a decode
    failure must be judged against is not lost; see ``_describe_decode_failure``.
    """
    try:
        data = parse_json_value(text)
    except ValueError as exc:
        raise _describe_decode_failure(text, exc) from exc
    if isinstance(data, list):
        data = {"proposals": data}
    return data


def _describe_decode_failure(text: str, exc: ValueError) -> ValueError:
    """``exc`` with a bounded preview of ``text`` appended to its message.

    The reply is the evidence a decode failure is judged against, and it was
    recorded nowhere: the 2026-09-21T01:25:27Z retry failure (attempt
    ``4216e797``) left only "structured model response contains no acceptable
    JSON value", so prose and a reply cut off at the output ceiling were
    indistinguishable after the fact — the ``finish_reason`` telemetry that
    could tell them apart only landed with commit 018d2be, after that failure.
    A preview also makes truncation self-evident, since a reply that ends
    mid-token announces itself where a bare message does not.

    A ``ValueError`` is returned rather than raised so the caller keeps the
    ``raise`` and the control flow stays in one place. The original message is
    retained verbatim as a prefix: the planner records ``str(exc)`` as the
    retry reason (``autonomous_planner.generate``), and that contract is
    unchanged in kind.
    """
    return ValueError(f"{exc} (reply preview: {text[:200]!r}, {len(text)} chars)")
