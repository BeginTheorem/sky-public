"""Tiered tool surfaces: the long tail rides as text, and a meta-call opens it.

Measured on her real registry (36 sense tools, 1,748 tool-carrying requests in the epoch):
full schemas 38,457 chars ~ 12,018 tokens per request, compact stubs 14,111 ~ 4,410, a text
catalogue 3,485 ~ 1,089. The catalogue also stops the request growing by 400 chars every time
a sense is added.

Measured on the live ledger, epoch 2026-10-18/19 (a 74-run regression of ``prompt_tokens``
against ``request_chars``, cross-checked against a reconstruction of the runtime registry):
the 52 tool schemas cost **17,956 tokens on every request** -- 30% of all prompt tokens in
the epoch -- while a run called a median of seven distinct tools and 21 of the 52 were never
called at all. The three sense servers are 36 of those tools and ~38k characters of JSON
Schema; the heaviest single schema (``search_papers``) is 4.6k characters on its own.

So a request carries full schemas for the house tools, plus full schemas for a sense family
only when the episode's own intention names that family. Everything else travels as a
compact stub: the name, the first line of the description, the argument names, and a pointer
at ``describe_tool``, which returns the real schema on demand.

Nothing becomes unreachable. Reachability is the invariant this module exists to preserve
while it shrinks the block -- a smaller block that quietly dropped a capability would be a
worse bug than the tokens it saves -- so a sense server this module does not know stays
full, and ``describe_tool`` answers for every tool in the registry, stub or not.

Tiering is the default. ``SKYNET_TOOL_CATALOG=0`` restores the untiered request byte for
byte, which is how the tests pin both behaviours.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .provider import Tool

ToolSchema = dict[str, Any]

#: Which words in an episode's own intention ask for which sense server. Kept narrow on
#: purpose: a keyword that matches everything would restore the block this exists to shrink
#: and do it invisibly.
SENSE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "arxiv": ("arxiv", "preprint", "paper", "papers", "latex", "citation", "abstract"),
    "github": ("github", "repository", "repo", "commit", "pull request", "issue", "issues"),
    "ddg": ("open web", "the web", "web search", "search the web", "online", "internet", "news"),
}

#: How much of a stub's description survives. The first line is what a model routes on.
_STUB_DESCRIPTION_CHARS = 160
_STUB_ARGUMENT_LIMIT = 8

#: The meta-tool a catalogue call goes through. The provider only lets the model call functions
#: the request declared, so a tool with no schema cannot be called by name -- which is exactly
#: why Hermes invokes its deferred tools through `tool_call` rather than a raw name.
INVOKE_TOOL_NAME = "invoke_tool"


def _mentions(text: str, words: Sequence[str]) -> bool:
    """Word-boundary match for single words, plain substring for phrases."""
    for word in words:
        if " " in word:
            if word in text:
                return True
        elif re.search(rf"\b{re.escape(word)}\b", text):
            return True
    return False


def sense_family(tool: object) -> str | None:
    """The MCP server a tool belongs to, or ``None`` for a house tool.

    Same probe the planning toolset uses to pick its sensors, so the two stay consistent
    about what a "sense" is.
    """
    client = getattr(tool, "client", None)
    server = str(getattr(client, "server_name", "") or "").strip().lower()
    return server or None


def _parameters(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return {}
    params = function.get("parameters")
    return params if isinstance(params, Mapping) else {}


def argument_names(schema: Mapping[str, Any]) -> list[str]:
    """Required arguments first, then the optional ones, each in schema order."""
    params = _parameters(schema)
    properties = params.get("properties")
    if not isinstance(properties, Mapping):
        return []
    names = [str(name) for name in properties]
    required = [str(name) for name in (params.get("required") or []) if str(name) in names]
    return required + [name for name in names if name not in required]


def compact_signature(schema: Mapping[str, Any], *, limit: int = _STUB_ARGUMENT_LIMIT) -> str:
    """``query, max_results, ...`` -- enough for a model to try a call without the schema."""
    names = argument_names(schema)[:limit]
    if not names:
        return "no arguments"
    if len(argument_names(schema)) > limit:
        return ", ".join(names) + ", ..."
    return ", ".join(names)


def stub_schema(tool: Tool) -> ToolSchema:
    """The compact stand-in for one tool: name, first description line, argument names."""
    full = tool.schema
    function = full.get("function") if isinstance(full, Mapping) else None
    function = function if isinstance(function, Mapping) else {}
    name = str(function.get("name") or tool.name)
    description = str(function.get("description") or "tool").strip().splitlines()
    first_line = (description[0] if description else "tool")[:_STUB_DESCRIPTION_CHARS]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                f"{first_line} [args: {compact_signature(full)}. "
                f"Compact stub: call describe_tool with name='{name}' for the full schema.]"
            ),
            "parameters": {"type": "object", "additionalProperties": True, "properties": {}},
        },
    }


def intention_text(start: object) -> str:
    """The episode's own words about what it is doing, and nothing else.

    Deliberately not the whole start envelope: an envelope carries memories, observations
    and goal titles, and a stray "search" in any of them would quietly pull a whole sense
    family back into the block. The intention is what the run is actually about.
    """
    parts: list[str] = []
    intention = getattr(start, "intention", None)
    if isinstance(intention, Mapping):
        parts.append(str(intention.get("title") or ""))
        parts.append(str(intention.get("why_now") or ""))
    pending = getattr(start, "pending_work", None)
    if isinstance(pending, Sequence) and not isinstance(pending, (str, bytes)):
        parts.extend(
            str(item.get("title") or "")
            for item in pending[:4]
            if isinstance(item, Mapping)
        )
    return " ".join(part for part in parts if part)


def armed_families(task_text: str) -> set[str]:
    """The sense families this episode's own words ask for, in full schema."""
    text = (task_text or "").casefold()
    return {family for family, words in SENSE_KEYWORDS.items() if _mentions(text, words)}


@dataclass(frozen=True, slots=True)
class ToolSurface:
    """What a request carries: full schemas for some tools, text for the rest."""

    schemas: list[ToolSchema]
    catalogue: str

    def system_suffix(self) -> str:
        """The catalogue as it rides in the system prompt, or an empty string."""
        return f"\n\n{self.catalogue}" if self.catalogue else ""


def render_catalogue(tools: Mapping[str, Tool], *, armed: set[str]) -> str:
    """One line per unarmed sense tool, grouped by server: the whole long tail as text.

    Hermes' own deferred catalogue is rendered exactly this way -- name and one line of
    description, grouped by namespace, with the schema fetched on demand -- and it is what
    makes a sense free to add: a line of text costs ~90 chars where its JSON Schema cost 1,000.
    """
    lines: list[str] = []
    by_server: dict[str, list[str]] = {}
    for name, tool in sorted(tools.items()):
        family = sense_family(tool)
        if family is None or family in armed or family not in SENSE_KEYWORDS:
            continue
        schema = tool.schema
        function = schema.get("function") if isinstance(schema, Mapping) else None
        function = function if isinstance(function, Mapping) else {}
        description = str(function.get("description") or "MCP tool").strip().splitlines()
        first_line = (description[0] if description else "MCP tool")[:110]
        by_server.setdefault(family, []).append(
            f"- {name} ({compact_signature(schema)}): {first_line}"
        )
    for server in sorted(by_server):
        lines.append(f"{server} sense:")
        lines.extend(sorted(by_server[server]))
    if not lines:
        return ""
    header = (
        "SENSE CATALOGUE -- tools not sent this episode. Call one with "
        f"{INVOKE_TOOL_NAME}(name=..., arguments={{...}}), or ask describe_tool(name) for its "
        "full schema first. Nothing here is unavailable; only its schema is not in this request."
    )
    return "\n".join([header, *lines])


def tool_surface(
    tools: Mapping[str, Tool],
    task_text: str,
    *,
    enabled: bool = True,
) -> ToolSurface:
    """The request-side surface: house tools full, armed senses full, the rest as text.

    Decided once per episode, from the episode's own intention: it does not change mid-run, and
    re-deciding per step would only make the request unstable for a provider's prefix cache.

    The catalogue is only produced when the registry can answer for it -- ``describe_tool`` and
    ``invoke_tool`` both present -- because a catalogue whose calls cannot be dispatched would
    trade a capability for tokens. That is the same reason an unknown sense server keeps its
    full schema.
    """
    dispatchable = "describe_tool" in tools and INVOKE_TOOL_NAME in tools
    if not enabled or not dispatchable:
        return ToolSurface([tool.schema for tool in tools.values()], "")
    armed = armed_families(task_text)
    schemas: list[ToolSchema] = []
    for tool in tools.values():
        family = sense_family(tool)
        if family is None or family in armed or family not in SENSE_KEYWORDS:
            schemas.append(tool.schema)
        # An unarmed sense gets no schema entry at all: the catalogue names it, and the call
        # goes through the meta-tool. Sending a stub as well would pay for both.
    return ToolSurface(schemas, render_catalogue(tools, armed=armed))


def select_tool_schemas(
    tools: Mapping[str, Tool],
    task_text: str,
    *,
    enabled: bool = True,
) -> list[ToolSchema]:
    """The schemas a request carries: the projection of ``tool_surface``."""
    return tool_surface(tools, task_text, enabled=enabled).schemas


class InvokeTool(Tool):
    """Call any tool by name: the way a catalogued sense is reached.

    The runner resolves this before dispatch, so the call it names travels the ordinary path --
    policy, effect memoisation, retries, telemetry -- and `invoke_tool` itself is never executed.
    If it ever reaches ``execute``, the caller bypassed that resolution, and saying so is better
    than pretending to be a tool.
    """

    name = INVOKE_TOOL_NAME
    capability_kind = "read"

    @property
    def schema(self) -> ToolSchema:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Invoke one of the tools listed in the sense catalogue, by name, with its "
                    "arguments. Ask describe_tool(name) first if you need the exact schema."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Tool name as listed in the catalogue."},
                        "arguments": {"type": "object", "description": "The tool's arguments."},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        return {
            "ok": False,
            "error": (
                f"{INVOKE_TOOL_NAME} was not resolved to a real tool; the runner resolves it "
                "before dispatch, so this call came from a path that does not."
            ),
        }


class DescribeTool(Tool):
    """Return the full JSON schema of one tool, so a compact stub stays usable.

    The registry holds every tool whether or not its schema travelled, and this answers for
    all of them: a tool whose schema was sent in full can still be described, which keeps
    the tool free of a "was this one stubbed" branch that would drift from the selection
    rule the moment either side changed.
    """

    name = "describe_tool"
    capability_kind = "read"

    def __init__(self, tools: Mapping[str, Any]) -> None:
        self._tools = tools

    @property
    def schema(self) -> ToolSchema:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Return the full JSON schema of one tool by name. The request carries compact "
                    "stubs for tool groups this episode did not ask for; call this before using one "
                    "of those so the arguments are the real ones instead of a guess."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Exact tool name as it appears in the request.",
                        }
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        wanted = str(arguments.get("name") or "").strip()
        tool = self._tools.get(wanted)
        if tool is None:
            similar = sorted(name for name in self._tools if wanted and wanted.casefold() in name.casefold())
            return {"ok": False, "error": f"unknown tool: {wanted}", "similar": similar[:8]}
        schema = getattr(tool, "schema", None)
        if schema is None:
            return {"ok": False, "error": f"tool has no schema: {wanted}"}
        return {"ok": True, "result": json.dumps(schema, ensure_ascii=False)}


def tool_catalog_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """``SKYNET_TOOL_CATALOG=0`` turns the tiering off; anything else leaves it on."""
    import os

    source = os.environ if environ is None else environ
    raw = str(source.get("SKYNET_TOOL_CATALOG", "1")).strip().casefold()
    return raw not in {"0", "false", "no", "off"}


__all__ = [
    "INVOKE_TOOL_NAME",
    "SENSE_KEYWORDS",
    "DescribeTool",
    "InvokeTool",
    "ToolSurface",
    "argument_names",
    "armed_families",
    "compact_signature",
    "intention_text",
    "render_catalogue",
    "select_tool_schemas",
    "sense_family",
    "stub_schema",
    "tool_catalog_enabled",
    "tool_surface",
]
