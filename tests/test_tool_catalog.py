"""Tiered tool schemas: the block shrinks, every tool stays reachable.

The saving this exists for is measured (2026-10-18/19: 52 schemas were 17,956 tokens on
every request, 30% of the epoch's prompt tokens, while a run called a median of seven
tools and 21 of the 52 were never called). These tests pin the other half of the bargain:
a stub carries the name and the argument names, `describe_tool` answers with the real
schema for anything in the registry, and a sense server this module does not know is left
full rather than guessed at.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from skynet.models import StartEnvelope
from skynet.tool_catalog import (
    INVOKE_TOOL_NAME,
    DescribeTool,
    InvokeTool,
    ToolSurface,
    argument_names,
    armed_families,
    compact_signature,
    intention_text,
    render_catalogue,
    select_tool_schemas,
    sense_family,
    stub_schema,
    tool_catalog_enabled,
    tool_surface,
)


class _FakeClient:
    """The one attribute the catalog reads off an MCP tool."""

    def __init__(self, server_name: str) -> None:
        self.server_name = server_name


class _SenseTool:
    """Stands in for `skynet.mcp.MCPTool`: a tool that belongs to a server."""

    def __init__(
        self,
        name: str,
        server: str,
        *,
        description: str = "MCP tool",
        properties: tuple[str, ...] = (),
        required: tuple[str, ...] = (),
        padding: int = 0,
    ) -> None:
        self.name = name
        self.description = description
        self.client = _FakeClient(server)
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {key: {"type": "string", "description": "x" * padding} for key in properties},
            "required": list(required),
            "additionalProperties": False,
        }

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        return {"ok": True}


class _HouseTool:
    """A builtin: no client, so no sense family, so never a stub."""

    def __init__(self, name: str = "house_tool") -> None:
        self.name = name

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "house tool",
                "parameters": {"type": "object", "properties": {"action": {"type": "string"}}},
            },
        }

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        return {"ok": True}


def _registry(*, with_describe: bool = True) -> dict[str, Any]:
    registry: dict[str, Any] = {
        "house": _HouseTool("house"),
        "search_papers": _SenseTool(
            "search_papers",
            "arxiv",
            description="Search arXiv for papers\nsecond line that must not travel",
            properties=("query", "max_results"),
            required=("query",),
            padding=3000,
        ),
        "list_issues": _SenseTool(
            "list_issues",
            "github",
            properties=("owner", "repo"),
            required=("owner", "repo"),
            padding=1500,
        ),
        "search": _SenseTool("search", "ddg", properties=("query",), required=("query",)),
    }
    if with_describe:
        # Both halves of the escape hatch have to exist for a catalogue to be honest:
        # `describe_tool` to hand over the schema, `invoke_tool` to make the call.
        registry["describe_tool"] = _HouseTool("describe_tool")
        registry[INVOKE_TOOL_NAME] = InvokeTool()
    return registry


def _by_name(schemas: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {schema["function"]["name"]: schema for schema in schemas}


def _size(schema: dict[str, Any]) -> int:
    return len(json.dumps(schema, ensure_ascii=False))


class SchemaSizeTests(unittest.TestCase):
    def test_a_stub_is_a_fraction_of_the_full_schema(self) -> None:
        tool = _registry()["search_papers"]
        stub = stub_schema(tool)
        self.assertLess(_size(stub), _size(tool.schema) / 4)

    def test_a_stub_keeps_the_name_and_the_argument_names(self) -> None:
        stub = stub_schema(_registry()["list_issues"])
        self.assertEqual(stub["function"]["name"], "list_issues")
        self.assertIn("owner, repo", stub["function"]["description"])
        self.assertIn("describe_tool", stub["function"]["description"])

    def test_a_stub_only_carries_the_first_description_line(self) -> None:
        stub = stub_schema(_registry()["search_papers"])
        self.assertIn("Search arXiv for papers", stub["function"]["description"])
        self.assertNotIn("second line", stub["function"]["description"])

    def test_required_arguments_are_listed_first(self) -> None:
        schema = _registry()["search_papers"].schema
        self.assertEqual(argument_names(schema), ["query", "max_results"])
        self.assertEqual(compact_signature(schema), "query, max_results")

    def test_a_schema_without_properties_says_so(self) -> None:
        self.assertEqual(compact_signature({"type": "function", "function": {"parameters": {}}}), "no arguments")


class SelectionTests(unittest.TestCase):
    def test_house_tools_always_travel_full(self) -> None:
        schemas = _by_name(select_tool_schemas(_registry(), "refactor the memory store"))
        self.assertEqual(schemas["house"], _registry()["house"].schema)

    def test_an_unasked_sense_family_is_named_in_the_catalogue_not_sent(self) -> None:
        surface = tool_surface(_registry(), "refactor the memory store", enabled=True)
        names = [schema["function"]["name"] for schema in surface.schemas]
        self.assertNotIn("search_papers", names, "an unarmed sense costs nothing as a schema entry")
        self.assertIn("search_papers", surface.catalogue)
        self.assertIn("arxiv sense:", surface.catalogue)

    def test_an_asked_family_travels_full_and_the_rest_go_to_the_catalogue(self) -> None:
        surface = tool_surface(
            _registry(), "Read a paper beyond its abstract and distil a decision", enabled=True
        )
        schemas = _by_name(surface.schemas)
        self.assertEqual(schemas["search_papers"], _registry()["search_papers"].schema)
        self.assertNotIn("list_issues", schemas)
        self.assertIn("list_issues", surface.catalogue)

    def test_the_generic_outward_look_keeps_every_sense(self) -> None:
        schemas = _by_name(
            select_tool_schemas(
                _registry(),
                "Consult one external source (arXiv, GitHub, or the open web) about a concrete bottleneck",
            )
        )
        for name in ("search_papers", "list_issues", "search"):
            self.assertEqual(schemas[name], _registry()[name].schema)

    def test_an_unknown_server_is_never_stubbed(self) -> None:
        registry = {"mystery": _SenseTool("mystery", "unknown-server", padding=300)}
        schemas = _by_name(select_tool_schemas(registry, "nothing relevant here"))
        self.assertEqual(schemas["mystery"], registry["mystery"].schema)

    def test_every_tool_stays_reachable_exactly_once(self) -> None:
        """Reachability, not presence: a catalogued sense is named once and still callable."""
        registry = _registry()
        surface = tool_surface(registry, "something", enabled=True)
        names = [schema["function"]["name"] for schema in surface.schemas]
        self.assertEqual(len(names), len(set(names)))
        catalogued = [name for name in registry if name not in names]
        self.assertEqual(sorted(catalogued), ["list_issues", "search", "search_papers"])
        for name in catalogued:
            self.assertEqual(surface.catalogue.count(f"- {name} ("), 1, f"{name} must be named once")
            self.assertIn(name, registry, "and still present in the registry it is called through")
        for name in names:
            self.assertIn(name, registry)

    def test_no_escape_hatch_means_no_catalogue(self) -> None:
        """A catalogue whose calls cannot be dispatched would trade a capability for tokens."""
        registry = _registry(with_describe=False)
        surface = tool_surface(registry, "refactor the memory store", enabled=True)
        self.assertEqual(surface.catalogue, "")
        schemas = _by_name(surface.schemas)
        for name, tool in registry.items():
            self.assertEqual(schemas[name], tool.schema, name)

    def test_disabled_tiering_is_the_unmodified_request(self) -> None:
        registry = _registry()
        self.assertEqual(
            select_tool_schemas(registry, "Read a paper", enabled=False),
            [tool.schema for tool in registry.values()],
        )

    def test_sense_family_reads_the_client_and_nothing_else(self) -> None:
        self.assertEqual(sense_family(_registry()["search_papers"]), "arxiv")
        self.assertIsNone(sense_family(_HouseTool()))
        self.assertIsNone(sense_family(object()))


class CatalogueTests(unittest.TestCase):
    def test_the_catalogue_lists_only_unarmed_senses(self) -> None:
        text = render_catalogue(_registry(), armed={"arxiv"})
        self.assertNotIn("- search_papers (", text)
        self.assertIn("- list_issues (", text)
        self.assertIn("- search (", text)
        self.assertIn(INVOKE_TOOL_NAME, text, "the catalogue has to say how a listed tool is called")

    def test_the_catalogue_is_empty_when_everything_is_armed(self) -> None:
        self.assertEqual(render_catalogue(_registry(), armed={"arxiv", "github", "ddg"}), "")

    def test_the_catalogue_is_cheaper_than_one_schema(self) -> None:
        registry = _registry()
        text = render_catalogue(registry, armed=set())
        self.assertLess(len(text), _size(registry["search_papers"].schema))

    def test_the_system_suffix_is_empty_or_separated(self) -> None:
        self.assertEqual(ToolSurface([], "").system_suffix(), "")
        self.assertEqual(ToolSurface([], "SENSE CATALOGUE").system_suffix(), "\n\nSENSE CATALOGUE")

    def test_armed_families_reads_the_words(self) -> None:
        self.assertEqual(armed_families("Read a paper beyond its abstract"), {"arxiv"})
        self.assertEqual(armed_families("refactor the memory store"), set())


class InvokeToolTests(unittest.TestCase):
    def test_its_schema_asks_for_a_name_and_arguments(self) -> None:
        schema = InvokeTool().schema
        self.assertEqual(schema["function"]["name"], INVOKE_TOOL_NAME)
        self.assertEqual(schema["function"]["parameters"]["required"], ["name"])
        self.assertIn("arguments", schema["function"]["parameters"]["properties"])

    def test_reaching_execute_is_an_error_not_a_pretend_call(self) -> None:
        """The runner resolves it before dispatch; a call that gets here bypassed that."""
        result = InvokeTool().execute({"name": "search_papers"}, idempotency_key="k")
        self.assertFalse(result["ok"])
        self.assertIn(INVOKE_TOOL_NAME, str(result["error"]))


class IntentionTextTests(unittest.TestCase):
    def test_the_intention_and_pending_titles_are_the_scope(self) -> None:
        envelope = StartEnvelope(
            "test",
            intention={"title": "Fix the planner", "why_now": "it stalls"},
            pending_work=[{"title": "second task"}],
        )
        text = intention_text(envelope)
        self.assertIn("Fix the planner", text)
        self.assertIn("it stalls", text)
        self.assertIn("second task", text)

    def test_observations_do_not_pull_a_family_in(self) -> None:
        # An envelope's observations and memories are not what the run is about; letting a
        # stray word in them select a whole sense family would undo the saving invisibly.
        envelope = StartEnvelope(
            "test",
            intention={"title": "Fix the planner", "why_now": "it stalls"},
            observations=[{"text": "the goal mentions arXiv and GitHub"}],
        )
        self.assertNotIn("arXiv", intention_text(envelope))

    def test_an_empty_envelope_yields_an_empty_scope(self) -> None:
        self.assertEqual(intention_text(StartEnvelope("test")), "")
        self.assertEqual(intention_text(object()), "")


class DescribeToolTests(unittest.TestCase):
    def test_it_returns_the_real_schema_json(self) -> None:
        registry = _registry()
        result = DescribeTool(registry).execute({"name": "search_papers"}, idempotency_key="k")
        self.assertTrue(result["ok"])
        self.assertEqual(json.loads(str(result["result"])), registry["search_papers"].schema)

    def test_it_answers_for_a_house_tool_too(self) -> None:
        registry = _registry()
        result = DescribeTool(registry).execute({"name": "house"}, idempotency_key="k")
        self.assertTrue(result["ok"])

    def test_an_unknown_name_is_a_structured_error_with_suggestions(self) -> None:
        result = DescribeTool(_registry()).execute({"name": "paper"}, idempotency_key="k")
        self.assertFalse(result["ok"])
        self.assertIn("paper", str(result["similar"]))

    def test_its_own_schema_requires_a_name(self) -> None:
        schema = DescribeTool({}).schema
        self.assertEqual(schema["function"]["name"], "describe_tool")
        self.assertEqual(schema["function"]["parameters"]["required"], ["name"])


class EnablementTests(unittest.TestCase):
    def test_the_default_is_on(self) -> None:
        self.assertTrue(tool_catalog_enabled({}))

    def test_the_off_switches_are_honoured(self) -> None:
        for value in ("0", "false", "FALSE", "no", "off"):
            self.assertFalse(tool_catalog_enabled({"SKYNET_TOOL_CATALOG": value}))

    def test_anything_else_leaves_it_on(self) -> None:
        self.assertTrue(tool_catalog_enabled({"SKYNET_TOOL_CATALOG": "1"}))
        self.assertTrue(tool_catalog_enabled({"SKYNET_TOOL_CATALOG": "yes"}))


if __name__ == "__main__":
    unittest.main()
