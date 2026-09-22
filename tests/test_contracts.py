"""Split from the former monolithic CoreTests suite."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from skynet.models import ModelTurn, RunStatus
from skynet.providers.errors import ProviderError
from skynet.reactor import Reactor, ReactorConfig


class CoreTests(unittest.TestCase):
    def test_parse_json_value_returns_arrays_and_scalars(self) -> None:
        from skynet.model_contracts import parse_json_value
        self.assertEqual(parse_json_value("[1, 2]"), [1, 2])
        self.assertEqual(parse_json_value("42"), 42)
        self.assertIsNone(parse_json_value("null"))
        self.assertEqual(parse_json_value('prose\n```json\n[1, 2]\n```'), [1, 2])

    def test_finish_and_memory_contracts_still_reject_non_objects(self) -> None:
        from skynet.memory import MemoryLoop
        from skynet.model_contracts import parse_json_object
        for text in ("[1, 2]", '"a string"', "42", "null", "true"):
            with self.assertRaises(ValueError):
                parse_json_object(text)
        # The Memory Loop parses through the same object-only contract.
        with self.assertRaises(ValueError):
            MemoryLoop._parse(ModelTurn(text="[1, 2]"))

    def test_provider_error_classification(self) -> None:
        self.assertFalse(ProviderError("auth", category="auth").retryable)
        self.assertTrue(ProviderError("busy", category="rate_limit", retryable=True).retryable)
    def test_finish_contract_rejects_missing_evidence(self) -> None:
        from skynet.model_contracts import FINISH_REPORT_SCHEMA, parse_json_object, validate_shape
        with self.assertRaises(ValueError):
            validate_shape(parse_json_object('{"status":"COMPLETED"}'), FINISH_REPORT_SCHEMA)
    def test_finish_summary_is_bounded_and_human_facing(self) -> None:
        from skynet.model_contracts import FINISH_REPORT_SCHEMA, SYSTEM_REACT
        summary = FINISH_REPORT_SCHEMA["properties"]["summary"]
        self.assertIn("summary", FINISH_REPORT_SCHEMA["required"])
        self.assertIsInstance(summary.get("maxLength"), int)
        self.assertTrue(summary.get("description"))
        # The Finish protocol must tell the model the summary is short and read by a human.
        self.assertIn("summary", SYSTEM_REACT)
        self.assertIn("short", SYSTEM_REACT)
        self.assertIn("human", SYSTEM_REACT)
    def test_planner_and_memory_contracts_reject_prose_wrappers(self) -> None:
        from skynet.model_contracts import parse_json_object
        with self.assertRaises(ValueError):
            parse_json_object('prefix {"proposals": []}')
    def test_fenced_finish_report_survives_a_prose_preamble(self) -> None:
        from skynet.model_contracts import parse_json_object
        # Regression (runs c1e074cb, 5116404c): one prose sentence before a
        # fenced, contract-valid Finish Report was rejected at char 0 and the
        # parser error was published instead of the report.
        preamble = 'Work is complete and verified. Returning the Finish Report.\n\n```json\n{"status": "COMPLETED", "summary": "ok"}\n```'
        self.assertEqual(parse_json_object(preamble), {"status": "COMPLETED", "summary": "ok"})
        trailer = 'Here is the report:\n```json\n{"status": "BLOCKED", "blocker": "x"}\n```\nHope that helps!'
        self.assertEqual(parse_json_object(trailer), {"status": "BLOCKED", "blocker": "x"})
        # A fence inside a JSON string must not cut the object short.
        quoted = 'Note:\n```json\n{"summary": "ran ```python\\nprint(1)\\n``` in the fix"}\n```'
        self.assertEqual(parse_json_object(quoted), {"summary": "ran ```python\nprint(1)\n``` in the fix"})
    def test_last_fenced_object_wins_over_a_format_example(self) -> None:
        from skynet.model_contracts import parse_json_object
        # A model that shows the format and then its answer puts the answer
        # last; returning the example would silently accept the wrong object.
        text = (
            'Here is the format:\n```json\n{"status": "COMPLETED", "summary": "example only"}\n```\n'
            'And the actual report:\n```json\n{"status": "BLOCKED", "summary": "real", "blocker": "x"}\n```'
        )
        self.assertEqual(parse_json_object(text), {"status": "BLOCKED", "summary": "real", "blocker": "x"})
    def test_unfenced_prose_and_non_json_fences_stay_rejected(self) -> None:
        from skynet.model_contracts import parse_json_object
        for text in (
            "I have finished thinking about it.",
            'prefix {"proposals": []}',
            "look:\n```python\nprint(1)\n```",
        ):
            with self.assertRaises(ValueError):
                parse_json_object(text)
        # A fenced JSON array is still not a JSON object; the object-only
        # contract rejects it with ValueError instead of leaking a TypeError
        # that the callers did not catch.
        with self.assertRaises(ValueError):
            parse_json_object("```json\n[1, 2]\n```")
    def test_parse_json_object_accepts_bare_and_fenced_objects(self) -> None:
        from skynet.model_contracts import parse_json_object
        self.assertEqual(parse_json_object('{"a": 1}'), {"a": 1})
        fenced = 'prose before\n```json\n{"a": 1}\n```\nprose after'
        self.assertEqual(parse_json_object(fenced), {"a": 1})

    def test_fenced_object_is_not_shadowed_by_a_later_fenced_scalar(self) -> None:
        from skynet.model_contracts import parse_json_object
        # Regression: the lenient fenced scan returned the first span that
        # parsed as ANY JSON value, so a trailing scalar hid an earlier object.
        text = '```json\n{"status": "COMPLETED", "summary": "real"}\n```\nExample:\n```json\n42\n```'
        self.assertEqual(parse_json_object(text), {"status": "COMPLETED", "summary": "real"})

    def test_fenced_object_is_not_shadowed_by_a_later_fenced_array(self) -> None:
        from skynet.model_contracts import parse_json_object
        text = '```json\n{"a": 1}\n```\nAnd as a list:\n```json\n[1, 2]\n```'
        self.assertEqual(parse_json_object(text), {"a": 1})

    def test_last_fenced_object_wins_among_objects(self) -> None:
        from skynet.model_contracts import parse_json_object
        text = '```json\n{"a": 1}\n```\nbetween\n```json\n{"b": 2}\n```\n```json\n[9]\n```'
        self.assertEqual(parse_json_object(text), {"b": 2})

    def test_parse_json_object_rejects_every_non_object(self) -> None:
        from skynet.model_contracts import parse_json_object
        for text in (
            "42",
            '"x"',
            "true",
            "null",
            "[1, 2]",
            "```json\n[1, 2]\n```",
            "I have no fence here",
        ):
            with self.assertRaises(ValueError):
                parse_json_object(text)

    def test_non_object_rejections_are_exactly_valueerror(self) -> None:
        from skynet.model_contracts import parse_json_object
        for text in ("42", '"x"', "true", "null", "[1, 2]", "```json\n[1, 2]\n```"):
            try:
                parse_json_object(text)
            except ValueError as exc:
                self.assertIs(type(exc), ValueError)
            else:
                self.fail(f"expected ValueError for {text!r}")

    def test_parse_json_value_stays_lenient(self) -> None:
        from skynet.model_contracts import parse_json_value
        self.assertEqual(parse_json_value("[1, 2]"), [1, 2])
        self.assertEqual(parse_json_value('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json_value("prose\n```json\n[1, 2]\n```"), [1, 2])
        self.assertEqual(parse_json_value("prose\n```json\n{\"a\": 1}\n```"), {"a": 1})
        with self.assertRaises(ValueError):
            parse_json_value("not json at all")

    def test_parse_json_value_scans_a_balanced_value_embedded_in_prose(self) -> None:
        from skynet.model_contracts import parse_json_value
        # The lenient planner decoder must not slice from the first `{` to the
        # last `}`: that destroyed a legitimate top-level array and a nested
        # object. The balanced scan returns the outermost embedded value.
        self.assertEqual(parse_json_value('Here is the answer: [{"a": 1}, {"b": 2}] hope it helps'), [{"a": 1}, {"b": 2}])
        self.assertEqual(parse_json_value('prose {"a": {"b": 1}} trailing'), {"a": {"b": 1}})
        # A bracket that starts no JSON value is skipped, not fatal.
        self.assertEqual(parse_json_value('the set {a, b} is not JSON, but [1, 2] is'), [1, 2])

    def test_unfenced_scan_does_not_relax_the_object_only_contract(self) -> None:
        from skynet.model_contracts import parse_json_object
        with self.assertRaises(ValueError):
            parse_json_object('prose {"a": 1} trailing')

    def test_clamp_lengths_truncates_only_over_length_strings(self) -> None:
        from skynet.model_contracts import FINISH_REPORT_SCHEMA, clamp_lengths
        report = {
            "status": "COMPLETED",
            "summary": "x" * 3000,
            "evidence": ["short", "y" * 900],
            "actions": [],
            "changes": [],
            "tests": [],
            "blocker": "",
            "next_hypothesis": "",
        }
        clamped = clamp_lengths(report, FINISH_REPORT_SCHEMA)
        self.assertIn("response.summary", clamped)
        self.assertEqual(len(report["summary"]), 1200)
        self.assertEqual(report["evidence"][0], "short", "in-range strings are untouched")
        self.assertEqual(report["evidence"][1], "y" * 900, "evidence items have no declared maxLength")

    def test_system_grants_owner_freedom_and_lists_real_tools(self) -> None:
        from skynet.model_contracts import ANTI_LOOP_PROTOCOL, SYSTEM, SYSTEM_MEMORY, SYSTEM_REACT
        # The old denial is gone; the owner channel is a choice, not a command.
        self.assertNotIn("There is no user to wait for", SYSTEM)
        self.assertIn("owner channel is open", SYSTEM)
        self.assertIn("observation, not an order", SYSTEM)
        for name in (
            "bash", "webfetch", "read", "grep", "db",
            "propose_self_improvement",
            "ask_user", "send_message_to_user", "request_rollback",
        ):
            self.assertIn(name, SYSTEM)
        # One anti-loop fragment, shared by all three phases.
        for prompt in (SYSTEM, SYSTEM_REACT, SYSTEM_MEMORY):
            self.assertIn(ANTI_LOOP_PROTOCOL, prompt)
    def test_integer_schema_rejects_string(self) -> None:
        from skynet.model_contracts import validate_shape
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"], "additionalProperties": False}
        validate_shape({"count": 3}, schema)
        with self.assertRaises(ValueError):
            validate_shape({"count": "3"}, schema)
        with self.assertRaises(ValueError):
            validate_shape({"count": True}, schema)
    def test_model_finish_report_does_not_mutate_orchestration_state(self) -> None:
        class ReportProvider:
            def complete(self, messages, *, max_tokens, tools=()):
                if len(messages) == 2:
                    return ModelTurn(text=json.dumps({"status": "completed", "new_goals": [{"title": "must not be applied"}], "evaluation": {"value_estimate": 1.0}}))
                return ModelTurn(text="plain report")

        with tempfile.TemporaryDirectory() as directory:
            reactor = Reactor(ReportProvider(), {}, ReactorConfig(state_path=Path(directory) / "state.sqlite3"))
            self.assertEqual(reactor.tick("report-test"), RunStatus.COMPLETED)
            self.assertIsNone(reactor.store.connection.execute("SELECT title FROM goals WHERE title='must not be applied'").fetchone())
            reactor.close()
