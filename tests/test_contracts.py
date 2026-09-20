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
        # Regression: one prose sentence before a fenced, contract-valid
        # Finish Report was rejected at char 0 and the parser error was
        # published instead of the report.
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
        # A fenced JSON array is still not a JSON object.
        with self.assertRaises(TypeError):
            parse_json_object("```json\n[1, 2]\n```")
    def test_system_grants_owner_freedom_and_lists_real_tools(self) -> None:
        from skynet.model_contracts import ANTI_LOOP_PROTOCOL, SYSTEM, SYSTEM_MEMORY, SYSTEM_REACT
        # The old denial is gone; the owner channel is a choice, not a command.
        self.assertNotIn("There is no user to wait for", SYSTEM)
        self.assertIn("owner channel is open", SYSTEM)
        self.assertIn("observation, not an order", SYSTEM)
        for name in (
            "bash", "webfetch", "read", "grep", "db",
            "propose_self_improvement", "promote_self_improvement",
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
