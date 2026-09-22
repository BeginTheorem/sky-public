"""The planner instrument is open; the planner decision stays closed.

``skynet/planner_contract.py`` is deliberately editable by self-improvement, so
these tests pin the security property that makes that safe: the open module can
only reshape text into a JSON value. Schema validation, proposal validation and
every dedup/fingerprint/learnability decision remain in the warn-set
``skynet.autonomous_planner`` module.
"""

import json
import unittest
from typing import Any

from skynet.autonomous_planner import AutonomousPlanner
from skynet.planner_contract import PLANNER_RETRY_INSTRUCTION, parse_planner_reply, planner_system_prompt
from skynet.self_improvement import GATE_PROTECTED_PATHS, SelfImprovementManager


def _valid_proposal(goal_id: str, kind: str = "engineering") -> dict[str, Any]:
    return {
        "goal_id": goal_id,
        "title": "bounded probe",
        "problem": "a bounded problem",
        "hypothesis": "the probe is observable",
        "expected_new_fact": "the probe yields one fact",
        "validation": "run the probe and assert the recorded result",
        "scope": ["skynet/planner.py"],
        "kind": kind,
    }


class PlannerContractParseTests(unittest.TestCase):
    """``parse_planner_reply`` decodes; it never judges."""

    def test_top_level_array_is_normalized_into_proposals(self) -> None:
        proposal = _valid_proposal("g")
        self.assertEqual(parse_planner_reply(json.dumps([proposal])), {"proposals": [proposal]})

    def test_bare_object_is_passed_through(self) -> None:
        self.assertEqual(parse_planner_reply(json.dumps({"proposals": []})), {"proposals": []})

    def test_fenced_array_with_prose_is_accepted(self) -> None:
        proposal = _valid_proposal("g")
        text = "Answer:\n```json\n" + json.dumps([proposal]) + "\n```"
        self.assertEqual(parse_planner_reply(text), {"proposals": [proposal]})

    def test_garbage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_planner_reply("I could not decide what to do.")

    def test_decode_failure_carries_a_bounded_reply_preview(self) -> None:
        # The reply is the evidence a decode failure is judged against; it used
        # to be recorded nowhere, so prose and a reply cut off at the output
        # ceiling were indistinguishable after the fact.
        text = "prose " + "z" * 500
        with self.assertRaises(ValueError) as raised:
            parse_planner_reply(text)
        message = str(raised.exception)
        self.assertIn("structured model response contains no acceptable JSON value", message)
        self.assertIn("reply preview:", message)
        self.assertIn(f", {len(text)} chars)", message)
        # Bounded: the preview never echoes the whole reply.
        self.assertLess(len(message), len(text))

    def test_scalar_is_returned_unchanged(self) -> None:
        # The decoder is not the validator: a scalar survives it and dies later.
        self.assertEqual(parse_planner_reply("42"), 42)

    def test_schema_violating_object_survives_the_decoder(self) -> None:
        proposal = _valid_proposal("g", kind="bogus")
        decoded = parse_planner_reply(json.dumps({"proposals": [proposal]}))
        self.assertEqual(decoded, {"proposals": [proposal]})

    def test_system_prompt_renders_the_schema(self) -> None:
        prompt = planner_system_prompt()
        self.assertIn("bounded autonomous planner", prompt)
        self.assertIn("proposals", prompt)

    def test_retry_instruction_forbids_a_list(self) -> None:
        self.assertIn("do not return a list", PLANNER_RETRY_INSTRUCTION)


class ProtectedValidationStillBitesTests(unittest.TestCase):
    """Opening the decoder cannot let the organism neuter the gate."""

    def test_third_proposal_with_invalid_kind_is_still_rejected(self) -> None:
        proposals = [
            _valid_proposal("g"),
            _valid_proposal("g"),
            _valid_proposal("g", kind="bogus"),
        ]
        text = json.dumps(proposals)
        # The open decoder returns the array (with the bad third proposal)...
        decoded = parse_planner_reply(text)
        self.assertEqual(decoded, {"proposals": proposals})
        # ...and the protected path refuses it, naming the live failure's path.
        with self.assertRaises(ValueError) as context:
            AutonomousPlanner._parse(text)
        self.assertIn("proposals[2].kind", str(context.exception))

    def test_unknown_top_level_field_is_still_rejected(self) -> None:
        text = json.dumps({"proposals": [], "override_gate": True})
        parse_planner_reply(text)  # decoded without complaint
        with self.assertRaises(ValueError):
            AutonomousPlanner._parse(text)

    def test_planner_contract_and_judge_are_both_open(self) -> None:
        # Owner decision 2026-09-21: the judge is no longer hard-protected.
        # The instrument stays outside the warn-set; the judge is inside it but
        # is only warned on first submission, never hard-held or rejected.
        self.assertNotIn("skynet/planner_contract.py", GATE_PROTECTED_PATHS)
        self.assertTrue(SelfImprovementManager._is_gate_protected("skynet/autonomous_planner.py"))


if __name__ == "__main__":
    unittest.main()
