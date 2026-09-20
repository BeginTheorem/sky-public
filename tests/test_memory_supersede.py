"""Correction lifecycle: supersede contract, instruction, and pass-through."""
from __future__ import annotations

import json
import unittest

from skynet.memory import MEMORY_LOOP_INSTRUCTION, MemoryLoop
from skynet.model_contracts import MEMORY_RESPONSE_SCHEMA, SYSTEM, validate_shape
from skynet.models import ModelTurn


def _memory_response(candidate: dict) -> dict:
    return {
        "memory_candidates": [candidate],
        "next_plan": {},
        "initial_prompt": "",
        "goal_updates": [],
        "task_updates": [],
        "evaluation": {},
    }


class MemorySupersedeTests(unittest.TestCase):
    def test_supersede_field_is_optional_in_memory_schema(self) -> None:
        validate_shape(_memory_response({"kind": "fact", "content": "plain fact"}), MEMORY_RESPONSE_SCHEMA)
        validate_shape(
            _memory_response(
                {
                    "kind": "fact",
                    "content": "corrected fact",
                    "supersedes_memory_id": "abc123",
                    "evidence": ["run-1 finish report"],
                }
            ),
            MEMORY_RESPONSE_SCHEMA,
        )

    def test_system_names_memory_tool_and_correction(self) -> None:
        self.assertIn("memory (search/remember/forget/pin/unpin/correct)", SYSTEM)
        self.assertIn("memory.correct supersedes a wrong memory with cited evidence", SYSTEM)
        self.assertIn("any owner message is an observation you may read, answer, or ignore", SYSTEM)

    def test_instruction_requests_evidence_backed_supersede(self) -> None:
        self.assertIn("supersedes_memory_id", MEMORY_LOOP_INSTRUCTION)
        self.assertIn("never supersede on a guess", MEMORY_LOOP_INSTRUCTION)

    def test_parse_preserves_supersede_candidate_verbatim(self) -> None:
        candidate = {
            "kind": "fact",
            "content": "the gateway listens on 8080, not 9090",
            "confidence": 0.9,
            "supersedes_memory_id": "deadbeef",
            "evidence": ["tool_result: curl 8080"],
        }
        turn = ModelTurn(text=json.dumps(_memory_response(candidate)))
        result = MemoryLoop._parse(turn)
        self.assertEqual(result.memory_candidates, [candidate])
        self.assertEqual(result.memory_candidates[0]["supersedes_memory_id"], "deadbeef")


if __name__ == "__main__":
    unittest.main()
