"""A refused proposal must name the hypothesis fields it is missing.

Measured defect (2026-10-31): eight consecutive ``propose_self_improvement``
refusals across two runs of the organism, every call carrying a ``hypothesis``
dict whose only key was ``problem``. The refusal was the bare sentence
"proposal requires a complete evidence-backed hypothesis", so the run brain
could see *that* the payload was refused but not *which* field the contract
wanted: each retry guessed a different channel (``files``, ``changes``, an
empty payload) instead of completing the hypothesis, and after two identical
refusals the repeat guard closed both runs ``blocked``.

The invariant these tests pin: a hypothesis-shape refusal names every field
that is missing or empty, and a complete hypothesis is never refused for its
shape.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from skynet.self_improvement import SelfImprovementManager, SelfImprovementTool

COMPLETE_HYPOTHESIS = {
    "problem": "a refused proposal does not say which hypothesis field is empty",
    "expected_behavior": "the refusal names every missing or empty field",
    "evidence": "2026-10-31: eight refusals, every hypothesis carrying `problem` only",
    "validation": "assert the named fields equal the empty ones",
    "rollback_condition": "revert if a complete hypothesis starts being refused",
}

ALL_HYPOTHESIS_FIELDS = set(COMPLETE_HYPOTHESIS)


def _named_fields(error: object) -> set[str]:
    """The field list a hypothesis-shape refusal puts after its colon."""
    text = str(error)
    marker = "missing or empty fields: "
    if marker not in text:
        return set()
    return {name.strip() for name in text.split(marker, 1)[1].split(",")}


def _tool(directory: str) -> SelfImprovementTool:
    return SelfImprovementTool(SelfImprovementManager(Path(directory)))


class HypothesisRefusalNamesTheMissingFieldsTests(unittest.TestCase):
    def test_a_problem_only_hypothesis_names_the_four_missing_fields(self) -> None:
        """The measured defect: `problem` alone, no clue about the rest."""
        with tempfile.TemporaryDirectory() as directory:
            result = _tool(directory).execute(
                {"files": {"new.txt": "x"}, "hypothesis": {"problem": "the draft is not durable under artifacts/"}},
                idempotency_key="problem-only",
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["failure_class"], "invalid_payload")
            self.assertEqual(
                _named_fields(result.get("error")),
                {"expected_behavior", "evidence", "validation", "rollback_condition"},
            )

    def test_empty_and_whitespace_only_fields_are_named_and_filled_ones_are_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = _tool(directory).execute(
                {
                    "files": {"new.txt": "x"},
                    "hypothesis": {
                        "problem": "p",
                        "expected_behavior": "e",
                        "evidence": "   ",
                        "validation": "v",
                        "rollback_condition": "",
                    },
                },
                idempotency_key="partially-filled",
            )
            self.assertFalse(result["ok"])
            self.assertEqual(_named_fields(result.get("error")), {"evidence", "rollback_condition"})

    def test_a_missing_hypothesis_key_names_all_five_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = _tool(directory).execute({"files": {"new.txt": "x"}}, idempotency_key="no-hypothesis")
            self.assertFalse(result["ok"])
            self.assertEqual(result["failure_class"], "invalid_payload")
            self.assertEqual(_named_fields(result.get("error")), ALL_HYPOTHESIS_FIELDS)

    def test_a_complete_hypothesis_is_never_refused_for_its_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = _tool(directory).execute(
                {"files": {"new.txt": "x"}, "hypothesis": dict(COMPLETE_HYPOTHESIS)},
                idempotency_key="complete",
            )
            self.assertEqual(_named_fields(result.get("error")), set())


if __name__ == "__main__":
    unittest.main()
