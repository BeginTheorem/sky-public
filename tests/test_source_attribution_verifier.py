"""Unit tests for SourceAttributionVerifier.

Covers extraction-stage verification of new extraction candidates:
a fabricated candidate that contradicts its cited source must be judged
Hallucination with a failing fact check.
"""

from __future__ import annotations

import unittest

from skynet.source_attribution_verifier import SourceAttributionVerifier


def _memory_ctx(text: str) -> dict[str, dict[str, object]]:
    return {"memory:src": {"accessible": True, "content": text}}


class SourceAttributionVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = SourceAttributionVerifier()

    def test_fabricated_candidate_with_contradictory_source_is_hallucination(self) -> None:
        # The candidate invents a date (2024-11-22) that contradicts the cited
        # source (2023-05-10); the key-fact contradiction must surface as
        # Hallucination with a failing fact check.
        src = "ACME Corp signed the partnership agreement on 2023-05-10."
        candidate = {
            "kind": "fact",
            "content": "ACME Corp signed the partnership agreement on 2024-11-22.",
            "evidence": ["memory:src"],
        }
        verdict, breakdown = self.verifier.verify(candidate, _memory_ctx(src))
        self.assertEqual(verdict, "Hallucination")
        self.assertEqual(breakdown["verdict"], "Hallucination")
        self.assertIn("memory:src", breakdown["evals"])
        self.assertIs(breakdown["evals"]["memory:src"]["fact_check"], False)

    def test_consistent_candidate_passes(self) -> None:
        src = "ACME Corp signed the partnership agreement on 2023-05-10."
        candidate = {
            "kind": "fact",
            "content": "ACME Corp signed the partnership agreement on 2023-05-10.",
            "evidence": ["memory:src"],
        }
        verdict, _ = self.verifier.verify(candidate, _memory_ctx(src))
        self.assertEqual(verdict, "Correct")

    def test_candidate_without_retrievable_source_is_unverifiable(self) -> None:
        verdict, _ = self.verifier.verify(
            {"kind": "fact", "content": "Some claim.", "evidence": []},
            {},
        )
        self.assertEqual(verdict, "unverifiable_link")


    def test_fabricated_candidate_with_contradictory_url_evidence_is_hallucination(self) -> None:
        # A new candidate that cites a retrieved url: source contradicting its
        # key facts must be judged Hallucination with a failing fact check.
        source_context = {
            "url:https://example.com/page": {
                "accessible": True,
                "content": "ACME Corp signed the partnership agreement on 2023-05-10.",
                "type": "url",
            }
        }
        candidate = {
            "kind": "fact",
            "content": "ACME Corp signed the partnership agreement on 2024-11-22.",
            "evidence": ["url:https://example.com/page"],
        }
        verdict, breakdown = self.verifier.verify(candidate, source_context)
        self.assertEqual(verdict, "Hallucination")
        self.assertEqual(breakdown["verdict"], "Hallucination")
        self.assertIn("url:https://example.com/page", breakdown["evals"])
        self.assertIs(breakdown["evals"]["url:https://example.com/page"]["fact_check"], False)

if __name__ == "__main__":
    unittest.main()
