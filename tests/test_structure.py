"""The structural mirror must be bounded, honest about absence, and never raise."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from skynet.structure import MAX_SUMMARY_CHARS, StructureTool, analyze, summarize

ROOT = Path("/repo")

REPORT: dict[str, Any] = {
    "summary": {
        "health_score": 81,
        "grade": "B",
        "average_complexity": 3.5935,
        "high_complexity_count": 26,
        "dead_code_count": 0,
        "clone_groups": 39,
        "code_duplication_percentage": 8.5948,
        "high_coupling_classes": 3,
        "average_coupling": 1.7469,
    },
    "system": {"architecture_analysis": {"compliance_score": 0.93, "total_violations": 12}},
    "complexity": {
        "functions": [
            {"name": "main", "file_path": "/repo/skynet/cli.py", "start_line": 438, "metrics": {"complexity": 62}},
            {"name": "tick", "file_path": "/repo/skynet/reactor.py", "start_line": 363, "metrics": {"complexity": 55}},
        ]
    },
    "clone": {
        "clone_pairs": [
            {
                "similarity": 1.0,
                "clone1": {"location": {"file_path": "/repo/skynet/cli.py", "start_line": 155, "end_line": 164}},
                "clone2": {"location": {"file_path": "/repo/skynet/cli.py", "start_line": 189, "end_line": 198}},
            }
        ]
    },
}


class SummarizeTests(unittest.TestCase):
    def test_summary_is_compact_and_names_the_hotspots(self) -> None:
        text = summarize(REPORT, ROOT)
        self.assertIn("health 81/100 (B)", text)
        self.assertIn("avg complexity 3.59", text)
        self.assertIn("architecture compliance 93%", text)
        self.assertIn("skynet/cli.py:438  main (62)", text)
        self.assertIn("sim 1  skynet/cli.py:155-164 <-> skynet/cli.py:189-198", text)
        self.assertLessEqual(len(text), MAX_SUMMARY_CHARS)

    def test_summary_tolerates_an_empty_report(self) -> None:
        text = summarize({}, ROOT)
        self.assertIn("Structural report for repo/skynet", text)


class AnalyzeTests(unittest.TestCase):
    def test_missing_binary_is_reported_not_raised(self) -> None:
        with patch("skynet.structure.pyscn_binary", return_value=None):
            result = analyze(ROOT)
        self.assertFalse(result["available"])
        self.assertIn("not installed", str(result["reason"]))

    def test_tool_reports_unavailable_without_failing_the_run(self) -> None:
        tool = StructureTool(ROOT)
        with patch("skynet.structure.analyze", return_value={"available": False, "reason": "pyscn is not installed in this environment"}):
            outcome = tool.execute({}, idempotency_key="k")
        self.assertFalse(outcome["ok"])
        self.assertIn("not installed", str(outcome["error"]))

    def test_tool_returns_the_bounded_summary(self) -> None:
        tool = StructureTool(ROOT)
        self.assertEqual(tool.name, "structure")
        with patch("skynet.structure.analyze", return_value={"available": True, "report": REPORT}):
            outcome = tool.execute({}, idempotency_key="k")
        self.assertTrue(outcome["ok"])
        self.assertIn("health 81/100", str(outcome["result"]))
        self.assertLessEqual(len(str(outcome["result"])), MAX_SUMMARY_CHARS)

    def test_schema_declares_no_arguments(self) -> None:
        schema = cast(dict[str, Any], StructureTool(ROOT).schema)
        self.assertEqual(schema["function"]["name"], "structure")
        self.assertEqual(schema["function"]["parameters"]["properties"], {})


if __name__ == "__main__":
    unittest.main()
