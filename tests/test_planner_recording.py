import json
import unittest
from unittest.mock import MagicMock

from skynet.autonomous_planner import AutonomousPlanner
from skynet.models import ModelTurn


class TestPlannerRecording(unittest.TestCase):
    def test_debug_generate_fixed_modelturn(self):
        mock_provider = MagicMock()
        mock_store = MagicMock()
        mock_store.connection = MagicMock()

        planner = AutonomousPlanner(mock_provider, mock_store)
        goals = [{"goal_id": "goal-1", "status": "active"}]

        raw_proposals = [{
            "goal_id": "wrong",
            "title": "T",
            "problem": "P",
            "hypothesis": "H",
            "expected_new_fact": "F",
            "validation": "V",
            "scope": ["S"],
            "kind": "engineering"
        }]

        turn = ModelTurn(text=json.dumps({"proposals": raw_proposals}))
        mock_provider.complete.return_value = turn

        planner.generate(generation=1, goals=goals, tasks=[], memories=[], previous={}, trigger="test")

        print("Store calls:")
        for call in mock_store.method_calls:
            print(call)

        self.assertTrue(any(call[0] == 'record_planner_proposal' for call in mock_store.method_calls))

if __name__ == "__main__":
    unittest.main()
