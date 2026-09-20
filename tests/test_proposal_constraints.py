import unittest
from unittest.mock import MagicMock

from skynet.autonomous_planner import AutonomousPlanner


class MockStore:
    def __init__(self):
        self.connection = MagicMock()
        self.connection.execute.return_value.fetchone.return_value = None
        self.tasks = {}
        self.proposals = []

    def start_planner_attempt(self, *args, **kwargs):
        return "attempt-123"

    def finish_planner_attempt(self, *args, **kwargs):
        pass

    def record_planner_proposal(self, proposal, attempt, status, reason=None, task_id=None):
        self.proposals.append({"proposal": proposal, "status": status, "reason": reason, "task_id": task_id})

    def add_task(self, title, goal_id, **kwargs):
        task_id = f"task-{len(self.tasks) + 1}"
        self.tasks[task_id] = {"title": title, "goal_id": goal_id}
        return task_id

    def append_event(self, event, data):
        pass

    def transaction(self):
        return MagicMock()

class TestProposalConstraints(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MagicMock()
        self.store = MockStore()
        self.planner = AutonomousPlanner(self.mock_provider, self.store)
        self.goals = [{"goal_id": "goal-1", "status": "active"}]
        self.tasks = []
        self.memories = []
        self.previous = {}

    def simulate_planner_turn(self, proposals):
        import json

        from skynet.models import ModelTurn
        self.mock_provider.complete.return_value = ModelTurn(text=json.dumps({"proposals": proposals}))

    def test_invalid_kind_rejection(self):
        # 'invalid_kind' is not in the allowed enum
        proposals = [{
            "goal_id": "goal-1",
            "title": "Test Task",
            "problem": "Problem",
            "hypothesis": "Hypothesis",
            "expected_new_fact": "Fact",
            "validation": "Validation",
            "scope": ["file.py"],
            "kind": "invalid_kind"
        }]
        self.simulate_planner_turn(proposals)
        result = self.planner.generate(generation=1, goals=self.goals, tasks=self.tasks, memories=self.memories, previous=self.previous, trigger="test")
        self.assertEqual(len(result), 0)
        # the payload fails the JSON schema, so it is dropped before any row is recorded
        self.assertEqual(self.store.proposals, [])

    def test_inactive_goal_rejection(self):
        # Goal is active in the list, but we'll try a goal that isn't
        proposals = [{
            "goal_id": "goal-999",
            "title": "Test Task",
            "problem": "Problem",
            "hypothesis": "Hypothesis",
            "expected_new_fact": "Fact",
            "validation": "Validation",
            "scope": ["file.py"],
            "kind": "engineering"
        }]
        self.simulate_planner_turn(proposals)
        result = self.planner.generate(generation=1, goals=self.goals, tasks=self.tasks, memories=self.memories, previous=self.previous, trigger="test")
        self.assertEqual(len(result), 0)
        # the goal is inactive: the payload passes the schema and is recorded as rejected
        self.assertEqual(self.store.proposals[0]["status"], "rejected")

    def test_title_length_rejection(self):
        # Title > 300 chars
        proposals = [{
            "goal_id": "goal-1",
            "title": "A" * 301,
            "problem": "Problem",
            "hypothesis": "Hypothesis",
            "expected_new_fact": "Fact",
            "validation": "Validation",
            "scope": ["file.py"],
            "kind": "engineering"
        }]
        self.simulate_planner_turn(proposals)
        result = self.planner.generate(generation=1, goals=self.goals, tasks=self.tasks, memories=self.memories, previous=self.previous, trigger="test")
        self.assertEqual(len(result), 0)
        # the payload fails the JSON schema, so it is dropped before any row is recorded
        self.assertEqual(self.store.proposals, [])

    def test_scope_length_rejection(self):
        # Scope > 12 files
        proposals = [{
            "goal_id": "goal-1",
            "title": "Test Task",
            "problem": "Problem",
            "hypothesis": "Hypothesis",
            "expected_new_fact": "Fact",
            "validation": "Validation",
            "scope": [f"file{i}.py" for i in range(13)],
            "kind": "engineering"
        }]
        self.simulate_planner_turn(proposals)
        result = self.planner.generate(generation=1, goals=self.goals, tasks=self.tasks, memories=self.memories, previous=self.previous, trigger="test")
        self.assertEqual(len(result), 0)
        # the payload fails the JSON schema, so it is dropped before any row is recorded
        self.assertEqual(self.store.proposals, [])

    def test_missing_required_field(self):
        # Missing 'validation'
        proposals = [{
            "goal_id": "goal-1",
            "title": "Test Task",
            "problem": "Problem",
            "hypothesis": "Hypothesis",
            "expected_new_fact": "Fact",
            "scope": ["file.py"],
            "kind": "engineering"
        }]
        self.simulate_planner_turn(proposals)
        result = self.planner.generate(generation=1, goals=self.goals, tasks=self.tasks, memories=self.memories, previous=self.previous, trigger="test")
        self.assertEqual(len(result), 0)
        # the payload fails the JSON schema, so it is dropped before any row is recorded
        self.assertEqual(self.store.proposals, [])

    def test_deduplication_via_fingerprint(self):
        # First proposal is accepted
        p1 = {
            "goal_id": "goal-1",
            "title": "Unique Task",
            "problem": "Problem A",
            "hypothesis": "Hypothesis A",
            "expected_new_fact": "Fact A",
            "validation": "Val A",
            "scope": ["file.py"],
            "kind": "engineering"
        }
        self.simulate_planner_turn([p1])
        self.planner.generate(generation=1, goals=self.goals, tasks=self.tasks, memories=self.memories, previous=self.previous, trigger="test")

        # Second proposal with same fingerprint
        # we need to make sure MockStore.connection.execute returns a hit
        self.store.connection.execute.return_value.fetchone.return_value = (1,)

        self.simulate_planner_turn([p1])
        result = self.planner.generate(generation=2, goals=self.goals, tasks=self.tasks, memories=self.memories, previous=self.previous, trigger="test")
        self.assertEqual(len(result), 0)
        self.assertEqual(self.store.proposals[-1]["status"], "deduplicated")

if __name__ == "__main__":
    unittest.main()
