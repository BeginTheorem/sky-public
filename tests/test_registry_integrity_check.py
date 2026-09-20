import json

from skynet.self_improvement import check_registry_integrity


def test_check_integrity_valid(tmp_path):
    registry = tmp_path / "proposals.json"
    data = {
        "p1": {"failure_class": "timeout", "status": "blocked_by_environment"},
        "p2": {"failure_class": "environment_failure", "status": "blocked_by_environment"},
        "p3": {"failure_class": "regression_failure", "status": "rejected"},
        "p4": {"failure_class": "", "status": "accepted"},
    }
    registry.write_text(json.dumps(data))
    assert check_registry_integrity(registry) == []

def test_check_integrity_violations(tmp_path):
    registry = tmp_path / "proposals.json"
    data = {
        "p1": {"failure_class": "timeout", "status": "rejected"}, # Violation
        "p2": {"failure_class": "environment_failure", "status": "rejected"}, # Violation
        "p3": {"failure_class": "administrative_test_failure", "status": "validated"}, # Violation
        "p4": {"failure_class": "regression_failure", "status": "rejected"}, # OK
    }
    registry.write_text(json.dumps(data))
    violations = check_registry_integrity(registry)
    assert len(violations) == 3
    pids = [v[0] for v in violations]
    assert "p1" in pids
    assert "p2" in pids
    assert "p3" in pids

def test_check_integrity_invalid_json(tmp_path):
    registry = tmp_path / "proposals.json"
    registry.write_text("invalid json")
    violations = check_registry_integrity(registry)
    assert len(violations) == 1
    assert violations[0][0] == "registry"


def test_check_integrity_flags_anchor_reason_code_drift(tmp_path):
    """A persisted anchor verdict must agree with the rest of its record.

    reason_code and match_count are written beside failure_class when an anchor
    fails, so persisting them only helps if a later reader can tell a consistent
    record from a drifted one: a stale anchor and a count-dropped ambiguous
    anchor share the same human-readable message, and nothing else in the
    registry compares the two fields.
    """
    registry = tmp_path / "proposals.json"
    registry.write_text(json.dumps({
        "class_drift": {"failure_class": "patch_mismatch", "status": "rejected", "reason_code": "anchor_not_found", "match_count": 0},
        "count_not_int": {"failure_class": "anchor_ambiguous", "status": "rejected", "reason_code": "anchor_ambiguous", "match_count": "two"},
        "count_bool": {"failure_class": "anchor_ambiguous", "status": "rejected", "reason_code": "anchor_ambiguous", "match_count": True},
        "count_missing": {"failure_class": "anchor_ambiguous", "status": "rejected", "reason_code": "anchor_ambiguous"},
        "code_missing": {"failure_class": "anchor_not_found", "status": "rejected"},
        "consistent_zero": {"failure_class": "anchor_not_found", "status": "rejected", "reason_code": "anchor_not_found", "match_count": 0},
        "consistent_multi": {"failure_class": "anchor_ambiguous", "status": "rejected", "reason_code": "anchor_ambiguous", "match_count": 2},
        "legacy_text_only": {"failure_class": "patch_mismatch", "status": "rejected", "failure": "patch anchor must match exactly once: skynet/autonomous_planner.py"},
        "non_anchor_code": {"failure_class": "path_violation", "status": "rejected", "reason_code": "path_violation"},
        "accepted_untouched": {"failure_class": "", "status": "accepted"},
    }))
    flagged = {proposal_id for proposal_id, _reason in check_registry_integrity(registry)}
    assert flagged == {"class_drift", "count_not_int", "count_bool", "count_missing", "code_missing"}
