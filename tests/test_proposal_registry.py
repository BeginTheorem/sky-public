import json

from skynet.proposal_registry import (
    EXHAUSTED_FAILURE_CLASS,
    audit_promotion_liveness,
    resolve_exhausted_environment_blocks,
    sweep_environment_blocks,
)


def test_audit_reports_only_promotions_whose_commit_left_head():
    """The invariant: a live-claiming record's commit must be an ancestor of HEAD.

    reconcile_awaiting_reboot never revisits an 'accepted' row, so without this
    check a promotion whose commit was dropped by an external rollback stays
    'accepted' forever with no recorded rollback reason.
    """
    proposals = {
        "live": {"status": "accepted", "commit": "aaa"},
        "dropped": {"status": "accepted", "commit": "bbb"},
        "awaiting": {"status": "awaiting_reboot", "promoted_commit": "bbb"},
        "never_committed": {"status": "accepted"},
        "rejected": {"status": "rejected", "commit": "bbb"},
        "unreadable": {"status": "accepted", "commit": "ccc"},
    }
    history = {"aaa": True, "bbb": False, "ccc": None}
    violations = audit_promotion_liveness(proposals, lambda commit: history[commit])
    assert [item["proposal_id"] for item in violations] == ["dropped", "awaiting"]
    assert violations[0]["status"] == "accepted"
    assert violations[1]["commit"] == "bbb"


def test_audit_is_silent_when_every_commit_is_an_ancestor():
    proposals = {"p1": {"status": "accepted", "commit": "aaa"}, "p2": {"status": "rejected"}}
    assert audit_promotion_liveness(proposals, lambda commit: True) == []


def test_audit_ignores_non_dict_records_and_undecidable_ancestry():
    proposals = {"junk": "not-a-record", "p1": {"status": "accepted", "commit": "aaa"}}
    assert audit_promotion_liveness(proposals, lambda commit: None) == []


def test_blocked_record_with_terminal_sibling_is_closed():
    """The retry guard refuses a fingerprint with ANY terminal sibling.

    A rejected/validated/awaiting_reboot/accepted sibling already makes
    ``propose_files`` throw for the identical patch, so a blocked sibling must
    not stay ``blocked_by_environment`` forever.
    """
    proposals = {
        "a": {"status": "rejected", "change_fingerprint": "F", "failure_class": "regression_failure"},
        "b": {"status": "blocked_by_environment", "change_fingerprint": "F", "environment_attempts": 1},
    }
    _, resolved = resolve_exhausted_environment_blocks(proposals, max_attempts=3)
    assert [item["proposal_id"] for item in resolved] == ["b"]
    assert proposals["b"]["status"] == "rejected"
    assert proposals["b"]["failure_class"] == EXHAUSTED_FAILURE_CLASS
    assert resolved[0]["spent_by_own_attempts"] is False


def test_terminal_sibling_statuses_all_close_the_group():
    for status in ("rejected", "validated", "awaiting_reboot", "accepted"):
        proposals = {
            "a": {"status": status, "change_fingerprint": "F"},
            "b": {"status": "blocked_by_environment", "change_fingerprint": "F", "environment_attempts": 1},
        }
        _, resolved = resolve_exhausted_environment_blocks(proposals, max_attempts=3)
        assert [item["proposal_id"] for item in resolved] == ["b"], status
        assert proposals["b"]["status"] == "rejected"


def test_retryable_blocked_record_without_terminal_sibling_is_untouched():
    proposals = {
        "b": {"status": "blocked_by_environment", "change_fingerprint": "F", "environment_attempts": 1},
    }
    _, resolved = resolve_exhausted_environment_blocks(proposals, max_attempts=3)
    assert resolved == []
    assert proposals["b"]["status"] == "blocked_by_environment"


def test_sweep_preserves_non_dict_top_level_records(tmp_path):
    """A rewrite must not discard records it does not understand."""
    path = tmp_path / "proposals.json"
    path.write_text(json.dumps({
        "a": {"status": "rejected", "change_fingerprint": "F"},
        "b": {"status": "blocked_by_environment", "change_fingerprint": "F", "environment_attempts": 1},
        "meta": "not-a-record",
        "count": 7,
    }))
    resolved = sweep_environment_blocks(path)
    assert [item["proposal_id"] for item in resolved] == ["b"]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["meta"] == "not-a-record"
    assert data["count"] == 7
    assert data["b"]["status"] == "rejected"
