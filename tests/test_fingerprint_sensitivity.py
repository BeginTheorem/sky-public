from skynet.planner import hypothesis_fingerprint, structural_fingerprint


def test_hypothesis_fingerprint_exact():
    f1 = hypothesis_fingerprint(area="Core", problem="Bug A", expected_behavior="Fix A", files=["file1.py", "file2.py"])
    f2 = hypothesis_fingerprint(area="Core", problem="Bug A", expected_behavior="Fix A", files=["file1.py", "file2.py"])
    assert f1 == f2

def test_hypothesis_fingerprint_normalization():
    f1 = hypothesis_fingerprint(area="Core", problem="Bug A!", expected_behavior="Fix A.", files=["file1.py"])
    f2 = hypothesis_fingerprint(area="core", problem="bug a", expected_behavior="fix a", files=["file1.py"])
    assert f1 == f2

def test_hypothesis_fingerprint_file_order():
    f1 = hypothesis_fingerprint(area="Core", problem="Bug A", expected_behavior="Fix A", files=["a.py", "b.py"])
    f2 = hypothesis_fingerprint(area="Core", problem="Bug A", expected_behavior="Fix A", files=["b.py", "a.py"])
    assert f1 == f2

def test_hypothesis_fingerprint_variance():
    f1 = hypothesis_fingerprint(area="Core", problem="Bug A", expected_behavior="Fix A", files=["a.py"])
    f2 = hypothesis_fingerprint(area="Core", problem="Bug A", expected_behavior="Fix B", files=["a.py"])
    assert f1 != f2

def test_structural_fingerprint_basic():
    s1 = structural_fingerprint(area="Core", target="Module X", behavior_kind="refactor")
    s2 = structural_fingerprint(area="Core", target="Module X", behavior_kind="refactor")
    assert s1 == s2

def test_structural_fingerprint_variance():
    s1 = structural_fingerprint(area="Core", target="Module X", behavior_kind="refactor")
    s2 = structural_fingerprint(area="Core", target="Module Y", behavior_kind="refactor")
    assert s1 != s2

def test_near_duplicate_sensitivity():
    # a "near duplicate" in this system's view is something that
    # passes normalize_hypothesis_text.
    # Let's check if a small wording change that isn't a "repetition marker"
    # creates a new fingerprint.
    f1 = hypothesis_fingerprint(area="Core", problem="The system crashes on start", expected_behavior="System starts", files=["a.py"])
    f2 = hypothesis_fingerprint(area="Core", problem="The system crashes during startup", expected_behavior="System starts", files=["a.py"])
    # These are conceptually identical but syntactically different.
    # The current implementation uses sha256 on normalized text.
    # "crashes on start" != "crashes during startup"
    assert f1 != f2, "Near-duplicates with different wording should have different fingerprints"

def run_tests():
    tests = [
        test_hypothesis_fingerprint_exact,
        test_hypothesis_fingerprint_normalization,
        test_hypothesis_fingerprint_file_order,
        test_hypothesis_fingerprint_variance,
        test_structural_fingerprint_basic,
        test_structural_fingerprint_variance,
        test_near_duplicate_sensitivity,
    ]
    for test in tests:
        try:
            test()
            print(f"PASSED: {test.__name__}")
        except AssertionError:
            print(f"FAILED: {test.__name__}")
            raise
    print("All tests passed!")

if __name__ == "__main__":
    run_tests()
