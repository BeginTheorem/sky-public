#!/usr/bin/env python3
"""
Standalone scratch-headroom verification script.

Reads the latest runtime_facts from the state database, runs df on the scratch
filesystem, compares the values, and prints a three-way verdict:

Verdict (a): new code live and facts correct -> exit 0
Verdict (b): new code live with wrong facts -> exit 1
Verdict (c): no scratch facts in runtime_facts -> exit 2
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def get_latest_runtime_facts(db_path: Path) -> dict | None:
    """Extract runtime_facts from the most recent run_started event."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT payload FROM event_log WHERE kind='run_started' ORDER BY sequence DESC LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        payload = json.loads(row[0])
        observations = payload.get("observations", [])

        for obs in observations:
            if obs.get("kind") == "runtime_facts":
                return obs
        return None
    except Exception as e:
        print(f"ERROR reading database: {e}", file=sys.stderr)
        return None


def get_df_output(scratch_dir: str) -> tuple[int, int] | None:
    """Run df on the scratch directory and return (free_bytes, total_bytes)."""
    try:
        result = subprocess.run(
            ["df", "--output=avail,size", "-B1", scratch_dir],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) < 2:
            return None
        parts = lines[1].split()
        if len(parts) < 2:
            return None
        free_bytes = int(parts[0])
        total_bytes = int(parts[1])
        return (free_bytes, total_bytes)
    except Exception as e:
        print(f"ERROR running df: {e}", file=sys.stderr)
        return None


def main() -> int:
    # Determine database path
    db_path = Path(os.getenv("SKYNET_STATE_DB", "state/skynet.sqlite3"))
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}", file=sys.stderr)
        return 2

    # Get runtime_facts
    facts = get_latest_runtime_facts(db_path)
    if facts is None:
        print("VERDICT (c): no scratch facts in runtime_facts")
        return 2

    # Check for scratch facts
    scratch_dir = facts.get("scratch_directory")
    headroom_bytes = facts.get("scratch_headroom_bytes")
    budget_bytes = facts.get("scratch_budget_bytes")
    low_headroom = facts.get("scratch_low_headroom")

    if scratch_dir is None or headroom_bytes is None or budget_bytes is None:
        print("VERDICT (c): no scratch facts in runtime_facts")
        print(f"  Available keys: {list(facts.keys())}")
        return 2

    # Run df on the scratch directory
    df_result = get_df_output(scratch_dir)
    if df_result is None:
        print(f"ERROR: Could not run df on {scratch_dir}", file=sys.stderr)
        return 2

    df_free, df_total = df_result

    # Print comparison
    print("Runtime facts:")
    print(f"  scratch_directory: {scratch_dir}")
    print(f"  scratch_headroom_bytes: {headroom_bytes}")
    print(f"  scratch_budget_bytes: {budget_bytes}")
    print(f"  scratch_low_headroom: {low_headroom}")
    print()
    print(f"df output (df --output=avail,size -B1 {scratch_dir}):")
    print(f"  free bytes: {df_free}")
    print(f"  total bytes: {df_total}")
    print()

    # Check if facts match df (allow small difference due to timing)
    # Using 1% tolerance
    headroom_match = abs(headroom_bytes - df_free) <= max(df_free * 0.01, 1024 * 1024)
    budget_match = abs(budget_bytes - df_total) <= max(df_total * 0.01, 1024 * 1024)

    if headroom_match and budget_match:
        print("VERDICT (a): new code live and facts correct")
        return 0

    print("VERDICT (b): new code live with wrong facts")
    print(f"  headroom_match: {headroom_match} (diff: {abs(headroom_bytes - df_free)})")
    print(f"  budget_match: {budget_match} (diff: {abs(budget_bytes - df_total)})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
