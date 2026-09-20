"""Shared fixtures for the SkyNet test suite."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from skynet.models import ModelTurn, ToolCall


def git_repo(root: Path) -> None:
    """Initialize a throwaway git repository with a deterministic identity."""
    root.mkdir(parents=True, exist_ok=True)
    for args in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "Test"],
    ):
        subprocess.run(args, cwd=root, check=True)


def commit_all(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "--all"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", message], cwd=root, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


class FakeProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, max_tokens, tools=()):
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(tool_calls=[ToolCall("fixture_tool", {})], usage_tokens=1)
        if self.calls == 2:
            return ModelTurn(
                text=json.dumps({"status": "COMPLETED", "summary": "finished the bounded episode", "evidence": ["fixture tool result"], "actions": [], "changes": [], "tests": [], "blocker": "", "next_hypothesis": ""}),
                usage_tokens=1,
            )
        return ModelTurn(
            text=json.dumps({"memory_candidates": [{"kind": "fact", "content": "cycle completed", "confidence": 0.9}], "next_plan": {"next": "continue"}, "initial_prompt": "continue", "goal_updates": [], "task_updates": [], "evaluation": {}}),
            usage_tokens=1,
        )


class FixtureTool:
    name = "fixture_tool"
    schema = {"type": "function", "function": {"name": name, "description": "test fixture", "parameters": {"type": "object", "additionalProperties": False}}}  # noqa: RUF012 - Tool protocol reads schema as an instance property

    def execute(self, arguments: dict[str, object], *, idempotency_key: str) -> dict[str, object]:
        return {"ok": True}


def count_open_fds() -> int:
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except OSError:
        return -1


def wait_for_threads_to_finish(names: set[str], timeout: float = 3.0) -> set[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        live = {thread.name for thread in threading.enumerate()} & names
        if not live:
            return set()
        time.sleep(0.05)
    return {thread.name for thread in threading.enumerate()} & names


def wait_for_fd_count(baseline: int, timeout: float = 3.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = count_open_fds()
        if current <= baseline:
            return current
        time.sleep(0.05)
    return count_open_fds()
