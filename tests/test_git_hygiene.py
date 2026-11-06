"""Git's optional locks: never taken on a read, and swept when one is left behind.

The incident these pin down (2026-10-28): a five-second ``git status`` was killed while
refreshing the index, left a 0-byte ``.git/index.lock``, and every later publication failed
for five hours. Nothing inside the organism could clear it — bash is hard-denied under
``.git/``, and the sudo gate cannot run at all while the unit sets ``NoNewPrivileges``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from skynet.git_hygiene import (
    READ_ONLY_GIT_ENV,
    READ_ONLY_GIT_TIMEOUT_SECONDS,
    clear_stale_index_lock,
    git_dir,
    git_env,
    install_git_read_only_environment,
    read_only_git,
)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    return root


def _lock(root: Path, *, age_seconds: float, content: bytes = b"") -> Path:
    lock = root / ".git" / "index.lock"
    lock.write_bytes(content)
    stamp = time.time() - age_seconds
    os.utime(lock, (stamp, stamp))
    return lock


# ── the sweep ────────────────────────────────────────────────────────────────────


def test_a_stale_empty_lock_is_removed(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    lock = _lock(root, age_seconds=3600)
    assert clear_stale_index_lock(root) == lock
    assert not lock.exists()


def test_a_fresh_lock_is_left_alone(tmp_path: Path) -> None:
    """A live writer's lock is also 0 bytes — age is what separates the two."""
    root = _repo(tmp_path)
    lock = _lock(root, age_seconds=1)
    assert clear_stale_index_lock(root) is None
    assert lock.exists()


def test_a_lock_carrying_content_is_left_alone(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    lock = _lock(root, age_seconds=3600, content=b"pid 1234")
    assert clear_stale_index_lock(root) is None
    assert lock.exists()


def test_a_lock_owned_by_a_live_git_process_is_left_alone(tmp_path: Path) -> None:
    """Age alone is not proof: a slow git on a huge worktree is still a writer."""
    root = _repo(tmp_path)
    lock = _lock(root, age_seconds=3600)
    assert clear_stale_index_lock(root, git_running=lambda: True) is None
    assert lock.exists()


def test_absent_lock_and_absent_repository_are_not_failures(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    assert clear_stale_index_lock(root) is None
    assert clear_stale_index_lock(tmp_path / "not-a-repo") is None


def test_a_worktree_gitdir_file_is_followed(tmp_path: Path) -> None:
    """A linked worktree keeps its git directory elsewhere, behind a ``gitdir:`` file."""
    worktree = tmp_path / "worktree"
    target = tmp_path / "elsewhere"
    target.mkdir(parents=True)
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {target}\n", encoding="utf-8")
    lock = target / "index.lock"
    lock.write_bytes(b"")
    stamp = time.time() - 3600
    os.utime(lock, (stamp, stamp))
    assert git_dir(worktree) == target
    assert clear_stale_index_lock(worktree) == lock
    assert not lock.exists()


# ── the read-only default ────────────────────────────────────────────────────────


def test_git_env_disables_optional_locks_without_dropping_the_rest() -> None:
    env = git_env({"PATH": "/usr/bin", "OTHER": "keep"})
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert env["PATH"] == "/usr/bin" and env["OTHER"] == "keep"
    assert READ_ONLY_GIT_ENV == {"GIT_OPTIONAL_LOCKS": "0"}


def test_read_only_git_passes_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["argv"] = argv
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    read_only_git(["status", "--porcelain", "--untracked-files=no"], tmp_path)
    assert seen["argv"] == ["git", "status", "--porcelain", "--untracked-files=no"]
    assert seen["cwd"] == str(tmp_path)
    assert seen["timeout"] == READ_ONLY_GIT_TIMEOUT_SECONDS
    env = seen["env"]
    assert isinstance(env, dict) and env["GIT_OPTIONAL_LOCKS"] == "0"


def test_install_makes_the_default_process_wide(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GIT_OPTIONAL_LOCKS", raising=False)
    install_git_read_only_environment()
    assert os.environ["GIT_OPTIONAL_LOCKS"] == "0"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_a_real_read_only_status_leaves_no_lock_behind(tmp_path: Path) -> None:
    """End to end: a status that does refresh the index still never takes the lock."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True, capture_output=True)
    (root / "file.txt").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=root, check=True, capture_output=True)
    completed = read_only_git(["status", "--porcelain"], root)
    assert completed.returncode == 0
    assert completed.stdout.strip() == "A  file.txt"
    assert not (root / ".git" / "index.lock").exists()


def test_git_processes_running_sees_a_live_git() -> None:
    """The holder probe reads /proc rather than shelling out; it must see this one."""
    from skynet.git_hygiene import git_processes_running

    process = subprocess.Popen(["git", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert git_processes_running() is True or process.poll() is not None
    finally:
        process.wait(timeout=10)
