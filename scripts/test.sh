#!/usr/bin/env bash
# Local verification entry point: static checks + the full test suite.
#
# Usage:
#   scripts/test.sh              # ruff, pyright and pytest
#   scripts/test.sh --coverage   # also print a per-module coverage report
#
# The self-improvement gate runs the same ruff/pyright/pytest stages (see
# skynet/self_improvement.py), so this script is the local mirror of the gate
# and stays well inside its 600s budget. The tree is pyright-clean; a new
# diagnostic is a regression, not a baseline.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
RUFF="${RUFF:-$ROOT_DIR/.venv/bin/ruff}"
PYRIGHT="${PYRIGHT:-$ROOT_DIR/.venv/bin/pyright}"

cd "$ROOT_DIR"

echo "== ruff =="
# The ruleset lives in pyproject.toml; the gate runs the same config.
"$RUFF" check skynet tests

echo "== pyright =="
# --pythonpath keeps import resolution identical to the editor's language
# server even when the tree is checked from a proposal worktree without .venv.
"$PYRIGHT" --pythonpath "$PYTHON" skynet tests

if [[ "${1:-}" == "--coverage" ]]; then
    echo "== pytest with coverage =="
    COVERAGE_FILE="${COVERAGE_FILE:-/tmp/skynet.coverage}" "$PYTHON" -m coverage run --branch --source=skynet -m pytest -q
    COVERAGE_FILE="${COVERAGE_FILE:-/tmp/skynet.coverage}" "$PYTHON" -m coverage report --sort=cover
else
    echo "== pytest =="
    "$PYTHON" -m pytest -q
fi
