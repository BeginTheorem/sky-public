#!/usr/bin/env bash

set -euo pipefail

ROOT="${SKYNET_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
REQUEST="$ROOT/state/reboot-request.json"
GUARD="$ROOT/state/reboot-guard.json"
SERVICE="${SKYNET_SERVICE:-skynet.service}"
SYSTEMCTL="${SKYNET_SYSTEMCTL:-systemctl}"

ROLLBACK_REQUEST="$ROOT/state/rollback-request.json"

if [[ ! -f "$REQUEST" && ! -f "$GUARD" && ! -f "$ROLLBACK_REQUEST" ]]; then
    exit 0
fi

# The request is the authoritative reboot intent, so a malformed or finished
# guard must never suppress it. The guard only gates a rollback when there is no
# request: a window that already finished is not a rollback warrant, and an
# unrelated later crash must never revert an accepted version.
if [[ -f "$REQUEST" || -f "$ROLLBACK_REQUEST" ]]; then
    :
elif [[ -f "$GUARD" ]]; then
    GUARD_STATE="$(python3 - "$GUARD" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        guard = json.load(stream)
except (OSError, ValueError):
    print("malformed")
    raise SystemExit(0)
if not isinstance(guard, dict):
    print("malformed")
elif guard.get("completed_at") or guard.get("rolled_back") or guard.get("active") is False:
    print("finished")
else:
    print("open")
PY
)"
    if [[ "$GUARD_STATE" != "open" ]]; then
        echo "reboot guard is $GUARD_STATE; no rollback" >&2
        exit 0
    fi
fi

MARKER="$REQUEST"
if [[ ! -f "$MARKER" ]]; then
    MARKER="$ROLLBACK_REQUEST"
fi
if [[ ! -f "$MARKER" ]]; then
    MARKER="$GUARD"
fi
ROLLBACK_COMMIT="$(python3 - "$MARKER" <<'PY' || true
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        marker = json.load(stream)
except (OSError, ValueError):
    marker = {}
print(marker.get("rollback_commit", "") if isinstance(marker, dict) else "")
PY
)"

# No usable rollback commit means the working tree must not be modified: the
# old runtime-backup fallback could quarantine .git/, .venv/ and tests/ (P0-2).
if [[ ! "$ROLLBACK_COMMIT" =~ ^[0-9a-fA-F]{7,64}$ ]]; then
    echo "no usable rollback commit: '$ROLLBACK_COMMIT' is empty or malformed; refusing to modify the working tree" >&2
    exit 1
elif ! git -C "$ROOT" cat-file -e "$ROLLBACK_COMMIT^{commit}" 2>/dev/null; then
    echo "rollback commit does not exist: $ROLLBACK_COMMIT; refusing to modify the working tree" >&2
    exit 1
elif ! git -C "$ROOT" merge-base --is-ancestor "$ROLLBACK_COMMIT" HEAD 2>/dev/null; then
    echo "rollback commit $ROLLBACK_COMMIT is not an ancestor of HEAD; refusing to reset" >&2
    exit 1
else
    git -C "$ROOT" reset --hard "$ROLLBACK_COMMIT"
fi

rm -f "$REQUEST"
rm -f "$GUARD"
rm -f "$ROLLBACK_REQUEST"
"$SYSTEMCTL" reset-failed "$SERVICE" || true
"$SYSTEMCTL" restart "$SERVICE"
