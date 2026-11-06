#!/usr/bin/env bash
# Deploy SkyNet to the target host as an unprivileged, lingering *user* service.
#
# First run bootstraps the whole instance on the host (git checkout, uv
# virtualenv, MCP runtime, rendered environment, user units, enable --now);
# later runs are a plain redeploy (push, refresh deps, re-upload env, re-render
# units, restart). No sudo is ever used: the service account owns its home and
# its own systemd user manager.
#
# Infrastructure values live in config/deploy.env (gitignored), never here.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ENV="$ROOT_DIR/config/deploy.env"
if [[ -f "$DEPLOY_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$DEPLOY_ENV"
fi

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_USER="${REMOTE_USER:-}"
REMOTE_DIR="${REMOTE_DIR:-$HOME/project-skynet}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/deploy_key}"
PYTHON_VERSION="${SKYNET_PYTHON_VERSION:-3.14}"

if [[ -z "$REMOTE_HOST" ]]; then
    printf 'REMOTE_HOST is not set; create config/deploy.env with REMOTE_HOST=<host>\n' >&2
    exit 2
fi
if [[ -z "$REMOTE_USER" ]]; then
    printf 'REMOTE_USER is not set; create config/deploy.env with REMOTE_USER=<account>\n' >&2
    exit 2
fi
if [[ ! -f "$SSH_KEY" ]]; then
    printf 'SSH key not found: %s\n' "$SSH_KEY" >&2
    exit 1
fi

RUNTIME_ENV="$ROOT_DIR/config/skynet.env"
if [[ ! -f "$RUNTIME_ENV" ]]; then
    printf 'Runtime environment file not found: %s\n' "$RUNTIME_ENV" >&2
    exit 1
fi

cd "$ROOT_DIR"
if [[ -n "$(git status --porcelain)" ]]; then
    printf 'Local worktree is dirty; commit or stash before deploying.\n' >&2
    exit 1
fi

SSH_ARGS=(
    -o BatchMode=yes
    -o PreferredAuthentications=publickey
    -o PasswordAuthentication=no
    -o ConnectTimeout=5
    -i "$SSH_KEY"
)
REMOTE_URL="ssh://${REMOTE_USER}@${REMOTE_HOST}${REMOTE_DIR}"
export GIT_SSH_COMMAND="ssh ${SSH_ARGS[*]}"

remote() {
    ssh "${SSH_ARGS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "$@"
}

# Every remote systemctl call must reach the target user's own manager. linger
# (loginctl enable-linger <user>) is a manual prerequisite of a fresh host.
USERCTL='export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user'

remote 'command -v systemctl >/dev/null' || {
    printf 'systemd not available on %s\n' "$REMOTE_HOST" >&2
    exit 1
}
REMOTE_HOME="$(remote 'printf %s "$HOME"')"

BOOTSTRAP=false
if ! remote "test -x '$REMOTE_DIR/.venv/bin/python' && test -f \"\$HOME/.config/systemd/user/skynet.service\""; then
    BOOTSTRAP=true
fi
printf 'Deploying %s to %s@%s:%s (%s)\n' \
    "$ROOT_DIR" "$REMOTE_USER" "$REMOTE_HOST" "$REMOTE_DIR" \
    "$([[ "$BOOTSTRAP" == true ]] && echo bootstrap || echo redeploy)"

# --- code -------------------------------------------------------------------
remote "set -eu; mkdir -p '$REMOTE_DIR'; if [ ! -d '$REMOTE_DIR/.git' ]; then git init -q '$REMOTE_DIR'; fi; git -C '$REMOTE_DIR' branch -m master main 2>/dev/null || true; git -C '$REMOTE_DIR' config receive.denyCurrentBranch updateInstead"

# Reconcile before pushing: the organism may have committed on the host since
# the last deploy. Merge those commits locally (abort on conflict) so the push
# stays a fast-forward and no history is discarded.
if git fetch --quiet "$REMOTE_URL" main 2>/dev/null; then
    if ! git merge-base --is-ancestor FETCH_HEAD HEAD 2>/dev/null; then
        printf 'Host has commits not present locally; merging them into the local branch.\n'
        if ! git merge --no-edit FETCH_HEAD; then
            git merge --abort || true
            printf 'Merge conflict with the host branch; reconcile manually before deploying.\n' >&2
            exit 1
        fi
    fi
fi
git push --quiet "$REMOTE_URL" HEAD:refs/heads/main
git push --quiet "$REMOTE_URL" --tags 2>/dev/null || true
DEPLOYED_COMMIT="$(git rev-parse HEAD)"

# --- virtualenv -------------------------------------------------------------
if [[ "$BOOTSTRAP" == true ]]; then
    remote "set -eu; command -v uv >/dev/null || { echo 'uv not found on host' >&2; exit 1; }; if [ ! -x '$REMOTE_DIR/.venv/bin/python' ]; then uv venv --python '$PYTHON_VERSION' '$REMOTE_DIR/.venv'; fi"
fi
remote "set -eu; VIRTUAL_ENV='$REMOTE_DIR/.venv' uv pip install -q -e '$REMOTE_DIR[dev]'"

# okama is not a project dependency any more: it is the `okama` MCP sensor, installed
# with pipx as the service account (see AGENTS.md, «Рынок»). The uninstall below keeps
# the project venv free of the old market extra — pip does not remove packages that are
# merely no longer required.
remote "set -eu; uv pip uninstall -q --python '$REMOTE_DIR/.venv/bin/python' okama >/dev/null 2>&1 || true"

# --- MCP runtime ------------------------------------------------------------
if [[ "$BOOTSTRAP" == true ]]; then
    remote "set -eu; bash '$REMOTE_DIR/scripts/install_mcp_runtime.sh'"
fi

# github-mcp-server is large; sync it only when its hash changed. It stays
# per-user: $SKYNET_MCP_GITHUB_COMMAND points at the account's own copy.
#
# gh is deliberately NOT shipped. One gh per box, the system one: the host
# provides /usr/local/bin/gh, deploy never uses sudo so it could not install a
# system binary anyway, and a per-user copy would only shadow it. The organism
# reads GitHub through the MCP server, never through gh.
LOCAL_GITHUB_MCP="${GITHUB_MCP_BINARY:-$HOME/.local/bin/github-mcp-server}"
if [[ -x "$LOCAL_GITHUB_MCP" ]]; then
    LOCAL_GITHUB_MCP_HASH="$(sha256sum "$LOCAL_GITHUB_MCP" | awk '{print $1}')"
    REMOTE_HASHES="$(remote "sha256sum \"\$HOME/.local/bin/github-mcp-server\" 2>/dev/null || true")"
    REMOTE_GITHUB_MCP_HASH="$(printf '%s\n' "$REMOTE_HASHES" | awk 'NF >= 2 {print $1; exit}')"
    if [[ "$LOCAL_GITHUB_MCP_HASH" != "$REMOTE_GITHUB_MCP_HASH" ]]; then
        scp -q "${SSH_ARGS[@]}" "$LOCAL_GITHUB_MCP" "$REMOTE_USER@$REMOTE_HOST:.github-mcp-upload"
    fi
    remote "set -eu; install -d \"\$HOME/.local/bin\"; if [ -f \"\$HOME/.github-mcp-upload\" ]; then install -m 755 \"\$HOME/.github-mcp-upload\" \"\$HOME/.local/bin/github-mcp-server\"; rm -f \"\$HOME/.github-mcp-upload\"; fi"
else
    printf 'WARNING: local github-mcp-server not found; skipping the GitHub MCP binary.\n' >&2
fi

# Drop a per-user gh left behind by an older deploy: it would shadow the system
# binary that every account on the host is meant to share.
remote "set -eu; rm -f \"\$HOME/.local/bin/gh\""
if ! remote "command -v gh >/dev/null 2>&1"; then
    printf 'WARNING: no gh on the host; SkyNet reads GitHub via MCP, so only interactive gh use is affected.\n' >&2
fi

# --- runtime environment ----------------------------------------------------
# The local checkout paths are rewritten to the host account's home so the same
# secret file works for any target user. Secrets stay mode 600 and are never
# committed.
RUNTIME_UPLOAD="$(mktemp "${TMPDIR:-/tmp}/skynet-env.XXXXXX")"
UNIT_DIR=""
cleanup() {
    rm -f "$RUNTIME_UPLOAD"
    [[ -n "$UNIT_DIR" ]] && rm -rf "$UNIT_DIR"
    return 0
}
trap cleanup EXIT
sed "s|${HOME}|${REMOTE_HOME}|g" "$RUNTIME_ENV" > "$RUNTIME_UPLOAD"
chmod 600 "$RUNTIME_UPLOAD"
scp -q "${SSH_ARGS[@]}" "$RUNTIME_UPLOAD" "$REMOTE_USER@$REMOTE_HOST:.skynet-env-upload"
remote "set -eu; install -m 600 \"\$HOME/.skynet-env-upload\" '$REMOTE_DIR/config/skynet.env'; rm -f \"\$HOME/.skynet-env-upload\"; printf '%s\n' '$DEPLOYED_COMMIT' > '$REMOTE_DIR/.deployed-commit'; touch '$REMOTE_DIR/.deployment-refresh'"

# --- user units -------------------------------------------------------------
UNIT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/skynet-units.XXXXXX")"
for unit in skynet.service skynet-telegram.service skynet-rollback.service skynet-alert@.service skynet-revive.service skynet-revive.timer; do
    sed "s|@ROOT@|$REMOTE_DIR|g" "$ROOT_DIR/deploy/${unit}.in" > "$UNIT_DIR/$unit"
    if grep -q '@ROOT@' "$UNIT_DIR/$unit"; then
        printf 'FATAL: unrendered @ROOT@ placeholder remains in %s\n' "$UNIT_DIR/$unit" >&2
        exit 1
    fi
done
UNIT_ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/skynet-units.XXXXXX.tar")"
trap 'rm -f "$RUNTIME_UPLOAD" "$UNIT_ARCHIVE"; [[ -n "$UNIT_DIR" ]] && rm -rf "$UNIT_DIR"; true' EXIT
tar -cf "$UNIT_ARCHIVE" -C "$UNIT_DIR" .
scp -q "${SSH_ARGS[@]}" "$UNIT_ARCHIVE" "$REMOTE_USER@$REMOTE_HOST:.skynet-units-upload.tar"
remote "set -eu; tmp=\$(mktemp -d \"\$HOME/.skynet-units.XXXXXX\"); trap 'rm -rf \"\$tmp\"' EXIT; tar -xf \"\$HOME/.skynet-units-upload.tar\" -C \"\$tmp\"; install -d \"\$HOME/.config/systemd/user\"; install -m 644 \"\$tmp/skynet.service\" \"\$tmp/skynet-telegram.service\" \"\$tmp/skynet-rollback.service\" \"\$tmp/skynet-alert@.service\" \"\$tmp/skynet-revive.service\" \"\$tmp/skynet-revive.timer\" \"\$HOME/.config/systemd/user/\"; rm -f \"\$HOME/.skynet-units-upload.tar\"; $USERCTL daemon-reload; $USERCTL enable skynet.service skynet-telegram.service skynet-revive.timer; $USERCTL reset-failed skynet.service skynet-telegram.service || true"

# --- start / restart --------------------------------------------------------
if [[ "$BOOTSTRAP" == true ]]; then
    remote "$USERCTL start skynet.service skynet-telegram.service"
else
    remote "$USERCTL restart skynet.service skynet-telegram.service" || \
        remote "$USERCTL start skynet.service skynet-telegram.service"
fi
# Arm the revive tick without waiting for a reboot: a start of an already active
# timer is a no-op, so this is idempotent across repeated deploys.
remote "$USERCTL start skynet-revive.timer"

# --- verify -----------------------------------------------------------------
# A restart that leaves a unit dead must fail the deploy, not print a status
# line and exit 0: the CAPABILITIES bug (an empty CapabilityBoundingSet= in a
# user unit) restarted five times and the deploy still reported success. Poll
# briefly for the units to settle, then assert each reached "active".
printf 'Deployed %s. State:\n' "$DEPLOYED_COMMIT"
STATES=""
for _ in 1 2 3 4 5; do
    STATES="$(remote "$USERCTL is-active skynet.service skynet-telegram.service" || true)"
    if ! printf '%s\n' "$STATES" | grep -qv '^active$'; then
        break
    fi
    sleep 2
done
printf '%s\n' "$STATES"
if printf '%s\n' "$STATES" | grep -qv '^active$'; then
    printf 'FATAL: a unit is not active after deploy (see journalctl --user -u <unit>)\n' >&2
    exit 1
fi
remote 'loginctl show-user "$USER" -p Linger 2>/dev/null || true'
