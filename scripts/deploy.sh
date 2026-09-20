#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Deployment target. Infrastructure values must not live in tracked files.
# Keep the real values in config/deploy.env, which is gitignored.
DEPLOY_ENV="$ROOT_DIR/config/deploy.env"
if [[ -f "$DEPLOY_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$DEPLOY_ENV"
fi

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_USER="${REMOTE_USER:-}"
REMOTE_DIR="${REMOTE_DIR:-$HOME/project-skynet}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/deploy_key}"

if [[ -z "$REMOTE_HOST" || -z "$REMOTE_USER" ]]; then
    printf 'REMOTE_HOST and REMOTE_USER must be set; create config/deploy.env with both\n' >&2
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

printf 'Deploying %s to %s@%s:%s\n' "$ROOT_DIR" "$REMOTE_USER" "$REMOTE_HOST" "$REMOTE_DIR"

# The server keeps its own repository: self-improvement commits live there and
# must not be wiped. Make sure it exists, tracks the shared branch and accepts
# a push into the checked-out worktree.
ssh "${SSH_ARGS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
    "set -eu; if [ ! -d '$REMOTE_DIR/.git' ]; then git init -q '$REMOTE_DIR'; fi; git -C '$REMOTE_DIR' branch -m master main 2>/dev/null || true; git -C '$REMOTE_DIR' config receive.denyCurrentBranch updateInstead"

# Reconcile before pushing: the organism may have committed on the server since
# the last deploy. Merge those commits locally (abort on conflict) so the push
# stays a fast-forward and no history is ever discarded.
if git fetch --quiet "$REMOTE_URL" main 2>/dev/null; then
    if ! git merge-base --is-ancestor FETCH_HEAD HEAD 2>/dev/null; then
        printf 'Server has commits not present locally; merging them into the local branch.\n'
        if ! git merge --no-edit FETCH_HEAD; then
            git merge --abort || true
            printf 'Merge conflict with the server branch; reconcile manually before deploying.\n' >&2
            exit 1
        fi
    fi
fi

git push --quiet "$REMOTE_URL" HEAD:refs/heads/main
git push --quiet "$REMOTE_URL" --tags 2>/dev/null || true
DEPLOYED_COMMIT="$(git rev-parse HEAD)"

# Refresh the project dependencies in the server's own virtualenv. This is
# best-effort and idempotent: the venv's own interpreter is used (never the
# system one, never --break-system-packages), pip gets a single bounded attempt,
# and an unreachable package index only warns instead of failing the deploy. If
# the venv is absent the pre-provisioned environment is left untouched.
ssh "${SSH_ARGS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
    "set -eu; if [ -x '$REMOTE_DIR/.venv/bin/python' ]; then if '$REMOTE_DIR/.venv/bin/python' -m pip install -q --timeout 15 --retries 0 -e '$REMOTE_DIR[dev]'; then printf 'Refreshed Python dependencies in %s/.venv\n' '$REMOTE_DIR'; else printf 'WARNING: could not refresh Python dependencies in %s/.venv (package index unreachable?); continuing with the existing environment.\n' '$REMOTE_DIR' >&2; fi; else printf 'WARNING: %s/.venv/bin/python not found; skipping dependency refresh (provision the virtualenv first).\n' '$REMOTE_DIR' >&2; fi"

# The runtime environment is uploaded out of band and stays mode 600.
RUNTIME_UPLOAD="$(mktemp "${TMPDIR:-/tmp}/skynet-env.XXXXXX")"
trap 'rm -f "$RUNTIME_UPLOAD"' EXIT
cp "$RUNTIME_ENV" "$RUNTIME_UPLOAD"
chmod 600 "$RUNTIME_UPLOAD"
scp -q "${SSH_ARGS[@]}" "$RUNTIME_UPLOAD" "$REMOTE_USER@$REMOTE_HOST:.skynet-env-upload"
ssh "${SSH_ARGS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
    "set -eu; sudo -n install -o root -g root -m 600 \"\$HOME/.skynet-env-upload\" \"$REMOTE_DIR/config/skynet.env\"; rm -f \"\$HOME/.skynet-env-upload\""

# GitHub runtime binaries are large; sync them only when their hash changed.
LOCAL_GH="${GH_BINARY:-$HOME/.local/bin/gh}"
LOCAL_GITHUB_MCP="${GITHUB_MCP_BINARY:-$HOME/.local/bin/github-mcp-server}"
if [[ ! -x "$LOCAL_GH" || ! -x "$LOCAL_GITHUB_MCP" ]]; then
    printf 'GitHub runtime binaries not found: %s and %s\n' "$LOCAL_GH" "$LOCAL_GITHUB_MCP" >&2
    exit 1
fi

LOCAL_GH_HASH="$(sha256sum "$LOCAL_GH" | awk '{print $1}')"
LOCAL_GITHUB_MCP_HASH="$(sha256sum "$LOCAL_GITHUB_MCP" | awk '{print $1}')"
REMOTE_HASHES="$(ssh "${SSH_ARGS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
    "sha256sum \"\$HOME/.local/bin/gh\" \"\$HOME/.local/bin/github-mcp-server\" 2>/dev/null || true")"
REMOTE_GH_HASH="$(printf '%s\n' "$REMOTE_HASHES" | awk 'NF >= 2 && $2 ~ /\/gh$/ {print $1; exit}')"
REMOTE_GITHUB_MCP_HASH="$(printf '%s\n' "$REMOTE_HASHES" | awk 'NF >= 2 && $2 ~ /\/github-mcp-server$/ {print $1; exit}')"

if [[ "$LOCAL_GH_HASH" != "$REMOTE_GH_HASH" ]]; then
    scp -q "${SSH_ARGS[@]}" "$LOCAL_GH" "$REMOTE_USER@$REMOTE_HOST:.gh-upload"
fi
if [[ "$LOCAL_GITHUB_MCP_HASH" != "$REMOTE_GITHUB_MCP_HASH" ]]; then
    scp -q "${SSH_ARGS[@]}" "$LOCAL_GITHUB_MCP" "$REMOTE_USER@$REMOTE_HOST:.github-mcp-upload"
fi
ssh "${SSH_ARGS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
    "set -eu; install -d \"\$HOME/.local/bin\"; if [ -f \"\$HOME/.gh-upload\" ]; then install -m 755 \"\$HOME/.gh-upload\" \"\$HOME/.local/bin/gh\"; rm -f \"\$HOME/.gh-upload\"; fi; if [ -f \"\$HOME/.github-mcp-upload\" ]; then install -m 755 \"\$HOME/.github-mcp-upload\" \"\$HOME/.local/bin/github-mcp-server\"; rm -f \"\$HOME/.github-mcp-upload\"; fi; printf '%s\n' '$DEPLOYED_COMMIT' | sudo -n tee \"$REMOTE_DIR/.deployed-commit\" >/dev/null; touch \"$REMOTE_DIR/.deployment-refresh\""

UNIT_UPLOAD="$(mktemp "${TMPDIR:-/tmp}/skynet-unit.XXXXXX")"
ROLLBACK_UNIT_UPLOAD="$(mktemp "${TMPDIR:-/tmp}/skynet-rollback-unit.XXXXXX")"
TELEGRAM_UNIT_UPLOAD="$(mktemp "${TMPDIR:-/tmp}/skynet-telegram-unit.XXXXXX")"
UNIT_ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/skynet-units.XXXXXX.tar")"
trap 'rm -f "$RUNTIME_UPLOAD" "$UNIT_UPLOAD" "$ROLLBACK_UNIT_UPLOAD" "$TELEGRAM_UNIT_UPLOAD" "$UNIT_ARCHIVE"' EXIT
cp "$ROOT_DIR/deploy/skynet.service" "$UNIT_UPLOAD"
chmod 644 "$UNIT_UPLOAD"
cp "$ROOT_DIR/deploy/skynet-rollback.service" "$ROLLBACK_UNIT_UPLOAD"
chmod 644 "$ROLLBACK_UNIT_UPLOAD"
cp "$ROOT_DIR/deploy/skynet-telegram.service" "$TELEGRAM_UNIT_UPLOAD"
chmod 644 "$TELEGRAM_UNIT_UPLOAD"
tar -cf "$UNIT_ARCHIVE" \
    -C "$(dirname "$UNIT_UPLOAD")" \
    "$(basename "$UNIT_UPLOAD")" \
    "$(basename "$ROLLBACK_UNIT_UPLOAD")" \
    "$(basename "$TELEGRAM_UNIT_UPLOAD")"
scp -q "${SSH_ARGS[@]}" "$UNIT_ARCHIVE" "$REMOTE_USER@$REMOTE_HOST:.skynet-units-upload.tar"
ssh "${SSH_ARGS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
    "set -eu; tmp=\$(mktemp -d \"\$HOME/.skynet-units.XXXXXX\"); trap 'rm -rf \"\$tmp\"' EXIT; tar -xf \"\$HOME/.skynet-units-upload.tar\" -C \"\$tmp\"; sudo -n install -o root -g root -m 644 \"\$tmp/$(basename "$UNIT_UPLOAD")\" /etc/systemd/system/skynet.service; sudo -n install -o root -g root -m 644 \"\$tmp/$(basename "$ROLLBACK_UNIT_UPLOAD")\" /etc/systemd/system/skynet-rollback.service; sudo -n install -o root -g root -m 644 \"\$tmp/$(basename "$TELEGRAM_UNIT_UPLOAD")\" /etc/systemd/system/skynet-telegram.service; rm -f \"\$HOME/.skynet-units-upload.tar\"; sudo -n systemctl daemon-reload; sudo -n systemctl enable skynet.service skynet-telegram.service"

printf 'Deploy completed at %s.\n' "$DEPLOYED_COMMIT"
