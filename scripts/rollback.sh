#!/usr/bin/env bash

set -euo pipefail

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
TARGET="${1:-}"

if [[ -z "$TARGET" ]]; then
    printf 'usage: %s <commit-or-deployed-tag>\n' "$0" >&2
    exit 2
fi

SSH_ARGS=(
    -o BatchMode=yes
    -o PreferredAuthentications=publickey
    -o PasswordAuthentication=no
    -o ConnectTimeout=5
    -i "$SSH_KEY"
)

# The server repository carries the shared history, so a rollback is a plain
# hard reset to a commit or tag that is already present. No mirror involved.
ssh "${SSH_ARGS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
    "set -eu; target=$REMOTE_DIR; git -C \"\$target\" rev-parse --verify \"$TARGET^{commit}\" >/dev/null; sudo -n systemctl stop skynet.service; git -C \"\$target\" reset --hard \"$TARGET\"; printf '%s\n' \"\$(git -C \"\$target\" rev-parse HEAD)\" | sudo -n tee \"\$target/.deployed-commit\" >/dev/null; touch \"\$target/.deployment-refresh\"; sudo -n systemctl start skynet.service"

printf 'Rollback requested for %s on %s.\n' "$TARGET" "$REMOTE_HOST"
