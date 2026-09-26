#!/usr/bin/env bash
# Reconcile the local repository with the server's own checkout.
#
# The server keeps the shared history and the organism commits to it directly,
# so which side is ahead flips depending on who worked last. This script is the
# single reintroduction point back into the local tree: it refuses a dirty local
# worktree (the precondition deploy.sh also enforces), fetches the server branch,
# and then either fast-forwards local or pushes local -- never a three-way merge,
# so a genuine divergence is reported instead of silently resolved.
#
#   scripts/sync.sh                    pull/push the shared branch
#   scripts/sync.sh --prune-worktrees  also drop the server's registered
#                                      self-improvement worktrees (root-owned,
#                                      removed via sudo -n) and prune stale ones
#   scripts/sync.sh --public           also run publish-public.sh --check
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

PRUNE_WORKTREES=false
RUN_PUBLIC=false
for arg in "$@"; do
    case "$arg" in
        --prune-worktrees) PRUNE_WORKTREES=true ;;
        --public) RUN_PUBLIC=true ;;
        -h|--help)
            printf 'usage: %s [--prune-worktrees] [--public]\n' "$0"
            exit 0
            ;;
        *)
            printf 'unknown argument: %s\n' "$arg" >&2
            exit 2
            ;;
    esac
done

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

cd "$ROOT_DIR"

# A dirty local tree means "uncommitted work would be lost or collide"; the same
# refusal deploy.sh makes. Sync moves commits only, never files.
if [[ -n "$(git status --porcelain)" ]]; then
    printf 'Local worktree is dirty; commit or stash before syncing.\n' >&2
    exit 1
fi

SSH_ARGS=(
    -o BatchMode=yes
    -o PreferredAuthentications=publickey
    -o PasswordAuthentication=no
    -o ConnectTimeout=5
    -i "$SSH_KEY"
)
export GIT_SSH_COMMAND="ssh ${SSH_ARGS[*]}"
REMOTE_URL="ssh://${REMOTE_USER}@${REMOTE_HOST}${REMOTE_DIR}"

if [[ "$PRUNE_WORKTREES" == true ]]; then
    printf 'Pruning server worktrees under %s...\n' "$REMOTE_DIR"
    ssh "${SSH_ARGS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
        "set -eu; cd '$REMOTE_DIR'; git worktree list --porcelain | awk '/^worktree /{print \$2}' | while read -r p; do if [ \"\$p\" = \"\$PWD\" ]; then continue; fi; printf '  remove %s\\n' \"\$p\"; sudo -n rm -rf \"\$p\"; sudo -n rm -rf \".git/worktrees/\$(basename \"\$p\")\"; done; sudo -n git -c safe.directory=\"\$PWD\" worktree prune"
fi

git fetch --quiet "$REMOTE_URL" main
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse FETCH_HEAD)"

if [[ "$LOCAL_SHA" == "$REMOTE_SHA" ]]; then
    printf 'Already in sync at %s.\n' "${LOCAL_SHA:0:7}"
elif git merge-base --is-ancestor "$LOCAL_SHA" "$REMOTE_SHA"; then
    printf 'Server is ahead; fast-forwarding local.\n'
    git merge --ff-only FETCH_HEAD
elif git merge-base --is-ancestor "$REMOTE_SHA" "$LOCAL_SHA"; then
    printf 'Local is ahead; pushing to server.\n'
    git push --quiet "$REMOTE_URL" HEAD:refs/heads/main
else
    printf 'Local and server have diverged; reconcile manually before syncing.\n' >&2
    exit 1
fi

git log -1 --format='Synced at %h %s'

if [[ "$RUN_PUBLIC" == true ]]; then
    "$ROOT_DIR/scripts/publish-public.sh" --check
fi