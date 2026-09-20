#!/usr/bin/env bash
# Owner alert for a failed systemd unit.
#
# This unit is wired as OnFailure= and must never fail itself: a failed alert
# must not cascade into a restart loop. Every path exits 0. The bot token is
# read from the environment and written into a mode-600 curl config file, never
# passed on the command line, because /proc/<pid>/cmdline is world-readable.
set -u

UNIT="${1:-unknown}"
REASON="${2:-unit failed}"
TOKEN="${SKYNET_TELEGRAM_BOT_TOKEN:-}"
CHAT_ID="${SKYNET_TELEGRAM_CHAT_ID:-}"
PROXY="${SKYNET_TELEGRAM_PROXY:-}"
TIMEOUT="${SKYNET_ALERT_TIMEOUT:-10}"

if [[ -z "$TOKEN" || -z "$CHAT_ID" ]]; then
    echo "alert skipped: SKYNET_TELEGRAM_BOT_TOKEN or SKYNET_TELEGRAM_CHAT_ID is not set" >&2
    exit 0
fi

HOST_VALUE="$(hostname 2>/dev/null || echo unknown)"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
MESSAGE="SkyNet alert: ${UNIT} — ${REASON}
host: ${HOST_VALUE}
time: ${STAMP}"

if command -v journalctl >/dev/null 2>&1; then
    JOURNAL="$(journalctl -u "$UNIT" -n 5 --no-pager 2>/dev/null || true)"
    if [[ -n "$JOURNAL" ]]; then
        MESSAGE="${MESSAGE}
journal (last 5):
${JOURNAL}"
    fi
fi

MESSAGE_FILE="$(mktemp 2>/dev/null || echo /tmp/skynet-alert-message.$$)"
CONFIG_FILE="$(mktemp 2>/dev/null || echo /tmp/skynet-alert-config.$$)"
chmod 600 "$MESSAGE_FILE" "$CONFIG_FILE" 2>/dev/null || true
cleanup() { rm -f "$MESSAGE_FILE" "$CONFIG_FILE"; }
trap cleanup EXIT

printf '%s' "$MESSAGE" > "$MESSAGE_FILE"
{
    printf 'url = "https://api.telegram.org/bot%s/sendMessage"\n' "$TOKEN"
    printf 'data-urlencode = "chat_id=%s"\n' "$CHAT_ID"
} > "$CONFIG_FILE"

CURL_EXTRA=()
if [[ -n "$PROXY" ]]; then
    CURL_EXTRA+=(--proxy "$PROXY")
fi

curl -sS --max-time "$TIMEOUT" -o /dev/null -K "$CONFIG_FILE" "${CURL_EXTRA[@]}" \
    --data-urlencode "text@${MESSAGE_FILE}" || true
exit 0
