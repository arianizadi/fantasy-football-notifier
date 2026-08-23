#!/usr/bin/env bash
# Pull the latest main and restart the notifier.
#
# .env, state JSON/SQLite files and .venv/ are gitignored, so `git reset --hard` leaves
# them alone. That is the design, but it is one .gitignore edit away from
# being false, and losing .env means re-issuing every credential. So this
# script snapshots them first and verifies they survived afterwards.
set -euo pipefail
umask 077

cd "$(dirname "$0")/.."
ROOT="$PWD"
BACKUPS="$HOME/backups"
REQUIRED_KEYS=(TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID OPENROUTER_API_KEY ESPN_SWID ESPN_S2)
STAGING=""

fail() { echo "FATAL: $*" >&2; exit 1; }
state_entry_count() {
  if [ ! -d state ]; then
    echo 0
    return
  fi
  find state -mindepth 1 -maxdepth 1 -print | wc -l
}
cleanup() {
  if [ -n "$STAGING" ] && [ -d "$STAGING" ]; then
    rm -rf -- "$STAGING"
  fi
}
trap cleanup EXIT

# --- pre-flight -----------------------------------------------------------
[ -f .env ] || fail ".env missing before deploy; refusing to run"
for key in "${REQUIRED_KEYS[@]}"; do
  grep -q "^${key}=" .env || fail ".env is missing ${key}; refusing to run"
done
chmod 600 .env
if [ -d state ]; then
  find state -type d -exec chmod 700 -- {} +
  find state -type f -exec chmod 600 -- {} +
fi

mkdir -p "$BACKUPS"
chmod 700 "$BACKUPS"
find "$BACKUPS" -maxdepth 1 -type f -name 'notifier-*.tar.gz' -exec chmod 600 -- {} +
STAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE="$BACKUPS/notifier-$STAMP.tar.gz"
STAGING=$(mktemp -d "$BACKUPS/.notifier-backup.XXXXXX")
mkdir -p "$STAGING/state"
cp -p .env "$STAGING/.env"
chmod 600 "$STAGING/.env"
if [ -d state ]; then
  cp -a state/. "$STAGING/state/"
fi

# Copying a live WAL database file-by-file can combine pages from different
# transactions.  SQLite's online backup API takes one consistent snapshot while
# the notifier keeps running.  Replace the staged raw copy and omit transient
# WAL/SHM files; all other state files remain copied above.
EVENTS_DB="state/news-events.sqlite3"
if [ -f "$EVENTS_DB" ]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
  [ -x "$PYTHON_BIN" ] || PYTHON_BIN=$(command -v python3) || fail "python3 is required to snapshot SQLite"
  rm -f -- \
    "$STAGING/$EVENTS_DB" \
    "$STAGING/$EVENTS_DB-wal" \
    "$STAGING/$EVENTS_DB-shm"
  "$PYTHON_BIN" - "$ROOT/$EVENTS_DB" "$STAGING/$EVENTS_DB" <<'PY'
import sqlite3
import sys
from urllib.parse import quote

source_path, destination_path = sys.argv[1:]
source_uri = f"file:{quote(source_path, safe='/')}?mode=ro"
with sqlite3.connect(source_uri, uri=True, timeout=30) as source:
    with sqlite3.connect(destination_path, timeout=30) as destination:
        source.backup(destination)
        destination.execute("PRAGMA journal_mode=DELETE")
        result = destination.execute("PRAGMA quick_check").fetchone()
        if result is None or result[0] != "ok":
            raise RuntimeError(f"SQLite backup quick_check failed: {result!r}")
PY
  chmod 600 "$STAGING/$EVENTS_DB"
fi

ARCHIVE_STAGED="$STAGING/notifier.tar.gz"
tar -C "$STAGING" -czf "$ARCHIVE_STAGED" .env state
chmod 600 "$ARCHIVE_STAGED"
mv "$ARCHIVE_STAGED" "$ARCHIVE"
ENV_SHA_BEFORE=$(sha256sum .env | cut -d' ' -f1)
STATE_BEFORE=$(state_entry_count)
echo "==> backed up to $ARCHIVE"

# Keep the ten most recent snapshots.
shopt -s nullglob
ARCHIVES=("$BACKUPS"/notifier-*.tar.gz)
if [ "${#ARCHIVES[@]}" -gt 10 ]; then
  rm -f -- "${ARCHIVES[@]:0:${#ARCHIVES[@]}-10}"
fi
shopt -u nullglob

# --- update ---------------------------------------------------------------
echo "==> current: $(git log --oneline -1)"
git fetch --quiet origin
if [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ]; then
  echo "==> already up to date"
else
  git reset --hard --quiet origin/main
  echo "==> updated: $(git log --oneline -1)"
fi

# --- post-flight: did anything eat the secrets? ---------------------------
if [ ! -f .env ]; then
  tar xzf "$ARCHIVE" -C "$ROOT"
  fail ".env vanished during update and was restored from $ARCHIVE. Check .gitignore."
fi
ENV_SHA_AFTER=$(sha256sum .env | cut -d' ' -f1)
[ "$ENV_SHA_BEFORE" = "$ENV_SHA_AFTER" ] || fail ".env changed during update; restore from $ARCHIVE"
STATE_AFTER=$(state_entry_count)
[ "$STATE_BEFORE" -eq "$STATE_AFTER" ] || echo "WARN: state file count $STATE_BEFORE -> $STATE_AFTER"
echo "==> .env and state intact"

# --- deps, tests, restart -------------------------------------------------
./.venv/bin/pip install -q -r requirements.txt
[ -f requirements-dev.txt ] && ./.venv/bin/pip install -q -r requirements-dev.txt
# The direct ESPN adapter replaced espn-api. pip does not remove dependencies
# that disappear from requirements, and the old package pins an insecure
# urllib3 release, so clean that one known orphan before validating the venv.
if ./.venv/bin/pip show espn-api >/dev/null 2>&1; then
  ./.venv/bin/pip uninstall -q -y espn-api
fi
./.venv/bin/python -m pip check

echo "==> tests"
./.venv/bin/python -m pytest -q

echo "==> restart"
OLD_PID=$(systemctl show fantasy-news-notifier -p MainPID --value)
if sudo -n systemctl restart fantasy-news-notifier >/dev/null 2>&1; then
  echo "==> requested restart through systemd"
else
  # Runner's service process runs as this deployment user, but sudo may
  # require an interactive password. Restart=always lets SIGINT use the
  # runner's KeyboardInterrupt/finally shutdown path (including the version
  # being upgraded from) without weakening sudoers.
  RESTART_POLICY=$(systemctl show fantasy-news-notifier -p Restart --value)
  [ "$RESTART_POLICY" = "always" ] || fail "sudo unavailable and Restart=$RESTART_POLICY"
  [ "$OLD_PID" -gt 1 ] 2>/dev/null || fail "service has no live MainPID"
  PID_OWNER=$(ps -o user= -p "$OLD_PID" | tr -d '[:space:]')
  [ "$PID_OWNER" = "$(id -un)" ] || fail "service process is owned by $PID_OWNER"
  kill -INT "$OLD_PID"
  echo "==> requested supervised restart through PID $OLD_PID"
fi

STATUS=""
NEW_PID=""
for _ in $(seq 1 45); do
  STATUS=$(systemctl is-active fantasy-news-notifier 2>/dev/null || true)
  NEW_PID=$(systemctl show fantasy-news-notifier -p MainPID --value 2>/dev/null || true)
  if [ "$STATUS" = "active" ] && [ "$NEW_PID" -gt 1 ] 2>/dev/null && [ "$NEW_PID" != "$OLD_PID" ]; then
    break
  fi
  sleep 2
done
echo "==> service: $STATUS (pid $NEW_PID)"
[ "$STATUS" = "active" ] || fail "service is not active after restart"
[ "$NEW_PID" -gt 1 ] 2>/dev/null || fail "service has no MainPID after restart"
[ "$NEW_PID" != "$OLD_PID" ] || fail "service PID did not change after restart"

# A process can briefly become active and then crash after imports or provider
# initialization. Require the same supervised PID to remain healthy for ten
# seconds before calling the deployment successful.
RESTARTS_AT_START=$(systemctl show fantasy-news-notifier -p NRestarts --value)
sleep 10
STABLE_STATUS=$(systemctl is-active fantasy-news-notifier 2>/dev/null || true)
STABLE_PID=$(systemctl show fantasy-news-notifier -p MainPID --value 2>/dev/null || true)
STABLE_RESTARTS=$(systemctl show fantasy-news-notifier -p NRestarts --value 2>/dev/null || true)
[ "$STABLE_STATUS" = "active" ] || fail "service failed during stabilization"
[ "$STABLE_PID" = "$NEW_PID" ] || fail "service PID changed during stabilization"
[ "$STABLE_RESTARTS" = "$RESTARTS_AT_START" ] || fail "service restarted during stabilization"

# The first upgraded start creates/migrates the durable news journal. Verify
# both SQLite integrity and the tables required by search/feedback before the
# backup made by the next deployment has to depend on it.
./.venv/bin/python - "$ROOT/state/news-events.sqlite3" <<'PY'
import sqlite3
import sys

database = sys.argv[1]
with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=10) as connection:
    result = connection.execute("PRAGMA quick_check").fetchone()
    if result is None or result[0] != "ok":
        raise RuntimeError(f"journal quick_check failed: {result!r}")
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    required = {"news_events", "news_events_fts", "event_store_meta"}
    missing = required - tables
    if missing:
        raise RuntimeError(f"journal tables missing: {sorted(missing)}")
PY

# Validate the bot credential and destination without logging the credential or
# sending a synthetic fantasy alert. Telegram errors are intentionally reduced
# to status codes so the token-bearing request URL never reaches deployment logs.
./.venv/bin/python - <<'PY'
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv(".env", override=True)
token = os.environ["TELEGRAM_BOT_TOKEN"]
chat_id = os.environ["TELEGRAM_CHAT_ID"]
try:
    response = requests.get(
        f"https://api.telegram.org/bot{token}/getChat",
        params={"chat_id": chat_id},
        timeout=15,
    )
except requests.RequestException:
    print("Telegram deployment probe failed: request error", file=sys.stderr)
    raise SystemExit(1)
if not response.ok:
    print(f"Telegram deployment probe failed: HTTP {response.status_code}", file=sys.stderr)
    raise SystemExit(1)
print("==> Telegram bot and destination: OK")
PY

RECENT_EVENTS=$(journalctl -t fantasy-news-notify -n 5 --no-pager -o cat \
  | grep -o '"event":"[a-z_.]*"' || true)
if [ -n "$RECENT_EVENTS" ]; then
  while IFS= read -r event; do
    printf '     %s\n' "$event"
  done <<< "$RECENT_EVENTS"
else
  echo "     no structured event in the last five journal lines"
fi
