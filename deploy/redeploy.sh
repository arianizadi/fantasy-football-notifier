#!/usr/bin/env bash
# Pull the latest main and restart the notifier.
#
# .env, state/*.json and .venv/ are gitignored, so `git reset --hard` leaves
# them alone. That is the design, but it is one .gitignore edit away from
# being false, and losing .env means re-issuing every credential. So this
# script snapshots them first and verifies they survived afterwards.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
BACKUPS="$HOME/backups"
REQUIRED_KEYS=(TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID OPENROUTER_API_KEY ESPN_SWID ESPN_S2)

fail() { echo "FATAL: $*" >&2; exit 1; }

# --- pre-flight -----------------------------------------------------------
[ -f .env ] || fail ".env missing before deploy; refusing to run"
for key in "${REQUIRED_KEYS[@]}"; do
  grep -q "^${key}=" .env || fail ".env is missing ${key}; refusing to run"
done

mkdir -p "$BACKUPS"
STAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE="$BACKUPS/notifier-$STAMP.tar.gz"
tar czf "$ARCHIVE" .env state/ 2>/dev/null
ENV_SHA_BEFORE=$(sha256sum .env | cut -d' ' -f1)
STATE_BEFORE=$(ls state/ 2>/dev/null | wc -l)
echo "==> backed up to $ARCHIVE"

# Keep the ten most recent snapshots.
ls -1t "$BACKUPS"/notifier-*.tar.gz 2>/dev/null | tail -n +11 | xargs -r rm -f

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
STATE_AFTER=$(ls state/ 2>/dev/null | wc -l)
[ "$STATE_BEFORE" -eq "$STATE_AFTER" ] || echo "WARN: state file count $STATE_BEFORE -> $STATE_AFTER"
echo "==> .env and state intact"

# --- deps, tests, restart -------------------------------------------------
./.venv/bin/pip install -q -r requirements.txt
[ -f requirements-dev.txt ] && ./.venv/bin/pip install -q -r requirements-dev.txt

echo "==> tests"
./.venv/bin/python -m pytest -q

echo "==> restart"
sudo systemctl restart fantasy-news-notifier
sleep 6
STATUS=$(systemctl is-active fantasy-news-notifier)
echo "==> service: $STATUS"
[ "$STATUS" = "active" ] || fail "service is not active after restart"
journalctl -t fantasy-news-notify -n 5 --no-pager -o cat | grep -o '"event":"[a-z_.]*"' | sed 's/^/     /'
