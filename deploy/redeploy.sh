#!/usr/bin/env bash
# Pull the latest main and restart the notifier.
# .env, state/*.json and .venv/ are gitignored, so they survive the reset.
set -euo pipefail
cd /home/arian/services/fantasy-news-notifier

echo "==> current: $(git log --oneline -1)"
git fetch --quiet origin
if [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ]; then
  echo "==> already up to date"
else
  git reset --hard --quiet origin/main
  echo "==> updated: $(git log --oneline -1)"
fi

./.venv/bin/pip install -q -r requirements.txt
if [ -f requirements-dev.txt ]; then ./.venv/bin/pip install -q -r requirements-dev.txt; fi

echo "==> tests"
./.venv/bin/python -m pytest -q

echo "==> restart"
sudo systemctl restart fantasy-news-notifier
sleep 6
systemctl is-active fantasy-news-notifier
journalctl -t fantasy-news-notify -n 4 --no-pager -o cat | grep -o '"event":"[a-z_.]*"' | sed 's/^/   /'
