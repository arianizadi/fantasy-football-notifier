#!/usr/bin/env python3
"""Capture real news items to disk for offline model evaluation.

Model evaluation has to replay identical inputs, but the X stream has no
backfill and RotoWire's feed holds five items. Without a stored corpus every
comparison runs on different data and the numbers mean nothing.

Each capture costs one X read per tweet scanned. Fixtures accumulate: a
capture merges into the existing file by guid rather than replacing it.

    bin/capture-fixtures.py --tweets 60 --rotowire
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from notifier.sources import rotowire  # noqa: E402
from notifier.sources.reporters import ALL_REPORTERS  # noqa: E402
from notifier.sources.sleeper import load_player_index  # noqa: E402
from notifier.sources.twitter import TwitterStream  # noqa: E402

FIXTURES = ROOT / "fixtures" / "news-items.json"
SEARCH_URL = "https://api.x.com/2/tweets/search/recent"


def load_existing() -> dict:
    if FIXTURES.exists():
        try:
            return {item["guid"]: item for item in json.loads(FIXTURES.read_text())}
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    return {}


def capture_tweets(session, token, limit, player_index) -> list[dict]:
    stream = TwitterStream(token, queue.Queue())
    stream.set_player_index(player_index)
    query = "(" + " OR ".join(f"from:{h}" for h in ALL_REPORTERS) + ") -is:retweet -is:reply"
    response = session.get(
        SEARCH_URL,
        headers={"Authorization": f"Bearer {token}"},
        params={
            "query": query,
            "max_results": min(limit, 100),
            "tweet.fields": "created_at,author_id,text",
            "expansions": "author_id",
            "user.fields": "username",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    users = payload.get("includes", {}).get("users", [])

    captured = []
    for tweet in payload.get("data", []):
        for item in stream._to_items({"data": tweet, "includes": {"users": users}}):
            captured.append(
                {
                    "guid": item.guid,
                    "source": item.source,
                    "player_name": item.player_name,
                    "headline": item.headline,
                    "body": item.body,
                    "url": item.url,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "expected_severity": None,  # fill in by hand to grade a model
                }
            )
    return captured


def capture_rotowire(session) -> list[dict]:
    items, _ = rotowire.fetch(session)
    return [
        {
            "guid": item.guid,
            "source": item.source,
            "player_name": item.player_name,
            "headline": item.headline,
            "body": item.body,
            "url": item.url,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "expected_severity": None,
        }
        for item in items
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tweets", type=int, default=0, help="tweets to capture (costs X reads)")
    parser.add_argument("--rotowire", action="store_true", help="capture the RSS feed")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    session = requests.Session()
    existing = load_existing()
    before = len(existing)

    if args.rotowire:
        for item in capture_rotowire(session):
            existing.setdefault(item["guid"], item)

    if args.tweets:
        token = os.environ.get("TWITTER_BEARER_TOKEN", "").strip()
        if not token:
            print("TWITTER_BEARER_TOKEN not set; skipping tweets")
        else:
            index = load_player_index(ROOT / "state", session)
            for item in capture_tweets(session, token, args.tweets, index):
                existing.setdefault(item["guid"], item)

    FIXTURES.parent.mkdir(parents=True, exist_ok=True)
    FIXTURES.write_text(json.dumps(list(existing.values()), indent=1))

    graded = sum(1 for i in existing.values() if i.get("expected_severity") is not None)
    print(f"fixtures: {before} -> {len(existing)} ({len(existing) - before} new)")
    print(f"  graded with expected_severity: {graded}")
    print(f"  file: {FIXTURES}")
    if graded == 0:
        print("  note: eval falls back to agreement-with-reference when none are graded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
