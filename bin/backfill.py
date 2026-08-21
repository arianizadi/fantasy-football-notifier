#!/usr/bin/env python3
"""Replay reporter tweets from a past window and send any missed alerts.

Needed when alerts were generated but never delivered - a Telegram rejection
marks the item seen while the message is lost, so normal dedupe guarantees it
is never retried. X's recent-search covers seven days, which is the recovery
limit.

Delivered alerts are labelled DELAYED so a two-day-old injury is not mistaken
for breaking news. Costs one X read per tweet scanned.

    bin/backfill.py --since 2026-08-20T01:00:00Z [--send]

Without --send it only reports what it would deliver.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifier.config import load_config  # noqa: E402
from notifier.logging_utils import configure_logging, structured_log  # noqa: E402
from notifier.notify import _post, format_alert  # noqa: E402
from notifier.pipeline import Notifier  # noqa: E402
from notifier.sources.reporters import ALL_REPORTERS  # noqa: E402
from notifier.sources.twitter import TwitterStream  # noqa: E402

SEARCH_URL = "https://api.x.com/2/tweets/search/recent"
MAX_PAGES = 3
SEARCH_WINDOW_DAYS = 7


def fetch_window(session, token, since):
    query = "(" + " OR ".join(f"from:{h}" for h in ALL_REPORTERS) + ") -is:retweet -is:reply"
    tweets, users, page_token = [], [], None
    for _ in range(MAX_PAGES):
        params = {
            "query": query,
            "max_results": 100,
            "tweet.fields": "created_at,author_id,text",
            "expansions": "author_id",
            "user.fields": "username",
            "start_time": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if page_token:
            params["next_token"] = page_token
        response = session.get(
            SEARCH_URL, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=30
        )
        response.raise_for_status()
        payload = response.json()
        tweets += payload.get("data", [])
        users += payload.get("includes", {}).get("users", [])
        page_token = payload.get("meta", {}).get("next_token")
        if not page_token:
            break
    return tweets, users


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", required=True, help="ISO8601 UTC, e.g. 2026-08-20T01:00:00Z")
    parser.add_argument("--send", action="store_true", help="actually deliver (default: report)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore dedupe; use when a failed send already marked items seen",
    )
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    configure_logging()
    config = load_config()

    since = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
    oldest = datetime.now(timezone.utc) - timedelta(days=SEARCH_WINDOW_DAYS)
    if since < oldest:
        print(f"--since is beyond X's {SEARCH_WINDOW_DAYS}-day search window; clamping.")
        since = oldest

    token = os.environ.get("TWITTER_BEARER_TOKEN", "").strip()
    if not token:
        print("TWITTER_BEARER_TOKEN not set")
        return 1

    session = requests.Session()
    notifier = Notifier(config)
    stream = TwitterStream(token, queue.Queue())
    stream.set_player_index(notifier._player_index)

    tweets, users = fetch_window(session, token, since)
    print(f"scanned {len(tweets)} tweets since {since:%Y-%m-%d %H:%M}Z")

    delivered = 0
    # --force skips the persisted dedupe store, but two reporters covering the
    # same event must still collapse to one message within a single run.
    seen_this_run: set[str] = set()
    for tweet in tweets:
        for item in stream._to_items({"data": tweet, "includes": {"users": users}}):
            if not item.player_name:
                continue
            alert = notifier._evaluate(item)
            if alert is None:
                continue
            event = alert.classification.event_type
            # Collapse repeats of the same event across reporters. --force
            # bypasses this to recover items already marked seen by a failed
            # delivery, which normal dedupe would otherwise suppress forever.
            run_key = notifier.seen.semantic_key(item.player_name, event)
            if run_key in seen_this_run:
                continue
            if not args.force and not notifier.seen.is_semantically_new(
                item.player_name, event
            ):
                continue
            seen_this_run.add(run_key)

            label = f"[{alert.classification.severity}/5] {item.player_name}"
            if not args.send:
                # A dry run must not mutate state. Recording here once made the
                # subsequent real run deliver nothing at all.
                print(f"  WOULD SEND {label}: {item.headline[:70]}")
                delivered += 1
                break

            notifier.seen.record_semantic(item.player_name, event)
            notifier.seen.record(item)

            text = (
                "<b>DELAYED</b> - recovered after a delivery outage, "
                f"posted {tweet.get('created_at', '')[:16].replace('T', ' ')}Z\n\n"
                + format_alert(alert)
            )
            if _post(session, config, {"text": text,
                                       "link_preview_options": {"is_disabled": True},
                                       "disable_notification": True}) is not None:
                delivered += 1
                print(f"  SENT {label}")
            break

    if args.send:
        notifier.seen.save()
    structured_log(logging.INFO, "backfill.complete", scanned=len(tweets), delivered=delivered)
    print(f"{'delivered' if args.send else 'would deliver'}: {delivered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
