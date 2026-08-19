#!/usr/bin/env python3
"""Verify reporter handles and measure their real posting volume + cost.

Reads are billed per post, so guessed handles are worse than useless: they
resolve to nothing and stream silently. Run this before adding any account.

WARNING: this script SPENDS MONEY. Every post it reads to compute the average
costs $0.005, the same as a production read. A full 4-page sweep of 15 accounts
reads ~1,000 posts and costs ~$5. It defaults to one page per account and
prints its own estimated cost before doing anything. Pass --deep for the full
sweep, --yes to skip the confirmation.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from notifier.sources.reporters import ALL_REPORTERS, BEAT_OPTIONAL  # noqa: E402

RATE_PER_POST = 0.005
SEASON_DAYS = 150
PAGE_SIZE = 100


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    token = os.environ.get("TWITTER_BEARER_TOKEN", "").strip()
    if not token:
        print("TWITTER_BEARER_TOKEN not set")
        return 1
    headers = {"Authorization": f"Bearer {token}"}

    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    deep = "--deep" in sys.argv
    assume_yes = "--yes" in sys.argv
    pages = 4 if deep else 1

    candidates = list(dict.fromkeys(ALL_REPORTERS + BEAT_OPTIONAL + argv))

    worst_case = len(candidates) * pages * PAGE_SIZE
    print(
        f"This reads up to {worst_case} posts at ${RATE_PER_POST}/post "
        f"= up to ${worst_case * RATE_PER_POST:.2f}."
    )
    if not deep:
        print("Using 1 page per account. Pass --deep for a 4-page sweep (4x the cost).")
    if not assume_yes:
        if input("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Aborted; nothing spent.")
            return 0
    resolved = requests.get(
        "https://api.x.com/2/users/by",
        headers=headers,
        params={"usernames": ",".join(candidates[:100])},
        timeout=25,
    ).json()
    found = {u["username"].lower() for u in resolved.get("data", [])}
    for handle in candidates:
        if handle.lower() not in found:
            print(f"  !! {handle} does not resolve on X")

    print(f"\n{'account':<20}{'posts/7d':>10}{'/day':>8}{'$/mo':>9}")
    print("-" * 47)
    total = 0.0
    for handle in candidates:
        if handle.lower() not in found:
            continue
        count, token_page = 0, None
        for _ in range(pages):
            params = {
                "query": f"from:{handle} -is:retweet -is:reply",
                "max_results": 100,
                "tweet.fields": "created_at",
            }
            if token_page:
                params["next_token"] = token_page
            response = requests.get(
                "https://api.x.com/2/tweets/search/recent",
                headers=headers,
                params=params,
                timeout=25,
            )
            if not response.ok:
                break
            payload = response.json()
            count += len(payload.get("data", []))
            token_page = payload.get("meta", {}).get("next_token")
            if not token_page:
                break
            time.sleep(0.3)
        # One page caps at 100, so a prolific account is undercounted; say so
        # rather than quietly reporting a floor as if it were the real rate.
        capped = count >= pages * PAGE_SIZE
        per_day = count / 7
        total += per_day
        flag = " (capped, real rate is higher)" if capped else ""
        print(
            f"{handle:<20}{count:>10}{per_day:>8.1f}"
            f"{per_day * RATE_PER_POST * 30:>9.2f}{flag}"
        )

    print("-" * 47)
    print(f"{'TOTAL':<20}{'':>10}{total:>8.1f}{total * RATE_PER_POST * 30:>9.2f}")
    print(f"\nseason estimate ({SEASON_DAYS}d): ${total * RATE_PER_POST * SEASON_DAYS:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
