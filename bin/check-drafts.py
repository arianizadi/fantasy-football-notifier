#!/usr/bin/env python3
"""Watch both leagues' draft status and sync the roster the moment one finishes.

The post-draft cron entries hardcode dates, which silently fire on the wrong
day if a commissioner moves the draft. This runs hourly, asks each provider
for the real draft state, and refreshes the snapshot as soon as a draft has
completed but that league's cached roster is still empty. It tracks providers
independently, so leagues drafting on different dates each trigger a sync.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifier.config import load_config  # noqa: E402
from notifier.drafts import completed_unsynced_league_keys  # noqa: E402
from notifier.logging_utils import configure_logging, structured_log  # noqa: E402
from notifier.notify import send_plain  # noqa: E402
from notifier.roster import load_snapshot, refresh_snapshot  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")


def sleeper_draft_state(session: requests.Session, league_id: str) -> tuple[str, datetime | None]:
    response = session.get(
        f"https://api.sleeper.app/v1/league/{league_id}/drafts", timeout=20
    )
    response.raise_for_status()
    drafts = response.json() or []
    if not drafts:
        return "unknown", None
    draft = drafts[0]
    start = draft.get("start_time")
    when = datetime.fromtimestamp(start / 1000, tz=timezone.utc) if start else None
    return str(draft.get("status") or "unknown"), when


def espn_draft_state(config) -> tuple[str, datetime | None]:
    url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/"
        f"{config.espn_year}/segments/0/leagues/{config.espn_league_id}"
    )
    response = requests.get(
        url,
        params={"view": "mSettings"},
        cookies={"SWID": config.espn_swid, "espn_s2": config.espn_s2},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=25,
    )
    response.raise_for_status()
    settings = (response.json().get("settings") or {}).get("draftSettings") or {}
    raw = settings.get("date")
    when = datetime.fromtimestamp(raw / 1000, tz=timezone.utc) if raw else None
    if when is None:
        return "unknown", None
    return ("complete" if when < datetime.now(timezone.utc) else "pre_draft"), when


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    configure_logging()
    config = load_config()
    session = requests.Session()
    snapshot = load_snapshot(config)

    states: dict[str, tuple[str, datetime | None]] = {}
    labels: dict[str, str] = {}
    if config.espn_enabled:
        league_key = f"espn:{config.espn_league_id}"
        labels[league_key] = "ESPN"
        try:
            states[league_key] = espn_draft_state(config)
        except requests.RequestException as error:
            structured_log(logging.WARNING, "draft.espn_check_failed", error=str(error))
    for league in snapshot.leagues:
        if league.provider == "sleeper":
            labels[league.key] = league.name
            try:
                states[league.key] = sleeper_draft_state(session, league.league_id)
            except requests.RequestException as error:
                structured_log(
                    logging.WARNING, "draft.sleeper_check_failed", error=str(error)
                )

    for league_key, (status, when) in states.items():
        structured_log(
            logging.INFO,
            "draft.status",
            league=labels.get(league_key, league_key),
            status=status,
            draftAt=when.astimezone(PACIFIC).isoformat() if when else None,
        )

    pending_keys = completed_unsynced_league_keys(snapshot, states)
    if not pending_keys:
        return 0

    structured_log(logging.INFO, "draft.completed_syncing", leagueKeys=pending_keys)
    refreshed = refresh_snapshot(config)
    synced_keys = [key for key in pending_keys if refreshed.is_drafted(key)]
    synced_players = sum(len(refreshed.mine(key)) for key in synced_keys)
    if synced_players:
        send_plain(
            session,
            config,
            f"Draft detected - synced <b>{synced_players}</b> players across "
            f"{len(synced_keys)} newly drafted league(s). Roster alerts are live.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
