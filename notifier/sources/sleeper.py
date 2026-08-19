"""Sleeper public API: trending adds and the player-id index.

Free and unauthenticated. Used as a corroborating "the market is reacting"
signal so waiver-relevant news about players not on the roster can still
surface, without paying for a social firehose.

Sleeper asks callers to stay under 1000 requests/minute; the player index is
~5MB so it is cached on disk and refreshed daily.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from ..logging_utils import structured_log

BASE_URL = "https://api.sleeper.app/v1"
PLAYER_INDEX_TTL_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT = 20
TRENDING_TIMEOUT = 10


def load_player_index(state_dir: Path, session: requests.Session) -> dict[str, Any]:
    """Map Sleeper player_id -> player record, cached on disk for a day."""
    cache_file = state_dir / "sleeper-players.json"
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < PLAYER_INDEX_TTL_SECONDS:
        try:
            return json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError) as error:
            structured_log(logging.WARNING, "sleeper.cache_unreadable", error=str(error))

    response = session.get(f"{BASE_URL}/players/nfl", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    players = response.json()

    # Keep only the fields needed for matching and depth-chart lookups so the
    # on-disk cache stays a few hundred KB instead of ~5MB.
    trimmed = {
        player_id: {
            "full_name": record.get("full_name") or "",
            "position": record.get("position") or "",
            "team": record.get("team") or "",
            "depth_chart_order": record.get("depth_chart_order"),
            "depth_chart_position": record.get("depth_chart_position") or "",
            "search_rank": record.get("search_rank"),
            "injury_status": record.get("injury_status") or "",
            "status": record.get("status") or "",
        }
        for player_id, record in players.items()
        if isinstance(record, dict) and record.get("full_name")
    }
    try:
        cache_file.write_text(json.dumps(trimmed, separators=(",", ":")))
    except OSError as error:
        structured_log(logging.WARNING, "sleeper.cache_write_failed", error=str(error))

    structured_log(logging.INFO, "sleeper.player_index_refreshed", playerCount=len(trimmed))
    return trimmed


def trending_adds(
    session: requests.Session,
    player_index: dict[str, Any],
    *,
    limit: int,
    lookback_hours: int = 6,
) -> dict[str, int]:
    """Return normalized player name -> add count over the lookback window."""
    response = session.get(
        f"{BASE_URL}/players/nfl/trending/add",
        params={"lookback_hours": lookback_hours, "limit": limit},
        timeout=TRENDING_TIMEOUT,
    )
    response.raise_for_status()

    from ..matcher import compact_key

    trending: dict[str, int] = {}
    for entry in response.json():
        record = player_index.get(str(entry.get("player_id")))
        if not record:
            continue
        trending[compact_key(record["full_name"])] = int(entry.get("count") or 0)
    return trending
