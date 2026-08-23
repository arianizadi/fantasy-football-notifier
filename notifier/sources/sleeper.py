"""Sleeper public API: trending adds and the player-id index.

Free and unauthenticated. Used as a corroborating "the market is reacting"
signal so waiver-relevant news about players not on the roster can still
surface, without paying for a social firehose.

Sleeper asks callers to stay under 1000 requests/minute; the player index is
~5MB and their API documentation says to fetch it at most once per day, so it
is cached on disk for twenty-four hours.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from ..logging_utils import structured_log

BASE_URL = "https://api.sleeper.app/v1"
# Sleeper documents this full player-map call as a once-daily operation.
PLAYER_INDEX_TTL_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT = 20
TRENDING_TIMEOUT = 10


class PlayerIndex(dict[str, Any]):
    """Sleeper records plus the timestamp of the data being represented.

    This remains a normal ``dict`` for existing callers while carrying cache
    freshness through to ``DepthCharts`` and ultimately the alert text.
    """

    def __init__(
        self,
        records: dict[str, Any] | None = None,
        *,
        refreshed_at: datetime | None = None,
        stale: bool = False,
    ) -> None:
        super().__init__(records or {})
        self.refreshed_at = refreshed_at
        self.stale = stale


def load_player_index(
    state_dir: Path,
    session: requests.Session,
    *,
    write_cache: bool = True,
) -> PlayerIndex:
    """Map Sleeper player_id -> player record, cached for twenty-four hours.

    ``write_cache=False`` supports a genuinely read-only dry run: an existing
    cache may be read, but a live fetch is kept in memory only.
    """
    cache_file = state_dir / "sleeper-players.json"
    cached_records: dict[str, Any] | None = None
    cached_refreshed_at: datetime | None = None
    if cache_file.exists():
        try:
            records = json.loads(cache_file.read_text())
            if not isinstance(records, dict) or not records:
                raise ValueError("Sleeper cache root is not a populated object")
            refreshed_at = datetime.fromtimestamp(cache_file.stat().st_mtime, timezone.utc)
            cached_records = records
            cached_refreshed_at = refreshed_at
            if (time.time() - cache_file.stat().st_mtime) < PLAYER_INDEX_TTL_SECONDS:
                return PlayerIndex(records, refreshed_at=refreshed_at)
        except (ValueError, OSError) as error:
            structured_log(logging.WARNING, "sleeper.cache_unreadable", error=str(error))

    try:
        response = session.get(f"{BASE_URL}/players/nfl", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        players = response.json()
        if not isinstance(players, dict) or not players:
            raise ValueError("Sleeper player response is not a populated object")
    except (requests.RequestException, ValueError):
        if cached_records is None:
            raise
        structured_log(
            logging.WARNING,
            "sleeper.stale_cache_fallback",
            refreshedAt=(
                cached_refreshed_at.isoformat()
                if cached_refreshed_at is not None
                else "unknown"
            ),
        )
        return PlayerIndex(
            cached_records,
            refreshed_at=cached_refreshed_at,
            stale=True,
        )

    # Keep only the fields needed for matching and depth-chart lookups so the
    # on-disk cache stays a few hundred KB instead of ~5MB.
    trimmed: dict[str, Any] = {
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
    refreshed_at = datetime.now(timezone.utc)
    if not write_cache:
        structured_log(
            logging.INFO,
            "sleeper.player_index_refreshed",
            playerCount=len(trimmed),
            cached=False,
        )
        return PlayerIndex(trimmed, refreshed_at=refreshed_at)

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{cache_file.name}.",
            suffix=".tmp",
            dir=cache_file.parent,
            text=True,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(trimmed, handle, separators=(",", ":"))
            os.replace(temporary, cache_file)
        finally:
            temporary.unlink(missing_ok=True)
        # Filesystems are the authority for cached-data age. Read it back so
        # the displayed timestamp matches what the next process will report.
        refreshed_at = datetime.fromtimestamp(cache_file.stat().st_mtime, timezone.utc)
    except OSError as error:
        structured_log(logging.WARNING, "sleeper.cache_write_failed", error=str(error))

    structured_log(logging.INFO, "sleeper.player_index_refreshed", playerCount=len(trimmed))
    return PlayerIndex(trimmed, refreshed_at=refreshed_at)


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
