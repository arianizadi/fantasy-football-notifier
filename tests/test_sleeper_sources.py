import json
import os
import threading
from datetime import datetime, timezone
from unittest.mock import Mock

import requests

from notifier.models import RosterSnapshot
from notifier.pipeline import (
    Notifier,
    PLAYER_INDEX_REFRESH_SECONDS,
    PLAYER_INDEX_RETRY_SECONDS,
    _next_player_index_refresh_at,
)
from notifier.sources.sleeper import (
    PLAYER_INDEX_TTL_SECONDS,
    PlayerIndex,
    load_player_index,
)
from notifier.sources.sleeper_league import (
    BENCH_SLOT,
    NFL_INACTIVE_SLOT,
    RESERVE_SLOT,
    STARTER_SLOT,
    TAXI_SLOT,
    fetch_league_rosters,
    roster_slot,
)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


def test_full_player_index_uses_sleeper_daily_cadence() -> None:
    assert PLAYER_INDEX_TTL_SECONDS == 24 * 60 * 60
    assert PLAYER_INDEX_REFRESH_SECONDS == PLAYER_INDEX_TTL_SECONDS


def test_expired_cache_falls_back_during_sleeper_outage(tmp_path) -> None:
    cache = tmp_path / "sleeper-players.json"
    cache.write_text(json.dumps({"1": {"full_name": "George Kittle"}}))
    refreshed_at = datetime(2026, 8, 21, 15, 30, tzinfo=timezone.utc)
    os.utime(cache, (refreshed_at.timestamp(), refreshed_at.timestamp()))
    session = Mock()
    session.get.side_effect = requests.RequestException("offline")

    index = load_player_index(tmp_path, session)

    assert index["1"]["full_name"] == "George Kittle"
    assert index.refreshed_at == refreshed_at
    assert index.stale is True


def test_empty_or_stale_index_schedules_short_retry() -> None:
    now = 1_000_000.0
    stale = PlayerIndex(
        {"1": {"full_name": "George Kittle"}},
        refreshed_at=datetime.fromtimestamp(1, timezone.utc),
        stale=True,
    )

    assert _next_player_index_refresh_at({}, now) == now + PLAYER_INDEX_RETRY_SECONDS
    assert _next_player_index_refresh_at(stale, now) == now + PLAYER_INDEX_RETRY_SECONDS


def test_failed_daily_refresh_keeps_last_good_player_index(monkeypatch, tmp_path) -> None:
    notifier = Notifier.__new__(Notifier)
    old_index = PlayerIndex(
        {"1": {"full_name": "George Kittle", "position": "TE", "team": "SF"}},
        refreshed_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )
    old_depth = object()
    notifier._state_lock = threading.RLock()
    notifier._player_index = old_index
    notifier.depth = old_depth
    notifier.snapshot = RosterSnapshot(generated_at=None)
    notifier.config = Mock(state_dir=tmp_path, dry_run=False)
    notifier.session = Mock()
    notifier.twitter = Mock()
    monkeypatch.setattr(
        "notifier.pipeline.sleeper.load_player_index",
        Mock(side_effect=requests.RequestException("offline")),
    )

    assert notifier._rebuild_depth_charts(reload_player_index=True) is False
    assert notifier._player_index is old_index
    assert notifier.depth is old_depth
    notifier.twitter.set_player_index.assert_not_called()


def test_sleeper_roster_slot_tracks_all_unavailable_states() -> None:
    common = {
        "starters": {"starter"},
        "reserve": {"reserve"},
        "taxi": {"taxi"},
    }
    assert roster_slot("starter", nfl_status="Active", **common) == STARTER_SLOT
    assert roster_slot("bench", nfl_status="Active", **common) == BENCH_SLOT
    assert roster_slot("reserve", nfl_status="Active", **common) == RESERVE_SLOT
    assert roster_slot("taxi", nfl_status="Active", **common) == TAXI_SLOT
    assert roster_slot("bench", nfl_status="Inactive", **common) == NFL_INACTIVE_SLOT


def test_fetch_league_rosters_preserves_reserve_taxi_and_nfl_status() -> None:
    rosters = [
        {
            "owner_id": "me",
            "roster_id": 1,
            "players": ["1", "2", "3", "4", "5"],
            "starters": ["1"],
            "reserve": ["3"],
            "taxi": ["4"],
        }
    ]
    users = [{"user_id": "me", "display_name": "Arian", "metadata": {}}]
    session = Mock()
    session.get.side_effect = [Response(rosters), Response(users)]
    index = {
        "1": {"full_name": "Starter", "position": "TE", "team": "SF", "status": "Active"},
        "2": {"full_name": "Bench", "position": "TE", "team": "SF", "status": "Active"},
        "3": {"full_name": "Reserve", "position": "TE", "team": "SF", "status": "Active"},
        "4": {"full_name": "Taxi", "position": "TE", "team": "SF", "status": "Active"},
        "5": {"full_name": "Inactive", "position": "TE", "team": "SF", "status": "PUP"},
    }

    _, players, capacity, scoring_format = fetch_league_rosters(
        session,
        {
            "league_id": "123",
            "name": "Dynasty",
            "roster_positions": [
                "QB",
                "RB",
                "WR",
                "TE",
                "BN",
                "BN",
                "BN",
                "BN",
                "BN",
            ],
            "settings": {"reserve_slots": 1},
            "scoring_settings": {"rec": 0.5},
        },
        "me",
        index,
    )
    slots = {player.name: player.lineup_slot for player in players}
    assert slots == {
        "Starter": STARTER_SLOT,
        "Bench": BENCH_SLOT,
        "Reserve": RESERVE_SLOT,
        "Taxi": TAXI_SLOT,
        "Inactive": NFL_INACTIVE_SLOT,
    }
    # The PUP player still occupies a normal bench slot in Sleeper's raw
    # roster even though lineup safety labels him NFL_INACTIVE.
    assert capacity.bench_used == 2
    assert capacity.bench_limit == 5
    assert capacity.ir_used == 1
    assert capacity.ir_limit == 1
    assert scoring_format == "HALF"


def test_cached_player_index_carries_cache_freshness(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "sleeper-players.json"
    cache.write_text(json.dumps({"1": {"full_name": "George Kittle"}}))
    refreshed_at = datetime(2026, 8, 23, 15, 30, tzinfo=timezone.utc)
    os.utime(cache, (refreshed_at.timestamp(), refreshed_at.timestamp()))
    monkeypatch.setattr(
        "notifier.sources.sleeper.time.time",
        lambda: refreshed_at.timestamp() + 60,
    )
    session = Mock()

    index = load_player_index(tmp_path, session)

    assert isinstance(index, PlayerIndex)
    assert index["1"]["full_name"] == "George Kittle"
    assert index.refreshed_at == refreshed_at
    session.get.assert_not_called()


def test_refreshed_index_keeps_sleeper_injury_status(tmp_path) -> None:
    session = Mock()
    session.get.return_value = Response(
        {
            "1": {
                "full_name": "George Kittle",
                "position": "TE",
                "team": "SF",
                "depth_chart_order": 1,
                "depth_chart_position": "TE",
                "search_rank": 91,
                "injury_status": "Questionable",
                "status": "Active",
                "ignored": "large unused field",
            }
        }
    )

    index = load_player_index(tmp_path, session)

    assert index["1"]["injury_status"] == "Questionable"
    assert "ignored" not in index["1"]
    assert index.refreshed_at is not None
