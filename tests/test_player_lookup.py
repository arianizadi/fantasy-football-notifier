from __future__ import annotations

from datetime import datetime, timezone

from notifier.models import LeagueRef, RosterPlayer, RosterSnapshot
from notifier.player_lookup import format_player_lookup
from notifier.sources.sleeper import PlayerIndex


def _index() -> PlayerIndex:
    return PlayerIndex(
        {
            "1": {
                "full_name": "George Kittle",
                "team": "SF",
                "position": "TE",
                "depth_chart_order": 1,
                "search_rank": 91,
                "status": "Active",
                "injury_status": "Questionable",
            },
            "2": {
                "full_name": "Jake Tonges",
                "team": "SF",
                "position": "TE",
                "depth_chart_order": 2,
                "search_rank": 178,
                "status": "Active",
                "injury_status": "",
            },
        },
        refreshed_at=datetime(2026, 8, 23, 18, 0, tzinfo=timezone.utc),
    )


def test_player_command_resolves_last_name_and_shows_provenance() -> None:
    league = LeagueRef("sleeper", "123456", "Home League", "My Team")
    snapshot = RosterSnapshot(
        generated_at=datetime(2026, 8, 23, 18, 5, tzinfo=timezone.utc),
        leagues=[league],
        players=[
            RosterPlayer(
                "George Kittle",
                "TE",
                "SF",
                "BE",
                True,
                "My Team",
                league.key,
            )
        ],
    )

    text = format_player_lookup("Kittle", _index(), snapshot)

    assert "<b>George Kittle</b>" in text
    assert "Sleeper status: Active" in text
    assert "injury: Questionable" in text
    assert "overall/search rank: #91" in text
    assert "Home League: YOU · BE" in text
    assert "TE2 Jake Tonges" in text
    assert "Sleeper refreshed:" in text
    assert "League ownership refreshed: 2026-08-23 18:05 UTC" in text


def test_player_command_escapes_query_and_handles_no_match() -> None:
    text = format_player_lookup("<unknown>", _index(), RosterSnapshot(None))
    assert "&lt;unknown&gt;" in text
    assert "<unknown>" not in text
