from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from notifier.models import LeagueRef, RosterCapacity, RosterPlayer, RosterSnapshot
from notifier.roster import SNAPSHOT_VERSION, _write_snapshot, load_snapshot


def _config(tmp_path):
    return SimpleNamespace(state_dir=tmp_path)


def test_capacity_round_trips_through_roster_snapshot(tmp_path) -> None:
    league = LeagueRef("espn", "123", "Home", "Mine")
    capacity = RosterCapacity(bench_used=5, bench_limit=5, ir_used=0, ir_limit=1)
    snapshot = RosterSnapshot(
        generated_at=datetime(2026, 8, 23, 18, tzinfo=timezone.utc),
        leagues=[league],
        players=[RosterPlayer("My QB", "QB", "ARI", "QB", True, "Mine", league.key)],
        capacities={league.key: capacity},
        scoring_formats={league.key: "PPR"},
    )

    _write_snapshot(tmp_path / "roster-snapshot.json", snapshot)
    restored = load_snapshot(_config(tmp_path))

    assert restored.capacities == {league.key: capacity}
    assert restored.scoring_formats == {league.key: "PPR"}


def test_raw_fantasy_starter_round_trips_separately_from_eligibility_slot(
    tmp_path,
) -> None:
    league = LeagueRef("sleeper", "123", "Home", "Mine")
    snapshot = RosterSnapshot(
        generated_at=datetime(2026, 8, 23, 18, tzinfo=timezone.utc),
        leagues=[league],
        players=[
            RosterPlayer(
                "George Kittle",
                "TE",
                "SF",
                "NFL_INACTIVE",
                True,
                "Mine",
                league.key,
                fantasy_starter=True,
            )
        ],
    )

    _write_snapshot(tmp_path / "roster-snapshot.json", snapshot)
    restored = load_snapshot(_config(tmp_path)).players[0]

    assert restored.lineup_slot == "NFL_INACTIVE"
    assert restored.fantasy_starter is True
    assert restored.is_fantasy_starter is True
    assert restored.can_be_started_from_bench is False


def test_legacy_version_two_snapshot_without_capacity_still_loads(tmp_path) -> None:
    payload = {
        "version": SNAPSHOT_VERSION,
        "generatedAt": "2026-08-23T18:00:00+00:00",
        "leagues": [
            {
                "provider": "espn",
                "leagueId": "123",
                "name": "Home",
                "myTeamName": "Mine",
            }
        ],
        "players": [
            {
                "name": "Legacy Starter",
                "position": "WR",
                "proTeam": "ARI",
                "lineupSlot": "WR",
                "onMyTeam": True,
                "fantasyTeam": "Mine",
                "leagueKey": "espn:123",
            }
        ],
    }
    (tmp_path / "roster-snapshot.json").write_text(json.dumps(payload))

    restored = load_snapshot(_config(tmp_path))

    assert restored.capacities == {}
    assert restored.scoring_formats == {}
    assert restored.players[0].fantasy_starter is None
    assert restored.players[0].is_fantasy_starter is True


def test_malformed_optional_capacity_section_is_ignored(tmp_path) -> None:
    payload = {
        "version": SNAPSHOT_VERSION,
        "generatedAt": None,
        "leagues": [],
        "players": [],
        "capacities": ["not", "a", "mapping"],
    }
    (tmp_path / "roster-snapshot.json").write_text(json.dumps(payload))

    assert load_snapshot(_config(tmp_path)).capacities == {}
