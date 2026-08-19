from datetime import datetime, timezone

from notifier.drafts import completed_unsynced_league_keys
from notifier.models import LeagueRef, RosterPlayer, RosterSnapshot
from notifier.plays import DepthCharts


def _snapshot() -> tuple[RosterSnapshot, LeagueRef, LeagueRef]:
    espn = LeagueRef("espn", "1", "ESPN League", "Mine")
    sleeper = LeagueRef("sleeper", "2", "Sleeper League", "Mine")
    snapshot = RosterSnapshot(
        generated_at=datetime.now(timezone.utc),
        leagues=[espn, sleeper],
        players=[
            RosterPlayer(
                name="Example Starter",
                position="RB",
                pro_team="LV",
                lineup_slot="RB",
                on_my_team=True,
                fantasy_team="Mine",
                league_key=espn.key,
            )
        ],
    )
    return snapshot, espn, sleeper


def test_undrafted_league_is_excluded_from_waiver_calculations() -> None:
    snapshot, espn, sleeper = _snapshot()
    player_index = {
        "1": {
            "full_name": "Example Starter",
            "position": "RB",
            "team": "LV",
            "depth_chart_order": 1,
            "search_rank": 20,
            "status": "Active",
        },
        "2": {
            "full_name": "Example Backup",
            "position": "RB",
            "team": "LV",
            "depth_chart_order": 2,
            "search_rank": 300,
            "status": "Active",
        },
    }

    charts = DepthCharts(player_index, snapshot)
    record, plays = charts.build(subject_names=("Example Starter",), snapshot=snapshot)
    assert record is not None
    context = charts.team_context(record, snapshot)
    assert context is not None

    assert snapshot.drafted_leagues() == [espn]
    assert [play.league for play in plays] == [espn]
    assert plays[0].claimable[0].name == "Example Backup"
    assert sleeper.key not in context.same_position[0].ownership
    assert set(context.same_position[0].ownership) == {espn.key}


def test_later_completed_league_remains_pending_until_its_roster_is_synced() -> None:
    snapshot, espn, sleeper = _snapshot()
    states = {
        espn.key: ("complete", None),
        sleeper.key: ("complete", None),
    }

    assert completed_unsynced_league_keys(snapshot, states) == [sleeper.key]
