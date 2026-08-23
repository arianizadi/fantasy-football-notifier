from datetime import datetime, timezone

import pytest

from notifier.models import (
    Alert,
    Classification,
    LeagueRef,
    NewsItem,
    RosterPlayer,
    RosterSnapshot,
)
from notifier.notify import format_alert
from notifier.plays import (
    Beneficiary,
    DepthCharts,
    LeaguePlays,
    event_allows_backup_moves,
    event_allows_lineup_substitution,
    plays_for_event,
)
from notifier.sources.sleeper import PlayerIndex


INDEX_REFRESHED_AT = datetime(2026, 8, 23, 17, 45, tzinfo=timezone.utc)


def _news(headline: str = "49ers activated George Kittle from active/PUP") -> NewsItem:
    return NewsItem(
        source="twitter",
        guid="twitter:kittle:return",
        player_name="George Kittle",
        headline=headline,
        body=headline,
        url="https://x.com/AdamSchefter/status/1",
        published_at=None,
    )


def _kittle_index() -> PlayerIndex:
    records = {
        "1": {
            "full_name": "George Kittle",
            "position": "TE",
            "team": "SF",
            "depth_chart_order": 1,
            "search_rank": 91,
            "injury_status": "Questionable",
            "status": "Active",
        },
        "2": {
            "full_name": "Jake Tonges",
            "position": "TE",
            "team": "SF",
            "depth_chart_order": 2,
            "search_rank": 178,
            "injury_status": "",
            "status": "Active",
        },
        "3": {
            "full_name": "Luke Farrell",
            "position": "TE",
            "team": "SF",
            "depth_chart_order": 3,
            "search_rank": 491,
            "injury_status": "",
            "status": "Active",
        },
        "4": {
            "full_name": "Josiah Deguara",
            "position": "TE",
            "team": "SF",
            "depth_chart_order": 4,
            "search_rank": 999,
            "injury_status": "",
            "status": "Active",
        },
        "5": {
            "full_name": "Brayden Willis",
            "position": "TE",
            "team": "SF",
            "depth_chart_order": 5,
            "search_rank": 315,
            "injury_status": "",
            "status": "Active",
        },
        "6": {
            "full_name": "Mike Evans",
            "position": "WR",
            "team": "SF",
            "depth_chart_order": 1,
            "search_rank": 28,
            "injury_status": "",
            "status": "Active",
        },
        "7": {
            "full_name": "Christian McCaffrey",
            "position": "RB",
            "team": "SF",
            "depth_chart_order": 1,
            "search_rank": 9,
            "injury_status": "",
            "status": "Active",
        },
        "8": {
            "full_name": "Brock Purdy",
            "position": "QB",
            "team": "SF",
            "depth_chart_order": 1,
            "search_rank": 55,
            "injury_status": "",
            "status": "Active",
        },
    }
    return PlayerIndex(records, refreshed_at=INDEX_REFRESHED_AT)


def test_kittle_preseason_return_golden_message() -> None:
    snapshot = RosterSnapshot(generated_at=None)
    charts = DepthCharts(_kittle_index(), snapshot)
    record, plays = charts.build(subject_names=("George Kittle",), snapshot=snapshot)
    assert record is not None
    assert plays == []
    context = charts.team_context(record, snapshot)
    assert context is not None

    alert = Alert(
        item=_news(),
        classification=Classification(
            event_type="return",
            severity=4,
            fantasy_impact="Activate Kittle in all leagues; he should return to starting TE1 status.",
            is_actionable=True,
            raw={},
        ),
        tier="preseason",
        context=context,
    )

    text = format_alert(alert)
    assert "[4/5] PRESEASON — RETURN" in text
    assert "Draft note: Return news improves availability" in text
    assert "Backup watch: Jake Tonges is next" in text
    assert "No pickup is recommended from return news" in text
    assert "Activate Kittle" not in text
    assert "ADD " not in text
    assert "[INJURED]" not in text
    assert "SUBJECT · RETURN" in text
    assert "Sleeper injury: Questionable" in text
    assert "Sleeper rank #91" in text
    assert "refreshed 2026-08-23 10:45 PT" in text
    assert "SF TE DEPTH / BACKUP WATCH · SLEEPER" in text
    assert "SF OTHER SLEEPER DEPTH LEADERS" in text
    assert "other starters" not in text.lower()
    assert "George Kittle" in text
    assert "Jake Tonges" in text
    assert "Luke Farrell" in text
    assert "Brayden Willis" not in text
    assert "Josiah Deguara" not in text


def test_kittle_return_shows_available_backups_without_recommending_them() -> None:
    league = LeagueRef("sleeper", "1234", "Home League", "Mine")
    snapshot = RosterSnapshot(
        generated_at=INDEX_REFRESHED_AT,
        leagues=[league],
        players=[
            RosterPlayer(
                "George Kittle",
                "TE",
                "SF",
                "TE",
                True,
                "Mine",
                league.key,
            )
        ],
    )
    charts = DepthCharts(_kittle_index(), snapshot)
    record, raw_plays = charts.build(
        subject_names=("George Kittle",), snapshot=snapshot
    )
    assert record is not None
    context = charts.team_context(record, snapshot)
    filtered = plays_for_event(raw_plays, "return", 4)

    text = format_alert(
        Alert(
            item=_news(),
            classification=Classification("return", 4, "", False, {}),
            tier="mine",
            per_league=filtered,
            context=context,
            all_leagues=[league],
        )
    )

    assert "TE2 Jake Tonges · FA" in text
    assert "TE3 Luke Farrell · FA" in text
    assert "Backup watch: Jake Tonges is next" in text
    assert "No pickup is recommended from return news" in text
    assert "ADD " not in text


@pytest.mark.parametrize("event_type", ["return", "signing", "trade", "depth_chart", "usage"])
def test_non_removal_events_never_create_mechanical_moves(event_type: str) -> None:
    league = LeagueRef("sleeper", "1234", "Home League", "Mine")
    raw = LeaguePlays(
        league=league,
        subject_state="mine",
        subject_owner="Mine",
        beneficiaries=[Beneficiary("Jake Tonges", "TE", 2, "free_agent")],
        bench_options=["Sam LaPorta"],
    )

    filtered = plays_for_event([raw], event_type, 5)[0]
    assert filtered.claimable == []
    assert filtered.bench_options == []
    assert not filtered.has_action
    assert not event_allows_backup_moves(event_type, 5)
    assert not event_allows_lineup_substitution(event_type, 5)

    alert = Alert(
        item=_news(),
        classification=Classification(event_type, 5, "", True, {}),
        tier="mine",
        per_league=[raw],
    )
    text = format_alert(alert)
    assert "ADD " not in text
    assert "START " not in text


def test_major_injury_retains_deterministic_add_and_start_moves() -> None:
    league = LeagueRef("espn", "9876", "Sunday Crew", "Mine")
    raw = LeaguePlays(
        league=league,
        subject_state="mine",
        subject_owner="Mine",
        beneficiaries=[Beneficiary("Jake Tonges", "TE", 2, "free_agent")],
        bench_options=["Sam LaPorta"],
    )
    filtered = plays_for_event([raw], "injury", 4)[0]
    assert filtered.claimable[0].name == "Jake Tonges"
    assert filtered.bench_options == ["Sam LaPorta"]

    alert = Alert(
        item=_news("George Kittle was ruled out"),
        classification=Classification(
            "injury", 4, "Add Tonges and start LaPorta.", True, {}
        ),
        tier="mine",
        per_league=[raw],
    )
    text = format_alert(alert)
    assert "Model summary:" not in text
    assert "LEAGUE-SPECIFIC MOVES" in text
    assert "Sunday Crew: ADD <b>Jake Tonges</b> (TE2)" in text
    assert "START <b>Sam LaPorta</b>" in text


def test_non_prescriptive_model_summary_can_still_be_shown() -> None:
    alert = Alert(
        item=_news(),
        classification=Classification(
            "return",
            4,
            "Kittle projects for his usual high-end TE role once fully cleared.",
            False,
            {},
        ),
        tier="preseason",
    )
    text = format_alert(alert)
    assert (
        "Model summary: Kittle projects for his usual high-end TE role once fully cleared."
        in text
    )


def test_moderate_injury_can_surface_backup_without_forcing_lineup_change() -> None:
    assert event_allows_backup_moves("injury", 3)
    assert not event_allows_lineup_substitution("injury", 3)


def test_only_active_bench_players_can_be_start_suggestions() -> None:
    league = LeagueRef("espn", "1", "Main League", "Mine")
    snapshot = RosterSnapshot(
        generated_at=None,
        leagues=[league],
        players=[
            RosterPlayer("George Kittle", "TE", "SF", "TE", True, "Mine", league.key),
            RosterPlayer("Active Bench", "TE", "DET", "BE", True, "Mine", league.key),
            RosterPlayer("ESPN IR", "TE", "MIN", "IR", True, "Mine", league.key),
            RosterPlayer("Sleeper Reserve", "TE", "NYG", "RESERVE", True, "Mine", league.key),
            RosterPlayer("Sleeper Taxi", "TE", "DAL", "TAXI", True, "Mine", league.key),
            RosterPlayer(
                "NFL Inactive", "TE", "NO", "NFL_INACTIVE", True, "Mine", league.key
            ),
        ],
    )
    charts = DepthCharts(_kittle_index(), snapshot)
    _, plays = charts.build(subject_names=("George Kittle",), snapshot=snapshot)
    assert plays[0].bench_options == ["Active Bench"]


def test_duplicate_league_names_are_disambiguated_by_provider_and_id() -> None:
    leagues = [
        LeagueRef("sleeper", "111111", "Weekend", "Mine"),
        LeagueRef("sleeper", "222222", "Weekend", "Mine"),
    ]
    per_league = [
        LeaguePlays(
            league=leagues[0],
            subject_state="mine",
            subject_owner="Mine",
            beneficiaries=[Beneficiary("Jake Tonges", "TE", 2, "free_agent")],
        )
    ]
    alert = Alert(
        item=_news("George Kittle was ruled out"),
        classification=Classification("injury", 4, "", True, {}),
        tier="mine",
        per_league=per_league,
        all_leagues=leagues,
    )
    text = format_alert(alert)
    assert "Weekend (SLEEPER 1111):" in text
    assert "Weekend:" not in text


def test_inactive_subject_remains_in_source_attributed_context() -> None:
    index = _kittle_index()
    index["1"]["status"] = "PUP"
    snapshot = RosterSnapshot(generated_at=None)
    charts = DepthCharts(index, snapshot)
    record = charts.lookup("George Kittle")
    assert record is not None
    context = charts.team_context(record, snapshot)
    assert context is not None
    subject = next(entry for entry in context.same_position if entry.is_subject)
    assert subject.sleeper_status == "PUP"

    text = format_alert(
        Alert(
            item=_news(),
            classification=Classification("return", 4, "", True, {}),
            tier="preseason",
            context=context,
        )
    )
    assert "Sleeper status: PUP" in text
    assert "SUBJECT · RETURN" in text
