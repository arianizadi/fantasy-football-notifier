import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from notifier.drafts import completed_unsynced_league_keys
from notifier.models import (
    Classification,
    LeagueRef,
    NewsItem,
    RosterPlayer,
    RosterSnapshot,
)
from notifier.notify import format_alert
from notifier.pipeline import Notifier
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


def test_mixed_draft_state_uses_in_season_espn_flow_only(monkeypatch) -> None:
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
    notifier = Notifier.__new__(Notifier)
    notifier._state_lock = threading.RLock()
    notifier.snapshot = snapshot
    notifier.depth = DepthCharts(player_index, snapshot)
    notifier.preseason = not snapshot.drafted_leagues()
    notifier.session = object()
    notifier.config = SimpleNamespace(
        dry_run=False,
        min_severity=2,
        min_severity_other=3,
    )
    notifier._refresh_ownership_just_in_time = Mock(return_value=True)
    monkeypatch.setattr(
        "notifier.pipeline.classify",
        Mock(
            return_value=Classification(
                event_type="injury",
                severity=4,
                fantasy_impact="The backup gains opportunity.",
                is_actionable=True,
                raw={"direction": "negative"},
            )
        ),
    )
    item = NewsItem(
        source="rotowire",
        guid="rotowire:mixed-draft-state",
        player_name="Example Starter",
        headline="Ruled out",
        body="Example Starter was ruled out.",
        url="https://example.test/report",
        published_at=None,
    )

    alert = notifier._evaluate(item)

    assert notifier.preseason is False
    assert alert is not None
    assert alert.tier == "mine"
    assert [plays.league for plays in alert.per_league] == [espn]
    assert alert.all_leagues == [espn]
    assert all(plays.league != sleeper for plays in alert.per_league)
    assert alert.context is not None
    assert all(
        sleeper.key not in entry.ownership
        for entry in [*alert.context.same_position, *alert.context.adjacent]
    )
    rendered = format_alert(alert)
    assert "Sleeper League" not in rendered
    assert sleeper.key not in rendered


def test_later_completed_league_remains_pending_until_its_roster_is_synced() -> None:
    snapshot, espn, sleeper = _snapshot()
    states = {
        espn.key: ("complete", None),
        sleeper.key: ("complete", None),
    }

    assert completed_unsynced_league_keys(snapshot, states) == [sleeper.key]


def test_fantasypros_formats_follow_each_provider_leagues_draft_state() -> None:
    snapshot, espn, sleeper = _snapshot()
    snapshot.scoring_formats = {
        espn.key: "PPR",
        sleeper.key: "HALF",
    }
    notifier = Notifier.__new__(Notifier)
    notifier._state_lock = threading.RLock()
    notifier.snapshot = snapshot

    # ESPN has drafted, while Sleeper's empty roster still means pre-draft.
    assert notifier._fantasypros_scoring_formats() == ("PPR",)

    notifier.snapshot = RosterSnapshot(
        generated_at=snapshot.generated_at,
        leagues=snapshot.leagues,
        players=[
            *snapshot.players,
            RosterPlayer(
                name="Sleeper Starter",
                position="WR",
                pro_team="SF",
                lineup_slot="WR",
                on_my_team=True,
                fantasy_team="Mine",
                league_key=sleeper.key,
            ),
        ],
        scoring_formats=snapshot.scoring_formats,
    )

    # Once Sleeper also has a roster, both provider formats become relevant.
    assert notifier._fantasypros_scoring_formats() == ("PPR", "HALF")


def test_daily_recap_reads_one_mixed_draft_state_under_the_state_lock(
    monkeypatch,
) -> None:
    snapshot, espn, sleeper = _snapshot()
    player_index = {
        "1": {"full_name": "Example Starter", "position": "RB", "team": "LV"}
    }
    now = datetime(2026, 8, 24, 15, tzinfo=timezone.utc)

    class TrackingLock:
        held = False

        def __enter__(self):
            self.held = True
            return self

        def __exit__(self, *_args):
            self.held = False

    lock = TrackingLock()

    class Probe:
        _state_lock = lock
        _refresh_ownership_just_in_time = Mock(return_value=True)
        events = SimpleNamespace(recent=Mock(return_value=[]))
        config = SimpleNamespace(
            daily_digest_timezone="America/Los_Angeles",
            dry_run=False,
        )

        @property
        def snapshot(self):
            assert lock.held
            return snapshot

        @property
        def _player_index(self):
            assert lock.held
            return player_index

    formatter = Mock(return_value=SimpleNamespace(parts=("recap",)))
    monkeypatch.setattr("notifier.pipeline.format_daily_recap", formatter)

    assert Notifier.daily_recap_parts(Probe(), now) == ("recap",)

    Probe._refresh_ownership_just_in_time.assert_called_once_with()
    Probe.events.recent.assert_called_once_with(
        since=now - timedelta(hours=24),
        until=now + timedelta(seconds=1),
        limit=2000,
    )
    kwargs = formatter.call_args.kwargs
    assert kwargs["roster_snapshot"] is snapshot
    assert kwargs["player_index"] is player_index
    assert snapshot.drafted_leagues() == [espn]
    assert sleeper not in snapshot.drafted_leagues()
