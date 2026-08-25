import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from notifier import roster as roster_module
from notifier.logging_utils import NotifierError
from notifier.models import (
    Classification,
    LeagueRef,
    NewsItem,
    RosterCapacity,
    RosterPlayer,
    RosterSnapshot,
)
from notifier.pipeline import Notifier
from notifier.plays import DepthCharts
from notifier.roster import load_snapshot, refresh_drafted_snapshot


def _player(
    name: str,
    league: LeagueRef,
    *,
    mine: bool,
    fantasy_team: str,
    position: str = "RB",
) -> RosterPlayer:
    return RosterPlayer(
        name=name,
        position=position,
        pro_team="LV",
        lineup_slot=position,
        on_my_team=mine,
        fantasy_team=fantasy_team,
        league_key=league.key,
    )


def _mixed_snapshot() -> tuple[RosterSnapshot, LeagueRef, LeagueRef]:
    espn = LeagueRef("espn", "1", "ESPN League", "Mine")
    sleeper = LeagueRef("sleeper", "2", "Sleeper Draft", "Mine")
    snapshot = RosterSnapshot(
        generated_at=datetime.now(timezone.utc),
        leagues=[espn, sleeper],
        players=[
            _player(
                "My Quarterback",
                espn,
                mine=True,
                fantasy_team="Mine",
                position="QB",
            ),
            _player("Example Starter", espn, mine=False, fantasy_team="Rival"),
            # A rival keeper must remain intact without activating this empty
            # user roster or forcing a live Sleeper request.
            _player("Sleeper Keeper", sleeper, mine=False, fantasy_team="Other"),
        ],
        capacities={
            espn.key: RosterCapacity(bench_used=5, bench_limit=5, ir_used=0, ir_limit=1),
            sleeper.key: RosterCapacity(
                bench_used=0,
                bench_limit=5,
                ir_used=0,
                ir_limit=1,
            ),
        },
        scoring_formats={espn.key: "PPR", sleeper.key: "HALF"},
    )
    return snapshot, espn, sleeper


def _config(tmp_path):
    return SimpleNamespace(
        state_dir=tmp_path,
        espn_enabled=True,
        espn_league_id=1,
        espn_year=2026,
        sleeper_username="arian",
        sleeper_league_ids=("2",),
        dry_run=False,
        min_severity=2,
        min_severity_other=3,
    )


def test_drafted_refresh_skips_predraft_provider_and_preserves_it(
    tmp_path, monkeypatch
) -> None:
    previous, espn, sleeper = _mixed_snapshot()
    fresh_capacity = RosterCapacity(
        bench_used=4,
        bench_limit=5,
        ir_used=1,
        ir_limit=1,
    )
    fresh_players = [
        _player(
            "My Quarterback",
            espn,
            mine=True,
            fantasy_team="Mine",
            position="QB",
        ),
        _player("Example Starter", espn, mine=False, fantasy_team="Rival"),
    ]
    sleeper_loader = Mock(side_effect=AssertionError("pre-draft Sleeper was queried"))
    monkeypatch.setattr(
        "notifier.roster._load_espn",
        lambda config, session: (espn, fresh_players, fresh_capacity, "PPR"),
    )
    monkeypatch.setattr("notifier.roster._load_sleeper", sleeper_loader)

    refreshed, _ = refresh_drafted_snapshot(_config(tmp_path), previous)

    assert sleeper_loader.call_count == 0
    assert refreshed.drafted_leagues() == [espn]
    assert refreshed.league(sleeper.key) == sleeper
    assert refreshed.capacities[sleeper.key] == previous.capacities[sleeper.key]
    assert refreshed.scoring_formats[sleeper.key] == "HALF"
    assert [
        player.name
        for player in refreshed.players
        if player.league_key == sleeper.key
    ] == ["Sleeper Keeper"]
    assert refreshed.capacities[espn.key] == fresh_capacity

    persisted = load_snapshot(_config(tmp_path))
    assert persisted.league(sleeper.key) == sleeper
    assert persisted.capacities[sleeper.key] == previous.capacities[sleeper.key]
    assert persisted.scoring_formats[sleeper.key] == "HALF"


def test_drafted_refresh_rejects_empty_active_roster_without_overwriting(
    tmp_path, monkeypatch
) -> None:
    previous, espn, _ = _mixed_snapshot()
    config = _config(tmp_path)
    monkeypatch.setattr(
        "notifier.roster._load_espn",
        lambda config, session: (
            espn,
            [_player("Rival Only", espn, mine=False, fantasy_team="Rival")],
            previous.capacities[espn.key],
            "PPR",
        ),
    )
    monkeypatch.setattr(
        "notifier.roster._load_sleeper",
        Mock(side_effect=AssertionError("pre-draft Sleeper was queried")),
    )

    with pytest.raises(NotifierError, match="returned an empty roster"):
        refresh_drafted_snapshot(config, previous)

    assert not (tmp_path / "roster-snapshot.json").exists()


def test_predraft_sleeper_failure_cannot_strip_espn_pickup(
    tmp_path, monkeypatch
) -> None:
    snapshot, espn, sleeper = _mixed_snapshot()
    config = _config(tmp_path)
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
    fresh_players = [
        _player(
            "My Quarterback",
            espn,
            mine=True,
            fantasy_team="Mine",
            position="QB",
        ),
        _player("Example Starter", espn, mine=False, fantasy_team="Rival"),
    ]
    sleeper_loader = Mock(side_effect=AssertionError("pre-draft Sleeper was queried"))
    monkeypatch.setattr(
        "notifier.roster._load_espn",
        lambda config, session: (
            espn,
            fresh_players,
            snapshot.capacities[espn.key],
            "PPR",
        ),
    )
    monkeypatch.setattr("notifier.roster._load_sleeper", sleeper_loader)
    monkeypatch.setattr(
        "notifier.pipeline.classify",
        lambda *args, **kwargs: Classification(
            event_type="injury",
            severity=4,
            fantasy_impact="",
            is_actionable=True,
            raw={"direction": "negative"},
        ),
    )

    notifier = Notifier.__new__(Notifier)
    notifier.config = config
    notifier.session = Mock()
    notifier._state_lock = threading.RLock()
    notifier._jit_roster_lock = threading.Lock()
    notifier._last_jit_roster_refresh = 0.0
    notifier._last_jit_roster_success = 0.0
    notifier._snapshot_mtime = 0.0
    notifier._player_index = player_index
    notifier.snapshot = snapshot
    notifier.depth = DepthCharts(player_index, snapshot)
    notifier.preseason = False
    notifier._recent_news_context = Mock(return_value="")
    notifier._journal_classification = Mock()
    notifier._enrich_fantasypros = Mock(side_effect=lambda plays: plays)

    alert = notifier._evaluate(
        NewsItem(
            source="twitter",
            guid="mixed-draft",
            player_name="Example Starter",
            headline="Example Starter left practice with an injury",
            body="Example Starter left practice with an injury.",
            url="https://example.com/mixed-draft",
            published_at=datetime.now(timezone.utc),
        )
    )

    assert sleeper_loader.call_count == 0
    assert alert is not None
    assert alert.tier == "claimable"
    assert alert.availability_refresh_failed is False
    assert alert.all_leagues == [espn]
    assert [plays.league for plays in alert.per_league] == [espn]
    assert [candidate.name for candidate in alert.per_league[0].claimable] == [
        "Example Backup"
    ]
    assert sleeper not in alert.all_leagues


def test_drafted_refresh_targets_only_active_sleeper_league(
    tmp_path, monkeypatch
) -> None:
    active = LeagueRef("sleeper", "10", "Drafted Sleeper", "Mine")
    waiting = LeagueRef("sleeper", "20", "Pre-draft Sleeper", "Mine")
    waiting_capacity = RosterCapacity(
        bench_used=0,
        bench_limit=5,
        ir_used=0,
        ir_limit=1,
    )
    previous = RosterSnapshot(
        generated_at=datetime.now(timezone.utc),
        leagues=[active, waiting],
        players=[
            _player("Active Roster Player", active, mine=True, fantasy_team="Mine"),
            _player("Waiting Rival Keeper", waiting, mine=False, fantasy_team="Other"),
        ],
        capacities={
            active.key: RosterCapacity(bench_used=5, bench_limit=5),
            waiting.key: waiting_capacity,
        },
        scoring_formats={active.key: "PPR", waiting.key: "HALF"},
    )
    fresh_players = [
        _player("Active Roster Player", active, mine=True, fantasy_team="Mine")
    ]
    sleeper_loader = Mock(
        return_value=[
            (
                active,
                fresh_players,
                RosterCapacity(bench_used=4, bench_limit=5),
                "PPR",
            )
        ]
    )
    monkeypatch.setattr("notifier.roster._load_sleeper", sleeper_loader)
    monkeypatch.setattr(
        "notifier.roster._load_espn",
        Mock(side_effect=AssertionError("ESPN was queried")),
    )
    config = SimpleNamespace(
        state_dir=tmp_path,
        espn_enabled=False,
        espn_year=2026,
        sleeper_username="arian",
        sleeper_league_ids=("10", "20"),
    )

    refreshed, _ = refresh_drafted_snapshot(config, previous)

    assert sleeper_loader.call_count == 1
    assert sleeper_loader.call_args.kwargs["league_ids"] == {"10"}
    assert refreshed.drafted_leagues() == [active]
    assert refreshed.league(waiting.key) == waiting
    assert refreshed.capacities[waiting.key] == waiting_capacity
    assert refreshed.scoring_formats[waiting.key] == "HALF"
    assert [
        player.name
        for player in refreshed.players
        if player.league_key == waiting.key
    ] == ["Waiting Rival Keeper"]


def test_provider_scoped_refresh_preserves_other_drafted_provider(
    tmp_path, monkeypatch
) -> None:
    espn = LeagueRef("espn", "1", "ESPN League", "Mine")
    sleeper = LeagueRef("sleeper", "2", "Sleeper League", "Mine")
    sleeper_capacity = RosterCapacity(
        bench_used=5,
        bench_limit=5,
        ir_used=0,
        ir_limit=1,
    )
    sleeper_player = _player(
        "Sleeper Roster Player",
        sleeper,
        mine=True,
        fantasy_team="Mine",
    )
    previous = RosterSnapshot(
        generated_at=datetime.now(timezone.utc),
        leagues=[espn, sleeper],
        players=[
            _player(
                "ESPN Quarterback",
                espn,
                mine=True,
                fantasy_team="Mine",
                position="QB",
            ),
            sleeper_player,
        ],
        capacities={
            espn.key: RosterCapacity(bench_used=5, bench_limit=5),
            sleeper.key: sleeper_capacity,
        },
        scoring_formats={espn.key: "PPR", sleeper.key: "HALF"},
    )
    fresh_espn_players = [
        _player(
            "ESPN Quarterback",
            espn,
            mine=True,
            fantasy_team="Mine",
            position="QB",
        )
    ]
    monkeypatch.setattr(
        "notifier.roster._load_espn",
        lambda config, session: (
            espn,
            fresh_espn_players,
            RosterCapacity(bench_used=4, bench_limit=5),
            "PPR",
        ),
    )
    sleeper_loader = Mock(
        side_effect=AssertionError("unrequested Sleeper provider was queried")
    )
    monkeypatch.setattr("notifier.roster._load_sleeper", sleeper_loader)

    refreshed, _ = refresh_drafted_snapshot(
        _config(tmp_path),
        previous,
        league_keys={espn.key},
    )

    assert sleeper_loader.call_count == 0
    assert refreshed.league(sleeper.key) == sleeper
    assert refreshed.mine(sleeper.key) == [sleeper_player]
    assert refreshed.capacities[sleeper.key] == sleeper_capacity
    assert refreshed.scoring_formats[sleeper.key] == "HALF"


def test_jit_merge_preserves_league_that_drafted_during_network_refresh(
    tmp_path, monkeypatch
) -> None:
    previous, espn, sleeper = _mixed_snapshot()
    config = _config(tmp_path)
    drafted_sleeper = LeagueRef(
        "sleeper",
        sleeper.league_id,
        "Sleeper Draft Complete",
        "Mine",
    )
    completed_capacity = RosterCapacity(
        bench_used=5,
        bench_limit=5,
        ir_used=1,
        ir_limit=1,
    )
    completed_snapshot = RosterSnapshot(
        generated_at=datetime.now(timezone.utc),
        leagues=[espn, drafted_sleeper],
        players=[
            _player(
                "My Quarterback",
                espn,
                mine=True,
                fantasy_team="Mine",
                position="QB",
            ),
            _player("Example Starter", espn, mine=False, fantasy_team="Rival"),
            _player(
                "New Sleeper Draft Pick",
                drafted_sleeper,
                mine=True,
                fantasy_team="Mine",
            ),
        ],
        capacities={
            espn.key: previous.capacities[espn.key],
            drafted_sleeper.key: completed_capacity,
        },
        scoring_formats={espn.key: "PPR", drafted_sleeper.key: "HALF"},
    )
    fresh_espn_players = [
        _player(
            "My Quarterback",
            espn,
            mine=True,
            fantasy_team="Mine",
            position="QB",
        ),
        _player("Example Starter", espn, mine=False, fantasy_team="Rival"),
    ]

    def finish_full_refresh_while_jit_is_in_flight(config, session):
        roster_module._write_snapshot(
            roster_module.roster_path(config),
            completed_snapshot,
        )
        return (
            espn,
            fresh_espn_players,
            previous.capacities[espn.key],
            "PPR",
        )

    sleeper_loader = Mock(side_effect=AssertionError("pre-draft Sleeper was queried"))
    monkeypatch.setattr("notifier.roster._load_espn", finish_full_refresh_while_jit_is_in_flight)
    monkeypatch.setattr("notifier.roster._load_sleeper", sleeper_loader)

    refreshed, _ = refresh_drafted_snapshot(config, previous)

    assert sleeper_loader.call_count == 0
    assert refreshed.drafted_leagues() == [espn, drafted_sleeper]
    assert refreshed.league(drafted_sleeper.key) == drafted_sleeper
    assert refreshed.capacities[drafted_sleeper.key] == completed_capacity
    assert refreshed.scoring_formats[drafted_sleeper.key] == "HALF"
    assert [
        player.name
        for player in refreshed.mine(drafted_sleeper.key)
    ] == ["New Sleeper Draft Pick"]

    persisted = load_snapshot(config)
    assert persisted.drafted_leagues() == [espn, drafted_sleeper]
    assert persisted.capacities[drafted_sleeper.key] == completed_capacity


def test_slow_jit_refresh_times_out_then_finishes_and_reuses_success(
    monkeypatch,
) -> None:
    league = LeagueRef("espn", "1", "ESPN League", "Mine")
    player = _player("My Running Back", league, mine=True, fantasy_team="Mine")
    original = RosterSnapshot(
        generated_at=datetime.now(timezone.utc),
        leagues=[league],
        players=[player],
    )
    fresh = RosterSnapshot(
        generated_at=datetime.now(timezone.utc),
        leagues=[league],
        players=[player],
    )
    started = threading.Event()
    release = threading.Event()

    def slow_refresh(config, previous, *, league_keys):
        started.set()
        assert release.wait(1.0)
        return fresh, 123.0

    refresh = Mock(side_effect=slow_refresh)
    wait_seconds = 0.05
    monkeypatch.setattr("notifier.pipeline.refresh_drafted_snapshot", refresh)
    monkeypatch.setattr(
        "notifier.pipeline.JIT_ROSTER_WAIT_SECONDS",
        wait_seconds,
    )

    notifier = Notifier.__new__(Notifier)
    notifier.config = SimpleNamespace()
    notifier._state_lock = threading.RLock()
    notifier._jit_roster_lock = threading.Lock()
    notifier._last_jit_roster_refresh = 0.0
    notifier._last_jit_roster_success = 0.0
    notifier._last_jit_roster_attempt_keys = frozenset()
    notifier._last_jit_roster_success_keys = frozenset()
    notifier._snapshot_mtime = 0.0
    notifier._player_index = {}
    notifier.snapshot = original

    before = time.monotonic()
    try:
        assert notifier._refresh_ownership_just_in_time() is False
        elapsed = time.monotonic() - before

        assert elapsed < wait_seconds + 0.2
        assert started.wait(1.0)
        assert not notifier._jit_roster_future.done()

        release.set()
        assert notifier._jit_roster_future.result(timeout=1.0) is True
        assert notifier._refresh_ownership_just_in_time() is True

        assert refresh.call_count == 1
        refresh.assert_called_once_with(
            notifier.config,
            original,
            league_keys={league.key},
        )
        assert notifier.snapshot is fresh
        assert notifier._snapshot_mtime == 123.0
    finally:
        release.set()
        executor = getattr(notifier, "_jit_roster_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
