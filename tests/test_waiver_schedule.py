from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import requests

from notifier.logging_utils import NotifierError
from notifier.models import LeagueRef, RosterPlayer, RosterSnapshot
from notifier.sources.sleeper import PlayerIndex
from notifier.telegram_control import ScheduledReport
from notifier import waiver_schedule
from notifier.waiver_schedule import (
    RETRY_INTERVAL_SECONDS,
    WaiverReportCoordinator,
    _dominant_espn_deadline,
    _next_sleeper_waiver,
)


NOW = datetime(2026, 8, 24, 17, tzinfo=timezone.utc)  # Monday 10 AM PT


def _entry(deadline: datetime) -> dict:
    return {"waiverProcessDate": int(deadline.timestamp() * 1000)}


def test_espn_uses_broad_pool_cohort_not_one_recent_drop() -> None:
    broad = datetime(2026, 8, 25, 7, tzinfo=timezone.utc)
    later_drop = broad + timedelta(days=1)
    entries = [_entry(broad) for _ in range(80)] + [_entry(later_drop)]

    assert _dominant_espn_deadline(entries, now=NOW) == broad


def test_espn_withholds_a_lone_drop_timer_as_weekly_deadline() -> None:
    entries = [_entry(NOW + timedelta(hours=24))]

    assert _dominant_espn_deadline(entries, now=NOW) is None


def test_sleeper_tuesday_after_means_wednesday_1205_am_pacific() -> None:
    deadline = _next_sleeper_waiver(
        NOW,
        {"waiver_day_of_week": 2, "daily_waivers": 0},
        timezone_name="America/Los_Angeles",
    )

    assert deadline == datetime(2026, 8, 26, 7, 5, tzinfo=timezone.utc)


def test_sleeper_custom_daily_schedule_fails_closed() -> None:
    assert (
        _next_sleeper_waiver(
            NOW,
            {"waiver_day_of_week": 2, "daily_waivers": 1},
            timezone_name="America/Los_Angeles",
        )
        is None
    )


class _FantasyPros:
    def signal(self, *args, **kwargs):
        del args, kwargs
        return None


class _Events:
    def recent(self, **kwargs):
        del kwargs
        return []


def _config(**overrides):
    values = {
        "waiver_report_enabled": True,
        "waiver_report_lead_hours": 8,
        "espn_enabled": True,
        "espn_league_id": 1,
        "espn_year": 2026,
        "sleeper_username": "user",
        "sleeper_league_ids": (),
        "daily_digest_timezone": "America/Los_Angeles",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _snapshot(*league_refs: LeagueRef) -> RosterSnapshot:
    players = [
        RosterPlayer(
            f"Player {league.league_id}",
            "RB",
            "SF",
            "BE",
            True,
            league.my_team_name,
            league.key,
        )
        for league in league_refs
    ]
    return RosterSnapshot(generated_at=NOW, leagues=list(league_refs), players=players)


def _coordinator(
    config,
    snapshot: RosterSnapshot,
    *,
    refresh_provider=lambda _league_keys: True,
    clock=lambda: 0.0,
    now_provider=lambda: NOW,
) -> WaiverReportCoordinator:
    return WaiverReportCoordinator(
        config,
        snapshot_provider=lambda: snapshot,
        player_index_provider=dict,
        refresh_provider=refresh_provider,
        event_store=_Events(),
        fantasypros=_FantasyPros(),
        completed_provider=lambda _key: False,
        session=requests.Session(),
        clock=clock,
        now_provider=now_provider,
    )


def _stub_report_formatting(monkeypatch, coordinator: WaiverReportCoordinator) -> None:
    monkeypatch.setattr(coordinator, "_candidate_evidence", lambda *args: ())
    monkeypatch.setattr(coordinator, "_context", lambda *args: object())
    monkeypatch.setattr(
        waiver_schedule,
        "build_waiver_report",
        lambda context, candidates: (context, candidates),
    )
    monkeypatch.setattr(
        waiver_schedule,
        "format_waiver_report_html",
        lambda report: ("waiver report",),
    )
    monkeypatch.setattr(
        waiver_schedule.sleeper,
        "trending_adds",
        lambda *args, **kwargs: {},
    )


def test_predraft_sleeper_is_not_called_or_allowed_to_block_active_espn(monkeypatch) -> None:
    espn = LeagueRef("espn", "1", "ESPN", "Mine")
    sleeper = LeagueRef("sleeper", "2", "Sleeper", "Mine")
    snapshot = RosterSnapshot(
        generated_at=NOW,
        leagues=[espn, sleeper],
        players=[
            RosterPlayer("Lamar Jackson", "QB", "BAL", "QB", True, "Mine", espn.key)
        ],
    )
    config = SimpleNamespace(
        waiver_report_enabled=True,
        waiver_report_lead_hours=8,
        espn_enabled=True,
        espn_league_id=1,
        sleeper_username="user",
        sleeper_league_ids=(),
        daily_digest_timezone="America/Los_Angeles",
    )
    coordinator = WaiverReportCoordinator(
        config,
        snapshot_provider=lambda: snapshot,
        player_index_provider=dict,
        refresh_provider=lambda _league_keys: True,
        event_store=_Events(),
        fantasypros=_FantasyPros(),
        completed_provider=lambda _key: False,
        session=requests.Session(),
    )
    monkeypatch.setattr(coordinator, "_fetch_espn_pool", lambda _now: None)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("pre-draft Sleeper must not be queried")

    monkeypatch.setattr(coordinator, "_fetch_sleeper_schedules", forbidden)

    assert coordinator._discover_due(NOW) == ()
    coordinator.close()


def test_cached_due_report_is_dropped_after_deadline_on_discovery_exception(
    monkeypatch,
) -> None:
    deadline = NOW + timedelta(hours=1)
    cached = ScheduledReport(
        key=f"waiver:espn:1:{int(deadline.timestamp())}",
        kind="waiver_report",
        parts=("cached",),
    )
    clock_value = [0.0]
    coordinator = _coordinator(
        _config(),
        _snapshot(LeagueRef("espn", "1", "ESPN", "Mine")),
        clock=lambda: clock_value[0],
    )
    calls = 0

    def discover(_now):
        nonlocal calls
        calls += 1
        if calls == 1:
            return (cached,)
        raise requests.RequestException("provider unavailable")

    monkeypatch.setattr(coordinator, "_discover_due", discover)

    assert coordinator.due_reports(NOW) == (cached,)
    clock_value[0] = 901.0
    assert coordinator.due_reports(deadline + timedelta(seconds=1)) == ()
    coordinator.close()


def test_sleeper_schedule_uses_pacific_not_recap_timezone(monkeypatch) -> None:
    league = LeagueRef("sleeper", "2", "Sleeper", "Mine")
    coordinator = _coordinator(
        _config(espn_enabled=False, daily_digest_timezone="Asia/Tokyo"),
        _snapshot(league),
    )
    monkeypatch.setattr(
        waiver_schedule.sleeper_league,
        "resolve_user_id",
        lambda _session, _username: "user-id",
    )
    monkeypatch.setattr(
        waiver_schedule.sleeper_league,
        "list_leagues",
        lambda _session, _user_id, _season: [
            {
                "league_id": "2",
                "status": "in_season",
                "settings": {"waiver_day_of_week": 2, "daily_waivers": 0},
            }
        ],
    )

    pools = coordinator._fetch_sleeper_schedules(_snapshot(league), NOW)

    assert pools[0]["deadline"] == datetime(2026, 8, 26, 7, 5, tzinfo=timezone.utc)
    coordinator.close()


@pytest.mark.parametrize("status", ["pre_draft", "drafting", "complete"])
def test_sleeper_non_active_statuses_are_skipped(monkeypatch, status: str) -> None:
    league = LeagueRef("sleeper", "2", "Sleeper", "Mine")
    snapshot = _snapshot(league)
    coordinator = _coordinator(_config(espn_enabled=False), snapshot)
    monkeypatch.setattr(
        waiver_schedule.sleeper_league,
        "resolve_user_id",
        lambda _session, _username: "user-id",
    )
    monkeypatch.setattr(
        waiver_schedule.sleeper_league,
        "list_leagues",
        lambda _session, _user_id, _season: [
            {
                "league_id": "2",
                "status": status,
                "settings": {"waiver_day_of_week": 2, "daily_waivers": 0},
            }
        ],
    )

    assert coordinator._fetch_sleeper_schedules(snapshot, NOW) == []
    coordinator.close()


def test_sleeper_in_season_status_is_accepted(monkeypatch) -> None:
    league = LeagueRef("sleeper", "2", "Sleeper", "Mine")
    snapshot = _snapshot(league)
    coordinator = _coordinator(_config(espn_enabled=False), snapshot)
    monkeypatch.setattr(
        waiver_schedule.sleeper_league,
        "resolve_user_id",
        lambda _session, _username: "user-id",
    )
    monkeypatch.setattr(
        waiver_schedule.sleeper_league,
        "list_leagues",
        lambda _session, _user_id, _season: [
            {
                "league_id": "2",
                "status": "in_season",
                "settings": {"waiver_day_of_week": 2, "daily_waivers": 0},
            }
        ],
    )

    pools = coordinator._fetch_sleeper_schedules(snapshot, NOW)

    assert [pool["league_key"] for pool in pools] == ["sleeper:2"]
    coordinator.close()


def test_sleeper_user_lookup_error_does_not_suppress_due_espn_report(
    monkeypatch,
) -> None:
    deadline = NOW + timedelta(hours=4)
    espn = LeagueRef("espn", "1", "ESPN", "Mine")
    sleeper = LeagueRef("sleeper", "2", "Sleeper", "Mine")
    coordinator = _coordinator(_config(), _snapshot(espn, sleeper))
    monkeypatch.setattr(
        coordinator,
        "_fetch_espn_pool",
        lambda _now: {
            "provider": "espn",
            "league_key": "espn:1",
            "deadline": deadline,
            "entries": [],
            "method": "Traditional rolling priority",
            "priority": 7,
        },
    )
    monkeypatch.setattr(
        waiver_schedule.sleeper_league,
        "resolve_user_id",
        lambda _session, _username: (_ for _ in ()).throw(
            NotifierError("Sleeper user not found")
        ),
    )
    _stub_report_formatting(monkeypatch, coordinator)

    reports = coordinator._discover_due(NOW)

    assert [report.key for report in reports] == [
        f"waiver:espn:1:{int(deadline.timestamp())}"
    ]
    coordinator.close()


def test_each_due_league_refreshes_independently_and_failure_retries_in_five_minutes(
    monkeypatch,
) -> None:
    deadline = NOW + timedelta(hours=4)
    espn = LeagueRef("espn", "1", "ESPN", "Mine")
    sleeper = LeagueRef("sleeper", "2", "Sleeper", "Mine")
    refresh_calls: list[set[str]] = []

    def refresh(keys: set[str]) -> bool:
        refresh_calls.append(keys)
        return keys == {"espn:1"}

    clock_value = 123.0
    coordinator = _coordinator(
        _config(),
        _snapshot(espn, sleeper),
        refresh_provider=refresh,
        clock=lambda: clock_value,
    )
    monkeypatch.setattr(
        coordinator,
        "_fetch_espn_pool",
        lambda _now: {
            "provider": "espn",
            "league_key": "espn:1",
            "deadline": deadline,
            "entries": [],
            "method": "Traditional rolling priority",
            "priority": 7,
        },
    )
    monkeypatch.setattr(
        coordinator,
        "_fetch_sleeper_schedules",
        lambda _snapshot, _now: [
            {
                "provider": "sleeper",
                "league_key": "sleeper:2",
                "deadline": deadline,
                "method": "Rolling priority",
                "priority": None,
            }
        ],
    )
    _stub_report_formatting(monkeypatch, coordinator)

    reports = coordinator.due_reports(NOW)

    assert refresh_calls == [{"espn:1"}, {"sleeper:2"}]
    assert [report.key for report in reports] == [
        f"waiver:espn:1:{int(deadline.timestamp())}"
    ]
    assert coordinator._next_check_at == clock_value + RETRY_INTERVAL_SECONDS
    coordinator.close()


def test_report_is_dropped_when_deadline_passes_during_provider_refresh(
    monkeypatch,
) -> None:
    deadline = NOW + timedelta(hours=4)
    league = LeagueRef("espn", "1", "ESPN", "Mine")
    provider_now = [NOW]

    def refresh(_league_keys: set[str]) -> bool:
        provider_now[0] = deadline
        return True

    coordinator = _coordinator(
        _config(),
        _snapshot(league),
        refresh_provider=refresh,
        now_provider=lambda: provider_now[0],
    )
    monkeypatch.setattr(
        coordinator,
        "_fetch_espn_pool",
        lambda _now: {
            "provider": "espn",
            "league_key": league.key,
            "deadline": deadline,
            "entries": [],
            "method": "Traditional rolling priority",
            "priority": 7,
        },
    )
    monkeypatch.setattr(
        coordinator,
        "_candidate_evidence",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("an expired pool must not be rendered")
        ),
    )

    assert coordinator._discover_due(NOW) == ()
    coordinator.close()


def test_report_is_dropped_when_deadline_passes_during_render(
    monkeypatch,
) -> None:
    deadline = NOW + timedelta(hours=4)
    league = LeagueRef("espn", "1", "ESPN", "Mine")
    provider_now = [NOW]
    coordinator = _coordinator(
        _config(),
        _snapshot(league),
        now_provider=lambda: provider_now[0],
    )
    monkeypatch.setattr(
        coordinator,
        "_fetch_espn_pool",
        lambda _now: {
            "provider": "espn",
            "league_key": league.key,
            "deadline": deadline,
            "entries": [],
            "method": "Traditional rolling priority",
            "priority": 7,
        },
    )
    monkeypatch.setattr(coordinator, "_candidate_evidence", lambda *args: ())
    monkeypatch.setattr(coordinator, "_context", lambda *args: object())

    def render_report(context, candidates):
        del context, candidates
        provider_now[0] = deadline
        return object()

    monkeypatch.setattr(waiver_schedule, "build_waiver_report", render_report)
    monkeypatch.setattr(
        waiver_schedule,
        "format_waiver_report_html",
        lambda _report: ("waiver report",),
    )
    monkeypatch.setattr(
        waiver_schedule.sleeper,
        "trending_adds",
        lambda *args, **kwargs: {},
    )

    assert coordinator._discover_due(NOW) == ()
    coordinator.close()


def test_evidence_starts_at_previous_processing_not_prior_monday(
    monkeypatch,
) -> None:
    deadline = datetime(2026, 8, 25, 7, tzinfo=timezone.utc)
    previous_deadline = datetime(2026, 8, 18, 7, tzinfo=timezone.utc)
    report_now = deadline - timedelta(hours=4)
    league = LeagueRef("espn", "1", "ESPN", "Mine")
    recent_calls: list[dict] = []

    class RecordingEvents:
        def recent(self, **kwargs):
            recent_calls.append(kwargs)
            return []

    coordinator = _coordinator(
        _config(),
        _snapshot(league),
        now_provider=lambda: report_now,
    )
    coordinator._events = RecordingEvents()
    monkeypatch.setattr(
        coordinator,
        "_fetch_espn_pool",
        lambda _now: {
            "provider": "espn",
            "league_key": league.key,
            "deadline": deadline,
            "entries": [],
            "method": "Traditional rolling priority",
            "priority": 7,
        },
    )
    _stub_report_formatting(monkeypatch, coordinator)

    reports = coordinator._discover_due(report_now)

    assert len(reports) == 1
    assert recent_calls[0]["since"] == previous_deadline
    assert recent_calls[0]["until"] == report_now + timedelta(seconds=1)
    coordinator.close()


def _candidate_row(
    player_name: str,
    event_type: str,
    *,
    subject_confident: bool = True,
) -> dict:
    return {
        "source": "twitter",
        "guid": f"twitter:{player_name}:{event_type}",
        "player_name": player_name,
        "headline": f"Team confirms {player_name} {event_type} update",
        "body": f"Team confirms {player_name} {event_type} update.",
        "url": "https://example.test/report",
        "published_at": (NOW - timedelta(hours=1)).isoformat(),
        "received_at": (NOW - timedelta(hours=1)).isoformat(),
        "event_type": event_type,
        "severity": 4,
        "subject_confident": subject_confident,
    }


def _candidate_evidence_for_row(row: dict):
    league = LeagueRef("sleeper", "2", "Sleeper", "Mine")
    snapshot = _snapshot(league)
    player_index = {
        "candidate": {
            "full_name": "Available Candidate",
            "position": "RB",
            "team": "SF",
            "status": "Active",
            "search_rank": 10,
            "depth_chart_order": 1,
            "injury_status": "",
        },
        "mine": {
            "full_name": "Player 2",
            "position": "RB",
            "team": "DAL",
            "status": "Active",
            "search_rank": 50,
            "depth_chart_order": 1,
            "injury_status": "",
        },
    }
    coordinator = _coordinator(_config(espn_enabled=False), snapshot)
    try:
        evidence = coordinator._candidate_evidence(
            {"provider": "sleeper", "league_key": league.key},
            snapshot,
            player_index,
            [row],
            {},
            {},
            NOW,
        )
    finally:
        coordinator.close()
    return next(
        candidate
        for candidate in evidence
        if candidate.name == "Available Candidate"
    )


@pytest.mark.parametrize("event_type", ["release", "suspension", "inactive"])
def test_direct_unavailability_events_remove_candidate_from_claim_pool(
    event_type: str,
) -> None:
    candidate = _candidate_evidence_for_row(
        _candidate_row("Available Candidate", event_type)
    )

    assert candidate.available is False


def test_direct_injury_caps_candidate_but_keeps_it_visible_to_watch() -> None:
    candidate = _candidate_evidence_for_row(
        _candidate_row("Available Candidate", "injury")
    )

    assert candidate.available is True
    assert candidate.recommendation_blocked is True
    assert candidate.injury_status == "Recent injury report"


def test_ambiguous_direct_injury_cannot_block_or_promote_candidate() -> None:
    candidate = _candidate_evidence_for_row(
        _candidate_row(
            "Available Candidate",
            "injury",
            subject_confident=False,
        )
    )

    assert candidate.available is True
    assert candidate.recommendation_blocked is False
    assert candidate.facts == ()


def test_backup_usage_article_does_not_assign_starters_injury_to_backup() -> None:
    coordinator = _coordinator(
        _config(espn_enabled=False),
        _snapshot(LeagueRef("sleeper", "2", "Sleeper", "Mine")),
    )
    starter = {
        "full_name": "Starting Runner",
        "position": "RB",
        "team": "ARI",
        "status": "Active",
        "search_rank": 20,
        "depth_chart_order": 1,
    }
    backup = {
        "full_name": "Backup Runner",
        "position": "RB",
        "team": "ARI",
        "status": "Active",
        "search_rank": 200,
        "depth_chart_order": 2,
    }
    records = {
        "startingrunner": starter,
        "backuprunner": backup,
    }
    row = {
        **_candidate_row("Backup Runner", "injury"),
        "headline": "Backup Runner handled work after Starting Runner left practice",
        "body": "Backup Runner handled work after Starting Runner left with a knee injury.",
    }
    try:
        facts, role, role_subject, _inferred, direct_concern = (
            coordinator._facts_for_candidate(
                "Backup Runner",
                backup,
                [row],
                records,
                tuple(records.items()),
            )
        )
    finally:
        coordinator.close()

    assert direct_concern == ""
    assert role == "uncertain_replacement"
    assert role_subject == "Starting Runner"
    assert any(fact.creates_opportunity for fact in facts)


def test_injured_backup_is_not_promoted_because_article_mentions_starter() -> None:
    coordinator = _coordinator(
        _config(espn_enabled=False),
        _snapshot(LeagueRef("sleeper", "2", "Sleeper", "Mine")),
    )
    starter = {
        "full_name": "Starting Runner",
        "position": "RB",
        "team": "ARI",
        "status": "Active",
        "search_rank": 20,
        "depth_chart_order": 1,
    }
    backup = {
        "full_name": "Backup Runner",
        "position": "RB",
        "team": "ARI",
        "status": "Active",
        "search_rank": 200,
        "depth_chart_order": 2,
    }
    records = {
        "startingrunner": starter,
        "backuprunner": backup,
    }
    row = {
        **_candidate_row("Backup Runner", "injury"),
        "headline": "Backup Runner injured his ankle while filling in for Starting Runner",
        "body": "Backup Runner injured his ankle while filling in for Starting Runner.",
    }
    try:
        facts, role, role_subject, _inferred, direct_concern = (
            coordinator._facts_for_candidate(
                "Backup Runner",
                backup,
                [row],
                records,
                tuple(records.items()),
            )
        )
    finally:
        coordinator.close()

    assert direct_concern == "injury"
    assert role == "depth_two"
    assert role_subject == ""
    assert not any(fact.creates_opportunity for fact in facts)


def test_healthy_starter_status_only_suppresses_news_it_is_fresh_enough_to_resolve(
) -> None:
    league = LeagueRef("sleeper", "2", "Sleeper", "Mine")
    snapshot = _snapshot(league)
    published = NOW - timedelta(days=5)
    records = {
        "starter": {
            "full_name": "Starting Runner",
            "position": "RB",
            "team": "ARI",
            "status": "Active",
            "injury_status": "",
            "search_rank": 20,
            "depth_chart_order": 1,
        },
        "backup": {
            "full_name": "Backup Runner",
            "position": "RB",
            "team": "ARI",
            "status": "Active",
            "injury_status": "",
            "search_rank": 200,
            "depth_chart_order": 2,
        },
    }
    row = {
        **_candidate_row("Backup Runner", "injury"),
        "headline": "Backup Runner handled work after Starting Runner left practice",
        "body": "Backup Runner handled work after Starting Runner left with a knee injury.",
        "published_at": published.isoformat(),
        "received_at": published.isoformat(),
    }

    def backup_evidence(refreshed_at: datetime):
        player_index = PlayerIndex(records, refreshed_at=refreshed_at)
        coordinator = _coordinator(_config(espn_enabled=False), snapshot)
        try:
            evidence = coordinator._candidate_evidence(
                {"provider": "sleeper", "league_key": league.key},
                snapshot,
                player_index,
                [row],
                {},
                {},
                NOW,
            )
        finally:
            coordinator.close()
        return next(item for item in evidence if item.name == "Backup Runner")

    current_index = backup_evidence(published)
    stale_index = backup_evidence(published - timedelta(seconds=1))

    assert not any(fact.creates_opportunity for fact in current_index.facts)
    assert any(fact.creates_opportunity for fact in stale_index.facts)


def test_direct_starter_decision_can_beat_a_stale_depth_order() -> None:
    coordinator = _coordinator(
        _config(espn_enabled=False),
        _snapshot(LeagueRef("sleeper", "2", "Sleeper", "Mine")),
    )
    candidate = {
        "full_name": "Named Starter",
        "position": "QB",
        "team": "CLE",
        "status": "Active",
        "search_rank": 200,
        "depth_chart_order": 2,
    }
    row = {
        **_candidate_row("Named Starter", "depth_chart"),
        "headline": "Browns named Named Starter their Week 1 starting quarterback",
        "body": "The Browns selected Named Starter to start Week 1.",
    }
    try:
        facts, role, _role_subject, _inferred, direct_concern = (
            coordinator._facts_for_candidate(
                "Named Starter",
                candidate,
                [row],
                {"namedstarter": candidate},
                (("namedstarter", candidate),),
            )
        )
    finally:
        coordinator.close()

    assert direct_concern == ""
    assert role == "confirmed_starter"
    assert any(fact.status == "role_starter" for fact in facts)


def test_stale_fantasypros_value_cannot_distort_drop_comparison() -> None:
    class RankedFantasyPros:
        def __init__(self, updated_at: datetime) -> None:
            self.updated_at = updated_at

        def signal(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(
                waiver_pos_rank="RB1",
                waiver_rank=1,
                ros_pos_rank="RB1",
                ros_rank=1,
                updated_at=self.updated_at,
            )

    league = LeagueRef("sleeper", "2", "Sleeper", "Mine")
    snapshot = _snapshot(league)
    player_index = {
        "mine": {
            "full_name": "Player 2",
            "position": "RB",
            "team": "SF",
            "status": "Active",
            "search_rank": 101,
            "depth_chart_order": 1,
        }
    }

    def assets(updated_at: datetime):
        coordinator = WaiverReportCoordinator(
            _config(espn_enabled=False, fantasypros_max_age_hours=12),
            snapshot_provider=lambda: snapshot,
            player_index_provider=lambda: player_index,
            refresh_provider=lambda _keys: True,
            event_store=_Events(),
            fantasypros=RankedFantasyPros(updated_at),
            completed_provider=lambda _key: False,
            session=requests.Session(),
        )
        try:
            return coordinator._roster_assets(
                league.key,
                snapshot,
                player_index,
                "PPR",
                NOW,
            )
        finally:
            coordinator.close()

    assert assets(NOW - timedelta(hours=13))[0].waiver_value == 75
    assert assets(NOW - timedelta(hours=1))[0].waiver_value == 100


@pytest.mark.parametrize(
    ("injury_status", "available", "blocked"),
    [
        ("OUT", True, True),
        ("INJURY_RESERVE", True, True),
        ("SUSPENSION", False, True),
        ("QUESTIONABLE", True, False),
        ("ACTIVE", True, False),
    ],
)
def test_live_espn_injury_status_gates_claims_before_sleeper_cache(
    injury_status: str,
    available: bool,
    blocked: bool,
) -> None:
    league = LeagueRef("espn", "1", "ESPN", "Mine")
    snapshot = _snapshot(league)
    player_index = {
        "candidate": {
            "full_name": "ESPN Candidate",
            "position": "RB",
            "team": "SF",
            "status": "Active",
            "search_rank": 25,
            "depth_chart_order": 1,
            # Deliberately stale/conflicting: the live ESPN pool must win.
            "injury_status": "",
        }
    }
    coordinator = _coordinator(_config(), snapshot)
    try:
        evidence = coordinator._candidate_evidence(
            {
                "provider": "espn",
                "league_key": league.key,
                "entries": [
                    {
                        "player": {
                            "fullName": "ESPN Candidate",
                            "defaultPositionId": 2,
                            "proTeamId": 25,
                            "injuryStatus": injury_status,
                            "ownership": {"percentOwned": 80},
                        }
                    }
                ],
            },
            snapshot,
            player_index,
            [],
            {},
            {},
            NOW,
        )[0]
    finally:
        coordinator.close()

    assert evidence.available is available
    assert evidence.recommendation_blocked is blocked
    if injury_status == "ACTIVE":
        assert evidence.injury_status == ""
    else:
        assert evidence.injury_status.startswith("ESPN ")
