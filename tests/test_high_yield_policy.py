from __future__ import annotations

from datetime import datetime, timezone

from notifier.models import ActionUrgency, Alert, Classification, NewsItem
from notifier.pipeline import _passes_high_yield_live_gate


def _alert(
    tier: str,
    *,
    severity: int = 4,
    level: str = "monitor",
    action: bool = False,
    relevant: bool = False,
    confident: bool = True,
    event: str = "injury",
) -> Alert:
    return Alert(
        item=NewsItem(
            source="twitter",
            guid=f"tweet:{tier}:{severity}:{level}",
            player_name="Example Player",
            headline="Example Player update",
            body="Example Player update",
            url="https://x.com/reporter/status/1",
            published_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
            subject_confident=confident,
        ),
        classification=Classification(
            event,
            severity,
            "",
            action,
            {"direction": "negative"},
        ),
        tier=tier,
        urgency=ActionUrgency(
            rule_level=level,
            level=level,
            action_available=action,
            roster_relevant=relevant,
            availability_verified=True,
            canonical_event_type=event,
        ),
    )


def test_owned_player_news_remains_immediate() -> None:
    assert _passes_high_yield_live_gate(
        _alert("mine", severity=2, relevant=True),
        None,
    )


def test_claimable_news_requires_a_verified_action_today() -> None:
    assert _passes_high_yield_live_gate(
        _alert("claimable", severity=3, level="act_today", action=True),
        {"position": "RB", "search_rank": 300},
    )
    assert not _passes_high_yield_live_gate(
        _alert("claimable", severity=5, level="monitor", action=False),
        {"position": "RB", "search_rank": 10},
    )


def test_general_news_requires_a_top_fantasy_player_and_severity_five() -> None:
    top_player = {"position": "RB", "search_rank": 25}
    assert _passes_high_yield_live_gate(
        _alert("league", severity=5),
        top_player,
    )
    assert not _passes_high_yield_live_gate(
        _alert("league", severity=4),
        top_player,
    )
    assert not _passes_high_yield_live_gate(
        _alert("league", severity=5),
        {"position": "RB", "search_rank": 600},
    )
    assert not _passes_high_yield_live_gate(
        _alert("league", severity=5),
        {"position": "DB", "search_rank": 10},
    )


def test_uncertain_subject_and_generic_other_news_never_interrupt() -> None:
    record = {"position": "WR", "search_rank": 20}
    assert not _passes_high_yield_live_gate(
        _alert("league", severity=5, confident=False),
        record,
    )
    assert not _passes_high_yield_live_gate(
        _alert("league", severity=5, event="other"),
        record,
    )
