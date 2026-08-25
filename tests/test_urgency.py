from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from notifier.embeddings import (
    INPUT_VERSION,
    EmbeddingVector,
    canonical_embedding_text,
    embedding_input_hash,
    pack_vector,
)
from notifier.event_store import EventStore
from notifier.models import (
    ActionUrgency,
    Alert,
    Classification,
    LeagueRef,
    NewsItem,
    RosterPlayer,
    RosterSnapshot,
)
from notifier.notify import format_alert
from notifier.outbox import DeliveryOutbox
from notifier.pipeline import _canonicalize_classification
from notifier.plays import Beneficiary, DepthCharts, LeaguePlays, plays_for_event
from notifier.urgency import (
    POLICY_VERSION,
    UrgencyService,
    apply_embedding_support,
    archive_rule_urgency,
    assess_rule_urgency,
)


def _item(
    player: str = "Mike Evans",
    headline: str = "Questionable after missing practice",
    *,
    confident: bool = True,
) -> NewsItem:
    return NewsItem(
        source="twitter",
        guid=f"tweet:{player}:{headline}",
        player_name=player,
        headline=headline,
        body=headline,
        url="https://x.com/reporter/status/1",
        published_at=datetime(2026, 8, 24, 12, tzinfo=timezone.utc),
        subject_confident=confident,
    )


def _alert(
    *,
    event: str = "injury",
    severity: int = 4,
    tier: str = "mine",
    subject_state: str = "mine",
    slot: str = "WR",
    subject_depth_order: int | None = 1,
    claimable: bool = False,
    bench: bool = False,
    confident: bool = True,
    availability_failed: bool = False,
    player: str = "Mike Evans",
    headline: str = "Questionable after missing practice",
) -> Alert:
    league = LeagueRef("espn", "1", "Home League", "Mine")
    plays = LeaguePlays(
        league=league,
        subject_state=subject_state,
        subject_owner="Mine" if subject_state == "mine" else "Rival",
        subject_depth_order=subject_depth_order,
        subject_lineup_slot=slot,
        beneficiaries=(
            [Beneficiary("Backup Player", "WR", 2, "free_agent")]
            if claimable
            else []
        ),
        bench_options=["Bench Player"] if bench else [],
    )
    return Alert(
        item=_item(player, headline, confident=confident),
        classification=Classification(
            event,
            severity,
            "Impact",
            True,
            {"direction": "negative" if event != "return" else "positive"},
        ),
        tier=tier,
        per_league=[] if tier in {"league", "preseason"} else [plays],
        all_leagues=[league],
        availability_refresh_failed=availability_failed,
    )


def _vector(values: tuple[float, ...] = (1.0, 0.0)) -> EmbeddingVector:
    return EmbeddingVector(
        model="provider/model",
        provider="openrouter",
        dimensions=2,
        input_version=INPUT_VERSION,
        input_hash="current",
        values=values,
        blob=pack_vector(values),
    )


def _history_row(
    player: str,
    *,
    level: str = "monitor",
    basis: str = "rules",
    event: str = "injury",
    direction: str = "negative",
    tier: str = "mine",
    event_status: str = "questionable",
    action_context: str = "mine_bench",
    values: tuple[float, ...] = (1.0, 0.0),
) -> dict:
    item = _item(player, f"{player} remains sidelined")
    return {
        "report_id": f"report:{player}",
        "source": item.source,
        "guid": item.guid,
        "player_name": player,
        "headline": item.headline,
        "body": item.body,
        "url": item.url,
        "published_at": item.published_at.isoformat(),
        "subject_confident": 1,
        "event_type": event,
        "direction": direction,
        "severity": 4,
        "tier": tier,
        "feedback": None,
        "embedding_model": "provider/model",
        "embedding_provider": "openrouter",
        "embedding_dimensions": 2,
        "embedding_input_version": INPUT_VERSION,
        "embedding_input_hash": embedding_input_hash(canonical_embedding_text(item)),
        "embedding_at": "2026-08-24T12:00:00+00:00",
        "embedding": pack_vector(values),
        "urgency_rule_level": level,
        "urgency_level": level,
        "urgency_basis": basis,
        "urgency_policy_version": POLICY_VERSION,
        "urgency_availability_verified": 1,
        "urgency_event_type": event,
        "urgency_direction": direction,
        "urgency_event_status": event_status,
        "urgency_action_context": action_context,
        "urgency_subject_is_starter": 0,
    }


def test_rule_urgency_reserves_act_now_for_live_roster_actions() -> None:
    assert (
        assess_rule_urgency(
            _alert(bench=True, headline="Ruled out for Week 1")
        ).level
        == "act_now"
    )
    assert (
        assess_rule_urgency(
            _alert(
                tier="claimable",
                subject_state="rostered",
                claimable=True,
                headline="Left practice after suffering a knee injury",
            )
        ).level
        == "act_now"
    )
    assert (
        assess_rule_urgency(
            _alert(
                severity=3,
                tier="claimable",
                subject_state="rostered",
                claimable=True,
            )
        ).level
        == "act_today"
    )


def test_rule_urgency_distinguishes_starter_from_bench_stash() -> None:
    starter = assess_rule_urgency(_alert(slot="WR", severity=4))
    bench = assess_rule_urgency(_alert(slot="BE", severity=3))

    assert starter.level == "act_today"
    assert bench.level == "monitor"


def test_bench_stash_never_becomes_immediate_from_its_depth_successor() -> None:
    tyson = assess_rule_urgency(
        _alert(
            player="Jordyn Tyson",
            slot="BE",
            severity=4,
            claimable=True,
        )
    )

    assert tyson.level == "monitor"
    assert tyson.action_available is False
    assert tyson.action_context == "mine_bench"


def test_questionable_starter_without_a_verified_move_is_not_act_now() -> None:
    evans = assess_rule_urgency(
        _alert(
            player="Mike Evans",
            headline="Questionable after missing practice",
            slot="WR",
            severity=4,
            claimable=False,
            bench=False,
        )
    )

    assert evans.level == "act_today"
    assert evans.action_available is False
    assert evans.event_status == "questionable"


def test_release_successor_is_capped_at_act_today() -> None:
    released = assess_rule_urgency(
        _alert(
            player="Depth Back",
            event="release",
            tier="claimable",
            subject_state="rostered",
            severity=4,
            claimable=True,
        )
    )

    assert released.level == "act_today"
    assert released.reason_codes[0] == "claimable_watch"


def test_known_return_release_and_preseason_controls() -> None:
    kittle = _alert(
        player="George Kittle",
        headline="Activated from active/PUP",
        event="return",
        tier="preseason",
        subject_state="free_agent",
    )
    noah = _alert(
        player="Noah Brown",
        headline="Raiders released veteran WR Noah Brown",
        event="other",
        tier="league",
        subject_state="free_agent",
    )

    assert assess_rule_urgency(kittle).level == "monitor"
    assert assess_rule_urgency(noah).level == "fyi"


def test_activation_with_limited_practice_is_still_a_return_not_an_injury() -> None:
    kittle = _alert(
        player="George Kittle",
        headline="49ers activated Kittle from active/PUP; he was limited in practice",
        event="injury",
        tier="mine",
        slot="TE",
        severity=4,
        claimable=True,
    )

    urgency = assess_rule_urgency(kittle)
    corrected = _canonicalize_classification(kittle.item, kittle.classification)
    filtered = plays_for_event(
        kittle.per_league,
        urgency.canonical_event_type,
        kittle.classification.severity,
    )

    assert urgency.canonical_event_type == "return"
    assert corrected.event_type == "return"
    assert corrected.raw["direction"] == "positive"
    assert urgency.direction == "positive"
    assert urgency.level == "monitor"
    assert urgency.action_available is False
    assert all(not plays.has_action for plays in filtered)


def test_activation_does_not_hide_a_same_report_reinjury() -> None:
    alert = _alert(
        player="Mike Evans",
        headline="Evans was activated from PUP but re-injured his knee in practice",
        event="injury",
        slot="WR",
        severity=4,
        bench=True,
    )

    urgency = assess_rule_urgency(alert)

    assert urgency.canonical_event_type == "injury"
    assert urgency.direction == "negative"
    assert urgency.level == "act_today"


@pytest.mark.parametrize(
    "headline",
    [
        "After being ruled out last week, Mike Evans returned to practice today",
        "Mike Evans was inactive Sunday but returned to practice Monday",
        "Mike Evans returned to practice today after being ruled out last week",
        "Mike Evans returned to practice Monday despite being inactive Sunday",
    ],
)
def test_historical_absence_before_a_current_return_does_not_create_act_now(
    headline: str,
) -> None:
    alert = _alert(
        player="Mike Evans",
        headline=headline,
        event="inactive",
        slot="WR",
        severity=4,
        bench=True,
    )

    urgency = assess_rule_urgency(alert)

    assert urgency.canonical_event_type == "return"
    assert urgency.direction == "positive"
    assert urgency.level == "monitor"


def test_return_followed_by_ruled_out_preserves_the_newer_absence() -> None:
    alert = _alert(
        player="Mike Evans",
        headline="Evans returned to practice Monday but was ruled out Tuesday",
        event="injury",
        slot="WR",
        severity=4,
        bench=True,
    )

    urgency = assess_rule_urgency(alert)

    assert urgency.canonical_event_type == "injury"
    assert urgency.direction == "negative"
    assert urgency.level == "act_now"


@pytest.mark.parametrize(
    "headline",
    [
        "The NFL lifted Mike Evans' suspension",
        "Mike Evans was reinstated after serving his suspension",
        "Mike Evans' suspension has ended",
    ],
)
def test_resolved_suspension_never_creates_a_removal_action(headline: str) -> None:
    alert = _alert(
        player="Mike Evans",
        headline=headline,
        event="suspension",
        slot="WR",
        severity=4,
        bench=True,
        claimable=True,
    )

    urgency = assess_rule_urgency(alert)
    corrected = _canonicalize_classification(alert.item, alert.classification)
    filtered = plays_for_event(
        alert.per_league,
        urgency.canonical_event_type,
        alert.classification.severity,
    )

    assert urgency.canonical_event_type == "return"
    assert urgency.direction == "positive"
    assert urgency.level == "monitor"
    assert corrected.event_type == "return"
    assert corrected.raw["direction"] == "positive"
    assert all(not plays.has_action for plays in filtered)


@pytest.mark.parametrize(
    "headline",
    [
        "Mike Evans was not reinstated after his suspension",
        "Mike Evans was reinstated Monday but suspended again Tuesday",
        "Mike Evans was reinstated Monday and then re-suspended Tuesday",
        "Mike Evans' suspension was reinstated Tuesday",
    ],
)
def test_negated_or_reversed_suspension_resolution_stays_negative(
    headline: str,
) -> None:
    alert = _alert(
        player="Mike Evans",
        headline=headline,
        event="suspension",
        slot="WR",
        severity=4,
        bench=True,
    )

    urgency = assess_rule_urgency(alert)

    assert urgency.canonical_event_type == "suspension"
    assert urgency.direction == "negative"
    assert urgency.level == "act_now"


def test_other_players_reinstatement_does_not_clear_subject_suspension() -> None:
    alert = _alert(
        player="Mike Evans",
        headline="Mike Evans was suspended; the NFL reinstated Chris Godwin",
        event="suspension",
        slot="WR",
        severity=4,
        bench=True,
    )

    urgency = assess_rule_urgency(alert)

    assert urgency.canonical_event_type == "suspension"
    assert urgency.direction == "negative"
    assert urgency.level == "act_now"


@pytest.mark.parametrize(
    "headline",
    [
        "Team re-signed Mike Evans after releasing him last week",
        "Team released Mike Evans Monday but re-signed him Tuesday",
        "Team waived Mike Evans and then claimed him back",
    ],
)
def test_reversed_release_becomes_a_positive_signing_without_backup_action(
    headline: str,
) -> None:
    alert = _alert(
        player="Mike Evans",
        headline=headline,
        event="release",
        slot="WR",
        severity=4,
        bench=True,
        claimable=True,
    )

    urgency = assess_rule_urgency(alert)
    corrected = _canonicalize_classification(alert.item, alert.classification)
    filtered = plays_for_event(
        alert.per_league,
        urgency.canonical_event_type,
        alert.classification.severity,
    )

    assert urgency.canonical_event_type == "signing"
    assert urgency.direction == "positive"
    assert urgency.level == "act_today"
    assert corrected.event_type == "signing"
    assert corrected.raw["direction"] == "positive"
    assert all(not plays.has_action for plays in filtered)


@pytest.mark.parametrize(
    "headline",
    [
        "Team released Mike Evans and did not re-sign him",
        "Team released Mike Evans but he wasn't re-signed",
        "Team re-signed Mike Evans Monday but released him Tuesday",
        "Team waived Mike Evans and he was not claimed back",
        "Team released Mike Evans after re-signing Chris Godwin",
        "Team waived Mike Evans and then claimed Chris Godwin back",
    ],
)
def test_failed_or_later_reversed_signing_stays_a_release(headline: str) -> None:
    alert = _alert(
        player="Mike Evans",
        headline=headline,
        event="release",
        slot="WR",
        severity=4,
        bench=True,
    )

    urgency = assess_rule_urgency(alert)

    assert urgency.canonical_event_type == "release"
    assert urgency.direction == "negative"
    assert urgency.level == "act_now"


def test_activation_overrides_inactive_model_label_but_not_explicit_negation() -> None:
    activated = _alert(
        player="Mike Evans",
        headline="Buccaneers activated Evans from active/PUP",
        event="inactive",
        slot="WR",
        severity=4,
        bench=True,
    )
    not_cleared = _alert(
        player="Mike Evans",
        headline="Evans has not been cleared to play after the injury",
        event="return",
        slot="WR",
        severity=4,
        bench=True,
    )

    activated_urgency = assess_rule_urgency(activated)
    not_cleared_urgency = assess_rule_urgency(not_cleared)

    assert activated_urgency.canonical_event_type == "return"
    assert activated_urgency.level == "monitor"
    assert not_cleared_urgency.canonical_event_type == "injury"
    assert not_cleared_urgency.level == "act_today"


@pytest.mark.parametrize(
    "headline",
    [
        "Evans has not yet been activated from active/PUP",
        "Evans hasn't yet been activated from active/PUP",
        "Evans has yet to be cleared to play",
        "Evans is not expected to be activated from active/PUP",
        "Evans was unable to be cleared to play",
    ],
)
def test_bounded_return_negation_never_becomes_a_positive_return(
    headline: str,
) -> None:
    alert = _alert(
        player="Mike Evans",
        headline=headline,
        event="return",
        slot="WR",
        severity=4,
        bench=True,
    )

    urgency = assess_rule_urgency(alert)

    assert urgency.canonical_event_type == "injury"
    assert urgency.direction == "negative"
    assert urgency.level == "act_today"


@pytest.mark.parametrize(
    "headline",
    [
        "Evans was not cleared Monday. He returned to practice Tuesday",
        "Evans was not cleared Monday but returned to practice Tuesday",
        "Evans was not only activated from PUP but returned to practice",
    ],
)
def test_earlier_negation_does_not_cancel_a_later_positive_return(
    headline: str,
) -> None:
    alert = _alert(
        player="Mike Evans",
        headline=headline,
        event="inactive",
        slot="WR",
        severity=4,
        bench=True,
    )

    urgency = assess_rule_urgency(alert)

    assert urgency.canonical_event_type == "return"
    assert urgency.direction == "positive"
    assert urgency.level == "monitor"


def test_other_players_return_does_not_clear_the_report_subject() -> None:
    alert = _alert(
        player="Mike Evans",
        headline="Mike Evans remains inactive; Chris Godwin returned to practice",
        event="inactive",
        slot="WR",
        severity=4,
        bench=True,
    )

    urgency = assess_rule_urgency(alert)

    assert urgency.canonical_event_type == "inactive"
    assert urgency.direction == "negative"
    assert urgency.level == "act_now"


def test_other_players_later_absence_does_not_reverse_subject_return() -> None:
    alert = _alert(
        player="Mike Evans",
        headline="Mike Evans returned to practice; Chris Godwin was ruled out",
        event="inactive",
        slot="WR",
        severity=4,
        bench=True,
    )

    urgency = assess_rule_urgency(alert)

    assert urgency.canonical_event_type == "return"
    assert urgency.direction == "positive"
    assert urgency.level == "monitor"


@pytest.mark.parametrize(
    "headline",
    [
        "Mike Evans is active for Week 1",
        "Mike Evans will play Sunday",
        "Mike Evans is available for Sunday",
    ],
)
def test_definitive_game_availability_clears_an_inactive_model_label(
    headline: str,
) -> None:
    alert = _alert(
        player="Mike Evans",
        headline=headline,
        event="inactive",
        slot="WR",
        severity=4,
        bench=True,
        claimable=True,
    )

    urgency = assess_rule_urgency(alert)
    corrected = _canonicalize_classification(alert.item, alert.classification)
    filtered = plays_for_event(
        alert.per_league,
        urgency.canonical_event_type,
        alert.classification.severity,
    )

    assert urgency.canonical_event_type == "return"
    assert urgency.direction == "positive"
    assert urgency.level == "monitor"
    assert urgency.action_available is False
    assert corrected.event_type == "return"
    assert corrected.raw["direction"] == "positive"
    assert all(not plays.has_action for plays in filtered)


@pytest.mark.parametrize(
    "headline",
    [
        "Mike Evans is not active for Week 1",
        "Mike Evans will not play Sunday",
        "Mike Evans is not available for Sunday",
        "Mike Evans is active for Week 1 but was ruled out after warmups",
    ],
)
def test_negated_or_later_reversed_game_availability_stays_inactive(
    headline: str,
) -> None:
    alert = _alert(
        player="Mike Evans",
        headline=headline,
        event="inactive",
        slot="WR",
        severity=4,
        bench=True,
    )

    urgency = assess_rule_urgency(alert)

    assert urgency.canonical_event_type == "inactive"
    assert urgency.direction == "negative"
    assert urgency.level == "act_now"


@pytest.mark.parametrize(
    "headline",
    [
        "Mike Evans remains inactive; Chris Godwin is active for Week 1",
        "Chris Godwin will play Sunday with Mike Evans watching",
        "Chris Godwin is available for Sunday; Mike Evans remains inactive",
    ],
)
def test_other_players_game_availability_does_not_clear_subject(
    headline: str,
) -> None:
    alert = _alert(
        player="Mike Evans",
        headline=headline,
        event="inactive",
        slot="WR",
        severity=4,
        bench=True,
    )

    urgency = assess_rule_urgency(alert)

    assert urgency.canonical_event_type == "inactive"
    assert urgency.direction == "negative"
    assert urgency.level == "act_now"


def test_other_players_later_inactive_status_does_not_reverse_subject_active() -> None:
    alert = _alert(
        player="Mike Evans",
        headline="Mike Evans is active for Week 1; Chris Godwin is inactive",
        event="inactive",
        slot="WR",
        severity=4,
        bench=True,
    )

    urgency = assess_rule_urgency(alert)

    assert urgency.canonical_event_type == "return"
    assert urgency.direction == "positive"
    assert urgency.level == "monitor"


@pytest.mark.parametrize(
    "headline",
    [
        "Mike Evans return to practice",
        "Mike Evans returns to practice",
        "Mike Evans returned to full practice",
        "Mike Evans resumes practice",
        "Mike Evans resumed full practice",
    ],
)
def test_return_morphology_overrides_an_inactive_model_label(headline: str) -> None:
    alert = _alert(
        player="Mike Evans",
        headline=headline,
        event="inactive",
        slot="WR",
        severity=4,
        bench=True,
    )

    urgency = assess_rule_urgency(alert)

    assert urgency.canonical_event_type == "return"
    assert urgency.direction == "positive"
    assert urgency.level == "monitor"


def test_questionable_starter_is_today_but_bench_player_is_monitor() -> None:
    evans = _alert(event="practice_report", severity=3, slot="WR")
    tyson = _alert(
        player="Jordyn Tyson",
        event="practice_report",
        severity=3,
        slot="BE",
    )

    assert assess_rule_urgency(evans).level == "act_today"
    assert assess_rule_urgency(tyson).level == "monitor"


def test_uncertain_subject_and_failed_availability_are_capped_at_monitor() -> None:
    assert assess_rule_urgency(_alert(severity=5, confident=False)).level == "monitor"
    failed = assess_rule_urgency(
        _alert(severity=5, claimable=True, availability_failed=True)
    )
    assert failed.level == "monitor"
    assert failed.availability_verified is False


def test_embedding_default_corroborates_without_changing_the_rule_band() -> None:
    alert = _alert(severity=3, slot="BE", bench=True)
    base = assess_rule_urgency(alert)
    assert base.level == "monitor"

    assessed = apply_embedding_support(
        alert,
        base,
        _vector(),
        [_history_row("Player One"), _history_row("Player Two")],
        threshold=0.70,
        min_neighbors=2,
    )

    assert assessed.rule_level == "monitor"
    assert assessed.level == "monitor"
    assert assessed.embedding_delta == 0
    assert assessed.embedding_support_count == 2
    assert assessed.basis == "rules+embedding_support"


def test_opt_in_lift_requires_three_context_matched_live_neighbors() -> None:
    alert = _alert(severity=2, slot="WR", bench=True)
    base = assess_rule_urgency(alert)
    rows = [
        _history_row("Player One", level="act_today", action_context="mine_starter"),
        _history_row("Player Two", level="act_today", action_context="mine_starter"),
        _history_row("Player Three", level="act_today", action_context="mine_starter"),
    ]

    assessed = apply_embedding_support(
        alert,
        base,
        _vector(),
        rows,
        threshold=0.70,
        min_neighbors=2,
        allow_lift=True,
    )

    assert assessed.level == "act_today"
    assert assessed.embedding_delta == 1
    assert assessed.embedding_support_count == 3
    assert assessed.basis == "rules+embedding_lift"


@pytest.mark.parametrize(
    "rows",
    [
        [_history_row("Mike Evans"), _history_row("Mike Evans")],
        [
            _history_row("Player One", basis="archive_replay"),
            _history_row("Player Two", basis="archive_replay"),
        ],
        [
            _history_row(
                "Player One",
                event="return",
                direction="positive",
                event_status="cleared",
            ),
            _history_row(
                "Player Two",
                event="return",
                direction="positive",
                event_status="cleared",
            ),
        ],
    ],
)
def test_embedding_lift_abstains_without_independent_live_compatible_history(
    rows: list[dict],
) -> None:
    alert = _alert(severity=3, slot="BE", bench=True)
    base = assess_rule_urgency(alert)

    assessed = apply_embedding_support(
        alert,
        base,
        _vector(),
        rows,
        threshold=0.70,
        min_neighbors=2,
    )

    assert assessed.level == base.level
    assert assessed.embedding_delta == 0


def test_embedding_history_never_promotes_to_act_now() -> None:
    alert = _alert(severity=2, slot="WR", bench=True)
    base = assess_rule_urgency(alert)
    rows = [
        _history_row("Player One", level="act_now", action_context="mine_starter"),
        _history_row("Player Two", level="act_now", action_context="mine_starter"),
        _history_row("Player Three", level="act_now", action_context="mine_starter"),
    ]

    assessed = apply_embedding_support(
        alert,
        base,
        _vector(),
        rows,
        threshold=0.70,
        min_neighbors=2,
        allow_lift=True,
    )

    assert assessed.level == "act_today"


def test_embedding_provider_failure_is_exact_rule_fallback() -> None:
    class BrokenEmbeddings:
        enabled = True

        @staticmethod
        def current_vector(_item):
            raise RuntimeError("offline")

    service = UrgencyService(SimpleNamespace(), BrokenEmbeddings())
    alert = _alert(severity=3, slot="BE")
    expected = assess_rule_urgency(alert)

    actual = service.assess(alert)

    assert actual.urgency == expected
    assert service.status().failures == 1


def test_archive_backfill_never_invents_live_action_context() -> None:
    urgency = archive_rule_urgency(
        {"event_type": "injury", "severity": 5, "tier": "mine"}
    )

    assert urgency is not None
    assert urgency.level == "monitor"
    assert urgency.basis == "archive_replay"
    assert urgency.action_available is False
    assert urgency.availability_verified is False


def test_lineup_slot_survives_depth_join() -> None:
    league = LeagueRef("espn", "1", "Home", "Mine")
    snapshot = RosterSnapshot(
        generated_at=None,
        leagues=[league],
        players=[
            RosterPlayer("Mike Evans", "WR", "TB", "WR", True, "Mine", league.key)
        ],
    )
    index = {
        "1": {
            "full_name": "Mike Evans",
            "team": "TB",
            "position": "WR",
            "status": "Active",
            "depth_chart_order": 1,
        }
    }

    _record, per_league = DepthCharts(index, snapshot).build(
        subject_names=("Mike Evans", ""), snapshot=snapshot
    )

    assert per_league[0].subject_lineup_slot == "WR"
    assert per_league[0].subject_is_starter is True


def test_outbox_round_trip_preserves_urgency_and_lineup_slot(tmp_path) -> None:
    alert = _alert(severity=3, slot="BE", bench=True)
    alert = replace(alert, urgency=assess_rule_urgency(alert))
    first = DeliveryOutbox(tmp_path)
    pending = first.add(alert)

    restored = DeliveryOutbox(tmp_path).get(pending.delivery_id)

    assert restored is not None
    assert restored.alert.urgency == alert.urgency
    assert restored.alert.per_league[0].subject_lineup_slot == "BE"


def test_event_store_persists_urgency_and_returns_only_prior_compatible_rows(
    tmp_path,
) -> None:
    store = EventStore(tmp_path)
    now = datetime.now(timezone.utc)
    classification = Classification(
        "injury", 4, "Impact", True, {"direction": "negative"}
    )
    items = [
        replace(
            _item(player, f"{player} remains sidelined"),
            published_at=now + timedelta(minutes=index),
        )
        for index, player in enumerate(("Player One", "Player Two", "Current Player"))
    ]
    try:
        for item in items:
            store.record_received(item)
            store.record_classification(item, classification, tier="mine")
            text = canonical_embedding_text(item)
            store.store_embedding(
                item,
                "provider/model",
                pack_vector((1.0, 0.0)),
                provider="openrouter",
                dimensions=2,
                input_version=INPUT_VERSION,
                input_hash=embedding_input_hash(text),
            )
        prior_urgency = ActionUrgency(
            "act_today",
            "act_today",
            policy_version=POLICY_VERSION,
            availability_verified=True,
            canonical_event_type="injury",
            direction="negative",
            event_status="unspecified",
            action_context="mine_bench",
        )
        before = store.get(items[0])["updated_at"]
        assert store.record_urgency(items[0], prior_urgency) is True
        assert store.record_urgency(items[1], prior_urgency) is True
        assert store.get(items[0])["updated_at"] == before

        rows = store.recent_urgency_candidates(
            items[2],
            event_type="injury",
            direction="negative",
            event_status="unspecified",
            action_context="mine_bench",
            tier="mine",
            model="provider/model",
            provider="openrouter",
            dimensions=2,
            input_version=INPUT_VERSION,
            since_days=365,
        )
    finally:
        store.close()

    assert {row["player_name"] for row in rows} == {"Player One", "Player Two"}
    assert all(row["urgency_rule_level"] == "act_today" for row in rows)
    assert all(row["urgency_event_type"] == "injury" for row in rows)
    assert all(row["urgency_action_context"] == "mine_bench" for row in rows)


def test_archive_write_cannot_replace_live_urgency_from_stale_snapshot(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item("Starter", "Starter ruled out")
    classification = Classification(
        "injury", 5, "Replacement needed", True, {"direction": "negative"}
    )
    try:
        store.record_received(item)
        store.record_classification(item, classification, tier="mine")
        stale = store.get(item)
        assert stale is not None
        chronology = (stale["received_at"], stale["updated_at"])
        live = ActionUrgency(
            "act_now",
            "act_now",
            basis="rules",
            policy_version=POLICY_VERSION,
            action_available=True,
            roster_relevant=True,
            availability_verified=True,
            canonical_event_type="injury",
            direction="negative",
            event_status="inactive",
            action_context="mine_starter",
            subject_is_starter=True,
        )
        assert store.record_urgency(item, live) is True

        archive = archive_rule_urgency(stale)
        assert archive is not None
        assert store.record_archive_urgency(item, archive) is False
        saved = store.get(item)
    finally:
        store.close()

    assert saved is not None
    assert saved["urgency_level"] == "act_now"
    assert saved["urgency_basis"] == "rules"
    assert saved["urgency_action_context"] == "mine_starter"
    assert (saved["received_at"], saved["updated_at"]) == chronology


def test_force_backfill_never_overwrites_live_basis(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item("Starter", "Starter ruled out for the season")
    classification = Classification(
        "injury", 5, "Replacement needed", True, {"direction": "negative"}
    )
    live = ActionUrgency(
        "act_now",
        "act_now",
        basis="rules",
        policy_version=POLICY_VERSION,
        action_available=True,
        roster_relevant=True,
        availability_verified=True,
        canonical_event_type="injury",
        direction="negative",
        event_status="season_out",
        action_context="mine_starter",
        subject_is_starter=True,
    )
    try:
        store.record_received(item)
        store.record_classification(item, classification, tier="mine")
        store.record_urgency(item, live)
        before = store.get(item)
    finally:
        store.close()

    environment = os.environ.copy()
    environment["NOTIFIER_STATE_DIR"] = str(tmp_path)
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "bin/backfill-urgency.py"), "--force"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "protected_live=1" in completed.stdout
    verify = EventStore(tmp_path)
    try:
        after = verify.get(item)
    finally:
        verify.close()
    assert before is not None and after is not None
    assert after["urgency_level"] == "act_now"
    assert after["urgency_basis"] == "rules"
    assert after["urgency_action_context"] == "mine_starter"
    assert after["urgency_at"] == before["urgency_at"]
    assert after["received_at"] == before["received_at"]
    assert after["updated_at"] == before["updated_at"]


def test_force_backfill_dry_run_matches_atomic_partial_row_guard(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item("Starter", "Starter ruled out for the season")
    classification = Classification(
        "injury", 5, "Replacement needed", True, {"direction": "negative"}
    )
    try:
        store.record_received(item)
        store.record_classification(item, classification, tier="mine")
    finally:
        store.close()
    connection = sqlite3.connect(tmp_path / "news-events.sqlite3")
    connection.execute("UPDATE news_events SET urgency_level = 'monitor'")
    connection.commit()
    connection.close()

    environment = os.environ.copy()
    environment["NOTIFIER_STATE_DIR"] = str(tmp_path)
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "bin/backfill-urgency.py"),
        "--force",
    ]
    dry_run = subprocess.run(
        [*command, "--dry-run"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    actual = subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert dry_run.returncode == 0, dry_run.stderr
    assert actual.returncode == 0, actual.stderr
    assert "saved=0" in dry_run.stdout
    assert "protected_live=1" in dry_run.stdout
    assert "saved=0" in actual.stdout
    assert "protected_live=1" in actual.stdout
    verify = EventStore(tmp_path)
    try:
        row = verify.get(item)
    finally:
        verify.close()
    assert row is not None
    assert row["urgency_level"] == "monitor"
    assert row["urgency_rule_level"] is None
    assert row["urgency_basis"] is None
    assert row["urgency_policy_version"] is None


@pytest.mark.parametrize("script", ["backfill-urgency.py", "eval-urgency.py"])
def test_urgency_audit_and_dry_run_do_not_create_a_database(
    tmp_path,
    script: str,
) -> None:
    environment = os.environ.copy()
    environment["NOTIFIER_STATE_DIR"] = str(tmp_path)
    root = Path(__file__).resolve().parents[1]
    arguments = [sys.executable, str(root / "bin" / script)]
    if script == "backfill-urgency.py":
        arguments.append("--dry-run")

    completed = subprocess.run(
        arguments,
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "news-events.sqlite3").exists()


def test_formatter_adds_one_short_urgency_line_and_omits_fyi() -> None:
    today = _alert(event="practice_report", severity=3)
    today = replace(today, urgency=assess_rule_urgency(today))
    info = _alert(event="release", tier="league", subject_state="free_agent")
    info = replace(info, urgency=assess_rule_urgency(info))

    assert "⏰ <b>ACT TODAY</b>" in format_alert(today)
    assert "<b>FYI</b>" not in format_alert(info)
