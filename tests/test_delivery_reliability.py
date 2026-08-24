import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from notifier.classify import _fallback
from notifier.dedupe import (
    SeenStore,
    event_fact_signature,
    event_status,
    role_decision_status,
    semantic_event_fact_signature,
    semantic_event_status,
    semantic_event_type,
)
from notifier.models import (
    Alert,
    Classification,
    LeagueRef,
    NewsItem,
    RosterCapacity,
    RosterSnapshot,
)
from notifier.outbox import DeliveryOutbox, _news_item
from notifier.pipeline import Notifier, _alert_supersedes
from notifier.notify import format_alert
from notifier.plays import Beneficiary, DepthEntry, LeaguePlays, TeamContext
from notifier.sources.twitter import TwitterStream


def _item(
    guid: str = "twitter:1:Example Player",
    source: str = "twitter",
    headline: str = "Example Player was ruled out",
) -> NewsItem:
    return NewsItem(
        source=source,
        guid=guid,
        player_name="Example Player",
        headline=headline,
        body=headline,
        url="https://example.test/news",
        published_at=datetime(2026, 8, 23, 12, tzinfo=timezone.utc),
    )


def _alert(item: NewsItem, severity: int = 4, event_type: str = "inactive") -> Alert:
    return Alert(
        item=item,
        classification=Classification(
            event_type=event_type,
            severity=severity,
            fantasy_impact="Availability changed",
            is_actionable=True,
            raw={"source": "test"},
        ),
        tier="league",
    )


def _notifier(tmp_path, *, dry_run: bool = False) -> Notifier:
    notifier = Notifier.__new__(Notifier)
    notifier.config = SimpleNamespace(dry_run=dry_run)
    notifier.session = object()
    notifier.seen = SeenStore(tmp_path / "seen-items.json")
    notifier.outbox = DeliveryOutbox(tmp_path)
    notifier._state_lock = threading.RLock()
    notifier._delivery_lock = threading.RLock()
    notifier._inflight_items = {}
    return notifier


def _event_row(
    alert: Alert,
    *,
    received_at: datetime,
    outcome: str = "delivered",
) -> dict:
    item = alert.item
    classification = alert.classification
    return {
        "guid": item.guid,
        "source": item.source,
        "player_name": item.player_name,
        "headline": item.headline,
        "body": item.body,
        "url": item.url,
        "published_at": item.published_at.isoformat() if item.published_at else None,
        "received_at": received_at.isoformat(),
        "updated_at": received_at.isoformat(),
        "event_type": classification.event_type,
        "severity": classification.severity,
        "summary": classification.fantasy_impact,
        "is_actionable": classification.is_actionable,
        "tier": alert.tier,
        "outcome": outcome,
    }


def test_failed_send_stays_in_durable_outbox_and_dedupe_is_unmodified(
    tmp_path, monkeypatch
) -> None:
    notifier = _notifier(tmp_path)
    item = _item()
    alert = _alert(item)
    monkeypatch.setattr("notifier.pipeline.send_alert", Mock(return_value=None))

    assert notifier._claim_item(item)
    assert notifier._complete_evaluation(item, alert) == 0

    assert notifier.seen.is_new(item)
    assert notifier.seen.is_semantically_new(
        item.player_name,
        alert.classification.event_type,
        alert.classification.severity,
        event_status(item, alert.classification.event_type),
    )
    assert len(notifier.outbox) == 1
    assert len(DeliveryOutbox(tmp_path)) == 1


def test_pending_raw_revision_survives_subject_attribution_upgrade(tmp_path) -> None:
    raw = replace(
        _item(guid="rotowire:beneficiary", source="rotowire"),
        player_name="Mike Washington",
        headline="Sees extra work after Jeanty injury",
        body="Ashton Jeanty left practice, creating extra work for Washington.",
    )
    corrected = replace(raw, player_name="Ashton Jeanty")
    outbox = DeliveryOutbox(tmp_path)
    outbox.add(_alert(raw, event_type="injury"))

    assert outbox.contains_item(corrected) is True


def test_pending_alert_replays_after_failure_then_finalizes_dedupe(
    tmp_path, monkeypatch
) -> None:
    notifier = _notifier(tmp_path)
    item = _item()
    alert = _alert(item)
    send = Mock(side_effect=[None, 77])
    monkeypatch.setattr("notifier.pipeline.send_alert", send)

    assert notifier._claim_item(item)
    assert notifier._complete_evaluation(item, alert) == 0
    # Recreate both stores to exercise the actual process-restart path.
    notifier.seen = SeenStore(tmp_path / "seen-items.json")
    notifier.outbox = DeliveryOutbox(tmp_path)
    pending = notifier.outbox.due(now=float("inf"))[0]

    with notifier._delivery_lock:
        assert notifier._attempt_pending_locked(pending) == 1

    assert send.call_count == 2
    assert len(notifier.outbox) == 0
    assert not notifier.seen.is_new(item)
    assert not notifier.seen.is_semantically_new(
        item.player_name,
        alert.classification.event_type,
        alert.classification.severity,
        event_status(item, alert.classification.event_type),
    )


def test_zero_attempt_restart_revalidates_beneficiary_before_replay(
    tmp_path, monkeypatch
) -> None:
    notifier = _notifier(tmp_path)
    item = _item("twitter:crash-before-send")
    league = LeagueRef("sleeper", "123", "Home", "Mine")
    plays = LeaguePlays(
        league=league,
        subject_state="mine",
        subject_owner="Mine",
        beneficiaries=[Beneficiary("Backup Player", "RB", 2, "free_agent")],
    )
    alert = replace(_alert(item), per_league=[plays])
    pending = notifier.outbox.add(alert, observed_at=10)
    assert pending.attempts == 0
    revalidated = replace(alert, delivery_delayed=True)
    notifier._revalidate_delayed_alert = Mock(return_value=revalidated)
    send = Mock(return_value=79)
    monkeypatch.setattr("notifier.pipeline.send_alert", send)

    assert notifier.deliver_pending() == 1

    notifier._revalidate_delayed_alert.assert_called_once_with(pending)
    assert send.call_args.args[2].delivery_delayed is True
    assert len(notifier.outbox) == 0


def test_removed_due_snapshot_entry_cannot_send_or_reschedule(
    tmp_path, monkeypatch
) -> None:
    notifier = _notifier(tmp_path)
    pending = notifier.outbox.add(_alert(_item("twitter:detached")))
    detached = notifier.outbox.due(float("inf"))[0]
    notifier.outbox.remove(pending.delivery_id)
    send = Mock(return_value=80)
    monkeypatch.setattr("notifier.pipeline.send_alert", send)

    assert notifier._attempt_pending_locked(detached, replay=True) == 0

    send.assert_not_called()
    assert len(notifier.outbox) == 0


def test_pending_literal_duplicate_is_blocked_but_same_guid_revision_escalates(
    tmp_path, monkeypatch
) -> None:
    notifier = _notifier(tmp_path)
    headline = "Example Player injury update"
    initial = replace(
        _item("twitter:shared-guid", headline=headline),
        body="Example Player is questionable with an ankle injury.",
    )
    escalated = replace(
        initial,
        body="Example Player has been ruled out with an ankle injury.",
    )
    send = Mock(side_effect=[None, 71])
    monkeypatch.setattr("notifier.pipeline.send_alert", send)

    assert notifier._claim_item(initial)
    assert notifier._complete_evaluation(initial, _alert(initial, 3, "injury")) == 0
    assert not notifier._claim_item(initial)

    assert notifier._claim_item(escalated)
    assert notifier._complete_evaluation(escalated, _alert(escalated, 5, "injury")) == 1

    assert send.call_count == 2
    assert send.call_args_list[1].args[2].item.body == escalated.body
    assert len(notifier.outbox) == 0


def test_outbox_delivery_ids_are_unambiguous_when_report_text_contains_pipes(
    tmp_path,
) -> None:
    outbox = DeliveryOutbox(tmp_path)
    first_item = replace(
        _item("twitter:pipe-collision", headline="Example|Player"),
        body="role update",
    )
    second_item = replace(
        first_item,
        headline="Example",
        body="Player|role update",
    )

    first = outbox.add(_alert(first_item, 3, "other"))
    second = outbox.add(_alert(second_item, 3, "other"))

    assert first.delivery_id != second.delivery_id
    assert len(outbox) == 2


@pytest.mark.parametrize(
    ("body", "severity", "remaining"),
    [
        ("Example Player has been ruled out with a concussion.", 5, 0),
        ("Example Player is questionable with a concussion.", 3, 1),
        (
            "Example Player is questionable with an ankle injury and is "
            "expected to miss 4 to 6 weeks.",
            3,
            0,
        ),
    ],
)
def test_same_headline_new_guid_reaches_status_condition_and_timetable_logic(
    tmp_path, monkeypatch, body, severity, remaining
) -> None:
    notifier = _notifier(tmp_path)
    headline = "Example Player injury update"
    initial = replace(
        _item("twitter:old", headline=headline),
        body="Example Player is questionable with an ankle injury.",
    )
    update = replace(
        _item("rotowire:new", source="rotowire", headline=headline),
        body=body,
    )
    send = Mock(side_effect=[None, 72])
    monkeypatch.setattr("notifier.pipeline.send_alert", send)

    assert notifier._claim_item(initial)
    assert notifier._complete_evaluation(initial, _alert(initial, 3, "injury")) == 0
    assert notifier._claim_item(update)
    assert notifier._complete_evaluation(update, _alert(update, severity, "injury")) == 1

    assert send.call_count == 2
    assert send.call_args_list[1].args[2].item.body == body
    assert len(notifier.outbox) == remaining


def test_newer_pending_return_retires_injury_even_with_zero_attempts(
    tmp_path, monkeypatch
) -> None:
    notifier = _notifier(tmp_path)
    injured_item = replace(
        _item("twitter:injury", headline="Example Player injury update"),
        body="Example Player was ruled out with an ankle injury.",
        published_at=None,
    )
    returned_item = replace(
        _item("twitter:return", headline="Example Player injury update"),
        body="Example Player was cleared and returned to practice.",
        published_at=None,
    )
    injured = notifier.outbox.add(
        _alert(injured_item, 4, "injury"),
        observed_at=10,
    )
    returned = notifier.outbox.add(
        _alert(returned_item, 4, "return"),
        observed_at=20,
    )
    send = Mock(return_value=81)
    monkeypatch.setattr("notifier.pipeline.send_alert", send)

    assert injured.attempts == 0
    assert notifier._attempt_pending_locked(injured) == 0

    send.assert_not_called()
    assert [entry.delivery_id for entry in notifier.outbox.due(float("inf"))] == [
        returned.delivery_id
    ]


def test_newer_pending_reinjury_retires_optimistic_return(
    tmp_path, monkeypatch
) -> None:
    notifier = _notifier(tmp_path)
    returned_item = replace(
        _item("twitter:return", headline="Example Player update"),
        body="Example Player was cleared and returned to practice.",
        published_at=None,
    )
    reinjured_item = replace(
        _item("twitter:reinjury", headline="Example Player update"),
        body="Example Player was ruled out again with an ankle injury.",
        published_at=None,
    )
    returned = notifier.outbox.add(
        _alert(returned_item, 3, "return"),
        observed_at=10,
    )
    reinjured = notifier.outbox.add(
        _alert(reinjured_item, 5, "injury"),
        observed_at=20,
    )
    send = Mock(return_value=82)
    monkeypatch.setattr("notifier.pipeline.send_alert", send)

    assert notifier._attempt_pending_locked(returned) == 0

    send.assert_not_called()
    assert [entry.delivery_id for entry in notifier.outbox.due(float("inf"))] == [
        reinjured.delivery_id
    ]


def test_newer_role_reversal_retires_stale_pending_starter(
    tmp_path, monkeypatch
) -> None:
    notifier = _notifier(tmp_path)
    starter_item = replace(
        _item(
            "twitter:starter",
            headline="Browns named Deshaun Watson their Week 1 starter",
        ),
        player_name="Deshaun Watson",
    )
    benched_item = replace(
        _item(
            "twitter:benched",
            headline="The Browns benched Deshaun Watson and named Sanders starter",
        ),
        player_name="Deshaun Watson",
    )
    starter = notifier.outbox.add(
        _alert(starter_item, 3, "depth_chart"),
        observed_at=10,
    )
    starter.attempts = 1
    benched = notifier.outbox.add(
        _alert(benched_item, 3, "depth_chart"),
        observed_at=20,
    )
    send = Mock(return_value=82)
    monkeypatch.setattr("notifier.pipeline.send_alert", send)

    assert notifier._attempt_pending_locked(benched) == 1

    send.assert_called_once()
    assert send.call_args.args[2].item == benched_item
    assert len(notifier.outbox) == 0


def test_expected_role_is_superseded_by_confirmed_role() -> None:
    expected = _alert(
        replace(
            _item(headline="Deshaun Watson is expected to start Week 1"),
            player_name="Deshaun Watson",
        ),
        3,
        "depth_chart",
    )
    confirmed = _alert(
        replace(
            _item(headline="The Browns named Deshaun Watson their Week 1 starter"),
            player_name="Deshaun Watson",
        ),
        3,
        "depth_chart",
    )

    assert _alert_supersedes(expected, confirmed) is True


def test_distinct_same_severity_usage_facts_remain_pending(tmp_path) -> None:
    notifier = _notifier(tmp_path)
    starter_item = replace(
        _item("twitter:starter", headline="Example Player role update"),
        body="Example Player will start this week.",
        published_at=None,
    )
    goal_line_item = replace(
        _item("twitter:goal-line", headline="Example Player role update"),
        body="Example Player will handle the goal-line work.",
        published_at=None,
    )
    starter = _alert(starter_item, 3, "usage")
    goal_line = _alert(goal_line_item, 3, "usage")
    older = notifier.outbox.add(starter, observed_at=10)
    newer = notifier.outbox.add(goal_line, observed_at=20)

    assert _alert_supersedes(starter, goal_line) is False
    assert notifier._pending_is_superseded(older) is False
    assert [entry.delivery_id for entry in notifier.outbox.due(float("inf"))] == [
        older.delivery_id,
        newer.delivery_id,
    ]


@pytest.mark.parametrize(
    ("old_body", "new_body", "old_severity", "new_severity"),
    [
        (
            "Example Player is questionable with an ankle injury.",
            "Example Player has been ruled out with an ankle injury.",
            3,
            5,
        ),
        (
            "Example Player was ruled out with an ankle injury.",
            "Example Player remains ruled out with an ankle injury.",
            4,
            4,
        ),
    ],
)
def test_older_retry_is_suppressed_after_newer_same_event_delivery(
    tmp_path, monkeypatch, old_body, new_body, old_severity, new_severity
) -> None:
    notifier = _notifier(tmp_path)
    old_time = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
    new_time = datetime(2026, 8, 23, 12, 5, tzinfo=timezone.utc)
    headline = "Example Player injury update"
    old_item = replace(
        _item("twitter:old", headline=headline),
        body=old_body,
        published_at=old_time,
    )
    new_item = replace(
        _item("rotowire:new", source="rotowire", headline=headline),
        body=new_body,
        published_at=new_time,
    )
    old = notifier.outbox.add(
        _alert(old_item, old_severity, "injury"),
        observed_at=old_time.timestamp(),
    )
    old.attempts = 1
    newer = _alert(new_item, new_severity, "injury")
    notifier.events = SimpleNamespace(
        recent_for_player=Mock(
            return_value=[_event_row(newer, received_at=new_time)]
        ),
        mark_outcome=Mock(),
    )
    send = Mock(return_value=91)
    monkeypatch.setattr("notifier.pipeline.send_alert", send)

    assert notifier._attempt_pending_locked(old) == 0

    send.assert_not_called()
    assert len(notifier.outbox) == 0


def test_later_distinct_condition_survives_older_delivered_condition(
    tmp_path, monkeypatch
) -> None:
    notifier = _notifier(tmp_path)
    old_time = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
    new_time = datetime(2026, 8, 23, 12, 5, tzinfo=timezone.utc)
    earlier_item = replace(
        _item("twitter:ankle", headline="Example Player injury update"),
        body="Example Player is questionable with an ankle injury.",
        published_at=old_time,
    )
    later_item = replace(
        _item("rotowire:concussion", source="rotowire", headline="Example Player injury update"),
        body="Example Player is questionable with a concussion.",
        published_at=new_time,
    )
    earlier = _alert(earlier_item, 3, "injury")
    later = _alert(later_item, 3, "injury")
    notifier.seen.record_semantic(
        earlier.item.player_name,
        earlier.classification.event_type,
        earlier.classification.severity,
        event_status(earlier.item, earlier.classification.event_type),
        event_fact_signature(earlier.item),
    )
    pending = notifier.outbox.add(later, observed_at=new_time.timestamp())
    pending.attempts = 1
    notifier.events = SimpleNamespace(
        recent_for_player=Mock(
            return_value=[_event_row(earlier, received_at=old_time)]
        ),
        mark_outcome=Mock(),
    )
    send = Mock(return_value=92)
    monkeypatch.setattr("notifier.pipeline.send_alert", send)

    assert notifier._attempt_pending_locked(pending) == 1

    send.assert_called_once()
    assert send.call_args.args[2].item.body == later_item.body
    assert len(notifier.outbox) == 0


def test_dry_run_never_mutates_seen_or_outbox(tmp_path, monkeypatch) -> None:
    notifier = _notifier(tmp_path, dry_run=True)
    item = _item()
    monkeypatch.setattr("notifier.pipeline.send_alert", Mock(return_value=-1))

    assert notifier._claim_item(item)
    assert notifier._complete_evaluation(item, _alert(item)) == 1

    assert notifier.seen.is_new(item)
    assert len(notifier.outbox) == 0
    assert not (tmp_path / "seen-items.json").exists()
    assert not (tmp_path / "pending-alerts.json").exists()


def test_outbox_round_trips_full_context(tmp_path) -> None:
    item = replace(_item(), subject_confident=False)
    refreshed = datetime(2026, 8, 23, 11, 30, tzinfo=timezone.utc)
    context = TeamContext(
        team="SF",
        subject_position="TE",
        same_position=[
            DepthEntry(
                name="Example Player",
                position="TE",
                depth_order=1,
                search_rank=91,
                is_subject=True,
                sleeper_injury_status="Questionable",
                sleeper_status="Active",
            )
        ],
        player_index_refreshed_at=refreshed,
    )
    league = LeagueRef("espn", "1", "Home", "Mine")
    plays = LeaguePlays(
        league=league,
        subject_state="mine",
        subject_owner="Mine",
        beneficiaries=[
            Beneficiary(
                "Backup",
                "RB",
                2,
                "free_agent",
                named_in_report=True,
                pro_team="ARI",
                fantasypros_waiver_rank=34,
                fantasypros_waiver_pos_rank="RB34",
                fantasypros_ros_rank=55,
                fantasypros_ros_pos_rank="RB55",
                fantasypros_scoring="PPR",
                fantasypros_updated_at="2026-08-23T18:00:00+00:00",
            )
        ],
        capacity=RosterCapacity(bench_used=5, bench_limit=5, ir_used=0, ir_limit=1),
        scoring_format="PPR",
    )
    alert = _alert(item)
    alert = Alert(
        item=alert.item,
        classification=alert.classification,
        tier=alert.tier,
        per_league=[plays],
        context=context,
        availability_refresh_failed=True,
        delivery_delayed=True,
    )

    store = DeliveryOutbox(tmp_path)
    store.add(alert)
    restored = DeliveryOutbox(tmp_path).due(now=float("inf"))[0].alert

    assert restored.item.published_at == item.published_at
    assert restored.context.player_index_refreshed_at == refreshed
    assert restored.context.same_position[0].sleeper_injury_status == "Questionable"
    assert restored.per_league[0].capacity == plays.capacity
    assert restored.per_league[0].beneficiaries[0].named_in_report is True
    assert restored.per_league[0].beneficiaries[0].pro_team == "ARI"
    assert restored.per_league[0].beneficiaries[0].fantasypros_waiver_rank == 34
    assert restored.per_league[0].beneficiaries[0].fantasypros_ros_pos_rank == "RB55"
    assert restored.per_league[0].scoring_format == "PPR"
    assert restored.availability_refresh_failed is True
    assert restored.delivery_delayed is True
    assert restored.item.subject_confident is False


def test_legacy_outbox_news_item_defaults_to_confident_subject() -> None:
    payload = {
        "source": "twitter",
        "guid": "old",
        "player_name": "Example Player",
        "headline": "Old alert",
        "body": "Old alert",
        "url": "",
        "published_at": None,
    }

    assert _news_item(payload).subject_confident is True


def test_semantic_dedupe_allows_severity_and_status_escalations(tmp_path) -> None:
    store = SeenStore(tmp_path / "seen.json")
    store.record_semantic("Example Player", "injury", 2, "limited")

    assert not store.is_semantically_new("Example Player", "injury", 2, "limited")
    assert store.is_semantically_new("Example Player", "injury", 3, "limited")

    store.record_semantic("Example Player", "injury", 4, "questionable")
    assert store.is_semantically_new("Example Player", "injury", 4, "inactive")
    assert not store.is_semantically_new("Example Player", "injury", 3, "limited")


def test_semantic_dedupe_treats_a_distinct_condition_as_a_new_fact(tmp_path) -> None:
    store = SeenStore(tmp_path / "seen.json")
    ankle = _item(headline="Example Player has an ankle injury")
    concussion = _item(
        guid="twitter:2:Example Player",
        headline="Example Player has a concussion",
    )
    store.record_semantic(
        "Example Player",
        "injury",
        4,
        "injury",
        event_fact_signature(ankle),
    )

    assert store.is_semantically_new(
        "Example Player",
        "injury",
        4,
        "injury",
        event_fact_signature(concussion),
    )


def test_semantic_dedupe_treats_a_new_injury_timetable_as_a_new_fact(tmp_path) -> None:
    store = SeenStore(tmp_path / "seen.json")
    initial = _item(headline="Example Player has an ankle injury")
    timetable = _item(
        guid="twitter:3:Example Player",
        headline="Example Player is expected to miss 4 to 6 weeks with an ankle injury",
    )
    store.record_semantic(
        "Example Player",
        "injury",
        4,
        "injury",
        event_fact_signature(initial),
    )

    assert store.is_semantically_new(
        "Example Player",
        "injury",
        4,
        "injury",
        event_fact_signature(timetable),
    )


def test_stable_injury_corroboration_crosses_other_and_injury_labels(tmp_path) -> None:
    store = SeenStore(tmp_path / "seen.json")
    first = replace(
        _item(
            guid="twitter:jeanty",
            headline=(
                "Raiders RB Ashton Jeanty, who had to be helped off the practice "
                "field today, is believed to have sprained his ankle"
            ),
        ),
        player_name="Ashton Jeanty",
        subject_confident=False,
    )
    corroboration = replace(
        _item(
            guid="rotowire:jeanty",
            source="rotowire",
            headline="Ashton Jeanty has an ankle sprain",
        ),
        player_name="Ashton Jeanty",
    )

    first_event = semantic_event_type(first, "other", "injury")
    assert first_event == "injury"
    store.record_semantic(
        first.player_name,
        first_event,
        4,
        semantic_event_status(first, first_event),
        semantic_event_fact_signature(first, first_event),
    )

    corroboration_event = semantic_event_type(corroboration, "injury")
    assert corroboration_event == "injury"
    assert not store.is_semantically_new(
        corroboration.player_name,
        corroboration_event,
        4,
        semantic_event_status(corroboration, corroboration_event),
        semantic_event_fact_signature(corroboration, corroboration_event),
    )


def test_pipeline_uses_original_medical_label_only_as_dedupe_hint(tmp_path) -> None:
    notifier = _notifier(tmp_path)
    first_item = replace(
        _item(
            guid="twitter:jeanty-ambiguous",
            headline=(
                "Raiders RB Ashton Jeanty, who had to be helped off the practice "
                "field, is believed to have sprained his ankle"
            ),
        ),
        player_name="Ashton Jeanty",
        subject_confident=False,
    )
    first = _alert(first_item, event_type="other")
    first = replace(
        first,
        classification=replace(
            first.classification,
            raw={"event_type": "injury", "subject_attribution": "uncertain"},
        ),
    )
    assert notifier._record_success(first)

    confirmation_item = replace(
        _item(
            guid="rotowire:jeanty-confirmation",
            source="rotowire",
            headline="Ashton Jeanty has an ankle sprain",
        ),
        player_name="Ashton Jeanty",
    )
    confirmation = _alert(confirmation_item, event_type="injury")

    assert not notifier._semantic_is_new(confirmation)


def test_restart_recovery_preserves_ambiguous_medical_dedupe_provenance(
    tmp_path, monkeypatch
) -> None:
    notifier = _notifier(tmp_path)
    item = replace(
        _item(
            guid="twitter:jeanty-accepted",
            headline=(
                "Raiders RB Ashton Jeanty, who had to be helped off the practice "
                "field, is believed to have sprained his ankle"
            ),
        ),
        player_name="Ashton Jeanty",
        subject_confident=False,
    )
    alert = replace(
        _alert(item, event_type="other"),
        classification=Classification(
            "other",
            4,
            "",
            False,
            {"event_type": "injury", "subject_attribution": "uncertain"},
        ),
    )
    pending = notifier.outbox.add(alert, observed_at=10)
    pending.attempts = 1
    notifier.events = SimpleNamespace(
        recent_for_player=Mock(
            return_value=[
                _event_row(
                    alert,
                    received_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
                )
            ]
        ),
        mark_outcome=Mock(),
    )
    send = Mock(return_value=99)
    monkeypatch.setattr("notifier.pipeline.send_alert", send)

    assert notifier._attempt_pending_locked(pending, replay=True) == 0

    send.assert_not_called()
    assert len(notifier.outbox) == 0
    confirmation_item = replace(
        _item(
            guid="rotowire:jeanty-after-restart",
            source="rotowire",
            headline="Ashton Jeanty has an ankle sprain",
        ),
        player_name="Ashton Jeanty",
    )
    assert not notifier._semantic_is_new(
        _alert(confirmation_item, event_type="injury")
    )


def test_six_hour_coalescing_is_checked_even_when_semantic_event_is_new(tmp_path) -> None:
    notifier = _notifier(tmp_path)
    alert = _alert(_item())
    notifier._semantic_is_new = Mock(return_value=True)
    notifier._can_coalesce = Mock(return_value=True)
    notifier._attempt_pending_locked = Mock(return_value=1)
    notifier._inflight_items[alert.item] = time.time()

    assert notifier._complete_evaluation(alert.item, alert) == 1

    notifier._can_coalesce.assert_called_once_with(alert)
    assert notifier._attempt_pending_locked.call_args.kwargs["force_semantic"] is True


def test_beneficiary_usage_is_not_recast_as_his_injury() -> None:
    report = replace(
        _item(
            headline=(
                "Mike Washington sees extra work after Ashton Jeanty left "
                "with a knee injury"
            )
        ),
        player_name="Mike Washington",
    )

    assert semantic_event_type(report, "other", "usage") == "other"


@pytest.mark.parametrize("hint", ["injury", "inactive", "practice_report"])
def test_uncertain_medical_display_label_keeps_its_model_family(hint: str) -> None:
    report = replace(
        _item(headline="Example Player has a sprained ankle"),
        subject_confident=False,
    )

    assert semantic_event_type(report, "other", hint) == hint


def test_usage_restriction_with_starter_wording_stays_a_distinct_event() -> None:
    restriction = replace(
        _item(
            headline=(
                "Deshaun Watson will start but play only one series before "
                "rotating every drive"
            )
        ),
        player_name="Deshaun Watson",
    )

    assert semantic_event_type(restriction, "usage") == "usage"


@pytest.mark.parametrize(
    ("headline", "expected_status"),
    [
        ("Deshaun Watson is not expected to start Week 1", "role_expected_not_starter"),
        ("Deshaun Watson isn't expected to start Week 1", "role_expected_not_starter"),
        ("Deshaun Watson is unlikely to start Week 1", "role_expected_not_starter"),
        ("Deshaun Watson may not start Week 1", "role_expected_not_starter"),
        ("Deshaun Watson won’t start Week 1", "role_not_starter"),
        ("Deshaun Watson was not named the starter", "role_not_starter"),
        ("The team did not name Deshaun Watson starter", "role_not_starter"),
        ("The team won’t name Deshaun Watson starter", "role_not_starter"),
        ("Deshaun Watson is expected not to start Week 1", "role_expected_not_starter"),
        ("Deshaun Watson not named starter", "role_not_starter"),
        ("Deshaun Watson not picked as QB1", "role_not_starter"),
        ("The Browns don't name Deshaun Watson starter", "role_not_starter"),
        ("The Browns do not name Deshaun Watson starter", "role_not_starter"),
        ("The Browns haven't named Deshaun Watson starter", "role_not_starter"),
        ("The Browns haven’t named Deshaun Watson starter", "role_not_starter"),
        ("Deshaun Watson doesn't get the start", "role_not_starter"),
        ("Deshaun Watson does not get the start", "role_not_starter"),
        ("Deshaun Watson won't get the start", "role_not_starter"),
        ("Deshaun Watson isn't starting Week 1", "role_not_starter"),
        ("Deshaun Watson isn’t starting Week 1", "role_not_starter"),
    ],
)
def test_negated_role_language_never_becomes_a_positive_starter_fact(
    headline: str,
    expected_status: str,
) -> None:
    report = replace(
        _item(headline=headline),
        player_name="Deshaun Watson",
    )

    assert role_decision_status(report) == expected_status
    assert semantic_event_fact_signature(report, "depth_chart") == (
        expected_status.replace("role_", "role:", 1)
    )


def test_projected_nonstarter_to_confirmed_benching_is_a_new_role_fact(tmp_path) -> None:
    store = SeenStore(tmp_path / "seen.json")
    projected = replace(
        _item(headline="Deshaun Watson may not start Week 1"),
        player_name="Deshaun Watson",
    )
    event_type = semantic_event_type(projected, "depth_chart")
    store.record_semantic(
        projected.player_name,
        event_type,
        3,
        semantic_event_status(projected, event_type),
        semantic_event_fact_signature(projected, event_type),
    )
    confirmed = replace(
        _item(headline="Deshaun Watson won’t start Week 1"),
        player_name="Deshaun Watson",
    )

    assert store.is_semantically_new(
        confirmed.player_name,
        semantic_event_type(confirmed, "depth_chart"),
        3,
        semantic_event_status(confirmed, "depth_chart"),
        semantic_event_fact_signature(confirmed, "depth_chart"),
    )


@pytest.mark.parametrize(
    "headline",
    [
        "Watson named starter",
        "Watson named QB1",
        "Browns name Watson QB1",
        "Browns select Watson to start",
        "Browns pick Watson to start",
        "Watson gets Week 1 start",
        "Watson starting Week 1",
        (
            "Sources: The Browns are naming Deshaun Watson as their starting "
            "QB for Week 1"
        ),
    ],
)
def test_common_confirmed_role_headlines_canonicalize_as_starter(
    headline: str,
) -> None:
    report = replace(
        _item(
            headline=headline,
            source="rotowire" if "Deshaun Watson" not in headline else "twitter",
        ),
        player_name="Deshaun Watson",
    )

    assert role_decision_status(report) == "role_starter"


@pytest.mark.parametrize(
    "headline",
    [
        (
            "The Browns are expected to name a starting quarterback between "
            "Deshaun Watson and Shedeur Sanders today"
        ),
        (
            "The starting quarterback battle between Deshaun Watson and "
            "Shedeur Sanders remains open"
        ),
    ],
)
def test_predecision_competition_language_is_never_a_confirmed_starter(
    headline: str,
) -> None:
    report = replace(
        _item(headline=headline),
        player_name="Deshaun Watson",
    )

    assert role_decision_status(report) in {"", "role_uncertain"}
    assert role_decision_status(report) != "role_starter"


@pytest.mark.parametrize(
    "headline",
    [
        "If Deshaun Watson starts Week 1, the Browns will lean on experience",
        "Whether Deshaun Watson starts Week 1 remains unclear",
        "It is unclear whether Deshaun Watson starts Week 1",
        "Deshaun Watson starts rehab Monday",
        "Deshaun Watson is starting rehab Monday",
        "Deshaun Watson will start rehab Monday",
        "Deshaun Watson starts the season on PUP",
        "Deshaun Watson will start the season on IR",
        "Deshaun Watson is starting camp on PUP",
        "Deshaun Watson is starting to throw again",
    ],
)
def test_non_role_start_language_never_becomes_a_starter_fact(headline: str) -> None:
    report = replace(
        _item(headline=headline),
        player_name="Deshaun Watson",
    )

    assert role_decision_status(report) != "role_starter"


@pytest.mark.parametrize(
    "headline",
    [
        "Deshaun Watson is no longer the starter",
        "Deshaun Watson is no longer QB1",
        "Deshaun Watson lost the starting job",
        "Deshaun Watson loses his starting role",
    ],
)
def test_common_role_reversal_headlines_are_not_starter_facts(headline: str) -> None:
    report = replace(
        _item(headline=headline),
        player_name="Deshaun Watson",
    )

    assert role_decision_status(report) == "role_not_starter"


def test_role_reversal_and_injury_timetable_remain_new_semantic_facts(tmp_path) -> None:
    store = SeenStore(tmp_path / "seen.json")
    starter = replace(
        _item(headline="Browns named Deshaun Watson their Week 1 starter"),
        player_name="Deshaun Watson",
    )
    role_event = semantic_event_type(starter, "other")
    store.record_semantic(
        starter.player_name,
        role_event,
        3,
        semantic_event_status(starter, role_event),
        semantic_event_fact_signature(starter, role_event),
    )
    benched = replace(
        _item(
            guid="twitter:watson-benched",
            headline="The Browns benched Deshaun Watson and named Sanders starter",
        ),
        player_name="Deshaun Watson",
    )
    assert store.is_semantically_new(
        benched.player_name,
        semantic_event_type(benched, "depth_chart"),
        3,
        semantic_event_status(benched, "depth_chart"),
        semantic_event_fact_signature(benched, "depth_chart"),
    )

    injury = replace(
        _item(
            guid="twitter:jeanty-injury",
            headline="Ashton Jeanty has a sprained ankle",
        ),
        player_name="Ashton Jeanty",
    )
    injury_event = semantic_event_type(injury, "injury")
    store.record_semantic(
        injury.player_name,
        injury_event,
        4,
        semantic_event_status(injury, injury_event),
        semantic_event_fact_signature(injury, injury_event),
    )
    timetable = replace(
        injury,
        guid="twitter:jeanty-timetable",
        headline="Ashton Jeanty will miss 4 to 6 weeks with a sprained ankle",
        body="Ashton Jeanty will miss 4 to 6 weeks with a sprained ankle",
    )
    assert store.is_semantically_new(
        timetable.player_name,
        semantic_event_type(timetable, "other"),
        4,
        semantic_event_status(timetable, "other"),
        semantic_event_fact_signature(timetable, "other"),
    )


def test_role_severity_model_drift_is_corroboration_but_injury_escalation_is_new(
    tmp_path,
) -> None:
    store = SeenStore(tmp_path / "seen.json")
    starter = replace(
        _item(headline="Browns named Deshaun Watson their Week 1 starter"),
        player_name="Deshaun Watson",
    )
    role_event = semantic_event_type(starter, "depth_chart")
    role_status = semantic_event_status(starter, role_event)
    role_facts = semantic_event_fact_signature(starter, role_event)
    store.record_semantic(starter.player_name, role_event, 3, role_status, role_facts)

    assert not store.is_semantically_new(
        starter.player_name,
        role_event,
        4,
        role_status,
        role_facts,
    )

    injury = replace(
        _item(headline="Ashton Jeanty has a sprained ankle"),
        player_name="Ashton Jeanty",
    )
    injury_event = semantic_event_type(injury, "injury")
    injury_status = semantic_event_status(injury, injury_event)
    injury_facts = semantic_event_fact_signature(injury, injury_event)
    store.record_semantic(
        injury.player_name,
        injury_event,
        3,
        injury_status,
        injury_facts,
    )
    assert store.is_semantically_new(
        injury.player_name,
        injury_event,
        4,
        injury_status,
        injury_facts,
    )


def test_concrete_injury_fact_lasts_24_hours_but_generic_usage_does_not(
    tmp_path, monkeypatch
) -> None:
    clock = [1_000_000.0]
    monkeypatch.setattr("notifier.dedupe.time.time", lambda: clock[0])
    store = SeenStore(tmp_path / "seen.json")
    injury = replace(
        _item(headline="Ashton Jeanty has a sprained ankle"),
        player_name="Ashton Jeanty",
    )
    event_type = semantic_event_type(injury, "injury")
    status = semantic_event_status(injury, event_type)
    facts = semantic_event_fact_signature(injury, event_type)
    store.record_semantic(injury.player_name, event_type, 4, status, facts)

    clock[0] += 16 * 60 * 60
    assert not store.is_semantically_new(
        injury.player_name,
        event_type,
        4,
        status,
        facts,
    )
    clock[0] += 8 * 60 * 60 + 1
    assert store.is_semantically_new(
        injury.player_name,
        event_type,
        4,
        status,
        facts,
    )

    usage = replace(
        _item(headline="Example Player handled first-team reps"),
        player_name="Example Player",
    )
    store.record_semantic("Example Player", "usage", 3, "usage", "unspecified")
    clock[0] += 91 * 60
    assert store.is_semantically_new(
        usage.player_name,
        "usage",
        3,
        "usage",
        "unspecified",
    )


def test_semantic_dedupe_accepts_true_ruled_out_corroboration(tmp_path) -> None:
    store = SeenStore(tmp_path / "seen.json")
    first = _item(headline="Example Player was ruled out")
    confirmation = _item(
        guid="rotowire:2",
        source="rotowire",
        headline="Team confirms Example Player will not play",
    )
    store.record_semantic(
        "Example Player",
        "injury",
        4,
        event_status(first, "injury"),
        event_fact_signature(first),
    )

    assert not store.is_semantically_new(
        "Example Player",
        "injury",
        4,
        event_status(confirmation, "injury"),
        event_fact_signature(confirmation),
    )


def test_lower_severity_corroboration_does_not_forget_prior_urgency(tmp_path) -> None:
    store = SeenStore(tmp_path / "seen.json")
    report = _item(headline="Example Player was ruled out")
    status = event_status(report, "injury")
    signature = event_fact_signature(report)
    store.record_semantic("Example Player", "injury", 4, status, signature)
    store.record_semantic("Example Player", "injury", 3, status, signature)

    assert not store.is_semantically_new(
        "Example Player",
        "injury",
        4,
        status,
        signature,
    )


def test_early_dedupe_allows_body_only_status_escalation_for_same_guid(tmp_path) -> None:
    store = SeenStore(tmp_path / "seen.json")
    initial = replace(
        _item(headline="Example Player injury update"),
        body="Example Player is questionable.",
    )
    escalated = replace(initial, body="Example Player has been ruled out.")
    store.record(initial)
    assert store.save()

    reloaded = SeenStore(tmp_path / "seen.json")
    assert reloaded.is_new(escalated) is True


def test_early_dedupe_allows_body_only_condition_change_for_same_guid(tmp_path) -> None:
    store = SeenStore(tmp_path / "seen.json")
    initial = replace(
        _item(headline="Example Player injury update"),
        body="Example Player has an ankle injury.",
    )
    new_condition = replace(initial, body="Example Player has a concussion.")
    store.record(initial)
    assert store.save()

    reloaded = SeenStore(tmp_path / "seen.json")
    assert reloaded.is_new(new_condition) is True


def test_early_dedupe_rejects_exact_raw_revision_after_save_reload(tmp_path) -> None:
    store = SeenStore(tmp_path / "seen.json")
    report = replace(
        _item(headline="Example Player role update"),
        body="Example Player remains in a committee.",
    )
    store.record(report)
    assert store.save()

    reloaded = SeenStore(tmp_path / "seen.json")
    assert reloaded.is_new(report) is False


def test_early_dedupe_allows_generic_body_revision_for_same_guid(tmp_path) -> None:
    store = SeenStore(tmp_path / "seen.json")
    initial = replace(
        _item(headline="Example Player role update"),
        body="Example Player remains in a committee.",
    )
    changed = replace(
        initial,
        body="Example Player will start and handle the goal-line work.",
    )
    store.record(initial)
    assert store.save()

    reloaded = SeenStore(tmp_path / "seen.json")
    assert reloaded.is_new(changed) is True


def test_legacy_seen_upgrade_suppresses_exact_report_and_allows_escalation(
    tmp_path,
) -> None:
    path = tmp_path / "seen.json"
    store = SeenStore(path)
    initial = replace(
        _item(headline="Example Player injury update"),
        body="Example Player is questionable with an ankle injury.",
    )
    escalated = replace(
        initial,
        body="Example Player has been ruled out with an ankle injury.",
    )
    store.record(initial)
    assert store.save()

    # Reproduce the production schema from before raw revision and condition
    # signature tracking were added. Status maps already existed in that
    # schema and remain sufficient to prove a real availability escalation.
    payload = json.loads(path.read_text())
    for field in (
        "guidFactSignatures",
        "fingerprintFactSignatures",
        "reportRevisions",
        "revisionAwareGuids",
        "revisionAwareFingerprints",
    ):
        payload.pop(field)
    path.write_text(json.dumps(payload))

    upgraded = SeenStore(path)
    assert upgraded.is_new(initial) is False
    assert upgraded.is_new(escalated) is True


def test_legacy_seen_without_status_metadata_does_not_replay_exact_report(
    tmp_path,
) -> None:
    path = tmp_path / "seen.json"
    report = replace(
        _item(headline="Example Player injury update"),
        body="Example Player has been ruled out with an ankle injury.",
    )
    current = SeenStore(path)
    current.record(report)
    assert current.save()
    payload = json.loads(path.read_text())
    path.write_text(
        json.dumps(
            {
                "guids": payload["guids"],
                "fingerprints": payload["fingerprints"],
                "semantic": {},
            }
        )
    )

    assert SeenStore(path).is_new(report) is False


def test_x_dispatcher_consumes_without_waiting_for_rss_poll(tmp_path) -> None:
    notifier = _notifier(tmp_path)
    notifier._tweet_queue = queue.Queue()
    notifier._stop = threading.Event()
    notifier._twitter_dispatcher = None
    consumed = threading.Event()
    notifier._submit_tweet = Mock(side_effect=lambda _item: consumed.set())

    notifier._start_twitter_dispatcher()
    notifier._tweet_queue.put(_item())

    assert consumed.wait(1.0)
    notifier._stop.set()
    notifier._twitter_dispatcher.join(timeout=1.0)
    notifier._submit_tweet.assert_called_once()


def test_source_priority_lets_x_claim_semantic_slot(tmp_path, monkeypatch) -> None:
    notifier = _notifier(tmp_path)
    twitter = _item(headline="Example Player was ruled out by the team")
    rotowire = _item(
        guid="rotowire:2",
        source="rotowire",
        headline="Team confirms Example Player will not play",
    )
    notifier._evaluate = lambda item: _alert(item)
    delivered_sources = []
    monkeypatch.setattr(
        "notifier.pipeline.send_alert",
        lambda _session, _config, alert: delivered_sources.append(alert.item.source) or 1,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        sent = notifier._process_items([rotowire, twitter], pool)

    assert sent == 1
    assert delivered_sources == ["twitter"]


def test_semantic_corroboration_reaches_existing_message_edit_path(
    tmp_path, monkeypatch
) -> None:
    notifier = _notifier(tmp_path)
    notifier.telegram_state = SimpleNamespace(
        coalescing_target=Mock(return_value=object())
    )
    earlier = _alert(_item("twitter:first"))
    notifier.seen.record_semantic(
        earlier.item.player_name,
        earlier.classification.event_type,
        earlier.classification.severity,
        event_status(earlier.item, earlier.classification.event_type),
        event_fact_signature(earlier.item),
    )
    corroboration = _alert(
        _item(
            "rotowire:corroboration",
            source="rotowire",
            headline="Team confirms Example Player was ruled out",
        )
    )
    send = Mock(return_value=77)
    monkeypatch.setattr("notifier.pipeline.send_alert", send)

    assert notifier._claim_item(corroboration.item)
    assert notifier._complete_evaluation(corroboration.item, corroboration) == 1

    send.assert_called_once()
    notifier.telegram_state.coalescing_target.assert_called()
    assert len(notifier.outbox) == 0
    assert not notifier.seen.is_new(corroboration.item)


def test_failed_coalesced_edit_remains_retryable(tmp_path, monkeypatch) -> None:
    notifier = _notifier(tmp_path)
    notifier.telegram_state = SimpleNamespace(
        coalescing_target=Mock(return_value=object())
    )
    earlier = _alert(_item("twitter:first"))
    notifier.seen.record_semantic(
        earlier.item.player_name,
        earlier.classification.event_type,
        earlier.classification.severity,
        event_status(earlier.item, earlier.classification.event_type),
        event_fact_signature(earlier.item),
    )
    corroboration = _alert(
        _item(
            "rotowire:retry-edit",
            source="rotowire",
            headline="Team again confirms Example Player was ruled out",
        )
    )
    send = Mock(side_effect=[None, 88])
    monkeypatch.setattr("notifier.pipeline.send_alert", send)

    assert notifier._claim_item(corroboration.item)
    assert notifier._complete_evaluation(corroboration.item, corroboration) == 0
    pending = notifier.outbox.due(now=float("inf"))[0]
    assert notifier._attempt_pending_locked(pending) == 1

    assert send.call_count == 2
    assert len(notifier.outbox) == 0


def test_recent_successful_jit_roster_refresh_is_reused(monkeypatch) -> None:
    notifier = Notifier.__new__(Notifier)
    notifier.config = SimpleNamespace()
    notifier._jit_roster_lock = threading.Lock()
    notifier._state_lock = threading.RLock()
    notifier._last_jit_roster_refresh = 0.0
    notifier._last_jit_roster_success = 0.0
    notifier._snapshot_mtime = 0.0
    notifier._player_index = {}
    notifier.snapshot = RosterSnapshot(generated_at=datetime.now(timezone.utc))
    fresh = RosterSnapshot(generated_at=datetime.now(timezone.utc))
    refresh = Mock(return_value=(fresh, 123))
    monkeypatch.setattr("notifier.pipeline.refresh_drafted_snapshot", refresh)
    current_mtime = Mock(return_value=456)
    monkeypatch.setattr("notifier.pipeline.snapshot_mtime", current_mtime)

    assert notifier._refresh_ownership_just_in_time() is True
    assert notifier._refresh_ownership_just_in_time() is True
    assert refresh.call_count == 1
    assert notifier.snapshot is fresh
    assert notifier._snapshot_mtime == 123
    # A full refresh written after the helper returned stays newer than the
    # exact JIT version, so the next normal reload can still observe it.
    current_mtime.assert_not_called()


def test_failed_jit_refresh_fails_closed_without_add_or_free_agent_claim(
    monkeypatch,
) -> None:
    notifier = Notifier.__new__(Notifier)
    league = LeagueRef("sleeper", "1234", "Home League", "Mine")
    # The scheduled snapshot says the backup is taken. We must still refresh:
    # the rival may have dropped him since this snapshot was written.
    candidate = Beneficiary("Backup Player", "RB", 2, "rostered", "Rival")
    plays = LeaguePlays(
        league=league,
        subject_state="mine",
        subject_owner="Mine",
        beneficiaries=[candidate],
    )
    depth = Mock()
    depth.build.return_value = ({"full_name": "Example Player"}, [plays])
    depth.team_context.return_value = None
    notifier._state_lock = threading.RLock()
    notifier.snapshot = RosterSnapshot(
        generated_at=datetime.now(timezone.utc),
        leagues=[league],
    )
    notifier.depth = depth
    notifier.preseason = False
    notifier.session = object()
    notifier.config = SimpleNamespace(
        dry_run=False,
        min_severity=2,
        min_severity_other=3,
    )
    notifier._refresh_ownership_just_in_time = Mock(return_value=False)
    monkeypatch.setattr(
        "notifier.pipeline.classify",
        Mock(
            return_value=Classification(
                "injury",
                3,
                "Add the backup.",
                True,
                {"direction": "negative"},
            )
        ),
    )

    alert = notifier._evaluate(_item())

    assert alert is not None
    notifier._refresh_ownership_just_in_time.assert_called_once_with()
    assert alert.availability_refresh_failed is True
    assert all(not entry.claimable for entry in alert.per_league)
    rendered = format_alert(alert)
    assert "no ADD or free-agent recommendation" in rendered
    assert "Backup Player" not in rendered


def test_ambiguous_multi_player_report_never_triggers_roster_moves(monkeypatch) -> None:
    notifier = Notifier.__new__(Notifier)
    league = LeagueRef("sleeper", "1234", "Home League", "Mine")
    plays = LeaguePlays(
        league=league,
        subject_state="mine",
        subject_owner="Mine",
        beneficiaries=[Beneficiary("Backup Player", "RB", 2, "free_agent")],
        bench_options=["Bench Player"],
    )
    depth = Mock()
    depth.build.return_value = ({"full_name": "Example Player"}, [plays])
    depth.team_context.return_value = None
    notifier._state_lock = threading.RLock()
    notifier.snapshot = RosterSnapshot(
        generated_at=datetime.now(timezone.utc),
        leagues=[league],
    )
    notifier.depth = depth
    notifier.preseason = False
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
                "injury",
                4,
                "Backup Player will take over.",
                True,
                {"direction": "negative"},
            )
        ),
    )
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: ["Jordan Mason"])
    item = stream._to_items(
        {
            "data": {
                "id": "ambiguous-subject",
                "author_id": "7",
                "text": "Jordan Mason will start with CMC ruled out.",
            },
            "includes": {"users": [{"id": "7", "username": "Reporter"}]},
        }
    )[0]
    stream._session.close()
    assert item.subject_confident is False

    alert = notifier._evaluate(item)

    assert alert is not None
    notifier._refresh_ownership_just_in_time.assert_not_called()
    assert alert.classification.event_type == "other"
    assert alert.classification.raw["subject_attribution"] == "uncertain"
    assert all(not plays.has_action for plays in alert.per_league)
    rendered = format_alert(alert)
    assert "automatic pickup and lineup moves are withheld" in rendered
    assert "<b>UPDATE</b>" in rendered
    assert "<b>INJURY</b>" not in rendered
    assert "Backup Player will take over" not in rendered
    assert "Pickup option" not in rendered
    assert "Start instead" not in rendered


def test_ambiguous_league_commentary_requires_severity_four() -> None:
    notifier = Notifier.__new__(Notifier)
    notifier.config = SimpleNamespace(
        min_severity=2,
        min_severity_other=3,
    )

    assert notifier._threshold_for("league", subject_confident=True) == 3
    assert notifier._threshold_for("league", subject_confident=False) == 4


def test_non_transaction_release_fails_closed_through_fallback_pipeline(monkeypatch) -> None:
    notifier = Notifier.__new__(Notifier)
    league = LeagueRef("sleeper", "1234", "Home League", "Mine")
    plays = LeaguePlays(
        league=league,
        subject_state="mine",
        subject_owner="Mine",
        beneficiaries=[Beneficiary("Backup Player", "RB", 2, "free_agent")],
        bench_options=["Bench Player"],
    )
    depth = Mock()
    depth.build.return_value = ({"full_name": "Jordan Mason"}, [plays])
    depth.team_context.return_value = None
    notifier._state_lock = threading.RLock()
    notifier.snapshot = RosterSnapshot(
        generated_at=datetime.now(timezone.utc),
        leagues=[league],
    )
    notifier.depth = depth
    notifier.preseason = False
    notifier.session = object()
    notifier.config = SimpleNamespace(
        dry_run=False,
        min_severity=2,
        min_severity_other=3,
    )
    notifier._refresh_ownership_just_in_time = Mock(return_value=True)
    monkeypatch.setattr(
        "notifier.pipeline.classify",
        lambda _session, _config, news_item, **_kwargs: _fallback(
            "test outage", news_item
        ),
    )
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: ["Jordan Mason"])
    item = stream._to_items(
        {
            "data": {
                "id": "transitive-release",
                "author_id": "7",
                "text": "The 49ers released Jordan Mason injury update.",
            },
            "includes": {"users": [{"id": "7", "username": "Reporter"}]},
        }
    )[0]
    stream._session.close()

    assert item.subject_confident is False
    assert _fallback("test outage", item).event_type == "release"
    alert = notifier._evaluate(item)

    assert alert is not None
    assert alert.classification.event_type == "other"
    assert alert.classification.raw["subject_attribution"] == "uncertain"
    assert all(not league_plays.has_action for league_plays in alert.per_league)
    rendered = format_alert(alert)
    assert "automatic pickup and lineup moves are withheld" in rendered
    assert "Pickup option" not in rendered
    assert "Start instead" not in rendered


def test_outbox_retries_preserve_chronological_order_across_sources(tmp_path) -> None:
    outbox = DeliveryOutbox(tmp_path)
    older = outbox.add(_alert(_item("rotowire:old", source="rotowire")))
    newer = outbox.add(_alert(_item("twitter:new", source="twitter")))
    older.queued_at = 10
    newer.queued_at = 20

    assert [entry.alert.item.guid for entry in outbox.due(now=float("inf"))] == [
        "rotowire:old",
        "twitter:new",
    ]


def test_outbox_orders_distinct_facts_by_observation_not_classification_finish(
    tmp_path,
) -> None:
    outbox = DeliveryOutbox(tmp_path)
    # The later report classified quickly and was enqueued first. The earlier
    # report finished slowly, but must still be delivered first on recovery.
    later_item = replace(
        _item("twitter:later", headline="Example Player injury update"),
        body="Example Player has a concussion.",
        published_at=None,
    )
    earlier_item = replace(
        _item("rotowire:earlier", source="rotowire", headline="Example Player injury update"),
        body="Example Player has an ankle injury.",
        published_at=None,
    )
    outbox.add(_alert(later_item, event_type="injury"), observed_at=20)
    outbox.add(_alert(earlier_item, event_type="injury"), observed_at=10)

    assert [entry.alert.item.guid for entry in outbox.due(float("inf"))] == [
        "rotowire:earlier",
        "twitter:later",
    ]
