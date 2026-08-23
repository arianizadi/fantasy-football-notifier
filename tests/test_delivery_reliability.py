import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from notifier.dedupe import SeenStore, event_status
from notifier.models import Alert, Classification, LeagueRef, NewsItem, RosterSnapshot
from notifier.outbox import DeliveryOutbox
from notifier.pipeline import Notifier
from notifier.notify import format_alert
from notifier.plays import Beneficiary, DepthEntry, LeaguePlays, TeamContext


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
    notifier._inflight_guids = set()
    notifier._inflight_fingerprints = set()
    return notifier


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
    item = _item()
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
    alert = _alert(item)
    alert = Alert(
        item=alert.item,
        classification=alert.classification,
        tier=alert.tier,
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
    assert restored.availability_refresh_failed is True
    assert restored.delivery_delayed is True


def test_semantic_dedupe_allows_severity_and_status_escalations(tmp_path) -> None:
    store = SeenStore(tmp_path / "seen.json")
    store.record_semantic("Example Player", "injury", 2, "limited")

    assert not store.is_semantically_new("Example Player", "injury", 2, "limited")
    assert store.is_semantically_new("Example Player", "injury", 3, "limited")

    store.record_semantic("Example Player", "injury", 4, "questionable")
    assert store.is_semantically_new("Example Player", "injury", 4, "inactive")
    assert not store.is_semantically_new("Example Player", "injury", 3, "limited")


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


def test_recent_successful_jit_roster_refresh_is_reused(monkeypatch) -> None:
    notifier = Notifier.__new__(Notifier)
    notifier.config = SimpleNamespace()
    notifier._jit_roster_lock = threading.Lock()
    notifier._state_lock = threading.RLock()
    notifier._last_jit_roster_refresh = 0.0
    notifier._last_jit_roster_success = 0.0
    notifier._snapshot_mtime = 0.0
    notifier._player_index = {}
    fresh = RosterSnapshot(generated_at=datetime.now(timezone.utc))
    refresh = Mock(return_value=fresh)
    monkeypatch.setattr("notifier.pipeline.refresh_snapshot", refresh)
    monkeypatch.setattr("notifier.pipeline.snapshot_mtime", Mock(return_value=123.0))

    assert notifier._refresh_ownership_just_in_time() is True
    assert notifier._refresh_ownership_just_in_time() is True
    assert refresh.call_count == 1
    assert notifier.snapshot is fresh
    assert notifier._snapshot_mtime == 123.0


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
