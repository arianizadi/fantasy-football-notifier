import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import requests

import notifier.pipeline as pipeline_module
from notifier.dedupe import SeenStore
from notifier.models import NewsItem, report_revision_identity
from notifier.pipeline import Notifier, _depth_report_text


def test_tweets_are_processed_when_rotowire_is_unavailable() -> None:
    tweet = NewsItem(
        source="twitter",
        guid="twitter:1:Example Player",
        player_name="Example Player",
        headline="Example Player was ruled out",
        body="Example Player was ruled out",
        url="https://x.com/example/status/1",
        published_at=None,
    )
    tweet_queue = queue.Queue()
    tweet_queue.put(tweet)
    seen = SimpleNamespace(is_new=Mock(return_value=True))
    process = Mock(return_value=0)
    notifier = SimpleNamespace(
        _reload_roster_if_changed=Mock(),
        deliver_pending=Mock(return_value=0),
        poller=SimpleNamespace(fetch=Mock(side_effect=requests.RequestException("offline"))),
        session=object(),
        _tweet_queue=tweet_queue,
        seen=seen,
        _pool=object(),
        _process_items=process,
    )

    assert Notifier.poll_once(notifier) == 0
    process.assert_called_once_with([tweet], notifier._pool)
    assert tweet_queue.empty()


def test_preseason_restart_does_not_send_a_chat_notice(monkeypatch) -> None:
    stop = threading.Event()
    stop.set()
    send_plain = Mock()
    monkeypatch.setattr(pipeline_module, "send_plain", send_plain)
    notifier = SimpleNamespace(
        snapshot=SimpleNamespace(leagues=[], mine=Mock(return_value=set())),
        config=SimpleNamespace(
            poll_seconds=15,
            poll_seconds_idle=60,
            openrouter_model="test-model",
            telegram_controls_enabled=False,
            daily_digest_enabled=False,
            dry_run=False,
        ),
        fantasypros=SimpleNamespace(enabled=False),
        _start_fantasypros_refresher=Mock(),
        twitter=None,
        preseason=True,
        check_roster_freshness=Mock(),
        _state_lock=threading.RLock(),
        _player_index={},
        _stop=stop,
    )

    Notifier.run_forever(notifier)

    send_plain.assert_not_called()


def test_pipeline_normalizes_rotowire_subject_before_claiming() -> None:
    item = NewsItem(
        source="rotowire",
        guid="rotowire:washington-beneficiary",
        player_name="Mike Washington",
        headline="Sees extra work after Jeanty injury",
        body=(
            "Washington took most of the carries after Ashton Jeanty (knee) "
            "left Sunday's practice."
        ),
        url="https://www.rotowire.com/football/player/mike-washington-999",
        published_at=None,
    )
    notifier = SimpleNamespace(
        _state_lock=threading.RLock(),
        _player_index={
            "1": {
                "full_name": "Ashton Jeanty",
                "position": "RB",
                "team": "LV",
                "depth_chart_order": 1,
                "status": "Active",
            },
            "2": {
                "full_name": "Mike Washington",
                "position": "RB",
                "team": "LV",
                "depth_chart_order": 2,
                "status": "Active",
            },
        },
    )

    normalized = Notifier._normalize_source_subject(notifier, item)

    assert normalized.player_name == "Ashton Jeanty"
    assert normalized.headline == item.headline
    assert normalized.body == item.body
    assert report_revision_identity(normalized) == report_revision_identity(item)
    assert _depth_report_text(normalized).startswith("mike washington ")


def test_subject_normalization_does_not_replay_revision_aware_report(tmp_path) -> None:
    raw = NewsItem(
        source="rotowire",
        guid="rotowire:washington-beneficiary",
        player_name="Mike Washington",
        headline="Sees extra work after Jeanty injury",
        body=(
            "Washington took most of the carries after Ashton Jeanty (knee) "
            "left Sunday's practice."
        ),
        url="https://www.rotowire.com/football/player/mike-washington-999",
        published_at=None,
    )
    normalized = NewsItem(
        source=raw.source,
        guid=raw.guid,
        player_name="Ashton Jeanty",
        headline=raw.headline,
        body=raw.body,
        url=raw.url,
        published_at=None,
    )
    seen = SeenStore(tmp_path / "seen.json")
    seen.record(raw)

    assert seen.is_new(normalized) is False


def test_reattribution_audit_logs_only_after_one_successful_claim(monkeypatch) -> None:
    item = NewsItem(
        source="rotowire",
        guid="rotowire:washington-beneficiary-new",
        player_name="Ashton Jeanty",
        headline="Sees extra work after Jeanty injury",
        body="Ashton Jeanty left practice.",
        url="https://www.rotowire.com/football/player/mike-washington-999",
        published_at=None,
    )
    log = Mock()
    monkeypatch.setattr(pipeline_module, "structured_log", log)
    notifier = SimpleNamespace(
        _state_lock=threading.RLock(),
        seen=SimpleNamespace(is_new=Mock(return_value=True)),
        outbox=SimpleNamespace(contains_item=Mock(return_value=False)),
        _inflight_items={},
        _journal_received=Mock(),
    )

    assert Notifier._claim_item(notifier, item) is True
    assert Notifier._claim_item(notifier, item) is False

    reattribution_logs = [
        call
        for call in log.call_args_list
        if len(call.args) > 1 and call.args[1] == "rotowire.subject_reattributed"
    ]
    assert len(reattribution_logs) == 1


def test_poll_checks_normalized_rotowire_item_against_seen_state() -> None:
    raw = NewsItem(
        source="rotowire",
        guid="rotowire:washington-beneficiary",
        player_name="Mike Washington",
        headline="Sees extra work after Jeanty injury",
        body="Ashton Jeanty left practice.",
        url="https://www.rotowire.com/football/player/mike-washington-999",
        published_at=None,
    )
    normalized = NewsItem(
        source=raw.source,
        guid=raw.guid,
        player_name="Ashton Jeanty",
        headline="Sees extra work after Jeanty injury",
        body=raw.body,
        url=raw.url,
        published_at=None,
    )
    seen = SimpleNamespace(is_new=Mock(return_value=True))
    process = Mock(return_value=0)
    notifier = SimpleNamespace(
        _reload_roster_if_changed=Mock(),
        deliver_pending=Mock(return_value=0),
        poller=SimpleNamespace(fetch=Mock(return_value=([raw], False, True))),
        session=object(),
        _tweet_queue=queue.Queue(),
        seen=seen,
        _pool=object(),
        _normalize_source_subject=Mock(return_value=normalized),
        _process_items=process,
    )

    assert Notifier.poll_once(notifier) == 0
    seen.is_new.assert_called_once_with(normalized)
    process.assert_called_once_with([normalized], notifier._pool)
