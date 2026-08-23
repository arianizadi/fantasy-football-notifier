from __future__ import annotations

from datetime import datetime, timezone

from notifier.event_store import EventStore
from notifier.models import Classification, NewsItem


def _item(guid: str = "twitter:42:George Kittle") -> NewsItem:
    return NewsItem(
        source="twitter",
        guid=guid,
        player_name="George Kittle",
        headline="49ers activated George Kittle from active/PUP",
        body="The 49ers activated TE George Kittle from the active/PUP list.",
        url="https://x.com/example/status/42",
        published_at=datetime(2026, 8, 23, 17, 30, tzinfo=timezone.utc),
    )


def test_event_journal_preserves_raw_report_and_structured_labels(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item()
    classification = Classification(
        "return",
        4,
        "Availability outlook improved, with workload still unconfirmed.",
        True,
        {"direction": "positive"},
    )

    store.record_classification(item, classification, tier="preseason")
    store.mark_outcome(item.guid, "delivered", message_id=501)
    row = store.get(item.guid)

    assert row is not None
    assert row["body"] == item.body
    assert row["player_name"] == "George Kittle"
    assert row["event_type"] == "return"
    assert row["direction"] == "positive"
    assert row["severity"] == 4
    assert row["tier"] == "preseason"
    assert row["outcome"] == "delivered"
    assert row["telegram_message_id"] == 501
    assert row["embedding"] is None
    store.close()


def test_event_journal_search_feedback_and_embedding_slot(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item()
    store.record_received(item)
    token = store.get(item.guid)["alert_token"]

    assert store.record_feedback(token, "useful") is True
    assert store.record_feedback(token, "invalid") is False
    assert store.store_embedding(item.guid, "future-provider/model", b"vector") is True
    assert store.recent_for_player("Kittle")[0]["feedback"] == "useful"
    assert store.search("active PUP")[0]["guid"] == item.guid
    assert store.get(item.guid)["embedding_model"] == "future-provider/model"
    store.close()


def test_event_journal_upsert_does_not_duplicate_guid(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item()
    store.record_received(item)
    store.record_received(item)

    assert len(store.search("Kittle")) == 1
    store.close()


def test_changed_raw_report_replaces_the_same_fts_row(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item()
    store.record_received(item)
    updated = NewsItem(
        source=item.source,
        guid=item.guid,
        player_name=item.player_name,
        headline="George Kittle cleared to practice",
        body="Kittle returned to team drills.",
        url=item.url,
        published_at=item.published_at,
    )

    store.record_received(updated)

    assert store.search("active PUP") == []
    assert store.search("team drills")[0]["guid"] == item.guid
    assert store._connection.execute(
        "SELECT COUNT(*) FROM news_events_fts"
    ).fetchone()[0] == 1
    store.close()


def test_classification_does_not_reindex_an_existing_raw_report(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item()
    store.record_received(item)
    indexed_before = store._connection.total_changes
    classification = Classification(
        "return",
        4,
        "Availability improved.",
        True,
        {"direction": "positive"},
    )

    store.record_classification(item, classification, tier="preseason")

    # One news_events UPDATE is the only write; FTS would add several changes.
    assert store._connection.total_changes - indexed_before == 1
    assert store.search("Availability improved") == []
    assert store.search("active PUP")[0]["event_type"] == "return"
    store.close()


def test_fts_rows_use_event_ids_and_legacy_rows_are_migrated(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item()
    store.record_received(item)
    event_id = store.get(item.guid)["id"]
    store._connection.execute(
        "DELETE FROM event_store_meta WHERE key = 'fts_rowid_schema'"
    )
    store._connection.execute("DELETE FROM news_events_fts")
    store._connection.execute(
        """
        INSERT INTO news_events_fts(rowid, guid, player_name, headline, body)
        VALUES (999, ?, ?, ?, ?)
        """,
        (item.guid, item.player_name, item.headline, item.body),
    )
    store._connection.commit()
    store.close()

    reopened = EventStore(tmp_path)
    fts_row = reopened._connection.execute(
        "SELECT rowid FROM news_events_fts WHERE news_events_fts MATCH ?",
        ('"Kittle"',),
    ).fetchone()

    assert fts_row["rowid"] == event_id
    assert reopened.search("Kittle")[0]["guid"] == item.guid
    reopened.close()


def test_in_memory_store_does_not_create_state_directory(tmp_path) -> None:
    state_dir = tmp_path / "dry-run-state"
    store = EventStore(state_dir, in_memory=True)

    store.record_received(_item())

    assert store.count() == 1
    assert store.path is None
    assert not state_dir.exists()
    store.close()


def test_recent_player_context_can_exclude_current_report(tmp_path) -> None:
    store = EventStore(tmp_path)
    first = _item("first")
    second = _item("second")
    store.record_received(first)
    store.record_received(second)

    rows = store.recent_for_player(
        "Kittle",
        exclude_guid="second",
        since_hours=72,
    )

    assert [row["guid"] for row in rows] == ["first"]
    store.close()
