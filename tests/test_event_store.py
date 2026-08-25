from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from notifier.event_store import EventStore, read_all_reports
from notifier.models import Classification, NewsItem, report_revision_identity


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
    store.mark_outcome(item, "delivered", message_id=501)
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
    assert store.store_embedding(item, "future-provider/model", b"vector") is True
    assert store.recent_for_player("Kittle")[0]["feedback"] == "useful"
    assert store.search("active PUP")[0]["guid"] == item.guid
    assert store.get(item.guid)["embedding_model"] == "future-provider/model"
    store.close()


def test_event_journal_upsert_does_not_duplicate_exact_revision(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item()
    store.record_received(item)
    store.record_received(item)

    assert len(store.search("Kittle")) == 1
    store.close()


def test_subject_confidence_persists_updates_and_is_returned_by_recent(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = replace(_item(), subject_confident=False)
    before = datetime.now(timezone.utc) - timedelta(minutes=1)

    store.record_received(item)

    assert store.get(item)["subject_confident"] == 0
    rows = store.recent(
        since=before,
        until=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    assert len(rows) == 1
    assert rows[0]["subject_confident"] == 0

    # Confidence is metadata, not part of the raw revision identity.  A more
    # confident parse of the exact same report updates the existing row.
    store.record_received(replace(item, subject_confident=True))

    assert store.count() == 1
    assert store.get(item)["subject_confident"] == 1
    store.close()


def test_changed_raw_report_with_reused_guid_preserves_both_fts_rows(tmp_path) -> None:
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

    assert store.search("active PUP")[0]["body"] == item.body
    assert store.search("team drills")[0]["guid"] == item.guid
    assert store._connection.execute(
        "SELECT COUNT(*) FROM news_events_fts"
    ).fetchone()[0] == 2
    assert store.count() == 2
    store.close()


def test_reused_guid_status_revisions_keep_rows_and_feedback_isolated(tmp_path) -> None:
    store = EventStore(tmp_path)
    questionable = NewsItem(
        source="rotowire",
        guid="rotowire:kittle-status",
        player_name="George Kittle",
        headline="George Kittle injury update",
        body="Kittle is questionable for Sunday with an ankle injury.",
        url="https://example.test/kittle",
        published_at=None,
    )
    ruled_out = NewsItem(
        source=questionable.source,
        guid=questionable.guid,
        player_name=questionable.player_name,
        headline=questionable.headline,
        body="Kittle has been ruled out for Sunday with an ankle injury.",
        url=questionable.url,
        published_at=None,
    )
    store.record_classification(
        questionable,
        Classification("injury", 3, "Availability uncertain.", True, {}),
    )
    store.mark_outcome(questionable, "delivered", message_id=41)
    store.record_classification(
        ruled_out,
        Classification("inactive", 5, "Unavailable for Sunday.", True, {}),
    )
    store.mark_outcome(ruled_out, "delivered", message_id=42)

    first = store.get(questionable)
    second = store.get(ruled_out)
    assert first is not None and second is not None
    assert first["id"] != second["id"]
    assert first["report_id"] != second["report_id"]
    assert first["alert_token"] != second["alert_token"]
    assert first["body"] == questionable.body
    assert first["event_type"] == "injury"
    assert first["telegram_message_id"] == 41
    assert second["body"] == ruled_out.body
    assert second["event_type"] == "inactive"
    assert second["telegram_message_id"] == 42
    assert store.record_feedback(first["alert_token"], "useful") is True
    assert store.record_feedback(second["alert_token"], "wrong") is True
    assert store.get(questionable)["feedback"] == "useful"
    assert store.get(ruled_out)["feedback"] == "wrong"
    assert store.count() == 2
    store.close()


def test_reused_guid_condition_revision_keeps_ankle_and_concussion_rows(tmp_path) -> None:
    store = EventStore(tmp_path)
    ankle = NewsItem(
        source="rotowire",
        guid="rotowire:kittle-condition",
        player_name="George Kittle",
        headline="George Kittle left practice",
        body="Kittle left practice with an ankle injury.",
        url="https://example.test/kittle-condition",
        published_at=None,
    )
    concussion = NewsItem(
        source=ankle.source,
        guid=ankle.guid,
        player_name=ankle.player_name,
        headline=ankle.headline,
        body="Kittle left practice and is being evaluated for a concussion.",
        url=ankle.url,
        published_at=None,
    )

    store.record_received(ankle)
    store.record_received(concussion)

    rows = store.recent_for_player("Kittle")
    assert {row["body"] for row in rows} == {ankle.body, concussion.body}
    assert len({row["report_id"] for row in rows}) == 2
    assert len({row["alert_token"] for row in rows}) == 2
    store.close()


def test_legacy_guid_unique_schema_migrates_without_breaking_old_feedback(
    tmp_path,
) -> None:
    item = _item()
    database = tmp_path / "news-events.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE news_events (
            id INTEGER PRIMARY KEY,
            guid TEXT NOT NULL UNIQUE,
            alert_token TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL,
            player_name TEXT NOT NULL DEFAULT '',
            headline TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL DEFAULT '',
            published_at TEXT,
            received_at TEXT NOT NULL,
            event_type TEXT,
            direction TEXT,
            severity INTEGER,
            summary TEXT,
            is_actionable INTEGER,
            tier TEXT,
            outcome TEXT NOT NULL DEFAULT 'received',
            telegram_message_id INTEGER,
            feedback TEXT,
            feedback_at TEXT,
            embedding_model TEXT,
            embedding BLOB,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO news_events(
            guid, alert_token, source, player_name, headline, body, url,
            published_at, received_at, outcome, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'delivered', ?)
        """,
        (
            item.guid,
            "legacy-guid-token",
            item.source,
            item.player_name,
            item.headline,
            item.body,
            item.url,
            item.published_at.isoformat(),
            "2026-08-23T17:31:00+00:00",
            "2026-08-23T17:31:00+00:00",
        ),
    )
    connection.commit()
    connection.close()

    store = EventStore(tmp_path)
    row = store.get(item)
    assert row is not None
    assert row["report_id"] == report_revision_identity(item)
    assert row["alert_token"] == report_revision_identity(item)[:16]
    assert row["legacy_alert_token"] == "legacy-guid-token"
    assert store.record_feedback("legacy-guid-token", "useful") is True
    assert store.get(item)["feedback"] == "useful"

    store.record_received(item)
    changed = NewsItem(
        source=item.source,
        guid=item.guid,
        player_name=item.player_name,
        headline=item.headline,
        body="Kittle returned to team drills.",
        url=item.url,
        published_at=item.published_at,
    )
    store.record_received(changed)
    assert store.count() == 2
    assert store.get(changed)["alert_token"] != store.get(item)["alert_token"]
    store.close()


def test_revision_schema_migration_adds_subject_confidence_default_true(
    tmp_path,
) -> None:
    item = _item()
    database = tmp_path / "news-events.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE news_events (
            id INTEGER PRIMARY KEY,
            report_id TEXT NOT NULL UNIQUE,
            guid TEXT NOT NULL,
            alert_token TEXT NOT NULL UNIQUE,
            legacy_alert_token TEXT,
            source TEXT NOT NULL,
            player_name TEXT NOT NULL DEFAULT '',
            headline TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL DEFAULT '',
            published_at TEXT,
            received_at TEXT NOT NULL,
            event_type TEXT,
            direction TEXT,
            severity INTEGER,
            summary TEXT,
            is_actionable INTEGER,
            tier TEXT,
            outcome TEXT NOT NULL DEFAULT 'received',
            telegram_message_id INTEGER,
            feedback TEXT,
            feedback_at TEXT,
            embedding_model TEXT,
            embedding BLOB,
            updated_at TEXT NOT NULL
        )
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    report_id = report_revision_identity(item)
    connection.execute(
        """
        INSERT INTO news_events(
            report_id, guid, alert_token, source, player_name, headline, body,
            url, published_at, received_at, outcome, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'delivered', ?)
        """,
        (
            report_id,
            item.guid,
            report_id[:16],
            item.source,
            item.player_name,
            item.headline,
            item.body,
            item.url,
            item.published_at.isoformat(),
            now,
            now,
        ),
    )
    connection.commit()
    connection.close()

    store = EventStore(tmp_path)

    columns = {
        row["name"] for row in store._connection.execute("PRAGMA table_info(news_events)")
    }
    assert "subject_confident" in columns
    assert store.get(item)["subject_confident"] == 1
    assert store.recent(
        since=datetime.now(timezone.utc) - timedelta(minutes=1),
        until=datetime.now(timezone.utc) + timedelta(minutes=1),
    )[0]["subject_confident"] == 1
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


def test_read_all_reports_preserves_active_wal_and_reads_its_rows(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item()
    store.record_received(item)
    database = tmp_path / "news-events.sqlite3"
    source_files = (
        database,
        tmp_path / "news-events.sqlite3-wal",
        tmp_path / "news-events.sqlite3-shm",
    )
    assert all(path.is_file() for path in source_files)
    before = {
        path.name: (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.stat().st_ctime_ns,
            path.read_bytes(),
        )
        for path in source_files
    }

    rows = read_all_reports(tmp_path)

    after = {
        path.name: (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.stat().st_ctime_ns,
            path.read_bytes(),
        )
        for path in source_files
    }
    assert [row["guid"] for row in rows] == [item.guid]
    assert after == before
    store.close()


def test_read_all_reports_works_on_locked_down_closed_wal_archive(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item()
    store.record_received(item)
    store.close()
    database = tmp_path / "news-events.sqlite3"
    wal = tmp_path / "news-events.sqlite3-wal"
    shm = tmp_path / "news-events.sqlite3-shm"
    assert not wal.exists()
    assert not shm.exists()
    before = database.read_bytes()
    original_mode = tmp_path.stat().st_mode & 0o777

    os.chmod(tmp_path, 0o555)
    try:
        rows = read_all_reports(tmp_path)
    finally:
        os.chmod(tmp_path, original_mode)

    assert [row["guid"] for row in rows] == [item.guid]
    assert database.read_bytes() == before
    assert not wal.exists()
    assert not shm.exists()


def test_concurrent_initializers_serialize_schema_migration(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "news-events.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE news_events (
            id INTEGER PRIMARY KEY,
            report_id TEXT NOT NULL UNIQUE,
            guid TEXT NOT NULL,
            alert_token TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL,
            player_name TEXT NOT NULL DEFAULT '',
            headline TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL DEFAULT '',
            published_at TEXT,
            received_at TEXT NOT NULL,
            event_type TEXT,
            direction TEXT,
            severity INTEGER,
            summary TEXT,
            is_actionable INTEGER,
            tier TEXT,
            outcome TEXT NOT NULL DEFAULT 'received',
            telegram_message_id INTEGER,
            feedback TEXT,
            feedback_at TEXT,
            embedding_model TEXT,
            embedding BLOB,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()

    original = EventStore._migrate_embedding_schema_locked
    counter_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def observed_migration(store) -> None:
        nonlocal active, maximum_active
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.03)
            original(store)
        finally:
            with counter_lock:
                active -= 1

    monkeypatch.setattr(
        EventStore,
        "_migrate_embedding_schema_locked",
        observed_migration,
    )
    worker_count = 8
    barrier = threading.Barrier(worker_count)
    errors: list[BaseException] = []

    def initialize() -> None:
        try:
            barrier.wait()
            store = EventStore(tmp_path)
            store.close()
        except BaseException as error:
            with counter_lock:
                errors.append(error)

    threads = [threading.Thread(target=initialize) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert maximum_active == 1
    verify = sqlite3.connect(database)
    columns = {row[1] for row in verify.execute("PRAGMA table_info(news_events)")}
    verify.close()
    assert "urgency_reason_codes" in columns


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
