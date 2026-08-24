"""Durable, searchable journal of every news item the notifier evaluates.

Structured labels answer operational questions (player, event, direction,
severity) more reliably than a vector alone. Nullable embedding columns are a
storage hook for later evaluation; a production similarity index may still
need explicit model, format, and dimension metadata. This module intentionally
makes no external model calls.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .logging_utils import structured_log
from .models import Classification, NewsItem, report_revision_identity

DATABASE_FILENAME = "news-events.sqlite3"
DIRECTIONS = frozenset({"positive", "negative", "mixed", "neutral", "unknown"})
NEGATIVE_EVENTS = frozenset({"injury", "inactive", "release", "suspension"})
POSITIVE_EVENTS = frozenset({"return"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token(item: NewsItem) -> str:
    return report_revision_identity(item)[:16]


def classification_direction(classification: Classification) -> str:
    """Prefer the model's constrained label, then use a conservative fallback."""
    candidate = str(classification.raw.get("direction") or "").strip().lower()
    if candidate in DIRECTIONS:
        return candidate
    event_type = classification.event_type.strip().lower()
    if event_type in NEGATIVE_EVENTS:
        return "negative"
    if event_type in POSITIVE_EVENTS:
        return "positive"
    if event_type == "practice_report":
        return "mixed"
    return "neutral"


class EventStore:
    """Thread-safe SQLite event journal used by poll and Telegram threads."""

    def __init__(self, state_dir: Path, *, in_memory: bool = False) -> None:
        """Open the journal, optionally without creating any files.

        ``in_memory`` is intended for dry runs and tests that must not mutate the
        configured state directory.
        """
        self.path: Path | None
        if in_memory:
            self.path = None
            database: str | Path = ":memory:"
        else:
            state_dir.mkdir(parents=True, exist_ok=True)
            self.path = state_dir / DATABASE_FILENAME
            database = self.path
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            database,
            timeout=10,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._fts_enabled = False
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._migrate_revision_schema_locked()
            self._create_news_events_table_locked()
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS news_events_guid_time "
                "ON news_events(guid, received_at DESC)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS news_events_legacy_alert_token "
                "ON news_events(legacy_alert_token)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS news_events_player_time "
                "ON news_events(player_name, received_at DESC)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS news_events_event_time "
                "ON news_events(event_type, received_at DESC)"
            )
            try:
                self._connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS news_events_fts USING fts5(
                        guid UNINDEXED,
                        player_name,
                        headline,
                        body
                    )
                    """
                )
                self._fts_enabled = True
                self._migrate_fts_rowids_locked()
            except sqlite3.OperationalError as error:
                self._fts_enabled = False
                structured_log(logging.INFO, "events.fts_unavailable", error=str(error))
            self._connection.commit()

    def _create_news_events_table_locked(self) -> None:
        """Create the revision-aware journal schema when it is absent."""
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS news_events (
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
                subject_confident INTEGER NOT NULL DEFAULT 1,
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

    def _migrate_revision_schema_locked(self) -> None:
        """Remove the legacy one-row-per-GUID constraint without data loss.

        Existing callback tokens are intentionally preserved so buttons on
        messages already visible in Telegram keep mapping to the same rows.
        New reports use revision-aware tokens after this migration.
        """
        exists = self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'news_events'"
        ).fetchone()
        if exists is None:
            return
        columns = [
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(news_events)")
        ]
        if "report_id" in columns:
            if "legacy_alert_token" not in columns:
                self._connection.execute(
                    "ALTER TABLE news_events ADD COLUMN legacy_alert_token TEXT"
                )
            if "subject_confident" not in columns:
                self._connection.execute(
                    "ALTER TABLE news_events ADD COLUMN "
                    "subject_confident INTEGER NOT NULL DEFAULT 1"
                )
            return

        self._connection.execute("DROP TABLE IF EXISTS news_events_fts")
        self._connection.execute(
            "ALTER TABLE news_events RENAME TO news_events_legacy_revision"
        )
        self._create_news_events_table_locked()
        legacy_rows = self._connection.execute(
            "SELECT * FROM news_events_legacy_revision ORDER BY id"
        ).fetchall()
        copied_columns = [
            column
            for column in columns
            if column not in {"report_id", "alert_token", "legacy_alert_token"}
        ]
        column_sql = ", ".join(
            ["report_id", "alert_token", "legacy_alert_token", *copied_columns]
        )
        placeholders = ", ".join("?" for _ in range(len(copied_columns) + 3))
        for row in legacy_rows:
            item = NewsItem(
                source=str(row["source"] or ""),
                guid=str(row["guid"] or ""),
                player_name=str(row["player_name"] or ""),
                headline=str(row["headline"] or ""),
                body=str(row["body"] or ""),
                url=str(row["url"] or ""),
                published_at=None,
            )
            report_id = report_revision_identity(item)
            self._connection.execute(
                f"INSERT INTO news_events({column_sql}) VALUES ({placeholders})",
                (
                    report_id,
                    report_id[:16],
                    str(row["alert_token"] or "") or None,
                    *(row[column] for column in copied_columns),
                ),
            )
        self._connection.execute("DROP TABLE news_events_legacy_revision")

        metadata_exists = self._connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'event_store_meta'"
        ).fetchone()
        if metadata_exists is not None:
            self._connection.execute(
                "DELETE FROM event_store_meta WHERE key = 'fts_rowid_schema'"
            )

    def _migrate_fts_rowids_locked(self) -> None:
        """Key legacy FTS rows by ``news_events.id`` once.

        Older databases let FTS assign unrelated row IDs, which forced updates
        and joins through the unindexed ``guid`` column.  A small metadata table
        makes the compatibility rebuild a one-time operation.
        """
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS event_store_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        migrated = self._connection.execute(
            "SELECT value FROM event_store_meta WHERE key = 'fts_rowid_schema'"
        ).fetchone()
        if migrated is not None and migrated["value"] == "1":
            return
        self._connection.execute("DELETE FROM news_events_fts")
        self._connection.execute(
            """
            INSERT INTO news_events_fts(rowid, guid, player_name, headline, body)
            SELECT id, guid, player_name, headline, body FROM news_events
            """
        )
        self._connection.execute(
            """
            INSERT INTO event_store_meta(key, value) VALUES ('fts_rowid_schema', '1')
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _sync_search_locked(self, row_id: int, item: NewsItem) -> None:
        if not self._fts_enabled:
            return
        self._connection.execute(
            """
            INSERT OR REPLACE INTO news_events_fts(
                rowid, guid, player_name, headline, body
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (row_id, item.guid, item.player_name, item.headline, item.body),
        )

    def _upsert_received_locked(self, item: NewsItem, now: str) -> None:
        """Insert one exact revision and collapse only true raw duplicates."""
        report_id = report_revision_identity(item)
        existing = self._connection.execute(
            """
            SELECT id, player_name, headline, body
            FROM news_events WHERE report_id = ?
            """,
            (report_id,),
        ).fetchone()
        search_changed = existing is None or (
            existing["player_name"],
            existing["headline"],
            existing["body"],
        ) != (item.player_name, item.headline, item.body)
        published = item.published_at.isoformat() if item.published_at else None
        cursor = self._connection.execute(
            """
            INSERT INTO news_events(
                report_id, guid, alert_token, source, player_name, headline, body, url,
                subject_confident, published_at, received_at, outcome, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'received', ?)
            ON CONFLICT(report_id) DO UPDATE SET
                source = excluded.source,
                player_name = excluded.player_name,
                headline = excluded.headline,
                body = excluded.body,
                url = excluded.url,
                subject_confident = excluded.subject_confident,
                published_at = COALESCE(excluded.published_at, news_events.published_at),
                updated_at = excluded.updated_at
            """,
            (
                report_id,
                item.guid,
                _token(item),
                item.source,
                item.player_name,
                item.headline,
                item.body,
                item.url,
                int(item.subject_confident),
                published,
                now,
                now,
            ),
        )
        if existing is None:
            row_id = int(cursor.lastrowid)
        else:
            row_id = int(existing["id"])
        if search_changed:
            self._sync_search_locked(row_id, item)

    def record_received(self, item: NewsItem) -> None:
        """Persist the original report before classification or filtering."""
        now = _utc_now()
        with self._lock:
            self._upsert_received_locked(item, now)
            self._connection.commit()

    def record_classification(
        self,
        item: NewsItem,
        classification: Classification,
        *,
        tier: str = "",
        outcome: str = "classified",
    ) -> None:
        """Attach labels without rewriting/reindexing an already-recorded report."""
        with self._lock:
            now = _utc_now()
            cursor = self._connection.execute(
                """
                UPDATE news_events SET
                    event_type = ?, direction = ?, severity = ?, summary = ?,
                    is_actionable = ?, tier = ?, outcome = ?, updated_at = ?
                WHERE report_id = ?
                """,
                (
                    classification.event_type,
                    classification_direction(classification),
                    classification.severity,
                    classification.fantasy_impact,
                    int(classification.is_actionable),
                    tier or None,
                    outcome,
                    now,
                    report_revision_identity(item),
                ),
            )
            if cursor.rowcount == 0:
                self._upsert_received_locked(item, now)
                self._connection.execute(
                    """
                    UPDATE news_events SET
                        event_type = ?, direction = ?, severity = ?, summary = ?,
                        is_actionable = ?, tier = ?, outcome = ?, updated_at = ?
                    WHERE report_id = ?
                    """,
                    (
                        classification.event_type,
                        classification_direction(classification),
                        classification.severity,
                        classification.fantasy_impact,
                        int(classification.is_actionable),
                        tier or None,
                        outcome,
                        now,
                        report_revision_identity(item),
                    ),
                )
            self._connection.commit()

    def mark_outcome(
        self,
        report: NewsItem | str,
        outcome: str,
        *,
        tier: str = "",
        message_id: int | None = None,
    ) -> None:
        with self._lock:
            if isinstance(report, NewsItem):
                predicate = "report_id = ?"
                identity = report_revision_identity(report)
            else:
                # Backward-compatible callers target the newest revision for
                # that source GUID. New delivery code should pass NewsItem.
                predicate = (
                    "id = (SELECT id FROM news_events WHERE guid = ? "
                    "ORDER BY received_at DESC, id DESC LIMIT 1)"
                )
                identity = report
            self._connection.execute(
                f"""
                UPDATE news_events SET
                    outcome = ?,
                    tier = COALESCE(NULLIF(?, ''), tier),
                    telegram_message_id = COALESCE(?, telegram_message_id),
                    updated_at = ?
                WHERE {predicate}
                """,
                (outcome, tier, message_id, _utc_now(), identity),
            )
            self._connection.commit()

    def record_feedback(self, alert_token: str, verdict: str) -> bool:
        if verdict not in {"useful", "wrong", "noisy"}:
            return False
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE news_events SET feedback = ?, feedback_at = ?, updated_at = ?
                WHERE alert_token = ? OR legacy_alert_token = ?
                """,
                (verdict, _utc_now(), _utc_now(), alert_token, alert_token),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    def store_embedding(
        self, report: NewsItem | str, model: str, embedding: bytes
    ) -> bool:
        """Attach provider-generated vector bytes without prescribing a provider."""
        if not model or not embedding:
            return False
        with self._lock:
            if isinstance(report, NewsItem):
                predicate = "report_id = ?"
                identity = report_revision_identity(report)
            else:
                predicate = (
                    "id = (SELECT id FROM news_events WHERE guid = ? "
                    "ORDER BY received_at DESC, id DESC LIMIT 1)"
                )
                identity = report
            cursor = self._connection.execute(
                f"""
                UPDATE news_events
                SET embedding_model = ?, embedding = ?, updated_at = ?
                WHERE {predicate}
                """,
                (model, sqlite3.Binary(embedding), _utc_now(), identity),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return dict(row)

    def get(self, report: NewsItem | str) -> dict[str, Any] | None:
        with self._lock:
            if isinstance(report, NewsItem):
                row = self._connection.execute(
                    "SELECT * FROM news_events WHERE report_id = ?",
                    (report_revision_identity(report),),
                ).fetchone()
            else:
                row = self._connection.execute(
                    "SELECT * FROM news_events WHERE guid = ? "
                    "ORDER BY received_at DESC, id DESC LIMIT 1",
                    (report,),
                ).fetchone()
            return self._row(row) if row is not None else None

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS total FROM news_events"
            ).fetchone()
            return int(row["total"] if row is not None else 0)

    def recent_for_player(
        self,
        query: str,
        *,
        limit: int = 5,
        exclude_guid: str = "",
        since_hours: int | None = None,
    ) -> list[dict[str, Any]]:
        pattern = f"%{query.strip()}%"
        cutoff = (
            (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()
            if since_hours is not None
            else ""
        )
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM news_events
                WHERE player_name LIKE ? COLLATE NOCASE
                  AND (? = '' OR guid != ?)
                  AND (? = '' OR received_at >= ?)
                ORDER BY received_at DESC, id DESC
                LIMIT ?
                """,
                (
                    pattern,
                    exclude_guid,
                    exclude_guid,
                    cutoff,
                    cutoff,
                    max(1, min(limit, 50)),
                ),
            ).fetchall()
            return [self._row(row) for row in rows]

    def recent(
        self,
        *,
        since: datetime,
        until: datetime | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return every saved report in a bounded UTC window.

        The daily recap intentionally reads the complete journal rather than
        the Telegram alert list.  That preserves lower-severity transactions
        and role notes which are useful in a morning football briefing even
        though they did not justify an immediate notification.
        """
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        since_utc = since.astimezone(timezone.utc).isoformat()
        until_value = until or datetime.now(timezone.utc)
        if until_value.tzinfo is None:
            until_value = until_value.replace(tzinfo=timezone.utc)
        until_utc = until_value.astimezone(timezone.utc).isoformat()
        bounded = max(1, min(int(limit), 2000))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM news_events
                WHERE received_at >= ? AND received_at < ?
                ORDER BY received_at DESC, id DESC
                LIMIT ?
                """,
                (since_utc, until_utc, bounded),
            ).fetchall()
            return [self._row(row) for row in rows]

    def search(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search with a LIKE fallback when SQLite lacks FTS5."""
        wanted = query.strip()
        if not wanted:
            return []
        bounded = max(1, min(limit, 100))
        with self._lock:
            if self._fts_enabled:
                tokens = [token.replace('"', "") for token in wanted.split() if token]
                expression = " AND ".join(f'"{token}"' for token in tokens)
                try:
                    rows = self._connection.execute(
                        """
                        SELECT events.* FROM news_events_fts AS search
                        JOIN news_events AS events ON events.id = search.rowid
                        WHERE news_events_fts MATCH ?
                        ORDER BY bm25(news_events_fts), events.received_at DESC, events.id DESC
                        LIMIT ?
                        """,
                        (expression, bounded),
                    ).fetchall()
                    return [self._row(row) for row in rows]
                except sqlite3.OperationalError as error:
                    structured_log(logging.WARNING, "events.search_failed", error=str(error))
            pattern = f"%{wanted}%"
            rows = self._connection.execute(
                """
                SELECT * FROM news_events
                WHERE player_name LIKE ? COLLATE NOCASE
                   OR headline LIKE ? COLLATE NOCASE
                   OR body LIKE ? COLLATE NOCASE
                ORDER BY received_at DESC, id DESC
                LIMIT ?
                """,
                (pattern, pattern, pattern, bounded),
            ).fetchall()
            return [self._row(row) for row in rows]
