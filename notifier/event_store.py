"""Durable, searchable journal of every news item the notifier evaluates.

Structured labels answer operational questions (player, event, direction,
severity) more reliably than a vector alone. Nullable embedding columns are a
storage hook for later evaluation; a production similarity index may still
need explicit model, format, and dimension metadata. This module intentionally
makes no external model calls.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .logging_utils import structured_log
from .models import Classification, NewsItem

DATABASE_FILENAME = "news-events.sqlite3"
DIRECTIONS = frozenset({"positive", "negative", "mixed", "neutral", "unknown"})
NEGATIVE_EVENTS = frozenset({"injury", "inactive", "release", "suspension"})
POSITIVE_EVENTS = frozenset({"return"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token(guid: str) -> str:
    return hashlib.sha256(guid.encode("utf-8")).hexdigest()[:16]


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
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS news_events (
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
        """Upsert raw report fields and reindex only when searchable text changed."""
        existing = self._connection.execute(
            """
            SELECT id, player_name, headline, body
            FROM news_events WHERE guid = ?
            """,
            (item.guid,),
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
                guid, alert_token, source, player_name, headline, body, url,
                published_at, received_at, outcome, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'received', ?)
            ON CONFLICT(guid) DO UPDATE SET
                source = excluded.source,
                player_name = excluded.player_name,
                headline = excluded.headline,
                body = excluded.body,
                url = excluded.url,
                published_at = COALESCE(excluded.published_at, news_events.published_at),
                updated_at = excluded.updated_at
            """,
            (
                item.guid,
                _token(item.guid),
                item.source,
                item.player_name,
                item.headline,
                item.body,
                item.url,
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
                WHERE guid = ?
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
                    item.guid,
                ),
            )
            if cursor.rowcount == 0:
                self._upsert_received_locked(item, now)
                self._connection.execute(
                    """
                    UPDATE news_events SET
                        event_type = ?, direction = ?, severity = ?, summary = ?,
                        is_actionable = ?, tier = ?, outcome = ?, updated_at = ?
                    WHERE guid = ?
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
                        item.guid,
                    ),
                )
            self._connection.commit()

    def mark_outcome(
        self,
        guid: str,
        outcome: str,
        *,
        tier: str = "",
        message_id: int | None = None,
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE news_events SET
                    outcome = ?,
                    tier = COALESCE(NULLIF(?, ''), tier),
                    telegram_message_id = COALESCE(?, telegram_message_id),
                    updated_at = ?
                WHERE guid = ?
                """,
                (outcome, tier, message_id, _utc_now(), guid),
            )
            self._connection.commit()

    def record_feedback(self, alert_token: str, verdict: str) -> bool:
        if verdict not in {"useful", "wrong", "noisy"}:
            return False
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE news_events SET feedback = ?, feedback_at = ?, updated_at = ?
                WHERE alert_token = ?
                """,
                (verdict, _utc_now(), _utc_now(), alert_token),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    def store_embedding(self, guid: str, model: str, embedding: bytes) -> bool:
        """Attach provider-generated vector bytes without prescribing a provider."""
        if not model or not embedding:
            return False
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE news_events
                SET embedding_model = ?, embedding = ?, updated_at = ?
                WHERE guid = ?
                """,
                (model, sqlite3.Binary(embedding), _utc_now(), guid),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return dict(row)

    def get(self, guid: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM news_events WHERE guid = ?", (guid,)
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
                ORDER BY received_at DESC
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
                        ORDER BY bm25(news_events_fts), events.received_at DESC
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
                ORDER BY received_at DESC
                LIMIT ?
                """,
                (pattern, pattern, pattern, bounded),
            ).fetchall()
            return [self._row(row) for row in rows]
