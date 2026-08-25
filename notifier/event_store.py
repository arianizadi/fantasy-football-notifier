"""Durable, searchable journal of every news item the notifier evaluates.

Structured labels answer operational questions (player, event, direction,
severity) more reliably than a vector alone. Nullable embedding columns retain
the model, provider, input-version, dimensions, and content hash so unlike
vector spaces can never be compared. This module intentionally makes no
external model calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any

from .logging_utils import structured_log
from .models import ActionUrgency, Classification, NewsItem, report_revision_identity

DATABASE_FILENAME = "news-events.sqlite3"
DIRECTIONS = frozenset({"positive", "negative", "mixed", "neutral", "unknown"})
NEGATIVE_EVENTS = frozenset({"injury", "inactive", "release", "suspension"})
POSITIVE_EVENTS = frozenset({"return"})
ARCHIVE_URGENCY_EMPTY_FIELDS = (
    "urgency_rule_level",
    "urgency_level",
    "urgency_basis",
    "urgency_policy_version",
)
ARCHIVE_URGENCY_GUARD_SQL = """
    AND (
        (
            urgency_rule_level IS NULL
            AND urgency_level IS NULL
            AND urgency_basis IS NULL
            AND urgency_policy_version IS NULL
        )
        OR urgency_basis = 'archive_replay'
    )
"""
SNAPSHOT_ATTEMPTS = 5


def archive_urgency_can_write(row: dict[str, Any]) -> bool:
    """Mirror the atomic archive-update guard for observational dry runs."""
    return bool(
        all(row.get(field) is None for field in ARCHIVE_URGENCY_EMPTY_FIELDS)
        or row.get("urgency_basis") == "archive_replay"
    )


def _file_signature(path: Path) -> tuple[int, int, int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _copy_stable_database_snapshot(database: Path, directory: Path) -> Path:
    """Copy one stable SQLite view without opening or touching the source.

    SQLite's nominal ``mode=ro`` can still create or update WAL sidecars. Audit
    commands instead copy the database and transactional sidecar into a private
    writable directory, then recover/read that copy. Source signatures on both
    sides reject a commit or checkpoint that overlapped the copy.
    """
    target = directory / DATABASE_FILENAME
    source_sidecars = (
        Path(f"{database}-wal"),
        Path(f"{database}-journal"),
    )
    target_sidecars = (
        Path(f"{target}-wal"),
        Path(f"{target}-journal"),
    )
    sources = (database, *source_sidecars)

    for _attempt in range(SNAPSHOT_ATTEMPTS):
        before = tuple(_file_signature(path) for path in sources)
        if before[0] is None:
            raise FileNotFoundError(database)
        try:
            shutil.copyfile(database, target)
            for source, copied, signature in zip(
                source_sidecars,
                target_sidecars,
                before[1:],
                strict=True,
            ):
                if signature is None:
                    copied.unlink(missing_ok=True)
                else:
                    shutil.copyfile(source, copied)
        except FileNotFoundError:
            continue
        after = tuple(_file_signature(path) for path in sources)
        if before == after:
            return target
    raise sqlite3.OperationalError(
        "news archive changed while creating a read-only audit snapshot"
    )


def read_all_reports(
    state_dir: Path,
    *,
    limit: int = 100_000,
) -> list[dict[str, Any]]:
    """Read the journal without creating or migrating it.

    Audit and dry-run commands must be observational. Opening ``EventStore``
    intentionally creates/migrates the production schema, while SQLite's URI
    read-only mode can still create WAL sidecars. These commands therefore read
    a private stable copy. A missing database or table is an empty archive, not
    permission to create one.
    """
    database = Path(state_dir) / DATABASE_FILENAME
    if not database.is_file():
        return []
    bounded = max(1, min(int(limit), 100_000))
    with tempfile.TemporaryDirectory(prefix="notifier-news-audit-") as temporary:
        snapshot = _copy_stable_database_snapshot(database, Path(temporary))
        connection = sqlite3.connect(snapshot, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            table = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'news_events'"
            ).fetchone()
            if table is None:
                return []
            rows = connection.execute(
                """
                SELECT * FROM news_events
                ORDER BY received_at, id
                LIMIT ?
                """,
                (bounded,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()


def read_fantasypros_corpus_snapshot(
    state_dir: Path,
    *,
    limit: int = 5_000,
) -> dict[str, Any]:
    """Observe the isolated FantasyPros corpus without opening production SQLite.

    The returned rows are the newest bounded chronological window, while the
    counts describe the complete corpus.  Query observations are restricted to
    the loaded rows.  As with :func:`read_all_reports`, all SQLite recovery and
    query work happens on a stable private copy, never on the configured source.
    Missing or older schemas are described rather than created or migrated.
    """

    database = Path(state_dir) / DATABASE_FILENAME
    empty: dict[str, Any] = {
        "database_exists": False,
        "database_size_bytes": 0,
        "database_sidecar_bytes": 0,
        "database_storage_bytes": 0,
        "corpus_table_exists": False,
        "observations_table_exists": False,
        "corpus_columns": [],
        "observation_columns": [],
        "corpus_count": 0,
        "vector_count": 0,
        "rows": [],
        "observations": [],
    }
    if not database.is_file():
        return empty

    bounded = max(1, min(int(limit), 100_000))
    with tempfile.TemporaryDirectory(prefix="notifier-fantasypros-audit-") as temporary:
        snapshot = _copy_stable_database_snapshot(database, Path(temporary))
        database_bytes = snapshot.stat().st_size
        sidecar_bytes = sum(
            path.stat().st_size
            for path in (Path(f"{snapshot}-wal"), Path(f"{snapshot}-journal"))
            if path.exists()
        )
        result = {
            **empty,
            "database_exists": True,
            "database_size_bytes": database_bytes,
            "database_sidecar_bytes": sidecar_bytes,
            "database_storage_bytes": database_bytes + sidecar_bytes,
        }
        connection = sqlite3.connect(snapshot, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            tables = {
                str(row["name"])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            corpus_table = "fantasypros_news_corpus" in tables
            observations_table = "fantasypros_corpus_observations" in tables
            result["corpus_table_exists"] = corpus_table
            result["observations_table_exists"] = observations_table

            if corpus_table:
                corpus_columns = [
                    str(row["name"])
                    for row in connection.execute(
                        "PRAGMA table_info(fantasypros_news_corpus)"
                    ).fetchall()
                ]
                result["corpus_columns"] = corpus_columns
                count = connection.execute(
                    "SELECT COUNT(*) AS total FROM fantasypros_news_corpus"
                ).fetchone()
                result["corpus_count"] = int(count["total"] if count is not None else 0)
                if "embedding" in corpus_columns:
                    count = connection.execute(
                        "SELECT COUNT(*) AS total FROM fantasypros_news_corpus "
                        "WHERE embedding IS NOT NULL"
                    ).fetchone()
                    result["vector_count"] = int(count["total"] if count is not None else 0)
                if {"id", "provider_item_id", "provider_created_at"}.issubset(
                    corpus_columns
                ):
                    rows = connection.execute(
                        """
                        SELECT * FROM (
                            SELECT * FROM fantasypros_news_corpus
                            ORDER BY provider_created_at DESC, id DESC
                            LIMIT ?
                        )
                        ORDER BY provider_created_at, id
                        """,
                        (bounded,),
                    ).fetchall()
                    result["rows"] = [dict(row) for row in rows]

            if observations_table:
                observation_columns = [
                    str(row["name"])
                    for row in connection.execute(
                        "PRAGMA table_info(fantasypros_corpus_observations)"
                    ).fetchall()
                ]
                result["observation_columns"] = observation_columns
                if (
                    result["rows"]
                    and corpus_table
                    and {
                        "run_id",
                        "provider_item_id",
                        "query_key",
                        "observed_at",
                    }.issubset(observation_columns)
                ):
                    observations = connection.execute(
                        """
                        WITH selected AS (
                            SELECT provider_item_id FROM fantasypros_news_corpus
                            ORDER BY provider_created_at DESC, id DESC
                            LIMIT ?
                        )
                        SELECT observations.*
                        FROM fantasypros_corpus_observations AS observations
                        JOIN selected USING (provider_item_id)
                        ORDER BY observed_at, run_id, query_key, provider_item_id
                        """,
                        (bounded,),
                    ).fetchall()
                    result["observations"] = [dict(row) for row in observations]
            return result
        finally:
            connection.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
            try:
                # Serialize schema inspection and DDL across independent
                # EventStore connections.  A deferred transaction would still
                # let two initializers observe the same missing column before
                # either one acquires SQLite's write lock.
                self._connection.execute("BEGIN IMMEDIATE")
                self._migrate_revision_schema_locked()
                self._create_news_events_table_locked()
                self._migrate_embedding_schema_locked()
                self._create_fantasypros_corpus_tables_locked()
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
                    if "fts5" not in str(error).casefold():
                        raise
                    self._fts_enabled = False
                    structured_log(
                        logging.INFO,
                        "events.fts_unavailable",
                        error=str(error),
                    )
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _create_fantasypros_corpus_tables_locked(self) -> None:
        """Create the isolated, append-only FantasyPros reference corpus.

        The corpus deliberately has no foreign key, trigger, or view into
        ``news_events``.  Merely collecting a reference row can therefore
        never make it eligible for a live alert, recap, search result, or
        urgency decision.
        """
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fantasypros_news_corpus (
                id INTEGER PRIMARY KEY,
                provider_item_id TEXT NOT NULL UNIQUE,
                sport TEXT NOT NULL CHECK (sport = 'NFL'),
                player_id INTEGER,
                team_id TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                impact TEXT NOT NULL DEFAULT '',
                categories_json TEXT NOT NULL DEFAULT '[]',
                author TEXT NOT NULL DEFAULT '',
                source_url TEXT NOT NULL DEFAULT '',
                provider_created_at TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                first_query_key TEXT NOT NULL,
                last_query_key TEXT NOT NULL,
                canonical_text TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_provider TEXT NOT NULL DEFAULT 'FantasyPros',
                attribution TEXT NOT NULL DEFAULT 'FantasyPros',
                usage_scope TEXT NOT NULL DEFAULT 'personal_reference',
                api_docs_url TEXT NOT NULL,
                embedding_model TEXT,
                embedding_provider TEXT,
                embedding_dimensions INTEGER,
                embedding_input_version TEXT,
                embedding_input_hash TEXT,
                embedding_at TEXT,
                embedding BLOB,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fantasypros_corpus_runs (
                run_id TEXT PRIMARY KEY,
                manifest_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('running', 'paused', 'failed', 'completed')
                ),
                stop_reason TEXT NOT NULL DEFAULT '',
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fantasypros_corpus_batches (
                run_id TEXT NOT NULL,
                query_key TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                item_count INTEGER NOT NULL,
                inserted_count INTEGER NOT NULL,
                updated_count INTEGER NOT NULL,
                duplicate_count INTEGER NOT NULL,
                completed_at TEXT NOT NULL,
                PRIMARY KEY (run_id, query_key),
                FOREIGN KEY (run_id) REFERENCES fantasypros_corpus_runs(run_id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fantasypros_corpus_observations (
                run_id TEXT NOT NULL,
                query_key TEXT NOT NULL,
                provider_item_id TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                PRIMARY KEY (run_id, query_key, provider_item_id),
                FOREIGN KEY (run_id, query_key)
                    REFERENCES fantasypros_corpus_batches(run_id, query_key),
                FOREIGN KEY (provider_item_id)
                    REFERENCES fantasypros_news_corpus(provider_item_id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fantasypros_corpus_embedding_spend (
                request_id INTEGER PRIMARY KEY,
                model TEXT NOT NULL,
                input_price_per_million_usd REAL NOT NULL CHECK (
                    input_price_per_million_usd >= 0
                ),
                conservative_tokens INTEGER NOT NULL CHECK (
                    conservative_tokens > 0
                ),
                reserved_cost_usd REAL NOT NULL CHECK (reserved_cost_usd >= 0),
                reserved_cost_nano_usd INTEGER NOT NULL CHECK (
                    reserved_cost_nano_usd >= 0
                ),
                lifetime_cap_usd REAL NOT NULL CHECK (lifetime_cap_usd > 0),
                lifetime_cap_nano_usd INTEGER NOT NULL CHECK (
                    lifetime_cap_nano_usd > 0
                ),
                status TEXT NOT NULL CHECK (
                    status IN ('reserved', 'completed', 'failed')
                ),
                reserved_at TEXT NOT NULL,
                finished_at TEXT,
                actual_prompt_tokens INTEGER CHECK (
                    actual_prompt_tokens IS NULL OR actual_prompt_tokens >= 0
                ),
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS fantasypros_corpus_created "
            "ON fantasypros_news_corpus(provider_created_at DESC, id DESC)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS fantasypros_corpus_player_created "
            "ON fantasypros_news_corpus(player_id, provider_created_at DESC)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS fantasypros_corpus_embedding_lookup "
            "ON fantasypros_news_corpus(embedding_model, embedding_dimensions, "
            "embedding_input_version, provider_created_at DESC)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS fantasypros_corpus_observation_item "
            "ON fantasypros_corpus_observations(provider_item_id, observed_at)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS fantasypros_corpus_embedding_spend_status "
            "ON fantasypros_corpus_embedding_spend(status, reserved_at)"
        )

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
                embedding_provider TEXT,
                embedding_dimensions INTEGER,
                embedding_input_version TEXT,
                embedding_input_hash TEXT,
                embedding_at TEXT,
                embedding BLOB,
                urgency_rule_level TEXT,
                urgency_level TEXT,
                urgency_reason_codes TEXT,
                urgency_basis TEXT,
                urgency_embedding_delta INTEGER,
                urgency_embedding_score REAL,
                urgency_embedding_support_count INTEGER,
                urgency_embedding_report_ids TEXT,
                urgency_policy_version TEXT,
                urgency_action_available INTEGER,
                urgency_roster_relevant INTEGER,
                urgency_availability_verified INTEGER,
                urgency_event_type TEXT,
                urgency_direction TEXT,
                urgency_event_status TEXT,
                urgency_action_context TEXT,
                urgency_subject_is_starter INTEGER,
                urgency_at TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )

    def _migrate_embedding_schema_locked(self) -> None:
        """Add vector provenance without rewriting the saved-news archive."""
        columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(news_events)")
        }
        additions = {
            "embedding_provider": "TEXT",
            "embedding_dimensions": "INTEGER",
            "embedding_input_version": "TEXT",
            "embedding_input_hash": "TEXT",
            "embedding_at": "TEXT",
            "urgency_rule_level": "TEXT",
            "urgency_level": "TEXT",
            "urgency_reason_codes": "TEXT",
            "urgency_basis": "TEXT",
            "urgency_embedding_delta": "INTEGER",
            "urgency_embedding_score": "REAL",
            "urgency_embedding_support_count": "INTEGER",
            "urgency_embedding_report_ids": "TEXT",
            "urgency_policy_version": "TEXT",
            "urgency_action_available": "INTEGER",
            "urgency_roster_relevant": "INTEGER",
            "urgency_availability_verified": "INTEGER",
            "urgency_event_type": "TEXT",
            "urgency_direction": "TEXT",
            "urgency_event_status": "TEXT",
            "urgency_action_context": "TEXT",
            "urgency_subject_is_starter": "INTEGER",
            "urgency_at": "TEXT",
        }
        for name, sql_type in additions.items():
            if name not in columns:
                self._add_news_event_column_locked(name, sql_type)
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS news_events_embedding_lookup "
            "ON news_events(player_name, embedding_model, embedding_dimensions, "
            "embedding_input_version, received_at DESC)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS news_events_urgency_context_history "
            "ON news_events(urgency_event_type, urgency_direction, "
            "urgency_action_context, tier, received_at DESC)"
        )

    def _add_news_event_column_locked(self, name: str, sql_type: str) -> None:
        """Add one known column, suppressing only a verified duplicate race."""
        try:
            self._connection.execute(
                f"ALTER TABLE news_events ADD COLUMN {name} {sql_type}"
            )
        except sqlite3.OperationalError as error:
            if "duplicate column name" not in str(error).casefold():
                raise
            columns = {
                str(row["name"])
                for row in self._connection.execute("PRAGMA table_info(news_events)")
            }
            if name not in columns:
                raise

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
                self._add_news_event_column_locked("legacy_alert_token", "TEXT")
            if "subject_confident" not in columns:
                self._add_news_event_column_locked(
                    "subject_confident",
                    "INTEGER NOT NULL DEFAULT 1",
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

    def record_urgency(
        self,
        item: NewsItem,
        urgency: ActionUrgency,
    ) -> bool:
        """Persist the auditable rule result and any bounded vector support.

        This intentionally does not change ``received_at`` or ``updated_at``;
        historical backfills must never make an old report look newly received
        to recap, chronology, or supersession code.
        """
        return self._record_urgency(item, urgency, archive_only=False)

    def record_archive_urgency(
        self,
        item: NewsItem,
        urgency: ActionUrgency,
    ) -> bool:
        """Attach an archive replay without replacing live roster context.

        The condition is part of the SQLite ``UPDATE`` so a live assessment
        written after a backfill snapshot was read still wins the race.  Force
        replays may refresh an existing archive replay, but they can never
        replace a rules-based live assessment.
        """
        if urgency.basis != "archive_replay":
            raise ValueError("archive urgency must use the archive_replay basis")
        return self._record_urgency(item, urgency, archive_only=True)

    def _record_urgency(
        self,
        item: NewsItem,
        urgency: ActionUrgency,
        *,
        archive_only: bool,
    ) -> bool:
        archive_guard = ARCHIVE_URGENCY_GUARD_SQL if archive_only else ""
        with self._lock:
            cursor = self._connection.execute(
                f"""
                UPDATE news_events SET
                    urgency_rule_level = ?, urgency_level = ?,
                    urgency_reason_codes = ?, urgency_basis = ?,
                    urgency_embedding_delta = ?, urgency_embedding_score = ?,
                    urgency_embedding_support_count = ?,
                    urgency_embedding_report_ids = ?, urgency_policy_version = ?,
                    urgency_action_available = ?, urgency_roster_relevant = ?,
                    urgency_availability_verified = ?, urgency_event_type = ?,
                    urgency_direction = ?, urgency_event_status = ?,
                    urgency_action_context = ?, urgency_subject_is_starter = ?,
                    urgency_at = ?
                WHERE report_id = ?
                {archive_guard}
                """,
                (
                    urgency.rule_level,
                    urgency.level,
                    json.dumps(list(urgency.reason_codes), separators=(",", ":")),
                    urgency.basis,
                    int(urgency.embedding_delta),
                    urgency.embedding_score,
                    int(urgency.embedding_support_count),
                    json.dumps(
                        list(urgency.embedding_report_ids), separators=(",", ":")
                    ),
                    urgency.policy_version,
                    int(urgency.action_available),
                    int(urgency.roster_relevant),
                    int(urgency.availability_verified),
                    urgency.canonical_event_type,
                    urgency.direction,
                    urgency.event_status,
                    urgency.action_context,
                    int(urgency.subject_is_starter),
                    _utc_now(),
                    report_revision_identity(item),
                ),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    def store_embedding(
        self,
        report: NewsItem | str,
        model: str,
        embedding: bytes,
        *,
        provider: str = "",
        dimensions: int | None = None,
        input_version: str = "",
        input_hash: str = "",
    ) -> bool:
        """Attach a validated vector and the metadata needed to compare it safely."""
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
                SET embedding_model = ?, embedding_provider = ?,
                    embedding_dimensions = ?, embedding_input_version = ?,
                    embedding_input_hash = ?, embedding_at = ?, embedding = ?
                WHERE {predicate}
                """,
                (
                    model,
                    provider,
                    dimensions,
                    input_version,
                    input_hash,
                    _utc_now(),
                    sqlite3.Binary(embedding),
                    identity,
                ),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    def recent_embedded_for_player(
        self,
        player_name: str,
        *,
        model: str,
        dimensions: int,
        input_version: str,
        exclude_report_id: str,
        active_message_id: int,
        active_alert_token: str,
        since_hours: int,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return comparable, previously delivered vectors for one player."""
        if (
            not player_name.strip()
            or not model
            or dimensions <= 0
            or active_message_id <= 0
            or not active_alert_token
        ):
            return []
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=max(1, since_hours))
        ).isoformat()
        with self._lock:
            current = self._connection.execute(
                "SELECT id, published_at FROM news_events WHERE report_id = ?",
                (exclude_report_id,),
            ).fetchone()
            if current is None:
                return []
            rows = self._connection.execute(
                """
                SELECT * FROM news_events
                WHERE player_name = ? COLLATE NOCASE
                  AND report_id != ?
                  AND received_at >= ?
                  AND embedding_model = ?
                  AND embedding_dimensions = ?
                  AND embedding_input_version = ?
                  AND embedding IS NOT NULL
                  AND telegram_message_id = ?
                  AND alert_token = ?
                  AND outcome = 'delivered'
                ORDER BY received_at DESC, id DESC
                LIMIT 100
                """,
                (
                    player_name.strip(),
                    exclude_report_id,
                    cutoff,
                    model,
                    int(dimensions),
                    input_version,
                    int(active_message_id),
                    active_alert_token,
                ),
            ).fetchall()
            current_published = _parse_datetime(current["published_at"])
            current_id = int(current["id"])
            older: list[dict[str, Any]] = []
            for row in rows:
                candidate_published = _parse_datetime(row["published_at"])
                if current_published is not None and candidate_published is not None:
                    precedes = candidate_published < current_published or (
                        candidate_published == current_published
                        and int(row["id"]) < current_id
                    )
                else:
                    precedes = int(row["id"]) < current_id
                if precedes:
                    older.append(self._row(row))
                if len(older) >= max(1, min(int(limit), 100)):
                    break
            return older

    def recent_urgency_candidates(
        self,
        item: NewsItem,
        *,
        event_type: str,
        direction: str,
        event_status: str,
        action_context: str,
        tier: str,
        model: str,
        provider: str,
        dimensions: int,
        input_version: str,
        since_days: int,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return chronological, comparable history for urgency evidence."""
        if (
            not item.player_name.strip()
            or not event_type
            or not direction
            or direction == "unknown"
            or not event_status
            or not action_context
            or not tier
            or not model
            or not provider
            or dimensions <= 0
        ):
            return []
        bounded = max(1, min(int(limit), 500))
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max(7, int(since_days)))
        ).isoformat()
        report_id = report_revision_identity(item)
        with self._lock:
            current = self._connection.execute(
                "SELECT id, published_at FROM news_events WHERE report_id = ?",
                (report_id,),
            ).fetchone()
            if current is None:
                return []
            rows = self._connection.execute(
                """
                SELECT * FROM news_events
                WHERE report_id != ?
                  AND received_at >= ?
                  AND urgency_event_type = ? COLLATE NOCASE
                  AND urgency_direction = ? COLLATE NOCASE
                  AND urgency_event_status = ?
                  AND urgency_action_context = ?
                  AND tier = ?
                  AND subject_confident = 1
                  AND (feedback IS NULL OR feedback NOT IN ('wrong', 'noisy'))
                  AND embedding_model = ?
                  AND embedding_provider = ?
                  AND embedding_dimensions = ?
                  AND embedding_input_version = ?
                  AND embedding IS NOT NULL
                ORDER BY received_at DESC, id DESC
                LIMIT 1000
                """,
                (
                    report_id,
                    cutoff,
                    event_type,
                    direction,
                    event_status,
                    action_context,
                    tier,
                    model,
                    provider,
                    int(dimensions),
                    input_version,
                ),
            ).fetchall()
            current_published = _parse_datetime(current["published_at"])
            current_id = int(current["id"])
            older: list[dict[str, Any]] = []
            for row in rows:
                candidate_published = _parse_datetime(row["published_at"])
                if current_published is not None and candidate_published is not None:
                    precedes = candidate_published < current_published or (
                        candidate_published == current_published
                        and int(row["id"]) < current_id
                    )
                else:
                    precedes = int(row["id"]) < current_id
                if precedes:
                    older.append(self._row(row))
                if len(older) >= bounded:
                    break
            return older

    def begin_fantasypros_corpus_run(
        self,
        run_id: str,
        query_keys: tuple[str, ...],
    ) -> set[str]:
        """Create or resume an exact FantasyPros crawl manifest."""
        if not run_id or not query_keys or len(set(query_keys)) != len(query_keys):
            raise ValueError("Corpus run and unique query keys are required")
        if any(not key or len(key) > 512 for key in query_keys):
            raise ValueError("Corpus query keys must be 1-512 characters")
        manifest = json.dumps(list(query_keys), separators=(",", ":"))
        now = _utc_now()
        with self._lock:
            existing = self._connection.execute(
                "SELECT manifest_json, status FROM fantasypros_corpus_runs "
                "WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if existing is None:
                self._connection.execute(
                    """
                    INSERT INTO fantasypros_corpus_runs(
                        run_id, manifest_json, status, stop_reason,
                        started_at, updated_at
                    ) VALUES (?, ?, 'running', '', ?, ?)
                    """,
                    (run_id, manifest, now, now),
                )
            elif str(existing["manifest_json"]) != manifest:
                raise ValueError("A resumed corpus run must use its original plan")
            elif str(existing["status"]) != "completed":
                self._connection.execute(
                    """
                    UPDATE fantasypros_corpus_runs
                    SET status = 'running', stop_reason = '', updated_at = ?
                    WHERE run_id = ?
                    """,
                    (now, run_id),
                )
            rows = self._connection.execute(
                "SELECT query_key FROM fantasypros_corpus_batches WHERE run_id = ?",
                (run_id,),
            ).fetchall()
            self._connection.commit()
            return {str(row["query_key"]) for row in rows}

    def store_fantasypros_corpus_batch(
        self,
        run_id: str,
        query_key: str,
        items: list[dict[str, Any]],
        *,
        fetched_at: datetime,
    ) -> dict[str, int]:
        """Atomically save one response and its resumable crawl checkpoint."""
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        fetched_iso = fetched_at.astimezone(timezone.utc).isoformat()
        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in items:
            provider_item_id = str(item.get("provider_item_id") or "").strip()
            canonical_text = str(item.get("canonical_text") or "").strip()
            content_hash = str(item.get("content_hash") or "").strip().lower()
            categories = item.get("categories")
            if (
                not provider_item_id
                or provider_item_id in seen_ids
                or str(item.get("sport") or "") != "NFL"
                or str(item.get("source_provider") or "") != "FantasyPros"
                or str(item.get("attribution") or "") != "FantasyPros"
                or str(item.get("usage_scope") or "") != "personal_reference"
                or not canonical_text
                or content_hash
                != hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
                or not isinstance(categories, list)
            ):
                raise ValueError("Invalid FantasyPros corpus record")
            seen_ids.add(provider_item_id)
            normalized.append(
                {
                    **item,
                    "provider_item_id": provider_item_id,
                    "canonical_text": canonical_text,
                    "content_hash": content_hash,
                    "categories_json": json.dumps(
                        [str(value) for value in categories],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )

        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                run = self._connection.execute(
                    "SELECT manifest_json FROM fantasypros_corpus_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if run is None:
                    raise ValueError("Corpus run does not exist")
                try:
                    manifest = tuple(json.loads(str(run["manifest_json"])))
                except (TypeError, ValueError, json.JSONDecodeError):
                    raise ValueError("Corpus run manifest is invalid") from None
                if query_key not in manifest:
                    raise ValueError("Corpus query is not part of this run")
                completed = self._connection.execute(
                    """
                    SELECT inserted_count, updated_count, duplicate_count
                    FROM fantasypros_corpus_batches
                    WHERE run_id = ? AND query_key = ?
                    """,
                    (run_id, query_key),
                ).fetchone()
                if completed is not None:
                    self._connection.commit()
                    return {
                        "inserted": int(completed["inserted_count"]),
                        "updated": int(completed["updated_count"]),
                        "duplicates": int(completed["duplicate_count"]),
                    }

                existing_hashes: dict[str, str] = {}
                if normalized:
                    placeholders = ",".join("?" for _ in normalized)
                    rows = self._connection.execute(
                        "SELECT provider_item_id, content_hash "
                        f"FROM fantasypros_news_corpus WHERE provider_item_id IN ({placeholders})",
                        tuple(item["provider_item_id"] for item in normalized),
                    ).fetchall()
                    existing_hashes = {
                        str(row["provider_item_id"]): str(row["content_hash"])
                        for row in rows
                    }

                inserted = sum(
                    item["provider_item_id"] not in existing_hashes
                    for item in normalized
                )
                updated = sum(
                    item["provider_item_id"] in existing_hashes
                    and existing_hashes[item["provider_item_id"]]
                    != item["content_hash"]
                    for item in normalized
                )
                duplicates = len(normalized) - inserted - updated
                now = _utc_now()
                for item in normalized:
                    self._connection.execute(
                        """
                        INSERT INTO fantasypros_news_corpus(
                            provider_item_id, sport, player_id, team_id, title,
                            description, impact, categories_json, author,
                            source_url, provider_created_at, first_seen_at,
                            last_seen_at, first_query_key, last_query_key,
                            canonical_text, content_hash, source_provider,
                            attribution, usage_scope, api_docs_url, updated_at
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?
                        )
                        ON CONFLICT(provider_item_id) DO UPDATE SET
                            sport = excluded.sport,
                            player_id = excluded.player_id,
                            team_id = excluded.team_id,
                            title = excluded.title,
                            description = excluded.description,
                            impact = excluded.impact,
                            categories_json = excluded.categories_json,
                            author = excluded.author,
                            source_url = excluded.source_url,
                            provider_created_at = excluded.provider_created_at,
                            last_seen_at = excluded.last_seen_at,
                            last_query_key = excluded.last_query_key,
                            canonical_text = excluded.canonical_text,
                            embedding_model = CASE
                                WHEN fantasypros_news_corpus.content_hash
                                     != excluded.content_hash THEN NULL
                                ELSE fantasypros_news_corpus.embedding_model END,
                            embedding_provider = CASE
                                WHEN fantasypros_news_corpus.content_hash
                                     != excluded.content_hash THEN NULL
                                ELSE fantasypros_news_corpus.embedding_provider END,
                            embedding_dimensions = CASE
                                WHEN fantasypros_news_corpus.content_hash
                                     != excluded.content_hash THEN NULL
                                ELSE fantasypros_news_corpus.embedding_dimensions END,
                            embedding_input_version = CASE
                                WHEN fantasypros_news_corpus.content_hash
                                     != excluded.content_hash THEN NULL
                                ELSE fantasypros_news_corpus.embedding_input_version END,
                            embedding_input_hash = CASE
                                WHEN fantasypros_news_corpus.content_hash
                                     != excluded.content_hash THEN NULL
                                ELSE fantasypros_news_corpus.embedding_input_hash END,
                            embedding_at = CASE
                                WHEN fantasypros_news_corpus.content_hash
                                     != excluded.content_hash THEN NULL
                                ELSE fantasypros_news_corpus.embedding_at END,
                            embedding = CASE
                                WHEN fantasypros_news_corpus.content_hash
                                     != excluded.content_hash THEN NULL
                                ELSE fantasypros_news_corpus.embedding END,
                            content_hash = excluded.content_hash,
                            source_provider = excluded.source_provider,
                            attribution = excluded.attribution,
                            usage_scope = excluded.usage_scope,
                            api_docs_url = excluded.api_docs_url,
                            updated_at = excluded.updated_at
                        """,
                        (
                            item["provider_item_id"],
                            item["sport"],
                            item.get("player_id"),
                            str(item.get("team_id") or ""),
                            str(item.get("title") or ""),
                            str(item.get("description") or ""),
                            str(item.get("impact") or ""),
                            item["categories_json"],
                            str(item.get("author") or ""),
                            str(item.get("source_url") or ""),
                            str(item.get("provider_created_at") or ""),
                            fetched_iso,
                            fetched_iso,
                            query_key,
                            query_key,
                            item["canonical_text"],
                            item["content_hash"],
                            item["source_provider"],
                            item["attribution"],
                            item["usage_scope"],
                            str(item.get("api_docs_url") or ""),
                            now,
                        ),
                    )
                self._connection.execute(
                    """
                    INSERT INTO fantasypros_corpus_batches(
                        run_id, query_key, fetched_at, item_count,
                        inserted_count, updated_count, duplicate_count,
                        completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        query_key,
                        fetched_iso,
                        len(normalized),
                        inserted,
                        updated,
                        duplicates,
                        now,
                    ),
                )
                for item in normalized:
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO fantasypros_corpus_observations(
                            run_id, query_key, provider_item_id, observed_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            query_key,
                            item["provider_item_id"],
                            fetched_iso,
                        ),
                    )
                self._connection.execute(
                    """
                    UPDATE fantasypros_corpus_runs
                    SET status = 'running', stop_reason = '', updated_at = ?
                    WHERE run_id = ?
                    """,
                    (now, run_id),
                )
                self._connection.commit()
                return {
                    "inserted": int(inserted),
                    "updated": int(updated),
                    "duplicates": int(duplicates),
                }
            except Exception:
                self._connection.rollback()
                raise

    def _set_fantasypros_corpus_run_status(
        self,
        run_id: str,
        status: str,
        reason: str = "",
    ) -> None:
        if status not in {"paused", "failed"}:
            raise ValueError("Invalid incomplete corpus run status")
        safe_reason = "".join(
            character
            for character in str(reason or "")[:64]
            if character.isalnum() or character in {"_", "-"}
        )
        with self._lock:
            self._connection.execute(
                """
                UPDATE fantasypros_corpus_runs
                SET status = ?, stop_reason = ?, updated_at = ?
                WHERE run_id = ? AND status != 'completed'
                """,
                (status, safe_reason, _utc_now(), run_id),
            )
            self._connection.commit()

    def pause_fantasypros_corpus_run(self, run_id: str, reason: str) -> None:
        self._set_fantasypros_corpus_run_status(run_id, "paused", reason)

    def fail_fantasypros_corpus_run(self, run_id: str, reason: str) -> None:
        self._set_fantasypros_corpus_run_status(run_id, "failed", reason)

    def complete_fantasypros_corpus_run(self, run_id: str) -> bool:
        """Complete a run only after every manifest query was checkpointed."""
        with self._lock:
            run = self._connection.execute(
                "SELECT manifest_json FROM fantasypros_corpus_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                return False
            try:
                manifest = set(json.loads(str(run["manifest_json"])))
            except (TypeError, ValueError, json.JSONDecodeError):
                return False
            completed = {
                str(row["query_key"])
                for row in self._connection.execute(
                    "SELECT query_key FROM fantasypros_corpus_batches WHERE run_id = ?",
                    (run_id,),
                ).fetchall()
            }
            if completed != manifest:
                return False
            now = _utc_now()
            self._connection.execute(
                """
                UPDATE fantasypros_corpus_runs
                SET status = 'completed', stop_reason = '',
                    completed_at = COALESCE(completed_at, ?), updated_at = ?
                WHERE run_id = ?
                """,
                (now, now, run_id),
            )
            self._connection.commit()
            return True

    def fantasypros_corpus_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM fantasypros_corpus_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            return self._row(row) if row is not None else None

    def latest_fantasypros_corpus_run(
        self,
        prefix: str,
    ) -> dict[str, Any] | None:
        """Return the newest valid run whose ID starts with a literal prefix.

        ``substr`` treats percent and underscore as ordinary characters, unlike
        ``LIKE``.  A malformed persisted manifest fails closed instead of
        handing a caller an unsafe or partial resume plan.
        """
        candidate = str(prefix or "")
        if not candidate or len(candidate) > 128 or any(
            ord(character) < 32 for character in candidate
        ):
            return None
        with self._lock:
            row = self._connection.execute(
                """
                SELECT rowid, * FROM fantasypros_corpus_runs
                WHERE substr(run_id, 1, length(?)) = ?
                ORDER BY started_at DESC, rowid DESC
                LIMIT 1
                """,
                (candidate, candidate),
            ).fetchone()
            if row is None:
                return None
            try:
                raw_manifest = json.loads(str(row["manifest_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
            if (
                not isinstance(raw_manifest, list)
                or not raw_manifest
                or not all(
                    isinstance(key, str) and 0 < len(key) <= 512
                    for key in raw_manifest
                )
                or len(set(raw_manifest)) != len(raw_manifest)
            ):
                return None
            status = str(row["status"])
            if status not in {"running", "paused", "failed", "completed"}:
                return None
            return {
                "run_id": str(row["run_id"]),
                "manifest": tuple(raw_manifest),
                "status": status,
                "stop_reason": str(row["stop_reason"] or ""),
                "started_at": str(row["started_at"]),
                "updated_at": str(row["updated_at"]),
                "completed_at": (
                    str(row["completed_at"])
                    if row["completed_at"] is not None
                    else None
                ),
            }

    @staticmethod
    def _fantasypros_spend_decimal(
        value: Any,
        *,
        name: str,
        allow_zero: bool,
    ) -> Decimal:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number")
        try:
            parsed = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError(f"{name} must be a finite number") from None
        if not parsed.is_finite() or parsed < 0 or (not allow_zero and parsed == 0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a finite {qualifier} number")
        return parsed

    def reserve_fantasypros_corpus_embedding_spend(
        self,
        *,
        model: str,
        input_price_per_million_usd: float,
        conservative_tokens: int,
        lifetime_cap_usd: float,
    ) -> int | None:
        """Durably reserve lifetime budget before one paid embedding request.

        Every inserted row remains part of the charged total forever, whatever
        its later status.  ``BEGIN IMMEDIATE`` makes the sum-and-insert decision
        atomic across independent EventStore connections and process restarts.
        Costs are enforced in integer nano-dollars, rounding each reservation
        up and the configured cap down so floating-point error cannot permit an
        over-budget request.
        """
        normalized_model = str(model or "").strip()
        if not normalized_model or len(normalized_model) > 512:
            raise ValueError("embedding model must be 1-512 characters")
        if (
            isinstance(conservative_tokens, bool)
            or not isinstance(conservative_tokens, int)
            or conservative_tokens <= 0
        ):
            raise ValueError("conservative_tokens must be a positive integer")
        price = self._fantasypros_spend_decimal(
            input_price_per_million_usd,
            name="input_price_per_million_usd",
            allow_zero=True,
        )
        cap = self._fantasypros_spend_decimal(
            lifetime_cap_usd,
            name="lifetime_cap_usd",
            allow_zero=False,
        )
        nano_per_usd = Decimal(1_000_000_000)
        reserved_cost = Decimal(conservative_tokens) * price / Decimal(1_000_000)
        reserved_nano = int(
            (reserved_cost * nano_per_usd).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        cap_nano = int(
            (cap * nano_per_usd).to_integral_value(rounding=ROUND_FLOOR)
        )
        sqlite_integer_max = 9_223_372_036_854_775_807
        if cap_nano <= 0 or reserved_nano > sqlite_integer_max or cap_nano > sqlite_integer_max:
            raise ValueError("embedding spend values exceed the supported range")

        now = _utc_now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    """
                    SELECT COALESCE(SUM(reserved_cost_nano_usd), 0) AS charged
                    FROM fantasypros_corpus_embedding_spend
                    """
                ).fetchone()
                charged_nano = int(row["charged"] if row is not None else 0)
                if charged_nano + reserved_nano > cap_nano:
                    self._connection.rollback()
                    return None
                cursor = self._connection.execute(
                    """
                    INSERT INTO fantasypros_corpus_embedding_spend(
                        model, input_price_per_million_usd,
                        conservative_tokens, reserved_cost_usd,
                        reserved_cost_nano_usd, lifetime_cap_usd,
                        lifetime_cap_nano_usd, status, reserved_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?)
                    """,
                    (
                        normalized_model,
                        float(price),
                        conservative_tokens,
                        float(reserved_cost),
                        reserved_nano,
                        float(cap),
                        cap_nano,
                        now,
                        now,
                    ),
                )
                request_id = int(cursor.lastrowid)
                self._connection.commit()
                return request_id
            except BaseException:
                self._connection.rollback()
                raise

    def finish_fantasypros_corpus_embedding_spend(
        self,
        request_id: int,
        *,
        actual_prompt_tokens: int | None,
        status: str,
    ) -> bool:
        """Finalize one reservation without ever releasing its charged cost."""
        if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id <= 0:
            return False
        normalized_status = str(status or "").strip().lower()
        if normalized_status not in {"completed", "failed"}:
            raise ValueError("embedding spend status must be completed or failed")
        if actual_prompt_tokens is not None and (
            isinstance(actual_prompt_tokens, bool)
            or not isinstance(actual_prompt_tokens, int)
            or actual_prompt_tokens < 0
        ):
            raise ValueError("actual_prompt_tokens must be a non-negative integer or None")
        if normalized_status == "completed" and actual_prompt_tokens is None:
            raise ValueError("completed embedding spend requires actual prompt tokens")

        now = _utc_now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    """
                    SELECT status, actual_prompt_tokens
                    FROM fantasypros_corpus_embedding_spend
                    WHERE request_id = ?
                    """,
                    (request_id,),
                ).fetchone()
                if row is None:
                    self._connection.rollback()
                    return False
                previous_status = str(row["status"])
                previous_tokens = row["actual_prompt_tokens"]
                if previous_status != "reserved":
                    idempotent = (
                        previous_status == normalized_status
                        and (
                            int(previous_tokens)
                            if previous_tokens is not None
                            else None
                        )
                        == actual_prompt_tokens
                    )
                    self._connection.rollback()
                    return idempotent
                self._connection.execute(
                    """
                    UPDATE fantasypros_corpus_embedding_spend
                    SET status = ?, actual_prompt_tokens = ?,
                        finished_at = ?, updated_at = ?
                    WHERE request_id = ? AND status = 'reserved'
                    """,
                    (
                        normalized_status,
                        actual_prompt_tokens,
                        now,
                        now,
                        request_id,
                    ),
                )
                self._connection.commit()
                return True
            except BaseException:
                self._connection.rollback()
                raise

    def fantasypros_corpus_embedding_spend_status(
        self,
        *,
        lifetime_cap_usd: float | None = None,
    ) -> dict[str, Any]:
        """Summarize the immutable lifetime charge ledger without mutation."""
        cap: Decimal | None = None
        cap_nano: int | None = None
        if lifetime_cap_usd is not None:
            cap = self._fantasypros_spend_decimal(
                lifetime_cap_usd,
                name="lifetime_cap_usd",
                allow_zero=False,
            )
            cap_nano = int(
                (cap * Decimal(1_000_000_000)).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            )
            if cap_nano <= 0:
                raise ValueError("lifetime_cap_usd is below one nano-dollar")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT
                    COUNT(*) AS reservations,
                    COALESCE(SUM(CASE WHEN status = 'reserved' THEN 1 ELSE 0 END), 0)
                        AS open_reservations,
                    COALESCE(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END), 0)
                        AS completed_reservations,
                    COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0)
                        AS failed_reservations,
                    COALESCE(SUM(conservative_tokens), 0) AS reserved_tokens,
                    COALESCE(SUM(actual_prompt_tokens), 0) AS actual_prompt_tokens,
                    COALESCE(SUM(
                        CASE WHEN actual_prompt_tokens IS NULL THEN 1 ELSE 0 END
                    ), 0)
                        AS unknown_actual_reservations,
                    COALESCE(SUM(reserved_cost_nano_usd), 0)
                        AS charged_cost_nano_usd,
                    COALESCE(SUM(
                        COALESCE(actual_prompt_tokens, 0)
                        * input_price_per_million_usd / 1000000.0
                    ), 0.0) AS actual_cost_usd
                FROM fantasypros_corpus_embedding_spend
                """
            ).fetchone()
        charged_nano = int(row["charged_cost_nano_usd"] if row is not None else 0)
        remaining_nano = (
            max(0, int(cap_nano) - charged_nano)
            if cap_nano is not None
            else None
        )
        return {
            "reservations": int(row["reservations"] if row is not None else 0),
            "open_reservations": int(
                row["open_reservations"] if row is not None else 0
            ),
            "completed_reservations": int(
                row["completed_reservations"] if row is not None else 0
            ),
            "failed_reservations": int(
                row["failed_reservations"] if row is not None else 0
            ),
            "reserved_tokens": int(
                row["reserved_tokens"] if row is not None else 0
            ),
            "actual_prompt_tokens": int(
                row["actual_prompt_tokens"] if row is not None else 0
            ),
            "unknown_actual_reservations": int(
                row["unknown_actual_reservations"] if row is not None else 0
            ),
            "charged_cost_usd": charged_nano / 1_000_000_000,
            "actual_cost_usd": float(row["actual_cost_usd"] if row is not None else 0),
            "lifetime_cap_usd": float(cap) if cap is not None else None,
            "remaining_usd": (
                remaining_nano / 1_000_000_000
                if remaining_nano is not None
                else None
            ),
            "budget_exhausted": (
                charged_nano >= int(cap_nano) if cap_nano is not None else False
            ),
        }

    def fantasypros_corpus_items(
        self,
        *,
        limit: int = 100_000,
    ) -> list[dict[str, Any]]:
        """Read only reference rows; live journal readers never call this."""
        bounded = max(1, min(int(limit), 100_000))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM fantasypros_news_corpus
                ORDER BY provider_created_at, id
                LIMIT ?
                """,
                (bounded,),
            ).fetchall()
            return [self._row(row) for row in rows]

    def fantasypros_corpus_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS total FROM fantasypros_news_corpus"
            ).fetchone()
            return int(row["total"] if row is not None else 0)

    def fantasypros_corpus_observations(
        self,
        *,
        provider_item_id: str = "",
        limit: int = 100_000,
    ) -> list[dict[str, Any]]:
        """Return preserved query/item weak-label observations."""
        bounded = max(1, min(int(limit), 100_000))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM fantasypros_corpus_observations
                WHERE (? = '' OR provider_item_id = ?)
                ORDER BY observed_at, run_id, query_key, provider_item_id
                LIMIT ?
                """,
                (provider_item_id, provider_item_id, bounded),
            ).fetchall()
            return [self._row(row) for row in rows]

    def fantasypros_corpus_embedding_backlog(
        self,
        *,
        model: str,
        dimensions: int,
        input_version: str,
        limit: int = 5000,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Return isolated corpus rows needing a comparable current vector."""
        bounded = max(1, min(int(limit), 100_000))
        predicate = "1 = 1" if force else """
            (embedding IS NULL
             OR embedding_model IS NULL OR embedding_model != ?
             OR embedding_provider IS NULL OR TRIM(embedding_provider) = ''
             OR embedding_dimensions IS NULL OR embedding_dimensions != ?
             OR embedding_input_version IS NULL OR embedding_input_version != ?
             OR embedding_input_hash IS NULL
             OR embedding_input_hash != content_hash
             OR embedding_at IS NULL OR TRIM(embedding_at) = '')
        """
        parameters: tuple[Any, ...] = (
            (bounded,)
            if force
            else (model, int(dimensions), input_version, bounded)
        )
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM fantasypros_news_corpus
                WHERE {predicate}
                ORDER BY provider_created_at, id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            return [self._row(row) for row in rows]

    def store_fantasypros_corpus_embedding(
        self,
        provider_item_id: str,
        model: str,
        embedding: bytes,
        *,
        provider: str,
        dimensions: int,
        input_version: str,
        input_hash: str,
    ) -> bool:
        """Attach a vector only if its canonical input is still current."""
        if (
            not provider_item_id
            or not model
            or not embedding
            or not provider
            or dimensions <= 0
            or not input_version
            or not input_hash
        ):
            return False
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE fantasypros_news_corpus
                SET embedding_model = ?, embedding_provider = ?,
                    embedding_dimensions = ?, embedding_input_version = ?,
                    embedding_input_hash = ?, embedding_at = ?, embedding = ?
                WHERE provider_item_id = ? AND content_hash = ?
                """,
                (
                    model,
                    provider,
                    int(dimensions),
                    input_version,
                    input_hash,
                    _utc_now(),
                    sqlite3.Binary(embedding),
                    provider_item_id,
                    input_hash,
                ),
            )
            self._connection.commit()
            return cursor.rowcount > 0

    def fantasypros_corpus_embedding_count(
        self,
        *,
        model: str = "",
        dimensions: int | None = None,
        input_version: str = "",
    ) -> int:
        clauses = ["embedding IS NOT NULL"]
        values: list[Any] = []
        if model:
            clauses.append("embedding_model = ?")
            values.append(model)
        if dimensions is not None:
            clauses.append("embedding_dimensions = ?")
            values.append(int(dimensions))
        if input_version:
            clauses.append("embedding_input_version = ?")
            values.append(input_version)
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS total FROM fantasypros_news_corpus WHERE "
                + " AND ".join(clauses),
                tuple(values),
            ).fetchone()
            return int(row["total"] if row is not None else 0)

    def embedding_backlog(
        self,
        *,
        model: str,
        dimensions: int,
        input_version: str,
        limit: int = 2000,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Return saved reports needing a vector for the configured space."""
        bounded = max(1, min(int(limit), 100_000))
        predicate = "1 = 1" if force else """
            (embedding IS NULL
             OR embedding_model IS NULL OR embedding_model != ?
             OR embedding_provider IS NULL OR TRIM(embedding_provider) = ''
             OR embedding_dimensions IS NULL OR embedding_dimensions != ?
             OR embedding_input_version IS NULL OR embedding_input_version != ?
             OR embedding_input_hash IS NULL OR TRIM(embedding_input_hash) = ''
             OR embedding_at IS NULL OR TRIM(embedding_at) = '')
        """
        parameters: tuple[Any, ...]
        if force:
            parameters = (bounded,)
        else:
            parameters = (model, int(dimensions), input_version, bounded)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM news_events
                WHERE {predicate}
                ORDER BY received_at, id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            return [self._row(row) for row in rows]

    def embedding_count(
        self,
        *,
        model: str = "",
        dimensions: int | None = None,
        input_version: str = "",
    ) -> int:
        clauses = ["embedding IS NOT NULL"]
        values: list[Any] = []
        if model:
            clauses.append("embedding_model = ?")
            values.append(model)
        if dimensions is not None:
            clauses.append("embedding_dimensions = ?")
            values.append(int(dimensions))
        if input_version:
            clauses.append("embedding_input_version = ?")
            values.append(input_version)
        with self._lock:
            row = self._connection.execute(
                f"SELECT COUNT(*) AS total FROM news_events WHERE {' AND '.join(clauses)}",
                tuple(values),
            ).fetchone()
            return int(row["total"] if row is not None else 0)

    def all_reports(self, *, limit: int = 100_000) -> list[dict[str, Any]]:
        """Return the complete saved-news archive in chronological order."""
        bounded = max(1, min(int(limit), 100_000))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM news_events
                ORDER BY received_at, id
                LIMIT ?
                """,
                (bounded,),
            ).fetchall()
            return [self._row(row) for row in rows]

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
