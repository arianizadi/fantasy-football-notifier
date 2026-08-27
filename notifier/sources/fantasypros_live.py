"""Budgeted FantasyPros NFL news backstop for the live alert pipeline.

X remains the fastest source and RotoWire remains an independent continuous
feed.  This source polls the documented FantasyPros NFL news endpoint on a
slower cadence so a local beat report that never reaches the configured X
accounts can still enter the same classify, roster, dedupe, and Telegram path.

The source borrows the application's one ``FantasyProsCache`` transport.  It
therefore shares the durable rolling request ledger, request-start cadence,
and API key with rankings and the isolated historical corpus.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import NewsItem
from .fantasypros import FantasyProsCache, FantasyProsRequestError
from .fantasypros_news import (
    NEWS_PATH,
    FantasyProsCorpusError,
    FantasyProsNewsQuery,
    parse_news_response,
)
from .reporters import PlayerNameIndex

LIVE_NEWS_REQUEST_BUCKET = "live_news"
LIVE_NEWS_STATE_FILENAME = "fantasypros-live-news.json"


@dataclass(frozen=True)
class FantasyProsLiveFetch:
    """One validated provider snapshot converted to live ``NewsItem`` rows."""

    items: tuple[NewsItem, ...]
    fetched_at: datetime


class FantasyProsLiveNews:
    """Fetch and deterministically attribute the latest FantasyPros NFL news."""

    def __init__(
        self,
        fantasypros: FantasyProsCache,
        *,
        enabled: bool,
        request_limit: int,
        request_reserve: int,
        state_dir: Path,
        clock: Any | None = None,
    ) -> None:
        self.fantasypros = fantasypros
        self.enabled = bool(enabled) and fantasypros.enabled
        self.request_limit = int(request_limit)
        self.request_reserve = int(request_reserve)
        if self.request_limit <= 0:
            raise ValueError("FantasyPros live-news request limit must be positive")
        if self.request_reserve <= 0:
            raise ValueError("FantasyPros live-news reserve must be positive")
        if (
            self.enabled
            and self.request_reserve >= self.fantasypros.status().request_cap
        ):
            raise ValueError("FantasyPros live-news reserve must be below the shared cap")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._names: PlayerNameIndex | None = None
        self._state_path = Path(state_dir) / LIVE_NEWS_STATE_FILENAME
        self._initialized = self._load_initialized()

    def _load_initialized(self) -> bool:
        if not self._state_path.exists():
            return False
        try:
            payload = json.loads(self._state_path.read_text())
            return bool(payload.get("initialized"))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def mark_initialized(self, *, fetched_at: datetime, item_count: int) -> bool:
        """Persist first-page priming before later polls may emit alerts."""
        payload = {
            "initialized": True,
            "primedAt": self._utc(fetched_at).isoformat(),
            "primedItems": max(0, int(item_count)),
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self._state_path.name}.",
                suffix=".tmp",
                dir=self._state_path.parent,
                text=True,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w") as handle:
                    json.dump(payload, handle, separators=(",", ":"))
                os.replace(temporary, self._state_path)
            finally:
                temporary.unlink(missing_ok=True)
        except OSError:
            return False
        self._initialized = True
        return True

    def set_player_index(self, player_index: dict[str, Any]) -> None:
        self._names = PlayerNameIndex(player_index)

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _to_item(self, record: dict[str, Any]) -> NewsItem:
        title = str(record.get("title") or "").strip()
        description = str(record.get("description") or "").strip()
        impact = str(record.get("impact") or "").strip()

        # FantasyPros titles conventionally begin with the article subject.
        # Restrict attribution to the title so a backup named only in the
        # impact paragraph cannot replace the report's actual subject.
        title_names = self._names.find(title) if self._names is not None else []
        player_name = title_names[0] if len(title_names) == 1 else ""
        subject_confident = len(title_names) == 1

        body_parts = [part for part in (description, impact) if part]
        body = "\n\n".join(dict.fromkeys(body_parts)) or title
        created = datetime.fromisoformat(
            str(record.get("provider_created_at") or "").replace("Z", "+00:00")
        )
        return NewsItem(
            source="fantasypros",
            guid=f"fantasypros:{record['provider_item_id']}",
            player_name=player_name,
            headline=title,
            body=body,
            url=str(record.get("source_url") or ""),
            published_at=self._utc(created),
            subject_confident=subject_confident,
        )

    def fetch(self) -> FantasyProsLiveFetch:
        """Fetch one newest-first page under the shared rolling budget."""
        if not self.enabled:
            raise FantasyProsRequestError("closed")
        fetched_at = self._clock()
        if not isinstance(fetched_at, datetime):
            raise FantasyProsRequestError("invalid_response")
        fetched_at = self._utc(fetched_at)
        query = FantasyProsNewsQuery(limit=100, order_by="created")
        request_cap = self.fantasypros.status().request_cap
        try:
            payload = self.fantasypros.get_json(
                NEWS_PATH,
                params=query.params,
                request_ceiling=request_cap - self.request_reserve,
                request_bucket=LIVE_NEWS_REQUEST_BUCKET,
                request_bucket_limit=self.request_limit,
            )
            records = parse_news_response(
                payload,
                query=query,
                fetched_at=fetched_at,
            )
            items = tuple(self._to_item(record) for record in records)
        except FantasyProsCorpusError as error:
            raise FantasyProsRequestError(
                error.code,
                request_reserved=error.request_reserved,
            ) from None
        except (KeyError, TypeError, ValueError, OverflowError):
            raise FantasyProsRequestError("invalid_response", request_reserved=True) from None
        return FantasyProsLiveFetch(items=items, fetched_at=fetched_at)
