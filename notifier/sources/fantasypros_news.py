"""Resumable FantasyPros NFL news ingestion for a reference-only corpus.

The public v2 news endpoint exposes no pagination cursor and returns at most
100 items for one general, category, or player query.  A useful historical
corpus therefore grows through explicit query plans and repeated snapshots.
These rows live in a separate table from ``news_events`` and can never become
alerts, recaps, or live urgency evidence merely by being downloaded.

The transport deliberately borrows the application's existing
``FantasyProsCache`` instance.  Its persistent request ledger, 425-request
application cap, and 1.05-second start cadence remain the single authority;
this module never creates a second API client or budget ledger.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from ..event_store import EventStore
from ..logging_utils import structured_log
from .fantasypros import (
    CORPUS_REQUEST_BUCKET,
    FantasyProsCache,
    FantasyProsRequestError,
)

NEWS_PATH = "nfl/news"
MAX_ITEMS_PER_REQUEST = 100
DEFAULT_LIVE_REQUEST_RESERVE = 75
DEFAULT_MAX_INGEST_REQUESTS = 100
NEWS_CATEGORIES = ("injury", "recap", "transaction", "rumor", "breaking")
NEWS_ORDERINGS = ("created", "updated")
ATTRIBUTION = "FantasyPros"
API_DOCS_URL = "https://api.fantasypros.com/public/v2/docs"
USAGE_SCOPE = "personal_reference"
CORPUS_INPUT_VERSION = "fantasypros-news-v1"

_SAFE_ERROR_CODES = frozenset(
    {
        "budget_exhausted",
        "budget_reserved",
        "cache_write_failed",
        "closed",
        "invalid_response",
        "ledger_untrusted",
        "request_failed",
        "request_limit",
    }
)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_HTML_TAG = re.compile(r"<[^>]*>")
_SPACE = re.compile(r"\s+")


class FantasyProsCorpusError(RuntimeError):
    """A secret-free provider or persistence failure."""

    def __init__(self, code: str, *, request_reserved: bool = False) -> None:
        safe_code = code if code in _SAFE_ERROR_CODES else "request_failed"
        super().__init__(safe_code)
        self.code = safe_code
        self.request_reserved = bool(request_reserved)


@dataclass(frozen=True)
class FantasyProsNewsQuery:
    """One documented v2 NFL news request."""

    category: str | None = None
    order_by: str = "created"
    fpid: int | None = None
    limit: int = MAX_ITEMS_PER_REQUEST

    def __post_init__(self) -> None:
        category = str(self.category or "").strip().lower() or None
        ordering = str(self.order_by or "").strip().lower()
        if category is not None and category not in NEWS_CATEGORIES:
            raise ValueError(f"Unsupported FantasyPros news category: {category}")
        if ordering not in NEWS_ORDERINGS:
            raise ValueError(f"Unsupported FantasyPros news ordering: {ordering}")
        if isinstance(self.limit, bool) or not 1 <= int(self.limit) <= 100:
            raise ValueError("FantasyPros news limit must be between 1 and 100")
        if self.fpid is not None and (
            isinstance(self.fpid, bool) or int(self.fpid) <= 0
        ):
            raise ValueError("FantasyPros player id must be a positive integer")
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "order_by", ordering)
        object.__setattr__(self, "limit", int(self.limit))
        object.__setattr__(self, "fpid", int(self.fpid) if self.fpid else None)

    @property
    def key(self) -> str:
        return (
            f"order={self.order_by};category={self.category or 'all'};"
            f"fpid={self.fpid or 'all'};limit={self.limit}"
        )

    @property
    def params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "limit": self.limit,
            "order_by": self.order_by,
        }
        if self.category is not None:
            params["category"] = self.category
        if self.fpid is not None:
            params["fpid"] = self.fpid
        return params


@dataclass(frozen=True)
class FantasyProsCorpusPlan:
    """A bounded request plan and its provider-side maximum yield."""

    queries: tuple[FantasyProsNewsQuery, ...]

    @property
    def request_count(self) -> int:
        return len(self.queries)

    @property
    def maximum_candidate_items(self) -> int:
        return sum(query.limit for query in self.queries)


@dataclass(frozen=True)
class FantasyProsCorpusResult:
    """Secret-free progress from one resumable ingestion invocation."""

    run_id: str
    planned_queries: int
    completed_queries: int
    requests_made: int
    inserted: int
    updated: int
    duplicates: int
    remaining_queries: int
    complete: bool
    stop_reason: str = ""


def build_news_plan(
    *,
    player_ids: Iterable[int] = (),
    include_general: bool = True,
    categories: Sequence[str] = NEWS_CATEGORIES,
    orderings: Sequence[str] = ("created",),
    player_categories: Sequence[str | None] = (None,),
    limit: int = MAX_ITEMS_PER_REQUEST,
) -> FantasyProsCorpusPlan:
    """Build a deterministic, duplicate-free global/category/player plan.

    A global sweep is one unfiltered query plus one query for each selected
    category.  Player queries can optionally repeat selected categories; this
    is how a caller can expand beyond the endpoint's latest 100 global rows.
    The helper only estimates the *maximum* number of returned rows because
    categories and players can overlap heavily.
    """
    normalized_orderings = tuple(dict.fromkeys(str(value).lower() for value in orderings))
    normalized_categories = tuple(
        dict.fromkeys(str(value).lower() for value in categories)
    )
    normalized_player_categories = tuple(
        dict.fromkeys(
            str(value).lower() if value is not None else None
            for value in player_categories
        )
    )
    normalized_players: list[int] = []
    for raw_player_id in player_ids:
        if isinstance(raw_player_id, bool) or int(raw_player_id) <= 0:
            raise ValueError("FantasyPros player ids must be positive integers")
        player_id = int(raw_player_id)
        if player_id not in normalized_players:
            normalized_players.append(player_id)

    queries: list[FantasyProsNewsQuery] = []
    for ordering in normalized_orderings:
        if include_general:
            queries.append(
                FantasyProsNewsQuery(order_by=ordering, limit=limit)
            )
        queries.extend(
            FantasyProsNewsQuery(
                category=category,
                order_by=ordering,
                limit=limit,
            )
            for category in normalized_categories
        )
        for player_id in normalized_players:
            queries.extend(
                FantasyProsNewsQuery(
                    category=category,
                    order_by=ordering,
                    fpid=player_id,
                    limit=limit,
                )
                for category in normalized_player_categories
            )

    unique = tuple({query.key: query for query in queries}.values())
    if not unique:
        raise ValueError("FantasyPros corpus plan must contain at least one query")
    return FantasyProsCorpusPlan(unique)


def clean_reference_text(value: Any) -> str:
    """Return deterministic plain text suitable for storage and embedding."""
    candidate = html.unescape(str(value or ""))
    candidate = _HTML_TAG.sub(" ", candidate)
    return _SPACE.sub(" ", candidate).strip()


def _clean_categories(values: Sequence[Any]) -> tuple[str, ...]:
    by_key: dict[str, str] = {}
    for raw in values:
        cleaned = clean_reference_text(raw)
        if cleaned:
            by_key.setdefault(cleaned.casefold(), cleaned)
    return tuple(sorted(by_key.values(), key=str.casefold))


def canonical_corpus_text(
    *,
    title: str,
    description: str,
    impact: str,
    categories: Sequence[str],
) -> str:
    """Build clean vector text without leaking evaluation category labels."""
    parts: list[str] = []
    cleaned_title = clean_reference_text(title)
    cleaned_description = clean_reference_text(description)
    cleaned_impact = clean_reference_text(impact)
    # Categories remain in structured storage and query observations. They
    # are excluded from the vector input so category-consistency evaluation
    # does not put the answer directly into the embedding.
    _ = categories
    if cleaned_title:
        parts.append(f"Headline: {cleaned_title}")
    if cleaned_description and cleaned_description.casefold() != cleaned_title.casefold():
        parts.append(f"Report: {cleaned_description}")
    if cleaned_impact:
        parts.append(f"FantasyPros impact: {cleaned_impact}")
    return "\n".join(parts)


def _optional_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        return None
    return int(number)


def _provider_created_at(value: Any) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("missing created time")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError("invalid created time") from None
    if parsed.tzinfo is None:
        # FantasyPros documents ``created_formated`` as UTC while ``created``
        # uses a timezone-free SQL timestamp.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_news_response(
    payload: Any,
    *,
    query: FantasyProsNewsQuery,
    fetched_at: datetime,
) -> list[dict[str, Any]]:
    """Validate one documented NFL response and return storage records."""
    if not isinstance(payload, dict) or str(payload.get("sport") or "").upper() != "NFL":
        raise FantasyProsCorpusError("invalid_response")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or len(raw_items) > query.limit:
        raise FantasyProsCorpusError("invalid_response")
    raw_count = payload.get("count")
    if isinstance(raw_count, bool):
        raise FantasyProsCorpusError("invalid_response")
    try:
        count = int(raw_count)
    except (TypeError, ValueError):
        raise FantasyProsCorpusError("invalid_response") from None
    if count < len(raw_items) or count < 0:
        raise FantasyProsCorpusError("invalid_response")

    normalized_fetched_at = fetched_at
    if normalized_fetched_at.tzinfo is None:
        normalized_fetched_at = normalized_fetched_at.replace(tzinfo=timezone.utc)
    fetched_iso = normalized_fetched_at.astimezone(timezone.utc).isoformat()
    parsed: dict[str, dict[str, Any]] = {}
    for raw in raw_items:
        try:
            if not isinstance(raw, dict):
                raise ValueError("item is not an object")
            required_fields = {
                "id",
                "created",
                "author",
                "player_id",
                "team_id",
                "title",
                "sport_id",
                "categories",
                "link",
                "desc",
                "impact",
            }
            if not required_fields.issubset(raw):
                raise ValueError("missing documented item fields")
            provider_item_id_value = _optional_positive_int(raw.get("id"))
            if provider_item_id_value is None:
                raise ValueError("invalid item id")
            provider_item_id = str(provider_item_id_value)
            if str(raw.get("sport_id") or "").strip().upper() != "NFL":
                raise ValueError("invalid item sport")
            title = clean_reference_text(raw.get("title"))
            if not title:
                raise ValueError("missing title")
            raw_categories = raw.get("categories")
            # The live public API returns JSON null for uncategorized rows,
            # including many results inside a documented category-filtered
            # response. Preserve the query category as an observation label,
            # while representing absent item metadata as an empty list.
            if raw_categories is None:
                raw_categories = []
            elif not isinstance(raw_categories, list):
                raise ValueError("invalid categories")
            categories = _clean_categories(raw_categories)
            description = clean_reference_text(raw.get("desc"))
            impact = clean_reference_text(raw.get("impact"))
            canonical_text = canonical_corpus_text(
                title=title,
                description=description,
                impact=impact,
                categories=categories,
            )
            if not canonical_text:
                raise ValueError("empty canonical text")
            content_hash = hashlib.sha256(
                canonical_text.encode("utf-8")
            ).hexdigest()
            record = {
                "provider_item_id": provider_item_id,
                "sport": "NFL",
                "player_id": _optional_positive_int(raw.get("player_id")),
                "team_id": clean_reference_text(raw.get("team_id")).upper(),
                "title": title,
                "description": description,
                "impact": impact,
                "categories": list(categories),
                "author": clean_reference_text(raw.get("author")),
                "source_url": str(raw.get("link") or "").strip(),
                "provider_created_at": _provider_created_at(
                    raw.get("created")
                ).isoformat(),
                "fetched_at": fetched_iso,
                "canonical_text": canonical_text,
                "content_hash": content_hash,
                "source_provider": ATTRIBUTION,
                "attribution": ATTRIBUTION,
                "usage_scope": USAGE_SCOPE,
                "api_docs_url": API_DOCS_URL,
            }
        except (TypeError, ValueError, OverflowError):
            raise FantasyProsCorpusError("invalid_response") from None

        previous = parsed.get(provider_item_id)
        if previous is not None and previous["content_hash"] != content_hash:
            # Conflicting copies of one provider identity are not safe to pick
            # by array order.
            raise FantasyProsCorpusError("invalid_response")
        parsed[provider_item_id] = record
    return list(parsed.values())


class _SharedFantasyProsNewsTransport:
    """Borrow one ranking-cache transport and its persistent request ledger."""

    def __init__(self, cache: FantasyProsCache) -> None:
        self.cache = cache

    def fetch(
        self,
        query: FantasyProsNewsQuery,
        *,
        request_ceiling: int,
        request_bucket_limit: int,
    ) -> Any:
        try:
            return self.cache.get_json(
                NEWS_PATH,
                params=query.params,
                request_ceiling=request_ceiling,
                request_bucket=CORPUS_REQUEST_BUCKET,
                request_bucket_limit=request_bucket_limit,
            )
        except FantasyProsRequestError as error:
            raise FantasyProsCorpusError(
                error.code,
                request_reserved=error.request_reserved,
            ) from None


class FantasyProsNewsCorpusIngestor:
    """Execute and safely resume a bounded reference-corpus query plan."""

    def __init__(
        self,
        store: EventStore,
        fantasypros: FantasyProsCache,
        *,
        clock: Any | None = None,
    ) -> None:
        self.store = store
        self.fantasypros = fantasypros
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._transport = _SharedFantasyProsNewsTransport(fantasypros)

    def ingest(
        self,
        run_id: str,
        plan: FantasyProsCorpusPlan,
        *,
        max_requests: int = DEFAULT_MAX_INGEST_REQUESTS,
        request_bucket_limit: int | None = None,
        live_request_reserve: int = DEFAULT_LIVE_REQUEST_RESERVE,
        target_items: int | None = None,
    ) -> FantasyProsCorpusResult:
        """Run a bounded invocation under a persistent rolling bucket limit.

        Completed batches are skipped on resume.  Each response and its crawl
        checkpoint commit in one SQLite transaction, so a crash can at worst
        cost one repeated provider request; provider-item uniqueness prevents
        a repeated request from duplicating corpus rows.
        """
        if not _RUN_ID.fullmatch(str(run_id or "")):
            raise ValueError("Corpus run id must be a safe 1-128 character token")
        if isinstance(max_requests, bool) or int(max_requests) <= 0:
            raise ValueError("max_requests must be a positive integer")
        if request_bucket_limit is not None and (
            isinstance(request_bucket_limit, bool)
            or int(request_bucket_limit) <= 0
        ):
            raise ValueError("request_bucket_limit must be a positive integer")
        if isinstance(live_request_reserve, bool) or int(live_request_reserve) < 0:
            raise ValueError("live_request_reserve must be a non-negative integer")
        if int(live_request_reserve) >= self.fantasypros.app_daily_cap:
            raise ValueError("live_request_reserve must be below the shared daily cap")
        if target_items is not None and (
            isinstance(target_items, bool) or int(target_items) <= 0
        ):
            raise ValueError("target_items must be a positive integer")

        queries_by_key = {query.key: query for query in plan.queries}
        if len(queries_by_key) != len(plan.queries) or not queries_by_key:
            raise ValueError("Corpus plan query keys must be unique and non-empty")
        completed = self.store.begin_fantasypros_corpus_run(
            run_id,
            tuple(queries_by_key),
        )
        requests_made = 0
        inserted = 0
        updated = 0
        duplicates = 0
        stop_reason = ""
        request_ceiling = (
            self.fantasypros.app_daily_cap - int(live_request_reserve)
        )
        rolling_bucket_limit = int(
            max_requests if request_bucket_limit is None else request_bucket_limit
        )

        for query_key, query in queries_by_key.items():
            if query_key in completed:
                continue
            if (
                target_items is not None
                and self.store.fantasypros_corpus_count() >= int(target_items)
            ):
                stop_reason = "target_reached"
                break
            if requests_made >= int(max_requests):
                stop_reason = "request_limit"
                break
            status = self.fantasypros.status()
            if status.requests_used >= status.request_cap - int(live_request_reserve):
                stop_reason = "budget_reserved"
                break
            try:
                payload = self._transport.fetch(
                    query,
                    request_ceiling=request_ceiling,
                    request_bucket_limit=rolling_bucket_limit,
                )
                requests_made += 1
                fetched_at = self._clock()
                if not isinstance(fetched_at, datetime):
                    raise FantasyProsCorpusError("invalid_response")
                items = parse_news_response(
                    payload,
                    query=query,
                    fetched_at=fetched_at,
                )
                result = self.store.store_fantasypros_corpus_batch(
                    run_id,
                    query_key,
                    items,
                    fetched_at=fetched_at,
                )
            except FantasyProsCorpusError as error:
                if error.request_reserved:
                    requests_made += 1
                stop_reason = error.code
                self.store.fail_fantasypros_corpus_run(run_id, error.code)
                break
            completed.add(query_key)
            inserted += result["inserted"]
            updated += result["updated"]
            duplicates += result["duplicates"]
            if (
                target_items is not None
                and self.store.fantasypros_corpus_count() >= int(target_items)
            ):
                stop_reason = "target_reached"
                break

        remaining = len(queries_by_key) - len(completed)
        complete = remaining == 0
        if complete:
            self.store.complete_fantasypros_corpus_run(run_id)
        elif stop_reason in {"request_limit", "budget_reserved", "target_reached"}:
            self.store.pause_fantasypros_corpus_run(run_id, stop_reason)

        structured_log(
            logging.INFO if complete else logging.WARNING,
            "fantasypros.corpus_ingestion",
            runId=run_id,
            requestsMade=requests_made,
            completedQueries=len(completed),
            remainingQueries=remaining,
            inserted=inserted,
            updated=updated,
            duplicates=duplicates,
            complete=complete,
            reason=stop_reason,
        )
        return FantasyProsCorpusResult(
            run_id=run_id,
            planned_queries=len(queries_by_key),
            completed_queries=len(completed),
            requests_made=requests_made,
            inserted=inserted,
            updated=updated,
            duplicates=duplicates,
            remaining_queries=remaining,
            complete=complete,
            stop_reason=stop_reason,
        )


def manifest_fingerprint(plan: FantasyProsCorpusPlan) -> str:
    """Stable identifier helper for callers that persist their own run names."""
    payload = json.dumps(
        [query.key for query in plan.queries],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
