"""Low-priority FantasyPros reference-corpus collection and embedding.

This coordinator is intentionally outside the breaking-news path.  It borrows
the application's single FantasyPros request ledger, writes only the isolated
reference tables, and never imports provider rows into alerts, recaps, search,
dedupe, or urgency labels.  A provider category is evaluation metadata, not a
claim that the row is human-labeled ground truth.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from .embeddings import EmbeddingUnavailable, OpenRouterEmbeddingClient
from .event_store import EventStore
from .logging_utils import structured_log
from .sources.fantasypros import (
    CORPUS_REQUEST_BUCKET,
    FantasyProsCache,
    FantasyProsRequestError,
)
from .sources.fantasypros_news import (
    CORPUS_INPUT_VERSION,
    NEWS_CATEGORIES,
    FantasyProsCorpusPlan,
    FantasyProsNewsCorpusIngestor,
    FantasyProsNewsQuery,
    build_news_plan,
    manifest_fingerprint,
)

PLAYER_INDEX_PATH = "nfl/players"
FANTASY_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
CORPUS_PLAN_VERSION = "v1"
CORPUS_EMBEDDING_BATCH_SIZE = 32
CORPUS_LOOP_RETRY_SECONDS = 60 * 60
CORPUS_LOOP_HEALTHY_SECONDS = 6 * 60 * 60
CORPUS_SNAPSHOT_INTERVAL = timedelta(hours=24)
BOOTSTRAP_RUN_PREFIX = f"bootstrap-{CORPUS_PLAN_VERSION}-"


@dataclass(frozen=True)
class FantasyProsCorpusStatus:
    enabled: bool
    running: bool
    corpus_items: int
    embedded_items: int
    requests_made: int
    inserted: int
    updated: int
    duplicates: int
    prompt_tokens: int
    conservative_tokens: int
    estimated_cost_usd: float
    last_success_at: datetime | None
    last_error: str


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        return None
    return int(number)


def _positions(value: Any) -> frozenset[str]:
    values = value if isinstance(value, list) else str(value or "").split(",")
    return frozenset(str(entry or "").strip().upper() for entry in values)


def parse_fantasypros_player_ids(
    payload: Any,
    *,
    limit: int,
) -> tuple[int, ...]:
    """Select current, ranked offensive players from the documented index."""
    if (
        not isinstance(payload, dict)
        or str(payload.get("sport") or "").strip().upper() != "NFL"
        or not isinstance(payload.get("players"), list)
    ):
        raise ValueError("invalid FantasyPros NFL player response")
    if isinstance(limit, bool) or int(limit) <= 0:
        raise ValueError("player limit must be positive")

    ranked: dict[int, int] = {}
    for row in payload["players"]:
        if not isinstance(row, dict):
            continue
        player_id = _positive_int(row.get("player_id"))
        if player_id is None:
            continue
        positions = _positions(row.get("positions") or row.get("position_id"))
        if not positions.intersection(FANTASY_POSITIONS):
            continue
        ranks = [
            rank
            for field in (
                "rank_ecr_ppr",
                "rank_ecr_half",
                "rank_ecr",
                "rank_adp_ppr",
                "rank_adp",
            )
            if (rank := _positive_int(row.get(field))) is not None
        ]
        if not ranks:
            continue
        best = min(ranks)
        ranked[player_id] = min(best, ranked.get(player_id, best))

    selected = sorted(ranked, key=lambda player_id: (ranked[player_id], player_id))[
        : int(limit)
    ]
    if not selected:
        raise ValueError("FantasyPros player index contains no usable players")
    # Keep the manifest stable when ECR ordering moves but membership does not.
    return tuple(sorted(selected))


def build_bootstrap_plan(player_ids: Iterable[int]) -> FantasyProsCorpusPlan:
    """Prioritize unique history, then add category observations if needed."""
    normalized = tuple(dict.fromkeys(int(player_id) for player_id in player_ids))
    if not normalized or any(player_id <= 0 for player_id in normalized):
        raise ValueError("at least one positive FantasyPros player id is required")

    seed = build_news_plan(
        categories=NEWS_CATEGORIES,
        player_categories=(),
    )
    unfiltered = tuple(
        FantasyProsNewsQuery(fpid=player_id) for player_id in normalized
    )
    category_observations = tuple(
        FantasyProsNewsQuery(fpid=player_id, category=category)
        for player_id in normalized
        for category in NEWS_CATEGORIES
    )
    queries = tuple(
        {query.key: query for query in (*seed.queries, *unfiltered, *category_observations)}.values()
    )
    return FantasyProsCorpusPlan(queries)


class FantasyProsCorpusManager:
    """Run bounded provider syncs and cheap vector backfills in the background."""

    def __init__(
        self,
        store: EventStore,
        fantasypros: FantasyProsCache,
        *,
        enabled: bool,
        openrouter_api_key: str,
        embedding_model: str,
        embedding_dimensions: int,
        embedding_timeout_seconds: int,
        target_items: int,
        max_requests: int,
        live_request_reserve: int,
        player_limit: int,
        embedding_budget_usd: float,
        embedding_price_per_million_usd: float,
        embedding_client: OpenRouterEmbeddingClient | None = None,
        clock: Any | None = None,
    ) -> None:
        self.store = store
        self.fantasypros = fantasypros
        self.enabled = bool(enabled)
        self.target_items = int(target_items)
        self.max_requests = int(max_requests)
        self.live_request_reserve = int(live_request_reserve)
        self.player_limit = int(player_limit)
        self.embedding_budget_usd = float(embedding_budget_usd)
        self.embedding_price_per_million_usd = float(
            embedding_price_per_million_usd
        )
        if self.target_items <= 0 or self.max_requests <= 0 or self.player_limit <= 0:
            raise ValueError("FantasyPros corpus limits must be positive")
        if self.live_request_reserve < 0:
            raise ValueError("FantasyPros corpus reserve cannot be negative")
        if not math.isfinite(self.embedding_budget_usd) or self.embedding_budget_usd <= 0:
            raise ValueError("FantasyPros corpus embedding budget must be positive")
        if (
            not math.isfinite(self.embedding_price_per_million_usd)
            or self.embedding_price_per_million_usd <= 0
        ):
            raise ValueError("FantasyPros corpus embedding price must be positive")
        if self.live_request_reserve >= self.fantasypros.app_daily_cap:
            raise ValueError("FantasyPros corpus reserve must be below the shared cap")

        self.embedding_model = str(embedding_model)
        self.embedding_dimensions = int(embedding_dimensions)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._ingestor = FantasyProsNewsCorpusIngestor(
            store,
            fantasypros,
            clock=self._clock,
        )
        self._embedding_client = embedding_client or OpenRouterEmbeddingClient(
            openrouter_api_key if self.enabled else "",
            self.embedding_model,
            self.embedding_dimensions,
            timeout_seconds=int(embedding_timeout_seconds),
        )
        self._lock = threading.RLock()
        self._running = False
        self._requests_made = 0
        self._inserted = 0
        self._updated = 0
        self._duplicates = 0
        self._last_success_at: datetime | None = None
        self._last_error = ""

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime):
            raise ValueError("corpus clock must return a datetime")
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _conservative_token_count(texts: Iterable[str]) -> int:
        # UTF-8 bytes are a deliberately pessimistic upper bound for normal
        # tokenizer output.  The durable ledger charges this estimate before
        # the provider call, so restarts, failures, and model changes cannot
        # make the lifetime fuse forget prior work.
        return sum(max(1, len(text.encode("utf-8"))) for text in texts)

    @staticmethod
    def _query_from_key(query_key: str) -> FantasyProsNewsQuery:
        fields: dict[str, str] = {}
        for component in str(query_key or "").split(";"):
            name, separator, value = component.partition("=")
            if not separator or name in fields:
                raise ValueError("invalid corpus query manifest")
            fields[name] = value
        if set(fields) != {"order", "category", "fpid", "limit"}:
            raise ValueError("invalid corpus query manifest")
        category = None if fields["category"] == "all" else fields["category"]
        fpid = None if fields["fpid"] == "all" else int(fields["fpid"])
        query = FantasyProsNewsQuery(
            order_by=fields["order"],
            category=category,
            fpid=fpid,
            limit=int(fields["limit"]),
        )
        if query.key != query_key:
            raise ValueError("invalid corpus query manifest")
        return query

    @classmethod
    def _plan_from_manifest(cls, manifest: Iterable[str]) -> FantasyProsCorpusPlan:
        queries = tuple(cls._query_from_key(key) for key in manifest)
        if not queries:
            raise ValueError("invalid corpus query manifest")
        return FantasyProsCorpusPlan(queries)

    @staticmethod
    def _stored_time(value: Any) -> datetime:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @property
    def _request_ceiling(self) -> int:
        return self.fantasypros.app_daily_cap - self.live_request_reserve

    def _request_headroom(self) -> tuple[int, str]:
        bucket_used = self.fantasypros.request_usage(bucket=CORPUS_REQUEST_BUCKET)
        if bucket_used >= self.max_requests:
            return 0, "request_limit"
        provider_status = self.fantasypros.status()
        global_remaining = self._request_ceiling - int(provider_status.requests_used)
        if global_remaining <= 0:
            return 0, "budget_reserved"
        return min(self.max_requests - bucket_used, global_remaining), ""

    def _embed_backlog(self, stop: threading.Event | None) -> tuple[int, int, int]:
        rows = self.store.fantasypros_corpus_embedding_backlog(
            model=self.embedding_model,
            dimensions=self.embedding_dimensions,
            input_version=CORPUS_INPUT_VERSION,
            limit=max(self.target_items * 2, 10_000),
        )
        saved = 0
        prompt_tokens = 0
        conservative_tokens = 0
        for start in range(0, len(rows), CORPUS_EMBEDDING_BATCH_SIZE):
            if stop is not None and stop.is_set():
                break
            batch = rows[start : start + CORPUS_EMBEDDING_BATCH_SIZE]
            texts = [str(row.get("canonical_text") or "") for row in batch]
            estimate = self._conservative_token_count(texts)
            reservation_id = self.store.reserve_fantasypros_corpus_embedding_spend(
                model=self.embedding_model,
                input_price_per_million_usd=(
                    self.embedding_price_per_million_usd
                ),
                conservative_tokens=estimate,
                lifetime_cap_usd=self.embedding_budget_usd,
            )
            if reservation_id is None:
                self._last_error = "embedding_budget_reached"
                break
            try:
                vectors = self._embedding_client.embed_many(texts)
            except EmbeddingUnavailable:
                self.store.finish_fantasypros_corpus_embedding_spend(
                    reservation_id,
                    actual_prompt_tokens=None,
                    status="failed",
                )
                self._last_error = "embedding_unavailable"
                break
            except Exception:
                self.store.finish_fantasypros_corpus_embedding_spend(
                    reservation_id,
                    actual_prompt_tokens=None,
                    status="failed",
                )
                raise

            actual_tokens = sum(max(0, int(vector.prompt_tokens)) for vector in vectors)
            try:
                if len(vectors) != len(batch):
                    raise ValueError("embedding response count mismatch")
                for row, vector in zip(batch, vectors, strict=True):
                    content_hash = str(row.get("content_hash") or "")
                    if (
                        vector.input_hash != content_hash
                        or vector.model != self.embedding_model
                        or int(vector.dimensions) != self.embedding_dimensions
                    ):
                        raise ValueError("embedding response identity mismatch")
                batch_saved = 0
                for row, vector in zip(batch, vectors, strict=True):
                    if self.store.store_fantasypros_corpus_embedding(
                        str(row.get("provider_item_id") or ""),
                        vector.model,
                        vector.blob,
                        provider=vector.provider,
                        dimensions=vector.dimensions,
                        input_version=CORPUS_INPUT_VERSION,
                        input_hash=str(row.get("content_hash") or ""),
                    ):
                        batch_saved += 1
                if batch_saved != len(batch):
                    raise ValueError("embedding corpus changed during storage")
                if not self.store.finish_fantasypros_corpus_embedding_spend(
                    reservation_id,
                    actual_prompt_tokens=actual_tokens,
                    status="completed",
                ):
                    raise ValueError("embedding spend finalization failed")
            except Exception:
                # A successful provider request stays charged even if its
                # response cannot be attached safely to the current corpus.
                self.store.finish_fantasypros_corpus_embedding_spend(
                    reservation_id,
                    actual_prompt_tokens=actual_tokens,
                    status="failed",
                )
                self._last_error = "embedding_storage_failed"
                break
            saved += batch_saved
            prompt_tokens += actual_tokens
            conservative_tokens += estimate
        return saved, prompt_tokens, conservative_tokens

    def _ingest_plan(
        self,
        run_id: str,
        plan: FantasyProsCorpusPlan,
        *,
        max_requests: int,
        target_items: int | None = None,
    ) -> Any:
        return self._ingestor.ingest(
            run_id,
            plan,
            max_requests=max_requests,
            request_bucket_limit=self.max_requests,
            live_request_reserve=self.live_request_reserve,
            target_items=target_items,
        )

    def sync_once(self, *, stop: threading.Event | None = None) -> FantasyProsCorpusStatus:
        if not self.enabled:
            return self.status()
        with self._lock:
            if self._running:
                return self.status()
            self._running = True
            self._last_error = ""
        requests_made = 0
        inserted = 0
        updated = 0
        duplicates = 0
        prompt_tokens = 0
        conservative_tokens = 0
        bucket_before = 0
        try:
            bucket_before = self.fantasypros.request_usage(
                bucket=CORPUS_REQUEST_BUCKET
            )
            corpus_count = self.store.fantasypros_corpus_count()
            headroom, wait_reason = self._request_headroom()
            result = None
            if corpus_count < self.target_items:
                latest = self.store.latest_fantasypros_corpus_run(
                    BOOTSTRAP_RUN_PREFIX
                )
                plan: FantasyProsCorpusPlan | None = None
                run_id = ""
                needs_player_index = latest is None
                if latest is not None and latest["status"] != "completed":
                    try:
                        plan = self._plan_from_manifest(latest["manifest"])
                        run_id = str(latest["run_id"])
                    except (TypeError, ValueError, OverflowError):
                        self._last_error = "invalid_manifest"
                elif latest is not None:
                    try:
                        completed_at = self._stored_time(
                            latest["completed_at"] or latest["updated_at"]
                        )
                    except (TypeError, ValueError, OverflowError):
                        self._last_error = "invalid_run_state"
                    else:
                        age = self._now() - completed_at
                        if age < CORPUS_SNAPSHOT_INTERVAL:
                            self._last_error = "awaiting_new_snapshot"
                        else:
                            needs_player_index = True

                if not self._last_error and needs_player_index:
                    if headroom <= 0:
                        self._last_error = wait_reason
                    else:
                        payload = self.fantasypros.get_json(
                            PLAYER_INDEX_PATH,
                            params={"ecr": "included", "show": "pos_rank"},
                            request_ceiling=self._request_ceiling,
                            request_bucket=CORPUS_REQUEST_BUCKET,
                            request_bucket_limit=self.max_requests,
                        )
                        player_ids = parse_fantasypros_player_ids(
                            payload,
                            limit=self.player_limit,
                        )
                        plan = build_bootstrap_plan(player_ids)
                        run_id = (
                            f"{BOOTSTRAP_RUN_PREFIX}"
                            f"{self._now().date().isoformat()}-"
                            f"{manifest_fingerprint(plan)}"
                        )
                        # Persist the exact manifest immediately after the
                        # player-index response so a restart resumes without
                        # spending another player-index request.
                        self.store.begin_fantasypros_corpus_run(
                            run_id,
                            tuple(query.key for query in plan.queries),
                        )

                if not self._last_error and plan is not None:
                    headroom, wait_reason = self._request_headroom()
                    if headroom <= 0:
                        self._last_error = wait_reason
                    else:
                        result = self._ingest_plan(
                            run_id,
                            plan,
                            max_requests=headroom,
                            target_items=self.target_items,
                        )
            else:
                if headroom <= 0:
                    self._last_error = wait_reason
                else:
                    plan = build_news_plan(
                        categories=NEWS_CATEGORIES,
                        player_categories=(),
                    )
                    run_id = f"delta-{self._now().date().isoformat()}"
                    result = self._ingest_plan(
                        run_id,
                        plan,
                        max_requests=min(headroom, plan.request_count),
                    )

            if result is not None:
                inserted += result.inserted
                updated += result.updated
                duplicates += result.duplicates
                if result.stop_reason not in {"", "target_reached"}:
                    self._last_error = result.stop_reason

            if stop is None or not stop.is_set():
                _saved, prompt_tokens, conservative_tokens = self._embed_backlog(stop)
            requests_made = max(
                0,
                self.fantasypros.request_usage(bucket=CORPUS_REQUEST_BUCKET)
                - bucket_before,
            )
            spend = self.store.fantasypros_corpus_embedding_spend_status(
                lifetime_cap_usd=self.embedding_budget_usd
            )
            with self._lock:
                self._requests_made += requests_made
                self._inserted += inserted
                self._updated += updated
                self._duplicates += duplicates
                if self._last_error in {
                    "",
                    "awaiting_new_snapshot",
                    "request_limit",
                    "budget_reserved",
                    "embedding_budget_reached",
                }:
                    self._last_success_at = self._now()
            structured_log(
                logging.INFO
                if self._last_error
                in {
                    "",
                    "awaiting_new_snapshot",
                    "request_limit",
                    "budget_reserved",
                    "embedding_budget_reached",
                }
                else logging.WARNING,
                "fantasypros.corpus_sync",
                corpusItems=self.store.fantasypros_corpus_count(),
                embeddedItems=self.store.fantasypros_corpus_embedding_count(
                    model=self.embedding_model,
                    dimensions=self.embedding_dimensions,
                    input_version=CORPUS_INPUT_VERSION,
                ),
                requestsMade=requests_made,
                rollingRequests=self.fantasypros.request_usage(
                    bucket=CORPUS_REQUEST_BUCKET
                ),
                inserted=inserted,
                updated=updated,
                duplicates=duplicates,
                promptTokens=prompt_tokens,
                conservativeTokens=conservative_tokens,
                lifetimePromptTokens=spend["actual_prompt_tokens"],
                lifetimeConservativeTokens=spend["reserved_tokens"],
                estimatedCostUsd=round(float(spend["charged_cost_usd"]), 6),
                reason=self._last_error,
            )
        except FantasyProsRequestError as error:
            requests_made = max(
                0,
                self.fantasypros.request_usage(bucket=CORPUS_REQUEST_BUCKET)
                - bucket_before,
            )
            with self._lock:
                self._requests_made += requests_made
                self._last_error = error.code
        except ValueError:
            with self._lock:
                if not self._last_error:
                    self._last_error = "invalid_corpus_state"
        except Exception as error:  # noqa: BLE001 - optional worker fails closed
            with self._lock:
                self._last_error = f"internal_{type(error).__name__}"
            structured_log(
                logging.WARNING,
                "fantasypros.corpus_sync_failed",
                errorType=type(error).__name__,
            )
        finally:
            with self._lock:
                self._running = False
        return self.status()

    def run_forever(self, stop: threading.Event) -> None:
        while not stop.is_set():
            status = self.sync_once(stop=stop)
            wait_states = {
                "awaiting_new_snapshot",
                "request_limit",
                "budget_reserved",
                "embedding_budget_reached",
            }
            delay = CORPUS_LOOP_HEALTHY_SECONDS
            if status.last_error and status.last_error not in wait_states:
                delay = CORPUS_LOOP_RETRY_SECONDS
            stop.wait(delay)

    def status(self) -> FantasyProsCorpusStatus:
        try:
            corpus_items = self.store.fantasypros_corpus_count()
            embedded_items = self.store.fantasypros_corpus_embedding_count(
                model=self.embedding_model,
                dimensions=self.embedding_dimensions,
                input_version=CORPUS_INPUT_VERSION,
            )
            spend = self.store.fantasypros_corpus_embedding_spend_status(
                lifetime_cap_usd=self.embedding_budget_usd
            )
        except Exception:  # noqa: BLE001 - status must survive shutdown
            corpus_items = 0
            embedded_items = 0
            spend = {
                "actual_prompt_tokens": 0,
                "reserved_tokens": 0,
                "charged_cost_usd": 0.0,
            }
        try:
            rolling_requests = self.fantasypros.request_usage(
                bucket=CORPUS_REQUEST_BUCKET
            )
        except Exception:  # noqa: BLE001 - status must survive shutdown
            rolling_requests = 0
        with self._lock:
            return FantasyProsCorpusStatus(
                enabled=self.enabled,
                running=self._running,
                corpus_items=corpus_items,
                embedded_items=embedded_items,
                requests_made=rolling_requests,
                inserted=self._inserted,
                updated=self._updated,
                duplicates=self._duplicates,
                prompt_tokens=int(spend["actual_prompt_tokens"]),
                conservative_tokens=int(spend["reserved_tokens"]),
                estimated_cost_usd=float(spend["charged_cost_usd"]),
                last_success_at=self._last_success_at,
                last_error=self._last_error,
            )
