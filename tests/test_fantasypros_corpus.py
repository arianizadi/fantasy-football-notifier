from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from notifier.event_store import EventStore
from notifier.sources.fantasypros import (
    MIN_REQUEST_INTERVAL_SECONDS,
    FantasyProsCache,
)
from notifier.sources.fantasypros_news import (
    API_DOCS_URL,
    CORPUS_INPUT_VERSION,
    FantasyProsCorpusError,
    FantasyProsNewsCorpusIngestor,
    FantasyProsNewsQuery,
    build_news_plan,
    canonical_corpus_text,
    parse_news_response,
)


NOW = datetime(2026, 8, 24, 19, 30, tzinfo=timezone.utc)
SECRET = "fp-secret-that-must-never-be-logged"


class FakeClock:
    def __init__(self, now: float = NOW.timestamp()) -> None:
        self._now = now
        self._lock = threading.Lock()
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self._now += seconds

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds

    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self(), timezone.utc)


class FakeResponse:
    def __init__(self, payload: Any, *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self.payload


class FakeSession:
    def __init__(
        self,
        clock: FakeClock,
        *,
        news_responses: list[FakeResponse | Exception] | None = None,
    ) -> None:
        self.clock = clock
        self.news_responses = list(news_responses or [])
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, "started": self.clock(), **kwargs})
        if url.endswith("/nfl/news"):
            response: FakeResponse | Exception = (
                self.news_responses.pop(0)
                if self.news_responses
                else FakeResponse(_payload())
            )
            if isinstance(response, Exception):
                raise response
            return response

        params = kwargs["params"]
        return FakeResponse(
            {
                "sport": "NFL",
                "year": "2026",
                "scoring": params["scoring"],
                "position_id": "ALL",
                "ranking_type_name": params["type"],
                "last_updated_ts": int(self.clock()),
                "players": [
                    {
                        "player_id": 1,
                        "player_name": "Test Player",
                        "player_team_id": "SF",
                        "player_positions": "RB",
                        "rank_ecr": 1,
                        "pos_rank": "RB1",
                    }
                ],
            }
        )


def _item(
    item_id: int = 519470,
    *,
    impact: str = "He should compete for backup touches.",
) -> dict[str, Any]:
    return {
        "id": item_id,
        "created": "2026-08-24 18:15:00",
        "created_formated": "Mon, Aug 24th 6:15pm UTC",
        "author": "Ari Koslow",
        "player_id": 6880,
        "team_id": "SF",
        "title": "<b>Backup</b> earns first-team work",
        "sport_id": "NFL",
        "categories": ["News", "Injury", "News"],
        "link": f"https://www.fantasypros.com/nfl/news/{item_id}/test.php",
        "desc": "The runner worked with the first team &amp; could see touches.",
        "impact": impact,
    }


def _payload(*items: dict[str, Any]) -> dict[str, Any]:
    values = list(items or (_item(),))
    return {
        "sport": "NFL",
        "title": "Fantasy Player News",
        "description": "Fantasy football news",
        "count": len(values),
        "items": values,
    }


def _cache(
    tmp_path: Path,
    clock: FakeClock,
    session: FakeSession,
    *,
    cap: int = 425,
) -> FantasyProsCache:
    return FantasyProsCache(
        tmp_path,
        SECRET,
        2026,
        session=session,
        app_daily_cap=cap,
        clock=clock,
        sleep=clock.sleep,
    )


def test_plan_exposes_documented_limit_categories_and_cost_ceiling() -> None:
    plan = build_news_plan(
        player_ids=[6880, 6880, 7000],
        categories=("injury", "transaction"),
        orderings=("created", "updated"),
        player_categories=(None, "injury"),
    )

    # Per ordering: general + two categories + two queries per unique player.
    assert plan.request_count == 14
    assert plan.maximum_candidate_items == 1400
    assert len({query.key for query in plan.queries}) == 14
    assert all(query.limit == 100 for query in plan.queries)
    assert FantasyProsNewsQuery(category="breaking").params == {
        "limit": 100,
        "order_by": "created",
        "category": "breaking",
    }
    with pytest.raises(ValueError):
        FantasyProsNewsQuery(limit=101)
    with pytest.raises(ValueError):
        FantasyProsNewsQuery(category="opinion")


def test_response_parser_keeps_attribution_and_canonical_clean_text() -> None:
    query = FantasyProsNewsQuery(category="injury")
    rows = parse_news_response(_payload(), query=query, fetched_at=NOW)

    assert len(rows) == 1
    row = rows[0]
    assert row["provider_item_id"] == "519470"
    assert row["player_id"] == 6880
    assert row["categories"] == ["Injury", "News"]
    assert row["title"] == "Backup earns first-team work"
    assert "& could see touches" in row["canonical_text"]
    assert "FantasyPros impact:" in row["canonical_text"]
    assert row["source_provider"] == "FantasyPros"
    assert row["usage_scope"] == "personal_reference"
    assert row["api_docs_url"] == API_DOCS_URL

    invalid = _payload()
    del invalid["items"][0]["impact"]
    with pytest.raises(FantasyProsCorpusError) as error:
        parse_news_response(invalid, query=query, fetched_at=NOW)
    assert error.value.code == "invalid_response"


def test_reference_rows_are_isolated_deduped_and_update_invalidates_vector(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path)
    query = FantasyProsNewsQuery(category="injury")
    first = parse_news_response(_payload(), query=query, fetched_at=NOW)
    store.begin_fantasypros_corpus_run("first-run", (query.key,))
    result = store.store_fantasypros_corpus_batch(
        "first-run", query.key, first, fetched_at=NOW
    )
    assert result == {"inserted": 1, "updated": 0, "duplicates": 0}
    assert store.complete_fantasypros_corpus_run("first-run") is True

    content_hash = first[0]["content_hash"]
    assert store.store_fantasypros_corpus_embedding(
        "519470",
        "embedding-model",
        b"four-byte-vector",
        provider="openrouter",
        dimensions=4,
        input_version=CORPUS_INPUT_VERSION,
        input_hash=content_hash,
    )
    assert store.fantasypros_corpus_embedding_count() == 1

    # A different query/run sees the same provider id, updates its changed
    # text in place, and clears the now-stale vector instead of growing a
    # duplicate row.
    changed = parse_news_response(
        _payload(_item(impact="He is expected to start immediately.")),
        query=query,
        fetched_at=NOW,
    )
    store.begin_fantasypros_corpus_run("second-run", (query.key,))
    result = store.store_fantasypros_corpus_batch(
        "second-run", query.key, changed, fetched_at=NOW
    )
    assert result == {"inserted": 0, "updated": 1, "duplicates": 0}
    row = store.fantasypros_corpus_items()[0]
    assert row["embedding"] is None
    assert row["embedding_model"] is None
    assert row["content_hash"] == changed[0]["content_hash"]
    assert json.loads(row["categories_json"]) == ["Injury", "News"]
    assert store.fantasypros_corpus_count() == 1

    # Corpus storage is not an implicit live-news import path.
    assert store.count() == 0
    assert store.all_reports() == []
    assert store.recent(
        since=NOW.replace(hour=0),
        until=NOW.replace(hour=23),
    ) == []
    assert store.search("backup touches") == []
    store.close()


def test_unchanged_cross_category_item_is_counted_as_provider_duplicate(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path)
    injury = FantasyProsNewsQuery(category="injury")
    breaking = FantasyProsNewsQuery(category="breaking")
    row = parse_news_response(_payload(), query=injury, fetched_at=NOW)

    store.begin_fantasypros_corpus_run("injury-run", (injury.key,))
    store.store_fantasypros_corpus_batch(
        "injury-run", injury.key, row, fetched_at=NOW
    )
    store.begin_fantasypros_corpus_run("breaking-run", (breaking.key,))
    result = store.store_fantasypros_corpus_batch(
        "breaking-run", breaking.key, row, fetched_at=NOW
    )

    assert result == {"inserted": 0, "updated": 0, "duplicates": 1}
    assert store.fantasypros_corpus_count() == 1
    observations = store.fantasypros_corpus_observations(
        provider_item_id="519470"
    )
    assert len(observations) == 2
    assert {row["query_key"] for row in observations} == {
        injury.key,
        breaking.key,
    }
    store.close()


def test_ingestion_shares_persistent_cap_and_cadence_with_rankings(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _cache(tmp_path, clock, session, cap=4)
    store = EventStore(tmp_path)
    plan = build_news_plan(categories=("injury",), player_categories=())
    ingestor = FantasyProsNewsCorpusIngestor(
        store, cache, clock=clock.datetime
    )

    result = ingestor.ingest(
        "shared-budget",
        plan,
        max_requests=2,
        live_request_reserve=1,
    )
    assert result.complete is True
    assert result.requests_made == 2
    assert cache.request_usage() == 2

    # Rankings consume the same two remaining reservations; a fifth request
    # never leaves the process.
    assert cache.refresh(("PPR",), force=True) is True
    assert cache.request_usage() == 4
    assert len(session.calls) == 4
    assert cache.refresh(("PPR",), force=True) is False
    assert len(session.calls) == 4
    starts = [call["started"] for call in session.calls]
    assert all(
        later - earlier >= MIN_REQUEST_INTERVAL_SECONDS
        for earlier, later in zip(starts, starts[1:])
    )
    store.close()


def test_atomic_request_ceiling_preserves_reserve_after_stale_status_race(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _cache(tmp_path, clock, session, cap=4)
    cache.get_json("nfl/news", params={})
    cache.get_json("nfl/news", params={})
    status_read = threading.Event()
    resume_status = threading.Event()

    class StatusRaceCache:
        app_daily_cap = cache.app_daily_cap

        def __init__(self) -> None:
            self._paused = False

        def status(self) -> Any:
            snapshot = cache.status()
            if not self._paused:
                self._paused = True
                status_read.set()
                assert resume_status.wait(timeout=2)
            return snapshot

        def get_json(self, *args: Any, **kwargs: Any) -> Any:
            return cache.get_json(*args, **kwargs)

    store = EventStore(tmp_path)
    ingestor = FantasyProsNewsCorpusIngestor(
        store,
        StatusRaceCache(),  # type: ignore[arg-type]
        clock=clock.datetime,
    )
    plan = build_news_plan(categories=(), player_categories=())
    results: list[Any] = []
    worker = threading.Thread(
        target=lambda: results.append(
            ingestor.ingest(
                "atomic-reserve-race",
                plan,
                max_requests=1,
                request_bucket_limit=3,
                live_request_reserve=1,
            )
        )
    )
    worker.start()
    assert status_read.wait(timeout=2)

    # A live/unbucketed request consumes the final non-reserved slot after the
    # corpus observed stale usage but before its own reservation attempt.
    cache.get_json("nfl/news", params={})
    assert cache.request_usage() == 3
    resume_status.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(results) == 1
    assert results[0].requests_made == 0
    assert results[0].stop_reason == "budget_reserved"
    assert cache.request_usage() == 3
    assert cache.request_usage(bucket="corpus") == 0
    assert len(session.calls) == 3
    store.close()
    cache.close()


def test_corpus_request_bucket_survives_restart_and_rolls_after_24_hours(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    first_session = FakeSession(clock)
    first_cache = _cache(tmp_path, clock, first_session, cap=10)
    store = EventStore(tmp_path)
    plan = build_news_plan(categories=("injury",), player_categories=())
    first_ingestor = FantasyProsNewsCorpusIngestor(
        store,
        first_cache,
        clock=clock.datetime,
    )

    first = first_ingestor.ingest(
        "rolling-first",
        plan,
        max_requests=2,
        request_bucket_limit=2,
        live_request_reserve=1,
    )
    assert first.complete is True
    assert first.requests_made == 2
    assert first_cache.request_usage() == 2
    assert first_cache.request_usage(bucket="corpus") == 2
    first_cache.close()

    restarted_session = FakeSession(clock)
    restarted_cache = _cache(tmp_path, clock, restarted_session, cap=10)
    restarted_ingestor = FantasyProsNewsCorpusIngestor(
        store,
        restarted_cache,
        clock=clock.datetime,
    )
    limited = restarted_ingestor.ingest(
        "rolling-second",
        plan,
        max_requests=2,
        request_bucket_limit=2,
        live_request_reserve=1,
    )

    assert limited.complete is False
    assert limited.requests_made == 0
    assert limited.stop_reason == "request_limit"
    assert restarted_session.calls == []
    assert restarted_cache.request_usage(bucket="corpus") == 2

    clock.advance(24 * 60 * 60 + 1)
    resumed = restarted_ingestor.ingest(
        "rolling-second",
        plan,
        max_requests=2,
        request_bucket_limit=2,
        live_request_reserve=1,
    )
    assert resumed.complete is True
    assert resumed.requests_made == 2
    assert len(restarted_session.calls) == 2
    assert restarted_cache.request_usage() == 2
    assert restarted_cache.request_usage(bucket="corpus") == 2
    store.close()
    restarted_cache.close()


def test_request_reserve_pauses_and_same_run_resumes_without_refetching_batches(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _cache(tmp_path, clock, session, cap=3)
    store = EventStore(tmp_path)
    plan = build_news_plan(categories=("injury", "transaction"), player_categories=())
    ingestor = FantasyProsNewsCorpusIngestor(
        store, cache, clock=clock.datetime
    )

    first = ingestor.ingest(
        "resumable",
        plan,
        max_requests=3,
        live_request_reserve=1,
    )
    assert first.complete is False
    assert first.requests_made == 2
    assert first.completed_queries == 2
    assert first.remaining_queries == 1
    assert first.stop_reason == "budget_reserved"
    assert store.fantasypros_corpus_run("resumable")["status"] == "paused"

    clock.advance(24 * 60 * 60 + 1)
    second = ingestor.ingest(
        "resumable",
        plan,
        max_requests=3,
        live_request_reserve=1,
    )
    assert second.complete is True
    assert second.requests_made == 1
    assert second.completed_queries == 3
    assert len(session.calls) == 3
    assert store.fantasypros_corpus_run("resumable")["status"] == "completed"
    store.close()


def test_target_items_stops_bootstrap_before_spending_remaining_requests(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _cache(tmp_path, clock, session, cap=10)
    store = EventStore(tmp_path)
    plan = build_news_plan(
        categories=("injury", "transaction"),
        player_categories=(),
    )
    ingestor = FantasyProsNewsCorpusIngestor(
        store, cache, clock=clock.datetime
    )

    result = ingestor.ingest(
        "target-bounded",
        plan,
        max_requests=3,
        live_request_reserve=1,
        target_items=1,
    )

    assert result.complete is False
    assert result.requests_made == 1
    assert result.completed_queries == 1
    assert result.remaining_queries == 2
    assert result.stop_reason == "target_reached"
    assert len(session.calls) == 1
    run = store.fantasypros_corpus_run("target-bounded")
    assert run is not None
    assert run["status"] == "paused"
    assert run["stop_reason"] == "target_reached"
    store.close()


def test_failed_request_is_secret_free_and_resume_skips_completed_query(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    session = FakeSession(
        clock,
        news_responses=[
            FakeResponse(_payload()),
            RuntimeError(f"headers included {SECRET}; body=private"),
            FakeResponse(_payload(_item(519471))),
        ],
    )
    cache = _cache(tmp_path, clock, session, cap=10)
    store = EventStore(tmp_path)
    plan = build_news_plan(categories=("injury",), player_categories=())
    ingestor = FantasyProsNewsCorpusIngestor(
        store, cache, clock=clock.datetime
    )

    with caplog.at_level(logging.INFO):
        first = ingestor.ingest(
            "retry-safe",
            plan,
            max_requests=2,
            request_bucket_limit=3,
            live_request_reserve=1,
        )
    assert first.complete is False
    assert first.completed_queries == 1
    assert first.requests_made == 2
    assert first.stop_reason == "request_failed"
    assert SECRET not in caplog.text
    assert "body=private" not in caplog.text
    assert store.fantasypros_corpus_run("retry-safe")["status"] == "failed"

    second = ingestor.ingest(
        "retry-safe",
        plan,
        max_requests=2,
        request_bucket_limit=3,
        live_request_reserve=1,
    )
    assert second.complete is True
    assert second.requests_made == 1
    assert second.completed_queries == 2
    assert len(session.calls) == 3
    assert store.fantasypros_corpus_count() == 2
    store.close()


def test_resume_rejects_changed_manifest(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    first = build_news_plan(categories=("injury",), player_categories=())
    changed = build_news_plan(categories=("transaction",), player_categories=())
    store.begin_fantasypros_corpus_run(
        "stable-manifest", tuple(query.key for query in first.queries)
    )

    with pytest.raises(ValueError, match="original plan"):
        store.begin_fantasypros_corpus_run(
            "stable-manifest", tuple(query.key for query in changed.queries)
        )
    store.close()


def test_canonical_text_is_deterministic() -> None:
    assert canonical_corpus_text(
        title="  Player &amp; role  ",
        description="<p>Player gets work.</p>",
        impact=" More touches. ",
        categories=("News", "News", "Injury"),
    ) == (
        "Headline: Player & role\n"
        "Report: Player gets work.\n"
        "FantasyPros impact: More touches."
    )
