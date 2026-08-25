from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from notifier.embeddings import EmbeddingUnavailable, EmbeddingVector, pack_vector
from notifier.event_store import EventStore
from notifier.fantasypros_corpus import (
    FANTASY_POSITIONS,
    FantasyProsCorpusManager,
    build_bootstrap_plan,
    parse_fantasypros_player_ids,
)
from notifier.sources.fantasypros_news import CORPUS_INPUT_VERSION


NOW = datetime(2026, 8, 24, 20, 0, tzinfo=timezone.utc)


def _player_payload() -> dict[str, Any]:
    return {
        "sport": "NFL",
        "players": [
            {
                "player_id": 30,
                "positions": ["WR"],
                "rank_ecr_ppr": 2,
            },
            {
                "player_id": 10,
                "position_id": "RB",
                "rank_ecr_half": 1,
            },
            {
                "player_id": 20,
                "positions": ["K"],
                "rank_ecr_ppr": 1,
            },
            {
                "player_id": 40,
                "positions": ["QB"],
                "rank_ecr": 3,
            },
        ],
    }


def _news_item(item_id: int) -> dict[str, Any]:
    return {
        "id": item_id,
        "created": "2026-08-24 19:00:00",
        "created_formated": "Mon, Aug 24th 7:00pm UTC",
        "author": "FantasyPros Staff",
        "player_id": 10,
        "team_id": "SF",
        "title": f"Player update {item_id}",
        "sport_id": "NFL",
        "categories": ["News", "Injury"],
        "link": f"https://www.fantasypros.com/nfl/news/{item_id}/item.php",
        "desc": "Player left practice with an injury.",
        "impact": "Monitor the next practice report.",
    }


class FakeFantasyPros:
    app_daily_cap = 425

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.used = 0
        self.bucket_used = 0

    def status(self) -> Any:
        return SimpleNamespace(requests_used=self.used, request_cap=425)

    def request_usage(self, *, bucket: str = "") -> int:
        return self.bucket_used if bucket else self.used

    def get_json(
        self,
        path: str,
        *,
        params: dict[str, Any],
        request_ceiling: int | None = None,
        request_bucket: str = "",
        request_bucket_limit: int | None = None,
    ) -> Any:
        if request_ceiling is not None and self.used >= request_ceiling:
            raise AssertionError("manager crossed the global request ceiling")
        if (
            request_bucket
            and request_bucket_limit is not None
            and self.bucket_used >= request_bucket_limit
        ):
            raise AssertionError("manager crossed the corpus request bucket")
        self.calls.append((path, dict(params)))
        self.used += 1
        if request_bucket:
            self.bucket_used += 1
        if path == "nfl/players":
            return _player_payload()
        item_id = 900_000 + len(self.calls)
        return {
            "sport": "NFL",
            "title": "Fantasy Player News",
            "description": "Fantasy football news",
            "count": 2,
            "items": [_news_item(item_id), _news_item(item_id + 1000)],
        }


class FakeEmbeddingClient:
    def __init__(self, *, model: str = "test/embedding", fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.model = model
        self.fail = fail

    def embed_many(self, texts: list[str]) -> list[EmbeddingVector]:
        self.calls.append(list(texts))
        if self.fail:
            raise EmbeddingUnavailable("synthetic failure")
        values = (1.0, 0.0, 0.0, 0.0)
        return [
            EmbeddingVector(
                model=self.model,
                provider="openrouter",
                dimensions=4,
                input_version="news-report-v1",
                input_hash=hashlib.sha256(text.encode()).hexdigest(),
                values=values,
                blob=pack_vector(values),
                prompt_tokens=7,
            )
            for text in texts
        ]


class MutableClock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


def _manager(
    state_dir: Path,
    fantasypros: FakeFantasyPros,
    embeddings: FakeEmbeddingClient,
    *,
    target: int = 2,
    budget: float = 0.25,
    max_requests: int = 10,
    player_limit: int = 3,
    model: str = "test/embedding",
    clock: Any = None,
) -> tuple[EventStore, FantasyProsCorpusManager]:
    store = EventStore(state_dir)
    manager = FantasyProsCorpusManager(
        store,
        fantasypros,  # type: ignore[arg-type]
        enabled=True,
        openrouter_api_key="not-used-by-fake",
        embedding_model=model,
        embedding_dimensions=4,
        embedding_timeout_seconds=1,
        target_items=target,
        max_requests=max_requests,
        live_request_reserve=75,
        player_limit=player_limit,
        embedding_budget_usd=budget,
        embedding_price_per_million_usd=0.01,
        embedding_client=embeddings,  # type: ignore[arg-type]
        clock=clock or (lambda: NOW),
    )
    return store, manager


def test_player_selection_is_ranked_relevant_and_manifest_stable() -> None:
    assert FANTASY_POSITIONS == {"QB", "RB", "WR", "TE"}
    # Selection uses ECR, while returned IDs are sorted to keep run manifests
    # stable across small rank-order changes.
    assert parse_fantasypros_player_ids(_player_payload(), limit=2) == (10, 30)

    plan = build_bootstrap_plan((30, 10))
    keys = [query.key for query in plan.queries]
    first_player_category = next(
        index
        for index, query in enumerate(plan.queries)
        if query.fpid is not None and query.category is not None
    )
    assert all(
        query.category is None
        for query in plan.queries[6:first_player_category]
    )
    assert len(keys) == len(set(keys))


def test_sync_reaches_target_embeds_rows_and_never_creates_live_events(
    tmp_path: Path,
) -> None:
    provider = FakeFantasyPros()
    embedding = FakeEmbeddingClient()
    store, manager = _manager(tmp_path, provider, embedding)
    try:
        status = manager.sync_once()

        assert status.corpus_items == 2
        assert status.embedded_items == 2
        assert status.requests_made == 2  # player index + first news query
        assert status.inserted == 2
        assert status.prompt_tokens == 14
        assert status.estimated_cost_usd <= 0.25
        assert store.count() == 0
        assert store.all_reports() == []
        rows = store.fantasypros_corpus_items()
        assert all(row["embedding_input_version"] == CORPUS_INPUT_VERSION for row in rows)
    finally:
        store.close()


def test_embedding_fuse_abstains_before_any_paid_request(tmp_path: Path) -> None:
    provider = FakeFantasyPros()
    embedding = FakeEmbeddingClient()
    store, manager = _manager(
        tmp_path,
        provider,
        embedding,
        budget=0.000000001,
    )
    try:
        status = manager.sync_once()

        assert status.corpus_items == 2
        assert status.embedded_items == 0
        assert status.last_error == "embedding_budget_reached"
        assert embedding.calls == []
    finally:
        store.close()


def test_one_request_budget_persists_manifest_without_fetching_news(
    tmp_path: Path,
) -> None:
    provider = FakeFantasyPros()
    embedding = FakeEmbeddingClient()
    store, manager = _manager(
        tmp_path,
        provider,
        embedding,
        target=1000,
        max_requests=1,
    )
    try:
        status = manager.sync_once()

        assert [path for path, _params in provider.calls] == ["nfl/players"]
        assert status.requests_made == 1
        assert status.corpus_items == 0
        assert status.last_error == "request_limit"
        run = store.latest_fantasypros_corpus_run("bootstrap-v1-")
        assert run is not None
        assert run["status"] == "running"
        assert run["manifest"]
    finally:
        store.close()


def test_incomplete_bootstrap_resumes_without_another_player_index(
    tmp_path: Path,
) -> None:
    provider = FakeFantasyPros()
    first_embeddings = FakeEmbeddingClient()
    store, first = _manager(
        tmp_path,
        provider,
        first_embeddings,
        target=1000,
        max_requests=2,
    )
    try:
        first_status = first.sync_once()
        assert first_status.last_error == "request_limit"
        assert [path for path, _params in provider.calls].count("nfl/players") == 1

        # Simulate the rolling request window expiring, then restart the
        # manager against the same durable corpus state.
        provider.used = 0
        provider.bucket_used = 0
        second_embeddings = FakeEmbeddingClient()
        second = FantasyProsCorpusManager(
            store,
            provider,  # type: ignore[arg-type]
            enabled=True,
            openrouter_api_key="not-used-by-fake",
            embedding_model="test/embedding",
            embedding_dimensions=4,
            embedding_timeout_seconds=1,
            target_items=1000,
            max_requests=2,
            live_request_reserve=75,
            player_limit=3,
            embedding_budget_usd=0.25,
            embedding_price_per_million_usd=0.01,
            embedding_client=second_embeddings,  # type: ignore[arg-type]
            clock=lambda: NOW + timedelta(days=1),
        )
        second.sync_once()

        assert [path for path, _params in provider.calls].count("nfl/players") == 1
        assert [path for path, _params in provider.calls][-2:] == [
            "nfl/news",
            "nfl/news",
        ]
    finally:
        store.close()


def test_completed_bootstrap_waits_then_starts_a_new_daily_snapshot(
    tmp_path: Path,
) -> None:
    provider = FakeFantasyPros()
    clock = MutableClock(datetime.now(timezone.utc))
    embedding = FakeEmbeddingClient()
    store, manager = _manager(
        tmp_path,
        provider,
        embedding,
        target=1000,
        max_requests=30,
        clock=clock,
    )
    try:
        first = manager.sync_once()
        assert first.last_error == ""
        assert [path for path, _params in provider.calls].count("nfl/players") == 1
        first_call_count = len(provider.calls)
        run = store.latest_fantasypros_corpus_run("bootstrap-v1-")
        assert run is not None and run["status"] == "completed"
        assert first.corpus_items < 1000

        provider.used = 0
        provider.bucket_used = 0
        waiting = manager.sync_once()
        assert waiting.last_error == "awaiting_new_snapshot"
        assert len(provider.calls) == first_call_count

        clock.advance(timedelta(hours=25))
        refreshed = manager.sync_once()
        assert refreshed.last_error == ""
        assert [path for path, _params in provider.calls].count("nfl/players") == 2
        latest = store.latest_fantasypros_corpus_run("bootstrap-v1-")
        assert latest is not None
        assert latest["run_id"] != run["run_id"]
    finally:
        store.close()


def test_embedding_spend_survives_failure_restart_and_model_change(
    tmp_path: Path,
) -> None:
    provider = FakeFantasyPros()
    failing = FakeEmbeddingClient(fail=True)
    store, manager = _manager(
        tmp_path,
        provider,
        failing,
        budget=0.000003,
    )
    try:
        failed = manager.sync_once()
        assert failed.last_error == "embedding_unavailable"
        ledger = store.fantasypros_corpus_embedding_spend_status(
            lifetime_cap_usd=0.000003
        )
        assert ledger["failed_reservations"] == 1
        assert ledger["charged_cost_usd"] > 0

        provider.used = 0
        provider.bucket_used = 0
        different_model = FakeEmbeddingClient(model="test/new-model")
        restarted = FantasyProsCorpusManager(
            store,
            provider,  # type: ignore[arg-type]
            enabled=True,
            openrouter_api_key="not-used-by-fake",
            embedding_model="test/new-model",
            embedding_dimensions=4,
            embedding_timeout_seconds=1,
            target_items=2,
            max_requests=10,
            live_request_reserve=75,
            player_limit=3,
            embedding_budget_usd=float(ledger["charged_cost_usd"]),
            embedding_price_per_million_usd=0.01,
            embedding_client=different_model,  # type: ignore[arg-type]
            clock=lambda: NOW + timedelta(days=1),
        )
        status = restarted.sync_once()

        assert status.last_error == "embedding_budget_reached"
        assert different_model.calls == []
        assert status.estimated_cost_usd == ledger["charged_cost_usd"]
    finally:
        store.close()
