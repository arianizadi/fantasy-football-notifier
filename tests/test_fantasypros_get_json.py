from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from notifier.sources.fantasypros import (
    BASE_URL,
    CACHE_FILENAME,
    CORPUS_REQUEST_BUCKET,
    MIN_REQUEST_INTERVAL_SECONDS,
    FantasyProsCache,
    FantasyProsRequestError,
)
from notifier.sources.fantasypros_news import (
    FantasyProsCorpusError,
    FantasyProsNewsQuery,
    _SharedFantasyProsNewsTransport,
)


SECRET = "fp-public-get-secret-that-must-not-escape"


class FakeClock:
    def __init__(self, now: float = 2_000_000_000.0) -> None:
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


class FakeResponse:
    def __init__(self, payload: Any, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(
        self,
        clock: FakeClock,
        *,
        payload: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.clock = clock
        self.payload = {"ok": True} if payload is None else payload
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, "started": self.clock(), **kwargs})
        if self.error is not None:
            raise self.error
        return FakeResponse(self.payload)


def _cache(
    state_dir: Path,
    clock: FakeClock,
    session: FakeSession,
    *,
    cap: int = 425,
) -> FantasyProsCache:
    return FantasyProsCache(
        state_dir,
        SECRET,
        2026,
        session=session,
        app_daily_cap=cap,
        clock=clock,
        sleep=clock.sleep,
    )


def test_get_json_shares_pacing_and_persistent_daily_budget(tmp_path: Path) -> None:
    clock = FakeClock()
    session = FakeSession(clock, payload={"items": [1]})
    cache = _cache(tmp_path, clock, session, cap=2)

    assert cache.get_json("nfl/news", params={"limit": 100}) == {"items": [1]}
    assert cache.get_json("nfl/news", params={"limit": 50}) == {"items": [1]}

    assert [call["url"] for call in session.calls] == [
        f"{BASE_URL}/nfl/news",
        f"{BASE_URL}/nfl/news",
    ]
    assert [call["params"] for call in session.calls] == [
        {"limit": 100},
        {"limit": 50},
    ]
    assert session.calls[1]["started"] - session.calls[0]["started"] >= (
        MIN_REQUEST_INTERVAL_SECONDS
    )

    state_text = (tmp_path / CACHE_FILENAME).read_text(encoding="utf-8")
    assert SECRET not in state_text
    assert len(json.loads(state_text)["requests"]) == 2

    restarted_session = FakeSession(clock)
    restarted = _cache(tmp_path, clock, restarted_session, cap=2)
    with pytest.raises(FantasyProsRequestError) as raised:
        restarted.get_json("nfl/news")
    assert raised.value.code == "budget_exhausted"
    assert raised.value.request_reserved is False
    assert str(raised.value) == "budget_exhausted"
    assert restarted_session.calls == []


def test_named_request_bucket_persists_and_does_not_limit_unbucketed_calls(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _cache(tmp_path, clock, session, cap=4)
    options = {
        "request_ceiling": 3,
        "request_bucket": CORPUS_REQUEST_BUCKET,
        "request_bucket_limit": 2,
    }

    cache.get_json("nfl/news", **options)
    cache.get_json("nfl/news", **options)

    assert cache.request_usage() == 2
    assert cache.request_usage(bucket=CORPUS_REQUEST_BUCKET) == 2
    on_disk = json.loads((tmp_path / CACHE_FILENAME).read_text(encoding="utf-8"))
    assert len(on_disk["requests"]) == 2
    assert len(on_disk["request_buckets"][CORPUS_REQUEST_BUCKET]) == 2

    restarted_session = FakeSession(clock)
    restarted = _cache(tmp_path, clock, restarted_session, cap=4)
    with pytest.raises(FantasyProsRequestError) as limited:
        restarted.get_json("nfl/news", **options)
    assert limited.value.code == "request_limit"
    assert limited.value.request_reserved is False
    assert restarted_session.calls == []

    # Ranking and diagnostic requests retain the original unbucketed behavior.
    assert restarted.get_json("nfl/news") == {"ok": True}
    assert restarted.request_usage() == 3
    assert restarted.request_usage(bucket=CORPUS_REQUEST_BUCKET) == 2


def test_legacy_ledger_without_request_buckets_remains_valid(tmp_path: Path) -> None:
    clock = FakeClock()
    (tmp_path / CACHE_FILENAME).write_text(
        json.dumps(
            {
                "version": 1,
                "requests": [clock() - 10],
                "datasets": {},
            }
        ),
        encoding="utf-8",
    )
    session = FakeSession(clock)
    cache = _cache(tmp_path, clock, session, cap=4)

    assert cache.enabled is True
    assert cache.request_usage() == 1
    assert cache.request_usage(bucket=CORPUS_REQUEST_BUCKET) == 0
    assert cache.get_json(
        "nfl/news",
        request_ceiling=3,
        request_bucket=CORPUS_REQUEST_BUCKET,
        request_bucket_limit=2,
    ) == {"ok": True}
    assert cache.request_usage() == 2
    assert cache.request_usage(bucket=CORPUS_REQUEST_BUCKET) == 1


@pytest.mark.parametrize(
    "request_buckets",
    [
        [],
        {CORPUS_REQUEST_BUCKET: "unknown"},
        {CORPUS_REQUEST_BUCKET: [2_000_000_000.0]},
        {"unsafe bucket": []},
    ],
)
def test_malformed_request_bucket_ledger_fails_closed(
    tmp_path: Path,
    request_buckets: Any,
) -> None:
    (tmp_path / CACHE_FILENAME).write_text(
        json.dumps(
            {
                "version": 1,
                "requests": [],
                "request_buckets": request_buckets,
                "datasets": {},
            }
        ),
        encoding="utf-8",
    )
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _cache(tmp_path, clock, session)

    assert cache.enabled is False
    assert cache.status().ledger_trusted is False
    assert cache.status().last_error == "ledger_invalid"
    with pytest.raises(FantasyProsRequestError) as failed:
        cache.get_json("nfl/news")
    assert failed.value.code == "ledger_untrusted"
    assert session.calls == []


def test_get_json_confines_host_and_sanitizes_transport_failures(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(
        clock,
        error=RuntimeError(f"headers contained x-api-key={SECRET}"),
    )
    cache = _cache(tmp_path, clock, session)

    with pytest.raises(FantasyProsRequestError) as invalid:
        cache.get_json("https://attacker.invalid/collect")
    assert invalid.value.code == "invalid_request"
    assert invalid.value.request_reserved is False
    assert session.calls == []
    assert cache.request_usage() == 0

    with pytest.raises(FantasyProsRequestError) as failed:
        cache.get_json("nfl/news")
    assert failed.value.code == "request_failed"
    assert failed.value.request_reserved is True
    assert SECRET not in str(failed.value)
    assert cache.request_usage() == 1
    assert SECRET not in (tmp_path / CACHE_FILENAME).read_text(encoding="utf-8")


def test_get_json_uses_the_same_serialization_lock_as_refresh(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _cache(tmp_path, clock, session)
    entered = threading.Event()

    def request() -> Any:
        entered.set()
        return cache.get_json("nfl/news")

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        with cache._refresh_lock:
            future = pool.submit(request)
            assert entered.wait(timeout=1)
            time.sleep(0.02)
            assert session.calls == []
        assert future.result(timeout=1) == {"ok": True}
        assert len(session.calls) == 1
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def test_news_transport_requires_only_the_public_cache_method() -> None:
    class PublicOnlyCache:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, ...]] = []

        def get_json(
            self,
            path: str,
            *,
            params: dict[str, Any],
            request_ceiling: int,
            request_bucket: str,
            request_bucket_limit: int,
        ) -> Any:
            self.calls.append(
                (
                    path,
                    params,
                    request_ceiling,
                    request_bucket,
                    request_bucket_limit,
                )
            )
            return {"sport": "NFL", "items": []}

    cache = PublicOnlyCache()
    transport = _SharedFantasyProsNewsTransport(cache)  # type: ignore[arg-type]
    query = FantasyProsNewsQuery(category="injury")

    assert transport.fetch(
        query,
        request_ceiling=350,
        request_bucket_limit=300,
    ) == {"sport": "NFL", "items": []}
    assert cache.calls == [
        (
            "nfl/news",
            query.params,
            350,
            CORPUS_REQUEST_BUCKET,
            300,
        )
    ]

    class FailingPublicOnlyCache(PublicOnlyCache):
        def get_json(self, path: str, **_kwargs: Any) -> Any:
            raise FantasyProsRequestError(
                "request_failed",
                request_reserved=True,
            )

    with pytest.raises(FantasyProsCorpusError) as failed:
        _SharedFantasyProsNewsTransport(  # type: ignore[arg-type]
            FailingPublicOnlyCache()
        ).fetch(
            query,
            request_ceiling=350,
            request_bucket_limit=300,
        )
    assert failed.value.code == "request_failed"
    assert failed.value.request_reserved is True
    assert SECRET not in str(failed.value)
