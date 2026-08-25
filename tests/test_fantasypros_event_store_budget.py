from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier

import pytest

from notifier.event_store import EventStore


NOW = datetime(2026, 8, 24, 22, 0, tzinfo=timezone.utc)


def test_latest_corpus_run_returns_literal_prefix_manifest_status_and_times(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path)
    try:
        first_manifest = ("order=created;category=all;fpid=all;limit=100",)
        second_manifest = ("order=created;category=injury;fpid=all;limit=100",)
        store.begin_fantasypros_corpus_run("bootstrap-v1-first", first_manifest)
        store.pause_fantasypros_corpus_run("bootstrap-v1-first", "request_limit")
        store.begin_fantasypros_corpus_run("not-bootstrap", first_manifest)
        store.begin_fantasypros_corpus_run("bootstrap-v1-second", second_manifest)
        store.store_fantasypros_corpus_batch(
            "bootstrap-v1-second",
            second_manifest[0],
            [],
            fetched_at=NOW,
        )
        assert store.complete_fantasypros_corpus_run("bootstrap-v1-second")

        latest = store.latest_fantasypros_corpus_run("bootstrap-v1-")

        assert latest is not None
        assert set(latest) == {
            "run_id",
            "manifest",
            "status",
            "stop_reason",
            "started_at",
            "updated_at",
            "completed_at",
        }
        assert latest["run_id"] == "bootstrap-v1-second"
        assert latest["manifest"] == second_manifest
        assert latest["status"] == "completed"
        assert latest["stop_reason"] == ""
        assert datetime.fromisoformat(latest["started_at"]).tzinfo is not None
        assert datetime.fromisoformat(latest["updated_at"]).tzinfo is not None
        assert datetime.fromisoformat(latest["completed_at"]).tzinfo is not None

        # Prefixes are literal, not LIKE patterns.
        assert store.latest_fantasypros_corpus_run("bootstrap%") is None
        assert store.latest_fantasypros_corpus_run("") is None
    finally:
        store.close()


def test_latest_corpus_run_fails_closed_on_corrupt_manifest(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    try:
        store.begin_fantasypros_corpus_run("delta-valid", ("query-one",))
        store._connection.execute(
            "UPDATE fantasypros_corpus_runs SET manifest_json = ? WHERE run_id = ?",
            ('{"not":"a-list"}', "delta-valid"),
        )
        store._connection.commit()

        assert store.latest_fantasypros_corpus_run("delta-") is None
    finally:
        store.close()


def test_embedding_spend_status_documents_empty_summary(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    try:
        status = store.fantasypros_corpus_embedding_spend_status(
            lifetime_cap_usd=0.25
        )

        assert status == {
            "reservations": 0,
            "open_reservations": 0,
            "completed_reservations": 0,
            "failed_reservations": 0,
            "reserved_tokens": 0,
            "actual_prompt_tokens": 0,
            "unknown_actual_reservations": 0,
            "charged_cost_usd": 0.0,
            "actual_cost_usd": 0.0,
            "lifetime_cap_usd": 0.25,
            "remaining_usd": 0.25,
            "budget_exhausted": False,
        }
    finally:
        store.close()


def test_every_reservation_stays_charged_after_success_failure_and_restart(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path)
    first = store.reserve_fantasypros_corpus_embedding_spend(
        model="test/embedding",
        input_price_per_million_usd=0.01,
        conservative_tokens=100_000,
        lifetime_cap_usd=0.003,
    )
    second = store.reserve_fantasypros_corpus_embedding_spend(
        model="test/embedding",
        input_price_per_million_usd=0.01,
        conservative_tokens=100_000,
        lifetime_cap_usd=0.003,
    )
    third = store.reserve_fantasypros_corpus_embedding_spend(
        model="test/embedding-v2",
        input_price_per_million_usd=0.01,
        conservative_tokens=100_000,
        lifetime_cap_usd=0.003,
    )
    assert isinstance(first, int)
    assert isinstance(second, int)
    assert isinstance(third, int)
    assert store.finish_fantasypros_corpus_embedding_spend(
        first,
        actual_prompt_tokens=90_000,
        status="completed",
    )
    assert store.finish_fantasypros_corpus_embedding_spend(
        second,
        actual_prompt_tokens=None,
        status="failed",
    )
    # ``third`` intentionally remains reserved, modeling a process crash.
    ledger = store._connection.execute(
        """
        SELECT request_id, model, input_price_per_million_usd,
               conservative_tokens, status, reserved_at, finished_at,
               actual_prompt_tokens
        FROM fantasypros_corpus_embedding_spend
        ORDER BY request_id
        """
    ).fetchall()
    assert [row["status"] for row in ledger] == [
        "completed",
        "failed",
        "reserved",
    ]
    assert ledger[0]["model"] == "test/embedding"
    assert ledger[0]["input_price_per_million_usd"] == 0.01
    assert ledger[0]["conservative_tokens"] == 100_000
    assert ledger[0]["actual_prompt_tokens"] == 90_000
    assert datetime.fromisoformat(ledger[0]["reserved_at"]).tzinfo is not None
    assert datetime.fromisoformat(ledger[0]["finished_at"]).tzinfo is not None
    assert ledger[2]["finished_at"] is None
    store.close()

    reopened = EventStore(tmp_path)
    try:
        denied = reopened.reserve_fantasypros_corpus_embedding_spend(
            model="test/embedding-v3",
            input_price_per_million_usd=0.01,
            conservative_tokens=1,
            lifetime_cap_usd=0.003,
        )
        assert denied is None

        status = reopened.fantasypros_corpus_embedding_spend_status(
            lifetime_cap_usd=0.003
        )
        assert status["reservations"] == 3
        assert status["completed_reservations"] == 1
        assert status["failed_reservations"] == 1
        assert status["open_reservations"] == 1
        assert status["unknown_actual_reservations"] == 2
        assert status["reserved_tokens"] == 300_000
        assert status["actual_prompt_tokens"] == 90_000
        assert status["charged_cost_usd"] == pytest.approx(0.003)
        assert status["actual_cost_usd"] == pytest.approx(0.0009)
        assert status["remaining_usd"] == 0.0
        assert status["budget_exhausted"] is True
    finally:
        reopened.close()


def test_embedding_spend_finish_is_terminal_and_idempotent(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    try:
        request_id = store.reserve_fantasypros_corpus_embedding_spend(
            model="test/embedding",
            input_price_per_million_usd=0.01,
            conservative_tokens=50,
            lifetime_cap_usd=1.0,
        )
        assert isinstance(request_id, int)
        assert store.finish_fantasypros_corpus_embedding_spend(
            request_id,
            actual_prompt_tokens=25,
            status="completed",
        )
        assert store.finish_fantasypros_corpus_embedding_spend(
            request_id,
            actual_prompt_tokens=25,
            status="completed",
        )
        assert not store.finish_fantasypros_corpus_embedding_spend(
            request_id,
            actual_prompt_tokens=None,
            status="failed",
        )
        assert not store.finish_fantasypros_corpus_embedding_spend(
            request_id,
            actual_prompt_tokens=26,
            status="completed",
        )
        with pytest.raises(ValueError, match="requires actual prompt tokens"):
            store.finish_fantasypros_corpus_embedding_spend(
                request_id + 1,
                actual_prompt_tokens=None,
                status="completed",
            )
    finally:
        store.close()


def test_embedding_spend_rounds_reservations_up_to_fail_closed(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    try:
        first = store.reserve_fantasypros_corpus_embedding_spend(
            model="tiny-price",
            input_price_per_million_usd=0.000001,
            conservative_tokens=1,
            lifetime_cap_usd=0.000000001,
        )
        second = store.reserve_fantasypros_corpus_embedding_spend(
            model="tiny-price",
            input_price_per_million_usd=0.000001,
            conservative_tokens=1,
            lifetime_cap_usd=0.000000001,
        )

        assert isinstance(first, int)
        assert second is None
        status = store.fantasypros_corpus_embedding_spend_status(
            lifetime_cap_usd=0.000000001
        )
        assert status["charged_cost_usd"] == 0.000000001
    finally:
        store.close()


def test_embedding_spend_reservation_is_atomic_across_connections(
    tmp_path: Path,
) -> None:
    first_store = EventStore(tmp_path)
    second_store = EventStore(tmp_path)
    barrier = Barrier(2)

    def reserve(store: EventStore) -> int | None:
        barrier.wait()
        return store.reserve_fantasypros_corpus_embedding_spend(
            model="test/embedding",
            input_price_per_million_usd=1.0,
            conservative_tokens=1_000_000,
            lifetime_cap_usd=1.0,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(reserve, (first_store, second_store)))

        assert sum(isinstance(result, int) for result in results) == 1
        assert results.count(None) == 1
        status = first_store.fantasypros_corpus_embedding_spend_status(
            lifetime_cap_usd=1.0
        )
        assert status["reservations"] == 1
        assert status["charged_cost_usd"] == 1.0
    finally:
        first_store.close()
        second_store.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "model": "",
                "input_price_per_million_usd": 0.01,
                "conservative_tokens": 1,
                "lifetime_cap_usd": 1.0,
            },
            "embedding model",
        ),
        (
            {
                "model": "model",
                "input_price_per_million_usd": float("nan"),
                "conservative_tokens": 1,
                "lifetime_cap_usd": 1.0,
            },
            "finite",
        ),
        (
            {
                "model": "model",
                "input_price_per_million_usd": 0.01,
                "conservative_tokens": 0,
                "lifetime_cap_usd": 1.0,
            },
            "positive integer",
        ),
        (
            {
                "model": "model",
                "input_price_per_million_usd": 0.01,
                "conservative_tokens": 1,
                "lifetime_cap_usd": 0,
            },
            "positive",
        ),
    ],
)
def test_embedding_spend_rejects_invalid_reservations(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    store = EventStore(tmp_path)
    try:
        with pytest.raises(ValueError, match=message):
            store.reserve_fantasypros_corpus_embedding_spend(**kwargs)  # type: ignore[arg-type]
        assert store.fantasypros_corpus_embedding_spend_status()["reservations"] == 0
    finally:
        store.close()
