from __future__ import annotations

from concurrent.futures import TimeoutError
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from notifier.embeddings import (
    INPUT_VERSION,
    EmbeddingService,
    canonical_embedding_text,
    embedding_input_hash,
    normalize_vector,
    pack_vector,
)
from notifier.event_store import EventStore
from notifier.models import Alert, Classification, NewsItem, report_revision_identity
from notifier.pipeline import Notifier


def _item() -> NewsItem:
    return NewsItem(
        source="twitter",
        guid="twitter:embedding-reliability",
        player_name="Example Player",
        headline="Example Player remained limited in practice",
        body="Example Player remained limited in practice",
        url="https://example.test/news",
        published_at=datetime(2026, 8, 24, 12, tzinfo=timezone.utc),
    )


def _alert(item: NewsItem) -> Alert:
    return Alert(
        item=item,
        classification=Classification(
            event_type="practice_report",
            severity=3,
            fantasy_impact="Monitor availability",
            is_actionable=True,
            raw={"direction": "mixed"},
        ),
        tier="league",
    )


def _stored_service(tmp_path) -> tuple[EventStore, EmbeddingService, NewsItem]:
    store = EventStore(tmp_path)
    item = _item()
    store.record_received(item)
    values = normalize_vector((3.0, 4.0), dimensions=2)
    assert store.store_embedding(
        item,
        "provider/model",
        pack_vector(values),
        provider="openrouter",
        dimensions=2,
        input_version=INPUT_VERSION,
        input_hash=embedding_input_hash(canonical_embedding_text(item)),
    )
    service = EmbeddingService(
        store,
        api_key="",
        mode="off",
        model="provider/model",
        dimensions=2,
    )
    return store, service, item


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("embedding_provider", ""),
        ("embedding_provider", "   "),
        ("embedding_at", None),
        ("embedding_at", "   "),
        ("embedding_input_hash", ""),
    ],
)
def test_incomplete_vector_provenance_fails_closed_and_reenters_backlog(
    tmp_path,
    column: str,
    value: str | None,
) -> None:
    store, service, item = _stored_service(tmp_path)
    try:
        assert service._row_vector(item) is not None
        store._connection.execute(
            f"UPDATE news_events SET {column} = ? WHERE report_id = ?",
            (value, report_revision_identity(item)),
        )
        store._connection.commit()

        assert service._row_vector(item) is None
        backlog = store.embedding_backlog(
            model="provider/model",
            dimensions=2,
            input_version=INPUT_VERSION,
        )
        assert [row["report_id"] for row in backlog] == [
            report_revision_identity(item)
        ]
    finally:
        service.close()
        store.close()


class _PendingFuture:
    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    @staticmethod
    def done() -> bool:
        return False

    def result(self, timeout: float | None = None):
        self.timeouts.append(timeout)
        raise TimeoutError


def test_urgency_wait_then_annotation_abstains_without_second_wait(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item()
    store.record_received(item)
    service = EmbeddingService(
        store,
        api_key="secret",
        mode="coalesce",
        model="provider/model",
        dimensions=2,
        wait_ms=250,
    )
    pending = _PendingFuture()
    service._futures[report_revision_identity(item)] = pending
    alert = _alert(item)
    try:
        # Urgency owns the one bounded wait for this in-flight vector.
        assert service.current_vector(item) is None
        assert pending.timeouts == [pytest.approx(0.25)]

        # Coalescing may use a finished result, but cannot wait another 250 ms.
        assert service.annotate(
            alert,
            active_message_id=42,
            active_alert_token="prior-token",
            wait_for_vector=False,
        ) == alert
        assert pending.timeouts == [pytest.approx(0.25)]
    finally:
        service.close()
        store.close()


def test_nonblocking_lookup_abstains_without_starting_another_request(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item()
    store.record_received(item)
    service = EmbeddingService(
        store,
        api_key="secret",
        mode="coalesce",
        model="provider/model",
        dimensions=2,
    )
    service.enqueue = Mock()
    try:
        assert service.current_vector(item, wait_for_result=False) is None
        service.enqueue.assert_not_called()
    finally:
        service.close()
        store.close()


def test_pipeline_marks_coalescing_as_nonblocking_after_urgency() -> None:
    item = _item()
    alert = _alert(item)
    annotate = Mock(return_value=alert)
    notifier = Notifier.__new__(Notifier)
    notifier.config = SimpleNamespace(dry_run=True)
    notifier.embeddings = SimpleNamespace(annotate=annotate)
    notifier.telegram_state = SimpleNamespace(
        active_edit_identity=Mock(return_value=(42, "prior-token"))
    )
    notifier._semantic_is_new = Mock(return_value=False)
    notifier._release_item = Mock()

    assert notifier._complete_evaluation(item, alert) == 0

    annotate.assert_called_once_with(
        alert,
        active_message_id=42,
        active_alert_token="prior-token",
        wait_for_vector=False,
    )
    notifier._release_item.assert_called_once_with(item)
